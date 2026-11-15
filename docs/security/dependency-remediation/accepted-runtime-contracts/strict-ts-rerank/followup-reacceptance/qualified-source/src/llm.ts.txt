/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */

import {
  getLlama,
  resolveModelFile,
  LlamaChatSession,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext,
  type Token as LlamaToken,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync, statSync, unlinkSync, readdirSync, readFileSync, writeFileSync } from "fs";
import {
  type EmbeddingDescriptor,
  computeEmbeddingSpaceId,
} from "./embedding/contract.js";
import {
  resolveEmbeddingConfig,
} from "./embedding/config.js";
import type { MlxEmbedClient, MlxEmbedConfig } from "./mlx.js";
import { LLMLifecycle, abortable, linkSignals, throwIfAborted } from "./llm/lifecycle.js";
import { stageCacheIdentity, actualStageIdentity, generateMlxWithIdentity, getMlxStageContract, type MlxStageContract } from "./llm/mlx-identity.js";

// Accounting lives on the owner, including structural mocks, not in a global manager.
const INSTANCE_LIFECYCLE = Symbol("qmd.llm.lifecycle");
function lifecycleFor(llm: LlamaCpp): LLMLifecycle {
  const owner = llm as unknown as { [INSTANCE_LIFECYCLE]?: LLMLifecycle };
  return owner[INSTANCE_LIFECYCLE] ??= new LLMLifecycle();
}

type DisposableLLMResource = { dispose?: () => void | Promise<void> };

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Detect if a model URI uses the Qwen3-Embedding format.
 * Qwen3-Embedding uses a different prompting style than nomic/embeddinggemma.
 */
export function isQwen3EmbeddingModel(modelUri: string): boolean {
  return /qwen.*embed/i.test(modelUri) || /embed.*qwen/i.test(modelUri);
}

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma (default).
 * Uses Qwen3-Embedding instruct format when a Qwen embedding model is active.
 */
export function formatQueryForEmbedding(query: string, modelUri?: string): string {
  const uri = modelUri ?? process.env.QMD_EMBED_MODEL ?? DEFAULT_EMBED_MODEL;
  if (isQwen3EmbeddingModel(uri)) {
    return `Instruct: Retrieve relevant documents for the given query\nQuery: ${query}`;
  }
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields (default).
 * Qwen3-Embedding encodes documents as raw text without special prefixes.
 */
export function formatDocForEmbedding(text: string, title?: string, modelUri?: string): string {
  const uri = modelUri ?? process.env.QMD_EMBED_MODEL ?? DEFAULT_EMBED_MODEL;
  if (isQwen3EmbeddingModel(uri)) {
    // Qwen3-Embedding: documents are raw text, no task prefix
    return title ? `${title}\n${text}` : text;
  }
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
  signal?: AbortSignal;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
  signal?: AbortSignal;
};

export type ExpandOptions = {
  context?: string;
  includeLexical?: boolean;
  intent?: string;
  signal?: AbortSignal;
};

export type ExpansionResult = { queries: Queryable[]; model: string };
export type SignalOptions = { signal?: AbortSignal } | AbortSignal;

function optionSignal(options: SignalOptions): AbortSignal | undefined {
  return 'aborted' in options ? options : options.signal;
}

/**
 * Options for LLM sessions
 */
export type LLMSessionOptions = {
  /** Max session duration in ms (default: 120 minutes; 0 disables the deadline) */
  maxDuration?: number;
  /** External abort signal */
  signal?: AbortSignal;
  /** Debug name for logging */
  name?: string;
};

/**
 * Session interface for scoped LLM access with lifecycle guarantees
 */
export interface ILLMSession {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]>;
  expandQuery(query: string, options?: ExpandOptions): Promise<Queryable[]>;
  expandQueryWithIdentity?(query: string, options?: ExpandOptions): Promise<ExpansionResult>;
  getCacheIdentity?(op: 'rerank' | 'expand', options?: { signal?: AbortSignal }): Promise<string | null>;
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
  getDescriptor?(options?: SignalOptions): Promise<EmbeddingDescriptor | null>;
  /** Whether this session is still valid (not released or aborted) */
  readonly isValid: boolean;
  /** Abort signal for this session (aborts on release or maxDuration) */
  readonly signal: AbortSignal;
}

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};


// =============================================================================
// LLM Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
// Override via QMD_EMBED_MODEL env var (e.g. hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf)
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
// const DEFAULT_GENERATE_MODEL = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

// Alternative generation models for query expansion:
// LiquidAI LFM2 - hybrid architecture optimized for edge/on-device inference
// Use these as base for fine-tuning with configs/sft_lfm2.yaml
export const LFM2_GENERATE_MODEL = "hf:LiquidAI/LFM2-1.2B-GGUF/LFM2-1.2B-Q4_K_M.gguf";
export const LFM2_INSTRUCT_MODEL = "hf:LiquidAI/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_K_M.gguf";

export const DEFAULT_EMBED_MODEL_URI = DEFAULT_EMBED_MODEL;
export const DEFAULT_RERANK_MODEL_URI = DEFAULT_RERANK_MODEL;
export const DEFAULT_GENERATE_MODEL_URI = DEFAULT_GENERATE_MODEL;

// Local model cache directory
const MODEL_CACHE_DIR = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, "qmd", "models")
  : join(homedir(), ".cache", "qmd", "models");
export const DEFAULT_MODEL_CACHE_DIR = MODEL_CACHE_DIR;

export type PullResult = {
  model: string;
  path: string;
  sizeBytes: number;
  refreshed: boolean;
};

type HfRef = {
  repo: string;
  file: string;
};

function parseHfUri(model: string): HfRef | null {
  if (!model.startsWith("hf:")) return null;
  const without = model.slice(3);
  const parts = without.split("/");
  if (parts.length < 3) return null;
  const repo = parts.slice(0, 2).join("/");
  const file = parts.slice(2).join("/");
  return { repo, file };
}

async function getRemoteEtag(ref: HfRef): Promise<string | null> {
  const url = `https://huggingface.co/${ref.repo}/resolve/main/${ref.file}`;
  try {
    const resp = await fetch(url, { method: "HEAD" });
    if (!resp.ok) return null;
    const etag = resp.headers.get("etag");
    return etag || null;
  } catch {
    return null;
  }
}

export async function pullModels(
  models: string[],
  options: { refresh?: boolean; cacheDir?: string } = {}
): Promise<PullResult[]> {
  const cacheDir = options.cacheDir || MODEL_CACHE_DIR;
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir, { recursive: true });
  }

  const results: PullResult[] = [];
  for (const model of models) {
    let refreshed = false;
    const hfRef = parseHfUri(model);
    const filename = model.split("/").pop();
    const entries = readdirSync(cacheDir, { withFileTypes: true });
    const cached = filename
      ? entries
          .filter((entry) => entry.isFile() && entry.name.includes(filename))
          .map((entry) => join(cacheDir, entry.name))
      : [];

    if (hfRef && filename) {
      const etagPath = join(cacheDir, `${filename}.etag`);
      const remoteEtag = await getRemoteEtag(hfRef);
      const localEtag = existsSync(etagPath)
        ? readFileSync(etagPath, "utf-8").trim()
        : null;
      const shouldRefresh =
        options.refresh || !remoteEtag || remoteEtag !== localEtag || cached.length === 0;

      if (shouldRefresh) {
        for (const candidate of cached) {
          if (existsSync(candidate)) unlinkSync(candidate);
        }
        if (existsSync(etagPath)) unlinkSync(etagPath);
        refreshed = cached.length > 0;
      }
    } else if (options.refresh && filename) {
      for (const candidate of cached) {
        if (existsSync(candidate)) unlinkSync(candidate);
        refreshed = true;
      }
    }

    const path = await resolveModelFile(model, cacheDir);
    const sizeBytes = existsSync(path) ? statSync(path).size : 0;
    if (hfRef && filename) {
      const remoteEtag = await getRemoteEtag(hfRef);
      if (remoteEtag) {
        const etagPath = join(cacheDir, `${filename}.etag`);
        writeFileSync(etagPath, remoteEtag + "\n", "utf-8");
      }
    }
    results.push({ model, path, sizeBytes, refreshed });
  }
  return results;
}

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;
  getDescriptor?(options?: SignalOptions): Promise<EmbeddingDescriptor | null>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(query: string, options?: ExpandOptions): Promise<Queryable[]>;
  expandQueryWithIdentity?(query: string, options?: ExpandOptions): Promise<ExpansionResult>;
  getCacheIdentity?(op: 'rerank' | 'expand', options?: { signal?: AbortSignal }): Promise<string | null>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// node-llama-cpp Implementation
// =============================================================================

// Embed via MLX when QMD_EMBED_BACKEND=mlx
export function getEmbedBackend(): 'gguf' | 'mlx' {
  if (process.env.QMD_STRICT_MLX === '1') return 'mlx';
  const raw = process.env.QMD_EMBED_BACKEND?.trim().toLowerCase() ?? '';
  return raw === 'mlx' ? 'mlx' : 'gguf';
}

export async function isMlxBackendAvailable(): Promise<boolean> {
  if (getEmbedBackend() !== 'mlx') return false;
  const { isMlxAvailable } = await import('./mlx.js');
  return isMlxAvailable();
}

export type LlamaCppConfig = {
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  modelCacheDir?: string;
  /**
   * Embedding backend: 'gguf' (node-llama-cpp, default) or 'mlx' (HTTP server).
   * Can also be set via QMD_EMBED_BACKEND env var.
   */
  embedBackend?: 'gguf' | 'mlx';
  /**
   * MLX server URL when embedBackend='mlx'.
   * Defaults to QMD_MLX_EMBED_URL env or http://127.0.0.1:8787
   */
  mlxUrl?: string;
  /** Validated MLX transport deadline, binary preference and output dimensions. */
  mlxTimeoutMs?: number;
  mlxBinary?: boolean;
  embedDims?: number;
  /**
   * @deprecated No-op: the daemon serves pre-quantized weights; precision
   * comes from the model repo (e.g. -4bit vs -8bit), not this flag.
   */
   mlxDtype?: 'float32' | 'float16' | 'bfloat16';
  /**
   * @deprecated Embeddings never fall back across vector spaces. Select GGUF
   * explicitly and rebuild the index instead. Retained for config compatibility.
   */
  mlxFallback?: boolean;
  /**
   * Strict all-MLX mode: disables all GGUF fallbacks across embed, rerank, and expand.
   * Hard-fails closed on daemon unreachability or missing models.
   */
  strictMlx?: boolean;
  /**
   * MLX concurrent request limit (default: 2). Increase to 3-4 if embedding
   * throughput is bottlenecked on network latency.
   */
  mlxConcurrency?: number;
  /**
   * Context size used for query expansion generation contexts.
   * Default: 2048. Can also be set via QMD_EXPAND_CONTEXT_SIZE.
   */
  expandContextSize?: number;
  /**
   * Inactivity timeout in ms before unloading contexts (default: 2 minutes, 0 to disable).
   *
   * Per node-llama-cpp lifecycle guidance, we prefer keeping models loaded and only disposing
   * contexts when idle, since contexts (and their sequences) are the heavy per-session objects.
   * @see https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
   */
  inactivityTimeoutMs?: number;
  /**
   * Whether to dispose models on inactivity (default: false).
   *
   * By default, only contexts are disposed to balance memory efficiency with performance.
   * Set to true to fully unload model weights from VRAM after inactivity.
   */
  disposeModelsOnInactivity?: boolean;
};

/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 5 minutes (keep models warm during typical search sessions)
const DEFAULT_INACTIVITY_TIMEOUT_MS = 5 * 60 * 1000;
const DEFAULT_EXPAND_CONTEXT_SIZE = 2048;

function resolveExpandContextSize(configValue?: number): number {
  if (configValue !== undefined) {
    if (!Number.isInteger(configValue) || configValue <= 0) {
      throw new Error(`Invalid expandContextSize: ${configValue}. Must be a positive integer.`);
    }
    return configValue;
  }

  const envValue = process.env.QMD_EXPAND_CONTEXT_SIZE?.trim();
  if (!envValue) return DEFAULT_EXPAND_CONTEXT_SIZE;

  const parsed = Number.parseInt(envValue, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    process.stderr.write(
      `QMD Warning: invalid QMD_EXPAND_CONTEXT_SIZE="${envValue}", using default ${DEFAULT_EXPAND_CONTEXT_SIZE}.\n`
    );
    return DEFAULT_EXPAND_CONTEXT_SIZE;
  }
  return parsed;
}

export class LlamaCpp implements LLM {
  private readonly _ciMode = !!process.env.CI;
  private readonly [INSTANCE_LIFECYCLE] = new LLMLifecycle();
  private readonly mlxOperationDeadlines = new WeakMap<AbortSignal, { expires: number; controller: AbortController }>();
  private llamaLoadPromise: Promise<Llama> | null = null;
  private llama: Llama | null = null;
  private embedModel: LlamaModel | null = null;
  private embedContexts: LlamaEmbeddingContext[] = [];
  private generateModel: LlamaModel | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContexts: Awaited<ReturnType<LlamaModel["createRankingContext"]>>[] = [];

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;
  private expandContextSize: number;

  // MLX backend state
  private embedBackend: 'gguf' | 'mlx';
  private mlxUrlOverride: string | null = null;
  private mlxDtype: 'float32' | 'float16' | 'bfloat16';
  private mlxConcurrency: number;
  private mlxTimeoutMs: number;
  private mlxBinary: boolean;
  private embedDims: number | undefined;
  private mlxFallback: boolean;
  private failClosed: boolean;
  private strictMlx: boolean;
  private useMlxRerank: boolean;
  private useMlxExpand: boolean;
  private mlxClient: MlxEmbedClient | null = null;
  private mlxClientLoad: { promise: Promise<void>; controller: AbortController; waiters: number } | null = null;
  private mlxDims: number | null = null;
  private mlxName: string | null = null;
  private mlxDescriptor: EmbeddingDescriptor | null = null;
  private mlxFailed = false;
  private mlxWarm = false;  // whether first request already warmed up
  private mlxRerankFallback: boolean;   // GGUF fallback allowed for rerank (default: true)
  private mlxExpandFallback: boolean;   // GGUF fallback allowed for expansion (default: true)
  private embedModelLoadPromise: Promise<LlamaModel> | null = null;
  private generateModelLoadPromise: Promise<LlamaModel> | null = null;
  private rerankModelLoadPromise: Promise<LlamaModel> | null = null;
  private rerankContextsCreatePromise: Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>[]> | null = null;

  // Inactivity timer for auto-unloading models
  private inactivityTimer: ReturnType<typeof setTimeout> | null = null;
  private inactivityTimeoutMs: number;
  private disposeModelsOnInactivity: boolean;

  // Track disposal state to prevent double-dispose
  private disposed = false;
  private disposePromise: Promise<void> | null = null;
  private idleUnloadPromise: Promise<void> | null = null;
  private resourceDisposals = new WeakMap<object, Promise<void>>();
  // Strong ownership is essential for failed per-call, partial-pool and late-load cleanup too.
  private failedResourceDisposals = new Map<DisposableLLMResource, { error: unknown; attempts: number }>();
  private resourceParents = new WeakMap<DisposableLLMResource, DisposableLLMResource>();


  constructor(config: LlamaCppConfig = {}) {
    const isStrictMlx = process.env.QMD_STRICT_MLX === '1' || config.strictMlx === true;
    const resolved = resolveEmbeddingConfig({
      backend: isStrictMlx ? 'mlx' : config.embedBackend,
      model: config.embedModel,
      dims: config.embedDims,
      mlxUrl: config.mlxUrl,
      mlxConcurrency: config.mlxConcurrency,
      mlxTimeoutMs: config.mlxTimeoutMs,
      mlxBinary: config.mlxBinary,
    });

    this.embedBackend = resolved.backend;
    this.mlxUrlOverride = resolved.mlxUrl;
    // DEPRECATED (no-op): the daemon serves pre-quantized weights, so there is
    // no runtime dtype to select. Kept for config compatibility; warns once when
    // explicitly set so stale configs don't silently believe it applies.
    this.mlxDtype = config.mlxDtype ?? 'float32';
    if (config.mlxDtype !== undefined) {
      console.warn(
        `QMD_MLX_DTYPE/mlxDtype is deprecated and ignored: the MLX daemon serves ` +
        `pre-quantized weights (pick precision via the model repo, e.g. -4bit vs -8bit).`
      );
    }
    this.mlxConcurrency = resolved.mlxConcurrency;
    this.mlxTimeoutMs = resolved.mlxTimeoutMs;
    this.mlxBinary = resolved.mlxBinary;
    this.embedDims = resolved.dims;
    this.strictMlx = isStrictMlx;
    // A transport retry must never substitute vectors from another embedding space.
    this.failClosed = resolved.backend === 'mlx';
    this.mlxFallback = false;
    this.useMlxRerank = resolved.backend === 'mlx' && (isStrictMlx || process.env.QMD_MLX_RERANK === '1');
    this.useMlxExpand = resolved.backend === 'mlx' && (isStrictMlx || process.env.QMD_MLX_EXPAND === '1');
    this.mlxRerankFallback = !isStrictMlx && (process.env.QMD_MLX_RERANK_FALLBACK ?? '1') !== '0';
    this.mlxExpandFallback = !isStrictMlx && (process.env.QMD_MLX_EXPAND_FALLBACK ?? '1') !== '0';

    this.embedModelUri = config.embedModel || process.env.QMD_EMBED_MODEL || (this.embedBackend === 'mlx' ? resolved.model : DEFAULT_EMBED_MODEL);
    this.generateModelUri = config.generateModel || process.env.QMD_GENERATE_MODEL || DEFAULT_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || process.env.QMD_RERANK_MODEL || DEFAULT_RERANK_MODEL;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
    this.expandContextSize = resolveExpandContextSize(config.expandContextSize);
    this.inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
    this.disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
    lifecycleFor(this).onActivity = () => this.touchActivity();
  }

  get embedModelName(): string {
    // Return MLX model name when MLX backend is active and loaded
    if (this.embedBackend === 'mlx' && !this.mlxFailed) {
      return this.mlxName ?? this.embedModelUri;
    }
    return this.embedModelUri;
  }

  /** True when this instance will route rerank calls through the MLX daemon. */
  private mlxRerankEnabled(): boolean {
    return this.useMlxRerank;
  }

  /** True when this instance will route expansion calls through the MLX daemon. */
  private mlxExpandEnabled(): boolean {
    return this.useMlxExpand;
  }

  /** Fresh effective namespace for cache reads; null means the identity is unprovable. */
  async getCacheIdentity(op: 'rerank' | 'expand', options: { signal?: AbortSignal } = {}): Promise<string | null> {
    const enabled = op === 'rerank' ? this.mlxRerankEnabled() : this.mlxExpandEnabled();
    return this.withOperation(optionSignal(options), async signal => {
      if (!enabled) return op === 'rerank' ? this.rerankModelUri : this.generateModelUri;
      try {
        return stageCacheIdentity(await this.freshMlxStageContract(signal), op);
      } catch (error) {
        this.checkOperation(signal);
        return null;
      }
    }, enabled);
  }

  /** Compatibility shim for callers which only understand MLX overrides. */
  async mlxModelFor(op: 'rerank' | 'expand'): Promise<string | null> {
    const identity = await this.getCacheIdentity(op);
    return identity?.startsWith('mlx:') ? identity : null;
  }

  async getDescriptor(options: SignalOptions = {}): Promise<EmbeddingDescriptor | null> {
    return this.withOperation(optionSignal(options), async signal => {
      if (this.embedBackend !== 'mlx') return null; // GGUF dimensions need an actual model probe.
      await this._ensureMlxClient(signal);
      const descriptor = await this.freshMlxDescriptor(signal);
      if (!descriptor || !this.mlxDescriptor) throw new Error('MLX embedding descriptor is unavailable');
      const effective = this.withOutputDimensions(descriptor);
      if (computeEmbeddingSpaceId(effective) !== computeEmbeddingSpaceId(this.mlxDescriptor)) {
        throw new Error('MLX embedding space changed after initialization; select the matching model or rebuild the index');
      }
      return structuredClone(this.mlxDescriptor);
    }, this.embedBackend === 'mlx');
  }

  private mlxConfig(signal?: AbortSignal): MlxEmbedConfig {
    return {
      url: this.mlxUrlOverride ?? undefined,
      timeoutMs: this.mlxTimeoutMs,
      binary: this.mlxBinary,
      concurrency: this.mlxConcurrency,
      signal: signal ?? lifecycleFor(this).signal,
    };
  }

  private async freshMlxDescriptor(signal: AbortSignal): Promise<EmbeddingDescriptor | null> {
    const { getMlxDescriptor } = await import('./mlx.js');
    this.checkOperation(signal);
    const descriptor = await getMlxDescriptor(this.mlxConfig(signal));
    this.checkOperation(signal);
    return descriptor;
  }

  private withOutputDimensions(descriptor: EmbeddingDescriptor): EmbeddingDescriptor {
    if (this.embedDims === undefined) return structuredClone(descriptor);
    if (this.embedDims > descriptor.nativeDimensions) throw new Error('Requested embedding dimensions exceed the MLX model dimensions');
    return { ...structuredClone(descriptor), outputDimensions: this.embedDims };
  }

  private async freshMlxStageContract(signal: AbortSignal): Promise<MlxStageContract | null> {
    this.checkOperation(signal);
    const descriptor = await getMlxStageContract(this.mlxConfig(signal));
    this.checkOperation(signal);
    return descriptor;
  }

  private checkOperation(signal?: AbortSignal): void {
    lifecycleFor(this).assertActive();
    const deadline = signal && this.mlxOperationDeadlines.get(signal);
    if (deadline && performance.now() >= deadline.expires) {
      deadline.controller.abort(new DOMException('MLX operation deadline exceeded', 'TimeoutError'));
    }
    throwIfAborted(signal);
  }

  /** Opt in only MLX routes; unrelated native inference/disposal has no MLX budget. */
  private withOperation<T>(signal: AbortSignal | undefined, fn: (signal: AbortSignal) => Promise<T>, mlx = false): Promise<T> {
    const lifecycle = lifecycleFor(this);
    lifecycle.assertActive();
    throwIfAborted(signal);
    // The public call owns this budget, including its wait for shared cold
    // initialization. Never put a waiter's deadline on the shared client/load.
    const deadline = mlx ? { expires: performance.now() + this.mlxTimeoutMs, controller: new AbortController() } : undefined;
    const release = lifecycle.acquireOperation();
    const linked = linkSignals([signal, lifecycle.signal, deadline?.controller.signal]);
    const timer = deadline ? setTimeout(() => {
      deadline.controller.abort(new DOMException('MLX operation deadline exceeded', 'TimeoutError'));
    }, Math.max(0, Math.ceil(deadline.expires - performance.now()))) : undefined;
    timer?.unref?.();
    if (deadline) this.mlxOperationDeadlines.set(linked.signal, deadline);
    const work = (async () => {
      try {
        if (this.idleUnloadPromise) await this.idleUnloadPromise;
        this.checkOperation(linked.signal);
        const result = await fn(linked.signal);
        this.checkOperation(linked.signal);
        return result;
      } finally {
        clearTimeout(timer);
        this.mlxOperationDeadlines.delete(linked.signal);
        linked.dispose();
        release();
      }
    })();
    // If the caller cancels, keep tracking the real native task until it has stopped.
    return abortable(lifecycle.track(work), linked.signal);
  }

  private clearInactivityTimer(): void {
    if (this.inactivityTimer) clearTimeout(this.inactivityTimer);
    this.inactivityTimer = null;
  }

  private touchActivity(): void {
    this.clearInactivityTimer();
    if (this.inactivityTimeoutMs > 0 && lifecycleFor(this).canUnload && !this.idleUnloadPromise && this.hasLoadedContexts()) {
      this.inactivityTimer = setTimeout(() => {
        this.unloadIdleResources().catch(err => console.error('Error unloading idle resources:', err));
      }, this.inactivityTimeoutMs);
      this.inactivityTimer.unref?.();
    }
  }

  private hasLoadedContexts(): boolean {
    return this.embedContexts.length > 0 || this.rerankContexts.length > 0 ||
      (this.disposeModelsOnInactivity && !!(this.embedModel || this.rerankModel || this.generateModel));
  }

  /** An atomic quiesce transition: new operations wait, active sessions/operations veto it. */
  unloadIdleResources(): Promise<void> {
    if (this.idleUnloadPromise) return this.idleUnloadPromise;
    if (!lifecycleFor(this).canUnload) return Promise.resolve();
    this.clearInactivityTimer();
    const work = lifecycleFor(this).track(this.releaseNativeResources(this.disposeModelsOnInactivity));
    this.idleUnloadPromise = work.finally(() => {
      this.idleUnloadPromise = null;
      this.touchActivity();
    });
    return this.idleUnloadPromise;
  }

  private disposeResource(resource: DisposableLLMResource | null, retryAtClose = false): Promise<void> {
    if (!resource) return Promise.resolve();
    const failed = this.failedResourceDisposals.get(resource);
    const previous = this.resourceDisposals.get(resource);
    // Cache in-flight/successful work, and failures unless final close authorizes one retry.
    if (previous && !(retryAtClose && failed && failed.attempts < 2)) return previous;
    // Native parent disposal cascades. It must not retry a failed child behind our
    // accounting, or wait forever on a disposal guard that child still owns.
    if (this.hasFailedDependent(resource)) return Promise.reject(new Error('LLM cleanup is blocked by an unreleased dependent resource'));
    const attempts = (failed?.attempts ?? 0) + 1;
    // Reserve the retry before starting it so overlapping cleanup cannot retry it again.
    if (failed) this.failedResourceDisposals.set(resource, { error: failed.error, attempts });
    const work = Promise.resolve().then(() => resource.dispose?.()).then(
      () => {
        this.forgetDisposedResource(resource);
        this.failedResourceDisposals.delete(resource);
        this.updateCleanupError();
      },
      error => {
        this.failedResourceDisposals.set(resource, { error, attempts });
        this.updateCleanupError();
        throw error;
      },
    );
    this.resourceDisposals.set(resource, work);
    return work;
  }

  /** Remove owning references only after the corresponding native release succeeds. */
  private forgetDisposedResource(resource: DisposableLLMResource): void {
    this.embedContexts = this.embedContexts.filter(context => context !== resource);
    this.rerankContexts = this.rerankContexts.filter(context => context !== resource);
    if (this.embedModel === resource) this.embedModel = null;
    if (this.generateModel === resource) this.generateModel = null;
    if (this.rerankModel === resource) this.rerankModel = null;
    if (this.llama === resource) this.llama = null;
    if (this.mlxClient === resource) this.mlxClient = null;
    this.resourceParents.delete(resource);
  }

  private rememberResourceParent<T extends DisposableLLMResource>(resource: T, parent: DisposableLLMResource | null): T {
    if (parent && resource !== parent) this.resourceParents.set(resource, parent);
    return resource;
  }

  private hasFailedDependent(resource: DisposableLLMResource): boolean {
    for (const failed of this.failedResourceDisposals.keys()) {
      const seen = new Set<DisposableLLMResource>();
      let parent = this.resourceParents.get(failed);
      while (parent && !seen.has(parent)) {
        if (parent === resource) return true;
        seen.add(parent);
        parent = this.resourceParents.get(parent);
      }
    }
    return false;
  }

  private async retryFailedResourcesAtClose(): Promise<void> {
    const remaining = new Set(this.failedResourceDisposals.keys());
    while (remaining.size) {
      const ready = [...remaining].filter(resource => !this.hasFailedDependent(resource));
      if (!ready.length) return; // Failed children keep their parents owned, not cascaded.
      for (const resource of ready) {
        remaining.delete(resource); // Each snapshot entry is tried at most once, even on failure.
        await this.disposeResource(resource, true).catch(() => {});
      }
    }
  }

  private updateCleanupError(): void {
    const failures = [...this.failedResourceDisposals.values()].map(failure => failure.error);
    lifecycleFor(this).setCleanupError(failures.length ? new AggregateError(
      [...new Set(failures)],
      'LLM resource cleanup failed; call dispose() to retry cleanup once. Further operations are disabled.',
    ) : null);
  }

  private async releaseNativeResources(models: boolean): Promise<void> {
    for (const context of this.embedContexts) this.rememberResourceParent(context, this.embedModel);
    for (const context of this.rerankContexts) this.rememberResourceParent(context, this.rerankModel);
    for (const model of [this.embedModel, this.generateModel, this.rerankModel]) {
      if (model) this.rememberResourceParent(model, this.llama);
    }
    const contexts = [...this.embedContexts, ...this.rerankContexts];
    const weights = models ? [this.embedModel, this.generateModel, this.rerankModel] : [];
    const errors: unknown[] = [];
    for (const resource of new Set([...contexts, ...weights])) {
      try { await this.disposeResource(resource); } catch (error) { errors.push(error); }
    }
    if (errors.length) throw new AggregateError(errors, 'Failed to dispose LLM resources');
  }

  /**
   * Ensure model cache directory exists
   */
  private ensureModelCacheDir(): void {
    if (!existsSync(this.modelCacheDir)) {
      mkdirSync(this.modelCacheDir, { recursive: true });
    }
  }

  /**
   * Initialize the llama instance (lazy)
   */
  private async ensureLlama(): Promise<Llama> {
    this.checkOperation();
    if (this.idleUnloadPromise) await this.idleUnloadPromise;
    this.checkOperation();
    if (this.strictMlx) throw new Error('Native GGUF initialization is disabled in strict MLX mode');
    if (this._ciMode) throw new Error("LLM operations are disabled in CI (set CI=true)");
    if (this.llama) return this.llama;
    if (this.llamaLoadPromise) return this.llamaLoadPromise;
    const work = (async () => {
      const gpuOverride = (process.env.QMD_LLAMA_GPU ?? "").toLowerCase();
      const forceCpu = ["false", "off", "none", "disable", "disabled", "0"].includes(gpuOverride);
      const load = (gpu: "auto" | false) => getLlama({ build: "autoAttempt", logLevel: LlamaLogLevel.error, gpu });
      let llama: Llama;
      if (forceCpu) llama = await load(false);
      else {
        try { llama = await load("auto"); }
        catch (error) {
          this.checkOperation();
          process.stderr.write(`QMD Warning: GPU init failed (${error instanceof Error ? error.message : String(error)}), falling back to CPU.\n`);
          llama = await load(false);
        }
      }
      if (!lifecycleFor(this).active) {
        await this.disposeResource(llama);
        this.checkOperation();
      }
      if (llama.gpu === false) process.stderr.write("QMD Warning: no GPU acceleration, running on CPU (slow). Run 'qmd status' for details.\n");
      this.llama = llama;
      return llama;
    })();
    this.llamaLoadPromise = lifecycleFor(this).track(work);
    try { return await work; }
    finally { if (this.llamaLoadPromise === work) this.llamaLoadPromise = null; }
  }

  /**
   * Resolve a model URI to a local path, downloading if needed
   */
  private async resolveModel(modelUri: string): Promise<string> {
    this.checkOperation();
    this.ensureModelCacheDir();
    // resolveModelFile handles HF URIs and downloads to the cache dir
    return await resolveModelFile(modelUri, { directory: this.modelCacheDir, signal: lifecycleFor(this).signal });
  }

  /**
   * Load embedding model (lazy)
   */
  private async ensureEmbedModel(): Promise<LlamaModel> {
    return this.ensureModel('embed');
  }

  private async ensureModel(kind: 'embed' | 'generate' | 'rerank'): Promise<LlamaModel> {
    this.checkOperation();
    if (this.idleUnloadPromise) await this.idleUnloadPromise;
    this.checkOperation();
    if (this.strictMlx) throw new Error('Native GGUF models are disabled in strict MLX mode');
    const key = `${kind}Model` as const;
    const loadKey = `${kind}ModelLoadPromise` as const;
    if (this[key]) return this[key]!;
    if (this[loadKey]) return this[loadKey]!;
    const work = (async () => {
      const llama = await this.ensureLlama();
      this.checkOperation();
      const modelPath = await this.resolveModel(this[`${kind}ModelUri`]);
      this.checkOperation();
      const model = this.rememberResourceParent(await llama.loadModel({ modelPath, loadSignal: lifecycleFor(this).signal }), llama);
      if (!lifecycleFor(this).active) {
        await this.disposeResource(model);
        this.checkOperation();
      }
      this[key] = model;
      return model;
    })();
    this[loadKey] = lifecycleFor(this).track(work);
    try { return await work; }
    finally { if (this[loadKey] === work) this[loadKey] = null; }
  }

  /**
   * Compute how many parallel contexts to create.
   *
   * GPU: constrained by VRAM (25% of free, capped at 8).
   * CPU: constrained by cores. Splitting threads across contexts enables
   *      true parallelism (each context runs on its own cores). Use at most
   *      half the math cores, with at least 4 threads per context.
   */
  private async computeParallelism(perContextMB: number): Promise<number> {
    const llama = await this.ensureLlama();

    if (llama.gpu) {
      try {
        const vram = await llama.getVramState();
        const freeMB = vram.free / (1024 * 1024);
        const maxByVram = Math.floor((freeMB * 0.25) / perContextMB);
        return Math.max(1, Math.min(8, maxByVram));
      } catch {
        return 2;
      }
    }

    // CPU: split cores across contexts. At least 4 threads per context.
    const cores = llama.cpuMathCores || 4;
    const maxContexts = Math.floor(cores / 4);
    return Math.max(1, Math.min(4, maxContexts));
  }

  /**
   * Get the number of threads each context should use, given N parallel contexts.
   * Splits available math cores evenly across contexts.
   */
  private async threadsPerContext(parallelism: number): Promise<number> {
    const llama = await this.ensureLlama();
    if (llama.gpu) return 0; // GPU: let the library decide
    const cores = llama.cpuMathCores || 4;
    return Math.max(1, Math.floor(cores / parallelism));
  }

  /**
   * Load embedding contexts (lazy). Creates multiple for parallel embedding.
   * Uses promise guard to prevent concurrent context creation race condition.
   */
  private embedContextsCreatePromise: Promise<LlamaEmbeddingContext[]> | null = null;

  private async ensureEmbedContexts(): Promise<LlamaEmbeddingContext[]> {
    this.checkOperation();
    if (this.idleUnloadPromise) await this.idleUnloadPromise;
    this.checkOperation();
    if (this.embedContexts.length > 0) return this.embedContexts;
    if (this.embedContextsCreatePromise) return this.embedContextsCreatePromise;
    const work = (async () => {
      const model = await this.ensureEmbedModel();
      this.checkOperation();
      const n = await this.computeParallelism(150);
      const threads = await this.threadsPerContext(n);
      const contexts = await this.createContextPool(n, () => model.createEmbeddingContext({
        contextSize: LlamaCpp.EMBED_CONTEXT_SIZE,
        createSignal: lifecycleFor(this).signal,
        ...(threads > 0 ? { threads } : {}),
      }), model);
      if (!lifecycleFor(this).active) {
        await this.settleAll(contexts.map(context => this.disposeResource(context)));
        this.checkOperation();
      }
      this.embedContexts = contexts;
      return contexts;
    })();
    this.embedContextsCreatePromise = lifecycleFor(this).track(work);
    try { return await work; }
    finally { if (this.embedContextsCreatePromise === work) this.embedContextsCreatePromise = null; }
  }

  private async createContextPool<T extends { dispose(): void | Promise<void> }>(count: number, create: () => Promise<T>, parent: DisposableLLMResource | null = null): Promise<T[]> {
    const contexts: T[] = [];
    try {
      for (let i = 0; i < count; i++) {
        this.checkOperation();
        contexts.push(this.rememberResourceParent(await create(), parent));
        this.checkOperation();
      }
      return contexts;
    } catch (error) {
      const cleanup = await Promise.allSettled(contexts.map(context => this.disposeResource(context)));
      const failures = cleanup.filter((entry): entry is PromiseRejectedResult => entry.status === 'rejected');
      if (failures.length) throw new AggregateError([error, ...failures.map(entry => entry.reason)], 'Context creation and cleanup failed');
      throw error;
    }
  }

  /**
   * Get a single embed context (for single-embed calls). Uses first from pool.
   */
  private async ensureEmbedContext(): Promise<LlamaEmbeddingContext> {
    const contexts = await this.ensureEmbedContexts();
    return contexts[0]!;
  }

  /**
   * Load generation model (lazy) - context is created fresh per call
   */
  private async ensureGenerateModel(): Promise<LlamaModel> {
    return this.ensureModel('generate');
  }

  /**
   * Load rerank model (lazy)
   */
  private async ensureRerankModel(): Promise<LlamaModel> {
    return this.ensureModel('rerank');
  }

  /**
   * Load rerank contexts (lazy). Creates multiple contexts for parallel ranking.
   * Each context has its own sequence, so they can evaluate independently.
   *
   * Ranking contexts inherit the model's default attention selection. The
   * installed API has no per-ranking-context flashAttention option.
   */
  // Qwen3 reranker template adds ~200 tokens overhead (system prompt, tags, etc.)
  // Default 2048 was too small for longer documents (e.g. session transcripts,
  // CJK text, or large markdown files) — callers hit "input lengths exceed
  // context size" errors even after truncation because the overhead estimate
  // was insufficient.  4096 comfortably fits the largest real-world chunks
  // while staying well below the 40 960-token auto size.
  // Override with QMD_RERANK_CONTEXT_SIZE env var if you need more headroom.
  private static readonly RERANK_CONTEXT_SIZE: number = (() => {
    const v = parseInt(process.env.QMD_RERANK_CONTEXT_SIZE ?? "", 10);
    return Number.isFinite(v) && v > 0 ? v : 4096;
  })();

  private static readonly EMBED_CONTEXT_SIZE: number = (() => {
    const v = parseInt(process.env.QMD_EMBED_CONTEXT_SIZE ?? "", 10);
    return Number.isFinite(v) && v > 0 ? v : 2048;
  })();
  private async ensureRerankContexts(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>[]> {
    this.checkOperation();
    if (this.idleUnloadPromise) await this.idleUnloadPromise;
    this.checkOperation();
    if (this.rerankContexts.length > 0) return this.rerankContexts;
    if (this.rerankContextsCreatePromise) return this.rerankContextsCreatePromise;
    const work = (async () => {
      const model = await this.ensureRerankModel();
      this.checkOperation();
      const n = Math.min(await this.computeParallelism(1000), 4);
      const threads = await this.threadsPerContext(n);
      const contextOptions = {
        contextSize: LlamaCpp.RERANK_CONTEXT_SIZE,
        createSignal: lifecycleFor(this).signal,
        ...(threads > 0 ? { threads } : {}),
      } satisfies NonNullable<Parameters<LlamaModel["createRankingContext"]>[0]>;
      const contexts = await this.createContextPool(n, async () => {
        // Retry a transient creation refusal once with the same supported options.
        try { return await model.createRankingContext(contextOptions); }
        catch (error) {
          this.checkOperation();
          return model.createRankingContext(contextOptions);
        }
      }, model);
      if (!lifecycleFor(this).active) {
        await this.settleAll(contexts.map(context => this.disposeResource(context)));
        this.checkOperation();
      }
      this.rerankContexts = contexts;
      return contexts;
    })();
    this.rerankContextsCreatePromise = lifecycleFor(this).track(work);
    try { return await work; }
    finally { if (this.rerankContextsCreatePromise === work) this.rerankContextsCreatePromise = null; }
  }

  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the embedding model's tokenizer
   * Returns tokenizer tokens (opaque type from node-llama-cpp)
   */
  async tokenize(text: string, options: { signal?: AbortSignal } = {}): Promise<readonly LlamaToken[]> {
    return this.withOperation(options.signal, async signal => {
    if (this.embedBackend === 'mlx' && !this.mlxFailed) {
      const words = text.match(/\p{L}+|\p{N}+|[^\s\p{L}\p{N}]/gu) || [];
      return words.map((_, i) => i as unknown as LlamaToken);
    }
    await this.ensureEmbedContext();  // Ensure model is loaded
    this.checkOperation(signal);
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.tokenize(text);
    }, this.embedBackend === 'mlx');
  }

  /**
   * Count tokens in text using the embedding model's tokenizer
   */
  async countTokens(text: string, options: { signal?: AbortSignal } = {}): Promise<number> {
    const tokens = await this.tokenize(text, options);
    return tokens.length;
  }

  /**
   * Detokenize token IDs back to text
   */
  async detokenize(tokens: readonly LlamaToken[], options: { signal?: AbortSignal } = {}): Promise<string> {
    return this.withOperation(options.signal, async signal => {
    await this.ensureEmbedContext();
    this.checkOperation(signal);
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.detokenize(tokens);
    });
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  /**
   * Truncate text to fit within the embedding model's context window.
   * Uses the model's own tokenizer for accurate token counting, then
   * detokenizes back to text if truncation is needed.
   * Returns the (possibly truncated) text and whether truncation occurred.
   */
  private async truncateToContextSize(text: string): Promise<{ text: string; truncated: boolean }> {
    if (!this.embedModel) return { text, truncated: false };

    const maxTokens = this.embedModel.trainContextSize;
    if (maxTokens <= 0) return { text, truncated: false };

    const tokens = this.embedModel.tokenize(text);
    if (tokens.length <= maxTokens) return { text, truncated: false };

    // Leave a small margin (4 tokens) for BOS/EOS overhead
    const safeLimit = Math.max(1, maxTokens - 4);
    const truncatedTokens = tokens.slice(0, safeLimit);
    const truncatedText = this.embedModel.detokenize(truncatedTokens);
    return { text: truncatedText, truncated: true };
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    return this.withOperation(options.signal, signal => this.embedInternal(text, { ...options, signal }), this.embedBackend === 'mlx');
  }

  private async embedInternal(text: string, options: EmbedOptions): Promise<EmbeddingResult | null> {
    if (this.embedBackend === 'mlx') return this._embedMlx(text, options);
    if (this._ciMode) throw new Error("LLM operations are disabled in CI (set CI=true)");
    try {
      const context = await this.ensureEmbedContext();
      this.checkOperation(options.signal);
      const { text: safeText, truncated } = await this.truncateToContextSize(text);
      this.checkOperation(options.signal);
      if (truncated) console.warn(`⚠ Text truncated to fit embedding context (${this.embedModel?.trainContextSize} tokens)`);
      const embedding = await context.getEmbeddingFor(safeText);
      this.checkOperation(options.signal);
      return { embedding: Array.from(embedding.vector), model: this.embedModelUri };
    } catch (error) {
      this.checkOperation(options.signal);
      console.error("Embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
   */
  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    return this.withOperation(options.signal, signal => this.embedBatchInternal(texts, { ...options, signal }), this.embedBackend === 'mlx');
  }

  private async embedBatchInternal(texts: string[], options: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    if (this.embedBackend === 'mlx') return this._embedBatchMlx(texts, options);
    if (this._ciMode) throw new Error("LLM operations are disabled in CI (set CI=true)");
    try {
      const contexts = await this.ensureEmbedContexts();
      this.checkOperation(options.signal);
      const chunkSize = Math.ceil(texts.length / contexts.length);
      const batches = contexts.map((context, index) => ({ context, texts: texts.slice(index * chunkSize, (index + 1) * chunkSize) }));
      const results = await this.settleAll(batches.map(async batch => {
        const embeddings: (EmbeddingResult | null)[] = [];
        for (const text of batch.texts) {
          this.checkOperation(options.signal);
          try {
            const { text: safeText, truncated } = await this.truncateToContextSize(text);
            this.checkOperation(options.signal);
            if (truncated) console.warn(`⚠ Batch text truncated to fit embedding context (${this.embedModel?.trainContextSize} tokens)`);
            const embedding = await batch.context.getEmbeddingFor(safeText);
            this.checkOperation(options.signal);
            embeddings.push({ embedding: Array.from(embedding.vector), model: this.embedModelUri });
          } catch (error) {
            this.checkOperation(options.signal);
            console.error("Embedding error for text:", error);
            embeddings.push(null);
          }
        }
        return embeddings;
      }));
      return results.flat();
    } catch (error) {
      this.checkOperation(options.signal);
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  /** Drain every native worker even if another one fails or cancels first. */
  private async settleAll<T>(promises: Promise<T>[]): Promise<T[]> {
    const settled = await Promise.allSettled(promises);
    const failed = settled.find((entry): entry is PromiseRejectedResult => entry.status === 'rejected');
    if (failed) throw failed.reason;
    return settled.map(entry => (entry as PromiseFulfilledResult<T>).value);
  }

  // ==========================================================================
  // MLX Embedding Backend
  // ==========================================================================

  /** Single-flight owner-scoped negotiation. A canceled waiter never cancels other waiters. */
  private async _ensureMlxClient(signal?: AbortSignal): Promise<void> {
    this.checkOperation(signal);
    if (this.mlxClient && this.mlxDescriptor) return;
    if (!this.mlxClientLoad) {
      const controller = new AbortController();
      const load = { controller, waiters: 0, promise: Promise.resolve() };
      const linked = linkSignals([controller.signal, lifecycleFor(this).signal]);
      const work = (async () => {
        const { MlxEmbedClient } = await import('./mlx.js');
        let client: MlxEmbedClient | null = null;
        try {
          this.checkOperation(linked.signal);
          const remote = await this.freshMlxDescriptor(linked.signal);
          if (!remote) throw new Error('MLX daemon did not provide a valid embedding descriptor');
          const expected = this.withOutputDimensions(remote);
          // Negotiate the daemon's native contract. Projected responses are
          // separately pinned to `expected` at the actual embedding request.
          client = new MlxEmbedClient({ ...this.mlxConfig(), expectedDescriptor: remote });
          await client.probeDims(linked.signal);
          this.checkOperation(linked.signal);
          this.mlxDims = expected.outputDimensions;
          this.mlxName = expected.model;
          this.mlxDescriptor = expected;
          this.mlxClient = client;
          this.mlxWarm = true;
        } catch (error) {
          await this.disposeResource(client);
          this.checkOperation(linked.signal);
          throw new Error(
            `MLX embedding server unreachable at ${this.mlxUrlOverride}. ` +
            `Start server with 'python scripts/mlx_embed_server.py --preload' or configure QMD_EMBED_BACKEND=gguf. (${error instanceof Error ? error.message : String(error)})`
          );
        } finally { linked.dispose(); }
      })();
      load.promise = lifecycleFor(this).track(work);
      this.mlxClientLoad = load;
      const clear = () => { if (this.mlxClientLoad === load) this.mlxClientLoad = null; };
      work.then(clear, clear);
    }
    const load = this.mlxClientLoad;
    load.waiters++;
    try { await abortable(load.promise, signal ?? lifecycleFor(this).signal); }
    finally {
      load.waiters--;
      if (load.waiters === 0 && this.mlxClientLoad === load) {
        this.mlxClientLoad = null;
        load.controller.abort(new DOMException('MLX initialization has no active callers', 'AbortError'));
      }
    }
    this.checkOperation(signal);
  }

  /**
   * Batch token counts via the MLX daemon. Returns null unless the MLX embed
   * path is active and the daemon answers — callers fall back to local
   * (GGUF or estimated) counts. Batched in one round-trip.
   */
  async mlxTokenize(texts: string[], options: { signal?: AbortSignal } = {}): Promise<number[] | null> {
    return this.withOperation(options.signal, async signal => {
      if (this.embedBackend !== 'mlx' || texts.length === 0) return null;
      try {
        const { tokenizeWithMlx } = await import('./mlx.js');
        this.checkOperation(signal);
        const result = await tokenizeWithMlx(texts, this.mlxConfig(signal));
        this.checkOperation(signal);
        return result;
      } catch (error) {
        this.checkOperation(signal);
        return null;
      }
    }, this.embedBackend === 'mlx');
  }

  private async _embedMlx(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    await this._ensureMlxClient(options.signal);
    this.checkOperation(options.signal);
    const mlxOptions = { ...options, dims: this.embedDims ?? this.mlxDims ?? undefined, expectedDescriptor: this.mlxDescriptor ?? undefined };
    const result = this.embedDims === undefined
      ? await this.mlxClient!.embed(text, mlxOptions)
      : await (await import('./mlx.js')).embedWithMlx(text, { ...this.mlxConfig(options.signal), expectedDescriptor: this.mlxDescriptor! }, mlxOptions);
    this.checkOperation(options.signal);
    if (!result) throw new Error('MLX embedding failed; cross-space GGUF fallback is disabled');
    return result;
  }

  /**
   * MLX rerank fast-path. Mirrors the GGUF contract: dedupe texts, score,
   * reassemble in original document order, sort by score descending.
   * Returns null when the daemon is unreachable or misconfigured; the
   * caller decides whether to fall back to GGUF.
   */
  private async _rerankMlx(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult | null> {
    try {
      const before = await this.freshMlxStageContract(options.signal ?? lifecycleFor(this).signal);
      this.checkOperation(options.signal);
      const { rerankWithMlx } = await import('./mlx.js');

      // Deduplicate identical texts (same discipline as the GGUF path).
      const textToDocs = new Map<string, { file: string; index: number }[]>();
      documents.forEach((doc, index) => {
        const existing = textToDocs.get(doc.text);
        if (existing) existing.push({ file: doc.file, index });
        else textToDocs.set(doc.text, [{ file: doc.file, index }]);
      });
      const uniqueTexts = Array.from(textToDocs.keys());
      if (uniqueTexts.length === 0) return { results: [], model: stageCacheIdentity(before, 'rerank') ?? 'mlx' };

      this.checkOperation(options.signal);
      const res = await rerankWithMlx(query, uniqueTexts, this.mlxConfig(options.signal));
      this.checkOperation(options.signal);
      if (!res) {
        // Unreachable or invalid — decide fallback by config.
        if (this.mlxRerankFallback) return null;
        throw new Error("MLX rerank failed and GGUF rerank fallback is disabled (QMD_MLX_RERANK_FALLBACK=0)");
      }

      const ranked = uniqueTexts
        .map((text, i) => ({ text, score: res.scores[i]! }))
        .sort((a, b) => b.score - a.score);

      const results: RerankDocumentResult[] = [];
      for (const item of ranked) {
        for (const docInfo of textToDocs.get(item.text) ?? []) {
          results.push({ file: docInfo.file, score: item.score, index: docInfo.index });
        }
      }
      const after = await this.freshMlxStageContract(options.signal ?? lifecycleFor(this).signal);
      return { results, model: actualStageIdentity('rerank', res, before, after) };
    } catch (err) {
      this.checkOperation(options.signal);
      if (this.mlxRerankFallback) {
        console.warn(`MLX rerank error, falling back to GGUF: ${err instanceof Error ? err.message : String(err)}`);
        return null;
      }
      throw err;
    }
  }

  /**
   * MLX query-expansion fast-path. Returns Queryable[] on success, null to signal GGUF fallback
   * when fallback is enabled, or throws if fallback is disabled.
   */
  private async _expandQueryMlx(query: string, options: ExpandOptions = {}): Promise<ExpansionResult | null> {
    const signal = options.signal ?? lifecycleFor(this).signal;
    try {
      const before = await this.freshMlxStageContract(signal);
      if (!before) {
        throw new Error(`MLX embedding server unreachable at ${this.mlxUrlOverride}. Start server with 'python scripts/mlx_embed_server.py --preload' or configure QMD_EMBED_BACKEND=gguf.`);
      }
      const { buildMlxExpansionPrompt, parseExpansionOutput } = await import('./expansion/protocol.js');
      this.checkOperation(signal);
      const prompt = buildMlxExpansionPrompt(query, { intent: options.intent });
      const result = await generateMlxWithIdentity(prompt, this.mlxConfig(signal));
      this.checkOperation(signal);
      if (!result?.text) {
        if (this.mlxExpandFallback) return null;
        throw new Error("MLX query expansion failed and GGUF fallback is disabled (QMD_MLX_EXPAND_FALLBACK=0)");
      }
      const parsed = parseExpansionOutput(result.text, { query, includeLexical: options.includeLexical ?? true });
      if (parsed.usable && parsed.queryables.length > 0) {
        const after = await this.freshMlxStageContract(signal);
        return { queries: parsed.queryables, model: actualStageIdentity('expand', result, before, after) };
      }
      if (this.mlxExpandFallback) {
        console.warn(`MLX expansion produced 0 usable typed items (${parsed.qualityMessage}), falling back to GGUF`);
        return null;
      }
      throw new Error(`MLX query expansion produced no valid typed expansions (${parsed.qualityMessage}) and GGUF fallback is disabled (QMD_MLX_EXPAND_FALLBACK=0)`);
    } catch (error) {
      this.checkOperation(signal);
      if (this.mlxExpandFallback) {
        console.warn(`MLX expansion error, falling back to GGUF: ${error instanceof Error ? error.message : String(error)}`);
        return null;
      }
      throw error;
    }
  }

  private async _embedBatchMlx(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    await this._ensureMlxClient(options.signal);
    this.checkOperation(options.signal);
    const mlxOptions = { ...options, dims: this.embedDims ?? this.mlxDims ?? undefined, expectedDescriptor: this.mlxDescriptor ?? undefined };
    const concurrent = texts.length > 64 && this.mlxConcurrency > 1;
    let results: (EmbeddingResult | null)[];
    if (this.embedDims !== undefined) {
      const transport = await import('./mlx.js');
      this.checkOperation(options.signal);
      const config = { ...this.mlxConfig(options.signal), expectedDescriptor: this.mlxDescriptor! };
      results = concurrent
        ? await transport.embedBatchConcurrent(texts, config, mlxOptions, 32)
        : await transport.embedBatchWithMlx(texts, config, mlxOptions);
    } else {
      results = concurrent
        ? await this.mlxClient!.embedBatchConcurrent(texts, mlxOptions, 32)
        : await this.mlxClient!.embedBatch(texts, mlxOptions);
    }
    this.checkOperation(options.signal);
    if (results.length !== texts.length || results.some(result => !result)) {
      throw new Error('MLX batch embedding failed; cross-space GGUF fallback is disabled');
    }
    return results;
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    return this.withOperation(options.signal, async signal => {
      if (this.strictMlx) {
        const before = await this.freshMlxStageContract(signal);
        const result = await generateMlxWithIdentity(prompt, this.mlxConfig(signal), { ...options, maxTokens: options.maxTokens ?? 150 });
        this.checkOperation(signal);
        if (!result) throw new Error('MLX generation failed; GGUF fallback is disabled');
        const after = await this.freshMlxStageContract(signal);
        return { text: result.text, model: actualStageIdentity('expand', result, before, after), done: true };
      }
      if (this._ciMode) throw new Error("LLM operations are disabled in CI (set CI=true)");
      const model = await this.ensureGenerateModel();
      this.checkOperation(signal);
      const context = this.rememberResourceParent(await model.createContext({ createSignal: signal }), model);
      try {
        this.checkOperation(signal);
        const sequence = context.getSequence();
        const session = new LlamaChatSession({ contextSequence: sequence });
        try {
          let text = '';
          await session.prompt(prompt, {
            signal,
            maxTokens: options.maxTokens ?? 150,
            temperature: options.temperature ?? 0.7,
            topK: 20,
            topP: 0.8,
            onTextChunk: chunk => { if (!signal.aborted) text += chunk; },
          });
          this.checkOperation(signal);
          return { text, model: this.generateModelUri, done: true };
        } finally {
          await session.dispose();
        }
      } finally {
        await this.disposeResource(context);
      }
    }, this.strictMlx);
  }

  async modelExists(modelUri: string): Promise<ModelInfo> {
    this.checkOperation();
    // For HuggingFace URIs, we assume they exist
    // For local paths, check if file exists
    if (modelUri.startsWith("hf:")) {
      return { name: modelUri, exists: true };
    }

    const exists = existsSync(modelUri);
    return {
      name: modelUri,
      exists,
      path: exists ? modelUri : undefined,
    };
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, options: ExpandOptions = {}): Promise<Queryable[]> {
    return (await this.expandQueryWithIdentity(query, options)).queries;
  }

  async expandQueryWithIdentity(query: string, options: ExpandOptions = {}): Promise<ExpansionResult> {
    return this.withOperation(options.signal, async signal => {
      const scoped = { ...options, signal };
      if (this.mlxExpandEnabled()) {
        const result = await this._expandQueryMlx(query, scoped);
        if (result) return result;
      }
      this.checkOperation(signal);
      return this.expandQueryGguf(query, scoped);
    }, this.mlxExpandEnabled());
  }

  private async expandQueryGguf(query: string, options: ExpandOptions): Promise<ExpansionResult> {
    if (this._ciMode) throw new Error("LLM operations are disabled in CI (set CI=true)");
    const llama = await this.ensureLlama();
    this.checkOperation(options.signal);
    const model = await this.ensureGenerateModel();
    this.checkOperation(options.signal);

    const includeLexical = options.includeLexical ?? true;


    const grammar = await llama.createGrammar({
      grammar: `
        root ::= line+
        line ::= type ": " content "\\n"
        type ::= "lex" | "vec" | "hyde"
        content ::= [^\\n]+
      `
    });

    const intent = options.intent;
    const prompt = intent
      ? `/no_think Expand this search query: ${query}\nQuery intent: ${intent}`
      : `/no_think Expand this search query: ${query}`;

    // Create a bounded context for expansion to prevent large default VRAM allocations.
    this.checkOperation(options.signal);
    const genContext = this.rememberResourceParent(await model.createContext({
      contextSize: this.expandContextSize,
      createSignal: options.signal,
    }), model);
    let session: LlamaChatSession | null = null;
    try {
      this.checkOperation(options.signal);
      const sequence = genContext.getSequence();
      session = new LlamaChatSession({ contextSequence: sequence });
      // Qwen3 recommended settings for non-thinking mode:
      // temp=0.7, topP=0.8, topK=20, presence_penalty for repetition
      // DO NOT use greedy decoding (temp=0) - causes infinite loops
      const result = await session.prompt(prompt, {
        signal: options.signal,
        grammar,
        maxTokens: 600,
        temperature: 0.7,
        topK: 20,
        topP: 0.8,
        repeatPenalty: {
          lastTokens: 64,
          presencePenalty: 0.5,
        },
      });

      this.checkOperation(options.signal);
      const lines = result.trim().split("\n");
      const queryLower = query.toLowerCase();
      const queryTerms = queryLower.replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);

      const hasQueryTerm = (text: string): boolean => {
        const lower = text.toLowerCase();
        if (queryTerms.length === 0) return true;
        return queryTerms.some(term => lower.includes(term));
      };

      const queryables: Queryable[] = lines.map(line => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!hasQueryTerm(text)) return null;
        return { type: type as QueryType, text };
      }).filter((q): q is Queryable => q !== null);

      // Filter out lex entries if not requested
      const filtered = includeLexical ? queryables : queryables.filter(q => q.type !== 'lex');
      if (filtered.length > 0) return { queries: filtered, model: this.generateModelUri };

      const fallback: Queryable[] = [
        { type: 'hyde', text: `Information about ${query}` },
        { type: 'lex', text: query },
        { type: 'vec', text: query },
      ];
      return { queries: includeLexical ? fallback : fallback.filter(q => q.type !== 'lex'), model: 'fallback:query' };
    } catch (error) {
      this.checkOperation(options.signal);
      console.error("Structured query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return { queries: fallback, model: 'fallback:query' };
    } finally {
      try { await session?.dispose(); }
      finally { await this.disposeResource(genContext); }
    }
  }

  // Qwen3 reranker chat template overhead (system prompt, tags, separators).
  // Measured at ~350 tokens on real queries; use 512 as a safe upper bound so
  // the truncation budget never lets a document slip past the context limit.
  private static readonly RERANK_TEMPLATE_OVERHEAD = 512;
  private static readonly RERANK_TARGET_DOCS_PER_CONTEXT = 10;

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    return this.withOperation(options.signal, signal => this.rerankInternal(query, documents, { ...options, signal }), this.mlxRerankEnabled());
  }

  private async rerankInternal(query: string, documents: RerankDocument[], options: RerankOptions): Promise<RerankResult> {
    if (this.mlxRerankEnabled()) {
      const mlxResult = await this._rerankMlx(query, documents, options);
      if (mlxResult) return mlxResult;
    }
    this.checkOperation(options.signal);
    if (documents.length === 0) return { results: [], model: this.rerankModelUri };
    if (this._ciMode) throw new Error("LLM operations are disabled in CI (set CI=true)");
    const contexts = await this.ensureRerankContexts();
    this.checkOperation(options.signal);
    const model = await this.ensureRerankModel();
    this.checkOperation(options.signal);

    // Truncate documents that would exceed the rerank context size.
    // Budget = contextSize - template overhead - query tokens
    const queryTokens = model.tokenize(query).length;
    const maxDocTokens = Math.max(1, LlamaCpp.RERANK_CONTEXT_SIZE - LlamaCpp.RERANK_TEMPLATE_OVERHEAD - queryTokens);
    const truncationCache = new Map<string, string>();

    const truncatedDocs = documents.map((doc) => {
      const cached = truncationCache.get(doc.text);
      if (cached !== undefined) {
        return cached === doc.text ? doc : { ...doc, text: cached };
      }

      const tokens = model.tokenize(doc.text);
      const truncatedText = tokens.length <= maxDocTokens
        ? doc.text
        : model.detokenize(tokens.slice(0, maxDocTokens));
      truncationCache.set(doc.text, truncatedText);

      if (truncatedText === doc.text) return doc;
      return { ...doc, text: truncatedText };
    });

    // Deduplicate identical effective texts before scoring.
    // This avoids redundant work for repeated chunks and fixes collisions where
    // multiple docs map to the same chunk text.
    const textToDocs = new Map<string, { file: string; index: number }[]>();
    truncatedDocs.forEach((doc, index) => {
      const existing = textToDocs.get(doc.text);
      if (existing) {
        existing.push({ file: doc.file, index });
      } else {
        textToDocs.set(doc.text, [{ file: doc.file, index }]);
      }
    });

    // Extract just the text for ranking
    const texts = Array.from(textToDocs.keys());

    // Split documents across contexts for parallel evaluation.
    // Each context has its own sequence with a lock, so parallelism comes
    // from multiple contexts evaluating different chunks simultaneously.
    const activeContextCount = Math.max(
      1,
      Math.min(
        contexts.length,
        Math.ceil(texts.length / LlamaCpp.RERANK_TARGET_DOCS_PER_CONTEXT)
      )
    );
    const activeContexts = contexts.slice(0, activeContextCount);
    const chunkSize = Math.ceil(texts.length / activeContexts.length);
    const chunks = Array.from({ length: activeContexts.length }, (_, i) =>
      texts.slice(i * chunkSize, (i + 1) * chunkSize)
    ).filter(chunk => chunk.length > 0);

    this.checkOperation(options.signal);
    const allScores = await this.settleAll(
      chunks.map((chunk, i) => activeContexts[i]!.rankAll(query, chunk))
    );
    this.checkOperation(options.signal);

    // Reassemble scores in original order and sort
    const flatScores = allScores.flat();
    const ranked = texts
      .map((text, i) => ({ document: text, score: flatScores[i]! }))
      .sort((a, b) => b.score - a.score);

    // Map back to our result format.
    const results: RerankDocumentResult[] = [];
    for (const item of ranked) {
      const docInfos = textToDocs.get(item.document) ?? [];
      for (const docInfo of docInfos) {
        results.push({
          file: docInfo.file,
          score: item.score,
          index: docInfo.index,
        });
      }
    }

    return {
      results,
      model: this.rerankModelUri,
    };
  }

  /**
   * Get device/GPU info for status display.
   * Initializes llama if not already done.
   */
  async getDeviceInfo(options: { signal?: AbortSignal } = {}): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    return this.withOperation(options.signal, async signal => {
    const llama = await this.ensureLlama();
    this.checkOperation(signal);
    const gpuDevices = await llama.getGpuDeviceNames();
    let vram: { total: number; used: number; free: number } | undefined;
    if (llama.gpu) {
      try {
        const state = await llama.getVramState();
        vram = { total: state.total, used: state.used, free: state.free };
      } catch { /* no vram info */ }
    }
    return {
      gpu: llama.gpu,
      gpuOffloading: llama.supportsGpuOffloading,
      gpuDevices,
      vram,
      cpuCores: llama.cpuMathCores,
    };
    });
  }

  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    const lifecycle = lifecycleFor(this);
    lifecycle.beginClose();
    this.clearInactivityTimer();
    this.disposePromise = (async () => {
      try {
        if (this.idleUnloadPromise) {
          // Its failures remain owned below; a successful retry must not rethrow stale errors.
          await this.idleUnloadPromise.catch(() => {});
        }
        await lifecycle.drain();
        // A fixed snapshot gives prior failures exactly one retry. New close-time failures
        // are retained and reported, not retried in a loop or by repeated dispose() calls.
        await this.retryFailedResourcesAtClose();
        await this.releaseNativeResources(true).catch(() => {});
        await Promise.allSettled([this.mlxClient, this.llama].map(resource => this.disposeResource(resource)));
      } finally {
        this.mlxDescriptor = null;
        this.mlxDims = null;
        this.mlxName = null;
        lifecycle.finishClose();
      }
      if (this.failedResourceDisposals.size) throw new AggregateError(
        [...new Set([...this.failedResourceDisposals.values()].map(failure => failure.error))],
        `LLM disposal failed: ${this.failedResourceDisposals.size} resource cleanup failure(s); failed resources and their owners retained after bounded cleanup; restart the process before loading another LLM`,
      );
    })();
    return this.disposePromise;
  }
}

// =============================================================================
// Session Management Layer
// =============================================================================

/**
 * Error thrown when an operation is attempted on a released or aborted session.
 */
export class SessionReleasedError extends Error {
  constructor(message = "LLM session has been released or aborted") {
    super(message);
    this.name = "SessionReleasedError";
  }
}

/**
 * Scoped LLM session with automatic lifecycle management.
 * Wraps LlamaCpp methods with operation tracking and abort handling.
 */
class LLMSession implements ILLMSession {
  private released = false;
  private abortController = new AbortController();
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
  private releaseLease: () => void;
  private unlink: () => void;

  constructor(private llm: LlamaCpp, options: LLMSessionOptions = {}) {
    const maxDuration = options.maxDuration ?? 120 * 60 * 1000;
    if (!Number.isSafeInteger(maxDuration) || maxDuration < 0 || maxDuration > 2_147_483_647) {
      throw new Error('maxDuration must be an integer between 0 and 2147483647 milliseconds');
    }
    const lifecycle = lifecycleFor(llm);
    this.releaseLease = lifecycle.acquireSession();
    const linked = linkSignals([options.signal, lifecycle.signal]);
    const onAbort = () => this.abortController.abort(linked.signal.reason);
    linked.signal.addEventListener('abort', onAbort, { once: true });
    if (linked.signal.aborted) onAbort();
    this.unlink = () => { linked.signal.removeEventListener('abort', onAbort); linked.dispose(); };
    if (maxDuration > 0 && !this.signal.aborted) {
      this.maxDurationTimer = setTimeout(() => {
        this.abortController.abort(new SessionReleasedError(`Session "${options.name ?? 'unnamed'}" exceeded max duration of ${maxDuration}ms`));
      }, maxDuration);
      this.maxDurationTimer.unref?.();
    }
  }

  get isValid(): boolean { return !this.released && !this.signal.aborted; }
  get signal(): AbortSignal { return this.abortController.signal; }

  assertValid(): void {
    if (!this.isValid) throw new SessionReleasedError(this.signal.reason?.message);
  }

  release(): void {
    if (this.released) return;
    this.released = true;
    if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
    this.maxDurationTimer = null;
    this.unlink();
    this.abortController.abort(new SessionReleasedError('Session released'));
    this.releaseLease();
  }

  private async withOperation<T>(signal: AbortSignal | undefined, fn: (signal: AbortSignal) => Promise<T>): Promise<T> {
    this.assertValid();
    throwIfAborted(signal);
    const lifecycle = lifecycleFor(this.llm);
    const release = lifecycle.acquireOperation();
    const linked = linkSignals([signal, this.signal]);
    const work = (async () => {
      try {
        this.assertValid();
        const result = await fn(linked.signal);
        this.assertValid();
        throwIfAborted(linked.signal);
        return result;
      } finally {
        linked.dispose();
        release();
      }
    })();
    try { return await abortable(lifecycle.track(work), linked.signal); }
    catch (error) { this.assertValid(); throw error; }
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    return this.withOperation(options.signal, signal => this.llm.embed(text, { ...options, signal }));
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    return this.withOperation(options.signal, signal => this.llm.embedBatch(texts, { ...options, signal }));
  }

  async expandQuery(query: string, options: ExpandOptions = {}): Promise<Queryable[]> {
    return this.withOperation(options.signal, signal => this.llm.expandQuery(query, { ...options, signal }));
  }

  async expandQueryWithIdentity(query: string, options: ExpandOptions = {}): Promise<ExpansionResult> {
    return this.withOperation(options.signal, async signal => {
      if (this.llm.expandQueryWithIdentity) return this.llm.expandQueryWithIdentity(query, { ...options, signal });
      return { queries: await this.llm.expandQuery(query, { ...options, signal }), model: 'unknown' };
    });
  }

  async getCacheIdentity(op: 'rerank' | 'expand', options: { signal?: AbortSignal } = {}): Promise<string | null> {
    return this.withOperation(options.signal, signal => this.llm.getCacheIdentity?.(op, { signal }) ?? Promise.resolve(null));
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    return this.withOperation(options.signal, signal => this.llm.rerank(query, documents, { ...options, signal }));
  }

  async getDescriptor(options: SignalOptions = {}): Promise<EmbeddingDescriptor | null> {
    return this.withOperation(optionSignal(options), signal => this.llm.getDescriptor?.({ signal }) ?? Promise.resolve(null));
  }
}

/**
 * Execute a function with a scoped LLM session.
 * The session provides lifecycle guarantees - resources won't be disposed mid-operation.
 *
 * @example
 * ```typescript
 * await withLLMSession(async (session) => {
 *   const expanded = await session.expandQuery(query);
 *   const embeddings = await session.embedBatch(texts);
 *   const reranked = await session.rerank(query, docs);
 *   return reranked;
 * }, { maxDuration: 10 * 60 * 1000, name: 'querySearch' });
 * ```
 */
export async function withLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  return withLLMSessionForLlm(getDefaultLlamaCpp(), fn, options);
}

/**
 * Execute a function with a scoped LLM session using a specific LlamaCpp instance.
 * Unlike withLLMSession, this does not use the global singleton.
 */
export async function withLLMSessionForLlm<T>(
  llm: LlamaCpp,
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const session = new LLMSession(llm, options);

  try {
    session.assertValid();
    const result = await fn(session);
    session.assertValid();
    return result;
  } finally {
    session.release();
  }
}

/**
 * Check if idle unload is safe (no active sessions or operations).
 * Used internally by LlamaCpp idle timer.
 */
export function canUnloadLLM(llm: LlamaCpp | null = defaultLlamaCpp): boolean {
  return llm ? lifecycleFor(llm).canUnload : true;
}

// =============================================================================
// Singleton for default LlamaCpp instance
// =============================================================================

let defaultLlamaCpp: LlamaCpp | null = null;

/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export function getDefaultLlamaCpp(): LlamaCpp {
  if (!defaultLlamaCpp) {
    defaultLlamaCpp = new LlamaCpp();
  }
  return defaultLlamaCpp;
}

/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm: LlamaCpp | null): void {
  defaultLlamaCpp = llm;
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  const llm = defaultLlamaCpp;
  if (llm) {
    // Keep a failed owner reachable and fenced; replacing it could allocate a
    // second native pool while its unreleased resources still exist.
    await llm.dispose();
    if (defaultLlamaCpp === llm) defaultLlamaCpp = null;
  }
}
