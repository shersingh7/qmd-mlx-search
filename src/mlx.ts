/** Bounded, cancellable HTTP bridge to the Python MLX daemon. */
import type { EmbeddingResult, EmbedOptions } from "./llm.js";
import { validateMlxServingResult, type MlxServingResult } from "./llm/mlx-identity.js";
import {
  MAX_EMBEDDING_DIMENSIONS, assertEmbeddingCompatibility, validateEmbeddingDescriptor,
  type EmbeddingDescriptor,
} from "./embedding/contract.js";
import {
  DEFAULT_MLX_URL, DEFAULT_MLX_TIMEOUT_MS, DEFAULT_MLX_CONCURRENCY,
  MAX_MLX_TIMEOUT_MS, MAX_MLX_CONCURRENCY, MAX_MLX_BATCH_SIZE, positiveInteger, validateMlxUrl,
} from "./embedding/config.js";
import {
  MAX_EMBEDDING_VALUES, EmbeddingProtocolError, decodeBinaryEmbeddings,
  decodeEmbeddingDescriptorHeader, validateJsonEmbedResponse,
} from "./embedding/protocol.js";

export interface MlxEmbedConfig {
  readonly url?: string;
  /** Complete API-call budget: negotiation, queueing, headers and bounded bodies. */
  readonly timeoutMs?: number;
  /** When omitted, binary is selected for batches >= 16. Explicit false is retained. */
  readonly binary?: boolean;
  readonly concurrency?: number;
  /** Response identity is mandatory when supplied; the production client always pins it. */
  readonly expectedDescriptor?: EmbeddingDescriptor;
  readonly signal?: AbortSignal;
}

type MlxEmbedOptions = EmbedOptions & { dims?: number; signal?: AbortSignal };
type MlxGenerateOptions = { maxTokens?: number; temperature?: number; signal?: AbortSignal };
type TransportConfig = Readonly<MlxEmbedConfig & { url: string; timeoutMs: number; concurrency: number }>;

export interface MlxHealth {
  status: string;
  state?: string;
  model?: string;
  dims?: number;
  ready: boolean;
  descriptor?: EmbeddingDescriptor;
  rerank_model?: string | null;
  generate_model?: string | null;
  error?: string | null;
}
export interface MlxMemory { active_mb: number; peak_mb: number; model_mb: number; }
export interface MlxStats { total_requests: number; avg_ms: number; compiled_shapes: number; uptime_sec: number; }
export type MlxRerankResult = MlxServingResult & { scores: number[]; model: string };

const METADATA_BYTES = 131_072;
const ERROR_BYTES = 4096;
const LOG_CHARS = 1024;
const MAX_JSON_BYTES = 64 * 1024 * 1024;
const MAX_REQUEST_BYTES = 16 * 1024 * 1024;
const BINARY_THRESHOLD = 16;

function validateSignal(signal?: AbortSignal): void {
  if (signal !== undefined && (typeof signal?.aborted !== "boolean" || typeof signal?.addEventListener !== "function" || typeof signal?.removeEventListener !== "function")) {
    throw new TypeError("signal must be an AbortSignal");
  }
}

function normalizeConfig(config: MlxEmbedConfig = {}): TransportConfig {
  const expectedDescriptor = config.expectedDescriptor === undefined ? undefined : validateEmbeddingDescriptor(config.expectedDescriptor);
  if (expectedDescriptor && expectedDescriptor.backend !== "mlx") throw new TypeError("Expected MLX embedding descriptor");
  if (config.binary !== undefined && typeof config.binary !== "boolean") throw new TypeError("binary must be a boolean");
  validateSignal(config.signal);
  return Object.freeze({
    url: validateMlxUrl(config.url ?? process.env.QMD_MLX_EMBED_URL ?? DEFAULT_MLX_URL),
    timeoutMs: positiveInteger(config.timeoutMs === undefined ? DEFAULT_MLX_TIMEOUT_MS : config.timeoutMs, "timeoutMs", MAX_MLX_TIMEOUT_MS),
    concurrency: positiveInteger(config.concurrency === undefined ? DEFAULT_MLX_CONCURRENCY : config.concurrency, "concurrency", MAX_MLX_CONCURRENCY),
    binary: config.binary, expectedDescriptor, signal: config.signal,
  });
}

function abortError(): DOMException { return new DOMException("MLX operation cancelled", "AbortError"); }
function throwIfCancelled(config: MlxEmbedConfig, signal?: AbortSignal): void {
  if (config.signal?.aborted || signal?.aborted) throw abortError();
}

/** Race explicitly as well as forwarding signal: adapters may ignore fetch aborts. */
function abortable<T>(promise: Promise<T>, signal: AbortSignal, onLateValue?: (value: T) => void): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", onAbort);
      reject(signal.reason ?? abortError());
    };
    if (signal.aborted) onAbort();
    else signal.addEventListener("abort", onAbort, { once: true });
    promise.then((value) => {
      if (settled) { onLateValue?.(value); return; }
      settled = true;
      signal.removeEventListener("abort", onAbort);
      resolve(value);
    }, (error) => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", onAbort);
      reject(error);
    });
  });
}

interface RequestScope { signal: AbortSignal; deadline: number; }
function checkScope(scope: RequestScope): void {
  if (scope.signal.aborted) throw scope.signal.reason;
  if (performance.now() >= scope.deadline) throw new DOMException("MLX request deadline exceeded", "TimeoutError");
}
async function withScope<T>(config: TransportConfig, signal: AbortSignal | undefined, metadata: boolean, run: (scope: RequestScope) => Promise<T>): Promise<T> {
  validateSignal(signal);
  throwIfCancelled(config, signal);
  const controller = new AbortController();
  const signals = [...new Set([config.signal, signal].filter((value): value is AbortSignal => value !== undefined))];
  const onAbort = () => controller.abort(abortError());
  for (const source of signals) source.addEventListener("abort", onAbort, { once: true });
  const duration = metadata ? Math.min(config.timeoutMs, 3000) : config.timeoutMs;
  const deadline = performance.now() + duration;
  const timer = setTimeout(() => controller.abort(new DOMException("MLX request deadline exceeded", "TimeoutError")),
    duration);
  timer.unref?.();
  try {
    // Recheck after registering to close the check/listen race.
    if (signals.some((source) => source.aborted)) onAbort();
    const result = await run({ signal: controller.signal, deadline });
    // Synchronous parsing cannot be preempted by a timer; never publish an
    // overdue result merely because its timeout callback has not run yet.
    if (controller.signal.aborted) throw controller.signal.reason;
    if (performance.now() >= deadline) {
      controller.abort(new DOMException("MLX request deadline exceeded", "TimeoutError"));
      throw controller.signal.reason;
    }
    return result;
  } finally {
    clearTimeout(timer);
    for (const source of signals) source.removeEventListener("abort", onAbort);
  }
}

function cancelBody(response: Response): void {
  if (!response.body || response.body.locked) return;
  try { void response.body.cancel().catch(() => {}); } catch { /* best effort, never await a non-cooperative cancel */ }
}

async function readBody(response: Response, scope: RequestScope, cap: number, prefixOnly = false): Promise<Uint8Array> {
  const length = response.headers.get("content-length");
  if (!prefixOnly && length !== null && /^\d+$/.test(length) && Number(length) > cap) {
    cancelBody(response);
    throw new EmbeddingProtocolError(`MLX response exceeds ${cap}-byte limit`);
  }
  if (!response.body) return new Uint8Array(0);
  const reader = response.body.getReader();
  let storage = new Uint8Array(Math.min(cap, 8192));
  let total = 0;
  let done = false;
  let emptyChunks = 0;
  try {
    while (true) {
      const next = await abortable(reader.read(), scope.signal);
      if (next.done) { done = true; break; }
      const bytes = next.value;
      if (bytes.byteLength === 0) {
        if (++emptyChunks > 64) throw new EmbeddingProtocolError("MLX body made no byte progress");
        continue;
      }
      emptyChunks = 0;
      if (total + bytes.byteLength > cap && !prefixOnly) {
        throw new EmbeddingProtocolError(`MLX response exceeds ${cap}-byte limit`);
      }
      const take = Math.min(bytes.byteLength, cap - total);
      // One object per one-byte chunk would not be a memory byte cap.
      if (total + take > storage.byteLength) {
        const grown = new Uint8Array(Math.min(cap, Math.max(storage.byteLength * 2, total + take)));
        grown.set(storage);
        storage = grown;
      }
      storage.set(bytes.subarray(0, take), total);
      total += take;
      if (prefixOnly && total === cap) break;
    }
    return storage.slice(0, total);
  } finally {
    if (!done) {
      try { void reader.cancel().catch(() => {}); } catch { /* cancellation cannot extend the deadline */ }
    }
    try { reader.releaseLock(); } catch { /* a non-cooperative reader may still have a pending read */ }
  }
}

class HttpError extends Error {
  constructor(readonly status: number, detail = "") { super(`HTTP ${status}${detail ? `: ${detail}` : ""}`); }
}

/** Count encoded JSON string bytes before allocating the serialized body. */
function requestByteLength(value: unknown): number {
  if (typeof value === "string") {
    let size = Buffer.byteLength(value, "utf8") + 2;
    for (let i = 0; i < value.length; i++) {
      const code = value.charCodeAt(i);
      if (code === 34 || code === 92 || [8, 9, 10, 12, 13].includes(code)) size++;
      else if (code < 32) size += 5;
      else if (code >= 0xd800 && code <= 0xdbff) {
        const low = value.charCodeAt(i + 1);
        if (low >= 0xdc00 && low <= 0xdfff) i++;
        else size += 3; // Lone surrogates become six-byte JSON escapes.
      } else if (code >= 0xdc00 && code <= 0xdfff) size += 3;
      if (size > MAX_REQUEST_BYTES) return size;
    }
    return size;
  }
  if (Array.isArray(value)) {
    let size = 2;
    for (const item of value) { size += requestByteLength(item) + 1; if (size > MAX_REQUEST_BYTES) break; }
    return size;
  }
  if (value && typeof value === "object") {
    let size = 2;
    for (const [key, item] of Object.entries(value)) {
      size += requestByteLength(key) + requestByteLength(item) + 2;
      if (size > MAX_REQUEST_BYTES) break;
    }
    return size;
  }
  return 32; // Conservative overhead for bounded numeric/boolean payload fields.
}

async function requestBytes(path: string, payload: Record<string, unknown> | undefined, config: TransportConfig,
  scope: RequestScope, cap: number, binary = false): Promise<{ bytes: Uint8Array; headers: Headers }> {
  checkScope(scope);
  let body: string | undefined;
  if (payload !== undefined) {
    if (requestByteLength(payload) > MAX_REQUEST_BYTES) throw new RangeError("MLX request exceeds 16 MiB byte limit; use smaller batches");
    body = JSON.stringify(payload);
    if (Buffer.byteLength(body, "utf8") > MAX_REQUEST_BYTES) throw new RangeError("MLX request exceeds 16 MiB byte limit; use smaller batches");
  }
  const response = await abortable(fetch(`${config.url}${path}`, {
    method: payload === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json; charset=utf-8", "Connection": "keep-alive",
      "Accept": binary ? "application/octet-stream" : "application/json" },
    body, signal: scope.signal,
  }), scope.signal, cancelBody);
  try {
    if (!response.ok) {
      if (payload === undefined) throw new HttpError(response.status); // Failed GET bodies are cancelled, not buffered.
      const prefix = await readBody(response, scope, ERROR_BYTES, true);
      throw new HttpError(response.status, new TextDecoder().decode(prefix).slice(0, LOG_CHARS));
    }
    return { bytes: await readBody(response, scope, cap), headers: response.headers };
  } finally { cancelBody(response); }
}

async function requestJson(path: string, payload: Record<string, unknown> | undefined, config: TransportConfig,
  scope: RequestScope, cap: number): Promise<unknown> {
  const { bytes } = await requestBytes(path, payload, config, scope, cap);
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
}

function failure(error: unknown, config: TransportConfig, signal?: AbortSignal, strictProtocol = false): null {
  throwIfCancelled(config, signal);
  if (strictProtocol && error instanceof EmbeddingProtocolError) throw error;
  console.error(`MLX request failed: ${(error instanceof Error ? error.message : String(error)).slice(0, LOG_CHARS)}`);
  return null;
}
function object(data: unknown): Record<string, unknown> {
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new EmbeddingProtocolError("Expected a JSON object");
  return data as Record<string, unknown>;
}
function validateTexts(texts: string[]): void {
  if (!Array.isArray(texts) || texts.length > MAX_MLX_BATCH_SIZE || texts.some((text) => typeof text !== "string")) {
    throw new RangeError(`MLX texts must be an array of at most ${MAX_MLX_BATCH_SIZE} strings; use concurrent batching for larger inputs`);
  }
}
function validateEmbedOptions(options?: MlxEmbedOptions): void {
  validateSignal(options?.signal);
  if (options?.dims !== undefined) positiveInteger(options.dims, "dims", MAX_EMBEDDING_DIMENSIONS);
}

export async function embedWithMlx(text: string, config?: MlxEmbedConfig, options?: MlxEmbedOptions): Promise<EmbeddingResult | null> {
  return (await embedBatchWithMlx([text], config, options))[0] ?? null;
}

export async function embedBatchWithMlx(texts: string[], rawConfig?: MlxEmbedConfig, options?: MlxEmbedOptions): Promise<(EmbeddingResult | null)[]> {
  const config = normalizeConfig(rawConfig);
  validateTexts(texts);
  validateEmbedOptions(options);
  throwIfCancelled(config, options?.signal);
  if (texts.length === 0) return [];
  const expected = config.expectedDescriptor;
  const dims = options?.dims ?? expected?.outputDimensions;
  if (expected && dims !== expected.outputDimensions) throw new EmbeddingProtocolError("Requested dimensions disagree with pinned embedding descriptor");
  const maxValues = Math.min(texts.length * (dims ?? MAX_EMBEDDING_DIMENSIONS), MAX_EMBEDDING_VALUES);
  const binary = config.binary ?? texts.length >= BINARY_THRESHOLD;
  const payload = { texts, ...(dims === undefined ? {} : { dims }), ...(options?.isQuery === undefined ? {} : { is_query: options.isQuery }) };
  try {
    return await withScope(config, options?.signal, false, async (scope) => {
      if (binary) {
        const { bytes, headers } = await requestBytes("/embed-bin", payload, config, scope, 8 + maxValues * 4, true);
        const actual = decodeEmbeddingDescriptorHeader(headers.get("X-QMD-Embedding-Descriptor"), expected);
        const decoded = decodeBinaryEmbeddings(bytes.buffer as ArrayBuffer, { count: texts.length, dims: dims ?? actual?.outputDimensions });
        if (actual && decoded.dims !== actual.outputDimensions) throw new EmbeddingProtocolError("Binary dimensions disagree with serving descriptor");
        return decoded.vectors.map((embedding) => ({ embedding, model: actual?.model ?? "mlx" }));
      }
      const data = await requestJson("/embed", payload, config, scope, Math.min(MAX_JSON_BYTES, METADATA_BYTES + maxValues * 32 + texts.length * 4));
      const valid = validateJsonEmbedResponse(data, { count: texts.length, dims, descriptor: expected });
      return valid.embeddings.map((embedding) => ({ embedding, model: valid.model }));
    });
  } catch (error) {
    failure(error, config, options?.signal, expected !== undefined);
    return texts.map(() => null);
  }
}

/** Bounded workers preserve exact input order/cardinality and stop dequeuing on abort. */
export async function embedBatchConcurrent(texts: string[], rawConfig?: MlxEmbedConfig, options?: MlxEmbedOptions,
  batchSize = 32): Promise<(EmbeddingResult | null)[]> {
  const config = normalizeConfig(rawConfig);
  positiveInteger(batchSize, "batchSize", MAX_MLX_BATCH_SIZE);
  validateEmbedOptions(options);
  if (!Array.isArray(texts) || texts.some((text) => typeof text !== "string")) throw new TypeError("texts must be a string array");
  throwIfCancelled(config, options?.signal);
  const results: (EmbeddingResult | null)[] = Array(texts.length).fill(null);
  if (texts.length === 0) return results;
  try {
    return await withScope(config, options?.signal, false, async (scope) => {
      const controller = new AbortController();
      const onAbort = () => controller.abort(scope.signal.reason);
      scope.signal.addEventListener("abort", onAbort, { once: true });
      const pipelineConfig = { ...config, signal: controller.signal };
      let next = 0;
      let failed = false;
      async function worker(): Promise<void> {
        while (!failed && next < texts.length) {
          checkScope(scope);
          throwIfCancelled(pipelineConfig, options?.signal);
          const start = next;
          next += batchSize;
          const batch = texts.slice(start, start + batchSize);
          try {
            const values = await embedBatchWithMlx(batch, pipelineConfig, options);
            checkScope(scope); // A delayed timer cannot publish an overdue batch.
            if (values.length === batch.length) for (let i = 0; i < values.length; i++) results[start + i] = values[i]!;
          } catch (error) { failed = true; controller.abort(abortError()); throw error; }
        }
      }
      const workers = Array.from({ length: Math.min(config.concurrency, Math.ceil(texts.length / batchSize)) }, worker);
      try {
        await Promise.all(workers);
        return results;
      } catch (error) {
        // Drain cancelled workers; none can dequeue another batch or publish late.
        controller.abort(abortError());
        await Promise.allSettled(workers);
        throw error;
      } finally { scope.signal.removeEventListener("abort", onAbort); }
    });
  } catch (error) {
    failure(error, config, options?.signal, true);
    return results;
  }
}

export async function embedBatchBinaryWithMlx(texts: string[], config?: MlxEmbedConfig, options?: MlxEmbedOptions): Promise<number[][] | null> {
  const results = await embedBatchWithMlx(texts, { ...config, binary: true }, options);
  if (results.some((entry) => entry === null)) return null;
  return results.map((entry) => entry!.embedding);
}

export async function rerankWithMlx(query: string, documents: string[], rawConfig?: MlxEmbedConfig,
  options?: { signal?: AbortSignal }): Promise<MlxRerankResult | null> {
  const config = normalizeConfig(rawConfig);
  validateSignal(options?.signal);
  validateTexts(documents);
  throwIfCancelled(config, options?.signal);
  if (!query.trim() || documents.length === 0) return null;
  try {
    return await withScope(config, options?.signal, false, async (scope) => {
      const data = object(await requestJson("/rerank", { query, documents }, config, scope, METADATA_BYTES + documents.length * 32));
      if (!Array.isArray(data.scores) || data.scores.length !== documents.length || data.scores.some((score) => typeof score !== "number" || !Number.isFinite(score))) {
        throw new EmbeddingProtocolError("Rerank scores must be finite and match document count");
      }
      if (typeof data.model !== "string" || !data.model.trim() || data.model.length > 8192) throw new EmbeddingProtocolError("Missing rerank serving model");
      return { scores: data.scores as number[], ...validateMlxServingResult("rerank", data as MlxServingResult), model: data.model };
    });
  } catch (error) { return failure(error, config, options?.signal); }
}

export async function generateWithMlx(prompt: string, rawConfig?: MlxEmbedConfig, options?: MlxGenerateOptions): Promise<string | null> {
  const config = normalizeConfig(rawConfig);
  validateSignal(options?.signal);
  const maxTokens = positiveInteger(options?.maxTokens === undefined ? 600 : options.maxTokens, "maxTokens", 65_536);
  if (options?.temperature !== undefined && (!Number.isFinite(options.temperature) || options.temperature < 0 || options.temperature > 2)) {
    throw new RangeError("temperature must be finite and between 0 and 2");
  }
  throwIfCancelled(config, options?.signal);
  if (!prompt.trim()) return null;
  const payload = { prompt, max_tokens: maxTokens, ...(options?.temperature === undefined ? {} : { temperature: options.temperature }) };
  try {
    return await withScope(config, options?.signal, false, async (scope) => {
      // Escaped JSON needs up to six bytes per code point; reserve generous token space.
      const data = object(await requestJson("/generate", payload, config, scope, METADATA_BYTES + maxTokens * 256));
      if (typeof data.text !== "string" || Buffer.byteLength(data.text) > maxTokens * 64) throw new EmbeddingProtocolError("Invalid or oversized generation text");
      validateMlxServingResult("expand", data as MlxServingResult);
      return data.text;
    });
  } catch (error) { return failure(error, config, options?.signal); }
}

export async function tokenizeWithMlx(texts: string[], rawConfig?: MlxEmbedConfig, options?: { signal?: AbortSignal }): Promise<number[] | null> {
  const config = normalizeConfig(rawConfig);
  validateSignal(options?.signal);
  validateTexts(texts);
  throwIfCancelled(config, options?.signal);
  if (texts.length === 0) return [];
  // /tokenize may return token IDs as well as counts; bound those by input bytes.
  const inputBytes = texts.reduce((sum, text) => sum + Buffer.byteLength(text), 0);
  try {
    return await withScope(config, options?.signal, false, async (scope) => {
      const data = object(await requestJson("/tokenize", { texts }, config, scope, Math.min(MAX_JSON_BYTES, METADATA_BYTES + inputBytes * 16 + texts.length * 64)));
      if (!Array.isArray(data.counts) || data.counts.length !== texts.length || data.counts.some((count) => !Number.isSafeInteger(count) || count < 0)) {
        throw new EmbeddingProtocolError("Token counts must be nonnegative safe integers matching input count");
      }
      return data.counts as number[];
    });
  } catch (error) { return failure(error, config, options?.signal); }
}

function validateHealth(data: unknown): MlxHealth {
  const value = object(data);
  if (typeof value.ready !== "boolean" || typeof value.status !== "string") throw new EmbeddingProtocolError("Invalid MLX health response");
  // The daemon reports null metadata while starting or running without an
  // embedding stage. Preserve that health report, but never invent an identity.
  const dims = value.dims == null ? undefined : positiveInteger(value.dims, "health dims", MAX_EMBEDDING_DIMENSIONS);
  const descriptor = value.descriptor == null ? undefined : validateEmbeddingDescriptor(value.descriptor);
  if (descriptor && descriptor.backend !== "mlx") throw new EmbeddingProtocolError("Health descriptor backend must be mlx");
  if (value.model != null && typeof value.model !== "string") throw new EmbeddingProtocolError("Invalid MLX health model");
  return { ...value, model: value.model ?? undefined, dims, descriptor } as unknown as MlxHealth;
}

async function metadata<T>(path: string, rawConfig: MlxEmbedConfig | undefined, validate: (data: unknown) => T): Promise<T | null> {
  const config = normalizeConfig(rawConfig);
  throwIfCancelled(config);
  try { return await withScope(config, undefined, true, async (scope) => validate(await requestJson(path, undefined, config, scope, METADATA_BYTES))); }
  catch (error) { throwIfCancelled(config); return null; }
}
export async function mlxHealth(config?: MlxEmbedConfig): Promise<MlxHealth | null> { return metadata("/health", config, validateHealth); }
export async function mlxMemory(config?: MlxEmbedConfig): Promise<MlxMemory | null> {
  return metadata("/memory", config, (data) => {
    const value = object(data);
    for (const key of ["active_mb", "peak_mb", "model_mb"]) if (typeof value[key] !== "number" || !Number.isFinite(value[key]) || (value[key] as number) < 0) throw new EmbeddingProtocolError("Invalid MLX memory response");
    return value as unknown as MlxMemory;
  });
}
export async function mlxStats(config?: MlxEmbedConfig): Promise<MlxStats | null> {
  return metadata("/stats", config, (data) => {
    const value = object(data);
    for (const key of ["total_requests", "avg_ms", "compiled_shapes", "uptime_sec"]) if (typeof value[key] !== "number" || !Number.isFinite(value[key]) || (value[key] as number) < 0) throw new EmbeddingProtocolError("Invalid MLX stats response");
    for (const key of ["total_requests", "compiled_shapes"]) if (!Number.isSafeInteger(value[key])) throw new EmbeddingProtocolError("Invalid MLX stats count");
    return value as unknown as MlxStats;
  });
}
export async function isMlxAvailable(config?: MlxEmbedConfig): Promise<boolean> { return (await mlxHealth(config))?.ready ?? false; }

export async function getMlxDescriptor(rawConfig?: MlxEmbedConfig, signal?: AbortSignal): Promise<EmbeddingDescriptor | null> {
  const config = normalizeConfig(rawConfig);
  validateSignal(signal);
  throwIfCancelled(config, signal);
  try {
    return await withScope(config, signal, true, async (scope) => {
      let data: unknown;
      try { data = await requestJson("/descriptor", undefined, config, scope, METADATA_BYTES); }
      catch (error) {
        if (!(error instanceof HttpError) || ![404, 405, 501].includes(error.status)) throw error;
        data = validateHealth(await requestJson("/health", undefined, config, scope, METADATA_BYTES)).descriptor;
      }
      const actual = validateEmbeddingDescriptor(data);
      if (actual.backend !== "mlx") throw new EmbeddingProtocolError("Descriptor backend must be mlx");
      if (config.expectedDescriptor) assertEmbeddingCompatibility(config.expectedDescriptor, actual, "MLX negotiation");
      return actual;
    });
  } catch (error) { throwIfCancelled(config, signal); return null; }
}

/** Production wrapper: single-flight negotiation, immutable identity, owned cancellation. */
export class MlxEmbedClient {
  private readonly _config: TransportConfig;
  private readonly _lifetime = new AbortController();
  private _descriptor: EmbeddingDescriptor | null = null;
  private _negotiation: {
    promise: Promise<EmbeddingDescriptor | null>;
    controller: AbortController;
    waiters: number;
  } | null = null;
  private _removeExternalListener?: () => void;

  constructor(rawConfig?: MlxEmbedConfig) {
    const config = normalizeConfig(rawConfig);
    if (config.signal) {
      const external = config.signal;
      const onAbort = () => { this._lifetime.abort(abortError()); this._removeExternalListener?.(); };
      external.addEventListener("abort", onAbort, { once: true });
      this._removeExternalListener = () => { external.removeEventListener("abort", onAbort); this._removeExternalListener = undefined; };
      if (external.aborted) onAbort();
    }
    this._config = Object.freeze({ ...config, signal: this._lifetime.signal });
  }
  get dims(): number | null { return this._descriptor?.outputDimensions ?? null; }
  get modelName(): string | null { return this._descriptor?.model ?? null; }
  get descriptor(): EmbeddingDescriptor | null { return this._descriptor; }
  config(): MlxEmbedConfig { return Object.freeze({ ...this._config, expectedDescriptor: this._descriptor ?? this._config.expectedDescriptor }); }

  /** Idempotent and terminal: pending requests abort; late results cannot publish state. */
  dispose(): void {
    this._removeExternalListener?.();
    this._lifetime.abort(abortError());
  }
  async getDescriptor(signal?: AbortSignal): Promise<EmbeddingDescriptor | null> {
    validateSignal(signal);
    throwIfCancelled(this._config, signal);
    if (this._descriptor) return this._descriptor;
    if (!this._negotiation) {
      const controller = new AbortController();
      const negotiation = { controller, waiters: 0, promise: Promise.resolve<EmbeddingDescriptor | null>(null) };
      negotiation.promise = getMlxDescriptor(this.config(), controller.signal).then((descriptor) => {
        throwIfCancelled(this._config, controller.signal);
        if (descriptor) this._descriptor = descriptor;
        return descriptor;
      }).finally(() => { if (this._negotiation === negotiation) this._negotiation = null; });
      this._negotiation = negotiation;
    }
    const negotiation = this._negotiation;
    negotiation.waiters++;
    try { return await abortable(negotiation.promise, signal ?? this._lifetime.signal); }
    finally {
      negotiation.waiters--;
      // One cancelled waiter must not poison another caller's negotiation.
      // With no remaining consumers, cancel the shared HTTP request as well.
      if (negotiation.waiters === 0 && this._negotiation === negotiation) {
        this._negotiation = null;
        negotiation.controller.abort(abortError());
      }
    }
  }
  async probeDims(signal?: AbortSignal): Promise<number> {
    const descriptor = await this.getDescriptor(signal);
    if (!descriptor) throw new Error("Could not probe MLX dimensions: server unreachable or missing a valid embedding descriptor");
    return descriptor.outputDimensions;
  }
  private async embeddingConfig(signal?: AbortSignal, dims?: number): Promise<MlxEmbedConfig> {
    const descriptor = await this.getDescriptor(signal);
    if (!descriptor) throw new EmbeddingProtocolError("MLX server did not provide a valid embedding descriptor");
    throwIfCancelled(this._config, signal);
    if (dims === undefined || dims === descriptor.outputDimensions) return this.config();
    if (dims > descriptor.nativeDimensions) throw new EmbeddingProtocolError("Requested embedding dimensions exceed native dimensions");
    // A projection changes only this request's expected output geometry. The
    // native negotiation stays pinned, and every other identity field is checked.
    return Object.freeze({ ...this.config(), expectedDescriptor: validateEmbeddingDescriptor({ ...descriptor, outputDimensions: dims }) });
  }
  private async withEmbedding<T>(options: MlxEmbedOptions | undefined, fallback: T,
    run: (config: MlxEmbedConfig, options: MlxEmbedOptions) => Promise<T>): Promise<T> {
    try {
      return await withScope(this._config, options?.signal, false, async (scope) => {
        const config = await this.embeddingConfig(scope.signal, options?.dims);
        checkScope(scope);
        return run(config, { ...options, signal: scope.signal });
      });
    } catch (error) {
      // Preserve caller/lifetime cancellation and protocol failures. A deadline
      // remains a backend timeout, even when it expires during shared init.
      failure(error, this._config, options?.signal, true);
      return fallback;
    }
  }
  async embed(text: string, options?: MlxEmbedOptions): Promise<EmbeddingResult | null> {
    validateTexts([text]);
    validateEmbedOptions(options);
    return this.withEmbedding(options, null, (config, scoped) => embedWithMlx(text, config, scoped));
  }
  async embedBatch(texts: string[], options?: MlxEmbedOptions): Promise<(EmbeddingResult | null)[]> {
    validateTexts(texts);
    validateEmbedOptions(options);
    throwIfCancelled(this._config, options?.signal);
    if (texts.length === 0) return [];
    return this.withEmbedding(options, texts.map(() => null), (config, scoped) => embedBatchWithMlx(texts, config, scoped));
  }
  async embedBatchConcurrent(texts: string[], options?: MlxEmbedOptions, batchSize?: number): Promise<(EmbeddingResult | null)[]> {
    positiveInteger(batchSize === undefined ? 32 : batchSize, "batchSize", MAX_MLX_BATCH_SIZE);
    validateEmbedOptions(options);
    if (!Array.isArray(texts) || texts.some((text) => typeof text !== "string")) throw new TypeError("texts must be a string array");
    throwIfCancelled(this._config, options?.signal);
    if (texts.length === 0) return [];
    return this.withEmbedding(options, texts.map(() => null), (config, scoped) => embedBatchConcurrent(texts, config, scoped, batchSize));
  }
  async embedBatchBinary(texts: string[], options?: MlxEmbedOptions): Promise<number[][] | null> {
    validateTexts(texts);
    validateEmbedOptions(options);
    throwIfCancelled(this._config, options?.signal);
    if (texts.length === 0) return [];
    return this.withEmbedding(options, null, (config, scoped) => embedBatchBinaryWithMlx(texts, config, scoped));
  }
  async health(): Promise<MlxHealth | null> { return mlxHealth(this.config()); }
  async memory(): Promise<MlxMemory | null> { return mlxMemory(this.config()); }
  async stats(): Promise<MlxStats | null> { return mlxStats(this.config()); }
  async isAvailable(): Promise<boolean> { return isMlxAvailable(this.config()); }
}
