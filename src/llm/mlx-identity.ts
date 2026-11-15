import { createHash } from "node:crypto";
import type { ReadableStreamDefaultReader } from "node:stream/web";
import { abortable, linkSignals, throwIfAborted } from "./lifecycle.js";

export type MlxStage = "rerank" | "expand";
export type MlxStageContract = { rerank?: unknown; generate?: unknown };
type StageDescriptor = Record<string, unknown> & { model: string };
type TransportConfig = { url?: string; timeoutMs?: number; signal?: AbortSignal };
export type MlxServingResult = { model?: string; revision?: string; descriptor?: unknown };

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function canonical(value: unknown, depth = 0): unknown {
  if (depth > 16) throw new Error('MLX stage descriptor nesting exceeds limit');
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('MLX stage descriptor contains a non-finite number');
  if (typeof value === 'string' && value.length > 16_384) throw new Error('MLX stage descriptor string exceeds limit');
  if (Array.isArray(value)) return value.map(item => canonical(item, depth + 1));
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => [key, canonical(item, depth + 1)]));
}

function stageDescriptor(descriptor: MlxStageContract | null, op: MlxStage): StageDescriptor | null {
  const stage = record(descriptor?.[op === 'rerank' ? 'rerank' : 'generate']);
  if (!stage || typeof stage.model !== 'string' || !stage.model.trim() || stage.model.length > 8192) return null;
  if (stage.revision !== undefined && typeof stage.revision !== 'string') return null;
  if (stage.backend !== undefined && stage.backend !== 'mlx') return null;
  if (stage.version !== undefined && stage.version !== 1) return null;
  if (stage.kind !== undefined && stage.kind !== (op === 'rerank' ? 'rerank' : 'generate')) return null;
  if ('pooling' in stage || 'nativeDimensions' in stage || 'outputDimensions' in stage) return null;
  canonical(stage); // Validate all fields before they can define a namespace.
  return stage as StageDescriptor;
}

/** Result metadata is authoritative, never a projection of the requested model. */
export function validateMlxServingResult(op: MlxStage, result: MlxServingResult): MlxServingResult {
  const { model, revision, descriptor } = result;
  if (model !== undefined && (typeof model !== 'string' || !model.trim() || model.length > 8192)) {
    throw new Error(`Invalid MLX ${op} serving model`);
  }
  if (revision !== undefined && (typeof revision !== 'string' || revision.length > 16_384)) {
    throw new Error(`Invalid MLX ${op} serving revision`);
  }
  let snapshot: unknown;
  if (descriptor !== undefined) {
    const value = record(descriptor);
    const contract = value && ('model' in value ? { [op === 'rerank' ? 'rerank' : 'generate']: value } : value);
    const stage = stageDescriptor(contract, op);
    if (!stage) throw new Error(`Invalid MLX ${op} serving descriptor`);
    if (model !== undefined && stage.model !== model) throw new Error(`MLX ${op} serving descriptor model disagrees with response`);
    if (revision !== undefined && stage.revision !== undefined && stage.revision !== revision) {
      throw new Error(`MLX ${op} serving descriptor revision disagrees with response`);
    }
    snapshot = canonical(descriptor);
  }
  return {
    ...(model === undefined ? {} : { model }), ...(revision === undefined ? {} : { revision }),
    ...(descriptor === undefined ? {} : { descriptor: snapshot }),
  };
}

/** Hash the whole stage contract: revision, quantization and tokenizer changes invalidate caches. */
export function stageCacheIdentity(descriptor: MlxStageContract | null, op: MlxStage): string | null {
  const stage = stageDescriptor(descriptor, op);
  if (!stage) return null;
  const extra = Object.keys(stage).some(key => key !== 'model');
  const suffix = extra ? `#${createHash('sha256').update(JSON.stringify(canonical(stage))).digest('hex').slice(0, 24)}` : '';
  return `mlx:${stage.model}${suffix}`;
}

/** Identity is local to this result, never a shared last-backend flag. */
export function actualStageIdentity(
  op: MlxStage,
  result: MlxServingResult,
  before: MlxStageContract | null,
  after: MlxStageContract | null,
): string {
  result = validateMlxServingResult(op, result);
  const model = result.model;
  // Legacy text-only responses remain usable, but are deliberately uncacheable.
  if (!model || model === 'mlx') return 'mlx';
  const first = stageDescriptor(before, op);
  const last = stageDescriptor(after, op);
  const predicted = stageCacheIdentity(before, op);
  if (first && last && predicted && predicted === stageCacheIdentity(after, op) && first.model === model && last.model === model) {
    if (result.revision !== undefined && result.revision !== (first.revision ?? '')) {
      throw new Error(`MLX ${op} serving revision changed during inference`);
    }
    const serving = record(result.descriptor);
    if (serving) {
      const contract = 'model' in serving ? { [op === 'rerank' ? 'rerank' : 'generate']: serving } : serving;
      if (stageCacheIdentity(contract, op) !== predicted) throw new Error(`MLX ${op} serving descriptor changed during inference`);
    }
    return predicted;
  }
  if (first || last) throw new Error(`MLX ${op} model identity changed during inference`);
  return 'mlx';
}

/** Bounded stage metadata/text transport: the embedding validator intentionally narrows capabilities. */
async function requestStageJson(
  path: string,
  config: TransportConfig,
  payload?: Record<string, unknown>,
  expires = performance.now() + (payload === undefined ? Math.min(config.timeoutMs ?? 300_000, 3000) : config.timeoutMs ?? 300_000),
): Promise<Record<string, unknown> | null> {
  const cap = payload === undefined ? 131_072 : 1_048_576;
  const duration = expires - performance.now();
  if (!(duration > 0) || !Number.isFinite(duration)) throw new DOMException('MLX request timed out', 'TimeoutError');
  const deadline = new AbortController();
  const timer = setTimeout(() => deadline.abort(new DOMException('MLX request timed out', 'TimeoutError')), duration);
  timer.unref?.();
  const linked = linkSignals([config.signal, deadline.signal]);
  let response: Response | undefined;
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try {
    throwIfAborted(linked.signal);
    response = await abortable(fetch(`${config.url ?? 'http://127.0.0.1:8787'}${path}`, {
      method: payload === undefined ? 'GET' : 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json', 'X-Request-Timeout': String(duration / 1000) },
      body: payload === undefined ? undefined : JSON.stringify(payload),
      signal: linked.signal,
    }), linked.signal, late => { void late.body?.cancel().catch(() => {}); });
    if (!response.ok || !response.body) return null;
    const length = response.headers.get('content-length');
    if (length !== null && /^\d+$/.test(length) && Number(length) > cap) throw new Error('MLX stage response exceeds byte limit');
    reader = response.body.getReader();
    let storage = new Uint8Array(Math.min(cap, 8192));
    let bytes = 0;
    let emptyChunks = 0;
    while (true) {
      const { value, done } = await abortable(reader.read(), linked.signal);
      throwIfAborted(linked.signal);
      if (done) break;
      if (value.byteLength === 0) {
        if (++emptyChunks > 64) throw new Error('MLX stage body made no progress');
        continue;
      }
      emptyChunks = 0;
      if (bytes + value.byteLength > cap) throw new Error('MLX stage response exceeds byte limit');
      if (bytes + value.byteLength > storage.byteLength) {
        const grown = new Uint8Array(Math.min(cap, Math.max(storage.byteLength * 2, bytes + value.byteLength)));
        grown.set(storage);
        storage = grown;
      }
      storage.set(value, bytes);
      bytes += value.byteLength;
    }
    const data: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(storage.subarray(0, bytes)));
    throwIfAborted(linked.signal);
    if (performance.now() >= expires) throw new DOMException('MLX request timed out', 'TimeoutError');
    return record(data);
  } finally {
    clearTimeout(timer);
    linked.dispose();
    if (reader) {
      try { void reader.cancel().catch(() => {}); } catch { /* cancellation cannot extend a deadline */ }
      try { reader.releaseLock(); } catch { /* a non-cooperative pending read may retain its lock */ }
    } else {
      try { void response?.body?.cancel().catch(() => {}); } catch { /* best effort */ }
    }
  }
}

/** Read raw stage contracts, not an embedding-only projection which drops stage revisions. */
export async function getMlxStageContract(config: TransportConfig): Promise<MlxStageContract | null> {
  try {
    // Fallback is another exchange in this operation, not another timeout budget.
    const expires = performance.now() + Math.min(config.timeoutMs ?? 300_000, 3000);
    let descriptor = await requestStageJson('/descriptor', config, undefined, expires);
    if (!descriptor) descriptor = record((await requestStageJson('/health', config, undefined, expires))?.descriptor);
    if (!descriptor) return null;
    const result: MlxStageContract = {};
    for (const op of ['rerank', 'expand'] as const) {
      const stage = stageDescriptor(descriptor, op);
      if (stage) result[op === 'rerank' ? 'rerank' : 'generate'] = stage;
    }
    // Without embeddings the daemon exposes its primary stage at the root:
    // generators declare kind=generate; legacy rerankers have no kind. Strip
    // sibling capabilities so the hash matches the inference stage descriptor.
    if (descriptor.version === 1 && descriptor.backend === 'mlx'
        && !('pooling' in descriptor || 'nativeDimensions' in descriptor || 'outputDimensions' in descriptor)) {
      const op = descriptor.kind === 'generate' ? 'expand' : 'rerank';
      const key = op === 'rerank' ? 'rerank' : 'generate';
      const { rerank: _rerank, generate: _generate, ...primary } = descriptor;
      const stage = stageDescriptor({ [key]: primary }, op);
      if (stage && result[key] === undefined) result[key] = stage;
    }
    if (performance.now() >= expires) throw new DOMException('MLX request timed out', 'TimeoutError');
    return result;
  } catch (error) {
    throwIfAborted(config.signal);
    return null;
  }
}

/** The legacy string-only generation API discards the serving model needed by the cache. */
export async function generateMlxWithIdentity(
  prompt: string,
  config: TransportConfig,
  options: { maxTokens?: number; temperature?: number } = {},
): Promise<({ text: string } & MlxServingResult) | null> {
  const maxTokens = options.maxTokens ?? 256;
  const temperature = options.temperature ?? 0.7;
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || maxTokens > 65_536) throw new Error('maxTokens must be a positive bounded integer');
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) throw new Error('temperature must be finite and between 0 and 2');
  if (Buffer.byteLength(prompt) > 1_048_576) throw new Error('MLX generation prompt exceeded 1 MiB');
  const data = await requestStageJson('/generate', config, { prompt, max_tokens: maxTokens, temperature });
  if (!data || typeof data.text !== 'string' || Buffer.byteLength(data.text) > maxTokens * 64) return null;
  return { text: data.text, ...validateMlxServingResult('expand', data as MlxServingResult) };
}
