/** Validated immutable embedding configuration: caller > environment > defaults. */
import { MAX_EMBEDDING_DIMENSIONS, type EmbeddingBackendType } from "./contract.js";

export interface EmbeddingConfigOptions {
  backend?: EmbeddingBackendType;
  model?: string;
  dims?: number;
  mlxUrl?: string;
  mlxTimeoutMs?: number;
  mlxConcurrency?: number;
  mlxBinary?: boolean;
  failClosed?: boolean;
}

export interface ResolvedEmbeddingConfig {
  readonly backend: EmbeddingBackendType;
  readonly model: string;
  readonly dims?: number;
  readonly mlxUrl: string;
  readonly mlxTimeoutMs: number;
  readonly mlxConcurrency: number;
  readonly mlxBinary: boolean;
  readonly failClosed: boolean;
}

export const DEFAULT_GGUF_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-Q8_0.gguf";
export const DEFAULT_MLX_URL = "http://127.0.0.1:8787";
export const DEFAULT_MLX_TIMEOUT_MS = 300_000;
export const DEFAULT_MLX_CONCURRENCY = 2;
export const MAX_MLX_TIMEOUT_MS = 3_600_000;
export const MAX_MLX_CONCURRENCY = 32;
export const MAX_MLX_BATCH_SIZE = 1024;

export function positiveInteger(value: unknown, name: string, ceiling: number): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0 || value > ceiling) {
    throw new RangeError(`${name} must be a finite positive integer <= ${ceiling}`);
  }
  return value;
}

function envInteger(value: string, name: string, ceiling: number): number {
  // Number/parseInt alone would accept whitespace, exponents or numeric suffixes.
  if (!/^[0-9]+$/.test(value)) throw new RangeError(`${name} must contain only decimal digits`);
  return positiveInteger(Number(value), name, ceiling);
}

export function validateMlxUrl(value: string): string {
  if (typeof value !== "string" || value !== value.trim()) throw new TypeError("MLX URL must be an HTTP(S) URL");
  const parsed = new URL(value);
  if (!["http:", "https:"].includes(parsed.protocol) || parsed.search || parsed.hash || parsed.username || parsed.password) {
    throw new TypeError("MLX URL must be HTTP(S) without credentials, query or fragment");
  }
  return value.replace(/\/+$/, "");
}

function booleanOption(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new TypeError(`${name} must be a boolean`);
  return value;
}

/** Reject malformed values rather than silently selecting another backend/limit. */
export function resolveEmbeddingConfig(options: EmbeddingConfigOptions = {}): ResolvedEmbeddingConfig {
  const backend = options.backend ?? process.env.QMD_EMBED_BACKEND?.toLowerCase() ?? "gguf";
  if (backend !== "mlx" && backend !== "gguf") throw new TypeError("Embedding backend must be mlx or gguf");
  const model = options.model ?? process.env.QMD_EMBED_MODEL ?? (backend === "mlx"
    ? process.env.MLX_EMBED_MODEL ?? "mlx-community/nomic-embed-text-v1.5"
    : DEFAULT_GGUF_EMBED_MODEL);
  if (typeof model !== "string" || !model.trim()) throw new TypeError("Embedding model must be a nonempty string");

  const dims = options.dims !== undefined
    ? positiveInteger(options.dims, "dims", MAX_EMBEDDING_DIMENSIONS)
    : process.env.QMD_EMBED_DIMS !== undefined
      ? envInteger(process.env.QMD_EMBED_DIMS, "QMD_EMBED_DIMS", MAX_EMBEDDING_DIMENSIONS)
      : undefined;
  let mlxUrl = options.mlxUrl ?? process.env.QMD_MLX_EMBED_URL;
  if (mlxUrl === undefined) {
    const port = process.env.MLX_EMBED_PORT === undefined ? 8787 : envInteger(process.env.MLX_EMBED_PORT, "MLX_EMBED_PORT", 65535);
    mlxUrl = `http://127.0.0.1:${port}`;
  }
  const mlxTimeoutMs = options.mlxTimeoutMs !== undefined
    ? positiveInteger(options.mlxTimeoutMs, "mlxTimeoutMs", MAX_MLX_TIMEOUT_MS)
    : process.env.QMD_MLX_TIMEOUT_MS !== undefined
      ? envInteger(process.env.QMD_MLX_TIMEOUT_MS, "QMD_MLX_TIMEOUT_MS", MAX_MLX_TIMEOUT_MS)
      : DEFAULT_MLX_TIMEOUT_MS;
  const mlxConcurrency = options.mlxConcurrency !== undefined
    ? positiveInteger(options.mlxConcurrency, "mlxConcurrency", MAX_MLX_CONCURRENCY)
    : process.env.QMD_MLX_CONCURRENCY !== undefined
      ? envInteger(process.env.QMD_MLX_CONCURRENCY, "QMD_MLX_CONCURRENCY", MAX_MLX_CONCURRENCY)
      : DEFAULT_MLX_CONCURRENCY;

  let mlxBinary = options.mlxBinary;
  if (mlxBinary === undefined && process.env.QMD_MLX_BINARY !== undefined) {
    const value = process.env.QMD_MLX_BINARY.toLowerCase();
    if (!["true", "false", "1", "0"].includes(value)) throw new TypeError("QMD_MLX_BINARY must be true, false, 1 or 0");
    mlxBinary = value === "true" || value === "1";
  }
  return Object.freeze({
    backend, model, dims, mlxUrl: validateMlxUrl(mlxUrl), mlxTimeoutMs, mlxConcurrency,
    mlxBinary: booleanOption(mlxBinary ?? true, "mlxBinary"),
    failClosed: booleanOption(options.failClosed ?? (backend === "mlx"), "failClosed"),
  });
}
