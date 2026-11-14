/**
 * job.ts — Durable indexing job orchestration and shadow database protections
 */

import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { Database } from "../db.js";
import { createSavepoint, withExclusiveDatabaseOperation, withoutDatabaseOperation, bindDatabaseOperation } from "../db.js";
import type { Store, EmbedResult, ChunkStrategy, EmbedProgress } from "../store.js";
import { generateEmbeddings, emptyEmbedResult } from "../store.js";
import {
  type IndexingJobCheckpoint,
  type IndexingFingerprint,
  computeDescriptorSignature,
  getActiveCheckpoint,
  saveCheckpoint,
  getCompletedFingerprint,
  saveCompletedFingerprint,
} from "./checkpoint.js";
import { getStoredChunkStrategy } from "./pending.js";
import type { EmbeddingDescriptor } from "../embedding/contract.js";

export interface IndexingJobOptions {
  jobId?: string;
  force?: boolean;
  model?: string;
  chunkStrategy?: ChunkStrategy;
  maxDocsPerBatch?: number;
  maxBatchBytes?: number;
  maxDuration?: number;
  onProgress?: (info: EmbedProgress, checkpoint?: IndexingJobCheckpoint) => void;
  signal?: AbortSignal;
}

export class ShadowDatabaseProtectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShadowDatabaseProtectionError";
  }
}

export class IndexingFingerprintMismatchError extends Error {
  readonly existingFingerprint: IndexingFingerprint;
  readonly expectedFingerprint: Partial<IndexingFingerprint>;
  readonly details: {
    sigMismatch: boolean;
    modelMismatch: boolean;
    strategyMismatch: boolean;
    dimMismatch: boolean;
  };

  constructor(
    message: string,
    existingFingerprint: IndexingFingerprint,
    expectedFingerprint: Partial<IndexingFingerprint>,
    details: {
      sigMismatch: boolean;
      modelMismatch: boolean;
      strategyMismatch: boolean;
      dimMismatch: boolean;
    }
  ) {
    super(message);
    this.name = "IndexingFingerprintMismatchError";
    this.existingFingerprint = existingFingerprint;
    this.expectedFingerprint = expectedFingerprint;
    this.details = details;
  }
}

export { IndexingFingerprintMismatchError as FingerprintMismatchError };

/**
 * Validates that a target database path is an explicit, safe shadow path
 * and NOT the active/live database file, symlink, or hardlink alias.
 */
export function validateShadowTarget(targetDbPath: string, liveDbPath?: string): void {
  if (!targetDbPath || typeof targetDbPath !== "string") {
    throw new ShadowDatabaseProtectionError("Shadow target database path must be a non-empty string.");
  }

  const defaultLivePath = path.resolve(process.env.HOME || "", ".cache/qmd/index.sqlite");
  const livePath = path.resolve(liveDbPath || defaultLivePath);

  // 1. Resolve canonical realpath for target and live
  let resolvedTarget = path.resolve(targetDbPath);
  try {
    if (fs.existsSync(resolvedTarget)) {
      resolvedTarget = fs.realpathSync(resolvedTarget);
    } else {
      const parentDir = path.dirname(resolvedTarget);
      if (fs.existsSync(parentDir)) {
        resolvedTarget = path.join(fs.realpathSync(parentDir), path.basename(resolvedTarget));
      }
    }
  } catch {}

  let resolvedLive = livePath;
  try {
    if (fs.existsSync(livePath)) {
      resolvedLive = fs.realpathSync(livePath);
    }
  } catch {}

  if (resolvedTarget === resolvedLive) {
    throw new ShadowDatabaseProtectionError(
      `Refusing to execute shadow workflow on live database: ${resolvedTarget}`
    );
  }

  // 2. Check stat / inode match (hardlinks) if both files exist
  try {
    if (fs.existsSync(resolvedTarget) && fs.existsSync(resolvedLive)) {
      const targetStat = fs.statSync(resolvedTarget);
      const liveStat = fs.statSync(resolvedLive);
      if (targetStat.dev === liveStat.dev && targetStat.ino === liveStat.ino) {
        throw new ShadowDatabaseProtectionError(
          `Refusing to execute shadow workflow on hardlink/alias of live database: ${resolvedTarget}`
        );
      }
    }
  } catch (err) {
    if (err instanceof ShadowDatabaseProtectionError) throw err;
  }

  // 3. Ensure shadow target has explicit isolation marker
  if (
    resolvedTarget.endsWith("index.sqlite") &&
    !resolvedTarget.includes("shadow") &&
    !resolvedTarget.includes("test") &&
    !resolvedTarget.includes("tmp") &&
    !resolvedTarget.includes("scratch")
  ) {
    throw new ShadowDatabaseProtectionError(
      `Shadow path '${resolvedTarget}' does not contain explicit isolation marker ('shadow', 'test', 'tmp', or 'scratch')`
    );
  }
}

/**
 * Runs a resumable, durable indexing job with checkpoint tracking.
 */
export async function runDurableIndexingJob(
  store: Store,
  options?: IndexingJobOptions,
): Promise<EmbedResult & { checkpoint: IndexingJobCheckpoint }> {
  return withExclusiveDatabaseOperation(store.db, "durable indexing", () => runDurableIndexingJobInternal(store, options));
}

async function runDurableIndexingJobInternal(
  store: Store,
  options?: IndexingJobOptions,
): Promise<EmbedResult & { checkpoint: IndexingJobCheckpoint }> {
  const db = store.db;
  const jobId = options?.jobId || randomUUID();
  const startedAt = new Date().toISOString();

  let descriptor: EmbeddingDescriptor | undefined = undefined;
  const newCheckpoint = (status: "in_progress" | "cancelled" = "in_progress"): IndexingJobCheckpoint => ({
    jobId,
    fingerprint: {
      model: options?.model || descriptor?.model || store.llm?.embedModelName || "embeddinggemma",
      dimensions: descriptor?.outputDimensions || 0,
      chunkStrategy: options?.chunkStrategy ?? "regex",
      descriptorSignature: computeDescriptorSignature(descriptor),
      revision: descriptor?.revision || undefined,
      quantization: descriptor?.quantization || undefined,
    },
    startedAt, updatedAt: startedAt, status,
    docsTotal: 0, docsCompleted: 0, chunksTotal: 0, chunksCommitted: 0,
  });
  const cancelled = () => ({ ...emptyEmbedResult("cancelled"), checkpoint: newCheckpoint("cancelled") });
  // Do not create/update even an empty job checkpoint for a preflight abort.
  if (options?.signal?.aborted) return cancelled();
  try {
    if (typeof store.getEmbeddingDescriptor === "function") {
      const d = await withoutDatabaseOperation(db, () => store.getEmbeddingDescriptor!({ signal: options?.signal }));
      descriptor = d ?? undefined;
    } else if (typeof (store as any).getDescriptor === "function") {
      const d = await withoutDatabaseOperation(db, () => (store as any).getDescriptor({ signal: options?.signal }));
      descriptor = d ?? undefined;
    }
  } catch (error) {
    if (options?.signal?.aborted) return cancelled();
    throw error;
  }
  if (options?.signal?.aborted) return cancelled();

  const currentSig = computeDescriptorSignature(descriptor);
  const effectiveModel = options?.model || descriptor?.model || store.llm?.embedModelName || "embeddinggemma";
  const targetStrategy: ChunkStrategy = options?.chunkStrategy ?? "regex";

  const existing = getActiveCheckpoint(db);
  const completed = getCompletedFingerprint(db);
  const storedStrategy = getStoredChunkStrategy(db);
  const differentModel = db.prepare(`SELECT model FROM content_vectors WHERE model != ? LIMIT 1`).get(effectiveModel) as { model: string } | undefined;
  const storedSpace = db.prepare(`SELECT value FROM store_config WHERE key = 'embedding_space_id'`).get() as { value: string } | undefined;
  const vectorTable = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string } | undefined;
  const storedDimensions = Number(vectorTable?.sql.match(/float\[(\d+)\]/)?.[1] ?? 0);
  const fingerprint = existing?.fingerprint ?? completed ?? (storedStrategy ? {
    model: differentModel?.model ?? effectiveModel,
    dimensions: storedDimensions || descriptor?.outputDimensions || 0,
    descriptorSignature: storedSpace?.value ?? currentSig,
    chunkStrategy: storedStrategy as ChunkStrategy,
  } : null);
  if (fingerprint && !options?.force) {
    const existingStrategy = fingerprint.chunkStrategy ?? "regex";
    const existingModel = fingerprint.model || "default";
    const sigMismatch = fingerprint.descriptorSignature !== currentSig;
    const modelMismatch = existingModel !== effectiveModel || Boolean(differentModel);
    const strategyMismatch = existingStrategy !== targetStrategy;
    const dimMismatch =
      descriptor?.outputDimensions !== undefined &&
      fingerprint.dimensions > 0 &&
      fingerprint.dimensions !== descriptor.outputDimensions;

    if (sigMismatch || modelMismatch || strategyMismatch || dimMismatch) {
      throw new IndexingFingerprintMismatchError(
        `Checkpoint fingerprint mismatch detected (descriptor: ${sigMismatch}, model: ${modelMismatch}, chunkStrategy: ${strategyMismatch}, dims: ${dimMismatch}). Automatic destructive migration is prohibited to prevent vector loss. Actionable message: explicit rebuild on isolated shadow index required (or specify options.force: true to explicitly overwrite).`,
        fingerprint,
        {
          model: effectiveModel,
          dimensions: descriptor?.outputDimensions || 0,
          chunkStrategy: targetStrategy,
          descriptorSignature: currentSig,
        },
        { sigMismatch, modelMismatch, strategyMismatch, dimMismatch }
      );
    }
  }

  const cp: IndexingJobCheckpoint = (existing && !options?.force) ? existing : newCheckpoint();

  const replacement = options?.force ? createSavepoint(db) : undefined;
  try {
    // Include checkpoint/fingerprint publication in the replacement boundary.
    if (!options?.force) saveCheckpoint(db, cp);

    let result = await generateEmbeddings(store, {
      force: Boolean(options?.force),
      model: effectiveModel,
      chunkStrategy: options?.chunkStrategy ?? "regex",
      maxDocsPerBatch: options?.maxDocsPerBatch,
      maxBatchBytes: options?.maxBatchBytes,
      maxDuration: options?.maxDuration,
      signal: options?.signal,
      onProgress: bindDatabaseOperation((info: EmbedProgress) => {
        cp.chunksTotal = info.totalChunks;
        cp.chunksCommitted = info.chunksEmbedded;
        cp.updatedAt = new Date().toISOString();
        if (!options?.force) saveCheckpoint(db, cp);
        withoutDatabaseOperation(db, () => options?.onProgress?.(info, cp));
      }),
    });

    // Cancellation may arrive between the generator's final await and durable
    // publication, including the empty replacement path.
    if (options?.signal?.aborted && result.status === "complete") {
      result = {
        ...result, status: "cancelled", firstFailureReason: "Embedding aborted by signal",
        ...(options?.force ? { docsProcessed: 0, chunksCommitted: 0, chunksEmbedded: 0, chunksPending: result.chunksAttempted } : {}),
      };
    }

    cp.status = result.status === "complete" ? "completed" : result.status === "cancelled" ? "cancelled" : "failed";
    cp.docsCompleted = result.docsProcessed;
    cp.docsTotal = result.docsSelected;
    cp.chunksCommitted = result.chunksCommitted;
    cp.updatedAt = new Date().toISOString();
    if (!options?.force || result.status === "complete") saveCheckpoint(db, cp);
    if (result.status === "complete") saveCompletedFingerprint(db, cp.fingerprint);
    if (result.status === "complete") replacement?.commit();
    else replacement?.rollback();

    return { ...result, checkpoint: cp };
  } catch (error) {
    replacement?.rollback();
    throw error;
  }
}
