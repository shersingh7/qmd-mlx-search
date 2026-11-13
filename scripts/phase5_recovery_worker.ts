/** Isolated production indexing entrypoint. No local inference or fallback. */
import { writeFileSync, writeSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { createStore, runDurableIndexingJob, validateShadowTarget, getActiveCheckpoint, type QMDStore } from "../src/index.js";
import { disposeDefaultLlamaCpp } from "../src/llm.js";
import { openDatabase, type Database } from "../src/db.js";

export const RECOVERY_BARRIER_PREFIX = "[BARRIER_REACHED] ";

export interface RecoveryWorkerOptions {
  dbPath: string;
  docsDir?: string;
  mlxPort?: number;
  mlxModelPath?: string;
  barrierAfterDocs?: number;
  barrierAfterChunks?: number;
  stopMode?: "exit" | "pause";
  action?: "index" | "resume";
  outputJson?: string;
  launchToken?: string;
  quiet?: boolean;
  pauseTimeoutMs?: number;
}

export interface RecoveryWorkerResult {
  status: "barrier_reached" | "completed" | "cancelled" | "failed";
  pid: number;
  dbPath: string;
  action: string;
  docsCompleted: number;
  chunksCommitted: number;
  totalVectors: number;
  checkpointStatus: string | null;
  checkpointJobId?: string;
  consecutiveSequenceVerified: boolean;
  hashes: Array<{ hash: string; seqs: number[] }>;
  error?: string;
}

function positiveInteger(name: string, value: number | undefined): void {
  if (value !== undefined && (!Number.isSafeInteger(value) || value < 1)) {
    throw new Error(`${name} must be a positive integer`);
  }
}

function inspectCommitted(db: Database) {
  const rows = db.prepare("SELECT hash, seq FROM content_vectors ORDER BY hash, seq").all() as Array<{ hash: string; seq: number }>;
  const sequences = new Map<string, number[]>();
  for (const { hash, seq } of rows) {
    const seqs = sequences.get(hash) || [];
    seqs.push(seq);
    sequences.set(hash, seqs);
  }
  // A document barrier means fully committed distinct contents, NOT chunks.
  const completed = db.prepare(`
    SELECT COUNT(*) AS count FROM (
      SELECT DISTINCT d.hash FROM documents d
      JOIN content_chunk_expectations e ON e.hash = d.hash AND e.strategy = 'regex'
      WHERE d.active = 1 AND e.total_chunks > 0
        AND (SELECT COUNT(*) FROM content_vectors v WHERE v.hash = d.hash) = e.total_chunks
    )
  `).get() as { count: number };
  return {
    docsCompleted: completed.count,
    totalVectors: rows.length,
    hashes: Array.from(sequences, ([hash, seqs]) => ({ hash, seqs })),
    consecutiveSequenceVerified: Array.from(sequences.values()).every(seqs => seqs.every((seq, i) => seq === i)),
  };
}

export async function runRecoveryWorker(options: RecoveryWorkerOptions): Promise<RecoveryWorkerResult> {
  const dbPath = path.resolve(options.dbPath);
  const action = options.action ?? "index";
  let store: QMDStore | null = null;
  let observer: Database | null = null;
  const envKeys = ["QMD_EMBED_BACKEND", "QMD_MLX_EMBED_URL", "QMD_MLX_FALLBACK"] as const;
  const previous = Object.fromEntries(envKeys.map(key => [key, process.env[key]]));
  try {
    positiveInteger("barrierAfterDocs", options.barrierAfterDocs);
    positiveInteger("barrierAfterChunks", options.barrierAfterChunks);
    positiveInteger("pauseTimeoutMs", options.pauseTimeoutMs);
    const port = options.mlxPort ?? 8797;
    if (!Number.isInteger(port) || port < 1 || port > 65535 || port === 8787) {
      throw new Error("mlxPort must be an isolated port from 1 to 65535, never live port 8787");
    }
    if (action !== "index" && action !== "resume") throw new Error(`Unknown action: ${action}`);
    const stopMode = options.stopMode ?? "exit";
    if (stopMode !== "exit" && stopMode !== "pause") throw new Error(`Unknown stopMode: ${stopMode}`);
    if (options.barrierAfterDocs !== undefined && options.barrierAfterChunks !== undefined) {
      throw new Error("Specify one barrier: barrierAfterDocs or barrierAfterChunks");
    }
    if (action === "resume" && (options.barrierAfterDocs !== undefined || options.barrierAfterChunks !== undefined)) {
      throw new Error("Resume does not accept an indexing barrier");
    }
    validateShadowTarget(dbPath);
    const mlxUrl = `http://127.0.0.1:${port}`;
    // The chunker still uses the legacy singleton: pin it to the very same
    // isolated endpoint as the per-store LLM, never its default live daemon.
    process.env.QMD_EMBED_BACKEND = "mlx";
    process.env.QMD_MLX_EMBED_URL = mlxUrl;
    process.env.QMD_MLX_FALLBACK = "0";
    const docsDir = options.docsDir ? path.resolve(options.docsDir) : path.join(path.dirname(dbPath), "public-docs");
    store = await createStore({ dbPath, config: {
      models: { embed: options.mlxModelPath || path.join(process.env.HOME || "~", ".cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine"),
        embedBackend: "mlx", mlxUrl, mlxFallback: false },
      collections: { public: { path: docsDir, pattern: "**/*.md" } },
    } });
    if (action === "index") await store.update();
    else if (!getActiveCheckpoint(store.internal.db)) throw new Error("Resume requires an active durable checkpoint");

    // Progress hooks deliberately lack the indexing connection's lease. Read
    // through a separate query-only connection: the barrier must prove durable
    // commits visible to another SQLite reader, not privileged/uncommitted state.
    observer = openDatabase(dbPath);
    observer.exec("PRAGMA query_only = ON");

    const vectorsBefore = (store.internal.db.prepare("SELECT COUNT(*) AS count FROM content_vectors").get() as { count: number }).count;
    const abortController = new AbortController();
    let barrierReached = false;
    let pauseExpired = false;
    const jobResult = await runDurableIndexingJob(store.internal, {
      maxDocsPerBatch: 1,
      signal: abortController.signal,
      onProgress: info => {
        if (barrierReached) return;
        const reachedChunks = options.barrierAfterChunks !== undefined && info.chunksEmbedded >= options.barrierAfterChunks;
        const reachedDocs = options.barrierAfterDocs !== undefined && inspectCommitted(observer!).docsCompleted >= options.barrierAfterDocs;
        if (reachedDocs || reachedChunks) {
          barrierReached = true;
          if (stopMode === "pause") {
            const snapshot = inspectCommitted(observer!);
            const checkpoint = observer!.prepare(`SELECT job_id, status, chunks_committed
              FROM indexing_checkpoints WHERE status = 'in_progress' ORDER BY updated_at DESC LIMIT 1`).get() as
              { job_id: string; status: string; chunks_committed: number } | undefined;
            if (checkpoint?.status !== "in_progress" || snapshot.totalVectors === 0
              || checkpoint.chunks_committed !== snapshot.totalVectors - vectorsBefore) {
              throw new Error("Committed barrier lacks a matching in-progress durable checkpoint");
            }
            const barrier: RecoveryWorkerResult = { status: "barrier_reached", pid: process.pid,
              dbPath, action, ...snapshot, chunksCommitted: snapshot.totalVectors - vectorsBefore,
              checkpointStatus: checkpoint.status, checkpointJobId: checkpoint.job_id };
            if (options.outputJson) writeFileSync(options.outputJson, JSON.stringify(barrier, null, 2));
            // onProgress is synchronous. Write the protocol synchronously, then
            // park THIS worker at the committed boundary without letting the
            // job return, cancel, close SQLite, or start the next batch. The
            // parent is another OS process and must SIGKILL us here.
            writeSync(process.stdout.fd, `${RECOVERY_BARRIER_PREFIX}${JSON.stringify(barrier)}\n`);
            Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, Math.min(options.pauseTimeoutMs ?? 10_000, 60_000));
            pauseExpired = true;
            throw new Error("Parent did not terminate worker before bounded barrier pause expired");
          }
          abortController.abort();
        }
      },
    });
    if (pauseExpired) throw new Error("Parent did not terminate worker before bounded barrier pause expired");
    const snapshot = inspectCommitted(store.internal.db);
    const active = getActiveCheckpoint(store.internal.db);
    const result: RecoveryWorkerResult = {
      status: barrierReached ? "barrier_reached" : jobResult.status === "complete" ? "completed" : jobResult.status === "cancelled" ? "cancelled" : "failed",
      // Report actual durable commits, including commits made before a session
      // cancellation unwinds; an inference return value is not DB evidence.
      pid: process.pid, dbPath, action, ...snapshot, chunksCommitted: snapshot.totalVectors - vectorsBefore,
      checkpointStatus: active?.status ?? null, checkpointJobId: jobResult.checkpoint.jobId,
      ...(jobResult.firstFailureReason && !barrierReached ? { error: jobResult.firstFailureReason } : {}),
    };
    if ((options.barrierAfterDocs !== undefined || options.barrierAfterChunks !== undefined) && !barrierReached) {
      throw new Error(`Indexing finished without reaching the requested committed barrier (${jobResult.status}, ${jobResult.chunksCommitted} chunks): ${jobResult.firstFailureReason || "barrier exceeds corpus"}`);
    }
    if (barrierReached) {
      if (jobResult.status !== "cancelled" || !active || snapshot.totalVectors === 0) {
        throw new Error("Committed barrier lacks a durable cancelled checkpoint and vectors");
      }
      // Non-crash diagnostic mode exits cleanly after cancellation. The
      // fresh-process proof above always pauses inside onProgress instead.
      if (options.outputJson) writeFileSync(options.outputJson, JSON.stringify(result, null, 2));
      console.log(`${RECOVERY_BARRIER_PREFIX}${JSON.stringify(result)}`);
    }
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!options.quiet) console.error(`[Recovery Worker Error] ${message}`);
    return { status: "failed", pid: process.pid, dbPath, action, docsCompleted: 0, chunksCommitted: 0,
      totalVectors: 0, checkpointStatus: null, consecutiveSequenceVerified: false, hashes: [], error: message };
  } finally {
    const cleanupErrors: string[] = [];
    try { observer?.close(); } catch (error) { cleanupErrors.push(`Observer cleanup: ${String(error)}`); }
    if (store) await store.close().catch(error => cleanupErrors.push(`Store cleanup: ${String(error)}`));
    await disposeDefaultLlamaCpp().catch(error => cleanupErrors.push(`LLM cleanup: ${String(error)}`));
    for (const key of envKeys) {
      if (previous[key] === undefined) delete process.env[key];
      else process.env[key] = previous[key];
    }
    if (cleanupErrors.length) throw new Error(`Recovery worker cleanup failed: ${cleanupErrors.join('; ')}`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  const getArg = (flag: string): string | undefined => {
    const index = args.indexOf(flag);
    if (index === -1) return undefined;
    if (!args[index + 1] || args[index + 1]!.startsWith("--")) throw new Error(`Missing value for ${flag}`);
    return args[index + 1];
  };
  try {
    const launchToken = getArg("--launch-token") || process.env.QMD_PHASE5_LAUNCH_TOKEN;
    if (!launchToken?.trim()) throw new Error("Direct TypeScript recovery worker execution rejected. Requires supervisor approval via --launch-token or QMD_PHASE5_LAUNCH_TOKEN.");
    const dbPath = getArg("--db");
    if (!dbPath) throw new Error("--db <path> is required");
    const numberArg = (flag: string) => getArg(flag) === undefined ? undefined : Number(getArg(flag));
    const outputJson = getArg("--output-json");
    const executionTimeoutMs = numberArg("--execution-timeout-ms") ?? 60_000;
    positiveInteger("executionTimeoutMs", executionTimeoutMs);
    const timer = setTimeout(() => {
      console.error("Recovery worker exceeded bounded execution deadline");
      process.exit(1);
    }, executionTimeoutMs);
    runRecoveryWorker({ dbPath, docsDir: getArg("--docs-dir"), mlxPort: numberArg("--mlx-port"),
      mlxModelPath: getArg("--mlx-model-path"), barrierAfterDocs: numberArg("--barrier-after-docs"),
      barrierAfterChunks: numberArg("--barrier-after-chunks"), pauseTimeoutMs: numberArg("--pause-timeout-ms"),
      stopMode: getArg("--stop-mode") as RecoveryWorkerOptions["stopMode"],
      action: getArg("--action") as RecoveryWorkerOptions["action"], outputJson, launchToken,
    }).then(result => {
      clearTimeout(timer);
      if (outputJson) writeFileSync(outputJson, JSON.stringify(result, null, 2));
      process.exit(result.status === "failed" ? 1 : 0);
    }).catch(error => {
      console.error(error);
      process.exit(1);
    });
  } catch (error) {
    console.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
}
