import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { createStore, getActiveCheckpoint, IndexingFingerprintMismatchError, runDurableIndexingJob, saveCheckpoint, type QMDStore } from "../src/index.js";
import { chunkDocument, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_WINDOW_TOKENS } from "../src/store.js";
import { disposeDefaultLlamaCpp } from "../src/llm.js";
import { encodeBinaryEmbeddings } from "../src/embedding/protocol.js";
import { executeRecoveryWorker, runPhase5Qualification, verifyRetrievalEquivalence, type StageMetrics } from "../scripts/phase5_e2e_runner.js";

const descriptor = {
  version: 1 as const, backend: "mlx" as const, model: "synthetic-recovery-only",
  pooling: "mean" as const, nativeDimensions: 4, outputDimensions: 4,
  maxTokens: 2048, normalized: true,
};

function embedding(text: string): number[] {
  const bytes = createHash("sha256").update(text).digest();
  const values = Array.from(bytes.subarray(0, 4), n => n + 1);
  const norm = Math.hypot(...values);
  return values.map(n => n / norm);
}

function logicalState(store: QMDStore) {
  const db = store.internal.db;
  return {
    content: db.prepare("SELECT * FROM content ORDER BY hash").all(),
    documents: db.prepare("SELECT * FROM documents ORDER BY collection, path").all(),
    vectors: db.prepare("SELECT * FROM content_vectors ORDER BY hash, seq").all(),
    vectorBytes: db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec ORDER BY hash_seq").all(),
    expectations: db.prepare("SELECT * FROM content_chunk_expectations ORDER BY hash, strategy").all(),
    config: db.prepare("SELECT * FROM store_config ORDER BY key").all(),
    checkpoints: db.prepare("SELECT * FROM indexing_checkpoints ORDER BY job_id").all(),
  };
}

function expectReaped(pid: number) {
  expect(() => process.kill(pid, 0)).toThrowError(/ESRCH/);
}

describe("W13 fresh OS-process recovery (synthetic HTTP, no models)", () => {
  let root: string;
  let server: http.Server;
  let port: number;
  let requests: string[];
  const envKeys = ["QMD_EMBED_BACKEND", "QMD_MLX_EMBED_URL", "QMD_MLX_FALLBACK"] as const;
  let previousEnv: Array<string | undefined>;
  const stores: QMDStore[] = [];

  beforeEach(async () => {
    previousEnv = envKeys.map(key => process.env[key]);
    root = mkdtempSync(path.join(tmpdir(), "qmd-w13-shadow-"));
    requests = [];
    server = http.createServer(async (req, res) => {
      const route = req.url || "/";
      requests.push(route);
      res.setHeader("Content-Type", "application/json");
      if (route === "/health") {
        res.end(JSON.stringify({ status: "ok", ready: true, model: descriptor.model, dims: 4, descriptor }));
      } else if (route === "/descriptor") {
        res.end(JSON.stringify(descriptor));
      } else {
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(Buffer.from(chunk));
        const { texts } = JSON.parse(Buffer.concat(chunks).toString()) as { texts: string[] };
        if (route === "/tokenize") {
          res.end(JSON.stringify({ counts: texts.map(text => Math.ceil(text.length / 4)) }));
        } else if (route === "/embed") {
          res.end(JSON.stringify({ embeddings: texts.map(embedding), model: descriptor.model, dims: 4, descriptor }));
        } else if (route === "/embed-bin") {
          res.setHeader("Content-Type", "application/octet-stream");
          res.setHeader("X-QMD-Embedding-Descriptor", encodeURIComponent(JSON.stringify(descriptor)));
          res.end(encodeBinaryEmbeddings(texts.map(embedding), 4));
        } else {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: `Unexpected synthetic request ${route}` }));
        }
      }
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    port = (server.address() as { port: number }).port;
    expect(port).not.toBe(8787);
    // Also pin the legacy singleton tokenizer to this owned ephemeral endpoint.
    process.env.QMD_EMBED_BACKEND = "mlx";
    process.env.QMD_MLX_EMBED_URL = `http://127.0.0.1:${port}`;
    process.env.QMD_MLX_FALLBACK = "0";
    await disposeDefaultLlamaCpp();
  });

  afterEach(async () => {
    try {
      for (const store of stores.splice(0)) await store.close();
      await disposeDefaultLlamaCpp();
    } finally {
      envKeys.forEach((key, i) => {
        if (previousEnv[i] === undefined) delete process.env[key];
        else process.env[key] = previousEnv[i];
      });
      server.closeAllConnections();
      await new Promise<void>(resolve => server.close(() => resolve()));
      rmSync(root, { recursive: true, force: true });
      expect(existsSync(root)).toBe(false);
    }
  });

  async function open(dbPath: string, docsDir: string) {
    const store = await createStore({ dbPath, config: {
      models: { embed: descriptor.model, embedBackend: "mlx", mlxUrl: `http://127.0.0.1:${port}`, mlxFallback: false },
      collections: { public: { path: docsDir, pattern: "**/*.md" } },
    } });
    stores.push(store);
    return store;
  }

  async function close(store: QMDStore) {
    await store.close();
    stores.splice(stores.indexOf(store), 1);
  }

  it("kills the real entrypoint at a committed barrier, resumes in another PID, and restores equivalent retrieval", async () => {
    const docsDir = path.join(root, "docs");
    mkdirSync(docsDir);
    const bodies = [
      "# Alpha\n\nRaft consensus elects leaders and commits replicated logs.\n",
      "# Beta\n\nSQLite transactions preserve durable database state.\n",
      `# Gamma\n\n${"Long recovery document with deterministic chunk boundaries. ".repeat(130)}\n`,
      "# Delta\n\nHTTP clients enforce bounded request deadlines.\n",
    ];
    bodies.forEach((body, i) => writeFileSync(path.join(docsDir, `${i}.md`), body));
    const expected = bodies.map(body => ({
      hash: createHash("sha256").update(body).digest("hex"),
      chunks: chunkDocument(body, CHUNK_SIZE_TOKENS * 3, CHUNK_OVERLAP_TOKENS * 3, CHUNK_WINDOW_TOKENS * 3),
    }));
    expect(expected[2]!.chunks.length).toBeGreaterThan(1);
    const dbPath = path.join(root, "shadow-recovery.sqlite");
    const common = { dbPath, docsDir, mlxPort: port, mlxModelPath: descriptor.model,
      launchToken: "offline-w13-owned", timeoutMs: 12_000, stateRoot: path.join(root, "child-state") };
    const first = await executeRecoveryWorker({ ...common, action: "index", barrierAfterDocs: 2,
      outputJson: path.join(root, "partial.json"), terminateAtBarrier: true });
    expect(first.error, first.stderr).toBeUndefined();
    expect(first.timedOut).toBe(false);
    expect(first.signal).toBe("SIGKILL");
    expect(first.exitCode).toBeNull();
    expect(first.pid).not.toBe(process.pid);
    expect(first.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, sessionTerminated: true,
      containmentVerified: true, pipesClosed: true, forcedPipeClose: false });
    expectReaped(first.pid!);
    // Production walks the durable keyset in content-hash order.
    const partialExpected = [...expected].sort((a, b) => a.hash.localeCompare(b.hash)).slice(0, 2);
    const partialCount = partialExpected.reduce((sum, doc) => sum + doc.chunks.length, 0);
    expect(first.report).toMatchObject({ status: "barrier_reached", pid: first.pid,
      docsCompleted: 2, chunksCommitted: partialCount, totalVectors: partialCount, checkpointStatus: "in_progress" });
    expect(first.report!.hashes).toEqual(partialExpected.map(doc => ({ hash: doc.hash, seqs: doc.chunks.map((_, seq) => seq) })));
    expect(JSON.parse(readFileSync(path.join(root, "partial.json"), "utf8"))).toEqual(first.report);

    const partial = await open(dbPath, docsDir);
    const partialCheckpoint = getActiveCheckpoint(partial.internal.db)!;
    expect(partialCheckpoint.status).toBe("in_progress");
    expect(partialCheckpoint.chunksCommitted).toBe(partialCount);
    expect(partialCheckpoint.jobId).toBe(first.report!.checkpointJobId);
    const committedBefore = logicalState(partial);
    expect(committedBefore.vectors).toHaveLength(partialCount);
    await close(partial);

    const resumed = await executeRecoveryWorker({ ...common, action: "resume", outputJson: path.join(root, "resumed.json") });
    expect(resumed.error, resumed.stderr).toBeUndefined();
    expect(resumed.exitCode).toBe(0);
    expect(resumed.signal).toBeNull();
    expect(resumed.pid).not.toBe(first.pid);
    expect(resumed.pid).not.toBe(process.pid);
    expect(resumed.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, sessionTerminated: true,
      containmentVerified: true, pipesClosed: true, forcedPipeClose: false });
    expectReaped(resumed.pid!);
    const expectedCount = expected.reduce((sum, doc) => sum + doc.chunks.length, 0);
    expect(resumed.report).toMatchObject({ status: "completed", pid: resumed.pid,
      totalVectors: expectedCount, checkpointStatus: null, checkpointJobId: partialCheckpoint.jobId,
      consecutiveSequenceVerified: true });

    const recovered = await open(dbPath, docsDir);
    expect(getActiveCheckpoint(recovered.internal.db)).toBeNull();
    const recoveredState = logicalState(recovered);
    expect(recoveredState.vectors).toHaveLength(expectedCount);
    expect(recoveredState.vectorBytes).toHaveLength(expectedCount);
    expect(recoveredState.vectors).toEqual(expect.arrayContaining(committedBefore.vectors));
    expect(recoveredState.vectorBytes).toEqual(expect.arrayContaining(committedBefore.vectorBytes));
    for (const doc of expected) {
      const rows = recovered.internal.db.prepare("SELECT seq, pos FROM content_vectors WHERE hash = ? ORDER BY seq").all(doc.hash);
      expect(rows).toEqual(doc.chunks.map((chunk, seq) => ({ seq, pos: chunk.pos })));
    }
    const finalCheckpoint = recovered.internal.db.prepare("SELECT status FROM indexing_checkpoints WHERE job_id = ?").get(partialCheckpoint.jobId);
    expect(finalCheckpoint).toEqual({ status: "completed" });

    saveCheckpoint(recovered.internal.db, { ...partialCheckpoint, jobId: "mismatch-owned",
      status: "in_progress", fingerprint: { ...partialCheckpoint.fingerprint, model: "wrong-model" } });
    const beforeMismatch = logicalState(recovered);
    await expect(runDurableIndexingJob(recovered.internal)).rejects.toBeInstanceOf(IndexingFingerprintMismatchError);
    expect(logicalState(recovered)).toEqual(beforeMismatch);
    recovered.internal.db.prepare("DELETE FROM indexing_checkpoints WHERE job_id = ?").run("mismatch-owned");

    async function rankings(store: QMDStore): Promise<StageMetrics> {
      const details: StageMetrics["queryDetails"] = [];
      for (const [i, query] of ["Raft consensus", "SQLite transactions", "HTTP deadlines"].entries()) {
        const hits = await store.searchVector(query, { limit: 4 });
        expect(hits).toHaveLength(4);
        expect(hits.every(hit => Number.isFinite(hit.score))).toBe(true);
        details.push({ queryId: `q${i}`, query, category: "synthetic", expectedDocIds: [],
          topDocIds: hits.map(hit => hit.docid), scores: hits.map(hit => hit.score),
          rank1: null, hitAt5: false, latencyMs: 0 });
      }
      return { recallAt1: 0, recallAt3: 0, recallAt5: 0, mrr: 0, ndcgAt5: 0,
        avgLatencyMs: 0, p50LatencyMs: 0, p95LatencyMs: 0, queryDetails: details };
    }
    const before = await rankings(recovered);
    const beforeBackup = logicalState(recovered);
    const restoredPath = path.join(root, "shadow-restored.sqlite");
    recovered.internal.db.prepare("VACUUM INTO ?").run(restoredPath);
    await close(recovered);
    const restored = await open(restoredPath, docsDir);
    expect(logicalState(restored)).toEqual(beforeBackup);
    const after = await rankings(restored);
    expect(verifyRetrievalEquivalence(before, after).identical).toBe(true);
    expect(requests).toContain("/tokenize");
    expect(requests).toContain("/embed");
    console.log(JSON.stringify({ evidence: "W13", partialPid: first.pid, partialSignal: first.signal,
      resumePid: resumed.pid, resumeExit: resumed.exitCode, reaped: true, vectors: expectedCount,
      restoreQueries: before.queryDetails.length, syntheticPort: port }));
  }, 30_000);

  it("proves integrated recovery/restore and enforces the MCP HTTP runtime contract", async () => {
    const report = await runPhase5Qualification({ skipGguf: true, skipMlx: true,
      mlxPort: port, mlxModelPath: descriptor.model, workerTimeoutMs: 12_000 });
    const bun = Boolean((process.versions as Record<string, string>).bun);
    if (bun) {
      // Production deliberately refuses Bun's uncancellable HTTP transport.
      // Prove that explicit refusal, not a fallback or fabricated green MCP probe.
      expect(report.errors).toEqual([expect.stringMatching(/^QMD MCP HTTP requires Node\.js >=22:/)]);
      expect(report.status).toBe("failed");
      expect(report.checks.mcpTransportProbePassed).toBe(false);
    } else {
      expect(report.errors).toEqual([]);
      expect(report.status, JSON.stringify(report)).toBe("passed");
      expect(report.checks.mcpTransportProbePassed).toBe(true);
    }
    expect(report.checks).toMatchObject({ durableIndexingCancellation: true, durableIndexingResumption: true,
      vectorDeduplicationReconciled: true, fingerprintMismatchRefused: true,
      sqliteOnlineBackupRestored: true, restoredRetrievalIdentical: true });
    expect(report.recovery.logicalStateInvariantOnMismatch).toBe(true);
    expect(report.recovery.workers).toHaveLength(2);
    for (const worker of report.recovery.workers) expectReaped(worker.pid);
    console.log(JSON.stringify({ evidence: "W13-integrated-synthetic", status: report.status,
      mcpHttpRuntimeSupported: !bun,
      workers: report.recovery.workers, restoredDocuments: report.recovery.backupDocCount,
      restoredVectors: report.recovery.backupVectorCount }));
  }, 30_000);

  it("bounds a stalled child, reports failure instead of recovery, and reaps its exact PID", async () => {
    const result = await executeRecoveryWorker({ dbPath: path.join(root, "shadow-unused.sqlite"),
      docsDir: root, mlxPort: port, launchToken: "owned", action: "resume",
      outputJson: path.join(root, "absent.json"), stateRoot: path.join(root, "stall-state"),
      timeoutMs: 200, command: [process.execPath, "-e", "setInterval(() => {}, 1000)"] });
    expect(result.timedOut).toBe(true);
    expect(result.error).toMatch(/timed out/i);
    expect(result.report).toBeUndefined();
    expect(result.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, pipesClosed: true, forcedPipeClose: false });
    expectReaped(result.pid!);
    console.log(JSON.stringify({ evidence: "W13-timeout", pid: result.pid, signal: result.signal, reaped: true }));
  });

  it("refuses malformed barrier options without opening a database or contacting inference", async () => {
    const result = await executeRecoveryWorker({ dbPath: path.join(root, "shadow-refused.sqlite"),
      docsDir: root, mlxPort: port, launchToken: "owned", barrierAfterDocs: 0,
      action: "index", outputJson: path.join(root, "invalid.json"),
      stateRoot: path.join(root, "invalid-state"), timeoutMs: 12_000 });
    expect(result.exitCode).toBe(1);
    expect(result.report).toMatchObject({ status: "failed", error: expect.stringMatching(/barrierAfterDocs.*positive integer/) });
    expect(existsSync(path.join(root, "shadow-refused.sqlite"))).toBe(false);
    expect(requests).toEqual([]);
    expectReaped(result.pid!);
  });

  it("bounds a spawn failure without waiting for a nonexistent exit event", async () => {
    const started = performance.now();
    const result = await executeRecoveryWorker({ dbPath: path.join(root, "shadow-no-spawn.sqlite"),
      outputJson: path.join(root, "spawn-failure.json"), stateRoot: path.join(root, "spawn-failure-state"),
      timeoutMs: 200, command: [path.join(root, "nonexistent-executable")] });
    expect(result.error).toMatch(/spawn failed/);
    expect(result.pid).toBeUndefined();
    expect(performance.now() - started).toBeLessThan(1_000);
    expect(result.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, pipesClosed: true, forcedPipeClose: false });
    expect(requests).toEqual([]);
  });

  it.each([
    { mutation: { checkpointStatus: "in_progress" }, reason: /Completed worker still has an active checkpoint/ },
    { mutation: { pid: -1 }, reason: /Worker report identity mismatch/ },
    { mutation: { action: "resume" }, reason: /Worker report identity mismatch/ },
    { mutation: { error: "synthetic cleanup failure" }, reason: /synthetic cleanup failure/ },
    { mutation: { error: null }, reason: /Worker report error must be a string/ },
  ])("refuses a zero-exit report with invalid identity/final state or cleanup error: %j", async ({ mutation, reason }) => {
    const dbPath = path.join(root, "shadow-authored-report.sqlite");
    const outputJson = path.join(root, "authored-report.json");
    const report = { status: "completed", dbPath, action: "index", docsCompleted: 1, chunksCommitted: 1,
      totalVectors: 1, checkpointStatus: null, consecutiveSequenceVerified: true, hashes: [] };
    const result = await executeRecoveryWorker({ dbPath, outputJson, stateRoot: path.join(root, "authored-state"), timeoutMs: 2_000,
      command: [process.execPath, "-e", `require('node:fs').writeFileSync(${JSON.stringify(outputJson)},
        JSON.stringify({ ...${JSON.stringify(report)}, pid: process.pid, ...${JSON.stringify(mutation)} }));`] });
    expect(result.exitCode).toBe(0);
    expect(result.error).toMatch(reason);
    expect(result.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, pipesClosed: true, forcedPipeClose: false });
    expectReaped(result.pid!);
    expect(requests).toEqual([]);
  });
});
