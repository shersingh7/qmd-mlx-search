/** Offline public-fixture, checkpoint, and restore-equivalence unit tests. */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { PUBLIC_CORPUS_DOCS, PUBLIC_HELD_OUT_QUERIES, computeIRMetrics } from "./fixtures/public_eval_dataset.js";
import { createStore, runDurableIndexingJob, IndexingFingerprintMismatchError, getActiveCheckpoint, saveCheckpoint, type QMDStore } from "../src/index.js";
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp } from "../src/llm.js";
import { verifyRetrievalEquivalence, type StageMetrics } from "../scripts/phase5_e2e_runner.js";

describe("Phase 5 Public Fixtures & Recovery Unit Tests", () => {
  it("public fixture corpus has 16 distinct documents with self-contained technical facts", () => {
    expect(PUBLIC_CORPUS_DOCS.length).toBe(16);
    expect(new Set(PUBLIC_CORPUS_DOCS.map(d => d.id)).size).toBe(16);
    for (const doc of PUBLIC_CORPUS_DOCS) {
      expect(doc.content.length).toBeGreaterThan(100);
      expect(doc.title.length).toBeGreaterThan(5);
    }
  });

  it("public held-out queries have 10 frozen judged queries with explicit expected doc IDs", () => {
    expect(PUBLIC_HELD_OUT_QUERIES.length).toBe(10);
    const ids = new Set(PUBLIC_CORPUS_DOCS.map(d => d.id));
    for (const q of PUBLIC_HELD_OUT_QUERIES) {
      expect(q.relevantDocIds.length).toBeGreaterThanOrEqual(1);
      for (const id of q.relevantDocIds) expect(ids.has(id)).toBe(true);
    }
  });

  it("computeIRMetrics calculates Recall@k, MRR, and nDCG@5 for synthetic rankings", () => {
    const relevant = [["doc-1"], ["doc-3"]];
    const perfect = computeIRMetrics([["doc-1", "doc-2"], ["doc-3", "doc-4"]], relevant, 5);
    expect(perfect.recallAt1).toBe(1);
    expect(perfect.recallAt5).toBe(1);
    expect(perfect.mrr).toBe(1);
    expect(perfect.ndcgAt5).toBe(1);
    const second = computeIRMetrics([["doc-other", "doc-1"], ["doc-other", "doc-3"]], relevant, 5);
    expect(second.recallAt1).toBe(0);
    expect(second.recallAt3).toBe(1);
    expect(second.recallAt5).toBe(1);
    expect(second.mrr).toBe(0.5);
    expect(second.ndcgAt5).toBeGreaterThan(0);
    expect(second.ndcgAt5).toBeLessThan(1);
  });

  describe("Isolated Shadow Database Recovery Lifecycle", () => {
    let tempDir: string;
    let store: QMDStore;
    const mockDescriptor = { version: 1 as const, backend: "mlx", model: "base-embed",
      pooling: "mean", outputDimensions: 4, nativeDimensions: 4, maxTokens: 128, normalized: true };

    beforeEach(async () => {
      tempDir = mkdtempSync(path.join(tmpdir(), "qmd-phase5-unit-"));
      // These unit tests have no network or model dependency, including the
      // legacy global tokenizer used by chunkDocumentByTokens.
      vi.spyOn(getDefaultLlamaCpp(), "mlxTokenize").mockResolvedValue(null);
      vi.spyOn(getDefaultLlamaCpp(), "tokenize").mockImplementation(async text =>
        Array.from({ length: Math.ceil(text.length / 4) }, (_, i) => i) as any);
      store = await createStore({ dbPath: path.join(tempDir, "shadow-test.sqlite"),
        config: { collections: { test: { path: tempDir, pattern: "**/*.md" } } } });
      store.internal.llm = {
        embedModelName: "base-embed", getDescriptor: async () => mockDescriptor,
        tokenize: async (text: string) => Array.from({ length: Math.ceil(text.length / 4) }, (_, i) => i),
        mlxTokenize: async (texts: string[]) => texts.map(text => Math.ceil(text.length / 4)),
        embed: async () => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "base-embed" }),
        embedBatch: async (texts: string[]) => texts.map(() => ({ embedding: [0.1, 0.2, 0.3, 0.4], model: "base-embed" })),
      } as any;
    });

    afterEach(async () => {
      await store.close();
      vi.restoreAllMocks();
      await disposeDefaultLlamaCpp();
      rmSync(tempDir, { recursive: true, force: true });
    });

    it("handles durable indexing cancellation and marks checkpoint cancelled", async () => {
      writeFileSync(path.join(tempDir, "test1.md"), "# Doc 1\n\nContent paragraph 1 for test\n");
      writeFileSync(path.join(tempDir, "test2.md"), "# Doc 2\n\nContent paragraph 2 for test\n");
      await store.update();
      const controller = new AbortController();
      const result = await runDurableIndexingJob(store.internal, { signal: controller.signal, maxDocsPerBatch: 1,
        onProgress: info => { if (info.chunksEmbedded > 0) controller.abort(); } });
      expect(result.status).toBe("cancelled");
      expect(getActiveCheckpoint(store.internal.db)?.status).toBe("cancelled");
      // Cancellation can unwind the inference session's result counters after
      // SQLite already committed a batch. Check durable rows, not that return value.
      expect((store.internal.db.prepare("SELECT COUNT(*) AS count FROM content_vectors").get() as { count: number }).count).toBeGreaterThan(0);
    });

    it("fingerprint mismatch refuses resume without changing existing logical state or vector bytes", async () => {
      writeFileSync(path.join(tempDir, "doc-keep.md"), "# Keep\n\nContent to preserve unchanged\n");
      await store.update();
      const seeded = await runDurableIndexingJob(store.internal, { model: "base-embed" });
      expect(seeded.status, JSON.stringify(seeded)).toBe("complete");
      saveCheckpoint(store.internal.db, {
        jobId: "mismatch-invariance-test", fingerprint: { model: "other-model", dimensions: 512,
          chunkStrategy: "regex", descriptorSignature: "sig-other" },
        startedAt: new Date().toISOString(), updatedAt: new Date().toISOString(), status: "in_progress",
        docsTotal: 5, docsCompleted: 1, chunksTotal: 10, chunksCommitted: 1,
      });
      const state = () => ({
        content: store.internal.db.prepare("SELECT * FROM content ORDER BY hash").all(),
        documents: store.internal.db.prepare("SELECT * FROM documents ORDER BY collection, path").all(),
        vectors: store.internal.db.prepare("SELECT * FROM content_vectors ORDER BY hash, seq").all(),
        vectorBytes: store.internal.db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec ORDER BY hash_seq").all(),
        checkpoints: store.internal.db.prepare("SELECT * FROM indexing_checkpoints ORDER BY job_id").all(),
        expectations: store.internal.db.prepare("SELECT * FROM content_chunk_expectations ORDER BY hash, strategy").all(),
        config: store.internal.db.prepare("SELECT * FROM store_config ORDER BY key").all(),
      });
      const before = state();
      expect(before.vectors).toHaveLength(1);
      await expect(runDurableIndexingJob(store.internal, { model: "base-embed" })).rejects.toThrow(IndexingFingerprintMismatchError);
      expect(state()).toEqual(before);
    });
  });

  describe("Retrieval Equivalence Verification Helper", () => {
    function stage(): StageMetrics {
      return { recallAt1: 1, recallAt3: 1, recallAt5: 1, mrr: 1, ndcgAt5: 1,
        avgLatencyMs: 12, p50LatencyMs: 10, p95LatencyMs: 15,
        queryDetails: [{ queryId: "Q01", query: "test query 1", category: "easy",
          expectedDocIds: ["doc-1"], topDocIds: ["doc-1", "doc-2"], scores: [0.9, 0.6],
          rank1: 1, hitAt5: true, latencyMs: 10 }] };
    }

    it("passes identical rankings with actual finite scores, ignoring timing", () => {
      const pre = stage();
      const post = structuredClone(pre);
      post.queryDetails[0]!.latencyMs = 100;
      expect(verifyRetrievalEquivalence(pre, post)).toMatchObject({ identical: true,
        details: [{ docOrderMatch: true, scoreToleranceMatch: true }] });
    });

    it("accepts score drift within tolerance and refuses drift outside it", () => {
      const pre = stage();
      const post = stage();
      post.queryDetails[0]!.scores[0]! += 0.00001;
      expect(verifyRetrievalEquivalence(pre, post).identical).toBe(true);
      post.queryDetails[0]!.scores[0]! += 0.01;
      expect(verifyRetrievalEquivalence(pre, post).identical).toBe(false);
    });

    it("refuses changed rankings with otherwise valid scores", () => {
      const pre = stage();
      const post = stage();
      post.queryDetails[0]!.topDocIds.reverse();
      post.queryDetails[0]!.rank1 = 2;
      expect(verifyRetrievalEquivalence(pre, post)).toMatchObject({ identical: false,
        details: [{ docOrderMatch: false }] });
    });

    // Register named zero-argument tests instead of bare .each array rows.
    // [] is an empty argument tuple in Bun, which treats the callback's lone
    // parameter as `done` and waits forever; other arrays are spread scalars.
    for (const { label, scores } of [
      { label: "missing", scores: undefined }, { label: "empty", scores: [] },
      { label: "too few", scores: [0.9] }, { label: "too many", scores: [0.9, 0.6, 0.2] },
      { label: "NaN", scores: [NaN, 0.6] }, { label: "positive infinity", scores: [Infinity, 0.6] },
      { label: "negative infinity", scores: [-Infinity, 0.6] },
    ]) {
      it(`refuses ${label} scores in either or both retrieval reports`, () => {
        for (const invalidSide of ["pre", "post", "both"]) {
          const pre = stage();
          const post = stage();
          if (invalidSide !== "post") (pre.queryDetails[0] as any).scores = scores;
          if (invalidSide !== "pre") (post.queryDetails[0] as any).scores = scores;
          expect(verifyRetrievalEquivalence(pre, post).identical).toBe(false);
        }
      });
    }

    it("refuses duplicate query IDs even when both reports repeat identical successful queries", () => {
      const pre = stage();
      pre.queryDetails.push(structuredClone(pre.queryDetails[0]!));
      expect(verifyRetrievalEquivalence(pre, structuredClone(pre)).identical).toBe(false);
    });

    it("refuses empty or different query sets", () => {
      const pre = stage();
      const post = stage();
      post.queryDetails[0]!.queryId = "Q02";
      expect(verifyRetrievalEquivalence(pre, post).identical).toBe(false);
      post.queryDetails = [];
      expect(verifyRetrievalEquivalence(pre, post).identical).toBe(false);
      expect(verifyRetrievalEquivalence(post, post).identical).toBe(false);
    });
  });
});
