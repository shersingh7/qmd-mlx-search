import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createStore, generateEmbeddings, insertEmbedding, chunkDocumentByTokens,
  deleteStoreCollection, renameStoreCollection, upsertStoreCollection,
  hybridQuery, type Store,
} from "../src/store.js";
import { runDurableIndexingJob, IndexingFingerprintMismatchError } from "../src/indexing/job.js";
import { LlamaCpp, setDefaultLlamaCpp } from "../src/llm.js";

const now = "2026-01-01T00:00:00.000Z";
const stores: Store[] = [];
function fixture() {
  const store = createStore(":memory:");
  stores.push(store);
  store.llm = backend() as any;
  return store;
}
function backend(model = "mock-a", dimensions = 4) {
  return {
    embedModelName: model,
    getDescriptor: vi.fn(async () => ({ version: 1, backend: "mlx", pooling: "mean", model, nativeDimensions: dimensions, outputDimensions: dimensions, maxTokens: 2048, normalized: true })),
    tokenize: vi.fn(async (text: string) => Array.from({ length: Math.max(1, Math.ceil(text.length / 8)) }, (_, i) => i)),
    embed: vi.fn(async () => ({ embedding: [1, ...Array(dimensions - 1).fill(0)], model })),
    embedBatch: vi.fn(async (texts: string[]) => texts.map(() => ({ embedding: [1, ...Array(dimensions - 1).fill(0)], model }))),
  };
}
function doc(store: Store, collection: string, path: string, hash: string, body: string) {
  store.insertContent(hash, body, now);
  store.insertDocument(collection, path, path, hash, now, now);
}
function logicalState(store: Store) {
  return {
    metadata: store.db.prepare("SELECT * FROM store_config ORDER BY key").all(),
    chunks: store.db.prepare("SELECT * FROM content_vectors ORDER BY hash, seq").all(),
    vectors: store.db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec ORDER BY hash_seq").all(),
    expectations: store.db.prepare("SELECT * FROM content_chunk_expectations ORDER BY hash, strategy").all(),
    checkpoints: store.db.prepare("SELECT * FROM indexing_checkpoints ORDER BY job_id").all(),
  };
}
afterEach(() => {
  for (const store of stores.splice(0)) store.close();
  setDefaultLlamaCpp(null);
  vi.restoreAllMocks();
});

describe("store audit: rebuild and chunk identity", () => {
  it("atomically replaces both vector space metadata and vectors on a forced model/dimension migration", async () => {
    const store = fixture();
    doc(store, "docs", "one.md", "hash-one", "synthetic replacement body");
    expect((await runDurableIndexingJob(store)).status).toBe("complete");
    store.llm = backend("mock-b", 6) as any;
    const rebuilt = await runDurableIndexingJob(store, { force: true });
    expect(rebuilt.status).toBe("complete");
    expect(store.db.prepare("SELECT DISTINCT model FROM content_vectors").all()).toEqual([{ model: "mock-b" }]);
    expect(store.db.prepare("SELECT value FROM store_config WHERE key='embedding_model'").get()).toEqual({ value: "mock-b" });
    expect(store.db.prepare("SELECT length(embedding) AS bytes FROM vectors_vec").get()).toEqual({ bytes: 24 });
  });

  it.each(["descriptor", "inference", "database", "checkpoint", "cancel"])("preserves the complete prior logical index after a forced %s failure", async (failure) => {
    const store = fixture();
    for (let i = 0; i < 3; i++) doc(store, "docs", `${i}.md`, `hash-${i}`, `synthetic old body ${i}`);
    expect((await runDurableIndexingJob(store)).status).toBe("complete");
    const before = logicalState(store);
    const next = backend("mock-b", 6);
    const controller = new AbortController();
    if (failure === "descriptor") next.getDescriptor.mockRejectedValue(new Error("backend unavailable"));
    if (failure === "inference") {
      next.embedBatch.mockRejectedValue(new Error("backend failed"));
      next.embed.mockRejectedValue(new Error("backend failed"));
    }
    if (failure === "database") store.db.exec("CREATE TRIGGER reject_vector BEFORE INSERT ON content_vectors BEGIN SELECT RAISE(FAIL, 'write blocked'); END");
    if (failure === "checkpoint") store.db.exec("CREATE TRIGGER reject_checkpoint BEFORE INSERT ON indexing_checkpoints BEGIN SELECT RAISE(FAIL, 'checkpoint blocked'); END");
    if (failure === "cancel") {
      let calls = 0;
      next.embedBatch.mockImplementation(async (texts: string[]) => {
        if (++calls === 2) controller.abort();
        return texts.map(() => ({ embedding: [1, 0, 0, 0, 0, 0], model: "mock-b" }));
      });
    }
    store.llm = next as any;
    if (failure === "descriptor") {
      await expect(runDurableIndexingJob(store, { force: true })).rejects.toThrow("backend unavailable");
    } else if (failure === "checkpoint") {
      await expect(runDurableIndexingJob(store, { force: true })).rejects.toThrow("checkpoint blocked");
    } else {
      const result = await runDurableIndexingJob(store, { force: true, maxDocsPerBatch: 1, signal: controller.signal });
      expect(result.status).toBe(failure === "cancel" ? "cancelled" : "failed");
      expect(result.chunksCommitted).toBe(0);
    }
    expect(logicalState(store)).toEqual(before);
    expect(await store.searchVec("old body", "mock-a", 3, undefined, undefined, [1, 0, 0, 0])).toHaveLength(3);
  });

  it("refuses a completed regex→auto change without mutation, then force rebuilds using real auto positions", async () => {
    const store = fixture();
    const body = Array.from({ length: 24 }, (_, i) => `export function fn${i}() {\n  const text = '${"synthetic payload ".repeat(35)}';\n  return text;\n}\n`).join("\n");
    const regex = await chunkDocumentByTokens(body, undefined, undefined, undefined, "source.ts", "regex", undefined, store.llm);
    const auto = await chunkDocumentByTokens(body, undefined, undefined, undefined, "source.ts", "auto", undefined, store.llm);
    expect(auto.map(c => c.pos)).not.toEqual(regex.map(c => c.pos));
    doc(store, "docs", "source.ts", "strategy-hash", body);
    expect((await runDurableIndexingJob(store, { chunkStrategy: "regex" })).status).toBe("complete");
    // Completed chunk identity must not depend on any surviving job record.
    store.db.exec("DELETE FROM indexing_checkpoints");
    const before = logicalState(store);
    await expect(runDurableIndexingJob(store, { chunkStrategy: "auto" })).rejects.toBeInstanceOf(IndexingFingerprintMismatchError);
    await expect(generateEmbeddings(store, { chunkStrategy: "auto" })).rejects.toThrow("Chunk strategy mismatch");
    expect(logicalState(store)).toEqual(before);
    expect((await runDurableIndexingJob(store, { chunkStrategy: "auto", force: true })).status).toBe("complete");
    expect(store.db.prepare("SELECT pos FROM content_vectors ORDER BY seq").all().map(r => r.pos)).toEqual(auto.map(c => c.pos));
    expect(store.db.prepare("SELECT DISTINCT strategy FROM content_chunk_expectations").all()).toEqual([{ strategy: "auto" }]);
  });

  it("failed low-level replacement and insertion never publish false completion metadata", async () => {
    const store = fixture();
    doc(store, "docs", "one.md", "hash-one", "synthetic body");
    store.ensureVecTable(4);
    expect(() => insertEmbedding(store.db, "hash-one", 0, 0, new Float32Array([1, 0]), "mock-a", now)).toThrow();
    expect(store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
    expect(store.getHashesNeedingEmbedding()).toBe(1);
    expect((await generateEmbeddings(store)).status).toBe("complete");
    const before = logicalState(store);
    expect(() => insertEmbedding(store.db, "hash-one", 0, 99, new Float32Array([1, 0]), "wrong", now)).toThrow();
    expect(logicalState(store)).toEqual(before);
  });

  it("refuses unsafe legacy vector schema migration without erasing its rows", () => {
    const store = fixture();
    store.db.exec("CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[4])");
    store.db.prepare("INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)").run("legacy_0", new Float32Array([1, 0, 0, 0]));
    expect(() => store.ensureVecTable(4)).toThrow("schema mismatch");
    expect(store.db.prepare("SELECT hash_seq FROM vectors_vec").all()).toEqual([{ hash_seq: "legacy_0" }]);
  });

  it("resets stale vectors and embedding metadata consistently for an empty forced rebuild", async () => {
    const store = fixture();
    doc(store, "docs", "one.md", "hash-one", "synthetic body");
    await runDurableIndexingJob(store);
    store.deactivateDocument("docs", "one.md");
    store.llm = backend("mock-b", 6) as any;
    expect((await runDurableIndexingJob(store, { force: true })).status).toBe("complete");
    expect(store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
    expect(store.db.prepare("SELECT * FROM content_chunk_expectations").all()).toEqual([]);
    expect(store.db.prepare("SELECT hash_seq FROM vectors_vec").all()).toEqual([]);
    expect(store.db.prepare("SELECT value FROM store_config WHERE key='embedding_model'").get()).toEqual({ value: "mock-b" });
  });
});

describe("store audit: incremental bounded jobs and own inference", () => {
  it("a durable no-op selects zero hashes, loads zero bodies, and performs zero tokenization/inference", async () => {
    const store = fixture();
    for (let i = 0; i < 19; i++) doc(store, "docs", `${i}.md`, `hash-${i}`, `synthetic ${i}`);
    expect((await runDurableIndexingJob(store, { maxDocsPerBatch: 3 })).docsProcessed).toBe(19);
    const own = store.llm as any;
    own.tokenize.mockClear(); own.embed.mockClear(); own.embedBatch.mockClear();
    const prepare = store.db.prepare.bind(store.db);
    const bodyQueries: string[] = [];
    vi.spyOn(store.db, "prepare").mockImplementation((sql: string) => {
      if (/doc as body/i.test(sql)) bodyQueries.push(sql);
      return prepare(sql);
    });
    const result = await runDurableIndexingJob(store, { maxDocsPerBatch: 3 });
    expect(result).toMatchObject({ status: "complete", docsSelected: 0, docsProcessed: 0, chunksAttempted: 0 });
    expect(bodyQueries).toEqual([]);
    expect(own.tokenize).not.toHaveBeenCalled();
    expect(own.embed).not.toHaveBeenCalled();
    expect(own.embedBatch).not.toHaveBeenCalled();
  });

  it("keyset-pages only pending hashes and respects both metadata/body page bounds", async () => {
    const store = fixture();
    for (let i = 0; i < 41; i++) doc(store, "docs", `${i}.md`, `hash-${i.toString().padStart(3, "0")}`, `synthetic document ${i}`);
    const prepare = store.db.prepare.bind(store.db);
    const metadataPageSizes: number[] = [];
    const bodyPageSizes: number[] = [];
    vi.spyOn(store.db, "prepare").mockImplementation((sql: string) => {
      const statement = prepare(sql);
      if (/WITH pending AS MATERIALIZED/.test(sql) && /LIMIT \?/.test(sql)) {
        const all = statement.all.bind(statement);
        statement.all = (...params: any[]) => { const rows = all(...params); metadataPageSizes.push(rows.length); return rows; };
      }
      if (/doc as body/i.test(sql)) {
        const all = statement.all.bind(statement);
        statement.all = (...params: any[]) => { const rows = all(...params); bodyPageSizes.push(rows.length); return rows; };
      }
      return statement;
    });
    const result = await runDurableIndexingJob(store, { maxDocsPerBatch: 5, maxBatchBytes: 45 });
    expect(result).toMatchObject({ status: "complete", docsSelected: 41, docsProcessed: 41, chunksCommitted: 41 });
    expect(Math.max(...metadataPageSizes)).toBeLessThanOrEqual(5);
    expect(Math.max(...bodyPageSizes)).toBeLessThanOrEqual(2);
    doc(store, "docs", "new.md", "hash-new", "newly pending synthetic document");
    expect((await runDurableIndexingJob(store)).docsSelected).toBe(1);
  });

  it("bound vector calls and indexing tokenization never escape to a global backend", async () => {
    const store = fixture();
    doc(store, "docs", "one.md", "hash-one", "own tokenizer and own vector body");
    const wrong = backend("GLOBAL");
    wrong.tokenize.mockRejectedValue(new Error("global tokenizer must not run"));
    wrong.embed.mockRejectedValue(new Error("global embed must not run"));
    setDefaultLlamaCpp(wrong as any);
    expect((await generateEmbeddings(store)).status).toBe("complete");
    expect(await store.searchVec("query", "mock-a")).toHaveLength(1);
    expect((store.llm as any).tokenize).toHaveBeenCalled();
    expect((store.llm as any).embed).toHaveBeenCalledTimes(1);
    expect(wrong.tokenize).not.toHaveBeenCalled();
    expect(wrong.embed).not.toHaveBeenCalled();
  });
});

describe("store audit: trustworthy inference cache identities", () => {
  it("does not trust legacy rows whose predicted namespace could contain fallback output", async () => {
    const store = fixture();
    store.setCachedResult(store.getCacheKey("rerank", { query: "q", model: "mlx:rank-a", chunk: "body" }), "0.99");
    store.setCachedResult(store.getCacheKey("expandQuery", { query: "q", model: "mlx:generator-a" }), JSON.stringify([{ type: "vec", query: "poisoned legacy expansion" }]));
    store.llm = {
      getCacheIdentity: async (op: string) => op === "rerank" ? "mlx:rank-a" : "mlx:generator-a",
      rerank: async () => ({ results: [{ file: "one", score: 0.2, index: 0 }], model: "mlx:rank-a" }),
      expandQueryWithIdentity: async () => ({ queries: [{ type: "vec", text: "fresh expansion" }], model: "mlx:generator-a" }),
    } as any;
    expect(await store.rerank("q", [{ file: "one", text: "body" }])).toEqual([{ file: "one", score: 0.2 }]);
    expect(await store.expandQuery("q")).toEqual([{ type: "vec", query: "fresh expansion" }]);
  });

  it("isolates actual configured GGUF models rather than caller/default aliases", async () => {
    const store = fixture();
    const a = new LlamaCpp({ embedBackend: "gguf", generateModel: "gguf:generator-a", rerankModel: "gguf:reranker-a" });
    const b = new LlamaCpp({ embedBackend: "gguf", generateModel: "gguf:generator-b", rerankModel: "gguf:reranker-b" });
    const aExpand = vi.fn(async () => ({ queries: [{ type: "vec", text: "model-a expansion" }], model: "gguf:generator-a" }));
    const bExpand = vi.fn(async () => ({ queries: [{ type: "vec", text: "model-b expansion" }], model: "gguf:generator-b" }));
    (a as any).expandQueryWithIdentity = aExpand;
    (b as any).expandQueryWithIdentity = bExpand;
    const rankMock = (model: string, score: number) => vi.fn(async (_query: string, docs: { file: string }[]) => ({ results: docs.map((doc, index) => ({ file: doc.file, index, score })), model }));
    const aRank = rankMock("gguf:reranker-a", 0.1);
    const bRank = rankMock("gguf:reranker-b", 0.9);
    a.rerank = aRank; b.rerank = bRank;
    try {
      const docs = [{ file: "one", text: "same synthetic chunk" }];
      store.llm = a;
      expect(await store.expandQuery("q")).toEqual([{ type: "vec", query: "model-a expansion" }]);
      expect((await store.rerank("q", docs))[0]?.score).toBe(0.1);
      store.llm = b;
      expect(await store.expandQuery("q")).toEqual([{ type: "vec", query: "model-b expansion" }]);
      expect((await store.rerank("q", docs))[0]?.score).toBe(0.9);
      store.llm = a;
      await store.expandQuery("q"); await store.rerank("q", docs);
      expect(aExpand).toHaveBeenCalledTimes(1); expect(aRank).toHaveBeenCalledTimes(1);
      expect(bExpand).toHaveBeenCalledTimes(1); expect(bRank).toHaveBeenCalledTimes(1);
    } finally { await a.dispose(); await b.dispose(); }
  });

  it("recomputes every document when a partial MLX cache hit is followed by actual GGUF scores", async () => {
    const store = fixture();
    const old = { file: "old", text: "old cached synthetic chunk" };
    const fresh = { file: "fresh", text: "new synthetic chunk" };
    const mlxKey = (text: string) => store.getCacheKey("rerank", { cacheVersion: 2, query: "q", model: "mlx:rank-a", chunk: text });
    const ggufKey = (text: string) => store.getCacheKey("rerank", { cacheVersion: 2, query: "q", model: "gguf:configured-rank", chunk: text });
    store.setCachedResult(mlxKey(old.text), "0.99");
    const rank = vi.fn(async (_query: string, docs: typeof old[]) => ({ results: docs.map((doc, index) => ({ file: doc.file, index, score: 0.2 })), model: "gguf:configured-rank" }));
    store.llm = { getCacheIdentity: async () => "mlx:rank-a", rerank: rank } as any;
    expect(await store.rerank("q", [old, fresh])).toEqual([{ file: "old", score: 0.2 }, { file: "fresh", score: 0.2 }]);
    expect(rank.mock.calls.map(args => args[1])).toEqual([[fresh], [old, fresh]]);
    expect(store.getCachedResult(mlxKey(fresh.text))).toBeNull();
    expect(store.getCachedResult(ggufKey(old.text))).toBe("0.2");
    expect(store.getCachedResult(ggufKey(fresh.text))).toBe("0.2");
  });

  it("writes expansions under the actual serving identity and bypasses unprovable cache reads", async () => {
    const store = fixture();
    let predicted: string | null = "mlx:generator-a";
    const expand = vi.fn(async () => ({ queries: [{ type: "vec", text: "actual expansion" }], model: "gguf:configured-generator" }));
    store.llm = { getCacheIdentity: async () => predicted, expandQueryWithIdentity: expand } as any;
    await store.expandQuery("q");
    const key = (model: string | null) => store.getCacheKey("expandQuery", { cacheVersion: 2, query: "q", model });
    expect(store.getCachedResult(key("mlx:generator-a"))).toBeNull();
    expect(store.getCachedResult(key("gguf:configured-generator"))).toBe(JSON.stringify([{ type: "vec", query: "actual expansion" }]));
    predicted = "gguf:configured-generator";
    await store.expandQuery("q");
    expect(expand).toHaveBeenCalledTimes(1);
    predicted = null;
    store.setCachedResult(key(null), JSON.stringify([{ type: "vec", query: "untrusted null namespace" }]));
    expect(await store.expandQuery("q")).toEqual([{ type: "vec", query: "actual expansion" }]);
    expect(expand).toHaveBeenCalledTimes(2);
  });

  it("still accepts simple legacy mocks without the new identity helpers", async () => {
    const store = fixture();
    store.llm = {
      expandQuery: async () => [{ type: "vec", text: "legacy expansion" }],
      rerank: async (_query: string, docs: { file: string }[]) => ({ results: docs.map(doc => ({ file: doc.file, score: 0.3 })) }),
    } as any;
    expect(await store.expandQuery("q")).toEqual([{ type: "vec", query: "legacy expansion" }]);
    expect(await store.rerank("q", [{ file: "one", text: "body" }])).toEqual([{ file: "one", score: 0.3 }]);
  });
});

describe("store audit: scoped retrieval and collection ownership", () => {
  it("expands KNN through duplicate chunks and joins their metadata by its indexed key", async () => {
    const store = fixture();
    doc(store, "docs", "long.md", "long_hash", "many synthetic chunks");
    doc(store, "docs", "other.md", "other_hash", "other synthetic document");
    store.ensureVecTable(4);
    for (let i = 0; i < 8; i++) insertEmbedding(store.db, "long_hash", i, i * 10, new Float32Array([1, 0, 0, 0]), "mock-a", now);
    insertEmbedding(store.db, "other_hash", 0, 0, new Float32Array([0, 1, 0, 0]), "mock-a", now);
    const prepare = store.db.prepare.bind(store.db);
    const plans: string[] = [];
    vi.spyOn(store.db, "prepare").mockImplementation((sql: string) => {
      const statement = prepare(sql);
      if (/WITH requested\(hash, seq, hash_seq\)/.test(sql)) {
        const all = statement.all.bind(statement);
        statement.all = (...params: any[]) => {
          plans.push(...prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...params).map(row => row.detail));
          return all(...params);
        };
      }
      return statement;
    });
    const result = await store.searchVec("query", "mock-a", 2, undefined, undefined, [1, 0, 0, 0]);
    expect(result.map(r => r.filepath)).toEqual(["qmd://docs/long.md", "qmd://docs/other.md"]);
    expect(plans.some(plan => /SEARCH cv USING INDEX.*hash=\? AND seq=\?/.test(plan))).toBe(true);
    expect(plans.some(plan => /SCAN cv\b/.test(plan))).toBe(false);
  });
  it("ranks FTS inside the scope even when the only matching document falls below a global 10× prefix", () => {
    const store = fixture();
    for (let i = 0; i < 31; i++) doc(store, "noise", `${i}.md`, `noise-${i}`, "raretarget");
    doc(store, "target", "wanted.md", "target-hash", `raretarget ${"padding ".repeat(1000)}`);
    expect(store.searchFTS("raretarget", 1, "target").map(r => r.filepath)).toEqual(["qmd://target/wanted.md"]);
  });

  it("combines and deduplicates singular/plural hybrid scopes without noise leakage", async () => {
    const store = fixture();
    for (const scope of ["a", "b", "noise"]) doc(store, scope, `${scope}.md`, `hash-${scope}`, `multiscope match ${scope}`);
    store.expandQuery = vi.fn(async () => []);
    const search = vi.spyOn(store, "searchFTS");
    const result = await hybridQuery(store, "multiscope", { collection: "a", collections: ["b", "a", "b"], skipRerank: true });
    expect(result.map(r => r.file).sort()).toEqual(["qmd://a/a.md", "qmd://b/b.md"]);
    expect(search.mock.calls.map(args => args[2])).toEqual(["a", "b"]);
  });

  it("renames indexed ownership, contexts and FTS atomically, then deletes only collection-owned rows", () => {
    const store = fixture();
    upsertStoreCollection(store.db, "old", { path: "/synthetic/old", context: { "/": "owned context" } });
    upsertStoreCollection(store.db, "other", { path: "/synthetic/other" });
    doc(store, "old", "one.md", "shared-hash", "ownershipneedle shared body");
    doc(store, "other", "one.md", "shared-hash", "ownershipneedle shared body");
    store.ensureVecTable(4);
    insertEmbedding(store.db, "shared-hash", 0, 0, new Float32Array([1, 0, 0, 0]), "mock-a", now);
    expect(renameStoreCollection(store.db, "old", "new")).toBe(true);
    expect(store.searchFTS("ownershipneedle", 5, "old")).toEqual([]);
    expect(store.searchFTS("ownershipneedle", 5, "new")[0]?.context).toBe("owned context");
    expect(deleteStoreCollection(store.db, "new")).toBe(true);
    expect(store.searchFTS("ownershipneedle").map(r => r.filepath)).toEqual(["qmd://other/one.md"]);
    expect(store.db.prepare("SELECT hash FROM content").all()).toEqual([{ hash: "shared-hash" }]);
    expect(store.db.prepare("SELECT hash FROM content_vectors").all()).toEqual([{ hash: "shared-hash" }]);
  });

  it("rolls back collection and FTS mutations on rename/delete failure", () => {
    const store = fixture();
    upsertStoreCollection(store.db, "old", { path: "/synthetic/old" });
    doc(store, "old", "one.md", "hash-one", "ownershipneedle body");
    store.db.exec("CREATE TRIGGER reject_rename BEFORE UPDATE ON documents BEGIN SELECT RAISE(FAIL, 'blocked'); END");
    expect(() => renameStoreCollection(store.db, "old", "new")).toThrow("blocked");
    expect(store.db.prepare("SELECT name FROM store_collections").all()).toEqual([{ name: "old" }]);
    expect(store.searchFTS("ownershipneedle")[0]?.filepath).toBe("qmd://old/one.md");
    store.db.exec("CREATE TRIGGER reject_delete BEFORE DELETE ON documents BEGIN SELECT RAISE(FAIL, 'blocked'); END");
    expect(() => deleteStoreCollection(store.db, "old")).toThrow("blocked");
    expect(store.db.prepare("SELECT name FROM store_collections").all()).toEqual([{ name: "old" }]);
    expect(store.searchFTS("ownershipneedle")[0]?.filepath).toBe("qmd://old/one.md");
  });
});
