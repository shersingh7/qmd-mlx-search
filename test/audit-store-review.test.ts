import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import http from "node:http";
import YAML from "yaml";
import { createStore as createSdk, type CollectionConfig, type QMDStore } from "../src/index.js";
import { createStore, generateEmbeddings, hybridQuery, structuredSearch, vectorSearchQuery, chunkDocumentByTokens, type Store } from "../src/store.js";
import { canUnloadLLM } from "../src/llm.js";
import { runDurableIndexingJob } from "../src/indexing/job.js";

const timestamp = "2026-01-01T00:00:00.000Z";
const stores: Store[] = [];
const sdkStores: QMDStore[] = [];
const directories: string[] = [];
function backend() {
  return {
    embedModelName: "review-model",
    getDescriptor: vi.fn(async () => ({ version: 1, backend: "mlx", pooling: "mean", model: "review-model", nativeDimensions: 4, outputDimensions: 4, maxTokens: 2048, normalized: true })),
    tokenize: vi.fn(async (text: string) => text.split(/\s+/).map((_, i) => i)),
    embed: vi.fn(async () => ({ embedding: [1, 0, 0, 0], model: "review-model" })),
    embedBatch: vi.fn(async (texts: string[]) => texts.map(() => ({ embedding: [1, 0, 0, 0], model: "review-model" }))),
  };
}
function fixture() {
  const store = createStore(":memory:");
  store.llm = backend() as any;
  stores.push(store);
  return store;
}
function seed(store: Store, hash = "hash-one", body = "public synthetic review fixture") {
  store.insertContent(hash, body, timestamp);
  store.insertDocument("docs", `${hash}.md`, hash, hash, timestamp, timestamp);
}
function state(store: Store) {
  return {
    config: store.db.prepare("SELECT * FROM store_config ORDER BY key").all(),
    collections: store.db.prepare("SELECT * FROM store_collections ORDER BY name").all(),
    content: store.db.prepare("SELECT * FROM content ORDER BY hash").all(),
    documents: store.db.prepare("SELECT * FROM documents ORDER BY id").all(),
    chunks: store.db.prepare("SELECT * FROM content_vectors ORDER BY hash, seq").all(),
    vectors: store.db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec ORDER BY hash_seq").all(),
    expectations: store.db.prepare("SELECT * FROM content_chunk_expectations ORDER BY hash, strategy").all(),
    checkpoints: store.db.prepare("SELECT * FROM indexing_checkpoints ORDER BY job_id").all(),
    cache: store.db.prepare("SELECT * FROM llm_cache ORDER BY hash").all(),
  };
}
function barrier() {
  let enter!: () => void;
  let release!: () => void;
  const entered = new Promise<void>(resolve => { enter = resolve; });
  const resumed = new Promise<void>(resolve => { release = resolve; });
  return { enter, release, entered, resumed };
}
async function openSdk(options: Parameters<typeof createSdk>[0]) {
  const sdk = await createSdk(options);
  sdk.internal.llm = backend() as any;
  sdkStores.push(sdk);
  return sdk;
}
function directory() {
  const path = mkdtempSync(join(tmpdir(), "qmd-store-review-"));
  directories.push(path);
  return path;
}
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(sdkStores.splice(0).map(store => store.close()));
  for (const store of stores.splice(0)) store.close();
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
  vi.unstubAllEnvs();
});

const busy = { code: "QMD_DATABASE_BUSY" };
describe("review: forced replacement connection ownership", () => {
  it.each(["inline", "yaml"] as const)("refuses unrelated public SDK mutations before DB/config writes while force awaits inference (%s)", async mode => {
    const root = directory();
    const config: CollectionConfig = {
      collections: { docs: { path: root, pattern: "**/*.md", context: { "/": "original context" } } },
      global_context: "original global",
    };
    const path = join(root, "config.yml");
    writeFileSync(path, YAML.stringify(config));
    const sdk = await openSdk({ dbPath: ":memory:", ...(mode === "inline" ? { config } : { configPath: path }) });
    seed(sdk.internal);
    await sdk.embed();
    sdk.internal.setCachedResult("unrelated-cache", "must survive");
    const before = state(sdk.internal);
    const configBefore = JSON.stringify(config);
    const yamlBefore = readFileSync(path, "utf8");
    const preparedWrite = sdk.internal.db.prepare("INSERT INTO store_config (key, value) VALUES ('foreign-write', 'must not run')");
    const preparedRead = sdk.internal.db.prepare("SELECT hash_seq FROM vectors_vec");
    const gate = barrier();
    const next = backend();
    next.embedBatch.mockImplementation(async () => { gate.enter(); await gate.resumed; throw new Error("synthetic inference failure"); });
    next.embed.mockRejectedValue(new Error("synthetic inference failure"));
    sdk.internal.llm = next as any;
    const replacement = sdk.embed({ force: true });
    try {
      await gate.entered;
      for (const mutate of [
        () => sdk.addCollection("not-acknowledged", { path: root }),
        () => sdk.addCollection("docs", { path: "/synthetic/replaced" }),
        () => sdk.renameCollection("docs", "renamed"),
        () => sdk.removeCollection("docs"),
        () => sdk.addContext("docs", "/new", "not acknowledged"),
        () => sdk.removeContext("docs", "/"),
        () => sdk.setGlobalContext("not acknowledged"),
        () => sdk.update(),
        () => sdk.embed({ force: true }),
        () => sdk.close(),
      ]) await expect(mutate()).rejects.toMatchObject(busy);
      expect(() => sdk.internal.insertContent("foreign-hash", "foreign body", timestamp)).toThrow(/Database is busy/);
      expect(() => sdk.internal.clearAllEmbeddings()).toThrow(/Database is busy/);
      expect(() => preparedWrite.run()).toThrow(/Database is busy/);
      expect(() => preparedRead.all()).toThrow(/Database is busy/);
      await expect(sdk.searchVector("query")).rejects.toMatchObject(busy);
      await expect(sdk.getStatus()).rejects.toMatchObject(busy);
      await expect(sdk.listCollections()).rejects.toMatchObject(busy);
      expect(JSON.stringify(config)).toBe(configBefore);
      expect(readFileSync(path, "utf8")).toBe(yamlBefore);
      const independent = await openSdk({ dbPath: ":memory:" });
      await independent.addCollection("accepted-peer-write", { path: root });
      expect((await independent.listCollections()).map(c => c.name)).toEqual(["accepted-peer-write"]);
    } finally {
      gate.release();
      expect((await replacement).status).toBe("failed");
    }
    expect(state(sdk.internal)).toEqual(before);
    expect(JSON.stringify(config)).toBe(configBefore);
    expect(readFileSync(path, "utf8")).toBe(yamlBefore);
    await sdk.addCollection("after-rebuild", { path: root });
    expect((await sdk.listCollections()).map(c => c.name).sort()).toEqual(["after-rebuild", "docs"]);
  });

  it("does not grant inference or progress callbacks the replacement's write ownership, and never deadlocks awaited callbacks", async () => {
    const config: CollectionConfig = { collections: { docs: { path: "/synthetic/docs", pattern: "**/*.md" } } };
    const sdk = await openSdk({ dbPath: ":memory:", config });
    seed(sdk.internal);
    await sdk.embed();
    const next = backend();
    let inferenceRefusal: unknown;
    next.embedBatch.mockImplementation(async (texts: string[]) => {
      try { await sdk.addCollection("from-inference", { path: "/synthetic/foreign" }); }
      catch (error) { inferenceRefusal = error; }
      return texts.map(() => ({ embedding: [1, 0, 0, 0], model: "review-model" }));
    });
    sdk.internal.llm = next as any;
    let progressWrite: Promise<unknown> | undefined;
    const result = await sdk.embed({ force: true, onProgress: () => {
      progressWrite ??= sdk.addContext("docs", "/foreign", "must refuse");
      void progressWrite.catch(() => {});
    } });
    expect(result.status).toBe("complete");
    expect(inferenceRefusal).toMatchObject(busy);
    await expect(progressWrite).rejects.toMatchObject(busy);
    expect(Object.keys(config.collections)).toEqual(["docs"]);
    expect(config.collections.docs?.context).toBeUndefined();
    // The owner still performed its own nested vector/checkpoint transactions.
    expect(sdk.internal.db.prepare("SELECT COUNT(*) AS count FROM vectors_vec").get()).toEqual({ count: 1 });
  });

  it("other connections see the prior committed index and refuse conflicting writes rather than acknowledging rolled-back config", async () => {
    const dbPath = join(directory(), "public-scratch.sqlite");
    const writer = await openSdk({ dbPath });
    await writer.addCollection("docs", { path: "/synthetic/docs" });
    seed(writer.internal);
    await writer.embed();
    const reader = await openSdk({ dbPath });
    reader.internal.db.exec("PRAGMA busy_timeout = 0");
    const before = state(reader.internal);
    const gate = barrier();
    const next = backend();
    next.embedBatch.mockImplementation(async () => { gate.enter(); await gate.resumed; throw new Error("synthetic failure"); });
    next.embed.mockRejectedValue(new Error("synthetic failure"));
    writer.internal.llm = next as any;
    const replacement = writer.embed({ force: true });
    try {
      await gate.entered;
      expect(state(reader.internal)).toEqual(before);
      expect(await reader.internal.searchVec("query", "review-model", 1, undefined, undefined, [1, 0, 0, 0])).toHaveLength(1);
      await expect(reader.addCollection("conflicting", { path: "/synthetic/foreign" })).rejects.toThrow(/locked|busy/i);
    } finally {
      gate.release();
      expect((await replacement).status).toBe("failed");
    }
    expect(state(reader.internal)).toEqual(before);
    await reader.addCollection("accepted-after-rollback", { path: "/synthetic/foreign" });
    expect((await writer.listCollections()).map(c => c.name).sort()).toEqual(["accepted-after-rollback", "docs"]);
  });
});

describe("review: cancelled zero-work replacements", () => {
  it.each([
    ["direct", "already"], ["direct", "empty-preflight"],
    ["durable", "already"], ["durable", "job-preflight"], ["durable", "empty-preflight"],
    ["direct", "empty-preflight-throws"], ["durable", "job-preflight-throws"], ["durable", "empty-preflight-throws"],
  ] as const)("preserves every prior row and checkpoint for %s cancellation at %s", async (runner, timing) => {
    const store = fixture();
    seed(store);
    await runDurableIndexingJob(store);
    store.deactivateDocument("docs", "hash-one.md");
    const before = state(store);
    const controller = new AbortController();
    let calls = 0;
    const descriptor = await store.getEmbeddingDescriptor!();
    store.getEmbeddingDescriptor = vi.fn(async () => {
      calls++;
      if (timing.startsWith("job-preflight") || calls === (runner === "durable" ? 2 : 1)) controller.abort();
      if (controller.signal.aborted && timing.endsWith("throws")) throw controller.signal.reason;
      return descriptor;
    });
    if (timing === "already") controller.abort();
    const result = runner === "durable"
      ? await runDurableIndexingJob(store, { force: true, signal: controller.signal })
      : await generateEmbeddings(store, { force: true, signal: controller.signal });
    expect(result).toMatchObject({ status: "cancelled", chunksCommitted: 0, docsProcessed: 0 });
    if (timing === "already") expect(store.getEmbeddingDescriptor).not.toHaveBeenCalled();
    expect(state(store)).toEqual(before);
  });

  it.each(["direct", "durable"] as const)("rechecks cancellation at the final publication boundary (%s)", async runner => {
    const store = fixture();
    seed(store);
    await runDurableIndexingJob(store);
    const before = state(store);
    const replacementBackend = backend();
    replacementBackend.embed.mockResolvedValue({ embedding: [0, 1, 0, 0], model: "review-model" });
    replacementBackend.embedBatch.mockImplementation(async texts => texts.map(() => ({ embedding: [0, 1, 0, 0], model: "review-model" })));
    store.llm = replacementBackend as any;
    const controller = new AbortController();
    let scheduled = false;
    const onProgress = (progress: { chunksEmbedded: number }) => {
      if (scheduled || progress.chunksEmbedded === 0) return;
      scheduled = true;
      // The session callback has already chosen its result when this fires,
      // but the replacement's outer publication continuation has not run.
      queueMicrotask(() => queueMicrotask(() => controller.abort()));
    };
    const result = await (runner === "direct" ? generateEmbeddings : runDurableIndexingJob)(store, {
      force: true, signal: controller.signal, onProgress,
    });
    expect(controller.signal.aborted).toBe(true);
    expect(result).toMatchObject({ status: "cancelled", docsProcessed: 0, chunksCommitted: 0, chunksEmbedded: 0 });
    expect(state(store)).toEqual(before);
  });

  it("public SDK forwards cancellation without clearing inactive embeddings", async () => {
    const sdk = await openSdk({ dbPath: ":memory:" });
    seed(sdk.internal);
    await sdk.embed();
    sdk.internal.deactivateDocument("docs", "hash-one.md");
    const before = state(sdk.internal);
    const controller = new AbortController();
    controller.abort();
    const result = await sdk.embed({ force: true, signal: controller.signal });
    expect(result.status).toBe("cancelled");
    expect(state(sdk.internal)).toEqual(before);
  });
});

describe("review: physical vector integrity", () => {
  it("repairs only a missing physical chunk, preserves valid chunks and other documents, then returns to a body-free no-op", async () => {
    const store = fixture();
    seed(store, "hash-broken", "synthetic content ".repeat(2000));
    seed(store, "hash-intact", "another synthetic document");
    await runDurableIndexingJob(store);
    const chunks = store.db.prepare("SELECT * FROM content_vectors WHERE hash = 'hash-broken' ORDER BY seq").all();
    expect(chunks.length).toBeGreaterThan(1);
    const key = `hash-broken_${chunks[0]!.seq}`;
    store.db.prepare("DELETE FROM vectors_vec WHERE hash_seq = ?").run(key);
    const survivingVectors = store.db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec ORDER BY hash_seq").all();
    const intactMetadata = store.db.prepare("SELECT * FROM content_vectors WHERE hash = 'hash-intact'").all();
    expect(store.getHashesNeedingEmbedding()).toBe(1);
    expect(store.getHashesForEmbedding().map(d => d.hash)).toEqual(["hash-broken"]);
    const own = store.llm as unknown as ReturnType<typeof backend>;
    own.embedBatch.mockClear();
    own.embed.mockClear();
    const result = await runDurableIndexingJob(store, { maxDocsPerBatch: 1 });
    expect(result).toMatchObject({ status: "complete", docsSelected: 1, docsProcessed: 1, chunksCommitted: 1 });
    expect(own.embedBatch.mock.calls.flatMap(args => args[0])).toHaveLength(1);
    expect(store.db.prepare("SELECT * FROM content_vectors WHERE hash = 'hash-intact'").all()).toEqual(intactMetadata);
    expect(store.db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec WHERE hash_seq != ? ORDER BY hash_seq").all(key)).toEqual(survivingVectors);
    expect(await store.searchVec("query", "review-model", 2, undefined, undefined, [1, 0, 0, 0])).toHaveLength(2);
    own.tokenize.mockClear(); own.embed.mockClear(); own.embedBatch.mockClear();
    const prepare = store.db.prepare.bind(store.db);
    const bodies: string[] = [];
    vi.spyOn(store.db, "prepare").mockImplementation((sql: string) => {
      if (/doc as body/i.test(sql)) bodies.push(sql);
      return prepare(sql);
    });
    expect(await runDurableIndexingJob(store)).toMatchObject({ status: "complete", docsSelected: 0, chunksAttempted: 0 });
    expect(bodies).toEqual([]);
    expect(own.tokenize).not.toHaveBeenCalled();
    expect(own.embed).not.toHaveBeenCalled();
    expect(own.embedBatch).not.toHaveBeenCalled();
    expect(store.getHashesNeedingEmbedding()).toBe(0);
  });

  it.each(["legacy-metadata", "missing-table"] as const)("never treats %s without a physical vector as complete", async mode => {
    const store = fixture();
    seed(store);
    if (mode === "legacy-metadata") {
      store.ensureVecTable(4);
      store.db.prepare("INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES ('hash-one', 0, 0, 'review-model', ?)").run(timestamp);
    } else {
      await runDurableIndexingJob(store);
      store.db.exec("DROP TABLE vectors_vec");
    }
    expect(store.getHashesNeedingEmbedding()).toBe(1);
    const result = await runDurableIndexingJob(store);
    expect(result).toMatchObject({ status: "complete", docsSelected: 1, chunksCommitted: 1 });
    expect(store.db.prepare("SELECT hash_seq FROM vectors_vec").all()).toEqual([{ hash_seq: "hash-one_0" }]);
    expect(await store.searchVec("query", "review-model", 1, undefined, undefined, [1, 0, 0, 0])).toHaveLength(1);
    expect((await runDurableIndexingJob(store)).docsSelected).toBe(0);
  });

  it("malformed dimensions never leave completion metadata or delete a previous physical vector", () => {
    const store = fixture();
    seed(store);
    store.ensureVecTable(4);
    expect(() => store.insertEmbedding("hash-one", 0, 0, new Float32Array([1, 0]), "review-model", timestamp)).toThrow();
    expect(store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
    expect(store.db.prepare("SELECT hash_seq FROM vectors_vec").all()).toEqual([]);
    store.insertEmbedding("hash-one", 0, 0, new Float32Array([1, 0, 0, 0]), "review-model", timestamp);
    const before = state(store);
    expect(() => store.insertEmbedding("hash-one", 0, 99, new Float32Array([1, 0]), "wrong-model", timestamp)).toThrow();
    expect(state(store)).toEqual(before);
  });
});

async function bounded<T>(promise: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([promise, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("review cancellation watchdog exceeded")), 1500);
    })]);
  } finally { clearTimeout(timer); }
}

describe("review: SDK search cancellation reaches actual work", () => {
  it.each(["structured", "hybrid", "vector", "expand", "hybrid-expand", "rerank"] as const)("aborts the actual MLX HTTP request when a public %s search is cancelled", async kind => {
    const descriptor = {
      version: 1, backend: "mlx", pooling: "mean", model: "fixture/encoder", nativeDimensions: 4, outputDimensions: 4, maxTokens: 2048, normalized: true,
      generate: { version: 1, backend: "mlx", kind: "generate", model: "fixture/generator" },
      rerank: { version: 1, backend: "mlx", kind: "rerank", model: "fixture/ranker" },
    } as const;
    const target = kind === "expand" || kind === "hybrid-expand" ? /^\/generate$/
      : kind === "rerank" ? /^\/rerank$/ : /^\/embed(?:-bin)?$/;
    const gate = barrier();
    let disconnect!: () => void;
    const disconnected = new Promise<void>(resolve => { disconnect = resolve; });
    const paths: string[] = [];
    const server = http.createServer((req, res) => {
      paths.push(req.url!);
      req.resume();
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/descriptor") res.end(JSON.stringify(descriptor));
      else if (req.url === "/health") res.end(JSON.stringify({ status: "ok", ready: true, descriptor }));
      else if (target.test(req.url!)) {
        res.once("close", () => { if (!res.writableEnded) disconnect(); });
        gate.enter(); // A real socket is awaiting its response, not a stubbed fetch.
      } else { res.writeHead(404); res.end(); }
    });
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
    vi.stubEnv("QMD_STRICT_MLX", "0");
    vi.stubEnv("QMD_MLX_EXPAND", "1");
    vi.stubEnv("QMD_MLX_RERANK", "1");
    vi.stubEnv("QMD_MLX_EXPAND_FALLBACK", "0");
    vi.stubEnv("QMD_MLX_RERANK_FALLBACK", "0");
    const sdk = await createSdk({ dbPath: ":memory:", config: { collections: {}, models: {
      embed: descriptor.model, embedBackend: "mlx", mlxUrl: base, mlxFallback: false,
    } } });
    sdkStores.push(sdk);
    // Cancellation must run against a valid, identity-bound index. Otherwise
    // query safety correctly refuses the request before inference begins.
    sdk.internal.ensureVecTable(4, descriptor);
    if (kind === "structured" || kind === "hybrid" || kind === "vector") {
      seed(sdk.internal, "hash-one", "unrelated indexed document");
      sdk.internal.insertEmbedding("hash-one", 0, 0, new Float32Array([1, 0, 0, 0]), descriptor.model, timestamp);
    }
    if (kind === "hybrid") vi.spyOn(sdk.internal, "expandQuery").mockResolvedValue([]);
    if (kind === "rerank") seed(sdk.internal);
    const controller = new AbortController();
    const pending: Promise<unknown> = kind === "vector" ? sdk.searchVector("synthetic query", { signal: controller.signal })
      : kind === "expand" ? sdk.expandQuery("synthetic query", { signal: controller.signal })
      : sdk.search({ ...(kind === "structured" ? { queries: [{ type: "vec" as const, query: "synthetic query" }] }
        : kind === "rerank" ? { queries: [{ type: "lex" as const, query: "synthetic" }] }
        : { query: "synthetic query" }), rerank: kind === "rerank", signal: controller.signal });
    void pending.catch(() => {});
    try {
      await bounded(gate.entered);
      controller.abort(new Error("cancel public search"));
      await expect(bounded(pending)).rejects.toThrow(/cancel public search|released|abort/i);
      await bounded(disconnected);
      expect(paths.filter(path => target.test(path))).toHaveLength(1);
      expect(sdk.internal.db.prepare("SELECT * FROM llm_cache").all()).toEqual([]);
    } finally {
      controller.abort();
      server.closeAllConnections();
      await new Promise<void>(resolve => server.close(() => resolve()));
      await Promise.allSettled([pending]);
    }
  });

  it("prompt cancellation retains the operation lease until an ignoring native-like backend actually settles", async () => {
    const sdk = await openSdk({ dbPath: ":memory:" });
    const own = backend();
    sdk.internal.ensureVecTable(4, await own.getDescriptor() as any);
    seed(sdk.internal);
    sdk.internal.insertEmbedding("hash-one", 0, 0, new Float32Array([1, 0, 0, 0]), own.embedModelName, timestamp);
    const gate = barrier();
    own.embedBatch.mockImplementation(async (texts: string[]) => {
      gate.enter();
      await gate.resumed;
      return texts.map(() => ({ embedding: [1, 0, 0, 0], model: "review-model" }));
    });
    sdk.internal.llm = own as any;
    const controller = new AbortController();
    const pending = sdk.search({ queries: [{ type: "vec", query: "synthetic query" }], signal: controller.signal });
    void pending.catch(() => {});
    try {
      await gate.entered;
      controller.abort(new Error("cancel native-like work"));
      await expect(bounded(pending)).rejects.toThrow(/cancel|released/i);
      expect(canUnloadLLM(own as any)).toBe(false);
    } finally { gate.release(); }
    await vi.waitFor(() => expect(canUnloadLLM(own as any)).toBe(true));
  });

  it.each(["expand", "rerank"] as const)("does not return or overwrite cached %s output when cancelled during identity preflight", async op => {
    const sdk = await openSdk({ dbPath: ":memory:" });
    const controller = new AbortController();
    const own = {
      ...backend(),
      getCacheIdentity: vi.fn(async () => {
        await Promise.resolve();
        controller.abort(new Error("cancel cache preflight"));
        return "cache-model";
      }),
      expandQuery: vi.fn(async () => [{ type: "vec", text: "must not run" }]),
      rerank: vi.fn(async () => ({ results: [], model: "must not run" })),
    };
    sdk.internal.llm = own as any;
    const cacheKey = sdk.internal.getCacheKey(op === "expand" ? "expandQuery" : "rerank", {
      cacheVersion: 2, query: "query", model: "cache-model", ...(op === "rerank" ? { chunk: "synthetic body" } : {}),
    });
    const cached = op === "expand" ? JSON.stringify([{ type: "vec", query: "cached query" }]) : "0.9";
    sdk.internal.setCachedResult(cacheKey, cached);
    const pending = op === "expand" ? sdk.expandQuery("query", { signal: controller.signal })
      : sdk.internal.rerank("query", [{ file: "one", text: "synthetic body" }], undefined, undefined, controller.signal);
    await expect(pending).rejects.toThrow(/cancel cache preflight|released/i);
    expect(own.expandQuery).not.toHaveBeenCalled();
    expect(own.rerank).not.toHaveBeenCalled();
    expect(sdk.internal.getCachedResult(cacheKey)).toBe(cached);
  });

  it("checks already-aborted signals before cache/no-work reads and rejects tokenization cancelled during its await", async () => {
    const sdk = await openSdk({ dbPath: ":memory:" });
    const controller = new AbortController();
    controller.abort(new Error("cancel before any work"));
    for (const call of [
      () => sdk.search({ queries: [], signal: controller.signal }),
      () => sdk.searchVector("query", { signal: controller.signal }),
      () => sdk.expandQuery("query", { signal: controller.signal }),
      () => hybridQuery(sdk.internal, "query", { signal: controller.signal }),
      () => structuredSearch(sdk.internal, [], { signal: controller.signal }),
      () => vectorSearchQuery(sdk.internal, "query", { signal: controller.signal }),
    ]) await expect(call()).rejects.toThrow(/cancel before any work/);
    const during = new AbortController();
    const own = backend();
    own.tokenize.mockImplementation(async () => { during.abort(new Error("cancel tokenization")); return [0]; });
    await expect(chunkDocumentByTokens("synthetic body", undefined, undefined, undefined, undefined, undefined, during.signal, own as any))
      .rejects.toThrow("cancel tokenization");
  });
});
