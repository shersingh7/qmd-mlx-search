import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import http from "node:http";
import { createStore as createSdk, type QMDStore } from "../src/index.js";
import { hybridQuery, structuredSearch, vectorSearchQuery, type Store } from "../src/store.js";
import { withExclusiveDatabaseOperation } from "../src/db.js";
import { LlamaCpp, canUnloadLLM } from "../src/llm.js";
import { type EmbeddingDescriptor } from "../src/embedding/contract.js";
import { encodeBinaryEmbeddings } from "../src/embedding/protocol.js";

const timestamp = "2026-01-01T00:00:00.000Z";
const vector = [1, 0, 0, 0];
const stores: QMDStore[] = [];
const llms: LlamaCpp[] = [];
const directories: string[] = [];
const servers: http.Server[] = [];
const descriptor = (revision = "A"): EmbeddingDescriptor => ({
  version: 1, backend: "mlx", model: "public/snapshot-encoder", revision,
  pooling: "mean", nativeDimensions: 4, outputDimensions: 4, maxTokens: 256, normalized: true,
});
function backend(revision = "A") {
  const state = { descriptor: descriptor(revision) };
  return {
    state, embedModelName: state.descriptor.model,
    getDescriptor: vi.fn(async () => state.descriptor),
    tokenize: vi.fn(async (text: string) => text.split(/\s+/).map((_, i) => i)),
    embed: vi.fn(async () => ({ embedding: [...vector], model: state.descriptor.model })),
    embedBatch: vi.fn(async (texts: string[]) => texts.map(() => ({ embedding: [...vector], model: state.descriptor.model }))),
    getCacheIdentity: vi.fn(async (op: string) => `fixture:${op}`),
    expandQueryWithIdentity: vi.fn(async (query: string) => ({ queries: [{ type: "vec" as const, text: `${query} expanded` }], model: "fixture:expand" })),
    rerank: vi.fn(async (_query: string, docs: { file: string }[]) => ({ results: docs.map((doc, index) => ({ file: doc.file, index, score: 0.9 })), model: "fixture:rerank" })),
  };
}
async function fixture(file = false) {
  let dbPath = ":memory:";
  if (file) {
    const directory = mkdtempSync(join(tmpdir(), "qmd-query-snapshot-"));
    directories.push(directory);
    dbPath = join(directory, "public.sqlite");
  }
  const sdk = await createSdk({ dbPath, config: { collections: { docs: { path: "/synthetic/public", pattern: "**/*.md" } } } });
  stores.push(sdk);
  const own = backend();
  const store = sdk.internal;
  store.llm = own as any;
  store.ensureVecTable(4, own.state.descriptor);
  for (const [name, embedding] of [["one", vector], ["two", [0, 1, 0, 0]]] as const) {
    store.insertContent(name, `${name} public synthetic snapshot document`, timestamp);
    store.insertDocument("docs", `${name}.md`, name, name, timestamp, timestamp);
    store.insertEmbedding(name, 0, 0, new Float32Array(embedding), own.embedModelName, timestamp);
  }
  return { sdk, store, own };
}
// Keep this self-contained: the real writer process uses the same logical readback.
function snapshot(store: Store) {
  const db = store.db;
  return {
    schema: db.prepare("SELECT name, sql FROM sqlite_master ORDER BY name").all(),
    config: db.prepare("SELECT * FROM store_config ORDER BY key").all(),
    documents: db.prepare("SELECT * FROM documents ORDER BY id").all(),
    content: db.prepare("SELECT * FROM content ORDER BY hash").all(),
    chunks: db.prepare("SELECT * FROM content_vectors ORDER BY hash, seq").all(),
    vectors: db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec ORDER BY hash_seq").all(),
    expectations: db.prepare("SELECT * FROM content_chunk_expectations ORDER BY hash, strategy").all(),
    checkpoints: db.prepare("SELECT * FROM indexing_checkpoints ORDER BY job_id").all(),
    cache: db.prepare("SELECT * FROM llm_cache ORDER BY hash").all(),
  };
}
function barrier() {
  let enter!: () => void;
  let release!: () => void;
  return { entered: new Promise<void>(resolve => { enter = resolve; }), resumed: new Promise<void>(resolve => { release = resolve; }), enter: () => enter(), release: () => release() };
}
const sdkRoutes = {
  vector: (sdk: QMDStore) => sdk.searchVector("snapshotneedle", { limit: 1 }),
  structured: (sdk: QMDStore) => sdk.search({ queries: [{ type: "vec", query: "snapshotneedle" }], rerank: false, limit: 1 }),
};
const expandedRoutes = {
  hybrid: (store: Store, hooks?: Parameters<typeof hybridQuery>[2]) => hybridQuery(store, "snapshotneedle", { skipRerank: true, ...hooks }),
  expanded: (store: Store, hooks?: Parameters<typeof vectorSearchQuery>[2]) => vectorSearchQuery(store, "snapshotneedle", hooks),
};

beforeEach(() => {
  vi.stubEnv("QMD_EMBED_BACKEND", "gguf");
  vi.stubEnv("QMD_STRICT_MLX", "0");
  vi.stubEnv("QMD_LLAMA_GPU", "false");
});
afterEach(async () => {
  vi.restoreAllMocks();
  for (const server of servers.splice(0)) {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
  for (const llm of llms.splice(0)) await llm.dispose();
  for (const sdk of stores.splice(0)) await sdk.close();
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
  vi.unstubAllEnvs();
});

describe("query validation, KNN and materialization share a deferred read snapshot", () => {
  it.each(["vector", "structured"] as const)("keeps matching SDK %s searches read-only", async route => {
    const { sdk, store } = await fixture(true);
    const before = snapshot(store);
    store.db.exec("PRAGMA query_only = ON");
    const results = await sdkRoutes[route](sdk);
    expect(results[0]).toMatchObject({ [route === "vector" ? "filepath" : "file"]: "qmd://docs/one.md" });
    expect(snapshot(store)).toEqual(before);
  });

  it.each([
    ["vector", "before-KNN"], ["structured", "before-KNN"],
    ["vector", "before-documents"], ["structured", "before-documents"],
  ] as const)("SDK %s never mixes A with B after a separate process commits %s", async (route, boundary) => {
    const { sdk, store, own } = await fixture(true);
    store.db.exec("PRAGMA query_only = ON");
    const prepare = store.db.prepare.bind(store.db);
    let committed: ReturnType<typeof snapshot> | undefined;
    let writerStatus: unknown;
    let pinnedRevision: unknown;
    own.embed.mockImplementation(async (...args: unknown[]) => {
      pinnedRevision = (args[1] as any)?.expectedDescriptor?.revision;
      return { embedding: [...vector], model: own.embedModelName };
    });
    own.embedBatch.mockImplementation(async (texts, ...args: unknown[]) => {
      pinnedRevision = (args[0] as any)?.expectedDescriptor?.revision;
      return texts.map(() => ({ embedding: [...vector], model: own.embedModelName }));
    });
    const hook = vi.spyOn(store.db, "prepare").mockImplementation(sql => {
      const target = boundary === "before-KNN" ? /WHERE embedding MATCH/.test(sql) : /WITH requested\(hash, seq, hash_seq\)/.test(sql);
      if (target && !committed) {
        const child = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", `
          import { createStore } from ${JSON.stringify(new URL("../src/index.ts", import.meta.url).href)};
          const sdk = await createStore({ dbPath: ${JSON.stringify(store.dbPath)} });
          const d = ${JSON.stringify(descriptor("B"))};
          sdk.internal.llm = {
            embedModelName: d.model, getDescriptor: async () => d,
            tokenize: async text => text.split(/\\s+/).map((_, i) => i),
            embedBatch: async texts => texts.map(text => ({ embedding: text.includes('one public') ? [0,1,0,0] : [1,0,0,0], model: d.model })),
          };
          const result = await sdk.embed({ force: true });
          sdk.internal.db.exec("UPDATE content SET doc = 'one public synthetic snapshot document B' WHERE hash = 'one'");
          console.log(JSON.stringify({ result, state: (${snapshot.toString()})(sdk.internal) }));
          await sdk.close();
        `], { cwd: new URL("..", import.meta.url), env: process.env, encoding: "utf8", timeout: 20_000 });
        expect(child.status, child.stderr).toBe(0);
        const readback = JSON.parse(child.stdout.trim());
        writerStatus = readback.result;
        committed = readback.state;
      }
      return prepare(sql);
    });
    let result: Awaited<ReturnType<typeof sdkRoutes[typeof route]>> | undefined;
    let failure: unknown;
    try { result = await sdkRoutes[route](sdk); } catch (error) { failure = error; }
    hook.mockRestore();
    expect(pinnedRevision).toBe("A");
    expect(writerStatus).toMatchObject({ status: "complete", chunksCommitted: 2 });
    expect(committed).toBeDefined(); // Writer must commit while the reader's snapshot is held.
    if (failure) expect(String(failure)).toMatch(/embedding|changed|schema|snapshot|locked/i);
    else expect(result?.[0]).toMatchObject({ [route === "vector" ? "filepath" : "file"]: "qmd://docs/one.md", body: "one public synthetic snapshot document" });
    expect(snapshot(store)).toEqual(committed); // No reader writes or rollback of the writer.
    store.llm = backend("B") as any;
    expect((await sdkRoutes[route](sdk))[0]).toMatchObject({ [route === "vector" ? "filepath" : "file"]: "qmd://docs/two.md" });
    expect(snapshot(store)).toEqual(committed);
  });

  it("nests safely in caller transactions and connection ownership, including failure", async () => {
    const { store } = await fixture();
    store.db.exec("BEGIN DEFERRED; INSERT INTO store_config VALUES ('caller-owned', 'keep'); SAVEPOINT caller_scope");
    await withExclusiveDatabaseOperation(store.db, "caller read", async () => {
      expect(await store.searchVec("raw", "caller", 1, undefined, undefined, vector)).toHaveLength(1);
      await expect(store.searchVec("raw", "caller", 1, undefined, undefined, [1, 0])).rejects.toThrow(/dimensions/i);
    });
    store.db.exec("ROLLBACK TO caller_scope; RELEASE caller_scope");
    expect(store.db.prepare("SELECT value FROM store_config WHERE key = 'caller-owned'").get()).toEqual({ value: "keep" });
    store.db.exec("ROLLBACK");
    expect(store.db.prepare("SELECT value FROM store_config WHERE key = 'caller-owned'").get()).toBeUndefined();
  });
});

async function httpBackend(store: Store, revision = "A") {
  vi.stubEnv("QMD_MLX_EXPAND", "1");
  vi.stubEnv("QMD_MLX_EXPAND_FALLBACK", "0");
  const state = { revision, generated: 0, embedded: 0 };
  const server = http.createServer(async (req, res) => {
    const d = { ...descriptor(state.revision), generate: { version: 1, backend: "mlx", kind: "generate", model: "public/generator", revision: "generator-A" } };
    const send = (value: unknown) => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(value)); };
    if (req.url === "/descriptor") { req.resume(); send(d); return; }
    if (req.url === "/health") { req.resume(); send({ status: "ok", ready: true, descriptor: d }); return; }
    let body = "";
    for await (const chunk of req) body += chunk;
    if (req.url === "/generate") { state.generated++; send({ text: "vec: snapshotneedle expanded public query", model: "public/generator" }); return; }
    if (req.url === "/embed" || req.url === "/embed-bin") {
      state.embedded++;
      const embeddings = JSON.parse(body).texts.map(() => vector);
      if (req.url === "/embed-bin") {
        res.setHeader("Content-Type", "application/octet-stream");
        res.setHeader("X-QMD-Embedding-Descriptor", encodeURIComponent(JSON.stringify(d)));
        res.end(encodeBinaryEmbeddings(embeddings, 4));
      } else send({ model: d.model, dims: 4, descriptor: d, embeddings });
      return;
    }
    res.writeHead(404); res.end();
  });
  servers.push(server);
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const llm = new LlamaCpp({ embedBackend: "mlx", mlxUrl: `http://127.0.0.1:${(server.address() as { port: number }).port}`, mlxFallback: false, mlxBinary: false });
  llms.push(llm);
  store.llm = llm;
  return { state, llm };
}

describe("inference caches publish only after the owning query succeeds", () => {
  it.each(["hybrid", "expanded"] as const)("real HTTP %s identity rejection cannot create or prune cache rows", async route => {
    const { store } = await fixture();
    const { state, llm } = await httpBackend(store, "B");
    store.setCachedResult("preexisting", "keep unchanged");
    const key = store.getCacheKey("expandQuery", { cacheVersion: 2, query: "snapshotneedle", model: await llm.getCacheIdentity("expand") });
    store.setCachedResult(key, "unparseable old cache row: must not be overwritten");
    store.db.exec("BEGIN");
    const insert = store.db.prepare("INSERT INTO llm_cache VALUES (?, 'unrelated old row', ?)");
    for (let i = 0; i < 1001; i++) insert.run(`old-${i}`, timestamp);
    store.db.exec("COMMIT");
    const before = snapshot(store);
    vi.spyOn(Math, "random").mockReturnValue(0); // Cleanup must also be deferred.
    await expect(expandedRoutes[route](store)).rejects.toThrow(/embedding space/i);
    expect(state.generated).toBe(1); // The valid generator really returned before rejection.
    expect(state.embedded).toBe(0);
    expect(snapshot(store)).toEqual(before);
  });

  it.each(["hybrid", "expanded"] as const)("discards %s expansion when embedding identity drifts after expansion", async route => {
    const { store } = await fixture();
    const { state } = await httpBackend(store);
    const before = snapshot(store);
    await expect(expandedRoutes[route](store, { hooks: { onExpand: () => { state.revision = "B"; } } })).rejects.toThrow(/embedding space/i);
    expect(state.generated).toBe(1);
    expect(snapshot(store)).toEqual(before);
  });

  it.each(["hybrid", "expanded"] as const)("publishes successful %s expansions in their actual namespace and reuses them", async route => {
    const { store } = await fixture();
    const { state, llm } = await httpBackend(store);
    expect(await expandedRoutes[route](store)).toHaveLength(2);
    const actualModel = await llm.getCacheIdentity("expand");
    const key = store.getCacheKey("expandQuery", { cacheVersion: 2, query: "snapshotneedle", model: actualModel });
    expect(JSON.parse(store.getCachedResult(key)!)).toEqual([{ type: "vec", query: "snapshotneedle expanded public query" }]);
    const before = snapshot(store);
    expect(await expandedRoutes[route](store)).toHaveLength(2);
    expect(state.generated).toBe(1);
    expect(snapshot(store)).toEqual(before);
  });

  it.each(["hybrid", "structured"] as const)("does not flush nested %s expansion/rerank writes if a later query hook fails", async route => {
    const { store, own } = await fixture();
    store.setCachedResult("preexisting", "keep unchanged");
    const before = snapshot(store);
    const checkUnpublished = () => expect(store.db.prepare("SELECT hash FROM llm_cache ORDER BY hash").all()).toEqual([{ hash: "preexisting" }]);
    const options = { hooks: {
      onEmbedStart: checkUnpublished,
      onRerankDone: () => { checkUnpublished(); throw new Error("synthetic post-rerank failure"); },
    } };
    await expect(route === "hybrid" ? hybridQuery(store, "snapshotneedle", options)
      : structuredSearch(store, [{ type: "vec", query: "snapshotneedle" }], options)).rejects.toThrow("synthetic post-rerank failure");
    expect(own.expandQueryWithIdentity).toHaveBeenCalledTimes(route === "hybrid" ? 1 : 0);
    expect(own.rerank).toHaveBeenCalledTimes(1);
    expect(snapshot(store)).toEqual(before);
  });

  it("publishes only its own buffer across concurrent same-DB and other-DB operations", async () => {
    const { store, own } = await fixture();
    const { store: peer } = await fixture();
    store.setCachedResult("preexisting", "keep unchanged");
    const gate = barrier();
    own.embedBatch.mockImplementation(async () => { gate.enter(); await gate.resumed; throw new Error("synthetic failed inference"); });
    const failed = hybridQuery(store, "abandoned-query", { skipRerank: true });
    const rejected = expect(failed).rejects.toThrow("synthetic failed inference");
    try {
      await gate.entered;
      await store.expandQuery("successful-same-db");
      await peer.expandQuery("successful-other-db");
    } finally { gate.release(); }
    await rejected;
    const key = (query: string) => store.getCacheKey("expandQuery", { cacheVersion: 2, query, model: "fixture:expand" });
    expect(store.getCachedResult(key("abandoned-query"))).toBeNull();
    expect(store.getCachedResult(key("successful-same-db"))).not.toBeNull();
    expect(store.getCachedResult(key("successful-other-db"))).toBeNull();
    expect(peer.getCachedResult(key("successful-other-db"))).not.toBeNull();
    expect(peer.getCachedResult(key("successful-same-db"))).toBeNull();
    expect(store.getCachedResult("preexisting")).toBe("keep unchanged");
  });

  it.each(["expansion", "embedding"] as const)("late cancelled native-like %s cannot publish abandoned cache writes", async stage => {
    const { store, own } = await fixture();
    store.setCachedResult("preexisting", "keep unchanged");
    const before = snapshot(store);
    const gate = barrier();
    if (stage === "embedding") own.embedBatch.mockImplementation(async texts => { gate.enter(); await gate.resumed; return texts.map(() => ({ embedding: [...vector], model: own.embedModelName })); });
    else own.expandQueryWithIdentity.mockImplementation(async query => { gate.enter(); await gate.resumed; return { queries: [{ type: "vec", text: `${query} expanded` }], model: "fixture:expand" }; });
    const controller = new AbortController();
    const pending = hybridQuery(store, "snapshotneedle", { signal: controller.signal, skipRerank: true });
    const rejected = expect(pending).rejects.toThrow(/cancel|released/i);
    try {
      await gate.entered;
      controller.abort(new Error("cancel abandoned query"));
      await rejected;
      expect(canUnloadLLM(own as any)).toBe(false);
      expect(snapshot(store)).toEqual(before);
    } finally { gate.release(); }
    await vi.waitFor(() => expect(canUnloadLLM(own as any)).toBe(true));
    expect(snapshot(store)).toEqual(before);
  });

  it("standalone expansion started by a query hook never inherits its failing parent's buffer", async () => {
    const { store } = await fixture();
    const key = (query: string) => store.getCacheKey("expandQuery", { cacheVersion: 2, query, model: "fixture:expand" });
    let independent: Promise<unknown> | undefined;
    await expect(hybridQuery(store, "outer-fails", { skipRerank: true, hooks: {
      onExpand: () => { independent = store.expandQuery("standalone"); },
      onEmbedStart: () => { throw new Error("outer failed"); },
    } })).rejects.toThrow("outer failed");
    await independent;
    expect(independent).toBeDefined();
    expect(store.getCachedResult(key("outer-fails"))).toBeNull();
    expect(store.getCachedResult(key("standalone"))).not.toBeNull();
  });

  it("rolls back only its synchronous cache publication if a later staged insert fails", async () => {
    const { store, own } = await fixture();
    store.setCachedResult("preexisting", "keep unchanged");
    const denied = store.getCacheKey("rerank", { cacheVersion: 2, query: "snapshotneedle", model: "fixture:rerank", chunk: "two public synthetic snapshot document" });
    store.db.exec(`CREATE TRIGGER reject_query_cache BEFORE INSERT ON llm_cache WHEN NEW.hash = '${denied}' BEGIN SELECT RAISE(FAIL, 'synthetic cache publication failure'); END`);
    const before = snapshot(store);
    await expect(hybridQuery(store, "snapshotneedle")).rejects.toThrow("synthetic cache publication failure");
    expect(snapshot(store)).toEqual(before);
    store.db.exec("DROP TRIGGER reject_query_cache");
    expect(await hybridQuery(store, "snapshotneedle")).toHaveLength(2);
    expect(store.getCachedResult(denied)).toBe("0.9");
    const expandedCalls = own.expandQueryWithIdentity.mock.calls.length;
    const rerankCalls = own.rerank.mock.calls.length;
    const published = snapshot(store);
    expect(await hybridQuery(store, "snapshotneedle")).toHaveLength(2);
    expect(own.expandQueryWithIdentity).toHaveBeenCalledTimes(expandedCalls);
    expect(own.rerank).toHaveBeenCalledTimes(rerankCalls);
    expect(snapshot(store)).toEqual(published);
  });

  it("bounds staged custom-backend output without rejecting an otherwise successful expansion", async () => {
    const { store, own } = await fixture();
    const oversized = "public synthetic ".repeat(20_000);
    own.expandQueryWithIdentity.mockResolvedValue({ queries: [{ type: "vec", text: oversized }], model: "fixture:expand" });
    expect(await store.expandQuery("large custom response")).toEqual([{ type: "vec", query: oversized }]);
    expect(store.db.prepare("SELECT * FROM llm_cache").all()).toEqual([]);
  });
});
