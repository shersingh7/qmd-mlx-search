import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore as createSdk, type CollectionConfig, type QMDStore } from "../src/index.js";
import { createStore, generateEmbeddings, type Store } from "../src/store.js";
import { runDurableIndexingJob } from "../src/indexing/job.js";
import { withExclusiveDatabaseOperation } from "../src/db.js";
import { loadConfig, setConfigSource } from "../src/collections.js";

const timestamp = "2026-01-01T00:00:00.000Z";
const stores: Store[] = [];
const sdkStores: QMDStore[] = [];
const directories: string[] = [];
function backend() {
  return {
    embedModelName: "quality-model",
    getDescriptor: vi.fn(async () => ({ version: 1, backend: "mlx", pooling: "mean", model: "quality-model", nativeDimensions: 4, outputDimensions: 4, maxTokens: 2048, normalized: true })),
    tokenize: vi.fn(async (text: string) => text.split(/\s+/).map((_, i) => i)),
    embed: vi.fn(async () => ({ embedding: [1, 0, 0, 0], model: "quality-model" })),
    embedBatch: vi.fn(async (texts: string[]) => texts.map(() => ({ embedding: [1, 0, 0, 0], model: "quality-model" }))),
  };
}
function fixture() {
  const store = createStore(":memory:");
  store.llm = backend() as any;
  stores.push(store);
  return store;
}
function seed(store: Store, hash: string, collection = "docs", body = `qualityneedle synthetic document ${hash}`) {
  store.insertContent(hash, body, timestamp);
  store.insertDocument(collection, `${hash}.md`, hash, hash, timestamp, timestamp);
}
function vectors(store: Store) {
  return {
    physical: store.db.prepare("SELECT hash_seq FROM vectors_vec ORDER BY hash_seq").all(),
    metadata: store.db.prepare("SELECT hash, seq FROM content_vectors ORDER BY hash, seq").all(),
  };
}
function removalState(store: Store) {
  return {
    collections: store.db.prepare("SELECT * FROM store_collections ORDER BY name").all(),
    documents: store.db.prepare("SELECT * FROM documents ORDER BY id").all(),
    fts: store.db.prepare("SELECT rowid, filepath, title, body FROM documents_fts ORDER BY rowid").all(),
    content: store.db.prepare("SELECT * FROM content ORDER BY hash").all(),
    ...vectors(store),
  };
}
function barrier() {
  let enter!: () => void;
  let release!: () => void;
  return {
    entered: new Promise<void>(resolve => { enter = resolve; }),
    resumed: new Promise<void>(resolve => { release = resolve; }),
    enter: () => enter(),
    release: () => release(),
  };
}
function directory() {
  const path = mkdtempSync(join(tmpdir(), "qmd-store-quality-"));
  directories.push(path);
  return path;
}
async function openSdk(options: Parameters<typeof createSdk>[0]) {
  const sdk = await createSdk(options);
  sdkStores.push(sdk);
  return sdk;
}
beforeEach(() => {
  vi.stubEnv("QMD_EMBED_BACKEND", "gguf");
  vi.stubEnv("QMD_STRICT_MLX", "0");
});
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.allSettled(sdkStores.splice(0).map(store => store.close()));
  for (const store of stores.splice(0)) store.close();
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
  setConfigSource(undefined);
  vi.unstubAllEnvs();
});

describe("quality: committed incremental work survives callback/session failures", () => {
  it.each(["direct", "durable"] as const)("retains truthful %s counters and checkpoint after cancellation following the first committed document", async runner => {
    const store = fixture();
    seed(store, "one"); seed(store, "two");
    const controller = new AbortController();
    const result = await (runner === "direct" ? generateEmbeddings : runDurableIndexingJob)(store, {
      maxDocsPerBatch: 1, signal: controller.signal,
      onProgress: info => { if (info.chunksEmbedded === 1) controller.abort(new Error("cancel after committed batch")); },
    });
    expect(vectors(store)).toEqual({ physical: [{ hash_seq: "one_0" }], metadata: [{ hash: "one", seq: 0 }] });
    expect(result).toMatchObject({ status: "cancelled", docsSelected: 2, docsProcessed: 1, chunksAttempted: 1, chunksCommitted: 1, chunksEmbedded: 1, chunksFailed: 0, errors: 0 });
    if (runner === "durable") {
      expect(store.db.prepare("SELECT status, docs_total, docs_completed, chunks_committed FROM indexing_checkpoints").all())
        .toEqual([{ status: "cancelled", docs_total: 2, docs_completed: 1, chunks_committed: 1 }]);
    }
  });

  it.each([
    ["direct", "callback-throw"], ["durable", "callback-throw"],
    ["direct", "abort-and-throw"], ["durable", "abort-and-throw"],
    ["direct", "microtask-abort"], ["durable", "microtask-abort"],
  ] as const)("keeps commit-verified accounting for %s at %s", async (runner, timing) => {
    const store = fixture();
    seed(store, "one"); seed(store, "two");
    const controller = new AbortController();
    let fired = false;
    const result = await (runner === "direct" ? generateEmbeddings : runDurableIndexingJob)(store, {
      maxDocsPerBatch: 1, signal: controller.signal,
      onProgress: info => {
        if (fired || info.chunksEmbedded !== 1) return;
        fired = true;
        if (timing === "microtask-abort") {
          queueMicrotask(() => queueMicrotask(() => controller.abort(new Error("cancel at session boundary"))));
        } else {
          if (timing === "abort-and-throw") controller.abort(new Error("cancel at progress callback"));
          throw new Error("synthetic progress failure");
        }
      },
    });
    expect(vectors(store).physical).toHaveLength(1);
    expect(vectors(store).metadata).toHaveLength(1);
    expect(result).toMatchObject({ status: timing === "callback-throw" ? "partial" : "cancelled", docsSelected: 2, docsProcessed: 1, chunksAttempted: 1, chunksCommitted: 1, chunksEmbedded: 1, chunksFailed: 0 });
    if (timing === "callback-throw") expect(result.firstFailureReason).toContain("synthetic progress failure");
    if (runner === "durable") {
      expect(store.db.prepare("SELECT docs_completed, chunks_committed FROM indexing_checkpoints").get())
        .toEqual({ docs_completed: 1, chunks_committed: 1 });
    }
  });

  it.each(["direct", "durable"] as const)("does not count a staged vector whose %s batch commit fails", async runner => {
    const store = fixture(); seed(store, "one");
    const exec = store.db.exec.bind(store.db);
    let fail = true;
    vi.spyOn(store.db, "exec").mockImplementation(sql => {
      if (fail && /^RELEASE SAVEPOINT/.test(sql) && store.db.prepare("SELECT 1 FROM content_vectors LIMIT 1").get()) {
        fail = false;
        throw new Error("synthetic batch commit failure");
      }
      exec(sql);
    });
    const result = await (runner === "direct" ? generateEmbeddings : runDurableIndexingJob)(store);
    expect(result).toMatchObject({ status: "failed", docsProcessed: 0, chunksAttempted: 1, chunksCommitted: 0, chunksFailed: 1 });
    expect(vectors(store)).toEqual({ physical: [], metadata: [] });
  });

  it.each(["direct", "durable"] as const)("retains %s committed counters when a session deadline expires during later tokenization", async runner => {
    const store = fixture(); seed(store, "one"); seed(store, "two");
    const gate = barrier();
    (store.llm as unknown as ReturnType<typeof backend>).tokenize.mockImplementation(async text => {
      if (text.endsWith("two")) { gate.enter(); await gate.resumed; }
      return text.split(/\s+/).map((_, i) => i);
    });
    vi.useFakeTimers();
    const pending = (runner === "direct" ? generateEmbeddings : runDurableIndexingJob)(store, { maxDocsPerBatch: 1, maxDuration: 1000 });
    try {
      await gate.entered;
      await vi.advanceTimersByTimeAsync(1000);
      gate.release();
      const result = await pending;
      expect(result).toMatchObject({ status: "cancelled", docsProcessed: 1, chunksAttempted: 1, chunksCommitted: 1 });
      expect(vectors(store).physical).toHaveLength(1);
      expect(vectors(store).metadata).toHaveLength(1);
    } finally { gate.release(); await pending; vi.useRealTimers(); }
  });

  it.each(["direct", "durable"] as const)("does not report a partly committed multi-chunk document as fully processed (%s)", async runner => {
    const store = fixture(); seed(store, "one", "docs", "synthetic chunk words ".repeat(6000));
    const controller = new AbortController();
    const result = await (runner === "direct" ? generateEmbeddings : runDurableIndexingJob)(store, {
      signal: controller.signal,
      onProgress: info => { if (info.chunksEmbedded > 0) controller.abort(); },
    });
    expect(store.db.prepare("SELECT total_chunks FROM content_chunk_expectations").get().total_chunks).toBeGreaterThan(32);
    expect(result).toMatchObject({ status: "cancelled", docsSelected: 1, docsProcessed: 0, chunksAttempted: 32, chunksCommitted: 32 });
    expect(vectors(store).physical).toHaveLength(32);
    expect(vectors(store).metadata).toHaveLength(32);
  });

  it.each(["direct", "durable"] as const)("zeros only rolled-back force work on a %s callback failure, preserving the prior index", async runner => {
    const store = fixture(); seed(store, "one"); seed(store, "two");
    const run = runner === "direct" ? generateEmbeddings : runDurableIndexingJob;
    await run(store);
    const before = vectors(store);
    const result = await run(store, {
      force: true, maxDocsPerBatch: 1,
      onProgress: info => { if (info.chunksEmbedded > 0) throw new Error("synthetic force callback failure"); },
    });
    expect(result).toMatchObject({ status: "failed", docsSelected: 2, docsProcessed: 0, chunksAttempted: 1, chunksCommitted: 0, chunksEmbedded: 0 });
    expect(vectors(store)).toEqual(before);
  });
});

describe("quality: close reserves SDK/connection ownership before awaiting disposal", () => {
  it.each(["resolve", "reject"] as const)("blocks update and all public operations while disposal will %s, then closes the DB exactly once", async disposition => {
    const root = directory();
    writeFileSync(join(root, "synthetic.md"), "qualityneedle closing fixture");
    const config: CollectionConfig = { collections: { docs: { path: root, pattern: "**/*.md" } } };
    const sdk = await openSdk({ dbPath: ":memory:", config });
    const gate = barrier();
    const failure = new Error("synthetic disposal failure");
    const dispose = vi.spyOn(sdk.internal.llm!, "dispose").mockImplementation(async () => {
      gate.enter(); await gate.resumed;
      if (disposition === "reject") throw failure;
    });
    const clear = vi.spyOn(sdk.internal, "clearCache");
    const close = vi.spyOn(sdk.internal, "close");
    const closePromise = sdk.close();
    void closePromise.catch(() => {});
    try {
      // Refuse synchronously, before even the deferred disposal starts.
      await expect(sdk.update()).rejects.toThrow(/closing|closed|busy/i);
      await gate.entered;
      expect(sdk.close()).toBe(closePromise);
      for (const operation of [
        () => sdk.update(), () => sdk.embed(), () => sdk.addCollection("foreign", { path: root }),
        () => sdk.removeCollection("docs"), () => sdk.renameCollection("docs", "foreign"),
        () => sdk.addContext("docs", "/", "foreign"), () => sdk.removeContext("docs", "/"),
        () => sdk.setGlobalContext("foreign"), () => sdk.expandQuery("foreign"),
        () => sdk.search({ query: "foreign" }), () => sdk.searchVector("foreign"), () => sdk.searchLex("foreign"),
        () => sdk.listCollections(), () => sdk.getDefaultCollectionNames(), () => sdk.getGlobalContext(),
        () => sdk.listContexts(), () => sdk.getStatus(), () => sdk.getIndexHealth(),
        () => sdk.get("qmd://docs/one.md"), () => sdk.getDocumentBody("qmd://docs/one.md"), () => sdk.multiGet("**/*.md"),
      ]) await expect(operation()).rejects.toThrow(/closing|closed|busy/i);
      expect(() => sdk.internal.db.prepare("SELECT 1")).toThrow(/busy/i);
      expect(clear).not.toHaveBeenCalled();
      expect(config).toEqual({ collections: { docs: { path: root, pattern: "**/*.md" } } });
    } finally { gate.release(); }
    if (disposition === "reject") await expect(closePromise).rejects.toBe(failure);
    else await expect(closePromise).resolves.toBeUndefined();
    expect(() => sdk.internal.db.prepare("SELECT 1").get()).toThrow(/not open|closed/i);
    expect(sdk.close()).toBe(closePromise);
    await expect(sdk.update()).rejects.toThrow(/closing|closed/i);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("a close rejected before acquiring the connection does not poison retry", async () => {
    const sdk = await openSdk({ dbPath: ":memory:" });
    const gate = barrier();
    const dispose = vi.spyOn(sdk.internal.llm!, "dispose");
    const pending = withExclusiveDatabaseOperation(sdk.internal.db, "synthetic active update", async () => {
      gate.enter(); await gate.resumed;
    });
    try {
      await gate.entered;
      await expect(sdk.close()).rejects.toMatchObject({ code: "QMD_DATABASE_BUSY" });
      expect(dispose).not.toHaveBeenCalled();
    } finally { gate.release(); await pending; }
    await expect(sdk.listCollections()).resolves.toEqual([]);
    await expect(sdk.close()).resolves.toBeUndefined();
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(() => sdk.internal.db.prepare("SELECT 1")).toThrow(/not open|closed/i);
  });
});

describe("quality: collection removal rolls back database and config failures", () => {
  async function removalFixture(mode: "yaml" | "inline") {
    const root = directory();
    const config: CollectionConfig = { collections: {
      docs: { path: root, pattern: "**/*.md", context: { "/": "owned context" } },
      keep: { path: root, pattern: "**/*.md" },
    }, global_context: "unchanged global" };
    const path = join(root, "config.yml");
    writeFileSync(path, `# retain original formatting on failed removal\n${YAML.stringify(config)}`);
    const sdk = await openSdk({ dbPath: ":memory:", ...(mode === "yaml" ? { configPath: path } : { config }) });
    sdk.internal.ensureVecTable(4);
    seed(sdk.internal, "one"); seed(sdk.internal, "shared", "docs"); seed(sdk.internal, "shared", "keep");
    for (const hash of ["one", "shared"]) sdk.internal.insertEmbedding(hash, 0, 0, new Float32Array([1, 0, 0, 0]), "quality-model", timestamp);
    return { root, config, path, sdk };
  }

  it.each(["unreadable", "unwritable-file", "unwritable-directory"] as const)("preserves indexed documents, FTS, config and shared vectors for an %s YAML source", async failure => {
    const { root, path, sdk } = await removalFixture("yaml");
    const before = removalState(sdk.internal);
    const original = readFileSync(path, "utf8");
    const saved = join(root, "saved.yml");
    if (failure === "unreadable") { renameSync(path, saved); mkdirSync(path); }
    else if (failure === "unwritable-file") chmodSync(path, 0o400);
    else chmodSync(root, 0o500);
    try { await expect(sdk.removeCollection("docs")).rejects.toThrow(); }
    finally {
      if (failure === "unreadable") { rmSync(path, { recursive: true }); renameSync(saved, path); }
      else { chmodSync(root, 0o700); chmodSync(path, 0o600); }
    }
    expect(removalState(sdk.internal)).toEqual(before);
    expect(readFileSync(path, "utf8")).toBe(original);
    expect(await sdk.searchLex("qualityneedle", { collection: "docs" })).toHaveLength(2);
  });

  it("does not delete DB documents when inline collections cannot be mutated", async () => {
    const { config, sdk } = await removalFixture("inline");
    Object.freeze(config.collections);
    const before = removalState(sdk.internal);
    await expect(sdk.removeCollection("docs")).rejects.toThrow();
    expect(removalState(sdk.internal)).toEqual(before);
    expect(Object.keys(config.collections)).toEqual(["docs", "keep"]);
  });

  it.each(["inline", "yaml"] as const)("compensates persisted %s config when the outer DB commit fails", async mode => {
    const { config, path, sdk } = await removalFixture(mode);
    const before = removalState(sdk.internal);
    const original = readFileSync(path, "utf8");
    const configBefore = structuredClone(config);
    const exec = sdk.internal.db.exec.bind(sdk.internal.db);
    let firstSavepoint: string | undefined;
    let failed = false;
    let publishedAtCommit = false;
    vi.spyOn(sdk.internal.db, "exec").mockImplementation(sql => {
      if (sql.startsWith("SAVEPOINT ") && !firstSavepoint) firstSavepoint = sql.slice("SAVEPOINT ".length);
      if (!failed && sql === `RELEASE SAVEPOINT ${firstSavepoint}`) {
        failed = true;
        const current = mode === "inline" ? config : YAML.parse(readFileSync(path, "utf8"));
        publishedAtCommit = !Object.hasOwn(current.collections, "docs");
        throw new Error("synthetic outer commit failure");
      }
      exec(sql);
    });
    await expect(sdk.removeCollection("docs")).rejects.toThrow("synthetic outer commit failure");
    expect(publishedAtCommit).toBe(true);
    expect(removalState(sdk.internal)).toEqual(before);
    expect(config).toEqual(configBefore);
    expect(readFileSync(path, "utf8")).toBe(original);
  });

  it("keeps config bytes untouched when the DB delete fails", async () => {
    const { path, sdk } = await removalFixture("yaml");
    const before = removalState(sdk.internal);
    const original = readFileSync(path, "utf8");
    sdk.internal.db.exec("CREATE TRIGGER refuse_removal BEFORE DELETE ON documents BEGIN SELECT RAISE(FAIL, 'synthetic delete failure'); END");
    await expect(sdk.removeCollection("docs")).rejects.toThrow("synthetic delete failure");
    expect(removalState(sdk.internal)).toEqual(before);
    expect(readFileSync(path, "utf8")).toBe(original);
  });

  it("reports both the commit and compensation failure instead of acknowledging a partial removal", async () => {
    const { root, path, sdk } = await removalFixture("yaml");
    const before = removalState(sdk.internal);
    const failure = new Error("synthetic outer commit failure");
    const exec = sdk.internal.db.exec.bind(sdk.internal.db);
    let firstSavepoint: string | undefined;
    let failed = false;
    vi.spyOn(sdk.internal.db, "exec").mockImplementation(sql => {
      if (sql.startsWith("SAVEPOINT ") && !firstSavepoint) firstSavepoint = sql.slice("SAVEPOINT ".length);
      if (!failed && sql === `RELEASE SAVEPOINT ${firstSavepoint}`) {
        failed = true;
        chmodSync(root, 0o500);
        throw failure;
      }
      exec(sql);
    });
    try {
      await expect(sdk.removeCollection("docs")).rejects.toMatchObject({ cause: failure, errors: [failure, expect.objectContaining({ code: "EACCES" })] });
      expect(removalState(sdk.internal)).toEqual(before);
      // Cross-resource recovery itself failed, so the exception must expose it.
      expect(YAML.parse(readFileSync(path, "utf8")).collections.docs).toBeUndefined();
    } finally { chmodSync(root, 0o700); }
  });

  it.each(["inline", "yaml"] as const)("successful %s removal leaves shared content/vectors and peer/CLI sources intact", async mode => {
    const { config, path, sdk } = await removalFixture(mode);
    const cli: CollectionConfig = { collections: { cli: { path: "/synthetic/cli", pattern: "**/*.md" } } };
    setConfigSource({ config: cli });
    const peerConfig: CollectionConfig = { collections: { peer: { path: "/synthetic/peer", pattern: "**/*.md" } } };
    const peer = await openSdk({ dbPath: ":memory:", config: peerConfig });
    const beforeVectors = vectors(sdk.internal);
    expect(await sdk.removeCollection("docs")).toBe(true);
    expect(await sdk.removeCollection("docs")).toBe(false);
    expect((await sdk.listCollections()).map(c => c.name)).toEqual(["keep"]);
    expect(sdk.internal.db.prepare("SELECT collection, path FROM documents").all()).toEqual([{ collection: "keep", path: "shared.md" }]);
    expect(sdk.internal.db.prepare("SELECT filepath FROM documents_fts").all()).toEqual([{ filepath: "keep/shared.md" }]);
    expect(vectors(sdk.internal)).toEqual(beforeVectors);
    expect(sdk.internal.db.prepare("SELECT hash FROM content ORDER BY hash").all()).toEqual([{ hash: "one" }, { hash: "shared" }]);
    expect(Object.keys((mode === "inline" ? config : YAML.parse(readFileSync(path, "utf8"))).collections)).toEqual(["keep"]);
    expect((await peer.listCollections()).map(c => c.name)).toEqual(["peer"]);
    expect(Object.keys(peerConfig.collections)).toEqual(["peer"]);
    expect(loadConfig()).toBe(cli);
  });
});
