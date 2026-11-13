import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore, type CollectionConfig, type QMDStore, type StoreOptions } from "../src/index.js";
import * as collections from "../src/collections.js";
import * as llmModule from "../src/llm.js";
import * as storeModule from "../src/store.js";

const timestamp = "2026-01-01T00:00:00.000Z";
let fixture: string;
let stores: QMDStore[];

beforeEach(() => {
  fixture = mkdtempSync(join(tmpdir(), "qmd-audit-sdk-"));
  stores = [];
  vi.stubEnv("HOME", join(fixture, "home"));
  vi.stubEnv("XDG_CONFIG_HOME", join(fixture, "config"));
  vi.stubEnv("XDG_CACHE_HOME", join(fixture, "cache"));
  vi.stubEnv("QMD_CONFIG_DIR", join(fixture, "config", "qmd"));
  vi.stubEnv("QMD_EMBED_BACKEND", "gguf");
  vi.stubEnv("QMD_STRICT_MLX", "0");
  collections.setConfigIndexName("index");
  collections.setConfigSource(undefined);
});

afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.allSettled(stores.map(store => store.close()));
  llmModule.setDefaultLlamaCpp(null);
  collections.setConfigSource(undefined);
  collections.setConfigIndexName("index");
  vi.unstubAllEnvs();
  rmSync(fixture, { recursive: true, force: true });
});

function config(name: string): CollectionConfig {
  return {
    collections: {
      [name]: { path: join(fixture, name), pattern: "**/*.md" },
    },
  };
}

async function open(options: Partial<StoreOptions> = {}): Promise<QMDStore> {
  const store = await createStore({ dbPath: ":memory:", ...options });
  stores.push(store);
  return store;
}

function yamlPath(name: string, value: CollectionConfig): string {
  const path = join(fixture, `${name}.yml`);
  writeFileSync(path, YAML.stringify(value));
  return path;
}

function readYaml(path: string): CollectionConfig {
  return YAML.parse(readFileSync(path, "utf8")) as CollectionConfig;
}

async function names(store: QMDStore): Promise<string[]> {
  return (await store.listCollections()).map(collection => collection.name).sort();
}

function seed(store: QMDStore, collection: string, path: string, body: string): string {
  const hash = createHash("sha256").update(body).digest("hex");
  store.internal.insertContent(hash, body, timestamp);
  store.internal.insertDocument(collection, path, path, hash, timestamp, timestamp);
  return hash;
}

describe("SDK configuration ownership", () => {
  test("interleaves two inline stores without changing either peer or the CLI source", async () => {
    const cli = config("cli");
    collections.setConfigSource({ config: cli });
    const aConfig = config("a");
    const bConfig = config("b");
    const a = await open({ config: aConfig });
    const b = await open({ config: bConfig });

    await Promise.all([
      a.addCollection("a-added", { path: join(fixture, "a-added"), ignore: ["drafts/**"] }),
      b.addContext("b", "/", "B context"),
    ]);
    await a.setGlobalContext("A global");
    await b.renameCollection("b", "b-renamed");
    expect(aConfig.collections["a-added"]?.ignore).toEqual(["drafts/**"]);
    expect(Object.keys(aConfig.collections).sort()).toEqual(["a", "a-added"]);
    expect(Object.keys(bConfig.collections)).toEqual(["b-renamed"]);
    expect(bConfig.collections["b-renamed"]?.context).toEqual({ "/": "B context" });
    expect(await names(a)).toEqual(["a", "a-added"]);
    expect(await names(b)).toEqual(["b-renamed"]);
    expect(await b.getGlobalContext()).toBeUndefined();
    expect(collections.loadConfig()).toBe(cli);

    await a.close();
    await b.addCollection("after-close", { path: join(fixture, "after-close") });
    await b.removeContext("b-renamed", "/");
    await b.setGlobalContext("B global");
    expect(bConfig.global_context).toBe("B global");
    expect(bConfig.collections["b-renamed"]?.context).toBeUndefined();
    expect(await names(b)).toEqual(["after-close", "b-renamed"]);
    expect(collections.loadConfig()).toBe(cli);
    expect(cli).toEqual(config("cli"));
    expect(existsSync(join(fixture, "config", "qmd", "index.yml"))).toBe(false);
  });

  test("interleaves two YAML stores and keeps the surviving writer bound after peer close", async () => {
    const cliPath = yamlPath("cli", config("cli"));
    collections.setConfigSource({ configPath: cliPath });
    const cliBefore = readFileSync(cliPath, "utf8");
    const aPath = yamlPath("a", config("a"));
    const bPath = yamlPath("b", config("b"));
    const a = await open({ configPath: aPath });
    const b = await open({ configPath: bPath });

    await a.addContext("a", "/one", "A context");
    await b.setGlobalContext("B global");
    await a.renameCollection("a", "a-renamed");
    await b.addCollection("b-added", { path: join(fixture, "b-added"), ignore: ["private/**"] });
    expect(Object.keys(readYaml(aPath).collections)).toEqual(["a-renamed"]);
    expect(readYaml(aPath).collections["a-renamed"]?.context).toEqual({ "/one": "A context" });
    expect(readYaml(bPath).collections["b-added"]?.ignore).toEqual(["private/**"]);
    expect(readYaml(bPath).global_context).toBe("B global");
    expect(await names(a)).toEqual(["a-renamed"]);
    expect(await names(b)).toEqual(["b", "b-added"]);

    await b.close();
    const bBefore = readFileSync(bPath, "utf8");
    await a.removeContext("a-renamed", "/one");
    await a.setGlobalContext("A survives");
    await a.addCollection("after-close", { path: join(fixture, "after-close") });
    expect(readYaml(aPath).global_context).toBe("A survives");
    expect(readYaml(aPath).collections["a-renamed"]?.context).toBeUndefined();
    expect(readFileSync(bPath, "utf8")).toBe(bBefore);
    expect(readFileSync(cliPath, "utf8")).toBe(cliBefore);
    expect(collections.getConfigPath()).toBe(cliPath);
  });

  test("mixes YAML, inline and DB-only mutations without external write-through from DB-only", async () => {
    const inline = config("inline");
    const filePath = yamlPath("mixed", config("file"));
    const file = await open({ configPath: filePath });
    const memory = await open({ config: inline });
    const dbPath = join(fixture, "db-only.sqlite");
    const dbOnly = await open({ dbPath });
    await dbOnly.addCollection("db", { path: join(fixture, "db") });
    await dbOnly.addContext("db", "/", "DB context");
    await dbOnly.renameCollection("db", "db-renamed");
    await dbOnly.setGlobalContext("DB global");
    await file.addContext("file", "/", "File context");
    await memory.setGlobalContext("Inline global");
    await file.close();
    const fileBefore = readFileSync(filePath, "utf8");
    await memory.addCollection("survivor", { path: join(fixture, "survivor") });
    await dbOnly.removeContext("db-renamed", "/");
    await dbOnly.addCollection("removed", { path: join(fixture, "removed") });
    await dbOnly.removeCollection("removed");
    expect(Object.keys(inline.collections).sort()).toEqual(["inline", "survivor"]);
    expect(inline.global_context).toBe("Inline global");
    expect(await names(memory)).toEqual(["inline", "survivor"]);
    expect(await names(dbOnly)).toEqual(["db-renamed"]);
    expect(await dbOnly.getGlobalContext()).toBe("DB global");
    expect(readFileSync(filePath, "utf8")).toBe(fileBefore);
    await memory.close();
    await dbOnly.close();
    const reopened = await open({ dbPath });
    expect(await names(reopened)).toEqual(["db-renamed"]);
    expect(await reopened.getGlobalContext()).toBe("DB global");
    expect(await reopened.listContexts()).toEqual([{ collection: "*", path: "/", context: "DB global" }]);
    expect(existsSync(join(fixture, "config", "qmd", "index.yml"))).toBe(false);
  });

  test.each(["inline", "yaml"] as const)("collection updates preserve context and settings in %s mode", async mode => {
    const original: CollectionConfig = {
      collections: {
        docs: {
          path: join(fixture, "docs"), pattern: "**/*.md", ignore: ["old/**"],
          context: { "/": "Docs context" }, includeByDefault: false, update: "echo synthetic",
        },
      },
    };
    const path = mode === "yaml" ? yamlPath("preserve", original) : undefined;
    const store = await open(path ? { configPath: path } : { config: original });
    await store.addCollection("docs", { path: join(fixture, "new-docs"), pattern: "**/*.txt", ignore: ["new/**"] });
    const expected = {
      path: join(fixture, "new-docs"), pattern: "**/*.txt", ignore: ["new/**"],
      context: { "/": "Docs context" }, includeByDefault: false, update: "echo synthetic",
    };
    expect((path ? readYaml(path) : original).collections.docs).toEqual(expected);
    expect(storeModule.getStoreCollection(store.internal.db, "docs")).toEqual({ name: "docs", ...expected });
    expect(await store.getDefaultCollectionNames()).toEqual([]);
    await store.addCollection("docs", { path: expected.path });
    expect(storeModule.getStoreCollection(store.internal.db, "docs")?.ignore).toEqual(["new/**"]);
    expect((path ? readYaml(path) : original).collections.docs?.ignore).toEqual(["new/**"]);
    await store.addContext("docs", "/empty", "");
    expect(await store.removeContext("docs", "/empty")).toBe(true);
    expect((path ? readYaml(path) : original).collections.docs?.context).toEqual({ "/": "Docs context" });
    expect(storeModule.getStoreCollection(store.internal.db, "docs")?.context).toEqual({ "/": "Docs context" });
  });
});

describe("SDK indexed collection lifecycle", () => {
  test.each(["inline", "yaml", "db-only"] as const)("rename/remove moves documents, FTS and contexts without deleting shared content (%s)", async mode => {
    const value = config("old");
    value.collections.keep = { path: join(fixture, "keep"), pattern: "**/*.md" };
    const path = mode === "yaml" ? yamlPath("lifecycle", value) : undefined;
    const store = await open(path ? { configPath: path } : mode === "inline" ? { config: value } : {});
    if (mode === "db-only") {
      await store.addCollection("old", { path: join(fixture, "old") });
      await store.addCollection("keep", { path: join(fixture, "keep") });
    }
    await store.addContext("old", "/", "Old context");
    const hash = seed(store, "old", "shared.md", "ownershipneedle shared content");
    seed(store, "keep", "shared.md", "ownershipneedle shared content");
    seed(store, "old", "only.md", "ownershipneedle exclusive content");
    expect(await store.renameCollection("old", "renamed")).toBe(true);
    expect(await store.searchLex("ownershipneedle", { collection: "old" })).toEqual([]);
    expect((await store.searchLex("ownershipneedle", { collection: "renamed" })).map(result => result.filepath).sort())
      .toEqual(["qmd://renamed/only.md", "qmd://renamed/shared.md"]);
    expect(await store.listContexts()).toContainEqual({ collection: "renamed", path: "/", context: "Old context" });
    expect("error" in await store.get("qmd://renamed/shared.md")).toBe(false);
    expect(store.internal.db.prepare("SELECT filepath FROM documents_fts WHERE filepath LIKE 'old/%'").all()).toEqual([]);

    expect(await store.removeCollection("renamed")).toBe(true);
    expect((await store.searchLex("ownershipneedle")).map(result => result.filepath)).toEqual(["qmd://keep/shared.md"]);
    expect(store.internal.db.prepare("SELECT collection FROM documents WHERE collection IN ('old', 'renamed')").all()).toEqual([]);
    expect(await store.listContexts()).toEqual([]);
    expect(store.internal.db.prepare("SELECT hash FROM content WHERE hash = ?").get(hash)).toEqual({ hash });
    expect((await store.get("qmd://keep/shared.md", { includeBody: true }))).toMatchObject({ body: "ownershipneedle shared content" });
    expect(await names(store)).toEqual(["keep"]);
    if (mode !== "db-only") expect(Object.keys((path ? readYaml(path) : value).collections)).toEqual(["keep"]);
  });
});

describe("SDK backend and search scope", () => {
  test("defaults MLX fallback off and preserves explicit per-store model/backend settings", async () => {
    const a = await open({ config: { collections: {}, models: {
      embedBackend: "mlx", embed: "synthetic-mlx", generate: "synthetic-generate", rerank: "synthetic-rerank",
      mlxUrl: "http://127.0.0.1:19991", mlxConcurrency: 3,
    } } });
    const b = await open({ config: { collections: {}, models: {
      embedBackend: "gguf", embed: "synthetic-gguf", mlxFallback: true, mlxConcurrency: 1,
    } } });
    expect(a.internal.llm).toMatchObject({
      embedBackend: "mlx", embedModelUri: "synthetic-mlx", generateModelUri: "synthetic-generate",
      rerankModelUri: "synthetic-rerank", mlxUrlOverride: "http://127.0.0.1:19991", mlxFallback: false, mlxConcurrency: 3,
    });
    // The legacy flag remains accepted, but the LLM contract forbids silently
    // replacing an embedding space even when an older config requests fallback.
    expect(b.internal.llm).toMatchObject({ embedBackend: "gguf", embedModelUri: "synthetic-gguf", mlxFallback: false, mlxConcurrency: 1 });
    await b.close();
    expect(a.internal.llm?.embedModelName).toBe("synthetic-mlx");
    const dbOnly = await open();
    expect(dbOnly.internal.llm).toMatchObject({ mlxFallback: false });
  });

  test("omitted backend settings keep environment precedence without adopting a peer config", async () => {
    vi.stubEnv("QMD_EMBED_BACKEND", "mlx");
    vi.stubEnv("QMD_MLX_CONCURRENCY", "4");
    vi.stubEnv("QMD_MLX_EMBED_URL", "http://127.0.0.1:19992");
    const a = await open({ config: { collections: {}, models: { embedBackend: "gguf", mlxConcurrency: 1 } } });
    const b = await open();
    expect(a.internal.llm).toMatchObject({ embedBackend: "gguf", mlxConcurrency: 1 });
    expect(b.internal.llm).toMatchObject({ embedBackend: "mlx", mlxConcurrency: 4, mlxUrlOverride: "http://127.0.0.1:19992", mlxFallback: false });
  });

  test("vector search calls only the owning backend and uses its model identity", async () => {
    const globalEmbed = vi.fn(async () => { throw new Error("global backend must not be used"); });
    llmModule.setDefaultLlamaCpp({ embed: globalEmbed } as any);
    const a = await open();
    const b = await open();
    const aHash = seed(a, "a", "one.md", "vector alpha");
    const bHash = seed(b, "b", "one.md", "vector beta");
    a.internal.ensureVecTable(3);
    b.internal.ensureVecTable(3);
    a.internal.insertEmbedding(aHash, 0, 0, new Float32Array([1, 0, 0]), "synthetic-a", timestamp);
    b.internal.insertEmbedding(bHash, 0, 0, new Float32Array([0, 1, 0]), "synthetic-b", timestamp);
    const aEmbed = vi.fn(async (_text: string, _options?: llmModule.EmbedOptions) => ({ embedding: [1, 0, 0], model: "synthetic-a" }));
    const bEmbed = vi.fn(async (_text: string, _options?: llmModule.EmbedOptions) => ({ embedding: [0, 1, 0], model: "synthetic-b" }));
    a.internal.llm = { embedModelName: "synthetic-a", embed: aEmbed } as any;
    b.internal.llm = { embedModelName: "synthetic-b", embed: bEmbed } as any;
    expect((await a.searchVector("alpha", { collection: "a", limit: 1 })).map(result => result.filepath)).toEqual(["qmd://a/one.md"]);
    expect((await b.searchVector("beta", { collection: "b", limit: 1 })).map(result => result.filepath)).toEqual(["qmd://b/one.md"]);
    expect(globalEmbed).not.toHaveBeenCalled();
    expect(aEmbed).toHaveBeenCalledTimes(1);
    expect(bEmbed).toHaveBeenCalledTimes(1);
    expect(aEmbed.mock.calls[0]?.[1]).toMatchObject({ model: "synthetic-a", isQuery: true });
    expect(bEmbed.mock.calls[0]?.[1]).toMatchObject({ model: "synthetic-b", isQuery: true });
  });

  test("simple and structured queries forward the complete deduplicated collection scope", async () => {
    const store = await open();
    for (const name of ["a", "b", "excluded"]) {
      await store.addCollection(name, { path: join(fixture, name) });
      seed(store, name, "one.md", `multiscope ${name} content`);
    }
    store.internal.expandQuery = vi.fn(async () => []);
    const hybrid = vi.spyOn(storeModule, "hybridQuery");
    const structured = vi.spyOn(storeModule, "structuredSearch");
    const scope = { collection: "a", collections: ["b", "a", "b"], rerank: false };
    const simpleResults = await store.search({ query: "multiscope", ...scope });
    const structuredResults = await store.search({ queries: [{ type: "lex", query: "multiscope" }], ...scope });
    expect(hybrid.mock.calls[0]?.[2]).toMatchObject({ collections: ["a", "b"] });
    expect(structured.mock.calls[0]?.[2]).toMatchObject({ collections: ["a", "b"] });
    expect(simpleResults.map(result => result.file).sort()).toEqual(["qmd://a/one.md", "qmd://b/one.md"]);
    expect(structuredResults.map(result => result.file).sort()).toEqual(["qmd://a/one.md", "qmd://b/one.md"]);
  });
});

describe("SDK exception-safe lifecycle", () => {
  test("invalid YAML closes the allocated database and leaves the CLI source alone", async () => {
    const cli = config("cli");
    collections.setConfigSource({ config: cli });
    const path = join(fixture, "invalid.yml");
    writeFileSync(path, "collections: [unterminated");
    const allocate = vi.spyOn(storeModule, "createStore");
    await expect(createStore({ dbPath: join(fixture, "invalid.sqlite"), configPath: path })).rejects.toThrow("Failed to parse");
    const internal = allocate.mock.results[0]?.value as storeModule.Store;
    expect(internal).toBeDefined();
    expect(() => internal.db.prepare("SELECT 1").get()).toThrow();
    expect(collections.loadConfig()).toBe(cli);
  });

  test("a partially failing config sync is rolled back and the database is closed", async () => {
    const dbPath = join(fixture, "rollback.sqlite");
    const baseline = await open({ dbPath, config: config("original") });
    await baseline.close();
    const originalSync = storeModule.syncConfigToDb;
    vi.spyOn(storeModule, "syncConfigToDb").mockImplementationOnce((db, value) => {
      originalSync(db, value);
      throw new Error("synthetic sync failure");
    });
    const allocate = vi.spyOn(storeModule, "createStore");
    await expect(createStore({ dbPath, config: config("replacement") })).rejects.toThrow("synthetic sync failure");
    const failed = allocate.mock.results[0]?.value as storeModule.Store;
    expect(() => failed.db.prepare("SELECT 1").get()).toThrow();
    const reopened = await open({ dbPath });
    expect(await names(reopened)).toEqual(["original"]);
  });

  test("LLM construction failure rolls back configuration and closes the database", async () => {
    const dbPath = join(fixture, "llm-failure.sqlite");
    const baseline = await open({ dbPath, config: config("original") });
    await baseline.close();
    const failure = new Error("synthetic LLM construction failure");
    const constructor = vi.spyOn(llmModule, "LlamaCpp").mockImplementationOnce(function () { throw failure; } as any);
    const allocate = vi.spyOn(storeModule, "createStore");
    await expect(createStore({ dbPath, config: config("replacement") })).rejects.toBe(failure);
    const failed = allocate.mock.results[0]?.value as storeModule.Store;
    expect(() => failed.db.prepare("SELECT 1").get()).toThrow();
    constructor.mockRestore();
    const reopened = await open({ dbPath });
    expect(await names(reopened)).toEqual(["original"]);
  });

  test("a commit failure disposes the constructed LLM and rolls back before closing", async () => {
    const dbPath = join(fixture, "commit-failure.sqlite");
    const baseline = await open({ dbPath, config: config("original") });
    await baseline.close();
    const failure = new Error("synthetic commit failure");
    const originalCreate = storeModule.createStore;
    const allocate = vi.spyOn(storeModule, "createStore").mockImplementationOnce(path => {
      const internal = originalCreate(path);
      const exec = internal.db.exec.bind(internal.db);
      vi.spyOn(internal.db, "exec").mockImplementation(sql => {
        if (sql === "COMMIT") throw failure;
        exec(sql);
      });
      return internal;
    });
    const dispose = vi.spyOn(llmModule.LlamaCpp.prototype, "dispose");
    await expect(createStore({ dbPath, config: config("replacement") })).rejects.toBe(failure);
    const failed = allocate.mock.results[0]?.value as storeModule.Store;
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(() => failed.db.prepare("SELECT 1").get()).toThrow();
    const reopened = await open({ dbPath });
    expect(await names(reopened)).toEqual(["original"]);
  });

  test("construction cleanup failure retains the original error and still closes the DB", async () => {
    const failure = new Error("synthetic config setup failure");
    const closeFailure = new Error("synthetic construction close failure");
    const originalCreate = storeModule.createStore;
    const allocate = vi.spyOn(storeModule, "createStore").mockImplementationOnce(path => {
      const internal = originalCreate(path);
      const close = internal.close;
      vi.spyOn(internal, "close").mockImplementation(() => { close(); throw closeFailure; });
      return internal;
    });
    vi.spyOn(storeModule, "syncConfigToDb").mockImplementationOnce(() => { throw failure; });
    await expect(createStore({ dbPath: ":memory:", config: config("failed") })).rejects.toMatchObject({
      cause: failure, errors: [failure, closeFailure],
    });
    const failed = allocate.mock.results[0]?.value as storeModule.Store;
    expect(() => failed.db.prepare("SELECT 1").get()).toThrow();
  });

  test("concurrent and repeated closes share one cleanup operation", async () => {
    const store = await open();
    const dispose = vi.spyOn(store.internal.llm!, "dispose");
    const close = vi.spyOn(store.internal, "close");
    const first = store.close();
    const second = store.close();
    expect(second).toBe(first);
    await Promise.all([first, second]);
    await store.close();
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);
    await expect(store.getStatus()).rejects.toThrow();
  });

  test("a rejecting LLM dispose still closes the database exactly once", async () => {
    const store = await open();
    const failure = new Error("synthetic dispose failure");
    const dispose = vi.spyOn(store.internal.llm!, "dispose").mockRejectedValue(failure);
    const close = vi.spyOn(store.internal, "close");
    const first = store.close();
    const second = store.close();
    const results = await Promise.allSettled([first, second]);
    expect(results).toEqual([{ status: "rejected", reason: failure }, { status: "rejected", reason: failure }]);
    expect(() => store.internal.db.prepare("SELECT 1").get()).toThrow();
    await expect(store.close()).rejects.toBe(failure);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);
  });

  test("reports both cleanup errors without skipping database cleanup", async () => {
    const store = await open();
    const disposeError = new Error("synthetic LLM disposal failure");
    const closeError = new Error("synthetic DB close failure");
    vi.spyOn(store.internal.llm!, "dispose").mockRejectedValue(disposeError);
    const originalClose = store.internal.close;
    const close = vi.spyOn(store.internal, "close").mockImplementation(() => {
      originalClose();
      throw closeError;
    });
    await expect(store.close()).rejects.toMatchObject({ errors: [disposeError, closeError] });
    expect(close).toHaveBeenCalledTimes(1);
    expect(() => store.internal.db.prepare("SELECT 1").get()).toThrow();
  });
});
