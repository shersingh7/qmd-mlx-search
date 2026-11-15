import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import { createStore as createSdk, type QMDStore } from "../src/index.js";
import { generateEmbeddings, structuredSearch, vectorSearchQuery, type Store } from "../src/store.js";
import { computeEmbeddingSpaceId, type EmbeddingDescriptor } from "../src/embedding/contract.js";
import { canUnloadLLM } from "../src/llm.js";

const timestamp = "2026-01-01T00:00:00.000Z";
const vector = [1, 0, 0, 0];
const stores: QMDStore[] = [];
const descriptor = (overrides: Partial<EmbeddingDescriptor> = {}): EmbeddingDescriptor => ({
  version: 1, backend: "mlx", model: "public-fixture-encoder", revision: "revision-A",
  tokenizer: "fixture-tokenizer", pooling: "mean", nativeDimensions: 4, outputDimensions: 4,
  maxTokens: 256, normalized: true, dtype: "float32", quantization: "none", ...overrides,
});
function backend(initial = descriptor()) {
  const state = { descriptor: initial };
  return {
    state,
    embedModelName: initial.model,
    getDescriptor: vi.fn(async () => state.descriptor),
    tokenize: vi.fn(async (text: string) => text.split(/\s+/).map((_, i) => i)),
    embed: vi.fn(async (_text: string, _options?: unknown) => ({ embedding: [...vector], model: state.descriptor.model })),
    embedBatch: vi.fn(async (texts: string[], _options?: unknown) => texts.map(() => ({ embedding: [...vector], model: state.descriptor.model }))),
    dispose: vi.fn(async () => {}),
  };
}
async function fixture() {
  const sdk = await createSdk({ dbPath: ":memory:", config: {
    collections: { docs: { path: "/synthetic/public", pattern: "**/*.md" } },
  } });
  stores.push(sdk);
  const own = backend();
  sdk.internal.llm = own as any;
  sdk.internal.insertContent("public-hash", "identityneedle synthetic public transaction note", timestamp);
  sdk.internal.insertDocument("docs", "note.md", "Public note", "public-hash", timestamp, timestamp);
  sdk.internal.ensureVecTable(4, own.state.descriptor);
  sdk.internal.insertEmbedding("public-hash", 0, 0, new Float32Array(vector), own.embedModelName, timestamp);
  // No generation/reranking/model downloads are needed by any search fixture.
  sdk.internal.expandQuery = vi.fn(async () => []);
  return { sdk, store: sdk.internal, own };
}
function state(store: Store) {
  return {
    schema: store.db.prepare("SELECT name, sql FROM sqlite_master ORDER BY name").all(),
    config: store.db.prepare("SELECT * FROM store_config ORDER BY key").all(),
    vectors: store.db.prepare("SELECT hash_seq, hex(embedding) AS embedding FROM vectors_vec ORDER BY hash_seq").all(),
    chunks: store.db.prepare("SELECT * FROM content_vectors ORDER BY hash, seq").all(),
    cache: store.db.prepare("SELECT * FROM llm_cache ORDER BY hash").all(),
  };
}
function barrier() {
  let enter!: () => void;
  let release!: () => void;
  return {
    entered: new Promise<void>(resolve => { enter = resolve; }),
    resumed: new Promise<void>(resolve => { release = resolve; }),
    enter: () => enter(), release: () => release(),
  };
}
const routes = {
  vector: (sdk: QMDStore) => sdk.searchVector("identityneedle"),
  structured: (sdk: QMDStore) => sdk.search({ queries: [{ type: "vec", query: "identityneedle" }], rerank: false }),
  hyde: (sdk: QMDStore) => sdk.search({ queries: [{ type: "hyde", query: "identityneedle" }], rerank: false }),
  hybrid: (sdk: QMDStore) => sdk.search({ query: "identityneedle", rerank: false }),
  expanded: (sdk: QMDStore) => vectorSearchQuery(sdk.internal, "identityneedle"),
};
const routeNames = Object.keys(routes) as (keyof typeof routes)[];
beforeEach(() => {
  vi.stubEnv("QMD_EMBED_BACKEND", "gguf");
  vi.stubEnv("QMD_STRICT_MLX", "0");
});
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(stores.splice(0).map(store => store.close()));
  vi.unstubAllEnvs();
});

describe("query-side persisted/serving embedding identity", () => {
  it.each(routeNames)("accepts matching identity without query writes (%s)", async route => {
    const { sdk, store } = await fixture();
    const before = state(store);
    expect(await routes[route](sdk)).toHaveLength(1);
    expect(state(store)).toEqual(before);
  });

  it.each(routeNames)("refuses same-dimensional revision drift before inference (%s)", async route => {
    const { sdk, store } = await fixture();
    const next = backend(descriptor({ revision: "revision-B" }));
    store.llm = next as any;
    const before = state(store);
    await expect(routes[route](sdk)).rejects.toThrow(/embedding space|rebuild|re-embed/i);
    expect(next.embed).not.toHaveBeenCalled();
    expect(next.embedBatch).not.toHaveBeenCalled();
    expect(state(store)).toEqual(before);
  });

  it.each([
    ["backend", { backend: "gguf" }], ["model", { model: "other-encoder" }],
    ["tokenizer", { tokenizer: "other-tokenizer" }], ["pooling", { pooling: "cls" }],
    ["queryPrefix", { queryPrefix: "different query: " }], ["docPrefix", { docPrefix: "different document: " }],
    ["context", { maxTokens: 512 }], ["normalization", { normalized: false }],
  ] as [string, Partial<EmbeddingDescriptor>][])("compares %s, not just dimensions", async (_field, change) => {
    const { sdk, store } = await fixture();
    store.llm = backend(descriptor(change)) as any;
    const before = state(store);
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/embedding space|rebuild|re-embed/i);
    await expect(routes.structured(sdk)).rejects.toThrow(/embedding space|rebuild|re-embed/i);
    expect(state(store)).toEqual(before);
  });

  it.each(routeNames)("rejects the actual legacy pre-maxTokens fingerprint (%s)", async route => {
    const { sdk, store, own } = await fixture();
    const d = own.state.descriptor;
    const oldCanonical = {
      v: d.version, b: d.backend, m: d.model, t: d.tokenizer ?? "", r: d.revision ?? "", p: d.pooling,
      qp: d.queryPrefix ?? null, dp: d.docPrefix ?? null, nd: d.nativeDimensions, od: d.outputDimensions,
      n: d.normalized, dt: d.dtype ?? "", q: d.quantization ?? "",
    };
    const legacyId = createHash("sha256").update(JSON.stringify(oldCanonical, Object.keys(oldCanonical).sort())).digest("hex").slice(0, 24);
    expect(legacyId).not.toBe(computeEmbeddingSpaceId(d));
    store.db.prepare("UPDATE store_config SET value = ? WHERE key = 'embedding_space_id'").run(legacyId);
    const before = state(store);
    await expect(routes[route](sdk)).rejects.toThrow(/embedding space|fingerprint|rebuild|re-embed/i);
    expect(own.embed).not.toHaveBeenCalled();
    expect(own.embedBatch).not.toHaveBeenCalled();
    expect(state(store)).toEqual(before);
  });

  it.each([
    ["missing ID", "DELETE FROM store_config WHERE key = 'embedding_space_id'"],
    ["missing descriptor", "DELETE FROM store_config WHERE key = 'embedding_descriptor'"],
    ["missing model", "DELETE FROM store_config WHERE key = 'embedding_model'"],
    ["forged ID", "UPDATE store_config SET value = 'legacy-fingerprint-policy' WHERE key = 'embedding_space_id'"],
    ["invalid JSON", "UPDATE store_config SET value = '{invalid' WHERE key = 'embedding_descriptor'"],
    ["invalid descriptor", "UPDATE store_config SET value = '{}' WHERE key = 'embedding_descriptor'"],
    ["wrong model", "UPDATE store_config SET value = 'another-model' WHERE key = 'embedding_model'"],
    ["all identity absent", "DELETE FROM store_config WHERE key IN ('embedding_space_id', 'embedding_model', 'embedding_descriptor')"],
  ])("refuses %s without repairing the database", async (_name, sql) => {
    const { sdk, store } = await fixture();
    store.db.exec(sql!);
    const before = state(store);
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/embedding|descriptor|metadata/i);
    await expect(routes.structured(sdk)).rejects.toThrow(/embedding|descriptor|metadata/i);
    expect(state(store)).toEqual(before);
    expect(await sdk.searchLex("identityneedle")).toHaveLength(1);
    expect(await sdk.search({ queries: [{ type: "lex", query: "identityneedle" }], rerank: false })).toHaveLength(1);
    expect(state(store)).toEqual(before);
  });

  it.each(["null", "absent"])("does not downgrade an MLX index when the serving descriptor is %s", async mode => {
    const { sdk, store } = await fixture();
    store.llm = {
      embedModelName: "public-fixture-encoder", embedBackend: "gguf",
      ...(mode === "null" ? { getDescriptor: async () => null } : {}),
      embed: async () => ({ embedding: vector, model: "public-fixture-encoder" }),
      embedBatch: async (texts: string[]) => texts.map(() => ({ embedding: vector, model: "public-fixture-encoder" })),
    } as any;
    const before = state(store);
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/descriptor|embedding/i);
    await expect(routes.structured(sdk)).rejects.toThrow(/descriptor|embedding/i);
    expect(state(store)).toEqual(before);
  });

  it.each(["vector", "structured"] as const)("pins the descriptor across %s inference, rejecting same-size daemon replacement", async route => {
    const { sdk, store, own } = await fixture();
    own.embed.mockImplementation(async () => {
      own.state.descriptor = descriptor({ revision: "revision-B" });
      return { embedding: [...vector], model: own.embedModelName };
    });
    own.embedBatch.mockImplementation(async texts => {
      own.state.descriptor = descriptor({ revision: "revision-B" });
      return texts.map(() => ({ embedding: [...vector], model: own.embedModelName }));
    });
    const before = state(store);
    await expect(routes[route](sdk)).rejects.toThrow(/embedding space|changed|rebuild/i);
    expect(state(store)).toEqual(before);
  });

  it("rejects a response's explicit incompatible descriptor even if the next preflight returns A again", async () => {
    const { sdk, own } = await fixture();
    own.embed.mockResolvedValue({ embedding: [...vector], model: own.embedModelName, descriptor: descriptor({ revision: "revision-B" }) } as any);
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/embedding space|descriptor/i);
  });

  it("passes an immutable detached descriptor to actual embedding work", async () => {
    const { sdk, own } = await fixture();
    let pinned: EmbeddingDescriptor | undefined;
    own.embed.mockImplementation(async (_text, options) => {
      pinned = (options as { expectedDescriptor: EmbeddingDescriptor }).expectedDescriptor;
      expect(Object.isFrozen(pinned)).toBe(true);
      (own.state.descriptor as any).revision = "revision-B";
      expect(pinned.revision).toBe("revision-A");
      return { embedding: [...vector], model: own.embedModelName };
    });
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/embedding space|changed/i);
    expect(pinned?.revision).toBe("revision-A");
  });

  it("rejects a query response from a different model even when descriptor probes agree", async () => {
    const { sdk, own } = await fixture();
    own.embed.mockResolvedValue({ embedding: [...vector], model: "unproven-other-encoder" });
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/model|embedding/i);
  });

  it.each(["descriptor", "embedding"])("rechecks persisted metadata after awaiting %s", async timing => {
    const { sdk, store, own } = await fixture();
    const changeMetadata = () => store.db.prepare("UPDATE store_config SET value = 'changed-during-query' WHERE key = 'embedding_space_id'").run();
    if (timing === "descriptor") own.getDescriptor.mockImplementation(async () => { changeMetadata(); return own.state.descriptor; });
    else own.embed.mockImplementation(async () => { changeMetadata(); return { embedding: [...vector], model: own.embedModelName }; });
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/embedding|changed/i);
    expect(store.db.prepare("SELECT value FROM store_config WHERE key = 'embedding_space_id'").get()).toEqual({ value: "changed-during-query" });
  });

  it("rechecks connection ownership after inference awaits a concurrent forced rebuild", async () => {
    const { sdk, store, own } = await fixture();
    const queryGate = barrier();
    own.embed.mockImplementation(async () => { queryGate.enter(); await queryGate.resumed; return { embedding: [...vector], model: own.embedModelName }; });
    const query = sdk.searchVector("identityneedle");
    const refusal = expect(query).rejects.toMatchObject({ code: "QMD_DATABASE_BUSY" });
    await queryGate.entered;
    const forceGate = barrier();
    const next = backend();
    next.embedBatch.mockImplementation(async () => { forceGate.enter(); await forceGate.resumed; throw new Error("synthetic rebuild failure"); });
    next.embed.mockRejectedValue(new Error("synthetic rebuild failure"));
    store.llm = next as any;
    const rebuild = generateEmbeddings(store, { force: true });
    try {
      await forceGate.entered;
      queryGate.release();
      await refusal;
    } finally {
      queryGate.release(); forceGate.release();
      expect((await rebuild).status).toBe("failed");
      await query.catch(() => {});
    }
  });

  it("validates batch identity once, not once per query/collection lookup", async () => {
    const { store, own } = await fixture();
    const result = await structuredSearch(store, [
      { type: "vec", query: "identityneedle" }, { type: "hyde", query: "transaction" }, { type: "vec", query: "public" },
    ], { collections: ["docs", "empty"], skipRerank: true });
    expect(result).toHaveLength(1);
    expect(own.embedBatch).toHaveBeenCalledTimes(1);
    expect(own.getDescriptor).toHaveBeenCalledTimes(2);
    expect(own.embedBatch.mock.calls[0]?.[1]).toMatchObject({ expectedDescriptor: descriptor() });
  });

  it("does not let the internal precomputed batch route outlive a coherent replacement identity", async () => {
    const { store } = await fixture();
    const replacement = descriptor({ revision: "revision-B" });
    await expect(structuredSearch(store, [{ type: "vec", query: "identityneedle" }], {
      skipRerank: true,
      hooks: { onEmbedDone: () => {
        store.db.prepare("UPDATE store_config SET value = ? WHERE key = 'embedding_space_id'").run(computeEmbeddingSpaceId(replacement));
        store.db.prepare("UPDATE store_config SET value = ? WHERE key = 'embedding_descriptor'").run(JSON.stringify(replacement));
      } },
    })).rejects.toThrow(/embedding|changed/i);
    expect(store.db.prepare("SELECT value FROM store_config WHERE key = 'embedding_space_id'").get())
      .toEqual({ value: computeEmbeddingSpaceId(replacement) });
  });

  it("does not reuse batch vectors across a same-identity DDL replacement", async () => {
    const { store, own } = await fixture();
    await expect(structuredSearch(store, [{ type: "vec", query: "identityneedle" }], {
      skipRerank: true,
      hooks: { onEmbedDone: () => {
        store.clearAllEmbeddings();
        store.ensureVecTable(4, own.state.descriptor);
        store.insertEmbedding("public-hash", 0, 0, new Float32Array(vector), own.embedModelName, timestamp);
      } },
    })).rejects.toThrow(/embedding|changed/i);
  });

  it.each(["vector", "structured"] as const)("preserves %s cancellation and the native-like lease with valid index identity", async route => {
    const { sdk, own } = await fixture();
    const gate = barrier();
    own.embed.mockImplementation(async () => { gate.enter(); await gate.resumed; return { embedding: [...vector], model: own.embedModelName }; });
    own.embedBatch.mockImplementation(async texts => { gate.enter(); await gate.resumed; return texts.map(() => ({ embedding: [...vector], model: own.embedModelName })); });
    const controller = new AbortController();
    const query = route === "vector" ? sdk.searchVector("identityneedle", { signal: controller.signal })
      : sdk.search({ queries: [{ type: "vec", query: "identityneedle" }], rerank: false, signal: controller.signal });
    const rejection = expect(query).rejects.toThrow(/cancel|released|abort/i);
    try {
      await gate.entered;
      controller.abort(new Error("cancel identity-verified query"));
      await rejection;
      expect(canUnloadLLM(own as any)).toBe(false);
    } finally { gate.release(); }
    await vi.waitFor(() => expect(canUnloadLLM(own as any)).toBe(true));
  });

  it("does not classify a declared MLX backend without descriptors as legacy GGUF", async () => {
    const { sdk, store } = await fixture();
    store.db.exec("DELETE FROM store_config WHERE key IN ('embedding_space_id', 'embedding_descriptor', 'embedding_model')");
    const mlx = { embedBackend: "mlx", embedModelName: "public-fixture-encoder",
      embed: vi.fn(async () => ({ embedding: [...vector], model: "public-fixture-encoder" })) };
    store.llm = mlx as any;
    const before = state(store);
    await expect(sdk.searchVector("identityneedle")).rejects.toThrow(/MLX|descriptor|embedding/i);
    expect(mlx.embed).not.toHaveBeenCalled();
    expect(state(store)).toEqual(before);
  });

  it.each(["tampered descriptor", "schema dimensions"])("validates stored %s before low-level raw vector queries too", async scenario => {
    const { store, own } = await fixture();
    const changed = descriptor(scenario === "tampered descriptor" ? { maxTokens: 512 } : { nativeDimensions: 8, outputDimensions: 8 });
    store.db.prepare("UPDATE store_config SET value = ? WHERE key = 'embedding_descriptor'").run(JSON.stringify(changed));
    if (scenario === "schema dimensions") store.db.prepare("UPDATE store_config SET value = ? WHERE key = 'embedding_space_id'").run(computeEmbeddingSpaceId(changed));
    const before = state(store);
    await expect(store.searchVec("unused", "caller-owned", 1, undefined, undefined, vector)).rejects.toThrow(/embedding|descriptor|dimension/i);
    expect(own.getDescriptor).not.toHaveBeenCalled();
    expect(state(store)).toEqual(before);
  });

  it("keeps explicit raw vectors caller-owned rather than claiming serving inference provenance", async () => {
    const { store } = await fixture();
    const changed = backend(descriptor({ revision: "revision-B" }));
    store.llm = changed as any;
    const before = state(store);
    expect(await store.searchVec("unused", "caller-owned", 1, undefined, undefined, vector)).toHaveLength(1);
    expect(changed.getDescriptor).not.toHaveBeenCalled();
    expect(changed.embed).not.toHaveBeenCalled();
    expect(state(store)).toEqual(before);
  });

  it("preserves descriptor-less legacy GGUF and caller-owned raw-vector compatibility", async () => {
    const { sdk, store } = await fixture();
    store.db.exec("DELETE FROM store_config WHERE key IN ('embedding_space_id', 'embedding_descriptor', 'embedding_model')");
    const gguf = { embedBackend: "gguf", embedModelName: "legacy-gguf", getDescriptor: async () => null,
      embed: vi.fn(async () => ({ embedding: [...vector], model: "legacy-gguf" })) };
    store.llm = gguf as any;
    const before = state(store);
    expect(await sdk.searchVector("identityneedle")).toHaveLength(1);
    expect(await store.searchVec("unused", "caller-owned", 1, undefined, undefined, vector)).toHaveLength(1);
    expect(gguf.embed).toHaveBeenCalledTimes(1);
    expect(state(store)).toEqual(before);
  });
});
