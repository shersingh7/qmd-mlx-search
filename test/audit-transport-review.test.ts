import { afterEach, describe, expect, test, vi } from "vitest";
import { getEventListeners } from "node:events";
import {
  MlxEmbedClient, embedBatchConcurrent, getMlxDescriptor, rerankWithMlx,
} from "../src/mlx.js";
import {
  actualStageIdentity, generateMlxWithIdentity, getMlxStageContract, stageCacheIdentity,
} from "../src/llm/mlx-identity.js";
import { validateJsonEmbedResponse } from "../src/embedding/protocol.js";
import type { EmbeddingDescriptor } from "../src/embedding/contract.js";

const url = "http://127.0.0.1:1"; // All HTTP is explicitly stubbed, never a live daemon.
const embedding: EmbeddingDescriptor = {
  version: 1, backend: "mlx", model: "fixture/encoder", revision: "embed-a", pooling: "mean",
  nativeDimensions: 4, outputDimensions: 4, maxTokens: 128, normalized: true,
};
const ranker = { version: 1, backend: "mlx", model: "fixture/ranker", revision: "rank-a", maxTokens: 128, yesTokenId: 1, noTokenId: 2 };
const generator = { version: 1, backend: "mlx", kind: "generate", model: "fixture/generator", revision: "gen-a", maxContext: 4096 };
const contract = { ...embedding, rerank: ranker, generate: generator };
const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

/** Real time, not fake timers: an 80 ms operation must settle before a 100 ms barrier. */
async function beforeBarrier<T>(work: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([work, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("Operation crossed the real 100 ms barrier")), 100);
    })]);
  } finally { clearTimeout(timer); }
}

function embeddingResponse(init: RequestInit): Response {
  const payload = JSON.parse(init.body as string);
  const descriptor = { ...embedding, outputDimensions: payload.dims ?? embedding.outputDimensions };
  return Response.json({ model: descriptor.model, descriptor, dims: descriptor.outputDimensions,
    embeddings: payload.texts.map(() => Array(descriptor.outputDimensions).fill(0.5)) });
}

describe("review: serving metadata is result-local", () => {
  test("rerank transports revision and the complete actual descriptor, not the advertised contract", async () => {
    const serving = { ...ranker, revision: "rank-b", quantization: "8bit" };
    vi.stubGlobal("fetch", async () => Response.json({ scores: [0.7], model: ranker.model, revision: serving.revision, descriptor: serving }));
    const result = await rerankWithMlx("q", ["d"], { url });
    expect(result).toEqual({ scores: [0.7], model: ranker.model, revision: serving.revision, descriptor: serving });
    expect(() => actualStageIdentity("rerank", result!, contract, contract)).toThrow(/serving revision/);
  });
  test.each(["rerank", "expand"] as const)("%s refuses descriptor drift even when model/revision match", async op => {
    const stage = op === "rerank" ? ranker : generator;
    const serving = { ...stage, quantization: "different-weights" };
    vi.stubGlobal("fetch", async () => Response.json(op === "rerank"
      ? { scores: [0.7], model: stage.model, revision: stage.revision, descriptor: serving }
      : { text: "vec: fixture query", model: stage.model, revision: stage.revision, descriptor: serving }));
    const result = op === "rerank" ? await rerankWithMlx("q", ["d"], { url })
      : await generateMlxWithIdentity("q", { url });
    expect(result?.descriptor).toEqual(serving);
    expect(() => actualStageIdentity(op, result!, contract, contract)).toThrow(/serving descriptor/);
  });
  test.each([null, [], "not-a-descriptor", { revision: "missing-model" }, { ...ranker, model: "wrong-model" }, { ...ranker, revision: 7 }])(
    "rerank refuses malformed or conflicting serving descriptor %j", async descriptor => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      vi.stubGlobal("fetch", async () => Response.json({ scores: [0.7], model: ranker.model, descriptor }));
      expect(await rerankWithMlx("q", ["d"], { url })).toBeNull();
    });
  test("generation rejects malformed serving metadata rather than publishing an old identity", async () => {
    vi.stubGlobal("fetch", async () => Response.json({ text: "vec: fixture", model: generator.model, descriptor: [] }));
    await expect(generateMlxWithIdentity("q", { url })).rejects.toThrow(/descriptor/i);
    expect(() => actualStageIdentity("expand", { model: generator.model, descriptor: null }, contract, contract)).toThrow(/descriptor/i);
  });
  test("complete consistent serving metadata retains its exact predicted identity", async () => {
    vi.stubGlobal("fetch", async () => Response.json({ scores: [0.7], model: ranker.model, revision: ranker.revision, descriptor: ranker }));
    const result = await rerankWithMlx("q", ["d"], { url });
    expect(actualStageIdentity("rerank", result!, contract, contract)).toBe(stageCacheIdentity(contract, "rerank"));
    expect(actualStageIdentity("expand", { model: generator.model, revision: generator.revision, descriptor: { generate: generator } }, contract, contract))
      .toBe(stageCacheIdentity(contract, "expand"));
  });
  test("the real LlamaCpp rerank path cannot attach advertised revision A to result B", async () => {
    const { LlamaCpp } = await import("../src/llm.js");
    vi.stubGlobal("fetch", async (input: string) => input.endsWith("/descriptor") ? Response.json(contract)
      : Response.json({ scores: [0.7], model: ranker.model, revision: "rank-b", descriptor: { ...ranker, revision: "rank-b" } }));
    const llm = new LlamaCpp({ strictMlx: true, mlxUrl: url, inactivityTimeoutMs: 0 });
    try {
      expect(await llm.getCacheIdentity("rerank")).toBe(stageCacheIdentity(contract, "rerank"));
      await expect(llm.rerank("q", [{ text: "d", file: "fixture.md" }])).rejects.toThrow(/serving revision/);
    } finally { await llm.dispose(); }
  });
});

describe("review: stage-only daemon descriptors", () => {
  test.each([
    { raw: ranker, expected: { rerank: ranker } },
    { raw: generator, expected: { generate: generator } },
    { raw: { ...ranker, generate: generator }, expected: { rerank: ranker, generate: generator } },
    { raw: contract, expected: { rerank: ranker, generate: generator } },
  ])("normalizes top-level and nested stage contracts ($raw.model)", async ({ raw, expected }) => {
    vi.stubGlobal("fetch", async () => Response.json(raw));
    expect(await getMlxStageContract({ url })).toEqual(expected);
  });
  test("health fallback normalizes stage-only descriptors as well", async () => {
    vi.stubGlobal("fetch", async (input: string) => input.endsWith("/descriptor") ? new Response(null, { status: 404 })
      : Response.json({ ready: true, status: "ok", descriptor: { ...ranker, generate: generator } }));
    expect(await getMlxStageContract({ url })).toEqual({ rerank: ranker, generate: generator });
  });
  test("a pure embedding contract does not invent stage identities", async () => {
    vi.stubGlobal("fetch", async () => Response.json(embedding));
    expect(await getMlxStageContract({ url })).toEqual({});
  });
  test.each(["rerank", "expand"] as const)("the real LlamaCpp path caches only its actual stage-only %s identity", async op => {
    const { LlamaCpp } = await import("../src/llm.js");
    const stage = op === "rerank" ? ranker : generator;
    vi.stubGlobal("fetch", async (input: string) => input.endsWith("/descriptor") ? Response.json(stage)
      : Response.json(op === "rerank" ? { scores: [0.7], model: stage.model, revision: stage.revision, descriptor: stage }
        : { text: "vec: fixture query", model: stage.model, revision: stage.revision, descriptor: stage }));
    const llm = new LlamaCpp({ strictMlx: true, mlxUrl: url, inactivityTimeoutMs: 0 });
    try {
      const predicted = await llm.getCacheIdentity(op);
      expect(predicted).toBe(stageCacheIdentity(contract, op));
      const actual = op === "rerank" ? await llm.rerank("q", [{ text: "d", file: "fixture.md" }]) : await llm.generate("q");
      expect(actual?.model).toBe(predicted);
    } finally { await llm.dispose(); }
  });
});

describe("review: absolute API deadlines across exchanges", () => {
  test.each([getMlxStageContract, getMlxDescriptor])("metadata fallback shares one 80 ms budget (%s)", async read => {
    const paths: string[] = [];
    vi.stubGlobal("fetch", async (input: string) => {
      paths.push(input);
      await sleep(60); // Intentionally ignores abort, exercising explicit racing.
      return input.endsWith("/descriptor") ? new Response(null, { status: 404 })
        : Response.json({ ready: true, status: "ok", descriptor: contract });
    });
    try {
      expect(await beforeBarrier(read({ url, timeoutMs: 80 }))).toBeNull();
      expect(paths.map(path => path.split("/").at(-1))).toEqual(["descriptor", "health"]);
    } finally { await sleep(65); }
  });
  test.each(["embed", "embedBatch", "embedBatchConcurrent", "embedBatchBinary"] as const)("cold %s includes negotiation in its 80 ms budget", async method => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const paths: string[] = [];
    vi.stubGlobal("fetch", async (input: string, init: RequestInit) => {
      paths.push(input);
      await sleep(60);
      return input.endsWith("/descriptor") ? Response.json(embedding) : embeddingResponse(init);
    });
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
    try {
      const operation = method === "embed" ? client.embed("x") : client[method](["x"]);
      expect(await beforeBarrier<unknown>(operation)).toEqual(method === "embed" || method === "embedBatchBinary" ? null : [null]);
      expect(paths.map(path => path.split("/").at(-1))).toHaveLength(2);
    } finally { client.dispose(); await sleep(65); }
  });
  test("concurrent queueing also uses the API budget, not one new budget per batch", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const fetchMock = vi.fn(async (_input: string, init: RequestInit) => { await sleep(60); return embeddingResponse(init); });
    vi.stubGlobal("fetch", fetchMock);
    try {
      expect(await beforeBarrier(embedBatchConcurrent(["a", "b", "c"], { url, timeoutMs: 80, concurrency: 1, binary: false }, undefined, 1)))
        .toEqual([{ model: embedding.model, embedding: [0.5, 0.5, 0.5, 0.5] }, null, null]);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally { await sleep(65); }
  });
  test("one caller's budget expiry does not cancel another shared negotiation waiter", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const descriptorSignal: AbortSignal[] = [];
    let release!: (response: Response) => void;
    const fetchMock = vi.fn(async (input: string, init: RequestInit) => {
      if (input.endsWith("/descriptor")) {
        descriptorSignal.push(init.signal!);
        return new Promise<Response>(resolve => { release = resolve; });
      }
      return embeddingResponse(init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
    const getDescriptor = client.getDescriptor.bind(client);
    // Model a scheduled negotiation start, so the first operation has already
    // spent some of its own budget when the shared metadata exchange begins.
    vi.spyOn(client, "getDescriptor").mockImplementation(async signal => { await sleep(30); return getDescriptor(signal); });
    try {
      const old = beforeBarrier(client.embed("old"));
      await sleep(35);
      const survivor = client.embed("survivor");
      expect(await old).toBeNull();
      expect(descriptorSignal[0]!.aborted).toBe(false);
      release(Response.json(embedding));
      expect(await survivor).toMatchObject({ model: embedding.model });
      expect(fetchMock.mock.calls.filter(([path]) => path.endsWith("/descriptor"))).toHaveLength(1);
      expect(getEventListeners(descriptorSignal[0]!, "abort")).toHaveLength(0);
    } finally { client.dispose(); }
  });
});

describe("review: JSON Float32 safety and validated projections", () => {
  test.each([1e100, -1e100, Number.MAX_VALUE])("rejects finite JSON scalar %s before Float32 overflow", value => {
    expect(() => validateJsonEmbedResponse({ embeddings: [[value]], model: "fixture", dims: 1 })).toThrow(/float32/i);
  });
  test("accepts finite Float32 endpoints and ordinary nonexact decimals", () => {
    const vector = [Math.fround(3.4028234663852886e38), -Math.fround(3.4028234663852886e38), 0.1, Number.MIN_VALUE];
    expect(validateJsonEmbedResponse({ embeddings: [vector], model: "fixture", dims: 4 }).embeddings[0]).toEqual(vector);
    expect([...new Float32Array(vector)].every(Number.isFinite)).toBe(true);
  });
  test.each([false, true])("direct client projects dimensions but validates the requested serving contract (binary=%s)", async binary => {
    const { encodeBinaryEmbeddings } = await import("../src/embedding/protocol.js");
    const fetchMock = vi.fn(async (input: string, init: RequestInit) => {
      if (input.endsWith("/descriptor")) return Response.json(embedding);
      const payload = JSON.parse(init.body as string);
      const actual = { ...embedding, outputDimensions: payload.dims };
      return binary ? new Response(encodeBinaryEmbeddings([[0.5, 0.5]], 2), {
        headers: { "X-QMD-Embedding-Descriptor": encodeURIComponent(JSON.stringify(actual)) },
      }) : embeddingResponse(init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new MlxEmbedClient({ url, binary });
    try {
      expect(await client.embed("x", { dims: 2 })).toMatchObject({ model: embedding.model, embedding: [0.5, 0.5] });
      expect(client.dims).toBe(4); // Per-call projections never mutate the negotiated native contract.
      expect(client.descriptor?.outputDimensions).toBe(4);
      await expect(client.embed("x", { dims: 5 })).rejects.toThrow(/dimensions/i);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally { client.dispose(); }
  });
  test("a reduced projection still rejects revision and descriptor mismatches", async () => {
    vi.stubGlobal("fetch", async (input: string, init: RequestInit) => input.endsWith("/descriptor") ? Response.json(embedding)
      : Response.json({ embeddings: [[0.5, 0.5]], model: embedding.model, dims: 2,
        descriptor: { ...embedding, outputDimensions: 2, revision: "changed" } }));
    const client = new MlxEmbedClient({ url, binary: false });
    try { await expect(client.embed("x", { dims: 2 })).rejects.toThrow(/mismatch/i); }
    finally { client.dispose(); }
  });
});
