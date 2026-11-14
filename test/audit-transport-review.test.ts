import { afterEach, describe, expect, test, vi } from "vitest";
import { getEventListeners } from "node:events";
import {
  MlxEmbedClient, embedBatchConcurrent, getMlxDescriptor, rerankWithMlx,
} from "../src/mlx.js";
import {
  actualStageIdentity, generateMlxWithIdentity, getMlxStageContract, stageCacheIdentity,
} from "../src/llm/mlx-identity.js";
import { EmbeddingProtocolError, encodeBinaryEmbeddings, validateJsonEmbedResponse } from "../src/embedding/protocol.js";
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
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(publish => { resolve = publish; });
  return { promise, resolve };
}

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

/** Real time, not fake timers: an 80 ms operation must settle before a 100 ms barrier. */
async function beforeBarrier<T>(work: Promise<T>, started = performance.now()): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const crossed = () => new Error("Operation crossed the real 100 ms barrier");
  try {
    const result = await Promise.race([work, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(crossed()), Math.max(0, 100 - (performance.now() - started)));
    })]);
    // A blocked event loop can settle work before an already-overdue timer.
    if (performance.now() - started >= 100) throw crossed();
    return result;
  } finally { clearTimeout(timer); }
}

function embeddingResponse(init: RequestInit): Response {
  const payload = JSON.parse(init.body as string);
  const descriptor = { ...embedding, outputDimensions: payload.dims ?? embedding.outputDimensions };
  return Response.json({ model: descriptor.model, descriptor, dims: descriptor.outputDimensions,
    embeddings: payload.texts.map(() => Array(descriptor.outputDimensions).fill(0.5)) });
}

const methods = ["embed", "embedBatch", "embedBatchConcurrent", "embedBatchBinary"] as const;
function invoke(client: MlxEmbedClient, method: typeof methods[number], signal?: AbortSignal) {
  return method === "embed" ? client.embed("x", { signal }) : client[method](["x"], { signal });
}
function servingResponse(input: string, init: RequestInit): Response {
  return input.endsWith("/embed-bin") ? new Response(encodeBinaryEmbeddings([[0.5, 0.5, 0.5, 0.5]], 4), {
    headers: { "X-QMD-Embedding-Descriptor": encodeURIComponent(JSON.stringify(embedding)) },
  }) : embeddingResponse(init);
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
    const exchanges: Promise<Response>[] = [];
    vi.stubGlobal("fetch", (input: string) => {
      paths.push(input);
      // Intentionally ignores abort, exercising explicit racing.
      const exchange = sleep(60).then(() => input.endsWith("/descriptor") ? new Response(null, { status: 404 })
        : Response.json({ ready: true, status: "ok", descriptor: contract }));
      exchanges.push(exchange);
      return exchange;
    });
    const operation = read({ url, timeoutMs: 80 });
    const settled = Promise.allSettled([operation]);
    try {
      expect(await beforeBarrier(operation)).toBeNull();
      expect(paths.map(path => path.split("/").at(-1))).toEqual(["descriptor", "health"]);
    } finally { await settled; await Promise.allSettled(exchanges); }
  });
  test.each(["embed", "embedBatch", "embedBatchConcurrent", "embedBatchBinary"] as const)("cold %s includes negotiation in its 80 ms budget", async method => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const paths: string[] = [];
    const exchanges: Promise<Response>[] = [];
    vi.stubGlobal("fetch", (input: string, init: RequestInit) => {
      paths.push(input);
      const exchange = (async () => {
        if (input.endsWith("/descriptor")) {
          // Spend real negotiation budget without assuming two independent
          // sleep callbacks will both be scheduled before the caller expires.
          const admitted = performance.now();
          while (performance.now() - admitted < 30) { /* synchronous metadata adapter */ }
          return Response.json(embedding);
        }
        // Keep the original 120 ms combined fixture work. A fresh 80 ms
        // inference budget would now settle beyond the real 100 ms barrier.
        await sleep(90);
        return servingResponse(input, init);
      })();
      exchanges.push(exchange);
      return exchange;
    });
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
    const started = performance.now(); // Fixture preparation is not API work.
    const operation = method === "embed" ? client.embed("x") : client[method](["x"]);
    const settled = Promise.allSettled([operation]);
    try {
      expect(await beforeBarrier<unknown>(operation, started)).toEqual(method === "embed" || method === "embedBatchBinary" ? null : [null]);
      expect(paths.map(path => path.split("/").at(-1))).toEqual(["descriptor", method === "embedBatchBinary" ? "embed-bin" : "embed"]);
    } finally {
      await settled;
      await Promise.allSettled(exchanges);
      client.dispose();
    }
  });
  test("concurrent queueing also uses the API budget, not one new budget per batch", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const exchanges: Promise<Response>[] = [];
    const fetchMock = vi.fn((_input: string, init: RequestInit) => {
      const exchange = sleep(60).then(() => embeddingResponse(init));
      exchanges.push(exchange);
      return exchange;
    });
    vi.stubGlobal("fetch", fetchMock);
    const operation = embedBatchConcurrent(["a", "b", "c"], { url, timeoutMs: 80, concurrency: 1, binary: false }, undefined, 1);
    const settled = Promise.allSettled([operation]);
    try {
      expect(await beforeBarrier(operation))
        .toEqual([{ model: embedding.model, embedding: [0.5, 0.5, 0.5, 0.5] }, null, null]);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally { await settled; await Promise.allSettled(exchanges); }
  });
});

describe("review: negotiation deadline classification", () => {
  test.each(["embed", "embedBatch", "embedBatchConcurrent", "embedBatchBinary"] as const)(
    "%s classifies synchronous metadata overrun before the deadline timer can run", async method => {
      vi.spyOn(console, "error").mockImplementation(() => {});
      const paths: string[] = [];
      const bytes = new TextEncoder().encode(JSON.stringify(embedding));
      vi.stubGlobal("fetch", async (input: string) => {
        paths.push(input);
        return new Response(new ReadableStream({
          pull(controller) {
            // Real monotonic time moves past both 80 ms scopes while timers
            // cannot run. The bytes are valid; no malformed descriptor exists.
            const stopAt = performance.now() + 85;
            while (performance.now() < stopAt) { /* deliberate synchronous adapter stall */ }
            controller.enqueue(bytes);
            controller.close();
          },
        }));
      });
      const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
      const getDescriptor = client.getDescriptor.bind(client);
      let observed: { descriptor: EmbeddingDescriptor | null; callerAborted: boolean } | undefined;
      vi.spyOn(client, "getDescriptor").mockImplementation(async signal => {
        const descriptor = await getDescriptor(signal);
        observed = { descriptor, callerAborted: signal!.aborted };
        return descriptor;
      });
      const started = performance.now();
      const operation = method === "embed" ? client.embed("x") : client[method](["x"]);
      const settled = Promise.allSettled([operation]);
      try {
        expect(await beforeBarrier<unknown>(operation, started)).toEqual(method === "embed" || method === "embedBatchBinary" ? null : [null]);
        expect(observed).toEqual({ descriptor: null, callerAborted: false });
        expect(paths).toEqual([`${url}/descriptor`]);
        expect(client.descriptor).toBeNull();
      } finally {
        await settled; // Preserve a failed assertion without abandoning work.
        console.log(JSON.stringify({ evidence: "synchronous-metadata-overrun", method, observed, paths, published: client.descriptor !== null }));
        client.dispose();
      }
    });

  test("an expired sole caller cannot publish metadata while its timer is delayed", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance"] });
    vi.spyOn(console, "error").mockImplementation(() => {});
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
    const getDescriptor = client.getDescriptor.bind(client);
    vi.spyOn(client, "getDescriptor").mockImplementation(signal => {
      // Spend caller budget before metadata admission. Metadata's own budget
      // still has time at t=85, but the only consumer's t=80 budget is over.
      vi.advanceTimersByTime(30);
      return getDescriptor(signal);
    });
    const paths: string[] = [];
    vi.stubGlobal("fetch", async (input: string) => {
      paths.push(input);
      await Promise.resolve();
      vi.spyOn(performance, "now").mockReturnValue(85); // Deliberately do not dispatch timers.
      return Response.json(embedding);
    });
    const operation = client.embed("expired");
    const settled = Promise.allSettled([operation]);
    try {
      expect(await operation).toBeNull();
      expect(paths).toEqual([`${url}/descriptor`]);
      expect(client.descriptor).toBeNull();
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      await settled;
      console.log(JSON.stringify({ evidence: "sole-caller-delayed-timer", paths, published: client.descriptor !== null, ownedTimers: vi.getTimerCount() }));
      client.dispose();
      vi.useRealTimers();
    }
  });
});

describe("review: explicit shared negotiation admission", () => {
  test("a caller held before admission is not a shared waiter when the old budget expires", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance"] });
    vi.spyOn(console, "error").mockImplementation(() => {});
    const heldCaller = deferred<void>();
    const oldResponse = deferred<Response>();
    const signals: AbortSignal[] = [];
    vi.stubGlobal("fetch", async (input: string, init: RequestInit) => {
      if (input.endsWith("/descriptor")) {
        signals.push(init.signal!);
        return signals.length === 1 ? oldResponse.promise : Response.json(embedding);
      }
      return embeddingResponse(init);
    });
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
    const getDescriptor = client.getDescriptor.bind(client);
    let calls = 0;
    let admissions = 0;
    vi.spyOn(client, "getDescriptor").mockImplementation(async signal => {
      if (++calls === 1) await sleep(30);
      else await heldCaller.promise;
      const pending = getDescriptor(signal); // Admission is synchronous before its first await.
      admissions++;
      return pending;
    });
    const tasks: Promise<unknown>[] = [];
    const old = client.embed("old");
    tasks.push(old);
    void old.catch(() => {});
    try {
      await vi.advanceTimersByTimeAsync(35);
      const survivor = client.embed("not-admitted-yet");
      tasks.push(survivor);
      void survivor.catch(() => {});
      await vi.advanceTimersByTimeAsync(45);
      expect(await old).toBeNull();
      expect(admissions).toBe(1); // The old fixture's sleep did not establish two admissions.
      expect(signals[0]!.aborted).toBe(true); // Correct last-waiter cancellation, not poisoning a survivor.
      console.log(JSON.stringify({ evidence: "held-survivor-not-admitted", admissions, descriptorAborted: signals[0]!.aborted, time: performance.now() }));
      heldCaller.resolve();
      expect(await survivor).toMatchObject({ model: embedding.model });
      expect(admissions).toBe(2);
      expect(signals).toHaveLength(2); // A genuinely later waiter gets a fresh negotiation.
      expect(getEventListeners(signals[0]!, "abort")).toHaveLength(0);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      heldCaller.resolve();
      oldResponse.resolve(Response.json(embedding));
      await Promise.allSettled(tasks);
      client.dispose();
      vi.useRealTimers();
    }
  });

  test("one caller's budget expiry does not cancel another shared negotiation waiter", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const firstDescriptorRequest = deferred<void>();
    const response = deferred<Response>();
    const descriptorSignals: AbortSignal[] = [];
    const fetchMock = vi.fn(async (input: string, init: RequestInit) => {
      if (input.endsWith("/descriptor")) {
        descriptorSignals.push(init.signal!);
        firstDescriptorRequest.resolve(); // Actual dispatch, not a possibly rejected method entry.
        return response.promise;
      }
      return embeddingResponse(init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
    const getDescriptor = client.getDescriptor.bind(client);
    let calls = 0;
    let descriptorEntries = 0;
    vi.spyOn(client, "getDescriptor").mockImplementation(async signal => {
      if (++calls === 1) await sleep(30); // Only the older operation spends budget before admission.
      const pending = getDescriptor(signal);
      descriptorEntries++;
      return pending;
    });
    const tasks: Promise<unknown>[] = [];
    const started = performance.now();
    const old = beforeBarrier(client.embed("old"), started);
    tasks.push(old);
    void old.catch(() => {});
    try {
      await beforeBarrier(firstDescriptorRequest.promise, started);
      expect(descriptorSignals).toHaveLength(1);
      expect(descriptorSignals[0]!.aborted).toBe(false);
      const survivor = client.embed("survivor"); // No sleep-based assumption: this registers synchronously.
      tasks.push(survivor);
      void survivor.catch(() => {});
      expect(descriptorEntries).toBe(2);
      expect(await old).toBeNull();
      expect(descriptorSignals[0]!.aborted).toBe(false);
      response.resolve(Response.json(embedding));
      expect(await survivor).toMatchObject({ model: embedding.model });
      expect(fetchMock.mock.calls.filter(([path]) => path.endsWith("/descriptor"))).toHaveLength(1);
      expect(getEventListeners(descriptorSignals[0]!, "abort")).toHaveLength(0);
      console.log(JSON.stringify({ evidence: "registered-survivor", descriptorEntries,
        descriptorRequests: descriptorSignals.length, descriptorAborted: descriptorSignals[0]!.aborted }));
    } finally {
      firstDescriptorRequest.resolve();
      response.resolve(Response.json(embedding));
      await Promise.allSettled(tasks); // Drain even when the intermediate ownership assertion fails.
      client.dispose();
    }
  });
});

describe("review: waiter publication and terminal cancellation", () => {
  test("a live registered survivor may publish while an older caller's timer is delayed", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance"] });
    vi.spyOn(console, "error").mockImplementation(() => {});
    const response = deferred<Response>();
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
    const getDescriptor = client.getDescriptor.bind(client);
    let admissions = 0;
    vi.spyOn(client, "getDescriptor").mockImplementation(signal => {
      if (admissions === 0) vi.advanceTimersByTime(30);
      const pending = getDescriptor(signal);
      admissions++;
      return pending;
    });
    const signals: AbortSignal[] = [];
    vi.stubGlobal("fetch", async (input: string, init: RequestInit) => {
      signals.push(init.signal!);
      return input.endsWith("/descriptor") ? response.promise : embeddingResponse(init);
    });
    const old = client.embed("old");
    const tasks: Promise<unknown>[] = [old];
    void old.catch(() => {});
    try {
      await vi.advanceTimersByTimeAsync(5);
      const survivor = client.embed("survivor");
      tasks.push(survivor);
      void survivor.catch(() => {});
      expect(admissions).toBe(2);
      vi.spyOn(performance, "now").mockReturnValue(85);
      response.resolve(Response.json(embedding));
      expect(await old).toBeNull();
      expect(await survivor).toMatchObject({ model: embedding.model });
      expect(client.descriptor).toEqual(embedding);
      expect(signals).toHaveLength(2); // One shared descriptor request and only the survivor's inference.
      expect(signals.every(signal => !signal.aborted)).toBe(true);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      response.resolve(Response.json(embedding));
      await Promise.allSettled(tasks);
      client.dispose();
      vi.useRealTimers();
    }
  });

  test("only the last cancelled waiter aborts shared work; late metadata is discarded and retry is clean", async () => {
    const response = deferred<Response>();
    const signals: AbortSignal[] = [];
    const fetchMock = vi.fn(async (_input: string, init: RequestInit) => {
      signals.push(init.signal!);
      return signals.length === 1 ? response.promise : Response.json(embedding);
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new MlxEmbedClient({ url, timeoutMs: 80 });
    const firstCaller = new AbortController();
    const lastCaller = new AbortController();
    const first = client.getDescriptor(firstCaller.signal);
    const last = client.getDescriptor(lastCaller.signal);
    const settled = Promise.allSettled([first, last]);
    let cancelledBody = false;
    try {
      firstCaller.abort();
      await expect(first).rejects.toMatchObject({ name: "AbortError" });
      expect(signals[0]!.aborted).toBe(false);
      lastCaller.abort();
      await expect(last).rejects.toMatchObject({ name: "AbortError" });
      expect(signals[0]!.aborted).toBe(true);
      response.resolve(new Response(new ReadableStream({ cancel() { cancelledBody = true; } })));
      await new Promise<void>(resolve => setImmediate(resolve));
      expect(cancelledBody).toBe(true);
      expect(client.descriptor).toBeNull();
      for (const signal of [firstCaller.signal, lastCaller.signal, signals[0]!]) expect(getEventListeners(signal, "abort")).toHaveLength(0);
      expect(await client.probeDims()).toBe(4);
      expect(fetchMock).toHaveBeenCalledTimes(2);
    } finally {
      response.resolve(Response.json(embedding));
      await settled;
      client.dispose();
    }
  });

  describe.each(methods)("%s preserves public failure shapes", method => {
    test("valid metadata and serving identity still succeed", async () => {
      const fetchMock = vi.fn(async (input: string, init: RequestInit) => input.endsWith("/descriptor") ? Response.json(embedding) : servingResponse(input, init));
      vi.stubGlobal("fetch", fetchMock);
      const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
      try {
        const value = { model: embedding.model, embedding: [0.5, 0.5, 0.5, 0.5] };
        expect(await invoke(client, method)).toEqual(method === "embed" ? value : method === "embedBatchBinary" ? [value.embedding] : [value]);
        expect(client.descriptor).toEqual(embedding);
        expect(fetchMock).toHaveBeenCalledTimes(2);
      } finally { client.dispose(); }
    });

    test("a genuinely malformed descriptor remains a protocol rejection", async () => {
      const fetchMock = vi.fn(async () => Response.json({ ...embedding, normalized: "true" }));
      vi.stubGlobal("fetch", fetchMock);
      const client = new MlxEmbedClient({ url, timeoutMs: 80 });
      try {
        await expect(invoke(client, method)).rejects.toBeInstanceOf(EmbeddingProtocolError);
        expect(client.descriptor).toBeNull();
        expect(fetchMock).toHaveBeenCalledTimes(1);
      } finally { client.dispose(); }
    });

    test("cancellation before metadata is terminal and dispatches nothing", async () => {
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      const caller = new AbortController();
      const client = new MlxEmbedClient({ url, timeoutMs: 80 });
      caller.abort();
      try {
        await expect(invoke(client, method, caller.signal)).rejects.toMatchObject({ name: "AbortError" });
        expect(fetchMock).not.toHaveBeenCalled();
        expect(client.descriptor).toBeNull();
        expect(getEventListeners(caller.signal, "abort")).toHaveLength(0);
      } finally { client.dispose(); }
    });

    test.each(["metadata", "inference"] as const)("caller cancellation during %s rejects, drains, and never dispatches late", async phase => {
      const reached = deferred<void>();
      const response = deferred<Response>();
      const signals: AbortSignal[] = [];
      let lateResponse!: () => Response;
      const fetchMock = vi.fn(async (input: string, init: RequestInit) => {
        signals.push(init.signal!);
        if ((phase === "metadata") === input.endsWith("/descriptor")) {
          lateResponse = () => input.endsWith("/descriptor") ? Response.json(embedding) : servingResponse(input, init);
          reached.resolve();
          return response.promise;
        }
        return Response.json(embedding);
      });
      vi.stubGlobal("fetch", fetchMock);
      const caller = new AbortController();
      const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false });
      const operation = invoke(client, method, caller.signal);
      const settled = Promise.allSettled([operation]);
      try {
        await beforeBarrier(reached.promise); // The chosen real HTTP phase must actually be admitted.
        caller.abort();
        await expect(operation).rejects.toMatchObject({ name: "AbortError" });
        response.resolve(lateResponse());
        await new Promise<void>(resolve => setImmediate(resolve));
        expect(fetchMock).toHaveBeenCalledTimes(phase === "metadata" ? 1 : 2);
        expect(signals.at(-1)!.aborted).toBe(true);
        expect(client.descriptor).toEqual(phase === "metadata" ? null : embedding);
        expect(getEventListeners(caller.signal, "abort")).toHaveLength(0);
        for (const signal of signals) expect(getEventListeners(signal, "abort")).toHaveLength(0);
      } finally {
        response.resolve(lateResponse?.() ?? Response.json(embedding));
        await settled;
        client.dispose();
      }
    });
  });

  test.each(["metadata", "inference"] as const)("disposal during %s is terminal and cannot publish late state", async phase => {
    const reached = deferred<void>();
    const response = deferred<Response>();
    const fetchMock = vi.fn(async (input: string, init: RequestInit) => {
      if ((phase === "metadata") === input.endsWith("/descriptor")) { reached.resolve(); return response.promise; }
      return input.endsWith("/descriptor") ? Response.json(embedding) : embeddingResponse(init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const external = new AbortController();
    const client = new MlxEmbedClient({ url, timeoutMs: 80, binary: false, signal: external.signal });
    const operation = client.embed("x");
    const settled = Promise.allSettled([operation]);
    try {
      await beforeBarrier(reached.promise);
      client.dispose(); // Contract action, not fixture teardown.
      await expect(operation).rejects.toMatchObject({ name: "AbortError" });
      const beforeLate = client.descriptor;
      response.resolve(Response.json(embedding));
      await new Promise<void>(resolve => setImmediate(resolve));
      expect(client.descriptor).toBe(beforeLate);
      expect(client.descriptor).toEqual(phase === "metadata" ? null : embedding);
      await expect(client.getDescriptor()).rejects.toMatchObject({ name: "AbortError" });
      await expect(client.embed("again")).rejects.toMatchObject({ name: "AbortError" });
      expect(fetchMock).toHaveBeenCalledTimes(phase === "metadata" ? 1 : 2);
      expect(getEventListeners(external.signal, "abort")).toHaveLength(0);
    } finally {
      response.resolve(Response.json(embedding));
      await settled;
      client.dispose();
    }
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
