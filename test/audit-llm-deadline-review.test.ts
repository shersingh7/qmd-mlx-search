import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { getEventListeners } from "node:events";
import type { EmbeddingDescriptor } from "../src/embedding/contract.js";

vi.mock("node-llama-cpp", () => ({
  getLlama: vi.fn(() => { throw new Error("Native initialization is forbidden in deadline tests"); }),
  resolveModelFile: vi.fn(),
  LlamaLogLevel: { error: "error" },
  LlamaChatSession: class {},
}));

// Exercise the real public owner and HTTP client, not a mocked MlxEmbedClient.
import { LlamaCpp, canUnloadLLM } from "../src/llm.js";
import "../src/mlx.js";
import "../src/expansion/protocol.js";

const url = "http://127.0.0.1:1";
const embedding: EmbeddingDescriptor = {
  version: 1, backend: "mlx", model: "fixture/encoder", revision: "embed-a", pooling: "mean",
  nativeDimensions: 4, outputDimensions: 4, maxTokens: 128, normalized: true,
};
const ranker = { version: 1, backend: "mlx", model: "fixture/ranker", revision: "rank-a", maxTokens: 128, yesTokenId: 1, noTokenId: 2 };
const generator = { version: 1, backend: "mlx", kind: "generate", model: "fixture/generator", revision: "gen-a", maxContext: 4096 };
const contract = { ...embedding, rerank: ranker, generate: generator };
const owners: LlamaCpp[] = [];
type Exchange = { path: string; signal: AbortSignal; started: number };

function owner(config: ConstructorParameters<typeof LlamaCpp>[0] = {}): LlamaCpp {
  const llm = new LlamaCpp({ strictMlx: true, mlxUrl: url, mlxTimeoutMs: 80, mlxBinary: false, inactivityTimeoutMs: 0, ...config });
  owners.push(llm);
  return llm;
}

function delay<T>(ms: number, value: () => T, signal?: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const onAbort = () => { clearTimeout(timer); signal?.removeEventListener("abort", onAbort); reject(signal?.reason); };
    const timer = setTimeout(() => { signal?.removeEventListener("abort", onAbort); resolve(value()); }, ms);
    if (signal?.aborted) onAbort();
    else signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function response(path: string, init: RequestInit): Response {
  if (path === "/descriptor") return Response.json(contract);
  if (path === "/health") return Response.json({ ready: true, status: "ok", descriptor: contract });
  if (path === "/embed") {
    const payload = JSON.parse(init.body as string);
    const descriptor = { ...embedding, outputDimensions: payload.dims ?? embedding.outputDimensions };
    return Response.json({ model: descriptor.model, descriptor, dims: descriptor.outputDimensions,
      embeddings: payload.texts.map(() => Array(descriptor.outputDimensions).fill(0.5)) });
  }
  if (path === "/tokenize") return Response.json({ counts: [2] });
  if (path === "/rerank") return Response.json({ scores: [0.7], model: ranker.model, revision: ranker.revision, descriptor: ranker });
  if (path === "/generate") return Response.json({ text: "vec: fixture query", model: generator.model, revision: generator.revision, descriptor: generator });
  throw new Error(`Unmocked fixture HTTP path: ${path}`);
}

function transport(schedule: (path: string, occurrence: number) => number = () => 60,
  answer: (path: string, init: RequestInit) => Response = response): Exchange[] {
  const exchanges: Exchange[] = [];
  const occurrences = new Map<string, number>();
  vi.stubGlobal("fetch", vi.fn((input: string | URL | Request, init: RequestInit = {}) => {
    const path = new URL(String(input)).pathname;
    const occurrence = (occurrences.get(path) ?? 0) + 1;
    occurrences.set(path, occurrence);
    exchanges.push({ path, signal: init.signal!, started: performance.now() });
    return delay(schedule(path, occurrence), () => answer(path, init), init.signal!);
  }));
  return exchanges;
}

function outcome<T>(work: Promise<T>) {
  return work.then(value => ({ ok: true as const, value, at: performance.now() }),
    error => ({ ok: false as const, error, at: performance.now() }));
}

async function expectDeadline(work: Promise<unknown>, start = performance.now()): Promise<void> {
  const settled = outcome(work);
  let done = false;
  void settled.then(() => { done = true; });
  await vi.advanceTimersByTimeAsync(79);
  expect(done).toBe(false);
  await vi.advanceTimersByTimeAsync(1);
  expect(done).toBe(true); // A renewed per-stage budget must fail here, not hang the test.
  const result = await settled;
  expect(result.ok).toBe(false);
  expect(result.at - start).toBe(80);
  if (!result.ok) expect(result.error).toMatchObject({ name: "TimeoutError", message: expect.stringMatching(/deadline|timed out/i) });
}

beforeEach(() => {
  vi.stubEnv("QMD_STRICT_MLX", "0");
  vi.stubEnv("QMD_MLX_RERANK", "0");
  vi.stubEnv("QMD_MLX_EXPAND", "0");
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance"] });
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(async () => {
  await Promise.all(owners.splice(0).map(llm => llm.dispose()));
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("public LlamaCpp MLX operations have one absolute 80 ms deadline", () => {
  test.each([undefined, 2])("cold embedding includes both negotiations (dimensions=%s)", async embedDims => {
    const exchanges = transport();
    const llm = owner({ embedDims });
    await expectDeadline(llm.embed("fixture"));
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/descriptor", "/descriptor"]);
    expect(exchanges.at(-1)!.signal.aborted).toBe(true);
  });

  test.each([undefined, 2])("cold inference uses only the unspent budget (dimensions=%s)", async embedDims => {
    const exchanges = transport(path => path === "/embed" ? 60 : 20);
    const llm = owner({ embedDims });
    await expectDeadline(llm.embed("fixture"));
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/descriptor", "/descriptor", "/embed"]);
    expect(exchanges.at(-1)).toMatchObject({ started: 40 });
    expect(exchanges.at(-1)!.signal.aborted).toBe(true);
  });

  test.each([
    { embedDims: undefined, count: 2 }, { embedDims: 2, count: 2 },
    { embedDims: undefined, count: 65 }, { embedDims: 2, count: 65 },
  ])("batch initialization and queued inference share the budget ($count texts, dims=$embedDims)", async ({ embedDims, count }) => {
    const exchanges = transport(path => path === "/embed" ? 60 : 20);
    const llm = owner({ embedDims, mlxConcurrency: 2 });
    await expectDeadline(llm.embedBatch(Array.from({ length: count }, (_, i) => `fixture-${i}`)));
    expect(exchanges.filter(exchange => exchange.path === "/descriptor")).toHaveLength(2);
    expect(exchanges.filter(exchange => exchange.path === "/embed")).toHaveLength(count === 65 ? 2 : 1);
    expect(exchanges.filter(exchange => exchange.path === "/embed").every(exchange => exchange.signal.aborted)).toBe(true);
  });

  test("descriptor refresh cannot reset the budget after cold initialization", async () => {
    const exchanges = transport((_path, occurrence) => occurrence === 3 ? 60 : 20);
    await expectDeadline(owner().getDescriptor());
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/descriptor", "/descriptor", "/descriptor"]);
    expect(exchanges.at(-1)).toMatchObject({ started: 40 });
  });

  test.each(["embed", "getDescriptor", "getCacheIdentity"] as const)("%s metadata fallback cannot return null or restart its budget", async method => {
    const exchanges = transport(() => 60, (path, init) => path === "/descriptor" ? new Response(null, { status: 404 }) : response(path, init));
    const llm = owner();
    const work = method === "embed" ? llm.embed("fixture") : method === "getDescriptor" ? llm.getDescriptor() : llm.getCacheIdentity("rerank");
    await expectDeadline(work);
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/descriptor", "/health"]);
    expect(exchanges.at(-1)!.signal.aborted).toBe(true);
  });

  test("tokenizer timeout fails closed rather than becoming optional metadata", async () => {
    const exchanges = transport(() => 120);
    await expectDeadline(owner().mlxTokenize(["fixture query"]));
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/tokenize"]);
    expect(exchanges[0]!.signal.aborted).toBe(true);
  });

  test.each(["rerank", "generate", "expandQueryWithIdentity"] as const)("%s metadata, inference and final identity checks share a deadline", async method => {
    const exchanges = transport(path => path === "/descriptor" ? 20 : 50);
    const llm = owner();
    const work = method === "rerank" ? llm.rerank("fixture", [{ text: "fixture", file: "fixture.md" }])
      : method === "generate" ? llm.generate("fixture") : llm.expandQueryWithIdentity("fixture query");
    await expectDeadline(work);
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/descriptor", method === "rerank" ? "/rerank" : "/generate", "/descriptor"]);
    expect(exchanges.at(-1)).toMatchObject({ started: 70 });
    expect(exchanges.at(-1)!.signal.aborted).toBe(true);
  });

  test("even synchronous work past the boundary cannot publish overdue success", async () => {
    const llm = owner();
    const client = llm as any;
    vi.spyOn(client, "_embedMlx").mockImplementation(async () => {
      vi.advanceTimersByTime(79);
      // Move the monotonic clock without executing the deadline callback.
      vi.spyOn(performance, "now").mockReturnValue(81);
      return { embedding: [0.5, 0.5, 0.5, 0.5], model: embedding.model };
    });
    await expect(llm.embed("fixture")).rejects.toMatchObject({ name: "TimeoutError" });
  });
});

describe("each shared initialization waiter owns its own deadline", () => {
  test("an older caller expires at 80 ms while the later survivor finishes at 100 ms", async () => {
    const exchanges = transport((path, occurrence) => path === "/descriptor" ? occurrence === 1 ? 60 : 30 : 10);
    const llm = owner();
    const first = outcome(llm.embed("older"));
    await vi.advanceTimersByTimeAsync(30);
    const survivor = outcome(llm.embed("survivor"));
    await vi.advanceTimersByTimeAsync(50);
    const expired = await first;
    expect(expired).toMatchObject({ ok: false, at: 80, error: { name: "TimeoutError" } });
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/descriptor", "/descriptor"]);
    expect(exchanges[1]!.signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(20);
    expect(await survivor).toEqual({ ok: true, at: 100, value: { embedding: [0.5, 0.5, 0.5, 0.5], model: embedding.model } });
    expect(exchanges.filter(exchange => exchange.path === "/descriptor")).toHaveLength(2);
    expect(getEventListeners(exchanges[1]!.signal, "abort")).toHaveLength(0);
    const warm = outcome(llm.embed("warm"));
    await vi.advanceTimersByTimeAsync(10);
    expect(await warm).toMatchObject({ ok: true, at: 110, value: { model: embedding.model } });
    expect(exchanges.filter(exchange => exchange.path === "/descriptor")).toHaveLength(2);
    expect(canUnloadLLM(llm)).toBe(true);
  });

  test("only the last expired waiter abandons shared initialization and a later call retries cleanly", async () => {
    const exchanges = transport((path, occurrence) => path === "/descriptor" && occurrence <= 2 ? 60 : 5);
    const llm = owner();
    const first = outcome(llm.embed("older"));
    await vi.advanceTimersByTimeAsync(10);
    const second = outcome(llm.embed("later"));
    await vi.advanceTimersByTimeAsync(70);
    expect(await first).toMatchObject({ ok: false, at: 80, error: { name: "TimeoutError" } });
    expect(exchanges[1]!.signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(10);
    expect(await second).toMatchObject({ ok: false, at: 90, error: { name: "TimeoutError" } });
    expect(exchanges[1]!.signal.aborted).toBe(true);
    expect((llm as any).mlxClient).toBeNull();
    expect((llm as any).mlxClientLoad).toBeNull();
    const retry = outcome(llm.embed("retry"));
    await vi.advanceTimersByTimeAsync(15);
    expect(await retry).toMatchObject({ ok: true, at: 105, value: { model: embedding.model } });
    expect(canUnloadLLM(llm)).toBe(true);
  });

  test("noncooperative late metadata cannot resurrect an abandoned initialization", async () => {
    const exchanges: Exchange[] = [];
    vi.stubGlobal("fetch", vi.fn((input: string, init: RequestInit) => {
      const path = new URL(input).pathname;
      exchanges.push({ path, signal: init.signal!, started: performance.now() });
      return delay(60, () => Response.json(contract)); // Deliberately ignores fetch cancellation.
    }));
    const llm = owner();
    await expectDeadline(llm.embed("abandoned"));
    expect(exchanges[1]!.signal.aborted).toBe(true);
    await vi.advanceTimersByTimeAsync(60);
    expect((llm as any).mlxClient).toBeNull();
    expect((llm as any).mlxDescriptor).toBeNull();
    expect((llm as any).mlxClientLoad).toBeNull();
    expect(exchanges.map(exchange => exchange.path)).toEqual(["/descriptor", "/descriptor"]);
    expect(canUnloadLLM(llm)).toBe(true);
  });
});

describe("the MLX timeout is not a native GGUF disposal deadline", () => {
  test("an expired MLX-to-native fallback retains its native lease until inference settles", async () => {
    vi.stubEnv("QMD_MLX_RERANK", "1");
    transport(() => 10, (path, init) => path === "/rerank" ? new Response(null, { status: 503 }) : response(path, init));
    const llm = owner({ strictMlx: false, embedBackend: "mlx" });
    const native = llm as any;
    native._ciMode = false;
    const context = {
      rankAll: vi.fn(() => delay(120, () => [0.5])),
      dispose: vi.fn(async () => {}),
    };
    native.rerankContexts = [context];
    vi.spyOn(native, "ensureRerankContexts").mockResolvedValue([context]);
    vi.spyOn(native, "ensureRerankModel").mockResolvedValue({ tokenize: () => [1], detokenize: () => "fixture" });
    await expectDeadline(llm.rerank("fixture", [{ text: "fixture", file: "fixture.md" }]));
    expect(context.rankAll).toHaveBeenCalledTimes(1);
    expect(canUnloadLLM(llm)).toBe(false);
    let closed = false;
    const closing = llm.dispose().then(() => { closed = true; });
    await vi.advanceTimersByTimeAsync(59);
    expect(closed).toBe(false);
    expect(context.dispose).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    await closing;
    expect(context.dispose).toHaveBeenCalledTimes(1);
  });

  test("native inference and native cleanup can each outlive the 80 ms MLX setting", async () => {
    const llm = owner({ strictMlx: false, embedBackend: "gguf" });
    const native = llm as any;
    native._ciMode = false;
    const context = {
      getEmbeddingFor: vi.fn(() => delay(120, () => ({ vector: [1, 0] }))),
      dispose: vi.fn(() => delay(120, () => undefined)),
    };
    native.embedContexts = [context];
    vi.spyOn(native, "ensureEmbedContext").mockResolvedValue(context);
    const inference = outcome(llm.embed("fixture"));
    await vi.advanceTimersByTimeAsync(120);
    expect(await inference).toMatchObject({ ok: true, at: 120, value: { embedding: [1, 0] } });
    let closed = false;
    const closing = llm.dispose().then(() => { closed = true; });
    await vi.advanceTimersByTimeAsync(80);
    expect(closed).toBe(false);
    await vi.advanceTimersByTimeAsync(40);
    await closing;
    expect(context.dispose).toHaveBeenCalledTimes(1);
  });
});
