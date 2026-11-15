import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const mlx = vi.hoisted(() => ({
  config: null as any,
  descriptor: null as any,
  probe: vi.fn(),
  embed: vi.fn(),
  embedBatch: vi.fn(),
  dispose: vi.fn(),
  getDescriptor: vi.fn(),
  rerank: vi.fn(),
  tokenize: vi.fn(),
  prompt: vi.fn(),
}));

vi.mock("../src/mlx.js", () => ({
  MlxEmbedClient: class {
    dims = 2;
    modelName = "fixture-embed";
    get descriptor() { return mlx.descriptor; }
    constructor(config: any) { mlx.config = config; }
    config() { return mlx.config; }
    probeDims = mlx.probe;
    embed = mlx.embed;
    embedBatch = mlx.embedBatch;
    embedBatchConcurrent = mlx.embedBatch;
    dispose = mlx.dispose;
  },
  embedWithMlx: (_text: string, _config: any, options: any) => mlx.embed(_text, options),
  embedBatchWithMlx: (_texts: string[], _config: any, options: any) => mlx.embedBatch(_texts, options),
  embedBatchConcurrent: (_texts: string[], _config: any, options: any) => mlx.embedBatch(_texts, options),
  getMlxDescriptor: mlx.getDescriptor,
  rerankWithMlx: mlx.rerank,
  tokenizeWithMlx: mlx.tokenize,
}));

vi.mock("node-llama-cpp", () => ({
  getLlama: vi.fn(() => { throw new Error("Real native initialization is forbidden in lifecycle tests"); }),
  resolveModelFile: vi.fn(),
  LlamaLogLevel: { error: "error" },
  LlamaChatSession: class { prompt = mlx.prompt; dispose = vi.fn(); },
}));

import {
  LlamaCpp, SessionReleasedError, canUnloadLLM, withLLMSessionForLlm,
} from "../src/llm.js";
import { getLlama } from 'node-llama-cpp';

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function fakeLlm(config: ConstructorParameters<typeof LlamaCpp>[0] = {}): any {
  const llm = new LlamaCpp({ embedBackend: "gguf", inactivityTimeoutMs: 0, ...config }) as any;
  llm._ciMode = false;
  return llm;
}

const descriptor = {
  version: 1, backend: "mlx", model: "fixture-embed", pooling: "mean",
  nativeDimensions: 2, outputDimensions: 2, maxTokens: 128, normalized: true,
  rerank: { model: "fixture-ranker", revision: "rev-a" },
  generate: { model: "fixture-generator", revision: "rev-a" },
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubEnv("QMD_STRICT_MLX", "0");
  vi.stubEnv("QMD_MLX_RERANK", "0");
  vi.stubEnv("QMD_MLX_EXPAND", "0");
  mlx.descriptor = structuredClone(descriptor);
  mlx.getDescriptor.mockImplementation(async () => structuredClone(mlx.descriptor));
  mlx.probe.mockResolvedValue(2);
  mlx.dispose.mockResolvedValue(undefined);
  mlx.embed.mockResolvedValue({ embedding: [1, 0], model: "fixture-embed" });
  mlx.embedBatch.mockResolvedValue([{ embedding: [1, 0], model: "fixture-embed" }]);
  mlx.rerank.mockResolvedValue({ scores: [0.9], model: "fixture-ranker" });
  mlx.prompt.mockResolvedValue("vec: fixture query");
  vi.stubGlobal('fetch', vi.fn(async (input: string | URL | Request) => {
    if (String(input).endsWith('/descriptor')) return new Response(JSON.stringify(mlx.descriptor));
    throw new Error(`Unmocked fixture HTTP request: ${input}`);
  }));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("per-instance leases and close fencing", () => {
  test("overlapping per-store sessions protect only their own instance", async () => {
    const a = fakeLlm();
    const b = fakeLlm();
    const ctx = { dispose: vi.fn() };
    a.embedContexts = [ctx];
    const outer = deferred();
    const inner = deferred();
    const first = withLLMSessionForLlm(a, async () => { await outer.promise; });
    const second = withLLMSessionForLlm(a, async () => { await inner.promise; });
    expect(canUnloadLLM(a)).toBe(false);
    expect(canUnloadLLM(b)).toBe(true);
    await a.unloadIdleResources();
    expect(ctx.dispose).not.toHaveBeenCalled();
    outer.resolve();
    await first;
    expect(canUnloadLLM(a)).toBe(false);
    inner.resolve();
    await second;
    await a.unloadIdleResources();
    expect(ctx.dispose).toHaveBeenCalledTimes(1);
    await Promise.all([a.dispose(), b.dispose()]);
  });

  test("direct evaluation retains its lease until ignored native cancellation settles", async () => {
    const llm = fakeLlm();
    const gate = deferred<any>();
    const started = deferred();
    const ctx = { dispose: vi.fn(), getEmbeddingFor: vi.fn(() => { started.resolve(); return gate.promise; }) };
    llm.embedContexts = [ctx];
    const abort = new AbortController();
    const pending = llm.embed("fixture", { signal: abort.signal });
    const rejected = expect(pending).rejects.toThrow();
    await started.promise;
    abort.abort(new Error("cancel fixture"));
    await rejected;
    await llm.unloadIdleResources();
    expect(ctx.dispose).not.toHaveBeenCalled();
    gate.resolve({ vector: [1, 0] });
    await new Promise(resolve => setImmediate(resolve));
    await llm.unloadIdleResources();
    expect(ctx.dispose).toHaveBeenCalledTimes(1);
    await llm.dispose();
  });

  test("new operations wait for an idle unload transition", async () => {
    const llm = fakeLlm();
    const disposing = deferred();
    const began = deferred();
    llm.embedContexts = [{ dispose: vi.fn(() => { began.resolve(); return disposing.promise; }) }];
    const idle = llm.unloadIdleResources();
    await began.promise;
    const getEmbeddingFor = vi.fn(async () => ({ vector: [1, 0] }));
    llm.ensureEmbedContext = vi.fn(async () => ({ getEmbeddingFor }));
    const work = llm.embed("fixture");
    await Promise.resolve();
    expect(getEmbeddingFor).not.toHaveBeenCalled();
    disposing.resolve();
    await idle;
    expect(await work).toMatchObject({ embedding: [1, 0] });
    await llm.dispose();
  });

  test('concurrent model families share one native engine initialization', async () => {
    const llm = fakeLlm();
    const gate = deferred<any>();
    const model = () => ({ dispose: vi.fn(async () => {}) });
    const engine = { gpu: 'mock', loadModel: vi.fn(async () => model()), dispose: vi.fn(async () => {}) };
    vi.mocked(getLlama).mockReturnValue(gate.promise);
    llm.resolveModel = vi.fn(async () => '/fixture/model.gguf');
    const embed = llm.ensureEmbedModel();
    const rerank = llm.ensureRerankModel();
    expect(getLlama).toHaveBeenCalledTimes(1);
    gate.resolve(engine);
    await Promise.all([embed, rerank]);
    expect(engine.loadModel).toHaveBeenCalledTimes(2);
    await llm.dispose();
    expect(engine.dispose).toHaveBeenCalledTimes(1);
  });

  test('a late native engine is disposed before publication', async () => {
    const llm = fakeLlm();
    const gate = deferred<any>();
    const engine = { gpu: 'mock', dispose: vi.fn(async () => {}) };
    vi.mocked(getLlama).mockReturnValue(gate.promise);
    const pending = llm.ensureLlama();
    const rejection = expect(pending).rejects.toThrow(/dispos|clos/i);
    const closing = llm.dispose();
    gate.resolve(engine);
    await rejection;
    await closing;
    expect(engine.dispose).toHaveBeenCalledTimes(1);
    expect(llm.llama).toBeNull();
  });

  test("dispose drains a late model load and does not resurrect it", async () => {
    const llm = fakeLlm();
    const load = deferred<any>();
    const started = deferred();
    const model = { dispose: vi.fn(async () => {}) };
    llm.llama = { loadModel: vi.fn(() => { started.resolve(); return load.promise; }), dispose: vi.fn(async () => {}) };
    llm.resolveModel = vi.fn(async () => "/fixture/model.gguf");
    const pending = llm.ensureEmbedModel();
    const rejected = expect(pending).rejects.toThrow(/dispos|clos/i);
    await started.promise;
    let closed = false;
    const closing = llm.dispose().then(() => { closed = true; });
    await Promise.resolve();
    expect(closed).toBe(false);
    load.resolve(model);
    await rejected;
    await closing;
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(llm.embedModel).toBeNull();
    await llm.dispose();
    expect(model.dispose).toHaveBeenCalledTimes(1);
    await expect(llm.embed("no reload")).rejects.toThrow(/dispos|clos/i);
  });

  test("rerank pool is initialized once and published atomically", async () => {
    const llm = fakeLlm();
    const gate = deferred();
    const createRankingContext = vi.fn(async () => {
      await gate.promise;
      return { dispose: vi.fn(async () => {}) };
    });
    llm.ensureRerankModel = vi.fn(async () => ({ createRankingContext }));
    llm.computeParallelism = vi.fn(async () => 4);
    llm.threadsPerContext = vi.fn(async () => 1);
    const calls = Array.from({ length: 5 }, () => llm.ensureRerankContexts());
    await new Promise(resolve => setImmediate(resolve));
    expect(llm.rerankContexts).toHaveLength(0);
    gate.resolve();
    const pools = await Promise.all(calls);
    expect(createRankingContext).toHaveBeenCalledTimes(4);
    expect(pools.every(pool => pool === pools[0])).toBe(true);
    await llm.dispose();
  });

  test("late context creation is cleaned rather than published", async () => {
    const llm = fakeLlm();
    const gate = deferred<any>();
    const started = deferred();
    const ctx = { dispose: vi.fn(async () => {}) };
    llm.ensureEmbedModel = vi.fn(async () => ({ createEmbeddingContext: () => { started.resolve(); return gate.promise; } }));
    llm.computeParallelism = vi.fn(async () => 1);
    llm.threadsPerContext = vi.fn(async () => 1);
    const pending = llm.ensureEmbedContexts();
    const rejected = expect(pending).rejects.toThrow(/dispos|clos/i);
    await started.promise;
    const closing = llm.dispose();
    gate.resolve(ctx);
    await rejected;
    await closing;
    expect(ctx.dispose).toHaveBeenCalledTimes(1);
    expect(llm.embedContexts).toHaveLength(0);
  });

  test("generation context is cleaned if sequence setup throws", async () => {
    const llm = fakeLlm();
    const ctx = { getSequence: () => { throw new Error("sequence failed"); }, dispose: vi.fn(async () => {}) };
    llm.generateModel = { createContext: vi.fn(async () => ctx), dispose: vi.fn(async () => {}) };
    await expect(llm.generate("fixture")).rejects.toThrow("sequence failed");
    expect(ctx.dispose).toHaveBeenCalledTimes(1);
    await llm.dispose();
  });

  test("generation-only weights obey the inactivity policy", async () => {
    vi.useFakeTimers();
    const llm = fakeLlm({ inactivityTimeoutMs: 10, disposeModelsOnInactivity: true });
    const ctx = { getSequence: vi.fn(() => ({})), dispose: vi.fn(async () => {}) };
    const model = { createContext: vi.fn(async () => ctx), dispose: vi.fn(async () => {}) };
    llm.generateModel = model;
    await llm.generate("fixture");
    await vi.advanceTimersByTimeAsync(11);
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(llm.generateModel).toBeNull();
    await llm.dispose();
  });
  test("failed pool creation cleans every partial context", async () => {
    const llm = fakeLlm();
    const ctx = { dispose: vi.fn(async () => {}) };
    const createEmbeddingContext = vi.fn().mockResolvedValueOnce(ctx).mockRejectedValue(new Error('allocation failed'));
    llm.ensureEmbedModel = vi.fn(async () => ({ createEmbeddingContext }));
    llm.computeParallelism = vi.fn(async () => 2);
    llm.threadsPerContext = vi.fn(async () => 1);
    await expect(llm.ensureEmbedContexts()).rejects.toThrow('allocation failed');
    expect(ctx.dispose).toHaveBeenCalledTimes(1);
    expect(llm.embedContexts).toEqual([]);
    await llm.dispose();
  });

  test("disposal is a shared promise and attempts all resources after cleanup errors", async () => {
    const llm = fakeLlm();
    const bad = { dispose: vi.fn(async () => { throw new Error('dispose fixture failed'); }) };
    const good = { dispose: vi.fn(async () => {}) };
    const model = { dispose: vi.fn(async () => {}) };
    llm.embedContexts = [bad, good];
    llm.generateModel = model;
    const first = llm.dispose();
    expect(llm.dispose()).toBe(first);
    await expect(first).rejects.toThrow('LLM disposal failed');
    expect(good.dispose).toHaveBeenCalledTimes(1);
    expect(model.dispose).toHaveBeenCalledTimes(1);
    for (const operation of [
      () => llm.embed('fixture'), () => llm.embedBatch([]), () => llm.generate('fixture'),
      () => llm.expandQuery('fixture'), () => llm.getDescriptor(), () => llm.getCacheIdentity('expand'),
      () => llm.tokenize('fixture'), () => llm.detokenize([]), () => llm.rerank('fixture', []),
    ]) await expect(operation()).rejects.toThrow(/dispos|clos/i);
  });
});

describe("session cancellation", () => {
  test("released sessions remove external listeners", async () => {
    const llm = fakeLlm();
    const abort = new AbortController();
    const add = vi.spyOn(abort.signal, "addEventListener");
    const remove = vi.spyOn(abort.signal, "removeEventListener");
    for (let i = 0; i < 8; i++) {
      await withLLMSessionForLlm(llm, async () => undefined, { signal: abort.signal });
    }
    expect(remove.mock.calls.filter(([event]) => event === "abort")).toHaveLength(8);
    expect(add.mock.calls.filter(([event]) => event === "abort")).toHaveLength(8);
    await llm.dispose();
  });

  test("aborted sessions cannot return success, even when a mocked backend ignores cancellation", async () => {
    const llm = fakeLlm();
    const gate = deferred<any>();
    const abort = new AbortController();
    llm.embed = vi.fn((_text, options) => {
      expect(options.signal).toBeInstanceOf(AbortSignal);
      return gate.promise;
    });
    const pending = withLLMSessionForLlm(llm, async session => {
      const work = session.embed("fixture");
      abort.abort(new Error("cancel session"));
      gate.resolve({ embedding: [1, 0], model: "mock" });
      return await work;
    }, { signal: abort.signal });
    await expect(pending).rejects.toBeInstanceOf(SessionReleasedError);
    await llm.dispose();
  });

  test("a callback that ignores its deadline still cannot succeed", async () => {
    const llm = fakeLlm();
    const abort = new AbortController();
    await expect(withLLMSessionForLlm(llm, async () => {
      abort.abort();
      return "not success";
    }, { signal: abort.signal })).rejects.toBeInstanceOf(SessionReleasedError);
    await llm.dispose();
  });
  test("the last canceled initialization waiter aborts the underlying metadata request", async () => {
    const llm = fakeLlm({ embedBackend: 'mlx' });
    const started = deferred<AbortSignal>();
    mlx.getDescriptor.mockImplementation((config: any) => new Promise((_resolve, reject) => {
      started.resolve(config.signal);
      config.signal.addEventListener('abort', () => reject(config.signal.reason), { once: true });
    }));
    const abort = new AbortController();
    const pending = llm.embed('fixture', { signal: abort.signal });
    const rejection = expect(pending).rejects.toThrow('cancel initialization');
    const requestSignal = await started.promise;
    abort.abort(new Error('cancel initialization'));
    await rejection;
    await new Promise(resolve => setImmediate(resolve));
    expect(requestSignal.aborted).toBe(true);
    await llm.dispose();
  });

  test("one canceled waiter does not cancel shared MLX initialization for another caller", async () => {
    const llm = fakeLlm({ embedBackend: 'mlx' });
    const gate = deferred<any>();
    const started = deferred<AbortSignal>();
    mlx.getDescriptor.mockImplementation((config: any) => { started.resolve(config.signal); return gate.promise; });
    const abort = new AbortController();
    const first = llm.embed('first', { signal: abort.signal });
    const rejection = expect(first).rejects.toThrow('cancel first');
    const second = llm.embed('second');
    const requestSignal = await started.promise;
    abort.abort(new Error('cancel first'));
    await rejection;
    expect(requestSignal.aborted).toBe(false);
    gate.resolve(mlx.descriptor);
    expect(await second).toMatchObject({ embedding: [1, 0] });
    expect(mlx.probe).toHaveBeenCalledTimes(1);
    await llm.dispose();
  });
});

describe("backend policy and effective cache identity", () => {
  test("strictMlx forces all MLX stages without opt-in environment flags", async () => {
    const llm = fakeLlm({ strictMlx: true });
    llm.ensureRerankContexts = vi.fn(() => { throw new Error("native rerank forbidden"); });
    const result = await llm.rerank("fixture", [{ file: "a", text: "fixture" }]);
    expect(result.model).toMatch(/^mlx:fixture-ranker/);
    expect(llm.ensureRerankContexts).not.toHaveBeenCalled();
    expect(llm.mlxExpandEnabled()).toBe(true);
    await llm.dispose();
  });

  test("explicit fallback never substitutes GGUF embeddings for the selected MLX space", async () => {
    const llm = fakeLlm({ embedBackend: "mlx", mlxFallback: true });
    const native = vi.fn(() => { throw new Error("native embed forbidden"); });
    llm.ensureEmbedContext = native;
    expect(await llm.getDescriptor()).toMatchObject({ model: "fixture-embed" });
    mlx.embed.mockResolvedValue(null);
    await expect(llm.embed("fixture")).rejects.toThrow(/MLX|embedding/i);
    expect(native).not.toHaveBeenCalled();
    await llm.dispose();
  });

  test("cold MLX failure also fails closed, regardless of explicit fallback", async () => {
    const llm = fakeLlm({ embedBackend: "mlx", mlxFallback: true });
    mlx.probe.mockRejectedValue(new Error("offline fixture"));
    llm.ensureEmbedContext = vi.fn(() => { throw new Error("native embed forbidden"); });
    await expect(llm.embed("fixture")).rejects.toThrow(/MLX/);
    expect(llm.ensureEmbedContext).not.toHaveBeenCalled();
    expect(mlx.dispose).toHaveBeenCalledTimes(1);
    await llm.dispose();
  });

  test("all resolved transport options reach the client", async () => {
    vi.stubEnv("QMD_MLX_TIMEOUT_MS", "17");
    vi.stubEnv("QMD_MLX_BINARY", "0");
    vi.stubEnv("QMD_EMBED_DIMS", "1");
    const llm = fakeLlm({ embedBackend: "mlx", mlxConcurrency: 3 });
    await llm.getDescriptor();
    await llm.embed("fixture");
    expect(mlx.config).toMatchObject({ timeoutMs: 17, binary: false, concurrency: 3, signal: expect.any(AbortSignal) });
    expect(mlx.embed.mock.calls[0]?.[1]).toMatchObject({ dims: 1, expectedDescriptor: expect.any(Object), signal: expect.any(AbortSignal) });
    await llm.dispose();
  });

  test("cache identities use fresh descriptor revisions and actual result labels", async () => {
    const llm = fakeLlm({ strictMlx: true });
    const a = await llm.getCacheIdentity("rerank");
    expect(a).toMatch(/^mlx:fixture-ranker/);
    const result = await llm.rerank("fixture", [{ file: "a", text: "fixture" }]);
    expect(result.model).toBe(a);
    mlx.descriptor.rerank.revision = "rev-b";
    const b = await llm.getCacheIdentity("rerank");
    expect(b).not.toBe(a);
    mlx.descriptor = null;
    expect(await llm.getCacheIdentity("rerank")).toBeNull();
    await llm.dispose();
  });

  test("rerank fallback is labeled with its actual GGUF URI", async () => {
    vi.stubEnv("QMD_MLX_RERANK", "1");
    const llm = fakeLlm({ embedBackend: "mlx", rerankModel: "fixture-ranker.gguf" });
    expect(await llm.getCacheIdentity("rerank")).toMatch(/^mlx:/);
    mlx.rerank.mockResolvedValue(null);
    llm.ensureRerankContexts = vi.fn(async () => [{ rankAll: vi.fn(async () => [0.1]) }]);
    llm.ensureRerankModel = vi.fn(async () => ({ tokenize: Array.from, detokenize: (v: string[]) => v.join("") }));
    const result = await llm.rerank("fixture", [{ file: "a", text: "fixture" }]);
    expect(result.model).toBe("fixture-ranker.gguf");
    await llm.dispose();
  });

  test('unproven expansion serving metadata never inherits the predicted MLX identity', async () => {
    const llm = fakeLlm({ strictMlx: true });
    vi.stubGlobal('fetch', vi.fn(async (input: string) => new Response(JSON.stringify(input.endsWith('/descriptor') ? mlx.descriptor : {
      text: 'vec: fixture query',
    }))));
    const predicted = await llm.getCacheIdentity('expand');
    expect(predicted).toMatch(/^mlx:/);
    const result = await llm.expandQueryWithIdentity('fixture query');
    expect(result.model).toBe('mlx');
    expect(result.model).not.toBe(predicted);
    await llm.dispose();
  });

  test('expansion fallback carries the actual GGUF identity', async () => {
    vi.stubEnv('QMD_MLX_EXPAND', '1');
    const llm = fakeLlm({ embedBackend: 'mlx' });
    vi.stubGlobal('fetch', vi.fn(async (input: string) => input.endsWith('/descriptor')
      ? new Response(JSON.stringify(mlx.descriptor)) : new Response('', { status: 503 })));
    const predicted = await llm.getCacheIdentity('expand');
    llm.expandQueryGguf = vi.fn(async () => ({ queries: [{ type: 'vec', text: 'fixture query' }], model: 'fixture-expand.gguf' }));
    const result = await llm.expandQueryWithIdentity('fixture query');
    expect(result.model).toBe('fixture-expand.gguf');
    expect(result.model).not.toBe(predicted);
    await llm.dispose();
  });

  test('revision changes during inference cannot publish a stale MLX label', async () => {
    const llm = fakeLlm({ strictMlx: true });
    mlx.rerank.mockImplementation(async () => {
      mlx.descriptor.rerank.revision = 'changed-during-inference';
      return { scores: [0.9], model: 'fixture-ranker' };
    });
    await expect(llm.rerank('fixture', [{ file: 'a', text: 'fixture' }])).rejects.toThrow(/identity changed/);
    await llm.dispose();
  });

  test("expansion identity comes from the actual response, not a predicted descriptor", async () => {
    const llm = fakeLlm({ strictMlx: true });
    vi.stubGlobal("fetch", vi.fn(async (input: string) => new Response(JSON.stringify(input.endsWith('/descriptor') ? mlx.descriptor : {
      text: "vec: fixture query", model: "fixture-generator",
    }), { status: 200, headers: { "content-type": "application/json" } })));
    const predicted = await llm.getCacheIdentity("expand");
    const result = await llm.expandQueryWithIdentity("fixture query");
    expect(result.queries).toEqual([{ type: "vec", text: "fixture query" }]);
    expect(result.model).toBe(predicted);
    await llm.dispose();
  });
});
