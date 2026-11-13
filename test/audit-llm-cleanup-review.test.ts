import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

vi.mock("node-llama-cpp", () => ({
  getLlama: vi.fn(),
  resolveModelFile: vi.fn(),
  LlamaLogLevel: { error: "error" },
  LlamaChatSession: class {
    prompt = vi.fn(async () => "fixture");
    dispose = vi.fn(async () => {});
  },
}));

import { getLlama } from "node-llama-cpp";
import {
  LlamaCpp, canUnloadLLM, disposeDefaultLlamaCpp, getDefaultLlamaCpp, setDefaultLlamaCpp,
} from "../src/llm.js";

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

function transientContext() {
  return {
    dispose: vi.fn().mockRejectedValueOnce(new Error("fixture transient cleanup failure")).mockResolvedValue(undefined),
    getEmbeddingFor: vi.fn(async () => ({ vector: [1, 0] })),
  };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubEnv("QMD_STRICT_MLX", "0");
  vi.stubEnv("QMD_LLAMA_GPU", "auto");
  vi.stubEnv("QMD_MLX_RERANK", "0");
  vi.stubEnv("QMD_MLX_EXPAND", "0");
  vi.mocked(getLlama).mockImplementation(() => { throw new Error("Real native initialization is forbidden in cleanup tests"); });
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Live HTTP is forbidden in cleanup tests"); }));
});

afterEach(() => {
  setDefaultLlamaCpp(null);
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("failed native cleanup retains ownership and fences reload", () => {
  test("failed idle context stays owned, blocks evaluation/allocation, and is retried by final dispose", async () => {
    const llm = fakeLlm();
    const context = transientContext();
    const model = { createEmbeddingContext: vi.fn(), dispose: vi.fn(async () => {}) };
    llm.embedModel = model;
    llm.embedContexts = [context];

    await expect(llm.unloadIdleResources()).rejects.toThrow(/dispose LLM resources/);
    expect(llm.embedContexts).toEqual([context]);
    expect(llm.failedResourceDisposals.size).toBe(1);
    expect(canUnloadLLM(llm)).toBe(false);
    await expect(llm.embed("must not reuse or reload")).rejects.toThrow(/cleanup.*dispose\(\)/i);
    await expect(llm.ensureEmbedContexts()).rejects.toThrow(/cleanup.*dispose\(\)/i);
    expect(context.getEmbeddingFor).not.toHaveBeenCalled();
    expect(model.createEmbeddingContext).not.toHaveBeenCalled();
    expect(context.dispose).toHaveBeenCalledTimes(1);

    await expect(llm.dispose()).resolves.toBeUndefined();
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(llm.failedResourceDisposals.size).toBe(0);
    expect(llm.embedContexts).toEqual([]);
    expect(llm.embedModel).toBeNull();
    expect(llm.inactivityTimer).toBeNull();
    expect(canUnloadLLM(llm)).toBe(false);
    await expect(llm.embed("closed")).rejects.toThrow(/closing|disposed/);
    expect(getLlama).not.toHaveBeenCalled();
  });

  test("idle transition retains pending contexts and shares one disposal attempt", async () => {
    const llm = fakeLlm();
    const started = deferred();
    const gate = deferred();
    const context = { dispose: vi.fn(() => { started.resolve(); return gate.promise; }) };
    llm.embedContexts = [context];
    const idle = llm.unloadIdleResources();
    await started.promise;
    const ownedWhilePending = llm.embedContexts.includes(context);
    const unloadableWhilePending = canUnloadLLM(llm);
    const sameTransition = llm.unloadIdleResources() === idle;
    gate.resolve();
    await idle;

    expect(ownedWhilePending).toBe(true);
    expect(unloadableWhilePending).toBe(false);
    expect(sameTransition).toBe(true);
    expect(context.dispose).toHaveBeenCalledTimes(1);
    expect(llm.embedContexts).toEqual([]);
    expect(canUnloadLLM(llm)).toBe(true);
    await llm.dispose();
    expect(context.dispose).toHaveBeenCalledTimes(1);
  });

  test("an operation waiting on failed idle cleanup cannot allocate, and overlapping close retries only once", async () => {
    const llm = fakeLlm();
    const started = deferred();
    const gate = deferred();
    const context = { dispose: vi.fn().mockImplementationOnce(() => { started.resolve(); return gate.promise; }).mockResolvedValue(undefined) };
    llm.embedContexts = [context];
    llm.ensureEmbedContext = vi.fn(async () => ({ getEmbeddingFor: vi.fn(async () => ({ vector: [1, 0] })) }));
    const idle = llm.unloadIdleResources();
    const idleRejection = expect(idle).rejects.toThrow(/dispose LLM resources/);
    await started.promise;
    const operation = llm.embed("waiting");
    const operationRejection = expect(operation).rejects.toThrow();
    const closing = llm.dispose();
    expect(llm.dispose()).toBe(closing);
    gate.reject(new Error("idle disposal failed before release"));
    await idleRejection;
    await operationRejection;
    await expect(closing).resolves.toBeUndefined();
    expect(llm.ensureEmbedContext).not.toHaveBeenCalled();
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(llm.failedResourceDisposals.size).toBe(0);
  });

  test("successful resources and aliases are not disposed again during cleanup retry", async () => {
    const llm = fakeLlm();
    const failed = transientContext();
    const successful = { dispose: vi.fn(async () => {}) };
    llm.embedContexts = [failed, successful];
    llm.rerankContexts = [failed, successful];
    await expect(llm.unloadIdleResources()).rejects.toThrow();
    expect(llm.embedContexts).toEqual([failed]);
    expect(llm.rerankContexts).toEqual([failed]);
    expect(failed.dispose).toHaveBeenCalledTimes(1);
    expect(successful.dispose).toHaveBeenCalledTimes(1);
    await llm.unloadIdleResources();
    expect(failed.dispose).toHaveBeenCalledTimes(1);

    const closing = llm.dispose();
    expect(llm.dispose()).toBe(closing);
    await closing;
    expect(failed.dispose).toHaveBeenCalledTimes(2);
    expect(successful.dispose).toHaveBeenCalledTimes(1);
    await llm.dispose();
    expect(failed.dispose).toHaveBeenCalledTimes(2);
  });

  test("permanent failure stays accounted after the single final retry, with an actionable terminal error", async () => {
    const llm = fakeLlm();
    const failure = new Error("fixture native release permanently failed");
    const context = { dispose: vi.fn(async () => { throw failure; }) };
    const successful = { dispose: vi.fn(async () => {}) };
    llm.embedContexts = [context, successful];
    await expect(llm.unloadIdleResources()).rejects.toThrow();
    const closing = llm.dispose();
    expect(llm.dispose()).toBe(closing);
    const error = await closing.catch((error: unknown) => error);
    expect(error).toBeInstanceOf(AggregateError);
    expect(error.message).toMatch(/LLM disposal failed.*1 resource.*retained.*restart/i);
    expect(error.errors).toContain(failure);
    expect(llm.embedContexts).toEqual([context]);
    expect(llm.failedResourceDisposals.get(context)).toMatchObject({ error: failure, attempts: 2 });
    expect(successful.dispose).toHaveBeenCalledTimes(1);
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(canUnloadLLM(llm)).toBe(false);
    await llm.unloadIdleResources();
    await expect(llm.dispose()).rejects.toBe(error);
    expect(context.dispose).toHaveBeenCalledTimes(2);
    await expect(llm.ensureEmbedContexts()).rejects.toThrow(/closing|disposed/);
  });

  test("permanent context failure retains its dependent model and engine without cascading extra cleanup attempts", async () => {
    const llm = fakeLlm({ disposeModelsOnInactivity: true });
    const context = { dispose: vi.fn(async () => { throw new Error("fixture permanent child failure"); }) };
    const model = { dispose: vi.fn(async () => { throw new Error("must not cascade into a failed context"); }) };
    const unrelatedModel = { dispose: vi.fn(async () => {}) };
    const engine = { dispose: vi.fn(async () => { throw new Error("must not cascade into a failed model"); }) };
    llm.embedContexts = [context];
    llm.embedModel = model;
    llm.generateModel = unrelatedModel;
    llm.llama = engine;
    await expect(llm.unloadIdleResources()).rejects.toThrow();
    expect(model.dispose).not.toHaveBeenCalled();
    expect(unrelatedModel.dispose).toHaveBeenCalledTimes(1);
    await expect(llm.dispose()).rejects.toThrow(/LLM disposal failed.*1 resource.*retained.*restart/i);
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(model.dispose).not.toHaveBeenCalled();
    expect(engine.dispose).not.toHaveBeenCalled();
    expect(llm.embedModel).toBe(model);
    expect(llm.llama).toBe(engine);
  });

  test("successful context retry precedes disposal of its model and engine", async () => {
    const llm = fakeLlm({ disposeModelsOnInactivity: true });
    const order: string[] = [];
    let released = false;
    const context = { dispose: vi.fn(async () => {
      order.push("context");
      if (order.length === 1) throw new Error("fixture transient child failure");
      released = true;
    }) };
    const model = { dispose: vi.fn(async () => { expect(released).toBe(true); order.push("model"); }) };
    const engine = { dispose: vi.fn(async () => { order.push("engine"); }) };
    llm.embedContexts = [context];
    llm.embedModel = model;
    llm.llama = engine;
    await expect(llm.unloadIdleResources()).rejects.toThrow();
    expect(model.dispose).not.toHaveBeenCalled();
    await expect(llm.dispose()).resolves.toBeUndefined();
    expect(order).toEqual(["context", "context", "model", "engine"]);
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(engine.dispose).toHaveBeenCalledTimes(1);
  });

  test("a failure first encountered in final dispose is not retried by repeated dispose calls", async () => {
    const llm = fakeLlm();
    const context = transientContext();
    llm.embedContexts = [context];
    const closing = llm.dispose();
    await expect(closing).rejects.toThrow(/LLM disposal failed.*retained.*restart/i);
    expect(llm.embedContexts).toEqual([context]);
    expect(llm.failedResourceDisposals.size).toBe(1);
    expect(llm.dispose()).toBe(closing);
    await expect(llm.dispose()).rejects.toThrow();
    expect(context.dispose).toHaveBeenCalledTimes(1);
  });

  test("idle weight cleanup failure preserves its model and blocks model reload until final cleanup", async () => {
    const llm = fakeLlm({ disposeModelsOnInactivity: true });
    const model = { dispose: transientContext().dispose, createContext: vi.fn() };
    llm.generateModel = model;
    await expect(llm.unloadIdleResources()).rejects.toThrow();
    expect(llm.generateModel).toBe(model);
    expect(canUnloadLLM(llm)).toBe(false);
    await expect(llm.generate("no reload")).rejects.toThrow(/cleanup.*dispose\(\)/i);
    expect(model.createContext).not.toHaveBeenCalled();
    await expect(llm.dispose()).resolves.toBeUndefined();
    expect(model.dispose).toHaveBeenCalledTimes(2);
    expect(llm.generateModel).toBeNull();
  });

  test("failed cleanup never rearms the inactivity timer into an automatic retry loop", async () => {
    vi.useFakeTimers();
    const llm = fakeLlm({ inactivityTimeoutMs: 10 });
    const context = transientContext();
    llm.embedContexts = [context];
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      llm.touchActivity();
      await vi.advanceTimersByTimeAsync(11);
      expect(context.dispose).toHaveBeenCalledTimes(1);
      expect(llm.embedContexts).toEqual([context]);
      expect(llm.inactivityTimer).toBeNull();
      expect(canUnloadLLM(llm)).toBe(false);
      await vi.advanceTimersByTimeAsync(10_000);
      expect(context.dispose).toHaveBeenCalledTimes(1);
      expect(logged).toHaveBeenCalledTimes(1);
      await llm.dispose();
      expect(context.dispose).toHaveBeenCalledTimes(2);
    } finally {
      logged.mockRestore();
    }
  });
});

describe("default owner cleanup fencing", () => {
  test("failed final cleanup retains the default owner instead of permitting a new native pool", async () => {
    const llm = fakeLlm();
    const context = { dispose: vi.fn(async () => { throw new Error("fixture permanent default cleanup failure"); }) };
    llm.embedContexts = [context];
    setDefaultLlamaCpp(llm);
    await expect(llm.unloadIdleResources()).rejects.toThrow();
    await expect(disposeDefaultLlamaCpp()).rejects.toThrow(/retained.*restart/i);
    expect(getDefaultLlamaCpp()).toBe(llm);
    expect(canUnloadLLM()).toBe(false);
    await expect(getDefaultLlamaCpp().embed("no unsafe replacement")).rejects.toThrow(/closing|disposed/);
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(llm.embedContexts).toEqual([context]);
    expect(getLlama).not.toHaveBeenCalled();
  });

  test("successful final cleanup releases the default owner for an explicitly fresh instance", async () => {
    const llm = fakeLlm();
    setDefaultLlamaCpp(llm);
    await disposeDefaultLlamaCpp();
    const replacement = getDefaultLlamaCpp();
    expect(replacement).not.toBe(llm);
    await disposeDefaultLlamaCpp();
    expect(getLlama).not.toHaveBeenCalled();
  });

  test("cleanup of an older default never clears a concurrently selected replacement", async () => {
    const llm = fakeLlm();
    const replacement = fakeLlm();
    const started = deferred();
    const gate = deferred();
    llm.embedContexts = [{ dispose: vi.fn(() => { started.resolve(); return gate.promise; }) }];
    setDefaultLlamaCpp(llm);
    const closing = disposeDefaultLlamaCpp();
    await started.promise;
    setDefaultLlamaCpp(replacement);
    gate.resolve();
    await closing;
    expect(getDefaultLlamaCpp()).toBe(replacement);
    await disposeDefaultLlamaCpp();
  });
});

describe("non-pool resources and failed loader disposal fencing", () => {
  test("failed per-call generation context cleanup is retained and retried at close", async () => {
    const llm = fakeLlm();
    const context = { ...transientContext(), getSequence: vi.fn(() => { throw new Error("fixture sequence setup failed"); }) };
    const model = { createContext: vi.fn(async () => context), dispose: vi.fn(async () => {}) };
    llm.generateModel = model;
    await expect(llm.generate("fixture")).rejects.toThrow("fixture transient cleanup failure");
    expect(llm.failedResourceDisposals.has(context)).toBe(true);
    expect(canUnloadLLM(llm)).toBe(false);
    await expect(llm.generate("must not create another context")).rejects.toThrow(/cleanup.*dispose\(\)/i);
    expect(model.createContext).toHaveBeenCalledTimes(1);
    await expect(llm.dispose()).resolves.toBeUndefined();
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(llm.failedResourceDisposals.size).toBe(0);
  });

  test("permanent per-call context failure also prevents dependent model and engine cascades", async () => {
    const llm = fakeLlm();
    const context = {
      getSequence: vi.fn(() => { throw new Error("fixture setup failure"); }),
      dispose: vi.fn(async () => { throw new Error("fixture permanent per-call cleanup failure"); }),
    };
    const model = { createContext: vi.fn(async () => context), dispose: vi.fn(async () => {}) };
    const engine = { dispose: vi.fn(async () => {}) };
    llm.generateModel = model;
    llm.llama = engine;
    await expect(llm.generate("fixture")).rejects.toThrow("fixture permanent per-call cleanup failure");
    await expect(llm.dispose()).rejects.toThrow(/LLM disposal failed.*1 resource.*retained.*restart/i);
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(model.dispose).not.toHaveBeenCalled();
    expect(engine.dispose).not.toHaveBeenCalled();
    expect(llm.generateModel).toBe(model);
    expect(llm.llama).toBe(engine);
  });

  test("partial pool cleanup failure remains owned even though the pool was never published", async () => {
    const llm = fakeLlm();
    const context = transientContext();
    const createEmbeddingContext = vi.fn().mockResolvedValueOnce(context).mockRejectedValue(new Error("fixture allocation failed"));
    llm.embedModel = { createEmbeddingContext, dispose: vi.fn(async () => {}) };
    llm.computeParallelism = vi.fn(async () => 2);
    llm.threadsPerContext = vi.fn(async () => 1);
    await expect(llm.ensureEmbedContexts()).rejects.toThrow(/Context creation and cleanup failed/);
    expect(llm.embedContexts).toEqual([]);
    expect(llm.embedContextsCreatePromise).toBeNull();
    expect(llm.failedResourceDisposals.has(context)).toBe(true);
    expect(canUnloadLLM(llm)).toBe(false);
    await expect(llm.ensureEmbedContexts()).rejects.toThrow(/cleanup.*dispose\(\)/i);
    expect(createEmbeddingContext).toHaveBeenCalledTimes(2);
    await expect(llm.dispose()).resolves.toBeUndefined();
    expect(context.dispose).toHaveBeenCalledTimes(2);
    expect(llm.failedResourceDisposals.size).toBe(0);
  });

  test("late model disposal failure is retained across loader rejection and retried after the loader drains", async () => {
    const llm = fakeLlm();
    const started = deferred();
    const load = deferred<any>();
    const model = { dispose: transientContext().dispose };
    const engine = { loadModel: vi.fn(() => { started.resolve(); return load.promise; }), dispose: vi.fn(async () => {}) };
    llm.llama = engine;
    llm.resolveModel = vi.fn(async () => "/fixture/model.gguf");
    const pending = llm.ensureEmbedModel();
    const rejected = expect(pending).rejects.toThrow("fixture transient cleanup failure");
    await started.promise;
    const closing = llm.dispose();
    load.resolve(model);
    await rejected;
    await expect(closing).resolves.toBeUndefined();
    expect(model.dispose).toHaveBeenCalledTimes(2);
    expect(engine.dispose).toHaveBeenCalledTimes(1);
    expect(llm.embedModel).toBeNull();
    expect(llm.embedModelLoadPromise).toBeNull();
    expect(llm.failedResourceDisposals.size).toBe(0);
  });

  test("late engine disposal failure is retried without publishing or double-disposing a successful engine", async () => {
    const llm = fakeLlm();
    const load = deferred<any>();
    const engine = { gpu: "fixture", dispose: transientContext().dispose };
    vi.mocked(getLlama).mockReturnValue(load.promise);
    const pending = llm.ensureLlama();
    const rejected = expect(pending).rejects.toThrow("fixture transient cleanup failure");
    const closing = llm.dispose();
    load.resolve(engine);
    await rejected;
    await expect(closing).resolves.toBeUndefined();
    expect(engine.dispose).toHaveBeenCalledTimes(2);
    expect(llm.llama).toBeNull();
    expect(llm.llamaLoadPromise).toBeNull();
    await llm.dispose();
    expect(engine.dispose).toHaveBeenCalledTimes(2);
  });

  test.each(["auto", "off"])("a rejected engine load still drains during close without reopening initialization (gpu=%s)", async gpu => {
    vi.stubEnv("QMD_LLAMA_GPU", gpu);
    const llm = fakeLlm();
    const load = deferred<any>();
    vi.mocked(getLlama).mockReturnValue(load.promise);
    const pending = llm.ensureLlama();
    const rejected = expect(pending).rejects.toThrow(gpu === "auto" ? /closing|disposed/ : "fixture engine load failed");
    let closed = false;
    const closing = llm.dispose().then(() => { closed = true; });
    await Promise.resolve();
    expect(closed).toBe(false);
    load.reject(new Error("fixture engine load failed"));
    await rejected;
    await closing;
    expect(llm.llamaLoadPromise).toBeNull();
    expect(llm.llama).toBeNull();
    await expect(llm.ensureLlama()).rejects.toThrow(/closing|disposed/);
    expect(getLlama).toHaveBeenCalledTimes(1);
  });

  test("a rejected model load cannot outlive final disposal or resume model initialization", async () => {
    const llm = fakeLlm();
    const started = deferred();
    const load = deferred<any>();
    const engine = { loadModel: vi.fn(() => { started.resolve(); return load.promise; }), dispose: vi.fn(async () => {}) };
    llm.llama = engine;
    llm.resolveModel = vi.fn(async () => "/fixture/model.gguf");
    const pending = llm.ensureEmbedModel();
    const rejected = expect(pending).rejects.toThrow("fixture model load failed");
    await started.promise;
    let closed = false;
    const closing = llm.dispose().then(() => { closed = true; });
    await Promise.resolve();
    expect(closed).toBe(false);
    expect(engine.dispose).not.toHaveBeenCalled();
    load.reject(new Error("fixture model load failed"));
    await rejected;
    await closing;
    expect(engine.dispose).toHaveBeenCalledTimes(1);
    expect(llm.embedModelLoadPromise).toBeNull();
    await expect(llm.ensureEmbedModel()).rejects.toThrow(/closing|disposed/);
    expect(engine.loadModel).toHaveBeenCalledTimes(1);
  });
});
