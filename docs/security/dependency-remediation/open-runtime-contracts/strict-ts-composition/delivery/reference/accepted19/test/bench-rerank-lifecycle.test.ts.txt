import { afterAll, beforeAll, describe, expect, test, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import * as ts from "typescript";

// Synthetic ownership/CLI tests only: no native library, models, GPU or inference.
// Run the complete actual module after changing ONLY its external dependency
// import and adding test exports. Verify the emitted executable bodies are exact.
type Context = {
  rankAll: ReturnType<typeof vi.fn<(query: string, docs: string[]) => Promise<number[]>>>;
  dispose: ReturnType<typeof vi.fn<() => Promise<void>>>;
};
function fixture() {
  const events: string[] = [];
  const contexts: Context[] = [];
  const model = {
    defaultContextFlashAttention: "auto" as const,
    trainContextSize: 4096,
    createRankingContext: vi.fn(async (_options: object) => {
      const index = contexts.length;
      const context: Context = {
        rankAll: vi.fn(async (_query: string, docs: string[]) => docs.map(() => 0.5)),
        dispose: vi.fn(async () => { events.push(`context:${index}`); }),
      };
      contexts.push(context);
      return context;
    }),
    dispose: vi.fn(async () => { events.push("model"); }),
  };
  const llama = {
    gpu: false as false | "metal",
    cpuMathCores: 8,
    getVramState: vi.fn(async () => ({ used: 100, total: 1000, free: 900 })),
    getGpuDeviceNames: vi.fn(async () => ["synthetic"]),
    loadModel: vi.fn(async (_options: object) => model),
    dispose: vi.fn(async () => { events.push("llama"); }),
  };
  return { contexts, model, llama, events };
}
type Fixture = ReturnType<typeof fixture>;
type BenchResult = { parallelism: number; times: number[]; medianMs: number; docsPerSec: number; flashAttention: boolean | "auto" };
type BenchModule = {
  benchmarkConfig(model: Fixture["model"], llama: Fixture["llama"], docs: string[], parallelism: number): Promise<BenchResult>;
  main(): Promise<void>;
};
let bench: BenchModule;
let native: { setBackend(llama: Fixture["llama"], resolve?: () => Promise<string>): void };
let directory: string;
let modulePath: string;
const docs = ["a", "b", "c", "d"];

const nativeStub = `
import { appendFileSync } from "node:fs";
const mode = process.env.QMD_BENCH_FAKE_MODE;
const trace = (event) => { if (process.env.QMD_BENCH_FAKE_TRACE) appendFileSync(process.env.QMD_BENCH_FAKE_TRACE, event + "\\n"); };
let backend;
let resolve;
export function setBackend(value, resolver = async () => "/synthetic/model.gguf") { backend = value; resolve = resolver; }
export const LlamaLogLevel = { error: "error" };
export async function resolveModelFile() {
  trace("resolve");
  if (resolve) return resolve();
  if (mode === "resolve-failure") throw new Error("resolve-fault");
  return "/synthetic/model.gguf";
}
export async function getLlama() {
  trace("llama:create");
  if (backend) return backend;
  if (!mode) throw new Error("synthetic backend not configured");
  let index = 0;
  const model = {
    defaultContextFlashAttention: "auto", trainContextSize: 4096,
    async createRankingContext(options) {
      if (Object.keys(options).some(key => !["contextSize", "threads"].includes(key))) throw new Error("unsupported-context-option");
      const id = index++;
      trace("context:create:" + id);
      return {
        async rankAll(_query, docs) {
          trace("rank:" + id);
          if (mode === "rank-failure" || mode === "operation-cleanup-failure") throw new Error("rank-fault");
          return docs.map(() => 0.5);
        },
        async dispose() {
          trace("context:dispose:" + id);
          if (mode === "context-cleanup-failure") throw new Error("context-cleanup-fault");
        }
      };
    },
    async dispose() {
      trace("model:dispose");
      if (mode === "model-cleanup-failure" || mode === "operation-cleanup-failure") throw new Error("model-cleanup-fault");
    }
  };
  return {
    gpu: false, cpuMathCores: 4,
    async loadModel() { trace("model:create"); return model; },
    async dispose() {
      trace("llama:dispose");
      if (mode === "operation-cleanup-failure") throw new Error("llama-cleanup-fault");
    }
  };
}
`;

beforeAll(async () => {
  directory = mkdtempSync(join(tmpdir(), "qmd-bench-lifecycle-"));
  const sourcePath = new URL("../src/bench-rerank.ts", import.meta.url);
  const source = readFileSync(sourcePath, "utf8");
  const ast = ts.createSourceFile("bench-rerank.ts", source, ts.ScriptTarget.Latest, true);
  const imports = ast.statements.filter(ts.isImportDeclaration);
  const dependency = imports.filter(node => ts.isStringLiteral(node.moduleSpecifier) && node.moduleSpecifier.text === "node-llama-cpp");
  expect(dependency).toHaveLength(1);
  const specifier = dependency[0]!.moduleSpecifier;
  const stubPath = join(directory, "synthetic-native.mjs");
  writeFileSync(stubPath, nativeStub, { flag: "wx" });
  const rewritten = source.slice(0, specifier.getStart(ast)) + JSON.stringify(pathToFileURL(stubPath).href) + source.slice(specifier.end);
  const compile = (text: string) => ts.transpileModule(text, {
    compilerOptions: { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
    reportDiagnostics: true,
  });
  const original = compile(source);
  const compiled = compile(rewritten + "\nexport { benchmarkConfig, main };\n");
  expect(compiled.diagnostics).toEqual([]);
  const bodies = (text: string) => {
    const parsed = ts.createSourceFile("bench.mjs", text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
    return parsed.statements.filter(node => !ts.isImportDeclaration(node) && !ts.isExportDeclaration(node))
      .map(node => node.getText(parsed)).join("\n");
  };
  expect(bodies(compiled.outputText)).toBe(bodies(original.outputText));
  if (process.env.QMD_BENCH_TEST_RECEIPT) writeFileSync(process.env.QMD_BENCH_TEST_RECEIPT, JSON.stringify({
    executable: process.execPath, versions: process.versions,
    sourceSha256: createHash("sha256").update(source).digest("hex"),
    exactExecutableBodiesVerified: true, syntheticResourcesOnly: true, nativeModelOrGpuRuns: 0,
  }, null, 2), { flag: "wx" });
  modulePath = join(directory, "actual-benchmark.mjs");
  writeFileSync(modulePath, compiled.outputText, { flag: "wx" });
  native = await import(pathToFileURL(stubPath).href) as typeof native;
  bench = await import(pathToFileURL(modulePath).href) as BenchModule;
});
afterAll(() => { if (directory) rmSync(directory, { recursive: true, force: true }); });

async function acquired(f: Fixture, count = 2) {
  for (let i = 0; i < count; i++) await f.model.createRankingContext({});
  f.model.createRankingContext.mockReset();
  for (const context of f.contexts) f.model.createRankingContext.mockResolvedValueOnce(context);
}
function allClosed(f: Fixture) {
  expect(f.contexts.map(context => context.dispose.mock.calls.length)).toEqual(f.contexts.map(() => 1));
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}
async function runMain(f: Fixture, resolver?: () => Promise<string>) {
  native.setBackend(f.llama, resolver);
  const log = vi.spyOn(console, "log").mockImplementation(() => {});
  const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
  try { await bench.main(); }
  finally { log.mockRestore(); write.mockRestore(); }
}

describe("rerank benchmark ownership", () => {
  test("successful runs retain measurement, default attention and CPU routing", async () => {
    const f = fixture();
    const result = await bench.benchmarkConfig(f.model, f.llama, docs, 2);
    expect(result.parallelism).toBe(2);
    expect(result.flashAttention).toBe("auto");
    expect(result.times).toHaveLength(3);
    expect(result.times.every(value => Number.isFinite(value) && value >= 0)).toBe(true);
    expect(f.model.createRankingContext.mock.calls).toEqual(Array.from({ length: 2 }, () => [{ contextSize: 2048, threads: 4 }]));
    expect(f.contexts[0]!.rankAll.mock.calls.slice(1).map(call => call[1])).toEqual(Array.from({ length: 3 }, () => docs.slice(0, 2)));
    expect(f.contexts[1]!.rankAll.mock.calls.map(call => call[1])).toEqual(Array.from({ length: 3 }, () => docs.slice(2)));
    allClosed(f);
  });
  test("contexts with empty chunks are released without starting rankings", async () => {
    const f = fixture();
    const result = await bench.benchmarkConfig(f.model, f.llama, ["only"], 4);
    expect(result.parallelism).toBe(4);
    expect(f.contexts.slice(1).map(context => context.rankAll.mock.calls.length)).toEqual([0, 0, 0]);
    allClosed(f);
  });
  test("partial capacity is still usable and owned", async () => {
    const f = fixture();
    await acquired(f, 1);
    f.model.createRankingContext.mockRejectedValueOnce(new Error("capacity"));
    const result = await bench.benchmarkConfig(f.model, f.llama, docs, 4);
    expect(result.parallelism).toBe(1);
    expect(f.model.createRankingContext).toHaveBeenCalledTimes(2);
    allClosed(f);
  });
  test("first-context retry stays bounded with identical supported options", async () => {
    const f = fixture();
    await acquired(f, 1);
    f.model.createRankingContext.mockReset().mockRejectedValueOnce(new Error("capacity")).mockResolvedValueOnce(f.contexts[0]!);
    const result = await bench.benchmarkConfig(f.model, f.llama, docs, 4);
    expect(result.parallelism).toBe(1);
    expect(f.model.createRankingContext.mock.calls).toEqual(Array.from({ length: 2 }, () => [{ contextSize: 2048, threads: 2 }]));
    allClosed(f);
  });
  test("exhausted first-context retry propagates and acquires nothing", async () => {
    const f = fixture();
    const fault = new Error("retry exhausted");
    f.model.createRankingContext.mockRejectedValue(fault);
    await expect(bench.benchmarkConfig(f.model, f.llama, docs, 4)).rejects.toBe(fault);
    expect(f.model.createRankingContext).toHaveBeenCalledTimes(2);
    expect(f.contexts).toEqual([]);
  });
  test("zero contexts refuse without ranking or disposal", async () => {
    const f = fixture();
    await expect(bench.benchmarkConfig(f.model, f.llama, docs, 0)).rejects.toThrow("No ranking contexts created");
    expect(f.model.createRankingContext).not.toHaveBeenCalled();
  });
  test("post-acquisition metadata failure releases every context", async () => {
    const f = fixture();
    f.llama.gpu = "metal";
    const fault = new Error("vram metadata");
    f.llama.getVramState.mockResolvedValueOnce({ used: 0, total: 1000, free: 1000 }).mockRejectedValueOnce(fault);
    await expect(bench.benchmarkConfig(f.model, f.llama, docs, 2)).rejects.toBe(fault);
    allClosed(f);
    expect(f.contexts[0]!.rankAll).not.toHaveBeenCalled();
    expect(f.model.createRankingContext.mock.calls).toEqual(Array.from({ length: 2 }, () => [{ contextSize: 2048 }]));
  });
  test("warmup failure releases every context", async () => {
    const f = fixture();
    await acquired(f);
    const fault = new Error("warmup");
    f.contexts[0]!.rankAll.mockRejectedValueOnce(fault);
    await expect(bench.benchmarkConfig(f.model, f.llama, docs, 2)).rejects.toBe(fault);
    allClosed(f);
  });
  test.each([NaN, -0.1, 1.1])("score validation failure releases every context (%s)", async score => {
    const f = fixture();
    await acquired(f);
    f.contexts[1]!.rankAll.mockResolvedValue([score]);
    await expect(bench.benchmarkConfig(f.model, f.llama, docs, 2)).rejects.toThrow("Invalid scores detected");
    allClosed(f);
  });
  test.each([false, true])("drains started rankings before disposal, including sync throw=%s", async sync => {
    const f = fixture();
    await acquired(f);
    const started = deferred<void>();
    const release = deferred<number[]>();
    const fault = new Error("ranking");
    f.contexts[0]!.rankAll.mockResolvedValueOnce([0.5]).mockImplementationOnce(() => {
      if (sync) throw fault;
      return Promise.reject(fault);
    });
    let active = false;
    f.contexts[1]!.rankAll.mockImplementationOnce(async () => {
      active = true;
      started.resolve();
      try { return await release.promise; } finally { active = false; }
    });
    for (const context of f.contexts) context.dispose.mockImplementation(async () => { expect(active).toBe(false); });
    let returned = false;
    const work = bench.benchmarkConfig(f.model, f.llama, docs, 2).finally(() => { returned = true; });
    const outcome = work.catch(error => error as unknown);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("sibling was not started")), 1000);
    });
    try {
      // Race is a test bound, never a production cancellation/disposal timeout.
      await Promise.race([started.promise, timeout]);
      await new Promise<void>(resolve => setImmediate(resolve));
      expect(returned).toBe(false);
      expect(f.contexts.map(context => context.dispose.mock.calls.length)).toEqual([0, 0]);
    } finally {
      clearTimeout(timer);
      release.resolve([0.5]);
      await outcome;
    }
    expect(await outcome).toBe(fault);
    allClosed(f);
  });
  test("retains multiple parallel ranking failures", async () => {
    const f = fixture();
    await acquired(f);
    const faults = [new Error("rank 0"), new Error("rank 1")];
    f.contexts[0]!.rankAll.mockResolvedValueOnce([0.5]).mockRejectedValueOnce(faults[0]);
    f.contexts[1]!.rankAll.mockRejectedValueOnce(faults[1]);
    const error = await bench.benchmarkConfig(f.model, f.llama, docs, 2).catch(error => error as AggregateError);
    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).errors).toEqual(faults);
    allClosed(f);
  });
  test.each([false, true])("attempts all cleanup and retains primary failure=%s", async primary => {
    const f = fixture();
    await acquired(f);
    const operation = new Error("operation");
    const cleanup = [new Error("dispose 0"), new Error("dispose 1")];
    if (primary) f.contexts[0]!.rankAll.mockRejectedValueOnce(operation);
    f.contexts.forEach((context, i) => context.dispose.mockRejectedValueOnce(cleanup[i]));
    const error = await bench.benchmarkConfig(f.model, f.llama, docs, 2).catch(error => error as AggregateError);
    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).errors).toEqual(primary ? [operation, ...cleanup] : cleanup);
    allClosed(f);
  });
});

describe("rerank benchmark top-level ownership", () => {
  test("resolution failure still releases llama", async () => {
    const f = fixture();
    const fault = new Error("resolve");
    await expect(runMain(f, async () => { throw fault; })).rejects.toBe(fault);
    expect(f.events).toEqual(["llama"]);
    expect(f.llama.loadModel).not.toHaveBeenCalled();
  });
  test("model-load failure still releases llama", async () => {
    const f = fixture();
    const fault = new Error("load");
    f.llama.loadModel.mockRejectedValueOnce(fault);
    await expect(runMain(f)).rejects.toBe(fault);
    expect(f.events).toEqual(["llama"]);
    expect(f.model.dispose).not.toHaveBeenCalled();
  });
  test("system metadata failure still releases llama", async () => {
    const f = fixture();
    f.llama.gpu = "metal";
    const fault = new Error("device metadata");
    f.llama.getGpuDeviceNames.mockRejectedValueOnce(fault);
    await expect(runMain(f)).rejects.toBe(fault);
    expect(f.events).toEqual(["llama"]);
    expect(f.llama.loadModel).not.toHaveBeenCalled();
  });
  test("post-model metadata failure releases model then llama", async () => {
    const f = fixture();
    const fault = new Error("metadata");
    Object.defineProperty(f.model, "trainContextSize", { get() { throw fault; } });
    await expect(runMain(f)).rejects.toBe(fault);
    expect(f.events).toEqual(["model", "llama"]);
  });
  test("no valid results is fatal and still cleans up in order", async () => {
    const f = fixture();
    f.llama.cpuMathCores = 4;
    await acquired(f, 1);
    const fault = new Error("rank");
    f.contexts[0]!.rankAll.mockRejectedValueOnce(fault);
    const error = await runMain(f).catch(error => error as AggregateError);
    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).errors).toEqual([fault]);
    expect(f.events).toEqual(["context:0", "model", "llama"]);
  });
  test("ordinary per-config failure can still produce partial valid results", async () => {
    const f = fixture();
    f.llama.cpuMathCores = 16;
    const original = f.model.createRankingContext.getMockImplementation()!;
    f.model.createRankingContext.mockImplementationOnce(async options => {
      const context = await original(options);
      context.rankAll.mockRejectedValueOnce(new Error("first config"));
      return context;
    });
    await runMain(f);
    expect(f.contexts).toHaveLength(7); // 1 failed + 2 + 4; 8 skipped by the unchanged CPU gate.
    allClosed(f);
    expect(f.events.slice(-2)).toEqual(["model", "llama"]);
  });
  test("context cleanup failure is fatal: no next config or summary", async () => {
    const f = fixture();
    f.llama.cpuMathCores = 32;
    await acquired(f, 1);
    const fault = new Error("context cleanup");
    f.contexts[0]!.dispose.mockRejectedValueOnce(fault);
    const error = await runMain(f).catch(error => error as AggregateError);
    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).errors).toEqual([fault]);
    expect(f.model.createRankingContext).toHaveBeenCalledTimes(1);
    expect(f.events).toEqual(["model", "llama"]);
  });
  test("successful summary follows all parent releases", async () => {
    const f = fixture();
    f.llama.cpuMathCores = 4;
    native.setBackend(f.llama);
    const log = vi.spyOn(console, "log").mockImplementation(message => {
      if (message === "  Results") f.events.push("summary");
    });
    const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    try {
      await bench.main();
      expect(f.events).toEqual(["context:0", "model", "llama", "summary"]);
    } finally { log.mockRestore(); write.mockRestore(); }
  });
  test("parent cleanup failures are all attempted and block summary publication", async () => {
    const f = fixture();
    f.llama.cpuMathCores = 4;
    const failures = [new Error("model cleanup"), new Error("llama cleanup")];
    f.model.dispose.mockRejectedValueOnce(failures[0]);
    f.llama.dispose.mockRejectedValueOnce(failures[1]);
    native.setBackend(f.llama);
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    try {
      const error = await bench.main().catch(error => error as AggregateError);
      expect(error).toBeInstanceOf(AggregateError);
      expect((error as AggregateError).errors).toEqual(failures);
      expect(log.mock.calls.flat().join("\n")).not.toMatch(/Results|Best:/);
      expect(f.model.dispose).toHaveBeenCalledTimes(1);
      expect(f.llama.dispose).toHaveBeenCalledTimes(1);
    } finally { log.mockRestore(); write.mockRestore(); }
  });
});

describe("actual benchmark entrypoint (synthetic dependency only)", () => {
  test("importing the complete module does not acquire models or emit benchmark output", () => {
    const trace = join(directory, "import.trace");
    writeFileSync(trace, "", { flag: "wx" });
    const result = spawnSync(process.execPath, ["--input-type=module", "-e", `await import(${JSON.stringify(pathToFileURL(modulePath).href)})`], {
      cwd: directory, env: { ...process.env, QMD_BENCH_FAKE_MODE: "success", QMD_BENCH_FAKE_TRACE: trace }, encoding: "utf8", timeout: 10000,
    });
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
    expect(result.stdout).toBe("");
    expect(result.stderr).toBe("");
    expect(readFileSync(trace, "utf8")).toBe("");
  });
  test.each(["success", "rank-failure"])("symlink entrypoint runs rather than silently exiting (%s)", mode => {
    const entry = join(directory, `${mode}-entry.mjs`);
    symlinkSync(modulePath, entry);
    const result = spawnSync(process.execPath, [entry, "--quick"], {
      cwd: directory, env: { ...process.env, QMD_BENCH_FAKE_MODE: mode }, encoding: "utf8", timeout: 10000,
    });
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(mode === "success" ? 0 : 1);
    expect(result.stdout).toContain("QMD Reranker Benchmark");
    if (mode === "success") expect(result.stdout).toContain("Best:");
  });
  test.each([
    ["success", 0, ["model:dispose", "llama:dispose"]],
    ["resolve-failure", 1, ["llama:dispose"]],
    ["rank-failure", 1, ["context:dispose:0", "model:dispose", "llama:dispose"]],
    ["context-cleanup-failure", 1, ["context:dispose:0", "model:dispose", "llama:dispose"]],
    ["model-cleanup-failure", 1, ["context:dispose:0", "model:dispose", "llama:dispose"]],
    ["operation-cleanup-failure", 1, ["context:dispose:0", "model:dispose", "llama:dispose"]],
  ] as const)("CLI mode %s has honest exit and cleanup", (mode, code, suffix) => {
    const trace = join(directory, `${mode}.trace`);
    writeFileSync(trace, "", { flag: "wx" });
    const result = spawnSync(process.execPath, [modulePath, "--quick"], {
      cwd: directory, env: { ...process.env, QMD_BENCH_FAKE_MODE: mode, QMD_BENCH_FAKE_TRACE: trace }, encoding: "utf8", timeout: 10000,
    });
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(code);
    const events = readFileSync(trace, "utf8").trim().split("\n");
    expect(events.slice(-suffix.length)).toEqual(suffix);
    if (code === 0) expect(result.stdout).toContain("Best:");
    else {
      expect(result.stdout).not.toMatch(/Results|Best:/);
      expect(result.stderr).not.toBe("");
    }
    if (mode === "operation-cleanup-failure") {
      for (const message of ["rank-fault", "model-cleanup-fault", "llama-cleanup-fault"]) expect(result.stderr).toContain(message);
    }
  });
});
