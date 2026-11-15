import { describe, expect, it } from "vitest";
import * as fs from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { captureRecoveryExecutionFailure } from "./helpers/recovery-execution-failure.js";

// Never import the registering admission suite. Compile WHOLE declarations from
// its actual AST; only process/clock bindings are synthetic. FS and collector are real.
const admissionPath = fileURLToPath(new URL("./audit-recovery-execution-admission.test.ts", import.meta.url));
const sourceText = fs.readFileSync(admissionPath, "utf8");
const ast = ts.createSourceFile(admissionPath, sourceText, ts.ScriptTarget.Latest, true);
const functions = Object.fromEntries(ast.statements.filter(ts.isFunctionDeclaration).map(node => [node.name!.text, node.getText(ast)]));
const compile = (source: string) => ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
const digest = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
const helperPath = fileURLToPath(new URL("./helpers/recovery-execution-cleanup.ts", import.meta.url));
const helpers: any = fs.existsSync(helperPath) ? await import(/* @vite-ignore */ new URL("./helpers/recovery-execution-cleanup.ts", import.meta.url).href) : {};
const output = process.env.QMD_CLEANUP_OUTPUT;
if (output) {
  fs.mkdirSync(output, { recursive: true });
  const extracted: Record<string, unknown> = {};
  for (const name of ["hostCase", "rawCase", "invalidBudgetFrame", "observeStageFailure", "reap", "cleanEnv", "instrument"]) {
    fs.writeFileSync(path.join(output, name + ".source.ts"), functions[name]!);
    extracted[name] = { sha256: digest(functions[name]!), bytes: Buffer.byteLength(functions[name]!) };
  }
  fs.writeFileSync(path.join(output, "source-binding.json"), JSON.stringify({ admissionPath, sourceSha256: digest(sourceText), helperSha256: fs.existsSync(helperPath) ? digest(fs.readFileSync(helperPath)) : null, functions: extracted, host: { execPath: process.execPath, node: process.versions.node, bun: process.versions.bun ?? null }, scope: "whole-source controls; synthetic process/custody/clock, real filesystem and actual collector; no native qualification" }, null, 2));
}

let sequence = 0;
const variants: Record<string, string | null> = {
  missing: null, empty: "", valid: '{"parent":710003,"pid":710004}\n', malformed: '{"parent":710003,"pid":',
  null: "null\n", partial: '{"parent":710003}\n', array: "[]\n", boolean: "false\n", negative: '{"parent":-1,"pid":710004}\n',
};
type Kind = "hostCase" | "rawCase" | "invalidBudgetFrame";
// Whole-caller lifecycle controls below use real local streams and asynchronous
// dispatch; these older protocol controls intentionally retain authored clocks.
type Options = { variant?: string; primary?: unknown; bodySuccess?: boolean; setupFailure?: string; releaseFailure?: string; reportFailure?: boolean; liveDescendant?: boolean; rootFailure?: boolean; hard?: boolean; listenerFailure?: string; timerFailure?: boolean; mutateAtRelease?: string; readFailureAtRelease?: boolean; rawStage?: string; rawFault?: string };
async function exercise(kind: Kind, options: Options = {}) {
  const area = fs.mkdtempSync(path.join(tmpdir(), "ea-cleanup-control-"));
  const evidence = output ? path.join(output, String(++sequence).padStart(3, "0")) : path.join(area, "evidence");
  fs.mkdirSync(evidence, { recursive: true });
  const receiptDir = path.join(area, "receipts");
  const saved = { flag: process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS, dir: process.env.QMD_RECOVERY_BOOTSTRAP_DIR };
  process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS = "1"; process.env.QMD_RECOVERY_BOOTSTRAP_DIR = receiptDir;
  const primary = Object.hasOwn(options, "primary") ? options.primary : Object.freeze({ primary: kind });
  const releaseError = Object.freeze({ secondaryRelease: kind });
  const reportError = Object.freeze({ secondaryReport: kind });
  const events: string[] = [], owners: any[] = [], handles: any[] = [], generated: any[] = [];
  const timers = new Set<number>();
  let fixtureRoot: string | undefined, spawns = 0, checks = 0, receipt: any, now = 1000, cleanupReadFailure = false;
  const fail = (label: string) => { events.push(label); if (options.setupFailure === label) throw primary; };
  class SyntheticHandle extends EventEmitter {
    pid: number; exitCode: number | null; signalCode: string | null = null;
    // Faithful local-stream adapter: terminal close must close each endpoint,
    // not just emit an impossible process-close with open authored pipes.
    stdout = new PassThrough(); stderr = new PassThrough(); stdin: any;
    constructor(public label: string, pid: number) {
      super(); this.pid = pid; this.exitCode = label === "sentinel" ? null : 0;
      const handleOn = this.on;
      this.on = function(event: string, listener: (...args: any[]) => void) {
        handleOn.call(this, event, listener);
        if (options.setupFailure === label + "-" + event + "-observer") throw primary;
        return this;
      };
      const stream = new PassThrough() as any;
      stream.write = (data: unknown) => { events.push(label + "-write-" + String(data).trim()); };
      stream.end = () => { events.push(label + "-end"); if (options.releaseFailure === label + "-end") throw releaseError; if (this.exitCode === null) { this.exitCode = 0; this.emit("exit", 0, null); this.finish(); } };
      this.stdin = stream;
      // Each actual endpoint event, not its early closed flag, gates process close.
      for (const endpoint of [stream, this.stdout, this.stderr]) endpoint.once("close", () => { this.closedPipes.add(endpoint); this.finish(); });
      for (const [name, target] of [["stdout", this.stdout], ["stderr", this.stderr]] as const) {
        const original = target.on;
        target.on = function(event: string, listener: (...args: any[]) => void) {
          events.push(label + "-" + name + "-listener");
          original.call(this, event, listener);
          if (event === "data" && options.setupFailure === label + "-" + name + "-listener") throw primary;
          return this;
        };
        const remove = target.removeListener;
        target.removeListener = function(event: string, listener: (...args: any[]) => void) {
          if (event === "data" && options.listenerFailure === label + "-" + name) { events.push("listener-removal-refused"); throw releaseError; }
          return remove.call(this, event, listener);
        };
      }
    }
    private terminal = false;
    private readonly closedPipes = new Set<unknown>();
    finish() { this.stdin.destroy(); this.stdout.destroy(); this.stderr.destroy(); if (this.closedPipes.size === 3 && !this.terminal) { this.terminal = true; this.emit("close", this.exitCode, this.signalCode); } }
    kill(signal: string) {
      events.push("synthetic-handle-kill-" + this.label + "-" + signal);
      if (options.releaseFailure === this.label) throw releaseError;
      this.signalCode = signal; this.emit("exit", null, signal); this.finish(); return true;
    }
  }
  const processBinding = { env: { QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS: "1", BUN_EXECUTABLE: "/synthetic/bun" }, execPath: "/synthetic/node", versions: { node: "synthetic" }, kill() { events.push("FORBIDDEN-NUMERIC-SIGNAL"); throw Error("no numeric signal authority"); } };
  const logger = { log(value: string) {
    const data = JSON.parse(value); events.push(data.evidence);
    if (data.evidence.endsWith("before-finally") && kind !== "hostCase" && !options.bodySuccess) throw primary;
    if (data.evidence.endsWith("-finally") && options.reportFailure) throw reportError;
  } };
  const observation = new Function("process", "console", compile(functions.observeStageFailure!) + ";return observeStageFailure;")(processBinding, logger);
  let realReap: any, realInstrument: any, realCleanEnv: any;
  const bindings: Record<string, any> = { ...helpers, path, tmpdir, Buffer, expect, runner: "file:///synthetic/runner.ts", source: "/synthetic/source.py", loader: "/synthetic/loader.mjs", process: processBinding, console: logger,
    mkdtempSync(prefix: string) { fail("root-create"); fixtureRoot = fs.mkdtempSync(prefix); events.push("fixture-created"); return fixtureRoot; },
    mkdirSync: fs.mkdirSync,
    cleanEnv(root: string) { fail("clean-env"); return realCleanEnv(root); },
    instrument(root: string, ...args: any[]) {
      fail("instrument"); const wrapper = realInstrument(root, ...args);
      fs.writeFileSync(path.join(root, "started.json"), '{"executable":"/synthetic/host","versions":{"node":"synthetic"}}');
      const metadata = variants[options.variant ?? "valid"];
      if (metadata !== null) fs.writeFileSync(path.join(root, "forks.jsonl"), metadata!);
      fs.writeFileSync(path.join(root, "reaps.jsonl"), '{"pid":710004}\n');
      if (options.bodySuccess) {
        fs.writeFileSync(path.join(root, "stage.json"), '{"stage":"supervisor-consumed"}');
        if (kind !== "invalidBudgetFrame" && !options.hard) fs.writeFileSync(path.join(root, "work"), "actual-user-work");
        if (kind === "invalidBudgetFrame") fs.rmSync(path.join(root, "forks.jsonl"), { force: true });
        fs.writeFileSync(path.join(root, "helper.json"), JSON.stringify({ elapsedMs: 1, guardFired: false, hooksBefore: [0], hooksAfter: [0], result: { launch: { state: "started", admissionGranted: true, supervisorReaped: true }, cleanup: { childReaped: true, groupTerminated: true, sessionTerminated: true, pipesClosed: true, forcedPipeClose: false, containmentVerified: false } } }));
        if (options.hard) fs.rmSync(path.join(root, "helper.json"));
      }
      return wrapper;
    },
    writeFileSync(file: string, data: any) {
      fail(file.endsWith("host.ts") ? "template-write" : "instrument-template-write");
      fs.writeFileSync(file, data);
      generated.push({ name: path.basename(file), bytes: Buffer.byteLength(data), sha256: digest(data), source: "actual extracted admission declaration; generated but NEVER executed" });
      fs.writeFileSync(path.join(evidence, path.basename(file) + ".generated-source"), data);
    },
    spawn() {
      const label = ++spawns === 1 ? "sentinel" : kind === "hostCase" ? "host" : "controller";
      fail("spawn-" + label); const child = new SyntheticHandle(label, label === "sentinel" ? 710001 : 710002); handles.push(child);
      if (label !== "sentinel" && kind !== "hostCase") queueMicrotask(() => {
        child.stdout.emit("data", Buffer.from(JSON.stringify({ type: "prepared" }) + "\n"));
        if (kind === "rawCase") child.stdout.emit("data", Buffer.from(JSON.stringify({ type: "ready", pid: 710004 }) + "\n" + JSON.stringify({ type: "cleanup", childReaped: true, sessionTerminated: true, groupTerminated: true, containmentVerified: false }) + "\n"));
        child.finish();
      });
      return child;
    },
    read: (p: string) => { if (cleanupReadFailure && p.endsWith("/forks.jsonl")) throw releaseError; return fs.existsSync(p) ? fs.readFileSync(p, "utf8") : null; }, existsSync: fs.existsSync,
    performance: { now: () => now }, sleep: async () => { now += 2000; },
    until: async (predicate: () => boolean) => { checks++; if (checks === 2 && !options.bodySuccess) { events.push("required-stage-failure"); throw primary; } expect(predicate()).toBe(true); events.push(checks === 2 ? "required-stage-passed" : "bootstrap-passed"); },
    alive(pid: number) { events.push("synthetic-alive-" + pid); return pid === 710001 || !!options.liveDescendant; },
    reap: async (child: any, killed: number[], observation: any) => {
      events.push("release-" + child.label);
      if (child.label === "host") {
        if (options.mutateAtRelease) fs.writeFileSync(path.join(fixtureRoot!, "forks.jsonl"), variants[options.mutateAtRelease]!);
        if (options.readFailureAtRelease) cleanupReadFailure = true;
      }
      if (options.releaseFailure === child.label) throw releaseError;
      return realReap(child, killed, observation);
    },
    observeStageFailure: observation,
    captureRecoveryExecutionFailure(context: any) { events.push("real-collector-entry"); receipt = captureRecoveryExecutionFailure(context); events.push("real-collector-return"); },
    rmSync(p: string, opts: any) { events.push("real-fixture-rm"); if (options.rootFailure) throw releaseError; fs.rmSync(p, opts); },
    setTimeout() { events.push("timer-start"); timers.add(1); return 1; }, clearTimeout(id: number) { events.push("timer-clear"); if (options.timerFailure) throw releaseError; timers.delete(id); },
    setInterval() { events.push("interval-start"); const id = timers.size + 2; timers.add(id); return id; }, clearInterval(id: number) { events.push("interval-clear"); timers.delete(id); },
  };
  if (helpers.RecoveryExecutionCleanup) bindings.RecoveryExecutionCleanup = class extends helpers.RecoveryExecutionCleanup {
    constructor() { super(); owners.push(this); }
    defer(label: string, release: () => unknown) {
      // Inject a release refusal at the actual owner's action boundary. All
      // non-faulted actions execute the real helper; no handwritten finally.
      super.defer(label, () => {
        if (label === "controller") { events.push("release-controller"); if (options.releaseFailure === "controller") throw releaseError; }
        return release();
      });
    }
  };

  realReap = new Function(...Object.keys(bindings), compile(functions.reap!) + ";return reap;")(...Object.values(bindings));
  realInstrument = new Function(...Object.keys(bindings), compile(functions.instrument!) + ";return instrument;")(...Object.values(bindings));
  realCleanEnv = new Function(...Object.keys(bindings), compile(functions.cleanEnv!) + ";return cleanEnv;")(...Object.values(bindings));
  const whole = new Function(...Object.keys(bindings), compile(functions[kind]!) + ";return " + kind + ";")(...Object.values(bindings));
  let thrown: unknown, threw = false;
  try {
    try { await (kind === "hostCase" ? whole("node", "supervisor-consumed", "delay", options.hard ? "hard" : options.bodySuccess ? "valid" : "alive") : kind === "rawCase" ? whole(options.rawStage ?? "none", "delay", "open", options.rawFault) : whole("Sbad\nK")); }
    catch (error) { threw = true; thrown = error; }
    const snapshot = {
      kind, options, scope: "source-extracted WHOLE case, actual cleanEnv/instrument/template generation and real cleanup helper; synthetic process/reap/alive/clock and authored sidecars; real owned FS and collector; generated subjects NEVER executed; NO native/custody proof",
      threw, primaryRetained: threw && Object.is(thrown, primary), thrownName: (thrown as any)?.name ?? null,
      events: [...events], fixtureExists: fixtureRoot !== undefined && fs.existsSync(fixtureRoot), fixtureRoot,
      fixtureFiles: fixtureRoot && fs.existsSync(fixtureRoot) ? fs.readdirSync(fixtureRoot).sort() : [],
      syntheticTimersRemaining: [...timers],
      generated,
      listenerCounts: handles.map(h => ({ label: h.label, close: h.listenerCount("close"), exit: h.listenerCount("exit"), stdout: h.stdout.listenerCount("data"), stderr: h.stderr.listenerCount("data"), stdin: h.stdin.listenerCount("error") })),
      cleanup: owners.map(o => o.diagnostics()), receipt,
    };
    fs.writeFileSync(path.join(evidence, "before-independent-cleanup.json"), JSON.stringify(snapshot, null, 2));
    if (fs.existsSync(receiptDir)) for (const name of fs.readdirSync(receiptDir)) fs.copyFileSync(path.join(receiptDir, name), path.join(evidence, name));
    return { ...snapshot, thrown, primary, releaseError, reportError, owners };
  } finally {
    if (fixtureRoot && fs.existsSync(fixtureRoot)) fs.rmSync(fixtureRoot, { recursive: true, force: true });
    if (output) fs.writeFileSync(path.join(evidence, "independent-cleanup.json"), JSON.stringify({ fixtureExistsAfter: fixtureRoot ? fs.existsSync(fixtureRoot) : false, authority: "call-owned filesystem only; NO PID action" }));
    fs.rmSync(area, { recursive: true, force: true });
    if (saved.flag === undefined) delete process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS; else process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS = saved.flag;
    if (saved.dir === undefined) delete process.env.QMD_RECOVERY_BOOTSTRAP_DIR; else process.env.QMD_RECOVERY_BOOTSTRAP_DIR = saved.dir;
  }
}
function released(result: Awaited<ReturnType<typeof exercise>>, labels: string[]) {
  expect(result.fixtureExists).toBe(false);
  for (const label of labels) expect(result.events).toContain("release-" + label);
  expect(result.events).toContain("real-fixture-rm");
  expect(result.events).not.toContain("FORBIDDEN-NUMERIC-SIGNAL");
}
describe("source-bound fixture lifecycle (NO native subjects)", () => {
  for (const variant of Object.keys(variants)) it(`hostCase preserves frozen primary and releases every acquired owner with ${variant} metadata`, async () => {
    const result = await exercise("hostCase", { variant });
    expect(result.primaryRetained).toBe(true); released(result, ["host", "sentinel"]);
    expect(result.receipt.status).toBe("captured");
    expect(result.events.indexOf("real-collector-return")).toBeLessThan(result.events.indexOf("release-host"));
    expect(result.cleanup[0].notes.some((n: any) => n.qualification === "unqualified")).toBe(true);
    const outcome = result.owners[0].outcomes.find((o: any) => o.label === "recovery-descendants");
    expect(outcome.status).toBe(["missing", "empty", "valid"].includes(variant) ? "completed" : "failed");
  });
  for (const primary of [undefined, null, 0, false, ""]) it(`hostCase retains exact falsy primary ${String(primary)}`, async () => {
    const result = await exercise("hostCase", { variant: "malformed", primary, releaseFailure: "host", reportFailure: true });
    expect(result.threw).toBe(true); expect(Object.is(result.thrown, primary)).toBe(true); released(result, ["host", "sentinel"]);
    expect(result.owners[0].outcomes.some((o: any) => o.error === result.releaseError)).toBe(true);
    expect(result.owners[0].outcomes.some((o: any) => o.error === result.reportError)).toBe(true);
  });
  for (const setupFailure of ["root-create", "clean-env", "instrument", "instrument-template-write", "template-write", "spawn-sentinel", "spawn-host", "sentinel-error-observer", "host-error-observer", "host-stdout-listener", "host-stderr-listener"]) it(`hostCase owns setup from ${setupFailure}`, async () => {
    const result = await exercise("hostCase", { setupFailure }); expect(result.primaryRetained).toBe(true);
    expect(result.fixtureExists).toBe(false);
    if (!["root-create", "clean-env", "instrument", "instrument-template-write", "template-write", "spawn-sentinel"].includes(setupFailure)) expect(result.events).toContain("release-sentinel");
    if (setupFailure.startsWith("host-")) {
      expect(result.events).toContain("release-host"); expect(result.listenerCounts.every(h => !h.stdout && !h.stderr && !h.exit && !h.close)).toBe(true);
    }
  });
  for (const kind of ["rawCase", "invalidBudgetFrame"] as const) {
    for (const setupFailure of ["clean-env", "instrument", "instrument-template-write", "spawn-controller", "controller-stdout-listener"]) it(`${kind} independently releases after ${setupFailure}`, async () => {
      const result = await exercise(kind, { setupFailure }); expect(result.primaryRetained).toBe(true); expect(result.fixtureExists).toBe(false);
      if (setupFailure === "spawn-controller" || setupFailure === "controller-stdout-listener") expect(result.events).toContain("release-sentinel");
      if (setupFailure === "controller-stdout-listener") expect(result.events).toContain("release-controller");
    });
    it(`${kind} preserves body primary after controller release and reporter failure`, async () => {
      const result = await exercise(kind, { releaseFailure: "controller", reportFailure: true }); expect(result.primaryRetained).toBe(true); released(result, ["controller", "sentinel"]);
    });
    it(`${kind} success keeps original functional assertions and no controller SIGKILL`, async () => {
      const result = await exercise(kind, { bodySuccess: true }); expect(result.threw).toBe(false); released(result, ["controller", "sentinel"]);
      expect(result.events).not.toContain("synthetic-handle-kill-controller-SIGKILL");
      expect(result.listenerCounts.every(h => !h.stdout && !h.stderr && !h.stdin && !h.close && !h.exit)).toBe(true);
    });
  }
  it("hostCase success keeps original functional assertions and is not a descendant certification", async () => {
    const result = await exercise("hostCase", { bodySuccess: true }); expect(result.threw).toBe(false); released(result, ["host", "sentinel"]);
    expect(result.cleanup[0].notes.some((n: any) => n.qualification === "unqualified")).toBe(true);
  });
  it("cleanup-only release/report failures surface rather than turning success green", async () => {
    const result = await exercise("hostCase", { bodySuccess: true, releaseFailure: "sentinel", reportFailure: true }); expect(result.threw).toBe(true); released(result, ["host", "sentinel"]);
    expect(result.owners[0].outcomes.filter((o: any) => o.status === "failed").length).toBeGreaterThanOrEqual(2);
  });
  it("failed scratch removal is explicit before independent filesystem cleanup", async () => {
    const result = await exercise("hostCase", { rootFailure: true }); expect(result.primaryRetained).toBe(true); expect(result.fixtureExists).toBe(true);
    expect(result.owners[0].outcomes.some((o: any) => o.label === "scratch" && o.status === "failed")).toBe(true);
  });
  it("live sidecar descendants remain nonqualified and are never numeric cleanup authority", async () => {
    const result = await exercise("hostCase", { liveDescendant: true }); expect(result.primaryRetained).toBe(true); released(result, ["host", "sentinel"]);
    expect(result.owners[0].outcomes.some((o: any) => o.label === "recovery-descendants" && o.status === "failed")).toBe(true);
  });
  it("frozen object primary survives simultaneous parser, release and logger failures", async () => {
    const result = await exercise("hostCase", { variant: "malformed", releaseFailure: "host", reportFailure: true });
    expect(result.primaryRetained).toBe(true); released(result, ["host", "sentinel"]);
    expect(Object.isFrozen(result.primary)).toBe(true); expect(Object.keys(result.primary as object)).toEqual(["primary"]);
    expect(result.owners[0].outcomes.filter((o: any) => o.status === "failed")).toHaveLength(3);
  });
  for (const failure of ["parse", "read"]) it(`cleanup-only metadata ${failure} failure surfaces after a successful whole host body`, async () => {
    const result = await exercise("hostCase", { bodySuccess: true, ...(failure === "parse" ? { mutateAtRelease: "malformed" } : { readFailureAtRelease: true }) });
    expect(result.threw).toBe(true); expect(result.thrownName).toBe("RecoveryExecutionCleanupError"); released(result, ["host", "sentinel"]);
    expect(result.owners[0].outcomes.some((o: any) => o.label === "recovery-descendants" && o.status === "failed")).toBe(true);
  });
  it("listener removal refusal is not deallocation and cannot skip host/sentinel/root", async () => {
    const result = await exercise("hostCase", { listenerFailure: "host-stdout" }); expect(result.primaryRetained).toBe(true); released(result, ["host", "sentinel"]);
    expect(result.listenerCounts.find(h => h.label === "host")!.stdout).toBe(1);
    expect(result.owners[0].outcomes.some((o: any) => o.label === "listener-data" && o.status === "failed")).toBe(true);
  });
  it("both raw intervals and guard are independently cleared after a body failure", async () => {
    const result = await exercise("rawCase", { rawStage: "child-validation", rawFault: "cancel-read-hold" });
    expect(result.primaryRetained).toBe(true); released(result, ["controller", "sentinel"]);
    expect(result.events.filter(e => e === "interval-clear")).toHaveLength(2); expect(result.syntheticTimersRemaining).toEqual([]);
  });
  it("guard-timer cleanup refusal preserves primary and attempts remaining raw releases", async () => {
    const result = await exercise("rawCase", { timerFailure: true });
    // The first clear occurs in the body, so this is the exact body primary.
    expect(result.thrown).toBe(result.releaseError); released(result, ["controller", "sentinel"]);
    expect(result.syntheticTimersRemaining).toEqual([1]);
    expect(result.owners[0].outcomes.some((o: any) => o.label === "guard-timer" && o.status === "failed")).toBe(true);
  });
  it("hard-parent-death action stays after a passed required-stage gate", async () => {
    const failed = await exercise("hostCase", { hard: true }); expect(failed.primaryRetained).toBe(true);
    expect(failed.events).not.toContain("synthetic-handle-kill-host-SIGKILL");
    const passed = await exercise("hostCase", { bodySuccess: true, hard: true }); expect(passed.threw).toBe(false);
    expect(passed.events.indexOf("required-stage-passed")).toBeLessThan(passed.events.indexOf("synthetic-handle-kill-host-SIGKILL")); released(passed, ["host", "sentinel"]);
  });
});

describe("actual non-registering cleanup helper (synthetic handles only)", () => {
  it("restoration and all releases run independently, retaining secondary identities", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(); const primary = Object.freeze({ primary: true }), secondary = Object.freeze({ cleanup: true });
    const events: string[] = []; let restored = false, thrown: unknown;
    owner.defer("restore", () => { restored = true; events.push("restore"); });
    owner.defer("release-a", () => { events.push("a"); throw secondary; });
    owner.defer("release-b", () => { events.push("b"); throw null; });
    owner.report = () => { events.push("report"); throw undefined; };
    try { await owner.run(async () => { throw primary; }); } catch (error) { thrown = error; }
    expect(thrown).toBe(primary); expect(restored).toBe(true); expect(events).toEqual(["b", "a", "restore", "report"]);
    expect(owner.outcomes.filter((o: any) => o.status === "failed").map((o: any) => o.error)).toEqual([null, secondary, undefined]);
  });
  it("cleanup-only reporting failure throws an aggregate that retains actual outcomes", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(), error = Object.freeze({ report: true }); owner.report = () => { throw error; };
    let thrown: any; try { await owner.run(async () => undefined); } catch (value) { thrown = value; }
    expect(thrown).toBeInstanceOf(helpers.RecoveryExecutionCleanupError); expect(thrown.outcomes).toBe(owner.outcomes); expect(owner.outcomes[0].error).toBe(error);
  });
  for (const failure of ["refusal", "timeout"]) it(`direct owned-handle ${failure} is explicit, with no surviving release listeners`, async () => {
    const child = Object.assign(new EventEmitter(), { pid: 710020, exitCode: null, signalCode: null, kill() { return failure !== "refusal"; } });
    let failed = false; try { await helpers.releaseOwnedProcess(child, [], 5); } catch { failed = true; }
    // Unknown direct resources retain containment; an actual terminal event,
    // not expiry of a release attempt, authorizes the original zero baseline.
    expect(child.listenerCount("error")).toBeGreaterThan(0); child.emit("close");
    expect(failed).toBe(true); expect(child.exitCode).toBeNull(); expect(child.listenerCount("exit")).toBe(0); expect(child.listenerCount("error")).toBe(0);
  });
  it("direct owned handle exit settles and removes listeners without any numeric binding", async () => {
    const child = Object.assign(new EventEmitter(), { pid: 710021, exitCode: null, signalCode: null, kill(this: EventEmitter) { this.emit("exit", null, "SIGKILL"); this.emit("close"); return true; } });
    const killed: number[] = []; await helpers.releaseOwnedProcess(child, killed, 5); expect(killed).toEqual([710021]); expect(child.eventNames()).toEqual([]);
  });
  for (const failure of ["end-refusal", "timeout", "none"]) it(`controller cooperative ${failure} never invokes kill and does not certify descendants`, async () => {
    let closeObserved = false, resolve!: () => void, killCalls = 0;
    const drained = new Promise<void>(done => { resolve = done; }); const refusal = Object.freeze({ eofRefused: true });
    // Preserve this bare-Emitter contract without confusing a real stream's
    // unrelated internal prefinish hook with an owner listener. Model its
    // local terminal boundary explicitly; no native lifetime is asserted.
    const stdin = Object.assign(new EventEmitter(), { closed: false, destroy(this: EventEmitter & { closed: boolean }) { this.closed = true; this.emit("close"); }, end(this: EventEmitter & { closed: boolean; destroy(): void }) { if (failure === "end-refusal") throw refusal; if (failure === "none") { closeObserved = true; this.destroy(); resolve(); } return this; } });
    const child = { stdin, kill() { killCalls++; throw Error("controller kill forbidden"); } };
    let thrown: unknown, failed = false; try { await helpers.releaseRecoveryController(child, { drained, isClosed: () => closeObserved }, 5); } catch (error) { failed = true; thrown = error; }
    if (failure !== "none") expect(stdin.listenerCount("error")).toBeGreaterThan(0);
    stdin.destroy(); await localTick();
    expect(killCalls).toBe(0); expect(failed).toBe(failure !== "none"); expect(closeObserved).toBe(failure === "none"); expect(stdin.eventNames()).toEqual([]);
    if (failure === "end-refusal") expect(thrown).toBe(refusal);
  });
  it("controller guard refusal rejects the body, then EOF still gets a bounded drain attempt", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(); const primary = Object.freeze({ guardExpired: true }); let ended = false, kills = 0;
    const child = Object.assign(new EventEmitter(), { stdin: Object.assign(new PassThrough(), { end(this: PassThrough) { ended = true; this.destroy(); child.emit("close"); return this; } }), kill() { kills++; return false; } });
    let thrown: unknown;
    try { await owner.run(async () => { const observed = owner.controller(child); observed.refuse(primary); await observed.closed; }); } catch (error) { thrown = error; }
    expect(thrown).toBe(primary); expect(ended).toBe(true); expect(kills).toBe(0); expect(child.eventNames()).toEqual([]);
  });
  it("controller observer attachment failure still releases the acquired handle", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(), primary = Object.freeze({ observer: true }); let ended = false;
    const child = Object.assign(new EventEmitter(), { stdin: Object.assign(new PassThrough(), { end(this: PassThrough) { ended = true; this.destroy(); child.emit("close"); return this; } }) });
    const on = child.on; child.on = function(event: string, listener: (...args: any[]) => void) { on.call(this, event, listener); if (event === "close") throw primary; return this; };
    let thrown: unknown; try { await owner.run(async () => { owner.controller(child); }); } catch (error) { thrown = error; }
    expect(thrown).toBe(primary); expect(ended).toBe(true); expect(child.eventNames()).toEqual([]);
  });
});

// Real LOCAL object lifetimes, not actual subprocesses, OS pipes or descendants.
const localTick = () => new Promise<void>(resolve => setImmediate(resolve));
const recordLocal = (name: string, value: unknown) => { if (output) fs.writeFileSync(path.join(output, "life-helper-" + name + ".json"), JSON.stringify(value, null, 2)); };
function localChild(label: string, exited = false) {
  const child: any = Object.assign(new EventEmitter(), { label, pid: label === "sentinel" ? 720001 : 720002,
    exitCode: exited ? 0 : null, signalCode: null, killCalls: 0, endCalls: 0, terminal: false,
    stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough(), fixtureObservers: [] });
  const closed = new Set<unknown>(); let closeRequested = false;
  const finishClose = () => { if (closeRequested && closed.size === 3 && !child.terminal) { child.terminal = true; child.emit("close"); } };
  child.terminalClose = () => { closeRequested = true; for (const stream of [child.stdin, child.stdout, child.stderr]) stream.destroy(); finishClose(); };
  for (const stream of [child.stdin, child.stdout, child.stderr]) stream.once("close", () => { closed.add(stream); finishClose(); });
  child.kill = (signal: string) => { child.killCalls++; child.signalCode = signal; child.emit("exit", null, signal); child.terminalClose(); return true; };
  const end = child.stdin.end;
  child.stdin.end = function(...args: any[]) { child.endCalls++; end.apply(this, args); queueMicrotask(() => { child.exitCode = 0; child.emit("exit", 0, null); child.terminalClose(); }); return this; };
  return child;
}
type LifecycleFault = "parse" | "stage-read" | "cancel-write" | "host-exit-before" | "host-exit-after" | "stdin-before" | "stdin-after" | "exited-pipes" | "host-callback" | "host-drain" | "frame-callback";
async function lifecycleWhole(fault: LifecycleFault, value?: unknown) {
  if (arguments.length < 2) value = Object.freeze({ callbackPrimary: true });
  const area = fs.mkdtempSync(path.join(tmpdir(), "ea-lifetime-whole-"));
  const dir = output ? path.join(output, "life-" + String(++sequence).padStart(3, "0")) : path.join(area, "evidence"); fs.mkdirSync(dir, { recursive: true });
  const children: any[] = [], owners: any[] = [], escaped: unknown[] = [], logs: any[] = [];
  const secondary = Object.freeze({ observation: fault });
  let root: string | undefined, checks = 0, observed: unknown = value, threw = false, thrown: unknown, finished = false, callbacksExecuted = 0, rootDuringStartedWork: boolean | undefined;
  const kind = fault.startsWith("host-") || fault === "exited-pipes" ? "hostCase" : fault.startsWith("stdin-") || fault === "frame-callback" ? "invalidBudgetFrame" : "rawCase";
  const timers = new Set<ReturnType<typeof setTimeout>>(), intervals = new Set<ReturnType<typeof setInterval>>();
  const processBinding = { env: { PATH: process.env.PATH }, execPath: "/authored/no-executable", versions: {}, kill() { throw Error("numeric authority forbidden"); } };
  const jsonBinding = { ...JSON, parse(text: string) { try { return JSON.parse(text); } catch (error) { observed = error; throw error; } }, stringify: JSON.stringify };
  const bindings: Record<string, any> = { ...helpers, path, Buffer, JSON: jsonBinding, expect, process: processBinding, performance,
    console: { log(text: string) { logs.push(JSON.parse(text)); } }, tmpdir: () => area,
    source: "/authored/source.py", runner: "file:///authored/runner.ts", loader: "/authored/loader.mjs",
    mkdtempSync(prefix: string) { root = fs.mkdtempSync(prefix); return root; }, mkdirSync: fs.mkdirSync, writeFileSync: fs.writeFileSync, existsSync: fs.existsSync, rmSync: fs.rmSync,
    read(file: string) { try { return fs.existsSync(file) ? fs.readFileSync(file, "utf8") : null; } catch (error) { observed = error; throw error; } },
    sleep: (ms: number) => new Promise(resolve => setTimeout(resolve, ms)),
    until: async (predicate: () => boolean) => {
      checks++; if (checks === 2) throw value; expect(predicate()).toBe(true);
      if (fault === "host-callback") await localTick();
      if (fault === "host-drain") await new Promise<void>(resolve => setTimeout(() => {
        rootDuringStartedWork = !!root && fs.existsSync(root);
        if (rootDuringStartedWork) fs.writeFileSync(path.join(root!, "started-work-drained"), "done");
        resolve();
      }, 20));
    },
    alive: (pid: number) => pid === 720001,
    captureRecoveryExecutionFailure() { throw Error("not a canonical/collector control"); },
    reap: helpers.releaseOwnedProcess,
    spawn() {
      const label = children.length === 0 ? "sentinel" : kind === "hostCase" ? "host" : "controller";
      const child = localChild(label, fault === "exited-pipes" && label === "host"); children.push(child);
      const hook = (target: EventEmitter, listener: (event: string) => void) => { target.on("newListener", listener); child.fixtureObservers.push({ target, listener }); };
      if (label === "host" && fault.startsWith("host-exit-")) {
        if (fault.endsWith("before")) hook(child, (event: string) => { if (event === "exit") throw secondary; });
        else { const on = child.on; child.on = function(event: string, listener: any) { on.call(this, event, listener); if (event === "exit") throw secondary; return this; }; }
      }
      if (label === "controller" && fault.startsWith("stdin-")) {
        // Acquisition now precedes data attachment: keep this control's earlier
        // setup primary before the separate stdin-observation refusal.
        hook(child, (event: string) => { if (event === "error") throw value; });
        hook(child.stdout, (event: string) => { if (event === "data") throw value; });
        if (fault.endsWith("before")) hook(child.stdin, (event: string) => { if (event === "error") throw secondary; });
        else { const on = child.stdin.on; child.stdin.on = function(event: string, listener: any) { on.call(this, event, listener); if (event === "error") throw secondary; return this; }; }
      }
      if (label === "controller" && fault === "cancel-write") child.stdin.write = (data: any) => { if (data === "K") throw value; return true; };
      if (label === "controller" && fault === "frame-callback") {
        const end = child.stdin.end; child.stdin.end = function(frame: any) { if (frame !== undefined) { child.endCalls++; throw value; } return end.call(this); };
        setImmediate(() => { callbacksExecuted++; try { child.stdout.emit("data", Buffer.from('{"type":"prepared"}\n')); } catch (error) { escaped.push(error); } });
      }
      if ((label === "controller" && fault === "parse") || (label === "host" && (fault === "host-callback" || fault === "host-drain"))) setImmediate(() => {
        callbacksExecuted++;
        try { child.stdout.emit("data", fault === "parse" ? Buffer.from("CORRUPT_JSON\n") : { toString() { throw value; } }); } catch (error) { escaped.push(error); }
      });
      return child;
    },
    setTimeout(callback: () => void, ms: number) { const timer = setTimeout(() => { timers.delete(timer); callback(); }, ms); timers.add(timer); return timer; },
    clearTimeout(timer: ReturnType<typeof setTimeout>) { timers.delete(timer); clearTimeout(timer); },
    setInterval(callback: () => void, ms: number) { const timer = setInterval(() => { try { callback(); } catch (error) { escaped.push(error); clearInterval(timer); intervals.delete(timer); } }, ms); intervals.add(timer); return timer; },
    clearInterval(timer: ReturnType<typeof setInterval>) { intervals.delete(timer); clearInterval(timer); },
  };
  bindings.RecoveryExecutionCleanup = class extends helpers.RecoveryExecutionCleanup { constructor() { super(); owners.push(this); } };
  // Only fixed declaration names from the actual source AST become executable
  // text. Fault values are bindings, never interpolated into Function source.
  const make = (name: string) => new Function(...Object.keys(bindings), compile(functions[name]!) + ";return " + name + ";")(...Object.values(bindings));
  bindings.cleanEnv = make("cleanEnv"); bindings.observeStageFailure = make("observeStageFailure");
  bindings.instrument = (...args: any[]) => { const result = make("instrument")(...args); if (kind === "hostCase") fs.writeFileSync(path.join(root!, "started.json"), "{}"); if (fault === "stage-read") fs.mkdirSync(path.join(root!, "stage.json")); if (fault === "cancel-write") fs.writeFileSync(path.join(root!, "gate-forwarded.json"), "{}"); return result; };
  const whole = make(kind);
  const started = performance.now();
  try {
    try { await (kind === "hostCase" ? whole("node", "supervisor-consumed", "delay", "alive") : kind === "invalidBudgetFrame" ? whole("Sbad\nK") : whole(fault === "stage-read" ? "child-validation" : "supervisor-consumed", "delay", "eof", fault === "cancel-write" ? "cancel-read-hold" : undefined)); }
    catch (error) { threw = true; thrown = error; }
    finished = true;
    const state = { fault, finished, threw, callbacksExecuted, rootDuringStartedWork, primaryExact: threw && Object.is(thrown, observed), elapsedMs: performance.now() - started,
      escaped: escaped.map(error => String(error)), rootExists: !!root && fs.existsSync(root), activeTimers: timers.size, activeIntervals: intervals.size,
      children: children.map(c => ({ label: c.label, killCalls: c.killCalls, endCalls: c.endCalls, terminal: c.terminal, stdoutDisposed: c.stdout.destroyed, stderrDisposed: c.stderr.destroyed })),
      cleanup: owners.map(o => o.diagnostics()), logs };
    fs.writeFileSync(path.join(dir, "before-independent-cleanup.json"), JSON.stringify(state, null, 2));
    return { ...state, thrown, observed, value, secondary, owners, escaped };
  } finally {
    for (const timer of timers) clearTimeout(timer); for (const interval of intervals) clearInterval(interval);
    // Explicit independent fixture disposal AFTER the serialized assertions.
    for (const child of children) { for (const hook of child.fixtureObservers) hook.target.removeListener("newListener", hook.listener); child.terminalClose(); }
    fs.rmSync(area, { recursive: true, force: true });
  }
}
describe("whole-source asynchronous lifecycle regressions (NO native subjects)", () => {
  // Check escaped.length as well as contents: native Bun's loose toEqual can
  // equate [undefined] with [], which is not safe for thrown-value presence.
  for (const fault of ["parse", "stage-read", "cancel-write"] as const) it(`rawCase executes and promptly propagates ${fault} callback failure`, async () => {
    const result = await lifecycleWhole(fault);
    expect(result.threw).toBe(true); expect(result.thrown).toBe(result.observed); expect(result.escaped).toEqual([]); expect(result.escaped.length).toBe(0);
    expect(result.elapsedMs).toBeLessThan(1000); expect(result.rootExists).toBe(false); expect(result.activeTimers).toBe(0); expect(result.activeIntervals).toBe(0);
    expect(result.children.find(c => c.label === "controller")!.endCalls).toBe(1);
    if (fault === "parse") expect(result.thrown).toBeInstanceOf(SyntaxError);
    if (fault === "stage-read") expect((result.thrown as any).code).toBe("EISDIR");
  });
  for (const value of [undefined, null, 0, false, "", Object.freeze({ callback: true })]) it(`hostCase awaits callback cancellation with exact ${String(value)} primary`, async () => {
    const result = await lifecycleWhole("host-callback", value);
    expect(result.threw).toBe(true); expect(Object.is(result.thrown, value)).toBe(true); expect(result.escaped).toEqual([]); expect(result.escaped.length).toBe(0); expect(result.rootExists).toBe(false); expect(result.callbacksExecuted).toBe(1);
  });
  for (const fault of ["host-exit-before", "host-exit-after", "stdin-before", "stdin-after"] as const) it(`whole caller independently releases despite ${fault} observation refusal`, async () => {
    const result = await lifecycleWhole(fault);
    expect(result.thrown).toBe(result.value); expect(result.rootExists).toBe(false); expect(result.children[0]!.killCalls).toBe(1);
    const target = result.children[1]!; expect(fault.startsWith("host-") ? target.killCalls : target.endCalls).toBe(1);
    expect(result.owners[0].outcomes.some((o: any) => o.status === "failed")).toBe(true);
  });
  it("already-exited whole host disposes its real local open output endpoints without a kill", async () => {
    const result = await lifecycleWhole("exited-pipes"); expect(result.thrown).toBe(result.value);
    const host = result.children[1]!; expect(host.killCalls).toBe(0); expect(host.stdoutDisposed).toBe(true); expect(host.stderrDisposed).toBe(true);
    expect(result.cleanup[0].notes.some((n: any) => n.qualification === "unqualified")).toBe(true);
  });
  it("whole host drains its actually started work before callback failure removes the root", async () => {
    const result = await lifecycleWhole("host-drain");
    expect(result.thrown).toBe(result.value); expect(result.callbacksExecuted).toBe(1); expect(result.escaped).toEqual([]); expect(result.escaped.length).toBe(0);
    expect(result.rootDuringStartedWork).toBe(true); expect(result.rootExists).toBe(false);
  });
  it("whole invalid frame propagates its executed callback error before cooperative EOF cleanup", async () => {
    const result = await lifecycleWhole("frame-callback");
    expect(result.thrown).toBe(result.value); expect(result.escaped).toEqual([]); expect(result.escaped.length).toBe(0); expect(result.callbacksExecuted).toBe(1);
    expect(result.children[1]!.endCalls).toBe(2); expect(result.children[1]!.killCalls).toBe(0); expect(result.rootExists).toBe(false);
  });
  it("callback primary presence and later callback/body errors survive an awaited work drain", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(), child = localChild("controller");
    const secondaryCallback = Object.freeze({ callbackSecondary: true }), secondaryBody = Object.freeze({ bodySecondary: true });
    let workDrained = false, releaseAfterDrain = false, threw = false, thrown: unknown;
    owner.defer("work-bound-resource", () => { releaseAfterDrain = workDrained; });
    try { await owner.run(async () => {
      owner.controller(child);
      await new Promise<void>(resolve => setImmediate(() => {
        owner.callback(() => { throw undefined; })(); owner.callback(() => { throw secondaryCallback; })();
        resolve();
      }));
      await localTick(); workDrained = true; throw secondaryBody;
    }); } catch (error) { threw = true; thrown = error; }
    expect(threw).toBe(true); expect(thrown).toBeUndefined(); expect(releaseAfterDrain).toBe(true);
    expect(owner.outcomes.some((o: any) => o.error === secondaryCallback)).toBe(true); expect(owner.outcomes.some((o: any) => o.error === secondaryBody)).toBe(true);
    recordLocal("secondary-errors-and-drain", { threw, primaryUndefined: thrown === undefined, releaseAfterDrain, callbackSecondaryExact: owner.outcomes.some((o: any) => o.error === secondaryCallback), bodySecondaryExact: owner.outcomes.some((o: any) => o.error === secondaryBody), cleanup: owner.diagnostics() });
  });
  it("failed local disposal cannot skip sibling endpoints and never forges their close", async () => {
    const child = localChild("host"), refusal = Object.freeze({ localDispose: true }), destroy = child.stdout.destroy;
    child.kill = (signal: string) => { child.killCalls++; child.signalCode = signal; child.emit("exit", null, signal); return true; };
    child.stdout.destroy = () => { throw refusal; };
    let thrown: any;
    try { await helpers.releaseOwnedProcess(child, [], 15); } catch (error) { thrown = error; }
    const before = { stdoutDestroyed: child.stdout.destroyed, stderrDestroyed: child.stderr.destroyed, stdinDestroyed: child.stdin.destroyed, terminal: child.terminal };
    expect(thrown).toBeInstanceOf(helpers.RecoveryExecutionCleanupError); expect(thrown.outcomes.some((o: any) => o.error === refusal)).toBe(true);
    expect(before).toEqual({ stdoutDestroyed: false, stderrDestroyed: true, stdinDestroyed: true, terminal: false });
    expect(child.listenerCount("error")).toBeGreaterThan(0);
    recordLocal("disposal-refusal-before-independent-cleanup", { before, outcomes: thrown.outcomes.map((o: any) => ({ label: o.label, status: o.status, exactRefusal: o.error === refusal })), ownedErrorObservers: child.listenerCount("error") });
    child.stdout.destroy = destroy; child.terminalClose(); await localTick(); expect(child.listenerCount("error")).toBe(0);
  });
  it("terminal observer refusal cannot suppress EOF or fabricate confirmed controller close", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(), child = localChild("controller"), refusal = Object.freeze({ closeObserver: true });
    const refuseCloseObservation = (event: string) => { if (event === "close") throw refusal; };
    child.on("newListener", refuseCloseObservation);
    let thrown: unknown; try { await owner.run(async () => { owner.controller(child); }); } catch (error) { thrown = error; }
    expect(thrown).toBe(refusal); expect(child.endCalls).toBe(1); expect(child.killCalls).toBe(0);
    expect(owner.outcomes.find((o: any) => o.label === "controller").status).toBe("failed");
    expect(owner.diagnostics().notes.some((n: any) => n.resource === "controller" && n.state === "pending" && n.closeObserved === false)).toBe(true);
    // Observation was refused, so unknown remains explicit even though this
    // authored object emitted close. Do not claim restoration that was unproved.
    expect(child.listenerCount("error")).toBeGreaterThan(0); expect(() => child.emit("error", Error("unobserved-terminal-late-error"))).not.toThrow();
    recordLocal("observer-refusal-before-independent-cleanup", { exactPrimary: thrown === refusal, eofCalls: child.endCalls, killCalls: child.killCalls, cleanup: owner.diagnostics(), ownedErrorObservers: child.listenerCount("error") });
    child.removeListener("newListener", refuseCloseObservation);
  });
  it("terminal observer removal failure is contained and cannot skip independent cleanup", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(), child = localChild("controller"), refusal = Object.freeze({ removeObserver: true }), unrelated = () => {};
    child.on("error", unrelated); child.on("close", unrelated);
    const remove = child.removeListener;
    child.removeListener = function(event: string, listener: any) { if (event === "error" && listener !== unrelated) throw refusal; return remove.call(this, event, listener); };
    let thrown: any; try { await owner.run(async () => { owner.controller(child); }); } catch (error) { thrown = error; }
    expect(thrown).toBeInstanceOf(helpers.RecoveryExecutionCleanupError); expect(owner.outcomes.some((o: any) => o.error === refusal)).toBe(true);
    expect(child.endCalls).toBe(1); expect(child.killCalls).toBe(0); expect(child.terminal).toBe(true);
    expect(child.listeners("error")).toContain(unrelated); expect(child.listeners("close")).toEqual([unrelated]);
    expect(child.stdout.closed).toBe(true); expect(child.stderr.closed).toBe(true);
    recordLocal("removal-refusal-before-independent-cleanup", { eofCalls: child.endCalls, killCalls: child.killCalls, terminal: child.terminal, restorationUnknown: owner.diagnostics().notes.some((n: any) => n.listenerRestoration === "unconfirmed"), cleanup: owner.diagnostics() });
    // Only the independent control restores its own injected method and removes
    // the precisely retained owner listener, preserving both unrelated hooks.
    child.removeListener = remove;
    for (const listener of child.listeners("error")) if (listener !== unrelated) child.removeListener("error", listener);
    expect(child.listeners("error")).toEqual([unrelated]);
  });
  it("800ms controller timeout hands off late errors until actual close, then restores unrelated listener baselines", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(), primary = Object.freeze({ pending: true }), child = localChild("controller");
    child.stdin.end = () => { child.endCalls++; return child.stdin; };
    const unrelated = () => {}; child.on("error", unrelated); child.stdin.on("error", unrelated); child.on("close", unrelated);
    const baseline = [child.listenerCount("error"), child.listenerCount("close"), child.stdin.listenerCount("error")];
    let thrown: unknown; const started = performance.now();
    try { await owner.run(async () => { const c = owner.controller(child); c.refuse(primary); await c.closed; }); } catch (error) { thrown = error; }
    expect(thrown).toBe(primary); expect(performance.now() - started).toBeGreaterThanOrEqual(750); expect(performance.now() - started).toBeLessThan(1200);
    expect(child.terminal).toBe(false); expect(child.killCalls).toBe(0); expect(owner.outcomes.find((o: any) => o.label === "controller").status).toBe("failed");
    expect(child.listenerCount("error")).toBeGreaterThan(baseline[0]!); expect(child.stdin.listenerCount("error")).toBeGreaterThan(baseline[2]!);
    const lateChild = Object.freeze({ lateChild: true }), lateStdin = Object.freeze({ lateStdin: true });
    await new Promise<void>(resolve => setImmediate(() => { expect(() => child.emit("error", lateChild)).not.toThrow(); expect(() => child.stdin.emit("error", lateStdin)).not.toThrow(); resolve(); }));
    expect(owner.outcomes.some((o: any) => o.error === lateChild)).toBe(true); expect(owner.outcomes.some((o: any) => o.error === lateStdin)).toBe(true);
    expect(owner.diagnostics().notes.some((n: any) => n.state === "pending")).toBe(true);
    recordLocal("timeout-before-terminal-handoff", { exactPrimary: thrown === primary, terminal: child.terminal, killCalls: child.killCalls, baseline, current: [child.listenerCount("error"), child.listenerCount("close"), child.stdin.listenerCount("error")], cleanup: owner.diagnostics() });
    child.terminalClose(); await localTick();
    expect([child.listenerCount("error"), child.listenerCount("close"), child.stdin.listenerCount("error")]).toEqual(baseline);
    expect(child.listeners("error")).toEqual([unrelated]); expect(child.stdin.listeners("error")).toEqual([unrelated]);
    recordLocal("timeout-after-real-terminal-handoff", { baseline, current: [child.listenerCount("error"), child.listenerCount("close"), child.stdin.listenerCount("error")], cleanup: owner.diagnostics() });
  });
  it("natural cooperative close preserves the result and removes only owned listeners", async () => {
    const owner = new helpers.RecoveryExecutionCleanup(), child = localChild("controller"), unrelated = () => {};
    child.on("error", unrelated); child.stdin.on("error", unrelated);
    const result = await owner.run(async () => { owner.controller(child); return "ok"; }); await localTick();
    expect(result).toBe("ok"); expect(child.endCalls).toBe(1); expect(child.killCalls).toBe(0); expect(child.terminal).toBe(true);
    expect(child.listeners("error")).toEqual([unrelated]); expect(child.stdin.listeners("error")).toEqual([unrelated]);
    expect(owner.outcomes.filter((o: any) => o.status === "failed")).toEqual([]);
    expect(owner.diagnostics().notes.some((n: any) => n.disposal === "forced-local-only")).toBe(false);
    recordLocal("natural-close", { result, eofCalls: child.endCalls, killCalls: child.killCalls, terminal: child.terminal, cleanup: owner.diagnostics() });
  });
});

// Acquisition-to-terminal contracts: real local streams, actual complete caller
// declarations and polling, authored handles/liveness, ZERO native subjects.
function acquisitionWhole(kind: Kind, onSpawn?: (child: any, context: any) => void, beforeSpawn?: (label: string, context: any) => void) {
  const area = fs.mkdtempSync(path.join(tmpdir(), "ea-acquisition-whole-"));
  const dir = output ? path.join(output, "acquire-" + String(++sequence).padStart(3, "0")) : path.join(area, "evidence");
  fs.mkdirSync(dir, { recursive: true });
  const ctx: any = { area, dir, kind, children: [], owners: [], events: [], timers: new Set(), intervals: new Set(), settled: false, threw: false, thrown: undefined };
  const bindings: Record<string, any> = { ...helpers, path, Buffer, JSON, expect, performance,
    process: { env: { PATH: process.env.PATH, QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS: "0" }, execPath: "/authored/not-executed", versions: {}, kill() { throw Error("numeric authority forbidden"); } },
    console: { log(text: string) { ctx.events.push({ kind: "log", value: JSON.parse(text) }); } }, tmpdir: () => area,
    source: "/authored/source.py", runner: "file:///authored/runner.ts", loader: "/authored/loader.mjs",
    mkdtempSync(prefix: string) { ctx.root = fs.mkdtempSync(prefix); return ctx.root; }, mkdirSync: fs.mkdirSync,
    writeFileSync: fs.writeFileSync, existsSync: fs.existsSync,
    read: (file: string) => fs.existsSync(file) ? fs.readFileSync(file, "utf8") : null,
    rmSync(file: string, options: any) { ctx.events.push({ kind: "scratch-release" }); return fs.rmSync(file, options); },
    sleep: (ms: number) => new Promise(resolve => setTimeout(resolve, ms)), alive: (pid: number) => pid === 730001,
    captureRecoveryExecutionFailure() { throw Error("collector not exercised in acquisition control"); },
    spawn(_executable: string, _args: any[], options: any) {
      const label = ctx.children.length === 0 ? "sentinel" : kind === "hostCase" ? "host" : "controller";
      beforeSpawn?.(label, ctx);
      const child: any = Object.assign(new EventEmitter(), { label, pid: label === "sentinel" ? 730001 : 730002,
        exitCode: null, signalCode: null, killCalls: 0, endCalls: 0, terminal: false, closeSeen: {} });
      const stdio = options.stdio === "ignore" ? ["ignore", "ignore", "ignore"] : options.stdio;
      for (const [index, name] of ["stdin", "stdout", "stderr"].entries()) {
        child[name] = stdio[index] === "pipe" ? new PassThrough() : null;
        if (child[name]) child[name].on("close", () => { child.closeSeen[name] = true; ctx.events.push({ kind: "endpoint-close", label, endpoint: name, wholeSettled: ctx.settled }); });
      }
      child.closeLocal = async () => {
        await Promise.all(["stdin", "stdout", "stderr"].map(name => {
          const stream = child[name]; if (!stream || child.closeSeen[name]) return;
          const closed = new Promise<void>(resolve => stream.once("close", resolve)); stream.destroy(); return closed;
        }));
        if (!child.terminal) { child.terminal = true; child.emit("close", child.exitCode, child.signalCode); }
      };
      child.kill = (signal: string) => { child.killCalls++; child.signalCode = signal; child.emit("exit", null, signal); void child.closeLocal(); return true; };
      if (child.stdin) {
        const end = child.stdin.end;
        child.stdin.end = function(...args: any[]) { child.endCalls++; end.apply(this, args); queueMicrotask(() => { child.exitCode = 0; child.emit("exit", 0, null); void child.closeLocal(); }); return this; };
      }
      ctx.children.push(child); onSpawn?.(child, ctx); return child;
    },
    setTimeout(callback: () => void, ms: number) { const timer = setTimeout(() => { ctx.timers.delete(timer); callback(); }, ms); ctx.timers.add(timer); return timer; },
    clearTimeout(timer: any) { ctx.timers.delete(timer); clearTimeout(timer); },
    setInterval(callback: () => void, ms: number) { const timer = setInterval(callback, ms); ctx.intervals.add(timer); return timer; },
    clearInterval(timer: any) { ctx.intervals.delete(timer); clearInterval(timer); },
  };
  bindings.RecoveryExecutionCleanup = class extends helpers.RecoveryExecutionCleanup { constructor() { super(); ctx.owners.push(this); } };
  // Only fixed declaration names extracted from the authenticated local AST
  // become code; fault values/objects are separate Function bindings.
  const make = (name: string) => new Function(...Object.keys(bindings), compile(functions[name]!) + ";return " + name + ";")(...Object.values(bindings));
  bindings.cleanEnv = make("cleanEnv"); bindings.until = make("until"); bindings.reap = make("reap"); bindings.observeStageFailure = make("observeStageFailure");
  bindings.instrument = (...args: any[]) => { const result = make("instrument")(...args); if (kind === "hostCase") fs.writeFileSync(path.join(ctx.root, "started.json"), "{}"); return result; };
  const whole = make(kind);
  const pending = kind === "hostCase" ? whole("node", "supervisor-consumed", "delay", "alive") : kind === "rawCase" ? whole("supervisor-consumed", "delay", "eof") : whole("Sbad\nK");
  ctx.pending = Promise.resolve(pending).then(() => { ctx.settled = true; }, error => { ctx.settled = true; ctx.threw = true; ctx.thrown = error; });
  ctx.snapshot = () => JSON.parse(JSON.stringify({ kind, settled: ctx.settled, threw: ctx.threw, rootExists: !!ctx.root && fs.existsSync(ctx.root),
    timers: ctx.timers.size, intervals: ctx.intervals.size, events: ctx.events, cleanup: ctx.owners.map((o: any) => o.diagnostics()),
    children: ctx.children.map((c: any) => ({ label: c.label, terminal: c.terminal, killCalls: c.killCalls, endCalls: c.endCalls,
      streams: Object.fromEntries(["stdin", "stdout", "stderr"].map(name => [name, c[name] ? { closed: c[name].closed, destroyed: c[name].destroyed, closeEventSeen: !!c.closeSeen[name], errors: c[name].listenerCount("error") } : null])) })) }));
  ctx.save = (name: string, value: unknown) => fs.writeFileSync(path.join(dir, name + ".json"), JSON.stringify(value, null, 2));
  ctx.dispose = async () => {
    expect(ctx.settled, "the entire actual caller must drain before independent cleanup").toBe(true);
    for (const timer of ctx.timers) clearTimeout(timer); for (const timer of ctx.intervals) clearInterval(timer);
    for (const child of ctx.children) await child.closeLocal(); fs.rmSync(area, { recursive: true, force: true });
  };
  return ctx;
}

describe("acquisition-first whole caller contracts (NO native subjects)", () => {
  for (const kind of ["hostCase", "rawCase", "invalidBudgetFrame"] as const) {
    for (const endpoint of kind === "hostCase" ? ["stdout", "stderr"] : ["stdin", "stdout", "stderr"]) it(`${kind} active ${endpoint} error rejects its actual awaited work`, async () => {
      const ctx = acquisitionWhole(kind), primary = Object.freeze(new Error("active-" + kind + "-" + endpoint));
      const escaped: unknown[] = [];
      try {
        await localTick(); const activeAtDispatch = !ctx.settled;
        try { ctx.children[1][endpoint].emit("error", primary); } catch (error) { escaped.push(error); }
        await new Promise(resolve => setTimeout(resolve, 30));
        const before = { whole: ctx.snapshot(), activeAtDispatch, escapedLength: escaped.length, exactPrimary: ctx.threw && ctx.thrown === primary,
          retained: ctx.owners[0].outcomes.some((o: any) => o.error === primary) };
        ctx.save("BEFORE-RESCUE", before);
        if (!ctx.settled) ctx.children[1].emit("error", Error("independent-after-snapshot-rescue"));
        await ctx.pending; ctx.save("AFTER-whole-drain", ctx.snapshot());
        expect(before.activeAtDispatch).toBe(true); expect(before.escapedLength).toBe(0); expect(before.exactPrimary).toBe(true);
        expect(before.retained).toBe(true); expect(before.whole.settled).toBe(true); expect(before.whole.rootExists).toBe(false);
        expect(ctx.timers.size).toBe(0); expect(ctx.intervals.size).toBe(0);
      } finally { await ctx.dispose(); }
    });
  }
  for (const direction of ["body-first", "callback-first"] as const) for (const attachment of ["before", "after"] as const) {
    for (const primary of [undefined, null, 0, false, "", Object.freeze({ causalPrimary: true })]) it(`whole host ${direction} ${attachment} attachment preserves exact ${String(primary)}`, async () => {
      const other = Object.freeze(new Error("other-causal-boundary")), order: string[] = [];
      let used = false;
      const ctx = acquisitionWhole("hostCase", child => {
        if (child.label !== "host") return;
        const fault = () => {
          if (used) return; used = true;
          if (direction === "body-first") {
            queueMicrotask(() => { order.push("callback"); child.emit("error", other); }); order.push("body"); throw primary;
          }
          order.push("callback"); child.emit("error", primary); order.push("body"); throw other;
        };
        if (attachment === "before") child.stdout.on("newListener", (event: string) => { if (event === "data") fault(); });
        else { const on = child.stdout.on; child.stdout.on = function(event: string, listener: any) { on.call(this, event, listener); if (event === "data") fault(); return this; }; }
      });
      try {
        await ctx.pending;
        const state = { order, exactPrimary: ctx.threw && Object.is(ctx.thrown, primary), otherRetained: ctx.owners[0].outcomes.some((o: any) => o.error === other), whole: ctx.snapshot() };
        ctx.save("causal-order-BEFORE-independent-cleanup", state);
        expect(state.order).toEqual(direction === "body-first" ? ["body", "callback"] : ["callback", "body"]);
        expect(state.exactPrimary).toBe(true); expect(state.otherRetained).toBe(true); expect(state.whole.rootExists).toBe(false);
      } finally { await ctx.dispose(); }
    });
  }
  it("real destroy(error) flags cannot precede owned error/close witnesses or root disposal", async () => {
    const primary = Object.freeze({ originalSetup: true }), streamError = Object.freeze(new Error("queued-release-error"));
    const events: any[] = []; let boundary: any;
    const ctx = acquisitionWhole("hostCase", (child, context) => {
      if (child.label !== "host") return;
      child.stdout.on("error", (error: unknown) => events.push({ kind: "actual-error", exact: error === streamError, wholeSettled: context.settled, rootExists: fs.existsSync(context.root) }));
      child.stdout.on("close", () => events.push({ kind: "actual-close", wholeSettled: context.settled, rootExists: fs.existsSync(context.root) }));
      child.stdout.on("newListener", (event: string) => { if (event === "data") throw primary; });
      child.kill = (signal: string) => {
        child.killCalls++; child.signalCode = signal; child.emit("exit", null, signal);
        child.stdout.destroy(streamError); child.stderr.destroy();
        boundary = { closed: child.stdout.closed, destroyed: child.stdout.destroyed, actualCloseSeen: !!child.closeSeen.stdout,
          errorObservers: child.stdout.listenerCount("error"), claimedClose: context.owners[0].diagnostics().notes.some((n: any) => n.resource === "host-local-stdout" && n.closeObserved) };
        context.save("AFTER-destroy-BEFORE-queued-events", boundary); void child.closeLocal(); return true;
      };
    });
    try {
      await ctx.pending; await localTick();
      const state = { boundary, events, exactPrimary: ctx.thrown === primary, streamErrorRetained: ctx.owners[0].outcomes.some((o: any) => o.error === streamError), whole: ctx.snapshot() };
      ctx.save("AFTER-events-BEFORE-independent-cleanup", state);
      expect(boundary.closed).toBe(true); expect(boundary.destroyed).toBe(true); expect(boundary.actualCloseSeen).toBe(false);
      expect(boundary.claimedClose).toBe(false); expect(boundary.errorObservers).toBeGreaterThan(1);
      expect(state.exactPrimary).toBe(true); expect(state.streamErrorRetained).toBe(true); expect(events.map(e => e.kind)).toEqual(["actual-error", "actual-close"]);
      expect(events.every(e => !e.wholeSettled && e.rootExists)).toBe(true); expect(state.whole.rootExists).toBe(false);
    } finally { await ctx.dispose(); }
  });
});

describe("acquisition interactions and explicit unknown history (NO native subjects)", () => {
  for (const kind of ["hostCase", "rawCase", "invalidBudgetFrame"] as const) for (const primary of [undefined, null, 0, false, "", Object.freeze({ spawnBoundary: true })]) {
    it(`${kind} synchronous body spawn boundary beats later callback for ${String(primary)}`, async () => {
      const later = Object.freeze(new Error("later-sentinel-error")), order: string[] = [];
      const ctx = acquisitionWhole(kind, undefined, (label, context) => {
        if (label === "sentinel") return;
        queueMicrotask(() => { order.push("callback"); context.children[0].emit("error", later); });
        order.push("body"); throw primary;
      });
      try {
        await ctx.pending;
        const state = { order, exactPrimary: ctx.threw && Object.is(ctx.thrown, primary), laterRetained: ctx.owners[0].outcomes.some((o: any) => o.error === later), whole: ctx.snapshot() };
        ctx.save("body-boundary-BEFORE-independent-cleanup", state);
        expect(state.order).toEqual(["body", "callback"]); expect(state.exactPrimary).toBe(true); expect(state.laterRetained).toBe(true);
        expect(state.whole.children).toHaveLength(1); expect(state.whole.children[0].killCalls).toBe(1); expect(state.whole.rootExists).toBe(false);
      } finally { await ctx.dispose(); }
    });
  }
  for (const attachment of ["before", "after"] as const) for (const event of ["error", "close"] as const) it(`whole frame ${attachment} ${event} observation refusal keeps sibling error ownership and EOF independent`, async () => {
    const primary = Object.freeze({ attachment, event }), later = Object.freeze(new Error("sibling-output-error")); let used = false;
    const ctx = acquisitionWhole("invalidBudgetFrame", child => {
      if (child.label !== "controller") return;
      const fault = () => { if (used) return; used = true; queueMicrotask(() => child.stderr.emit("error", later)); throw primary; };
      if (attachment === "before") child.stdin.on("newListener", (name: string) => { if (name === event) fault(); });
      else { const on = child.stdin.on; child.stdin.on = function(name: string, listener: any) { on.call(this, name, listener); if (name === event) fault(); return this; }; }
    });
    try {
      await ctx.pending; const whole = ctx.snapshot();
      const state = { exactPrimary: ctx.thrown === primary, laterRetained: ctx.owners[0].outcomes.some((o: any) => o.error === later), whole };
      ctx.save("partial-observation-BEFORE-independent-cleanup", state);
      expect(state.exactPrimary).toBe(true); expect(state.laterRetained).toBe(true); expect(whole.children[1].endCalls).toBe(1);
      expect(whole.children[1].killCalls).toBe(0); expect(whole.children[0].killCalls).toBe(1); expect(whole.rootExists).toBe(false);
      expect(ctx.owners[0].outcomes.find((o: any) => o.label === "controller").status).toBe("failed");
      if (attachment === "before" && event === "close") expect(whole.cleanup[0].notes.some((n: any) => n.resource === "controller-local-stdin" && n.state === "pending" && !n.closeObserved)).toBe(true);
    } finally { await ctx.dispose(); }
  });
  it("whole host acquired after destroy but before queued events still contains the real stream error", async () => {
    const primary = Object.freeze(new Error("destroy-before-acquisition")); let atAcquisition: any;
    const ctx = acquisitionWhole("hostCase", child => {
      if (child.label !== "host") return;
      child.stdout.destroy(primary); atAcquisition = { closed: child.stdout.closed, destroyed: child.stdout.destroyed, actualClose: !!child.closeSeen.stdout };
    });
    try {
      await ctx.pending; const state = { atAcquisition, exactPrimary: ctx.thrown === primary, retained: ctx.owners[0].outcomes.some((o: any) => o.error === primary), whole: ctx.snapshot() };
      ctx.save("queued-at-acquisition-BEFORE-independent-cleanup", state);
      expect(atAcquisition).toEqual({ closed: true, destroyed: true, actualClose: false }); expect(state.exactPrimary).toBe(true); expect(state.retained).toBe(true);
      expect(state.whole.rootExists).toBe(false); expect(state.whole.children[1].streams.stdout.closeEventSeen).toBe(true);
    } finally { await ctx.dispose(); }
  });
  it("a truly historical unobserved close stays unknown rather than becoming a flag witness", async () => {
    const child = localChild("host", true), unrelated = () => {};
    child.stdout.on("error", unrelated); child.stdout.destroy(); await localTick();
    expect(child.stdout.closed).toBe(true);
    const baseline = child.stdout.listeners("error"); let thrown: any;
    try { await helpers.releaseOwnedProcess(child, [], 15); } catch (error) { thrown = error; }
    const pending = { failed: thrown instanceof helpers.RecoveryExecutionCleanupError,
      historicalCloseUnconfirmed: thrown?.outcomes.some((o: any) => o.label === "local-stdout-close" && o.status === "failed"), errors: child.stdout.listenerCount("error"), baseline: baseline.length };
    recordLocal("historical-close-BEFORE-independent-cleanup", pending);
    expect(pending.failed).toBe(true); expect(pending.historicalCloseUnconfirmed).toBe(true); expect(pending.errors).toBeGreaterThan(pending.baseline);
    // No second close is invented. Independent disposal removes precisely the
    // unreachable test-local owner hooks, never relabelling unknown as drained.
    for (const event of ["error", "close"]) for (const listener of child.stdout.listeners(event)) if (listener !== unrelated) child.stdout.removeListener(event, listener);
    child.terminalClose(); await localTick(); expect(child.stdout.listeners("error")).toEqual(baseline);
  });
});

// Evaluate the WHOLE authenticated helper in a fresh, call-local module. Only
// note() is observed to retain the standalone fallback owner; failure methods,
// release bodies and actual local stream events are never replaced. This is
// not a global prototype patch, handler registry or mutation of thrown values.
const phaseHelperText = fs.readFileSync(helperPath, "utf8");
const phaseHelperCode = ts.transpileModule(phaseHelperText, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText;
recordLocal("phase-source-binding", { helperSha256: digest(phaseHelperText), emittedSha256: digest(phaseHelperCode), host: { execPath: process.execPath, node: process.versions.node, bun: process.versions.bun ?? null } });
type ReleaseMethod = "releaseOwnedProcess" | "releaseRecoveryController";
type ReleasePhase = "standalone" | "shared-clean" | "shared-body";
function phaseFixture(method: ReleaseMethod, phase: ReleasePhase, timerBindings: Record<string, any> = {}) {
  const local: any = {};
  new Function("exports", "setTimeout", "clearTimeout", phaseHelperCode)(local, timerBindings.setTimeout ?? setTimeout, timerBindings.clearTimeout ?? clearTimeout);
  let owner: any;
  const note = local.RecoveryExecutionCleanup.prototype.note;
  local.RecoveryExecutionCleanup.prototype.note = function(value: any) { owner = this; return note.call(this, value); };
  if (phase !== "standalone") owner = new local.RecoveryExecutionCleanup();
  const child = localChild(method === "releaseOwnedProcess" ? "host" : "controller");
  const escaped: unknown[] = [], order: string[] = [], restores: Array<() => void> = [];
  const closeSeen = new Set<any>();
  for (const stream of [child.stdin, child.stdout, child.stderr]) stream.once("close", () => closeSeen.add(stream));
  let closed = false, resolve!: () => void;
  const drained = new Promise<void>(done => { resolve = done; });
  child.once("close", () => { closed = true; resolve(); });
  const baseline = new Map<any, Record<string, Function[]>>();
  for (const target of [child, child.stdin, child.stdout, child.stderr]) {
    const unrelated = () => {};
    target.on("error", unrelated); target.on("close", unrelated);
    baseline.set(target, { error: [unrelated], close: [unrelated] });
  }
  const bodyPrimary = Object.freeze({ priorBody: method });
  const dispatch = (value: unknown, target: any = child.stderr) => {
    order.push("resource-error");
    try { target.emit("error", value); } catch (error) { escaped.push(error); }
  };
  const run = async () => {
    let threw = false, thrown: any;
    try {
      if (phase === "standalone") {
        if (method === "releaseOwnedProcess") await local.releaseOwnedProcess(child, [], 8);
        else await local.releaseRecoveryController(child, { drained, isClosed: () => closed }, 8);
      } else await owner.run(async () => {
        if (method === "releaseOwnedProcess") owner.ownedHandle(child, "direct", (observation: any) => local.releaseOwnedProcess(child, [], observation));
        else owner.controller(child);
        if (phase === "shared-body") throw bodyPrimary;
        return "body-ok";
      });
    } catch (error) { threw = true; thrown = error; }
    return { threw, thrown };
  };
  const snapshot = <Identities extends Record<string, unknown>>(name: string, result: any, identities: Identities = {} as Identities) => {
    const state = { method, phase, order: [...order], threw: result.threw, thrownName: result.thrown?.name ?? null, escapedLength: escaped.length,
      identities: Object.fromEntries(Object.entries(identities).map(([key, value]) => [key, { thrownExact: result.threw && Object.is(result.thrown, value), retainedExact: owner.outcomes.some((o: any) => o.status === "failed" && Object.is(o.error, value)) }])) as { [Key in keyof Identities]: { thrownExact: boolean; retainedExact: boolean } },
      child: { terminal: child.terminal, exitListeners: child.listenerCount("exit"), errorListeners: child.listenerCount("error"), killCalls: child.killCalls, endCalls: child.endCalls,
        streams: [child.stdin, child.stdout, child.stderr].map(stream => ({ destroyed: stream.destroyed, closedFlag: stream.closed, actualCloseSeen: closeSeen.has(stream), errorListeners: stream.listenerCount("error") })) }, cleanup: owner.diagnostics(),
      scope: "whole helper bytes; call-local owner and authored handles; real local stream events; PRE independent rescue; ZERO native subjects" };
    recordLocal("phase-" + String(++sequence).padStart(3, "0") + "-" + name + "-PRE-independent-rescue", state);
    return state;
  };
  const dispose = async () => {
    for (const restore of restores.reverse()) restore();
    const waits = [child.stdin, child.stdout, child.stderr].filter(stream => !closeSeen.has(stream)).map(stream => new Promise<void>(done => stream.once("close", done)));
    child.terminalClose(); await Promise.all(waits); await localTick();
    // Restore only retained hooks of this test-local owner after actual closes.
    // Intentional removal-refusal tests serialize the failed state first.
    for (const target of [child, child.stdin, child.stdout, child.stderr]) for (const event of ["error", "close"]) {
      const retained = baseline.get(target)![event]!;
      for (const listener of target.listeners(event)) if (!retained.includes(listener)) target.removeListener(event, listener);
      expect(target.listeners(event)).toEqual(retained);
    }
  };
  return { local, child, escaped, order, restores, closeSeen, bodyPrimary, dispatch, run, snapshot, dispose, get owner() { return owner; } };
}

describe("phase-aware standalone and shared release ordering (NO native subjects)", () => {
  for (const method of ["releaseOwnedProcess", "releaseRecoveryController"] as const) for (const phase of ["standalone", "shared-clean", "shared-body"] as const) {
    for (const direction of ["operation-first", "callback-first"] as const) for (const primary of [undefined, null, 0, false, "", Object.freeze({ opaqueFirst: true })]) {
      it(`${method} ${phase} ${direction} retains exact ${String(primary)} and its secondary`, async () => {
        const f = phaseFixture(method, phase), other = Object.freeze(new Error("other-local-boundary"));
        const action = () => {
          try {
            if (direction === "operation-first") {
              queueMicrotask(() => f.dispatch(other)); f.order.push("operation-throw"); throw primary;
            }
            f.dispatch(primary); f.order.push("operation-throw"); throw other;
          } finally {
            // Shared teardown can finish naturally AFTER both observed faults;
            // standalone negatives deliberately remain pending at the snapshot.
            if (phase !== "standalone") queueMicrotask(() => { f.child.exitCode = 0; f.child.emit("exit", 0, null); f.child.terminalClose(); });
          }
        };
        if (method === "releaseOwnedProcess") f.child.kill = () => { f.child.killCalls++; return action(); };
        else f.child.stdin.end = () => { f.child.endCalls++; return action(); };
        try {
          const result = await f.run(), state = f.snapshot(direction, result, { first: primary, other, body: f.bodyPrimary });
          expect(state.order).toEqual(direction === "operation-first" ? ["operation-throw", "resource-error"] : ["resource-error", "operation-throw"]);
          expect(result.threw).toBe(true); expect(f.escaped.length).toBe(0);
          expect(state.identities.other.retainedExact).toBe(true);
          if (phase === "standalone") expect(Object.is(result.thrown, primary)).toBe(true);
          else {
            expect(state.identities.first.retainedExact).toBe(true);
            if (phase === "shared-clean") { expect(result.thrown).toBeInstanceOf(f.local.RecoveryExecutionCleanupError); expect(result.thrown.outcomes).toBe(f.owner.outcomes); }
            else expect(result.thrown).toBe(f.bodyPrimary);
          }
          expect(f.child.listenerCount("exit")).toBe(0);
          expect(f.child.terminal).toBe(phase !== "standalone");
          expect(f.child.listenerCount("error")).toBe(phase === "standalone" ? 2 : 1);
          expect(f.child.killCalls).toBe(method === "releaseOwnedProcess" ? 1 : 0); expect(f.child.endCalls).toBe(method === "releaseRecoveryController" ? 1 : 0);
          expect(f.child.stdout.destroyed).toBe(true); expect(f.child.stderr.destroyed).toBe(true);
          if (method === "releaseRecoveryController" && phase === "standalone") { expect(f.child.stdin.destroyed).toBe(false); expect(f.child.stdin.listenerCount("error")).toBe(2); }
          if (phase !== "standalone") expect(f.closeSeen.size).toBe(3);
          if (primary && typeof primary === "object") { expect(Object.isFrozen(primary)).toBe(true); expect(Object.keys(primary)).toEqual(["opaqueFirst"]); }
        } finally { await f.dispose(); }
      });
    }
  }
});

describe("phase-aware owned-operation boundaries (NO native subjects)", () => {
  for (const method of ["releaseOwnedProcess", "releaseRecoveryController"] as const) for (const phase of ["standalone", "shared-clean"] as const) {
    for (const fault of ["observe-error", "observe-close", "deadline-create", "deadline-clear", "unexpected-access"] as const) {
      it(`${method} ${phase} captures ${fault} where it is first observed`, async () => {
        const primary = undefined, later = Object.freeze(new Error("later-after-" + fault));
        let f: ReturnType<typeof phaseFixture>, used = false;
        const first = (withLater = true, finish = false): never => {
          used = true; f.order.push("operation-throw");
          if (withLater) queueMicrotask(() => { f.dispatch(later, f.child); if (finish) { f.child.exitCode = 0; f.child.emit("exit", 0, null); f.child.terminalClose(); } });
          throw primary;
        };
        f = phaseFixture(method, phase, {
          setTimeout(callback: any, ms: number) { if (fault === "deadline-create" && !used) return first(); return setTimeout(callback, ms); },
          clearTimeout(timer: any) { clearTimeout(timer); if (fault === "deadline-clear" && !used) first(false); },
        });
        // Keep the handle genuinely pending until the scheduled local events;
        // ordinary EOF/kill method calls here are authored, never OS signals.
        if (fault !== "deadline-clear") {
          const finishObservationControl = () => { if (fault.startsWith("observe-")) queueMicrotask(() => { f.child.exitCode = 0; f.child.terminalClose(); }); };
          f.child.kill = () => { f.child.killCalls++; f.child.emit("exit", null, "SIGKILL"); finishObservationControl(); return true; };
          f.child.stdin.end = () => { f.child.endCalls++; finishObservationControl(); return f.child.stdin; };
        }
        if (fault.startsWith("observe-")) {
          const on = f.child.stdout.on;
          f.child.stdout.on = function(event: string, listener: any) { const result = on.call(this, event, listener); if (event === fault.slice(8) && !used) first(); return result; };
          f.restores.push(() => { f.child.stdout.on = on; });
        } else if (fault === "unexpected-access") {
          const target = method === "releaseOwnedProcess" ? f.child : f.child.stdin, key = method === "releaseOwnedProcess" ? "exitCode" : "end";
          const descriptor = Object.getOwnPropertyDescriptor(target, key)!;
          Object.defineProperty(target, key, { configurable: true, get() { if (!used) first(true, true); return descriptor.value; }, set(value) { Object.defineProperty(target, key, { ...descriptor, value }); } });
          f.restores.push(() => { Object.defineProperty(target, key, descriptor); });
        }
        try {
          const result = await f.run(), state = f.snapshot(fault, result, { primary, later });
          expect(used).toBe(true); expect(result.threw).toBe(true); expect(f.escaped.length).toBe(0);
          const active = phase === "standalone" || fault.startsWith("observe-");
          if (active) expect(Object.is(result.thrown, primary)).toBe(true);
          else { expect(result.thrown).toBeInstanceOf(f.local.RecoveryExecutionCleanupError); expect(state.identities.primary.retainedExact).toBe(true); }
          if (fault !== "deadline-clear") expect(state.identities.later.retainedExact).toBe(true);
          if (fault === "deadline-create") { expect(f.child.stdout.destroyed).toBe(true); expect(f.child.stderr.destroyed).toBe(true); }
          if (fault.startsWith("observe-")) expect(method === "releaseOwnedProcess" ? f.child.killCalls : f.child.endCalls).toBe(1);
          expect(f.child.killCalls).toBe(method === "releaseRecoveryController" ? 0 : fault === "unexpected-access" ? 0 : 1);
        } finally { await f.dispose(); }
      });
    }
    it(`${method} ${phase} keeps endpoint disposal independent and phase-correct`, async () => {
      const f = phaseFixture(method, phase), refusal = Object.freeze({ endpointDispose: true }), later = Object.freeze(new Error("after-dispose-refusal"));
      f.child.kill = () => { f.child.killCalls++; f.child.emit("exit", null, "SIGKILL"); return true; };
      f.child.stdin.end = () => { f.child.endCalls++; return f.child.stdin; };
      const destroy = f.child.stdout.destroy;
      f.child.stdout.destroy = () => { queueMicrotask(() => f.dispatch(later, f.child)); throw refusal; };
      f.restores.push(() => { f.child.stdout.destroy = destroy; });
      try {
        const result = await f.run(), state = f.snapshot("disposal", result, { refusal, later });
        expect(result.threw).toBe(true); expect(state.identities.refusal.retainedExact).toBe(true); expect(state.identities.later.retainedExact).toBe(true); expect(f.escaped.length).toBe(0);
        if (method === "releaseRecoveryController" && phase === "standalone") {
          const drain = f.owner.outcomes.find((o: any) => o.label === "controller-drain");
          expect(result.thrown).toBe(drain.error);
        } else expect(result.thrown).toBeInstanceOf(f.local.RecoveryExecutionCleanupError);
        expect(f.child.stdout.destroyed).toBe(false); expect(f.child.stderr.destroyed).toBe(true); expect(f.child.terminal).toBe(false);
        expect(f.child.stdin.destroyed).toBe(method === "releaseOwnedProcess");
      } finally { await f.dispose(); }
    });
  }
  for (const phase of ["standalone", "shared-clean"] as const) for (const operation of ["on", "removeListener"] as const) {
    it(`releaseOwnedProcess ${phase} independently kills and removes exit after ${operation} refusal`, async () => {
      const f = phaseFixture("releaseOwnedProcess", phase), primary = false, later = Object.freeze(new Error("after-exit-observation-refusal"));
      const original = f.child[operation]; let used = false;
      f.child[operation] = function(event: string, listener: any) {
        const value = original.call(this, event, listener);
        if (event === "exit" && !used) { used = true; queueMicrotask(() => f.dispatch(later, f.child)); f.order.push("operation-throw"); throw primary; }
        return value;
      };
      f.restores.push(() => { f.child[operation] = original; });
      f.child.kill = () => { f.child.killCalls++; f.child.emit("exit", null, "SIGKILL"); return true; };
      try {
        const result = await f.run(), state = f.snapshot("exit-" + operation, result, { primary, later });
        expect(used).toBe(true); expect(f.child.killCalls).toBe(1); expect(f.child.listenerCount("exit")).toBe(0); expect(f.escaped.length).toBe(0); expect(state.identities.later.retainedExact).toBe(true);
        if (phase === "standalone") expect(result.threw && Object.is(result.thrown, primary)).toBe(true);
        else { expect(result.thrown).toBeInstanceOf(f.local.RecoveryExecutionCleanupError); expect(state.identities.primary.retainedExact).toBe(true); }
      } finally { await f.dispose(); }
    });
  }
  for (const method of ["releaseOwnedProcess", "releaseRecoveryController"] as const) for (const event of ["error", "close"] as const) {
    for (const direction of ["operation-first", "callback-first"] as const) it(`${method} active terminal ${event} removal ${direction} keeps actual first primary`, async () => {
      const f = phaseFixture(method, "shared-clean"), primary = undefined, other = Object.freeze(new Error("other-after-terminal-removal"));
      const remove = f.child.stdout.removeListener, on = f.child.stdout.on; let used = false, ownedListener: any;
      // Retain the exact helper listener. EventEmitter.once removes its own
      // fixture wrapper before callback entry; that is NOT this owner boundary.
      f.child.stdout.on = function(name: string, listener: any) { if (name === event) ownedListener = listener; return on.call(this, name, listener); };
      f.child.stdout.removeListener = function(name: string, listener: any) {
        const result = remove.call(this, name, listener);
        if (name === event && listener === ownedListener && !used) {
          used = true;
          if (direction === "operation-first") { queueMicrotask(() => f.dispatch(other)); f.order.push("operation-throw"); throw primary; }
          f.dispatch(primary); f.order.push("operation-throw"); throw other;
        }
        return result;
      };
      f.restores.push(() => { f.child.stdout.removeListener = remove; f.child.stdout.on = on; });
      let result: any;
      try {
        let threw = false, thrown: unknown;
        try { await f.owner.run(async () => {
          if (method === "releaseOwnedProcess") f.owner.ownedHandle(f.child, "direct", (observation: any) => f.local.releaseOwnedProcess(f.child, [], observation));
          else f.owner.controller(f.child);
          f.child.stdout.destroy(); await localTick();
          expect(f.closeSeen.has(f.child.stdout)).toBe(true); expect(f.child.stderr.destroyed).toBe(false);
        }); } catch (error) { threw = true; thrown = error; }
        result = { threw, thrown };
        const state = f.snapshot("active-remove-" + event + "-" + direction, result, { primary, other });
        expect(used).toBe(true); expect(threw).toBe(true); expect(Object.is(thrown, primary)).toBe(true); expect(state.identities.other.retainedExact).toBe(true); expect(f.escaped.length).toBe(0);
        expect(f.order).toEqual(direction === "operation-first" ? ["operation-throw", "resource-error"] : ["resource-error", "operation-throw"]);
      } finally { await f.dispose(); }
    });
    it(`${method} shared teardown terminal ${event} removal remains cleanup-only`, async () => {
      const f = phaseFixture(method, "shared-clean"), refusal = undefined, remove = f.child.stdout.removeListener, on = f.child.stdout.on; let used = false, ownedListener: any;
      f.child.stdout.on = function(name: string, listener: any) { if (name === event) ownedListener = listener; return on.call(this, name, listener); };
      f.child.stdout.removeListener = function(name: string, listener: any) { const result = remove.call(this, name, listener); if (name === event && listener === ownedListener && !used) { used = true; throw refusal; } return result; };
      f.restores.push(() => { f.child.stdout.removeListener = remove; f.child.stdout.on = on; });
      try {
        const result = await f.run(), state = f.snapshot("teardown-remove-" + event, result, { refusal });
        expect(used).toBe(true); expect(result.thrown).toBeInstanceOf(f.local.RecoveryExecutionCleanupError); expect(state.identities.refusal.retainedExact).toBe(true); expect(f.child.terminal).toBe(true);
      } finally { await f.dispose(); }
    });
  }
  it("pure diagnostic recording does not admit a body primary", async () => {
    const f = phaseFixture("releaseOwnedProcess", "shared-clean");
    try {
      let threw = false, thrown: unknown;
      try { await f.owner.run(() => { f.owner.failed("diagnostic-only", undefined); f.owner.callback(() => { throw false; })(); }); } catch (error) { threw = true; thrown = error; }
      f.snapshot("diagnostic-not-admission", { threw, thrown }, { diagnostic: undefined, callback: false });
      expect(threw).toBe(true); expect(Object.is(thrown, false)).toBe(true); expect(f.owner.outcomes[0].label).toBe("diagnostic-only");
    } finally { await f.dispose(); }
  });
});

// Wait rejection must be admitted before an independently failing finalizer.
// Bind a fresh complete helper per test; never replace global timer functions.
describe("drain failure before timer finalization", () => {
  const errors: Array<[string, unknown]> = [
    ["undefined", undefined], ["null", null], ["false", false], ["zero", 0],
    ["empty string", ""], ["frozen object", Object.freeze({ original: true })],
    ["frozen Error", Object.freeze(new Error("original drain failure"))],
  ];
  for (const [label, early] of errors) for (const inject of [false, true]) {
    it(`${label}: ${inject ? "failing" : "ordinary"} timer disposal preserves the original rejection`, async () => {
      const active = new Set<ReturnType<typeof setTimeout>>();
      const later = Object.freeze(new Error("later timer disposal failure"));
      let injected = false;
      const schedule = (callback: () => void, delay: number) => {
        const timer = setTimeout(() => { active.delete(timer); callback(); }, delay);
        active.add(timer); return timer;
      };
      const cancel = (timer: ReturnType<typeof setTimeout> | undefined) => {
        clearTimeout(timer); if (timer !== undefined) active.delete(timer);
        if (inject && !injected) { injected = true; throw later; }
      };
      const emitted = compile(fs.readFileSync(helperPath, "utf8")).replace(/^export /gm, "");
      const local = new Function("setTimeout", "clearTimeout", emitted + ";return { releaseRecoveryController };")(schedule, cancel);
      let killed = 0, eof = 0;
      const child = Object.assign(new EventEmitter(), {
        stdin: new PassThrough(), stdout: new PassThrough(), stderr: new PassThrough(),
        exitCode: null, signalCode: null,
        kill() { killed++; throw new Error("controller kill is forbidden"); },
      });
      const end = child.stdin.end;
      child.stdin.end = function (...args: any[]) { eof++; return (end as any).apply(this, args); };
      const streams = [child.stdin, child.stdout, child.stderr];
      const closes = streams.map(stream => new Promise<void>(resolve => stream.once("close", resolve)));
      let reject!: (error: unknown) => void;
      const drained = new Promise<void>((_resolve, fail) => { reject = fail; });
      let threw = false, thrown: unknown;
      try {
        const releasing = local.releaseRecoveryController(child, { drained, isClosed: () => false });
        reject(early);
        try { await releasing; } catch (error) { threw = true; thrown = error; }
        expect(threw).toBe(true);
        expect(Object.is(thrown, early)).toBe(true);
        expect(injected).toBe(inject);
        expect(active.size).toBe(0);
        expect(eof).toBe(1);
        expect(killed).toBe(0);
      } finally {
        // This is independent local-fixture disposal, not native custody proof.
        for (const timer of active) clearTimeout(timer);
        for (const stream of streams) stream.destroy();
        child.emit("close", 0, null);
        await Promise.all(closes);
      }
    });
  }
});
