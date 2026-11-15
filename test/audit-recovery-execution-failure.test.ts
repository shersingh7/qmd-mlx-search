const testRuntime = (process.versions as Record<string, string>).bun ? "bun:test" : "vitest";
const { describe, expect, it }: typeof import("vitest") = await import(testRuntime);
import fs from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import path from "node:path";
import ts from "typescript";
import { fileURLToPath } from "node:url";
import { captureRecoveryExecutionFailure as capture, EXECUTION_FAILURE_LIMITS as limits,
  type FailureCaptureHooks, type FailureCaptureIO, type RecoveryExecutionFailureContext as Context } from "./helpers/recovery-execution-failure.js";

// Filesystem-only contracts. No integration module import/registration, Python,
// subprocess, signalling, version query, protocol observer or recovery launch.
const admissionPath = fileURLToPath(new URL("./audit-recovery-execution-admission.test.ts", import.meta.url));
const admissionText = fs.readFileSync(admissionPath, "utf8");
const admission = ts.createSourceFile(admissionPath, admissionText, ts.ScriptTarget.Latest, true);
const definitions = admission.statements.filter(ts.isFunctionDeclaration);
const observeText = definitions.find(node => node.name?.text === "observeStageFailure")!.getText(admission);
const hostCase = definitions.find(node => node.name?.text === "hostCase")!;
let callbackText = "";
function findCallback(node: ts.Node) {
  if (ts.isCallExpression(node) && node.expression.getText(admission) === "observeStageFailure") callbackText = node.arguments[1]!.getText(admission);
  ts.forEachChild(node, findCallback);
}
findCallback(hostCase);
if (!callbackText) throw new Error("actual hostCase snapshot callback not found");
const compile = (text: string) => ts.transpileModule(text, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
const digest = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
const envKeys = ["QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS", "QMD_RECOVERY_BOOTSTRAP_DIR", "GITHUB_RUN_ID", "GITHUB_JOB", "GITHUB_RUN_ATTEMPT", "EA_AUTHORED_SECRET_CONTROL"];
const fixedNames = ["started.json", "helper.json", "stage.json", "forks.jsonl", "reaps.jsonl", "gate-forwarded.json", "validation.json", "queued-eof.json", "control-reads.jsonl", "probe.json", "execution-boundary.py", "host.ts"];
let sequence = 0;
// Raw receipts are unit-authored fixture observations, NOT canonical executions.
async function fixture(run: (ctx: Context, out: string) => unknown | Promise<unknown>) {
  const root = fs.mkdtempSync(path.join(tmpdir(), "ea-failure-unit-")), out = path.join(root, "receipts");
  const saved = Object.fromEntries(envKeys.map(k => [k, process.env[k]]));
  Object.assign(process.env, { QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS: "1", QMD_RECOVERY_BOOTSTRAP_DIR: out,
    GITHUB_RUN_ID: "123456789", GITHUB_JOB: "unit-job", GITHUB_RUN_ATTEMPT: "2", EA_AUTHORED_SECRET_CONTROL: "authored-do-not-dump-environment" });
  const ctx: Context = { root, runtime: "node", mode: "alive", stage: "supervisor-consumed", pause: "delay",
    host: { pid: 1234, exitCode: 0, signalCode: null }, sentinelPid: 2345, executable: "/authored/requested-node", args: ["--import", "/authored/loader", "host.ts"],
    fixtureElapsedMs: 1801.25, stdout: "authored stdout é�", stderr: "authored stderr" };
  try { await run(ctx, out); }
  finally {
    if (process.env.QMD_EA_UNIT_OUTPUT) {
      const dest = path.join(process.env.QMD_EA_UNIT_OUTPUT, String(++sequence)); fs.mkdirSync(dest, { recursive: true });
      fs.writeFileSync(path.join(dest, "unit-host.json"), JSON.stringify({ scope: "filesystem-only-authored-fixture", observations: "authored fixture files and ChildProcess-shaped fields, not a recovery execution",
        fifoQualification: process.env.QMD_EA_FIFO ? "owned-real-FIFO-never-opened-for-reading" : "authored-FIFO-stat-seam",
        execPath: process.execPath, pid: process.pid, versions: { node: process.versions.node, bun: process.versions.bun ?? null } }));
      if (fs.existsSync(out)) for (const name of fs.readdirSync(out)) {
        const file = path.join(out, name); if (fs.lstatSync(file).isFile() && name.endsWith(".json")) fs.copyFileSync(file, path.join(dest, name));
      }
    }
    for (const k of envKeys) { if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k]; }
    fs.rmSync(root, { recursive: true, force: true });
  }
}
function load(ctx: Context, hooks?: FailureCaptureHooks) {
  const result = capture(ctx, hooks); expect(result.status).toBe("captured"); expect(result.path).toBeDefined();
  const bytes = fs.readFileSync(result.path!); expect(bytes.length).toBeLessThanOrEqual(limits.receiptBytes);
  // Receipt schema assertions below are independent of implementation TS types.
  return { result, bytes, data: JSON.parse(bytes.toString("utf8")) };
}
function io(overrides: Partial<FailureCaptureIO> = {}): FailureCaptureIO {
  return { lstatSync: fs.lstatSync, openSync: fs.openSync, fstatSync: fs.fstatSync, readSync: fs.readSync, closeSync: fs.closeSync,
    mkdirSync: fs.mkdirSync, writeSync: fs.writeSync, linkSync: fs.linkSync, unlinkSync: fs.unlinkSync, ...overrides };
}
function noWork(): FailureCaptureHooks {
  const fail = () => { throw new Error("unexpected capture work"); };
  return { io: new Proxy({} as FailureCaptureIO, { get: fail }), serialize: fail, nonce: fail };
}
function observe(logger: { log: (...args: unknown[]) => unknown }) {
  return new Function("process", "console", compile(observeText) + ";return observeStageFailure;")(process, logger) as
    (check: () => Promise<void>, snapshot: () => unknown) => Promise<void>;
}
function callback(ctx: Context, captureFn: typeof capture = capture) {
  const bindings = { captureRecoveryExecutionFailure: captureFn, root: ctx.root, runtime: ctx.runtime, mode: ctx.mode, stage: ctx.stage,
    pause: ctx.pause, probe: ctx.probe, host: ctx.host, sentinel: { pid: ctx.sentinelPid }, executable: ctx.executable, args: ctx.args,
    started: 0, performance: { now: () => ctx.fixtureElapsedMs }, stdout: ctx.stdout, stderr: ctx.stderr,
    env: { QMD_RECOVERY_PYTHON: ctx.supervisorPython },
    path, read: (p: string) => fs.existsSync(p) ? fs.readFileSync(p, "utf8") : null, existsSync: fs.existsSync,
    forks: () => (fs.existsSync(path.join(ctx.root, "forks.jsonl")) ? fs.readFileSync(path.join(ctx.root, "forks.jsonl"), "utf8") : "").trim().split("\n").filter(Boolean).map(line => JSON.parse(line)) };
  return new Function(...Object.keys(bindings), compile("const snapshot = " + callbackText) + ";return snapshot;")(...Object.values(bindings)) as () => unknown;
}
function receipts(out: string) { return fs.existsSync(out) ? fs.readdirSync(out).filter(n => n.endsWith(".json")) : []; }
function expectStream(stream: { capturedSha256: unknown; capturedBytes: number; byteLength: number; truncated: boolean }, value: string) {
  // Independent oracle: encode the COMPLETE authored string, then take bytes.
  // Only tests may allocate this full encoding; no UTF-16 boundary logic here.
  // Buffer.byteLength is not an oracle: Bun 1.3.8 undercounts lone surrogates.
  const complete = Buffer.from(value, "utf8"), prefix = complete.subarray(0, limits.textBytes);
  const capturedSha256 = { algorithm: "sha256", scope: "captured-bytes", value: digest(prefix) };
  expect(stream.capturedSha256).toEqual(capturedSha256);
  expect(stream.capturedBytes).toBeLessThanOrEqual(stream.byteLength);
  expect(stream.truncated).toBe(complete.length > prefix.length);
  expect(stream).toEqual({ source: "existing-JS-string-encoded-as-UTF8-not-original-stream-bytes", byteLength: complete.length,
    capturedBytes: prefix.length, rawBase64: prefix.toString("base64"), truncated: complete.length > prefix.length, capturedSha256 });
}

// These execute the unmodified observer and the actual changed callback, not a
// duplicate observer, text-presence check or import of a registering test module.
// Function bodies come only from this public repository test's parsed AST;
// fixture strings and external input are passed as bindings, never source text.
describe("failure-only recovery execution evidence", () => {
  it("does no capture work on success, disabled diagnostics, missing destination or controls", () => fixture(async ctx => {
    let calls = 0; const wrapper = observe({ log: () => { throw new Error("unexpected log"); } });
    await wrapper(async () => {}, () => { calls++; return callback(ctx)(); }); expect(calls).toBe(0);
    delete process.env.QMD_RECOVERY_BOOTSTRAP_DIR;
    expect(capture(ctx, noWork()).status).toBe("no-destination");
    process.env.QMD_RECOVERY_BOOTSTRAP_DIR = ""; expect(capture(ctx, noWork()).status).toBe("no-destination");
    process.env.QMD_RECOVERY_BOOTSTRAP_DIR = path.join(ctx.root, "unused"); process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS = "0";
    expect(capture(ctx, noWork()).status).toBe("disabled");
    const primary = { primary: true }; let observed = false;
    await expect(wrapper(async () => { throw primary; }, () => { observed = true; throw new Error("must not observe"); })).rejects.toBe(primary);
    expect(observed).toBe(false); process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS = "1";
    for (const control of [{ probe: "startup-delay" }, { mode: "valid" as const }, { stage: "none" }, { stage: "child-validation" }, { pause: "stop" as const }])
      expect(capture({ ...ctx, ...control }, noWork()).status).toBe("noncanonical");
    expect(fs.existsSync(process.env.QMD_RECOVERY_BOOTSTRAP_DIR!)).toBe(false);
  }));

  it("captures before the actual legacy fork parser can fail, retaining identical primary", () => fixture(async (ctx, out) => {
    fs.writeFileSync(path.join(ctx.root, "forks.jsonl"), '{"pid":');
    for (const primary of [new Error("stage missing"), { marker: 1 }, "non-Error primary", null, undefined]) {
      const logs: unknown[] = []; let cleanup = false;
      try { await expect(observe({ log: x => logs.push(x) })(async () => { throw primary; }, callback(ctx))).rejects.toBe(primary); }
      finally { cleanup = true; }
      expect(cleanup).toBe(true); expect(logs).toEqual([]);
    }
    expect(receipts(out)).toHaveLength(5);
    for (const name of receipts(out)) {
      const data = JSON.parse(fs.readFileSync(path.join(out, name), "utf8"));
      expect(data.fixtureCleanupStarted).toBe(false); expect(data.files["forks.jsonl"].content.json).toBe("invalid-json");
    }
  }));

  it("keeps legacy snapshots/logging when capture read, write or serialization fails", () => fixture(async ctx => {
    const failures: FailureCaptureHooks[] = [
      { io: io({ lstatSync: (() => { throw Object.assign(new Error("authored read failure"), { code: "EACCES" }); }) as typeof fs.lstatSync }) },
      { io: io({ writeSync: (() => { throw Object.assign(new Error("authored write failure"), { code: "EIO" }); }) as typeof fs.writeSync }) },
      { serialize: () => { throw new Error("authored serialize failure"); } },
    ];
    fs.writeFileSync(path.join(ctx.root, "helper.json"), JSON.stringify({ elapsedMs: 211, guardFired: false, result: { launch: { state: "started" }, error: "full helper remains raw" } }));
    for (const hooks of failures) {
      const logs: string[] = [], primary = Symbol("identical primary");
      await expect(observe({ log: x => logs.push(String(x)) })(async () => { throw primary; }, callback(ctx, c => capture(c, hooks)))).rejects.toBe(primary);
      expect(logs).toHaveLength(1); expect(JSON.parse(logs[0]!)).toMatchObject({ evidence: "recovery-missing-stage-before-finally", helper: { elapsedMs: 211, launch: { state: "started" } } });
    }
    let logCalls = 0; const primary = 17;
    await expect(observe({ log: () => { logCalls++; throw new Error("log failure"); } })(async () => { throw primary; }, callback(ctx))).rejects.toBe(primary);
    expect(logCalls).toBe(1);
    await expect(observe({ log: () => {} })(async () => { throw primary; }, () => ({ toJSON() { throw new Error("snapshot serialization"); } }))).rejects.toBe(primary);
  }));

  it("recognizes exactly the twenty canonical registrations and excludes probe self-tests", () => fixture((ctx, out) => {
    const titles = new Set<string>();
    for (const runtime of ["node", "bun"] as const) {
      for (const stage of ["supervisor-consumed", "supervisor-forwarded", "child-admitted"]) for (const mode of ["alive", "graceful", "hard"] as const) {
        const { data } = load({ ...ctx, runtime, stage, mode }); titles.add(data.case.title);
        expect(data.case).toMatchObject({ runtimeRequested: runtime, stage, mode, pause: "delay", classification: "canonical" });
      }
      const { data } = load({ ...ctx, runtime, mode: "hard", pause: "stop" }); titles.add(data.case.title);
      expect(data.case.title).toBe(`cleans real ${runtime} SIGKILL with its supervisor SIGSTOPped`);
    }
    expect(titles.size).toBe(20); expect(receipts(out)).toHaveLength(20);
    expect(capture({ ...ctx, probe: "startup-delay" }).status).toBe("noncanonical"); expect(receipts(out)).toHaveLength(20);
  }));

  it("retains exact fixed raw bytes/full helper, state, public-source hashes and separate actual identities", () => fixture((ctx, out) => {
    const authored: Record<string, Buffer> = {};
    for (const name of fixedNames) {
      const value = name === "helper.json" ? JSON.stringify({ elapsedMs: 211.2, guardFired: false, result: { error: "specific failure é�", timedOut: true, pid: 42, exitCode: 0, report: null, launch: { state: "started", admissionGranted: true }, cleanup: { childReaped: true } } })
        : name === "started.json" ? JSON.stringify({ executable: "/authored/actual-bun", versions: { bun: "authored-version", node: "compatibility-only" } })
        : name.endsWith(".jsonl") ? '{"pid":42}\n' : name.endsWith(".json") ? '{"pid":42}' : `# exact generated bytes ${name} é�\n`;
      authored[name] = Buffer.from(value); fs.writeFileSync(path.join(ctx.root, name), authored[name]!);
    }
    const beforeListeners = process.eventNames().map(e => [e, process.listenerCount(e)]);
    const beforeGlobals = { open: fs.openSync, read: fs.readSync, write: fs.writeSync, stringify: JSON.stringify };
    const { result, bytes, data } = load(ctx);
    expect(path.dirname(result.path!)).toBe(out); expect(path.basename(result.path!)).toMatch(/^recovery-execution-failure-.*\.json$/);
    expect(fs.statSync(result.path!).mode & 0o777).toBe(0o600); expect(fs.readdirSync(out)).toHaveLength(1);
    expect(data.scope).toBe("canonical-missing-stage-existing-evidence-only"); expect(data.fixtureCleanupStarted).toBe(false);
    expect(Object.keys(data.files).sort()).toEqual([...fixedNames].sort());
    for (const [name, value] of Object.entries(authored)) {
      const file = data.files[name]; expect(file.status).toBe("captured"); expect(file.truncated).toBe(false); expect(file.stability).toBe("stable-at-observations");
      expect(Buffer.from(file.rawBase64, "base64")).toEqual(value); expect(file.capturedBytes).toBe(value.length);
      expect(file.capturedSha256).toEqual({ algorithm: "sha256", scope: "captured-bytes", value: digest(value) }); expect(file.fullFileSha256).toBe(digest(value));
    }
    expect(data.parentTestHost).toMatchObject({ executable: process.execPath, pid: process.pid, runtime: process.versions.bun ? "bun" : "node" });
    expect(data.case.runtimeRequested).toBe("node"); expect(data.callerObserved).toMatchObject({ status: "observed-existing-started-json", executable: "/authored/actual-bun", versions: { bun: "authored-version" } });
    expect(data.pythonIdentity).toEqual({ status: "unknown-not-observed", requestedExecutable: "/usr/bin/python3", selectedExecutable: null, version: null });
    expect(data.github).toEqual({ GITHUB_RUN_ID: "123456789", GITHUB_JOB: "unit-job", GITHUB_RUN_ATTEMPT: "2" });
    expect(bytes.toString()).not.toContain("authored-do-not-dump-environment");
    for (const relative of ["scripts/phase5_recovery_process.py", "test/audit-recovery-execution-admission.test.ts", "scripts/phase5_e2e_runner.ts", "scripts/phase5_recovery_process_loader.mjs"]) {
      const expected = fs.readFileSync(fileURLToPath(new URL("../" + relative, import.meta.url))); const source = data.sources[relative];
      expect(source.fullFileSha256).toBe(digest(expected)); expect(Buffer.from(source.rawBase64, "base64")).toEqual(expected);
    }
    expect(data.fixture).toMatchObject({ root: ctx.root, host: ctx.host, sentinelPid: ctx.sentinelPid, elapsedMs: ctx.fixtureElapsedMs });
    expect(process.eventNames().map(e => [e, process.listenerCount(e)])).toEqual(beforeListeners);
    expect({ open: fs.openSync, read: fs.readSync, write: fs.writeSync, stringify: JSON.stringify }).toEqual(beforeGlobals);
  }));

  it("retains the configured interpreter literal without inventing an observed identity", () => fixture(ctx => {
    const requested = "/qualified/../runtime/python with spaces";
    const { data } = load({ ...ctx, supervisorPython: requested });
    expect(data.pythonIdentity).toEqual({ status: "unknown-not-observed", requestedExecutable: requested, selectedExecutable: null, version: null });
    for (const invalid of ["", "relative-python", "/x\0y"]) {
      expect(capture({ ...ctx, supervisorPython: invalid })).toMatchObject({ status: "failed", code: "INVALID_ABSOLUTE_PATH" });
    }
  }));

  it("distinguishes absent, empty, invalid UTF-8/JSON, valid U+FFFD and JSONL without inventing partial publication", () => fixture(ctx => {
    fs.writeFileSync(path.join(ctx.root, "helper.json"), ""); fs.writeFileSync(path.join(ctx.root, "stage.json"), Buffer.from([0xff]));
    fs.writeFileSync(path.join(ctx.root, "validation.json"), '{"partial":'); fs.writeFileSync(path.join(ctx.root, "probe.json"), '{"valid":"é�"}');
    fs.writeFileSync(path.join(ctx.root, "forks.jsonl"), '{"pid":1}\n{"pid":2}\n'); fs.writeFileSync(path.join(ctx.root, "reaps.jsonl"), '{"pid":1}\nnope\n');
    fs.writeFileSync(path.join(ctx.root, "control-reads.jsonl"), '\u00a0\n');
    const { data } = load(ctx);
    expect(data.files["started.json"].status).toBe("absent"); expect(data.callerObserved.status).toBe("unknown");
    expect(data.files["helper.json"].content).toEqual({ utf8: "valid", json: "empty" }); expect(data.files["helper.json"].fullFileSha256).toBe(digest(""));
    expect(data.files["stage.json"].content).toEqual({ utf8: "invalid", json: "invalid-utf8" });
    expect(data.files["validation.json"].content.json).toBe("invalid-json"); expect(data.files["validation.json"].stability).toBe("stable-at-observations");
    expect(data.files["probe.json"].content).toEqual({ utf8: "valid", json: "valid-json" });
    expect(data.files["forks.jsonl"].content.json).toBe("valid-jsonl"); expect(data.files["reaps.jsonl"].content.json).toBe("invalid-json");
    expect(data.files["control-reads.jsonl"].content).toEqual({ utf8: "valid", json: "invalid-json" });
    expect(data.interpretation).toContain("parse errors do not prove partial publication");
  }));

  it("bounds every input and aggregate reads/output and never labels a prefix digest as full", () => fixture(ctx => {
    const large = Buffer.alloc(limits.fileBytes + 512, 120);
    for (const name of fixedNames) fs.writeFileSync(path.join(ctx.root, name), large);
    let readBytes = 0, maxRequested = 0;
    const boundIO = io({ readSync: ((fd: number, buffer: Buffer, offset: number, length: number, position: number) => {
      maxRequested = Math.max(maxRequested, length); const n = fs.readSync(fd, buffer, offset, length, position); readBytes += n; return n;
    }) as typeof fs.readSync });
    const { bytes, data } = load(ctx, { io: boundIO });
    expect(readBytes).toBeLessThanOrEqual(limits.totalReadBytes); expect(maxRequested).toBeLessThanOrEqual(limits.fileBytes + 1);
    expect(data.budget.readBytes).toBe(readBytes); expect(bytes.length).toBeLessThanOrEqual(limits.receiptBytes);
    const first = data.files["started.json"]; expect(first.truncated).toBe(true); expect(first.capturedBytes).toBe(limits.fileBytes);
    expect(first.fullFileSha256).toBeNull(); expect(first.capturedSha256.scope).toBe("captured-bytes"); expect(first.content.json).toBe("not-evaluated-incomplete");
    expect(Object.values(data.files).some((f: any) => f.status === "budget-exhausted")).toBe(true);
    expect(data.budget.exhausted).toBe(true);
  }));

  it("refuses directories, symlinks (also ancestors), dangling links and owned FIFOs before open", () => fixture(ctx => {
    // The owned qualification harness supplies a real FIFO, created by tool
    // Python without ever reading it. Ordinary CI uses a labelled synthetic
    // FIFO stat seam instead of launching mkfifo from a filesystem-only test.
    const fifo = process.env.QMD_EA_FIFO;
    const stageFile = path.join(ctx.root, "stage.json");
    if (fifo) fs.linkSync(fifo, stageFile); else fs.writeFileSync(stageFile, "synthetic FIFO metadata control");
    fs.mkdirSync(path.join(ctx.root, "helper.json"));
    fs.symlinkSync(path.join(ctx.root, "missing"), path.join(ctx.root, "started.json"));
    fs.writeFileSync(path.join(ctx.root, "target"), "must not read"); fs.symlinkSync(path.join(ctx.root, "target"), path.join(ctx.root, "probe.json"));
    const opened: string[] = [];
    const refusalIO = io({
      lstatSync: ((p: fs.PathLike, ...args: any[]) => {
        const stat = (fs.lstatSync as any)(p, ...args);
        if (!fifo && String(p) === stageFile) stat.mode = typeof stat.mode === "bigint"
          ? (stat.mode & ~BigInt(fs.constants.S_IFMT)) | BigInt(fs.constants.S_IFIFO)
          : (stat.mode & ~fs.constants.S_IFMT) | fs.constants.S_IFIFO;
        return stat;
      }) as typeof fs.lstatSync,
      openSync: ((p: fs.PathLike, ...args: any[]) => { opened.push(String(p)); return (fs.openSync as any)(p, ...args); }) as typeof fs.openSync,
    });
    const { data } = load(ctx, { io: refusalIO });
    for (const name of ["stage.json", "helper.json", "started.json", "probe.json"]) {
      expect(data.files[name].status).toBe("unsupported-object"); expect(opened).not.toContain(path.join(ctx.root, name));
    }
    const alias = path.join(ctx.root, "alias"); fs.symlinkSync(ctx.root, alias);
    const other = load({ ...ctx, root: alias }).data; expect(other.files["stage.json"].status).toBe("unsupported-object");
    expect(other.files["stage.json"].reason).toBe("symlink-ancestor");
  }));

  it("uses no-follow/nonblocking opens and refuses descriptor identity races without reading", () => fixture(ctx => {
    const file = path.join(ctx.root, "stage.json"); fs.writeFileSync(file, '{"stage":"one"}');
    let raced = false, reads = 0, closed = 0;
    const racedIO = io({
      openSync: ((p: fs.PathLike, flags: number, mode?: number) => {
        if (String(p) === file) {
          expect(flags & fs.constants.O_NOFOLLOW).not.toBe(0); expect(flags & fs.constants.O_NONBLOCK).not.toBe(0);
          fs.renameSync(file, file + ".original"); fs.writeFileSync(file, '{"stage":"two"}'); raced = true;
        }
        return fs.openSync(p, flags, mode);
      }) as typeof fs.openSync,
      readSync: ((...args: any[]) => { if (raced && closed === 0) reads++; return (fs.readSync as any)(...args); }) as typeof fs.readSync,
      closeSync: (fd: number) => { if (raced) closed++; return fs.closeSync(fd); },
    });
    const { data } = load(ctx, { io: racedIO }); expect(data.files["stage.json"].status).toBe("unstable"); expect(data.files["stage.json"].reason).toBe("descriptor-identity-changed");
    expect(reads).toBe(0); expect(closed).toBeGreaterThan(0);
  }));

  it("retains same-length mutations as unstable bytes, not full-file certification", () => fixture(ctx => {
    const file = path.join(ctx.root, "helper.json"); fs.writeFileSync(file, '{"n":1}'); let targetFD = -1, mutated = false;
    const { data } = load(ctx, { io: io({
      openSync: ((p: fs.PathLike, flags: number, mode?: number) => { const fd = fs.openSync(p, flags, mode); if (String(p) === file) targetFD = fd; return fd; }) as typeof fs.openSync,
      readSync: ((...args: any[]) => { const n = (fs.readSync as any)(...args); if (args[0] === targetFD && !mutated) { fs.writeFileSync(file, '{"n":2}'); mutated = true; } return n; }) as typeof fs.readSync,
    }) });
    const observed = data.files["helper.json"]; expect(mutated).toBe(true); expect(observed.status).toBe("unstable"); expect(observed.fullFileSha256).toBeNull();
    expect(Buffer.from(observed.rawBase64, "base64").toString()).toBe('{"n":1}'); expect(observed.content.json).toBe("not-evaluated-incomplete");
  }));

  it("keeps exact stat identities and refuses full hashes when high-resolution stability is unsupported", () => fixture(ctx => {
    const file = path.join(ctx.root, "helper.json"); fs.writeFileSync(file, '{}'); let targetFD = -1;
    const { data } = load(ctx, { io: io({
      openSync: ((p: fs.PathLike, ...a: any[]) => { const fd = (fs.openSync as any)(p, ...a); if (String(p) === file) targetFD = fd; return fd; }) as typeof fs.openSync,
      fstatSync: ((fd: number, ...a: any[]) => { const stat = (fs.fstatSync as any)(fd, ...a); if (fd === targetFD) { delete stat.mtimeNs; delete stat.ctimeNs; } return stat; }) as typeof fs.fstatSync,
      closeSync: fd => { fs.closeSync(fd); if (fd === targetFD) targetFD = -1; },
    }) });
    const observed = data.files["helper.json"];
    for (const field of ["dev", "ino", "size", "nlink", "mtimeNs", "ctimeNs"]) expect(observed.before[field]).toMatch(/^\d+$/);
    expect(observed.reason).toBe("timestamp-precision-unavailable"); expect(observed.stability).toBe("unknown-timestamp-precision");
    expect(observed.fullFileSha256).toBeNull(); expect(observed.capturedSha256.value).toBe(digest('{}')); expect(Buffer.from(observed.rawBase64, "base64").toString()).toBe('{}');
  }));

  it("retains partial bytes/read errors and closes every acquired descriptor", () => fixture(ctx => {
    fs.writeFileSync(path.join(ctx.root, "helper.json"), '{"record":true}');
    const acquired = new Set<number>(), target = new Set<number>(); let reads = 0;
    const { data } = load(ctx, { io: io({
      openSync: ((p: fs.PathLike, ...a: any[]) => { const fd = (fs.openSync as any)(p, ...a); acquired.add(fd); if (String(p).endsWith("/helper.json")) target.add(fd); return fd; }) as typeof fs.openSync,
      readSync: ((fd: number, buffer: Buffer, offset: number, length: number, position: number) => {
        if (target.has(fd)) { reads++; if (reads > 1) throw Object.assign(new Error("sensitive error text not retained"), { code: "EIO" }); length = 3; }
        return fs.readSync(fd, buffer, offset, length, position);
      }) as typeof fs.readSync,
      closeSync: (fd: number) => { acquired.delete(fd); target.delete(fd); fs.closeSync(fd); },
    }) });
    expect(acquired.size).toBe(0); expect(data.files["helper.json"].status).toBe("read-error"); expect(data.files["helper.json"].errorCode).toBe("EIO");
    expect(Buffer.from(data.files["helper.json"].rawBase64, "base64").toString()).toBe('{"r'); expect(data.files["helper.json"].fullFileSha256).toBeNull();
    expect(JSON.stringify(data)).not.toContain("sensitive error text");
  }));

  it("reports metadata/open/recheck/close errors without leaks or throwing", () => fixture(ctx => {
    fs.writeFileSync(path.join(ctx.root, "helper.json"), '{}');
    for (const operation of ["openSync", "fstatSync", "lstatSync", "closeSync"] as const) {
      const acquired = new Set<number>(); let target = -1;
      const hooksIO = io({ openSync: ((p: fs.PathLike, ...a: any[]) => { if (String(p).endsWith("/helper.json") && operation === "openSync") throw Object.assign(new Error("open"), { code: "EACCES" }); const fd = (fs.openSync as any)(p, ...a); acquired.add(fd); if (String(p).endsWith("/helper.json")) target = fd; return fd; }) as typeof fs.openSync,
        closeSync: (fd: number) => { fs.closeSync(fd); acquired.delete(fd); const wasTarget = fd === target; if (wasTarget) target = -1; if (wasTarget && operation === "closeSync") throw Object.assign(new Error("closed then failed"), { code: "EIO" }); },
        fstatSync: ((fd: number, ...a: any[]) => { if (fd === target && operation === "fstatSync") throw Object.assign(new Error("stat"), { code: "EIO" }); return (fs.fstatSync as any)(fd, ...a); }) as typeof fs.fstatSync,
        lstatSync: ((p: fs.PathLike, ...a: any[]) => { if (String(p).endsWith("/helper.json") && operation === "lstatSync") throw Object.assign(new Error("stat"), { code: "EACCES" }); return (fs.lstatSync as any)(p, ...a); }) as typeof fs.lstatSync });
      const { data } = load(ctx, { io: hooksIO }); expect(data.files["helper.json"].status).toBe("read-error"); expect(acquired.size).toBe(0);
    }
  }));

  it("publishes only complete closed bytes atomically and exclusively with partial writes", () => fixture((ctx, out) => {
    const fds = new Set<number>(); let pending: number | undefined, temp = "", writes = 0, linked = false;
    const hooksIO = io({
      openSync: ((p: fs.PathLike, flags: number, mode?: number) => { const fd = fs.openSync(p, flags, mode); fds.add(fd); if (String(p).endsWith(".tmp")) { pending = fd; temp = String(p); expect(flags & fs.constants.O_EXCL).not.toBe(0); expect(mode).toBe(0o600); } return fd; }) as typeof fs.openSync,
      closeSync: fd => { fs.closeSync(fd); fds.delete(fd); },
      writeSync: ((fd: number, buffer: Buffer, offset: number, length: number) => { writes++; expect(fd).toBe(pending); expect(receipts(out)).toEqual([]); return fs.writeSync(fd, buffer, offset, Math.min(length, 17)); }) as typeof fs.writeSync,
      linkSync: (a, b) => { expect(fds.has(pending!)).toBe(false); expect(String(a)).toBe(temp); expect(fs.existsSync(b)).toBe(false); expect(() => JSON.parse(fs.readFileSync(a, "utf8"))).not.toThrow(); fs.linkSync(a, b); linked = true; },
    });
    const { result } = load(ctx, { io: hooksIO }); expect(linked).toBe(true); expect(writes).toBeGreaterThan(1); expect(fds.size).toBe(0); expect(fs.existsSync(temp)).toBe(false); expect(receipts(out)).toEqual([path.basename(result.path!)]);
  }));

  it("preserves preexisting final/temp files, symlink/hardlink aliases and collision targets", () => fixture((ctx, out) => {
    const nonce = () => "fixed-collision-control";
    const first = load(ctx, { nonce }); const before = fs.readFileSync(first.result.path!);
    expect(capture(ctx, { nonce }).status).toBe("failed"); expect(fs.readFileSync(first.result.path!)).toEqual(before);
    fs.unlinkSync(first.result.path!); fs.writeFileSync(path.join(ctx.root, "foreign"), "foreign-bytes"); fs.linkSync(path.join(ctx.root, "foreign"), first.result.path!);
    expect(capture(ctx, { nonce }).status).toBe("failed"); expect(fs.readFileSync(path.join(ctx.root, "foreign"), "utf8")).toBe("foreign-bytes");
    fs.unlinkSync(first.result.path!); fs.symlinkSync(path.join(ctx.root, "foreign"), first.result.path!);
    expect(capture(ctx, { nonce }).status).toBe("failed"); expect(fs.lstatSync(first.result.path!).isSymbolicLink()).toBe(true);
    fs.unlinkSync(first.result.path!);
    const tmp = path.join(out, `.recovery-execution-failure-${process.pid}-${nonce()}.tmp`); fs.symlinkSync(path.join(ctx.root, "foreign"), tmp);
    expect(capture(ctx, { nonce }).status).toBe("failed"); expect(fs.lstatSync(tmp).isSymbolicLink()).toBe(true); expect(fs.readFileSync(path.join(ctx.root, "foreign"), "utf8")).toBe("foreign-bytes");
    fs.unlinkSync(tmp); fs.linkSync(path.join(ctx.root, "foreign"), tmp); expect(capture(ctx, { nonce }).status).toBe("failed"); expect(fs.readFileSync(tmp, "utf8")).toBe("foreign-bytes");
  }));

  it("handles serialization, output size, mkdir/write/close/link failures and owned-temp cleanup", () => fixture((ctx, out) => {
    const acquired = new Set<number>();
    for (const phase of ["serialize", "oversize", "mkdir", "write", "zero-write", "close", "link"]) {
      const hooks: FailureCaptureHooks = { io: io({
        mkdirSync: ((...a: any[]) => { if (phase === "mkdir") throw new Error("mkdir fail"); return (fs.mkdirSync as any)(...a); }) as typeof fs.mkdirSync,
        openSync: ((...a: any[]) => { const fd = (fs.openSync as any)(...a); acquired.add(fd); return fd; }) as typeof fs.openSync,
        writeSync: ((...a: any[]) => { if (phase === "write") throw new Error("write fail"); if (phase === "zero-write") return 0; return (fs.writeSync as any)(...a); }) as typeof fs.writeSync,
        closeSync: fd => { fs.closeSync(fd); acquired.delete(fd); if (phase === "close") throw new Error("closed then failed"); },
        linkSync: (a, b) => { if (phase === "link") throw new Error("link fail"); fs.linkSync(a, b); },
      }), serialize: phase === "serialize" ? () => { throw new Error("serialization failure"); } : phase === "oversize" ? () => "x".repeat(limits.receiptBytes + 1) : undefined };
      const result = capture(ctx, hooks); expect(result.status).toBe("failed"); expect(acquired.size).toBe(0);
      expect(receipts(out)).toEqual([]); if (fs.existsSync(out)) expect(fs.readdirSync(out)).toEqual([]);
    }
  }));

  it("keeps exact-boundary files complete and bounds existing streams and argument metadata", () => fixture(ctx => {
    // Valid JSON whose encoded bytes land exactly on the documented file bound.
    const bytes = Buffer.from('"' + "x".repeat(limits.fileBytes - 2) + '"');
    fs.writeFileSync(path.join(ctx.root, "helper.json"), bytes);
    const { data } = load({ ...ctx, stdout: "é�".repeat(limits.textBytes), stderr: "z".repeat(limits.textBytes + 1), args: Array(10).fill("q".repeat(5000)),
      executable: "x".repeat(5000), host: { ...ctx.host, signalCode: "q".repeat(100) } });
    const helper = data.files["helper.json"]; expect(helper.capturedBytes).toBe(limits.fileBytes); expect(helper.eofObserved).toBe(true);
    expect(helper.truncated).toBe(false); expect(helper.fullFileSha256).toBe(digest(bytes)); expect(helper.content.json).toBe("valid-json");
    for (const stream of [data.fixture.stdout, data.fixture.stderr]) { expect(stream.capturedBytes).toBe(limits.textBytes); expect(stream.truncated).toBe(true); expect(Buffer.from(stream.rawBase64, "base64").length).toBe(limits.textBytes); }
    expect(data.fixture.requestedArgs).toHaveLength(8); expect(data.fixture.requestedArgsTruncated).toBe(true); expect(data.fixture.requestedArgs.every((arg: string) => arg.length === 4096)).toBe(true);
    expect(data.fixture.requestedExecutable).toHaveLength(4096); expect(data.fixture.requestedExecutableTruncated).toBe(true);
    expect(data.fixture.host.signalCode).toHaveLength(64); expect(data.fixture.host.signalCodeTruncated).toBe(true);
  }));

  for (const stream of ["stdout", "stderr"] as const) {
    it(`retains the exact UTF-8 ${stream} byte prefix across code-unit and byte cutoffs`, () => fixture(ctx => {
      // Run the demonstrated negative first, retaining the contract-correct
      // digest even against the old implementation. A partial final UTF-8
      // sequence belongs in rawBase64; do not trim it to a character boundary.
      const splitPair = "a".repeat(limits.textBytes - 1) + "😀" + "tail";
      const observed = load({ ...ctx, [stream]: splitPair }).data.fixture[stream];
      expect(observed.capturedSha256.value).toBe("ed423d90f254819bd19b4c7d7d39cb68254a093db095097d33f3025b1c194d1b");
      expectStream(observed, splitPair);
      expect(Buffer.from(observed.rawBase64, "base64").at(-1)).toBe(0xf0);
      for (const suffix of ["😀", "é", "漢", "\uFFFD"]) for (const offset of [-4, -3, -2, -1, 0, 1]) {
        if (suffix === "😀" && offset === -1) continue; // already checked above
        const value = "a".repeat(limits.textBytes + offset) + suffix + "tail";
        expectStream(load({ ...ctx, [stream]: value }).data.fixture[stream], value);
      }
    }));

    it(`preserves empty, ASCII, multilingual, literal U+FFFD and unpaired-surrogate ${stream}`, () => fixture(ctx => {
      for (const value of ["", "short ASCII\n", "é漢字 😀 \uFFFD\n", "literal \uFFFD", "lone-high:\uD83D", "lone-low:\uDE00",
        "x".repeat(limits.textBytes), "x".repeat(limits.textBytes + 1),
        "a".repeat(limits.textBytes - 1) + "\uD83Dtail", "a".repeat(limits.textBytes - 1) + "\uDE00tail",
        "a".repeat(limits.textBytes - 1) + "\uD83D\uD83D\uDE00tail", "a".repeat(limits.textBytes - 2) + "\uD83D\uDE00\uDE00tail",
        "é".repeat(limits.textBytes / 2 - 1) + "😀tail", "é漢😀\uFFFD\n".repeat(limits.textBytes)]) {
        expectStream(load({ ...ctx, [stream]: value }).data.fixture[stream], value);
      }
    }));

    for (const [kind, unit] of [["high", "\uD83D"], ["low", "\uDE00"]] as const) {
      it(`reports exact UTF-8 totals and truncation for repeated lone-${kind} ${stream}`, () => fixture(ctx => {
        const repeats = Math.floor(limits.textBytes / 3);
        // Encode controlled COMPLETE inputs, not the implementation's counting
        // algorithm. Keep the false-not-truncated regression first, while
        // collecting every variant before assertions against the collector.
        const values = [unit.repeat(limits.textBytes / 2), "lone:" + unit,
          unit.repeat(repeats), ...[1, 2, 3].map(n => unit.repeat(repeats) + "a".repeat(n)),
          unit.repeat(repeats + 1), unit.repeat(limits.textBytes / 2 + 1),
          "", "short ASCII\n", "\u007f\u0080\u07ff\u0800\uD7ff\uE000\uFFFF", "é漢字", "literal \uFFFD",
          "\uD83D\uDE00", "\uD83D\uD83D\uDE00", "\uDE00\uD83D", "é".repeat(limits.textBytes / 2)];
        const observations = values.map(value => {
          const complete = Buffer.from(value, "utf8"), encoded = new TextEncoder().encode(value);
          expect(new Uint8Array(complete)).toEqual(encoded);
          const { result, data } = load({ ...ctx, [stream]: value });
          return { value, receiptName: path.basename(result.path!), observed: data.fixture[stream],
            expected: { byteLength: complete.length, capturedBytes: Math.min(complete.length, limits.textBytes),
              truncated: complete.length > limits.textBytes } };
        });
        if (process.env.QMD_EA_UNIT_OUTPUT) fs.writeFileSync(path.join(process.env.QMD_EA_UNIT_OUTPUT, `metadata-${stream}-${kind}.json`), JSON.stringify({
          scope: "actual collector on authored strings; complete Buffer.from and TextEncoder agree", stream, kind, observations,
        }, null, 2));
        for (const { value, observed, expected } of observations) {
          expect(observed.truncated).toBe(expected.truncated);
          expectStream(observed, value);
        }
      }));
    }
  }

  it("keeps large supplied stream encodings and their buffers bounded", () => fixture(ctx => {
    const stdout = "é漢😀\uFFFD".repeat(256 * 1024), stderr = "z".repeat(2 * 1024 * 1024);
    const encodings: { codeUnits: number; allocatedBytes: number }[] = [];
    const originalFrom = Buffer.from; let reportAssembled = false;
    // Synchronous real-collector call-through only. Stop observing at the
    // serializer boundary, so this measures stream encodings, not receipt JSON.
    // Restore before assertions/awaits; no concurrent tests use this interposer.
    Buffer.from = ((input: unknown, ...args: unknown[]) => {
      const streamEncoding = !reportAssembled && typeof input === "string";
      if (streamEncoding && input.length > limits.textBytes + 1) throw new Error("unbounded stream encoding input");
      const bytes = Reflect.apply(originalFrom, Buffer, [input, ...args]) as Buffer;
      if (streamEncoding) encodings.push({ codeUnits: input.length, allocatedBytes: bytes.length });
      return bytes;
    }) as typeof Buffer.from;
    let observed: ReturnType<typeof load>;
    try { observed = load({ ...ctx, stdout, stderr }, { serialize: report => { reportAssembled = true; return JSON.stringify(report); } }); }
    finally { Buffer.from = originalFrom; }
    expect(encodings).toHaveLength(2);
    for (const encoding of encodings) {
      expect(encoding.codeUnits).toBeLessThanOrEqual(limits.textBytes + 1);
      expect(encoding.allocatedBytes).toBeLessThanOrEqual(3 * (limits.textBytes + 1));
    }
    expectStream(observed.data.fixture.stdout, stdout); expectStream(observed.data.fixture.stderr, stderr);
    if (process.env.QMD_EA_UNIT_OUTPUT) fs.writeFileSync(path.join(process.env.QMD_EA_UNIT_OUTPUT, "bounded-stream-encoding.json"), JSON.stringify({
      scope: "actual synchronous collector Buffer.from stream encoding inputs/returned buffers; not a whole-runtime heap bound",
      suppliedCodeUnits: { stdout: stdout.length, stderr: stderr.length }, encodings,
      expectedPrefixes: { stdout: observed.data.fixture.stdout.capturedSha256, stderr: observed.data.fixture.stderr.capturedSha256 },
    }, null, 2));
  }));

  it("does not repair invalid BOM/UTF-8, unsupported identity objects or unsafe GitHub identifiers", () => fixture(ctx => {
    fs.writeFileSync(path.join(ctx.root, "helper.json"), Buffer.from('\uFEFF{}'));
    fs.writeFileSync(path.join(ctx.root, "probe.json"), Buffer.from([0x22, 0xc0, 0xaf, 0x22]));
    fs.writeFileSync(path.join(ctx.root, "started.json"), '{"executable":"/authored/caller","versions":[]}');
    process.env.GITHUB_RUN_ID = "x".repeat(129); process.env.GITHUB_JOB = "line\nbreak"; delete process.env.GITHUB_RUN_ATTEMPT;
    const { data } = load(ctx);
    expect(data.files["helper.json"].content).toEqual({ utf8: "valid", json: "invalid-json" });
    expect(Buffer.from(data.files["helper.json"].rawBase64, "base64")).toEqual(Buffer.from('\uFEFF{}'));
    expect(data.files["probe.json"].content).toEqual({ utf8: "invalid", json: "invalid-utf8" });
    expect(data.callerObserved).toMatchObject({ status: "unknown", reason: "unrecognized-started-json-object" });
    expect(data.github).toEqual({ GITHUB_RUN_ID: null, GITHUB_JOB: null, GITHUB_RUN_ATTEMPT: null });
    expect(data.githubStatus).toEqual({ GITHUB_RUN_ID: { state: "omitted-invalid-or-over-limit", originalCharacters: 129 }, GITHUB_JOB: { state: "omitted-invalid-or-over-limit", originalCharacters: 10 }, GITHUB_RUN_ATTEMPT: { state: "absent", originalCharacters: null } });
  }));

  it("records disappearance and a symlink swap as unstable, without opening link targets", () => fixture(ctx => {
    for (const phase of ["before-open", "after-read", "symlink-before-open"]) {
      const target = path.join(ctx.root, "helper.json"); fs.writeFileSync(target, '{}'); let targetFD = -1, changed = false;
      const { data } = load(ctx, { io: io({
        openSync: ((p: fs.PathLike, flags: number, mode?: number) => {
          if (String(p) === target && phase !== "after-read") { fs.unlinkSync(target); if (phase === "symlink-before-open") fs.symlinkSync(path.join(ctx.root, "unread-target"), target); }
          const fd = fs.openSync(p, flags, mode); if (String(p) === target) targetFD = fd; return fd;
        }) as typeof fs.openSync,
        readSync: ((...a: any[]) => { const n = (fs.readSync as any)(...a); if (a[0] === targetFD && phase === "after-read" && !changed) { fs.unlinkSync(target); changed = true; } return n; }) as typeof fs.readSync,
        closeSync: fd => { fs.closeSync(fd); if (fd === targetFD) targetFD = -1; },
      }) });
      expect(data.files["helper.json"].status).toBe("unstable"); expect(data.files["helper.json"].fullFileSha256).toBeNull();
      if (phase === "after-read") expect(Buffer.from(data.files["helper.json"].rawBase64, "base64").toString()).toBe('{}');
      else expect(data.files["helper.json"].readBytes).toBe(0);
      if (fs.existsSync(target) || phase === "symlink-before-open") fs.unlinkSync(target);
    }
  }));

  it("reports an unremovable owned temp or unknown temp identity without touching foreign files", () => fixture((ctx, out) => {
    const result = capture(ctx, { io: io({ unlinkSync: p => { if (String(p).endsWith(".tmp")) throw Object.assign(new Error("unlink fail"), { code: "EACCES" }); fs.unlinkSync(p); } }) });
    expect(result.status).toBe("captured"); expect(result.temporaryCleanup).toBe("cleanup-error-owned-temp-may-remain");
    expect(receipts(out)).toHaveLength(1); expect(fs.readdirSync(out).filter(n => n.endsWith(".tmp"))).toHaveLength(1);
    for (const name of fs.readdirSync(out)) fs.unlinkSync(path.join(out, name));
    const acquired = new Set<number>(); let outputFD = -1;
    const failed = capture(ctx, { io: io({
      openSync: ((p: fs.PathLike, ...a: any[]) => { const fd = (fs.openSync as any)(p, ...a); acquired.add(fd); if (String(p).endsWith(".tmp")) outputFD = fd; return fd; }) as typeof fs.openSync,
      fstatSync: ((fd: number, ...a: any[]) => { if (fd === outputFD) throw Object.assign(new Error("stat fail"), { code: "EIO" }); return (fs.fstatSync as any)(fd, ...a); }) as typeof fs.fstatSync,
      closeSync: fd => { fs.closeSync(fd); acquired.delete(fd); },
    }) });
    expect(failed.status).toBe("failed"); expect(failed.temporaryCleanup).toBe("unknown-identity-retained"); expect(acquired.size).toBe(0);
    expect(receipts(out)).toEqual([]); expect(fs.readdirSync(out)).toHaveLength(1);
  }));

  it("never unlinks another inode replacing an owned temp and rejects unsafe output ancestors", () => fixture((ctx, out) => {
    let foreignTemp = "";
    const result = capture(ctx, { io: io({ writeSync: ((fd: number, ...a: any[]) => {
      const name = fs.readdirSync(out).find(n => n.endsWith(".tmp"))!; foreignTemp = path.join(out, name);
      fs.unlinkSync(foreignTemp); fs.writeFileSync(foreignTemp, "not-call-owned"); throw new Error("replacement race");
    }) as typeof fs.writeSync }) });
    expect(result.status).toBe("failed"); expect(fs.readFileSync(foreignTemp, "utf8")).toBe("not-call-owned"); expect(receipts(out)).toEqual([]);
    const target = path.join(ctx.root, "target"); fs.mkdirSync(target); const alias = path.join(ctx.root, "output-alias"); fs.symlinkSync(target, alias);
    process.env.QMD_RECOVERY_BOOTSTRAP_DIR = path.join(alias, "nested"); expect(capture(ctx).status).toBe("failed"); expect(fs.readdirSync(target)).toEqual([]);
    process.env.QMD_RECOVERY_BOOTSTRAP_DIR = "relative-output"; expect(capture(ctx, noWork()).status).toBe("failed");
  }));
});
