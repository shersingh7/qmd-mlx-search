/** Opt-in, AFTER-canonical, model-free observation. Never cleanup qualification.
 * The production caller, interpreter, argv flags and 200/800ms clocks are unchanged.
 * Python instrumentation adds imports/IO and runs warmed: it cannot explain a
 * historical cold invocation. Each process has its own relative clock domain.
 */
import childProcess, { type SpawnOptions } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, constants, fstatSync, lstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { release } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { TextDecoder } from "node:util";
import type { RecoveryProcessResult } from "../../scripts/phase5_e2e_runner.js";
import { observeOwnedChild } from "./recovery-bootstrap-diagnostics.mjs";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const SOURCE = path.join(ROOT, "scripts/phase5_recovery_process.py");
export const LIMITS = Object.freeze({ records: 32, sidecarBytes: 16_384, jsonBytes: 262_144, cases: [200, 5000] as const });
const EVENTS = ["python-entry", "source-verified", "module-loaded", "prepared-before-send", "prepared-sent", "admission-read", "execution-read",
  "cancel-read", "parent-eof", "ready-sent", "worker-exit-sent", "cleanup-sent", "not-started-sent", "supervisor-error-sent", "wrapper-return"] as const;
type Runtime = "node" | "bun";
type Event = typeof EVENTS[number];
type PythonRecord = { event: Event; seq: number; pid: number; tMs: number; version?: string; executable?: string; budgetMs?: number };
type SidecarStatus = "absent" | "partial" | "complete" | "malformed" | "oversized" | "unsafe" | "io-error";
type Sidecar = { status: SidecarStatus; bytes: number; trailingBytes: number; records: PythonRecord[];
  selectedInterpreter: { version: string; executable: string; pid: number } | null; milestones: Record<Event, number | null> };
const hash = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
const object = (value: unknown): Record<string, unknown> => value !== null && typeof value === "object" ? value as Record<string, unknown> : {};

export function actualHost(expected?: string) {
  if (expected !== undefined && expected !== "node" && expected !== "bun") throw new Error("checked-in runtime enum required");
  const bun = process.versions.bun ?? null;
  const runtime: Runtime = bun === null ? "node" : "bun";
  if (expected !== undefined && expected !== runtime) throw new Error("actual host mismatch");
  return { runtime, executable: process.execPath, node: process.versions.node, bun, uv: process.versions.uv ?? null,
    platform: process.platform, arch: process.arch, kernel: release(), pid: process.pid };
}

function sidecar(status: SidecarStatus, bytes = 0, records: PythonRecord[] = [], trailingBytes = 0): Sidecar {
  const milestones = Object.fromEntries(EVENTS.map(event => [event, records.find(item => item.event === event)?.tMs ?? null])) as Record<Event, number | null>;
  const entry = records.find(item => item.event === "python-entry");
  return { status, bytes, trailingBytes, records, selectedInterpreter: entry ? { version: entry.version!, executable: entry.executable!, pid: entry.pid } : null, milestones };
}

/** Retain only a validated prefix; neither partial EOF nor malformed data means success. */
export function parsePythonSidecar(text: string | null): Sidecar {
  if (text === null) return sidecar("absent");
  const bytes = Buffer.byteLength(text);
  if (bytes > LIMITS.sidecarBytes) return sidecar("oversized", bytes);
  const parts = text.split("\n");
  const tail = parts.pop()!;
  const records: PythonRecord[] = [];
  for (const line of parts) {
    try {
      if (records.length >= LIMITS.records || Buffer.byteLength(line) > 2048) throw new Error();
      const value: unknown = JSON.parse(line);
      const item = object(value);
      const keys = ["event", "seq", "pid", "tMs", ...(item.event === "python-entry" ? ["version", "executable"] : []), ...(item.event === "admission-read" ? ["budgetMs"] : [])];
      if (!EVENTS.includes(item.event as Event) || Object.keys(item).length !== keys.length || Object.keys(item).some(key => !keys.includes(key))
        || item.seq !== records.length || !Number.isSafeInteger(item.pid) || (item.pid as number) < 1
        || typeof item.tMs !== "number" || !Number.isFinite(item.tMs) || item.tMs < 0
        || (records.length === 0 && item.event !== "python-entry")
        || (records.length > 0 && (item.pid !== records[0]!.pid || item.tMs < records[records.length - 1]!.tMs))
        || records.some(record => record.event === "wrapper-return")
        || (item.event === "python-entry" && (records.length !== 0 || typeof item.version !== "string" || item.version.length > 512
          || typeof item.executable !== "string" || item.executable.length > 1024 || !item.executable.startsWith("/")))
        || (item.event === "admission-read" && (!Number.isSafeInteger(item.budgetMs) || (item.budgetMs as number) < 1 || (item.budgetMs as number) > 5000))) throw new Error();
      records.push(item as PythonRecord);
    } catch { return sidecar("malformed", bytes, records, Buffer.byteLength(tail)); }
  }
  return sidecar(tail.length || records.at(-1)?.event !== "wrapper-return" ? "partial" : "complete", bytes, records, Buffer.byteLength(tail));
}

/** O_NONBLOCK prevents an accidental special-file open from waiting. No links. */
export function readPythonSidecar(file: string): Sidecar {
  let fd: number | undefined;
  try {
    const before = lstatSync(file);
    if (!before.isFile() || before.nlink !== 1) return sidecar("unsafe");
    if (before.size > LIMITS.sidecarBytes) return sidecar("oversized", before.size);
    fd = openSync(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const opened = fstatSync(fd);
    if (!opened.isFile() || opened.nlink !== 1 || opened.dev !== before.dev || opened.ino !== before.ino) return sidecar("unsafe");
    const buffer = Buffer.alloc(LIMITS.sidecarBytes + 1);
    let bytes = 0;
    while (bytes < buffer.length) {
      const count = readSync(fd, buffer, bytes, buffer.length - bytes, null);
      if (count === 0) break;
      bytes += count;
    }
    if (bytes > LIMITS.sidecarBytes) return sidecar("oversized", bytes);
    const raw = buffer.subarray(0, bytes);
    const trailingBytes = bytes - raw.lastIndexOf(10) - 1;
    const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
    let text = "", start = 0;
    for (let end = raw.indexOf(10); end !== -1; end = raw.indexOf(10, start)) {
      try { text += decoder.decode(raw.subarray(start, end)) + "\n"; }
      catch { return sidecar("malformed", bytes, parsePythonSidecar(text).records, trailingBytes); }
      start = end + 1;
    }
    // An unterminated row (including a cut multibyte sequence) is raw partial
    // evidence only. Never decode it with replacement or use it as a record.
    const parsed = parsePythonSidecar(text);
    return sidecar(parsed.status === "malformed" ? "malformed" : trailingBytes ? "partial" : parsed.status, bytes, parsed.records, trailingBytes);
  } catch (error) { return sidecar((error as NodeJS.ErrnoException).code === "ENOENT" ? "absent" : "io-error"); }
  finally { if (fd !== undefined) closeSync(fd); }
}

/** Create components without following a pre-existing symlink. Leaf is owned. */
function directoryTree(directory: string): void {
  if (!path.isAbsolute(directory) || path.resolve(directory) === path.parse(directory).root) throw new Error("explicit absolute non-root directory required");
  let current = path.parse(directory).root;
  for (const part of path.resolve(directory).slice(current.length).split(path.sep)) {
    current = path.join(current, part);
    try { mkdirSync(current, { mode: 0o700 }); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; }
    const info = lstatSync(current);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("unsafe output directory or symlink");
  }
  if (process.getuid && lstatSync(directory).uid !== process.getuid()) throw new Error("output directory must be call-user owned");
}
export function createOwnedRun(directory: string, runtime: Runtime): string {
  actualHost(runtime);
  directoryTree(directory);
  return mkdtempSync(path.join(directory, `${runtime}-`));
}

export function writeExclusiveJson(file: string, value: unknown): void {
  const text = JSON.stringify(value, null, 2) + "\n";
  if (Buffer.byteLength(text) > LIMITS.jsonBytes) throw new Error("bounded JSON output exceeded");
  const fd = openSync(file, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  try { writeFileSync(fd, text); } finally { closeSync(fd); }
}

/** Fixed allowlist. PATH is the only ambient executable-discovery field. */
export function isolatedEnvironment(directory: string): NodeJS.ProcessEnv {
  directoryTree(directory);
  const env: NodeJS.ProcessEnv = { PATH: process.env.PATH ?? "/usr/bin:/bin", LANG: "C.UTF-8", CI: "1", REQUIRE_BUN: "1", QMD_LLAMA_GPU: "false",
    HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_DATASETS_OFFLINE: "1", UV_OFFLINE: "1", PYTHONDONTWRITEBYTECODE: "1", PYTHONNOUSERSITE: "1",
    TOKENIZERS_PARALLELISM: "false", npm_config_offline: "true" };
  const folders = { HOME: "home", HERMES_HOME: "hermes", HERMES_DIR: "hermes", HERMES_KNOWLEDGE_DIR: "knowledge", KNOWLEDGE_DIR: "knowledge",
    XDG_CONFIG_HOME: "config", XDG_CACHE_HOME: "cache", XDG_DATA_HOME: "data", XDG_STATE_HOME: "state", HF_HOME: "hf", HF_HUB_CACHE: "hf/hub",
    TRANSFORMERS_CACHE: "hf/transformers", TMPDIR: "tmp", TMP: "tmp", TEMP: "tmp", BUN_INSTALL_CACHE_DIR: "bun", npm_config_cache: "npm", UV_CACHE_DIR: "uv" };
  for (const [key, folder] of Object.entries(folders)) { env[key] = path.join(directory, folder); directoryTree(env[key]!); }
  return env;
}

/** The current caller has no public override option: use its existing fixture
 * seam, a synchronous, exact-launch-only argv replacement; restore immediately.
 * No flags/config/stdio/owner methods are replaced, and no protocol bytes added.
 */
export function replaceSupervisorArgv(argv: readonly string[], source: string, wrapper: string): string[] {
  if (argv.length !== 5 || argv[0] !== "/usr/bin/python3" || argv[1] !== "-S" || argv[2] !== "-u" || argv[3] !== source
    || !path.isAbsolute(wrapper)) throw new Error("unexpected supervisor argv");
  return [...argv.slice(0, 3), wrapper, argv[4]!];
}

export function makeSupervisorWrapper({ source, expectedSha256, sidecar: file }: { source: string; expectedSha256: string; sidecar: string }): string {
  if (!path.isAbsolute(source) || !path.isAbsolute(file) || !/^[0-9a-f]{64}$/.test(expectedSha256)) throw new Error("invalid owned wrapper binding");
  return `# Test-only self-observation; no shared-clock or cold-causality claim.
import time
origin = time.monotonic()
import os, sys, json
owner = os.getpid()
record_fd = os.open(${JSON.stringify(file)}, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
os.set_inheritable(record_fd, False)
sequence = 0
written_bytes = 0
def witness(event, **extra):
    global sequence, written_bytes
    if os.getpid() != owner or sequence >= ${LIMITS.records}:
        return
    value = dict(event=event, seq=sequence, pid=owner, tMs=(time.monotonic() - origin) * 1000)
    value.update(extra)
    data = (json.dumps(value, separators=(',', ':')) + '\\n').encode()
    if written_bytes + len(data) > ${LIMITS.sidecarBytes}:
        return
    sequence += 1
    written_bytes += len(data)
    try:
        while data:
            count = os.write(record_fd, data)
            data = data[count:]
    except OSError:
        pass  # Evidence can remain partial; it never changes a cleanup receipt.
witness('python-entry', version=sys.version[:512], executable=sys.executable[:1024])
# Add only isolated offline/cache defaults missing from the unchanged caller env.
for key, folder in dict(HERMES_DIR='hermes', HERMES_KNOWLEDGE_DIR='knowledge', KNOWLEDGE_DIR='knowledge', XDG_DATA_HOME='data', XDG_STATE_HOME='state',
                        HF_HOME='hf', HF_HUB_CACHE='hf/hub', TRANSFORMERS_CACHE='hf/transformers', UV_CACHE_DIR='uv', BUN_INSTALL_CACHE_DIR='bun', npm_config_cache='npm').items():
    os.environ.setdefault(key, os.path.join(os.environ['HOME'], folder))
    os.makedirs(os.environ[key], mode=0o700, exist_ok=True)
os.environ.update(QMD_LLAMA_GPU='false', REQUIRE_BUN='1', UV_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
import hashlib
source = ${JSON.stringify(source)}
with open(source, 'rb') as source_file:
    code = source_file.read()
if hashlib.sha256(code).hexdigest() != ${JSON.stringify(expectedSha256)}:
    raise RuntimeError('production source binding mismatch')
witness('source-verified')
sys.path.insert(0, os.path.dirname(source))
sys.argv[0] = source
scope = dict(__name__='owned_startup_observation')
exec(compile(code, source, 'exec'), scope)
witness('module-loaded')
original_send = scope['send']
def observed_send(message):
    kind = message.get('type')
    if kind == 'prepared':
        witness('prepared-before-send')
    original_send(message)
    event = {'prepared':'prepared-sent', 'ready':'ready-sent', 'exit':'worker-exit-sent', 'cleanup':'cleanup-sent',
             'not-started':'not-started-sent', 'supervisor-error':'supervisor-error-sent'}.get(kind)
    if event:
        witness(event)
scope['send'] = observed_send
original_read = scope['read_control']
def observed_read(limit=4096):
    # One original read only, returned unchanged. Never consume a second byte,
    # read the child lease, inspect a pipe elsewhere, or synthesize S/E.
    data = original_read(limit)
    if data == b'':
        witness('parent-eof')
    elif data == b'E':
        witness('execution-read')
    elif data and (b'T' in data or b'K' in data):
        witness('cancel-read')
    elif data and data.startswith(b'S') and data.endswith(b'\\n') and data[1:-1].isdigit():
        witness('admission-read', budgetMs=int(data[1:-1]))
    return data
scope['read_control'] = observed_read
try:
    try:
        scope['main']()
    except BaseException as failure:
        # Same error boundary as the verified production __main__; original send.
        try:
            scope['send']({'type':'supervisor-error', 'error':f'Recovery supervisor failed: {failure}'})
        except OSError:
            pass
        sys.exit(1)
finally:
    witness('wrapper-return')
    os.close(record_fd)
`;
}

type Invocation = { originalArgv: string[]; actualArgv: string[]; restored: boolean };
type NodeObservation = ReturnType<ReturnType<typeof observeOwnedChild>["finish"]>;
function launchWithOverride(wrapper: string, start: () => Promise<RecoveryProcessResult>): {
  result: Promise<RecoveryProcessResult>; invocation: Invocation; finishNode: (result: RecoveryProcessResult) => NodeObservation | null; releaseNode: () => void;
} {
  const invocation: Invocation = { originalArgv: [], actualArgv: [], restored: false };
  let trace: ReturnType<typeof observeOwnedChild> | undefined;
  // Transfer the call-local handle before releasing it: finish already disposes
  // in its own finally. The asynchronous finally must not release it twice.
  const releaseNode = () => { const owned = trace; trace = undefined; owned?.dispose(); };
  const finishNode = (result: RecoveryProcessResult) => { const owned = trace; trace = undefined; return owned?.finish(result) ?? null; };
  const capture = (argv: readonly string[]) => {
    if (invocation.originalArgv.length) throw new Error("more than one owned supervisor launch");
    invocation.originalArgv = [...argv]; invocation.actualArgv = replaceSupervisorArgv(argv, SOURCE, wrapper);
    return invocation.actualArgv;
  };
  let restore: () => void;
  if (process.versions.bun) {
    // Same supported native-Bun fixture seam, not a second FD/process owner.
    const bun = (globalThis as unknown as { Bun: { spawn: (argv: string[], options: unknown) => unknown } }).Bun;
    const original = bun.spawn;
    bun.spawn = function(argv, options) {
      return original.call(bun, argv[0] === "/usr/bin/python3" && argv[3] === SOURCE ? capture(argv) : argv, options);
    };
    restore = () => { bun.spawn = original; };
  } else {
    const original = childProcess.spawn;
    const replacement = (executable: string, args: readonly string[] | SpawnOptions = [], options?: SpawnOptions) => {
      if (Array.isArray(args) && executable === "/usr/bin/python3" && args[2] === SOURCE) {
        const argv = capture([executable, ...args]);
        const child = original(argv[0]!, argv.slice(1), options!);
        trace = observeOwnedChild(child);
        return child;
      }
      return Array.isArray(args) ? original(executable, args, options!) : original(executable, args as SpawnOptions);
    };
    childProcess.spawn = replacement as typeof original; syncBuiltinESMExports();
    restore = () => { childProcess.spawn = original; syncBuiltinESMExports(); };
  }
  try { return { result: start(), invocation, finishNode, releaseNode }; }
  catch (error) {
    try { releaseNode(); } catch { /* Preserve the primary start failure; no successful observation is published. */ }
    throw error;
  }
  finally { restore(); invocation.restored = true; }
}

/** PID-existence sample only; not an identity/ownership claim or signal authority. */
function pidPresent(pid: number | undefined): boolean | null {
  if (pid === undefined) return null;
  try { process.kill(pid, 0); return true; }
  catch (error) { return (error as NodeJS.ErrnoException).code === "ESRCH" ? false : null; }
}
export function classifyObservation(result: RecoveryProcessResult, workValidated: boolean): string {
  if (result.cleanup.containmentVerified) return "unexpected-containment-claim";
  if (result.timedOut) return "timed-out";
  if (result.launch.state === "not-started" || result.launch.state === "not-attempted") return result.launch.state;
  if (!result.launch.supervisorReaped || !result.cleanup.childReaped || !result.cleanup.groupTerminated || !result.cleanup.sessionTerminated
    || !result.cleanup.pipesClosed || result.cleanup.forcedPipeClose) return "incomplete-cleanup";
  if (!workValidated || result.exitCode !== 0 || result.signal !== null || result.report?.status !== "completed") return "work-not-validated";
  return "authored-work-observed-uncontained";
}

const DOC = "# Public startup observation\n\nSynthetic authored work only; no database, model, server, or personal index.\n";
async function runCase(directory: string, timeoutMs: 200 | 5000) {
  mkdirSync(directory, { mode: 0o700 });
  const oldEnvironment = process.env;
  process.env = isolatedEnvironment(path.join(directory, "caller-environment"));
  try {
    const docs = path.join(directory, "public-docs"); mkdirSync(docs, { mode: 0o700 });
    const doc = path.join(docs, "public.md"); writeFileSync(doc, DOC, { flag: "wx", mode: 0o600 });
    const outputJson = path.join(directory, "worker-report.json");
    const dbPath = path.join(directory, "unused.sqlite");
    const workFile = path.join(directory, "authored-work.json");
    const wrapper = path.join(directory, "self-observing-supervisor.py");
    const pythonFile = path.join(directory, "python-self-observation.jsonl");
    const sourceSha256 = hash(readFileSync(SOURCE));
    writeFileSync(wrapper, makeSupervisorWrapper({ source: SOURCE, expectedSha256: sourceSha256, sidecar: pythonFile }), { flag: "wx", mode: 0o600 });
    // Literal authored JS, with call-owned paths serialized as data, no shell.
    const code = `const fs=require('node:fs'),crypto=require('node:crypto');
const bytes=fs.readFileSync(${JSON.stringify(doc)});
const report={status:'completed',pid:process.pid,dbPath:${JSON.stringify(dbPath)},action:'index',docsCompleted:1,chunksCommitted:0,totalVectors:0,checkpointStatus:null,consecutiveSequenceVerified:true,hashes:[]};
fs.writeFileSync(${JSON.stringify(workFile)},JSON.stringify({pid:process.pid,bytes:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex')}),{flag:'wx',mode:384});
fs.writeFileSync(${JSON.stringify(outputJson)},JSON.stringify(report),{flag:'wx',mode:384});`;
    const command = [process.execPath, "-e", code];
    // Importing the public caller does not invoke qualification or any model.
    const { executeRecoveryWorker } = await import("../../scripts/phase5_e2e_runner.js");
    const started = performance.now();
    const pending = launchWithOverride(wrapper, () => executeRecoveryWorker({ dbPath, docsDir: docs, outputJson,
      stateRoot: path.join(directory, "state"), timeoutMs, command }));
    let result: RecoveryProcessResult, returnedAtMs: number, nodeOwner: NodeObservation | null;
    let finished = false;
    try {
      result = await pending.result;
      returnedAtMs = performance.now() - started;
      nodeOwner = pending.finishNode(result);
      finished = true;
    } finally {
      if (finished) pending.releaseNode();
      else {
        // The caller/finish exception remains primary; collection stays incomplete
        // even if disposal also fails. Never signal or close the caller's child.
        try { pending.releaseNode(); } catch { /* No cleanup-success claim. */ }
      }
    }
    // Capture everything BEFORE any fixture cleanup. None is performed on the subject.
    const snapshot = { fixtureCleanupStarted: false, subjectFixtureSignals: 0, supervisorPidPresent: pidPresent(result.launch.supervisorPid),
      workerPidPresent: pidPresent(result.pid), pidCheckScope: "reported-PID-existence-only-not-kernel-identity", databaseCreated: (() => {
        try { lstatSync(dbPath); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
      })() };
    let workValidated = false;
    let work: unknown = null;
    try {
      if (lstatSync(workFile).isFile() && lstatSync(workFile).size <= 2048 && lstatSync(workFile).nlink === 1) {
        work = JSON.parse(readFileSync(workFile, "utf8"));
        const observed = object(work);
        workValidated = result.pid !== undefined && result.report !== undefined && observed.pid === result.pid
          && observed.bytes === Buffer.byteLength(DOC) && observed.sha256 === hash(DOC)
          && result.report.pid === result.pid && result.report.docsCompleted === 1 && !snapshot.databaseCreated;
      }
    } catch { /* Missing/partial work is not validated. Raw owned files are retained. */ }
    const observation = { label: timeoutMs === 200 ? "short-original-caller-inclusive-budget" : "separate-valid-positive-control-not-a-retry",
      timeoutMs, teardownMs: 800, productQualified: false, containmentQualification: "explicit-command-always-uncontained",
      callerClock: "this-host-performance-now-relative-to-before-real-call; production-clock-remains-internal",
      callerInvokedMs: 0, callerReturnedMs: returnedAtMs, unsupportedCallerMilestones: { preparedReceiptMs: null, remainingBudgetMs: null, sendSMs: null, sendEMs: null,
        cancellationMs: null, reason: "no production control-reader/writer/deadline hooks; Python read timestamps are not caller send/receipt timestamps" },
      invocation: pending.invocation, sourceSha256, wrapperSha256: hash(readFileSync(wrapper)), result, snapshot,
      work, workValidated, classification: classifyObservation(result, workValidated), python: readPythonSidecar(pythonFile), nodeOwner,
      nativeBunOwnerObservation: "unsupported; no extra reader, fd owner, or native owner instrumentation" };
    writeExclusiveJson(path.join(directory, "before-fixture-cleanup.json"), observation);
    return observation;
  } finally { process.env = oldEnvironment; }
}

export async function runStartupObservation(directory: string, expected: Runtime) {
  if (process.env.QMD_RECOVERY_STARTUP_OBSERVATION !== "1") throw new Error("explicit diagnostic opt-in required");
  const host = actualHost(expected);
  const owned = createOwnedRun(directory, expected);
  const sourceNames = ["scripts/phase5_e2e_runner.ts", "scripts/phase5_recovery_process.py", "scripts/phase5_recovery_worker.ts", "scripts/qmd_mlx/process_anchor.py",
    "test/helpers/recovery-bootstrap-diagnostics.mjs", "test/helpers/recovery-startup-observation.ts", "test/helpers/recovery-startup-observation.selftest.mjs", ".github/workflows/ci.yml", "package.json", "bun.lock"];
  const report = { schemaVersion: 1, scope: "after-canonical-warmed-instrumented-real-caller-observation", actualHost: host,
    productQualified: false, diagnosticExitZeroMeans: "observations-collected-not-product-or-C1-cleanup-qualification", coldCausality: "not-established",
    clocks: "Python and caller/Node-observer deltas are independent process-local domains; no cross-process epoch subtraction",
    instrumentationEffects: ["minimal-import Python entry, then extra offline cache setup, source hashing/compile, bounded sidecar IO and function wrappers",
      "original read_control called once and returned unchanged; no second control read, changed deadline, raw S200 or permission injection",
      "Node-only existing event observer attaches after spawn; native Bun owner observation remains unsupported"],
    sourceSha256: Object.fromEntries(sourceNames.map(name => [name, hash(readFileSync(path.join(ROOT, name)))])),
    cases: [] as Awaited<ReturnType<typeof runCase>>[], collectionErrors: [] as string[] };
  // Published first, so a later import/probe failure cannot erase all inventory.
  writeExclusiveJson(path.join(owned, "invocation.json"), { ...report, limits: LIMITS });
  try {
    for (const timeout of LIMITS.cases) {
      const item = await runCase(path.join(owned, `case-${timeout}`), timeout);
      report.cases.push(item);
      if (!item.result.launch.supervisorReaped && item.result.launch.state !== "not-attempted") {
        report.collectionErrors.push("prior-supervisor-unreaped; further calls refused; partial evidence retained"); break;
      }
    }
  } catch {
    // Do not echo arbitrary exception text/environment. Earlier raw records stay.
    report.collectionErrors.push("observation-incomplete; inspect retained owned artifacts");
  }
  writeExclusiveJson(path.join(owned, "observation.json"), report);
  return { directory: owned, report };
}

// Explicit CLI boundary. A normal import must not allocate files or run probes.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    if (process.argv.length !== 4 || process.argv[2] !== "--runtime" || !["node", "bun"].includes(process.argv[3] ?? "")) throw new Error("literal --runtime node|bun required");
    actualHost(process.argv[3]);
    if (process.env.QMD_RECOVERY_STARTUP_OBSERVATION !== "1" || !process.env.QMD_RECOVERY_STARTUP_DIR) throw new Error("explicit diagnostic opt-in and absolute output directory required");
    const { directory, report } = await runStartupObservation(process.env.QMD_RECOVERY_STARTUP_DIR, process.argv[3] as Runtime);
    console.log(JSON.stringify({ evidence: "after-canonical-startup-observation", directory, actualHost: report.actualHost,
      cases: report.cases.map(item => ({ timeoutMs: item.timeoutMs, classification: item.classification, python: item.python.status })),
      productQualified: false, collectionErrors: report.collectionErrors }));
    if (report.collectionErrors.length) process.exitCode = 1;
  } catch { console.error("Startup observation refused or failed; explicit opt-in, actual host enum and owned absolute output required."); process.exitCode = 2; }
}
