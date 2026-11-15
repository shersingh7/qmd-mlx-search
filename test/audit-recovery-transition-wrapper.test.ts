const testRuntime = (process.versions as Record<string, string>).bun ? "bun:test" : "vitest";
const { describe, expect, it }: typeof import("vitest") = await import(testRuntime);
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { recoveryTransitionWrapper, type RecoveryTransitionPause, type RecoveryTransitionStage } from "./helpers/recovery-transition-wrapper.js";

const source = fileURLToPath(new URL("../scripts/phase5_recovery_process.py", import.meta.url));
const stages = ["before-fork", "fork-return", "registered", "before-ready", "after-ready"] as const;
const traced = (stage: RecoveryTransitionStage) => !["fork-return", "after-ready"].includes(stage);

// Executes the actual generated wrapper, not a text/presence test. The real-source
// probe stops on entry to require_waitid_support, before any session/fork/signal
// operation. Synthetic semantic probes observe (do not emit) the wrapper's hold,
// with only sleep/SIGSTOP replaced; the separate ten-case suite uses real holds.
const driver = String.raw`import ast, builtins, json, os, signal, sys, time, types
payload = json.loads(sys.argv[1])
namespace = {}
counts = dict(astImports=0, astParses=0, traceInstalls=0, targetCalls=0, targetLines=0, otherCalls=0, otherLines=0)
counts.update(sourceReads=0, sourceCompiles=0, mainInvocations=0, acquisitionAttempts=0)
no_child = payload['mode'] != 'semantic'
holds, children, reaped, finally_killed = [], [], [], []
original_import, original_parse, original_trace = builtins.__import__, ast.parse, sys.settrace
original_compile, original_setsid = builtins.compile, os.setsid
original_fork, original_exit, original_wait, original_kill = os.fork, os._exit, os.waitpid, os.kill
original_sleep = time.sleep
root = os.path.realpath(payload['root'])
public = os.path.realpath(payload['public'])
stdlib = os.path.realpath(sys.base_prefix)
def inside(value, directory):
    return value == directory or value.startswith(directory + os.sep)
def refuse_acquisition(operation):
    counts['acquisitionAttempts'] += 1
    raise AssertionError('no-child probe attempted ' + operation)
def audit(event, args):
    if no_child and event in ('os.fork', 'os.forkpty', 'os.kill', 'os.killpg', 'os.posix_spawn', 'os.exec', 'subprocess.Popen'):
        refuse_acquisition(event)
    if event == 'open' and not isinstance(args[0], int):
        value = os.path.realpath(os.fsdecode(args[0]))
        writing = isinstance(args[1], str) and any(flag in args[1] for flag in 'wa+')
        if not writing and value == os.path.realpath(payload['sourceFile']): counts['sourceReads'] += 1
        if not (inside(value, root) or (not writing and (inside(value, public) or inside(value, stdlib)))):
            raise PermissionError('transition-unit read/write outside public/owned closure')
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'socket.connect', 'socket.bind'):
        raise PermissionError('transition-unit unexpected process/network operation')
sys.addaudithook(audit)
try:
    open('/Users/shersingh/github/qmd-mlx-search/package.json', 'rb')
except PermissionError:
    outside_refused = True
else:
    raise AssertionError('unit audit did not refuse outside-root read')
def imported(name, *args, **kwargs):
    if name == 'ast': counts['astImports'] += 1
    return original_import(name, *args, **kwargs)
def parsed(*args, **kwargs):
    counts['astParses'] += 1
    return original_parse(*args, **kwargs)
def compiled(text, filename, mode, flags=0, dont_inherit=False, optimize=-1, **kwargs):
    if filename == payload['sourceFile'] and mode == 'exec' and not flags & ast.PyCF_ONLY_AST:
        counts['sourceCompiles'] += 1
    return original_compile(text, filename, mode, flags, dont_inherit, optimize, **kwargs)
def installed(callback):
    if callback is None: return original_trace(None)
    counts['traceInstalls'] += 1
    def observed(frame, event, arg):
        # Observe the subject's selected code/globals, not arbitrary main metadata.
        qualified = frame.f_code is namespace.get('main_code') and frame.f_globals is namespace.get('scope')
        if event == 'call': counts['targetCalls' if qualified else 'otherCalls'] += 1
        if event == 'line': counts['targetLines' if qualified else 'otherLines'] += 1
        local = callback(frame, event, arg)
        return observed if local is not None else None
    return original_trace(observed)
def snapshot_hold(kind, value):
    with open(payload['stageFile']) as record: observation = json.load(record)
    holds.append(dict(kind=kind, value=value, observation=observation,
                      events=list(namespace['scope'].get('events', []))))
def slept(seconds): snapshot_hold('delay', seconds)
def stopped(pid, number):
    assert pid == os.getpid() and number == signal.SIGSTOP
    snapshot_hold('stop', int(number))
def forked():
    if no_child: refuse_acquisition('fork')
    pid = original_fork()
    if pid: children.append(pid)
    return pid
def waited(pid, flags):
    result = original_wait(pid, flags)
    if result[0] > 0: reaped.append(result[0])
    return result
def exited(code):
    # Captured in the real child before _exit: line tracing survives fork only
    # for the inherited real-main frame, not its nested/helper functions.
    original_trace(None)
    with open(payload['childFile'], 'w') as record:
        json.dump(dict(pid=os.getpid(), counts=counts, paused=namespace.get('paused')), record)
    original_exit(code)
class ProbeBoundary(Exception): pass
wrapper_code = None
wrapper_call_line = len(payload['wrapper'].splitlines())
entry_code, entry_globals = None, None
def profile(frame, event, arg):
    global entry_code, entry_globals
    if event != 'call': return
    caller = frame.f_back
    dispatched = (caller is not None and caller.f_code is wrapper_code
                  and caller.f_globals is namespace and caller.f_lineno == wrapper_call_line)
    # Establish one code/globals pair at the actual wrapper's terminal dispatch.
    # Retain it across alias calls and name rebinding; recursive frames need not
    # be the first frame selected by the subject's separate line-trace contract.
    if entry_code is None and dispatched:
        entry_code, entry_globals = frame.f_code, frame.f_globals
    entry = frame.f_code is entry_code and frame.f_globals is entry_globals
    if entry: counts['mainInvocations'] += 1
    boundary_target = namespace.get('scope', {}).get('require_waitid_support')
    boundary_entry = (type(boundary_target) is types.FunctionType
                      and frame.f_code is boundary_target.__code__
                      and frame.f_globals is boundary_target.__globals__)
    # The authenticated dispatch is also a no-child tripwire, not evidence that
    # the subject accepted a callable's shape (hook-only stages do not preflight).
    if ((payload['mode'] == 'real-source' and boundary_entry)
            or (payload['mode'] == 'invalid-source' and (entry or dispatched))):
        raise ProbeBoundary()
def session(): refuse_acquisition('setsid')
def killed(pid, number):
    if no_child: refuse_acquisition('kill')
    return original_kill(pid, number)
builtins.__import__, ast.parse, sys.settrace = imported, parsed, installed
builtins.compile = compiled
os.fork, os._exit, os.waitpid = forked, exited, waited
os.setsid = session if no_child else original_setsid
os.kill = killed
# Only unit semantic probes replace holds. No canonical API case uses this driver.
if payload['mode'] == 'semantic':
    time.sleep, os.kill = slept, stopped
if payload['mode'] == 'real-source':
    sys.argv = ['semantic-transition.py', json.dumps(dict(command=['/usr/bin/python3'], timeoutMs=200, contained=False))]
if payload['mode'] in ('real-source', 'invalid-source'):
    sys.setprofile(profile)
error, boundary = None, False
try:
    wrapper_code = compile(payload['wrapper'], '<generated-transition-wrapper>', 'exec')
    exec(wrapper_code, namespace)
except ProbeBoundary:
    boundary = True
except BaseException as failure:
    origin = failure.__traceback__
    while origin.tb_next is not None: origin = origin.tb_next
    # The innermost actual code/globals identity distinguishes a subject raise
    # from an observer/source exception with the same type, message or filename.
    error = dict(type=type(failure).__name__, message=str(failure),
                 fromWrapperCode=origin.tb_frame.f_code is wrapper_code and origin.tb_frame.f_globals is namespace)
finally:
    original_trace(None)
    sys.setprofile(None)
    builtins.__import__, ast.parse, sys.settrace = original_import, original_parse, original_trace
    builtins.compile, os.setsid, os.kill = original_compile, original_setsid, original_kill
    # Each numeric PID below is a direct child whose wait ownership remains
    # retained; never signal a reaped identity or any global process/group.
    for pid in children:
        if pid in reaped: continue
        deadline = time.monotonic() + 1
        while True:
            result = original_wait(pid, os.WNOHANG)
            if result[0]: reaped.append(pid); break
            if time.monotonic() >= deadline:
                finally_killed.append(pid)
                original_kill(pid, signal.SIGKILL)
                original_wait(pid, 0); reaped.append(pid); break
            original_sleep(0.001)
print(json.dumps(dict(counts=counts, error=error, boundary=boundary, holds=holds, children=children,
                     reaped=reaped, finallyKilled=finally_killed, outsideReadRefused=outside_refused,
                     forkLine=namespace.get('fork_line'), readyLine=namespace.get('ready_line'),
                     events=namespace.get('scope', {}).get('events'), pid=os.getpid(),
                     interpreter=dict(executable=sys.executable, version=sys.version))))
`;

type Counts = { astImports: number; astParses: number; traceInstalls: number; targetCalls: number; targetLines: number; otherCalls: number; otherLines: number;
  sourceReads: number; sourceCompiles: number; mainInvocations: number; acquisitionAttempts: number };
type Probe = { counts: Counts; error: { type: string; message: string; fromWrapperCode: boolean } | null; boundary: boolean;
  holds: { kind: string; value: number; observation: { stage: string; pid: number }; events: string[] }[];
  children: number[]; reaped: number[]; finallyKilled: number[]; outsideReadRefused: boolean;
  forkLine: number | null; readyLine: number | null; pid: number; events: string[] | null;
  child: { pid: number; counts: Counts; paused: boolean } | null };
let sequence = 0;
function probe(stage: RecoveryTransitionStage, pause: RecoveryTransitionPause, mode: "real-source" | "semantic" | "invalid-source", synthetic?: string): Probe {
  // Include JSON-sensitive pathname bytes in the actual executed Python strings.
  const root = mkdtempSync(path.join(tmpdir(), "qmd-transition-wrapper-'\"\\é-"));
  const sourceFile = synthetic === undefined ? source : path.join(root, "source-'\"\\é.py");
  const stageFile = path.join(root, "stage-'\"\\é.json");
  const forkFile = path.join(root, "forks-'\"\\é.jsonl");
  const childFile = path.join(root, "child.json");
  const wrapper = recoveryTransitionWrapper({ source: sourceFile, stageFile, forkFile, stage, pause });
  try {
    if (synthetic !== undefined) writeFileSync(sourceFile, synthetic);
    const result = spawnSync("/usr/bin/python3", ["-S", "-c", driver, JSON.stringify({ wrapper, root, public: path.dirname(source), sourceFile, stageFile, childFile, mode })],
      { encoding: "utf8", timeout: 8_000, maxBuffer: 1024 * 1024 });
    expect(result.error, result.stderr).toBeUndefined();
    expect(result.signal, result.stderr).toBeNull();
    expect(result.status, result.stderr).toBe(0);
    const parsed = JSON.parse(result.stdout) as Omit<Probe, "child">;
    const receipt: Probe = { ...parsed, child: existsSync(childFile) ? JSON.parse(readFileSync(childFile, "utf8")) : null };
    if (process.env.QMD_TRANSITION_UNIT_OUTPUT) {
      const dest = path.join(process.env.QMD_TRANSITION_UNIT_OUTPUT, String(++sequence));
      mkdirSync(dest, { recursive: true });
      writeFileSync(path.join(dest, "wrapper.py"), wrapper);
      const sha256 = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
      writeFileSync(path.join(dest, "receipt.json"), JSON.stringify({ stage, pause, mode,
        classification: mode === "semantic" ? "REAL_FORK_SYNTHETIC_HOLD_UNIT" : "NO_CHILD_PRE_ACQUISITION_UNIT",
        sourceSha256: sha256(synthetic === undefined ? readFileSync(source) : synthetic),
        wrapperSha256: sha256(wrapper), driverSha256: sha256(driver), ...receipt }, null, 2));
      if (synthetic !== undefined) writeFileSync(path.join(dest, "source.py"), synthetic);
      writeFileSync(path.join(dest, "stdout.txt"), result.stdout);
      writeFileSync(path.join(dest, "stderr.txt"), result.stderr);
    }
    expect(receipt.outsideReadRefused).toBe(true);
    expect(receipt.finallyKilled).toEqual([]);
    expect([...receipt.reaped].sort()).toEqual([...receipt.children].sort());
    return receipt;
  } finally { rmSync(root, { recursive: true, force: true }); }
}

const semanticSource = `import json, os
# A nested shadow must not supply the worker-fork or ready anchor.
events = []
def send(message):
    events.append('ready-sent')
def noise():
    value = 1
    value += 1
    events.append('noise')
def main():
    def shadow():
        worker = pretend.fork()
        ready()
    def ready():
        events.append('ready-entry')
        send(dict(type='ready'))
    noise()
    try:
        worker = os.fork()
        if worker == 0:
            noise()
            os._exit(0)
        events.append('registered')
        ready()
    finally:
        if worker > 0:
            os.waitpid(worker, 0)
    # The later cleanup-ready call is not the startup-ready sibling anchor.
    ready()
`;

describe.skipIf(process.platform === "win32")("source-bound transition fixture instrumentation", () => {
  for (const stage of stages) {
    it(`${stage} performs only its necessary AST/line-tracing work on the real source`, () => {
      const receipt = probe(stage, "delay", "real-source");
      expect(receipt.error).toBeNull();
      expect(receipt.boundary).toBe(true);
      expect(receipt.children).toEqual([]);
      expect(receipt.holds).toEqual([]);
      console.log(JSON.stringify({ evidence: "transition-wrapper-work-counts", stage, ...receipt.counts }));
      expect(receipt.counts.astImports).toBe(traced(stage) ? 1 : 0);
      expect(receipt.counts.astParses).toBe(traced(stage) ? 1 : 0);
      expect(receipt.counts.traceInstalls).toBe(traced(stage) ? 1 : 0);
      expect(receipt.counts.otherLines).toBe(0);
      if (traced(stage)) {
        // sys.settrace still receives unrelated CALL events. Zero unrelated
        // LINE events is the contract, not a false claim of zero callbacks.
        expect(receipt.counts.otherCalls).toBeGreaterThan(0);
        expect(receipt.counts.targetCalls).toBe(1);
        expect(receipt.counts.targetLines).toBeGreaterThan(0);
        const lines = readFileSync(source, "utf8").split("\n");
        expect(lines[receipt.forkLine! - 1]!.trim()).toBe("worker = os.fork()");
        expect(lines[receipt.readyLine! - 1]!.trim()).toBe("ready()");
      }
    });
  }
  for (const stage of stages) for (const pause of ["delay", "stop"] as const) {
    it(`${stage}/${pause} holds at its real semantic anchor and retains tracing across fork`, () => {
      const receipt = probe(stage, pause, "semantic", semanticSource);
      expect(receipt.error).toBeNull();
      expect(receipt.boundary).toBe(false);
      expect(receipt.holds).toHaveLength(1);
      expect(receipt.holds[0]!.observation).toEqual({ stage, pid: receipt.pid });
      expect(receipt.holds[0]!.kind).toBe(pause);
      if (pause === "delay") expect(receipt.holds[0]!.value).toBe(1.300);
      const expected = stage === "after-ready" ? ["noise", "registered", "ready-entry", "ready-sent"]
        : stage === "before-ready" ? ["noise", "registered"] : ["noise"];
      expect(receipt.holds[0]!.events).toEqual(expected);
      expect(receipt.children).toHaveLength(1);
      expect(receipt.child!.pid).toBe(receipt.children[0]);
      expect(receipt.counts.otherLines).toBe(0);
      expect(receipt.child!.counts.otherLines).toBe(0);
      if (traced(stage)) expect(receipt.child!.counts.targetLines).toBeGreaterThan(receipt.counts.targetCalls);
      // before-fork holds before cloning; all other stages hold parent-only.
      expect(receipt.child!.paused).toBe(stage === "before-fork");
    });
  }
  const invalid = {
    "missing real main": semanticSource.replace("def main():", "def not_main():"),
    "duplicate top-level main": semanticSource + "\ndef main():\n    pass\n",
    "wrong fork receiver": semanticSource.replace("worker = os.fork()", "worker = pretend.fork()"),
    "ambiguous worker fork": semanticSource.replace("worker = os.fork()", "worker = os.fork()\n        worker = os.fork()"),
    "nested-only ready": semanticSource.replace("        ready()\n    finally:", "        noise()\n    finally:"),
    "ambiguous startup ready": semanticSource.replace("        ready()\n    finally:", "        ready()\n        ready()\n    finally:"),
    "rebound main code": semanticSource + "\nmain = lambda: None\n",
  };
  for (const [label, text] of Object.entries(invalid)) {
    it(`refuses ${label} before invoking main or acquiring children`, () => {
      const receipt = probe("before-ready", "delay", "invalid-source", text);
      expect(receipt.error?.type).toBe("ValueError");
      expect(receipt.children).toEqual([]);
      expect(receipt.holds).toEqual([]);
      expect(receipt.counts.traceInstalls).toBe(0);
    });
  }
  // These are preflight refusal contracts, not missing-stage assertions. The
  // profile boundary and acquisition guards keep even their RED runs no-child.
  const identityInvalid = {
    "wrong-loaded-main-globals": semanticSource + "\nimport types\nmain = types.FunctionType(main.__code__, dict(globals()))\n",
    "different-code-same-filename-and-firstline": semanticSource + `
import types
old_main = main
def substitute():
    events.append('wrong-main-ran')
main = types.FunctionType(substitute.__code__.replace(co_filename=old_main.__code__.co_filename, co_firstlineno=old_main.__code__.co_firstlineno, co_name='main'), globals())
`,
    "equal-but-distinct-code": semanticSource + `
replacement = main.__code__.replace()
assert replacement == main.__code__ and replacement is not main.__code__
main.__code__ = replacement
`,
    "forged-callable-metadata-shell": semanticSource + `
class Shell:
    __code__ = main.__code__
    __globals__ = globals()
    def __call__(self):
        events.append('wrong-main-ran')
main = Shell()
`,
    "non-callable-metadata-shell": semanticSource + "\nimport types\nmain = types.SimpleNamespace(__code__=main.__code__, __globals__=globals())\n",
    "non-function-main": semanticSource + "\nmain = 42\n",
  };
  for (const [label, text] of Object.entries(identityInvalid)) {
    it(`source identity rejects ${label} before tracing, invocation or acquisition`, () => {
      const receipt = probe("before-ready", "delay", "invalid-source", text);
      expect(receipt.error?.type).toBe("ValueError");
      expect(receipt.error?.message).toBe("Loaded main does not match the qualified source anchor");
      expect(receipt.error?.fromWrapperCode).toBe(true);
      expect(receipt.events).toEqual([]);
      expect(receipt.boundary).toBe(false);
      expect(receipt.counts.traceInstalls).toBe(0);
      expect(receipt.counts.mainInvocations).toBe(0);
      expect(receipt.counts.acquisitionAttempts).toBe(0);
      expect(receipt.counts.sourceReads).toBe(1);
      expect(receipt.counts.sourceCompiles).toBe(1);
      expect(receipt.counts.otherLines).toBe(0);
      expect(receipt.children).toEqual([]);
      expect(receipt.holds).toEqual([]);
    });
  }
  // The observer must not evaluate fixture metadata merely to classify a frame.
  const descriptorSource = semanticSource + String.raw`
import sys
_saved_main = main
class MetadataShell:
    @property
    def __code__(self):
        events.append('metadata-read:' + sys._getframe(1).f_code.co_name)
        return _saved_main.__code__
    @property
    def __globals__(self):
        events.append('metadata-globals-read:' + sys._getframe(1).f_code.co_name)
        return globals()
    def __call__(self):
        events.append('shell-invoked')
main = MetadataShell()
`;
  for (const raising of [false, true]) {
    it(`observer transparency leaves ${raising ? "raising" : "returning"} metadata descriptors untouched before subject refusal`, () => {
      const text = raising ? descriptorSource.replace("return _saved_main.__code__", "raise ValueError('metadata-property-origin-not-subject-preflight')") : descriptorSource;
      const receipt = probe("before-ready", "delay", "invalid-source", text);
      expect(receipt.error?.type).toBe("ValueError");
      expect(receipt.error?.message).toBe("Loaded main does not match the qualified source anchor");
      expect(receipt.error?.fromWrapperCode).toBe(true);
      expect(receipt.events).toEqual([]);
      expect(receipt.boundary).toBe(false);
      expect(receipt.counts.traceInstalls).toBe(0);
      expect(receipt.counts.mainInvocations).toBe(0);
      expect(receipt.counts.acquisitionAttempts).toBe(0);
      expect(receipt.counts.sourceReads).toBe(1);
      expect(receipt.counts.sourceCompiles).toBe(1);
      expect(receipt.counts.otherLines).toBe(0);
      expect(receipt.children).toEqual([]);
      expect(receipt.holds).toEqual([]);
    });
  }
  it("observer transparency does not consult a callable shell's metaclass metadata", () => {
    const text = semanticSource + `
class MetadataType(type):
    def __getattribute__(cls, name):
        if name == '__call__':
            events.append('metadata-call-read')
            raise ValueError('metadata-property-origin-not-subject-preflight')
        return type.__getattribute__(cls, name)
class Shell(metaclass=MetadataType):
    def __call__(self):
        events.append('shell-invoked')
main = Shell()
`;
    const receipt = probe("before-ready", "delay", "invalid-source", text);
    expect(receipt.error).toEqual({ type: "ValueError", message: "Loaded main does not match the qualified source anchor", fromWrapperCode: true });
    expect(receipt.events).toEqual([]);
    expect(receipt.boundary).toBe(false);
    expect(receipt.counts.traceInstalls).toBe(0);
    expect(receipt.counts.mainInvocations).toBe(0);
    expect(receipt.counts.acquisitionAttempts).toBe(0);
    expect(receipt.counts.sourceReads).toBe(1);
    expect(receipt.counts.sourceCompiles).toBe(1);
    expect(receipt.counts.otherLines).toBe(0);
    expect(receipt.children).toEqual([]);
    expect(receipt.holds).toEqual([]);
  });
  for (const message of ["metadata-property-origin-not-subject-preflight", "Loaded main does not match the qualified source anchor"]) {
    it(`observer transparency distinguishes observer-origin ${message} from subject preflight`, () => {
      const text = semanticSource + `
import sys
def failing_observer(frame, event, arg):
    if event == 'call':
        events.append('observer-origin')
        raise ValueError(${JSON.stringify(message)})
# A matching filename is not code identity or evidence of a subject raise.
failing_observer.__code__ = failing_observer.__code__.replace(co_filename='<generated-transition-wrapper>')
sys.setprofile(failing_observer)
noise()
`;
      const receipt = probe("before-ready", "delay", "invalid-source", text);
      expect(receipt.error).toEqual({ type: "ValueError", message, fromWrapperCode: false });
      expect(receipt.events).toEqual(["observer-origin"]);
      expect(receipt.boundary).toBe(false);
      expect(receipt.counts.traceInstalls).toBe(0);
      expect(receipt.counts.mainInvocations).toBe(0);
      expect(receipt.counts.acquisitionAttempts).toBe(0);
      expect(receipt.counts.sourceReads).toBe(1);
      expect(receipt.counts.sourceCompiles).toBe(1);
      expect(receipt.counts.otherLines).toBe(0);
      expect(receipt.children).toEqual([]);
      expect(receipt.holds).toEqual([]);
    });
  }
  const boundarySource = semanticSource.replace("def main():\n", `def require_waitid_support():
    raise AssertionError('pre-acquisition profile boundary was missed')
def main():
    require_waitid_support()
`);
  it("observer transparency traces the selected frame after metadata-shell rebinding", () => {
    const text = (boundarySource + descriptorSource.slice(semanticSource.length).replace("\nmain = MetadataShell()\n", "\n"))
      .replace("    require_waitid_support()\n", "    global main\n    main = MetadataShell()\n    require_waitid_support()\n");
    const receipt = probe("before-ready", "delay", "real-source", text);
    expect(receipt.error).toBeNull();
    expect(receipt.boundary).toBe(true);
    expect(receipt.events).toEqual([]);
    expect(receipt.counts.traceInstalls).toBe(1);
    expect(receipt.counts.mainInvocations).toBe(1);
    expect(receipt.counts.targetCalls).toBe(1);
    expect(receipt.counts.targetLines).toBeGreaterThan(0);
    expect(receipt.counts.otherLines).toBe(0);
    expect(receipt.counts.acquisitionAttempts).toBe(0);
    expect(receipt.counts.sourceReads).toBe(1);
    expect(receipt.counts.sourceCompiles).toBe(1);
    expect(receipt.children).toEqual([]);
    expect(receipt.holds).toEqual([]);
  });
  const reboundRecursionSource = (semanticSource.replace("def main():\n", `def require_waitid_support():
    raise AssertionError('pre-acquisition profile boundary was missed')
def main(reentrant=False):
    events.append('actual-main-entry')
    if reentrant:
        return
    global main
    main = MetadataShell()
    unrelated()
    _foreign_main(True)
    assert _foreign_events == ['actual-main-entry']
    events.append('foreign-main-returned')
    _saved_main(True)
    require_waitid_support()
`) + descriptorSource.slice(semanticSource.length).replace("\nmain = MetadataShell()\n", "\n")) + `
def unrelated():
    def main():
        events.append('unrelated-main-entry')
    main()
import types
_foreign_events = []
_foreign_main = types.FunctionType(_saved_main.__code__, dict(globals(), events=_foreign_events))
`;
  for (const { label, stage, raising } of [
    { label: "returning metadata shell", stage: "before-ready", raising: false },
    { label: "raising metadata shell", stage: "before-ready", raising: true },
    { label: "hook-only raising metadata shell", stage: "after-ready", raising: true },
  ] as const) {
    it(`observer transparency counts saved-alias recursion after rebinding to a ${label}`, () => {
      const text = raising ? reboundRecursionSource
        .replace("return _saved_main.__code__", "raise ValueError('observer-must-not-read-rebound-code')")
        .replace("return globals()", "raise ValueError('observer-must-not-read-rebound-globals')") : reboundRecursionSource;
      const receipt = probe(stage, "delay", "real-source", text);
      expect(receipt.error).toBeNull();
      expect(receipt.boundary).toBe(true);
      // Body events, the independent code/globals trace observer and profile
      // CALL accounting must agree without counting either unrelated frame.
      expect(receipt.events).toEqual(["actual-main-entry", "unrelated-main-entry", "foreign-main-returned", "actual-main-entry"]);
      expect(receipt.events!.filter(event => event === "actual-main-entry")).toHaveLength(2);
      expect(receipt.counts.mainInvocations).toBe(2);
      expect(receipt.counts.astImports).toBe(traced(stage) ? 1 : 0);
      expect(receipt.counts.astParses).toBe(traced(stage) ? 1 : 0);
      expect(receipt.counts.traceInstalls).toBe(traced(stage) ? 1 : 0);
      expect(receipt.counts.targetCalls).toBe(traced(stage) ? 2 : 0);
      if (traced(stage)) expect(receipt.counts.targetLines).toBeGreaterThan(0);
      else expect(receipt.counts.targetLines).toBe(0);
      expect(receipt.counts.otherLines).toBe(0);
      expect(receipt.counts.acquisitionAttempts).toBe(0);
      expect(receipt.counts.sourceReads).toBe(1);
      expect(receipt.counts.sourceCompiles).toBe(1);
      expect(receipt.children).toEqual([]);
      expect(receipt.holds).toEqual([]);
    });
  }
  const identityValid = {
    "qualified-top-level-main-with-nested-shadow": boundarySource.replace("    def shadow():", "    def main():\n        pass\n    def shadow():"),
    "same-code-and-globals-new-function": boundarySource + "\nimport types\nmain = types.FunctionType(main.__code__, globals())\n",
    "identity-decorated-main": boundarySource.replace("def main():", "def identity(function):\n    return function\n@identity\ndef main():"),
    "recursive-main-frame": boundarySource.replace("def main():\n", "def main(reentrant=False):\n    if reentrant:\n        return\n    main(True)\n"),
  };
  for (const [label, text] of Object.entries(identityValid)) {
    it(`source identity accepts ${label} and reaches the pre-acquisition boundary`, () => {
      const receipt = probe("before-ready", "delay", "real-source", text);
      expect(receipt.error).toBeNull();
      expect(receipt.boundary).toBe(true);
      expect(receipt.counts.traceInstalls).toBe(1);
      expect(receipt.counts.mainInvocations).toBe(label === "recursive-main-frame" ? 2 : 1);
      expect(receipt.counts.targetCalls).toBe(label === "recursive-main-frame" ? 2 : 1);
      expect(receipt.counts.targetLines).toBeGreaterThan(0);
      expect(receipt.counts.otherLines).toBe(0);
      expect(receipt.counts.acquisitionAttempts).toBe(0);
      expect(receipt.counts.sourceReads).toBe(1);
      expect(receipt.counts.sourceCompiles).toBe(1);
      expect(receipt.children).toEqual([]);
      expect(receipt.holds).toEqual([]);
    });
  }
});
