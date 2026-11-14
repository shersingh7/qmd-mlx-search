// Positive acceptance at semantic transitions. The independent frozen line-242
// reproducer is intentionally unchanged; this wrapper resolves its own source
// locations from Python's AST and never manufactures supervisor proof messages.
const testRuntime = (process.versions as Record<string, string>).bun ? "bun:test" : "vitest";
const { describe, expect, it }: typeof import("vitest") = await import(testRuntime);
if ((process.versions as Record<string, string>).bun) console.log(JSON.stringify({
  evidence: "recovery-transition-native-runtime", bun: (process.versions as Record<string, string>).bun, executable: process.execPath,
}));
import childProcess from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeRecoveryWorker, type RecoveryProcessResult } from "../scripts/phase5_e2e_runner.js";

const source = fileURLToPath(new URL("../scripts/phase5_recovery_process.py", import.meta.url));
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") return false; throw error; }
}
function intercept(wrapper: string, observe: (pid: number, kill: () => void) => void): () => void {
  if ((process.versions as Record<string, string>).bun) {
    const bun = (globalThis as any).Bun;
    const original = bun.spawn;
    bun.spawn = function(command: string[], options: unknown) {
      if (command[3] === source) {
        const child = original.call(bun, [...command.slice(0, 3), wrapper, ...command.slice(4)], options);
        observe(child.pid, () => child.kill("SIGKILL")); return child;
      }
      return original.call(bun, command, options);
    };
    return () => { bun.spawn = original; };
  }
  const original = childProcess.spawn;
  (childProcess as any).spawn = function(executable: string, args: string[], options: unknown) {
    if (args[2] === source) {
      const child = original(executable, [...args.slice(0, 2), wrapper, ...args.slice(3)], options as any);
      observe(child.pid!, () => { child.kill("SIGKILL"); }); return child;
    }
    return original(executable, args, options as any);
  };
  syncBuiltinESMExports();
  return () => { childProcess.spawn = original; syncBuiltinESMExports(); };
}

type Stage = "before-fork" | "fork-return" | "registered" | "before-ready" | "after-ready";
async function transitionCase(stage: Stage, pause: "delay" | "stop") {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-transition-"));
  const wrapper = path.join(root, "semantic-transition.py");
  const stageFile = path.join(root, "stage.json");
  const forkFile = path.join(root, "forks.jsonl");
  const workFile = path.join(root, "unexpected-work.json");
  writeFileSync(wrapper, `import ast, json, os, signal, sys, time
source = ${JSON.stringify(source)}
code = open(source, 'rb').read()
tree = ast.parse(code, source)
# Select the real worker creation and ready emission, not joining-anchor forks.
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
fork_line = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'worker' for target in node.targets) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'fork')
ready_line = min(node.lineno for node in ast.walk(main) if isinstance(node, ast.Expr) and node.lineno > fork_line and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'ready')
sys.path.insert(0, os.path.dirname(source))
scope = dict(__name__='semantic_transition_regression')
exec(compile(code, source, 'exec'), scope)
paused = False
def hold(label):
    global paused
    if paused: return
    paused = True
    with open(${JSON.stringify(stageFile)}, 'w') as record:
        json.dump(dict(stage=label, pid=os.getpid()), record)
    ${pause === "stop" ? "os.kill(os.getpid(), signal.SIGSTOP)" : "time.sleep(1.300)"}
original_send = scope['send']
def observed_send(message):
    original_send(message)
    if message['type'] == 'ready' and ${stage === "after-ready" ? "True" : "False"}: hold('after-ready')
scope['send'] = observed_send
original_fork = os.fork
def observed_fork():
    pid = original_fork()
    if pid:
        with open(${JSON.stringify(forkFile)}, 'a') as record:
            record.write(json.dumps(dict(parent=os.getpid(), pid=pid)) + '\\n')
        if ${stage === "fork-return" ? "True" : "False"}: hold('fork-return')
    return pid
os.fork = observed_fork
def trace(frame, event, arg):
    if not paused and event == 'line' and frame.f_code.co_filename == source and frame.f_code.co_name == 'main':
        if ${stage === "before-fork" ? "True" : "False"} and frame.f_lineno == fork_line: hold('before-fork')
        if ${stage === "registered" ? "True" : "False"} and frame.f_lineno > fork_line and frame.f_locals.get('worker', 0) > 0: hold('registered')
        if ${stage === "before-ready" ? "True" : "False"} and frame.f_lineno == ready_line: hold('before-ready')
    return trace
sys.settrace(trace)
scope['main']()
`);
  const sentinel = childProcess.spawn("/usr/bin/python3", ["-S", "-c", "import time; time.sleep(60)"], { detached: true, stdio: "ignore" });
  let supervisorPid: number | undefined;
  let killSupervisor: (() => void) | undefined;
  const restore = intercept(wrapper, (pid, kill) => { supervisorPid = pid; killSupervisor = kill; });
  let helper: Promise<RecoveryProcessResult> | undefined;
  let guard: ReturnType<typeof setTimeout> | undefined;
  let guardFired = false;
  const finallyKilled: number[] = [];
  const hooks = ["exit", "SIGTERM", "SIGINT"].map(event => process.listenerCount(event));
  try {
    const started = performance.now();
    helper = executeRecoveryWorker({ dbPath: path.join(root, "unused.sqlite"), docsDir: root,
      outputJson: path.join(root, "absent.json"), stateRoot: path.join(root, "state"), timeoutMs: 200,
      command: ["/usr/bin/python3", "-S", "-u", "-c", `import json,os,time
with open(${JSON.stringify(workFile)}, 'w') as record: json.dump(dict(pid=os.getpid()), record)
time.sleep(60)`] });
    const result = await Promise.race([helper, new Promise<never>((_resolve, reject) => {
      guard = setTimeout(() => { guardFired = true; reject(new Error("Transition exceeded 1800ms failure guard")); }, 1_800);
    })]);
    clearTimeout(guard);
    const elapsedMs = performance.now() - started;
    const read = (file: string) => existsSync(file) ? readFileSync(file, "utf8") : null;
    const forks = () => existsSync(forkFile) ? readFileSync(forkFile, "utf8").trim().split("\n").map(line => JSON.parse(line) as { pid: number }) : [];
    const atReturn = { evidence: "recovery-transition-before-finally", stage, pause, elapsedMs, supervisorPid,
      supervisorAbsent: supervisorPid !== undefined && !alive(supervisorPid), sentinelAlive: alive(sentinel.pid!),
      observation: read(stageFile), forks: forks(), work: read(workFile), guardFired, result };
    console.log(JSON.stringify(atReturn));
    // Observe past the original 1300ms hold before any fixture cleanup. A late
    // fork or side effect is a failure even if the receipt reports an error.
    await sleep(1_450);
    console.log(JSON.stringify({ evidence: "recovery-transition-no-late-work-before-finally", stage, pause,
      supervisorAbsent: !alive(supervisorPid!), workersAbsent: forks().every(item => !alive(item.pid)),
      sentinelAlive: alive(sentinel.pid!), forks: forks(), work: read(workFile) }));
    expect(read(stageFile), JSON.stringify(atReturn)).not.toBeNull();
    expect(JSON.parse(read(stageFile)!).stage).toBe(stage);
    expect(result.launch.admissionGranted, JSON.stringify(atReturn)).toBe(true);
    expect(result.launch.supervisorReaped, JSON.stringify(atReturn)).toBe(true);
    expect(atReturn.supervisorAbsent, JSON.stringify(atReturn)).toBe(true);
    expect(result.timedOut).toBe(true);
    expect(result.cleanup).toMatchObject({ groupTerminated: true, sessionTerminated: true, pipesClosed: true,
      forcedPipeClose: false, containmentVerified: false });
    if (stage === "before-fork") {
      expect(result.launch.state).toBe("not-started");
      expect(result.launch.noStartProof).toBe("supervisor-acknowledgement");
      expect(result.pid).toBeUndefined();
      expect(result.cleanup).toMatchObject({ childReaped: false, processGroups: [] });
      expect(forks()).toEqual([]);
    } else {
      expect(result.launch.state).toBe("started");
      expect(result.cleanup.childReaped).toBe(true);
      expect(forks().map(item => item.pid)).toContain(result.pid);
    }
    expect(forks().every(item => !alive(item.pid))).toBe(true);
    expect(read(workFile)).toBeNull();
    expect(alive(supervisorPid!)).toBe(false);
    expect(alive(sentinel.pid!)).toBe(true);
    expect(guardFired).toBe(false);
    expect(elapsedMs).toBeGreaterThanOrEqual(200);
    expect(elapsedMs).toBeLessThan(1_500);
    expect(["exit", "SIGTERM", "SIGINT"].map(event => process.listenerCount(event))).toEqual(hooks);
  } finally {
    restore(); if (guard) clearTimeout(guard); if (helper) await helper;
    if (supervisorPid !== undefined && alive(supervisorPid)) { finallyKilled.push(supervisorPid); killSupervisor!(); }
    if (existsSync(forkFile)) for (const line of readFileSync(forkFile, "utf8").trim().split("\n")) {
      const { pid } = JSON.parse(line); if (alive(pid)) { finallyKilled.push(pid); process.kill(pid, "SIGKILL"); }
    }
    if (sentinel.exitCode === null && sentinel.signalCode === null) {
      const exited = new Promise<void>(resolve => sentinel.once("exit", () => resolve())); sentinel.kill("SIGKILL"); await exited;
    }
    rmSync(root, { recursive: true, force: true });
    console.log(JSON.stringify({ evidence: "recovery-transition-finally", stage, pause, finallyKilled, scratchRemoved: !existsSync(root) }));
    expect(finallyKilled).toEqual([]);
  }
}

describe.skipIf(process.platform === "win32")("admitted recovery transition retains cancellation ownership", () => {
  for (const stage of ["before-fork", "fork-return", "registered", "before-ready", "after-ready"] as const) {
    for (const pause of ["delay", "stop"] as const) {
      it(`cancels ${stage} after S across a real ${pause}`, () => transitionCase(stage, pause));
    }
  }
});
