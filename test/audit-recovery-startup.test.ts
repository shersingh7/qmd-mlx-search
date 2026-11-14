// Real delayed supervisors: the public 200ms budget includes cold bootstrap.
const testRuntime = (process.versions as Record<string, string>).bun ? "bun:test" : "vitest";
const { describe, expect, it }: typeof import("vitest") = await import(testRuntime);
if ((process.versions as Record<string, string>).bun) console.log(JSON.stringify({
  evidence: "recovery-startup-native-runtime", bun: (process.versions as Record<string, string>).bun, executable: process.execPath,
}));
import childProcess from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeRecoveryWorker, type RecoveryProcessResult } from "../scripts/phase5_e2e_runner.js";

const supervisorScript = fileURLToPath(new URL("../scripts/phase5_recovery_process.py", import.meta.url));
function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") return false; throw error; }
}
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Intercept only this process's supervisor launch, keeping real OS handles,
// runtime-native transport, production Python bytes, and the original deadline.
function interceptSupervisor(wrapper: string, observe: (pid: number, kill: () => void) => void): () => void {
  if ((process.versions as Record<string, string>).bun) {
    const bun = (globalThis as any).Bun;
    const original = bun.spawn;
    bun.spawn = function(command: string[], options: unknown) {
      if (command[0] === "/usr/bin/python3" && command[3] === supervisorScript) {
        const child = original.call(bun, [...command.slice(0, 3), wrapper, ...command.slice(4)], options);
        observe(child.pid, () => child.kill("SIGKILL"));
        return child;
      }
      return original.call(bun, command, options);
    };
    return () => { bun.spawn = original; };
  }
  const original = childProcess.spawn;
  (childProcess as any).spawn = function(executable: string, args: string[], options: unknown) {
    if (executable === "/usr/bin/python3" && args[2] === supervisorScript) {
      const child = original(executable, [...args.slice(0, 2), wrapper, ...args.slice(3)], options as any);
      observe(child.pid!, () => child.kill("SIGKILL"));
      return child;
    }
    return original(executable, args, options as any);
  };
  syncBuiltinESMExports();
  return () => { childProcess.spawn = original; syncBuiltinESMExports(); };
}

async function coldSupervisorCase(bootstrapDelayMs: number, canonical = false) {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-cold-supervisor-"));
  const forkFile = path.join(root, "forks.jsonl");
  const supervisorFile = path.join(root, "supervisor.json");
  const workerFile = path.join(root, "unexpected-work.json");
  const wrapper = path.join(root, "delayed-supervisor.py");
  // The observation wrapper never forges protocol messages. Delay only cold
  // bootstrap, then run the actual source. A post-fork scheduling pause makes
  // any regression's otherwise short late-work window observable.
  writeFileSync(wrapper, `import json, os, sys, time
with open(${JSON.stringify(supervisorFile)}, 'w') as record:
    json.dump(dict(pid=os.getpid(), sid=os.getsid(0), pgid=os.getpgrp()), record)
time.sleep(${bootstrapDelayMs / 1000})
_original_fork = os.fork
def observed_fork():
    pid = _original_fork()
    if pid:
        with open(${JSON.stringify(forkFile)}, 'a') as record:
            record.write(json.dumps(dict(parent=os.getpid(), pid=pid)) + '\\n')
        time.sleep(0.080)
    return pid
os.fork = observed_fork
sys.path.insert(0, ${JSON.stringify(path.dirname(supervisorScript))})
sys.argv[0] = ${JSON.stringify(supervisorScript)}
exec(compile(open(${JSON.stringify(supervisorScript)}, 'rb').read(), ${JSON.stringify(supervisorScript)}, 'exec'))
`);
  const sentinel = childProcess.spawn("/usr/bin/python3", ["-S", "-c", "import time; time.sleep(60)"], { detached: true, stdio: "ignore" });
  let supervisorPid: number | undefined;
  let supervisorObservedAlive = false;
  let killSupervisor: (() => void) | undefined;
  const restore = interceptSupervisor(wrapper, (pid, kill) => {
    supervisorPid = pid; supervisorObservedAlive = alive(pid); killSupervisor = kill;
  });
  let guard: ReturnType<typeof setTimeout> | undefined;
  let guardFired = false;
  let helper: Promise<RecoveryProcessResult> | undefined;
  const finallyKilled: number[] = [];
  try {
    const started = performance.now();
    helper = executeRecoveryWorker({ dbPath: path.join(root, "unused.sqlite"), docsDir: root,
      outputJson: path.join(root, "absent.json"), stateRoot: path.join(root, "state"), timeoutMs: 200,
      ...(canonical ? {} : { command: ["/usr/bin/python3", "-S", "-c", `import json,os,signal,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(${JSON.stringify(workerFile)}, 'w') as record:
    json.dump(dict(pid=os.getpid(), sid=os.getsid(0), pgid=os.getpgrp()), record)
time.sleep(60)`] }) });
    const result = await Promise.race([helper, new Promise<never>((_resolve, reject) => {
      guard = setTimeout(() => { guardFired = true; reject(new Error("200ms cold supervisor exceeded bounded teardown")); }, 1_800);
    })]);
    clearTimeout(guard);
    const elapsedMs = performance.now() - started;
    const before = { evidence: "recovery-cold-supervisor-before-finally", bootstrapDelayMs, canonical,
      elapsedMs, supervisorPid, supervisorObservedAlive, supervisorAbsent: supervisorPid !== undefined && !alive(supervisorPid),
      supervisorIdentity: existsSync(supervisorFile) ? JSON.parse(readFileSync(supervisorFile, "utf8")) : null,
      sentinelAlive: alive(sentinel.pid!), forks: existsSync(forkFile) ? readFileSync(forkFile, "utf8") : null,
      work: existsSync(workerFile) ? readFileSync(workerFile, "utf8") : null, guardFired, result };
    console.log(JSON.stringify(before));
    // Observe past the original delayed bootstrap, BEFORE asserting/cleaning.
    // A returned missing PID alone must never hide a later fork or side effect.
    const lateEnd = performance.now() + bootstrapDelayMs + 150;
    const lateLiveWorkers: number[] = [];
    while (performance.now() < lateEnd) {
      if (existsSync(workerFile)) {
        const worker = JSON.parse(readFileSync(workerFile, "utf8"));
        if (alive(worker.pid) && !lateLiveWorkers.includes(worker.pid)) lateLiveWorkers.push(worker.pid);
      }
      await delay(10);
    }
    console.log(JSON.stringify({ evidence: "recovery-cold-supervisor-no-late-work-before-finally", bootstrapDelayMs,
      supervisorPid, supervisorAbsent: !alive(supervisorPid!), sentinelAlive: alive(sentinel.pid!), lateLiveWorkers,
      forks: existsSync(forkFile) ? readFileSync(forkFile, "utf8") : null,
      work: existsSync(workerFile) ? readFileSync(workerFile, "utf8") : null }));
    expect(supervisorObservedAlive, JSON.stringify(before)).toBe(true);
    expect(before.supervisorAbsent, JSON.stringify(before)).toBe(true);
    expect(result.timedOut, JSON.stringify(before)).toBe(true);
    expect(result.error).toMatch(/timed out after 200ms/);
    expect(result.pid).toBeUndefined();
    expect(result.report).toBeUndefined();
    expect(result.launch).toEqual({ state: "not-started", supervisorPid, admissionGranted: false, supervisorReaped: true });
    // childReaped stays false: there was no worker to reap. The empty owned
    // session is proved by withheld admission AND the actual supervisor reap.
    expect(result.cleanup).toMatchObject({ childReaped: false, groupTerminated: true, sessionTerminated: true,
      pipesClosed: true, forcedPipeClose: false, containmentVerified: false, processGroups: [] });
    expect(existsSync(forkFile)).toBe(false);
    expect(existsSync(workerFile)).toBe(false);
    expect(lateLiveWorkers).toEqual([]);
    expect(alive(supervisorPid!)).toBe(false);
    expect(alive(sentinel.pid!)).toBe(true);
    expect(guardFired).toBe(false);
    expect(elapsedMs).toBeGreaterThanOrEqual(200);
    expect(elapsedMs).toBeLessThan(1_500);
  } finally {
    restore();
    if (guard) clearTimeout(guard);
    if (helper) await helper;
    if (supervisorPid !== undefined && alive(supervisorPid)) { finallyKilled.push(supervisorPid); killSupervisor!(); }
    // Failure-only recorded worker cleanup, never a numerical group signal.
    if (existsSync(workerFile)) {
      const worker = JSON.parse(readFileSync(workerFile, "utf8"));
      if (alive(worker.pid)) { finallyKilled.push(worker.pid); process.kill(worker.pid, "SIGKILL"); }
    }
    if (sentinel.exitCode === null && sentinel.signalCode === null) {
      const exited = new Promise<void>(resolve => sentinel.once("exit", () => resolve()));
      sentinel.kill("SIGKILL"); await exited;
    }
    rmSync(root, { recursive: true, force: true });
    console.log(JSON.stringify({ evidence: "recovery-cold-supervisor-finally", bootstrapDelayMs, finallyKilled, scratchRemoved: !existsSync(root) }));
    expect(finallyKilled).toEqual([]);
  }
}

describe.skipIf(process.platform === "win32")("cold recovery supervisor admission", () => {
  it("does not fork when bootstrap exceeds the 200ms execution deadline", () => coldSupervisorCase(350));
  it("reaps an unadmitted supervisor before returning even when bootstrap exceeds teardown", () => coldSupervisorCase(1_300));
  it("does not claim containment or success for a canonical worker that was never started", () => coldSupervisorCase(1_300, true));
});

async function rawAdmissionCase(mode: "closed-parent" | "queued-cancel" | "signal" | "no-grant") {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-admission-protocol-"));
  const wrapper = path.join(root, "proof-to-stdout.py");
  const forkFile = path.join(root, "unexpected-fork");
  const workFile = path.join(root, "unexpected-work");
  // Redirect the actual private channel to an ordinary captured stdout pipe;
  // this avoids adding any second owner for Bun's extra-stdio descriptors.
  // Messages and supervision still come from the complete production source.
  writeFileSync(wrapper, `import os, sys
os.dup2(1, 3)
_original_fork = os.fork
def observed_fork():
    open(${JSON.stringify(forkFile)}, 'w').write('fork')
    return _original_fork()
os.fork = observed_fork
sys.path.insert(0, ${JSON.stringify(path.dirname(supervisorScript))})
exec(compile(open(${JSON.stringify(supervisorScript)}, 'rb').read(), ${JSON.stringify(supervisorScript)}, 'exec'))
`);
  const child = childProcess.spawn("/usr/bin/python3", ["-S", "-u", wrapper,
    JSON.stringify({ command: ["/usr/bin/python3", "-S", "-c", `open(${JSON.stringify(workFile)}, 'w').write('work')`],
      timeoutMs: 200, contained: true })], { detached: true, stdio: ["pipe", "pipe", "pipe"] });
  let protocol = "";
  let pending = "";
  let stderr = "";
  let observedLive = false;
  let guard: ReturnType<typeof setTimeout> | undefined;
  let guardFired = false;
  child.stdout!.on("data", chunk => {
    protocol += chunk.toString(); pending += chunk.toString();
    let newline: number;
    while ((newline = pending.indexOf("\n")) !== -1) {
      const message = JSON.parse(pending.slice(0, newline)); pending = pending.slice(newline + 1);
      if (message.type === "prepared") {
        observedLive = alive(child.pid!);
        if (mode === "queued-cancel") child.stdin!.write("S200\nK");
        else if (mode === "signal") child.kill("SIGTERM");
      } else if (message.type === "ready") child.stdin!.write("K"); // Failure-only cancellation.
    }
  });
  child.stderr!.on("data", chunk => { stderr += chunk.toString(); });
  if (mode === "closed-parent") child.stdin!.end("S200\n"); // An old grant must not survive EOF.
  try {
    const exit = await Promise.race([new Promise<number | null>(resolve => child.once("close", resolve)),
      new Promise<never>((_resolve, reject) => { guard = setTimeout(() => {
        guardFired = true; child.kill("SIGKILL"); reject(new Error("Admission protocol did not terminate"));
      }, 1_800); })]);
    clearTimeout(guard);
    const messages = protocol.trim().split("\n").map(line => JSON.parse(line));
    console.log(JSON.stringify({ evidence: "recovery-admission-refusal-before-finally", mode, supervisorPid: child.pid,
      supervisorAbsent: !alive(child.pid!), observedLive, guardFired, exit, messages, stderr,
      forked: existsSync(forkFile), worked: existsSync(workFile) }));
    expect(exit, stderr).toBe(0);
    expect(guardFired).toBe(false);
    expect(messages).toEqual([
      { type: "prepared", supervisorPid: child.pid, sessionId: child.pid },
      { type: "not-started", supervisorPid: child.pid, sessionId: child.pid,
        reason: mode === "closed-parent" || mode === "queued-cancel" ? "cancelled" : mode === "signal" ? "parent-interrupted" : "admission-timeout" },
    ]);
    // closed-parent may already have exited by the time its buffered greeting
    // is consumed. All live-parent cancellation modes must observe the handle.
    if (mode !== "closed-parent") expect(observedLive).toBe(true);
    expect(alive(child.pid!)).toBe(false);
    expect(existsSync(forkFile)).toBe(false);
    expect(existsSync(workFile)).toBe(false);
  } finally {
    if (guard) clearTimeout(guard);
    child.stdin!.end();
    if (child.exitCode === null && child.signalCode === null) {
      const exited = new Promise<void>(resolve => child.once("exit", () => resolve()));
      child.kill("SIGKILL"); await exited;
    }
    rmSync(root, { recursive: true, force: true });
  }
}

async function malformedAdmissionCase(mode: "wrong-supervisor" | "unadmitted-worker" | "unadmitted-cleanup" | "truncated") {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-bad-admission-"));
  const wrapper = path.join(root, "malformed-supervisor.py");
  const forkFile = path.join(root, "unexpected-fork");
  writeFileSync(wrapper, `import json, os, sys
sys.path.insert(0, ${JSON.stringify(path.dirname(supervisorScript))})
scope = dict(__name__='protocol_negative_control')
exec(compile(open(${JSON.stringify(supervisorScript)}, 'rb').read(), ${JSON.stringify(supervisorScript)}, 'exec'), scope)
original_send = scope['send']
def malformed_send(message):
    if message['type'] == 'prepared':
        mode = ${JSON.stringify(mode)}
        if mode == 'wrong-supervisor':
            message['supervisorPid'] += 1
        elif mode == 'unadmitted-worker':
            message = dict(type='ready', pid=os.getpid(), sessionId=os.getpid(), processGroupId=os.getpid())
        elif mode == 'unadmitted-cleanup':
            message = dict(type='cleanup', childReaped=True, groupTerminated=True, sessionTerminated=True, containmentVerified=True, sigkillUsed=False, processGroups=[])
        elif mode == 'truncated':
            os.write(3, json.dumps(message).encode())
            return
    original_send(message)
scope['send'] = malformed_send
def forbidden_fork():
    open(${JSON.stringify(forkFile)}, 'w').write('fork')
    raise RuntimeError('Unexpected launch in negative protocol control')
os.fork = forbidden_fork
scope['main']()
`);
  let supervisorPid: number | undefined;
  let killSupervisor: (() => void) | undefined;
  const restore = interceptSupervisor(wrapper, (pid, kill) => { supervisorPid = pid; killSupervisor = kill; });
  let guard: ReturnType<typeof setTimeout> | undefined;
  let helper: Promise<RecoveryProcessResult> | undefined;
  try {
    helper = executeRecoveryWorker({ dbPath: path.join(root, "unused.sqlite"), docsDir: root,
      outputJson: path.join(root, "absent.json"), stateRoot: path.join(root, "state"), timeoutMs: 2_000,
      command: ["/usr/bin/python3", "-S", "-c", "pass"] });
    const result = await Promise.race([helper, new Promise<never>((_resolve, reject) => {
      guard = setTimeout(() => reject(new Error("Malformed admission exceeded bounded teardown")), 3_000);
    })]);
    clearTimeout(guard);
    console.log(JSON.stringify({ evidence: "recovery-malformed-admission-before-finally", mode, supervisorPid,
      supervisorAbsent: !alive(supervisorPid!), forked: existsSync(forkFile), result }));
    expect(result.error).toMatch(/supervisor protocol failed/);
    expect(result.error).toMatch(/lacks complete owned-session proof/);
    expect(result.launch).toEqual({ state: "unconfirmed", supervisorPid, admissionGranted: false, supervisorReaped: true });
    expect(result.pid).toBeUndefined();
    expect(result.cleanup).toMatchObject({ childReaped: false, groupTerminated: false, sessionTerminated: false,
      containmentVerified: false, pipesClosed: true, forcedPipeClose: false, processGroups: [] });
    expect(alive(supervisorPid!)).toBe(false);
    expect(existsSync(forkFile)).toBe(false);
  } finally {
    restore();
    if (guard) clearTimeout(guard);
    if (helper) await helper;
    if (supervisorPid !== undefined && alive(supervisorPid)) killSupervisor!();
    rmSync(root, { recursive: true, force: true });
  }
}

describe.skipIf(process.platform === "win32")("recovery admission protocol refuses unsafe launch", () => {
  for (const mode of ["closed-parent", "queued-cancel", "signal", "no-grant"] as const) {
    it(`does not fork for ${mode}`, () => rawAdmissionCase(mode));
  }
  for (const mode of ["wrong-supervisor", "unadmitted-worker", "unadmitted-cleanup", "truncated"] as const) {
    it(`does not certify ${mode} protocol`, () => malformedAdmissionCase(mode));
  }
});

async function gatedCancellationCase(mode: "eof" | "SIGTERM" | "SIGINT" | "cancel" | "malformed" | "duplicate" | "no-grant") {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-execution-gate-"));
  const wrapper = path.join(root, "proof-to-stdout.py");
  const workFile = path.join(root, "unexpected-work");
  // Python exec compiles only this fixed local source; no shell/user input.
  writeFileSync(wrapper, `import os,sys
os.dup2(1,3)
sys.path.insert(0, ${JSON.stringify(path.dirname(supervisorScript))})
exec(compile(open(${JSON.stringify(supervisorScript)}, 'rb').read(), ${JSON.stringify(supervisorScript)}, 'exec'))
`);
  const sentinel = childProcess.spawn("/usr/bin/python3", ["-S", "-c", "import time;time.sleep(60)"], { detached: true, stdio: "ignore" });
  const child = childProcess.spawn("/usr/bin/python3", ["-S", "-u", wrapper, JSON.stringify({
    command: ["/usr/bin/python3", "-S", "-c", `open(${JSON.stringify(workFile)}, 'w').write('work')`], timeoutMs: 200, contained: false,
  })], { detached: true, stdio: ["pipe", "pipe", "pipe"] });
  const messages: any[] = [];
  let pending = "";
  let stderr = "";
  let workerPid: number | undefined;
  let guard: ReturnType<typeof setTimeout> | undefined;
  let guardFired = false;
  const finallyKilled: number[] = [];
  child.stdout!.on("data", chunk => {
    pending += chunk.toString(); let newline: number;
    while ((newline = pending.indexOf("\n")) !== -1) {
      const message = JSON.parse(pending.slice(0, newline)); pending = pending.slice(newline + 1); messages.push(message);
      if (message.type === "prepared") child.stdin!.write("S200\n");
      if (message.type === "ready") {
        workerPid = message.pid;
        if (mode === "eof") child.stdin!.end();
        else if (mode === "SIGTERM" || mode === "SIGINT") child.kill(mode);
        else if (mode === "cancel") child.stdin!.write("K");
        else if (mode === "malformed") child.stdin!.write("X");
        else if (mode === "duplicate") child.stdin!.write("EE");
      }
    }
  });
  child.stderr!.on("data", chunk => { stderr += chunk.toString(); });
  try {
    const started = performance.now();
    const code = await Promise.race([new Promise<number | null>(resolve => child.once("close", resolve)),
      new Promise<never>((_resolve, reject) => { guard = setTimeout(() => {
        guardFired = true; reject(new Error("Gated cancellation exceeded 1800ms guard"));
      }, 1_800); })]);
    clearTimeout(guard);
    const elapsedMs = performance.now() - started;
    console.log(JSON.stringify({ evidence: "recovery-execution-gate-before-finally", mode, elapsedMs, supervisorPid: child.pid,
      supervisorAbsent: !alive(child.pid!), workerPid, workerAbsent: workerPid !== undefined && !alive(workerPid),
      sentinelAlive: alive(sentinel.pid!), guardFired, messages, stderr, work: existsSync(workFile) }));
    expect(code, stderr).toBe(0);
    expect(messages.map(message => message.type)).toEqual(["prepared", "ready", "exit", "cleanup"]);
    expect(messages[3]).toMatchObject({ childReaped: true, groupTerminated: true, sessionTerminated: true,
      containmentVerified: false, timedOut: mode === "no-grant" });
    if (mode === "malformed" || mode === "duplicate") expect(messages[3].errors.join(";")).toMatch(/Invalid recovery execution admission/);
    expect(workerPid).toBeTypeOf("number");
    expect(alive(workerPid!)).toBe(false);
    expect(alive(child.pid!)).toBe(false);
    expect(alive(sentinel.pid!)).toBe(true);
    expect(existsSync(workFile)).toBe(false);
    expect(guardFired).toBe(false);
    expect(elapsedMs).toBeLessThan(1_500);
  } finally {
    if (guard) clearTimeout(guard); child.stdin!.end();
    if (child.exitCode === null && child.signalCode === null) {
      finallyKilled.push(child.pid!); const exited = new Promise<void>(resolve => child.once("exit", () => resolve())); child.kill("SIGKILL"); await exited;
    }
    if (workerPid !== undefined && alive(workerPid)) { finallyKilled.push(workerPid); process.kill(workerPid, "SIGKILL"); }
    if (sentinel.exitCode === null && sentinel.signalCode === null) {
      const exited = new Promise<void>(resolve => sentinel.once("exit", () => resolve())); sentinel.kill("SIGKILL"); await exited;
    }
    rmSync(root, { recursive: true, force: true });
    console.log(JSON.stringify({ evidence: "recovery-execution-gate-finally", mode, finallyKilled }));
    expect(finallyKilled).toEqual([]);
  }
}

describe.skipIf(process.platform === "win32")("created recovery worker requires execution acknowledgement", () => {
  for (const mode of ["eof", "SIGTERM", "SIGINT", "cancel", "malformed", "duplicate", "no-grant"] as const) {
    it(`reaps the gated child without user work on ${mode}`, () => gatedCancellationCase(mode));
  }
});
