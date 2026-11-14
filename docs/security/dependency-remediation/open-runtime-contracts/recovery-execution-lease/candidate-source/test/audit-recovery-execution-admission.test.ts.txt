import { describe, expect, it } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { RecoveryProcessResult } from "../scripts/phase5_e2e_runner.js";

const source = fileURLToPath(new URL("../scripts/phase5_recovery_process.py", import.meta.url));
const runner = new URL("../scripts/phase5_e2e_runner.ts", import.meta.url).href;
const loader = fileURLToPath(new URL("../scripts/phase5_recovery_process_loader.mjs", import.meta.url));
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const read = (file: string) => existsSync(file) ? readFileSync(file, "utf8") : null;
function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") return false; throw error; }
}
function cleanEnv(root: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { PATH: process.env.PATH, LANG: "en_US.UTF-8", CI: "1", QMD_LLAMA_GPU: "false",
    HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_DATASETS_OFFLINE: "1", PYTHONDONTWRITEBYTECODE: "1" };
  for (const [key, name] of Object.entries({ HOME: "home", HERMES_HOME: "hermes", XDG_CACHE_HOME: "cache",
    XDG_CONFIG_HOME: "config", TMPDIR: "tmp", TMP: "tmp", TEMP: "tmp" })) {
    mkdirSync(path.join(root, name), { recursive: true }); env[key] = path.join(root, name);
  }
  return env;
}
async function until(predicate: () => boolean, deadline: number, message: string) {
  while (!predicate() && performance.now() < deadline) await sleep(2);
  expect(predicate(), message).toBe(true);
}
async function reap(child: ChildProcess, killed: number[]) {
  if (child.exitCode === null && child.signalCode === null) {
    killed.push(child.pid!);
    const exited = new Promise<void>(resolve => child.once("exit", () => resolve())); child.kill("SIGKILL"); await exited;
  }
}

type Stage = "none" | "supervisor-consumed" | "supervisor-forwarded" | "child-consumed" | "child-validation" | "child-admitted";
type Fault = "control-read-error" | "proof-write-error" | "cancel-read-hold";
function instrument(root: string, stage: Stage, pause: "delay" | "stop", proofToStdout: boolean, fault?: Fault, forwardDelayMs = 250): string {
  const wrapper = path.join(root, "execution-boundary.py");
  writeFileSync(wrapper, `import json,os,select,signal,sys,time
source=${JSON.stringify(source)}
${proofToStdout ? "os.dup2(1,3)" : ""}
sys.path.insert(0,os.path.dirname(source))
scope=dict(__name__='execution_admission_regression')
exec(compile(open(source,'rb').read(),source,'exec'),scope)
original_read,original_write,original_fork,original_waitpid=os.read,os.write,os.fork,os.waitpid
owner=os.getpid()
held=False
def hold(label):
    global held
    if held or label!=${JSON.stringify(stage)}: return
    held=True
    with open(${JSON.stringify(path.join(root, "stage.json"))},'w') as out:
        json.dump(dict(stage=label,pid=os.getpid(),owner=owner,at=time.monotonic()),out)
    ${pause === "stop" ? "os.kill(os.getpid(),signal.SIGSTOP)" : "time.sleep(1.300)"}
def read(fd,n):
    data=original_read(fd,n)
    if os.getpid()==owner and fd==0 and data==b'K' and ${fault === "cancel-read-hold" ? "True" : "False"}:
        frame=sys._getframe().f_back
        while frame and not (frame.f_code.co_filename==source and frame.f_code.co_name=='main'):
            frame=frame.f_back
        # Consuming K after real launch-pipe EOF (exec OR child exit) is the
        # legitimate handoff, not pre-admission revocation erasure.
        pre_exec=frame is not None and frame.f_locals.get('launch_eof') is False
        with open(${JSON.stringify(path.join(root, "control-reads.jsonl"))},'a') as out:
            out.write(json.dumps(dict(pid=os.getpid(),preExec=pre_exec))+'\\n')
        if pre_exec:
            with open(${JSON.stringify(path.join(root, "stage.json"))},'w') as out:
                json.dump(dict(stage='cancel-consumed',pid=os.getpid()),out)
            time.sleep(1.300)
    if data==b'E':
        hold('supervisor-consumed' if os.getpid()==owner and fd==0 else 'child-consumed')
        # At the post-forward boundary, the child must revalidate a grant that
        # actually aged past its lease, not already execute before parent death.
        if os.getpid()!=owner and ${stage === "supervisor-forwarded" && forwardDelayMs > 0 ? "True" : "False"}: time.sleep(${forwardDelayMs / 1000})
        if os.getpid()!=owner and ${fault === "control-read-error" || fault === "cancel-read-hold" ? "True" : "False"}: time.sleep(.050)
    return data
def write(fd,data):
    if os.getpid()==owner and fd!=3 and data==b'E' and ${stage === "supervisor-forwarded" && forwardDelayMs === 0 ? "True" : "False"}:
        # Isolate live-channel revocation from lease expiry: observe REAL queued
        # EOF before forwarding, rather than spending 50ms of a 200ms lease on
        # an artificial child sleep. EOF is persistent, not consumed authority.
        readable,_,_=select.select([0],[],[],.050)
        assert readable and original_read(0,1)==b''
        with open(${JSON.stringify(path.join(root, "queued-eof.json"))},'w') as out:
            json.dump(dict(pid=os.getpid(),at=time.monotonic()),out)
    written=original_write(fd,data)
    if os.getpid()==owner and fd!=3 and data==b'E':
        with open(${JSON.stringify(path.join(root, "gate-forwarded.json"))},'w') as out:
            json.dump(dict(pid=os.getpid()),out)
        if ${fault === "control-read-error" ? "True" : "False"}: os.close(0)
        hold('supervisor-forwarded')
        # Preserve the reviewer's real runnable-child slice after forwarding.
        if ${stage === "supervisor-consumed" ? "True" : "False"}: time.sleep(.050)
    return written
def fork():
    pid=original_fork()
    if pid:
        with open(${JSON.stringify(path.join(root, "forks.jsonl"))},'a') as out:
            out.write(json.dumps(dict(parent=os.getpid(),pid=pid))+'\\n')
    return pid
def waitpid(*args):
    result=original_waitpid(*args)
    with open(${JSON.stringify(path.join(root, "reaps.jsonl"))},'a') as out:
        out.write(json.dumps(dict(owner=os.getpid(),pid=result[0],status=result[1]))+'\\n')
    return result
os.read,os.write,os.fork,os.waitpid=read,write,fork,waitpid
if ${fault === "proof-write-error" ? "True" : "False"}:
    original_send=scope['send']
    proof_closed=False
    def send(message):
        global proof_closed
        if message['type']=='ready' and not proof_closed:
            proof_closed=True
            os.close(3) # A real EBADF on the next production proof write.
        original_send(message)
    scope['send']=send
if 'execution_allowed' in scope:
    original_allowed=scope['execution_allowed']
    def allowed(*args):
        hold('child-validation')
        remaining=(args[1]-time.monotonic())*1000
        result=original_allowed(*args)
        with open(${JSON.stringify(path.join(root, "validation.json"))},'w') as out:
            json.dump(dict(pid=os.getpid(),allowed=result,remainingMs=remaining),out)
        if result: hold('child-admitted')
        return result
    scope['execution_allowed']=allowed
scope['main']()
`);
  return wrapper;
}

async function rawCase(stage: Stage, pause: "delay" | "stop", grant: "open" | "eof" | "none", fault?: Fault, forwardDelayMs = 250) {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-execution-admission-raw-"));
  const env = cleanEnv(root);
  const wrapper = instrument(root, stage, pause, true, fault, forwardDelayMs);
  const work = path.join(root, "work");
  const positive = !fault && stage === "none" && grant === "open";
  const sentinel = spawn("/usr/bin/python3", ["-B", "-S", "-c", "import time;time.sleep(60)"], { env, detached: true, stdio: "ignore" });
  const child = spawn("/usr/bin/python3", ["-B", "-S", "-u", wrapper, JSON.stringify({
    command: ["/bin/sh", "-c", 'printf actual-user-work > "$1"', "owned-shell", work], timeoutMs: positive ? 5_000 : 200, contained: false,
  })], { env, detached: true, stdio: ["pipe", "pipe", "pipe"] });
  const messages: any[] = [];
  let pending = "", stderr = "", worker: number | undefined;
  let guardFired = false;
  const killed: number[] = [];
  const started = performance.now();
  const closed = new Promise<void>(resolve => child.once("close", () => resolve()));
  child.stdout!.on("data", chunk => {
    pending += chunk.toString(); let newline: number;
    while ((newline = pending.indexOf("\n")) !== -1) {
      const message = JSON.parse(pending.slice(0, newline)); pending = pending.slice(newline + 1); messages.push(message);
      if (message.type === "prepared") child.stdin!.write(positive ? "S5000\n" : "S200\n");
      if (message.type === "ready") {
        worker = message.pid;
        if (grant !== "none") child.stdin!.write("E");
        if (grant !== "open" && !stage.startsWith("child-")) child.stdin!.end();
      }
    }
  });
  child.stdin!.on("error", () => {});
  child.stderr!.on("data", chunk => { stderr += chunk.toString(); });
  // Child-boundary probes first prove that the actual child reached the held
  // boundary. Otherwise a fast supervisor EOF cleanup could miss that boundary.
  const closeAfterAdmission = stage.startsWith("child-") ? setInterval(() => {
    if (existsSync(path.join(root, "stage.json"))) child.stdin!.end();
  }, 2) : undefined;
  let cancelSent = false;
  const cancelAfterForward = fault === "cancel-read-hold" ? setInterval(() => {
    if (!cancelSent && existsSync(path.join(root, "gate-forwarded.json"))) { cancelSent = true; child.stdin!.write("K"); }
  }, 2) : undefined;
  const guard = setTimeout(() => { guardFired = true; child.kill("SIGKILL"); }, positive ? 5_000 : 1_800);
  try {
    await closed; clearTimeout(guard);
    const elapsedMs = performance.now() - started;
    // No assertion/finally cleanup before the original scheduling window ends.
    if (stage !== "none") await sleep(Math.max(0, 1_450 - elapsedMs));
    const forkRecords = (read(path.join(root, "forks.jsonl")) || "").trim().split("\n").filter(Boolean).map(line => JSON.parse(line) as { pid: number });
    worker ??= forkRecords[0]?.pid;
    const reaps = (read(path.join(root, "reaps.jsonl")) || "").trim().split("\n").filter(Boolean).map(line => JSON.parse(line) as { pid: number });
    const validationText = read(path.join(root, "validation.json"));
    const validation = validationText ? JSON.parse(validationText) as { allowed: boolean; remainingMs: number } : null;
    const controlReads = (read(path.join(root, "control-reads.jsonl")) || "").trim().split("\n").filter(Boolean).map(line => JSON.parse(line) as { preExec: boolean });
    const observation = { evidence: "recovery-execution-admission-before-finally", stage, pause, grant, fault, forwardDelayMs, elapsedMs, guardFired,
      supervisorPid: child.pid, supervisorAbsent: !alive(child.pid!), workerPid: worker, workerAbsent: worker !== undefined && !alive(worker),
      sentinelAlive: alive(sentinel.pid!), work: read(work), stageRecord: read(path.join(root, "stage.json")), queuedEof: read(path.join(root, "queued-eof.json")),
      reaps, validation, controlReads, messages, stderr };
    console.log(JSON.stringify(observation));
    expect(guardFired, JSON.stringify(observation)).toBe(false);
    expect(observation.supervisorAbsent).toBe(true); expect(observation.workerAbsent).toBe(true);
    expect(observation.sentinelAlive).toBe(true);
    expect(observation.work).toBe(positive ? "actual-user-work" : null);
    // A broken proof channel cannot provide a cleanup receipt. The fixture
    // records the REAL waitpid return, never a forged supervisor message.
    expect(reaps.some(item => item.pid === worker)).toBe(true);
    if (fault !== "proof-write-error") expect(messages.at(-1)).toMatchObject({ type: "cleanup", childReaped: true,
      sessionTerminated: true, groupTerminated: true, containmentVerified: false });
    if (stage !== "none") expect(JSON.parse(observation.stageRecord!).stage).toBe(stage);
    if (stage === "supervisor-forwarded" && forwardDelayMs === 0) {
      expect(read(path.join(root, "queued-eof.json"))).not.toBeNull();
      expect(validation, JSON.stringify(observation)).not.toBeNull();
      expect(validation!.allowed).toBe(false); expect(validation!.remainingMs).toBeGreaterThan(0);
    }
    if (stage === "supervisor-forwarded" && grant === "open") {
      expect(validation, JSON.stringify(observation)).not.toBeNull();
      expect(validation!.allowed).toBe(false); expect(validation!.remainingMs).toBeLessThan(0);
    }
    if (stage !== "none" && grant === "open") expect(messages.at(-1).timedOut).toBe(true);
    if (fault === "cancel-read-hold") {
      expect(cancelSent).toBe(true);
      expect(observation.stageRecord, "supervisor must not erase pre-admission K").toBeNull();
      expect(controlReads.every(item => !item.preExec)).toBe(true);
    }
  } finally {
    clearTimeout(guard); if (closeAfterAdmission) clearInterval(closeAfterAdmission); if (cancelAfterForward) clearInterval(cancelAfterForward);
    child.stdin!.end(); await reap(child, killed);
    if (worker !== undefined && alive(worker)) { killed.push(worker); process.kill(worker, "SIGKILL"); }
    const sentinelKills: number[] = []; await reap(sentinel, sentinelKills);
    console.log(JSON.stringify({ evidence: "recovery-execution-admission-finally", stage, pause, grant, fault, forwardDelayMs, finallyKilled: killed }));
    rmSync(root, { recursive: true, force: true }); expect(killed).toEqual([]);
  }
}

async function hostCase(runtime: "node" | "bun", stage: Stage, pause: "delay" | "stop", mode: "valid" | "alive" | "graceful" | "hard") {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-execution-admission-host-"));
  const env = cleanEnv(root);
  const wrapper = instrument(root, stage, pause, false);
  const hostScript = path.join(root, "host.ts");
  writeFileSync(hostScript, `import childProcess from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
import {existsSync,writeFileSync} from 'node:fs';
import {executeRecoveryWorker} from ${JSON.stringify(runner)};
const source=${JSON.stringify(source)},wrapper=${JSON.stringify(wrapper)},root=${JSON.stringify(root)};
if ((process.versions as Record<string,string>).bun) {
  const bun=(globalThis as any).Bun,original=bun.spawn;
  bun.spawn=function(command:string[],options:unknown) {
    return original.call(bun,command[3]===source?[...command.slice(0,3),wrapper,...command.slice(4)]:command,options);
  };
} else {
  const original=childProcess.spawn;
  (childProcess as any).spawn=function(executable:string,args:string[],options:unknown) {
    return original(executable,args[2]===source?[...args.slice(0,2),wrapper,...args.slice(3)]:args,options as any);
  }; syncBuiltinESMExports();
}
const events=['exit','SIGTERM','SIGINT'];
const hooksBefore=events.map(event=>process.listenerCount(event));
writeFileSync(root+'/started.json',JSON.stringify({executable:process.execPath,versions:process.versions}));
const started=performance.now();
const helper=executeRecoveryWorker({dbPath:root+'/unused.sqlite',docsDir:root,outputJson:root+'/absent-report.json',stateRoot:root+'/state',
  timeoutMs:${mode === "valid" ? 5_000 : 200},command:['/bin/sh','-c','printf actual-user-work > "$1"','owned-shell',root+'/work']});
const exitPoll=${mode === "graceful" ? "setInterval(()=>{if(existsSync(root+'/stage.json'))process.exit(17);},2)" : "undefined"};
let guardFired=false;
const guard=setTimeout(()=>{guardFired=true;process.exitCode=99;},${mode === "valid" ? 6_000 : 1_800});
const result=await helper;
clearTimeout(guard);if(exitPoll)clearInterval(exitPoll);
writeFileSync(root+'/helper.json',JSON.stringify({elapsedMs:performance.now()-started,hooksBefore,hooksAfter:events.map(event=>process.listenerCount(event)),guardFired,result}));
`);
  const executable = runtime === "node" ? process.execPath : process.env.BUN_EXECUTABLE || "bun";
  const args = runtime === "node" ? ["--import", loader, hostScript] : [hostScript];
  const sentinel = spawn("/usr/bin/python3", ["-B", "-S", "-c", "import time;time.sleep(60)"], { env, detached: true, stdio: "ignore" });
  const host = spawn(executable, args, { env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "", stderr = "";
  host.stdout!.on("data", chunk => { stdout += chunk.toString(); }); host.stderr!.on("data", chunk => { stderr += chunk.toString(); });
  const killed: number[] = [];
  const forks = () => (read(path.join(root, "forks.jsonl")) || "").trim().split("\n").filter(Boolean).map(line => JSON.parse(line) as { parent: number; pid: number });
  let guardFired = false;
  try {
    await until(() => existsSync(path.join(root, "started.json")) || host.exitCode !== null, performance.now() + 5_000, "host bootstrap");
    expect(read(path.join(root, "started.json")), stderr).not.toBeNull();
    const started = performance.now();
    const deadline = started + (mode === "valid" ? 6_000 : 1_800);
    if (mode !== "valid") {
      await until(() => existsSync(path.join(root, "stage.json")), deadline, `actual ${stage} must be reached: ${stderr}`);
      if (mode === "hard") host.kill("SIGKILL"); // An actual abrupt death, with NO JS exit hook.
    }
    while (host.exitCode === null && host.signalCode === null && performance.now() < deadline) await sleep(2);
    guardFired = host.exitCode === null && host.signalCode === null;
    const atHostExit = { elapsedMs: performance.now() - started, exitCode: host.exitCode, signal: host.signalCode,
      recoveryAlive: forks().filter(item => alive(item.parent) || alive(item.pid)), work: read(path.join(root, "work")) };
    if (mode !== "valid") await sleep(Math.max(0, 1_500 - (performance.now() - started)));
    const children = forks();
    const helperText = read(path.join(root, "helper.json"));
    const helper = helperText ? JSON.parse(helperText) as { elapsedMs: number; guardFired: boolean; hooksBefore: number[]; hooksAfter: number[]; result: RecoveryProcessResult } : null;
    const observation = { evidence: "recovery-parent-death-before-finally", runtime, mode, stage, pause, executable,
      nativeRuntime: JSON.parse(read(path.join(root, "started.json"))!), atHostExit, guardFired, forks: children,
      supervisorsAbsent: children.length === 1 && children.every(item => !alive(item.parent)),
      workersAbsent: children.length === 1 && children.every(item => !alive(item.pid)), sentinelAlive: alive(sentinel.pid!),
      work: read(path.join(root, "work")), stageRecord: read(path.join(root, "stage.json")), helper, stdout, stderr };
    console.log(JSON.stringify(observation));
    expect(guardFired, JSON.stringify(observation)).toBe(false);
    expect(observation.supervisorsAbsent, JSON.stringify(observation)).toBe(true); expect(observation.workersAbsent).toBe(true);
    expect(observation.sentinelAlive).toBe(true); expect(observation.work).toBe(mode === "valid" ? "actual-user-work" : null);
    if (mode === "alive" || mode === "valid") {
      expect(helper, JSON.stringify(observation)).not.toBeNull(); expect(helper!.guardFired).toBe(false);
      expect(helper!.hooksAfter).toEqual(helper!.hooksBefore);
      expect(helper!.result.launch).toMatchObject({ state: "started", admissionGranted: true, supervisorReaped: true });
      expect(helper!.result.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, sessionTerminated: true,
        pipesClosed: true, forcedPipeClose: false, containmentVerified: false });
      if (mode === "alive") {
        expect(helper!.result.timedOut).toBe(true); expect(helper!.elapsedMs).toBeGreaterThanOrEqual(200); expect(helper!.elapsedMs).toBeLessThan(1_500);
      }
    } else {
      expect(helper).toBeNull();
      if (mode === "hard") expect(host.signalCode).toBe("SIGKILL"); else expect(host.exitCode).toBe(17);
    }
  } finally {
    await reap(host, killed);
    for (const item of forks()) for (const pid of [item.parent, item.pid]) if (alive(pid)) { killed.push(pid); process.kill(pid, "SIGKILL"); }
    const sentinelKills: number[] = []; await reap(sentinel, sentinelKills);
    console.log(JSON.stringify({ evidence: "recovery-parent-death-finally", runtime, mode, stage, pause, finallyKilled: killed }));
    rmSync(root, { recursive: true, force: true }); expect(killed).toEqual([]);
  }
}

async function invalidBudgetFrame(frame: string) {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-execution-budget-frame-"));
  const env = cleanEnv(root);
  const wrapper = instrument(root, "none", "delay", true);
  const work = path.join(root, "work");
  const sentinel = spawn("/usr/bin/python3", ["-B", "-S", "-c", "import time;time.sleep(60)"], { env, detached: true, stdio: "ignore" });
  const child = spawn("/usr/bin/python3", ["-B", "-S", "-u", wrapper, JSON.stringify({ command: ["/bin/sh", "-c",
    'printf actual-user-work > "$1"', "owned-shell", work], timeoutMs: 200, contained: false })],
  { env, detached: true, stdio: ["pipe", "pipe", "pipe"] });
  let protocol = "", stderr = "", sent = false, guardFired = false;
  const killed: number[] = [];
  child.stdout!.on("data", chunk => {
    protocol += chunk.toString();
    if (!sent && protocol.includes('"prepared"')) { sent = true; child.stdin!.end(frame); }
  });
  child.stdin!.on("error", () => {});
  child.stderr!.on("data", chunk => { stderr += chunk.toString(); });
  const closed = new Promise<void>(resolve => child.once("close", () => resolve()));
  const guard = setTimeout(() => { guardFired = true; child.kill("SIGKILL"); }, 1_800);
  try {
    await closed; clearTimeout(guard);
    const observation = { evidence: "recovery-budget-frame-before-finally", frame, protocol, stderr, guardFired,
      supervisorAbsent: !alive(child.pid!), forked: existsSync(path.join(root, "forks.jsonl")), work: read(work), sentinelAlive: alive(sentinel.pid!) };
    console.log(JSON.stringify(observation));
    expect(sent).toBe(true); expect(guardFired).toBe(false); expect(observation.supervisorAbsent).toBe(true);
    expect(observation.forked).toBe(false); expect(observation.work).toBeNull(); expect(observation.sentinelAlive).toBe(true);
    expect(protocol).not.toContain('"ready"');
  } finally {
    clearTimeout(guard); await reap(child, killed); const sentinelKills: number[] = []; await reap(sentinel, sentinelKills);
    console.log(JSON.stringify({ evidence: "recovery-budget-frame-finally", frame, finallyKilled: killed }));
    rmSync(root, { recursive: true, force: true }); expect(killed).toEqual([]);
  }
}

describe.skipIf(process.platform === "win32")("execution authority cannot replay after the caller dies", () => {
  it("actually executes a valid raw grant (separate 5s readiness control)", () => rawCase("none", "delay", "open"));
  it("does not execute on EOF without E", () => rawCase("none", "delay", "none"));
  it("expires consumed E autonomously with a still-open parent and no cancellation byte", () => rawCase("supervisor-consumed", "delay", "open"));
  it("expires an already-forwarded grant at the child with a still-open parent", () => rawCase("supervisor-forwarded", "delay", "open"));
  it("rejects caller EOF at the child while its execution lease is still valid", () => rawCase("supervisor-forwarded", "delay", "eof", undefined, 0));
  for (const fault of ["control-read-error", "proof-write-error"] as const) {
    it(`retains anchored cleanup after a real terminal ${fault}`, () => rawCase("none", "delay", "open", fault));
  }
  it("does not erase queued cancellation before the child's last gate", () => rawCase("none", "delay", "open", "cancel-read-hold"));
  for (const stage of ["supervisor-consumed", "supervisor-forwarded", "child-consumed", "child-validation", "child-admitted"] as const) {
    it(`rejects stale permission across a 1300ms ${stage} hold and EOF`, () => rawCase(stage, "delay", "eof"));
  }
  for (const stage of ["supervisor-consumed", "child-consumed"] as const) {
    it(`autonomously cleans a real stopped ${stage} on EOF, without fixture CONT`, () => rawCase(stage, "stop", "eof"));
  }
  for (const frame of ["S0\n", "S201\n", "S200\nS200\n", "S200", "S999999999999999999999\n", "S+1\n", "Sbad\nK"] as const) {
    it(`refuses bounded malformed/duplicate/truncated budget ${JSON.stringify(frame)}`, () => invalidBudgetFrame(frame));
  }
  for (const runtime of ["node", "bun"] as const) {
    it(`actually executes in real ${runtime} (separate 5s readiness control)`, () => hostCase(runtime, "none", "delay", "valid"));
    for (const stage of ["supervisor-consumed", "supervisor-forwarded", "child-admitted"] as const) {
      for (const mode of ["alive", "graceful", "hard"] as const) {
        it(`cleans ${runtime} ${mode} after ${stage} without late work`, () => hostCase(runtime, stage, "delay", mode));
      }
    }
    it(`cleans real ${runtime} SIGKILL with its supervisor SIGSTOPped`, () => hostCase(runtime, "supervisor-consumed", "stop", "hard"));
  }
});
