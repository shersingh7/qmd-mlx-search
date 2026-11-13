// Direct native-Bun acceptance probe: no Vitest host, mock, or transport adapter.
import assert from "node:assert/strict";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { closeSync, existsSync, fstatSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { executeRecoveryWorker, type RecoveryProcessResult } from "../../scripts/phase5_e2e_runner.js";

const bunVersion = (process.versions as Record<string, string>).bun;
assert.ok(bunVersion, "This regression must execute in native Bun, not a Node-hosted runner");
const bun = (globalThis as unknown as { Bun: { gc(full: boolean): void } }).Bun;
const scenario = process.argv[2];
assert.ok(scenario === "sequential-gc" || scenario === "fd-reuse", "Unknown native recovery scenario");
console.log(JSON.stringify({ evidence: "native-recovery-runtime", scenario, bun: bunVersion, executable: process.execPath }));

function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") return false; throw error; }
}
async function waitAbsent(pid: number): Promise<void> {
  const deadline = performance.now() + 1_000;
  while (alive(pid) && performance.now() < deadline) await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(alive(pid), false, `Fixture PID ${pid} must be absent`);
}
async function stopSentinel(child: ChildProcess): Promise<void> {
  if (child.exitCode === null && child.signalCode === null) {
    const exited = new Promise<void>(resolve => child.once("exit", () => resolve()));
    child.kill("SIGKILL");
    await exited;
  }
  await waitAbsent(child.pid!);
}

// Observe real parent-side descriptors without replacing/intercepting spawn.
function pipes(): number[] {
  const result: number[] = [];
  for (let fd = 3; fd < 512; fd++) {
    try { const stat = fstatSync(fd); if (stat.isFIFO() || stat.isSocket()) result.push(fd); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "EBADF") throw error; }
  }
  return result;
}

type Identity = { pid: number; pgid: number; sid: number };
async function call(root: string, sentinel: ChildProcess, index: number, timeoutMs: number, timeout: boolean,
  forceGC: boolean, priorFDs: Set<number>, fdSentinel: number): Promise<void> {
  const stateRoot = path.join(root, String(index));
  const readyPath = path.join(root, `ready-${index}.json`);
  const releasePath = path.join(root, `release-${index}`);
  const outputJson = path.join(root, `report-${index}.json`);
  const dbPath = path.join(root, "unused.sqlite");
  const marker = `native-owned-output-${index}`;
  const worker = `
import json, os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(${JSON.stringify(readyPath + ".tmp")}, 'w') as record:
    json.dump(dict(pid=os.getpid(), pgid=os.getpgrp(), sid=os.getsid(0)), record)
os.replace(${JSON.stringify(readyPath + ".tmp")}, ${JSON.stringify(readyPath)})
${timeout ? "time.sleep(60)" : `while not os.path.exists(${JSON.stringify(releasePath)}):
    time.sleep(0.002)
print(${JSON.stringify(marker)}, flush=True)
with open(${JSON.stringify(outputJson)}, 'w') as report:
    json.dump(dict(status='completed', pid=os.getpid(), dbPath=${JSON.stringify(dbPath)}, action='resume', docsCompleted=1, chunksCommitted=1, totalVectors=1, checkpointStatus=None, consecutiveSequenceVerified=True, hashes=[]), report)`}
`;
  const started = performance.now();
  const beforeFDs = new Set(pipes());
  let identity: Identity | undefined;
  let guard: ReturnType<typeof setTimeout> | undefined;
  let guardFired = false;
  let gcDuringActive = false;
  let settled: RecoveryProcessResult | undefined;
  let activeFDs: number[] = [];
  let reusedFDs: number[] = [];
  const finallyKilled: number[] = [];
  const helper = executeRecoveryWorker({ dbPath, docsDir: root, action: "resume", outputJson, stateRoot,
    timeoutMs, command: ["/usr/bin/python3", "-S", "-u", "-c", worker] })
    .then(result => { settled = result; return result; });
  try {
    const observeThenRun = async () => {
      if (forceGC) {
        while (!existsSync(readyPath) && !settled) await new Promise(resolve => setTimeout(resolve, 5));
        assert.ok(existsSync(readyPath), `Worker must be ready before active-call GC: ${JSON.stringify(settled)}`);
        identity = JSON.parse(readFileSync(readyPath, "utf8")) as Identity;
        assert.equal(alive(identity.pid), true);
        assert.equal(alive(sentinel.pid!), true);
        activeFDs = pipes().filter(fd => !beforeFDs.has(fd));
        assert.ok(activeFDs.length > 0, "Must observe real active recovery pipes");
        reusedFDs = activeFDs.filter(fd => priorFDs.has(fd));
        for (const fd of activeFDs) priorFDs.add(fd);
        // The previous invocation has returned; collect its unreachable owners
        // only AFTER the next invocation has acquired its real pipes. Exercise
        // both a full collection and intervening unrelated descriptor reuse.
        bun.gc(true);
        const churn = Array.from({ length: 24 }, () => openSync(path.join(root, "fd-churn"), "a+"));
        try {
          bun.gc(true);
          for (const fd of churn) { assert.equal(fstatSync(fd).isFile(), true); writeSync(fd, "."); }
          assert.equal(fstatSync(fdSentinel).isFile(), true);
          writeSync(fdSentinel, `${index}\n`);
        } finally { for (const fd of churn) closeSync(fd); }
        gcDuringActive = !settled && alive(identity.pid);
        assert.equal(gcDuringActive, true, "GC must occur while this worker and its proof channel are active");
      }
      if (!timeout) writeFileSync(releasePath, "release\n");
      return await helper;
    };
    const result = await Promise.race([observeThenRun(), new Promise<never>((_resolve, reject) => {
      guard = setTimeout(() => { guardFired = true; reject(new Error("Native recovery exceeded execution + teardown budget")); }, timeoutMs + 800);
    })]);
    clearTimeout(guard);
    const elapsedMs = performance.now() - started;
    if (!identity && existsSync(readyPath)) identity = JSON.parse(readFileSync(readyPath, "utf8")) as Identity;
    // Preserve failure evidence BEFORE any assertion/finally cleanup can mask it.
    console.log(JSON.stringify({ evidence: "native-recovery-before-finally", scenario, index, timeoutMs, elapsedMs,
      gcDuringActive, activeFDs, reusedFDs, identity, parentAbsent: result.pid !== undefined && !alive(result.pid),
      sentinelPid: sentinel.pid, sentinelAlive: alive(sentinel.pid!), descriptorSentinelOpen: fstatSync(fdSentinel).isFile(), guardFired, result }));
    assert.equal(guardFired, false);
    assert.ok(elapsedMs < (timeoutMs === 200 ? 1_500 : timeoutMs + 500));
    assert.ok(result.launch.supervisorPid !== undefined);
    assert.equal(result.launch.supervisorReaped, true);
    await waitAbsent(result.launch.supervisorPid!);
    const noStart = result.launch.state === "not-started";
    if (noStart) {
      // Only a short, non-active cold call can withhold admission. Active GC /
      // descriptor-reuse calls retain every original started-worker assertion.
      assert.equal(timeoutMs, 200);
      assert.equal(forceGC, false);
      assert.equal(result.launch.admissionGranted, false);
      assert.equal(result.pid, undefined);
      assert.equal(identity, undefined);
      assert.deepEqual(result.cleanup.processGroups, []);
      assert.equal(result.exitCode, null);
      assert.equal(result.signal, null);
    } else {
      assert.equal(result.launch.state, "started");
      assert.equal(result.launch.admissionGranted, true);
      assert.ok(result.pid !== undefined);
      assert.ok(result.cleanup.processGroups.includes(result.pid!));
    }
    if (identity) {
      assert.equal(result.pid, identity.pid);
      assert.equal(result.cleanup.processGroupId, identity.pgid);
      assert.equal(result.cleanup.sessionId, identity.sid);
      assert.equal(identity.pgid, identity.pid);
      assert.notEqual(identity.sid, identity.pid);
    }
    assert.equal(result.cleanup.childReaped, !noStart);
    assert.equal(result.cleanup.groupTerminated, true);
    assert.equal(result.cleanup.sessionTerminated, true);
    assert.equal(result.cleanup.pipesClosed, true);
    assert.equal(result.cleanup.forcedPipeClose, false);
    assert.equal(result.cleanup.containmentVerified, false);
    assert.match(result.error!, /command injection is uncontained.*cannot be certified/);
    assert.doesNotMatch(result.error!, /lacks complete owned-session proof|bounded cleanup proof|protocol failed|proof stream failed|pipe.*(?:close|failed)/i);
    assert.equal(result.timedOut, timeout);
    if (timeout) {
      assert.ok(elapsedMs >= timeoutMs);
      assert.equal(result.exitCode, null);
      // Like the standalone 200ms contract, the first cold invocation need
      // not have installed its handler. All active-GC calls prove readiness.
      if (identity) {
        assert.equal(result.signal, "SIGKILL");
        assert.equal(result.cleanup.sigkillUsed, true);
      } else if (!noStart) assert.ok(result.signal === "SIGTERM" || result.signal === "SIGKILL");
      assert.match(result.error!, new RegExp(`timed out after ${timeoutMs}ms`));
    } else {
      assert.equal(result.exitCode, 0);
      assert.equal(result.signal, null);
      assert.equal(result.report?.status, "completed");
      assert.ok(result.stdout.includes(marker));
    }
    if (!noStart) await waitAbsent(result.pid!);
    assert.equal(alive(sentinel.pid!), true);
    if (scenario === "fd-reuse" && index > 1) assert.ok(reusedFDs.length > 0, "Stress must actually reuse recovery descriptor numbers");
  } finally {
    if (guard) clearTimeout(guard);
    const result = await helper;
    if (!identity && existsSync(readyPath)) identity = JSON.parse(readFileSync(readyPath, "utf8")) as Identity;
    if (guardFired) console.error(JSON.stringify({ evidence: "native-recovery-guard-before-finally", scenario, index,
      identity, gcDuringActive, activeFDs, reusedFDs, parentAbsent: result.pid !== undefined && !alive(result.pid),
      sentinelPid: sentinel.pid, sentinelAlive: alive(sentinel.pid!), guardFired, result }));
    // A failure-only net: a PID file is not permission to kill a recycled PID.
    if (identity && alive(identity.pid)) {
      const lookup = spawnSync("/usr/bin/python3", ["-S", "-c",
        "import json,os,sys; p=int(sys.argv[1]); print(json.dumps(dict(pid=p,pgid=os.getpgid(p),sid=os.getsid(p))))", String(identity.pid)], { encoding: "utf8" });
      if (lookup.status === 0 && JSON.stringify(JSON.parse(lookup.stdout)) === JSON.stringify(identity)) {
        finallyKilled.push(identity.pid); process.kill(identity.pid, "SIGKILL");
      }
    }
    if (result.pid !== undefined) await waitAbsent(result.pid);
    console.log(JSON.stringify({ evidence: "native-recovery-call-finally", scenario, index, guardFired, finallyKilled }));
    assert.deepEqual(finallyKilled, []);
  }
}

const root = mkdtempSync(path.join(tmpdir(), "qmd-native-bun-ownership-"));
const sentinel = spawn("/usr/bin/python3", ["-S", "-c", "import time; time.sleep(60)"], { detached: true, stdio: "ignore" });
const fdSentinel = openSync(path.join(root, "descriptor-sentinel"), "a+");
try {
  const priorFDs = new Set<number>();
  await call(root, sentinel, 0, 200, true, false, priorFDs, fdSentinel);
  if (scenario === "sequential-gc") await call(root, sentinel, 1, 5_000, true, true, priorFDs, fdSentinel);
  else for (let index = 1; index <= 8; index++) await call(root, sentinel, index, 5_000, false, true, priorFDs, fdSentinel);
  console.log(JSON.stringify({ evidence: "native-recovery-passed", scenario, calls: scenario === "sequential-gc" ? 2 : 9 }));
} finally {
  closeSync(fdSentinel);
  await stopSentinel(sentinel);
  rmSync(root, { recursive: true, force: true });
  console.log(JSON.stringify({ evidence: "native-recovery-finally", scenario, sentinelPid: sentinel.pid,
    sentinelAbsent: !alive(sentinel.pid!), scratchRemoved: !existsSync(root) }));
  assert.equal(existsSync(root), false);
}
