import { describe, expect, it } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeRecoveryWorker, type RecoveryProcessResult } from "../scripts/phase5_e2e_runner.js";

function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") return false; throw error; }
}

async function waitAbsent(pid: number): Promise<void> {
  const deadline = performance.now() + 1_000;
  while (alive(pid) && performance.now() < deadline) await new Promise(resolve => setTimeout(resolve, 10));
  expect(alive(pid), `Fixture PID ${pid} must be absent`).toBe(false);
}

async function stopSentinel(child: ChildProcess): Promise<void> {
  if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  if (child.exitCode === null && child.signalCode === null) await new Promise<void>(resolve => child.once("exit", () => resolve()));
  await waitAbsent(child.pid!);
}

// A real helper invocation with a real TERM-ignoring descendant. The independent
// guard is failure-only; pre-finally assertions cannot pass by using its cleanup.
async function treeCase(mode: "timeout" | "leader-exit" | "real-worker" | "escaped-pipe") {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-tree-"));
  const pidFile = path.join(root, "owned-pids.json");
  const outputJson = path.join(root, "worker-report.json");
  const dbPath = path.join(root, "shadow-unused.sqlite");
  const sentinel = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { detached: true, stdio: "ignore" });
  let server: http.Server | undefined;
  let helper: Promise<RecoveryProcessResult> | undefined;
  let guard: ReturnType<typeof setTimeout> | undefined;
  let pids: { parent: number; grandchild: number } | undefined;
  let guardFired = false;
  let finallyKilled: number[] = [];
  let workerRequests = 0;
  try {
    // IPC readiness proves the grandchild's TERM handler exists before the
    // leader exits or the helper's execution deadline starts teardown.
    const grandchildCode = "process.on('SIGTERM', () => {}); process.send('ready'); setInterval(() => {}, 1000);";
    const prelude = `
      const grandchild = spawn(process.execPath, ['-e', ${JSON.stringify(grandchildCode)}], {
        detached: ${mode === "escaped-pipe"}, stdio: ['ignore', 1, 2, 'ipc']
      });
      await new Promise(resolve => grandchild.once('message', resolve));
      grandchild.disconnect();
      grandchild.unref();
      writeFileSync(${JSON.stringify(pidFile)}, JSON.stringify({ parent: process.pid, grandchild: grandchild.pid }));
      process.on('SIGTERM', () => {});
    `;
    let command: string[];
    let port = 8797;
    if (mode === "real-worker") {
      const descriptor = { version: 1, backend: "mlx", model: "synthetic-recovery-stall",
        pooling: "mean", nativeDimensions: 4, outputDimensions: 4, maxTokens: 2048, normalized: true };
      writeFileSync(path.join(root, "owned-fixture.md"), "# Authored fixture\nRecovery must terminate a stalled embedding request.\n");
      server = http.createServer(async (req, res) => {
        // Metadata has its own short public deadline. Succeed through that
        // prerequisite, then stall actual inference so this test proves work
        // started before the supervisor's independent execution deadline.
        res.setHeader("Content-Type", "application/json");
        if (req.url === "/descriptor") res.end(JSON.stringify(descriptor));
        else if (req.url === "/health") res.end(JSON.stringify({ status: "ok", ready: true, descriptor }));
        else if (req.url === "/embed" || req.url === "/embed-bin") {
          workerRequests++;
          req.resume(); // Deliberately leave this owned, non-model response open.
        } else if (req.url === "/tokenize") {
          const chunks: Buffer[] = [];
          for await (const chunk of req) chunks.push(Buffer.from(chunk));
          const { texts } = JSON.parse(Buffer.concat(chunks).toString()) as { texts: string[] };
          res.end(JSON.stringify({ counts: texts.map(text => Math.ceil(text.length / 4)) }));
        } else { res.statusCode = 404; res.end(JSON.stringify({ error: "Unexpected fixture route" })); }
      });
      await new Promise<void>(resolve => server!.listen(0, "127.0.0.1", resolve));
      port = (server.address() as { port: number }).port;
      expect(port).not.toBe(8787);
      const preload = path.join(root, "spawn-descendant.mjs");
      writeFileSync(preload, `import { spawn } from 'node:child_process'; import { writeFileSync } from 'node:fs'; ${prelude}`);
      command = [process.execPath, "--import", preload,
        ...((process.versions as Record<string, string>).bun ? [] : ["--import",
          fileURLToPath(new URL("../node_modules/tsx/dist/loader.mjs", import.meta.url))]),
        fileURLToPath(new URL("../scripts/phase5_recovery_worker.ts", import.meta.url)),
        "--launch-token", "owned-tree-real-worker", "--db", dbPath, "--docs-dir", root,
        "--mlx-model-path", descriptor.model,
        "--mlx-port", String(port), "--output-json", outputJson, "--execution-timeout-ms", "10000"];
    } else {
      const report = { status: "completed", dbPath, action: "resume", docsCompleted: 1, chunksCommitted: 1,
        totalVectors: 1, checkpointStatus: null, checkpointJobId: "authored-command-fixture",
        consecutiveSequenceVerified: true, hashes: [{ hash: "synthetic", seqs: [0] }] };
      const ending = mode === "timeout" ? "setInterval(() => {}, 1000);" : `
        writeFileSync(${JSON.stringify(outputJson)}, JSON.stringify({ ...${JSON.stringify(report)}, pid: process.pid }));
        process.exit(0);`;
      command = [process.execPath, "--input-type=module", "-e",
        `import { spawn } from 'node:child_process'; import { writeFileSync } from 'node:fs'; ${prelude} ${ending}`];
    }
    const started = performance.now();
    // A real TypeScript worker has cold-import/native-addon startup costs that
    // vary across CI hosts. Give that fixture an explicit execution budget,
    // while keeping the helper's post-deadline cleanup allowance unchanged.
    const timeoutMs = mode === "real-worker" ? 5_000 : 200;
    const elapsedBoundMs = mode === "real-worker" ? timeoutMs + 500 : 1_500;
    helper = executeRecoveryWorker({ dbPath, docsDir: root, mlxPort: port, launchToken: "owned-tree",
      action: mode === "real-worker" ? "index" : "resume", outputJson, stateRoot: path.join(root, "state"), timeoutMs, command });
    const result = await Promise.race([helper, new Promise<never>((_resolve, reject) => {
      guard = setTimeout(() => { guardFired = true; reject(new Error("Helper exceeded its own timeout + teardown budget")); }, elapsedBoundMs + 300);
    })]);
    clearTimeout(guard);
    const elapsedMs = performance.now() - started;
    expect(existsSync(pidFile), result.stderr).toBe(true);
    pids = JSON.parse(readFileSync(pidFile, "utf8"));
    expect(result.pid).toBe(pids!.parent);
    expect(guardFired).toBe(false);
    expect(elapsedMs).toBeLessThan(elapsedBoundMs);
    // These observations are made BEFORE any fixture/outer-guard cleanup.
    await waitAbsent(pids!.parent);
    if (mode !== "escaped-pipe") await waitAbsent(pids!.grandchild);
    expect(alive(sentinel.pid!)).toBe(true);
    if (mode === "real-worker") {
      expect(workerRequests, result.stderr).toBeGreaterThan(0);
      expect(existsSync(dbPath)).toBe(true);
    }
    expect(result.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, pipesClosed: true });
    if (mode === "timeout" || mode === "real-worker") {
      expect(result.timedOut).toBe(true);
      expect(result.error).toMatch(/timed out/);
      expect(result.cleanup).toMatchObject({ sigkillUsed: true, forcedPipeClose: false });
    } else if (mode === "leader-exit") {
      expect(result.timedOut).toBe(false);
      expect(result.exitCode).toBe(0);
      // Command injection has no OS containment contract, even though the
      // owned-session cleanup itself is proved independently of its report.
      expect(result.error).toMatch(/command injection is uncontained/);
      expect(result.cleanup).toMatchObject({ containmentVerified: false, sigkillUsed: true, forcedPipeClose: false });
    } else {
      // A process that deliberately creates another session is not in our
      // owned group. Refuse success and bound pipe drain, never signal an
      // arbitrary other group inferred from logs or report data.
      expect(result.error).toMatch(/pipe/i);
      expect(result.cleanup).toMatchObject({ forcedPipeClose: true });
      expect(alive(pids!.grandchild)).toBe(true);
    }
    console.log(JSON.stringify({ evidence: "recovery-tree-before-finally", mode, timeoutMs, elapsedMs: Math.round(elapsedMs), workerRequests,
      helper: result.pid, grandchild: pids!.grandchild, parentAbsent: !alive(pids!.parent),
      grandchildAbsent: !alive(pids!.grandchild), sentinelAlive: alive(sentinel.pid!), guardFired,
      error: result.error ?? null, cleanup: result.cleanup }));
  } finally {
    if (guard) clearTimeout(guard);
    if (!pids && existsSync(pidFile)) pids = JSON.parse(readFileSync(pidFile, "utf8"));
    // Failure-only safety net, plus the explicitly escaped fixture's cleanup.
    for (const pid of pids ? [pids.parent, pids.grandchild] : []) {
      if (alive(pid)) { finallyKilled.push(pid); process.kill(pid, "SIGKILL"); }
    }
    if (helper) await helper;
    for (const pid of pids ? [pids.parent, pids.grandchild] : []) await waitAbsent(pid);
    await stopSentinel(sentinel);
    if (server) {
      server.closeAllConnections();
      await new Promise<void>(resolve => server!.close(() => resolve()));
    }
    rmSync(root, { recursive: true, force: true });
    expect(existsSync(root)).toBe(false);
    console.log(JSON.stringify({ evidence: "recovery-tree-finally", mode, guardFired, finallyKilled,
      allFixturePidsAbsent: true, scratchRemoved: !existsSync(root) }));
    if (mode !== "escaped-pipe") expect(finallyKilled).toEqual([]);
  }
}

async function closedPipeEscapeCase(mode: "new-pgid" | "new-session") {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-closed-pipes-"));
  const pidFile = path.join(root, "owned-pids.json");
  const outputJson = path.join(root, "worker-report.json");
  const dbPath = path.join(root, "shadow-unused.sqlite");
  const sentinel = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { detached: true, stdio: "ignore" });
  let pids: { parent: number; grandchild: number; parentSID: number; childSID: number; childPGID: number } | undefined;
  let guard: ReturnType<typeof setTimeout> | undefined;
  let guardFired = false;
  let helper: Promise<RecoveryProcessResult> | undefined;
  const finallyKilled: number[] = [];
  try {
    const grandchildCode = `
import json, os, signal, sys, time
${mode === "new-session" ? "os.setsid()" : "os.setpgid(0, 0)"}
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1] + '.tmp', 'w') as record:
    json.dump(dict(parent=int(sys.argv[2]), grandchild=os.getpid(), parentSID=int(sys.argv[3]), childSID=os.getsid(0), childPGID=os.getpgrp()), record)
os.replace(sys.argv[1] + '.tmp', sys.argv[1])
time.sleep(60)
`;
    const leaderCode = `
import json, os, subprocess, sys, time
subprocess.Popen([sys.executable, '-c', ${JSON.stringify(grandchildCode)}, ${JSON.stringify(pidFile)}, str(os.getpid()), str(os.getsid(0))], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
deadline = time.monotonic() + 2
while not os.path.exists(${JSON.stringify(pidFile)}) and time.monotonic() < deadline:
    time.sleep(0.002)
assert os.path.exists(${JSON.stringify(pidFile)})
with open(${JSON.stringify(outputJson)}, 'w') as report:
    json.dump(dict(status='completed', pid=os.getpid(), dbPath=${JSON.stringify(dbPath)}, action='resume', docsCompleted=1, chunksCommitted=1, totalVectors=1, checkpointStatus=None, consecutiveSequenceVerified=True, hashes=[]), report)
`;
    const started = performance.now();
    helper = executeRecoveryWorker({ dbPath, docsDir: root, action: "resume", outputJson,
      stateRoot: path.join(root, "state"), timeoutMs: 1_000, command: ["/usr/bin/python3", "-c", leaderCode] });
    const result = await Promise.race([helper, new Promise<never>((_resolve, reject) => {
      guard = setTimeout(() => { guardFired = true; reject(new Error("Closed-pipe helper exceeded bounded teardown")); }, 1_800);
    })]);
    clearTimeout(guard);
    const elapsedMs = performance.now() - started;
    pids = JSON.parse(readFileSync(pidFile, "utf8"));
    expect(result.pid).toBe(pids!.parent);
    expect(result.cleanup.sessionId).toBe(pids!.parentSID);
    expect(pids!.childPGID).toBe(pids!.grandchild);
    expect(pids!.childPGID).not.toBe(pids!.parent);
    expect(pids!.childSID === pids!.parentSID).toBe(mode === "new-pgid");
    expect(result.exitCode, result.stderr).toBe(0);
    expect(result.signal).toBeNull();
    expect(result.timedOut).toBe(false);
    expect(result.error).toMatch(/command injection is uncontained.*escape.*cannot be certified/);
    expect(result.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, sessionTerminated: true,
      pipesClosed: true, forcedPipeClose: false, containmentVerified: false });
    expect(elapsedMs).toBeLessThan(1_500);
    expect(guardFired).toBe(false);
    // These are the real recorded processes, checked BEFORE fixture cleanup.
    await waitAbsent(pids!.parent);
    if (mode === "new-pgid") {
      await waitAbsent(pids!.grandchild);
      expect(result.cleanup.processGroups).toContain(pids!.childPGID);
      expect(result.cleanup.sigkillUsed).toBe(true);
    } else {
      expect(alive(pids!.grandchild)).toBe(true);
      expect(result.cleanup.processGroups).not.toContain(pids!.childPGID);
    }
    expect(alive(sentinel.pid!)).toBe(true);
    console.log(JSON.stringify({ evidence: "recovery-closed-pipes-before-finally", mode, ...pids,
      elapsedMs: Math.round(elapsedMs), parentAbsent: !alive(pids!.parent), grandchildAbsent: !alive(pids!.grandchild),
      sentinelAlive: alive(sentinel.pid!), guardFired, error: result.error, cleanup: result.cleanup }));
  } finally {
    if (guard) clearTimeout(guard);
    if (!pids && existsSync(pidFile)) pids = JSON.parse(readFileSync(pidFile, "utf8"));
    // Only fixture-recorded PIDs, never global pkill/PGID heuristics.
    for (const pid of pids ? [pids.parent, pids.grandchild] : []) {
      if (alive(pid)) { finallyKilled.push(pid); process.kill(pid, "SIGKILL"); }
    }
    if (helper) await helper;
    for (const pid of pids ? [pids.parent, pids.grandchild] : []) await waitAbsent(pid);
    await stopSentinel(sentinel);
    rmSync(root, { recursive: true, force: true });
    expect(existsSync(root)).toBe(false);
    if (mode === "new-pgid") expect(finallyKilled).toEqual([]);
    console.log(JSON.stringify({ evidence: "recovery-closed-pipes-finally", mode, guardFired, finallyKilled, scratchRemoved: !existsSync(root) }));
  }
}

// Exercise the same OS boundary used by the canonical entrypoint, not a mocked
// child_process API. If containment regresses, only recorded probe PIDs are killed.
async function noForkCase() {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-recovery-no-fork-"));
  const proofPath = path.join(root, "proof.json");
  const escapedPath = path.join(root, "unexpected-child.json");
  // Python exec below evaluates only this fixed fixture source, never a shell
  // command or user input. Append so a regression records EVERY created PID.
  const childCode = `import json, os, time; open(${JSON.stringify(escapedPath)}, 'a').write(json.dumps(os.getpid()) + chr(10)); time.sleep(60)`;
  const code = `
import errno, json, os, subprocess, sys
proof = {}
try:
    pid = os.fork()
    if pid == 0:
        exec(${JSON.stringify(childCode)})
        os._exit(0)
    proof['forkDenied'] = False
except OSError as failure:
    proof['forkDenied'] = failure.errno == errno.EPERM
try:
    subprocess.Popen([sys.executable, '-c', ${JSON.stringify(childCode)}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    proof['spawnDenied'] = False
except OSError as failure:
    proof['spawnDenied'] = failure.errno == errno.EPERM
with open(${JSON.stringify(proofPath)}, 'w') as result:
    json.dump(proof, result)
`;
  const supervisor = spawn("/usr/bin/python3", ["-S", "-u",
    fileURLToPath(new URL("../scripts/phase5_recovery_process.py", import.meta.url)),
    JSON.stringify({ command: ["/usr/bin/python3", "-c", code], timeoutMs: 500, contained: true })],
    { detached: true, stdio: ["pipe", "pipe", "pipe", "pipe"] });
  let protocol = "";
  let stderr = "";
  let guard: ReturnType<typeof setTimeout> | undefined;
  (supervisor.stdio[3] as import("node:stream").Readable).on("data", chunk => { protocol += chunk.toString(); });
  supervisor.stderr!.on("data", chunk => { stderr += chunk.toString(); });
  try {
    const exit = await Promise.race([new Promise<number | null>(resolve => supervisor.once("close", resolve)),
      new Promise<never>((_resolve, reject) => { guard = setTimeout(() => {
        supervisor.stdin!.write("K"); reject(new Error("No-fork probe exceeded its bounded supervisor"));
      }, 1_500); })]);
    clearTimeout(guard);
    expect(exit, stderr).toBe(0);
    expect(JSON.parse(readFileSync(proofPath, "utf8"))).toEqual({ forkDenied: true, spawnDenied: true });
    const messages = protocol.trim().split("\n").map(line => JSON.parse(line));
    const ready = messages.find(message => message.type === "ready");
    const cleanup = messages.find(message => message.type === "cleanup");
    expect(cleanup).toMatchObject({ childReaped: true, groupTerminated: true,
      sessionTerminated: true, containmentVerified: true, errors: [] });
    await waitAbsent(ready.pid);
    expect(existsSync(escapedPath)).toBe(false);
    console.log(JSON.stringify({ evidence: "recovery-kernel-no-fork", pid: ready.pid,
      platform: process.platform, forkDenied: true, spawnDenied: true, cleanup }));
  } finally {
    if (guard) clearTimeout(guard);
    supervisor.stdin!.end("K");
    if (existsSync(escapedPath)) {
      const pids = readFileSync(escapedPath, "utf8").trim().split("\n").map(line => JSON.parse(line) as number);
      for (const pid of pids) {
        if (alive(pid)) process.kill(pid, "SIGKILL");
        await waitAbsent(pid);
      }
    }
    rmSync(root, { recursive: true, force: true });
  }
}

describe.skipIf(process.platform === "win32")("recovery helper owns its POSIX worker group", () => {
  it("bounds a 200ms timeout across TERM-ignoring inherited-pipe descendants", async () => treeCase("timeout"));
  it("cleans the group even after its successful leader exits", async () => treeCase("leader-exit"));
  it("bounds the actual recovery-worker entrypoint with a preloaded inherited-pipe descendant", async () => treeCase("real-worker"));
  it("fails closed with bounded pipe fallback for a deliberately escaped session", async () => treeCase("escaped-pipe"));
  it("cleans a recorded new PGID in the owned session even with stdio DEVNULL", async () => closedPipeEscapeCase("new-pgid"));
  it("explicitly refuses arbitrary new-session escape even with stdio DEVNULL", async () => closedPipeEscapeCase("new-session"));
  it("enforces the canonical no-fork boundary in the real OS", noForkCase);
});
