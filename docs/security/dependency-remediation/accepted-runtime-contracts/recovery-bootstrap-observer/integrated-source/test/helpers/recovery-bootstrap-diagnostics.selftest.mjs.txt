// Explicit after-canonical Node-only controls; not Vitest's **/*.test.ts surface.
import assert from "node:assert/strict";
import { spawn, ChildProcess } from "node:child_process";
import { channel } from "node:diagnostics_channel";
import { errorMonitor } from "node:events";
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { beginRecoveryBootstrapDiagnostics, observeOwnedChild, readCheckoutHead } from "./recovery-bootstrap-diagnostics.mjs";

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const childEnv = () => Object.fromEntries([
  "PATH", "HOME", "HERMES_HOME", "HERMES_DIR", "HERMES_KNOWLEDGE_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
  "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TMPDIR", "TMP", "TEMP", "BUN_INSTALL_CACHE_DIR", "npm_config_cache",
  "CI", "REQUIRE_BUN", "QMD_LLAMA_GPU", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "UV_OFFLINE",
  "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "TOKENIZERS_PARALLELISM",
].filter(name => process.env[name] !== undefined).map(name => [name, process.env[name]]));
const launch = code => spawn(process.execPath, ["--input-type=module", "-e", code], {
  env: childEnv(), stdio: ["ignore", "pipe", "pipe", "pipe"],
});
async function until(predicate, label) {
  const deadline = performance.now() + 2_000;
  while (!predicate() && performance.now() < deadline) await delay(5);
  assert.ok(predicate(), label);
}
function eventPromise(emitter, name) {
  let listener;
  const promise = new Promise(resolve => { listener = (...args) => resolve(args); emitter.once(name, listener); });
  return { promise, dispose: () => emitter.removeListener(name, listener) };
}
async function release(child) {
  if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  await until(() => child.exitCode !== null || child.signalCode !== null, "owned fixture exit");
  for (const stream of child.stdio) stream?.destroy();
}
const drain = child => { for (const stream of child.stdio.slice(1)) stream.resume(); };
const ends = report => Object.fromEntries(["stdout", "stderr", "control"].map(name => [name,
  report.events.some(event => event.event === `${name}.end`)]));
const state = report => report.finalState;

// Fixture drains are intentional fixture behavior, never observer operations.
test("natural stream EOF and aggregate close preserve method identity/flow and remove observers", async () => {
  const child = launch("process.stdout.write('public'); process.stderr.write('fixture');");
  const streams = child.stdio.slice(1);
  const originals = { spawn: ChildProcess.prototype.spawn, kill: child.kill, emit: child.emit,
    streams: streams.map(stream => ({ read: stream.read, resume: stream.resume, destroy: stream.destroy,
      flowing: stream.readableFlowing, data: stream.listenerCount("data"), readable: stream.listenerCount("readable"),
      end: stream.listenerCount("end"), close: stream.listenerCount("close"), error: stream.listenerCount("error"), monitor: stream.listenerCount(errorMonitor) })) };
  const trace = observeOwnedChild(child);
  try {
    streams.forEach((stream, index) => {
      const before = originals.streams[index];
      assert.equal(stream.readableFlowing, before.flowing);
      assert.equal(stream.listenerCount("data"), before.data);
      assert.equal(stream.listenerCount("readable"), before.readable);
      assert.equal(stream.listenerCount("error"), before.error);
      assert.equal(stream.read, before.read); assert.equal(stream.resume, before.resume); assert.equal(stream.destroy, before.destroy);
    });
    drain(child);
    await until(() => trace.snapshot().events.some(event => event.event === "owner.close"), "natural aggregate close");
    const report = trace.finish();
    assert.deepEqual(ends(report), { stdout: true, stderr: true, control: true });
    assert.equal(state(report).closeCounter.needed, 4);
    assert.equal(state(report).closeCounter.got, 4);
    console.log(JSON.stringify({ evidence:"diagnostic-control-natural-close", ends:ends(report), closeCounter:state(report).closeCounter }));
    assert.equal(report.restoration.listenersRemoved, true);
    assert.equal(ChildProcess.prototype.spawn, originals.spawn); assert.equal(child.kill, originals.kill); assert.equal(child.emit, originals.emit);
    streams.forEach((stream, index) => assert.equal(stream.listenerCount(errorMonitor), originals.streams[index].monitor));
    assert.ok(!JSON.stringify(report).includes("public")); assert.ok(!JSON.stringify(report).includes("fixture"));
  } finally { trace.dispose(); await release(child); }
});

test("one independently owned writer retaining only FD3 is observed without attributing kernel cause", async () => {
  const root = mkdtempSync(path.join(tmpdir(), "bootstrap-known-writer-"));
  const releaseFile = path.join(root, "release");
  const readyFile = path.join(root, "ready");
  // The root owns the writer handle and passes its actual WRITE endpoint 3,
  // not this parent's opposite socket endpoint. The public synthetic writer
  // has a cooperative release plus a 2s self-exit bound; no PID signals.
  const writerCode = `import { existsSync, writeFileSync } from 'node:fs';
    writeFileSync(${JSON.stringify(readyFile)}, 'ready');
    setInterval(() => { if (existsSync(${JSON.stringify(releaseFile)})) process.exit(0); }, 5);
    setTimeout(() => process.exit(2), 2000);`;
  const child = launch(`import { spawn } from 'node:child_process'; import { existsSync } from 'node:fs';
    const writer = spawn(process.execPath, ['--input-type=module', '-e', ${JSON.stringify(writerCode)}],
      { env: process.env, stdio: ['ignore', 'ignore', 'ignore', 3] });
    writer.unref();
    const deadline = performance.now() + 1000;
    while (!existsSync(${JSON.stringify(readyFile)}) && performance.now() < deadline) await new Promise(resolve => setTimeout(resolve, 5));
    if (!existsSync(${JSON.stringify(readyFile)})) process.exit(3);
    process.exit(0);`);
  const trace = observeOwnedChild(child);
  try {
    drain(child);
    await until(() => child.exitCode !== null, "root reaped while holder remains owned");
    await until(() => ends(trace.snapshot()).stdout && ends(trace.snapshot()).stderr, "unretained output EOF");
    await delay(25);
    const held = trace.snapshot();
    assert.deepEqual(ends(held), { stdout: true, stderr: true, control: false });
    assert.equal(state(held).streams.control.readableEnded, false);
    assert.equal(state(held).closeCounter.needed, 4);
    assert.equal(state(held).closeCounter.got, 3);
    assert.equal(held.events.some(event => event.event === "owner.close"), false);
    assert.equal(held.causality, "not-established-by-runtime-events");
    console.log(JSON.stringify({ evidence:"diagnostic-control-known-fixture-FD3-writer", ends:ends(held), closeCounter:state(held).closeCounter, causality:held.causality }));
    writeFileSync(releaseFile, "release");
    await until(() => trace.snapshot().events.some(event => event.event === "owner.close"), "natural close after fixture releases known writer");
    assert.deepEqual(ends(trace.finish()), { stdout: true, stderr: true, control: true });
  } finally {
    writeFileSync(releaseFile, "release");
    try { await until(() => child.stdio[3].closed, "bounded fixture writer release"); }
    finally { trace.dispose(); await release(child); rmSync(root, { recursive:true, force:true }); }
  }
});

test("late reader observation is not diagnosed as a retained writer", async () => {
  const child = launch("process.stdout.write('short-buffer');");
  const trace = observeOwnedChild(child);
  const exit = eventPromise(child, "exit");
  try {
    await exit.promise; // Node itself schedules flushStdio only after emitting exit.
    const before = trace.snapshot();
    assert.equal(before.events.some(event => event.event === "owner.exit"), true);
    assert.equal(before.events.some(event => event.event === "owner.close"), false);
    // Deliberately delay the reader event loop. This says nothing about kernel holders.
    const deadline = performance.now() + 40;
    while (performance.now() < deadline) { /* bounded fixture-only reader scheduling delay */ }
    const delayed = trace.snapshot();
    assert.equal(delayed.events.some(event => event.event === "owner.close"), false);
    assert.equal(delayed.causality, "not-established-by-runtime-events");
    drain(child);
    await until(() => trace.snapshot().events.some(event => event.event === "owner.close"), "delayed reader then natural closure");
    const report = trace.finish();
    const tExit = report.events.find(event => event.event === "owner.exit").tMs;
    const tClose = report.events.find(event => event.event === "owner.close").tMs;
    assert.ok(tClose - tExit >= 40);
    assert.deepEqual(ends(report), { stdout: true, stderr: true, control: true });
    console.log(JSON.stringify({ evidence:"diagnostic-control-delayed-reader", closeDelayMs:tClose-tExit, ends:ends(report), causality:report.causality }));
  } finally { exit.dispose(); trace.dispose(); await release(child); }
});

test("forced destruction and pre-destroy receipt never become EOF or cleanup certification", async () => {
  const child = launch("setInterval(() => {}, 1000)");
  const trace = observeOwnedChild(child);
  try {
    const receipt = { launch: { state: "unconfirmed", admissionGranted: false, supervisorReaped: false },
      cleanup: { forcedPipeClose: true, pipesClosed: false, containmentVerified: false },
      transportFailure: { prepared: false, supervisorExited: false, supervisorCloseObserved: false,
        stdout: { bytes: 0, ended: false, closed: false, destroyed: false },
        stderr: { bytes: 0, ended: false, closed: false, destroyed: false },
        proof: { bytes: 0, ended: false, closed: false, destroyed: false } },
      stdout: "private-content-must-not-escape", error: "private-error", command: ["private-command"] };
    for (const stream of child.stdio.slice(1)) stream.destroy();
    await until(() => child.stdio.slice(1).every(stream => stream.closed), "destroyed fixture streams");
    const report = trace.finish(receipt);
    assert.deepEqual(ends(report), { stdout: false, stderr: false, control: false });
    assert.equal(state(report).streams.stdout.destroyed, true);
    assert.deepEqual(report.receipt.preForcedClosure.transportFailure, receipt.transportFailure);
    assert.equal(report.receipt.preForcedClosure.tMs, null);
    assert.equal(report.receipt.cleanup.pipesClosed, false);
    assert.equal(report.receipt.launch.state, "unconfirmed");
    for (const forbidden of ["private-content-must-not-escape", "private-error", "private-command"]) assert.ok(!JSON.stringify(report).includes(forbidden));
    assert.equal("cleanupCertified" in report, false);
    console.log(JSON.stringify({ evidence:"diagnostic-control-forced-destroy", ends:ends(report), preForcedClosure:report.receipt.preForcedClosure, cleanup:report.receipt.cleanup }));
  } finally { trace.dispose(); await release(child); }
});

test("bounded events, non-consuming errors and idempotent restoration", async () => {
  const child = launch("setInterval(() => {}, 1000)");
  const targets = [child, ...child.stdio.slice(1)];
  const relevant = ["spawn", "exit", "end", "close", "error", errorMonitor];
  const originalListeners = targets.map(target => relevant.map(event => target.listeners(event)));
  const trace = observeOwnedChild(child);
  try {
    for (let index = 0; index < 100; index++) child.stdout.emit(errorMonitor, Object.assign(new Error("private error details"), { code: "EIO" }));
    const report = trace.finish();
    assert.equal(report.events.length, 32);
    assert.ok(report.droppedEvents > 0);
    assert.ok(JSON.stringify(report).length < 100_000);
    assert.ok(!JSON.stringify(report).includes("private error details"));
    trace.dispose(); trace.dispose();
    targets.forEach((target, index) => relevant.forEach((event, eventIndex) => assert.deepEqual(target.listeners(event), originalListeners[index][eventIndex])));
    assert.equal(child.stdout.listenerCount(errorMonitor), 0);
    assert.equal(child.stdout.listenerCount("error"), 0);
    assert.throws(() => child.stdout.emit("error", new Error("still unhandled")), /still unhandled/);
  } finally { trace.dispose(); await release(child); }
});

test("actual checkout reader handles authored detached, loose, packed and malformed refs without a git process", () => {
  const root = mkdtempSync(path.join(tmpdir(), "bootstrap-head-fixture-"));
  const sha = "0123456789abcdef0123456789abcdef01234567"; // synthetic, not a candidate SHA
  try {
    assert.equal(readCheckoutHead(root), null);
    mkdirSync(path.join(root, ".git", "refs", "heads"), { recursive:true });
    writeFileSync(path.join(root, ".git", "HEAD"), sha + "\n");
    assert.equal(readCheckoutHead(root), sha);
    writeFileSync(path.join(root, ".git", "HEAD"), "ref: refs/heads/fixture\n");
    writeFileSync(path.join(root, ".git", "refs", "heads", "fixture"), sha + "\n");
    assert.equal(readCheckoutHead(root), sha);
    rmSync(path.join(root, ".git", "refs", "heads", "fixture"));
    writeFileSync(path.join(root, ".git", "packed-refs"), sha + " refs/heads/fixture\n");
    assert.equal(readCheckoutHead(root), sha);
    writeFileSync(path.join(root, ".git", "HEAD"), "ref: refs/../../private\n");
    assert.equal(readCheckoutHead(root), null);
    writeFileSync(path.join(root, ".git", "HEAD"), "malformed literal SHA\n");
    assert.equal(readCheckoutHead(root), null);
  } finally { rmSync(root, { recursive:true, force:true }); }
});

test("disabled mode is a no-op; unmatched constructor subscription is restored even on exceptions", () => {
  const root = mkdtempSync(path.join(tmpdir(), "bootstrap-observer-off-"));
  const dc = channel("child_process");
  const before = dc.hasSubscribers;
  try {
    const disabled = beginRecoveryBootstrapDiagnostics({ enabled: false, directory: root });
    assert.equal(dc.hasSubscribers, before);
    disabled.attachAfterLaunch(); disabled.finish(); disabled.dispose();
    assert.deepEqual(readdirSync(root), []);
    const active = beginRecoveryBootstrapDiagnostics({ enabled: true, directory: root });
    try { throw new Error("authored caller failure"); }
    catch { active.finish(); }
    assert.equal(dc.hasSubscribers, before);
    const files = readdirSync(root);
    assert.equal(files.length, 1);
    const artifact = JSON.parse(readFileSync(path.join(root, files[0]), "utf8"));
    assert.equal(artifact.status, "no-matching-node-supervisor-observed");
    assert.equal(artifact.restoration.channelUnsubscribed, true);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("constructor capture matches only the actual fixed supervisor, without reading command/config", async () => {
  const root = mkdtempSync(path.join(tmpdir(), "bootstrap-observer-capture-"));
  const observer = beginRecoveryBootstrapDiagnostics({ enabled: true, directory: root });
  const child = launch("setInterval(() => {}, 1000)");
  try {
    observer.attachAfterLaunch();
    observer.finish({ launch: { state: "unconfirmed" }, stdout: "must-not-log" });
    const report = JSON.parse(readFileSync(path.join(root, readdirSync(root)[0]), "utf8"));
    assert.equal(report.status, "no-matching-node-supervisor-observed");
    assert.equal(report.restoration.channelUnsubscribed, true);
    assert.ok(!JSON.stringify(report).includes("must-not-log"));
    assert.equal(child.listenerCount(errorMonitor), 0);
  } finally { observer.dispose(); await release(child); rmSync(root, { recursive: true, force: true }); }
});
