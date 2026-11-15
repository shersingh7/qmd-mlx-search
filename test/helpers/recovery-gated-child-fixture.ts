import childProcess, { type ChildProcess } from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { executeRecoveryWorker } from "../../scripts/phase5_e2e_runner.js";

type Helper = ReturnType<typeof executeRecoveryWorker>;
type Timer = ReturnType<typeof setTimeout>;

// Test-only, non-registering ownership scope shared by the real API case and
// its failure controls. These seams inject failures/observe acquisitions; they
// do not replace the API or manufacture a product cleanup acknowledgement.
export interface GatedChildFixtureOperations {
  makeRoot(): string;
  spawnSentinel(): ChildProcess;
  syncExports(): void;
  clearGuard(timer: Timer): void;
  clearSentinelTimer(timer: Timer): void;
  removeRoot(root: string): void;
}
export interface GatedChildFixture {
  root: string;
  kernel: string;
  work: string;
  sentinel: ChildProcess;
  hooks: number[];
  readonly supervisorPid: number | undefined;
  readonly guardFired: boolean;
  waitForHelper(helper: Helper): Helper;
}

// Retain the actual ChildProcess through signal AND exit/error observation.
// No delayed numeric PID, global listener, forced GC, or unobserved rejection.
function observeSentinel(child: ChildProcess, clearTimer: (timer: Timer) => void) {
  let done = false;
  let failurePresent = false;
  let failure: unknown;
  let operationFailure: { error: unknown } | undefined;
  const failures: { release: string; error: unknown }[] = [];
  const attempt = (release: string, action: () => void) => {
    try { action(); }
    catch (error) { failures.push({ release, error }); }
  };
  let resolve!: () => void;
  const terminal = new Promise<void>(settled => { resolve = settled; });
  const onExit = () => { done = true; resolve(); };
  const onError = (error: Error) => { failurePresent = true; failure = error; done = true; resolve(); };
  // Return custody before fallible observer setup. A call-through attachment
  // report cannot prevent the caller retaining this exact handle's disposer.
  return {
    failures,
    get operationFailure() { return operationFailure; },
    install() {
      const firstFailure = failures.length;
      attempt("sentinel-exit-observer", () => { child.once("exit", onExit); });
      attempt("sentinel-error-observer", () => { child.once("error", onError); });
      if (child.exitCode !== null || child.signalCode !== null) onExit();
      if (failures.length > firstFailure) throw failures[firstFailure]!.error;
    },
    async release() {
      const firstFailure = failures.length;
      let timer: Timer | undefined;
      try {
        if (!done && child.pid !== undefined && !child.kill("SIGKILL")) {
          throw new Error("Owned sentinel signal was not delivered");
        }
        await Promise.race([terminal, new Promise<never>((_resolve, reject) => {
          timer = setTimeout(() => reject(new Error("Owned sentinel did not exit within 2000ms")), 2_000);
        })]);
        if (failurePresent) throw failure;
      } catch (error) {
        operationFailure = { error }; // Presence is separate from throw truthiness.
        failures.push({ release: "sentinel", error });
      } finally {
        // A failed release/report never skips the other independent releases.
        if (timer !== undefined) attempt("sentinel-timer", () => clearTimer(timer!));
        attempt("sentinel-exit-listener", () => { child.removeListener("exit", onExit); });
        attempt("sentinel-error-listener", () => { child.removeListener("error", onError); });
      }
      // Setup failures are already authoritative at the outer body boundary.
      // Retain every secondary in owned state; never attach it to the primary.
      if (failures.length > firstFailure) throw failures[firstFailure]!.error;
    },
  };
}

export async function withGatedChildFixture<T>(
  source: string,
  fixture: string,
  body: (owned: GatedChildFixture) => Promise<T>,
  operations: Partial<GatedChildFixtureOperations> = {},
): Promise<T> {
  const ops: GatedChildFixtureOperations = {
    makeRoot: () => mkdtempSync(path.join(tmpdir(), "qmd-gated-api-")),
    spawnSentinel: () => childProcess.spawn("/usr/bin/python3", ["-S", "-c", "import time; time.sleep(60)"], { detached: true, stdio: "ignore" }),
    syncExports: syncBuiltinESMExports,
    clearGuard: clearTimeout,
    clearSentinelTimer: clearTimeout,
    removeRoot: root => rmSync(root, { recursive: true, force: true }),
    ...operations,
  };
  let root: string | undefined;
  let sentinelOwner: ReturnType<typeof observeSentinel> | undefined;
  let restore: (() => void) | undefined;
  let helper: Helper | undefined;
  let guard: Timer | undefined;
  let guardFired = false;
  let supervisorPid: number | undefined;
  let failed = false;
  let primary: unknown;
  let value!: T;
  const failures: { release: string; error: unknown }[] = [];
  const stopGuard = () => {
    if (guard !== undefined) {
      const timer = guard;
      guard = undefined;
      ops.clearGuard(timer);
    }
  };
  try {
    // The cleanup scope starts BEFORE the first acquisition or interception.
    root = ops.makeRoot();
    const ownedRoot = root;
    const sentinel = ops.spawnSentinel();
    sentinelOwner = observeSentinel(sentinel, ops.clearSentinelTimer);
    sentinelOwner.install();
    if ((process.versions as Record<string, string>).bun) {
      const bun = (globalThis as any).Bun;
      const original = bun.spawn;
      restore = () => { bun.spawn = original; ops.syncExports(); };
      bun.spawn = function(command: string[], options: unknown) {
        if (command[3] === source) {
          const child = original.call(bun, [...command.slice(0, 3), fixture, "--supervisor", source, "live-miss", ownedRoot, "3", ...command.slice(4)], options);
          supervisorPid = child.pid;
          return child;
        }
        return original.call(bun, command, options);
      };
    } else {
      const original = childProcess.spawn;
      // Register restoration BEFORE patching and the fallible synchronization.
      restore = () => { childProcess.spawn = original; ops.syncExports(); };
      (childProcess as any).spawn = function(executable: string, args: string[], options: unknown) {
        if (args[2] === source) {
          const child = original(executable, [...args.slice(0, 2), fixture, "--supervisor", source, "live-miss", ownedRoot, "3", ...args.slice(3)], options as any);
          supervisorPid = child.pid;
          return child;
        }
        return original(executable, args, options as any);
      };
    }
    ops.syncExports();
    value = await body({
      root, kernel: path.join(root, "kernel.jsonl"), work: path.join(root, "unexpected-work"), sentinel,
      hooks: ["exit", "SIGTERM", "SIGINT"].map(event => process.listenerCount(event)),
      get supervisorPid() { return supervisorPid; },
      get guardFired() { return guardFired; },
      waitForHelper(started: Helper): Helper {
        if (helper) throw new Error("Gated fixture accepts exactly one helper");
        helper = started;
        return Promise.race([started, new Promise<never>((_resolve, reject) => {
          guard = setTimeout(() => { guardFired = true; reject(new Error("Gated acquisition exceeded 1800ms guard")); }, 1_800);
        })]).then(result => { stopGuard(); return result; });
      },
    });
  } catch (error) {
    failed = true;
    primary = error; // A separate flag preserves even throw undefined/null/0.
  } finally {
    const attempt = async (release: string, action: () => unknown) => {
      try { await action(); }
      catch (error) { failures.push({ release, error }); }
    };
    if (restore) await attempt("interception", restore);
    await attempt("guard", stopGuard);
    if (helper) await attempt("helper", async () => {
      try { await helper; }
      catch (error) { if (!failed || error !== primary) throw error; }
    });
    // A rejected helper or release never skips another acquired resource.
    if (sentinelOwner) {
      const earlierFailures = failures.length;
      await attempt("sentinel", sentinelOwner.release);
      // An operation failure inside this disposer is primary to its own
      // release/report faults, not a replacement for an earlier outer error.
      if (!failed && earlierFailures === 0 && sentinelOwner.operationFailure) {
        failed = true; primary = sentinelOwner.operationFailure.error;
      }
    }
    if (root !== undefined) await attempt("scratch", () => ops.removeRoot(root!));
    await attempt("receipt", () => console.log(JSON.stringify({ evidence: "gated-child-api-finally",
      fixtureWorkerSignals: [], scratchRemoved: root === undefined || !existsSync(root),
      cleanupFailures: failures.map(item => item.release),
      sentinelFailures: sentinelOwner?.failures.map(item => item.release) ?? [] })));
    if (failures.length || sentinelOwner?.failures.length) await attempt("error-report", () => console.error("Gated fixture cleanup failures (not deallocation proof)", failures, sentinelOwner?.failures ?? []));
  }
  if (failed) throw primary;
  if (failures.length === 1) throw failures[0]!.error;
  if (failures.length) throw new AggregateError(failures.map(item => item.error), "Gated fixture cleanup failed");
  return value;
}
