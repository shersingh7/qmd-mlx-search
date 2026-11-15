const testRuntime = (process.versions as Record<string, string>).bun ? "bun:test" : "vitest";
const { describe, expect, it }: typeof import("vitest") = await import(testRuntime);
import childProcess, { type ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import { syncBuiltinESMExports } from "node:module";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeRecoveryWorker } from "../scripts/phase5_e2e_runner.js";
import { withGatedChildFixture, type GatedChildFixtureOperations } from "./helpers/recovery-gated-child-fixture.js";

const source = fileURLToPath(new URL("../scripts/phase5_recovery_process.py", import.meta.url));
const fixture = fileURLToPath(new URL("./python/test_recovery_gated_child_ownership.py", import.meta.url));
const hookCounts = () => ["exit", "SIGTERM", "SIGINT"].map(event => process.listenerCount(event));
async function caught(action: () => Promise<unknown>) {
  try { await action(); return { threw: false, value: undefined as unknown }; }
  catch (value) { return { threw: true, value }; }
}

// Independent custody exists even on a failing assertion. All observations and
// required assertions precede rescue. Rescue is never counted as fixture success.
function custody(rescueOperations: Partial<{
  syncExports(): void;
  clearTimer(timer: ReturnType<typeof setTimeout>): void;
  removeRoot(root: string): void;
}> = {}) {
  const spawn = childProcess.spawn;
  const bun = (globalThis as any).Bun;
  const bunSpawn = bun?.spawn;
  const hooks = hookCounts();
  const roots: string[] = [];
  const children: ChildProcess[] = [];
  const kills = new Map<ChildProcess, ChildProcess["kill"]>();
  const listeners = new Map<ChildProcess, number[]>();
  const rescueFailures: { release: string; error: unknown }[] = [];
  const rescueOps = { syncExports: syncBuiltinESMExports, clearTimer: clearTimeout,
    removeRoot: (root: string) => rmSync(root, { recursive: true, force: true }), ...rescueOperations };
  const operations: Partial<GatedChildFixtureOperations> = {
    makeRoot: () => { const root = mkdtempSync(path.join(tmpdir(), "qmd-gated-cleanup-")); roots.push(root); return root; },
    spawnSentinel: () => {
      const child = spawn("/usr/bin/python3", ["-S", "-c", "import time; time.sleep(60)"], { detached: true, stdio: "ignore" });
      children.push(child); kills.set(child, child.kill);
      listeners.set(child, [child.listenerCount("exit"), child.listenerCount("error")]);
      return child;
    },
  };
  const observe = (label: string) => {
    const state = { evidence: "gated-fixture-before-independent-rescue", label,
      host: { executable: process.execPath, node: process.versions.node, bun: process.versions.bun ?? null },
      acquiredSentinels: children.length,
      liveSentinels: children.filter(child => child.exitCode === null && child.signalCode === null).length,
      remainingRoots: roots.filter(root => existsSync(root)),
      bindingsRestored: childProcess.spawn === spawn && (!bun || bun.spawn === bunSpawn),
      hooksRestored: JSON.stringify(hookCounts()) === JSON.stringify(hooks),
      sentinelListenersRestored: children.every(child => JSON.stringify([child.listenerCount("exit"), child.listenerCount("error")]) === JSON.stringify(listeners.get(child))) };
    console.log(JSON.stringify(state));
    return state;
  };
  const rescue = async () => {
    const firstFailure = rescueFailures.length;
    const attempt = async (release: string, action: () => unknown) => {
      try { await action(); }
      catch (error) { rescueFailures.push({ release, error }); }
    };
    await attempt("node-spawn", () => { childProcess.spawn = spawn; });
    if (bun) await attempt("bun-spawn", () => { bun.spawn = bunSpawn; });
    await attempt("synchronization", rescueOps.syncExports);
    const rescued: number[] = [];
    for (const child of children) {
      await attempt("sentinel-kill-binding", () => { child.kill = kills.get(child)!; });
      await attempt("sentinel", async () => {
        if (child.exitCode !== null || child.signalCode !== null) return;
        // Real owned handle, never a historical PID or process-list identity.
        let timer: ReturnType<typeof setTimeout> | undefined;
        let onExit: (() => void) | undefined;
        let onError: ((error: Error) => void) | undefined;
        let timedOut = false, failed = false;
        let failure: unknown;
        try {
          // Settle rather than reject: a synchronous signal refusal must not
          // leave an unobserved rejecting observer behind.
          let settle!: () => void;
          const done = new Promise<void>(resolve => { settle = resolve; });
          onExit = settle;
          onError = error => { failed = true; failure = error; settle(); };
          child.once("exit", onExit); child.once("error", onError);
          timer = setTimeout(() => { timedOut = true; settle(); }, 2_000);
          if (!kills.get(child)!.call(child, "SIGKILL")) throw new Error("Independent sentinel signal refused");
          await done;
          if (failed) throw failure;
          if (timedOut) throw new Error("Independent sentinel rescue timed out");
          rescued.push(child.pid!);
        } catch (error) {
          // Record the earlier release failure before timer/listener failures.
          rescueFailures.push({ release: "sentinel", error });
        } finally {
          if (timer !== undefined) await attempt("sentinel-timer", () => rescueOps.clearTimer(timer!));
          if (onExit) await attempt("sentinel-exit-listener", () => child.removeListener("exit", onExit!));
          if (onError) await attempt("sentinel-error-listener", () => child.removeListener("error", onError!));
        }
      });
    }
    for (const root of roots) await attempt("scratch", () => rescueOps.removeRoot(root));
    await attempt("receipt", () => console.log(JSON.stringify({ evidence: "gated-fixture-independent-rescue", rescuedSentinels: rescued,
      allSentinelsExited: children.every(child => child.exitCode !== null || child.signalCode !== null),
      allScratchRemoved: roots.every(root => !existsSync(root)), releaseFailures: rescueFailures.slice(firstFailure).map(item => item.release) })));
    if (rescueFailures.length > firstFailure) throw rescueFailures[firstFailure]!.error;
  };
  return { operations, roots, children, observe, rescue, rescueFailures };
}
// JavaScript cannot inspect an unwinding exception inside rescue(). Carry its
// presence explicitly, independent of truthiness, without mutating the value.
async function withCustody<T>(owner: ReturnType<typeof custody>, body: () => Promise<T>): Promise<T> {
  let primaryPresent = false;
  try { return await body(); }
  catch (primary) { primaryPresent = true; throw primary; }
  finally {
    try { await owner.rescue(); }
    catch (secondary) { if (!primaryPresent) throw secondary; }
  }
}

async function reportingControl(mode: "normal" | "report" | "serialization", body: () => Promise<unknown>) {
  const owner = custody();
  let acquisitions = 0;
  const refuse = () => { acquisitions++; throw new Error("Reporting-only control attempted acquisition"); };
  owner.operations.makeRoot = refuse;
  owner.operations.spawnSentinel = refuse;
  const log = console.log, stringify = JSON.stringify;
  const diagnostic = new Error(`independent rescue ${mode} diagnostic failure`);
  const reports: string[] = [];
  let failure: Awaited<ReturnType<typeof caught>>;
  try {
    console.log = value => { if (mode === "report") throw diagnostic; reports.push(value); };
    if (mode === "serialization") JSON.stringify = () => { throw diagnostic; };
    failure = await caught(() => withCustody(owner, body));
  } finally { console.log = log; JSON.stringify = stringify; }
  expect(acquisitions).toBe(0);
  expect(owner.roots).toEqual([]); expect(owner.children).toEqual([]);
  return { owner, diagnostic, failure, reports };
}

describe("gated-child independent rescue reporting (no acquisitions)", () => {
  it("reporting regression preserves exact active primaries when final reporting throws", async () => {
    for (const primary of [undefined, null, 0, "", Symbol("primary"), Object.freeze({ primary: true }), Object.freeze(new Error("frozen primary"))]) {
      const { owner, diagnostic, failure } = await reportingControl("report", async () => { throw primary; });
      expect(failure.threw).toBe(true); expect(failure.value).toBe(primary);
      expect(owner.rescueFailures.map(item => item.error)).toEqual([diagnostic]);
    }
  });

  it("reporting regression preserves an exact nonmutable primary when serialization throws", async () => {
    const primary = Object.freeze(new Error("serialization must remain secondary"));
    const { owner, diagnostic, failure } = await reportingControl("serialization", async () => { throw primary; });
    expect(failure.threw).toBe(true); expect(failure.value).toBe(primary);
    expect(owner.rescueFailures.map(item => item.error)).toEqual([diagnostic]);
  });

  it("reporting regression normal reports preserve thrown values and successful return identity", async () => {
    for (const primary of [undefined, null, 0, "", Symbol("normal-report"), Object.freeze({ primary: true })]) {
      const { failure, reports } = await reportingControl("normal", async () => { throw primary; });
      expect(failure.threw).toBe(true); expect(failure.value).toBe(primary); expect(reports).toHaveLength(1);
    }
    const returned = Object.freeze({ returned: true });
    const { failure } = await reportingControl("normal", async () => returned);
    expect(failure.threw).toBe(false); expect(failure.value).toBeUndefined();
    const owner = custody();
    expect(await withCustody(owner, async () => returned)).toBe(returned);
  });

  it("reporting regression surfaces report-only failure when no primary exists", async () => {
    const { failure, diagnostic } = await reportingControl("report", async () => {});
    expect(failure.threw).toBe(true); expect(failure.value).toBe(diagnostic);
  });

  it("reporting regression surfaces serialization-only failure when no primary exists", async () => {
    const { failure, diagnostic } = await reportingControl("serialization", async () => {});
    expect(failure.threw).toBe(true); expect(failure.value).toBe(diagnostic);
  });

  it("keeps the earlier cleanup-only failure authoritative when reporting also fails, without acquisitions", async () => {
    const cleanup = new Error("independent synchronization report failure"), diagnostic = new Error("later reporter failure");
    const owner = custody({ syncExports: () => { syncBuiltinESMExports(); throw cleanup; } });
    const log = console.log;
    let failure: Awaited<ReturnType<typeof caught>>;
    try {
      console.log = () => { throw diagnostic; };
      failure = await caught(() => withCustody(owner, async () => {}));
    } finally { console.log = log; }
    expect(failure.threw).toBe(true); expect(failure.value).toBe(cleanup);
    expect(owner.rescueFailures.map(item => item.error)).toEqual([cleanup, diagnostic]);
    expect(owner.roots).toEqual([]); expect(owner.children).toEqual([]);
  });
});

function expectReleased(state: ReturnType<ReturnType<typeof custody>["observe"]>) {
  expect(state.liveSentinels).toBe(0);
  expect(state.remainingRoots).toEqual([]);
  expect(state.bindingsRestored && state.hooksRestored && state.sentinelListenersRestored).toBe(true);
}

describe.skipIf(process.platform === "win32")("gated-child fixture failure ownership", () => {
  it("releases real sentinel/scratch after the real public API initial mkdir rejects, preserving that error", async () => {
    const owner = custody();
    let primary: unknown;
    let supervisorPid: number | undefined;
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async owned => {
        const stateRoot = path.join(owned.root, "state");
        writeFileSync(stateRoot, "real file blocks API stateRoot/home mkdir");
        try {
          await owned.waitForHelper(executeRecoveryWorker({ dbPath: path.join(owned.root, "unused.sqlite"), docsDir: owned.root,
            outputJson: path.join(owned.root, "absent.json"), stateRoot, timeoutMs: 200,
            command: ["/usr/bin/python3", "-S", "-c", "raise RuntimeError('must not launch')"] }));
        } catch (error) { primary = error; throw error; }
        finally { supervisorPid = owned.supervisorPid; }
      }, owner.operations));
      const state = owner.observe("actual-api-mkdir-refusal");
      expect(failure.threw).toBe(true);
      expect(failure.value).toBe(primary);
      expect(["ENOTDIR", "EEXIST"]).toContain((primary as NodeJS.ErrnoException).code);
      expect(supervisorPid).toBeUndefined();
      expect(state.acquiredSentinels).toBe(1);
      expectReleased(state);
    });
  });

  it("preserves unusual exact primary thrown values, including undefined, before the first acquisition", async () => {
    for (const primary of [undefined, null, 0, "", Symbol("primary"), Object.freeze({ primary: true })]) {
      let launched = false;
      const failure = await caught(() => withGatedChildFixture(source, fixture, async () => {}, {
        makeRoot: () => { throw primary; }, spawnSentinel: () => { launched = true; throw new Error("Unexpected launch"); },
      }));
      expect(failure.threw).toBe(true);
      expect(failure.value).toBe(primary);
      expect(launched).toBe(false);
    }
  });

  it("releases acquired scratch when sentinel setup throws synchronously", async () => {
    const owner = custody();
    const primary = Object.freeze({ phase: "sentinel-launch" });
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async () => {}, {
        ...owner.operations, spawnSentinel: () => { throw primary; },
      }));
      const state = owner.observe("sentinel-launch-refusal");
      expect(failure.threw).toBe(true); expect(failure.value).toBe(primary);
      expect(state.acquiredSentinels).toBe(0); expectReleased(state);
    });
  });

  it("registers restoration before fallible interception synchronization", async () => {
    const owner = custody();
    const primary = new Error("synchronization failed after the real builtin synchronization");
    let syncs = 0;
    let bodyRan = false;
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async () => { bodyRan = true; }, {
        ...owner.operations, syncExports: () => { syncBuiltinESMExports(); if (++syncs === 1) throw primary; },
      }));
      const state = owner.observe("interception-synchronization-failure");
      expect(failure.threw).toBe(true); expect(failure.value).toBe(primary);
      expect(syncs).toBe(2); expect(bodyRan).toBe(false);
      expect(state.acquiredSentinels).toBe(1); expectReleased(state);
    });
  });

  it("drains in-flight helper work before scratch disposal when the body already failed", async () => {
    const owner = custody();
    const primary = Symbol("body failure");
    const order: string[] = [];
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async owned => {
        // Synthetic helper scheduling control, NOT an API/product receipt.
        const helper = new Promise<Awaited<ReturnType<typeof executeRecoveryWorker>>>(resolve => {
          setTimeout(() => { order.push("helper-settled"); resolve(undefined as never); }, 20);
        });
        void owned.waitForHelper(helper).catch(() => {});
        throw primary;
      }, { ...owner.operations, removeRoot: root => { order.push("scratch"); rmSync(root, { recursive: true, force: true }); } }));
      const state = owner.observe("in-flight-drain-before-disposal");
      expect(failure.threw).toBe(true); expect(failure.value).toBe(primary);
      expect(order).toEqual(["helper-settled", "scratch"]); expectReleased(state);
    });
  });

  it("attempts independent releases despite restoration, timer and scratch failures without replacing throw undefined", async () => {
    const owner = custody();
    const attempts: string[] = [];
    let syncs = 0;
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async owned => {
        await owned.waitForHelper(Promise.reject(undefined));
      }, { ...owner.operations,
        syncExports: () => { syncBuiltinESMExports(); if (++syncs > 1) { attempts.push("restore"); throw new Error("injected restoration report failure after real restoration"); } },
        clearGuard: timer => { clearTimeout(timer); attempts.push("guard"); throw new Error("injected timer report failure after real clear"); },
        removeRoot: () => { attempts.push("scratch"); throw new Error("injected scratch removal refusal: attempt only, no deallocation proof"); },
      }));
      const state = owner.observe("mock-release-refusals-attempts-not-deallocation");
      expect(failure.threw).toBe(true); expect(failure.value).toBeUndefined();
      expect(attempts).toEqual(["restore", "guard", "scratch"]);
      expect(state.liveSentinels).toBe(0); expect(state.remainingRoots).toEqual(owner.roots);
      expect(state.bindingsRestored && state.hooksRestored && state.sentinelListenersRestored).toBe(true);
    });
  });

  it("does not turn cleanup-only scratch failure into a passing test", async () => {
    const owner = custody();
    const failureValue = new Error("injected remove report failure after actual deallocation");
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async () => {}, {
        ...owner.operations, removeRoot: root => { rmSync(root, { recursive: true, force: true }); throw failureValue; },
      }));
      const state = owner.observe("cleanup-only-failure-with-real-release");
      expect(failure.threw).toBe(true); expect(failure.value).toBe(failureValue); expectReleased(state);
    });
  });

  it("surfaces a rejected drained helper even when the body did not await it", async () => {
    const owner = custody();
    const primary = null;
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async owned => {
        // Synthetic rejected helper control, not a product cleanup result.
        void owned.waitForHelper(Promise.reject(primary)).catch(() => {});
      }, owner.operations));
      const state = owner.observe("cleanup-only-helper-rejection");
      expect(failure.threw).toBe(true); expect(failure.value).toBe(primary); expectReleased(state);
    });
  });

  it("reports sentinel signal refusal honestly while still removing scratch, before independent exact-handle rescue", async () => {
    const owner = custody();
    const primary = new Error("mocked sentinel signal refusal: not deallocation proof");
    await withCustody(owner, async () => {
      const failure = await caught(() => withGatedChildFixture(source, fixture, async owned => {
        owned.sentinel.kill = () => { throw primary; };
      }, owner.operations));
      const state = owner.observe("mock-sentinel-signal-refusal-before-rescue");
      expect(failure.threw).toBe(true); expect(failure.value).toBe(primary);
      expect(state.liveSentinels).toBe(1); expect(state.remainingRoots).toEqual([]);
      expect(state.bindingsRestored && state.hooksRestored && state.sentinelListenersRestored).toBe(true);
    });
  });

  it("continues independent rescue after synchronization, timer and first-root refusal with truthful residual state", async () => {
    const synchronization = new Error("injected independent synchronization failure after actual synchronization");
    const timerFailure = new Error("injected first timer failure after actual clear");
    const rootRefusal = new Error("injected first root removal refusal: attempt is not deallocation");
    let inject = true, timers = 0;
    let refusedRoot: string | undefined;
    const rootAttempts: string[] = [];
    const owner = custody({
      syncExports: () => { syncBuiltinESMExports(); if (inject) throw synchronization; },
      clearTimer: timer => { clearTimeout(timer); if (inject && ++timers === 1) throw timerFailure; },
      removeRoot: root => {
        if (inject) { rootAttempts.push(root); if (root === refusedRoot) throw rootRefusal; }
        rmSync(root, { recursive: true, force: true });
      },
    });
    await withCustody(owner, async () => {
      try {
        refusedRoot = owner.operations.makeRoot!(); owner.operations.makeRoot!();
        owner.operations.spawnSentinel!(); owner.operations.spawnSentinel!();
        const before = owner.observe("independent-rescue-fault-control-before-subject");
        expect(before.liveSentinels).toBe(2); expect(before.remainingRoots).toEqual(owner.roots);
        // This is the rescue subject, NOT a product fixture success receipt.
        const failure = await caught(() => owner.rescue());
        const state = owner.observe("independent-rescue-fault-control-before-final-independent-cleanup");
        expect(failure.threw).toBe(true); expect(failure.value).toBe(synchronization);
        expect(owner.rescueFailures.map(item => item.error)).toEqual([synchronization, timerFailure, rootRefusal]);
        expect(timers).toBe(2); expect(rootAttempts).toEqual(owner.roots);
        expect(state.liveSentinels).toBe(0); expect(state.remainingRoots).toEqual([owner.roots[0]]);
        expect(state.bindingsRestored && state.hooksRestored && state.sentinelListenersRestored).toBe(true);
      } finally { inject = false; } // Nonfallible; outer primary-aware owner retries only owned residue.
    });
    expectReleased(owner.observe("independent-rescue-fault-control-after-final-independent-cleanup"));
  });
});

// This control owns the actual handle and independent rescue BEFORE installing
// any fault. Unrelated baseline listeners must survive the subject unchanged.
async function sentinelFaultControl(options: {
  timerReport?: boolean;
  exitReport?: boolean;
  errorReport?: boolean;
  reportFailure?: boolean;
  bodyPrimary?: { value: unknown };
  signalPrimary?: { value: unknown };
  registration?: { event: "exit" | "error" | "both"; primary: unknown; secondary?: unknown };
}, check: (result: {
  failure: { threw: boolean; value: unknown };
  returned: object;
  bodyRan: boolean;
  state: { liveSentinels: number; remainingRoots: string[]; terminalSignal: string | null;
    bindingsRestored: boolean; hooksRestored: boolean; listenersRestored: boolean };
  attachments: string[];
  removals: string[];
  clearedTimers: ReturnType<typeof setTimeout>[];
  errors: { timer: Error; exit: Error; error: Error; report: Error };
  outerFailures: { release: string; error: unknown }[];
  sentinelFailures: { release: string; error: unknown }[];
}) => void, control: {
  // Synthetic caller-boundary controls only; omitted by every real subject.
  owner?: ReturnType<typeof custody>;
  finalize?: (release: string, action: () => void) => void;
} = {}) {
  const owner = control.owner ?? custody();
  const spawn = owner.operations.spawnSentinel!;
  const nodeSpawn = childProcess.spawn, bun = (globalThis as any).Bun, bunSpawn = bun?.spawn;
  const hooks = hookCounts();
  const returned = Object.freeze({ successfulReturn: true });
  const errors = { timer: new Error("sentinel timer report after actual clear"),
    exit: new Error("sentinel exit-listener report after actual removal"),
    error: new Error("sentinel error-listener report after actual removal"),
    report: new Error("later sentinel diagnostic report failure") };
  const attachments: string[] = [], removals: string[] = [];
  const clearedTimers: ReturnType<typeof setTimeout>[] = [];
  let child: ChildProcess | undefined;
  let once: ChildProcess["once"] | undefined, remove: ChildProcess["removeListener"] | undefined;
  const baseline = new Map<string, Set<Function>>();
  const unrelatedExit = () => {}, unrelatedError = () => {};
  let timerCleared = false, signalRefused = false, bodyRan = false;
  let outerFailures: { release: string; error: unknown }[] = [];
  let sentinelFailures: { release: string; error: unknown }[] = [];
  const diagnostic = console.error;
  await withCustody(owner, async () => {
    let primaryPresent = false;
    try {
      console.error = (_message, outer, sentinel) => {
        // Retain the actual call-owned arrays, including a later reporter fault.
        outerFailures = outer; sentinelFailures = sentinel ?? [];
        if (options.reportFailure) throw errors.report;
      };
      const operations: Partial<GatedChildFixtureOperations> = {
        ...owner.operations,
        spawnSentinel: () => {
          child = spawn(); // owner has already retained the handle and kill binding
          once = child.once; remove = child.removeListener;
          child.on("exit", unrelatedExit); child.on("error", unrelatedError);
          for (const event of ["exit", "error"]) baseline.set(event, new Set(child.rawListeners(event)));
          child.once = function(event, listener) {
            const result = (once! as EventEmitter["once"]).call(this, event, listener) as ChildProcess;
            if (event === "exit" || event === "error") {
              attachments.push(event);
              if (options.registration?.event === event || options.registration?.event === "both") {
                throw event === "error" && options.registration.event === "both"
                  ? options.registration.secondary : options.registration.primary;
              }
            }
            return result;
          };
          child.removeListener = function(event, listener) {
            const result = remove!.call(this, event, listener);
            // EventEmitter's once self-removal happens BEFORE terminal settles.
            // Inject only the explicit disposer, not that native self-removal.
            if ((timerCleared || signalRefused) && !baseline.get(String(event))?.has(listener)) {
              removals.push(String(event));
              if (event === "exit" && options.exitReport) throw errors.exit;
              if (event === "error" && options.errorReport) throw errors.error;
            }
            return result;
          };
          if (options.signalPrimary) child.kill = () => {
            signalRefused = true; throw options.signalPrimary!.value;
          };
          return child;
        },
        clearSentinelTimer: timer => {
          clearTimeout(timer); clearedTimers.push(timer); timerCleared = true;
          if (options.timerReport) throw errors.timer;
        },
      };
      let failure: { threw: boolean; value: unknown };
      try {
        const value = await withGatedChildFixture(source, fixture, async () => {
          bodyRan = true;
          if (options.bodyPrimary) throw options.bodyPrimary.value;
          return returned;
        }, operations);
        failure = { threw: false, value };
      } catch (value) { failure = { threw: true, value }; }
      console.error = diagnostic;
      const state = {
        liveSentinels: owner.children.filter(value => value.exitCode === null && value.signalCode === null).length,
        remainingRoots: owner.roots.filter(root => existsSync(root)), terminalSignal: child?.signalCode ?? null,
        bindingsRestored: childProcess.spawn === nodeSpawn && (!bun || bun.spawn === bunSpawn),
        hooksRestored: JSON.stringify(hookCounts()) === JSON.stringify(hooks),
        listenersRestored: !!child && ["exit", "error"].every(event => {
          const actual = child!.rawListeners(event), expected = baseline.get(event)!;
          return actual.length === expected.size && actual.every(listener => expected.has(listener));
        }),
      };
      console.log(JSON.stringify({ evidence: "sentinel-fault-before-independent-rescue", state, bodyRan,
        host: { executable: process.execPath, node: process.versions.node, bun: process.versions.bun ?? null },
        attachments, removals, actualTimerClears: clearedTimers.length,
        sentinelFailures: sentinelFailures.map(item => item.release), outerFailures: outerFailures.map(item => item.release) }));
      check({ failure, returned, bodyRan, state, attachments, removals, clearedTimers, errors, outerFailures, sentinelFailures });
    } catch (primary) {
      primaryPresent = true; throw primary;
    } finally {
      // Disable only these exact-handle injections before independent rescue.
      // Required observations and assertions above precede every rescue action.
      const firstFailure = owner.rescueFailures.length;
      const attempt = (release: string, action: () => void) => {
        try { if (control.finalize) control.finalize(release, action); else action(); }
        catch (error) { owner.rescueFailures.push({ release, error }); }
      };
      attempt("sentinel-control-diagnostic", () => { console.error = diagnostic; });
      const removed: string[] = [];
      if (child && once) attempt("sentinel-control-once-binding", () => { child!.once = once!; });
      if (child && remove) {
        attempt("sentinel-control-remove-binding", () => { child!.removeListener = remove!; });
        for (const event of ["exit", "error"]) {
          let extras: ReturnType<ChildProcess["rawListeners"]> = [];
          attempt(`sentinel-control-${event}-listeners`, () => {
            extras = child!.rawListeners(event).filter(listener => !baseline.get(event)?.has(listener));
          });
          for (const [index, listener] of extras.entries()) {
            attempt(`sentinel-control-extra-${event}-listener-${index}`, () => {
              remove!.call(child, event, listener as (...args: any[]) => void); removed.push(event);
            });
          }
        }
        attempt("sentinel-control-unrelated-exit-listener", () => { remove!.call(child, "exit", unrelatedExit); });
        attempt("sentinel-control-unrelated-error-listener", () => { remove!.call(child, "error", unrelatedError); });
      }
      for (const [index, timer] of clearedTimers.entries()) attempt(`sentinel-control-timer-${index}`, () => clearTimeout(timer));
      attempt("sentinel-control-receipt", () => console.log(JSON.stringify({ evidence: "sentinel-fault-independent-listener-rescue", removed })));
      // Retain secondaries in call-owned state; never decorate an arbitrary
      // primary. Cleanup/report-only failure still makes this caller fail.
      if (!primaryPresent && owner.rescueFailures.length > firstFailure) throw owner.rescueFailures[firstFailure]!.error;
    }
  });
  expectReleased(owner.observe("sentinel-fault-after-independent-rescue"));
}

function expectSentinelReleased(result: Parameters<Parameters<typeof sentinelFaultControl>[1]>[0]) {
  expect(result.state.liveSentinels).toBe(0);
  expect(result.state.terminalSignal).toBe("SIGKILL");
  expect(result.state.remainingRoots).toEqual([]);
  expect(result.state.bindingsRestored && result.state.hooksRestored && result.state.listenersRestored).toBe(true);
  expect(result.attachments).toEqual(["exit", "error"]);
  expect(result.removals).toEqual(["exit", "error"]);
  expect(result.clearedTimers).toHaveLength(1);
}

describe.skipIf(process.platform === "win32")("gated-child sentinel disposer independence", () => {
  it("clears the real sentinel timer then reports failure without skipping either listener", async () => {
    await sentinelFaultControl({ timerReport: true }, result => {
      expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(result.errors.timer);
      expect(result.sentinelFailures.map(item => item.error)).toEqual([result.errors.timer]);
      expectSentinelReleased(result);
    });
  });

  it("removes the actual exit listener then reports failure without skipping the error listener", async () => {
    await sentinelFaultControl({ exitReport: true }, result => {
      expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(result.errors.exit);
      expect(result.sentinelFailures.map(item => item.error)).toEqual([result.errors.exit]);
      expectSentinelReleased(result);
    });
  });

  it("surfaces an error-listener removal report after all actual releases", async () => {
    await sentinelFaultControl({ errorReport: true }, result => {
      expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(result.errors.error);
      expect(result.sentinelFailures.map(item => item.error)).toEqual([result.errors.error]);
      expectSentinelReleased(result);
    });
  });

  it("preserves opaque body primaries through every sentinel release report and diagnostic failure", async () => {
    for (const primary of [undefined, null, 0, "", Symbol("sentinel-body"), Object.freeze({ primary: true }), Object.freeze(new Error("frozen sentinel body"))]) {
      await sentinelFaultControl({ bodyPrimary: { value: primary }, timerReport: true, exitReport: true, errorReport: true, reportFailure: true }, result => {
        expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(primary);
        expect(result.sentinelFailures.map(item => item.error)).toEqual([result.errors.timer, result.errors.exit, result.errors.error]);
        expect(result.outerFailures.map(item => item.error)).toEqual([result.errors.timer, result.errors.report]);
        expectSentinelReleased(result);
      });
    }
  });

  it("preserves a signal-refusal primary through listener and diagnostic failures without claiming termination", async () => {
    for (const primary of [undefined, null, 0, Object.freeze(new Error("frozen sentinel signal refusal"))]) {
      await sentinelFaultControl({ signalPrimary: { value: primary }, exitReport: true, errorReport: true, reportFailure: true }, result => {
        expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(primary);
        expect(result.sentinelFailures.map(item => item.error)).toEqual([primary, result.errors.exit, result.errors.error]);
        expect(result.outerFailures.map(item => item.error)).toEqual([primary, result.errors.report]);
        expect(result.state.liveSentinels).toBe(1); expect(result.state.terminalSignal).toBeNull();
        expect(result.state.remainingRoots).toEqual([]);
        expect(result.state.bindingsRestored && result.state.hooksRestored && result.state.listenersRestored).toBe(true);
        expect(result.removals).toEqual(["exit", "error"]); expect(result.clearedTimers).toEqual([]);
      });
    }
  });

  it("retains release capability before either observer attachment reports failure", async () => {
    for (const event of ["exit", "error"] as const) for (const primary of [undefined, Symbol("observer-primary"), Object.freeze(new Error("frozen observer report"))]) {
      await sentinelFaultControl({ registration: { event, primary }, timerReport: true, exitReport: true, errorReport: true, reportFailure: true }, result => {
        expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(primary); expect(result.bodyRan).toBe(false);
        expect(result.sentinelFailures.map(item => item.error)).toEqual([primary, result.errors.timer, result.errors.exit, result.errors.error]);
        expect(result.outerFailures.map(item => item.error)).toEqual([result.errors.timer, result.errors.report]);
        expectSentinelReleased(result);
      });
    }
  });

  it("attempts both observer registrations and retains the first exact setup primary plus secondary outcomes", async () => {
    await sentinelFaultControl({ registration: { event: "both", primary: undefined, secondary: null }, timerReport: true, exitReport: true, errorReport: true }, result => {
      expect(result.failure.threw).toBe(true); expect(result.failure.value).toBeUndefined(); expect(result.bodyRan).toBe(false);
      expect(result.sentinelFailures.map(item => item.error)).toEqual([undefined, null, result.errors.timer, result.errors.exit, result.errors.error]);
      expectSentinelReleased(result);
    });
  });

  it("preserves successful return identity and unrelated listeners when no sentinel operation fails", async () => {
    await sentinelFaultControl({}, result => {
      expect(result.failure.threw).toBe(false); expect(result.failure.value).toBe(result.returned);
      expect(result.sentinelFailures).toEqual([]); expect(result.outerFailures).toEqual([]);
      expectSentinelReleased(result);
    });
  });
});

// No real root, ChildProcess, helper, signal or recovery/API invocation. The
// actual fixture and caller execute against one explicitly synthetic emitter.
// This is caller-finalization evidence, never native deallocation evidence.
async function sentinelCallerFinalizationControl(options: {
  mode?: "normal" | "report" | "serialization";
  primary?: { value: unknown };
  faults?: { release: string; error: unknown; before?: boolean }[];
}) {
  const owner = custody();
  const root = path.join(tmpdir(), "qmd-never-created-caller-finalization");
  const child = Object.assign(new EventEmitter(), {
    pid: Symbol("synthetic handle, not numeric signal authority"), exitCode: null, signalCode: null,
  }) as unknown as ChildProcess & { signalCode: NodeJS.Signals | null };
  const once = child.once, remove = child.removeListener;
  const nodeSpawn = childProcess.spawn, bun = (globalThis as any).Bun, bunSpawn = bun?.spawn;
  const log = console.log, diagnosticBinding = console.error, stringify = JSON.stringify;
  const diagnostic = Object.freeze(new Error(`caller listener-rescue ${options.mode ?? "normal"} failure`));
  const attempts: string[] = [], reports: string[] = [];
  let roots = 0, sentinels = 0, scratchReleases = 0, syntheticSignals = 0, reportAttempts = 0, checkRan = false;
  // These overrides neither acquire nor register anything with independent
  // custody. Default acquisition functions remain unreachable in this control.
  owner.operations.makeRoot = () => { roots++; return root; };
  owner.operations.spawnSentinel = () => { sentinels++; return child; };
  owner.operations.removeRoot = value => { expect(value).toBe(root); scratchReleases++; };
  child.kill = signal => {
    expect(signal).toBe("SIGKILL"); syntheticSignals++;
    queueMicrotask(() => { child.signalCode = "SIGKILL"; child.emit("exit", null, "SIGKILL"); });
    return true;
  };
  let failure: Awaited<ReturnType<typeof caught>>;
  let bindingsRestored = false, listenersRemaining = -1;
  try {
    JSON.stringify = ((value: any, ...args: any[]) => {
      if (value?.evidence === "sentinel-fault-independent-listener-rescue") {
        reportAttempts++;
        if (options.mode === "serialization") throw diagnostic;
      }
      return (stringify as any)(value, ...args);
    }) as typeof JSON.stringify;
    console.log = value => {
      if (options.mode === "report" && JSON.parse(value).evidence === "sentinel-fault-independent-listener-rescue") throw diagnostic;
      reports.push(value);
    };
    failure = await caught(() => sentinelFaultControl({}, result => {
      checkRan = true;
      // Required observations precede independent rescue, also in this fake.
      expect(result.failure.threw).toBe(false); expect(result.failure.value).toBe(result.returned);
      expect(result.state.listenersRestored).toBe(true);
      expect(result.sentinelFailures).toEqual([]); expect(result.outerFailures).toEqual([]);
      // Two independently owned extras per event prove a first release failure
      // cannot skip another listener or the unrelated-baseline removals.
      for (const event of ["exit", "error"]) { child.on(event, () => {}); child.on(event, () => {}); }
      if (options.primary) throw options.primary.value;
    }, { owner, finalize: (release, action) => {
      attempts.push(release);
      const fault = options.faults?.find(item => item.release === release);
      if (fault?.before) throw fault.error;
      action();
      if (fault) throw fault.error;
    } }));
    bindingsRestored = child.once === once && child.removeListener === remove && console.error === diagnosticBinding
      && childProcess.spawn === nodeSpawn && (!bun || bun.spawn === bunSpawn);
    listenersRemaining = child.listenerCount("exit") + child.listenerCount("error");
  } finally {
    console.log = log; JSON.stringify = stringify; console.error = diagnosticBinding;
    // Synthetic test hygiene only, after the observations above, not subject
    // success. The captured remover acts on each exact remaining listener.
    child.once = once; child.removeListener = remove;
    for (const event of ["exit", "error"]) for (const listener of child.rawListeners(event)) remove.call(child, event, listener as (...args: any[]) => void);
  }
  expect(checkRan).toBe(true);
  expect(roots).toBe(1); expect(sentinels).toBe(1); expect(scratchReleases).toBe(1); expect(syntheticSignals).toBe(1);
  expect(owner.roots).toEqual([]); expect(owner.children).toEqual([]); expect(existsSync(root)).toBe(false);
  expect(console.log).toBe(log); expect(JSON.stringify).toBe(stringify); expect(console.error).toBe(diagnosticBinding);
  return { owner, failure, diagnostic, attempts, reports, reportAttempts, bindingsRestored, listenersRemaining };
}

describe("sentinel caller inner finalization (synthetic, zero real acquisitions)", () => {
  for (const [label, primary] of [
    ["undefined", undefined], ["null", null], ["zero", 0], ["frozen Error", Object.freeze(new Error("exact inner caller primary"))],
  ] as const) {
    for (const mode of ["normal", "report", "serialization"] as const) {
      it(`caller finalization preserves exact ${label} check primary with ${mode} reporting`, async () => {
        const result = await sentinelCallerFinalizationControl({ mode, primary: { value: primary } });
        expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(primary);
        expect(result.owner.rescueFailures.map(item => item.error)).toEqual(mode === "normal" ? [] : [result.diagnostic]);
        expect(result.owner.rescueFailures.map(item => item.release)).toEqual(mode === "normal" ? [] : ["sentinel-control-receipt"]);
        expect(result.reportAttempts).toBe(1); expect(result.bindingsRestored).toBe(true); expect(result.listenersRemaining).toBe(0);
      });
    }
    it(`caller finalization continues independent restoration and listener releases with ${label} primary`, async () => {
      const releases = ["sentinel-control-diagnostic", "sentinel-control-once-binding", "sentinel-control-remove-binding",
        "sentinel-control-extra-exit-listener-0", "sentinel-control-extra-error-listener-0",
        "sentinel-control-unrelated-exit-listener", "sentinel-control-unrelated-error-listener", "sentinel-control-timer-0"];
      const faults = releases.map(release => ({ release, error: Object.freeze(new Error(release)) }));
      const result = await sentinelCallerFinalizationControl({ mode: "report", primary: { value: primary }, faults });
      expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(primary);
      expect(result.owner.rescueFailures.map(item => item.release)).toEqual([...releases, "sentinel-control-receipt"]);
      expect(result.owner.rescueFailures.map(item => item.error)).toEqual([...faults.map(item => item.error), result.diagnostic]);
      expect(result.attempts).toEqual(["sentinel-control-diagnostic", "sentinel-control-once-binding", "sentinel-control-remove-binding",
        "sentinel-control-exit-listeners", "sentinel-control-extra-exit-listener-0", "sentinel-control-extra-exit-listener-1",
        "sentinel-control-error-listeners", "sentinel-control-extra-error-listener-0", "sentinel-control-extra-error-listener-1",
        "sentinel-control-unrelated-exit-listener", "sentinel-control-unrelated-error-listener", "sentinel-control-timer-0", "sentinel-control-receipt"]);
      expect(result.reportAttempts).toBe(1); expect(result.bindingsRestored).toBe(true); expect(result.listenersRemaining).toBe(0);
    });
    it(`caller finalization surfaces earliest cleanup-only ${label} before a later reporter error`, async () => {
      const release = "sentinel-control-once-binding";
      const result = await sentinelCallerFinalizationControl({ mode: "report", faults: [{ release, error: primary }] });
      expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(primary);
      expect(result.owner.rescueFailures.map(item => item.release)).toEqual([release, "sentinel-control-receipt"]);
      expect(result.owner.rescueFailures.map(item => item.error)).toEqual([primary, result.diagnostic]);
      expect(result.reportAttempts).toBe(1); expect(result.bindingsRestored).toBe(true); expect(result.listenersRemaining).toBe(0);
    });
  }
  for (const mode of ["report", "serialization"] as const) {
    it(`caller finalization surfaces ${mode}-only failure after a successful body`, async () => {
      const result = await sentinelCallerFinalizationControl({ mode });
      expect(result.failure.threw).toBe(true); expect(result.failure.value).toBe(result.diagnostic);
      expect(result.owner.rescueFailures.map(item => item.error)).toEqual([result.diagnostic]);
      expect(result.owner.rescueFailures.map(item => item.release)).toEqual(["sentinel-control-receipt"]);
      expect(result.reportAttempts).toBe(1); expect(result.bindingsRestored).toBe(true); expect(result.listenersRemaining).toBe(0);
    });
  }
  it("caller finalization retains normal successful return identity and separately reports rescue", async () => {
    const result = await sentinelCallerFinalizationControl({ mode: "normal" });
    expect(result.failure.threw).toBe(false); expect(result.failure.value).toBeUndefined();
    expect(result.owner.rescueFailures).toEqual([]); expect(result.reportAttempts).toBe(1);
    expect(result.reports.map(value => JSON.parse(value).evidence)).toEqual([
      "gated-child-api-finally", "sentinel-fault-before-independent-rescue", "sentinel-fault-independent-listener-rescue",
      "gated-fixture-independent-rescue", "gated-fixture-before-independent-rescue"]);
    expect(result.bindingsRestored).toBe(true); expect(result.listenersRemaining).toBe(0);
  });
  it("caller finalization continues known independent releases after a pre-effect refusal, without claiming release", async () => {
    const release = "sentinel-control-extra-exit-listener-0", secondary = Object.freeze(new Error("synthetic pre-effect refusal"));
    const result = await sentinelCallerFinalizationControl({ primary: { value: undefined }, faults: [{ release, error: secondary, before: true }] });
    expect(result.failure.threw).toBe(true); expect(result.failure.value).toBeUndefined();
    expect(result.owner.rescueFailures.map(item => item.error)).toEqual([secondary]);
    expect(result.owner.rescueFailures.map(item => item.release)).toEqual([release]);
    expect(result.attempts).toContain("sentinel-control-extra-exit-listener-1");
    expect(result.attempts).toContain("sentinel-control-extra-error-listener-1");
    expect(result.attempts).toContain("sentinel-control-unrelated-error-listener");
    expect(result.attempts).toContain("sentinel-control-timer-0"); expect(result.reportAttempts).toBe(1);
    expect(result.bindingsRestored).toBe(true); expect(result.listenersRemaining).toBe(1);
  });
});
