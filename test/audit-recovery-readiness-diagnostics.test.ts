// Shared assertions run in Node/Vitest and native `bun test`, not full Bun/Vitest.
const testRuntime = (process.versions as Record<string, string>).bun ? "bun:test" : "vitest";
const { describe, expect, it }: typeof import("vitest") = await import(testRuntime);
import { lstatSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { RecoveryProcessResult } from "../scripts/phase5_e2e_runner.js";
import { readRecoveryFixturePids } from "./helpers/recovery-readiness-diagnostics.js";

// Authored result mock only: these scalars are not a native cleanup receipt.
function authoredResult(): RecoveryProcessResult {
  return { launch: { state: "not-started", admissionGranted: false, supervisorReaped: true },
    exitCode: null, signal: null, timedOut: true, stdout: "PRIVATE-STDOUT", stderr: "PRIVATE-STDERR", error: "PRIVATE-ERROR",
    cleanup: { childReaped: false, groupTerminated: true, sessionTerminated: true, pipesClosed: true,
      forcedPipeClose: false, containmentVerified: false, sigkillUsed: false, processGroups: [] } };
}

function fixture() {
  const root = mkdtempSync(path.join(tmpdir(), "qmd-readiness-diagnostics-"));
  return { root, pidFile: path.join(root, "owned-pids.json"), outputJson: path.join(root, "worker-report.json") };
}

// Scoped synchronous mocks, restored even if the tested operation throws.
function scoped<T>(enabled: string | undefined, log: (text: string) => void, run: () => T): T {
  const previous = process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS;
  const original = console.error;
  try {
    if (enabled === undefined) delete process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS;
    else process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS = enabled;
    console.error = log;
    return run();
  } finally {
    console.error = original;
    if (previous === undefined) delete process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS;
    else process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS = previous;
  }
}

function thrown(run: () => unknown): unknown {
  try { run(); } catch (error) { return error; }
  throw new Error("Expected a primary read/parse failure");
}

describe("failure-only recovery fixture PID readiness diagnostics", () => {
  it("does no diagnostic observation or output unless the inherited flag is exactly 1", () => {
    for (const enabled of [undefined, "0", "true", "01"]) {
      const primary = new Error("authored read failure");
      let snapshots = 0;
      let outputs = 0;
      const error = scoped(enabled, () => { outputs++; }, () => thrown(() => readRecoveryFixturePids(
        () => { throw primary; }, () => { snapshots++; throw new Error("must not observe"); })));
      expect(error).toBe(primary);
      expect(snapshots).toBe(0);
      expect(outputs).toBe(0);
    }
  });

  it("returns successful real PID reads unchanged without observing or logging, enabled or disabled", () => {
    const f = fixture();
    try {
      writeFileSync(f.pidFile, '{"parent":17,"grandchild":19}'); // Authored data, never signalled.
      for (const enabled of [undefined, "1"]) {
        let reads = 0;
        let snapshots = 0;
        let outputs = 0;
        const actual = scoped(enabled, () => { outputs++; }, () => readRecoveryFixturePids(
          () => { reads++; return JSON.parse(readFileSync(f.pidFile, "utf8")); },
          () => { snapshots++; throw new Error("success must not observe"); }));
        expect(actual).toEqual({ parent: 17, grandchild: 19 });
        expect(reads).toBe(1);
        expect(snapshots).toBe(0);
        expect(outputs).toBe(0);
      }
    } finally { rmSync(f.root, { recursive: true, force: true }); }
  });

  it("records genuine owned ENOENT evidence before cleanup and rethrows the exact primary object", () => {
    const f = fixture();
    const events: string[] = [];
    const records: string[] = [];
    let primary: unknown;
    let caught: unknown;
    try {
      caught = scoped("1", text => {
        expect(lstatSync(f.root).isDirectory()).toBe(true);
        events.push("diagnostic"); records.push(text);
      }, () => thrown(() => readRecoveryFixturePids(() => {
        events.push("read");
        try { return JSON.parse(readFileSync(f.pidFile, "utf8")); }
        catch (error) { primary = error; throw error; }
      }, () => {
        events.push("snapshot");
        return { mode: "new-session", elapsedMs: 1001.25, guardFired: false, result: authoredResult(), ...f };
      })));
    } finally {
      events.push("cleanup"); rmSync(f.root, { recursive: true, force: true });
    }
    expect(primary).toMatchObject({ code: "ENOENT" });
    expect(caught).toBe(primary);
    expect(events).toEqual(["read", "snapshot", "diagnostic", "cleanup"]);
    expect(records).toHaveLength(1);
    const record = JSON.parse(records[0]!);
    expect(record).toMatchObject({ evidence: "recovery-closed-pipes-readiness-failure-before-finally",
      mode: "new-session", elapsedMs: 1001.25, guardFired: false,
      pathFacts: { pidRecord: { exists: false }, pidTemporary: { exists: false }, workerReport: { exists: false } },
      result: { launch: { state: "not-started", admissionGranted: false, supervisorReaped: true },
        cleanup: { childReaped: false, containmentVerified: false, processGroups: [] } } });
    expect(record.result).not.toHaveProperty("pid");
    expect(record).not.toHaveProperty("finallyKilled");
    expect(record).not.toHaveProperty("allFixturePidsAbsent");
    expect(records[0]).not.toMatch(/PRIVATE-|argv|environment|allClean/);
  });

  it("keeps a genuine JSON parse failure and no-follow byte-size facts without opening paths", () => {
    const f = fixture();
    try {
      writeFileSync(f.pidFile, "{");
      writeFileSync(f.pidFile + ".tmp", "partial");
      // The report is a link to a much larger authored file: lstat must report
      // link bytes, never target bytes/content. The helper does not read it.
      const target = path.join(f.root, "authored-target");
      writeFileSync(target, "x".repeat(10_000));
      symlinkSync(target, f.outputJson);
      let primary: unknown;
      const records: string[] = [];
      const caught = scoped("1", text => records.push(text), () => thrown(() => readRecoveryFixturePids(
        () => {
          try { return JSON.parse(readFileSync(f.pidFile, "utf8")); }
          catch (error) { primary = error; throw error; }
        }, () => ({ mode: "new-pgid", elapsedMs: 42, guardFired: true, result: authoredResult(), ...f }))));
      expect(primary).toBeInstanceOf(SyntaxError);
      expect(caught).toBe(primary);
      const { pathFacts, guardFired } = JSON.parse(records[0]!);
      expect(pathFacts).toEqual({ pidRecord: { exists: true, kind: "file", bytes: 1 },
        pidTemporary: { exists: true, kind: "file", bytes: 7 },
        workerReport: { exists: true, kind: "symlink", bytes: lstatSync(f.outputJson).size } });
      expect(guardFired).toBe(true);
    } finally { rmSync(f.root, { recursive: true, force: true }); }
  });

  it("uses metadata only for nonregular objects, including a directory", () => {
    const f = fixture();
    try {
      mkdirSync(f.pidFile);
      const primary = new Error("authored boundary error on a nonregular record");
      const records: string[] = [];
      scoped("1", text => records.push(text), () => expect(thrown(() => readRecoveryFixturePids(
        () => { throw primary; }, () => ({ mode: "new-session", elapsedMs: 1, guardFired: false, result: authoredResult(), ...f })))).toBe(primary));
      expect(JSON.parse(records[0]!).pathFacts.pidRecord).toEqual({ exists: true, kind: "directory", bytes: lstatSync(f.pidFile).size });
    } finally { rmSync(f.root, { recursive: true, force: true }); }
  });

  it("preserves the primary when diagnostic context observation throws", () => {
    const primary = new Error("authored primary");
    let outputs = 0;
    const caught = scoped("1", () => { outputs++; }, () => thrown(() => readRecoveryFixturePids(
      () => { throw primary; }, () => { throw new Error("authored observation failure"); })));
    expect(caught).toBe(primary);
    expect(outputs).toBe(0);
  });

  it("preserves the primary when path observation throws", () => {
    const f = fixture();
    try {
      const primary = new Error("authored primary");
      let outputs = 0;
      const caught = scoped("1", () => { outputs++; }, () => thrown(() => readRecoveryFixturePids(
        () => { throw primary; }, () => ({ mode: "new-session", elapsedMs: 1, guardFired: false,
          result: authoredResult(), ...f, get pidFile(): string { throw new Error("authored path observation failure"); } }))));
      expect(caught).toBe(primary);
      expect(outputs).toBe(0);
    } finally { rmSync(f.root, { recursive: true, force: true }); }
  });

  it("preserves the primary when JSON serialization throws", () => {
    const f = fixture();
    const stringify = JSON.stringify;
    try {
      const primary = new Error("authored primary");
      let outputs = 0;
      const caught = scoped("1", () => { outputs++; }, () => {
        JSON.stringify = () => { throw new Error("authored serialization failure"); };
        return thrown(() => readRecoveryFixturePids(() => { throw primary; },
          () => ({ mode: "new-session", elapsedMs: 1, guardFired: false, result: authoredResult(), ...f })));
      });
      expect(caught).toBe(primary);
      expect(outputs).toBe(0);
    } finally { JSON.stringify = stringify; rmSync(f.root, { recursive: true, force: true }); }
  });

  it("preserves the primary when the diagnostic output sink throws, without a fallback log", () => {
    const f = fixture();
    try {
      const primary = new Error("authored primary");
      let outputs = 0;
      const caught = scoped("1", () => { outputs++; throw new Error("authored output failure"); }, () => thrown(() => readRecoveryFixturePids(
        () => { throw primary; }, () => ({ mode: "new-session", elapsedMs: 1, guardFired: false, result: authoredResult(), ...f }))));
      expect(caught).toBe(primary);
      expect(outputs).toBe(1);
    } finally { rmSync(f.root, { recursive: true, force: true }); }
  });

  it("bounds the returned-result projection and excludes arbitrary text/report/argv/env fields", () => {
    const f = fixture();
    try {
      const result = authoredResult();
      result.cleanup.processGroups = Array.from({ length: 100 }, (_, i) => i + 1); // Authored IDs; never signalled.
      result.transportFailure = { prepared: false, supervisorExited: true, supervisorCloseObserved: false,
        stdout: { bytes: 23, ended: false, closed: false, destroyed: false },
        stderr: { bytes: 31, ended: false, closed: true, destroyed: false },
        proof: { bytes: 47, ended: true, closed: true, destroyed: true } };
      const records: string[] = [];
      scoped("1", text => records.push(text), () => thrown(() => readRecoveryFixturePids(() => { throw new Error("primary"); },
        () => ({ mode: "new-session", elapsedMs: 1, guardFired: false, result, ...f }))));
      const record = JSON.parse(records[0]!);
      expect(record.result.cleanup.processGroups).toEqual(result.cleanup.processGroups.slice(0, 16));
      expect(record.result.cleanup.processGroupsOmitted).toBe(84);
      expect(record.result.outputBytes).toEqual({ stdout: Buffer.byteLength(result.stdout), stderr: Buffer.byteLength(result.stderr), error: Buffer.byteLength(result.error!) });
      expect(record.result.transportFailure).toEqual(result.transportFailure);
      expect(Object.keys(record.result).sort()).toEqual(["cleanup", "exitCode", "launch", "outputBytes", "signal", "timedOut", "transportFailure"]);
      expect(records[0]!.length).toBeLessThan(4096);
    } finally { rmSync(f.root, { recursive: true, force: true }); }
  });
});
