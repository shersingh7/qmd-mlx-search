import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import childProcess from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { executeRecoveryWorker } from "../scripts/phase5_e2e_runner.js";

// These routing/preflight controls intercept acquisition, not native execution.
describe.skipIf(process.platform === "win32")("recovery supervisor interpreter selection", () => {
  let root: string;
  let previous: string | undefined;
  let intercepted: ReturnType<typeof vi.spyOn>;
  const options = () => ({ dbPath: path.join(root, "unused.sqlite"), docsDir: root,
    outputJson: path.join(root, "absent.json"), stateRoot: path.join(root, "state"),
    timeoutMs: 200, command: [process.execPath, "--version"] });

  beforeEach(() => {
    root = mkdtempSync(path.join(tmpdir(), "qmd-supervisor-python-"));
    previous = process.env.QMD_RECOVERY_PYTHON;
    delete process.env.QMD_RECOVERY_PYTHON;
    intercepted = vi.spyOn(childProcess, "spawn").mockImplementation(() => { throw new Error("acquisition intercepted"); });
    syncBuiltinESMExports();
  });
  afterEach(() => {
    vi.restoreAllMocks(); syncBuiltinESMExports();
    if (previous === undefined) delete process.env.QMD_RECOVERY_PYTHON;
    else process.env.QMD_RECOVERY_PYTHON = previous;
    rmSync(root, { recursive: true, force: true });
  });

  it("preserves the literal legacy default when unconfigured", async () => {
    await expect(executeRecoveryWorker(options())).rejects.toThrow("acquisition intercepted");
    expect(intercepted).toHaveBeenCalledTimes(1);
    expect(intercepted.mock.calls[0]![0]).toBe("/usr/bin/python3");
    expect((intercepted.mock.calls[0]![1] as string[]).slice(0, 2)).toEqual(["-S", "-u"]);
  });

  it("uses the explicitly configured absolute executable without a probe spawn", async () => {
    process.env.QMD_RECOVERY_PYTHON = process.execPath;
    await expect(executeRecoveryWorker(options())).rejects.toThrow("acquisition intercepted");
    expect(intercepted).toHaveBeenCalledTimes(1);
    expect(intercepted.mock.calls[0]![0]).toBe(process.execPath);
  });

  it("lets a call-local interpreter override the environment", async () => {
    process.env.QMD_RECOVERY_PYTHON = "invalid-relative-python";
    await expect(executeRecoveryWorker({ ...options(), supervisorPython: process.execPath })).rejects.toThrow("acquisition intercepted");
    expect(intercepted).toHaveBeenCalledTimes(1);
    expect(intercepted.mock.calls[0]![0]).toBe(process.execPath);
  });

  for (const kind of ["empty", "relative", "missing", "directory", "not-executable"] as const) {
    it(`refuses ${kind} configuration before any acquisition, without fallback`, async () => {
      const file = path.join(root, "not-executable");
      writeFileSync(file, "fixture"); chmodSync(file, 0o600);
      process.env.QMD_RECOVERY_PYTHON = { empty: "", relative: "python3", missing: path.join(root, "missing"), directory: root, "not-executable": file }[kind];
      const result = await executeRecoveryWorker(options());
      expect(intercepted).not.toHaveBeenCalled();
      expect(result.launch).toEqual({ state: "not-attempted", admissionGranted: false, supervisorReaped: false });
      expect(result.error).toMatch(/supervisor interpreter/i);
      expect(result.timedOut).toBe(false);
    });
  }
});
