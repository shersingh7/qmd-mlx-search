import { describe, expect, it } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Node-only installations may omit Bun. Advertised Bun CI jobs MUST set
// REQUIRE_BUN=1 (and may set BUN_EXECUTABLE to an absolute local executable).
const executable = process.env.BUN_EXECUTABLE || "bun";
const version = spawnSync(executable, ["--version"], { encoding: "utf8", timeout: 5_000 });
const missing = (version.error as NodeJS.ErrnoException | undefined)?.code === "ENOENT";
const required = process.env.REQUIRE_BUN === "1" || process.env.BUN_EXECUTABLE !== undefined;
const repository = fileURLToPath(new URL("../", import.meta.url));

type Evidence = Record<string, any>;
function runNative(args: string[]): { records: Evidence[]; stderr: string } {
  expect(process.platform, "Native recovery ownership requires POSIX").not.toBe("win32");
  expect(version.error, `Native Bun is required: ${executable}`).toBeUndefined();
  expect(version.status, version.stderr).toBe(0);
  expect(version.stdout.trim()).toMatch(/^\d+\.\d+\.\d+/);
  const root = mkdtempSync(path.join(tmpdir(), "qmd-native-bun-wrapper-"));
  for (const name of ["home", "cache", "config", "hermes"]) mkdirSync(path.join(root, name));
  try {
    const result = spawnSync(executable, args, { cwd: repository, encoding: "utf8", timeout: 25_000,
      maxBuffer: 2 * 1024 * 1024, env: { ...process.env, HOME: path.join(root, "home"),
        XDG_CACHE_HOME: path.join(root, "cache"), XDG_CONFIG_HOME: path.join(root, "config"),
        HERMES_HOME: path.join(root, "hermes"), TMPDIR: tmpdir(), CI: "true", QMD_LLAMA_GPU: "false",
        HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_DATASETS_OFFLINE: "1" } });
    console.log(result.stdout);
    const diagnostics = `${result.error ?? ""}\n${result.stdout}\n${result.stderr}`;
    expect(result.error, diagnostics).toBeUndefined();
    expect(result.signal, diagnostics).toBeNull();
    expect(result.status, diagnostics).toBe(0);
    const records = result.stdout.split("\n").filter(line => line.startsWith("{\"evidence\":")).map(line => JSON.parse(line) as Evidence);
    const runtime = records.find(record => record.evidence === "native-recovery-runtime" || record.evidence === "recovery-tree-native-runtime");
    expect(runtime, diagnostics).toBeDefined();
    expect(runtime!.bun).toBe(version.stdout.trim());
    expect(path.isAbsolute(runtime!.executable)).toBe(true);
    return { records, stderr: result.stderr };
  } finally { rmSync(root, { recursive: true, force: true }); }
}

describe.skipIf(!required && (process.platform === "win32" || missing))("real native-Bun recovery ownership", () => {
  it("runs all eight unchanged ownership/escape/no-fork cases in the Bun test host", () => {
    // --timeout changes only the outer runner's default 5s test limit, NOT the
    // helper's 200ms / 5000ms execution deadlines or its 800ms teardown allowance.
    const { records, stderr } = runNative(["test", "./test/audit-recovery-tree.test.ts", "--timeout", "30000"]);
    expect(stderr).toMatch(/8 pass/);
    expect(stderr).toMatch(/0 fail/);
    const observations = records.filter(record => /before-finally$/.test(record.evidence) || record.evidence === "recovery-kernel-no-fork");
    expect(observations).toHaveLength(8);
  });

  it("preserves proof across sequential 200ms/5000ms calls and GC while the second is active", () => {
    const { records } = runNative(["./test/fixtures/bun_recovery_ownership.ts", "sequential-gc"]);
    const observations = records.filter(record => record.evidence === "native-recovery-before-finally");
    expect(observations.map(record => record.timeoutMs)).toEqual([200, 5_000]);
    expect(observations[1]!.gcDuringActive).toBe(true);
    expect(records.find(record => record.evidence === "native-recovery-passed")).toMatchObject({ calls: 2, scenario: "sequential-gc" });
  });

  it("survives eight active-call GC cycles with observed real descriptor-number reuse", () => {
    const { records } = runNative(["./test/fixtures/bun_recovery_ownership.ts", "fd-reuse"]);
    const observations = records.filter(record => record.evidence === "native-recovery-before-finally");
    expect(observations).toHaveLength(9);
    expect(observations.slice(1).every(record => record.gcDuringActive)).toBe(true);
    expect(observations.slice(2).every(record => record.reusedFDs.length > 0)).toBe(true);
    expect(records.find(record => record.evidence === "native-recovery-passed")).toMatchObject({ calls: 9, scenario: "fd-reuse" });
  });
});
