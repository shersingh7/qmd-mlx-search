const testRuntime = (process.versions as Record<string, string>).bun ? "bun:test" : "vitest";
const { describe, expect, it }: typeof import("vitest") = await import(testRuntime);
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeRecoveryWorker } from "../scripts/phase5_e2e_runner.js";
import { withGatedChildFixture } from "./helpers/recovery-gated-child-fixture.js";

const source = fileURLToPath(new URL("../scripts/phase5_recovery_process.py", import.meta.url));
const fixture = fileURLToPath(new URL("./python/test_recovery_gated_child_ownership.py", import.meta.url));
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") return false; throw error; }
}

describe.skipIf(process.platform === "win32")("gated-child ownership survives one missed discovery snapshot", () => {
  it("acquires a real live child after kernel WNOHANG=0 without extending its execution lease", async () => {
    await withGatedChildFixture(source, fixture, async owned => {
      const { root, kernel, work, sentinel, hooks } = owned;
      const started = performance.now();
      const result = await owned.waitForHelper(executeRecoveryWorker({ dbPath: path.join(root, "unused.sqlite"), docsDir: root,
        outputJson: path.join(root, "absent.json"), stateRoot: path.join(root, "state"), timeoutMs: 200,
        command: ["/usr/bin/python3", "-S", "-c", `open(${JSON.stringify(work)}, 'w').write('escaped')`] }));
      const { supervisorPid, guardFired } = owned;
      const elapsedMs = performance.now() - started;
      // Retain the actual API refusal/transport result even if a required
      // kernel witness is missing; this does not relax any assertion below.
      console.log(JSON.stringify({ evidence: "gated-child-api-return", supervisorPid, result }));
      const records = readFileSync(kernel, "utf8").trim().split("\n").map(line => JSON.parse(line));
      const boundary = records.find(record => record.kind === "boundary");
      const observed = { evidence: "gated-child-api-before-finally", host: { executable: process.execPath,
        node: process.versions.node, bun: (process.versions as Record<string, string>).bun ?? null },
        supervisorPid, supervisorAbsent: supervisorPid !== undefined && !alive(supervisorPid),
        sentinelAlive: alive(sentinel.pid!), workExists: existsSync(work), elapsedMs, guardFired, records, result };
      console.log(JSON.stringify(observed));
      await sleep(1_450);
      console.log(JSON.stringify({ evidence: "gated-child-api-no-late-work-before-finally", supervisorAbsent: !alive(supervisorPid!),
        workerAbsent: !alive(boundary.pid), sentinelAlive: alive(sentinel.pid!), workExists: existsSync(work) }));
      expect(boundary.stage, JSON.stringify(observed)).toBe("fork-return");
      expect(records.filter(record => record.kind === "discovery-omission")).toHaveLength(1);
      expect(records.find(record => record.kind === "owned-live-at-zero")).toMatchObject({ pid: boundary.pid, ownedUnreaped: true });
      expect(records.at(-1)).toEqual({ kind: "fixture-cleanup", recovered: [] });
      expect(result.launch).toMatchObject({ state: "started", admissionGranted: true, supervisorReaped: true });
      expect(result.pid).toBe(boundary.pid);
      expect(result.cleanup).toMatchObject({ childReaped: true, groupTerminated: true, sessionTerminated: true,
        pipesClosed: true, forcedPipeClose: false, containmentVerified: false });
      expect(result.timedOut).toBe(true);
      expect(observed.supervisorAbsent && observed.sentinelAlive && !observed.workExists).toBe(true);
      expect(!alive(boundary.pid) && !existsSync(work) && alive(sentinel.pid!)).toBe(true);
      expect(guardFired).toBe(false);
      expect(elapsedMs).toBeGreaterThanOrEqual(200);
      expect(elapsedMs).toBeLessThan(1_500);
      expect(["exit", "SIGTERM", "SIGINT"].map(event => process.listenerCount(event))).toEqual(hooks);
    });
  });
});
