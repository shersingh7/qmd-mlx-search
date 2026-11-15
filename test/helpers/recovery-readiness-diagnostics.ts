// Test-only, opt-in evidence at the existing PID read/parse boundary. No waits,
// stream readers, signals, interpreter launches, or cleanup authority live here.
import { lstatSync } from "node:fs";
import type { RecoveryProcessResult } from "../../scripts/phase5_e2e_runner.js";

type Snapshot = {
  mode: "new-pgid" | "new-session";
  elapsedMs: number;
  guardFired: boolean;
  result: RecoveryProcessResult;
  pidFile: string;
  outputJson: string;
};

function pathFact(file: string) {
  try {
    const stat = lstatSync(file); // Never follow a link or open/read its target.
    return { exists: true, kind: stat.isSymbolicLink() ? "symlink" : stat.isFile() ? "file" : stat.isDirectory() ? "directory" : "other", bytes: stat.size };
  } catch (error) {
    if (["ENOENT", "ENOTDIR"].includes((error as NodeJS.ErrnoException).code ?? "")) return { exists: false };
    throw error;
  }
}

function resultFacts(result: RecoveryProcessResult) {
  const { launch, cleanup, transportFailure } = result;
  const stream = (value: NonNullable<typeof transportFailure>["stdout"]) => ({
    bytes: value.bytes, ended: value.ended, closed: value.closed, destroyed: value.destroyed,
  });
  return {
    pid: result.pid, exitCode: result.exitCode, signal: result.signal, timedOut: result.timedOut,
    launch: { state: launch.state, supervisorPid: launch.supervisorPid, admissionGranted: launch.admissionGranted,
      supervisorReaped: launch.supervisorReaped, noStartProof: launch.noStartProof },
    cleanup: { processGroupId: cleanup.processGroupId, sessionId: cleanup.sessionId, childReaped: cleanup.childReaped,
      groupTerminated: cleanup.groupTerminated, sessionTerminated: cleanup.sessionTerminated, pipesClosed: cleanup.pipesClosed,
      forcedPipeClose: cleanup.forcedPipeClose, containmentVerified: cleanup.containmentVerified, sigkillUsed: cleanup.sigkillUsed,
      processGroups: cleanup.processGroups.slice(0, 16), processGroupsOmitted: Math.max(0, cleanup.processGroups.length - 16) },
    // Preserve output volume, not arbitrary command output, report data, argv or env.
    outputBytes: { stdout: Buffer.byteLength(result.stdout), stderr: Buffer.byteLength(result.stderr), error: Buffer.byteLength(result.error ?? "") },
    transportFailure: transportFailure === undefined ? undefined : {
      prepared: transportFailure.prepared, supervisorExited: transportFailure.supervisorExited,
      supervisorCloseObserved: transportFailure.supervisorCloseObserved,
      stdout: stream(transportFailure.stdout), stderr: stream(transportFailure.stderr), proof: stream(transportFailure.proof),
    },
  };
}

export function readRecoveryFixturePids<T>(readPids: () => T, snapshot: () => Snapshot): T {
  try { return readPids(); }
  catch (primary) {
    try {
      if (process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS === "1") {
        const value = snapshot();
        console.error(JSON.stringify({ evidence: "recovery-closed-pipes-readiness-failure-before-finally",
          mode: value.mode, elapsedMs: value.elapsedMs, guardFired: value.guardFired,
          result: resultFacts(value.result),
          pathFacts: { pidRecord: pathFact(value.pidFile), pidTemporary: pathFact(value.pidFile + ".tmp"), workerReport: pathFact(value.outputJson) } }));
      }
    } catch { /* Best effort only: observation, serialization or logging must not replace primary. */ }
    throw primary;
  }
}
