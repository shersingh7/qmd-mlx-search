// Test-only post-failure preservation. Importing this helper does no capture IO,
// launches nothing and installs no listeners, protocol observers or fs patches.
import fs from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

/** Each file retains at most 128 KiB, with one additional detection byte.
 * ALL descriptor reads (including detection bytes) share a 768 KiB budget.
 * Existing in-memory stdout/stderr each retain at most 8 KiB; metadata is fixed
 * or scalar-bounded. Serialized output is refused above 2 MiB, never sliced JSON.
 * Hashes describe retained bytes; only stable, EOF-observed, untruncated files
 * receive fullFileSha256. These observations are not an atomic FS snapshot.
 */
export const EXECUTION_FAILURE_LIMITS = Object.freeze({
  fileBytes: 128 * 1024, totalReadBytes: 768 * 1024, receiptBytes: 2 * 1024 * 1024, textBytes: 8 * 1024,
  metadataCharacters: 4096, versionCharacters: 256, argumentCount: 8, githubCharacters: 128, signalCharacters: 64,
});
export type FailureCaptureIO = Pick<typeof fs, "lstatSync" | "openSync" | "fstatSync" | "readSync" | "closeSync" | "mkdirSync" | "writeSync" | "linkSync" | "unlinkSync">;
// Call-local fault seams for filesystem-only tests; no global patch is needed.
export type FailureCaptureHooks = { io?: FailureCaptureIO; serialize?: (value: unknown) => string; nonce?: () => string };
export type RecoveryExecutionFailureContext = {
  root: string; runtime: "node" | "bun"; mode: "valid" | "alive" | "graceful" | "hard";
  stage: string; pause: "delay" | "stop"; probe?: string;
  host: { pid?: number; exitCode: number | null; signalCode: string | null };
  sentinelPid?: number; executable: string; args: string[]; fixtureElapsedMs: number; stdout: string; stderr: string;
  supervisorPython?: string;
};
export type FailureCaptureResult = { status: "disabled" | "no-destination" | "noncanonical" | "captured" | "failed"; path?: string; code?: string | null; temporaryCleanup?: string; descriptorCloseErrorCode?: string | null };

type Identity = { dev: string; ino: string; size: string; mode: number; nlink: string; mtimeNs: string | null; ctimeNs: string | null };
type Content = { utf8: string; json: string };
type Evidence = {
  status: string; reason?: string; errorCode?: string | null; closeErrorCode?: string | null;
  before?: Identity; opened?: Identity; after?: Identity; pathnameAfter?: Identity;
  capturedBytes: number; readBytes: number; rawBase64: string | null; truncated: boolean; eofObserved: boolean;
  stability: string; content: Content; fullFileSha256: string | null;
  capturedSha256: { algorithm: "sha256"; scope: "captured-bytes"; value: string } | null;
};
type Budget = { remaining: number; readBytes: number };
const FILES = ["started.json", "helper.json", "stage.json", "forks.jsonl", "reaps.jsonl", "gate-forwarded.json", "validation.json", "queued-eof.json", "control-reads.jsonl", "probe.json", "execution-boundary.py", "host.ts"] as const;
const SOURCES = ["scripts/phase5_recovery_process.py", "test/audit-recovery-execution-admission.test.ts", "scripts/phase5_e2e_runner.ts", "scripts/phase5_recovery_process_loader.mjs", "scripts/qmd_mlx/process_anchor.py", "test/helpers/recovery-execution-failure.ts"] as const;
const PUBLIC_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const errorCode = (error: unknown): string | null => {
  try { const code = (error as NodeJS.ErrnoException)?.code; return typeof code === "string" && /^[A-Z][A-Z0-9_]{0,63}$/.test(code) ? code : null; }
  catch { return null; }
};
// BigInt stats keep inode/size identity exact and expose nanosecond timestamps.
// Native Bun's number Stats round to milliseconds and miss real same-ms writes.
const identity = (stat: fs.BigIntStats): Identity => ({ dev: stat.dev.toString(), ino: stat.ino.toString(), size: stat.size.toString(), mode: Number(stat.mode), nlink: stat.nlink.toString(),
  mtimeNs: typeof stat.mtimeNs === "bigint" ? stat.mtimeNs.toString() : null, ctimeNs: typeof stat.ctimeNs === "bigint" ? stat.ctimeNs.toString() : null });
const sameInode = (a: Identity, b: Identity) => a.dev === b.dev && a.ino === b.ino && (a.mode & fs.constants.S_IFMT) === (b.mode & fs.constants.S_IFMT);
const unchanged = (a: Identity, b: Identity) => sameInode(a, b) && a.mode === b.mode && a.size === b.size && a.mtimeNs !== null && a.ctimeNs !== null && a.mtimeNs === b.mtimeNs && a.ctimeNs === b.ctimeNs && a.nlink === b.nlink;
class UnsafePath extends Error { constructor(readonly reason: string) { super(reason); } }
function absolute(value: string): boolean {
  return typeof value === "string" && value.length <= 4096 && !value.includes("\0") && path.isAbsolute(value) && path.normalize(value) === value;
}
// Refuse existing symlink/non-directory ancestors too. No openat/dirfd API is
// available here: hostile concurrent ancestor replacement is not atomically
// excluded. Descriptor and post-read pathname identities are checked separately.
function directories(directory: string, io: FailureCaptureIO, create: boolean) {
  let current = path.parse(directory).root;
  for (const part of path.relative(current, directory).split(path.sep).filter(Boolean)) {
    current = path.join(current, part);
    let stat: fs.Stats;
    try { stat = io.lstatSync(current); }
    catch (error) {
      if (!create || errorCode(error) !== "ENOENT") throw error;
      try { io.mkdirSync(current, { mode: 0o700 }); } catch (e) { if (errorCode(e) !== "EEXIST") throw e; }
      stat = io.lstatSync(current);
    }
    if (stat.isSymbolicLink()) throw new UnsafePath("symlink-ancestor");
    if (!stat.isDirectory()) throw new UnsafePath("non-directory-ancestor");
  }
}
function emptyEvidence(): Evidence {
  return { status: "read-error", capturedBytes: 0, readBytes: 0, rawBase64: null, truncated: false, eofObserved: false, stability: "not-observed",
    content: { utf8: "not-evaluated", json: "unavailable" }, fullFileSha256: null, capturedSha256: null };
}
function content(bytes: Buffer, kind: "json" | "jsonl" | "text", complete: boolean): Content {
  if (!complete) return { utf8: "not-evaluated", json: "not-evaluated-incomplete" };
  let text: string;
  try { text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes); }
  catch { return { utf8: "invalid", json: kind === "text" ? "not-applicable" : "invalid-utf8" }; }
  if (kind === "text") return { utf8: "valid", json: "not-applicable" };
  if (bytes.length === 0) return { utf8: "valid", json: "empty" };
  try {
    if (kind === "jsonl") for (const line of text.split("\n")) { if (!/^[ \t\r]*$/.test(line)) JSON.parse(line); }
    else JSON.parse(text);
    return { utf8: "valid", json: kind === "jsonl" ? "valid-jsonl" : "valid-json" };
  } catch { return { utf8: "valid", json: "invalid-json" }; }
}
function readEvidence(file: string, kind: "json" | "jsonl" | "text", budget: Budget, io: FailureCaptureIO): Evidence {
  const result = emptyEvidence();
  let fd: number | undefined, bytes = Buffer.alloc(0), eof = false;
  try {
    directories(path.dirname(file), io, false);
    const before = io.lstatSync(file, { bigint: true }); result.before = identity(before);
    if (!before.isFile() || before.isSymbolicLink()) { result.status = "unsupported-object"; result.reason = before.isSymbolicLink() ? "symlink" : "nonregular"; return result; }
    if (budget.remaining === 0) { result.status = "budget-exhausted"; result.truncated = before.size > 0n; return result; }
    if (!fs.constants.O_NOFOLLOW || !fs.constants.O_NONBLOCK) { result.status = "unsupported-object"; result.reason = "safe-open-flags-unavailable"; return result; }
    fd = io.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
    const opened = io.fstatSync(fd, { bigint: true }); result.opened = identity(opened);
    if (!opened.isFile() || !sameInode(result.before, result.opened)) {
      result.status = "unstable"; result.reason = "descriptor-identity-changed"; result.stability = "changed-at-observations"; return result;
    }
    const buffer = Buffer.alloc(Math.min(EXECUTION_FAILURE_LIMITS.fileBytes + 1, budget.remaining));
    try {
      while (result.readBytes < buffer.length) {
        const n = io.readSync(fd, buffer, result.readBytes, buffer.length - result.readBytes, result.readBytes);
        if (n === 0) { eof = true; break; }
        result.readBytes += n; budget.readBytes += n; budget.remaining -= n;
      }
    } finally { bytes = buffer.subarray(0, Math.min(result.readBytes, EXECUTION_FAILURE_LIMITS.fileBytes)); }
    result.after = identity(io.fstatSync(fd, { bigint: true }));
    result.pathnameAfter = identity(io.lstatSync(file, { bigint: true }));
    const stable = unchanged(result.before, result.opened) && unchanged(result.opened, result.after) && unchanged(result.after, result.pathnameAfter);
    result.status = stable ? "captured" : "unstable";
    result.stability = stable ? "stable-at-observations" : "changed-at-observations";
    if (!stable) {
      const noPrecision = [result.before, result.opened, result.after, result.pathnameAfter].some(s => s.mtimeNs === null || s.ctimeNs === null);
      result.reason = noPrecision ? "timestamp-precision-unavailable" : "metadata-or-pathname-changed";
      if (noPrecision) result.stability = "unknown-timestamp-precision";
    }
  } catch (error) {
    if (error instanceof UnsafePath) { result.status = "unsupported-object"; result.reason = error.reason; }
    else if (errorCode(error) === "ENOENT" && !result.before) result.status = "absent";
    else if (["ENOENT", "ELOOP"].includes(errorCode(error) ?? "") && result.before) {
      result.status = "unstable"; result.reason = fd === undefined ? "pathname-changed-before-open" : "pathname-removed-after-read";
      result.errorCode = errorCode(error); result.stability = "changed-at-observations";
    }
    else { result.status = "read-error"; result.errorCode = errorCode(error); }
  } finally {
    if (fd !== undefined) {
      try { io.closeSync(fd); }
      catch (error) { result.status = "read-error"; result.closeErrorCode = errorCode(error); }
    }
  }
  result.capturedBytes = bytes.length; result.eofObserved = eof;
  result.truncated = result.readBytes > bytes.length || BigInt(result.before?.size ?? "0") > BigInt(bytes.length) || BigInt(result.after?.size ?? "0") > BigInt(bytes.length);
  if (result.opened) { result.rawBase64 = bytes.toString("base64"); result.capturedSha256 = { algorithm: "sha256", scope: "captured-bytes", value: hash(bytes) }; }
  const complete = result.status === "captured" && eof && !result.truncated && result.after?.size === String(bytes.length);
  result.content = content(bytes, kind, complete);
  if (complete) result.fullFileSha256 = hash(bytes);
  return result;
}
function canonical(context: RecoveryExecutionFailureContext): string | null {
  if (context.probe !== undefined || !["node", "bun"].includes(context.runtime) || !["alive", "graceful", "hard"].includes(context.mode)) return null;
  if (context.pause === "stop") return context.mode === "hard" && context.stage === "supervisor-consumed" ? `cleans real ${context.runtime} SIGKILL with its supervisor SIGSTOPped` : null;
  return context.pause === "delay" && ["supervisor-consumed", "supervisor-forwarded", "child-admitted"].includes(context.stage)
    ? `cleans ${context.runtime} ${context.mode} after ${context.stage} without late work` : null;
}
function callerObserved(started: Evidence): Record<string, unknown> {
  if (started.fullFileSha256 === null || started.content.json !== "valid-json" || started.rawBase64 === null) return { status: "unknown", source: "started.json", fileStatus: started.status, content: started.content };
  try {
    const value = JSON.parse(Buffer.from(started.rawBase64, "base64").toString("utf8"));
    if (value === null || typeof value !== "object" || Array.isArray(value) || typeof value.executable !== "string" || value.executable.length > 4096
      || value.versions === null || typeof value.versions !== "object" || Array.isArray(value.versions)) throw new Error("unrecognized started object");
    const entries = Object.entries(value.versions);
    if (entries.length > 64 || entries.some(([k, v]) => !/^[a-z0-9_]{1,64}$/.test(k) || typeof v !== "string" || v.length > 256)) throw new Error("unrecognized versions");
    return { status: "observed-existing-started-json", source: "started.json", executable: value.executable, versions: Object.fromEntries(entries) };
  } catch { return { status: "unknown", source: "started.json", reason: "unrecognized-started-json-object" }; }
}
function textEvidence(value: string) {
  // B code units cover at least B UTF-8 bytes unless the string ends first.
  // Do not sever a real surrogate pair before encoding: include its low half.
  // This encodes at most B+1 code units (at most 3*(B+1) bytes), then keeps
  // the exact byte prefix, even when its final UTF-8 sequence is incomplete.
  let end = Math.min(value.length, EXECUTION_FAILURE_LIMITS.textBytes);
  const last = value.charCodeAt(end - 1), next = value.charCodeAt(end);
  if (last >= 0xd800 && last <= 0xdbff && next >= 0xdc00 && next <= 0xdfff) end++;
  const bytes = Buffer.from(value.slice(0, end), "utf8").subarray(0, EXECUTION_FAILURE_LIMITS.textBytes);
  // Count the same UTF-8 encoding without a second full-stream allocation.
  // Unpaired surrogate units encode as U+FFFD (3 bytes); valid pairs use 4.
  // Do not rely on Buffer.byteLength, which can undercount unpaired units.
  let length = 0;
  for (let i = 0; i < value.length; i++) {
    const unit = value.charCodeAt(i);
    if (unit < 0x80) length++;
    else if (unit < 0x800) length += 2;
    else {
      const next = value.charCodeAt(i + 1);
      if (unit >= 0xd800 && unit <= 0xdbff && next >= 0xdc00 && next <= 0xdfff) { length += 4; i++; }
      else length += 3;
    }
  }
  return { source: "existing-JS-string-encoded-as-UTF8-not-original-stream-bytes", byteLength: length, capturedBytes: bytes.length, rawBase64: bytes.toString("base64"), truncated: length > bytes.length,
    capturedSha256: { algorithm: "sha256", scope: "captured-bytes", value: hash(bytes) } };
}
function github() {
  const values: Record<string, string | null> = {}, status: Record<string, { state: string; originalCharacters: number | null }> = {};
  for (const key of ["GITHUB_RUN_ID", "GITHUB_JOB", "GITHUB_RUN_ATTEMPT"]) {
    const value = process.env[key], valid = typeof value === "string" && /^[A-Za-z0-9_.-]{1,128}$/.test(value);
    values[key] = valid ? value! : null;
    status[key] = { state: value === undefined ? "absent" : valid ? "retained" : "omitted-invalid-or-over-limit", originalCharacters: value?.length ?? null };
  }
  return { values, status };
}

/** Called only at the start of hostCase's failure snapshot, before legacy reads
 * and finally cleanup. Never throws or logs, even for IO/serialization failures;
 * observeStageFailure remains the sole owner of primary error/logging semantics.
 * Success/disabled/no-destination/control paths perform no capture filesystem IO.
 */
export function captureRecoveryExecutionFailure(context: RecoveryExecutionFailureContext, hooks: FailureCaptureHooks = {}): FailureCaptureResult {
  let io: FailureCaptureIO | undefined, temp: string | undefined, owned: Identity | undefined, fd: number | undefined;
  const result: FailureCaptureResult = { status: "failed" };
  try {
    if (process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS !== "1") return { status: "disabled" };
    const directory = process.env.QMD_RECOVERY_BOOTSTRAP_DIR;
    if (!directory) return { status: "no-destination" };
    const title = canonical(context); if (title === null) return { status: "noncanonical" };
    if (!absolute(directory) || !absolute(context.root)) return { status: "failed", code: "INVALID_ABSOLUTE_PATH" };
    if (context.supervisorPython !== undefined && (typeof context.supervisorPython !== "string" || context.supervisorPython.length > 4096
      || context.supervisorPython.includes("\0") || !path.isAbsolute(context.supervisorPython))) return { status: "failed", code: "INVALID_ABSOLUTE_PATH" };
    io = hooks.io ?? fs;
    const budget: Budget = { remaining: EXECUTION_FAILURE_LIMITS.totalReadBytes, readBytes: 0 };
    const files = Object.fromEntries(FILES.map(name => [name, readEvidence(path.join(context.root, name), name.endsWith(".jsonl") ? "jsonl" : name.endsWith(".json") ? "json" : "text", budget, io!)]));
    const sources = Object.fromEntries(SOURCES.map(name => [name, readEvidence(path.join(PUBLIC_ROOT, name), "text", budget, io!)]));
    const githubIdentity = github();
    const versionKeys = ["node", "bun", "uv", "v8"];
    const report = {
      schemaVersion: 1, scope: "canonical-missing-stage-existing-evidence-only", fixtureCleanupStarted: false,
      case: { classification: "canonical", title, runtimeRequested: context.runtime, mode: context.mode, stage: context.stage, pause: context.pause },
      parentTestHost: { runtime: process.versions.bun ? "bun" : "node", executable: process.execPath.slice(0, 4096), executableTruncated: process.execPath.length > 4096, pid: process.pid, ppid: process.ppid,
        platform: process.platform, arch: process.arch, versions: Object.fromEntries(versionKeys.map(key => [key, process.versions[key]?.slice(0, 256) ?? null])),
        versionsTruncated: versionKeys.some(key => (process.versions[key]?.length ?? 0) > 256) },
      callerObserved: callerObserved(files["started.json"]!),
      pythonIdentity: { status: "unknown-not-observed", requestedExecutable: context.supervisorPython ?? "/usr/bin/python3", selectedExecutable: null, version: null },
      fixture: { root: context.root, host: { pid: context.host.pid ?? null, exitCode: context.host.exitCode, signalCode: context.host.signalCode?.slice(0, 64) ?? null,
          signalCodeTruncated: (context.host.signalCode?.length ?? 0) > 64 },
        sentinelPid: context.sentinelPid ?? null, elapsedMs: context.fixtureElapsedMs, elapsedClock: "parent-test-host-relative-performance.now-ms-from-existing-stage-wait-start", requestedExecutable: context.executable.slice(0, 4096), requestedExecutableTruncated: context.executable.length > 4096,
        requestedArgs: context.args.slice(0, 8).map(arg => arg.slice(0, 4096)), requestedArgsTruncated: context.args.length > 8 || context.args.some(arg => arg.length > 4096),
        stdout: textEvidence(context.stdout), stderr: textEvidence(context.stderr), processObservation: "existing ChildProcess fields only; no new signals or kernel probes" },
      github: githubIdentity.values, githubStatus: githubIdentity.status, limits: EXECUTION_FAILURE_LIMITS,
      statEncoding: "dev/ino/size/nlink are exact decimal integer strings; mode is an integer; mtimeNs/ctimeNs are exact decimal nanoseconds or explicitly null",
      budget: { readBytes: budget.readBytes, remainingReadBytes: budget.remaining, exhausted: budget.remaining === 0 }, files, sources,
      interpretation: "existing failure-time files only; parse errors do not prove partial publication; missing auxiliary files do not prove no fork/admission; no parent-exit, cleanup or root-cause qualification",
      limitations: ["not an atomic filesystem snapshot; stable-at-observations is not atomic immutability", "no dirfd-relative reads/publication: hostile concurrent ancestor/temp pathname replacement is not atomically excluded",
        "full-file hashes are failure-time observations, not independently attested executed bytes", "selected Python identity was not observed; later controls cannot fill it", "no power-loss durability or complete-cleanup certification"],
    };
    const serialized = (hooks.serialize ?? JSON.stringify)(report);
    if (typeof serialized !== "string" || Buffer.byteLength(serialized) + 1 > EXECUTION_FAILURE_LIMITS.receiptBytes) return { status: "failed", code: "RECEIPT_SIZE_LIMIT" };
    const output = Buffer.from(serialized + "\n");
    directories(directory, io, true);
    const nonce = (hooks.nonce ?? randomUUID)();
    if (!/^[A-Za-z0-9-]{1,96}$/.test(nonce)) return { status: "failed", code: "INVALID_NONCE" };
    temp = path.join(directory, `.recovery-execution-failure-${process.pid}-${nonce}.tmp`);
    fd = io.openSync(temp, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW, 0o600);
    owned = identity(io.fstatSync(fd, { bigint: true }));
    let offset = 0;
    while (offset < output.length) {
      const n = io.writeSync(fd, output, offset, output.length - offset);
      if (n <= 0) throw Object.assign(new Error("no write progress"), { code: "EIO" });
      offset += n;
    }
    const closing = fd; fd = undefined;
    try { io.closeSync(closing); } catch (error) { result.descriptorCloseErrorCode = errorCode(error); throw error; }
    if (!sameInode(owned, identity(io.lstatSync(temp, { bigint: true })))) throw Object.assign(new Error("temp changed"), { code: "ESTALE" });
    const final = path.join(directory, `recovery-execution-failure-${context.runtime}-${context.mode}-${context.stage}-${context.pause}-${process.pid}-${nonce}.json`);
    // Hard-link publication is atomic visibility + no-replace, unlike rename(),
    // which overwrites an existing receipt. The complete file is already closed.
    io.linkSync(temp, final);
    if (!sameInode(owned, identity(io.lstatSync(final, { bigint: true })))) throw Object.assign(new Error("published identity changed"), { code: "ESTALE" });
    result.status = "captured"; result.path = final;
  } catch (error) { result.code = errorCode(error); }
  finally {
    if (fd !== undefined && io) { try { io.closeSync(fd); } catch (error) { result.descriptorCloseErrorCode = errorCode(error); } }
    if (temp && owned && io) {
      try {
        if (sameInode(owned, identity(io.lstatSync(temp, { bigint: true })))) { io.unlinkSync(temp); result.temporaryCleanup = "removed-owned-temp"; }
        else result.temporaryCleanup = "different-inode-retained";
      } catch (error) { result.temporaryCleanup = errorCode(error) === "ENOENT" ? "already-absent" : "cleanup-error-owned-temp-may-remain"; }
    } else if (temp && fd !== undefined) result.temporaryCleanup = "unknown-identity-retained";
  }
  return result;
}
