// Test-only Node observer. Never reads streams, opens borrowed FDs or owns cleanup.
import { channel } from "node:diagnostics_channel";
import { errorMonitor } from "node:events";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { lstatSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, release, version } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const EVENT_LIMIT = 32;
const SUPERVISOR = fileURLToPath(new URL("../../scripts/phase5_recovery_process.py", import.meta.url));
const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const isBun = () => Boolean(process.versions.bun);
/** @param {unknown} value */
const record = value => value !== null && typeof value === "object" ? /** @type {Record<string, unknown>} */ (value) : {};
/** @param {unknown} value */
const number = value => typeof value === "number" && Number.isFinite(value) ? value : null;
/** Fixed scalar allowlist: never serialize child args, payloads, environment or errors. @param {unknown} source @param {string[]} keys */
function fields(source, keys) {
  const input = record(source);
  return Object.fromEntries(keys.filter(key => ["boolean", "number"].includes(typeof input[key])
    || (key === "state" && ["not-attempted", "not-started", "started", "unconfirmed"].includes(String(input[key]))))
    .map(key => [key, input[key]]));
}
/** @param {unknown} source */
function receiptFields(source) {
  const receipt = record(source);
  const failure = record(receipt.transportFailure);
  return {
    launch: fields(receipt.launch, ["state", "supervisorPid", "admissionGranted", "supervisorReaped"]),
    timedOut: receipt.timedOut === true,
    cleanup: fields(receipt.cleanup, ["childReaped", "groupTerminated", "sessionTerminated", "pipesClosed", "forcedPipeClose", "containmentVerified"]),
    // Time is deliberately null: existing production receipt precedes destroy,
    // but this test-only observer has no hook at that exact production line.
    preForcedClosure: receipt.transportFailure === undefined ? null : {
      source: "existing-result.transportFailure-before-destroy", tMs: null,
      transportFailure: {
        ...fields(failure, ["prepared", "supervisorExited", "supervisorCloseObserved"]),
        stdout: fields(failure.stdout, ["bytes", "ended", "closed", "destroyed"]),
        stderr: fields(failure.stderr, ["bytes", "ended", "closed", "destroyed"]),
        proof: fields(failure.proof, ["bytes", "ended", "closed", "destroyed"]),
      },
    },
    protocolEventTimes: "not-observed-no-data-or-write-hooks",
  };
}

/** Observe only a child already owned by the caller. @param {import('node:child_process').ChildProcess} child */
export function observeOwnedChild(child) {
  const started = performance.now();
  const relative = () => Math.round((performance.now() - started) * 1_000) / 1_000;
  /** @type {Array<{event:string,tMs:number,state:ReturnType<typeof snapshotState>,code?:string|null}>} */
  const events = [];
  /** @type {Array<{target:import('node:events').EventEmitter,event:string|symbol,listener:(...args: unknown[])=>void}>} */
  const listeners = [];
  let droppedEvents = 0;
  let disposed = false;
  const streams = { stdout: child.stdout, stderr: child.stderr,
    control: /** @type {import('node:stream').Readable|null} */ (child.stdio[3]) };
  const owner = /** @type {{_closesNeeded?:number,_closesGot?:number,_handle?:unknown}} */ (child);
  function snapshotState() {
    return {
      ownerExited: child.exitCode !== null || child.signalCode !== null,
      nativeProcessHandlePresent: Boolean(owner._handle),
      closeCounter: { needed: number(owner._closesNeeded), got: number(owner._closesGot), scope: "Node-private-read-only-not-kernel-state" },
      streams: Object.fromEntries(Object.entries(streams).map(([name, stream]) => {
        const socket = /** @type {import('node:stream').Readable & {bytesRead?:number,_handle?:{fd?:number}} | null} */ (stream);
        return [name, socket === null ? null : {
          readableEnded: socket.readableEnded, closed: socket.closed, destroyed: socket.destroyed,
          readableLength: socket.readableLength, readableFlowing: socket.readableFlowing,
          socketBytesRead: number(socket.bytesRead), fdAtObservation: number(socket._handle?.fd),
        }];
      })),
    };
  }
  /** @param {string} event @param {unknown} [error] */
  function append(event, error) {
    if (disposed) return;
    if (events.length >= EVENT_LIMIT) { droppedEvents = Math.min(Number.MAX_SAFE_INTEGER, droppedEvents + 1); return; }
    const code = record(error).code;
    events.push({ event, tMs: relative(), state: snapshotState(),
      ...(event.endsWith(".error") ? { code: typeof code === "string" && /^[A-Z][A-Z0-9_]{0,63}$/.test(code) ? code : null } : {}) });
  }
  /** @param {import('node:events').EventEmitter} target @param {string|symbol} event @param {string} label */
  function listen(target, event, label) {
    /** @param {...unknown} args */
    const listener = (...args) => append(label, event === errorMonitor ? args[0] : undefined);
    target.on(event, listener);
    listeners.push({ target, event, listener });
  }
  listen(child, "spawn", "owner.spawn");
  listen(child, "exit", "owner.exit");
  listen(child, "close", "owner.close");
  listen(child, errorMonitor, "owner.error");
  for (const [name, stream] of Object.entries(streams)) if (stream) {
    listen(stream, "end", `${name}.end`);
    listen(stream, "close", `${name}.close`);
    listen(stream, errorMonitor, `${name}.error`);
  }
  append("spawn-returned");
  function dispose() {
    if (disposed) return;
    for (const { target, event, listener } of listeners) target.removeListener(event, listener);
    disposed = true;
  }
  /** @param {unknown} [receipt] */
  function snapshot(receipt) {
    return { schemaVersion: 1, scope: "node-owned-supervisor-runtime-events", runtime: { node: process.versions.node, bun: process.versions.bun ?? null },
      ownerPid: child.pid ?? null, clock: "observer-local-relative-performance-now-ms", eventLimit: EVENT_LIMIT,
      causality: "not-established-by-runtime-events", nativeBunOwnerDiagnostics: "not-implemented",
      events: events.slice(), droppedEvents, finalStateAtMs: relative(), finalState: snapshotState(), receipt: receiptFields(receipt) };
  }
  /** @param {unknown} [receipt] */
  function finish(receipt) {
    try { return { ...snapshot(receipt), restoration: { listenersRemoved: true } }; }
    finally { dispose(); }
  }
  return { snapshot, finish, dispose };
}

/** Opt-in and lexical launch scope; never a global spawn wrapper. @param {{enabled?:boolean,directory?:string}} [options] */
export function beginRecoveryBootstrapDiagnostics(options = {}) {
  const enabled = options.enabled ?? process.env.QMD_RECOVERY_BOOTSTRAP_DIAGNOSTICS === "1";
  const directory = options.directory ?? process.env.QMD_RECOVERY_BOOTSTRAP_DIR;
  if (!enabled) return { attachAfterLaunch() {}, finish(/** @type {unknown} */ _receipt) {}, dispose() {} };
  const dc = channel("child_process");
  /** @type {import('node:child_process').ChildProcess[]} */
  const candidates = [];
  /** @type {ReturnType<typeof observeOwnedChild>|undefined} */
  let trace;
  let active = false;
  let attached = false;
  let finished = false;
  let status = isBun() ? "native-bun-owner-diagnostics-not-implemented" : "no-matching-node-supervisor-observed";
  /** Constructor notification has no args/streams yet. @param {unknown} message */
  const subscriber = message => {
    const candidate = record(message).process;
    if (candidates.length < 4 && candidate && typeof record(candidate).on === "function") {
      candidates.push(/** @type {import('node:child_process').ChildProcess} */ (candidate));
    }
  };
  if (!isBun()) { dc.subscribe(subscriber); active = true; }
  function unsubscribe() { if (active) { dc.unsubscribe(subscriber); active = false; } }
  function attachAfterLaunch() {
    if (attached || finished) return;
    attached = true;
    try {
      const child = candidates.find(child => child.spawnfile === "/usr/bin/python3"
        && child.spawnargs[1] === "-S" && child.spawnargs[2] === "-u" && child.spawnargs[3] === SUPERVISOR);
      if (child) { trace = observeOwnedChild(child); status = "observed-node-supervisor"; }
    } finally { candidates.length = 0; unsubscribe(); }
  }
  function dispose() { unsubscribe(); candidates.length = 0; trace?.dispose(); }
  /** @param {unknown} [receipt] */
  function finish(receipt) {
    if (finished) return;
    finished = true;
    try {
      const observation = trace?.finish(receipt);
      const report = { ...(observation ?? { schemaVersion: 1, nativeBunOwnerDiagnostics: "not-implemented" }), status,
        invocation:"existing-fast-timeout-case-in-place; other canonical files may run concurrently; no global-first-supervisor claim",
        restoration: { channelUnsubscribed: true, listenersRemoved: true } };
      dispose();
      if (!directory || !path.isAbsolute(directory)) throw Object.assign(new Error("explicit artifact directory required"), { code: "EINVAL" });
      mkdirSync(directory, { recursive: true });
      writeFileSync(path.join(directory, `fast-timeout-${isBun() ? "bun" : "node"}-${process.pid}.json`), JSON.stringify(report) + "\n", { flag: "wx", mode: 0o600 });
    } catch (error) {
      // Do not replace the canonical assertion/error with an instrumentation IO error.
      const code = record(error).code;
      console.error(JSON.stringify({ evidence: "recovery-bootstrap-diagnostic-write-failed", code: typeof code === "string" && /^[A-Z][A-Z0-9_]{0,63}$/.test(code) ? code : null }));
    } finally { dispose(); }
  }
  return { attachAfterLaunch, finish, dispose };
}

/** Read-only checkout identity; never start Apple's git/CLT launcher. @param {string} root */
export function readCheckoutHead(root) {
  try {
    let gitDir = path.join(root, ".git");
    if (lstatSync(gitDir).isFile()) {
      const pointer = readFileSync(gitDir, "utf8").trim();
      if (!pointer.startsWith("gitdir: ")) return null;
      gitDir = path.resolve(root, pointer.slice(8));
    }
    const head = readFileSync(path.join(gitDir, "HEAD"), "utf8").trim();
    if (/^[0-9a-f]{40}$/.test(head)) return head;
    const ref = head.startsWith("ref: ") ? head.slice(5) : "";
    if (!/^refs\/[A-Za-z0-9_./-]+$/.test(ref) || ref.split("/").some(part => !part || part === "." || part === "..")) return null;
    try {
      const sha = readFileSync(path.join(gitDir, ref), "utf8").trim();
      if (/^[0-9a-f]{40}$/.test(sha)) return sha;
    } catch { /* packed ref below */ }
    const packed = readFileSync(path.join(gitDir, "packed-refs"), "utf8");
    return packed.split("\n").map(line => line.split(" ")).find(parts => parts[1] === ref && /^[0-9a-f]{40}$/.test(parts[0] ?? ""))?.[0] ?? null;
  } catch { return null; }
}

/** No Python, xcrun, xcode-select, network, event text or ambient environment dump. @param {string} directory */
export function writeStaticMetadata(directory) {
  if (!path.isAbsolute(directory)) throw new Error("absolute artifact directory required");
  mkdirSync(directory, { recursive: true });
  /** @type {Record<string,string>} */
  const env = {};
  for (const [key, folder] of Object.entries({ HOME:"home", HERMES_HOME:"hermes", HERMES_DIR:"hermes", HERMES_KNOWLEDGE_DIR:"knowledge", KNOWLEDGE_DIR:"knowledge", XDG_CONFIG_HOME:"config",
    XDG_CACHE_HOME:"cache", XDG_DATA_HOME:"data", HF_HOME:"hf", HF_HUB_CACHE:"hf/hub", TRANSFORMERS_CACHE:"hf/transformers", TMPDIR:"tmp", TMP:"tmp", TEMP:"tmp", BUN_INSTALL_CACHE_DIR:"bun" })) {
    env[key] = path.join(directory, "metadata-environment", folder); mkdirSync(env[key], { recursive:true });
  }
  Object.assign(env, { PATH: process.env.PATH ?? "/usr/bin:/bin", CI:"1", REQUIRE_BUN:"1", QMD_LLAMA_GPU:"false", HF_HUB_OFFLINE:"1", TRANSFORMERS_OFFLINE:"1", HF_DATASETS_OFFLINE:"1",
    GIT_CONFIG_NOSYSTEM:"1", GIT_CONFIG_GLOBAL:"/dev/null", GIT_OPTIONAL_LOCKS:"0", GIT_TERMINAL_PROMPT:"0" });
  /** @param {string} executable @param {string[]} args */
  const query = (executable, args) => { try { return execFileSync(executable, args, { cwd:ROOT, env, timeout:2_000, maxBuffer:4096, stdio:["ignore", "pipe", "pipe"], encoding:"utf8" }).trim(); } catch { return null; } };
  /** @param {string} name */
  const value = name => process.env[name]?.slice(0, 256) ?? null;
  const hashes = Object.fromEntries(["scripts/phase5_e2e_runner.ts", "scripts/phase5_recovery_process.py", "scripts/phase5_recovery_process_loader.mjs", "scripts/phase5_recovery_worker.ts", "scripts/qmd_mlx/process_anchor.py",
    "test/audit-recovery-tree.test.ts", "test/helpers/recovery-bootstrap-diagnostics.mjs", "test/helpers/recovery-bootstrap-diagnostics.selftest.mjs", ".github/workflows/ci.yml", "package.json", "bun.lock"].map(name => [name, createHash("sha256").update(readFileSync(path.join(ROOT, name))).digest("hex")]));
  const report = { schemaVersion:1, scope:"static-runner-metadata-no-added-interpreter-launch", freshHomeIsNotColdOS:true,
    execution:process.env.GITHUB_ACTIONS === "true" ? "github-actions-before-canonical-suite" : "local-metadata-command-not-CI-proof",
    git: { sourceHead:value("QMD_DIAGNOSTIC_HEAD_SHA"), eventSha:value("GITHUB_SHA"), actualCheckout:readCheckoutHead(ROOT), checkoutSource:"read-only .git/HEAD/ref; no git/CLT launcher" },
    runner: { os:value("RUNNER_OS"), arch:value("RUNNER_ARCH"), imageOS:value("ImageOS"), imageVersion:value("ImageVersion"), runId:value("GITHUB_RUN_ID"), runAttempt:value("GITHUB_RUN_ATTEMPT"), job:value("GITHUB_JOB") },
    host: { platform:process.platform, arch:process.arch, kernel:release(), osVersion:version(), swVers:process.platform === "darwin" ? query("/usr/bin/sw_vers", []) : null, cpuModel:cpus()[0]?.model ?? null },
    runtime: { metadataNode:process.versions.node, executable:process.execPath, uv:process.versions.uv, packageLauncher:value("QMD_DIAGNOSTIC_MATRIX_RUNTIME"), expectedMainSuiteHost:"Node (package.json test explicitly invokes node); actual observed host is in fast-timeout artifact" },
    systemPython: { requested:"/usr/bin/python3", launcherIsRegularFile:lstatSync("/usr/bin/python3").isFile(), firstSelectedInterpreterVersion:"not-observed", versionQuery:"not-run; no Python/xcrun/xcode-select diagnostic warmup" },
    sourceSha256:hashes };
  writeFileSync(path.join(directory, "runner-metadata.json"), JSON.stringify(report, null, 2)+"\n", { flag:"wx", mode:0o600 });
  return report;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] !== "--metadata" || !process.argv[3]) throw new Error("usage: node recovery-bootstrap-diagnostics.mjs --metadata ABSOLUTE_DIRECTORY");
  writeStaticMetadata(process.argv[3]);
}
