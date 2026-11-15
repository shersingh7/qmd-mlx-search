# macOS bootstrap/open-stream diagnosis — unresolved, no production repair

## Outcome

The exact CI failure is verified, but its causal source is still **unproven**. The original helper correctly returned an **unconfirmed** failure after forced stream closure; it must not be relabelled as no-start or clean cancellation. The current local host did not reproduce the unexplained CI stream hold. Local observations instead show same-PID launcher exec and natural EOF, with working positive controls for kernel fork detection and retained-writer pipe detection.

All work is under `/Users/shersingh/.hermes/cache/scratch/qmd-mac-bootstrap-C4GaOR`. No mutable recovery source was imported, tested, frozen, or edited. The only mutable repository file read was the expressly permitted parent-owned context plan. No installed dependencies, services, configurations, skills, index/model state, or Git refs/worktree files were changed. No installs, registry/model downloads, GPU execution, CI dispatch/reruns, commits, pushes or merge were performed.

## Authenticated CI identity (important corrections)

- Repository: `shersingh7/qmd-mlx-search`.
- Run: https://github.com/shersingh7/qmd-mlx-search/actions/runs/43503677544, attempt 1, `pull_request`, failed.
- API `head_sha`: `dcf98c159b671b4d08c3759215c4371502f49fc0`.
- Actual checkout in both logs: merge commit `99d061d2d373f8affd62210f74bf0b6d5ce0c898`, merging the exact head into `1d6cfef0353810eea55d3fec879d9c05ef5ebc4c`. Inspected frozen source/workflow/fixtures have byte-identical SHA-256 at the head and merge checkout.
- **132155679910 is Node (macos-15)**, runner `GitHub Actions 1000000233`, runner ID `1000000233`, Node test step 12 failed.
- **132155679594 is Bun (macos-15)**, runner `GitHub Actions 1000000228`, runner ID `1000000228`, package-script Vitest step 13 failed. The initial task-context job labels were reversed.
- Both logs: macOS **15.8.2 / 24H232**, image **macos-15-arm64**, image version **20261104.0514.1**, Node **22.22.3 arm64**, Bun **1.3.8+b64edcb49**. Actual CI CPU model is not recorded; do not equate it to this M2 host.
- Setup-python installed CPython **3.12.10**, but the recovery TS code launches **`/usr/bin/python3` explicitly**, not the setup-python PATH interpreter. The selected system interpreter's resolved executable/version was not recorded in the failed CI case.
- Both main Vitest suites actually run **Node**: frozen `package.json:34` explicitly says `node node_modules/vitest/vitest.mjs ...`; Bun log line 266 prints that command. Both integrated receipts have `mcpHttpRuntimeSupported:true`, agreeing with the non-Bun branch. Native Bun recovery probes run separately later and their fast-timeout controls pass. Do not attribute the missing-executable failures to Bun's native reader implementation.
- Fresh job-log downloads are byte-identical to the retained originals. Annotations are a test-step exit-code failure, action-runtime deprecation warning, and a macOS arm64 **queue-capacity notice**. Both jobs were assigned and executed; that notice is not evidence that the stream failure was an unassigned-runner infrastructure failure.

Primary identity artifacts: `evidence/collection-summary.json`, `evidence/additional-source-identity.json`, `evidence/run.json`, `evidence/jobs.json`, original log copies and annotations.

## Exact failing assertion and what the receipt proves

Node log **727** (`2026-11-13T20:40:58.2167420Z`), `test/audit-recovery-tree.test.ts` fast case:

- execution budget 200 ms; elapsed **1007 ms**; 1800 ms failure guard did not fire;
- supervisor absent and runtime-reaped; unrelated sentinel still alive;
- no admission granted, no worker PID, empty stdout/stderr;
- `launch.state="unconfirmed"`, `supervisorReaped=true`, `forcedPipeClose=true`;
- `childReaped=false`, `groupTerminated=false`, `sessionTerminated=false`, empty processGroups, no containment certification;
- finally receipt: no worker fixture kills.

The precise failure is line **75** of the test: its two accepted ownership outcomes are positively proved `not-started`, or positively owned `started`; `unconfirmed` matches neither. The caller did return inside the original `<1500 ms` guard. **The failure is missing natural stream/cleanup proof, not a missed outer timing assertion.**

Frozen TS lines 250–252 require withheld admission, no protocol failure, no worker PID, supervisor identity/reap, aggregate child close, and no forced closure for no-start. Lines 289–304 wait 800 ms after stop, then destroy stdout/stderr and the control stream and finish honestly unconfirmed.

Crucially, `closed` is set only by the aggregate `child.close` event. Exact Node 22.22.3 `internal/child_process` counts **each i>0 stdio socket**, including extra FD3. Thus the original log does **not** establish whether stdout, stderr, FD3, multiple streams, or delayed close-event handling caused the missing close. Destroying stdout/stderr in the fallback is not a prior observation that those particular streams were open. The CI receipt lacks per-stream end/close timestamps and kernel fork/FD-owner evidence.

The separate missing-executable failures in both main suites assert `/spawn failed/` at `test/audit-recovery.test.ts:301`, but receive a 200 ms timeout with proved not-started instead. That check lives in Python `main()` after interpreter bootstrap and `require_waitid_support()`. This confirms preflight placement relative to the caller's deadline, but does not causally explain the unexplained first Node stream hold.

Later native Bun fast controls: Node-job log 1849 **206 ms**, Bun-job log 1847 **250 ms**, both proved not-started with no forced closure. They are controls, not erasure of the earlier red result.

Parsed original evidence: `evidence/ci-extracted-receipts.json`.

## Executable, fork and FD contract

Frozen `scripts/phase5_recovery_process.py` imports only stdlib plus the exact copied `scripts/qmd_mlx/process_anchor.py`. It validates waitid support and its fresh session, checks the command, advertises `prepared`, and consumes `S` before its first workload fork. Worker fork closes private FD3, redirects stdin, creates a worker process group, applies the canonical no-fork boundary where requested, then execs. Joining ownership anchors replace FDs 0–3 with `/dev/null`, so they are not intended output/proof-pipe holders. Worker stdout/stderr remain inherited until the anchored worker/session cleanup is complete.

The Node runtime owns the supervisor process and sockets. The Bun branch uses `Bun.spawn` as the native process/FD owner and borrows readers through `Bun.file(fd).stream()`, retaining that owner through exit and all drains. No changes or GC workaround were made to either path.

Locally, `otool` proves `/usr/bin/python3` links libxcselect and calls `xcselect_invoke_xcrun`. **That call alone does not prove a fork.** Kernel-observed local launch instead changes executable paths within the same reserved PID: Homebrew gate → `/usr/bin/python3` → CommandLineTools Python framework. The actual local selected interpreter is **3.9.6**, not the orchestration Python 3.15.1 or CI setup-python 3.12.10.

## Fresh bounded experiments and controls

Host: macOS **27.2.1 / 26B131**, **Apple M2 Pro**, **Mac14,9**, arm64. Native runtimes independently verified as Node **22.22.3** / libuv **1.51.0**, Bun **1.3.8**, both arm64. This is not the CI image or a virgin boot/OS cache.

The closure extracts the exact runner interface/function (frozen lines 37–451), builtin imports, and exact worker types/barrier constant. No model-bearing production imports or npm packages are loaded. The Python helper and process anchor are exact commit copies. A separate observed variant only adds timestamped event/stream-state listeners and before-destroy/finish snapshots; original 200/800/1800/<1500 bounds remain unchanged. The valid-report positive control has its separate existing-style 2000 ms budget, not an extension of the fast case.

| Local closure case | Node ms | Native Bun ms | Result |
|---|---:|---:|---|
| First exact fast case | 268.153125 | 291.342041 | started, worker/supervisor reaped, natural streams |
| Observed fast case | 286.165709 | 243.141083 | same, no forced close |
| Missing executable | 82.706750 | 79.001542 | spawn-failed diagnostic, proved not-started |
| Valid authored report | 155.184708 | 119.713792 | started, completed report and clean owned cleanup; injected-command refusal retained |

All **8** closure commands exit 0 and all declared assertions are checked programmatically. No outer guard kills, no recovery-helper/worker finally kills; intentional sentinel cleanup is explicitly recorded for every case. These are frozen-source diagnostic controls, not a full repository/canonical suite or release acceptance.

Raw probes register kernel kqueue events before target exec, retain the exact direct child with frozen `waitid(WNOWAIT)`, and never send `S`:

- Corrected system-launcher unadmitted case: **204.922875 ms**, SIGKILL through its owned Popen handle, all three streams naturally EOF, no root NOTE_FORK.
- Direct CLT executable control: **206.156125 ms**, same result; not proposed as an interpreter-selection repair.
- Positive control creating a test-owned Popen child: kernel records **one NOTE_FORK**, child exits 0, root reaps it normally. This validates the root-fork observer rather than merely trusting an empty trace.
- Native automatic child tracking (`NOTE_TRACK`) is unsupported here (errno 45). The working fallback records root NOTE_FORK/EXEC/EXIT only. It must not be advertised as complete descendant tracking or CI proof.

Separate real pipe controls use only test-owned Popen handles. Killing a root at 200 ms without another writer yields natural EOF at **217.684459 ms**. Adding a second owned writer leaves **all three streams without EOF at 1001.348875 ms**, even though the root is exited and unreaped. The red original-bound observation is recorded before its holder is finally killed. This demonstrates the POSIX inheritance mechanism and the monitor's ability to detect it; it does **not** identify a real CI bootstrap descendant.

Artifacts: `evidence/investigation-summary.json`, per-case JSON plus `-command.json` and `-finally.json`, `evidence/raw3-summary.json`, `evidence/*writer-positive*.json`, and `evidence/probe-source-manifest.json`.

## Platform clock warning for the active protocol owner

Empirical `evidence/clock-domains.json` shows clocks cannot be transmitted as though they share an epoch:

- system Python 3.9.6 `time.monotonic_ns()` returned `46698375`, while its `clock_gettime_ns(CLOCK_MONOTONIC)` returned `347238047923000`;
- orchestration Python 3.15.1 monotonic returned `306983828569375`;
- native Node `process.hrtime.bigint()` returned `347240867782250`;
- native Bun `process.hrtime.bigint()` returned `5000708`.

Even two Python implementations reporting `mach_absolute_time()` as their implementation did not expose the same effective epoch. Do not implement a cross-runtime expiry by copying raw Node/Bun `performance.now()` or `hrtime` values into Python `time.monotonic()`. This warning is independent of the CI stream-cause hypothesis and does not change this lane's frozen protocol.

## Minimum next patch/design proposal (no production patch applied)

1. **Add test-only, bounded observability before choosing a repair.** Record runtime/architecture, spawn return, `prepared`/admission, exact supervisor exit, each stdout/stderr/FD3 end/close/error, close-counter state on Node, and native-owner drain state on Bun. Snapshot before the existing forced-close fallback; do not infer EOF from `.destroyed`. Preserve the original failure receipt.
2. **Capture the first cold CI invocation, not a warmed retry.** Log static image/system launcher metadata without executing Python beforehand. Obtain actual `/usr/bin/python3` selected-executable/version and kernel root-fork/exec/FD-holder evidence in a separately instrumented pristine runner probe. The raw gate/kqueue harness is a starting point with an explicit NOTE_TRACK limitation; it is not an unchanged canonical launch.
3. **Only then choose the smallest causal fix.** A proved launcher-owned descendant calls for an ownership boundary before that launcher's possible forks, or a verified direct-interpreter contract. A proved EOF-with-delayed-reader event calls for a runtime reader/event-lifetime fix. No evidence currently authorizes choosing one, blindly replacing the interpreter, killing sampled descendants, or widening budgets. Keep canonical no-fork and kernel process anchors unchanged.
4. Keep the confirmed missing-executable preflight and execution-permission redesign separate; neither, by itself, qualifies the original open-stream failure.

Remaining acceptance: parent/active owner must qualify the settled candidate on the actual macOS arm64 image in both matrix jobs, capture the first 200 ms case and required native Bun controls with no forced-close certification, retain exact candidate head/checkout identity, and run the unchanged canonical suites/guards. Record no-start versus started proof and pre-finally process/sentinel/FD observations. Until those receipts exist, this independent bootstrap gate remains **open**.

## Investigation limitations / retained errors

- GitHub CLI was available only by absolute path, and auth required the explicitly allowed normal HOME for GH-only invocations. Every test subprocess used an allowlisted synthetic clear environment, call-owned HOME/Hermes/HF/XDG/tmp paths and offline flags; no normal user HOME was passed to tests.
- First raw instrumentation attempt failed with errno 45 before target exec. Its exit-1 command/stdout/stderr and original source are preserved. That first version registered the monitor before entering its fixture try/finally; gate stdin closes on orchestration exit, but no independent owned-handle exit receipt was captured. No historical PID was subsequently probed or signalled. The corrected harness always enters owned-handle cleanup.
- The next raw supervisor fixtures accidentally named nonexistent `/bin/true`; they truthfully produced preflight refusal and are retained as such. Distinct corrected receipts use the verified existing `/usr/bin/true`. They are not a retry-until-green source qualification.
- Fresh synthetic HOME is not a claim of cold kernel/Xcode/dyld caches. Local positive results cannot clear a failure on another macOS build, nor rule out a cold-only Apple launcher path.
- No skills were saved because this lane explicitly forbids skill/config writes; reusable observations are retained in this scratch report.

Commands actually executed are fully captured with argv, cwd, environment, exit code and outer cleanup in `evidence/*-command.json`. The entrypoints were `run_probes.py baseline`, `observed`, `raw`, and `raw3` through `/usr/bin/env -i ... /opt/homebrew/bin/python3 -I -B`, not imports of the mutable project. Do not rerun into this evidence directory; copy the harness/frozen closure to a fresh call-owned scratch root for future probes.
