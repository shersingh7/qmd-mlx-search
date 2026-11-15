# Recovery-bootstrap diagnostics — local proposal verified, CI gate still open

## Deliverable

Own root: `/Users/shersingh/.hermes/cache/scratch/qmd-ci-observer-mTSwsD`.

- `diagnostics-proposal.patch` — four-path proposal, applied successfully to a separate owned non-worktree tree and every applied output hash checked.
- `input.json` — original 244-file hashes and 274 dependency read-use aliases.
- `source-hash-manifest.json` — proposed source hashes, original/proposed changed-path hashes, byte-preservation and patch-application proof.
- `verification-summary.json` — parsed actual focused results, controls, metadata, shell syntax, limits and limitations.
- `artifact-hash-manifest.json` — hashes of verification inputs/artifacts present before the final runner log/report (scope stated in that manifest).
- `DESIGN.md`, `harness/`, `evidence/` — design written before edits plus explicit execution addendum, runnable owned harnesses, raw stdout/stderr, argv/cwd/clear-environment/exit receipts, source-bound final invocations and retained original CI bytes.

Patch SHA-256: `69810f46ab0a458e53dd9a13a951a82e11379611c2bcdfa366939f1498cb174f`.

| Proposed path | Original SHA-256 | Proposed SHA-256 |
|---|---|---|
| `.github/workflows/ci.yml` | `4afa73ffff380ca3fffcdf22118f91a75ca0d9bef780084b523208aa7b046e8c` | `f4a8f301be216c80a07668b8ddbbb8d9e3e44f12acfc3fb48fd51670dfbdf67f` |
| `test/audit-recovery-tree.test.ts` | `bffbd2c578cf357e3b34da75cca79a917c9537c6c211cd6e7ae08c274c3cfa1d` | `8f4d31a4348f9c041eead28c8379d9a6053793054fab8fc6f71a246d635dd515` |
| `test/helpers/recovery-bootstrap-diagnostics.mjs` | new | `3c762b713596d47c3e0622195418a6182fb150c290c38aa16146c7017fba41de` |
| `test/helpers/recovery-bootstrap-diagnostics.selftest.mjs` | new | `b25346753f581711188206b525cac5f972d1d71e2301611fe55b9daa6287046e` |

No production `src`/recovery helper/worker/loader/anchor, manifest, lock, dependency or compiler-config change is in the patch. Frozen TS remains `6204b26a3a8218cd1da6ce03ec856065656d904cc9b936d9110900f8d79ac875`; Python remains `52a00fb5b891752f8d53f8d8b08fae3ce467d392461b6524f762376ade344bbf`. All 244 original own/parent frozen input file hashes were checked; all unrelated proposed source bytes remain identical.

## What the proposal observes

Node 22.22.3's constructor-time internal `diagnostics_channel('child_process')` notification captures the real ChildProcess without replacing any method. Immediately after the existing fast-case synchronous launch returns its Promise, the helper matches exactly `/usr/bin/python3 -S -u <frozen helper path>`, attaches to that actual owner, and unsubscribes the constructor channel.

It records at most **32** relative events: spawn return/spawn, owner exit/close/errors, and individual stdout/stderr/control-FD3 end/close/errors. Snapshots read Node close counters, actual handle presence, stream EOF/close/destroy/flow state, readable length, socket bytes and observed FD values. Error hooks use `errorMonitor`, not consuming error listeners. There are no data/readable listeners, reads, resume/pause, borrowed FD opens, polling timers, signals, cleanup actions, global/prototype patches or command/config/error-message dumps in the observer. All hooks are removed in finally; disabled mode performs no subscription/listener/artifact IO.

The candidate's existing `result.transportFailure` supplies the true **pre-destroy** snapshot. Its event timestamp is explicitly null. Prepared/admission are recorded only where available in that existing snapshot/final receipt; their exact times are **not observed**. Node counters are timestamped at real runtime events/return, not retrospectively at the production fallback line. EOF is never inferred from destroyed/closed flags, and no diagnostic field certifies cleanup.

Native Bun owner tracing is explicitly **not implemented**. The main package-script jobs actually host Vitest in Node. Native Bun receives a separate unsupported-owner annotation without creating a second descriptor owner.

## Preservation and canonical order

- Removing only the new import/begin/attach/finish calls reconstructs the entire original test file **byte for byte**. Every existing assertion, all eight test declarations, fixture commands and launch order remain intact.
- Original **200 ms execution / 800 ms teardown / 1800 ms guard / exclusive <1500 ms elapsed bound** remain unchanged. No retry, skip, timeout widening, interpreter replacement or cleanup-certification adjustment was added.
- No extra diagnostic Python/recovery launch occurred before the first local changed fast-case invocation. The original sentinel still launches Node, not Python. First local fast-case elapsed was 216 ms; this is not a cold-OS/CI claim.
- CI metadata does not execute Python, git/CLT, xcrun or xcode-select. It reads actual `.git/HEAD`/refs directly, records OS/image/CPU and source hashes, and identifies requested system Python while marking its first selected executable/version **not observed**.
- The same existing workflow, PR/push triggers, matrix, runtime versions/architecture selection and canonical command order are retained. Observer opt-in/path arguments are supplied **after** the offline wrapper strips ambient variables. Trusted `$RUNNER_TEMP`/`$GITHUB_ENV` references are quoted; PR head SHA is an env value, not shell source.
- New selftests are `.selftest.mjs`, outside canonical Vitest `**/*.test.ts` discovery, and the workflow runs them explicitly only **after** the existing full suite/smoke steps. Existing full-suite scheduling can still run other canonical files concurrently: the artifact does not claim to identify the globally first supervisor or a virgin OS cache.

## Actual local verification

Host: macOS **27.2.1 / 26B131**, Apple M2 Pro, arm64; Node **22.22.3**, libuv **1.51.0**. This is not the failed CI image. Own node_modules containers/per-package aliases and own Vite/results/cache paths existed before the first red run. Workloads used clear allowlisted environments and own HOME/Hermes/knowledge/HF/XDG/Bun/temp paths, offline flags, REQUIRE_BUN=1 and QMD_LLAMA_GPU=false. No normal HOME, ambient PYTHONPATH/NODE_OPTIONS or secrets were forwarded by the harness.

| Final source-bound check | Actual result |
|---|---|
| Existing fast case, diagnostics on | **1 pass, 0 fail, 7 CLI-filtered**, 246 ms; real Node owner matched; stdout/stderr/FD3 natural end+close, owner exit+close, counter **4/4** |
| Existing fast case, diagnostics off | **1 pass, 0 fail, 7 CLI-filtered**, 221 ms; no forced closure, sentinel alive, no recovery fixture finally kills |
| New Node helper selftests | **8 pass, 0 fail, 0 skip** |
| Native Bun import/unsupported-owner control | Bun **1.3.8**, **1 pass, 0 fail, 7 filtered**; 249 ms; explicit unsupported-native-owner artifact, no tracing claim |
| Separate after-first actual candidate fallback experiment | Exit **0**, expected failed transport receipt observed at **966.232834 ms**, Node counter **2/4**; truthful `pipesClosed=false`, `forcedPipeClose=true`, `containmentVerified=false` retained |
| Helper strict JSDoc/type check and both JS syntax checks | Exit **0** |
| Same guarded TS source surface, original vs proposed | **65 → 65** diagnostics, **0 new**; not a green canonical root check |
| Actual newly provided official actionlint **1.7.12** | Exit **0**, no diagnostics; binary hash checked against supplied receipt; earlier successful lint retained |
| Workflow shell syntax | All **11** JavaScript-job run scripts passed `bash -n`; shellcheck unavailable/explicitly disabled |
| Patch application | Exit **0** in owned `patch-application-check/`; all four resulting hashes match proposal |

Selftests exercise real streams rather than fake API responses:

1. Natural closure: three EOF observations and Node counter 4/4.
2. Authored self-bounded writer inheriting **only FD3's write endpoint**: stdout/stderr EOF present, FD3 EOF absent, counter 3/4, no aggregate close; cooperative release then natural close. Holder identity comes from fixture construction, not inference from runtime events.
3. Deliberately delayed reader event loop: owner exit observed, aggregate close delayed **40.37 ms**, then all EOF/close events. This is not labelled a retained writer.
4. Real forced stream destruction remains separate from EOF. An explicitly authored schema-control receipt tests projection/privacy. A separate actual-candidate helper experiment additionally verified copying a **real** production pre-destroy receipt: its known escaped stdout/stderr holder leaves proof-FD3 ended and counter 2/4 before fixture release. No historical/numerically guessed PIDs were signalled, no causal CI claim made.
5. Event/output bounds, original method/flow/error-propagation identity, exact listener restoration, idempotent disposal, disabled mode, exception-path subscription restoration, nonmatching-child rejection and authored detached/loose/packed/malformed checkout refs are asserted.

The final verifier reads exact artifacts back, matches owner PID to the original pre-finally receipt, checks all final source hashes, counts declared tests programmatically and verifies original assertion/guard/sentinel outcomes. All final workload commands returned without an outer harness cleanup signal.

## Retained failures and scope issue

- Initial red: missing helper module, before implementation. Retained unchanged.
- Initial strict-helper check: one JSDoc rest-parameter typing error, fixed and rerun successfully.
- First selftest run: privacy assertion accidentally matched the observer's fixed descriptive label `Node-private-read-only-not-kernel-state`; corrected to check the three exact authored secret sentinels. Raw 6-pass/1-fail result retained; final eight-test source passed.
- First final-readback parser failed on Vitest ANSI prefixes. Raw logs were not changed; parser now uses Node's `stripVTControlCharacters`, and the fresh final readback passed.
- **Two exploratory whole-snapshot tsc commands followed archived documentation `.mjs` absolute imports into live public source.** Both were parse-only and reported 67 diagnostics; no live source execution/write or private-knowledge access occurred. They are explicitly disqualified for frozen-source qualification, retained in raw logs, and those harness modes are now blocked. Final bounded validation excludes that historical documentation JS and guards every source read to the own snapshot or approved node_modules. Its 65 unchanged diagnostics are not the parent's separate 61-error canonical/root gate and do not repair it.
- Original run **43503677544**, Node job **132155679910**, checkout merge **99d061d2d373f8affd62210f74bf0b6d5ce0c898**, head **dcf98c159b671b4d08c3759215c4371502f49fc0**, line-727 **1007 ms / unconfirmed / forced closure** receipt is preserved verbatim under `evidence/original-ci/`. It is not replaced by local green controls. Original cause remains **unproven**; this work does not identify which original stream was held or a retained writer versus reader-event lag.

## Proposed real-CI artifact/readback contract — parent owned, not executed

For each existing macOS Node/Bun matrix job, upload via `always()`:

- Artifact name `recovery-bootstrap-<node|bun>-macos-15-<run_attempt>`.
- `runner-metadata.json` with source-head, event SHA and actual checkout **separately**, image/OS/CPU, requested system interpreter, package-script launcher and static source hashes.
- `fast-timeout-node-<pid>.json` from the existing case. Require status `observed-node-supervisor`; missing/unavailable observations are not qualification.
- Native annotations may additionally appear as `fast-timeout-bun-<pid>.json`; unsupported-owner status is not native tracing proof.
- `after-canonical-selftests.tap`. No metadata environment/cache directories are uploaded.

The upload's any-file existence check does **not** establish that the required fast-case artifact exists. Parent must read back exact artifact IDs/files from the exact candidate run, require the Node fast-case artifact in both matrix jobs, match its source hashes/runtime/owner PID to unmodified canonical logs and validate actual first-case scheduling. Preserve original console failure receipts and logs, not only projected diagnostics. Require the original canonical bounds/assertions and native ownership suites; later native passes cannot erase an earlier Node failure.

If a failure recurs, use the per-stream end/close state and actual Node counters to identify what was observed. A missing EOF/close alone does not prove a kernel writer. Prepared/admission timings, first selected system-interpreter identity, kernel holder ownership and full descendant tracing remain unavailable. Prior local diagnoser root-fork/retained-writer controls are design input only; NOTE_TRACK was unsupported (errno 45). Any optional shim/compiler/pre-exec kernel probe must be separately identified on a fresh job or after canonical execution, never called an unchanged first launch or silently introduced as a production dependency.

**Next acceptance:** parent independent spec/quality review of these four paths and their lease-candidate interaction; compare preimage hashes and integrate narrowly (do not restore an old mirror wholesale); parent canonical build/full-suite/type/package integration; then actual exact-head/checkout macOS CI readback. No CI execution, rerun, dispatch, network/install, model/GPU/private-knowledge work, commit, push, stage, merge, daemon/config/skill change or production publication was performed here.
