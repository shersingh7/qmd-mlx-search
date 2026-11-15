# SPEC REVIEW — PASS_SCOPED

**Verdict:** `SPEC_PASS_SCOPED` for the frozen four-file test-only proposal, not production/CI acceptance. This verdict was recorded before beginning the separate QUALITY gate. No blocking source-bound specification finding. The original macOS CI/bootstrap gate remains open.

## Exact source binding

Delivery (read-only): `/Users/shersingh/.hermes/cache/scratch/qmd-ci-observer-mTSwsD`.
Reviewer root: `/Users/shersingh/.hermes/cache/scratch/qmd-observer-independent-g4hwpj_2`.

- Patch SHA-256: `69810f46ab0a458e53dd9a13a951a82e11379611c2bcdfa366939f1498cb174f`.
- Source-manifest SHA-256: `d4f94e439de1704994cd2eb94e6b697d55c0625a833f7881832071529f0fed87`.
- Authenticated all 244 original sources, all 246 proposed sources, and all 104 files declared by the supplied artifact manifest **within its stated pre-final scope**. Later direct receipts outside that manifest are not mistaken for missing declared artifacts. Rechecked current source bytes separately before every workload.
- Replayed the patch with `/usr/bin/patch -p1 --batch --forward -i ...` in an owned non-worktree: exit 0; all four output hashes match the delivery.

| Path | Original SHA-256 | Proposed SHA-256 |
|---|---|---|
| `.github/workflows/ci.yml` | `4afa73ffff380ca3fffcdf22118f91a75ca0d9bef780084b523208aa7b046e8c` | `f4a8f301be216c80a07668b8ddbbb8d9e3e44f12acfc3fb48fd51670dfbdf67f` |
| `test/audit-recovery-tree.test.ts` | `bffbd2c578cf357e3b34da75cca79a917c9537c6c211cd6e7ae08c274c3cfa1d` | `8f4d31a4348f9c041eead28c8379d9a6053793054fab8fc6f71a246d635dd515` |
| `test/helpers/recovery-bootstrap-diagnostics.mjs` | new | `3c762b713596d47c3e0622195418a6182fb150c290c38aa16146c7017fba41de` |
| `test/helpers/recovery-bootstrap-diagnostics.selftest.mjs` | new | `b25346753f581711188206b525cac5f972d1d71e2301611fe55b9daa6287046e` |

No production, loader, worker, anchor, dependency, manifest, lock, tsconfig, or other source change. TS remains `6204b26a3a8218cd1da6ce03ec856065656d904cc9b936d9110900f8d79ac875`; Python remains `52a00fb5b891752f8d53f8d8b08fae3ce467d392461b6524f762376ade344bbf`. Bound the existing lease receipt and both report hashes; all 12 lease candidate paths match, using the original test preimage for its one intentionally instrumented path. No protocol re-audit was performed.

## Preservation / observer interaction

- Removing only the declared import/begin/attach/finish changes recovers the **entire** canonical test byte-for-byte. AST comparison separately preserves all 8 test declarations, all 7 spawn/execute calls, exact call text, and order. Limits remain 200 ms execution, 800 ms teardown, 1800 ms guard, and exclusive elapsed `<1500 ms`.
- `test/audit-recovery-tree.test.ts:43-51,90-94`: observer surrounds the existing synchronous launch and finishes before existing fixture cleanup. No diagnostic Python or extra recovery invocation preceded the first focused changed case. The pre-existing sentinel still launches Node.
- `test/helpers/recovery-bootstrap-diagnostics.mjs:122-153`: real constructor-time Node `child_process` diagnostic channel captures the actual ChildProcess, matches the fixed interpreter/flags/helper after synchronous launch, then promptly unsubscribes. An independent real runtime control verified the notification precedes spawn fields and contains the actual object; unmatched children are refused. Native Bun never subscribes or acquires another FD owner.
- `:49-119`: bounded 32 scalar snapshots; only spawn/exit/close, per-stream end/close, and `errorMonitor`. No data/readable/error-consuming listeners, stream reads/resume/pause, borrowed-FD opens, timers, signals, or global/prototype/method replacement in the observer. Direct checks verified active ordinary errors remain unhandled, method/flow identity, exact raw-listener restoration, finish-projection exception disposal, primary caller error preservation on IO failure, and existing channel subscriber preservation.
- `:27-46`: actual `transportFailure` is projected as the pre-destroy receipt with `tMs:null`; return-time Node counters are not relabelled as fallback-time counters. EOF, closed, and destroyed are separate. The separate real-helper fallback control matched the actual owner PID and exact pre-destroy snapshot, retaining `pipesClosed=false`, `forcedPipeClose=true`, `containmentVerified=false`, and Node `2/4`. This is failed transport evidence, not successful cleanup.
- Disabled mode has no subscriptions, receipt getter accesses, listeners or artifact IO in independent controls. Fixed fields omit argv/config/env/output/error messages. Active error monitor tests retained the exact original thrown error without consuming it. Synthetic messages in these controls are labelled fixtures, not real OS errors.

## Workflow and evidence boundaries

- `.github/workflows/ci.yml:44-101`: triggers, matrix, fail-fast, runtime versions, architecture gate, Python jobs, and all original canonical commands/order compare equal after removing only the declared diagnostic env prefixes/added steps. Opt-in is passed **after** the offline wrapper. PR head appears in an env field, not shell source; trusted RUNNER_TEMP/GITHUB_ENV are quoted.
- Metadata reads `.git/HEAD`/refs without git, CLT, Python, xcrun or xcode-select; only allowed `sw_vers` is executed. Source head, event SHA, and actual checkout are distinct. Local mirror checkout is correctly unavailable/null, not fabricated. First selected interpreter/version stays unobserved. A fresh HOME is explicitly not cold OS evidence.
- `.selftest.mjs` is outside unchanged `test/**/*.test.ts` discovery. Controls are explicitly after canonical suite/smoke and `always()` artifacts have 7-day retention. Other canonical files may run concurrently; the helper correctly disclaims globally-first-supervisor/virgin-cache claims.
- `ci.yml:95-99` is an any-file upload check, **not** proof of fast-case artifact presence. This limitation is documented in the supplied implementation report:90. A directory containing only metadata/TAP cannot qualify the gate.
- Retained original CI log/readback: run `43503677544`, Node job `132155679910`, Bun job `132155679594`, source head `dcf98c159b671b4d08c3759215c4371502f49fc0`, checkout `99d061d2d373f8affd62210f74bf0b6d5ce0c898`. Node fast receipt at line 727 is 1007 ms and the actual assertion is `unconfirmed` vs `started` (log lines 742/1955), **not** the elapsed bound. Original source hashes differ from the later accepted lease: do not imply the new lease already ran in that historical CI. The original stream/holder/reader-delay cause remains UNPROVEN.

## Independently executed evidence

Actual host: macOS 27.2.1 / 26B131, Apple M2 Pro arm64; Node 22.22.3 / libuv 1.51.0; native Bun 1.3.8. Not the original macos-15 CI image.

| Workload | Actual result |
|---|---|
| First changed fast case, observer on | exit 0; 1 pass, 7 CLI-filtered; 248 ms; actual supervisor PID 39647 matches trace; natural EOF/close and Node 4/4 |
| Same case, observer off | exit 0; 1 pass, 7 CLI-filtered; 222 ms; no artifact |
| Provided direct Node helper selftests | exit 0; 8 executed/pass, 0 fail/skip |
| Independent heldouts | exit 0; 6 executed/pass, 0 fail/skip |
| Native Bun import/no-op control | exit 0; 1 pass, 7 filtered; 216 ms; explicit unsupported native-owner annotation, no trace |
| Separate actual helper fallback | exit 0 for expected negative control; 946.096916 ms; real failed transport, 2/4, null pre-destroy timestamp; fixture cooperatively released |
| Independent preservation/patch/workflow shell syntax | exit 0; all 11 JavaScript-job run blocks pass bash -n |
| Static metadata command | exit 0; exact source hashes read back; no selected-Python identity claim |

Raw commands, full explicit environments, source hashes, stdout/stderr, exits/signals, and no-outer-cleanup receipts are in `evidence/*-command.json` and logs. Main readback: `evidence/spec-readback.json`; preservation: `evidence/static-preservation.json`; authenticity: `evidence/authentication.json`; lease: `evidence/lease-binding.json`.

All workloads used owned real node_modules containers, per-package read-use aliases and Vite/Vitest/Bun/HOME/Hermes/knowledge/HF/XDG/tmp/cache paths established before first run; no ambient NODE_OPTIONS/PYTHONPATH/credentials, CI=1, REQUIRE_BUN=1, GPU disabled and HF offline. At most two workload workers ran concurrently. No full suite/build, CI/gh/network/install/download, model/GPU/private knowledge, staging/commit, live edit, or service/config/skill write.

## Qualifications / required parent gates

- **C1 retained, not a new observer defect:** the existing lease's extra 1300 ms post-startup sole-owner pause returns truthful incomplete cleanup around 922–927 ms without forbidden work; owners naturally reap later. Observation does not turn this into complete-cleanup proof.
- **C2, evidence scope:** Node-private counters and runtime end/close events cannot identify kernel holders or causality; prepared/admission event times and selected system-interpreter identity are unobserved; native-owner tracing remains unsupported.
- **C3, acceptance gate:** parent must narrowly integrate against exact preimages, run canonical full suite/build/type/package gates, and obtain actual candidate-head macOS CI. For BOTH main macOS matrix jobs read back run/job identity and exact artifact IDs/files; require `observed-node-supervisor` fast JSON, match owner PID/runtime to unmodified canonical logs, retain any earlier failure, and verify original assertions/bounds/scheduling. Compare candidate source head, event SHA, and actual PR merge checkout separately. Read exact workflow/test/helper/lease source bytes at that checkout and compare hashes to the reviewed candidate and metadata. A missing/unavailable fast JSON, missing checkout source readback, or metadata/TAP-only artifact leaves the gate OPEN. Later native passes cannot erase an earlier Node failure.

## Retained review-only errors

Initial independent heldout was 5 pass / 1 fail because the reviewer wrongly expected constructor `spawnfile` to be undefined rather than Node's null. Initial source and raw failure are retained; only the nullish-absence oracle was corrected, and six fresh controls passed. Initial readback incorrectly expected per-assertion status `pending`; actual Vitest uses `skipped` while aggregate field is numPendingTests=7. Its raw failure/source are retained and corrected parser readback passed. Neither was a product/source failure. Supplied earlier exploratory whole-snapshot tsc runs remain disqualified; they were not replayed.
