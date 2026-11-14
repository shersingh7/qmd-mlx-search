# Recovery startup closure — bounded ownership repair

Status: **execution-lease repair scoped-accepted with qualification C1; integration and native CI remain open**. Parent-owned integration addendum to the repository audit/remediation plan. No deployment or live-index changes are authorized by this document.

## Evidence and root cause

Candidate `dcf98c159b671b4d08c3759215c4371502f49fc0` correctly stops an **unadmitted** delayed supervisor. Independent Node and Bun checks passed that contract. It does not safely close the **admitted but not ready** transition:

1. Python consumes `S` and drains queued cancellation.
2. It can be delayed or OS-stopped before `os.fork()`.
3. The parent no longer terminates its process handle after admission, yet has no worker ownership proof.
4. The caller returns an honest failure after the bounded teardown; the supervisor can later fork and execute user work.

Independent evidence: `/Users/shersingh/.hermes/cache/scratch/qmd-cold-admission-independent-cl2z6zva/review-receipt.json`. Both scheduling-delay and real SIGSTOP/SIGCONT controls reproduced the gap. Parent reruns in Node 22.22.3 and Bun 1.3.8 also observed a live supervisor at return and a later side-effect file; neither needed fixture cleanup kills. Original negative receipts must remain unchanged.

Exact-head CI also has two separate unresolved contracts in run `43503677544`:

- A missing worker executable is not diagnosed before a cold Python startup consumes the 200 ms execution budget. The fixture expects a spawn failure but receives a timeout/no-start receipt.
- One Node/Mac fast-timeout case reaped the unadmitted supervisor but did not receive aggregate `child.close` before fallback, forced pipe closure, and correctly returned `launch.state=unconfirmed`. The receipt does not identify an open stream or distinguish retained writers from delayed close events. Do not relabel this as clean cancellation. `/usr/bin/python3` is locally an xcselect/xcrun launcher; bootstrap descendant inheritance is a hypothesis, not an established CI cause.

Logs: `/Users/shersingh/.hermes/cache/scratch/qmd-ci-dcf98c1-92jhwzjl/`.

## Design requirements

- Make cancellation and ownership explicit across actual process creation, child bookkeeping and readiness. Another cooperative check immediately before a fork does not remove a scheduling gap after that check.
- Never blindly kill an admitted supervisor if that can discard a live worker's ownership anchor. Never signal guessed/reused numeric process groups.
- Acknowledged no-start and preflight refusal may need distinct truthful states. Do not claim a worker or supervisor was reaped if it never existed; do not infer no-start merely from a missing PID or forced pipe closure.
- Preserve the 200 ms execution deadline, 800 ms teardown, 1800 ms test failure guard and less-than-1500 ms helper bound. No cache warmup, retry-until-green, timeout inflation or skipped mandatory runtimes.
- Preserve real Node and native Bun handle ownership, descriptor reuse/GC behavior, started-worker anchored cleanup, canonical kernel no-fork containment and the rule that arbitrary injected commands cannot be certified as contained.
- Validate mandatory prerequisites before expensive work. Determine why aggregate Mac stream closure is missing using per-stream/native-owner observations before changing runtime selection or cleanup behavior.
- Keep protocol data bounded and private to the supervisor; maintain caller hook/timer/descriptor cleanup on all exits.

## Implementation sequence

1. **Reproduce and instrument.** Retain existing red evidence. Add portable repository regressions for admitted-before-fork delay and OS pause, recording process/sentinel/side-effect state before fixture cleanup. Diagnose the two exact CI failures separately; report any unreproduced cause honestly.
2. **Choose the ownership mechanism.** Write a short state/ownership table before production edits. Identify the process handle/anchor that authorizes each cancellation and the point at which user work can begin. Cover cancellation before admission, after admission but before creation, across creation/registration, before ready, after ready, EOF and signals. Prefer a small provable design; do not add a native/compiler dependency without discussing the scope with the parent.
3. **Implement one transition at a time.** Keep one exclusive writer across the TS/Python protocol and its fixtures. Add safe prerequisite refusal and repair the diagnosed stream/bootstrap cause without changing unrelated retrieval/model code or dependency graphs.
4. **Prove the contracts.** Exercise both actual runtimes, delayed startup, actual OS pause, malformed/truncated protocol, queued cancellation/EOF, unrelated sentinels, canonical containment, process-tree cleanup and native Bun FD reuse. Relocate retained probes only as necessary for new semantic transition locations; preserve their behavioral assertions and original versions. If a stronger no-start acknowledgement changes an old characterization assertion, document that explicitly and require positive evidence rather than making identity optional.
5. **Independent acceptance and integration.** A fresh reviewer must close all reproduced findings. Parent then runs settled-source canonical tests/build, changed-language checks and relevant package tests; publishes a reviewed candidate and verifies every exact-head CI matrix job before merge.

## Exclusive live-write scope

- `scripts/phase5_e2e_runner.ts`
- `scripts/phase5_recovery_process.py`
- `test/audit-recovery-tree.test.ts`
- `test/audit-recovery.test.ts`
- `test/phase5-recovery.test.ts`
- `test/audit-native-bun-recovery.test.ts`
- `test/fixtures/bun_recovery_ownership.ts`
- `test/audit-recovery-startup.test.ts`
- `test/audit-recovery-transition.test.ts`

A tightly scoped new recovery-specific helper/test may be proposed if genuinely necessary. No edits to manifests, locks, workflows, other review-owned files, live services/configuration, existing independent evidence, or immutable November 11 acceptance evidence. The TypeScript repair lane is an isolated mirror, not another live writer; its later patch must preserve this lane's resulting protocol.

## Acceptance limits

Passing the old focused suites is insufficient: they missed the admitted-before-fork gap. A truthful `unconfirmed` error is necessary when proof fails, but is not evidence that late work or leaked processes have been prevented. Do not claim broad real-time guarantees beyond the actual ownership model and tested failure domain.

## Revision: execution permission must not survive parent death

The first ownership repair passes its 70 focused tests, but independent review found a second stale-permission boundary: Python can consume `E`, pause, then forward it after the execution deadline and queued EOF. Actual Node and native Bun parent-SIGKILL probes executed late work; ordinary live-parent timeout and graceful-exit controls remained safe. The parent independently reproduced the raw-protocol failure with unchanged source and positive controls. Evidence: `/Users/shersingh/.hermes/cache/scratch/qmd-transition-independent-ruk8fjg0/` and `/Users/shersingh/.hermes/cache/scratch/qmd-parent-stale-execution-8z0ri3dz/parent-receipt.json`.

This is an execution-authority design defect, not permission to extend a timeout or add another supervisor-only check immediately before forwarding a byte.

1. **Design the actual execution boundary before editing.** Record which process/descriptor owns cancellation after the caller is SIGKILLed, where execution permission becomes effective, and how a delayed grant expires or is revoked. Examine both the supervisor and the child gate; moving the same scheduling gap to the next line is not acceptance.
2. **Keep the original budget authoritative.** Do not restart the execution timeout after delayed startup/grant consumption. If clocks or a lease cross runtime boundaries, establish their semantics rather than assuming Node, Bun and Python clocks share an epoch. Preserve the existing 200/800/1800/<1500 cancellation bounds.
3. **Make abrupt death independent of caller exit hooks.** Queued cancellation/EOF and terminal control-channel errors must reach anchored cleanup without granting stale execution or discarding a live worker. Preserve native owned handles, kernel-authenticated anchors, no-fork containment and honest receipts.
4. **Reproduce, repair and cross the new boundary.** Retain original RED evidence; add permanent positive/negative tests for valid execution, EOF without E, E consumed before a scheduling hold, and alive/graceful/hard parent exit under both real runtimes. Probe grant forwarding/consumption and the last trusted execution gate, not only hardcoded source-line locations. Record absence, side effects, sentinels and any fixture kills before cleanup can hide a defect.
5. **State the supported failure model.** Do not silently exclude abrupt parent death or relabel stale execution as success. Also do not promise impossible hard-real-time behavior while the entire supervising system is unscheduled; document detection/admission/cleanup boundaries and escalate a genuine platform limitation with evidence.
6. **Review the repaired protocol again, then qualify native CI.** The earlier Mac bootstrap/open-stream failure remains a separate unresolved gate. Type-only integration must preserve the resulting protocol rather than restore an older mirror prefix.

## Execution-lease candidate: scoped acceptance with qualification C1

The five-file repair delta and 12-file source closure are archived under `docs/security/dependency-remediation/open-runtime-contracts/recovery-execution-lease/`. The parent reproduced **114 passing tests in seven files**, with no failed/skipped tests, and replayed the exact entry delta. All 44 new pre-finally cases retained their sentinels and needed no fixture cleanup kills; actual Node and native Bun each exercised valid, alive-parent, graceful-exit and hard-SIGKILL cases. Existing deliberately escaped, uncertified fixture descendants remain separately disclosed rather than counted as natural cleanup.

The `S<remaining-ms>\n` lease is anchored before Python emits PREPARED; it does not compare raw JS/Python clocks or restart a deadline on receipt. The parent also verified the selected Python's already-initialized clock is inherited across fork. The child checks the lease and caller-channel liveness after containment, and refuses stale execution. Before real exec/exit closes the existing CLOEXEC status pipe, the supervisor polls rather than consumes cancellation bytes that the child still needs. Independent review now passes this scoped clock, execution-boundary, signal and reader-handoff contract; the original failed review remains negative evidence.

The completed review and parent replay are archived under `docs/security/dependency-remediation/accepted-runtime-contracts/recovery-execution-lease/`. The reviewer freshly passed the seven-file, 114-test suite and reproduced both earlier REDs. The parent authenticated all 12 candidate hashes and 244 public source files, then freshly replayed 15 unchanged semantic heldouts, seven real gate controls, three fork-clock controls and a kernel cross-session join-refusal control. These additional invocations are not added to the 114 distinct tests.

**C1 — execution refusal is not unconditional successful-cleanup certification.** An additional 1300 ms hold inside the sole supervisor's post-startup readability poll outlasts the caller's existing teardown allowance. On both actual Node and Bun, the parent independently reproduced an unexpired child-gate refusal with no work executed, but the caller returned with `supervisorReaped=false`, `childReaped=false`, `pipesClosed=false` and `forcedPipeClose=true`. Both owners naturally terminated/reaped after the hold, without fixture intervention. This qualifies the supported scheduling/cleanup boundary: preserve the truthful incomplete receipt and do not count it as a clean-cleanup pass or promise unrestricted real-time successful cleanup. It does not reopen the reproduced stale-grant, EOF, abrupt-parent-death or queued-cancellation execution defects.

Lease review is complete; the separate test/CI observer has now passed scoped independent review and been applied as described below. It remains outside the lease proof and must not insert Python/recovery warmup before the **existing** canonical fast case, change production protocol, or weaken any limit/assertion. Neither lane can claim real CI or full-workspace acceptance. Type-only integration must preserve the frozen new protocol and qualification C1.

## Verified Mac diagnosis: cause and acceptance still open

The read-only investigation and parent verification are archived in `docs/security/dependency-remediation/open-runtime-contracts/mac-bootstrap/`. This is an **open-gate evidence archive**, not accepted recovery code.

- **Correct the job identities:** Node is `132155679910`; Bun is `132155679594`. Both main package-script Vitest suites actually run Node 22.22.3; native Bun 1.3.8 controls are separate. The CI image is `macos-15-arm64`, macOS 15.8.2 / 24H232, image `20261104.0514.1`. Its CPU model is unrecorded, not established as M2. The actual PR checkout was merge `99d061d2d373f8affd62210f74bf0b6d5ce0c898`; inspected source bytes match head `dcf98c159b671b4d08c3759215c4371502f49fc0`.
- **Preserve the exact negative:** the first Node fast-case receipt returned in 1007 ms, inside its 1500 ms bound. Test line 75 rejected unconfirmed ownership. Node's aggregate close includes stdout, stderr **and FD3**; missing aggregate close does not identify which reader/writer was responsible. Local same-PID exec and natural-drain controls, plus a genuine retained-writer reproduction, do not establish the CI cause or erase its failure.
- **Review deadline domains explicitly:** the parent independently measured system Python 3.9.6, project Python 3.12.13, Node and Bun. Raw monotonic/hrtime values are not interchangeable across these implementations. A matching implementation name is insufficient. Require a proven conversion/shared clock or conservatively anchored duration protocol that preserves the original deadline across startup, transmission, queued grants and fork/exec boundaries. Do not restart the budget on receipt. The numerical warning was not delivered to a no-longer-live implementation worker; it remains an explicit parent/reviewer obligation for that handoff.
- **Instrument before selecting a repair:** capture bounded stdout/stderr/control end/close/error events, native-owner state and a snapshot before existing forced closure. Do not infer natural EOF from `.destroyed`, change stream flow with instrumentation, or change the 200/800/1800/<1500 bounds. Capture the first un-warmed recovery invocation on the actual CI image, plus selected-interpreter and kernel fork/FD evidence in a separately identified instrumented probe. A fresh HOME is not a virgin OS cache, and an instrumented launcher is not an unchanged canonical launch.
- **Keep gates separate:** missing-executable preflight and stale execution-grant repairs do not independently close this bootstrap gate. Interpreter replacement, wider timeouts, forced-close certification and sampled descendant kills remain unsupported without causal evidence. Preserve exact candidate/checkout identity and both actual matrix-job receipts before merge.

## Test-only observer: scoped review passed and four paths integrated

The four-path patch is archived in `docs/security/dependency-remediation/open-runtime-contracts/mac-bootstrap/diagnostics-proposal/`. It adds an opt-in Node observer to the existing fast case, a separate helper/selftest pair and CI metadata/artifact handling. Production recovery/worker/loader/anchor bytes, dependencies, deadlines, assertions and canonical launch order remain unchanged. Removing the observation hooks recovers the original test file byte for byte.

The parent independently applied the patch in an owned derivative and verified all 246 proposed source hashes. The existing fast case passed with diagnostics on and off (one executed case and seven filtered cases each); all eight Node diagnostic selftests passed. Strict helper typing, syntax and actionlint passed. These local results neither establish the original CI cause nor qualify the actual CI image.

The observer distinguishes individual stream EOF/close events and Node close counters without reading/draining streams or owning descriptors. It copies the existing pre-destroy receipt without inventing its timestamp or inferring kernel causality. Native Bun owner tracing remains explicitly unsupported. Extra controls run after the existing canonical suite; concurrent canonical files mean this is not a globally-first-supervisor or cold-OS claim.

Retain the disclosed verification-scope incident: two exploratory no-emit checks followed historical documentation imports into live public source. They performed parsing only and are disqualified for frozen-source acceptance. The later guarded 65-to-65 diagnostic comparison is not a canonical root type-check pass.

Independent review `deleg_eb890b44` returned **SPEC_PASS_SCOPED / QUALITY_PASS_WITH_QUALIFICATIONS**, with no blocking source finding. The parent authenticated 244 original and 246 proposed source files, 104 supplied artifacts and 103 review artifacts, then freshly replayed all six independent heldouts and a separate real-helper negative transport control. The latter matched the actual owner and exact pre-destroy receipt, with a null pre-destroy timestamp and observed Node counter 2/4; it deliberately retained failed-transport/forced-closure semantics and cooperatively released its fixture writer. It is not successful-cleanup or historical-CI-cause proof.

Only `.github/workflows/ci.yml`, the canonical recovery-tree test and the two new diagnostic helper/selftest files were applied, against exact preimages. All four final hashes match the review; production TS/Python lease bytes are unchanged. The live helper's eight Node selftests and live actionlint passed. Review and parent evidence are archived in `docs/security/dependency-remediation/accepted-runtime-contracts/recovery-bootstrap-observer/`. This is file-level integration, not canonical full-workspace/build/type/package acceptance.

**Native CI acceptance remains open.** In both main macOS matrix jobs, require the actual `observed-node-supervisor` fast-case JSON, owner/runtime/log agreement and unchanged assertions/bounds. Read back the candidate head, event SHA and actual merge checkout separately, then verify the exact workflow/test/helper/lease source bytes at that checkout against the candidate and metadata. An upload containing only metadata/TAP, a missing/unavailable fast trace or missing checkout-source evidence does not qualify. Later native Bun passes cannot erase an earlier Node failure. Runtime events do not identify kernel holders or establish the original 1007 ms failure's cause; selected-interpreter identity and native Bun owner tracing remain unobserved/unsupported.

## B supplemental startup probe: reviewed and locally integrated

The three-path after-canonical probe is now independently **SPEC PASS → full QUALITY PASS** and applied without changing production recovery code, canonical order, deadlines or assertions. Its earlier observer-lifetime and malformed-UTF-8 defects were reproduced RED and closed against unchanged reproducers; six lifecycle held-outs and 33 byte/schema cases per actual runtime passed. See `docs/security/dependency-remediation/accepted-runtime-contracts/recovery-startup-observation/` for the exact source, review and execution evidence.

Fresh parent integration passed the canonical 1524-test suite (66 skipped, 54 files), canonical build, 156-root strict checks under actual Node and Bun, 14 startup selftests and eight original observer selftests. Four separately labelled local 200/5000 ms comparisons executed authored public work and retained their uncontained/unqualified status. Installed-package evidence was carried forward only after every regular tarball entry matched the current build; it was not represented as a new smoke execution.

**CI remains open.** At `fd9b9025b496ad7a3a8e2f9ca1d0ffa68c289c07`, run `43605300047` had 55 recovery and two transport test failures. The 32 unexpected missing-stage receipts are complete: 31 were not admitted, and one was admitted without reaching its target stage. All reported reaped supervisors, no authored work and no forced pipe closure; these facts do not prove why a stage was unexercised. B was absent from that run. Diagnose the transport deadline/shared-waiter failures and recovery startup/fixture behavior independently, then read the new after-canonical CI probe alongside—not in place of—the original failure evidence. No local warmed observation identifies an earlier CI interpreter or closes the historical scheduling/descriptor cause.
