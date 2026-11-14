# Independent execution-lease specification review

## Verdict

**SPEC_PASS for the scoped execution-admission repair and named retained ownership contracts.** The reproduced stale-E/EOF/abrupt-parent-SIGKILL and pre-exec queued-cancellation defects are closed in the frozen candidate. This is not an unconditional real-time cleanup guarantee or release/CI acceptance. The additional post-startup owner-pause diagnostic below is an explicit qualification, not passing complete-cleanup evidence.

No code, tests, configuration, documentation, skills, manifests, locks, workflows, live services or external evidence were edited. All execution used call-owned scratch clones and caches.

## Exact source binding

HEAD remained `dcf98c159b671b4d08c3759215c4371502f49fc0`. Independently authenticated the worker's entry bytes, replayed only its five-file patch, and matched all 12 final candidate hashes against the frozen closure, live checkout, parent mirror, independent mirror and independent patch replay. All 244 public-source snapshot files matched their manifest before and after execution. Canonical `process_anchor.py`, `__init__.py`, worker and loader remain unchanged.

Primary source digests:
- `scripts/phase5_e2e_runner.ts`: `6204b26a3a8218cd1da6ce03ec856065656d904cc9b936d9110900f8d79ac875`
- `scripts/phase5_recovery_process.py`: `52a00fb5b891752f8d53f8d8b08fae3ce467d392461b6524f762376ade344bbf`
- `test/audit-recovery-execution-admission.test.ts`: `6a516aebe57bcd1580adb8d83a99d5d5e4bb41a5685f5d334e06d2d721b79010`

See `source-binding.json`, `final-source-binding.json`, `dependency-links.json`. Dependencies were 274 per-package aliases inside a real owned node_modules directory; Vite and Bun caches were owned before execution. No installs or downloads.

## Reproduced RED, then repaired behavior

1. **Original stale-E:** byte-identical original harness, relocated only, freshly failed against entry source: actual-user-work at **1740.536ms**, original 200ms budget / 1300ms hold / 50ms runnable-child slice / 1800ms guard retained. Positive work and no-E/EOF controls behaved correctly. The candidate harness changes exactly one S transport literal to `S200\n`, retaining every assertion and bound. It passes: positive work **118.704ms**, no-E/EOF no work, stale-E/EOF no work and natural cleanup **368.544ms**. Every sentinel survived; no fixture kill or guard fire. `original-replay-summary.json` and `original-transport-adaptation.patch` retain the comparison.
2. **Queued cancellation:** freshly replayed the exact retained pre-handoff RED source and its test in a separate owned clone. One selected test genuinely fails with actual-user-work while the child still has **146.936ms** of lease; 43 unselected tests are pending, not passes. Real processes were reaped/absent and sentinel alive, no guard or fixture kill. `queued-cancel-red/receipt.json` binds source hashes and failure. Candidate focused test and independent K/T/unexpected-byte final-gate controls refuse with a still-valid lease; the candidate never consumes the pre-exec revocation in the supervisor. The held-out reader-handoff pause also reaches the actual poll boundary and refuses work before expiry.

## Clock, protocol and ownership checks

- TS computes floor(original deadline minus current time) only after verified PREPARED (`phase5_e2e_runner.ts:390-404`). Python anchors before PREPARED and caps to that anchor plus budget (`phase5_recovery_process.py:310-347`). No Node/Bun/Python epoch comparison or per-stage timeout reset exists.
- Three fresh real `/usr/bin/python3` fork-clock controls prove the initialized 3.9.6 clock is inherited. Actual executable resolves to `/Library/Developer/CommandLineTools/usr/bin/python3`; this does not establish a shared epoch for separately launched runtimes.
- A real delayed-PREPARED control verifies the fork/child inherit exactly `prepared_at + remaining_ms/1000`; its 800ms original budget is conservatively charged, with timeout/cleanup at **694.406ms**, not restarted on delivery. Valid fragmented S executes; fragmented expiry, delayed consumed S, delayed fork and slow real canonical containment do not refresh the lease or run work.
- Child's duplicate is genuinely nonblocking and CLOEXEC. Real empty/live, EOF, K, T, unexpected-byte, EBADF and expired-deadline controls exercise the exact gate function (`:214-232`), not mocked reads or timestamps.
- Canonical containment precedes the final check (`:391-397`). The existing CLOEXEC launch-writer EOF authorizes supervisor reader handoff only after exec or child exit (`:540-589`). No supervisor pre-exec K consumption occurred in held-outs.
- Refusal blocks its own TERM, successfully kernel-joins its existing same-session supervisor group, and only then sends TERM and CONT while membership pins authority (`:235-251`). Observed real signal calls prove pinned membership and blocked TERM. A real cross-session join fails EPERM and authorizes **zero** killpg attempts; unrelated sentinel remains alive.
- Real EBADF on child gate, supervisor control and proof write enters safe refusal/retained cleanup. Proof failure supplies no fabricated cleanup message; actual waitpid evidence is retained. A deterministic second actual TERM at the AST-resolved StartupCancelled exception-transfer boundary does not unwind owners.

## Focused result and runtime identities

Fresh direct **Node-host Vitest: 7 files, 114 passed, 0 failed, 0 skipped**: all 70 retained plus all 44 new. Canonical source config retained through an isolated cache-only wrapper; maxWorkers=2, allowlisted synthetic environment, offline/model-disabled flags, REQUIRE_BUN=1, no outer Seatbelt wrapper.

- New surface: 15 raw cases + 7 malformed/budget cases + 22 real hosts.
- Actual Node **22.22.3**, `/Users/shersingh/.hermes/node/bin/node`: 11 hosts (valid 1 / alive 3 / graceful 3 / actual hard SIGKILL 4).
- Actual Bun **1.3.8**, `/Users/shersingh/.bun/bin/bun`: the same 11 hosts. Bun's emulated Node 24.3.0 field is not the Node test-host version.
- All 44 new pre-finally observations: processes naturally absent, sentinels alive, guards unfired, no forbidden work, zero finally kills. Both real stopped-supervisor cases recover without fixture CONT. Six live-helper times: **222.317–254.134ms**; existing 200/800/1800/<1500 contracts are unchanged.
- EOF-only gate refusal occurs with **197.672ms** remaining. Expired forwarded E with an open caller separately refuses at **-201.991ms** remaining. Causes are not conflated.
- Additional executions, not additional unique top-level tests: native Bun startup 18 + transition 10 + tree 8, and four native descriptor/GC/signal scenarios / 13 ownership calls. The 20 retained transition and 14 retained gated observations remain covered. Four pre-existing deliberate escape descendants require disclosed finally kills (escaped-pipe/new-session once per runtime); those are not natural-cleanup proof.
- **15 independent semantic held-out cases pass**, plus 7 direct real gate controls, 3 fork clocks and 1 cross-session join control. One initial reviewer proof-error wrapper incorrectly closed fd3 again during retried READY; retained original error, corrected only own wrapper to close once, and ran a fresh successful fixture. No product assertion was weakened or source changed.

Raw evidence/counts: `focused-invocation.json`, `focused-tests.{log,json}`, `focused-observations.json`, `review-evidence-summary.json`, `heldout-final-results.json`, per-case `heldouts/`, `heldouts-proof-corrected/`, `fork-clock-results.json`, `gate-controls.json`, `join-refusal/`.

## C1 — additional cleanup qualification, not concealed as a pass

A separate actual Node/Bun diagnostic pauses the supervisor **after startup**, inside its real pre-exec readability poll, for 1300ms. Real parent SIGTERM queues K and activates normal caller cancellation. The child refuses K with **63.116ms (Node) / 2.273ms (Bun)** still available, authenticates TERM/CONT to its owner group, and executes no work. However the post-startup signal handler only latches cancellation; it does not interrupt this artificial hold. The sole WNOWAIT/reap owner is still held when the helper's existing 800ms teardown expires.

Consequently the helper returns at **921.149 / 926.373ms** with **truthful incomplete proof** (`supervisorReaped=false`, `childReaped=false`, `pipesClosed=false`, `forcedPipeClose=true`); actual supervisor and exited-but-not-yet-reaped worker remain observable at return. Both naturally disappear after the hold; real waitpid is recorded, no fixture signal or kill. This is not late execution and does not reopen the two repaired execution-authority defects. It is a concrete counterexample to any stronger claim that every arbitrarily paused post-startup cleanup owner is fully reaped within the caller's teardown allowance. Keep this qualified separately: prevention of unauthorized work is stronger here than bounded successful cleanup certification. Do not advertise these two diagnostic invocations as complete-cleanup passes. `monitor-pause-diagnostic-results.json` and per-runtime `at-helper-return.json` preserve the failure states before any cleanup.

## Boundary and remaining gates

Successful last child validation linearizes revocable admission, not atomic pipe/time/exec. Independently scheduled supervisor owns later revocation. No hard-real-time guarantee is possible while necessary trusted owners cannot run; arbitrary exec cannot autonomously resume a permanently stopped supervisor. Original hard parent SIGKILL/stale-E cases are explicitly included and pass, not excluded by this boundary.

Parent still owns type-only proposal rebase, canonical build/full workspace suite, integration/publishing, and **actual candidate-head native Mac CI**. This local review does not qualify the separate failed bootstrap/open-stream run `43503677544` (Node job `132155679910`, Bun job `132155679594`). The pre-existing 61 TypeScript diagnostics and pending 19-path type proposal are not counted as lease failures. Quality review may begin now that this scoped spec gate has passed; it must retain C1 and these qualification boundaries.
