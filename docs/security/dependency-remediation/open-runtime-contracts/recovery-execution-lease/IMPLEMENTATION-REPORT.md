# Recovery execution-permission candidate

**Local candidate verified, not self-accepted.** Independent spec/quality review, parent integration/type-only rebase, canonical suite/build and native CI remain separate gates.

## Deliverables and attribution

- `candidate-receipt.json`: exact 12-file hash closure, five-file entry delta and verification handles.
- `entry-delta.patch`: ONLY this writer's delta against the reauthenticated dirty entry snapshot, not against HEAD. Entry matched the prior nine-file freeze 9/9. HEAD remains `dcf98c159b671b4d08c3759215c4371502f49fc0`.
- `candidate-source/`, `candidate-source-manifest.json`: exact bytes exercised in the final run.
- `DESIGN.md`, `REVOCATION-DESIGN.md`: ownership/linearization tables recorded before the respective production changes.
- `candidate-evidence-summary.json`, `candidate-observations.json`: programmatically counted pre-finally observations, actual helper receipts, kernel-reap observations and finally kills.
- `candidate-invocation.json`, `logs/candidate-focused.{json,log}`: exact clean environment/argv and complete final results.

Live changes are limited to:
1. `scripts/phase5_recovery_process.py` (+156/-26 against entry).
2. `scripts/phase5_e2e_runner.ts` (+6/-2).
3. `test/audit-recovery-startup.test.ts` (three S-frame transport substitutions).
4. `test/audit-recovery-tree.test.ts` (one S-frame transport substitution).
5. New `test/audit-recovery-execution-admission.test.ts` (380 lines, 44 tests).

The existing semantic AST transition fixture is unchanged. Its before-fork/fork-return/registered/before-ready/after-ready assertions still run at their actual semantic transitions. No stale TypeScript mirror was applied. Native Bun ownership/GC fixtures, the canonical worker/loader/kernel policy, `process_anchor.py`, manifests, locks, docs, workflows and unrelated accepted work are unchanged by this writer. No helper dependency, install, external harness, subdelegation, stage/commit/push or remote job was used.

## Mechanism and execution boundary

- S is now the bounded frame `S<remaining integer ms>\n`. Python anchors at PREPARED, which necessarily precedes the caller's remaining-budget subtraction. Charging the duration from that earlier anchor is conservative. Node, Bun and Python clock epochs are never compared. The inherited local deadline is not restarted after E or startup.
- E creates no replayable execution capability. The child independently expires while waiting for the private gate. After E and canonical no-fork preparation, its last trusted gate validates the inherited deadline and a CLOEXEC duplicate of the caller channel. EOF/data/error denies admission.
- A refused gated child can wake a paused or truly SIGSTOPped supervisor after abrupt caller death: block its own TERM/INT, **kernel-join the existing supervisor group in its own session**, send TERM then CONT only while its own membership pins that group, then exit. Join failure authorizes no signal. There is no guessed/recycled numeric PID/PGID fallback. Existing runtime-owned native TERM/CONT remains intact, including Bun's platform-number-through-owned-handle workaround.
- Pre-exec cancellation bytes cannot be erased by the supervisor. It polls without consuming until the existing CLOEXEC launch-writer EOF proves exec or child exit crossed the reader-handoff boundary. Any pre-exec readability/error cancels anchored work while leaving K/T/EOF independently visible to the child.
- Terminal control/proof errors enter anchored cleanup. Reentrant cancellation is latched before raising, so a second TERM cannot interrupt cleanup transfer. A Darwin stopped-child non-exit observation is not relabeled as an exit: the recovery caller records the observer error, cancels the still-retained exact child, and obtains actual subsequent exit/reap evidence. The canonical observer is unchanged.

`logs/clock-proof.json` verifies the actual macOS Python 3.9.6 clock initialized before fork has the same inherited epoch in parent/child. This is not an assumption about separately launched runtimes.

Admission linearizes at successful final child validation. POSIX does not atomically combine pipe liveness, time and execve. Cancellation after that validation revokes already-admitted work through the independently scheduled supervisor. A stopped supervisor across E is protected by the gated child; a stopped/delayed child across/after its gate is protected by the supervisor. There is no hard-real-time guarantee while every trusted cancellation agent is unscheduled, nor automatic recovery of a permanently stopped supervisor after arbitrary exec has replaced the Python gate. Actual abrupt caller SIGKILL in the reproduced stale-E domain is explicitly included and tested without exit-hook assistance or fixture CONT.

## Retained RED versus current controls

The independent original harness remains byte-identical in `original-reproduce_stale_e.py`; external evidence roots were never written. Its fresh entry-source run exited 1: stale E plus EOF performed actual-user-work after 1732.263ms, with the 1800ms guard unfired. Its positive really executed and EOF without E refused.

`post-repair-final/` copies the current exact Python closure and adapts ONLY `S` to `S200\n`. The original behavioral assertions, 1300ms hold, runnable-child slice, 200ms execution budget and 1800ms guard are preserved. Exit 0: positive work at 104.938ms; EOF-without-E no work; stale E plus EOF no work at 388.181ms. All supervisor/workers absent, sentinels alive, zero fixture cleanup kills.

An additional real RED was found during review of the first green checkpoint: consuming K before the child gate erased cancellation while the supervisor paused. Actual work ran with approximately 65ms of lease still available. `revocation-red-receipt.json`, `revocation-red-source/` and `logs/revocation-red.log` retain it. The new poll/handoff design closes that distinct case; the final record has no pre-exec K consumption, no work and an actual reap.

## Final fresh verification

Direct Vitest, clean allowlisted environment, synthetic HOME/HERMES/cache/TMPDIR, offline/model-disabled flags, required actual Bun, no outer Seatbelt sandbox, maxWorkers=2:

**7 files, 114 top-level tests passed, 0 failed, 0 skipped = all 70 retained + 44 new.**

- Retained direct Node tests: startup 18, transition 10, tree 8, audit 10, phase5 17; seven native-Bun wrapper tests also pass.
- Retained actual Bun: startup 18 + transition 10 + tree 8 = 36 native tests, plus four native fixture scenarios / 13 actual ownership calls, retaining descriptor reuse and active-call GC coverage.
- New Python/raw protocol cases: 15 execution-boundary/error cases + seven malformed/duplicate/truncated/oversized/cancel-priority budget cases.
- New real Node 22.22.3: 11 hosts (valid 1, alive 3, graceful 3, hard SIGKILL 4).
- New real Bun 1.3.8: the same 11 hosts. Bun's emulated `process.versions.node=24.3.0` is not mislabeled as the Node test runtime.
- All 44 new observations: natural pre-finally absence (or no fork for invalid frames), sentinels alive, no negative work, guard unfired and **zero finally kills**. Both OS-STOP supervisor cases clean up without fixture CONT. Valid raw/Node/Bun positives actually write actual-user-work and use explicitly separate 5s readiness controls.
- Six new live-caller helper receipts: 216.620–243.976ms, with original 200ms execution / 800ms teardown / 1800ms guard / <1500ms helper contracts intact.
- EOF-only child-gate isolation: real queued EOF observed before forwarding; child refuses with **197.459ms still available**. Expired-forwarded/open-parent isolation independently refuses with an expired lease. Neither inference depends on the other cause.
- Real EBADF control-read and proof-write cases actually reap the child. The broken proof channel supplies **no invented cleanup receipt**; the fixture records the real waitpid return and pre-finally absence.
- All 20 retained transition and 14 retained gated cases pass with no late work and natural cleanup. Canonical no-fork passes on both runtimes. Forced-close positives remain honestly `pipesClosed=false`, `forcedPipeClose=true`.
- Four existing deliberately escaped, uncertified fixture descendants still require recorded finally kills (escaped-pipe and new-session in each runtime). These are disclosed in the summary and are NOT natural-cleanup evidence for this repair.

Actual current Python bytes compile under the real interpreter; `git diff --check` exits 0. Root TypeScript still exits 2 with **61 pre-existing diagnostics before and after, no new diagnostics**, normalized by source line rather than shifted line numbers (`ts-candidate-diagnostics.json`).

## Iteration evidence and limits

Earlier logs are retained, not retried unchanged or relabeled green. The first added boundary tests sometimes sent EOF before the child reached the requested boundary; they now wait for that actual stage before closing. The first integrated run exposed real duplicate-TERM unwind and Darwin stopped-child observation failures; production cleanup was repaired rather than assertions weakened. A cancellation probe initially also counted legitimate K consumption after child-exit launch EOF; it now distinguishes the actual reader-handoff phase. An EOF-isolation fixture's artificial 50ms child sleep once exhausted the lease: execution was safely denied, but the intended unexpired-EOF coverage was unmet. That probe now observes real queued EOF before forwarding without an artificial child sleep; its 200/800/1800 bounds and refusal assertions were not increased or weakened.

The separate failed Mac CI bootstrap/open-stream case is neither diagnosed nor qualified by this repair. Per parent steering, its exact-failed-head read-only diagnosis belongs to the other lane. No interpreter change or bootstrap hypothesis fix was made here. Parent owns independent review, mirror rebase, integration and exact-head native CI; this report makes no release/merge acceptance claim.
