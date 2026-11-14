# macOS recovery CI follow-up

## Observed failure, not an inferred cause

At PR head `9697a6fd9707a2787ab417cd72213d3fa7d7be66`, Actions run `43587714173` passes both Linux JavaScript jobs and both Python jobs. The macOS JavaScript jobs fail recovery cases: 30 failures in the Bun package-launcher matrix, 32 in the Node matrix. Both full Vitest suites are actually hosted by Node 22.22.3; native Bun children are separate controls.

The in-place Node observer is present in both uploaded macOS artifacts. In the Bun-labelled job, the observed supervisor exits around the 200ms deadline, stdout reaches EOF, but stderr and the proof stream do not; the unchanged 800ms teardown later forces closure. The other job observes eventual EOF and truthful not-started cleanup. Several transition/admission fixtures never reach their intended stage before the fixed deadline.

These observations do **not** identify the kernel holder, selected interpreter version, root cause, or a valid production repair. The literal `/usr/bin/python3` is recorded as a regular file; no earlier Python/xcrun/version probe warmed the canonical test. The runner image is macOS 15.8.2 / virtual Apple M1, distinct from the local development Mac where the full suite passes.

## Bounded work plan

1. Bind downloaded logs, the specific canonical-case JSON, static source hashes and actual merge checkout; distinguish package launcher from test host.
2. Review the current admission/lease/cleanup code and correlate all failed assertions. Separate bootstrap latency, clock-domain and transport hypotheses instead of inferring one from a missing stage file.
3. Run only small owned local controls that distinguish mechanisms. Label their limits: local controls and warmed probes are not causal proof of the original CI invocation.
4. Propose the smallest causally justified repair, or a bounded opt-in diagnostic for the actual CI image. Keep the first canonical invocation in place; put supplemental comparisons afterward.
5. Preserve the 200ms execution lease, 800ms teardown, 1800ms fixture bounds, assertions and pre-forced-close evidence. Do not substitute another interpreter/runtime, skip cases, restart a clock or certify destroyed pipes as successful cleanup merely to obtain green CI.
6. Independently review any implementation and require fresh actual macOS matrix execution before merge.

The Nix fixed-output mismatches are separate: both hashes were measured on native hosts in run `43587714131`, bound to the exact package/lock/flake inputs. The subsequent native build and installed-surface run `43590280284` passed on both hosts at `9ca0fee`; this does not resolve the recovery failures.

## Evidence-led implementation wave

Diagnosis `deleg_e3151dd8` reconciled the original 62 failing outer tests without attributing a kernel holder or a selected CI interpreter. Parent verified its 47 artifacts and 17 unchanged source bindings, then independently repeated its three raw controls. The incomplete-stage negative and atomic-publication positive reproduced; all subject processes were already absent before fixture cleanup and unrelated sentinels survived. These raw PREPARED-anchored controls are not caller-inclusive startup qualification.

A later Linux failure in run `43590280244` is **different**: its stage record is null, but real child validation rejected caller EOF with about 197ms left, no work, and complete cleanup. Do not claim that the empty-file repair resolves this missing auxiliary witness. Reproduce and preserve the actual validated boundary rather than weakening the test or inventing a stage event.

### Lane A — fixture repair and failure evidence

Own only `test/audit-recovery-execution-admission.test.ts` and `test/fixtures/bun_recovery_ownership.ts`.

1. Deliberately compose the diagnosed exclusive-temp/rename stage publication with the opt-in missing-stage/FD-set observations. Their supplied patches share a preimage; validate the composition rather than blindly stacking them.
2. Add deterministic held-write regression coverage in the existing admission-test file. Execute the actual generated fixture and unchanged supervisor; require complete publication before existence triggers EOF, preserving original deadlines and assertions.
3. Diagnose the later Linux null-marker case using its actual receipt. If a fixture repair is supported, keep authoritative child validation, positive remaining lease, real EOF, no-work and cleanup assertions; no fabricated marker or silent weakening of a required boundary. Otherwise retain the case as unresolved.
4. Verify diagnostic-disabled behavior, diagnostic-enabled failure snapshots before finally, strict typing, focused actual Node/native-Bun controls and source complements. Produce an owned candidate for independent SPEC then QUALITY review; no live integration by the worker.

### Lane B — bounded after-canonical startup observations

Own only `.github/workflows/ci.yml`, a new `test/helpers/recovery-startup-observation.ts`, and its new `test/helpers/recovery-startup-observation.selftest.mjs`.

1. Add an explicitly supplemental probe **after** the existing canonical suite/smokes, never before them. Preserve the current canonical commands, two-worker setting, runtime pins, first-case observer and all existing assertions/deadlines.
2. Exercise the real `executeRecoveryWorker` caller under actual Node and native Bun with a finite short-budget/positive-control comparison. The short caller budget stays 200ms; any separate 5000ms positive control must remain separately labelled, not relabel a short failure as success. Keep literal `/usr/bin/python3` selection and all production source unchanged.
3. Use bounded, owned opt-in Python self-observation for selected version/executable and process-local entry/module/PREPARED milestones, with explicit instrumentation effects and clock-domain limits. Preserve caller-inclusive budget accounting; never send a fresh raw S200 merely because PREPARED arrived. Record unsupported/missing observations as unknown.
4. No broad process/FD scans, privileged tracing, interpreter substitution, extra production hooks, or cleanup certification from destroyed streams. Do not let diagnostic failure hide canonical failure or prevent preservation of the original observations.
5. Provide source/runtime-bound receipts, safe owned cleanup, focused regression tests, strict typing and actionlint before independent review. Actual macOS execution remains a parent gate.

The lanes have disjoint write sets and can proceed alongside the separate AST review. The parent composes only independently accepted changes, runs settled-source tests/build/package checks, publishes the candidate and reads the exact native CI receipts. The main branch and alert-closure gates remain closed until required CI actually passes.
