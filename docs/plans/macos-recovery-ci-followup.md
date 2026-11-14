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

The Nix fixed-output mismatches are separate: both hashes were measured on native hosts in run `43587714131`, bound to the exact package/lock/flake inputs. Updating those two measured values does not resolve the macOS recovery failures.
