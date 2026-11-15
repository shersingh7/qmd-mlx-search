# Recovery-fixture publication — scoped local acceptance

**Accepted scope:** `test/audit-recovery-execution-admission.test.ts` and `test/fixtures/bun_recovery_ownership.ts` only. Independent SPEC review passed before QUALITY review. This is not whole-release or actual-CI approval.

The fixture now publishes readiness only after a complete, exclusively created temporary record is closed and atomically renamed. A separate zero-delay forwarding race is witnessed at the child's actual `E` read, with the observer explicitly labelled; required stage, EOF, lease, reap and no-work assertions remain intact. Opt-in failure diagnostics preserve the primary error before fixture cleanup. Native Bun descriptor observations are parent-side numeric fstat sets, not kernel-writer identity.

## Parent integration verification

- Canonical `pnpm run test`: **1,524 passed, 66 skipped, 54 files**, with the individual reporter rows reconciled against the summary.
- Canonical `pnpm run build`: passed with active compiler read guards.
- Default strict compilation: **154 roots on actual Node and native Bun**, zero diagnostics and zero refused reads.
- Actual native-Bun changed-behavior selection: **6 passed, 43 filtered**; not the full native Vitest suite.
- Enabled descriptor-observation wrapper: **4 passed, 3 filtered**, containing **13 real native-Bun calls and 13 observations**. Original reuse and sentinel checks remain enforced.
- Installed Node and Bun package smoke tests passed against the same tarball. Bun HTTP still refuses with the documented Node remedy; that is not HTTP feature acceptance.

## Preserved failures and limits

The parent's first native focused attempt mistakenly used a compiler-only read guard for runtime fixtures. It refused call-owned temporary stage files, producing **1 pass / 5 fail / 43 filtered**. That attempt, its raw logs and cleanup readbacks are retained. A separate runtime policy permits only the public source/dependencies and the exact call-owned scratch root; an outside-root negative control remains refused. The corrected six-case run passed without changing product/test bytes. Previously completed successful compiler invocations were reused explicitly, not represented as fresh reruns.

Reviewer harness failures and refused ancestor-package probes remain in the review archive. Whole native-Bun Vitest allocator SIGTRAP, historical startup/FD-holder causality and the paused-owner C1 qualification remain open. The separate three-path startup-observation proposal failed its first SPEC review and is **not applied or approved here**.

The archived baseline CI at `a614490` still had 31/11 failed recovery cases in its Node-/Bun-labelled macOS jobs (both main suites actually Node). Its Linux JavaScript/Python jobs and native Nix installed checks passed. This fixture integration requires a new exact-source CI run; later passes must not erase earlier negative receipts.

## Evidence

- [`acceptance-summary.json`](acceptance-summary.json): accepted postimages, scope and archive hashes.
- [`local-qualification-summary.json`](local-qualification-summary.json): actual parent commands and results.
- [`independent-review.tar.gz`](independent-review.tar.gz): original SPEC/QUALITY reports, source bindings, RED/atomic-only/GREEN and held-out controls, including failures.
- [`parent-integration.tar.gz`](parent-integration.tar.gz): exact application, canonical/full-root/native/package results, corrected and failed harness attempts, and source-bound baseline CI readback.
- The corresponding `*-index.json` files bind every archive member; the parent reopened both archives and verified every declared entry. Scripts retained inside the archives are evidence, not additional production/compiler roots.
