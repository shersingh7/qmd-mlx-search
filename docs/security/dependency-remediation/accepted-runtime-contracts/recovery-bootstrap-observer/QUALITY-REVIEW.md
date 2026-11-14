# QUALITY REVIEW — PASS_WITH_QUALIFICATIONS

**Verdict:** `QUALITY_PASS_WITH_QUALIFICATIONS`, after the exact-source `SPEC_PASS_SCOPED` report was recorded. No blocking or source-fix finding. This approves the bounded test-only observer proposal, not release integration, causal diagnosis, or CI qualification.

Source binding is unchanged from SPEC: patch `69810f46ab0a458e53dd9a13a951a82e11379611c2bcdfa366939f1498cb174f`, source manifest `d4f94e439de1704994cd2eb94e6b697d55c0625a833f7881832071529f0fed87`, and the exact four proposed hashes listed in `SPEC-REVIEW.md`. QUALITY prerequisites and the SPEC report hash are retained in `evidence/quality-prerequisites.json`.

## Findings

No P1/P2/P3 defect requiring proposal changes was found in the scoped review. Retain the following **qualifications**, not silently promoted guarantees:

1. **Q1 — Internal runtime observation is deliberately narrow.** `test/helpers/recovery-bootstrap-diagnostics.mjs:49-119,127-153` relies on the pinned Node constructor diagnostic notification and Node-private close/handle fields. The unavailable status is appropriate if a different runtime no longer exposes it; it is not permissible to install a spawn wrapper, borrow another FD owner or infer kernel holder identity. Native Bun is explicitly unsupported. Reproducer/control: `harness/run.mjs heldouts <new-label>` verifies the actual 22.22.3 constructor event; `bun-fast-on <new-label>` verifies actual Bun 1.3.8 unsupported-owner behavior. Counters observed in local Node controls are 4/4 natural, 3/4 authored FD3-only writer, and 2/4 actual helper fallback, not interchangeable causal classifications.
2. **Q2 — Artifact upload success is weaker than acceptance.** `.github/workflows/ci.yml:90-101`; supplied `IMPLEMENTATION-REPORT.md:80-94`. `if-no-files-found:error` can succeed with metadata/TAP but no fast-case JSON. The documentation correctly leaves exact-candidate artifact/source readback to the parent. Do not count a skipped/failed suite as qualified just because after-canonical controls or upload succeeded. Reproduction condition: suite never reaches the fast case while static metadata and always-run controls produce files. No CI was executed in this review.
3. **Q3 — Preserve truthful negative evidence.** Helper `:27-46`, production `scripts/phase5_e2e_runner.ts:336-357` (unchanged) and existing lease C1. A pre-destroy receipt with null timestamp is the correct data source; final return snapshots cannot reconstruct earlier counters. Destroyed/closed does not establish EOF or complete cleanup. Our actual fallback probe succeeded **as a negative control** with failed transport at 946.096916 ms and 2/4; it is not clean-cleanup proof or the historical CI cause. The prior 1300 ms owner-pause qualification remains unchanged.

## Review assessment

- The two lifecycle hooks plus one exception-safe finally insertion are small and lexically scoped. Original assertions/calls/order remain byte-exact when hooks are removed. No recovery/protocol logic was folded into the observer.
- Bounded scalar projection is separate from IO and metadata. No argv/config/env/payload/error-message dumps appear in observer reports. Static metadata uses a limited explicit allowlist and hashes selected source bytes. PR head cannot become shell source. Actual source head/event SHA/checkout remain separate, and missing checkout metadata remains unavailable instead of guessed.
- `errorMonitor` preserves ordinary error behavior; independent tests exercised it **while active**, beyond the provided selftest's post-disposal ordinary-error check. Exact raw-listener identities/methods/flow restored. Receipt-projection exceptions and artifact-directory IO failures retain primary errors and remove installed observation state; disabled mode did not touch a throwing receipt getter or create artifacts.
- Helper controls intentionally use fixture drains/timers/signals on their own children, not observer operations. Selftests are explicit after-canonical Node tests, outside unchanged Vitest `*.test.ts` discovery. They do not become cold-bootstrap warmups.
- Single local on/off observations (248 vs 222 ms) establish only that the original assertions passed in these executions. They do not establish zero overhead, a statistical performance equivalence, or cold CI behavior.

## Actually executed QUALITY gates

All commands use `harness/run.mjs`, owned explicit allowlisted environments and source binding. Raw argv/env/cwd/exit receipts and logs are retained for each label.

| Gate / label | Actual result |
|---|---|
| Supplied official actionlint 1.7.12, `independent-actionlint` | exit 0; no diagnostics; `-shellcheck=` explicitly disables unavailable shellcheck |
| Actionlint binary authentication | SHA-256 `8db11704dc296f096216db4db65d86cd7f0ebfdf4c38453a1da276b137b88388`, equal to supplied receipt; no install/download |
| Strict helper JSDoc/JS typecheck, `independent-helper-types` | exit 0: `tsc --noEmit --allowJs --checkJs --strict --skipLibCheck --module NodeNext --moduleResolution NodeNext --target ESNext test/helpers/recovery-bootstrap-diagnostics.mjs` |
| JS syntax, `independent-helper-syntax`, `independent-selftest-syntax` | both installed Node `--check` commands exit 0 |
| Bounded same TS entry comparison, `independent-bounded-types-final` | exit 0 for unchanged diagnostics: **65 before → 65 after, 0 new**; 89 identical TS entrypoints; 668/669 compiler source files; 11 JS entrypoints excluded symmetrically; exact diagnostic objects match |
| JavaScript-job shell syntax (SPEC gate) | 11 run scripts pass `/bin/bash -n` |

The bounded validator keeps the product tsconfig unchanged. It never adds archived documentation `.mjs` entrypoints, and guards both lexical and real source read paths to its own snapshot or the exact explicitly approved per-package read-use dependency targets/aliases. Imported observer JS is still included (the after-source count increases by one). This is **not a green full/root typecheck**, not the parent's distinct canonical 61-error baseline/type proposal, and not a replacement for canonical build/package/full suite execution.

The initial bounded-validator run safely refused an already-approved own mirror dependency alias while checking the original tree. Raw exit-1 evidence and the initial harness remain. The corrected guard adds only the explicitly enumerated OWN original/mirror per-package aliases, retaining realpath closure protection; the final bounded run passed unchanged-diagnostic comparison. It did not allow arbitrary `node_modules` paths or other live source. The earlier supplied exploratory whole-snapshot 67-diagnostic runs are disqualified and were not replayed.

## Remaining acceptance

Parent owns canonical build/full suite/type/package integration and actual exact-head macOS CI. Require the actual Node fast artifact in both matrix jobs, owner/runtime/log/source/hash readback and unmodified assertions, not just metadata/TAP or native later passes. Require candidate head, event SHA and actual merge checkout/file bytes independently. Original Node run 43503677544 remains an authenticated negative (1007 ms, ownership assertion, cause unproven); these local controls neither reran it nor closed its cause gate. Selected system-interpreter version, prepared/admission event timestamps and native/kernel holder tracing remain unavailable.

No proposal/source fix was made. All new artifacts and reviewer harness corrections are confined to the reviewer root. No CI, network/install, live edit, GPU/model/private-knowledge, config/service/skill write, staging/commit or publishing action was performed.
