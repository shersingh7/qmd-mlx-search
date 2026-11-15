# Final TypeScript composition — proposal awaiting independent review

## Scope and frozen inputs

The owned candidate is `candidate/` in this directory. It composes the accepted T1 17-path proposal, the delivered T2 two-path metric rebase, and T3's three development-dependency/lock changes over the current 295-file public source baseline. The resulting snapshot has **298 public source files and 22 changed paths**. All three original patches were actually checked/applied, and the final combined patch was replayed in `patch-replay-final/` with all 298 postimages matched. No TypeScript or dependency changes were applied to the live checkout.

Input deliveries remain unchanged:

- T1: `/Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p`; 17 partial paths, two held. Index SHA256 `104aaf1f4b4c56af6415aa2dbce692998d635a38b97abb1762d2397df8e661ac`.
- T2: `/Users/shersingh/.hermes/cache/scratch/qmd-types-metrics-ppnll4m1`; 194 indexed artifacts verified. Index SHA256 `b64e19f60ac5ec3b7675f802a03f92cc0e632bd9aa7b189973c975c6f1bf272d`. Original patch SHA256 `222dcc9efbb2fe4b29b1f5d87ef3b68dafef161b83efb75038f3773f30e1eda2`.
- T3: `/Users/shersingh/.hermes/cache/scratch/qmd-types-declarations-1yp0w89j`; 263 indexed artifacts verified. Index SHA256 `35b02bf22d08f008adc60c684af3a2d56ae83260df6e97c79a697c7b91bc11dd`. Original patch SHA256 `174a639eec01e7b135deb5823335fb7b6b41360b059b456709b9cc4037dcf7f3`.

T3 metadata has a separate independent review (`deleg_3f9188dc`). Its three official declaration packages were authenticated against archived SHA512 SRI/SHA256 and actual tar members. Only 44 genuine declarations/top-level metadata files were copied into owned package directories; existing dependencies remain individual read-use aliases in owned containers. No lifecycle script, native/model materialization or network action was used for this composition.

## Parent-found report defects and bounded repair

The unmodified T1+T2+T3 composition passed its guarded compiler invocations but failed all three new tests in `heldouts/metrics-inputs.test.ts`:

1. A sparse score array could produce `identical: true` without a score at every document position.
2. Matching sparse document-ID arrays could be certified despite a missing ID.
3. An explicit undefined query detail threw while reading `queryId`, rather than refusing equivalence.

The source-bound RED is `evidence/report-boundaries-red.json` (three actual test failures, no skips). Original input delivery files and `pre-parent-candidate-manifest.json` retain the predecessor identities.

The parent changed **only `verifyRetrievalEquivalence`** beyond T2's already-delivered runner hunks. Dense `for...of` validation now visits query and document holes, validates nonempty unique IDs and short-circuits; a bounded indexed score loop visits every position and rejects missing/nonfinite scores. Valid dense score/tolerance behavior is unchanged. No casts, compiler suppression, deadline changes, recovery protocol changes or whole-file restoration were introduced. Three permanent tests were inserted into `test/phase5-recovery.test.ts`; existing T2 test bytes remain intact. The only other parent source change is a comment explaining that the existing Vitest string policy matches Zod-prefixed packages, not exclusively `zod`.

`evidence/parent-source-preservation.json` records actual TypeScript 5.9.3 source and emitted-JS complement equality outside the one authorized metric function, exact preservation of previous tests around the insertion, and identical emitted JavaScript for the Vitest comment clarification. Emission comparisons explicitly remove comments; full-source complement comparison separately protects unchanged bytes. The `parentSemanticChanges` manifest list includes the comment-only path; it does not imply a runtime policy change there.

## Actual executions — not summed into unique coverage

| Invocation | Result |
|---|---|
| Parent boundary RED on pre-parent composition | 3 failed, 0 skipped, exit 1 |
| Same unchanged three heldouts, repaired candidate, Node | 3 passed, exit 0 |
| Same unchanged heldouts, actual Bun host | 3 passed, exit 0 |
| Node focused union: prior 13 files plus native-Bun ownership wrappers | 256 passed, 40 existing skips, 14 files, exit 0 |
| Actual Bun focused union: prior 13 files | 227 passed, 40 existing skips, 13 files, exit 0 |
| Canonical build script, repaired candidate | Exit 0 |
| Guarded canonical compiler, Node and actual Bun | Each exit 0 / zero diagnostics, with two historical live-import refusals retained |
| Parent source/emission preservation probe | Exit 0 |
| Final 22-path patch check/apply in fresh owned replay | Both exit 0; all 298 postimages match |

The compiler refusals mean **whole-root acceptance is not established**, even with zero diagnostics. Historical `.mjs` imports, all compiler options and roots remain unchanged. The parent must perform canonical compilation in the actual integrated public checkout after review. C1 remains qualified: truthful incomplete cleanup under an artificially paused sole owner is not universal complete-cleanup acceptance.

Actual command arrays, clean environments and raw stdout/stderr are in `evidence/*-execution.json`; Vitest JSON and runtime/guard readback JSONL are alongside them. `run.py` starts each invocation with an empty allowlist and owned HOME/Hermes/knowledge/XDG/HF/TMP/cache directories, offline flags and at most two workers. Native-model/network entrypoints are refused by the copied test guard. Use a **new root or new labels** for any replay; do not write into this frozen delivery.

Example invocation shape in a fresh owned copy:

```
<verified-python> -I -B -S <new-root>/run.py green-check candidate <verified-node> node_modules/vitest/vitest.mjs run --config ../candidate-isolated.config.mjs --no-cache --maxWorkers=2 test/phase5-recovery.test.ts
```

For actual Bun tests use the verified Bun executable with `--bun`; for compiler checks include its explicit `--preload ../probes/compiler-read-guard.cjs`. Build before comparing canonical root selection. The runner sets the guard automatically for `build-` and `types-` labels; do not infer unguarded compilation from its exit status.

## Retained parent tooling failure

The first final-patch serialization omitted `new file mode` headers for its three new test files. Git refused the patch (`dev/null: No such file or directory`) before changing the replay. Original patch and raw failure are retained under `evidence/failed-new-file-format.*`. Only patch metadata was corrected; product bytes were unchanged. A new replay, not the failed one, supplies the final success.

## Remaining gates

Independent SPEC then QUALITY review of the composed TypeScript/metric semantics and preservation; separate T3 dependency review; actual live integration and whole-project checks; fresh complete dependency/security scans; real Linux/Darwin Nix fixed-output hashes; exact-head native image/macOS CI; authorized commit/push/merge and default-branch alert readback. No model/GPU throughput, native image or release acceptance is claimed here.
