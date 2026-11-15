# Integration wave 5 — workflow revision and accepted-code composition

## Baseline and authority

Parent owns live integration, commits/push, CI readback and merge. Workers write only fresh owned managed-scratch derivatives. The current public baseline is frozen at `/Users/shersingh/.hermes/cache/scratch/qmd-integration-wave5-sr5w96q5/baseline-manifest.json` (248 files). Its local canonical build passed; the full Node-hosted pnpm suite passed 1,404 tests with 66 explicit skips. These results are a checkpoint, not acceptance of later compositions or native image CI.

The active approved worker route is `openai-codex / gpt-6.1-sol-900k`. No provider/harness fallback, model/GPU execution, live-index access, registry publication or deployment is authorized by this plan. Existing November 11 acceptance evidence and all earlier RED receipts remain immutable.

## W1 — repair the rejected image-workflow proposal

Owner: one new implementation worker in an isolated derivative. Production scope: `.github/workflows/gguf-native-image-qualification.yml`; focused portable regressions may use `test/security/test_gguf_workflow_controller.py` and `test/security/fixtures/gguf_workflow_fake_cli.py`. No launcher, qualifier, existing workflow, dependency or application edits.

Source-bound review `deleg_15b024ec` failed SPEC on workflow `a384be7b...`; QUALITY was deferred. Authenticate the full manifest/digests rather than using these abbreviated labels. The review retained 74 passing supplied controls and four failed contracts among 34 independent challenges:

1. **S1 / P2:** direct and mapping pattern captures rebind `GGUF_IMAGE` without the AST guard noticing. Add complete direct pattern-binder coverage (`MatchAs`, `MatchStar`, `MatchMapping.rest` as applicable), while keeping the current launcher and unrelated valid patterns valid. Never import/execute launcher code or claim general Python runtime immutability.
2. **S2 / P2:** required receipt persistence fails after an already-emitted successful receipt. Decide/catch required-write outcomes before emitting final qualification; preserve prior primary errors and keep the final logged receipt consistent with the process exit. Test valid output, storage failures and combined primary/storage failure.
3. **S3 / P2:** timeout handling reaps the direct group leader after TERM and skips KILL, permitting a TERM-resistant member to write after controller return. Design an owned lifetime/termination boundary before coding. Keep a reliable lifetime anchor until the last group signal, including leader exit and interruption paths; never add delayed unanchored PGID signaling. Verify unrelated sentinels survive and distinguish killed/reaped/running states. Preserve command budgets and honest cleanup failures; do not promise unrestricted cleanup under an unscheduled or killed owner.

Sequence: reproduce the four existing RED contracts on original bytes; record the ownership/state design; implement the smallest cohesive repair; retain unchanged positive/refusal assertions; add focused source-controlled regressions; rerun the supplied 74 and independent 34 cases plus new boundary checks. Harness adapters may relocate paths or intercept the equivalent active execution boundary if ownership changes, with hit assertions—never change behavioral expectations to hide a failure. The retained S3 runner first requires a late-side-effect completion file, making it an old-defect characterization rather than a valid post-kill oracle. Preserve it and its RED unchanged; use a separate positive contract with ready/ownership evidence, no late work through the original observation window, verified owned termination or truthful failure, and an unrelated sentinel. Do not demand a killed child's normal-completion marker. All Docker, image and compiler-looking answers remain explicitly synthetic. No real Docker, probe/compiler, model, network or CI execution in this lane.

Acceptance: exact source/patch manifest; genuine prior REDs; positive post-repair contracts; Bash/Python syntax and official actionlint; independent SPEC then QUALITY reacceptance. This is revision 1 for these findings, not a restart of unrelated audits.

**W1 checkpoint:** independent review `deleg_34d9c9b4` passed SPEC and then QUALITY for the exact three-file repair. The parent authenticated 5,811 review artifacts and its 5,812-entry bundle, rechecked 12,962 frozen input records, and freshly passed the 20 independent edges. It then integrated the three previously absent files, preserved 292 existing source paths, and passed all 26 regressions against the actual checkout plus live actionlint. Counts are separate executions; original negative evidence remains retained. Real Docker/native Linux image execution and whole-release CI remain open. Durable acceptance: `docs/security/dependency-remediation/accepted-runtime-contracts/gguf-native-image-workflow/acceptance-summary.json`.

## T1 — compose the accepted TypeScript/reranker proposal

Owner: a separate isolated worker. Read the authenticated 19-path final manifest at `qmd-independent-final-deltas-ho4piwdo/report-index.json` and its source-bound review, not a guessed file list. The only overlaps with newer recovery work are `scripts/phase5_e2e_runner.ts` and `test/phase5-recovery.test.ts`.

1. Start from the current 248-file baseline, not the older full mirror.
2. Apply the already-reviewed changes, preserving all newer lease protocol/test bytes. Rebase only necessary type changes in the two overlapping paths; prove emitted runtime JS remains unchanged for production lease changes and preserve embedded fixtures/assertions.
3. Carry the accepted entrypoint realpath fix and Bun Zod-prefix policy. The prepared comment clarification may be adopted only with identical emitted JS.
4. Keep production tsconfig/dependencies and the four newly integrated observer paths unchanged. Guard compiler reads and retain complete canonical root selection; do not hide diagnostics by dropping JS/docs entries or following unreviewed absolute imports.
5. Run strict type/build checks in the owned derivative and focused relevant Node/native-Bun suites, not a redundant full repository test run. Keep exact runtime, skip, cache and environment evidence.

Acceptance: a minimal patch against the current baseline, actual changed-path union/preimages/final hashes, explicit lease preservation evidence and focused integration results. New semantic repairs require a separate finding and review; do not silently expand scope.

## T1 result and narrowly authorized follow-ups

T1 returned a **17-path partial composition, not acceptance**. The parent authenticated its 188 indexed artifacts, 248 baseline sources, 251 candidate/replay sources and all 17 current preimages. Two paths stayed unchanged because the accepted metric guards and sparse-slot regression are executable, not type-erased. The strict root still reports 35 diagnostics per host, including 31 on that held runner and four missing-declaration diagnostics. The original REDs and guard refusals remain evidence.

### T2 — finish the two held metric paths

Parent now explicitly authorizes the five already-reviewed executable runner hunks and the one already-reviewed sparse-query regression in `scripts/phase5_e2e_runner.ts` and `test/phase5-recovery.test.ts`. Start from T1's 251-source derivative. Retain the current lease, deadlines, clock domain, execution grants, observer and all existing assertions unchanged outside those exact hunks; do not replace the files from an older mirror. Prove unchanged source/emission outside precisely identified metric/CLI spans, replay the sparse-slot RED and GREEN, then run focused Node/Bun metric and recovery checks. This is a bounded semantic composition requiring independent re-review, not permission to redesign recovery or silence types with assertions.

### T3 — declare the actual development type dependencies

A separate owner may change only `package.json`, `pnpm-lock.yaml` and `bun.lock` in an owned derivative. Resolve genuine compatible Bun and picomatch type declarations, pin their versions and regenerate both package graphs using the real package managers. Anonymous public registry metadata and integrity-verified type-only package downloads are authorized in this lane; lifecycle scripts, native/model installs, GPU execution, private/live data and publication are not. Preserve existing runtime dependencies and unrelated graph entries. Keep package aliases read-only, caches owned, and show frozen graph checks plus the concrete declaration diagnostics removed.

Historical `.mjs` probes in `docs/reviews/apple-silicon-audit-evidence/` are frozen acceptance evidence, not a new rewrite target. Their absolute imports make an isolated guarded whole-root pass impossible; preserve the guard refusals, do not narrow roots or alter those files merely to report green. The parent must perform the final canonical whole-root check against the actual integrated public checkout, where those same-source imports can be source-bound. Any remaining environment or compiler problem stays explicit. Changed JavaScript lock graphs require fresh security and actual Nix-hash verification before release.

T2, T3 and W1 write scopes were programmatically checked as disjoint. All follow-up deliveries remain proposals until parent readback and independent review; no live source was changed by T1.

## Parent composition findings and final scoped reviews

The parent authenticated T2's 194 indexed artifacts and T3's 263, applied all three patches to a fresh owned public-source snapshot, and verified the 22-path composition (295 baseline files → 298 candidate files). The combined Node and Bun compilers each returned zero diagnostics, but both historical-import read refusals still prevent isolated whole-root acceptance. No TypeScript/dependency changes have been applied to the live checkout.

Fresh parent-held tests then reproduced three additional **pre-existing report-validation defects** in `verifyRetrievalEquivalence`: sparse scores could be certified as equivalent, matching sparse document IDs could pass, and an explicit undefined query detail threw instead of refusing. The original three failing tests and exact source snapshot are retained at `qmd-types-final-composition-mer5fi8x`. Parent now owns a narrow additional repair inside that function plus permanent regressions in `test/phase5-recovery.test.ts`; the T2 worker has finished, so there is no concurrent writer. Replace hole-skipping array callbacks with dense, short-circuiting iteration, preserve valid dense/tolerance behavior and all code outside the metric function/previously reviewed hunks, and require independent review of the additional semantics. A comment-only clarification of Vitest's existing Zod-prefix matching may be included with identical emitted JavaScript.

Run two independent scoped reviews in parallel: one for the metric repair/composed TypeScript runtime behavior and lease preservation; one for T3's genuine declarations and both lock graphs. Neither may narrow compiler roots, rewrite historical evidence, claim a guarded zero-diagnostic run is full-project acceptance, or approve its own implementation. The parent owns the settled full-checkout checks, fresh security audit, real platform-specific Nix hashes, exact-head CI and final publication/merge.

**Composition checkpoint:** the parent repaired the three observed report-input defects and verified exact source/emitted-JS preservation outside the authorized metric function, prior-test preservation, and comment-only configuration emission identity. Fresh focused results: Node 256 passed/40 skipped across 14 files; Bun 227 passed/40 skipped across 13; the unchanged three parent heldouts pass on both hosts. The final 22-path patch replays to all 298 postimages. Declaration review `deleg_3f9188dc` passed SPEC then QUALITY: the parent authenticated all 339 artifacts, 33 command receipts and the exact three composed postimages. Its real frozen-manager checks are lockfile-only, not full application/native installs. Metrics/composition review `deleg_01f17941` remains pending; no live TypeScript/dependency integration yet. Declaration evidence is archived under `accepted-runtime-contracts/development-declarations/`.

**Security checkpoint:** a fresh, paginated GitHub inventory still has 262 open alerts, including 17 critical. Against the exact candidate graphs, the retained-advisory check passed across 1,780 instances, fresh npm audit passed for 375 package/version identities, and OSV passed for 489 queries with zero findings. This is candidate evidence, not default-branch closure or independent source approval. Nix is not on the current PATH or at its usual default profile path; actual Linux/Darwin CI must measure and verify the changed dependency hashes rather than inventing them. Durable proposal/scan evidence: `docs/security/dependency-remediation/open-runtime-contracts/final-types-composition/proposal-summary.json`.

## Final integrated-checkout checkpoint

Both final independent reviews passed and the 22-path patch is integrated. All 298 postimages matched, then a real frozen Bun install verified genuine declarations without changing either lock or the 50 existing native artifacts. The actual checkout now passes default-root strict compilation on Node and Bun with 153 roots, no diagnostics and no refused reads; none of the historical probes or compiler options were removed.

The canonical build and full Node suite passed (1,495 passed, 66 skipped, 53 files). Full Python selections passed: runtime 1,487/2 skipped/13 deselected, security 500, fine-tuning 1,787/49 skipped. A generated test helper was made private without changing its 26 unittest cases; both pytest and unittest now pass. Eight initial fine-tuning failures were qualification-harness PATH errors, resolved using the actual locked virtualenv binaries with unchanged assertions. Original failures remain archived.

Real isolated package installation/smokes passed on Node and Bun. Native Bun CLI, SDK, SQLite and MCP stdio work; its HTTP mode explicitly refuses with a Node remedy. An extra full native-Bun Vitest attempt crashed during Wasm-thread teardown and is not accepted as successful; the supported CI uses Node for that full suite, with independent native-Bun application/ownership checks.

The AST error-path ownership follow-up is separate (`ast-error-path-ownership.md`). Native Linux GGUF image execution, measured Linux/Darwin Nix dependency hashes, final exact-head CI and default-branch alert readback remain release gates. All current parent receipts are under `accepted-runtime-contracts/final-types-composition/`.

## G1 — parent-owned GGUF/launcher composition while workers run

The accepted seven-file exporter set is `finetune/{train.py,train_unsloth.py,convert_gguf.py,gguf_toolchain.py}` plus `finetune/tests/{test_finetune_correctness.py,test_gguf_toolchain.py,test_gguf_structure.py}`. Compose it with accepted launcher/test/LOCKED-JOBS changes and the already integrated qualifier pair, using exact manifest preimages and protected-body checks.

The union of the seven exporter paths and the accepted 43-file runtime closure is 46 paths. Current metadata/locks already match the accepted closure; do not regenerate or rewrite them merely because earlier planning text said they were pending. Apply only actual reviewed deltas, preserve raw-PEFT and secret-refusal repairs, and verify all sixteen graphs plus the full launcher suite including six real-source closure cases. Native export evidence remains bound to the unchanged helper, but is not image execution.

**G1 checkpoint:** ten reviewed file deltas are now applied. The five-file Python selection passed 372 cases with six unrelated numerical PEFT cases deselected, including all six real-source closure cases. All sixteen offline lock checks passed with unchanged lock bytes; the one initial interpreter-discovery failure is preserved and was resolved by explicitly selecting the already installed Python. The retained 262-alert baseline shows zero affected instances, not a new online scan or default-branch closure. Durable evidence: `docs/security/dependency-remediation/accepted-runtime-contracts/gguf-exporter/live-composition/integration-summary.json`.

## Parent completion sequence

1. Authenticate and retain the workflow rejection/reproducers without relabeling them green.
2. Integrate accepted T1/G1 deltas only after exact preservation/readback; re-review substantive rebases as needed.
3. Independently reaccept W1, then integrate its settled source and tests. Bind it to accepted qualifier `292f01ce...`, not obsolete interface-only `6717b377...`.
4. Run settled-source canonical tests/build, separately omitted Python suites, strict types, package/launcher and security/graph checks; refresh affected Nix inputs only if actual package graphs changed.
5. Commit/push the authorized reviewed candidate, then require exact-head/actual-checkout CI. Read both macOS Node fast-case artifacts and source/log identities; metadata/TAP-only uploads do not qualify. Execute/read back the real Linux amd64 image workflow, digest, architecture, probe and owned cleanup.
6. Preserve the unknown historical macOS cause and C1 qualification honestly. Merge only after all required gates pass; then re-enumerate default-branch dependency alerts instead of assuming closure.

Logical write scopes were checked programmatically: W1, the 19 T1 paths and the 46-path G1 source union do not overlap. All physical worker writes remain in separate owned roots; only the parent writes the live repository.
