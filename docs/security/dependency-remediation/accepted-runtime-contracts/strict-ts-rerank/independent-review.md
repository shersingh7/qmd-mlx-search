# Independent strict-TypeScript + reranker supplement review

## Verdicts

**SPEC: PASS for the frozen combined proposal and exercised Node 22.22.3 / Bun 1.3.8 paths.** This is not live integration, recovery-transition acceptance, package acceptance or release approval.

**QUALITY: APPROVE WITH ONE MINOR FINDING.** No critical or important defect attributable to this source delta was found. The broader Bun limitation below remains an explicit failed check, not a green suite.

Machine-readable results: `review-result.json`. Final preservation checks: `evidence/integrity-final.json`. Commands, cleared child environments, actual host identities, stdout/stderr and JSON results are retained in `evidence/` and `logs/`.

## Source and scope authentication

- Authenticated all **242** original snapshot entries; verified original mirror hashes, derivative mirror hashes and own replay source. Reconstructed all **20** original delivery files and all **4** supplement files by applying the actual patches in own scratch. The integration union is exactly **16** source/test files; metadata intentions and historical-import relocation are separate.
- Original source patch SHA-256: `e7b8d38aa36df031a3d28b4ba4044d5a6e16607b7f1ddfe3fa515f7fbe4fa620`.
- Supplement SHA-256: `531b457942ab2c4d3e17780a69251a9b4b8d23b2287f436f76c05b3981eed4cc`.
- Independently authenticated the original 14 source files against actual `git show 346d905666ccafca1ddea76d5f994df8d574f4b0:<path>` bytes and delivery hashes. Fresh closed settled-snapshot compiler replay produced **61** genuine diagnostics. The repaired root produced **0** under both hosts.
- Compiler configurations are byte-identical. `strict`, `noUncheckedIndexedAccess` and root `noEmit` remain enabled. Root inventory: **146 → 147**, no removals, only `test/bench-rerank-lifecycle.test.ts` added. No new compiler suppression or broad ambient-any shim. Legitimate installed Bun/picomatch declarations are used; metadata remains an intent for parent integration.
- The three live archived `.mjs` originals still match their actual historical manifest, not merely the delivery's claim. Mirror-only relocations were never applied to live evidence.
- O/F contain the **old** 24,780-byte recovery prefix, SHA `0d35ce1805fed69b6a7c8cbe844227159181dc29198fb74e06d7291230f74f90`. Original type repairs do not alter it. The separate newer live recovery freeze's **9** files matched at entry and exit. Parent must rebase only type changes outside the protected recovery implementation/interface; no old whole-file replacement is approved.

## Original 14-file repair review

Reviewed the actual patch, not only diagnostics:

- `scripts/phase5_e2e_runner.ts`: pair-presence and peer-score narrowing, percentile absence refusal, indexed iteration and CLI argument narrowing are outside the frozen recovery implementation. Dense behavior is preserved; sparse query slots now produce a refusal rather than a dereference crash.
- `test/fixtures/public_eval_dataset.ts`: first-element checks and `entries()` preserve metrics. Independent exact-emitted-function probes compare **100** deterministic dense/empty/partial fixtures against original bytes on each runtime.
- `src/llm/mlx-identity.ts`: the explicit Node web-reader type avoids Bun's global reader augmentation; whole-module runtime emit is byte-identical. `src/mcp/server.ts:1266` passes the existing URL's `.href`; independent constructor-only probes preserve URL, method, headers, body and signal behavior without sending requests.
- `src/test-preload.ts`, `test/cli.test.ts`, `test/eval-bm25.test.ts`, `test/eval.test.ts`, `test/rrf-trace.test.ts`, `test/store.helpers.unit.test.ts`: NodeNext `.js` specifiers resolve real source APIs rather than hiding roots. CLI health parsing validates the real status/uptime/PID/UUID shape before existing ownership checks/signalling; independent extracted-initializer checks accept valid data and reject malformed identities.
- `test/llm.test.ts`: the mocked result is annotated with the real public rerank return type, not a fabricated shape. `test/multi-collection-filter.test.ts`: optional access still fails the existing expected-value assertion if missing. `test/phase5-recovery.test.ts` adds the sparse-slot regression without relabeling older recovery evidence.
- `src/bench-rerank.ts`: typed factory results and the real model attention-default property replace unsupported per-ranking-context flash configuration and falsely distinct flash comparisons.

## Supplement contract and policy review

`src/bench-rerank.ts:126-210` now encloses acquisition/measurement in cleanup, drains every started outer ranking with `allSettled` (including synchronous refusal), attempts every context release, retains primary and cleanup faults, and marks failed cleanup fatal. `:275-312` retains ordinary per-config partial results but stops on cleanup failure, releases model then llama, and withholds summary until successful parent cleanup. `:355-360` guards imports and gives fatal CLI failures nonzero exit.

`src/llm.ts:1094-1106` removes only the runtime-ignored option, checks supported options with `satisfies`, and retains bounded retry and lifecycle checks. Independently compared every `LlamaCpp` member: only `ensureRerankContexts` changed. `ensureModel`, `ensureLlama`, `computeParallelism`, `threadsPerContext`, `createContextPool`, `dispose`, both context-size constants and benchmark policy constants/helpers are unchanged.

Installed node-llama-cpp **3.22.3** declaration and runtime bytes match the supplied hashes. Its actual `_create` destructures only the supported six options; an independent invocation of that actual JS factory with a synthetic model boundary proves an extra throwing `flashAttention` getter is not even read. The regular context constructor defaults attention from the model. A negative type probe rejects `flashAttention` with **TS2353**; Node-reader, Request-href and supported-option positive type probes pass. No model loading/default-attention policy was changed.

## Fresh execution results

All executions had cleared allowlisted child environments, fictional HOME/Hermes/HF/config/cache roots, managed-scratch TMPDIR, CI/offline flags, `REQUIRE_BUN=1` and `QMD_LLAMA_GPU=false`. Vite/results caches were owned **before the first test**; dependency sharing used owned directories with per-package symlinks, not an externally writable top-level cache. Bun Vitest was explicitly hosted with `bun --bun`, confirmed by the in-test runtime receipt.

| Check | Node 22.22.3 | Bun 1.3.8 |
|---|---:|---:|
| Unchanged strict root `tsc --noEmit`, TS 5.9.3 | exit 0 | exit 0 |
| Five focused lifecycle/LLM files | **124 pass, 31 intentional CI model skips** | **124 pass, 31 intentional CI model skips** |
| Six additional original-repair files | **77 pass, 9 CI model skips** | **59 pass, 9 CI skips; 1 suite import failure** |
| Exact-body old-source lifecycle controls | **3 expected RED** | **3 expected RED** |
| Same exact-body proposed-source controls | **3 pass** | **3 pass** |
| New independent lifecycle/CLI/compatibility heldouts | **4 pass** | **4 pass** |
| Actual production cold-pool concurrency probe | pass | pass |
| Five original-typefix heldout groups | pass | pass |

The new independent lifecycle probes preserve complete emitted executable bodies while changing only the dependency import and adding test exports. They check arbitrary refusal values, multiple pending siblings, asynchronous release failures, complete nested error retention, parent-release ordering and actual CLI status/output. The production pool probe invokes actual source with synthetic factory boundaries: five cold callers share four contexts, one bounded retry, supported options/signal, and exactly four releases. Synthetic throughput/RSS output is **not** native performance or memory evidence.

**Broader Bun limitation:** `test/phase5-recovery.test.ts` fails during import at `src/mcp/server.ts:42` (`z.object` on undefined), before collecting tests. The identical failure reproduces against the unpatched snapshot in a separate owned baseline mirror. This is a baseline-level Bun/Vitest import limitation in this isolated layout, not a new patch finding or a passing six-file Bun run. Raw standalone before/after logs are retained.

## Quality finding

**Q1 — Minor/P3: legacy Node fallback is not symlink-aware (`src/bench-rerank.ts:353-355`).** The fallback compares a lexical `process.argv[1]` URL with Node's real module URL. With only the native `import.meta.main` property forced undefined on the current Node host, direct entry runs, but symlink entry exits **0 without launching**. Native current Node/Bun symlink tests pass; Bun's forced-fallback symlink characterization also runs. No older Node binary was run, so this is a concretely exercised fallback-branch finding, not an older-runtime qualification. Normalize fallback entry identity or document the narrow legacy limitation. It affects this development utility, not the canonical packaged CLI.

Original type/import changes are low regression risk. Benchmark ownership/error-flow changes are a larger behavioral delta, justified by observed RED controls and broad synthetic failure coverage. Native library promises that never settle and native allocator/disposal behavior are not qualified here; no timeout/GC workaround was introduced.

## Preservation, harness issues and remaining gates

All protected source, metadata/configuration and original artifacts remained unchanged. Original parent RED SHA remains `055073c2b89162b97bdae97f66217fc5b3e911acebf37d59761be80381865bbe`. The previously disclosed F write to O's shared Vite results cache is preserved, not restored; its hash and nanosecond mtime stayed unchanged throughout this review.

Own heldout harness initially placed a fallback-control prefix before a retained shebang, causing a syntax failure; corrected only the own harness. A subsequent Bun characterization initially used a Node-specific expected symlink result; corrected only the host-specific characterization. Both failed scripts/logs and their partial observations remain retained. No product source/assertions or immutable evidence were rewritten.

Created only this managed-scratch review tree: own mirrors/reconstructions, owned caches, probes, raw logs, command/runtime receipts and reports. No live integration, commits, installs, network requests, credentials/private knowledge/live indices, native model/GPU/inference/training or external writes. Parent retains recovery rebase, final integrated root checks/full suites/package checks, real declarations/lock/audit/Nix integration, native CI and release acceptance.
