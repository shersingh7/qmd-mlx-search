# SPEC review — PASS, source-bound and qualified

Quality review may start only after this result. `evidence/spec-result.json` records `PASS_SOURCE_BOUND_QUALIFIED` with `qualityStarted: false` after the fresh checks below. This is proposal approval within the stated scope, **not whole-root, T3-provenance, live-integration or release acceptance**.

## Bound input and scope

- Frozen input: `/Users/shersingh/.hermes/cache/scratch/qmd-types-final-composition-mer5fi8x`.
- Independently verified its report-index SHA256 `20b7291d264daa47e2017070c8747ba962db1160ce5802bd38eac8e6b82859cb`, all 50 indexed artifacts, both exact source manifests (295 baseline / 298 candidate files), and the actual 22-path diff/preimages.
- Candidate manifest SHA256 `cada46c60b45ccd367282c06c6daecc58f233ff2c831c96c53318b597e29d3b0`.
- Final combined patch SHA256 `f2db636f1c6b113f15b44fe28742764dc5c7a3accb67fca9f5119cbe54469f1c`.
- Final runner SHA256 `4626805b4c9cefa234911d896395da6a56266c79e720164eee6a048e09c81630`; permanent metric tests SHA256 `12da9479dc81affd3e83b2c53fde1b4dab3b36786a940a65214203030abd130a`.
- All work is in this new owned review directory. No live product edits, integration, staging, commits, installation, network, model/GPU loading, actual product indexing/daemon/cloud/native-image/CI execution, external coding harness or further delegation.

## Fresh contract executions

Counts are per invocation and must not be summed as unique coverage.

| Invocation | Files | Passed | Failed | Skipped | Raw exit |
|---|---:|---:|---:|---:|---:|
| Original parent three boundary assertions, reconstructed exact precursor, Node | 1 | 0 | 3 | 0 | 1 |
| Same boundary assertions, reconstructed exact precursor, actual Bun | 1 | 0 | 3 | 0 | 1 |
| Unchanged parent boundary tests, final candidate, Node | 1 | 3 | 0 | 0 | 0 |
| Unchanged parent boundary tests, final candidate, actual Bun | 1 | 3 | 0 | 0 | 0 |
| Reviewer independent metric matrix, Node | 1 | 77 | 0 | 0 | 0 |
| Reviewer independent metric matrix, actual Bun | 1 | 77 | 0 | 0 | 0 |
| Focused product union including native-Bun ownership wrappers, Node | 14 | 256 | 0 | 40 | 0 |
| Focused product union, actual Bun | 13 | 227 | 0 | 40 | 0 |
| Retained real SDK/Zod-prefix probes, Node | 2 | 4 | 0 | 0 | 0 |
| Retained real SDK/Zod-prefix probes, actual Bun | 2 | 4 | 0 | 0 | 0 |
| Selected execution-lease controls, Node host with real Node and Bun children | 1 | 12 | 0 | 32 | 0 |
| Selected execution-lease controls, actual Bun host, real Bun child | 1 | 11 | 0 | 33 | 0 |

The 40 skips are existing model-dependent suite skips. The 32/33 lease skips are explicit selection, not successful coverage of the full 44-case lease file or C1. The permanent metric file has 21 passing cases within each focused union. Executables and versions are independently recorded: Node `/Users/shersingh/.hermes/node/bin/node`, version `22.22.3`; Bun `/Users/shersingh/.bun/bin/bun`, version `1.3.8` (its Node compatibility version is separately `24.3.0`); TypeScript `5.9.3`. Python review harness version is `3.12.13`, distinct from the product's preserved literal `/usr/bin/python3` helper selection.

## Parent repair correctness

At `scripts/phase5_e2e_runner.ts:549-599`, dense query/doc iteration observes holes and explicit undefined query entries, rejects empty/duplicate query and document IDs, and leaves the existing query-error refusal intact. The indexed score loop observes every score position on both sides, rejects missing/nonfinite values, preserves finite zero/negative values, and retains inclusive absolute tolerance semantics. This review does not expand `StageMetrics` into an arbitrary untrusted-JSON schema contract.

The independent 77-case matrix exercises first/middle/final holes on either/both sides; query holes and explicit undefined details; duplicate/empty identities; score/result lengths; nonfinite scores and invalid tolerances; genuine dense multi-query success; timing independence; ranking/rank/hit/aggregate changes; zero tolerance and exact binary-representable tolerance boundaries. One case compares complete result objects with the verified T2 predecessor across 120 deterministic dense positive/negative controls, not just type-checking or `identical` flags.

Original parent RED evidence remains unmodified. The reviewer reconstructed the pre-parent 298-file manifest using only hash-verified T2 public files and exact current public postimages. Its RED copy changes only the candidate import pathname; all behavioral assertion bytes are preserved. The final GREEN copy of the parent's heldout file is byte-exact.

## Composition, source and lease preservation

- T1's 17-path partial patch is bound to its supplied index. Sixteen final source postimages are byte-identical to the independently accepted T1 files; `vitest.config.ts` differs only by the explicit comment clarification and has identical comments-removed TypeScript emission to both T1 and T2.
- T2's actual two-path patch independently passes `git apply --check --whitespace=error` and apply on a fresh 295-file owned baseline. Every replay postimage matches the expected baseline with exactly its two accepted changes.
- Final 22-path patch independently passes the same check/apply in a separate fresh owned replay; all 298 postimages match.
- Independent TypeScript 5.9.3 ordered-AST/source proof establishes complete source and comments-removed emitted-JS complement equality baseline→T2 outside the authorized metric function, percentile guard, ranking loop and CLI argument spans. It establishes complete T2→final complement equality outside the single top-level metric function. No flat name dictionary drops nested/repeated definitions. All previous T2 test bytes survive around the three new test insertions.
- All 276 baseline source files outside the changed set remain byte-identical, including the Python supervisor, loader, worker, observer/helper sources, ownership fixtures and admission/startup/transition/tree tests. The runner's clocks, execution grants, ownership, Python command selection, source/worker payload construction and teardown protocol outside the explicitly authorized metric/CLI hunks are unchanged. No full historical runner copy was restored.
- Selected actual lease controls retain the 200ms execution lease and unchanged 1800ms fixture guard, prove valid raw and native-runtime grants execute, reject persistent caller EOF while lease time remains, reject an already-forwarded expired lease, and reject seven malformed/duplicate/truncated budget frames. All selected before-finally snapshots preserve the unrelated sentinel, and all failure-cleanup lists are empty. Existing native-Bun wrappers also execute their unmodified startup/transition/tree/FD-reuse/parent-signal controls. No production deadline or 800ms teardown allowance was relaxed.
- **C1 remains qualified:** an artificially paused sole cleanup owner may truthfully return incomplete cleanup and naturally reap later. The selected replays do not establish or claim universal complete cleanup, and do not relabel the retained C1 limitation as green.

## Entrypoint/reranker and real SDK boundaries

The accepted complete benchmark module is hash-bound and its executable bodies are checked before dependency-only synthetic substitution. Fresh recorded entrypoint cases: 44 on Node (native/forced fallback × default/preserve-symlinks-main) and 22 on actual Bun (native/forced fallback, default mode). Direct/relative/symlink/multihop paths with URL-significant characters launch once, imports stay inert, and lifecycle cases retain settled-sibling and all-release/error ownership assertions. Forced fallback qualifies that branch on the installed runtime, not an older Node binary or native-model throughput.

Real, external MCP SDK probes preserve SDK identity and CJS behavior, register/list/validate both object and raw-shape Zod schemas over in-memory transports, reject malformed inputs/outputs, and characterize the Bun inlined-vs-native Zod instance boundary and real `zod-to-json-schema` prefix matching. No SDK replacement or network listener was used. Recorded egress/native-load guard attempt lists are empty; model constructors/resolvers are refused at the copied package boundary. No model or GPU execution is claimed.

## Compiler/build boundary — remains incomplete for whole-root acceptance

The canonical `node scripts/build.mjs` exits 0 with no diagnostics/refusals. Canonical default-root `tsc --noEmit --pretty false` on Node and actual Bun each exits 0 with zero diagnostics, but each records **two refused live absolute reads**: the live checkout's `package.json` and `src/mcp/server.ts`. The lexical **and** real-path read guard is retained, as are all compiler options, `allowJs: true`, and the three historical `.mjs` roots. Root selection has 105 files before build and 153 after the canonical build; generated dist roots are not hidden or excluded.

Accordingly, **whole-root acceptance remains false**. The actual integrated checkout's canonical whole-root compiler is a separate parent gate; these guarded exits are not a substitute.

## Isolation, remaining gates and tooling issues

Real owned `node_modules` containers have individual read-use aliases, not root symlinks; `.vite`, `.vite-temp`, `.cache`, and `.vitest` are owned before RED. The three T3 metadata postimages are byte-exact to T3, and each compiler tree uses the 44 hash-bound genuine declaration/top-level metadata files. T3 provenance/graph acceptance belongs to its separate reviewer; this report does not duplicate or grant it.

Fresh subprocess environments are allowlisted with owned HOME/Hermes/knowledge/XDG/HF/TMP/tool caches and offline flags, `QMD_LLAMA_GPU=false`, `CI=1`, and `REQUIRE_BUN=1`. All 18,642 shared dependency metadata entries remain unchanged; dependency `AGENTS.md` content was never read. The immutable parent's 50 artifacts and 295/298 source files were rechecked unchanged after execution. No unowned cache was restored.

Reviewer tooling initially assumed T3's index was at the root; the actual `evidence/report-index.json` was discovered before source copying. The reviewer preservation probe then conservatively refused ambiguous AST selectors (duplicate variable/containing-loop matches and TS-emitted CLI syntax). These failed raw receipts are retained; corrected precise AST scope/header selection passes in `spec-preservation-r4`, without any product edits or weakened comparisons. The parent's separately retained first patch-serialization failure also remains untouched.

Remaining gates: separate T3 review, parent integration, actual-checkout whole-root checks, fresh security/scanner/install/native-image/Nix/CI/release qualification and any authorized publication. No new blocking SPEC finding was found in the authorized composition/metric delta.

## Evidence map

- `evidence/spec-result.json`, `test-summary.json`, `lease-controls-summary.json`, `benchmark-replay-summary.json`
- `evidence/independent-preservation.json`, `patch-verification.json`, `compiler-summary.json`
- Per-command `*-execution.json`: exact argv, environment, raw stdout/stderr, exit
- Per-suite Vitest JSON, `*-hosts.jsonl`, `*-compiler-reads-*.jsonl`, entrypoint/lifecycle receipts
- `inputs/*manifest.json`, `evidence/*source-after.json`, dependency before/final metadata
- `heldouts/independent-metrics.test.ts`, copied unchanged public probes/fixtures, and owned `probes/` verifiers
