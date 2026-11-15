# QUALITY review — PASS, same source-bound qualifications

Started after the completed SPEC PASS and frozen `SPEC-REVIEW.md` (SHA256 `09a6e3bd86e72baaef8eb0833d9f773f75b9ec24b1edfc231b84e47cc39eb68a`). No new blocking or nonblocking product finding was identified in the authorized 22-path composition/parent metric delta. No implementation or live edits were made.

## Reviewed code and evidence

1. **Metric validation (`scripts/phase5_e2e_runner.ts:541-627`).** Query/document dense `for...of` loops now observe missing slots and reject empty/duplicate IDs rather than relying on sparse-skipping callbacks. The score loop explicitly reads both sides at every bounded index, rejects nonnumeric/nonfinite values and preserves inclusive tolerance, dense ranking, finite zero/negative score and timing-independent behavior. Early invalid-identity returns remain separate from per-query mismatch details. The retained post-validation missing-query guard is conservative; it was not removed to make a type-only story. No casts, `any`, compiler suppressions, asynchronous mutation, new deadline or protocol change appears in the parent metric function. The fresh structural checker confirms these properties with the actual TypeScript 5.9.3 AST; the 77-case matrix and 120-case full-result differential supply behavioral rather than merely structural evidence.

2. **Other T2 executable spans (`scripts/phase5_e2e_runner.ts:691-702,763-770,1404-1408`).** The latency percentile guard refuses a missing selected sample; the `.entries()` rank loop preserves dense indices/IDs; the port argument extraction retains ordinary argv truthiness, parsing and default behavior. Actual patch replay and complete source/emitted-JS complements bind their scope. They are acknowledged semantic hunks, not disguised casts or whole-runner restoration. Everything outside the authorized metric/latency/rank/argv spans—including repeated/nested declarations—is retained in the proof.

3. **Accepted reranker boundary (`src/llm.ts:1083-1116`).** The exact accepted T1 postimage remains present. Supported options are constrained with `satisfies`, attention policy remains the model default, the retry uses the same supported options/signal and checks cancellation before retry, and the existing four-context/resource-parent/lifecycle mechanisms remain intact. Fresh focused assertions exercise installed declaration/runtime agreement, supported option keys, zero/nonzero CPU threads, bounded retry, exhausted retry cleanup, cancellation and late context ownership. This is not a native throughput/memory claim.

4. **Accepted benchmark ownership (`src/bench-rerank.ts:102-211,218-315,354-370`).** `Promise.allSettled` drains every started rank operation, including synchronous refusals captured by async callbacks, before disposal. All owned releases are attempted and primary plus cleanup errors remain visible. Context cleanup failure is fatal to subsequent configurations; model/llama releases precede successful summary publication. The entrypoint uses native main flags where available and a guarded, realpath/URL-correct fallback otherwise. Existing lifecycle failure/positive cases and the exact complete executable-body proof passed on Node and actual Bun. Recorded entrypoint coverage is 44/22 respectively; forced fallback on these installed runtimes does not establish an older Node binary's compatibility.

5. **Vitest/Zod policy (`vitest.config.ts:10-16`; `test/zod-import.test.ts:1-15`).** The clarification accurately describes the existing prefix-matching policy rather than claiming package-exclusive Zod inlining. T1/T2/final comments-removed config emission is identical. The independently replayed retained probes use the actual external MCP SDK, actual native Zod observation bridge and actual `zod-to-json-schema`, with in-memory transports. They demonstrate instance boundaries, object/raw-shape registration/listing, valid calls and noncoercing input/output refusals on both hosts; SDK identity and CJS semantics remain intact. No mock SDK stands in for this evidence.

6. **Test and source preservation.** The original T2 tests are byte-preserved around the parent insertion; prior fake payloads/assertions and accepted T1 files are hash-bound. Baseline/candidate manifests, all 298 combined patch replay postimages, all 295 T2 replay postimages, 276 unchanged source files and all parent indexed inputs are independently verified. Owned before/after artifact manifests preserve earlier raw failures as well as final passing receipts. Shared dependency metadata is unchanged; no cache restoration conceals writes.

## Fresh results carried forward from SPEC

- Parent three assertions: actual Node and Bun RED each 3 failures/exit 1 on the exact reconstructed precursor; same assertions GREEN each 3 passes/exit 0 on final.
- Independent metric matrix: 77 passes/exit 0 per host, including complete-result parity across 120 dense controls.
- Node focused union: 14 files, 256 passes, 40 existing skips, exit 0. Actual Bun union: 13 files, 227 passes, 40 existing skips, exit 0. Permanent metric file: 21 passing cases within each.
- Real SDK/prefix probes: 2 files, 4 passes, no skips, exit 0 per host.
- Selected lease controls: Node 12 passes/32 selection skips; actual Bun 11 passes/33 selection skips; exits 0. Raw valid grant and native-runtime positive work are observed; valid-lease EOF and expired grant refuse work; seven malformed frames do not fork. Before-fixture-cleanup sentinel/guard observations and empty finally-kill lists are retained. Node-host native children are verified as actual Node and Bun, not substituted binaries.
- Canonical build: exit 0. Canonical guarded compiler: each host exit 0/zero diagnostics **with two denied live absolute reads**, not whole-root acceptance. After-build root selections are independently identical on both compiler hosts, retain 153 roots, all three historical `.mjs` files and `allowJs: true`.

Counts are per invocation, not cumulative unique coverage.

## Remaining gates and explicit non-acceptances

### Q1 — whole-root compiler gate remains blocked (known qualification, not a new delta regression)

Historical absolute imports are unchanged at:

- `docs/reviews/apple-silicon-audit-evidence/mcp-positive-contract-and-admission-review.mjs:7`
- `docs/reviews/apple-silicon-audit-evidence/mcp-postrepair-http-admission-proof.mjs:8`
- `docs/reviews/apple-silicon-audit-evidence/quality-mcp-heldout-cancel.mjs:5`

Reproducer in this owned copy: run the recorded `types-canonical-node` or `types-canonical-bun` argv through `run.py`. Inspect the corresponding `*-compiler-reads-*.jsonl`, not only exit 0: both lexical and real-path guards deny live `package.json` and `src/mcp/server.ts` reads. No historical script was executed, rewritten, excluded or narrowed. The parent must resolve whole-root acceptance in the actual integrated public checkout; this reviewer does not grant it.

### Q2 — C1 is still qualified

The 200ms execution lease, runner's 800ms teardown allowance (`scripts/phase5_e2e_runner.ts:228`) and existing 1800ms fixture bounds remain unchanged. A paused sole cleanup owner may truthfully return incomplete cleanup and reap naturally afterward. Neither the focused unions nor explicitly selected lease subset certifies universal complete cleanup. Existing C1 evidence/qualification is preserved, not erased or relabeled.

### Q3 — independent T3 and release gates are separate

Only T3's exact `package.json`, `pnpm-lock.yaml`, `bun.lock` composition and 44 genuine copied declaration/metadata files were bound here. Its separate reviewer owns provenance/graph approval. No install, audit-provider/network scan, model/GPU materialization, real index/daemon/cloud run, native image, Nix/CI or publication approval occurred.

## Reviewer tooling limitations

The reviewer preservation tool initially refused ambiguous AST matches and a NodeNext-emitted CLI spelling. Raw failed receipts are retained. Its final ordered, scope-specific selectors and entire-complement comparisons pass; no source or fixture was repaired to satisfy it. The copied execution harness only adds two owned optional receipt path variables for benchmark tests; product assertions and model/egress guard behavior remain unchanged.

**Decision:** QUALITY PASS for the frozen authorized composition/metric repair, subject to Q1–Q3 and the SPEC scope. The independent structural check exits 0 and binds this review to runner SHA256 `4626805b4c9cefa234911d896395da6a56266c79e720164eee6a048e09c81630`.
