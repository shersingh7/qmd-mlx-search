# Owned TypeScript/reranker composition — PARTIAL / BLOCKED

## Deliverable and scope

- Owned root: `/Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p`; product derivative: `mirror/`.
- Frozen baseline: **248 authenticated public source files**; accepted proposal: **19 hashes authenticated** against report-index SHA-256 `949e00889d5c073c9d61bdb69ee01dfde307dffc6bc8c39f2f8046dd82fba9dd`.
- Applied **17 accepted paths byte-for-byte**, including supported reranker factory options, async drain/release, realpath-aware direct/symlink entrypoint guard, backend identity import and SDK URL typing. **Two lease-overlap paths deliberately held unchanged.** No unreviewed type assertions, suppressions, policy narrowing, compiler/config-root edits or live writes.
- Minimal patch: `parent-integration.partial.patch` (SHA-256 `8bdd69cdafc5bc1edc6d82347b1f2bb5cf8c2d09cf615466819ea70ed9c7b353`). This is **not** a completed 19-path integration or an approval. `path-preimages.json` supplies exact per-path SHA-256 preimages/absence and postimages.
- Real `git apply --check` and apply passed on a fresh owned baseline replay; all **251** replayed source hashes equal the derivative. `before-manifest.json`, `after-manifest.json`, `composition-result.json` and `evidence/authentication.json` retain the bindings.

## Blocking findings

### B1 — accepted overlap changes are not type-erased

`reference/original-source.patch:1–57` is authenticated to the accepted source patch. Its five runner hunks add executable missing-query-slot/score/percentile guards and alter iteration/argument evaluation. `reference/diffs/test__phase5-recovery.test.ts.diff` adds a behavioral sparse-query-slot test, not a type-only change.

Scope requires only a type-only rebase on `scripts/phase5_e2e_runner.ts` and `test/phase5-recovery.test.ts`. These two paths therefore remain byte-identical to the newer lease baseline. The accepted sparse-slot test was safely relocated to `held/` using only owned public imports and actually run: **1 failure, 17 pattern-unselected skips** (`node-held-sparse`), throwing at the unchanged runner's `qPre.topDocIds`. This is evidence of the scope conflict/pre-existing sparse-report behavior, not a new execution-lease defect. No replacement behavior or non-null assertions were silently introduced.

### B2 — canonical root attempts forbidden live imports

All canonical roots remain: **150 → 153**, only the three accepted new tests added; no roots removed. This includes generated build JS/declarations and all three existing `.mjs` documentation probes. Their absolute imports target live `src/mcp/server.ts`. The installed compiler actually attempted reads of live `package.json` and `src/mcp/server.ts`; the lexical+real guard blocked both before content read. Raw guard JSONL remains in `evidence/*-compiler-reads-*.jsonl`. No documentation imports, exclusions, `allowJs` or compiler options were changed to conceal this. Even a nominal zero diagnostic count would not establish an isolated canonical-root pass while these reads are refused.

### B3 — authoritative dependency declaration surface differs

The allowed installed root has **274 actual per-package aliases**. `@types/bun`, `bun-types` and `@types/picomatch` are absent; no install or ambient declaration fabrication was attempted. Strict root reports `bun:sqlite`, `bun:test`, `globalThis.Bun` and picomatch typing errors in addition to the held runner's errors. `logs/inspect-deps.stdout.log` records the discovery.

## Actual verification

| Invocation | Real result |
|---|---|
| Canonical build script `node scripts/build.mjs`, unchanged package build command | **exit 0**, both owned baseline and 17-path derivative |
| Baseline strict root after its canonical build | **exit 2, 61 diagnostics** |
| Derivative strict canonical root, Node 22.22.3 | **exit 2, 35 diagnostics**, plus blocked live-import attempts |
| Derivative strict canonical root, actual Bun 1.3.8 compiler host | **exit 2, 35 diagnostics**, plus blocked live-import attempts |
| 13-file focused Node Vitest union | **245 passed, 40 existing CI/model skips**, exit 0 |
| Same union, actual Bun Vitest host (`bun --bun`) | **223 passed, 40 existing CI/model skips**, exit 0 |
| Actual SDK public in-memory boundary + real converter, Node | **4 passed, 0 skips** |
| Same SDK/converter checks, actual Bun | **4 passed, 0 skips** |
| Native-Bun recovery ownership wrappers (outer Node Vitest) | **7 passed, 0 skips**; actual child runtime **Bun 1.3.8** verified |
| Safely relocated accepted sparse-slot regression against unchanged runner | **1 expected diagnostic failure, 17 pattern-unselected skips** |

Focused pass counts are one below the accepted historical 246/224 because the added sparse-slot test is explicitly blocked, not passed or silently skipped in the derivative. SDK receipts freshly record **20 tool calls per host: 4 valid inputs, 14 rejected invalid inputs, 2 rejected invalid outputs**, closed transports and zero network/native attempts. Bun genuinely uses distinct inlined Zod factories/classes while the SDK remains external; the real converter prefix check passes. The approved string-inline policy still matches **Zod-prefixed packages**, not exclusively the `zod` package.

The first Node focused attempt had an **own guard-fixture error**: a whole-path `mlx` regex accidentally rejected the SQLite library because the public dependency path contains `qmd-mlx-search`. It was corrected only in the owned test guard to check the library basename/package boundary. Original fixture, failing JSON/logs (**235 pass, 2 fail, 48 skipped**) and corrected-run evidence are retained. No product/test assertion was weakened. The pre-build type attempt's **39 diagnostics** is also retained separately; generated dist naturally resolves four timing-script errors after the canonical build.

## Preservation

- All **12 newer lease candidate paths**, all **4 observer paths**, package metadata/locks and both compiler configurations are byte-identical to baseline.
- Installed TypeScript **5.9.3** emitted comparisons prove exact runtime-JS identity for both held overlap files; source bytes are identical too (`evidence/source-emission-proof.json`, `evidence/emitted/`). These are preservation proofs, **not** proof of a completed type rebase.
- All **18,642** authoritative dependency entries retain identical hashes/size/nanosecond-mtime/mode/link metadata. Node_modules containers are real owned directories with per-package read-use aliases, owned `.vite/.cache/.vitest`, and isolated tool caches configured before any RED run.
- Frozen baseline's **248** input hashes and all **19** accepted input hashes were rechecked unchanged. No live integration, installs, network, model/GPU/materialization, private knowledge/config, service, commit/push, skill or CI operations.
- **C1 remains qualified:** an artificially paused sole cleanup owner may return truthful incomplete-cleanup proof, without forbidden work, then naturally reap later. No universal complete-cleanup claim is made.

## Reproduction and parent gates

Every invocation's exact argv/cwd/allowlisted child environment and exit code is in `evidence/<label>.json`; complete stdout/stderr is under `logs/`. `run` starts from an empty environment and `probes/run.py` creates a fresh per-label owned HOME/Hermes/knowledge/XDG/HF/TMP/cache environment. Use a **new label** for each rerun (directories are exclusive); `types-*`/`build-*` labels enable the compiler read guard, and Bun type runs additionally use explicit `--preload ../probes/compiler-read-guard.cjs`.

`REPRODUCE.md` contains runnable owned-root commands. Parent gates: decide whether to authorize/review the accepted executable metric guards on the two held paths; resolve historical absolute compiler imports and declaration provenance without hiding roots/errors; independently review any substantive rebase; then run the parent-owned full suite/integration gate. No self-approval, full-suite, actual CI, installed/native-model or release acceptance is claimed.
