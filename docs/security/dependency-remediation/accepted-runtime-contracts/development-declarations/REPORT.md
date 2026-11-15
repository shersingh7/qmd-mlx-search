# Independent T3 review — SPEC PASS, then QUALITY PASS

**Verdict: PASS for T3's scoped three-file development-declaration repair. No blocking or nonblocking product findings. This is not approval of the full checkout, integration, security advisories, Nix hashes, CI or publication.**

Reviewer-owned root: `/Users/shersingh/.hermes/cache/scratch/qmd-declarations-review-C2FZRvxD`.

Read-only delivery reviewed: `/Users/shersingh/.hermes/cache/scratch/qmd-types-declarations-1yp0w89j`.

## SPEC

The SPEC verdict was written and read back before the QUALITY gate executed. Its evidence is `evidence/spec-verdict.json`, `replay-acceptance.json`, `diagnostic-comparison.json`, `independent-graph.json`, `provenance.json` and `preservation.json`.

### Exact product and provenance

Only `package.json`, `pnpm-lock.yaml` and `bun.lock` change. The actual patch SHA256 is `174a639eec01e7b135deb5823335fb7b6b41360b059b456709b9cc4037dcf7f3`. All 263 producer-indexed artifacts were independently rehashed against the supplied index SHA256 `35b02bf22d08f008adc60c684af3a2d56ae83260df6e97c79a697c7b91bc11dd`.

The 251 source preimages were also matched to the separately bound predecessor manifest/index. Actual patch check, apply and reverse-check each exited 0 in the owned derivative. Every one of the 251 postimages was compared. The final source-root inventory is exactly those 251 sources plus 48 generated files, excluding the deliberately separate owned `node_modules` tree; no extra product file or symlink remains.

| Product path | Verified candidate SHA256 |
|---|---|
| package.json | `7607a98a802ec455328ce0452fb2e43fb6c955e774d30e17cc49befac5e7ef1a` |
| pnpm-lock.yaml | `c8dc5bf3011bef4a9417f49dbb37f47fa8aaa45dad97ba13366709554f2e308e` |
| bun.lock | `c1dca7fd35ac82ef213872fc997d023218028d51b9fffb92b0e12b0ac070d329` |

All three **already archived** genuine tarballs were checked against the retained anonymous registry metadata, URL, SHA512 SRI, SHA1, SHA256, identity, unpacked size and complete ordered file-member inventories. No registry request or signature-verification claim was made.

| Declaration identity | Tarball SHA256 | Original files materialized |
|---|---|---:|
| @types/bun@1.3.8 | `49844bf49db894cb893a628a47c3f8368d0e64a601c35da84d5c677464e121eb` | 4 |
| bun-types@1.3.8 | `ac218b0baa43612cffb24f8cd1b574d4ded7bc5f5b9c7b10d026da2c85a2836c` | 31 |
| @types/picomatch@4.0.3 | `f36c5d59dbeb3cb55a1690fef8cf4da182d04a1c25d3771edf2ddb07394c2312` | 9 |

The 44 installed files were checked in both producer trees and in the fresh owned replay. Bun's 323 extra documentation members were intentionally not installed. The preliminary `fileCountMatchesMetadata:false` values are reconciled by **4 files + 1 directory = 5 members** for @types/bun and **9 files + 2 directories = 11 members** for @types/picomatch, not by ignoring any checksum error. Bun-types has 354 files and no directory members.

Official dependencies are exactly `@types/bun -> bun-types: 1.3.8` and `bun-types -> @types/node: '*'`. No declaration was synthesized, no ambient shim or override was added, and no install lifecycle was executed.

### Complete graph preservation

An independent parser/structural comparison checks the **entire three parsed files**: removing exactly the two new direct development pins and three permitted new graph records/snapshots makes the candidate objects identical to their preimages. This covers all prior packages, snapshots, dependency/optional/peer edges, integrity fields, importer/workspace metadata, settings, scripts, existing pins and unrelated top-level fields—not just a direct-dependency sample.

Both managers retain the same complete inventory and add only:

- `@types/bun@1.3.8`
- `@types/picomatch@4.0.3`
- `bun-types@1.3.8`

Distinct package/version identities are **372 -> 375**. The existing `@types/node@24.10.1` is the sole Node-type identity, reused for the wildcard request. Bun's 385 -> 388 keys include aliases; its frozen command's 376-package display includes the workspace and is not the third-party identity count.

### Actual offline command results

`evidence/verified-raw-commands.json` binds every per-command receipt and raw stdout/stderr hash. **33 sandboxed commands** ran: 21 exited 0, 8 expected negative controls exited 1, and 4 strict compiler invocations exited 2. No setup failure is relabeled a product failure or pass.

Verified hosts are the actual supplied executable paths: Node **22.22.3**, Bun **1.3.8**, pnpm **10.33.0**, Python **3.12.13**, and installed TypeScript **5.9.3**. Bun's own `process.execPath`/`process.versions.bun`, not a Node launcher, are retained in the receipts and type-binding evidence.

| Check | Fresh result |
|---|---|
| Canonical `node scripts/build.mjs`, before and after | Both exit 0; all 48 generated hashes, sizes and modes identical to each other and the producer manifest |
| Actual Node strict CLI, before -> after | 35 -> 31 diagnostics; exit 2 in both runs |
| Actual Bun strict CLI, before -> after | 35 -> 31 diagnostics; exit 2 in both runs |
| Canonical `--showConfig` | Identical options and 153 roots, retaining all three historical `.mjs` probes and generated `dist` roots |
| Real frozen pnpm | Exit 0; `--lockfile-only --frozen-lockfile --ignore-scripts --offline`, owned initially empty store |
| Real frozen Bun | Exit 0; `--lockfile-only --frozen-lockfile --ignore-scripts --save-text-lockfile`, owned initially empty cache, OS network denial; no unsupported `--offline` flag |
| Stale pnpm manifest control | Exit 1, specifically `ERR_PNPM_OUTDATED_LOCKFILE` |
| Stale Bun removal-only manifest control | Exit 1, specifically `lockfile had changes, but lockfile is frozen` |
| Existing `test/dependency-locks.test.ts` | One file; 5 passed, 0 failed/skipped; one worker, `--configLoader runner` |
| Missing/tampered declaration/package-metadata controls | Three exit-1 refusals at the intended missing-file/hash checks |
| Existing integrity/edge/metadata graph-tamper controls | Three exit-1 refusals at their intended whole-graph checks |

Both manager positives and the two valid stale controls left their scoped package stores/caches at **zero files**. Frozen fixtures' manifests/locks were unchanged. This proves these lockfile-only operations, **not a full application/native install**. Producer raw receipts were also reviewed; its earlier Bun version-change failure remains a metadata-resolution failure, not stale-lock proof.

The only removed diagnostics are the official Bun sqlite module, globalThis.Bun, picomatch declaration and Bun test module errors. There are no added diagnostics. The same 31 `scripts/phase5_e2e_runner.ts` diagnostics remain, owned by the separate T2 repair.

Both real hosts bind `bun:sqlite` and `bun:test` to the installed genuine bun-types files, picomatch to `@types/picomatch@4.0.3`, and `globalThis.Bun` to **`typeof import("bun")`, not `any`**. Bound paths and declaration SHA256 values are in `evidence/type-binding-summary.json` and the full `resolution-{node,bun}-bindings.json` artifacts.

### Isolation and preservation

Every tested child starts with a new allowlisted environment and synthetic HOME/Hermes/knowledge/XDG/HF/temp/cache paths; HF and uv offline flags, `QMD_LLAMA_GPU=false`, `CI=1`, `REQUIRE_BUN=1`, owned npm/Bun configs and disabled scripts are explicit. Every tested command uses an OS network/write sandbox. An independent owned-target probe demonstrated **EPERM for both a disallowed write and loopback network attempt**.

The source's `node_modules` is a real owned directory with 274 per-package read-use aliases and owned `.vite`, `.vite-temp`, `.cache`, `.vitest` directories created before the first build/compiler run. Only the verified declaration files were materialized. No application/native package installation, model/GPU workload, database/index/daemon, live-config/service change, external harness, online audit, staging, commit or publication was performed.

All **1,203 manifest-bound public input files** are unchanged in bytes and tracked stat metadata. The shared dependency inventory has **18,643 rows including the root**; tracked stat metadata and all permitted content hashes are identical before/after and match the authoritative snapshot. One explicit exception is `node-llama-cpp/llama/llama.cpp/AGENTS.md`: inventory was **metadata-only**, never a content read/hash. Access time is not part of the preservation claim. No unowned cache was cleaned or restored.

Each strict invocation still refuses exactly the unchanged historical absolute reads of `/Users/shersingh/github/qmd-mlx-search/package.json` and `/Users/shersingh/github/qmd-mlx-search/src/mcp/server.ts` **before content read**. Independent controls on each host additionally refuse both paths plus an owned lexical symlink escape (three refusals per host). Consequently this evidence certifies the **relative four-diagnostic T3 repair only**, not whole-root acceptance. The roots/configuration/archive were not narrowed or rewritten to suppress the refusals.

## QUALITY — evaluated only after SPEC PASS

**PASS; no findings.** The change is the minimal two exact direct development declarations plus their genuine transitive wrapper dependency. Both lock formats preserve every pre-existing record and agree on the new trust identities. There is no executable product-code change, runtime dependency change, ignored compiler error, widened declaration `any`, configuration workaround or unrelated cleanup. Identical build emissions and genuine binding on both hosts support the change's compatibility within the stated scope.

Machine verdict: `evidence/quality-verdict.json`. Evidence: `quality-evidence.json`, `verified-raw-commands.json`, `supplier-raw-receipts-reviewed.json` and `type-binding-summary.json`.

## Remaining parent gates

- Final integrated whole-checkout canonical build, strict check and tests after the separate runner repair, with whole-root handling of historical imports.
- Fresh online vulnerability/dependency-alert retrieval for the changed complete graphs.
- Actual Linux/Darwin Nix dependency fixed-output hash computation/verification.
- Parent-owned integration, CI, merge and any publication/release decision.

There are no reviewer setup failures or unresolved T3 product findings. All reviewer-created files are under the owned root above. The artifact index is `evidence/review-index.json`; its SHA256 is supplied in the handoff rather than embedded recursively here.
