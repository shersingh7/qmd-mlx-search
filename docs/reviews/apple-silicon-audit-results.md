# QMD MLX Apple Silicon audit — implementation and verification

## Acceptance state

**LOCAL CODE ACCEPTANCE PASSED.** Implementation, full local suites, real-model checks, installed-package checks and the independent spec/quality closures are complete. No known critical or important finding remains open within the reviewed contracts. This is acceptance of the inactive local artifact—not production activation or large-model/full-corpus/soak qualification.

- Repository: `/Users/shersingh/github/qmd-mlx-search`
- Working branch: `hermes/apple-silicon-audit`
- Baseline commit: `1d6cfef0353810eea55d3fec879d9c05ef5ebc4c`
- Execution: direct Hermes work plus parallel Hermes implementation/review waves using the task-authorized GPT-6.1 route. No external coding harness was launched.
- The work is local and uncommitted. No GitHub push, registry publication, global QMD replacement, launchd activation, live-index migration or personal-corpus indexing was performed.

The [implementation plan](../plans/apple-silicon-audit-implementation.md) contains 15 initial workstreams and the review-derived closure contracts. This was a targeted architectural overhaul, not an indiscriminate rewrite of the retrieval algorithm.

## Bottom line

The most important discovery was correctness, not speed: the earlier BERT/MiniLM path omitted learned segment-zero embeddings. Its own singleton/batch results agreed while disagreeing substantially with the actual trained model. That path is repaired and checked against the same cached Hugging Face model on CPU.

The implementation now has explicit ownership and bounds across GPU work, model residency, requests, database publication and process cleanup. The measured opt-in MiniLM FP16/bulk profile improves useful throughput while retaining reference agreement; it does not justify consuming all RAM or changing every model's defaults.

## Architectural changes

| Area | Implemented behavior |
|---|---|
| GPU/runtime ownership | One execution owner creates, evaluates and disposes MLX state. Interactive requests can run between bounded bulk microbatches; CPU preparation can overlap without concurrent unsafe Metal ownership. |
| Adapter correctness | Explicit supported encoder/causal families, trained pooling, integer token/segment inputs, compute-compatible masks and float32 pooling/normalization. Actual precision/storage policy is validated instead of merely echoed. |
| Request identity | Atomic tokenizer/descriptor preparation snapshots; identity checked across microbatches and reloads; completed results carry their own immutable descriptor through JSON/binary serialization. |
| Memory/lifecycle | Cancellation distinguishes queued work from an already-running native kernel. Weight eviction respects leases; shutdown cannot be followed by a stale CPU caller reloading weights off-owner. Partial loads and tensor-retention paths are covered. |
| TypeScript transport | Bounded response/cardinality/framing validation, finite Float32 values, serving-identity checks, abort propagation and one absolute public MLX operation deadline across metadata and inference. |
| Store/SDK | Per-store backend/config ownership; truthful incremental counters; forced replacement rollback; durable checkpoint reconciliation; close-before-await fencing; rollback/compensation for failed collection configuration changes. |
| Query consistency | Stored IDs are checked literally. Final identity validation, KNN and document materialization share one synchronous deferred SQLite snapshot, including concurrent writers on other connections. |
| Cache publication | Expansion/rerank cache mutations are staged per query/database and published only after successful completion. Failed identity/cancellation paths cannot insert, prune or rewrite cache state. |
| HTTP/MCP | Independent request cancellation ownership handles numeric `0` and empty IDs, rejects duplicate active IDs before overwrite, preserves typed IDs and separates sessions. Admission bounds apply before collecting bodies; rejected initialization does not evict a valid bystander session. |
| Supervision/recovery | Unreaped leader anchors prevent recycled-PGID signalling. Failed process readback is unknown, not absence. CLI success is finalized after cleanup. Canonical recovery uses a narrow verified no-fork boundary; arbitrary injected commands cannot claim containment. |
| Qualification | HTTP status and error-free finite telemetry are required at every memory snapshot. Node/local-tsx readiness precedes inference startup; Python Phase5 uses the supported Node HTTP host. |
| Packaging/testing | Pinned dependency contracts, source builds and isolated real tarball installs. Development CLIs use Node's tsx import hook instead of the IPC wrapper. Default Vitest workers match CI without weakening process deadlines. |

Single GPU ownership does **not** mean using one GPU core: batched MLX operations still expose device parallelism. The point is to avoid racing model allocation, inference and teardown while using batching to amortize overhead.

## Important repaired failure cases

- Wrong-but-self-consistent MiniLM embeddings and half-precision attention-mask crashes.
- Cancelled shared loads, incomplete shutdown, failed eviction, post-shutdown reload and mutable response identity.
- Descriptor drift, malformed binary/JSON vectors, renewed timeout budgets and lost wrapper cancellation.
- Native-context disposal races and leaked/reused failed cleanup promises.
- Partial indexing reported as zero committed work; cancellation arriving immediately before forced publication.
- Close/update races and destructive collection removal surviving a configuration persistence failure.
- Same-dimensional model drift, legacy fingerprints and cross-process rebuilds bypassing query validation.
- Failed searches creating/pruning caches before an identity failure was known.
- Zero/empty request IDs ignoring cancellation; duplicate IDs orphaning operations.
- Unbounded incomplete uploads and global-capacity rejection evicting an unrelated idle session.
- PID-file-only daemon termination, false startup success, recycled process-group authority and cleanup errors returning success.
- Recovery success inferred from closed pipes despite escaped descendants; unsupported containment now cannot certify success.
- Qualification passing numeric HTTP-error bodies or choosing Bun for Node-only HTTP hosting.
- Intermittent source-CLI failures caused by Unix IPC path collisions under long managed temporary directories. A controlled long/short-path reproduction and the direct-loader regression are retained.

These are the repaired and exercised contracts—not a proof that every future bug or leak is impossible.

## Local verification

Latest full release-candidate invocations:

| Gate | Actual result |
|---|---|
| `pnpm run build` | Passed |
| Full TypeScript/Vitest suite | **1,318 passed, 66 skipped; 45 files** |
| Full model-free Python suite | **1,417 passed, 2 skipped** |
| Real encoder runtime configurations | **8 runtime runs across 5 profiles passed** |
| Independent same-model numerical references | **5 first-run reference checks passed**; repeats checked against those qualified vectors |
| Real FP32 and FP16 local stack | Both passed: 8 public documents, exact JSON/binary parity, backup/restore ranking equality, SDK/REST/MCP/SSE and foreign-origin rejection |
| Real stack cleanup | Owned subprocess group cleaned, server/worker/HTTP threads stopped, final active and cached Metal allocations zero |
| Isolated Node package | CLI, SDK, SQLite/sqlite-vec, MCP stdio, REST/MCP HTTP, SSE and bounded partial upload passed |
| Isolated Bun package | CLI, SDK, SQLite/sqlite-vec and MCP stdio passed; explicit HTTP refusal passed—not HTTP feature acceptance |
| Package/source reconciliation | Both pack/install runs produced the same SHA-256; all **126 packaged files** matched the current worktree bytes |
| Shell/plist/CI checks | Shell syntax and launchd plist passed; CI parsed and test paths/dependency prerequisites checked locally |
| Whitespace checks | `git diff --check` passed |

Skipped tests are not counted as passed. They include optional integration/model coverage; these runs do not turn those cases into verified features. Counts above come from single full invocations, not sums of overlapping focused runs. GitHub-hosted CI was configured but was not run remotely.

Independent spec and quality review closures include runtime/model ownership, immutable response identity, public deadlines, transactional store accounting/close/removal, query snapshots/cache immutability, process anchors, failed-readback behavior, telemetry integrity, prelaunch runtime readiness and HTTP admission. The final HTTP review passed 24 held-out positive probes, including pre-body bounds, capacity recovery, control-plane progress and preservation of unrelated sessions. No known critical or important finding remains open within these reviewed contracts.

## Measured Apple Silicon performance

Host: **Apple M2 Pro, 32 GiB**. Local cached MiniLM and BGE-small weights; all model tests were offline and serialized. The primary MiniLM configurations were repeated in reverse order.

| MiniLM configuration | 64-text batch p50 across two runs | Retained active Metal | Peak active Metal |
|---|---:|---:|---:|
| FP32, 512-token bulk target | 85.6–85.8 ms | 86.7 MiB | 177.6 MiB |
| FP16, 512-token bulk target | 77.0–77.8 ms | 43.3 MiB | 130.0 MiB |
| FP16, 4096-token bulk target | 61.6–61.9 ms | 43.3 MiB | 379.7 MiB |

The explicit FP16/4096 configuration delivered **1.39× measured bulk throughput** versus corrected FP32/512, with about **28% lower batch latency**. FP16 halved retained active Metal allocations. Larger batches raise peak allocation and can lengthen the wait behind a non-preemptible forward, so the conservative 512-token default remains.

This is a precision/batching configuration comparison on one synthetic encoder workload—not a universal code-only speedup, an MLX-versus-GGUF comparison or a claim that every M-series GPU is saturated. Active Metal, allocator cache, peak allocation and process RSS are distinct. Zero final allocator readings followed an explicit observation barrier; immediate readings remain in the raw evidence.

See the [complete measurements and limits](../benchmarks/apple-silicon-audit/RESULTS.md), [machine-readable summary](../benchmarks/apple-silicon-audit/rc-summary.json) and [CSV](../benchmarks/apple-silicon-audit/rc-summary.csv). Failed and preliminary benchmark receipts were retained rather than relabeled as successful measurements.

## Compatibility and adoption

1. **Rebuild incompatible vector indexes explicitly.** Corrected BERT numerics, context-aware fingerprints, changed precision/pooling/revision or chunk strategy must not reuse mislabeled vectors. Back up and validate a separate rebuilt index before cutover.
2. **Use Node 22+ for MCP/REST HTTP.** Bun 1.3.8 remains supported for the tested CLI/SDK/stdio surfaces; its HTTP disconnect limitations are refused explicitly. Native SQLite bindings must match the chosen runtime.
3. **Persistent daemon startup requires an explicitly selected, already-cached model.** Testing did not install or start a persistent service. Legacy unverified PID-only daemon records are not treated as termination authority.
4. **Embedding, query expansion and reranking are separate backend choices.** A fast small-encoder result is not evidence to replace a production reranker or larger embedding model.
5. **Low-level raw precomputed vectors are a trusted caller boundary.** Untagged arrays do not prove serving-runtime provenance. Public SDK queries do not accept such raw arrays; internal generated vectors carry private provenance.
6. **Recoverable process scope is explicit.** The general watchdog owns an inherited process group, not arbitrary deliberate `setsid` escapes. Canonical recovery's no-fork guard is not a general untrusted-code sandbox. Darwin arm64 behavior was exercised; Linux seccomp was reviewed, not executed on this host.
7. **Install this fork, not an unrelated upstream registry build.** The package name stays `@tobilu/qmd` for compatibility, but the audited code is in this checkout/verified tarball. Nothing was published to that registry scope.

## Still unqualified / not performed

- Real 4B embedding, MLX query-generation and reranker model performance/quality.
- Every M-series machine, Windows or a live Linux execution environment.
- Extended soak, maximum-memory stress and aggregate memory guarantees across unrelated apps/frameworks.
- Representative private-corpus or full-corpus IR quality. Authored one-hot qualification fixtures test orchestration only; they are not model-quality evidence.
- Production rollout, live-index migration, global installation, GitHub publication or registry release.

The appropriate conclusion is **substantially repaired, measured local code**, not “all possible leaks are proven absent.”

## Evidence and deliverables

- Plan: `docs/plans/apple-silicon-audit-implementation.md`
- Baseline: `docs/reviews/apple-silicon-audit-baseline.md`
- Benchmark results and raw vectors: `docs/benchmarks/apple-silicon-audit/`
- Full-suite/package receipts: `docs/reviews/apple-silicon-audit-evidence/` (collected during final sealing)
- Verified npm tarball SHA-256: `9ed5a2766f9693844c96e8faa6a81c1c50c2b8a25aaafe3fdeb1cb6e29ddec37`

The local evidence includes source fingerprints, raw results and failure receipts. Personal/live indexes were not used. Final delivery bundles preserve the source and evidence rather than relying on temporary workspace paths.
