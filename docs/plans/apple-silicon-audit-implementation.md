# Apple Silicon Reliability and Throughput Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task. David explicitly selected the currently configured Hermes subagent route, `openai-codex` / `gpt-6.1-sol-900k`. Do not launch an external coding harness or follow historical plans' AGY routing.

**Goal:** Repair demonstrated correctness, retention, cancellation and recovery defects; improve useful Apple Silicon throughput without corrupting embedding spaces or exhausting the daily-driver Mac.

**Architecture:** Retain TypeScript CLI/SDK/MCP, SQLite FTS5/sqlite-vec, Python MLX adapters, and a single GPU execution owner. Harden ownership and bounded flow at each boundary rather than replacing proven components with a speculative rewrite. Keep embedding identity fail-closed; cache only under a verified serving identity; expose a deliberate bulk-throughput setting independently of the interactive fairness target.

**Tech Stack:** Node 22/Bun, TypeScript, Vitest, SQLite, Python 3.12, MLX/Metal, NumPy, pytest, local HTTP. Existing cached MiniLM weights support real hardware checks without downloads.

---

## 1. Scope and baseline

Baseline `1d6cfef`, branch `hermes/apple-silicon-audit`, repository `/Users/shersingh/github/qmd-mlx-search`.

Verified hardware: M2 Pro, 10 CPU cores, 16 GPU cores, 32 GiB unified memory. Native baseline build passed; Python 246 passed/13 skipped; TypeScript 804 passed/72 skipped/1 failed. See `docs/reviews/apple-silicon-audit-baseline.md` for commands, model snapshot and bounded real Metal measurements.

Allowed: repository code/tests/docs, repo-local dependencies, public/synthetic fixture databases, owned loopback test children, local artifacts. Not allowed: personal corpora/live indexes, global install/link, live daemon/launchd/config changes, reactivating cron, push/publish, new model downloads without a separate need/decision. All subprocesses must override HOME/XDG/TMPDIR before imports. No benchmark requires exhausting RAM, unrestricted parallel GPU jobs, or sysctl changes.

**Completion is evidence-based:** neither these audits nor a green unit suite proves every possible leak is gone. Report exact supported configurations, executed gates, skipped model paths, unresolved failures and performance comparisons. Do not turn an unavailable comparator into a pass.

## 2. Shared contracts (freeze before parallel implementation)

### 2.1 Embedding response identity

- Preserve binary payload `[count:i32 LE][dims:i32 LE][values:f32 LE]`.
- JSON embedding responses add `descriptor`, the actual embedding descriptor used for that result.
- Binary embedding responses add `X-QMD-Embedding-Descriptor`, containing UTF-8 JSON percent-encoded with Python `urllib.parse.quote(json.dumps(descriptor), safe='')`; TypeScript decodes with `JSON.parse(decodeURIComponent(header))`.
- Both reflect requested output dimensions. Server identity comes from its runtime, never from a request's model label.
- `MlxEmbedConfig` gains optional `expectedDescriptor` and `signal`. A production `MlxEmbedClient` negotiates/pins the descriptor and requires compatible response metadata; a missing/mismatched descriptor fails closed rather than silently falling back. Low-level transport helpers can remain usable against legacy fixtures when no expectation is explicitly configured, but cannot provide production index safety without negotiation.
- Descriptor validation requires finite positive integral dimensions/context, valid enum fields, nonempty model and normalized=true where required. `maxTokens` must participate in space identity; identity-policy changes are explicitly documented as requiring reindexing, not quietly accepted into old indexes.
- Preserve public `number[]` embedding API. Typed views/copies may be optimized internally; do not claim end-to-end zero-copy.

### 2.2 Cancellation and lifecycle

- Existing executor `submit` signatures remain compatible. A cancelled caller may stop waiting promptly, but an active Metal kernel remains owned until completion; no model unload while its forward is active.
- Shared load completion belongs to the actual load job, not the first request's deadline. Failed unloads remain accounted for and block unsafe residency admission.
- Runtime constructor gets additive `max_bulk_microbatch_tokens` option (default 512 for compatibility). `--max-bulk-microbatch-tokens` exposes it. It is a scheduling target, not an absolute single-sequence cap. Hard supported context and memory bounds remain independent.
- TypeScript `EmbedOptions`, `GenerateOptions`, `RerankOptions`, expansion options gain optional `signal`. Client wrapper `dispose()` aborts its owned requests. Per-instance operation/session leases protect idle unload; final disposal is idempotent, drains/fences loaders, and cannot resurrect resources.

### 2.3 Store/SDK and cache interfaces

- Store bound search/tokenization paths use `store.llm`, not the default singleton.
- `HybridQueryOptions` gains additive `collections?: string[]`; combine/deduplicate with existing `collection`. SDK forwards the whole list.
- Existing `deleteStoreCollection`/`renameStoreCollection` names remain; their transactional semantics include indexed document records, contexts and FTS effects, preserving content shared by other collections.
- Config injection becomes scoped/per-instance; no SDK instance may change another store's source or reset a global source on close.
- LLM owner exposes `getCacheIdentity(op: 'rerank'|'expand'): Promise<string|null>` and `expandQueryWithIdentity(query, options): Promise<{queries: Queryable[]; model: string}>`; `expandQuery` remains a compatible wrapper. `RerankResult.model` reports the actual prefixed backend/model identity for MLX and configured URI for GGUF. Cache predictions must match actual returned identity; on mismatch do not mix cached old scores with fresh fallback scores. Prefer bypass/recompute to guessing. Fake LLMs without new helpers retain backward-compatible fallback behavior in tests, not unsafe production behavior.

## 3. Parallel ownership manifest

Only the named owner edits each source file. Newly created regression files use that owner's unique prefix. No child stages, commits, changes skills, installs packages, launches a harness or runs the full suite. Parent owns integration, plan/results/README/CHANGELOG, evidence artifacts and final acceptance.

| Lane | Exclusive source ownership | New regression prefix |
|---|---|---|
| W01 | `scripts/qmd_mlx/executor.py` | `test/python/test_audit_executor.py` |
| W02 | `scripts/qmd_mlx/model_manager.py` | `test/python/test_audit_residency.py` |
| W03 | `scripts/qmd_mlx/runtime.py`, `batching.py` | `test/python/test_audit_batching_runtime.py` |
| W04 | `scripts/qmd_mlx/adapters/embedding.py`, `adapters/tokenization.py` | `test/python/test_audit_embedding_adapters.py` |
| W05 | `scripts/qmd_mlx/rerank.py`, `generate.py` | `test/python/test_audit_stage_adapters.py` |
| W06 | `scripts/qmd_mlx/server.py`, `protocol.py`, `fake_server.py`, `scripts/mlx_embed_server.py` | `test/python/test_audit_http.py` |
| W07 | `scripts/qmd_mlx/supervisor.py`, `watchdog.py` | `test/python/test_audit_supervision.py` |
| W08 | `src/mlx.ts`, `src/embedding/{config,contract,protocol}.ts`, `src/expansion/protocol.ts`; existing `test/mlx.test.ts`, `test/embedding-contract.test.ts` | `test/audit-transport.test.ts` |
| W09 | `src/llm.ts`, optional new `src/llm/*`; existing `test/llm.test.ts`, `test/expansion.test.ts` | `test/audit-llm-lifecycle.test.ts` |
| W10 | `src/store.ts`, `src/indexing/*`, `src/db.ts`, `src/maintenance.ts`; existing store/indexing/chunking tests as needed, excluding SDK/MCP/phase5 tests | `test/audit-store.test.ts` |
| W11 | `src/index.ts`, `src/collections.ts`; existing `test/sdk.test.ts`, `test/collections-config.test.ts` | `test/audit-sdk.test.ts` |
| W12 | `src/mcp/server.ts`; existing `test/mcp.test.ts` | `test/audit-mcp-http.test.ts` |
| W13 | `scripts/phase5_recovery_worker.ts`, `phase5_e2e_runner.ts`, `scripts/qmd_mlx/phase5_e2e_qualification.py`; existing `test/phase5-recovery.test.ts`, `test/python/test_phase5_qualification.py` | `test/audit-recovery.test.ts` |
| W14 | `scripts/qmd_mlx/sustained.py`, `scripts/qmd_gguf/benchmark.py`, `scripts/qmd-mlx-sustained.py`, `scripts/qmd-gguf-benchmark.py`; existing sustained/GGUF Python tests | `test/python/test_audit_qualification.py` |
| W15 | package/lock/build/CI files, Python dependency files, `bin/qmd`, daemon/launchd/install/smoke/launcher scripts, `scripts/apply-fork-dist-patches.py`; no README/CHANGELOG edits | `test/python/test_audit_packaging.py` |

All lanes may run in parallel against these contracts. W08/W09/W10/W11 integration, W01/W02/W03/W05 scheduling, W04/W06 metadata and W07/W13 subprocess behavior are dependency gates: focused tests can be temporarily blocked pending a sibling, but they cannot be reported fully accepted until parent integration passes.

## 4. Implementation tasks and red/green checks

Each numbered item below is a small test/fix/retest step inside its lane. Save failing evidence before fixing; run focused tests after each meaningful change. Report exact changed paths and tests, not generalized success.

### W01 — Queue cancellation, capacity and fairness

1. Add a barrier-controlled test: one running task, a cancelled pending task, then another submission. Pending cancellation must release queue capacity promptly.
2. Poll/wake on cancellation while waiting without declaring a running native job finished. Remove only pending cancelled/expired jobs; retain active-job progress truthfully.
3. Build a mixed priority heap and assert the oldest eligible bulk sequence runs at the fairness boundary, not the first heap-array match.
4. Test shutdown rejects new work, wakes pending callers and cannot execute a cancelled pending closure.
5. Run `pytest test/python/test_audit_executor.py test/python/test_mlx_executor.py -q` through the isolated runner.

### W02 — Transactional residency/load ownership

1. Reproduce a short-timeout initiating loader and longer-lived concurrent waiter; the latter must await actual completion and succeed.
2. Make the actual load operation own generation-safe completion/event state. After `load()` validate `is_loaded()`, and unconditionally attempt rollback of partially published weights on failure.
3. Inject failed unload before references are dropped. Failed residency must stay visible/accounted, and admission of another model must fail closed when over budget. Cover explicit, idle and LRU eviction.
4. Test retries cannot mark a partial adapter READY and no waiter is left unsignaled.
5. Run unique tests plus existing model-manager suite.

### W03 — Runtime rollback, memory-effective retries and configurable bulk flow

1. Unsupported model and failed eager-load constructors must not leave owned executor threads running; preserve externally supplied ownership.
2. Exit the exception/traceback scope before OOM bisection so failed forward arrays are collectible during retries. Regression uses weakrefs; retain result order/cardinality and bounded retry depth.
3. Separate `max_bulk_microbatch_tokens` fairness setting from hardware/model max batch budget. Validate configuration and retain safe default; test eight long documents under low/high settings with exact padded-token accounting.
4. Replace concatenate-plus-row-copy restoration with one output allocation and vectorized scatter, returning the original array for an identity single batch where safe. Validate result cardinality before scatter.
5. Keep all load/forward/dispose work serialized; orderly runtime shutdown unloads owned resources before stopping its owner when possible, never using a caller thread against a live kernel.
6. Run runtime/batching tests. Parent alone compares real Metal throughput and memory before/after.

### W04 — Truthful, model-aware embedding adapters

1. Test tokenizer and weights resolve the same revision. Load supported encoder weights from one resolved immutable/local snapshot; reject unsupported requested quantization instead of merely labeling it.
2. Apply and report actual floating compute dtype; accumulate mean pooling/norm in float32. Ensure failures roll back partial adapter state.
3. Measure/resolve descriptor revision and precision truthfully, including local weight identity where practical. Preserve tokenizer caching only as intentional CPU retention.
4. Correct model-family support: reject unqualified Gemma-to-Qwen mapping; BGE requires supported BERT architecture with CLS semantics, MiniLM mean, Qwen last token. Fail unsupported Nomic variants explicitly rather than fabricating support.
5. Use a batched tokenizer fast path when available, with encode-only fallback and exact once-only IDs/lengths/order; no silent truncation. Remove redundant already-float32 copies only after ownership/parity tests.
6. Run adapter/tokenization regression tests with spies. Parent verifies cached MiniLM on actual Metal; no worker GPU use.

### W05 — Reranker/generator correctness and useful GPU work

1. Reranker load is transactional: yes/no token resolution failure unloads weights; retry genuinely reloads and validates.
2. Recognize tied output projection (`embed_tokens.as_linear`) and project only final hidden states, not `[B,S,V]`; test selected scores and projection shape against a reference fake.
3. Tokenize/format outside the GPU owner where safe; submit bounded rerank microbatches individually to permit interactive embedding interleaving. Protect all actual forward/load work with stage leases.
4. Enforce prompt token length + **requested** `max_tokens` against context. Test `100+4096 >4096` refuses and `3500+1 <=4096` accepts under an otherwise compatible fixture.
5. Cover cancellation at boundaries, unload/reload and malformed token IDs. Run existing and new stage tests.

### W06 — Bounded HTTP and result identity

1. Add the missing socket import and close/deregister on every rejection/stop path; test peer EOF, registry count and semaphore recovery after repeated overload and half headers.
2. Establish absolute header/body read budgets before consuming data; a regular drip below inactivity timeout must not hold a handler indefinitely. Request timeout cannot restart after upload.
3. Propagate peer disconnect to request cancellation without misidentifying a normal HTTP half-close or unloading active kernels. Ensure helper monitors are bounded and joined.
4. Validate finite integral dims and numeric options, rejecting NaN/Infinity/fractions/bools with typed 4xx errors. Fail malformed framing before allocation.
5. Add descriptor result envelopes/percent-encoded binary header exactly as shared contract; synthetic fake server follows the same wire shape but retains explicitly synthetic metadata.
6. Wire CLI bulk microbatch target to runtime. Test help/startup with no models; run HTTP/protocol/startup tests.

### W07 — Honest supervision and bounded cleanup

1. Watchdog rejects dead worker/failed state after startup grace, while allowing documented idle states.
2. Make HTTP total deadlines actually interrupt slow-drip reads and close every response. Avoid unbounded helper threads.
3. Seek/read bounded log tails; define finite on-disk child-log behavior without loading the whole file.
4. Spawn owned subprocess groups and terminate/reap the entire owned tree on timeout/breach, never an unrelated sentinel; do not treat process exit as group cleanup.
5. Run synthetic socket and parent/grandchild fixtures with short deadlines and retained evidence. No launchd/live process changes.

### W08 — TS bounded transport and contracts

1. Full deadline + caller abort covers headers/body; bounded streaming reader applies separate error/JSON/binary limits, discards/cancels non-success bodies, limits logged error text and frees timers/listeners.
2. Validate immutable config, finite positive integral concurrency/batch size/timeouts/dims. Preserve `binary:false` and never extend a caller deadline with `Math.max`.
3. Pin and validate serving descriptors as shared contract; reject daemon A→B switches, including same-dimensional ones. Validate every JSON/binary count/dimension/value and exact empty frame length.
4. Include context limit in descriptor identity and make unknown-family prompt behavior explicit rather than applying Gemma prefixes to all encoders.
5. Bound expansion response bytes/line counts; do not treat ASCII-only overlap as semantic verification for non-Latin text.
6. Fix the existing false timeout test so the server actually stalls `/slow/embed`. Test real streamed oversized and delayed bodies, cancellation and cardinality.
7. Run unique tests plus MLX/contract/expansion protocol tests, respecting existing-file ownership.

### W09 — Per-instance LLM lifecycle and actual identity

1. Per-instance shared session/operation tracker prevents its idle unload during its own operation, without global singleton gating unrelated stores.
2. Serialize context pool creation, fence/drain every late async loader on dispose, clean partial generation contexts in `finally`, make close idempotent and arm generation-only idle cleanup.
3. Detach external abort listeners and propagate session cancellation to transport/model calls. Validate maxDuration and reject post-release success.
4. Enforce strictMlx regardless of missing opt-in environment flags. Prohibit cross-space embedding fallback after selection; no stale MLX descriptor paired with GGUF vectors.
5. Forward all resolved client settings and implement the shared cache identity/expansion envelope API. Actual rerank/generation identity follows the serving backend, not intent.
6. Add races using controlled promises and no real models. Run LLM/expansion and unique regressions; report dependent integration gates honestly.

### W10 — Store/index integrity and scalability

1. Forced rebuild must preflight the backend before deleting anything and replace/reset vector metadata consistently. If safe atomic replacement cannot be completed, refuse and preserve the previous index; never erase then reject stale metadata.
2. Persist completed chunk strategy/identity, reject or explicitly rebuild incompatible strategy even when no active checkpoint remains. Compare positions/content against a fresh strategy index.
3. Select only genuinely missing/incomplete documents; avoid the active-checkpoint condition selecting all hashes. Use bounded keyset pages/streamed batches and skip completed documents before fetching bodies. A no-op durable job loads zero bodies.
4. Use the per-store LLM for vector queries and chunk tokenization. Honor all collections; move collection FTS filtering inside ranking rather than an incomplete global overfetch prefix.
5. Implement transactional indexed-document rename/removal semantics shared with SDK, preserving other collections' content and index integrity.
6. Namespace caches by configured/actual serving identity and avoid mixed-backend cached/fresh scores; use the shared LLM helper when present. Preserve intentional test-double compatibility.
7. Focused tests cover forced success/failure preservation, completed-strategy change, same-sized different spaces, scoped FTS beyond prior cutoff, incremental no-op SQL, cache changes, and store-specific tokenizer/backend. Maintain public compatibility.

### W11 — SDK/config isolation

1. Replace SDK reliance on global configSource with explicit scoped/per-instance config handling, including initialization and failure cleanup.
2. Interleave two inline, two YAML and mixed DB-only stores through mutations/close. Assert only the intended config/DB changes and no default path writes.
3. Forward all collection scopes to hybrid search, default embedding fallback false, and correct per-store vector configuration.
4. Close is idempotent and cleans DB/model resources even on failure; construction failures close owned DB/LLM.
5. Test rename/remove against real seeded indexed docs once W10 lands; never patch W10 files from this lane.

### W12 — MCP/REST streaming and security

1. Replace unbounded `collectBody` with bounded body parsing; validate REST payload types/ranges and map malformed input to 4xx.
2. Stream Web Response bodies with backpressure, flush SSE headers promptly, propagate disconnect/abort and close/cancel readers on stop. No `arrayBuffer()` on an unbounded SSE response.
3. Implement loopback Host/Origin protection in versioned source for both REST and MCP; reject foreign/null/unexpected origins before work. CLI clients with no Origin remain supported under valid Host.
4. Bound session count and idle lifetime; dispose session/server/transport resources on expiry, failed initialization and shutdown.
5. Exercise actual owned localhost HTTP listeners with MCP initialize, query/get, SSE GET disconnect, oversized body and origin attacks; no GUI acceptance is needed for this CLI/server project.

### W13 — Actual fresh-process recovery and restoration

1. Repair undefined worker variables (`abortCtrl`, `barrierDocs`, `docCount`) and bounded barrier protocol. Execute worker main, not import-only code.
2. Launch separate child processes against a fake inference fixture; stop after committed partial work, terminate, reopen and resume. Verify exact vector/chunk/checkpoint state, not just nonzero counts.
3. Bound all subprocess waits and coordinate owned-group cleanup with W07; preserve unrelated sentinel.
4. Correct the stale positive equivalence fixture by providing finite matching scores. Add missing/nonfinite/wrong-length/duplicate-query refusals without weakening helper validation.
5. Exercise actual restore on disposable DB and compare ranked document IDs and scores before/after; assert mismatch rejection preserves existing logical data.
6. Tests must remain offline and runnable in CI; model/synthetic qualifications clearly separated.

### W14 — Qualification must fail closed

1. `evaluate_pilot_acceptance` must reject nonoverlap, unstable verdict, missing/nonfinite memory telemetry and violated Metal/process resource criteria. Test each independent negative plus a complete valid positive.
2. Token throughput uses actual tokenizer counts only; otherwise report unavailable or explicitly named words/sec, never substitute words for tokens.
3. Enforce documented pilot limits consistently across MLX/GGUF CLIs, preserve honest differences in soak support, reject contradictory acceptance data.
4. Run qualification tests without models. Historical failed artifacts stay unchanged; new strict evaluation may classify old artifacts as unqualified, not retroactively passed.

### W15 — Reproducible package/daemon/CI surface

1. Check in an appropriate tested Python dependency lock/constraints without polluting global Python. Add portable offline Python safety CI plus build gate; dependencies unavailable on Linux are explicitly handled, not silently skipped tests.
2. Make source install/pack produce required dist files and include nested Python runtime/requirements. Verify with `npm pack`/isolated tarball install (no publish). Build/install scripts must not mask exit codes.
3. Modern npm native dependency approvals must be narrow and deterministic; current parent-owned package delta approves better-sqlite3. Preserve or improve it intentionally.
4. Render launchd plist structurally with escaped paths and explicit selected env/ports/models/log dirs. Stub launchctl tests validate custom settings, failed startup, missing models and listener ownership; never install a real job.
5. Retire the external-home live-dist patch dependency in favor of source-based security (W12). Never patch the user's installed package.
6. Test real bin/qmd runtime/argv behavior, fix smoke copy of nested source files and command pipelines. Run shell syntax/plutil checks and package inventory. Parent owns README/CHANGELOG final edits.

## 5. Test execution and evidence

An existing parent helper executes each command with fresh fixture HOME/XDG/HERMES_HOME/TMPDIR and offline flags:

```sh
python3 /Users/shersingh/.hermes/cache/scratch/qmd-audit/run-gate.py W01-tests \
  /Users/shersingh/github/qmd-mlx-search/.venv/bin/python -m pytest \
  test/python/test_audit_executor.py test/python/test_mlx_executor.py -q

python3 /Users/shersingh/.hermes/cache/scratch/qmd-audit/run-gate.py W08-tests \
  node node_modules/vitest/vitest.mjs run --maxWorkers=1 \
  test/audit-transport.test.ts test/mlx.test.ts
```

Use unique labels. Focused suites only during parallel writes; no concurrent full suites or GPU workers. All failures keep real nonzero status. New negative tests must fail on the relevant baseline behavior before fixes where practical; preserve characterization scripts separately from acceptance tests.

## 6. Parent integration and two-stage independent review

1. Read every worker's diff/new files and rerun focused regressions; resolve shared contracts. Check ownership manifest for overlapping writers before each dispatch.
2. **Spec review** by fresh read-only workers: findings-to-fix coverage, no unsafe scope expansion, missing cases, full identity and lifecycle path composition. Repair gaps before quality signoff.
3. **Code/release review** by different workers after spec passes: concurrency/resource cleanup, package/daemon behavior, recovery through actual subprocesses, bounded allocations, skipped dependencies and truthful docs.
4. Run full build, full isolated Vitest and Python suites, real launcher/package smoke, shell syntax, diff check and status. Script entrypoints excluded by standard tsc require explicit runtime/type validation.
5. Run actual MLX HTTP JSON/binary + SDK/MCP against cached MiniLM/public disposable docs, then numerical parity and repeated-load/unload resource checks. Sequential hardware probes only.
6. Compare baseline/candidate on identical inputs/model/precision. Report p50/p95/throughput and active/cache/current-RSS separately. Repeat experiments to avoid treating thermal/noise changes as optimization. High-throughput bulk mode is a tradeoff with interactive latency and stays explicit unless evidence supports a safe new default.
7. Do not assert 4B, every M-series Mac, GGUF superiority or representative retrieval quality from a small MiniLM fixture. A lengthy unattended soak or max-memory stress requires a separate operator window.

## 7. Deliverables and acceptance matrix

- This plan and baseline report.
- Reviewed working local code plus regression tests (no push/deployment).
- Durable benchmark runner/results and package validation receipt.
- `docs/reviews/apple-silicon-audit-results.md`: each lane implemented/verified/blocked, commands/counts, measured before/after, breaking identity changes/reindex guidance, resource limitations and remaining work.
- Updated README/CHANGELOG with measured claims only; no blanket 'all leaks fixed', '2–5x faster', automatic universal MLX support or zero-copy end-to-end assertion.
- Final user report leads with actual completion state and the strongest verified improvements, supplies plan/results paths, and explicitly distinguishes code acceptance from production activation.

## 8. Review-derived closure work (additional mandatory acceptance)

These are extensions of the original lanes, not permission to deploy. A reproduced negative remains failed evidence; only a new positive-contract run can close it. Final status belongs in the results report, not in optimistic plan checkboxes.

### R01 — Bind CPU preparation, GPU work and response identity

**Files:** `scripts/qmd_mlx/adapters/embedding.py`, `runtime.py`, `server.py`; new `test/python/test_runtime_generation_ownership.py`.

1. Pause the inherited tokenizer implementation after admission, finish shutdown, resume it, and assert no off-owner weight load or retained model appears. Test owned and supplied executors.
2. Remove implicit weight loading from CPU tokenization. Acquire an atomic tokenizer-only snapshot and verified serving identity without retaining a stale weight lease that blocks shutdown.
3. Change local weight/tokenizer assets and evict between microbatches; assert the request refuses mixed revisions rather than returning `[A, B, B]` under descriptor B.
4. Carry the completed operation's captured descriptor to JSON and binary serialization. A subsequent request must not relabel earlier vectors through a mutable current-descriptor lookup.
5. Preserve cancellation, fairness, token-once batching, BERT segment-zero embeddings and compute-dtype attention masks. Run generation-ownership and HTTP/runtime regressions before sequential real-Metal reruns.

### R02 — Enforce the public LLM deadline, not only transport deadlines

**Files:** `src/llm.ts`, `src/llm/lifecycle.ts`; `test/audit-llm-deadline-review.test.ts`.

1. Through public `LlamaCpp.embed`, set an 80 ms budget while descriptor and inference exchanges each take 60 ms; overdue vectors must never be returned.
2. Cover native and projected dimensions, batches, metadata fallback and tokenizer calls with one operation budget.
3. Cancel one initialization waiter while preserving another; cancel the shared load only after its final waiter leaves. Do not apply an MLX transport timeout to unrelated GGUF cleanup.
4. Run fake-clock regressions and separate real-clock probes, then independently recheck the public route.

### R03 — Truthful transactional store lifecycle

**Files:** `src/store.ts`, `src/index.ts`, `src/db.ts`, `src/indexing/job.ts` as coordinated; `test/audit-store-review.test.ts`, `test/audit-store-quality-regressions.test.ts`.

1. Cancel two microtasks after final progress but before forced-rebuild publication. Direct and durable paths must roll back, report zero committed replacement chunks and preserve every prior row/vector/checkpoint.
2. Cancel incremental indexing after a committed batch. Preserve truthful committed/attempted/document counters and checkpoint progress; do not apply forced-rebuild rollback accounting to retained incremental writes.
3. Start `close()`, hold disposal, then attempt update/config/search operations. Reserve closing ownership before awaits; no cached failed close may permanently strand an open database.
4. Make YAML unreadable/unwritable during collection removal. Destructive document/FTS changes must remain rollbackable until persistence succeeds; test successful removal and compensation/error reporting separately.
5. Keep foreign reads/writes and callback transactions from inheriting an active replacement's ownership; never acknowledge a mutation that another operation can roll back.

### R04 — Validate index identity at query time

**Files:** `src/store.ts`, optional SDK forwarding in `src/index.ts`; `test/audit-query-identity.test.ts`.

1. Seed an index under revision A, serve revision B with the same dimensions, then exercise public vector and structured/hybrid search. Both must refuse incompatible vector use.
2. Preserve the stored space ID as historical evidence. An old fingerprint-policy ID must not be accepted merely by recomputing a new ID from its descriptor.
3. Validate stored metadata against the actual serving/response identity, including internal precomputed embedding paths. Recheck safely around awaits and concurrent rebuild ownership.
4. Query validation must be read-only: no `ensureVecTable`, metadata rewrite, deletion or silent relabeling. Keep lexical search usable and explicitly define legacy GGUF/raw-vector compatibility boundaries.
5. Re-run `query-identity-gap.mts` as a positive refusal gate; preserve the original failing receipt showing incompatible results accepted.
6. Enclose final identity validation, KNN and document materialization in one coherent SQLite read snapshot. Test a real separate-process forced replacement at the final boundary; a per-connection guard cannot prove cross-connection consistency.
7. Stage expansion/rerank cache mutations per query and publish only after successful completion. Initial mismatch preflight must avoid wasted work, and drift after expansion must still leave persistent cache state unchanged. Do not hold an async SQLite write transaction across inference.

### R05 — Non-recyclable process authority and honest cleanup verdicts

**Files:** Python watchdog/supervisor and public watchdog CLI; TypeScript recovery runner/worker and narrowly scoped helpers; corresponding process-tree regression files.

1. Simulate PGID reuse after the owned leader exits and between TERM/KILL. Numeric group existence alone must never authorize signalling; retain a non-recyclable ownership anchor or revoke authority when continuity is unprovable.
2. Preserve valid orphan-grandchild cleanup without affecting unrelated sentinels. Do not solve reuse by quietly abandoning all legitimate post-leader cleanup.
3. Inject signal denial through the real watchdog CLI. Finalize exit status and machine-readable overall success only after teardown; cleanup failure must be nonzero and explicit.
4. Exercise descendants that create another group in the owned session and descendants that detach sessions, both with redirected/closed standard streams. Pipe closure and disappearance of one PGID do not prove descendant cleanup.
5. Account for the supported owned-session boundary and refuse unsupported/unprovable containment. Never claim a generic process sandbox from a one-shot PPID scan, and never kill unrelated processes to make a test green.
6. Assert targeted PID absence before outer fixture cleanup. A fixture's emergency cleanup is not product acceptance.
7. Inject failed/diagnostic-bearing `ps` group readbacks while the leader is a retained zombie. Unknown telemetry must retain the anchor and report nonzero `cleanup_failed`, never absence/success. Keep the legitimate clean-empty status-1 case working.

### R06 — Runtime compatibility and fail-closed qualification

**Files:** MCP server, CLI daemon sections, launchd helper, qualification/recovery scripts, package smokes and CI; README/CHANGELOG owned by the parent.

1. Propagate disconnect/deadline/stop/MCP cancellation through the SDK to actual tokenizer and embedding HTTP work, not merely a raced response promise.
2. Explicitly require Node 22+ for HTTP hosting where Bun 1.3.8 disconnect semantics cannot meet the contract. Preserve and independently exercise native Bun CLI/SDK/SQLite/MCP stdio; test HTTP refusal without calling it HTTP feature acceptance.
3. Use `bun run test` for the Vitest suite. Verify the test host really is Node; separately install tarballs and run native Node/Bun product checks. Launch TypeScript CLI fixtures with Node and an absolute `tsx` loader import, not the `tsx` CLI's Unix IPC server; test concurrent invocations under long managed TMPDIR paths without escaping to system temporary storage.
4. Require explicit cached model selection for persistent MLX daemon startup. Verify MCP listener PID/token before reporting startup or acting on PID records; reject legacy/unverified ownership.
5. Reconcile every concurrent workload sample bidirectionally with interval receipts. Reject missing/orphan/duplicate observations, frozen telemetry, insufficient interval coverage and nonfinite metrics.
6. Phase5 composition must fail on subordinate cleanup errors, unavailable final telemetry or invalid preflight values even when a child exits zero.

### R07 — Own MCP cancellation independently of request-ID edge cases

**Files:** `src/mcp/server.ts`, `test/audit-mcp-request-ownership.test.ts` and related HTTP/stdio fixtures.

1. Use valid numeric ID `0` and empty-string IDs in real cancellation notifications. Downstream work must abort and settle; a 202 notification acknowledgement alone is not cancellation evidence.
2. Reject duplicate IDs within an entire batch and across active requests in the same session before SDK ownership can be overwritten. Preserve number/string identity and allow the same ID in different sessions.
3. Keep operation ownership until actual completion/transport cleanup, then permit legitimate ID reuse. Stop, disconnect, deadline and session deletion must reach every accepted operation.
4. Bound active ownership/admission and prove overflow starts no work. Release every reservation on errors and cancellation; do not fix duplicates with an unbounded registry.
5. Keep the workaround in versioned source using public SDK/transport contracts where possible. Do not modify installed dependencies or blindly upgrade the SDK. Exercise HTTP and shared stdio behavior.
6. Rerun the held-out cancellation probe, focused suites and independent review before the final package/integration refresh.

### Final frozen-tree gate

1. Wait for all writers to finish; receive independent targeted spec and quality verdicts and repair outstanding important findings.
2. Freeze/fingerprint the final implementation. Run full isolated TypeScript and model-free Python suites, build, shell/plist checks and diff/status validation without masking failures.
3. Re-run real MiniLM and BGE numerical references and bounded Metal cleanup checks against the final runtime; compare only semantically correct, clearly labelled configurations. No long-soak or large-model claim follows from these fixtures.
4. Re-run real MLX → SQLite/SDK → Node REST/MCP/SSE integration on authored public data.
5. Repack and reinstall both supported runtime variants from the final tree. Earlier successful package receipts are preliminary if code or packaged documentation has changed.
6. Preserve failed and preliminary receipts, collect final durable evidence, write the acceptance/limitations report and deliver the local artifact. No push, global installation, live-index migration or production activation is implied.
