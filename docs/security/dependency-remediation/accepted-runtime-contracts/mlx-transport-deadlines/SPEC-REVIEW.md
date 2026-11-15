# Independent SPEC review — revised MLX transport proposal

**Verdict: PASS for this exact two-file proposal. No blocking SPEC finding.** This is a source review, not integration, packaging, remote-CI, merge, or release approval. QUALITY review is a subsequent stage.

## Source binding and isolation

- Frozen candidate: `/Users/shersingh/.hermes/cache/scratch/qmd-transport-parent-repair-dbt2fr5r`.
- Independent probe root: `/Users/shersingh/.hermes/cache/scratch/qmd-transport-review-revised-zr5vj8qi`.
- Authenticated INDEX SHA-256: `946172a89b3ee0d688d856e866f9e135f88e198761fd4cc27985ff4e75f34659`; all **1,369** indexed artifacts passed path/hash validation and byte authentication before replay, and passed again after execution. Each variant's source-input closure has **298** files. The deliverable differs from that base only at the two owned paths.
- Full proposal SHA-256: `a2e31a613318e78801203bb068b9f79e2e5a54faf7fb6b051eea5c1d37b48640`.
- Parent-only test correction SHA-256: `acf23a5d7c4822e70fa546e7d0da91ff1674d2c684e11d7f7b3cc4b7d83bb38d`.

| Path | Exact preimage SHA-256 | Exact reviewed postimage SHA-256 |
|---|---|---|
| `src/mlx.ts` | `dbd4dc66ab506d1611b4b48ee502a97b6e5d7ecc14092836980ecea1e844f6a7` | `2cd9aa0d12b0405b4b7147d2a2d7c12ccd7f9b0448b351c1b92115957a845751` |
| `test/audit-transport-review.test.ts` | `50706fec09208aebd8391a29543a6953f282e8c07ebe8b8e4333a2772c14e127` | `b1489d0735ce1dccc351eca930110d6dc0235ea82d595a9252c8394489b0947d` |

No live or frozen artifact was edited. Only this new scratch root contains authored probes, copies, caches, or evidence. All executing variants have owned node_modules/cache containers and **281** freshly resolved read-only package aliases. Subprocess environments are allowlisted, synthetic-HOME/Hermes/knowledge/XDG/HF/TMP/tool-cache and offline; no model, GPU, daemon, install, network, or unrelated recovery work was used. Active runtime/compiler read, dependency-write, and connection/fetch refusal controls passed on both Node and Bun. These are in-process guards, not an OS sandbox or complete Bun native-loader mediation.

## Findings against the specification

### 1. Deadline classification is based on monotonic expiry, not timer dispatch — PASS

Inspected the complete production delta (+22/-11), `RequestScope`, `withScope`, `abortable`, the four public client methods, metadata fallback, concurrent workers, and the `LlamaCpp` composition callers.

`embeddingConfig` now retains the actual outer scope, associates its signal with its absolute deadline during descriptor negotiation, removes that association in `finally`, and calls `checkScope(scope)` **before** interpreting null metadata as invalid protocol (`src/mlx.ts:553–575`). The original cancellation/error path remains intact. This addresses the demonstrated situation where valid descriptor bytes arrive after both scopes expire while timeout callbacks cannot yet run.

Independent execution on exact old production bytes with byte-identical revised permanent tests produced **5 failed / 60 filtered**: all four real synchronous-overrun methods failed with `EmbeddingProtocolError` at the original embeddingConfig path, plus the expired-sole-consumer publication assertion. Raw observations confirm `descriptor:null` and `callerAborted:false` for all four methods. The revised source passed those cases as part of **87/87** permanent tests. This establishes the local source mechanism, not the exact unobserved remote-CI scheduling or promise identity.

Held-out tests independently checked t=79, t=80, and t=85 after delayed metadata admission: only t=79 may publish or dispatch inference; t=80/t=85 return timeout/null without publication. A nondeliverable `<`→`<=` publication mutant caused **only t=80 to fail** (2 passed / 1 failed / 27 filtered), at the cache-publication assertion, not an import or setup error.

### 2. Publication belongs to an actual live registered consumer — PASS

`_negotiation.waiters` holds waiter identities, not a count; an embedding waiter carries its own scope deadline. Publication requires at least one non-aborted waiter whose deadline is still strictly in the future (`src/mlx.ts:521–545`). Direct metadata callers retain `Infinity` as their consumer deadline while the shared metadata request still has its own bounded scope. Therefore an expired sole embedding consumer cannot cache a descriptor, but an independently live embedding or direct-metadata consumer can.

Independent held-outs passed for:

- sole expired consumer and the exact equality boundary;
- explicit-signal and signal-less direct-metadata survivors after embedding expiry;
- direct-metadata cancellation with either a still-live or expired embedding consumer;
- embedding cancellation with an actual direct-metadata survivor;
- direct-first and embedding-first mixed consumers;
- two calls sharing one external signal but owning different internal scope signals/deadlines;
- cached-descriptor fast paths and early getDescriptor rejection;
- abandoned request completion while its replacement negotiation is already pending, including disposal of that replacement.

The direct-owner negative leaves registration unchanged but excludes Infinity-deadline consumers only at publication. Both mixed-consumer cases then fail at the cache assertion **after** the direct promise successfully returns metadata (**2 failed / 28 filtered**). This qualifies the held-out ownership oracle, rather than counting a failed setup premise as targeted coverage.

### 3. Parent's shared-request witness correction is causal — PASS

The permanent real-clock test now resolves readiness from the **actual descriptor fetch** and requires one existing non-aborted descriptor request before starting the younger call (`test/audit-transport-review.test.ts:347–397`). Its unresolved response prevents a cache-hit path. For the younger synchronous invocation, the known live lifetime, fresh scope, and held existing negotiation make registration reachable; the method-entry counter alone is not treated as sufficient evidence.

An observational copy of that exact positive fixture recorded:

1. first entry: 1 request, 1 actual registered waiter, non-aborted signal;
2. request-readiness barrier resolved;
3. younger entry: still 1 request, **2 actual registered waiters** with distinct deadlines;
4. older API barrier resolved null;
5. surviving request remained non-aborted and the younger operation succeeded;
6. both operations fulfilled before teardown disposal, with no negotiation and zero request abort listeners.

The independent **85 ms pre-entry negative** preserved the original 80 ms budget, 100 ms barrier, and existing oracles. It failed **1 / 64 filtered**. Additional labelled observations resolve the handoff's ambiguity between the two identically worded barriers:

- the old entry saw an already-aborted signal, no negotiation, **0 waiters**;
- the **older API barrier resolved null**;
- the **request-readiness barrier rejected**, with 0 descriptor requests and exactly 1 method entry;
- no younger entry started.

Thus the revised fixture rejects the invalid shared-request premise; it is not failing merely because the older API barrier happened to expire. The retained parent negative and raw output remain negative evidence, not relabelled green. No product poisoning defect is inferred from the superseded witness defect.

### 4. Absolute 80 ms API budgets and documented result shapes remain enforced — PASS

The cold fixture's API marker follows fixture preparation and immediately precedes invocation. The real synchronous overrun starts inside stream.pull, after the request scopes exist. The unchanged real limits are **80 ms / 100 ms**, and cold combined work remains **30 ms synchronous metadata + 90 ms noncooperative inference**. Fake-clock boundary tests supplement these real-clock tests.

The existing signal-forwarding budget-renewal mutant still fails **all four** cold methods at the real 100 ms barrier (**4 failed / 61 filtered**). The revised code passes the real cold/fallback/queue cases; no phase budget was inflated or restarted by this proposal. The public timeout shapes remain null for embed/binary and null entries for batch/concurrent. The independent Node run also passed the real `LlamaCpp` owner-composition file's **22** deadline tests alongside the transport file's **65** tests.

### 5. Cancellation, stale results, retry, and cleanup — PASS

Caller cancellation and client disposal remain terminal. Production cancellation checks precede cached descriptor returns and state publication; the last exiting waiter cancels only its own negotiation identity. The finally identity guard cannot clear a newer negotiation. Late response bodies are cancelled by the existing explicit abort race; no expired path dispatches inference in the exercised controls.

Permanent tests cover all four methods' pre-entry and in-flight caller cancellation, metadata/inference disposal, last-waiter cancellation, body cancellation, and clean retry. Held-outs additionally prove old completion cannot clear or overwrite a replacement already in flight. Temporary WeakMap entries are absent after success, timeout, cached fast-path, malformed metadata, and early rejection; waiter sets, timers, and caller/request abort listeners are released. Removing only WeakMap deletion causes the four early-rejection checks to fail at the association-cleanup assertion (**4 failed / 26 filtered**), after the intended cancellation seam was reached.

An injected intermediate assertion failure was reached with **2 actual registered waiters**. The permanent finally released the response and drained both operations before disposal; the recorded lifetime was still active, negotiation absent, and request listener count zero. The run remains an expected **1 failed / 64 filtered** control, with no reported unhandled errors. The real cold negative also completes its noncooperative exchange cleanup before test teardown. Disposal-as-action remains distinct from teardown disposal.

### 6. Strict metadata, serving identity, dimensions, and projection refusal — PASS

The proposal does not broadly convert protocol errors to null, weaken the pinned descriptor, or introduce fallback. Live-budget malformed negotiation remains an `EmbeddingProtocolError` for all four APIs. Additional held-outs refuse model/revision/backend/geometry/normalization/pooling mismatch and serving revision drift. Existing JSON/binary projection tests still validate the requested serving contract and leave the negotiated native contract pinned. No public signature, daemon protocol, backend selection, model choice, or configuration-root change is present.

## Independent executions (not borrowed parent passes)

Exact argv, source manifests, runtime receipts, observed dependency hashes, raw stdout/stderr, parsed observations, and individual failed assertions are in `evidence/independent-review-summary.json` and each `review-*.receipt.json` / `.vitest.json` / `.review-*.json`.

| Execution label | Actual result |
|---|---|
| `review-frozen-node` | **87 passed**, 2 files: revised transport 65 + composition 22 |
| `review-native-bun` | **45 passed / 20 filtered**, actual Bun launcher **and** fork worker |
| `review-heldout-node-v2` | **30 passed**, 1 reviewer fixture file |
| `review-heldout-bun-v2` | **30 passed**, actual Bun launcher and worker |
| `review-strict-node` / `review-strict-bun` | Both exit 0; **381 declaration reads per host** |
| `review-exact-old-red` | Expected **5 failed / 60 filtered** |
| `review-renewal-negative` | Expected **4 failed / 61 filtered** |
| `review-witness-positive` | **1 passed / 64 filtered**, observed real waiter counts 1→2 |
| `review-witness-negative` | Expected **1 failed / 64 filtered**, request-readiness barrier specifically identified |
| `review-intermediate-failure` | Expected **1 failed / 64 filtered**, both operations drained before disposal |
| `review-boundary-mutant` | Expected **1 failed / 2 passed / 27 filtered** |
| `review-direct-owner-mutant` | Expected **2 failed / 28 filtered** |
| `review-weakmap-leak-mutant` | Expected **4 failed / 26 filtered** |

All test/type runs had **zero guard refusals**, zero import/setup failures, and zero reported unhandled errors. Expected-negative guard self-tests separately recorded their intended refusals. The earlier 27-case held-out Node/Bun runs are retained as v1, not counted as additional unique coverage. RED/GREEN differ only in `src/mlx.ts`, share the exact permanent test postimage, and agree on **114** common observed public dependency hashes.

One evidence-reducer attempt failed because a prefix glob mixed v1/v2 run labels; its real traceback/receipt is retained. Exact-label reduction succeeded on the unchanged raw executions, without rerunning tests or discarding evidence. This was a reviewer-harness reduction issue, not a product failure.

## Limits and next stage

The native Bun selection excludes 20 dynamic/native-LLM-related transport cases. No full native Vitest allocator-SIGTRAP qualification, whole-project canonical suite, package/installed artifact check, combined-live build, or current-head remote CI was attempted. The parent's standalone snapshot build and earlier qualification are supplied provenance only, not my executions. Current live source ownership was checked by bytes for the two owned preimages; unrelated accepted changes were neither imported nor reverted. Auxiliary canonical-workspace recognition is not claimed.

**SPEC PASS permits the subsequent affected-scope QUALITY review only. Parent integration and acceptance gates remain open.**
