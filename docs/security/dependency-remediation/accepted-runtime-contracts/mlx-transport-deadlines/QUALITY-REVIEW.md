# Independent QUALITY review — revised MLX transport proposal

**Verdict: PASS within the complete affected scope of the exact reviewed two-file delta. No blocking or nonblocking source-quality finding.** This review began only after the recorded SPEC PASS; see `evidence/review-stage-order.json`. This is not integration, installed-package, remote-CI, merge, or release acceptance.

## Binding and reviewed surface

The full proposal SHA-256 is `a2e31a613318e78801203bb068b9f79e2e5a54faf7fb6b051eea5c1d37b48640`.

- `src/mlx.ts`: `2cd9aa0d12b0405b4b7147d2a2d7c12ccd7f9b0448b351c1b92115957a845751`.
- `test/audit-transport-review.test.ts`: `b1489d0735ce1dccc351eca930110d6dc0235ea82d595a9252c8394489b0947d`.

Reviewed the entire production +22/-11 and test +435/-40 delta, not just the parent's three corrected fixture spans. The review includes `RequestScope`/`withScope`, explicit abort races, request/body cleanup, metadata fallback, descriptor validation, JSON/binary serving identity and projection validation, all four public client entrypoints, concurrent batching, and affected `LlamaCpp` initialization/embedding/deadline/lifecycle composition. The public source closure remains 298 files; the separate reviewer test and mutants are explicitly nondeliverable probes.

## Assessment

| Concern | Assessment |
|---|---|
| Deadline/error precedence | `embeddingConfig` now checks the actual outer monotonic deadline before treating null metadata as a strict protocol failure. This is a narrow classification correction, not an expanded catch or a blanket protocol-to-null conversion. Caller/lifetime cancellation still takes precedence through the existing failure path. Live-budget malformed descriptor and serving-identity failures still reject. |
| Publication ownership | A Set of waiter objects is the appropriate representation when publication depends on each consumer's deadline and cancellation state. Distinct scope identities prevent same-external-signal calls from borrowing one another's budgets. `Infinity` retains the established direct-metadata semantics without removing the shared metadata request's own timeout. |
| Stale-completion safety | Publication checks the lifetime and negotiation controller before cache mutation. Last-waiter departure clears/cancels only the matching negotiation. The identity-qualified finally cannot clear a replacement. Held-out overlapping old/new completion and disposal controls passed, including before fixture cleanup. |
| Cache/geometry invariants | Validated descriptor snapshots remain immutable and pinned. Projection remains per-request and cannot replace native negotiation geometry. Requested dimensions above native, malformed metadata, negotiation mismatch, serving revision drift, and binary/JSON response mismatch remain strict. Existing cached getters after disposal are not new state publication; active APIs still reject before taking the cache fast path. |
| API compatibility | No exported signature, return cardinality, HTTP path/payload/header, backend/model selection, or configuration root changed. The private `embeddingConfig` parameter change is closed over its actual caller. All four public timeout/null-entry shapes and empty-input behavior remain inherited from the existing wrappers. |
| Cancellation and listeners | The patch adds no listener or timer. Existing `abortable` removes per-waiter listeners and keeps rejection handlers on underlying promises. Individual waiter deletion occurs in finally, and only the last waiter aborts shared work. Disposal remains terminal and late bodies remain best-effort cancelled without awaiting noncooperative cancel handlers. |
| Memory/lifetime bounds | The WeakMap uses the unique per-operation scope signal and deletes it on all descriptor completion paths. The waiter Set contains only active descriptor waits and deletes those exact identities in finally. Held-outs assert absence after success, timeout, cache hit, invalid metadata, and early rejection. There is no new persistent history or per-exchange timer. |
| Iteration cost | Publication scans active waiters once per completed negotiation. `[...waiters].some(...)` has O(active waiters) transient allocation and scanning, not per-vector/per-batch growth. Replacing this small one-time conversion with manual iteration is not a correctness or proportional release gate. No speculative micro-optimization is requested. |
| Maintainability | The private WeakMap bridges a public signal-only metadata API to internal absolute scope ownership without changing that public API. Comments explain both the delayed-timer publication rule and the timeout-vs-protocol rule. Set identity cleanup and negotiation identity checks remain locally readable. No unrelated refactor is mixed in. |

## Permanent test delta

- The real-clock barrier now checks elapsed monotonic time even if an overdue timer loses the race to synchronous work. For the changed cold/overrun fixtures, the API timestamp is immediately before invocation, after setup. The overrun stall begins inside the response adapter after scope creation.
- The real 80/100 ms limits and 120 ms combined cold work are preserved. The renewed-budget mutant still fails all four methods. These real-time cases remain scheduler-sensitive by nature; one-worker bounded review passes do not certify the parent's later canonical CI scheduling.
- Readiness for the shared-request fixture is now actual descriptor dispatch with a held response. The renamed entry counter is not presented as independent admission proof. The reviewed source path and additional observations establish that the younger call joins the existing negotiation.
- Changed asynchronous fixtures retain started operations, release held gates/responses in finally, and drain before teardown disposal. The controlled intermediate assertion failure reached two real waiters and confirmed both operations fulfilled before teardown disposal, no remaining negotiation, and zero request listeners. Expected-negative errors remain recorded as failed controls rather than being hidden or relabelled.
- Valid metadata/serving positives accompany strict-negative and cancellation cases across all four methods. Disposal-as-action remains visibly different from routine teardown. Fake-time ownership tests supplement, rather than replace, the unchanged real deadlines.
- The additions are confined to public client APIs and explicit synthetic fetch fixtures. Reflection in the independent probe is observation/testing, not a new production interface. No fabricated daemon, model, or CI result is used as execution evidence.

## Execution grounding

The completed SPEC stage independently executed:

- **87 permanent tests passed** on actual Node 22.22.3 (65 transport + 22 real owner-composition tests).
- **45 passed / 20 filtered** on actual Bun 1.3.8; both launcher and fork worker receipts identify Bun, independently of its Node-compatibility version.
- **30 held-out tests passed per host**, Node and native Bun, covering absolute-boundary ownership, both ordering directions of metadata/embedding mixtures, shared external signals, cache/early-rejection cleanup, overlapping stale/replacement negotiation, and strict identity/geometry refusal.
- Scoped strict Node and Bun checks both exited 0 with **381 declaration reads per host**. These checks use explicit relative typeRoots in the probe configuration, not altered production compiler roots.
- Exact-old source, budget-renewal, invalid admission, cleanup-failure, equality-boundary, direct-owner, and WeakMap-cleanup controls failed at their intended assertions. All counts are parsed from actual selected cases; import/setup failures were not counted as targeted failures.

All test/type execution guards recorded zero refusals. Active negative controls separately proved compiler/runtime read refusal, dependency-write refusal, and network refusal on both hosts. Current public dependency targets were re-resolved and actual observed file reads hash-bound. RED/GREEN agree on 114 common observed dependency hashes and share the exact permanent test bytes. No new source/test execution was necessary for QUALITY beyond the source-bound qualification already completed; this stage adds the full affected-scope lifecycle, compatibility, cost, and maintainability assessment.

## Limitations and authority

- Guards are in-process instrumentation, not OS containment or complete native Bun loader mediation.
- Native Bun's 20 filtered permanent cases, full native Vitest allocator behavior, canonical full suite, combined live build, installed package, current-head CI, and merge remain parent gates.
- The parent's successful standalone snapshot build is authenticated borrowed evidence, not my build execution or combined-live acceptance.
- No unrelated live workflow/recovery changes were imported or reverted. Auxiliary canonical-workspace recognition is not asserted.
- One reviewer reducer prefix-glob issue is retained with its real failed receipt; corrected exact-label parsing succeeded without rerunning tests. It changes no substantive source verdict.

**The exact two-file candidate passes SPEC followed by affected-scope QUALITY. No fix or acceptance transition was applied.**
