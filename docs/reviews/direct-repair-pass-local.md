# Direct repair pass: local recovery and verification fixes

## Status and scope

The initial focused pass was implemented directly in Hermes after the user asked to conserve usage and skip a separate review agent. No new subagents or external coding harness were launched. At that checkpoint the changes were local only. The user subsequently authorized full verification followed by commit/push if green, initially without merging; that later verification is recorded below. The latest instruction also authorizes merging after exact-head CI passes.

This is a **verified focused repair**, not full-release acceptance or a claim that every possible leak is eliminated. No live knowledge/index, model daemon, global installation or personal corpus was changed.

## Repairs

1. **Recovery ownership:** integrated the bounded gated-child discovery repair. A kernel `WNOHANG=0` is no longer confused with `ECHILD` when a discovery snapshot misses a live child. Ownership/exit acquisition retries within the original teardown allowance; acquiring the identity later does not restart that clock.
2. **Failure precedence:** admit an observed drain rejection before timer finalization. A later cleanup failure cannot replace the earlier primary, including `undefined`, `null`, `false`, zero, an empty string or frozen objects. Shared teardown retains its separate aggregate-error semantics.
3. **Test-selection accuracy:** the audit harness records an independent public `beforeEach` execution-phase witness. A callback that runs and then skips can no longer disappear behind its final state. Every final identity is reconciled with the complete collected registry, using linear-time set/map lookups.
4. **Filesystem-policy correctness:** the Python qualification guard binds fd-relative cleanup to the actual directory descriptor and exact trusted stdlib caller. It validates kernel path/descriptor identity and applies the normal source/write policy; an arbitrary nonnegative descriptor no longer bypasses protection. This is macOS test instrumentation, not an OS sandbox.
5. **Integration typing:** corrected inherited fixture types without changing emitted JavaScript or weakening assertions. Preserved the earlier source versions and verified emitted-JavaScript equality for both changed test files.

The initial focused-pass application/test delta covers 16 paths, with one production runtime file (`scripts/phase5_recovery_process.py`); the remaining paths are regression fixtures, lifecycle helpers and diagnostic integration. Audit-harness fixes are delivered separately in the repair bundle rather than installed as application runtime code.

## Actual verification

| Scope | Node | Native Bun |
|---|---:|---:|
| Lifecycle / error precedence | 266 passed | 266 passed |
| Transition-wrapper contracts | 41 passed | 41 passed |
| Sentinel cleanup / restoration | 48 passed | 48 passed |
| Failure collector / readiness diagnostics | 40 passed | 40 passed |
| Gated-child public API regression | 1 passed | 1 passed |
| **Targeted test executions per runtime** | **396 passed** | **396 passed** |

These are 396 distinct TypeScript tests exercised on both hosts, not twice that many distinct contracts. Source-only type adjustments after those executions have identical emitted JavaScript; their exact before/after bindings are retained.

- Original Python selection: **24/24 methods passed**, no skips or unrun methods. Seven actual supervisors were observed; current direct-owner waits, streams, pipes, descriptor baseline and diagnostic release checks passed. This closes the earlier one-pass/23-unrun guard blockage for this execution.
- Reporter controls: six contracts passed on each runtime. Real Vitest positive selection passed; an intentionally executed unselected-then-skipped callback correctly made qualification fail. Framework auxiliaries were awaited after public shutdown.
- Filesystem guard controls: seven passed, including genuine owned cleanup and refusal of outside/protected directory mutations.
- Canonical package build passed. Default-root strict TypeScript checking passed on both Node and Bun with zero reported diagnostics or guard read refusals.
- Built CLI smoke passed: help, creation/indexing of two synthetic documents, and matching Node/Bun BM25 search results. Only an isolated disposable index was used; embeddings/models were not loaded.
- `git diff --check` passed.

The new regression was genuinely RED before the repair: its seven injected-finalizer cases failed while seven ordinary-disposal neighbors passed. Original negatives and instrumentation failures remain retained rather than rewritten as successes.

## Initial focused-pass qualifications and retained evidence limits

- Transition-wrapper fork tests use real forks with synthetic hold instrumentation. They are not an OS-pause timing qualification.
- Sentinel-failure tests deliberately exercise independent fixture rescue. Such rescue is not relabelled natural product cleanup.
- One Bun guard-end snapshot was absent because the native test host did not emit the assumed exit hook. Its earlier zero-subject/vacuous-closure aggregate is explicitly superseded: actual admission records show 31 sentinels, and the test assertions remain scoped to their own lifecycle observations. This missing aggregate is not fabricated retroactively.
- Early harness attempts that blocked Bun's approved compatibility call-through, omitted required public dependency reads, or intercepted a framework self-SIGTERM are retained as instrumentation failures. Corrected runs use separate labels.
- The full canonical test suite, exact-head CI matrix, all recovery timing/custody cases, and MLX throughput/retrieval benchmarks were **not rerun in this pass**. Historical CI causes and unrestricted real-time cleanup remain unqualified. No new GPU-performance claim is made.
- A separate independent review was intentionally omitted at the user's request. Local verification is not represented as independent approval.

## Reproduction and artifacts

The downloadable repair bundle contains the final application patch, changed source files, audit-harness corrections, regression controls, source manifests, concise receipts and retained raw test outputs. The broader historical checkpoint is separate and unchanged. Apply a patch only after checking its baseline and running `git apply --check`; the current local checkout already contains these changes.


## Subsequent full local verification and publication authorization

The final candidate retains the original two-worker parallel configuration and unchanged 200 ms execution / existing teardown limits. Experimental serial scheduling and additional transition instrumentation were not sufficient to fix bootstrap variability; they were preserved as diagnostic evidence and removed from the candidate.

### Interpreter binding repair

The concrete bottleneck was reproduced with fresh disposable homes: Apple's Command Line Tools Python redirects standard-library bytecode into `HOME/Library/Caches/com.apple.python`. The legacy interpreter created 40 cache files per characterized call. Across 12 no-work, no-admission samples, readiness ranged from 102.6–526.3 ms (median 176.5 ms). Disabling writes alone did not remove all over-budget samples. The project's already installed Python 3.12 completed the equivalent 12 fresh-home samples in 26.5–39.5 ms (median 27.1 ms). These are local bootstrap/runtime comparisons, **not MLX inference benchmarks or universal speedup claims**.

`executeRecoveryWorker` now accepts a call-local `supervisorPython` or the `QMD_RECOVERY_PYTHON` environment setting. Both must identify an explicit absolute executable; empty, relative, missing, directory and non-executable configurations fail before acquisition, without silent fallback or a probe process. The unconfigured legacy default remains `/usr/bin/python3`, and its original `-S -u` arguments remain unchanged. Injected worker argv is never rewritten.

The existing offline runner binds the interpreter it was actually launched with (`sys.executable`):

```sh
python scripts/run-offline.py npm test
```

Thus a Python 3.12 CI/tool environment is no longer silently replaced by Apple's unrelated system interpreter. Nested qualification hosts and failure/ownership observers retain that explicit selection; requested identity remains distinct from observed runtime identity. Regression tests cover fallback preservation, override precedence, fail-closed validation, environment scrubbing and diagnostic literal preservation.

### Final executed gates

All 13 local gate commands completed with exit 0 on the settled candidate:

| Gate | Observed result |
|---|---|
| Complete JavaScript suite, direct Node | 1,959 passed; 66 skipped; 61 files |
| Complete JavaScript suite, Bun package launcher | 1,959 passed; 66 skipped; 61 files |
| Model-free Python suite | 1,511 passed; 2 skipped; 13 real-model cases deselected; 11 subtests passed |
| Dependency-security suite | 457 passed; 43 subtests passed |
| Canonical source build / strict TypeScript | Both passed |
| Real launcher controls | 11 passed |
| Isolated packaged install, Node and Bun | Both passed; identical tarball bytes |
| Bootstrap-observer selftests | 8 passed |
| Real Node/Bun supplementary startup observations | Both completed and retained separately from canonical acceptance |
| Startup-observation selftests, with actual-host evidence | 15 passed; no skips |

The Bun package script explicitly invokes Node/Vitest; it is **not** 1,959 tests running in Bun's native test runner. Native Bun ownership, transition, GC and descriptor-reuse cases run as real subprocess tests inside those suites, and native Bun package/SDK/SQLite/MCP-stdio behavior was exercised separately. Native Bun HTTP is deliberately rejected with an explicit Node.js remedy; that refusal test is not HTTP feature acceptance.

Both canonical JavaScript invocations produced a real Node supervisor observation for the selected interpreter, with admission, supervisor/child reap, closed pipes, and no forced pipe closure in the existing fast-timeout case. Native Bun's missing low-level observer remains explicitly labelled unsupported rather than fabricated.

The retained 262-advisory baseline reports zero affected instances in the candidate. This does not close default-branch alerts before merge, and remote exact-head CI is a separate post-push observation. Independent agent review and new GPU/retrieval benchmarks were not performed in this pass. The latest user authorization covers commit, push and merge after exact-head CI passes; deployment, live-index operations and global installation remain outside scope.
