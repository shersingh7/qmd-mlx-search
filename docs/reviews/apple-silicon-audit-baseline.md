# Apple Silicon audit — independently executed baseline

## Scope

- Baseline revision: `1d6cfef` (fresh clone of `https://github.com/shersingh7/qmd-mlx-search.git`).
- Work branch: `hermes/apple-silicon-audit`.
- Hardware verified live: Apple M2 Pro, 10 CPU cores, 16 GPU cores, 32 GiB unified memory; macOS 27.2.1, arm64.
- Code/test/benchmark work only. No production indexes, global QMD install, daemon registration, remote pushes or private corpus evaluation.
- Worker route explicitly selected by David for this task: Hermes delegation using the currently configured `openai-codex` / `gpt-6.1-sol-900k`. This is not an external Codex CLI launch.

## Environment

- Node 22.22.3, Bun 1.3.8, npm 12.0.2.
- Repo-local Python 3.12.13 venv; freshly resolved requirements: MLX 0.32.4, mlx-lm 0.32.2, mlx-embedding-models 0.0.11, Transformers 5.22.2, NumPy 2.5.4, pytest 9.2.0.
- `bun install --frozen-lockfile --ignore-scripts` succeeded. The expected missing native SQLite binding was resolved by approving only `better-sqlite3` and rebuilding it. npm added `allowScripts.better-sqlite3=true` to `package.json`; this is an installation delta, not a product-code fix. SQLite runtime reported 3.51.3.
- Tests run with disposable HOME, XDG cache/config, HERMES_HOME and TMPDIR. All model-download flags are offline, ambient QMD/MLX settings removed. Actual inherited TMPDIR originally pointed to macOS system temp despite the session description; the gate runner overrides it on every subprocess invocation.

## Gate results

| Gate | Actual result |
|---|---|
| `bun run build` | Passed before any source changes |
| `.venv/bin/python -m pytest test/python/ -q` | 246 passed, 13 skipped; exit 0 |
| `node node_modules/vitest/vitest.mjs run --reporter=dot --maxWorkers=2 test/` | 804 passed, 72 skipped, 1 failed; 25 files passed / 1 failed; exit 1 |
| Cached real MiniLM inference through production MLX runtime | Passed numerical single/mixed-batch parity and finite/unit-norm checks |

Baseline TypeScript failure: `test/phase5-recovery.test.ts` → Retrieval Equivalence Verification Helper → `verifyRetrievalEquivalence passes on identical query details and rankings`. Root cause is subject to the audit; no expectation has been weakened.

## Bounded real Metal baseline

The parent ran production `MLXEmbeddingRuntime` with the already cached `sentence-transformers/all-MiniLM-L6-v2` snapshot `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, max length 256, float32, 384-dimensional mean-pooled vectors. No model download or private data was used. The inputs are four authored public technical strings at mixed lengths; they are synthetic workload text, but **inference and measurements are real**.

| Workload | Repetitions | Median | p95 |
|---|---:|---:|---:|
| One medium-length query | 20 | 2.690 ms | 3.275 ms |
| 64 mixed-length documents | 8 | 88.600 ms | 145.903 ms |
| Four mixed-length documents repeated | 200 | 8.456 ms | 9.492 ms |

- Model load: 3.719 seconds.
- Minimum singleton/mixed cosine: 0.99999988; maximum elementwise difference: 1.1921e-7.
- Active Metal memory during the repeated workload: 90,267,668 bytes before and after.
- Metal allocator cache: 380,415,176 bytes before and after; a cache is not automatically a leak.
- After shutdown, Python GC and explicit cache clearing: active/cache Metal memory both zero.
- Peak process RSS is a high-water mark, not current RSS; do not interpret it as retained memory.
- This short MiniLM run does **not** prove long-run leak freedom, 4B performance, concurrent interactive latency, or retrieval quality. It is a valid pre-change comparison point for the same workload/model/runtime.

## Retained local evidence

Parent-owned artifacts under `/Users/shersingh/.hermes/cache/scratch/qmd-audit/`:
- `run-gate.py`: isolated offline command runner and exit-code receipts.
- `baseline-python.{json,log}`.
- `baseline-typescript.{json,log}`.
- `probe-mlx.py`: real Metal baseline/comparison runner.
- `baseline-metal.json`, `baseline-metal-vectors.npz`, `baseline-metal-run.{json,log}`.

Final delivery must copy necessary reproducibility artifacts into the repository or a durable deliverable; scratch evidence alone is temporary.
