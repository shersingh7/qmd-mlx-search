# Independent review replay notes

Frozen delivery: `/Users/shersingh/.hermes/cache/scratch/qmd-ci-observer-mTSwsD` (read-only).
Owned review root: `/Users/shersingh/.hermes/cache/scratch/qmd-observer-independent-g4hwpj_2`.

`harness/run.mjs` creates a fresh explicit environment for every label and validates all 246 frozen proposed source hashes before executing a workload. No ambient NODE_OPTIONS/PYTHONPATH/credentials are inherited. The default node_modules directories in both own trees are real containers with per-package read-use aliases and owned `.vite`, `.cache`, `.vitest`; never replace either root container with a shared symlink. Vite, Vitest, Bun, npm, HOME, Hermes, HF, XDG and temporary paths belong to this root.

For another bounded run, choose a **new label** so append-only evidence is not overwritten. Example (outer orchestration is also cleared; the runner constructs the complete child allowlist):

```sh
ROOT=/Users/shersingh/.hermes/cache/scratch/qmd-observer-independent-g4hwpj_2
/usr/bin/env -i HOME="$ROOT/home" TMPDIR="$ROOT/tmp" \
 PATH=/Users/shersingh/.hermes/node/bin:/Users/shersingh/.bun/bin:/usr/bin:/bin:/usr/sbin:/sbin \
 CI=1 REQUIRE_BUN=1 QMD_LLAMA_GPU=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
 /Users/shersingh/.hermes/node/bin/node "$ROOT/harness/run.mjs" heldouts parent-review-unique-label
```

Runnable bounded modes with independent labels:

- `fast-on` / `fast-off`: exact changed canonical fast test only, at most two Vitest workers, original assertions/limits. Seven other declarations are CLI-filtered, not executed coverage.
- `selftests`: direct Node `--test --test-concurrency=1` helper controls.
- `heldouts`: direct Node independent active-error, exact-restoration, exception, write-failure, disabled, constructor and ref-refusal controls (6 test declarations).
- `bun-fast-on`: actual installed Bun fast-case import/no-op control; no native observer implementation.
- `actual-fallback`: separate AFTER-first real-helper negative transport probe with bounded cooperative writer release; not cold CI or successful-cleanup proof.
- `helper-types`, `syntax-helper`, `syntax-selftest`, `actionlint`, `metadata`: exact commands are in the corresponding command receipts; lint uses the supplied read-only official binary and disables unavailable shellcheck.

`static.mjs`, `bounded-types.mjs`, and `spec-readback.mjs` are retained full runnable verification logic. Their output paths are intentionally exclusive/fixed to preserve this review's evidence. For repeated aggregate generation, use a fresh OWN derivative with the same source bytes/per-package aliases/caches and copied harness+provided inputs, or change ONLY your own output destinations. Do not remove or overwrite this review's retained failures/evidence. `bounded-types.mjs` never executes docs or uses whole-snapshot JS entries: it selects the identical 89 TS entrypoints, excludes 11 JS entrypoints on both sides and guards compiler source reads. It is not the canonical root type gate.

Initial reviewer oracle/guard errors and final corrected fresh runs are retained with distinct labels. No source fix, stage, CI/gh, network/install, GPU/model/private-knowledge, global/prototype patches or settings/services/skills writes are part of replay. Do not reinterpret fresh HOME as cold OS or demand global-first scheduling when other canonical files can run concurrently. Parent alone owns full integration and exact-candidate CI readback.
