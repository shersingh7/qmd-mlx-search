# Parent repair: exact uv release identity

Status: **verified parent-owned candidate; independent reacceptance pending**. No image/container/cloud execution or live integration.

## Original failure

Independent review `deleg_3eb51d4a` found that the qualifier compared the whole uv version banner with `uv 0.11.18`. The parent authenticated all 42 reviewed input files and reproduced the original positive-contract suite: four tests, two legitimate upstream suffix failures, two passing controls, no errors/skips. The actual installed uv also failed the old full qualifier despite reporting `uv 0.11.18 (e32666915 2026-06-01 aarch64-apple-darwin)`.

Original review: `/Users/shersingh/.hermes/cache/scratch/qmd-independent-launcher-review-ua_xj6f_/REPORT.md`. Original failing runs are retained in `evidence/parent-red.{json,xml}` and `evidence/real-uv-red.json`.

## Two-file repair

- `finetune/tools/qualify_gguf_image.py`: a private stdlib-regex parser extracts the exact release token from the bare or parenthesized-metadata banner. The existing guard compares that token exactly with `0.11.18`. It preserves `+commits`, prerelease and development suffixes as different versions, rejects malformed/prefix matches, and preserves the full original banner in the receipt. Python/tool checks, native compilation, command limits and scope labels are unchanged.
- `finetune/tests/test_locked_job_submission.py`: append-only regressions. Three accepted displays exercise the full CLI with real host Git/cc/c++/make and only a synthetic uv version display. Ten wrong/malformed displays must refuse before building. The entire original test file remains a byte-identical prefix.

Launcher, documentation, image pins, overrides, source closures, lock files and all other reviewed inputs are unchanged. Only the owned repair mirror was edited.

## Fresh verification

- Original four positive-contract cases: **4 passed, 0 failures/errors/skips**. `test_probe_contract_green.py` changes only the explicit expected source SHA from the old probe to the repaired probe; `green-test-binding.patch` records this, and the original test remains unchanged.
- Complete launcher suite: **162 passed, 0 failures/errors/skips**, including every original test and the new regressions. See `evidence/candidate-suite.{json,xml}`.
- Full probe with the actual installed uv, no version shim and no `--native-only`: **exit 0**, real host C/C++ compilation and execution returned `qmd-c-ok` and `qmd-cxx-ok`. Actual platform is Darwin/arm64; Python 3.12.13. See `evidence/real-uv-green.json`. This is **not** Linux image qualification.
- All 42 original reviewer inputs remain unchanged. `repair-manifest.json` binds the exact two-file delta and full composed source; `repair.patch` is replayable against the reviewed four-file proposal.

## Remaining gates

Independent review must close R1 and finish the previously deferred quality review of the composed launcher/qualifier proposal. A separate scratch-only worker (`deleg_6bb65ced`) owns the pre-merge image-bound CI workflow proposal; it may not modify this repair. Parent integration, actual Docker execution on native Linux amd64, source/digest/architecture/readback verification, combined build/tests and exact-head CI remain unqualified. No claim is made for Linux arm64, locked converter/training, GPU, HF scheduling or publication.
