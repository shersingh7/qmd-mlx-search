# Independent frozen launcher/image review

## Verdict — SPEC first

**SPEC: FAIL for the delivery as a whole.** The launcher closure/image delta passes the scoped contract; the new image qualifier has a reproducible release-version refusal. **QUALITY: deferred** because this important prerequisite contract fails. This is not image acceptance and does not close the parent's gate.

### R1 — P1: correct upstream uv 0.11.18 is rejected by the qualifier

- Target: `finetune/tools/qualify_gguf_image.py:45-51`, specifically the whole-line equality at line 50. The workflow calls this failing path at `qualification-workflow.yml.proposed:58` (without `--native-only`).
- The required release's upstream `uv --version` includes a parenthesized target triple, with optional commit/date. The immutable upstream revision recorded in the reviewed registry configs is `e32666915e7fd99bbd0f167f5c381f4fb4198486`. At that exact revision, `crates/uv-cli/src/lib.rs:109,160-162` binds `--version` to `SelfVersionInfo`; `crates/uv-cli/src/version.rs:66-85` **always** appends either ` (commit date target)` or ` (target)`. Its tests at lines 152-176 confirm both formats. The release Dockerfile builds that CLI; the derived Python image copies it (`.github/workflows/build-docker.yml:274-280`). No image execution is being inferred from those sources.
- Fresh positive-contract tests: **4 tests, 2 failed, 2 passed, 0 errors/skips; pytest exit 1**. With real host Git/cc/c++/make and only `uv --version` replaced by a display fixture:
  - `uv 0.11.18 (x86_64-unknown-linux-musl)` → probe **exit 2**, `image probe requires uv 0.11.18` (contract expects 0).
  - `uv 0.11.18 (e32666915 2026-06-01 x86_64-unknown-linux-musl)` → same **exit 2**.
  - Bare `uv 0.11.18` control → **exit 0**, actually compiled and ran `qmd-c-ok`/`qmd-cxx-ok` through make.
  - `uv 0.11.17 (x86_64-unknown-linux-musl)` control → **exit 2**, correctly refused.
- Impact: the default proposed CI invocation cannot qualify the intended official release; the existing missing-tool/wrong-version tests do not cover the actual accepted version's display format. The host `--native-only` success bypasses this check and therefore cannot detect it.
- Required repair direction: compare the exact parsed release identity, not the entire human-readable version banner; retain wrong-release refusal and add both correct upstream suffix formats as positive regressions. No fix was applied here.
- Receipts: `evidence/probe-positive-contract.{json,xml}`, `evidence/contract-*.json`, `evidence/uv-version-counterexample.json`. Pinned public source and retrieval hashes: `public-source/` and `evidence/public-source-fetch.json`.

## Passing launcher SPEC scope

- Actual callers inspected as text/AST, never imported/executed: direct converter import/export (`convert_gguf.py:66,126,143`), deferred shared exporter (`train.py:72-88,188-189`), native-to-shared exporter (`train_unsloth.py:73-85,159-161,210-215`), training eval subprocess (`train.py:114-126`), eval reward import and query path (`eval.py:56-62`). The helper's imports add no further local project module.
- Exactly the three GGUF-capable closures add the helper and select the non-slim default. All twelve other closures retain their slim image. Script arguments, including `--no-gguf` and `--image` after the separator, do not affect selection.
- Explicit image values, including empty/malformed and leading-dash equals-form values, remain literal. Mocked submission sends and reads back those same values; **there is no new image validation/refusal policy**.
- Unsafe/missing helper files refuse before SDK access, including real permission-denied, directory, FIFO, symlink, hardlink, and symlink-ancestor held-outs. Regular content changes are allowed and produce a different preview digest; old prepared bytes stage immutably. No nonexistent expected-source-digest requirement was invented.
- Accepted secret/argv/input/privacy/single-run/readback/locked-command source bytes are identical outside the precise closure/image delta. Original test bytes remain an identical prefix. Fresh reverse/check/apply/check replay reproduces all four manifest hashes.

## Fresh execution, separated by suite

| Surface | Actual result |
|---|---|
| Frozen proposed focused pytest suite, own exact-source fixture | **149 passed**, 0 failures/errors/skips; exit 0; pytest 2.56 s |
| Independent held-outs | **85 passed**, 0 failed; includes 24 isolated `-I -B -S` preview audit processes, all 15 defaults, unsafe-helper/snapshot cases, synthetic SDK privacy/readback/error controls |
| Positive qualifier release contract | **2 passed / 2 failed**, 0 errors/skips; exit 1; R1 |
| Mutation controls | **7 independently detected**, each changed only its named source path and failed at its named assertion; case counts recorded individually, not added to acceptance totals |
| Real native-only host probe | exit 0, Darwin/arm64, Python 3.12.13, Apple clang 21.0.0, GNU Make 3.81; compiled/runs both CPU binaries |
| Recipe | bash `-n` exit 0; **not executed**; actionlint absent and no installation authorized |
| Image / cloud | **not executed**; no pulls, jobs, uploads, services or GPU/model materialization |

Each test invocation uses an environment allowlist with owned HOME/Hermes/XDG/HF/cache/TMP roots, offline flags, CI, no credentials/live flags/PYTHONPATH, and disabled pytest plugin autoload/cache. Only launcher/probe source is executed; GGUF helper, converters and trainers remain staging data. Preview audit hooks observe no child/network/ML/SDK import or named-secret read; explicit source-read sets exclude implicit private inputs. Synthetic SDK success is not cloud success.

Mutation controls: omit each GGUF closure → 1/3 failures each; changed helper export → 3/3; all-slim → 3/3; empty-override fallback → 5/15; bypassed image readback → 3/3. Every run has zero errors/skips and an isolated failure-site assertion in `evidence/mutations-v2.json`.

## Source binding and preservation

| Artifact | SHA-256 |
|---|---|
| Four-file proposal patch | `d858a625c036ed4c3bd826383f76b7b369271a2dcf1a389e8bf443ec3362719b` |
| Proposed launcher | `f61076c5a71e5b251db42aadb1bbe15fb1689a2d3293f1085edcdefc1958a83b` |
| Proposed tests | `3bd023524ce03e38e55d4055862b27e90249642e132ecaa98f9c5ae3564db1e1` |
| Proposed docs | `be69f428d973a52ea94c64a9dd196b02a7c978a7e8318f7d85c70b12e4caf8df` |
| Proposed qualifier | `6717b3770708789be996da14ef3ce614f81f15c1ce0713ed03aa457201729ecf` |
| Staging-only helper, not launcher patch | `5c36cc3f73ec03cd7c89f7f8f7083987824f4b8bae19ee720a02177fa03ea028` |
| Proposed workflow | `2c145115e53c2948249ee36d2593c7e9419bdaacbd7fe8555f2da01f323cb23e` |

The full image remains `ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie@sha256:c8fe61165767c52118d7d3e67db5d0be5454c2e0da1f9fe15c697ebdd72f304e`; slim remains `ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie-slim@sha256:9f9a7c30d604c8297ac14321b91b42f8516ff207f42b76b1bf29974fad523f6e`.

Before/after hash verification: all **42 own fixture inputs**, **42 frozen source inputs**, **6 frozen delivery files**, and **3 original live launcher/tests/docs** are unchanged. Eighteen policy/metadata/lock files and sixteen PEP header blocks match the main mirror. Twelve retained registry objects were independently body-hashed/sized and checked against index/manifest/config parent descriptors; this is retained metadata verification, not a fresh registry pull or image test. Details: `intake.json`, `evidence/proof.json`, `evidence/final-verification.json`.

## Scope and remaining gates

- Quality review is intentionally deferred, not silently passed for the new qualifier or whole delivery.
- Parent's already-known CI integration obligations remain: premerge-capable trigger (new manual workflow requires default-branch presence), bind the tested image to actual `GGUF_IMAGE` rather than driftable hardcoding, actionlint, exact head/source/digest/engine/architecture readback and a real image run. The optional recipe is not an installed workflow. Its container resource/network/readonly/nonroot boundaries and CID cleanup are static text only; host-only probe execution does not establish those boundaries or process-tree cleanup.
- Actual Linux amd64 image execution, arm64 execution, helper correctness, initial Python metadata/all graphs, locked converter/training, GPU/CUDA, HF scheduling/private storage and publication remain separate and unqualified.
- No shared/live/frozen source or delivery artifact was changed. All created fixtures, evidence and review code are below this report's owned directory.
- Reviewer harness issues are retained, not disguised as product failures: the first pytest run lacked the parent basetemp directory (148 setup errors), fixed before the fresh `frozen-suite-final` run; the first mutation driver caught a no-op tuple-removal mutation before counting it, corrected and rerun under separate `mutations-v2` paths. Earlier receipts remain in `evidence/` and are excluded from acceptance counts.
