# GGUF Image Qualification Implementation Plan

> **For Hermes:** Use subagent-driven-development for independent review; the parent owns integration, publication and CI acceptance.

**Goal:** Give the three GGUF-capable remote closures their required native build tools without making the other runners heavier, and prove the selected image actually runs before release.

**Architecture:** Keep explicit source/lock/helper staging and existing privacy/argv safeguards. Select the digest-pinned non-slim uv image only for closures containing `finetune/gguf_toolchain.py`; keep the existing slim default elsewhere. Separate source review, locked Python-environment installation, native image execution and real converter qualification.

**Tech Stack:** Python 3.12, uv 0.11.18, pytest, OCI/GHCR metadata, Docker and GitHub Actions.

## 1. Authenticate and independently review the proposal

Files: `finetune/tools/submit_locked_job.py`, `finetune/tests/test_locked_job_submission.py`, `finetune/tools/qualify_gguf_image.py`, `finetune/LOCKED-JOBS.md`.

- Frozen proposal: `/Users/shersingh/.hermes/cache/scratch/qmd-launcher-image-mLaqSSB1/delivery/manifest.json`; patch SHA-256 `d858a625c036ed4c3bd826383f76b7b369271a2dcf1a389e8bf443ec3362719b`.
- Parent authenticated four proposed files, twelve registry objects and replayed all **149** staging/launcher tests in its own isolated tree. Native host C/C++ compilation also passed on Darwin/arm64. These are not container or converter acceptance.
- Review spec first, then quality. Preserve the existing launcher bytes outside the declared closure/image changes. Inspect explicit-image semantics and failure cleanup without inventing an expected-source-digest policy that the launcher does not implement.

## 1a. Repair the confirmed qualifier refusal

Independent review `deleg_3eb51d4a` passed the launcher closure/image-selection spec but found P1 in `qualify_gguf_image.py:50`: whole-banner equality rejects upstream uv's required target/build suffix. The parent reproduced the four positive-contract tests (two legitimate suffix cases fail; bare and wrong-release controls behave as expected) and the same refusal with the actual installed uv 0.11.18 release.

1. In an owned mirror, parse the release token without normalizing it or dropping meaningful version suffixes; compare that token exactly with the required release. Parent owns this two-file qualifier/regression delta. Preserve the reported full version banner, Python/tool prerequisites, command bounds and real compilation checks.
2. Append regressions for both upstream suffix formats and bare-release compatibility. Reject wrong releases, prefix collisions and post-release/prerelease tokens; preserve the existing launcher/test-byte prefix and all original negative evidence.
3. Freshly run the original positive-contract suite with only its explicit source-hash binding updated for repaired bytes, the complete launcher suite, and the full probe using actual installed uv and host compilers. These establish host/parser behavior, not image execution.
4. Independently reaccept the repaired source and finish the deferred quality review. A separate scratch-only workflow proposal owns pre-merge triggers and binding to the actual launcher constant; it must not edit the parent-owned qualifier or alter image-selection policy.

## 1b. Close the remaining empty-version diagnostic edge case

Independent reacceptance `deleg_85a0cf7f` closes R1/P1 for the exact four-file proposal: four positive contracts, all 162 launcher cases and 59 independent held-outs pass. Its nonblocking Q1/P3 remains reproducible: a prerequisite that exits successfully with empty version stdout raises an uncaught `IndexError` before any build or success receipt.

1. Parent owns a new isolated two-file follow-up: the qualifier and a separate regression test. Preserve the accepted launcher/docs/existing 162-test bytes and all original RED evidence.
2. First require clean exit-2 refusal, no traceback/success receipt/build and temporary-directory cleanup for empty, newline-only and whitespace-only output from every actually queried prerequisite, in full and native-only modes. Do not query uv in native-only mode.
3. Replace only the unsafe first-line indexing with an explicit empty-output refusal. Preserve the exact uv token parser, successful banner output, Python gates, command bounds, native compilation and receipt scope.
4. Re-run the unchanged original launcher suite, new regressions and retained positive/held-out contracts with only their source-hash bindings updated, plus the real installed-uv host probe. Obtain bounded independent acceptance of this delta; retain the earlier approval where bytes/contracts are unchanged. This remains separate from native Linux image qualification.

## 2. Bind the actual integrated closure

- Integrate the separately accepted repaired GGUF helper, not the obsolete fixture. The launcher stages helper bytes; it does not execute or qualify them.
- Apply parent-owned initial dependency metadata and refresh every affected PEP 723 lock. Re-run all sixteen graph/freshness checks and security gates.
- Re-run `pytest -q finetune/tests/test_locked_job_submission.py` with cleared credentials, isolated HOME/cache/temp roots and plugins disabled. Require every real-source staging case to execute; do not omit the six cases requiring the composed helper.

## 3. Make image CI runnable before merge

Create `.github/workflows/gguf-native-image-qualification.yml` from the reviewed recipe, with parent-owned changes:

- Do **not** leave the new workflow dispatch-only. GitHub documents that `workflow_dispatch` receives events only when the workflow exists on the default branch: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch . Add an appropriate pre-merge `pull_request` or candidate-branch `push` trigger. Do not merge merely to bootstrap the gate.
- Bind the tested image to the launcher's actual `GGUF_IMAGE` value; prevent a hardcoded recipe pin drifting independently.
- Preserve read-only checkout credentials, minimal mounted public probe, empty client credentials, non-root execution, no network inside the container, read-only root, resource limits and owned-container cleanup.
- Validate with `actionlint` and bash syntax checks. Confirm trigger contexts and exact checked-out candidate identity.

## 4. Run and read back native image evidence

Candidate image: `ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie@sha256:c8fe61165767c52118d7d3e67db5d0be5454c2e0da1f9fe15c697ebdd72f304e`.

The parent checked the configured local Unix Docker endpoint; no usable engine responded. No engine was started and no image was pulled. Use ordinary reviewed GitHub CI, not paid HF Jobs or a registry publication.

- Read back the pulled digest and image architecture. Check the engine/runner architecture too before calling execution native.
- Execute the probe with `python -I -B -S /probe.py`. Require exit zero, the expected Python/uv versions and actual `qmd-c-ok` / `qmd-cxx-ok` outputs.
- Retain candidate/probe hashes, architecture, image identity and real job logs. A host-only probe or registry config history is not image execution.
- The initial recipe targets Linux amd64. Claim arm64 image execution only after a separate actual native arm64 run; metadata availability is not qualification.

## 5. Close only the verified scope

Native compiler prerequisites do not qualify the locked converter graph, full exports, model quality, CUDA/GPU, HF scheduling or publication. Keep those source-bound gates separate. After relevant writers settle, run combined canonical tests/build/package checks and exact-candidate CI before merge; verify default-branch dependency-alert outcomes afterward.
