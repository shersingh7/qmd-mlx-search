# Independent GGUF reacceptance and whole-proposal review

## Decision

**SPEC: PASS. QUALITY: PASS for the whole composed seven-file cold-export code proposal.** No blocking or actionable nonblocking code-review findings remain. This is not live-tree, dependency-graph, remote-image, model-quality, CI, or release acceptance.

Reviewed the original six-file proposal plus final repair delta, in that order:

- Original patch SHA-256: `a2cee04e4f1d482b6080311c1233aa99b182594eaa3b38d771d545871e0ccbec`.
- Repair patch SHA-256: `6952897a0338fe1d1081f2335b893632143549b1fa37ee958beec54c281d2430`.
- Reconstructed both patches from the recorded public Git baseline into this owned scratch directory. Both `git apply --check --whitespace=error-all` and actual application succeeded; all seven resulting files match the final frozen proposal exactly.

### Composed source SHA-256

| Path | SHA-256 |
|---|---|
| `finetune/train.py` | `3e8d06816928626f733992df2f650aea493180f935f8daa63601c9ade302387e` |
| `finetune/train_unsloth.py` | `257f517da20d4f6a745e51dc3d1026f821ef369db136ef4beedb9444a104a1f5` |
| `finetune/convert_gguf.py` | `40b19f2e7be09993bb0f9f22e2ef069b5eeb89f2a6dca7cbf415e57555bf87ab` |
| `finetune/gguf_toolchain.py` | `5c36cc3f73ec03cd7c89f7f8f7083987824f4b8bae19ee720a02177fa03ea028` |
| `finetune/tests/test_finetune_correctness.py` | `2e023eb88997dc63a54b4c6e7eb40bb1a19023415dc900e306dbff0d735b88f3` |
| `finetune/tests/test_gguf_toolchain.py` | `9d68ac180502a2b6b54b78606e7640333e6b30c6a9f4d32518aa88d626513941` |
| `finetune/tests/test_gguf_structure.py` | `a39e8d9e442ff9426731e61c7cc8b3ba4f939eb0461fc1598e6f16b6ee992617` |

## SPEC evidence

| Fresh run | Passed | Failed / errors / skipped | Deliberately deselected |
|---|---:|---:|---:|
| `portable-installed/pytest.xml` | 183 | 0 / 0 / 0 | 6 numerical PEFT cases |
| `portable-pinned/pytest.xml` | 183 | 0 / 0 / 0 | Same 6 cases |
| `counterexamples/pytest.xml` | 10 | 0 / 0 / 0 | 0 |
| `independent-v2-installed/pytest.xml` | 119 | 0 / 0 / 0 | 0 |
| `independent-v2-pinned/pytest.xml` | 119 | 0 / 0 / 0 | 0 |

**614 passing case executions**, not 614 distinct tests. Counts and declared JUnit totals were compared programmatically. `evidence/SPEC.json` records the passing local SPEC gate before the deferred QUALITY decision. Original counterexample function ASTs are identical; only the target path was relocated.

Reader identities, compiled from exact source bytes rather than old `.pyc`:

- Installed **GGUF 0.19.0** reader: `/Users/shersingh/.hermes/cache/scratch/qmd-frozen-project-install-nszwouqz/environment/lib/python3.12/site-packages/gguf/gguf_reader.py`, SHA-256 `91cb9e81f1a67f5856f012cf3f526ad9ffdb6b887d2aa66e2593ee6f63764aad`.
- Exact upstream reader at Git `5193ef1d4f7c95b006ab3b9f857ed27be9a63235`: `qmd-gguf-toolchain-qtll2q_e/upstream/gguf-py/gguf/gguf_reader.py`, SHA-256 `d0ea743200e19d7ef0a4c969edcc4a6dba8e3656a790b217e3bdc7b31170718e`. Installed distribution metadata does not identify this different source tree.
- Independently matched five upstream source blobs against that exact Git object, including base/Qwen filters and reader/writer.

### Former blockers

1. **MTP namespace loss repaired** (`gguf_toolchain.py:156–184`). The classifier applies the actual pinned base normalization before checking MTP prefixes and never rewrites saved identifiers. Fresh original counterexamples invoke the actual pinned filter. Independent heldouts add eight namespace forms, both Qwen3.5 architectures, first/last positions among four genuinely saved shards, unchanged-key/hash checks, empty/non-MTP controls, and plain-Qwen3 controls. The maintained 183-test suite additionally exercises every three-shard position.
2. **Truncated outputs refused before publication** (`gguf_toolchain.py:117–153, 187–222`). Real reader parsing and explicit tensor byte bounds replace the magic-only acceptance. Fresh process-boundary tests exercise shared exporter, actual native main, and standalone main with zero-status truncations and nonzero-status structurally valid outputs. Previous complete sets survive; GGUF uploads, evaluation, success logging, and publication do not proceed. Positive outputs come from actual maintained writers or retained real converter artifacts, never repaired negative fixtures or validator stubs.

### Independent resource/lifetime checks

For **each reader**, nine genuine writer-output truncation locations × 17 repetitions produced **153 actual mappings, all closed**, with GC disabled and all 153 outward exceptions retained. FD counts remained **12 → 12**. Constructor/header, metadata, tensor-info, and final tensor-byte failures are included. Outward exceptions have neither parse cause nor context; helper traceback locals contain no reader/ndarray views, and helper-local diagnostic rendering succeeds. The preserved portable suite separately repeats valid and malformed reader paths 24 times each.

Nearby positive controls include 24 real writer combinations spanning four tensor encodings, two endiannesses, three alignments, and two tensors. These zero-block structures are explicitly not usable models.

All **14 retained real old-proposal outputs** pass the new helper and scalar tensor/type/bounds checks under both readers. The seven stock-tokenizer outputs retain 25 tensors, qwen35 architecture, and 248077 vocabulary entries. Independently verified the stock manifest (`d909c6414b617b37aa28000fbb80650702c55a6948c5bdabb1e1d5284ec0c813`), all 93 listed evidence files, seven artifact hashes, and 22 recorded zero-status real commands including two old fresh builds. This authenticates earlier conversion positives, **not fresh changed-helper native execution or MTP-bearing model quality**.

## Deferred whole-proposal QUALITY review

Completed after the local SPEC gate. Reviewed all seven files, the complete 222-line helper, original body patch, repair delta, caller publication contracts, and metadata/closure intent—not just the two repaired functions.

- **Prerequisite and dependency policy** (`gguf_toolchain.py:22–43`, `train.py:175–189`, `train_unsloth.py:141–162`, `convert_gguf.py:122–144`): initial locked packages/native tools, no runtime installer, checks precede expensive model/data loading, dry-run bypass and native `--no-gguf` preserved. Standalone full preparation precedes loads; skip-quantize avoids native build.
- **Source/tool ownership** (`gguf_toolchain.py:65–114`): fresh non-reusable owned work directory, exact public URL/revision, checked init/fetch/checkout/HEAD, disabled Git credential/config/hooks overrides, matching pinned converter package, CPU-only configuration and `-j 2`; no shared checkout or binary. All command status/launch failures remain authoritative.
- **Conversion and publication** (`gguf_toolchain.py:46–54,187–222`, `train.py:72–93`, `train_unsloth.py:73–115,204–216`, `convert_gguf.py:143–195`): correct Q4_K_M/Q6_K/Q8_0 set, F16/skip behavior, complete-before-publication flow, owned temporary cleanup, refusal/exception propagation, and no-upload semantics preserved. Existing adapter/tokenizer uploads before native GGUF export deliberately remain; no all-Hub transaction claim.
- **Resource implementation**: `__new__` retains the actual maintained reader even when its constructor fails; the actual mmap closes explicitly in `finally`; outward diagnostics are raised after the parse handler without retaining a closed-view parse traceback. No production GC or tensor-wide copying introduced.
- **Regression integrity**: exact AST checks confirm the other five helper definitions unchanged by repair and caller changes confined to the intended cold-export definitions relative to baseline. The unrelated PEFT numerical test body remains identical. Typed writer fixtures strengthen successful fake artifacts; failure/partial/stale assertions remain. The legacy adapter only replaces its exact old success marker after checked zero status, never validation or negative fixtures. Its old source remains `b24de803e2a3a48dbed4f19dfa9daebccc52e395467ff201c0512331321249cf`; previously recorded 32 routing passes / 15 numerical deselections were authenticated, **not freshly rerun or relabelled as 47 passes**.

No additional actionable defects found within this scoped proposal. Structural validity is not semantic model completeness, correct quantization numerics, or inference quality. Complete validation before publication does not promise group-atomic rollback for a later filesystem rename or Hub-upload failure; the specified previous-output guarantee concerns processor/validation failures.

## Isolation, issues, and remaining gates

- Only this new owned scratch directory was written. No live repository, original proposal/evidence, environment, skill, config, service, credential, lock, or launcher was modified. Thirty intake-protected inputs and all 93 stock evidence files remain hash-identical. The audited environment retains its identical 30765-entry tree and dependency metadata hashes.
- Initial independent harness run had 32 import failures because it tried loading the newer pinned converter with installed 0.19 constants. A complete-test-frame diagnostic rendering also grew recursively; the outer tool call timed out while starting its second phase. Initial receipts remain untouched. `evidence/harness-issues.md` documents the narrow corrections; final actual-filter comparison uses the matching pinned package, and diagnostic rendering captures helper locals only. No product code/assertion or negative output was repaired to obtain green.
- No install, download, native compiler/converter, model construction/forward/training, GPU, network, Hub/cloud write, commit, or push was performed here.
- Parent still owns selective rebase onto separately advanced accepted files/headers, initial cmake/protobuf/sentencepiece declarations, all 16 graphs, three helper closures, build-capable image, combined numerical/native/package/CI qualification, merge and default-alert readback. Do not copy this frozen mirror wholesale into the live repository.

## Runnable handles

Root: `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-reaccept-ku9v_mpa`

- `bootstrap_review.py`: immutable-input authentication and exact two-patch reconstruction (already executed; intended for this original intake).
- `run_checks.py`: credential-free child environment, source-freshness receipts and bounded checks. Example rerun, writing only this review's own outputs:
  ```sh
  /usr/bin/env -i PATH=/usr/bin:/bin \
    HOME=/Users/shersingh/.hermes/cache/scratch/qmd-gguf-reaccept-ku9v_mpa/home \
    TMPDIR=/Users/shersingh/.hermes/cache/scratch/qmd-gguf-reaccept-ku9v_mpa/tmp \
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
    /Users/shersingh/.hermes/cache/scratch/qmd-gguf-reaccept-ku9v_mpa/run_checks.py \
    portable-installed portable-pinned counterexamples independent-v2-installed independent-v2-pinned
  ```
- `test_independent_contracts.py`: new independent writer/resource/namespace/public-entrypoint heldouts.
- Each run directory contains exact argv/environment/status, stdout/stderr, JUnit and source receipts; independent directories additionally contain resource, retained-artifact and entrypoint JSONL evidence.
- `spec_gate.py`, `quality_gate.py`, `evidence/intake.json`, `evidence/SPEC.json`, `evidence/QUALITY.json`: machine-readable reconstruction, counts, preservation, review sequence, and scoped decisions.
