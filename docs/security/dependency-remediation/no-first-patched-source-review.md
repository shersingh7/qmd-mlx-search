# No-first-patched advisory investigation

## Outcomes

- **GHSA-x9r9-c232-4q39 / CVE-2026-80047 (Transformers, original alert 176): an actual fix exists in 5.17.0.** The current advisory's `first_patched_version: null` and bounded range ending at 5.8.1 do not describe actual source behavior accurately. 5.9.0 and 5.16.1 retain the unsafe download-before-consent ordering. 5.17.0 contains the referenced fix; 5.22.2 retains it.
- **GHSA-4j2p-28q2-5m79 / CVE-2026-69112 (Accelerate, original alert 161): no fixed published upstream release was found.** The latest public PyPI release, 1.15.0, still joins untrusted `weight_map` values without containment or regular-file validation. Current upstream main at `f5cc53877677eb7fa0ef7091fe41536a777fee6a` retains the same code. Upgrading to 1.15.0 can evade the advisory's `<=1.14.0` range but is **not an actual vulnerability fix**.
- All investigation writes are confined to `/Users/shersingh/.hermes/cache/scratch/qmd-dependency-remediation/`. No repository source, dependency manifests, alerts, GitHub settings, or dismissals were changed by this investigation. Public package wheels were downloaded only for inspection, not installed. No models, GPU, training, cloud jobs, credentials, private data, or live index were used.

## Transformers provenance

1. Official GitHub advisory (retrieved live) still declares `>=4.49.0, <=5.8.1`, with no first patched version:
   - https://github.com/advisories/GHSA-x9r9-c232-4q39
   - https://api.github.com/advisories/GHSA-x9r9-c232-4q39
2. Its referenced PR was merged on 2026-09-08:
   - https://github.com/huggingface/transformers/pull/48620
   - Actual fix commit: https://github.com/huggingface/transformers/commit/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55
   - The diff replaces the initial `get_cached_module_file(...)` with an existence-only `has_file(...)`, and moves the module download **after** `resolve_trust_remote_code(...)`.
3. Release 5.17.0 explicitly lists PR #48620 and was published 2026-09-09. GitHub compare reports `ahead`, `behind_by: 0`, `ahead_by: 8` from the fix commit to the release tag, proving ancestry:
   - https://github.com/huggingface/transformers/releases/tag/v5.17.0
   - https://api.github.com/repos/huggingface/transformers/compare/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55...v5.17.0
4. Tag source and SHA256-verified PyPI wheel have identical generation source:
   - Fixed 5.17.0 source: https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py#L605-L646
   - Fixed 5.22.2 source: https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/generation/utils.py#L618-L658
   - Still-unsafe 5.9.0: https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/generation/utils.py#L462-L486
   - Still-unsafe previous release 5.16.1: https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/generation/utils.py
   - Public artifact metadata: https://pypi.org/pypi/transformers/json

### Defensible Transformers remediation requirement

- Require **at least 5.17.0 for this advisory**; do not use `>5.8.1` or `>=5.9.0` as a security floor. Pin/lock a verified fixed artifact. Other Transformers advisories can require a higher floor.
- 5.22.2 is verified to retain this fix and is the latest public PyPI release observed. It requires **Python >=3.11**, while 5.17.0 requires >=3.10. Align `pyproject.toml` and independently resolved PEP 723 script environments if selecting 5.22.2.
- Add a regression that simulates remote custom generation with trust declined and asserts that no module download/cache function is called before consent. Explicit `trust_remote_code=False` alone does not repair vulnerable releases: the bug is the write *before* that flag is evaluated.
- The project has no direct `load_custom_generate` or `custom_generate` references in its finetune Python source, but it loads models and calls `.generate()`. Do not equate absence of a direct call with a package fix or assume custom generation cannot be reached through model metadata.

## Accelerate provenance and blocker

1. Official GitHub advisory (retrieved live):
   - https://github.com/advisories/GHSA-4j2p-28q2-5m79
   - https://api.github.com/advisories/GHSA-4j2p-28q2-5m79
   - Declares `<=1.14.0`, `first_patched_version: null`.
2. Latest public release/artifact:
   - https://github.com/huggingface/accelerate/releases/tag/v1.15.0
   - https://pypi.org/pypi/accelerate/json
   - Both the tag and the SHA256-verified 1.15.0 wheel retain:
     ```python
     checkpoint_files = sorted(list(set(index.values())))
     checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
     ```
   - https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/modeling.py#L1936-L1944
   - Both public entrypoints reach this code: https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/big_modeling.py#L634-L647
   - Current main pinned to an immutable commit also retains it: https://github.com/huggingface/accelerate/blob/f5cc53877677eb7fa0ef7091fe41536a777fee6a/src/accelerate/utils/modeling.py#L1949-L1957
3. Advisory-linked PRs were **not merged**:
   - https://github.com/huggingface/accelerate/pull/4070 (`merged: false`), path-traversal half only.
   - https://github.com/huggingface/accelerate/pull/4138 (`merged: false`), explicitly leaves FIFO handling out.
4. A later proposed fix covers both traversal and non-regular files, but was rejected, not merged:
   - https://github.com/huggingface/accelerate/pull/4214
   - Maintainer rejection: https://github.com/huggingface/accelerate/pull/4214#issuecomment-5586211932
   - Issue comment explicitly says they are not accepting PRs for this: https://github.com/huggingface/accelerate/issues/4067#issuecomment-5586233400
   - Current policy treats loading untrusted artifacts as an operator trust decision: https://github.com/huggingface/accelerate/blob/f5cc53877677eb7fa0ef7091fe41536a777fee6a/SECURITY.md

### Defensible Accelerate remediation requirement

An upgrade-only closure is not defensible. Choose and document one of:

1. **A controlled patched dependency**: vendor/fork a precisely pinned upstream revision, implement the missing validation in the loader itself, lock the actual patched artifact/source, and run negative and compatibility tests. Do not describe PR #4070, #4138, or #4214 as a merged upstream security fix.
2. **A project-level fail-closed loading boundary**: ensure every relevant model/checkpoint load path uses validation or disables the vulnerable loader before it can run, including indirect PEFT/TRL callers and independently run scripts. This is an application remediation, **not a fixed upstream release**; it needs explicit integration coverage. If a complete boundary cannot be established, report an honest unresolved upstream blocker rather than dismissing the alert or treating 1.15.0 as fixed.

Validation requirements for either local approach:

- Validate checkpoint index schema and shard entry types. Reject empty/malformed names, NULs, absolute paths, foreign drive/UNC forms where relevant, and normalized paths escaping the checkpoint root. Use path-component/commonpath containment, not a vulnerable textual `startswith(root)` check.
- Reject special-file shards **before opening**: FIFOs, sockets, character/block devices, directories, missing files, and broken links. Require the final target to be a regular file. `safetensors`, `weights_only=True`, `trust_remote_code=False`, and `local_files_only=True` do not independently prevent path traversal or a special-file open.
- Apply protections to directory and explicit-index inputs, and to both `load_checkpoint_in_model` and `load_checkpoint_and_dispatch`. Check the final shard-load boundary too, so alternate discovery paths cannot select a special file as `pytorch_model.bin` or `model.safetensors`.
- Treat symlinks deliberately. A blanket resolved-path-within-snapshot check breaks normal HF cache layouts (`snapshots/<revision>/...` -> `../../blobs/<hash>`), as the proposed PRs explain. Conversely, allowing *every* symlink to an external regular file is not arbitrary-file-read containment. Either reject escapes for local artifact directories, or allow only positively identified trusted cache blob targets under an approved root; test that exception. Account for mutation races by using controlled/read-only staging or equivalent safe opening semantics rather than claiming a preflight `stat` solves hostile concurrent replacement.
- Ensure early initialization and alias coverage if using runtime guards. `accelerate.big_modeling` imports `load_checkpoint_in_model` by value; replacing only `accelerate.utils.modeling.load_checkpoint_in_model` after that import can leave a bypass. PEFT/TRL/other libraries can hold additional aliases. A validated central patched dependency is less fragile than ad hoc callsite wrappers.
- Offline tests should reject relative traversal, absolute/foreign-root entries, prefix-sibling escapes, symlink escapes, FIFO/special files, malformed indexes, and mixed entry types **without attempting any rejected shard open**. Positive tests should cover ordinary, nested, explicit-index, and supported HF-cache layouts. Exercise public APIs and actual project launch paths, not only a standalone validator.

## Local source observations for the finetune writer

Read-only inspection found these loading surfaces (line numbers are observations during concurrent work, not an immutable revision):

- `finetune/train.py:359` gives a string model ID directly to `SFTTrainer`; `:523` calls `AutoModelForCausalLM.from_pretrained`, and `:527` calls `PeftModel.from_pretrained`.
- `finetune/eval.py:59,71`, `jobs/eval.py:319,325,330`, `experiments/grpo/grpo.py:80,84`, conversion scripts, retrieval eval, benchmarks, and `train_unsloth.py:74` have additional load paths.
- `convert_gguf.py:120,131`, `convert_onnx.py:73,84,225`, and `dataset/prepare_data_lfm2.py:48` explicitly enable remote code. Removing unjustified opt-ins is worthwhile defense-in-depth, not a substitute for either fix.
- PEP 723 headers independently declare Transformers/Accelerate dependencies in multiple runners; fixing only the project lock does not constrain these isolated environments.
- At the later read, concurrent worker changes had already resolved the local lock to Transformers 5.22.2 and Accelerate 1.15.0. That is sufficient for the Transformers advisory researched here, but **not for the Accelerate vulnerability** without additional remediation.
- No direct use of either named Accelerate checkpoint API was found in project Python source. This does not prove transitive loaders or independently imported aliases unreachable. No installed-package end-to-end reachability claim is made here.

## Executed verification and artifacts

`probe_upstream_no_first_patched.py` compiles the actual upstream function bodies with future annotations and dependency mocks. It does not import ML packages or modify their code. Consent is stubbed to deny; checkpoint reading is intercepted before the first shard read. Thus these are **source-level executable regressions, not installed-package integration tests**.

Executed command:

```sh
TMPDIR=/Users/shersingh/.hermes/cache/scratch/qmd-dependency-remediation PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 /Users/shersingh/.hermes/cache/scratch/qmd-dependency-remediation/probe_upstream_no_first_patched.py
```

Results: **34 probes passed**, comprising:

- **10 Transformers cases**: 5.8.1 / 5.9.0 / 5.16.1 attempt download before denied consent; 5.17.0 / 5.22.2 do existence-check only, then deny before download. Each tested `trust_remote_code=False` and `None`.
- **24 Accelerate cases**: the released 1.15.0 function bodies pass traversal, absolute, FIFO, and directory shard paths to the mocked loader, through both public APIs, with both directory and direct-index arguments. Ordinary nested files and a hub-like regular-blob symlink also reach the mocked loader. No FIFO was actually opened, no arbitrary host file was read, and no model was loaded.

SHA256-verified wheels; selected source members match their GitHub tag byte-for-byte:

| Package | Version | Wheel SHA256 |
|---|---|---|
| transformers | 5.17.0 | `78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801` |
| transformers | 5.22.2 | `829c2820aff91730b8ec45a4b8ff4be07f586a140b68a9073998b36fa35297dc` |
| accelerate | 1.15.0 | `97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec` |

Evidence files in the scratch directory include:

- `no-first-patched-probe-results.json` — full executable probe results.
- `probe_upstream_no_first_patched.py` — replayable hermetic source-level probes.
- `no-first-patched-evidence-manifest.json` — retrieval timestamp and file SHA256 inventory.
- `transformers-*-artifact-evidence.json`, `accelerate-1.15.0-artifact-evidence.json` — wheel URLs, timestamps, digest checks, Python requirements, source-match result.
- Official advisory/PR/comment/release API JSON, pinned upstream source, and the inspected public wheels.

Remaining blocker: **Accelerate has no verified upstream fixed release.** Project integration tests and any implementation of a local fix belong to the finetune writer; this investigation did not change or certify project remediation.
