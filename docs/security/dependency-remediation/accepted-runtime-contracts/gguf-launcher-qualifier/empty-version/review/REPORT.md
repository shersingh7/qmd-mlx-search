# Empty prerequisite version refusal: independent two-file review

## SPEC: PASS

Source-bound acceptance of exactly `finetune/tools/qualify_gguf_image.py` plus
new `finetune/tests/test_qualify_empty_version.py`. No live integration.

- Patch SHA-256: `3408d9b639ea3b0cadb1d3522fa0b04b304acacdc00f7d05ce0ab5da5bf9113c`.
- Base accepted review result: `0cd28c76456d3dcec36ab4784cac64d865bef91b30c27a1a333d1a159843cbed`.
- Qualifier preimage: `72ef9788f08b2f331de19653302b1e96ec630b7495463d3b19a8264a03101ccc`.
- Qualifier candidate: `292f01cedb9f2b61f3dce87baba3680749f72ea32566953f937146f09a0114d2`.
- New regression file: `362ea09d114af4573d58f44f3577b076d3443d93b6183f2e60260701775a0fa8`.
- Independently checked preimages, applied only the two-file patch in an owned
  derivative (`git apply --check` and replay both exit 0), then matched all 43
  candidate source hashes. Base has 42 files; candidate preserves 41 byte-for-byte,
  changes one existing file, and adds one. All 42 baseline files remain preserved.

### Fresh execution evidence

| Surface | Cases | Result |
|---|---:|---|
| Authored regression file against exact old qualifier | 27 | 27 expected failures; pytest exit 1, no errors/skips |
| Same regression file against candidate | 27 | passed |
| Retained uv release positive/wrong-release contracts | 4 | passed |
| Retained parser and public CLI contracts | 59 | passed |
| Candidate combined scoped suite | 90 | exit 0; 0 failures/errors/skips |

The combined suite ran once in 29.68s; its three groups
are not added again to a separate total. Exact executed case-name multisets
match the source-bound retained fixtures. The 27 old-source failures all reach
the exit-code assertion at regression line 65 and show `IndexError: list index
out of range`, exit 1. Thus call-order, no-build-sentinel, and temporary-directory
cleanup assertions preceding that line did not fail. This is genuine RED, not
an import/setup failure and not an acceptance requirement for the buggy state.

The candidate's 27 real CLI negatives cover full mode's five queried tools and
native-only mode's four, each with empty, newline-only, and whitespace-only
successful stdout. They require exit 2, exact named refusal, empty stdout, no
traceback, no build attempt, expected call prefix/order and no workspace leak.
The fixtures are explicitly authored version-only/no-build controls, not claims
of native success.

Retained 4+59 cases independently reran the accepted upstream bare/suffixed uv
banners, exact release tokens including +N/prerelease refusals, malformed banners,
missing uv, native-only uv omission, simulated wrong-Python gates, default full
mode, help and argument refusals. Their assertions are unchanged; only the prior
candidate source-hash literal was adapted upstream. Reviewer's copied support
added missing owned `KNOWLEDGE_DIR`, `TMP`, and `TEMP` values only.

### Real host controls (not pytest cases)

1. Actual installed `/Users/shersingh/.local/bin/uv`, no fake uv and no
   `--native-only`: exit 0, actual Python 3.12.13, Darwin/arm64, actual Apple clang
   C/C++ and make, outputs `qmd-c-ok` and `qmd-cxx-ok`. Receipt is bound to the
   candidate qualifier and explicitly denies converter/training/GPU/cloud
   qualification. Temporary directory emptied on return.
2. Real-tool wrappers add whitespace/additional version lines while delegating
   every real command; make build stdout is deliberately suppressed. Exit 0,
   identical first-line version receipt and real C/C++ outputs; cleanup verified.
   This checks unchanged nonempty banner semantics and proves the new blank guard
   is version-only, not a global ban on quiet native commands.

### Contract preservation

Qualifier lines 54–59 are the entire production delta. `run()` already strips
stdout (line 26); the new check routes blank results through existing RuntimeError
handling (lines 88–90) rather than hiding IndexError globally. Version commands
remain timeout 10s and same-order; build remains 60s; native executables remain
10s. `run`, `_uv_release`, and `main` have identical ASTs to the accepted base;
all production bytes outside that extraction block are identical. Python 3.12,
uv 0.11.18 identity, native-only selection, output verification and receipt scope
are unchanged. The new tests' literal prerequisite names include `c++` correctly.

## QUALITY: PASS (assessed after SPEC PASS)

No in-scope defects found. The fix is minimal, localized to extraction, keeps
refusal inside TemporaryDirectory cleanup, preserves real stdout and release
parsing semantics, and does not weaken positive/native evidence. Existing Q1/P3
empty-version diagnostic is closed on this exact candidate; prior R1/P1 closure
remains preserved. No source repair was performed by this reviewer.

The original 162-case launcher test file, launcher, docs, locks, manifest and
helper remain byte-identical to the accepted base. The helper's preserved hash is
`5c36cc3f73ec03cd7c89f7f8f7083987824f4b8bae19ee720a02177fa03ea028`; it was not imported/executed and is not
newly qualified. All protected parent/prior source and selected evidence hashes
match before/after snapshots.

## Review-only issues and count clarification

- Initial post-test aggregation expected nonempty JUnit classnames outside
  `--rootdir`; pytest emitted empty classnames for the 63 external retained cases.
  Product suite already passed 90/90. Corrected only reviewer aggregation to exact
  case-name multiset comparison and resumed from original receipts, without
  rerunning, overwriting or double-counting tests. Initial runner and issue receipt
  are retained under `inputs/run_review-initial.py` and
  `evidence/reviewer-aggregation-issue.json`.
- Parent manifest's `originalInputsUnchanged: 42` must not be read as 42 unchanged
  candidate source files: independently measured counts are 42 preserved baseline
  files and 43 candidate files (41 unchanged existing + 1 changed + 1 new).
  No manifest/lock was modified.

## Scope and remaining gates

This is only the two-file repair review, not release or image acceptance. Parent
integration, native Linux amd64 isolated image execution, whole-workspace build
and exact-head CI remain open. Original 162 launcher cases were preserved, not
freshly rerun here; the parent's 252-case green artifact is retained but is not
this review's fresh count. Older 85 heldouts and seven mutation controls were
not freshly rerun. No workflow proposal review, network, resolver/install,
Docker/container, cloud, GPU, model, converter, training or evaluation execution.

## Artifact locations

- Root: `/Users/shersingh/.hermes/cache/scratch/qmd-empty-version-review-v6k8m2`
- Frozen candidate: `repo/`; baseline: `baseline/`; old-source RED derivative: `old-repo/`.
- `evidence/source-authentication.json`, `execution-summary.json`,
  `preservation-verification.json`, `protected-before.json`, `protected-after.json`.
- `evidence/independent-empty-red.{json,xml}` and
  `evidence/independent-empty-and-retained-green.{json,xml}`.
- `evidence/independent-real-installed-uv-full.json` and
  `evidence/independent-real-multiline-quiet-full.json`.
- `review-result.json` and `artifact-manifest.json` bind the report/evidence;
  review-owned files are frozen read-only after completion (not OS immutable).
