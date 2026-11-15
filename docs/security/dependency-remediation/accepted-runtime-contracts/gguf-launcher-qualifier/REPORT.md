# Independent uv-repair reacceptance and composed quality review

## Verdict

- **SPEC: PASS for the exact composed four-file source proposal. R1/P1 is closed.**
- **QUALITY: PASS WITH ONE NONBLOCKING P3 DIAGNOSTIC NIT.** No important new failure; quality is no longer deferred.
- This accepts the scoped launcher/qualifier source proposal, **not** a Linux image, integrated checkout, locked converter/training graph, GPU path or HF job. The parent's integration/CI gate remains open.

## Exact source binding

| Artifact | SHA-256 |
|---|---|
| Original four-file proposal patch | `d858a625c036ed4c3bd826383f76b7b369271a2dcf1a389e8bf443ec3362719b` |
| Settled two-file repair patch | `50744121d38fdb998cce014e716302bf0e2010a0eff0a221d0649c94b0a48710` |
| Composed launcher, unchanged by repair | `f61076c5a71e5b251db42aadb1bbe15fb1689a2d3293f1085edcdefc1958a83b` |
| Composed docs, unchanged by repair | `be69f428d973a52ea94c64a9dd196b02a7c978a7e8318f7d85c70b12e4caf8df` |
| Repaired qualifier | `72ef9788f08b2f331de19653302b1e96ec630b7495463d3b19a8264a03101ccc` |
| Append-only repaired tests | `cb512f84519f3e6013a7750c1d699f9a0a0a598afb1afe878357cfaf301502d4` |
| Staging-data-only helper, never executed | `5c36cc3f73ec03cd7c89f7f8f7083987824f4b8bae19ee720a02177fa03ea028` |

The repair manifest was independently authenticated against all 42 original reviewer bindings and all 42 parent candidate inputs. Fresh `git apply --check`, apply and reverse-check succeeded in an owned fixture. A second owned fixture reversed both patches to the original three files/no qualifier, verified the original manifest identities, then reapplied both patches and reproduced all 42 composed input hashes. No staging, commits or shared-source edits occurred.

The original 39,234 test bytes are an identical prefix of the 42,117-byte candidate. All original live test bytes are also preserved as a prefix. The positive-contract adaptation changes only one explicit expected probe SHA; its original assertions, inputs and expected exits are byte-identical. The original contract file, original red evidence, and the parent's original artifacts remain untouched.

## Fresh execution

Each row is a separately executed surface; overlapping cases are not summed into a novel acceptance total.

| Surface | Actual result |
|---|---|
| Original four-case positive-contract suite against old probe in owned fixture | **2 passed / 2 failed**, 0 errors/skips, exit 1; only the two valid upstream suffix cases fail |
| Same four cases against exact repaired probe, source-binding-only adaptation | **4 passed**, 0 failures/errors/skips, exit 0 |
| Complete composed launcher suite | **162 passed**, 0 failures/errors/skips, exit 0 |
| Independent narrow parser/CLI/receipt held-outs, final run | **59 passed**, 0 failures/errors/skips, exit 0 |
| Full probe with actual installed uv, no shim and no `--native-only` | **exit 0**, actual `qmd-c-ok` and `qmd-cxx-ok` compilation/execution |
| Real host `--native-only` probe | **exit 0**, actual C/C++ compilation/execution; narrower receipt retained |

The fresh 162-case XML contains **every original 149 case**, verified with multiset comparison rather than inferred from totals. All **six real-source closure cases** ran: three AST/call-site contracts and three public preview/staged-helper-hash/argv contracts. The additional 13 cases are the parent's append-only three accepted displays and ten wrong/malformed refusals.

The 59 held-outs include exact token retention for `+0`, `+23`, other build suffixes, prerelease/development suffixes and wrong prefixes; full grammar matching; four independently compiled full-CLI positive displays; refusal before build; strict CLI option boundaries; no uv execution in native-only mode even with wrong/missing uv; unchanged Python-minor gate controls; and truthful receipt scope/source hash. Simulated Python-minor controls test gating only, **not** actual Python 3.11/3.13 runtime compatibility. Thirty-five isolated child CLI receipts preserve argv, environment, exit, stdout/stderr and observed tool calls.

Actual full host identity: **Darwin/arm64, Python 3.12.13**, installed uv banner `uv 0.11.18 (e32666915 2026-06-01 aarch64-apple-darwin)`, Apple clang 21.0.0, GNU Make 3.81 and Git 2.53.0. This is **not Linux container execution**. The receipt explicitly retains `native_only: false` and `locked_python_converter_training_gpu_cloud_qualified: false`.

Key receipts:

- `red/evidence/red-positive-contract.{json,xml}`
- `evidence/green-positive-contract.{json,xml}` and `evidence/contract-*.json`
- `evidence/candidate-suite.{json,xml}` and `evidence/suite-coverage.json`
- `evidence/repair-heldouts-final.{json,xml}` and `evidence/heldout-*.json`
- `evidence/real-uv-full.json`, `evidence/real-native-only.json`
- `evidence/repair-source-proof.json`, `evidence/final-verification.json`

## R1 reasoning and completed quality review

The repaired `_uv_release` fullmatches the bare or single-parenthesized-metadata display and returns the **entire exact release token**. The existing guard still compares it with exactly `0.11.18`. It does not use prefix matching, normalization, floating-point comparison, or strip `+commits`/prerelease suffixes into a match. The full displayed first-line banner, including metadata, remains in the receipt. Parenthesized metadata is opaque display information, **not** source/build authenticity evidence.

The retained upstream source at revision `e32666915e7fd99bbd0f167f5c381f4fb4198486` was independently body-hash/size checked without network access. `crates/uv-cli/src/version.rs:66-85,179-193` confirms that `+N` is part of the release token and that target or commit/date/target belongs in the parenthesized suffix. All eight successfully retained public-source files match the original retrieval receipt.

The composed launcher, qualifier, full test file and docs were read; affected converter/trainer/eval/helper call sites were inspected as text/AST only. No exporter, trainer, eval or GGUF helper was imported/executed.

Quality conclusions:

- Closure additions are narrowly limited to the three GGUF-capable runners. Image selection derives from helper membership, avoiding a second production runner list. Independent test expectations cover all 15 defaults, with the other twelve closures unchanged.
- `None`, not truthiness, selects defaults. Explicit image values remain literal; opaque script argv cannot change launcher image selection. No new custom-image validation policy was introduced.
- Seven exact inverse replacements reproduce the original launcher **byte-for-byte**, establishing preservation of all unrelated source: safe reads, lock/policy checks, secret/input handling, staging/snapshot semantics, fixed-shell argv, private upload/readback, single-run submission and remote command/image readback. The repair changes none of these bytes.
- The parser repair is small, stdlib-only, private and documented. Native tool checks, Python gate, command timeouts, actual build/output verification and qualification flags are unchanged. Positive regressions compile with real host tools and replace only the uv display, avoiding fake-build acceptance.
- The real-source closure tests traverse actual imports/call sites and retain independent closure expectations; staging/hash success is not misrepresented as helper correctness. The docs distinguish registry metadata, host probe, actual image execution, graph readiness and remote/GPU qualification.
- The prior **85 launcher held-outs and seven mutation controls are retained evidence under unchanged launcher bytes**, not fresh runs and not added to the fresh 59-case count.

### Q1 — P3, nonblocking: empty successful version output escapes the clean refusal diagnostic

At `finetune/tools/qualify_gguf_image.py:54`, `run(...).splitlines()[0]` raises `IndexError` if a prerequisite exits 0 with empty stdout. `main` at line 84 does not catch it. An empty synthetic uv display reproduced **exit 1 and a traceback**, with no success receipt and no build attempt. The original qualifier reproduces the same behavior, so it was not introduced by the release-parser repair and does not reopen the valid-upstream refusal.

This is a diagnostic robustness nit, not a false qualification or normal official-uv-path blocker. Optional follow-up: reject an empty first-line result explicitly with `RuntimeError` and add a clean refusal test. **No product fix was made or required for this bounded R1 closure.** Evidence: `evidence/heldout-grammar-0.json` and `evidence/old-probe-empty-banner.json`.

## Preservation, issues and limits

Final before/after checks confirm unchanged: **42 frozen reviewer inputs, 42 parent candidate inputs, six frozen delivery artifacts, three original live files, six prior review artifacts and eight parent repair artifacts**. The 42 owned candidate inputs and 42 owned red inputs also retain their exact expected hashes. Every write is beneath this review's own scratch directory.

Two reviewer setup issues are disclosed and retained, not counted as product failures:

1. The supplied executable spelling `security-venv/bin/python3.12.13` did not exist (exit 127, before any fixture write). The same approved venv's `bin/python` was resolved and verified as actual **3.12.13** before use; no install or runtime substitution outside that venv occurred.
2. The first held-out run had one reviewer assertion expecting argparse's generic unknown-option message for `--native-only=false`; argparse correctly refused with its specific `ignored explicit argument 'false'` message. The initial **58-pass/1-fail** XML/JSON, test bytes and all 35 initial CLI receipts are preserved. Only that message expectation was corrected; the same 59 cases then passed fresh with all refusal/no-tool-call assertions intact.

Child invocations use allowlisted, owned HOME/Hermes/Knowledge/XDG/HF/cache/TMP roots, offline flags, CI, disabled bytecode/plugin/cache generation and no credentials/live flags/PYTHONPATH. The probe itself further restricts the native child environment. There were no dependency installs, network requests, Docker commands/pulls, model/GPU work, jobs, uploads, harness launches, subdelegations or shared-source/config/doc/skill/workflow/manifest/lock edits.

The separate new workflow proposal was **not** reviewed, touched or polled. Its absence is not used as a source-SPEC failure. Parent obligations remain explicit: independently review/integrate that workflow and the composed sources, then obtain actual native Linux amd64 engine/runner/image/digest/source/readback/probe/cleanup evidence and combined build/tests plus exact-head CI. Arm64 manifest availability is not arm64 execution. Locked converter/training, initial dependency graphs, GPU/CUDA, HF scheduling/storage and publication remain separate unqualified gates.
