# Independent GGUF workflow review

## SPEC: FAIL — changes required

**QUALITY: not started**, because SPEC did not pass. Three P2 findings below arise from four failed independent synthetic contracts. The supplied 74-case suite still passes; no negative result has been relabeled green. No proposal/source changes were made.

Reviewed merge target: `.github/workflows/gguf-native-image-qualification.yml` (one new file only).

- Workflow SHA-256: `a384be7b604411d9a41b15568dd9d37324632416d3fe7c2346b06f3f7d60b306`.
- Patch SHA-256: `dd03de5b3075b9a39b0bf2d4ff7cd03199ac34881dbb4d0cf009a0b6251d042c`.
- Manifest SHA-256: `e84773ff3b1a44694df7fd55d2a1305486185a2f8ee8c41aed0bb0d17f2f1dfe`.
- Independently authenticated **518 unique manifest-bound paths**, with matching hashes/sizes. Single new-file patch reconstructed byte-for-byte and `git apply --check` / owned fresh apply both exited 0.
- Embedded controller body (without terminating newline): `b0cc257ab4ede6c2e6ea9e887e4a4e5747edea9ba43d5ed52a86f19590606df4`. The generated `.py` file adds one newline and hashes to `606e0ea2d8389f1cb57cc3f5360f411e7ed047c00cb2c83a797f5a0f93c51d98`. These are distinct encodings of the same extracted controller; fixture execution leaves the embedded production bytes unchanged.

## Findings

### S1 — P2: pattern capture escapes the source-binding ambiguity guard

**Location:** workflow lines 206–214. `ast.MatchAs.name` stores capture names as strings, not `ast.Name(Store)`. The current scanner checks Name stores/deletes, definitions/args, imports and exception targets, but omits direct pattern binders. Inserting this immediately after the literal assignment is accepted:

```python
match 'ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie@sha256:' + 'd' * 64:
    case GGUF_IMAGE:
        pass
```

The retained executable **synthetic-only** snippet uses the fully spelled-out string; it contains only the original literal assignment and this match, never launcher/probe code. It confirms the post-match value is the `d` digest. Both `ast-match-direct-capture` and `ast-match-mapping-capture` then run the unmodified controller over the synthetic source variant: it pulls/runs the original `c8fe...` literal, exits 0 and prints qualification. Expected behavior is ambiguity refusal before pull.

**Scope:** the actual frozen launcher `f61076c5...` has one Store at line 49 and a Load at line 285; its existing literal is selected correctly. This finding is a concrete direct syntactic rebinding hole in the claimed guard, not a claim that the present launcher chooses a different image and not a demand to prove arbitrary Python runtime immutability.

**Repair direction:** recognize direct pattern-capture binders, or define a narrow source grammar and fail closed outside it, without importing/executing the launcher.

Reproducers/readbacks: `heldout/cases/ast-match-*/{syntactic-rebinding-reproducer.py,syntactic-rebinding-result.json,observed-receipt.json,stdout.log}`.

### S2 — P2: required receipt write can contradict the emitted success receipt

**Location:** workflow lines 279–285. Success status and `final_exit=0` are printed before `receipt.json` is written. `receipt-write-failure` uses only the fake CLI to occupy the owned output path with a directory, a deterministic storage-error fixture. The controller prints a qualified/final-0 receipt, then `write_text` raises `IsADirectoryError` and the process exits 1.

**Impact/scope:** this is not a green CI job; it is a false success receipt in the chosen CI-log evidence channel, with an unhandled final-write error. Equivalent write failures can arise from storage exhaustion or permissions. Required receipt persistence must be resolved/caught before publishing final qualification; preserve any earlier primary failure and log a consistent nonqualified exit on storage failure.

Evidence: `heldout/cases/receipt-write-failure/{observed-receipt.json,stdout.log,stderr.log}`.

### S3 — P2: timeout group cleanup stops when only the leader has exited

**Location:** workflow lines 128–138. The TERM/KILL loop breaks as soon as `child.wait()` reaps the direct leader. A group member ignoring TERM is therefore never sent KILL. `timeout-group-descendant` creates only a bounded Python fixture in that same owned process group, accelerates the fake 180-second wait to 0.3 seconds using the supplied adapter, and records the descendant alive at controller return, with an owned marker written afterward.

**Scope:** exit 124 remains preserved and the simulated Docker cleanup readback is not misreported as real Docker evidence. This demonstrates missing host-command group drainage, not an actual Docker/container leak. No guessed/recycled PID signals were sent by the test; the child is bounded to its natural completion point. Real Docker may never spawn this exact child, which this offline review cannot establish.

**Repair direction:** retain a reliable owned lifetime anchor through group termination/drainage. Do not add a blind delayed `killpg` after leader reap; that introduces recycled-PGID risk.

Evidence: `heldout/cases/timeout-group-descendant/{descendant-ready.json,descendant-late-side-effect.json,observed-receipt.json}`, plus timestamps in `heldout/summary.json`.

## Checks that passed / boundaries retained

- Fresh Bash `-n`, stdlib inline-Python `compile`, and checksum-authenticated **actionlint 1.7.12** all pass. Optional shellcheck/pyflakes were disabled and not run.
- Fresh relocated supplied suite: **74/74 passed**, 66 expected nonzero, 0 skipped. This is the latest 74-case suite, not a sum of archived runs.
- Independent heldouts: **30/34 passed**, **4 failed contracts**, 0 fixture errors/skips; 27 expected nonzero and 25 actual nonzero. The two false-green AST cases are the difference; receipt failure and timeout both still return nonzero but violate their additional contracts.
- Duplicate JSON keys and malformed/unexpected object/array types fail closed across engine, image, probe, cleanup identity/list. CID directory/FIFO/oversize cases are refused without removal. Harmless extra fields/digests and duplicate unrelated list IDs remain valid controls rather than arbitrary new rejection requirements.
- Correct pre-merge PR event types, candidate-branch push and manual triggers; `contents: read`; supplied checkout v6 SHA; no persisted credentials; exactly the two public sparse-checkout files. Quoted event environment and JSON-escaped raw command logs do not become shell/workflow commands.
- Current launcher image derives from AST, not a second pin; pull/run use that exact input. Full RepoDigest membership is tested as JSON, not substring. Host, runner, daemon and image platform gates are separate. Distinct event head/context/checkout plus both source hashes are retained.
- Only the frozen no-follow, bounded, singly linked regular public probe copy is mounted read-only. Credential-empty Docker configuration, fixed local Unix endpoint, stripped/allowlisted environment, no network, nonroot 65534, read-only root, cap drop, no-new-privileges, CPU/memory/PID bounds and owned exec tmpfs are present. No workspace/socket/credential/GPU mount or install.
- Primary refusal/nonzero/signal/timeout paths stay nonzero; cleanup requires no-follow owned CID, full-ID/label readback before removal and successful full-ID daemon listing for absence. Wrong ownership/daemon errors/missing CID remain nonqualification. A benign rm race plus proven absence is distinct from rm success alone. These are synthetic control-flow observations, not proof of actual Docker `--rm`/`--cidfile` behavior.
- Full probe mode is mandatory: no `--native-only` or banner-parser bypass. Python 3.12, Linux/x86_64, exact outputs, source-hash agreement and nonempty tool-version strings are checked. The repaired public qualifier owns exact uv-release gating. Static text/AST inspection of parent followup `292f01ce...` confirms the same CLI/receipt interface; it was never imported or executed.

## Evidence honesty, preservation and remaining gates

All daemon/image/probe/compiler-looking answers in this review are explicitly **SYNTHETIC**. Physical host is Darwin/arm64, approved Python is actual 3.12.13. No real Docker CLI/daemon, CI, image, registry, compiler, application launcher, qualifier, model, SDK, GPU, cloud, install or download was executed. Only local syntax tools, owned git patch replay and Python synthetic subprocess fixtures ran.

All **518 original manifest-bound paths** and the manifest itself remain unchanged after review, including the original RED receipt and current 74-case receipts. Archived smoke/71-case summaries, which are outside those 518 paths, remain present and are recorded separately in `evidence/retained-archive-summaries.json`; their archived counts are not added to fresh results. Nine protected input hashes still match their original handoff values. The only pre-existing exception is the parent-authorized live plan section 1b: removing exactly that span independently reconstructs `be37fc5d96af167b87fb396561c4791d511d95f3affdae5327d0a6f372b6ed58`. All ten actual before/after protected-input hashes remain unchanged during this review. No live/shared source, proposal, config, services, docs, skills, locks or other agent artifacts were modified.

The copied old qualifier `6717b377...` remained interface data only; its known banner refusal is not a workflow-source failure. Parent acceptance record for R1 `72ef9788...` was read, not re-executed; the empty-version followup and final accepted source handoff remain parent-owned.

One review-fixture preflight failed before any cases because a newline-terminated controller file was compared with the body-without-newline hash. `evidence/heldout-preflight-red.json` retains both hashes; only the fixture hash boundary was corrected, not controller/product behavior or a test expectation.

Before acceptance: repair/re-review S1–S3; finish accepted qualifier integration; run and read back actual native Linux amd64 candidate CI (source/checkout hashes, immutable digest/platform, compiler/uv results, real Docker lifecycle); then complete combined build/tests and exact-head CI. **No arm64, locked converter, training, GPU or cloud qualification is granted.**

Machine-readable result: `review-result.json`. Main receipts: `evidence/validation-final.json`, `evidence/synthetic-final-v2/summary.json`, `heldout/summary.json`, `evidence/authentication.json`, `evidence/preservation-after.json`.
