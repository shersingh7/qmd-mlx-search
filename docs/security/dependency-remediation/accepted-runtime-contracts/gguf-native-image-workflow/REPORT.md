# Independent W1 GGUF workflow review

**SPEC: PASS. QUALITY: PASS (started only after the recorded SPEC pass).**
No blocking or nonblocking code findings in the exact three-file repair. This is offline controller/test acceptance only; parent integration and real native Linux amd64 image/CI qualification remain gates.

## Observed executions — distinct runs, not summed

| Execution | Raw runner exit | Cases | Passed | Failed / fixture errors |
|---|---:|---:|---:|---:|
| Preserved supplied suite | 0 | 74 | 74 | 0 / 0 |
| Authorized positive heldouts | 0 | 34 | 34 | 0 / 0 |
| Permanent unittest discovery | 0 | 26 | 26 | 0 / 0 |
| New bounded independent edges | 0 | 20 | 20 | 0 / 0 |
| Original S3 characterization | 1 | 34 | 33 | 1 / 1 |

All primary passing runs have zero skips. The 74-case suite includes eight actual zero-exit positive controls and 66 refusals/failures; the 34-case positive-contract suite includes seven zero-exit controls and 27 nonzero cases. Every admitted CLI has a matching fake hit. Original characterization still fails only because it requires the killed worker’s normal-completion marker; its actual S3 controller receipt is 124/not-qualified. It was not relabeled green.

## SPEC conclusions and concrete witnesses

| Contract / source lines | Result | Independent witness |
|---|---|---|
| S1 — workflow 306–326 | PASS | MatchAs, MatchStar, MatchMapping.rest refused before Docker, including nested/OR/class/unused-function edges; unrelated patterns, actual launcher and replacement digest controls pass. Single literal selection is not a proof of arbitrary Python immutability. |
| S2 — workflow 87–89, 205–208, 238–240, 387–417 | PASS | Actual required-directory storage failure never logs qualified, preserves raw -2 → public 130 and primary failure; exclusive mkstemp collisions avoid symlink/hardlink alias damage; successful required-path replacement leaves alias targets unchanged. Diagnostic create/replace/unlink failures remain nonqualified with explicit errors, no concealed deletion. |
| S3 — workflow 125–250, 282–290 | PASS | Unreaped anchor signal-before-wait guards; same-group TERM-resistant worker termination, early CLI success/failure, INT/TERM during registration, registration held beyond deadline, helper startup expiry with no fake admission, actual exec failure, early anchor exit and controller SIGKILL/EOF. No post-reap/recycled-PGID signal. |

Seven permanent, one preserved-positive and four new independent lifecycle runs retain return-time snapshots before fixture cleanup and observe the original three-second post-return window without late work. Worker lifetime FDs close; unrelated sentinels are alive at return and naturally exit 0. Normal cases record anchor reap. Abrupt controller death is raw -9 with no qualification receipt; independent anchor FD closure proves termination, **not** controller reap. The permanent abrupt snapshot initially observes a live worker, then EOF revocation closes its FD within the observation window; no instant-death claim is made.

Actual Python is 3.12.13 on Darwin/arm64. A same-executable child monotonic sample falls between parent samples; helper/controller use that executable and original absolute deadlines. Startup/transport delays consume rather than restart the synthetic budget. Redundant Darwin EPERM is explicit nonqualification, not fabricated signal delivery.

Preserved positive/refusal controls exercise source and checkout/head/context identity, exact JSON/digest membership, host/runner/daemon/image architecture gates, bounded no-follow snapshot, empty Docker credentials and explicit Unix endpoint, container hardening, full probe/no native-only bypass, JSON-safe logging, full CID + label ownership and successful absence readback, primary-vs-cleanup failures. Current accepted launcher/probe hashes are statically compatible; the additional accepted-probe-data positive binds the current 292f01ce… bytes without executing/importing them.

## Assertion / oracle / transport preservation

- All 74 runner bytes are identical to the authenticated original. All 34 CASES AST entries are identical. After excluding only the authorized S3 fixture setup/oracle and corrected source-hash literal, the **whole heldout AST** is unchanged, not merely selected assertions.
- Positive S3 worker differs only by a held lifetime FD; the original two-second side-effect code remains unchanged. No completion marker or timestamp was invented. Fake transports move only the active CLI boundary, preserve helper argv/session/fds and original deadlines, and fail closed without Docker/git fallback.
- The three W1 development-oracle corrections were checked against retained actual states: no CLI hit after expired admission, and truthful nonqualification for two already-self-KILLed/EPERM paths. Final assertions retain worker FD, anchor reap, sentinel and no-late-work requirements. The initial full test source was not separately archived, so no literal before/after hash claim is made for that development version.
- Review-only environment transport narrows shell bootstrap PATH to env/mktemp/mkdir, inner Python PATH to an empty owned directory, supplements explicit Hermes aliases and UV_OFFLINE=1 before imports, and leaves controller bytes unchanged. Synthetic polluted outer values remain in the credential-stripping controls. No inherited real credentials/context are forwarded or printed.

## QUALITY after SPEC

PASS: cohesive single-owner design, stdlib-only generated helper, no third sparse source/dependency, explicit lifetime limits, separate CLI/anchor/public statuses, fail-closed cleanup/persistence composition, and named source-bound regressions. Reviewed workflow 125–250/282–290/387–417, test 53–132/173–343/346–365, fake 11–30/36–44/67–109. No implementation changes requested.

## Identity, syntax and scope

- All 12,170 delivery regular files, 518 original-delivery files and 267 original-review files reauthenticated before and after. All 12,962 input-hash records remain unchanged, including the accepted public inputs and supplied actionlint. The 47 source special fixtures were lstat-only, never opened/followed/copied.
- Both the repair-baseline patch and all-new-files integration patch independently pass actual git apply --check and apply in separate fresh nonrepository trees; each yields exact candidate bytes for all three files. Baseline workflow preimage is not confused with parent-declared absent live preimages.
- Python controller/helper/test/fake compile, Bash -n, and authenticated actionlint 1.7.12 pass. Shellcheck/pyflakes are explicitly disabled. No Docker/daemon/registry/network/qualifier/launcher/compiler/converter/model/GPU/cloud/CI execution, live edits, installation, staging/commit/push or skills writes occurred.
- Group KILL plus anchor reap does not certify all descendant reap. Escaped groups, sole-owner descheduling/death, runner hardkill/loss and uninterruptible work are not unconditional guarantees. No real image, container lifecycle, Linux native or combined-integration acceptance is inferred.

## Reviewer fixture issues retained, not product failures

The superseded executable path ending python3.12.13 failed with raw 127 before any Python execution; the user erratum supplied the actual bin/python. My first permanent replay had 19/26 passing and seven reviewer allowlist failures before sentinel creation (fake.py --sentinel was omitted). Those raw artifacts remain; only the reviewer transport allowlist changed, with an exact fixture hash/argv guard. A fresh case root then passed 26/26 with unchanged product bytes. Neither run is folded into the primary passing counts.

## Evidence / replay handles

- SPEC decision: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/spec-verdict.json`
- QUALITY decision: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/quality-verdict.json`
- Source identities: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/source-manifest.json`
- Primary raw executions: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/supplied-execution.json`, `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/positive-execution.json`, `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/permanent-final-execution.json`, `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/independent-heldout-execution.json`
- New edge configurations, syscall/ownership audits, raw logs, snapshots: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/independent-heldouts`
- Syntax / real patch replay results: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/validation.json`
- Clock proof: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/clock-domain.json`
- Whole-AST oracle delta: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/positive-oracle.diff`
- Archived W1 first-oracle state audit: `/Users/shersingh/.hermes/cache/scratch/qmd-w1-independent-n8m2r7/evidence/development-first-artifact-audit.json`

Actual replay commands were run with run-isolated.sh (clean allowlist before Python imports), run_replays.py, and independent_heldouts.py under this fresh owned root. Reproduce in a **new** qmd-w1-independent-* root using the recorded source hashes and copied regular input closure, not in this evidence root or the original deliveries; the fresh-create guards deliberately reject overwriting evidence. Scripts/configs plus per-execution literal argv are included in the archive.

Reusable review lesson (recorded here only; skill writes out of scope): preserve the unreaped ownership anchor through the final signal, distinguish prevention oracles from killed-worker completion characterizations, and publish required persistence before success.
