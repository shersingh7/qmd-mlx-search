# W1 GGUF workflow P2 repair — offline proposal

**Independent SPEC/QUALITY reacceptance is pending.** Parent alone owns live integration/publication. No live repo/source/config/service/skill writes or Docker/CI/network/install/native execution occurred.

## Changes and ownership design

- **S1:** AST binding scanner now recognizes `MatchAs.name`, `MatchStar.name`, and `MatchMapping.rest`; walking nested/OR patterns preserves conservative direct-rebinding refusal before pull. The frozen real launcher remains valid; no launcher import/execution or second image pin.
- **S2:** required receipt is written to an exclusively created, unpredictable owned temporary file and atomically replaced before the sole final JSON log. Caught storage errors yield `not-qualified`, final 1 if no earlier primary failure, otherwise preserve the earlier primary. Required failing path is not deleted to hide the failure. Raw CLI signal status is separate from public `128+signal` exit.
- **S3:** workflow generates a small stdlib `command-owner.py` in its owned directory. It is the session/group leader and remains anchored after the CLI exits. Controller never waits/polls it until after its last group KILL; an unexpected zombie still reserves identity. CLI status and private control/result pipes are separate. EOF/deadline cause the still-owned helper to KILL its own group; controller also KILLs before bounded anchor wait. Nonthrowing first-signal-wins handlers cover spawn/registration and repeat cancellation. No application recovery lease, external helper checkout, PID guessing, delayed post-reap kill or new dependency.

See `DESIGN.md` for the preimplementation state table and deadline/ownership boundaries. The native workflow retains exactly two sparse-checkout sources. Generated helper uses the same interpreter/system-wide monotonic clock and original pre-spawn deadline; synthetic 0.3s waits subtract from that original timestamp rather than restarting at receipt.

## Exact merge delta

| Path | Preimage SHA-256 | Final SHA-256 | + / - lines |
|---|---|---|---|
| `.github/workflows/gguf-native-image-qualification.yml` | `a384be7b604411d9a41b15568dd9d37324632416d3fe7c2346b06f3f7d60b306` | `c1439878b3342a7cbdae7fc34ebee7c2db13f91f04c7802af107bffac9486ec7` | +156 / -25 |
| `test/security/test_gguf_workflow_controller.py` | `ABSENT` | `9c62c564bcaaf9c671f6b73fb88c91259e1dc48e1d7f8e8ff27689309e3e75ee` | +371 / -0 |
| `test/security/fixtures/gguf_workflow_fake_cli.py` | `ABSENT` | `87873345779febe5fdcea6fb162379259bcfacfa243fa5575817c574e2988a78` | +109 / -0 |

Source-bound repair patch: `delivery/workflow-repair.patch` — `db0e846e4d8f06a538815a8d9a765462772f90dccf3253698dbc205df94edd88`.
Optional parent-only unintegrated new-file patch: `delivery/unintegrated-new-files.patch` — `ba2463a0f8417134b85ebe51b001a1b29b29208c8881a1943f44239f03aee932`.
Both patches were freshly `git apply --check`ed/applied in separate owned, non-repository trees: all four commands exit 0 and exact three-file output matches candidate bytes (`evidence/patch-replay.json`). No stage/commit/push/gh/CI.

## Actual execution results

| Run | Exit | Cases / passed / failed | Meaning |
|---|---:|---|---|
| Untouched original supplied suite | 0 | 74 / 74 / 0 | Original controller and assertions |
| Untouched original independent runner | 1 | 34 / 30 / 4 | Freshly reproduced all four RED contracts, zero fixture errors |
| Final repaired supplied suite | 0 | 74 / 74 / 0 | Same 74-case runner bytes/assertions; relocated active-boundary fake transport |
| Final repaired **original-characterization** runner | 1 | 34 / 33 / 1 | One S3 old-oracle fixture error: killed descendant cannot write its normal-completion marker; not relabeled green |
| Final repaired **positive-contract** heldouts | 0 | 34 / 34 / 0 | 33 protected heldouts plus explicitly authorized separate S3 positive oracle; zero fixture errors/skips |
| Permanent focused portable unittest discovery | 0 | 26 / 26 / 0 | 12 pattern, four receipt, seven lifetime, three failure/budget cases; zero skips |

Primary baseline receipts and old REDs are in `baseline-safe-*`; the first baseline runs remain separately in `baseline-supplied` / `baseline-heldout`, exact originals in `original/`, final runs in `verified-*`. Counts are distinct runs, not added together. Machine results and absolute raw handles are in `delivery/result.json`.

### Oracle and transport changes (explicit)

The original S3 runner requires `descendant-late-side-effect.json` to appear before deciding whether it appeared after return. This characterizes the old defect, not a killed worker. Its original bytes/RED and final failing characterization remain retained. Parent authorized a separate positive oracle: actual ready/group evidence; held worker lifetime FD; return-time snapshot before fixture cleanup; no late effect through the original 3-second observation window; owned termination plus anchor reap or an explicit nonqualified cleanup failure; unrelated naturally draining sentinel. The original 2-second worker side-effect code is unchanged in the positive heldout; closure comes from actual process death, not fabricated completion data.
All 74 original runner bytes are identical; all 34 CASES AST entries and non-S3/non-sourcehash assertion instances remain protected. Exact fixture deltas are in `evidence/verified-fixture-delta.json`. Source hash is the body **without** final newline, not the generated `.py` encoding. The moved transport verifies helper argv/fds/session, replaces only its actual CLI tail with owned Python, records every hit and fails closed on unexpected spawns. Original signal fixture now requests the controller to self-signal instead of signaling a numeric grandparent; worker observes actual PGID. Every admitted CLI in 74/34/34 cases has a proven matching hit (`evidence/assertion-and-transport-preservation.json`).
Isolation disclosure: the old workflow env -i explicitly stripped, but did not set, HERMES_DIR/HERMES_KNOWLEDGE_DIR. The first exact-baseline inner processes therefore had these two aliases **unset** (not inherited), although HOME/HERMES_HOME/KNOWLEDGE_DIR were owned and no application/private-knowledge module was imported. Those runs remain archived rather than hidden. Both full original suites were freshly repeated with a shell-only environment transport adding these aliases before Python imports; original runner/controller/assertion bytes and old controller hash are unchanged, all four REDs reproduce. Every 74/34 case has an environment-transport receipt proving the body unchanged. These fully explicit-allowlist repeats are the primary reported baseline evidence. The repaired workflow already sets both aliases directly.

### Syntax and lifecycle

- Bash `-n`, controller/helper/focused-test Python `compile`, and authenticated supplied actionlint **1.7.12** all pass; actionlint stdout/stderr empty. Shellcheck/pyflakes explicitly disabled/not run. Python actual **3.12.13**, physical host **Darwin/arm64**. No native Linux/image result inferred.
- Seven permanent lifecycle cases cover timeout, early success/failure leader exit, TERM/INT, repeated signals during spawn registration, primary timeout with simulated failed Docker cleanup, and actual owned controller SIGKILL/EOF. Each retains a pre-fixture-cleanup return state and observes no late effect for at least the original three-second window; each real worker lifetime FD closes, unrelated sentinel naturally exits 0. See `evidence/lifecycle-summary.json` and individual `return-state.json`/`observation.json`.
- Signal-delivery and reap failures are separately exercised with declared diagnostic seams; a real owned process is drained before an injected reap timeout to avoid fixture leaks. No failure turns green. On Darwin a redundant `killpg` after helper self-KILL may report EPERM; that stays a recorded command-cleanup failure/nonqualification, not claimed signal delivery. Worker FD closure and anchor reap are independently observed.
- An initial 26-test development run had three **new-oracle** errors (expired command incorrectly required a fake hit; two self-KILL/EPERM cases incorrectly required a second delivered signal). First fixture states remain in `evidence/permanent-first/`. Corrections require no fake invocation after expired admission and explicit nonqualified failure when signal delivery is unproven; no original protected expectation or deadline was weakened. Final v1 and strengthened v2 runs both pass 26/26; only v2 binds final test source.

## Preservation and limitations

- Reauthenticated all **518** original delivery paths and **267** independent review paths before and after: manifests, source, original REDs and receipts unchanged. No live whole-tree identity assertion was used; parent concurrent accepted GGUF/CI/observer/plan transitions remain outside this patch. Frozen launcher hash stays `f61076c5a71e5b251db42aadb1bbe15fb1689a2d3293f1085edcdefc1958a83b`.
- Frozen obsolete qualifier is interface/hash data only and was never imported/executed. The accepted parent qualifier is not edited or requalified here. Workflow remains dynamically bound to actual source bytes and full-mode probe interface.
- PR types/candidate push/manual trigger, contents read, pinned checkout v6/no credentials, AST-derived exact image/RepoDigest membership, distinct head/context/checkout, Linux amd64 host/runner/daemon/image gates, no-follow bounded public snapshot, empty Docker credentials/local Unix endpoint, nonroot/network-none/read-only/capability/resource/tmpfs hardening, full probe/no native-only bypass, JSON-safe logging and CID/label/full-ID-list cleanup contracts remain checked by retained positive/refusal fixtures.
- Group KILL delivery is not all-descendant reap certification; non-child descendants are not reaped by this controller, and deliberately escaped groups are outside its ownership. EOF/deadline work only while the surviving helper can run; sole owner descheduling/SIGKILL, runner loss, job hard timeout and uninterruptible kernel work cannot be called unrestricted cleanup. Incomplete signal/reap evidence forces nonqualification.
- Synthetic Docker answers do not qualify Docker `--rm`/`--cidfile`, daemon/container lifecycle or real image behavior. No registry request, native C/C++ execution, actual qualifier/launcher/compiler/converter/model/GPU/cloud, CI or release qualification occurred. Parent commissions independent SPEC then QUALITY; real native Linux amd64 CI/readback and combined integration remain separate.

## Durable handles

- Owned root: `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-w1-repair.WDt8LlYR`
- Design: `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-w1-repair.WDt8LlYR/DESIGN.md`
- Source-bound patch/result: `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-w1-repair.WDt8LlYR/delivery/workflow-repair.patch`, `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-w1-repair.WDt8LlYR/delivery/result.json`
- Manifest: `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-w1-repair.WDt8LlYR/delivery/manifest.json`
- Syntax results: `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-w1-repair.WDt8LlYR/evidence/validation-final.json`
- Final portable run: `/Users/shersingh/.hermes/cache/scratch/qmd-gguf-w1-repair.WDt8LlYR/evidence/permanent-final-v2-execution.json`

Reusable lesson (recorded only here because skill writes are out of scope): keep an owned group identity alive/unreaped until the final signal, distinguish a normal-completion characterization from a prevention oracle, and publish required persistence before success.
