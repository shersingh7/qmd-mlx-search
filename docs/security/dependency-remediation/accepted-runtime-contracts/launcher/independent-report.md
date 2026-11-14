# Malformed-secret-option repair — independent reacceptance

## Verdict

**SPEC PASS. QUALITY PASS.** The previously reproduced malformed `--secret-env` reflection defect is closed for the reviewed bytes. This is a narrow acceptance of that repair, not cloud/image/runtime or whole-repository acceptance.

## Scope and source evidence

Read-only repository: `/Users/shersingh/github/qmd-mlx-search`.

Exact diff inspected against `346d905`, limited to:

- `finetune/tools/submit_locked_job.py`
- `finetune/tests/test_locked_job_submission.py`
- `finetune/LOCKED-JOBS.md`

The launcher diff replaces only the interpolated exception text in `check_env_name` with a constant launcher-authored diagnostic and adds its explanatory comment. AST comparison proves the rejection predicate and every other part of the launcher are unchanged. Invalid input is not split, trimmed, normalized, or accepted as an inline secret. `ValueError` and the public refusal exit code 2 remain unchanged.

The repository test diff adds the six equals/space/newline × preview/submit regressions. AST comparison proves no pre-existing repository test was altered. The original independent 101-case file was executed without modification. The documentation now explicitly distinguishes the initial Python lock from external converter source, build tools, and later dependency installations (`LOCKED-JOBS.md:193-200`).

## Spec acceptance

- Recognized malformed secret options in both preview and submit mode return 2, leave stdout empty, and print only the generic refusal; no inline value is reflected in stdout or stderr.
- Both `--secret-env VALUE` and `--secret-env=VALUE` spellings are covered. Additional bounded probes cover tab/CRLF/outer whitespace without normalization or acceptance.
- Malformed options are refused in preflight, before staging, secret environment reads, or SDK initialization. In-process sentinels forbid those operations. Twelve fresh `-B -I -S` processes independently forbid SDK/ML imports, network, child processes, staging, synthetic HOME access, and secret reads; all twelve audit receipts contain zero violations.
- Related recognized `--env`/`--secret-env` syntax, reserved-name, duplicate, and name-collision errors remain refused without printing synthetic values.
- Valid names, their order, non-secret values, and explicitly selected secret values are preserved literally. Mock submission verifies exact names/values and that only selected environment names are read; manifests do not contain secret values.

The fresh-process guard allows only stdlib locale/terminal-dimension lookups (`LANGUAGE`, `LC_ALL`, `LC_MESSAGES`, `LANG`, `COLUMNS`, `LINES`). These are not secret-value reads.

## Quality acceptance (after spec passed)

The repair is localized, uses constant safe-to-print text, preserves the existing shared validator and exception type, and adds focused regressions without weakening old assertions. Both reviewed Python sources compile in memory without bytecode writes. Scoped `git diff --check 346d905 -- <three files>` succeeds. No quality findings remain in this repair.

## Actual execution results

Python: `/Users/shersingh/.hermes/cache/scratch/qmd-dependency-remediation/security-venv/bin/python`, **3.12.13**.

| Execution | Passed | Failed/errors/skipped | Pytest time | Evidence subdirectory |
| --- | ---: | --- | --- | --- |
| Exact unmodified independent suite | 101 | 0 / 0 / 0 | 1.18s | `independent/` |
| Current repository launcher suite | 78 | 0 / 0 / 0 | 0.71s | `repository/` |
| Bounded in-process option/privacy/forwarding held-outs | 42 | 0 / 0 / 0 | 0.42s | `bounded/` |
| Bounded fresh-process malformed-option held-outs | 12 | 0 / 0 / 0 | 0.81s | `bounded-cli-final/` |
| **Final acceptance executions** | **233** | **0 / 0 / 0** | | |

Common actual command shape:

```text
/Users/shersingh/.hermes/cache/scratch/qmd-dependency-remediation/security-venv/bin/python -B -I -m pytest -q -p no:cacheprovider <exact-test-file> --basetemp <evidence-subdirectory>/fixtures --junitxml <evidence-subdirectory>/tests.xml
```

Each subdirectory contains `receipt.json` with the complete actual argv, CWD, explicitly constructed environment, return code, test-file hash and parsed counts, plus `tests.xml`, `stdout.txt`, and `stderr.txt`. The exact test targets are the original scratch `test_independent_launcher.py`, repository `test_locked_job_submission.py`, and the two new scratch `test_secret_option_heldouts.py` / `test_secret_cli_heldouts.py` files. HOME/caches/TMPDIR were synthetic managed scratch paths; bytecode and plugin autoload were disabled; HF/Transformers/Datasets/uv offline flags were set.

### Harness issues, retained rather than hidden

Two early attempts at the new fresh-process guard each produced **12 failed, 0 passed** because the guard incorrectly prohibited non-secret stdlib environment lookups before option validation: first `argparse` → `gettext` locale lookup, then `shutil.get_terminal_size` indexed lookup. Full traces and failed receipts remain in `bounded-cli/` and `bounded-cli-corrected/`; the corresponding guard sources are preserved as `initial_overstrict_cli_probe.py.txt` and `intermediate_overstrict_cli_probe.py.txt`. Only those explicit locale/terminal keys were allowed in the corrected guard. Product/refusal assertions were not relaxed; repository files and the original independent suite were not edited. Final corrected execution is separately recorded as 12 passed in `bounded-cli-final/`.

## Evidence integrity

`initial-snapshot.json` and `final-evidence.json` confirm all three reviewed files and the original independent test file remained byte-identical throughout this review. `reviewed-diff.patch` records the exact scoped diff. XML declared counts were checked against actual testcase/failure/error/skipped elements.

| Reviewed file | SHA-256 |
| --- | --- |
| `finetune/tools/submit_locked_job.py` | `9f7333dec035129ec084fed14a5656552a29ea803e99b90740ec567815e51ae9` |
| `finetune/tests/test_locked_job_submission.py` | `4f37a131b666d723d22428551f354333e5aae4ab30e5fdfd49cb0b60dd6936ab` |
| `finetune/LOCKED-JOBS.md` | `6b798d82f3f2a0624e182c248e3d85355ae031c49c9f751b8751ebdabc452d9b` |

Original independent test SHA-256: `2649aab306cf95573ff9286e9af8dab0796d586ac98398c7a77434beef2def03`.

Prior red evidence was not overwritten or relabeled: `qmd-locked-launcher-independent-review/final-heldout-tests.xml` remains **100 passed / 1 failed**, SHA-256 `62aa83dc5494cff81ce8c5f3dcbdc9e77f689fe8a204f8e20cde6251e6a68dc0`; `qmd-dependency-remediation/parent-secret-refusal-red.xml` remains **6 failed**, SHA-256 `7f4926f0f869ce2437711de719424cd2eb0fb3191f85ea8a96cc77c44e33666a`. The named parent green receipts were also checked unchanged.

## Exclusions and files written

No repository edits, commits, configuration/service changes, private-data reads, authentication, real SDK clients, uploads/jobs, installs, model downloads, GPU use, training, delegation, or Node tests/builds were performed. Only new files and synthetic fixtures under `/Users/shersingh/.hermes/cache/scratch/qmd-locked-secret-reacceptance/` were written.

Concurrent-writer changes to eval bodies, dependency graph verification, and recovery surfaces are excluded. The retained 101-case suite was rerun as requested; the prior broad closure/descriptor review was not restarted. Parent integration remains separate. Cloud scheduling, image execution, credentials/storage permissions, GPU/CUDA, converter/build/later-install surfaces, training and publication remain **unqualified**. This review does not claim arbitrary unknown-argument redaction, process-argv/shell-history secrecy, or an end-to-end runtime lock.
