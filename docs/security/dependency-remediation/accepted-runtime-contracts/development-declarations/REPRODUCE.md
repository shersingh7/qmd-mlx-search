# Reproduction / inspection notes

This is a completed **independent review**, not an implementation delivery. Do not execute helpers in the immutable producer root, the predecessor root, an existing reviewer root, or a previous reproduction root.

## Evidence inspection

- Read `REPORT.md`, then `evidence/spec-verdict.json` and `evidence/quality-verdict.json`.
- `evidence/verified-raw-commands.json` lists the exact 33 executed sandboxed argv, per-command receipt hashes and stdout/stderr hashes.
- Each `evidence/<label>.json` contains the complete actual child environment, cwd, argv, exit code and bound OS sandbox profile.
- `logs/<label>.stdout.log` and `.stderr.log` are the real command output, including expected strict failures and eight independent negative controls.
- `evidence/diagnostic-comparison.json` retains all diagnostics, the four removed errors, unchanged roots/options, actual runtime identity and two historical read refusals per strict command.
- `evidence/type-binding-summary.json` and full resolution bindings retain declaration-file paths/hashes and non-any global Bun binding on both hosts.
- `protected-input-{before,after}.json` and `shared-{before,after}.json` are full byte/stat preservation records. `dependency-inventory-policy.json` states the explicit AGENTS.md metadata-only exception.
- `evidence/review-index.json` binds public reviewer artifacts, exact sandbox profiles, explicit source/generated/installed manifests and the producer handoff anchors. The index excludes itself and private operational homes/caches.

## Fresh owned replay

The inspected reviewer harness is transparent Python/JavaScript, not an external coding harness. To repeat, first read `probes/review.py`, `quality-review.py`, `graph-review.mjs`, `verify-owned.py`, `safety-probe.cjs` and `readguard-control.cjs`. Make a **new** `qmd-declarations-review-*` root under `/Users/shersingh/.hermes/cache/scratch`; copy these six helpers into its `probes/` directory without modifying the originals. `review.py` derives its write root only from its own location.

Launch the specified Python 3.12.13 with `-I`, a clean `env -i` allowlist, and synthetic HOME/Hermes/knowledge/XDG/HF/temp/cache paths inside the new root, before imports. Retain the offline HF/uv flags, `QMD_LLAMA_GPU=false`, `CI=1`, `REQUIRE_BUN=1` and `PYTHONDONTWRITEBYTECODE=1`. The three phases are:

1. `python -I <NEW>/probes/review.py preflight`
2. `python -I <NEW>/probes/review.py execute`
3. `python -I <NEW>/probes/review.py preserve`

Only after SPEC PASS run `python -I <NEW>/probes/quality-review.py`.

All actual tested subprocess environments and commands are reconstructed explicitly by the owned `cleanenv`/`run` functions. The preflight authenticates the supplied immutable producer index and all 263 indexed artifacts, binds all 251 source preimages, rejects shared-dependency drift, supplies real owned caches, and reads only the three already archived declaration payloads. It never installs an application/native package or accesses the registry. It does not run producer bootstrap/offline-replay helpers, whose path-derived writes and broad AGENTS content hashing are unnecessary for this review.

Keep genuine source/readguard/sandbox setup errors separate from product errors. Expected strict exits are 2 (35 -> 31 diagnostics), not 0. Expected stale-manager/verification controls are 1 at their specific rejection checks. The same 153 roots/options and two denied historical content reads must remain. No full-root pass is claimed.
