# Dependency remediation and main-branch merge

## Authorized scope

Review the audited branch, repair dependency vulnerabilities, verify the combined result and merge it into the repository's actual default branch (`main`). Do not deploy QMD, modify personal indexes, publish registry packages, launch training jobs or download model weights. Keep the November 11 acceptance evidence immutable: it describes an earlier dependency snapshot, not this new release.

## Inventory and count reconciliation

The push banner reported 261 open alerts (17 critical). Fresh paginated REST enumeration on November 13 returned **260 unique open alerts**, corroborated by an independent all-state enumeration: 276 historical records = 260 open, 14 fixed and 2 auto-dismissed. Both enumerations retain every raw record. The missing banner item is not credited as a fix; the exact reason for the discrepancy is not established.

Initial open records: 17 critical, 92 high, 121 medium, 30 low. Manifests: `pnpm-lock.yaml` (101), `package.json` (2), `finetune/uv.lock` (157). Raw source of truth and per-alert ledger: `docs/security/dependency-remediation/`.

A later fresh REST enumeration and GraphQL total both returned **262** open alerts: 17 critical, 93 high, 122 medium, 30 low. The subsequent push banner also agreed with 262. Newly enumerated IDs 292 and 293 are Pillow advisories; the candidate resolves Pillow 13.0.0. Keep the initial snapshot immutable, and use `alerts-current.json` plus `alerts-current-summary.json` for the expanded gate. No alert was dismissed to reconcile these counts.

## Workstreams and ownership

| Lane | Exclusive write scope | Contract |
|---|---|---|
| JavaScript dependency repair | `package.json`, `bun.lock`, `pnpm-lock.yaml`, new `test/dependency-locks.test.ts` | Supported Node/Bun runtime compatibility; regenerate both locks; no known affected resolved versions; no blind incompatible overrides. Keep lifecycle-script approval narrow. |
| Fine-tuning dependency repair | `finetune/**` only | Preserve actual training/evaluation functionality, update real dependencies and lock graph, review script-local dependency declarations. No deleting lockfiles or features merely to hide alerts. No paid/GPU training or model downloads. |
| Independent integration review | Read only | Review the audited merge candidate and security/portability boundaries. Return reproduced important defects rather than speculative new features. |
| Parent integration and security gates | New security audit/guard scripts/tests, `.github/workflows/**`, documentation/ledger | Reconcile every alert with actual resolved versions, verify both ecosystems, add regression/CI checks, run settled-source suites and installed package checks, collect independent acceptance and merge. |

Shared manifest changes have one owner. Model/GPU checks remain serialized in the parent. Use managed scratch explicitly on every subprocess, strip ambient model/application settings, and never read live knowledge/index data.

## Execution plan

1. Export and deduplicate all alert pages; retain original identifiers, ranges and names. Build one ledger row for every open alert, preserving original package spelling and a separately derived ecosystem-canonical lookup key.
2. Inspect dependency provenance and patched release metadata. Prioritize all critical/high cases, but close medium/low cases too. Advisory records without a first-patched value require explicit upstream/range investigation, not assumptions.
3. Upgrade or replace affected dependencies with compatible supported releases. Regenerate the real resolver outputs, check lock freshness and inspect every resolved instance, not just direct versions.
4. Validate Node transport/native addons and fine-tuning API compatibility with bounded public/synthetic fixtures. Record unexercised GPU training separately from model-free API/lock verification.
5. Add offline baseline-advisory regression coverage and fresh online ecosystem scans. Do not dismiss alerts, disable scanners, delete dependency manifests or weaken tests to obtain a zero count.
6. Run full TypeScript/Python suites, source build, native installed-package smoke checks, meaningful real-Metal/reference checks when affected, and independent spec/quality review on settled bytes. Preserve old negative receipts.
7. Push the candidate and open/read back a PR targeting `main`; inspect exact-head GitHub Actions. Repair failures, distinguishing real cross-platform defects from infrastructure failures. Do not bypass failures or merge only because the branch is unprotected.
8. Merge only the reviewed and tested head. Verify the remote `main` commit and post-merge checks. Re-fetch every Dependabot page until GitHub's default-branch graph is reconciled; compare original IDs and new alerts. If upstream has no usable fix or access/CI blocks completion, report the exact blocker rather than dismissing or inventing closure.

## Source-aware patch and cross-platform qualification

1. **Do not confuse scanner clearance with source repair.** Source inspection found official Accelerate 1.15.0 still vulnerable to GHSA-4j2p-28q2-5m79 even though the advisory stops at 1.14.0. Retain that blocker until a reviewed, hash-bound artifact is actually selected. Transformers GHSA-x9r9-c232-4q39 uses a source-verified safe floor of 5.17.0 rather than trusting the advisory's 5.8.1 upper bound. Evidence: `no-first-patched-source-review.md`.
2. **Use a reproducible controlled dependency, not a version-label workaround.** Review the exact-context patch, installed package aliases, regular-file capabilities, path/race checks, HF cache compatibility and unchanged tensor/offload behavior. Preserve upstream license and original wheel. The custom wheel is a local patch, not an upstream advisory closure.
3. **Validate both native filesystem implementations.** Run actual wheel-installed CPU Torch/safetensors cases on macOS and Linux, including negative controls, malformed indexes, before-open interception, FD cleanup and race probes. A Mac pass does not qualify Linux's O_PATH/procfs implementation. Retain real failures; repair the cause and rerun rather than skip tests or infer a pass.
4. **Bind every actual resolver path.** After independent review and native platform acceptance, pin the exact local wheel in `finetune/pyproject.toml`/`uv.lock`. Synchronize all **15** dependency-bearing PEP 723 runners to the actual immutable published artifact URL plus SHA256. Every runner gets the same source/constraint policy; lightweight data-only scripts must not gain an unnecessary Torch installation. Actual uv metadata-only probes demonstrated both transitive source selection and absence of unused Accelerate/Torch in lightweight scripts.
5. **Verify bindings, not just versions.** The release gate must match the wheel bytes, metadata, local source, uv wheel hash, immutable URL and complete runner inventory before granting the one advisory-specific mitigation. No manually supplied version tag, missing source, stale script block or additional unverified resolved instance is accepted.
6. **Keep live scans fail-closed.** Validate actual npm bulk and OSV response schemas, duplicate keys, error-field presence, pagination and bounded response bodies. Bun 1.3.8 exit 0 is not evidence of a valid audit response. Retain real HTTP malformed-response regressions and independent live-provider controls.
7. **Complete installed Nix checks.** Derive both platform hashes from real native CI outputs for the exact settled package/lock graph. Keep integrity validation enabled. Both native builds must pass compiled CLI/SDK, SQLite/vector, WASM, recursive helper inventory and disposable HTTP retrieval checks.
8. **Review all combined changes again before merge.** Include fine-tuning API/CLI compatibility, recovery-tree timing/ownership coverage, source-binding verification and the final exact-head CI. Publish only to the draft feature branch while these gates remain open; a published candidate wheel is not production acceptance.

## Acceptance

- Every originally enumerated alert is accounted for by ID and resolved dependency version/path; no affected resolved instance remains in the candidate graphs.
- Fresh scans and regression guards pass, including duplicate/version/marker cases and the two advisory records initially lacking first-patched metadata.
- Current-head required/relevant CI passes before merge, the actual remote default branch contains the reviewed commit, and GitHub returns zero open Dependabot alerts after graph refresh.
- No live service, index, credentials, scheduler or registry release was changed. No paid training or model-weight download was performed.
- Deliver the PR/merge handles, per-alert closure evidence, final tests and any expressly unqualified surfaces. Claims distinguish source testing, installed runtimes, synthetic orchestration and actual model/hardware behavior.
