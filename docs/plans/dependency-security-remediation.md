# Dependency remediation and main-branch merge

## Authorized scope

Review the audited branch, repair dependency vulnerabilities, verify the combined result and merge it into the repository's actual default branch (`main`). Do not deploy QMD, modify personal indexes, publish registry packages, launch training jobs or download model weights. Keep the November 11 acceptance evidence immutable: it describes an earlier dependency snapshot, not this new release.

## Inventory and count reconciliation

The push banner reported 261 open alerts (17 critical). Fresh paginated REST enumeration on November 13 returned **260 unique open alerts**, corroborated by an independent all-state enumeration: 276 historical records = 260 open, 14 fixed and 2 auto-dismissed. Both enumerations retain every raw record. The missing banner item is not credited as a fix; the exact reason for the discrepancy is not established.

Current open records: 17 critical, 92 high, 121 medium, 30 low. Manifests: `pnpm-lock.yaml` (101), `package.json` (2), `finetune/uv.lock` (157). Raw source of truth and per-alert ledger: `docs/security/dependency-remediation/`.

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

## Acceptance

- Every originally enumerated alert is accounted for by ID and resolved dependency version/path; no affected resolved instance remains in the candidate graphs.
- Fresh scans and regression guards pass, including duplicate/version/marker cases and the two advisory records initially lacking first-patched metadata.
- Current-head required/relevant CI passes before merge, the actual remote default branch contains the reviewed commit, and GitHub returns zero open Dependabot alerts after graph refresh.
- No live service, index, credentials, scheduler or registry release was changed. No paid training or model-weight download was performed.
- Deliver the PR/merge handles, per-alert closure evidence, final tests and any expressly unqualified surfaces. Claims distinguish source testing, installed runtimes, synthetic orchestration and actual model/hardware behavior.
