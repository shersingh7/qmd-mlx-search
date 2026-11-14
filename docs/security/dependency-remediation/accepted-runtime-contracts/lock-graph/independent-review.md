# Independent R2 reacceptance and deferred graph-helper quality review

## SPEC: PASS — R2 closed for the reviewed working-tree sources

The original literal-left wildcard defect is repaired. The production change is
exactly five added lines in `MarkerSpace.parse` at
`scripts/uv_lock_graph.py:226-231`: the original literal-left/variable-right form
is inspected **before** swapping, and any `*` in the literal for either existing
Python variable raises `ValueError`. It does not choose packaging's or uv's
wildcard interpretation, normalize the literal, or change partition, source,
extra, range, graph schema or policy algorithms.

The independently generated `independent-production-delta.patch` is byte-exact:
removing those five lines reproduces the previously reviewed helper bytes.
`verify_patched_dependencies.py` is unchanged from the previous review.

### Fresh independent acceptance

All paths below are under this review directory unless absolute.

- **453 passed + 43 subtests**, zero failures/errors/skips: frozen current
  `test/security`, `finetune/tests/test_script_source_policy.py` and
  `finetune/tests/test_dependency_policy.py`. XML has 453 testcase elements and
  496 declared tests including subtests: `full-targeted.xml`.
- **Original four R2 assertions pass fresh**, including real audit CLI root,
  model and data refusal. Only `BASE`/`ROOT` were relocated to immutable original
  fixtures and this review's fresh source freeze. Behavioral function ASTs are
  identical, and the original test, RED XML, report and installer evidence were
  not changed: `test_original_r2_replay.py`, `replay-relocation.json`,
  `original-and-new-r2.xml`.
- **162 additional independent R2 cases pass**: both Python variables; multiple
  star-bearing literals and operators; alternate quoting, parentheses, OR/dead
  branches; 14 graph contexts including unused optional edges and inactive
  orphans; ordinary numeric-reverse equality/order; variable-left wildcards;
  categorical literal stars remain admitted. `test_r2_heldout.py` compiles actual
  source bytes and asserts their SHA-256 rather than trusting an ambient `.pyc`.
- **51 additional actual audit CLI invocations** use current public inputs for
  root/model/data: 24 literal-left wildcard **refusals even with otherwise
  complete graphs**, 24 numeric-reverse/variable-left-wildcard positives, and
  three complete baselines. Every refusal returns exit 1, `passed: false`, the
  exact unsupported-marker error and **zero proofs**; positives return exit 0
  and 11 proofs. Standalone header/manifest prerequisites are checked with the
  real `_script_lock_metadata`. `r2-cli-receipts-v2.json`.
- All **16 actual graphs** pass both static validation and fresh real uv
  `lock --check --no-build --offline --no-config --python <supplied-3.12>`:
  **931 Python package instances**, with all lock bytes unchanged.
  Five data graphs remain **0 Accelerate / 0 Torch**, with counts
  **5 / 36 / 36 / 5 / 5**. `all-graphs.json`.
- Live repository audit: exit **0**, **1772 instances / 262 retained alerts /
  0 affected / 11 source-binding proofs**. `live-audit.json` and its exact report
  under `cli-reports/`.

## QUALITY: PASS — no blocking findings or required code changes

The deferred review covered the whole **616-line helper**, its binding
integration and proof-publication order, after the fresh specification gate.

### Mathematical and semantic review

1. **Exact finite quotient, not sampling.** Numeric predicates are unions of
   half-open intervals on lexicographically ordered unbounded final Python
   triples. All finite endpoints enter one shared partition, whose last cell
   has no upper cap. `python_version` projects that same axis to major/minor;
   it is not an independent environment coordinate. Categorical equality and
   inequality use every mentioned string plus a cofinite OTHER class. Boolean
   closure therefore preserves predicate constancy on every partition cell.
   Bitset subtraction checks universal coverage within the admitted language;
   finite test environments are corroboration, not its proof.
2. **Admission and partition integrity.** Unsupported predicates fail during
   registration even in impossible/unused contexts. Sealing occurs after all
   declarations, ranges, edges and metadata predicates have been registered.
   Top-level resolution domains must be disjoint and cover the Python domain;
   overlapping or duplicate same-name records fail rather than permitting an
   arbitrary selected record.
3. **Requirements, edges and sources.** `targets` checks the complete relevant
   domain and every compatible version/source selector. Metadata requirements
   additionally need corresponding recorded edges with requested extras.
   Unqualified edges mean the declared source or a registry package, not any
   arbitrary URL/path. Invalid and ambiguous forms fail with generic errors
   rather than reflecting source credentials.
4. **Extras and constraints.** Extras propagate by monotone domain additions
   keyed by record/extra; cycles terminate without dropping later-arriving
   domains. Unknown selected extras refuse. Unselected extras do not force
   packages, although their marker/edge syntax is still parsed. Constraints do
   not create requirements, and apply to intersecting record domains; only
   redundant unconditional overrides are admitted. Project optional/dev
   profiles and every advertised orphan base-edge domain are checked without
   removing records from advisory assessment.
5. **Boundedness and maintenance.** The caller provides bounded, no-follow
   regular-file reads. The helper refuses its seven declared dimensional/input
   budgets, recursion and overflow rather than truncating proof state. The
   private packaging AST dependency is explicit, typed/shape-checked and covered
   by the pinned **packaging 26.3** security toolchain. The separation between
   marker space, graph validation and reviewed-artifact policy remains clear;
   the narrow R2 repair does not broaden approval authority.
6. **Binding order.** Project validation is before project binding collection at
   `verify_patched_dependencies.py:571-573`; every standalone metadata and graph
   check precedes its binding collection at `:588-592`. `_record_proofs` is called
   only after the entire configured closure at `:594`. Root/model/data refusal
   tests verify that no earlier successful graph leaks partial proof keys.

### Fresh quality probes

**52 passed**, zero failures/errors/skips: 29 unchanged prior independent graph
contracts, plus 23 newly written quality contracts. These check every budget,
exact partition-budget admission/refusal boundary, source/edge selector agreement,
conditional constraints, multidomain extra cycles, unknown extras, required edge
extras, generic error text, orphan retention, redundant overrides and unbounded
Python/OTHER cells. Both successful and refused validations preserve input
objects. `test_previous_heldout_contracts.py`, `test_quality_contracts.py`,
`quality-contracts.xml`.

The original R1 omitted-direct-package fixture freshly returns exit 1 with the
missing-coverage error and no proofs: `original-r1-repaired-report.json`.

Canonical tests also freshly exercised uv 0.11.18 with only tiny synthetic
loopback wheels: complete locked execution succeeded; wrong stored hash failed;
an omitted required record still passed uv `--check` but failed locked execution
with exit 2 and was independently refused by the static guard. Receipts are in
`tmp/full-targeted/test_real_uv_{supported_source_,missing_manifest_}0/`.

The old literal-left tiny-installer receipts remain historical, immutable
counterexample evidence. The old frozen-guard probe was **not** rerun and its
former static acceptance was not relabeled as current behavior.

## Source identity and read-only verification

Reviewed HEAD: `dcf98c159b671b4d08c3759215c4371502f49fc0` plus these exact
working-tree bytes, not a claim of committed delivery:

```text
scripts/verify_patched_dependencies.py 3925f9ae6602688848530aa37f59d015c6230b1562a78d7a2ed64f67059706ea
scripts/uv_lock_graph.py e1d133c16346ed0138eaae115d1ab3c98a9b200a1adcb2fdb3cd69acc9f7c596
test/security/test_patched_dependencies.py 5a6b7df95337a05a7393c5576eeeae32b150b71b81711add23cbbffe50515dcb
test/security/test_uv_lock_graph.py 578b385e3940d17b9807793bc4acc9bf11275d64de7de89c1921aa7aa57da432
```

Final readback verifies **4/4 reviewed sources**, **9 other protected inputs**,
**15/15 actual PEP headers**, **15/15 standalone locks**, project manifest/root
lock/policy, **74/74 frozen files**, and **8 original evidence files** unchanged.
Wheel SHA-256 remains
`abfd4ac2b7c632dd0c005e8c928c44112ed67fb3c43d5b320414d7d44b684ad6`.
The preserved original R2 XML still records four failures; the fresh relocated
positive receipt is separate. `git diff --check` passes. Full identities,
commands, per-file counts and assertions are in `review-evidence.json`.

Total fresh pytest receipts: **671 testcase elements / 714 declared tests
including 43 subtests**, with zero failures/errors/skips. This total is distinct
from the separately enumerated 51-invocation public CLI matrix.

Only this owned scratch review directory was written. Execution used supplied
**Python 3.12.13**, **uv 0.11.18** and **packaging 26.3**, isolated HOME/cache/TMPDIR,
owned empty bytecode prefix and `-B`, offline HF/Transformers and disabled Python
downloads. No product runner, model/GPU workload, application installation,
external advisory scan, repository/config/skills/docs edit, staging, commit,
push, or delegation was performed.

### Limits and encountered issue

This is **declared/recorded graph consistency**, not full dependency resolution,
reconstruction/authentication of missing upstream metadata, ordinary registry
artifact authentication, platform-wheel availability, or application/native-CI
qualification. Reads are bounded per-file, not an atomic concurrently changing
repository snapshot. Dimensional budgets are not a global operation-count or
wall-clock SLA; maximum-budget adversarial performance was not qualified.
Parent retains final integration/build, native CI and default-branch closure.

The first extra CLI probe stopped after completing its root cases because the
full current `train.py` body contains another `"gguf"` literal. The revised
probe alters only the PEP header, preserves script bodies, and passed a complete
fresh v2 matrix. Earlier outputs remain separate; no product inputs or acceptance
assertions were weakened.
