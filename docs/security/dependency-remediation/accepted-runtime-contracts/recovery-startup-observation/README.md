# B: bounded after-canonical startup observation

## Accepted scope, not CI closure

The independently reviewed three-path proposal adds `test/helpers/recovery-startup-observation.ts`, its permanent selftests, and after-canonical macOS workflow steps. Its exact postimages passed **SPEC, then full three-path QUALITY**, with no blocking finding. Parent intake authenticated all **557** review artifacts before applying the unchanged proposal. Production recovery, loader, lease, anchor and worker code were not changed.

The supplemental probe records the selected Python process's own entry/module/admission observations and bounded raw sidecars for actual Node and native Bun callers. It uses the original caller-inclusive **200 ms** budget plus a separately labelled **5000 ms positive control**. It runs only after the existing canonical suite and package smoke. Neither diagnostic success nor the longer control changes the canonical test result, warms its original launch, enlarges its deadline, or certifies contained cleanup for injected commands.

## Two rejected-proposal defects now closed

- **B-SPEC-1 — observer lifetime:** release call-owned callbacks on delayed rejection, synchronous failure, final-snapshot failure and ordinary completion without losing primary failures or removing caller-owned listeners. The original reproducer changed from 13 retained callbacks to zero; its assertions were unchanged. Independent and parent held-outs exercised six lifecycle modes, including rejection after an asynchronous wait and a secondary disposal error.
- **B-SPEC-2 — raw byte validation:** reject malformed UTF-8 at complete-record boundaries before decoding can turn corrupt bytes into valid-looking JSON. Preserve the validated prefix, actual raw byte counts and unparsed tail; keep valid literal/escaped U+FFFD and multibyte text accepted. Both actual runtimes passed 33 additional byte/schema cases, including size boundaries and unsafe filesystem-object rejection. Node also exercised inode-mismatch refusal and byte-split short reads with matched opens/closes. Bun's named-FS fault-injection seam is unsupported and **not qualified**.

The archive retains the original REDs, unchanged reproducers, repaired-source GREENs, review order, exact patch bindings and parent executions. These are source-bound proofs, not a claim that every possible runtime failure is fixed.

## Fresh parent integration checks

- Canonical `pnpm run test`: **1524 passed, 66 skipped, 54 files**; raw reporter rows reconcile with the declared totals.
- Canonical `pnpm run build`: exit 0, **610** guarded compiler records, zero refusals.
- Whole-root strict checks: **156 roots** under actual Node and actual Bun, zero diagnostics or guard refusals.
- Startup selftests: **14 passed**; original observer selftests: **8 passed**.
- Four real local supplemental comparisons: Node and native Bun at 200/5000 ms. Each observed authored public work, retained `productQualified=false`, and reported no subject PID remaining before fixture cleanup. These local instrumented runs do not establish cold or historical CI causality.
- Parent replay of six lifecycle modes and 33 byte/schema cases per runtime passed. Actionlint passed with shellcheck/pyflakes explicitly disabled, not qualified.
- Installed-package evidence is **carried forward, not rerun**: all **130 regular entries** of the previously exercised Node/native-Bun tarball match the current build byte-for-byte. These three test/workflow files are not package entries. Tarball SHA-256: `ce36bcaca2037d7ae1c6e0a63e71260554e6c9e0a837158f24821b4fe69ad3be`.

The reviewer lane's auxiliary canonical-workspace notice remains unverified and is not rewritten. The parent separately ran the recognized canonical test/build commands above; those receipts, not a fabricated reviewer gate, support this integration.

## Remaining gates and limits

Actual macOS CI acceptance remains open. Run `43605300047` at preceding head `fd9b9025b496ad7a3a8e2f9ca1d0ffa68c289c07` retained 55 recovery and two transport test failures. The newly added probe had not run there. Reconcile its later CI artifacts against the event SHA, actual checkout and exact helper/production bytes; do not retroactively attribute their interpreter or timings to an earlier canonical invocation.

The paused sole-owner C1 limitation, historical incomplete/forced-close receipts, and separate full-native-Bun Vitest allocator SIGTRAP remain disclosed. Parent-side PID/fstat observations do not identify kernel resource holders. Merge and default-branch dependency-alert closure are separate gates.
