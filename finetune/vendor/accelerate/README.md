# Controlled Accelerate checkpoint security patch

**Artifact:** `accelerate-1.15.0+qmd.1-py3-none-any.whl`

**SHA256:** `9b778265a3952140fa4b64b97a022b11f288bd56627b64e71c7fd556dabdeadd`

This is a controlled local build, **not** an official Hugging Face release or an
upstream advisory closure. The official Accelerate 1.15.0 wheel still contains
GHSA-4j2p-28q2-5m79 / CVE-2026-69112. The local version identifies an actual
source fix; changing the version alone is not the mitigation. No upstream fix or
merged PR is claimed.

## Contents and provenance

- `upstream/accelerate-1.15.0-py3-none-any.whl`: original official PyPI input.
  SHA256 `97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec`.
- `source.patch`: exact-context unified source patch for
  `accelerate/__init__.py` and `accelerate/utils/modeling.py`.
- `checkpoint_security.py`: added package module, installed as
  `accelerate/utils/checkpoint_security.py` under Apache-2.0.
- `../../tools/build_patched_accelerate.py`: standard-library-only builder.
- `provenance.json`: input hashes, exact upstream HTTPS URL, output wheel hash,
  license preservation, and hashes of every wheel member.
- `LICENSE`: original Apache-2.0 license, also preserved byte-for-byte at the
  original license member's new versioned dist-info location inside the wheel.
- `evidence/`: actual installed-wheel JUnit results and machine-readable runtime
  and integrity evidence. These are local execution results, not cloud results.

The builder refuses an input that does not match the pinned upstream SHA256. It
applies strict patch context, adds the module and a patch notice, updates the
local version and wheel RECORD, and leaves other upstream source byte-identical.
The wheel uses sorted ZIP_STORED members, fixed 1980 timestamps and Unix 0644
attributes; there is no clock, build backend, compiler or compression-version
input. Two independent output-directory builds compare byte-for-byte, including
provenance, in the tests.

From the repository root, an offline rebuild is:

```sh
env HOME=/Users/shersingh/.hermes/cache/scratch/qmd-accelerate-home \
  TMPDIR=/Users/shersingh/.hermes/cache/scratch PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 finetune/tools/build_patched_accelerate.py
```

`--output-dir DIRECTORY` builds elsewhere. `--upstream FILE` accepts only the
same SHA256-pinned input. Optional `--download-upstream` retrieves the exact
immutable PyPI artifact URL and verifies its hash; no download is needed when
using the included input wheel.

## Security boundary

Both `load_checkpoint_in_model` and `load_checkpoint_and_dispatch` use the same
patched implementation inside the installed package. Imported aliases held by
`accelerate`, `accelerate.utils`, `accelerate.big_modeling` and downstream code
therefore do not bypass a runtime wrapper or depend on startup ordering.

1. Directory and explicit-index input paths are supported. Discovery preserves
   upstream whole-model-file precedence and the legacy flat index mapping.
2. Indexes must be nonempty JSON objects with a nonempty mapping of nonempty
   weight-name strings to nonempty shard-name strings. Wrapped `weight_map`
   indexes may have object-valued `metadata`. Duplicate JSON keys, malformed
   JSON, NULs and mixed/invalid shard types fail before tensor deserialization.
3. Shard paths must be relative POSIX components. Empty/dot/dot-dot components,
   absolute paths, Windows drive/UNC/root forms, backslashes and colon/alternate
   stream forms are rejected rather than silently normalized.
4. All discovered shards are preflighted in each reading process before its
   first shard data open. Missing
   files, broken links, FIFOs, sockets, devices and directories are rejected.
   This check also covers whole-model files and the index file itself.
5. **Preflight is not the race guarantee.** A directory capability is retained
   for the load. Walks use descriptor-relative metadata operations and
   `O_DIRECTORY | O_NOFOLLOW` for intermediate components. Symlinks are read and
   validated explicitly; an untrusted resolved pathname is never data-opened.
   Root identity is checked while establishing the initial capability.
6. The final `load_state_dict` boundary independently resolves and pins the
   actual regular-file inode, including when that API is called directly. Torch
   receives a held binary file object and retains `weights_only=True`.
   Safetensors receives an FD-backed pathname and retains mmap/direct-device
   loading. Path replacement after pinning cannot redirect those reads.

### Deliberate Hugging Face cache compatibility

A symlink resolving inside the checkpoint root is supported. An escape is
supported **only** for a positively qualified snapshot under the operator's
configured `huggingface_hub.constants.HF_HUB_CACHE`, with layout
`models--.../snapshots/<40-or-64-lowercase-hex-revision>/...`, and only to a direct
`blobs/<40-or-64-lowercase-hex-name>` file of the **same** model repository.
Nested checkpoint subfolders and symlinked indexes are covered.

A directory that merely looks like a cache does not gain this permission. Repo
and blob directories are walked without following links, and the final blob
must itself be a regular non-symlink file. Links to other repositories, arbitrary
external files, symlinked blob directories and symlinked blob leaves are denied.
This is a trusted-root containment rule, **not** content-digest authentication.
A custom cache root must be explicitly configured before importing HF libraries;
this patch never infers permission from an arbitrary snapshot-looking path.

### Platforms and limits

- **Linux implementation:** `O_PATH | O_NOFOLLOW` pins a metadata-only handle;
  `fstat` verifies the actual inode before its first data open through
  `/proc/self/fd`. It requires usable procfs FD links. This branch has **not been
  executed in this macOS environment**: the installed Docker client could not
  connect to a daemon. Run this same suite on Linux before Linux deployment.
- **macOS, executed:** there is no `O_PATH` equivalent providing the same device
  safety. A private mode-0700 temporary directory receives a descriptor-relative
  **hardlink** to the candidate. The pinned staging entry is checked before any
  data open, then opened, unlinked and removed while its FD remains held.
  A regular-to-FIFO/device/symlink replacement cannot become an unchecked data
  open. `O_NONBLOCK` by itself is explicitly not relied upon for this guarantee.
- macOS requires trusted `TMPDIR` storage on the checkpoint's filesystem and
  permission to hardlink the file. Cross-device, read-only-filesystem, immutable
  file or permission failures fail closed with an actionable error; there is no
  unsafe open/copy fallback. This can reject otherwise readable checkpoints on
  such filesystems. Use a compatible trusted staging location or Linux.
- Neither strategy copies a whole shard for security staging. Safetensors mmap
  through the held FD is exercised with buffered reads forbidden. Torch still
  performs its normal CPU deserialization/tensor allocations.
- Other platforms (including Windows/BSD) fail closed; this wheel's universal
  Python tag is not a promise of filesystem support everywhere.
- The caller's initially selected checkpoint root and configured cache root are
  trusted authorities. This is not protection against a hostile same-UID process
  that can change the loader, its FDs, or private staging permissions, nor against
  privileged mount manipulation. An already-authorized inode can still be
  modified/truncated in place by a writer; FD pinning prevents name redirection,
  not corruption, SIGBUS/DoS or inconsistent tensor bytes. Use immutable,
  integrity-verified artifact storage when that threat matters.
- This does not sandbox pickle/remote Python code or fix independent checkpoint
  loaders in Transformers, PEFT, TRL or other packages. GPU/real distributed
  execution has not been tested. Non-reading rank-0-broadcast paths retain
  index-only compatibility: schema/path syntax is always checked, but all-shard
  filesystem preflight is deferred until this process actually reads a shard.
  A CPU branch-contract test stubs the final Torch collective; it is not a real
  multi-process distributed test.

## Executed validation

Environment: macOS arm64; Python 3.12.13, Torch 2.15.0, safetensors 0.8.0,
Transformers 5.22.2, PEFT 0.21.3, TRL 1.17.1. See runtime evidence for the exact
installed path and artifact/source hashes.

- **Unmodified installed wheel:** 48 negative-proof probes passed. Both APIs,
  all retained public aliases, directory/direct-index inputs, traversal, absolute
  paths, FIFO and directory shards reach an intercepted loader. All fixtures are
  owned synthetic files; no real private file or blocking FIFO is opened.
- **Patched installed wheel:** 1139 tests passed, 48 baseline-only cases skipped.
  The package is actually wheel-installed in the isolated finetune environment;
  tests compare installed source hashes to the wheel/provenance. There is no
  repository-local validator import or application guard standing in for it.
- Cases cover invalid schemas, special files, path/foreign-root forms,
  prefix-siblings, external symlinks, valid/nested/flat-index checkpoints,
  approved and forged caches, final-load bypass attempts, aliases, FD cleanup,
  and mutation before/after preflight, during directory/symlink resolution,
  at atomic pinning and after pinning. Rejected reads are intercepted before
  syscalls/deserializers, so a regression cannot open a FIFO and hang the test.
- Real tiny CPU Torch and safetensors weights are loaded with mapped/unmapped
  devices, dtype conversion and CPU state-dict offload. Original deserializer
  exceptions remain original exceptions. An AST check confirms that upstream's
  complete model/device/offload loop is unchanged inside the new root context.
  Character/block-device cases use
  synthetic stat metadata for owned files: no privileged device creation or
  real host-device access was performed.
- Actual PEFT adapter save/reload and TRL SFTTrainer construction execute against
  this installed Accelerate. Both guarded APIs then load/reject checkpoints on
  their real model objects using aliases captured before transitive imports.
  Runtime tracing confirms these PEFT/TRL versions **do not themselves call**
  either vulnerable checkpoint API in those operations; no invented indirect
  exploit route or security claim for their independent adapter loaders is made.
  No training step, model download, GPU or cloud job was run by this suite.
- A reproducibility test rebuilds the wheel in a separate output directory and
  compares both wheel and provenance bytes. A wrong upstream hash is refused.

Re-run patched integration using the already-discovered isolated environment:

```sh
env HOME=/Users/shersingh/.hermes/cache/scratch/qmd-accelerate-home \
  TMPDIR=/Users/shersingh/.hermes/cache/scratch PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= \
  ACCELERATE_USE_CPU=true \
  /Users/shersingh/.hermes/cache/scratch/qmd-finetune-venv/bin/python -m pytest \
  -q -p no:cacheprovider finetune/tests/test_accelerate_checkpoint_security.py
```

The included unmodified wheel can be installed into a separate `--target`
`site-packages` directory with no dependencies. Prepend that directory via
process-only `PYTHONPATH`, set `QMD_ACCELERATE_BASELINE=1`, and run the same test
file with `-k baseline`. Never run baseline probes by weakening the patched
loader or by pointing at private files. The recorded baseline used
`/Users/shersingh/.hermes/cache/scratch/qmd-accelerate-baseline/site-packages`.

## Required integration/publication gate — NOT completed here

This stage creates the controlled artifact, patch, builder and tests only. It
**does not change manifests, locks or PEP 723 job headers**, does not publish a
commit, and does not claim isolated/cloud runners are fixed.

The parent must independently review this patch, validate Linux, commit/push the
reviewed artifact on the authorized feature branch, and only then use that
**actual immutable commit's raw artifact URL plus this wheel's SHA256** in every
independently resolved PEP 723 runner. Do not invent a commit URL. Pin the exact
local version and a local wheel source in the finetune uv project, regenerate
its lock, and exercise fresh installed environments and actual launch paths.
A registry `accelerate>=1.15.0` selection alone remains vulnerable and must not be
used as proof of remediation. Keep the upstream-advisory limitation visible.
