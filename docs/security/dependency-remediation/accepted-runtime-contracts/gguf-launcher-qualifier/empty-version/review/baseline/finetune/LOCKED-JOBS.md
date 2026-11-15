# Lock-qualified Hugging Face Jobs

Use `tools/submit_locked_job.py` for the lock-qualified submission route. It is
**dry-run by default**: it performs local preflight, builds a temporary staging
tree, prints a JSON file/hash/command manifest, and removes the temporary tree.
It does not import the HF SDK, authenticate, upload, resolve/install packages,
download models, or start training in this mode. Python 3.11+ on macOS/Linux is
sufficient for preflight; `--submit` additionally needs the project's `huggingface_hub` client
with the public Jobs/Buckets APIs (2.4+).

## Why the plain command is not sufficient

Existing examples such as `hf jobs uv run jobs/sft.py` describe the underlying HF
Jobs capability, **not a lock-qualified dependency route**. In the HF Hub 2.4
implementation, `run_uv_job` discovers files only from the script and existing
file-valued script arguments. It neither automatically carries the adjacent
`jobs/sft.py.lock` nor adds `uv run --locked`; it also flattens uploaded filenames.
A PEP 723 header alone does not establish the reviewed resolved graph.

The reviewed Accelerate publication URL retains a `#sha256=...` fragment in
`docs/security/dependency-remediation/patched-dependencies.json`. The runnable
`tool.uv.sources` URL has **no fragment**: uv 0.11.18 rejects a fragment there,
and a URL fragment passed to `uv run` is not sufficient hash enforcement.
The adjacent script lock's wheel `hash = "sha256:..."`, together with
`uv run --locked`, supplies the artifact hash enforcement. Do not drop the lock
or replace `--locked` with `--frozen`, `--no-sources`, or an unlocked invocation.

## Preview first; submit only by explicit opt-in

Run from the checkout (the launcher itself does not depend on your current
working directory). The namespace, hardware flavor, and deadline are required;
there is no inferred identity or default paid hardware. Replace placeholders
with your intended namespace/resources; these examples have not been submitted.

```sh
# Offline/local preview: no credentials are read and no cloud job is created.
python3 finetune/tools/submit_locked_job.py \
  --runner finetune/jobs/sft.py \
  --namespace YOUR_NAMESPACE --flavor a10g-large --timeout 2h \
  -- --dry-run

# Explicitly enable private uploads and paid compute. The final --dry-run is
# still the SCRIPT's dry-run flag, not a substitute for the launcher's preview.
# Provide HF_TOKEN securely in your local environment; never put its value in
# chat, --env, a script argument, or a command-line literal.
python3 finetune/tools/submit_locked_job.py \
  --runner finetune/jobs/sft.py \
  --namespace YOUR_NAMESPACE --flavor a10g-large --timeout 2h \
  --secret-env HF_TOKEN --submit \
  -- --dry-run
```

To intentionally train, review the manifest and the runner's own side effects,
then remove the **script** `--dry-run` argument. Selecting `--submit` can incur
charges even if a script immediately exits. Existing runners can download models
and data and publish models/evaluations after they start. This launcher does not
make those runtime actions offline or harmless. Use the existing HF Jobs
inspect/log/cancel commands with the returned job ID. If readback fails after job
creation, the error names the created ID: inspect/cancel it before retrying to
avoid duplicate paid jobs. A successful submission is not a successful training
run.

Everything after the first exact `--` is forwarded in its original order and
spelling, including subsequent `--`, `--help`, spaces, and glob literals. No
argument is inspected for local-file existence or automatically uploaded. Quote
wildcards and shell metacharacters in your own shell if they are intended as
literal arguments.

## Exactly what can leave the machine

Before any SDK/network operation, the launcher validates the root publication
policy, selected allowlisted runner, required adjacent locks, script source and
constraint binding, and the locked Accelerate wheel URL/version/SHA-256 wherever
that package occurs. Missing or mismatched locks are blocking inputs, not an
invitation to silently regenerate dependencies. Data-only graphs are not forced
to install Accelerate. uv performs the full graph freshness check with
`--locked` in the job; local static preflight is not a replacement for that check
or for source review.

The literal `CODE_CLOSURE` in the launcher is the only automatic upload list:

- The selected policy-listed script and its adjacent `.py.lock`.
- Dataset tools: `dataset/schema.py`; scoring also includes `reward.py`.
- `eval.py`: `reward.py` and the public `evals/queries.txt` asset.
- Training entrypoints: sibling `eval.py` and its lock, `reward.py`, and queries,
  preserving post-training evaluation and its relative `__file__` paths.
- `train_unsloth.py` also includes `train.py` and its lock because its GGUF
  publication path imports the shared exporter from that module.
- `convert_gguf.py`, `train.py`, and `train_unsloth.py` include the shared
  `gguf_toolchain.py` prerequisite/export helper, with its exact bytes and hash in
  the preview. Missing or unsafe helper files block preflight before SDK access.
- `jobs/sft.py` and GRPO: the reviewed `jobs/eval_common.py`, avoiding the
  standalone helper's fallback download.

Other self-contained runners need only their script and lock. Repository-relative
subdirectories are preserved; two `eval.py` basenames are never flattened or
silently renamed. No checkout-wide glob, directory upload, Git metadata,
`pyproject.toml`, project `uv.lock`, vendor wheel, cache, local model, output
folder, configuration folder, or live dataset is automatically included.
Unlisted runners are refused rather than assigned a guessed dependency closure.

### Optional, explicitly consented local inputs

Each input must be a single absolute, regular, nonsymlink file, with a literal
relative destination **under the remote `finetune/` directory**. Directories,
globs, traversal, hardlinks, symlinks in any ancestor, duplicate/overlapping
destinations, code/uv-config destinations, and common credential paths/names are
refused. Inputs are limited to data/config/text suffixes, 64 MiB each, and the
whole staged set to 128 MiB. Larger data and model trees belong in explicitly
supplied remote volumes, not a recursive local upload.

```sh
python3 finetune/tools/submit_locked_job.py \
  --runner finetune/dataset/score_data.py \
  --namespace YOUR_NAMESPACE --flavor cpu-basic --timeout 10m \
  --input /absolute/path/reviewed-examples.jsonl=data/examples.jsonl \
  --allow-input-upload \
  -- finetune/data/examples.jsonl
```

In that example the file is staged as `finetune/data/examples.jsonl`; the script
argument is also `finetune/data/examples.jsonl` because `score_data.py` resolves
relative paths from the repository root. Use `/inputs/...` only
for an explicitly mounted remote volume, as below. Other scripts commonly
resolve paths from the working directory, `finetune/`, so they use
`data/examples.jsonl`. Arguments are deliberately **not rewritten**. Review
individual scripts' path conventions and manifests before submission.

Use `--input /absolute/reviewed-config.yaml=configs/sft.yaml` with the same
consent flag for a local training configuration; no configs/data are inferred
from its contents. An explicit input may itself be private: consent means you
have reviewed and authorized every byte to be uploaded. Filename checks cannot
determine an arbitrary file's sensitivity; do not supply credentials or caches
under innocent names. Input contents are not printed in the manifest.

### Explicit remote resources and secrets

`--env NAME=VALUE` adds a **non-secret** remote environment value.
`--secret-env NAME` forwards just that explicitly selected local environment
variable as an SDK secret **only on `--submit`**; previews show names, never read
or print their values. There is no wholesale forwarding of your environment.
uv/Python/path-injection environment overrides are refused. `--resource-group-id`
is optional and explicit.

Remote volumes are never discovered from local paths. Specify each one:

```sh
python3 finetune/tools/submit_locked_job.py \
  --runner finetune/dataset/score_data.py \
  --namespace YOUR_NAMESPACE --flavor cpu-basic --timeout 10m \
  --volume dataset=YOUR_NAMESPACE/reviewed-data=/inputs/data \
  --writable-volume bucket=YOUR_NAMESPACE/results=/outputs/results \
  -- /inputs/data/examples.jsonl
```

Supported volume types are `bucket`, `dataset`, `model`, and `space`. Mounts must
be under `/inputs` or `/outputs`; overlapping mounts and reserved code/runtime
paths are refused. `--volume` sets `read_only=True`; only
`--writable-volume` permits remote writes. Without a writable volume or the
runner's own publication mechanism, runtime outputs in the container are
**ephemeral**.

## Submission implementation and limits

The opt-in path uses only public SDK APIs: create/check the private
`NAMESPACE/jobs-artifacts` bucket, `sync_job_volume` the isolated staging tree to
a fresh opaque subfolder, read back its exact file paths/sizes, `run_job`, then
`inspect_job` to verify the job ID, image, flavor, and exact command. An existing
public artifacts bucket is a hard refusal, not the SDK's default warning-and-
continue behavior. Staging is mounted read-only at `/qmd-code`. A fixed shell
wrapper copies it into a writable temporary worktree, changes to `finetune/`,
and executes these separate argv elements without shell-interpolating script
arguments:

```text
uv run --locked --no-project --script jobs/sft.py <original script args>
```

`UV_NO_CONFIG=1` and `--no-project` keep unstaged project/user configuration out
of script resolution. Both code and locks retain their relative paths. Nested
evaluation subprocesses reuse the selected script's Python environment rather
than performing an additional unlocked resolution. Bucket readback checks
paths/sizes, not an independent re-download/SHA-256 of remote bytes; the manifest
contains SHA-256 of the exact local staged bytes. Remote bucket visibility is
not transactionally locked against a separate owner's concurrent changes.
Uploads remain in the private bucket after local staging cleanup; the launcher
does not silently delete remote artifacts or cancel a job on a later error.

### Runner-specific images and the separate native execution gate

Both digest-pinned official Astral images retain uv **0.11.18** and Python 3.12:

- `convert_gguf.py`, `train.py`, and `train_unsloth.py` default to
  `ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie@sha256:c8fe61165767c52118d7d3e67db5d0be5454c2e0da1f9fe15c697ebdd72f304e`.
  The non-slim base retains the native build-tool layer; its registry config
  records Git, GCC/G++, and make provisioning. This is metadata evidence, not
  proof that those commands or the converter execute successfully.
- The other twelve runners keep the lightweight default unchanged:
  `ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie-slim@sha256:9f9a7c30d604c8297ac14321b91b42f8516ff207f42b76b1bf29974fad523f6e`.

The public GHCR index, digest-addressed index, Linux amd64/arm64 manifests and
config digests were checked read-only for both tags. See
[Astral's image guidance](https://docs.astral.sh/uv/guides/integration/docker/)
for the supported non-slim Python base. No image was pulled or executed locally:
the bounded read-only Docker check found no usable engine. Container execution
therefore remains a separate CI gate, not a mocked-SDK test result.

`--image IMAGE` before the first `--` overrides either default **literally**.
The launcher does not repair, normalize, or replace an empty/malformed explicit
value with a default, nor validate whether a custom image can run the job.
Defaults depend only on the runner's reviewed closure, not its opaque script
arguments (including `--no-gguf`). The preview, SDK request, and image readback
use the same selected value.

For GGUF export, provision `git`, `cc`, `c++`, and `make` in the image. CMake,
sentencepiece, and protobuf belong in the initially reviewed Python/script locks;
the native image does not qualify or replace those locks. The pinned helper must
not bootstrap apt/pip/uv or install mutable upstream requirements at runtime.
Parent integration must supply the separately repaired helper and matching
initial locks before claiming converter readiness; staging the helper's hash
does not establish that its implementation is correct.

On an ordinary Linux CI worker with Docker already provided, qualify the native
image separately with the stdlib-only `tools/qualify_gguf_image.py` probe. Pull
only in that explicitly authorized CI gate, bound the pull/run, then execute a
fresh container with `--pull never --network none --read-only --cpus 2 --memory 1g
--pids-limit 64 --cap-drop ALL --security-opt no-new-privileges`, an executable
128 MiB temporary filesystem, and only the probe file mounted read-only. Run
`python -I -B -S /probe.py`; it checks uv/Python versions, executes
`git/cc/c++/make --version`, compiles C and C++ fixtures using make, and runs both
CPU binaries. Remove only that gate's own container on timeout. A passing
`--native-only` host invocation merely exercises the probe, not the image.

The launcher is a bounded submission helper, not all HF Jobs orchestration.
Cloud scheduling, credentials, storage permissions, GPU/CUDA compatibility,
model downloads, the full locked converter/training path and publication remain
**unqualified** by offline tests or by the native-only image gate. A locked
submission is not an end-to-end lock of every external tool a runner may invoke.

## Equivalent local locked commands

From `finetune/`, after reviewing the script and its adjacent lock:

```sh
uv run --locked --no-project --script jobs/sft.py --dry-run
uv run --locked --no-project --script experiments/grpo/grpo.py --dry-run
uv run --locked --no-project --script dataset/score_data.py finetune/data/examples.jsonl
uv run --locked --no-project --script train.py sft --config configs/sft.yaml --dry-run
```

These are real runtime commands, **not** launcher previews. Even a script's
`--dry-run` can require dependency installation; do not run them merely to view
the upload manifest. To update a lock, use the separately reviewed dependency
maintenance workflow and review/commit the resulting lock, never regenerate it
implicitly during submission.
