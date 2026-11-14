# Fine-tuning dependency and API compatibility

## Supported environment

The project and dependency-bearing PEP 723 runners now require **Python 3.11+**.
Python 3.10 is deliberately retired: Pillow 13 and the selected training stack
require 3.11. `uv.lock` is a real universal uv lock, including Python/platform
forks and Linux CUDA dependencies; it is not a macOS-only replacement.

Validated release matrix:

| Package | Locked version |
| --- | --- |
| torch | 2.15.0 |
| transformers | 5.22.2 |
| trl | 1.17.1 |
| peft | 0.21.3 |
| accelerate | 1.15.0 |
| datasets | 5.2.0 |
| huggingface-hub | 2.4.0 |
| trackio | 0.45.1 |
| Pillow | 13.0.0 |
| starlette / anyio | 1.10.0 / 4.16.0 |
| python-multipart | 0.0.33 |
| urllib3 / certifi | 2.8.0 / 2026.10.23 |
| optional onnx / onnxruntime | 1.23.2 / 1.32.0 |

Public release metadata, rather than version guesses, was used for these
choices: [PyPI JSON API](https://docs.pypi.org/api/json/), including the exact
[Trackio 0.45.1](https://pypi.org/pypi/trackio/0.45.1/json),
[datasets 5.2.0](https://pypi.org/pypi/datasets/5.2.0/json),
[Unsloth 2026.11.4](https://pypi.org/pypi/unsloth/2026.11.4/json), and
[optimum-onnx 0.1.0](https://pypi.org/pypi/optimum-onnx/0.1.0/json) metadata.

## Security policy and graph changes

`[tool.uv].constraint-dependencies` sets explicit floors for all 23 distinct
PEP-503 lookup names represented in the original 157 fine-tuning alerts.
Constraints do **not** install otherwise unused dependencies. Original package
identifiers and advisory IDs in the source alert snapshot are not rewritten.
`tools/sync_script_dependencies.py` propagates the same constraints and project
requirements to every dependency-bearing PEP 723 script. Standalone `uv run
script.py` and HF Jobs resolve independently of the project lock, so updating
only `uv.lock` would not secure those entrypoints.

For the original alert snapshot, every package version in the regenerated
universal project lock is outside the recorded affected ranges: **92 alerts
select patched versions, 63 refer to packages no longer in the upstream graph,
and 2 require independent advisory review**. The latter are deliberately not
claimed fixed merely because a selected version is above the recorded range:

- `GHSA-x9r9-c232-4q39` (transformers; null first-patched version).
- `GHSA-4j2p-28q2-5m79` (accelerate; null first-patched version).

The removed vulnerable nodes are `aiohttp`, `multidict`, `gradio`, `authlib`,
and `cryptography`. This follows **upstream** graph changes: datasets 5.2 no
longer declares aiohttp, and Trackio 0.45.1 ships its own dashboard rather than
requiring Gradio. Monitoring remains installed and enabled by the existing
training configuration/token opt-in; its init, log, finish and Transformers
callback APIs are exercised locally. No monitoring feature was deleted to
make alerts disappear. Floors for those absent packages remain in the policy
in case a later dependency reintroduces them.

## Deliberate API adaptations

- Transformers 5.22 replaced `warmup_ratio` with fractional `warmup_steps`.
  Existing YAML `warmup_ratio: 0.03` retains its meaning (three warmup steps in
  a 100-step run). YAML learning rates such as `2e-4` are converted to floats.
- TRL uses `processing_class`, not `tokenizer`, and `SFTConfig.max_length`,
  not `max_seq_length`. Existing CLI option names are retained.
- Trackio's project is passed via `TrainingArguments.project`; relying only
  on the old environment variable would lose the configured project name.
- GRPO rewards accept both plain text and conversational message inputs.
- Training job scripts are import-safe and have offline `--dry-run` previews.
  No authentication, helper fetch, model/dataset load, training or upload is
  performed for previews.
- CUDA-only bitsandbytes remains declared for Linux GPU runners, with a
  platform marker so Apple Silicon metadata resolution/previews remain usable.
  The native Qwen3.5 CLI checks availability before tokenizer/model downloads
  or loading and refuses a missing 8-bit optimizer dependency; it never silently
  substitutes a different CUDA optimizer.
- Local evaluators retain CUDA BF16 loading where available and have an
  explicit float32 CPU fallback for portable tests.
  Both evaluators honor the exact supplied Hub ID or local full/merged-model
  directory. Local or remote `adapter_config.json` selects its declared base
  plus the requested adapter, preferring the adapter's own tokenizer when
  present. Only a confirmed missing Hub metadata file permits full-model/base-
  tokenizer routing; authentication, network and offline-cache errors propagate
  rather than falling back to an unrelated Qwen model. Full models retain
  their configured embedding tying.

### Qwen3.5 / legacy `train_unsloth.py`

Current Unsloth and unsloth-zoo releases require **datasets<5**, conflicting
with the patched **datasets>=5.0.1** floor. They also cap Transformers/TRL
below this validated matrix. There is no honest compatible Unsloth graph to
install with dependency overrides.

`train_unsloth.py` therefore retains the existing model/epoch/batch/LoRA/data/
output/Hub/GGUF/evaluation CLI, but explicitly defaults to the audited native
**Transformers + PEFT + TRL** backend. Its console identifies the backend.
`--backend unsloth` fails before loading anything and explains the upstream
conflict. `--device cpu` supports small portable fixtures; CUDA keeps BF16 and
the 8-bit optimizer. Unsloth-specific acceleration and performance are **not
qualified** by this remediation. Do not install the old capped stack or use
`--no-deps` to force it into the environment.

With GGUF enabled, `--push-hub REPO` publishes the adapter/tokenizer and the
newly exported q4_k_m, q6_k and q8_0 GGUFs through `HfApi.upload_file`. Conversion
runs in a fresh staging directory and checks all expected nonempty GGUF files
before moving them to `OUTPUT/gguf` and uploading. Failed/incomplete conversion
cannot reuse stale GGUFs from an earlier run; export and upload failures are
explicit errors. `--no-gguf` retains adapter/tokenizer-only publication.

### ONNX conversion

`optimum-onnx` 0.1.0 requires Transformers<4.58, incompatible with the security
floor. Conversion now uses PyTorch's native ONNX exporter for the existing
Qwen2/Qwen3 presets, preserving flattened `past_key_values.*`/`present.*` I/O,
dynamic batch/sequence/cache axes, tokenizer/config assets, and q4/q8/FP16
modes. ONNX Runtime's maintained FP16 conversion avoids the stale Cast type
metadata produced by onnxconverter-common. Q4 uses the current
`save_model_to_file` API. External tensor storage is enabled for large outputs.

The pinned PyTorch release still supports the explicitly selected TorchScript
ONNX exporter; it emits deprecation/tracing warnings. Tiny-model tests verify
empty-cache prefill, nonempty-cache decode and batch/sequence variation against
PyTorch, including unequal-length left-padded batches and successive single-
token/chunked decode with real propagated caches. Causal masks use physical
cache/query offsets independently of per-sequence rotary position IDs. Tests
also check CPU inference for all three quantization modes. Full-size exports
and browser/WebGPU execution remain unqualified.

## Reproducing bounded verification

Run from `finetune/`. Prefer project-locked invocations locally; PEP 723 is for
standalone runners. Use an isolated environment/cache/home for validation, not
an existing application or MLX environment.

```bash
uv lock --check
uv export --frozen --all-groups --no-emit-project --output-file /path/to/scratch/finetune-requirements.txt
uv sync --frozen --group onnx
uv run --frozen --group onnx python -m pytest -q
python tools/sync_script_dependencies.py --check
uv run --frozen python train.py sft --config configs/sft.yaml --dry-run
uv run --frozen python jobs/sft.py --dry-run
```

The tests set an empty fixture HOME/HF cache, offline flags, disable telemetry,
use synthetic public examples and tiny random CPU-only models, and block
network sockets (local Unix IPC is allowed). They exercise actual SFT and GRPO
optimizer steps, adapter save/reload/generation, a real tiny Qwen3.5 hybrid
training CLI run, an LFM2 forward/backward pass, dataset preparation/JSON
loading, local Trackio callbacks, and CLI previews. These are API/compatibility
fixtures, **not trained-model quality or performance results**.

Independent validation also regenerated and checked locks plus frozen exports
for all 16 PEP 723 scripts (15 with third-party dependencies), inspecting every
locked target-version variant against the original alert ranges.

**Not qualified:** real CUDA or MPS training, multi-GPU/DDP, CUDA driver/runtime
compatibility on HF Jobs images, bitsandbytes GPU kernels, production model
quality/performance, full-size GGUF/ONNX conversion, Hub uploads, or any paid
HF Jobs execution. The universal Linux lock now selects PyTorch's CUDA 13.2
runtime; validate the GPU host's driver and runtime before launching training.
No model weights, personal datasets or live indices are needed by verification.
