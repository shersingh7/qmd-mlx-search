#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers>=5.22.2,<5.23",
#     "peft>=0.21.3,<0.22",
#     "torch>=2.13.0,<3",
#     "accelerate==1.15.0+qmd.2",
#     "huggingface_hub>=2.4.0,<3",
#     "sentencepiece>=0.2.1",
#     "protobuf>=3.20.0",
#     "numpy",
#     "onnx>=1.23.2",
#     "onnxruntime>=1.32.0",
#     "onnx-ir>=1.1.0",
#     "onnxscript>=0.7.2",
# ]
# [tool]
# [tool.uv]
# constraint-dependencies = [
#     "accelerate==1.15.0+qmd.2",
#     "aiohttp>=3.14.4",
#     "anyio>=4.14.2",
#     "authlib>=1.9.1",
#     "brotli>=1.2.1",
#     "certifi>=2026.10.18",
#     "cryptography>=50.0.0",
#     "datasets>=5.0.1",
#     "filelock>=4.2.1",
#     "fsspec>=2026.6.0",
#     "gradio>=6.16.0",
#     "idna>=3.15",
#     "multidict>=6.9.1",
#     "pillow>=13.0.0",
#     "pyarrow>=23.0.1",
#     "pygments>=2.20.0",
#     "python-multipart>=0.0.31",
#     "requests>=2.33.0",
#     "setuptools>=84.1.0",
#     "starlette>=1.3.1",
#     "torch>=2.13.0",
#     "transformers>=5.17.0",
#     "urllib3>=2.8.0",
# ]
# [tool.uv.sources]
# [tool.uv.sources.accelerate]
# url = "https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/f452290f116934a4f126fe8e513c565e97a8c94f/finetune/vendor/accelerate/accelerate-1.15.0%2Bqmd.2-py3-none-any.whl"
# ///
"""
Convert QMD query expansion model to ONNX format for Transformers.js.

Loads the base model, merges SFT and GRPO adapters, then exports to ONNX
with quantization for browser deployment via Transformers.js + WebGPU.

Usage:
    uv run convert_onnx.py --size 1.7B
    uv run convert_onnx.py --size 1.7B --no-upload
    uv run convert_onnx.py --base Qwen/Qwen3-1.7B \
                           --sft tobil/qmd-query-expansion-1.7B-sft \
                           --grpo tobil/qmd-query-expansion-1.7B-grpo \
                           --output tobil/qmd-query-expansion-1.7B-ONNX

Quantization options:
    --quantize q4    MatMulNBits 4-bit (default, smallest)
    --quantize q8    8-bit dynamic quantization
    --quantize fp16  FP16 (requires GPU export)
    --quantize none  No quantization (FP32, ~7GB)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PRESETS = {
    "1.7B": {
        "base": "Qwen/Qwen3-1.7B",
        "sft": "tobil/qmd-query-expansion-1.7B-sft",
        "grpo": "tobil/qmd-query-expansion-1.7B-grpo",
        "output": "tobil/qmd-query-expansion-1.7B-ONNX",
    },
    "4B": {
        "base": "Qwen/Qwen3-4B",
        "sft": "tobil/qmd-query-expansion-4B-sft",
        "grpo": "tobil/qmd-query-expansion-4B-grpo",
        "output": "tobil/qmd-query-expansion-4B-ONNX",
    },
}


def merge_adapters(base_model: str, sft_model: str, grpo_model: str) -> tuple:
    """Load base model, merge SFT + GRPO adapters, return (model, tokenizer)."""
    print(f"\nStep 1: Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float32, trust_remote_code=True,
    )

    print(f"Step 2: Merging SFT adapter {sft_model}...")
    model = PeftModel.from_pretrained(model, sft_model)
    model = model.merge_and_unload()

    print(f"Step 3: Merging GRPO adapter {grpo_model}...")
    model = PeftModel.from_pretrained(model, grpo_model)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    return model, tokenizer


def export_onnx(model, tokenizer, output_dir: str):
    """Export Qwen2/Qwen3 with flattened KV-cache I/O for Transformers.js.

    optimum-onnx 0.1.0 requires transformers<4.58, which cannot satisfy the
    security floor. Use the supported PyTorch exporter instead of overriding
    upstream metadata or silently downgrading Transformers.
    """
    from transformers.cache_utils import DynamicCache

    if model.config.model_type not in {"qwen2", "qwen3"}:
        raise ValueError("The native ONNX exporter supports Qwen2/Qwen3 models only")

    class WithPast(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids, attention_mask, position_ids, *past):
            cache = DynamicCache(
                ddp_cache_data=tuple(zip(past[::2], past[1::2])),
                config=self.wrapped.config,
            )
            # Build an explicit additive mask. This avoids tracing the
            # Python/vmap mask dispatcher and keeps batch/sequence/past axes
            # dynamic, including an empty cache on the first generation call.
            key_positions = torch.arange(attention_mask.shape[-1], device=input_ids.device)
            # Rotary position_ids count real tokens and can differ by batch
            # row under left padding. Causality instead uses physical slots:
            # the queries follow the cached keys, including padding slots.
            query_positions = torch.arange(input_ids.shape[-1], device=input_ids.device) + past[0].shape[-2]
            allowed = (key_positions <= query_positions.unsqueeze(-1)) & attention_mask[:, None, :].bool()
            dtype = self.wrapped.dtype
            mask = torch.where(allowed[:, None, :, :], 0.0, torch.finfo(dtype).min).to(dtype)
            result = self.wrapped(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            present = tuple(t for layer in result.past_key_values.layers for t in (layer.keys, layer.values))
            return (result.logits, *present)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = model.to(device="cpu", dtype=torch.float32).eval()
    model.set_attn_implementation("eager")
    config = model.config
    layers = config.num_hidden_layers
    heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    input_ids = torch.ones((1, 3), dtype=torch.long)
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    position_ids = torch.arange(2, 5).unsqueeze(0)
    past = tuple(torch.zeros((1, heads, 2, head_dim)) for _ in range(layers * 2))
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    for i in range(layers):
        for kind in ("key", "value"):
            inp = f"past_key_values.{i}.{kind}"
            out = f"present.{i}.{kind}"
            input_names.append(inp)
            output_names.append(out)
            dynamic_axes[inp] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[out] = {0: "batch_size", 2: "total_sequence_length"}

    print(f"\nExporting ONNX with KV cache to {output}...")
    with torch.inference_mode():
        torch.onnx.export(
            WithPast(model),
            (input_ids, attention_mask, position_ids, *past),
            str(output / "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
            external_data=True,
        )
    model.config.save_pretrained(output)
    model.generation_config.save_pretrained(output)
    tokenizer.save_pretrained(output)


def _find_onnx_model(onnx_dir: str) -> Path:
    """Find the main ONNX model file in the output directory."""
    model_path = Path(onnx_dir) / "model.onnx"
    if model_path.exists():
        return model_path
    candidates = list(Path(onnx_dir).glob("*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"No .onnx files found in {onnx_dir}")
    return candidates[0]


def quantize_onnx(onnx_dir: str, quantize_type: str):
    """Quantize the exported ONNX model."""
    if quantize_type == "none":
        print("\nSkipping quantization (FP32).")
        return

    model_path = _find_onnx_model(onnx_dir)
    print(f"\nStep 6: Quantizing {model_path.name} ({quantize_type})...")

    if quantize_type == "q4":
        _quantize_q4(model_path)
    elif quantize_type == "q8":
        _quantize_q8(model_path)
    elif quantize_type == "fp16":
        _convert_fp16(model_path)


def _quantize_q4(model_path: Path):
    """4-bit MatMulNBits quantization via onnxruntime. Needs ~16GB RAM for 1.7B models."""
    from onnxruntime.quantization import matmul_nbits_quantizer

    q_path = model_path.with_name(model_path.stem + "_q4" + model_path.suffix)
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=str(model_path),
        block_size=32,
        is_symmetric=True,
        bits=4,
    )
    quant.process()
    quant.model.save_model_to_file(str(q_path), use_external_data_format=True)

    # Remove original FP32 files, keep only quantized
    if q_path.exists():
        _report_size(q_path)
        model_path.unlink(missing_ok=True)
        data_path = model_path.with_name(model_path.name + "_data")
        data_path.unlink(missing_ok=True)
        # Rename quantized to model.onnx for Transformers.js compatibility
        q_path.rename(model_path)
        print(f"  Renamed {q_path.name} -> {model_path.name}")


def _quantize_q8(model_path: Path):
    """8-bit dynamic quantization via onnxruntime."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    q_path = model_path.with_name(model_path.stem + "_q8" + model_path.suffix)
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(q_path),
        weight_type=QuantType.QUInt8,
        use_external_data_format=True,
    )

    if q_path.exists():
        _report_size(q_path)
        model_path.unlink(missing_ok=True)
        data_path = model_path.with_name(model_path.name + "_data")
        data_path.unlink(missing_ok=True)
        q_path.rename(model_path)
        print(f"  Renamed {q_path.name} -> {model_path.name}")


def _convert_fp16(model_path: Path):
    """Convert ONNX model weights to FP16."""
    from onnxruntime.transformers import float16
    import onnx

    print("  Converting to FP16...")
    model = onnx.load(str(model_path), load_external_data=True)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    fp16_path = model_path.with_name(model_path.stem + "_fp16" + model_path.suffix)
    onnx.save_model(
        model_fp16, str(fp16_path), save_as_external_data=True,
        all_tensors_to_one_file=True, location=fp16_path.name + ".data",
        size_threshold=1024,
    )

    if fp16_path.exists():
        _report_size(fp16_path)
        model_path.unlink(missing_ok=True)
        data_path = model_path.with_name(model_path.name + "_data")
        data_path.unlink(missing_ok=True)
        fp16_path.rename(model_path)
        print(f"  Renamed {fp16_path.name} -> {model_path.name}")


def _report_size(path: Path):
    """Print file size in MB."""
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  {path.name}: {size_mb:.1f} MB")



def validate_onnx(onnx_dir: str, base_model: str):
    """Run a sample inference through the ONNX model to verify it works."""
    import onnxruntime as ort
    import numpy as np

    model_path = _find_onnx_model(onnx_dir)
    print(f"\nValidation: loading {model_path.name}...")

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir, trust_remote_code=True)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    # Tokenize a test prompt
    test_query = "/no_think Expand this search query: distributed consensus"
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": test_query}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(chat_prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Build feed dict with all required inputs
    seq_len = input_ids.shape[1]
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Add position_ids if needed
    all_inputs = {inp.name: inp for inp in session.get_inputs()}
    if "position_ids" in all_inputs:
        feed["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    # Initialize past_key_values to zeros if the model expects them
    for name, inp in sorted(all_inputs.items()):
        if name.startswith("past_key_values"):
            shape = []
            for dim in inp.shape:
                shape.append(dim if isinstance(dim, int) else 0)
            # batch dim = 1
            if shape and shape[0] == 0:
                shape[0] = 1
            feed[name] = np.zeros(shape, dtype=np.float32)

    # Run inference
    output_names = [o.name for o in session.get_outputs()]
    results = session.run(output_names, feed)

    # Check logits shape
    logits = results[0]
    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Greedy decode next token
    next_token_id = int(np.argmax(logits[0, -1, :]))
    next_token = tokenizer.decode([next_token_id])
    print(f"  Next token: {repr(next_token)} (id={next_token_id})")

    # Check KV cache outputs exist
    kv_outputs = [n for n in output_names if n.startswith("present")]
    if kv_outputs:
        print(f"  KV cache outputs: {len(kv_outputs)} tensors (generation-ready)")
    else:
        print("  WARNING: No KV cache outputs — model may not support efficient generation")

    # Sanity checks
    assert logits.shape[0] == 1, "Batch size mismatch"
    assert logits.shape[1] == input_ids.shape[1], "Sequence length mismatch"
    assert logits.max() > logits.min(), "Logits are constant (broken model)"
    assert not np.isnan(logits).any(), "Logits contain NaN"
    assert not np.isinf(logits).any(), "Logits contain Inf"

    print("  Validation PASSED")


def write_transformers_js_config(onnx_dir: str, quantize_type: str = "q4"):
    """Write Transformers.js compatibility config."""
    config_path = Path(onnx_dir) / "transformers_js_config.json"
    config = {
        "model_type": "text-generation",
        "quantized": quantize_type != "none",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  Wrote {config_path.name}")


def upload_to_hub(
    onnx_dir: str,
    output_repo: str,
    base_model: str,
    sft_model: str,
    grpo_model: str,
    quantize_type: str = "q4",
):
    """Upload ONNX model to HuggingFace Hub."""
    print(f"\nStep 7: Uploading to {output_repo}...")
    api = HfApi()
    api.create_repo(repo_id=output_repo, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=onnx_dir,
        repo_id=output_repo,
        commit_message="Upload ONNX model",
    )

    # Map quantize_type to Transformers.js dtype values
    dtype_map = {"q4": "q4", "q8": "q8", "fp16": "fp16", "none": "fp32"}
    tj_dtype = dtype_map.get(quantize_type, "fp32")
    format_desc = "FP32 (no quantization)" if quantize_type == "none" else f"{quantize_type.upper()} quantization"
    repo_name = output_repo.split("/")[-1]

    readme = f"""---
base_model: {base_model}
tags: [onnx, transformers.js, webgpu, query-expansion, qmd]
library_name: transformers.js
---
# {repo_name}

ONNX conversion of the QMD Query Expansion model for use with
[Transformers.js](https://huggingface.co/docs/transformers.js) and WebGPU.

## Details
- **Base:** {base_model}
- **SFT:** {sft_model}
- **GRPO:** {grpo_model}
- **Task:** Query expansion (lex/vec/hyde format)
- **Format:** ONNX with {format_desc}

## Usage with Transformers.js

```javascript
import {{ AutoTokenizer, AutoModelForCausalLM }} from "@huggingface/transformers";

const tokenizer = await AutoTokenizer.from_pretrained("{output_repo}");
const model = await AutoModelForCausalLM.from_pretrained("{output_repo}", {{
  dtype: "{tj_dtype}",
  device: "webgpu",
}});
```

## Prompt Format
```
<|im_start|>user
/no_think Expand this search query: your query here<|im_end|>
<|im_start|>assistant
```
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=output_repo,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert QMD model to ONNX")
    parser.add_argument(
        "--size", choices=PRESETS.keys(), help="Use preset config for model size",
    )
    parser.add_argument("--base", help="Base model (overrides preset)")
    parser.add_argument("--sft", help="SFT adapter (overrides preset)")
    parser.add_argument("--grpo", help="GRPO adapter (overrides preset)")
    parser.add_argument("--output", help="Output HF repo (overrides preset)")
    parser.add_argument(
        "--quantize",
        choices=["q4", "q8", "fp16", "none"],
        default="q4",
        help="Quantization type (default: q4)",
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Don't upload to HF Hub",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run inference validation on exported model",
    )
    parser.add_argument(
        "--validate-only", metavar="DIR",
        help="Skip export, only validate an existing ONNX dir",
    )
    args = parser.parse_args()

    # Validate-only mode: skip export, just run validation
    if args.validate_only:
        validate_onnx(args.validate_only, "")
        return

    # Resolve config
    if args.size:
        preset = PRESETS[args.size]
        base_model = args.base or preset["base"]
        sft_model = args.sft or preset["sft"]
        grpo_model = args.grpo or preset["grpo"]
        output_repo = args.output or preset["output"]
    elif args.base and args.sft and args.grpo and args.output:
        base_model = args.base
        sft_model = args.sft
        grpo_model = args.grpo
        output_repo = args.output
    else:
        parser.error(
            "Either --size or all of --base/--sft/--grpo/--output are required",
        )

    model_name = output_repo.split("/")[-1]
    print(f"QMD ONNX Conversion: {model_name}")
    print("=" * 60)

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace...")
        login(token=hf_token)

    # Merge adapters
    model, tokenizer = merge_adapters(base_model, sft_model, grpo_model)

    # Export to ONNX
    onnx_dir = f"/tmp/onnx_output/{model_name}"
    os.makedirs(onnx_dir, exist_ok=True)
    export_onnx(model, tokenizer, onnx_dir)

    # Quantize
    quantize_onnx(onnx_dir, args.quantize)

    # Write Transformers.js config
    write_transformers_js_config(onnx_dir, args.quantize)

    # Validate
    if args.validate:
        validate_onnx(onnx_dir, base_model)

    # Upload
    if not args.no_upload:
        upload_to_hub(onnx_dir, output_repo, base_model, sft_model, grpo_model, args.quantize)

    print(f"\nDone! ONNX files at: {onnx_dir}")
    if not args.no_upload:
        print(f"Repository: https://huggingface.co/{output_repo}")


if __name__ == "__main__":
    main()
