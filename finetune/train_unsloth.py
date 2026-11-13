#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.13.0,<3",
#     "trl>=1.17.1,<1.18",
#     "peft>=0.21.3,<0.22",
#     "transformers>=5.22.2,<5.23",
#     "accelerate>=1.15.0,<1.16",
#     "huggingface_hub>=2.4.0,<3",
#     "datasets>=5.2.0,<5.3",
#     "trackio>=0.45.1,<0.46",
#     "pyyaml>=6.0.3,<7",
#     "gguf",
#     "bitsandbytes>=0.50.2; sys_platform == \"linux\"",
# ]
# [tool.uv]
# constraint-dependencies = [
#     "accelerate>=1.15.0",
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
#     "transformers>=5.10.0",
#     "urllib3>=2.8.0",
# ]
# ///
"""Qwen3.5 SFT with the original train_unsloth.py command-line interface.

The default backend is native Transformers/PEFT/TRL. Unsloth 2026.11.4
requires datasets<5, conflicting with the patched datasets>=5.0.1 floor.
We do not override that metadata or silently install a vulnerable version.

    uv run train_unsloth.py --model 0.8B
    uv run train_unsloth.py --model 2B --dry-run

See DEPENDENCIES.md for the explicit Unsloth compatibility limitation.
"""
import argparse
import os
import sys
from pathlib import Path

MODEL_MAP = {
    "0.8B": "unsloth/Qwen3.5-0.8B",
    "2B": "unsloth/Qwen3.5-2B",
    "4B": "unsloth/Qwen3.5-4B",
    "9B": "unsloth/Qwen3.5-9B",
    "27B": "unsloth/Qwen3.5-27B",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP), help="Model size to train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--data", default="data/train/train.jsonl")
    parser.add_argument("--output", help="Output directory (default: outputs/qwen3.5-{size})")
    parser.add_argument("--push-hub", help="Optional Hugging Face repo for the trained adapter")
    parser.add_argument("--backend", choices=["transformers", "unsloth"], default="transformers",
                        help="Native audited stack (default); Unsloth is currently blocked by its datasets<5 cap")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-gguf", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model_name = MODEL_MAP[args.model]
    output_dir = args.output or f"outputs/qwen3.5-{args.model}"
    print(f"QMD SFT: {model_name}; backend={args.backend}; output={output_dir}")
    print(f"Data={args.data}; epochs={args.epochs}; batch={args.batch_size} x {args.grad_accum}; lr={args.lr}")
    if args.dry_run:
        print("Dry run — no authentication, model loading, or training.")
        return
    if args.backend == "unsloth":
        parser.error("Unsloth 2026.11.4 requires datasets<5, but security remediation requires datasets>=5.0.1. Use --backend transformers; see DEPENDENCIES.md.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    use_cpu = args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available())
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32 if use_cpu else torch.bfloat16,
    )
    dataset = load_dataset("json", data_files=args.data, split="train").shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    report_to = "trackio" if os.environ.get("HF_TOKEN") else "none"
    config = SFTConfig(
        output_dir=output_dir,
        max_length=args.max_seq_len,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=200,
        use_cpu=use_cpu,
        bf16=not use_cpu,
        optim="adamw_torch" if use_cpu else "adamw_8bit",
        seed=3407,
        dataset_num_proc=None if use_cpu else 4,
        report_to=report_to,
        project="qmd-query-expansion",
        run_name=f"sft-qwen3.5-{args.model}",
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=split["train"], eval_dataset=split["test"], args=config,
        peft_config=LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_rank, lora_dropout=0,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if args.push_hub:
        trainer.model.push_to_hub(args.push_hub)
        tokenizer.push_to_hub(args.push_hub)
    if not args.no_gguf:
        from train import export_gguf
        merged = trainer.model.merge_and_unload()
        export_gguf(merged, tokenizer, output_dir, Path(output_dir).name)
    if not args.no_eval:
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name("eval.py")), output_dir], check=True)
    print(f"Done! Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
