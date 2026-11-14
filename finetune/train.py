# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.13.0,<3",
#     "trl>=1.17.1,<1.18",
#     "peft>=0.21.3,<0.22",
#     "transformers>=5.22.2,<5.23",
#     "accelerate==1.15.0+qmd.2",
#     "huggingface_hub>=2.4.0,<3",
#     "trackio>=0.45.1,<0.46",
#     "nvidia-ml-py",
#     "datasets>=5.2.0,<5.3",
#     "bitsandbytes>=0.50.2,<0.51; sys_platform == 'linux'",
#     "pyyaml>=6.0.3,<7",
#     "gguf",
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
Unified training script for QMD query expansion models.

Primary pipeline is SFT-only:
  sft  - Supervised fine-tuning on labeled examples

GRPO was moved to `experiments/grpo/` and is not part of the main training
pipeline by default.

Usage:
    uv run train.py sft  --config configs/sft.yaml
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from transformers import TrainerCallback


def export_gguf(model, tokenizer, output_dir: str, model_name: str):
    """Export a complete Q4_K_M/Q6_K/Q8_0 set without installing dependencies."""
    import tempfile
    from gguf_toolchain import convert_model, prepare_toolchain

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Both tools and artifacts are fresh, private to this call, and cleaned on
    # failure. Never replace previous exports until every process succeeded.
    with tempfile.TemporaryDirectory(prefix=".gguf-convert-", dir=output_path) as tmp:
        work = Path(tmp)
        tools = prepare_toolchain(work / "tools")
        merged_path = work / "merged"
        print("Saving merged model for GGUF conversion...")
        model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)
        files = convert_model(tools, merged_path, model_name)
        gguf_dir = output_path / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        for path in files:
            path.replace(gguf_dir / path.name)
    print(f"GGUF files saved to: {gguf_dir}")


class TimedSaveCallback(TrainerCallback):
    """Trigger periodic checkpoint saves based on elapsed wall-clock time."""

    def __init__(self, interval_minutes: float):
        self.interval_seconds = float(interval_minutes) * 60.0
        self.last_save_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", False):
            return control

        now = time.time()
        if now - self.last_save_time >= self.interval_seconds:
            control.should_save = True
            self.last_save_time = now
        return control


def run_eval(model_path: str) -> float | None:
    """Run eval.py on the trained model and return average score."""
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)

    eval_script = Path(__file__).parent / "eval.py"
    result = subprocess.run(
        [sys.executable, str(eval_script), model_path],
        cwd=str(Path(__file__).parent),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    avg = None
    for line in (result.stdout or "").splitlines():
        if line.strip().startswith("Average:"):
            try:
                avg = float(line.split("Average:", 1)[1].split("%", 1)[0].strip())
            except ValueError:
                pass
            break
    return avg


def build_sft_config(cfg, **overrides):
    """Build current TRL arguments while preserving the existing YAML schema."""
    from trl import SFTConfig

    training = cfg["training"]
    kwargs = dict(
        output_dir=cfg["model"]["output"],
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        # PyYAML treats values such as 2e-4 as strings.
        learning_rate=float(training["learning_rate"]),
        max_length=training["max_length"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=training.get("save_steps", 200),
        save_total_limit=training.get("save_total_limit", 2),
        eval_strategy="steps",
        eval_steps=training.get("eval_steps", 200),
        # Fractional warmup_steps replaces warmup_ratio in Transformers 5.22.
        warmup_steps=float(training["warmup_ratio"]),
        lr_scheduler_type=training["lr_scheduler"],
        ddp_find_unused_parameters=training.get("ddp_find_unused_parameters", False),
        dataset_text_field=cfg["dataset"].get("text_field", "text"),
        project=cfg.get("tracking", {}).get("project", "huggingface"),
        bf16=True,
        report_to="none",
    )
    kwargs.update(overrides)
    return SFTConfig(**kwargs)


def cmd_sft(args):
    """Run supervised fine-tuning."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Configuration previews must not import GPU extensions, authenticate, or
    # touch the Hub/model/dataset cache.
    if args.dry_run:
        print("SFT Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    # Validate the locked Python/native prerequisites before any data/model work.
    from gguf_toolchain import check_prerequisites
    check_prerequisites()

    import torch
    from datasets import load_dataset
    import torch.distributed as dist
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    from trl import SFTTrainer

    os.environ.setdefault("HF_LOG_CUDA_MEMORY", "0")

    dataset_name = cfg["dataset"]["name"]
    print(f"Loading dataset: {dataset_name}...")

    # Support local JSONL files and glob patterns
    if dataset_name.startswith("data/") or dataset_name.endswith(".jsonl"):
        from pathlib import Path
        import glob

        # Handle glob patterns like "data/*.jsonl"
        if "*" in dataset_name:
            jsonl_files = sorted(glob.glob(dataset_name))
            if not jsonl_files:
                raise ValueError(f"No files found matching: {dataset_name}")
            print(
                f"  Found {len(jsonl_files)} JSONL files: {[Path(f).name for f in jsonl_files]}"
            )
            dataset = load_dataset("json", data_files=jsonl_files, split="train")
        else:
            data_path = Path(dataset_name)
            if data_path.is_dir():
                train_file = data_path / "train.jsonl"
                dataset = load_dataset(
                    "json", data_files=str(train_file), split="train"
                )
            else:
                dataset = load_dataset("json", data_files=dataset_name, split="train")
    else:
        dataset = load_dataset(dataset_name, split=cfg["dataset"]["split"])
    print(f"Dataset loaded: {len(dataset)} examples")

    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=cfg["dataset"]["eval_split"], seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Check if output looks like a HF Hub path (contains /)
    output_name = cfg["model"]["output"]
    push_to_hub = "/" in output_name and not output_name.startswith("outputs/")
    if "push_to_hub" in cfg["model"]:
        push_to_hub = bool(cfg["model"]["push_to_hub"])
    output_dir = output_name.split("/")[-1] if push_to_hub else output_name

    report_to = "none"
    if os.environ.get("HF_TOKEN"):
        try:
            import trackio  # noqa: F401

            report_to = "trackio"
        except Exception:
            print("Trackio not installed; disabling tracking.")

    tracking = cfg.get("tracking", {})
    if report_to == "trackio":
        project = tracking.get("project")
        if project:
            os.environ.setdefault("TRACKIO_PROJECT", project)

    run_name = tracking.get("run_name")
    if run_name and "{" in run_name:
        from datetime import datetime

        now = datetime.now()
        run_name = run_name.replace("{day}", now.strftime("%b %d")).replace(
            "{time}", now.strftime("%H:%M")
        )

    save_interval_minutes = cfg["training"].get("save_interval_minutes")
    save_steps = cfg["training"].get("save_steps", 200)
    save_total_limit = cfg["training"].get("save_total_limit", 2)
    if save_interval_minutes:
        # Prefer wall-clock checkpointing (for long jobs / preemption safety)
        save_steps = max(save_steps, 10_000_000)

    callbacks = []
    if save_interval_minutes:
        try:
            interval_value = float(save_interval_minutes)
        except (TypeError, ValueError):
            interval_value = None
        if interval_value and interval_value > 0:
            callbacks.append(TimedSaveCallback(interval_value))

    config = build_sft_config(
        cfg,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_model_id=output_name if push_to_hub else None,
        hub_strategy="every_save" if push_to_hub else "end",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        report_to=report_to,
        run_name=run_name if report_to == "trackio" else None,
    )

    # LoRA config with modules_to_save for embedding layers
    # This prevents token ID mismatches during inference
    peft_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
        modules_to_save=["embed_tokens", "lm_head"],  # Critical for special tokens
        ensure_weight_tying=True,
    )

    print("Loading tokenizer...")
    base_model = cfg["model"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print("Starting SFT training...")
    trainer.train()

    is_main = os.environ.get("RANK", "0") == "0"
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not is_main:
        return

    if push_to_hub:
        print("Pushing to Hub...")
        trainer.push_to_hub()
        print(f"Done! Model: https://huggingface.co/{output_name}")
    else:
        trainer.save_model()
        print(f"Done! Model saved to: {output_dir}")

    # Export GGUF
    print("\nExporting to GGUF...")
    # Need to get the merged model for GGUF
    print("Loading model for GGUF export...")
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto"
    )
    base.config.tie_word_embeddings = False
    model = PeftModel.from_pretrained(base, output_dir, local_files_only=True)
    model = model.merge_and_unload()
    export_gguf(model, tokenizer, output_dir, Path(output_dir).name)

    # Run eval
    eval_avg = run_eval(output_dir)
    if report_to == "trackio":
        try:
            import trackio

            if eval_avg is not None:
                trackio.log({"eval.avg": eval_avg})
        except Exception:
            pass


def cmd_grpo(args):
    """Run GRPO reinforcement learning on top of merged SFT weights."""
    print(
        "GRPO is not part of the main training pipeline and has been moved to `experiments/grpo/`."
    )
    print("To run experimental GRPO, use:")
    print("  cd finetune && uv run python experiments/grpo/grpo.py")
    return

    import torch
    import torch.distributed as dist
    import os
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    from trl import GRPOTrainer, GRPOConfig

    # Import reward from the shared module
    sys.path.insert(0, os.path.dirname(__file__))
    from reward import QMDRewardFunction

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("HF_LOG_CUDA_MEMORY", "0")

    if args.dry_run:
        print("GRPO Training Configuration:")
        print(yaml.dump(cfg, default_flow_style=False))
        return

    # Tracking
    report_to = "none"
    if os.environ.get("HF_TOKEN"):
        try:
            import trackio  # noqa: F401

            report_to = "trackio"
        except Exception:
            print("Trackio not installed; disabling tracking.")

    tracking = cfg.get("tracking", {})
    if report_to == "trackio":
        project = tracking.get("project")
        if project:
            os.environ.setdefault("TRACKIO_PROJECT", project)

    run_name = tracking.get("run_name")
    if run_name and "{" in run_name:
        from datetime import datetime

        now = datetime.now()
        run_name = run_name.replace("{day}", now.strftime("%b %d")).replace(
            "{time}", now.strftime("%H:%M")
        )

    # Load tokenizer
    base_model_name = cfg["model"]["base"]
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    print("Loading dataset...")
    dataset = load_dataset(cfg["dataset"]["name"], split="train")

    def extract_prompt(example):
        content = example[cfg["dataset"]["prompt_field"]][0]["content"]
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": formatted}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    max_samples = cfg["dataset"].get("max_samples", len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(min(max_samples, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load base model, merge SFT adapter
    sft_model_name = cfg["model"]["sft"]
    print(f"Loading SFT model from {sft_model_name}...")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        if available == 0:
            raise RuntimeError("CUDA is available but no devices were detected.")
        if local_rank >= available:
            print(
                f"Warning: LOCAL_RANK={local_rank} but only {available} CUDA device(s) visible. "
                "Falling back to the last available device."
            )
            local_rank = available - 1
        torch.cuda.set_device(local_rank)
    dtype_name = cfg["model"].get("torch_dtype", "bfloat16")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": {"": local_rank} if torch.cuda.is_available() else "auto",
    }

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, sft_model_name)
    model = model.merge_and_unload()
    print("SFT adapter merged.")

    # Add fresh LoRA for GRPO with modules_to_save
    grpo_lora_config = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
        modules_to_save=["embed_tokens", "lm_head"],  # Critical for special tokens
        ensure_weight_tying=True,
    )
    model = get_peft_model(model, grpo_lora_config)
    model.print_trainable_parameters()

    # Build GRPO config
    output_name = cfg["model"]["output"]
    push_to_hub = "/" in output_name and not output_name.startswith("outputs/")
    if "push_to_hub" in cfg["model"]:
        push_to_hub = bool(cfg["model"]["push_to_hub"])
    output_dir = output_name.split("/")[-1] if push_to_hub else output_name

    grpo_cfg = cfg.get("grpo", {})
    learning_rate = cfg["training"]["learning_rate"]
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)

    save_interval_minutes = cfg["training"].get("save_interval_minutes")
    save_steps = cfg["training"].get("save_steps", 200)
    save_total_limit = cfg["training"].get("save_total_limit", 2)
    save_strategy = cfg["training"].get("save_strategy", "epoch")
    if save_interval_minutes:
        # Prefer wall-clock checkpointing (for long jobs / preemption safety)
        save_steps = max(save_steps, 10_000_000)
        save_strategy = "steps"

    callbacks = []
    if save_interval_minutes:
        try:
            interval_value = float(save_interval_minutes)
        except (TypeError, ValueError):
            interval_value = None
        if interval_value and interval_value > 0:
            callbacks.append(TimedSaveCallback(interval_value))

    config = GRPOConfig(
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_model_id=output_name if push_to_hub else None,
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 200),
        beta=grpo_cfg.get("beta", 0.04),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        max_grad_norm=cfg["training"]["max_grad_norm"],
        max_steps=cfg["training"].get("max_steps", -1),
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=True,
        skip_memory_metrics=True,
        report_to=report_to,
        run_name=run_name if report_to == "trackio" else None,
    )

    # Train
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[QMDRewardFunction()],
        callbacks=callbacks,
    )

    print("Starting GRPO training...")
    trainer.train()

    is_main = os.environ.get("RANK", "0") == "0"
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if not is_main:
        return

    if push_to_hub:
        print("Pushing to Hub...")
        trainer.push_to_hub()

    trainer.save_model()
    if report_to == "trackio":
        try:
            import trackio

            trackio.finish()
        except Exception:
            pass
    print(f"Done! Model saved to: {output_dir}")

    # Export GGUF
    print("\nExporting to GGUF...")
    merged = model.merge_and_unload()
    export_gguf(merged, tokenizer, output_dir, Path(output_dir).name)

    # Run eval
    eval_avg = run_eval(output_dir)
    if report_to == "trackio" and eval_avg is not None:
        try:
            import trackio

            trackio.log({"eval.avg": eval_avg})
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="QMD Query Expansion Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run train.py sft  --config configs/sft.yaml
        """,
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    sft_parser = sub.add_parser("sft", help="Supervised fine-tuning")
    sft_parser.add_argument("--config", required=True, help="Path to SFT config YAML")
    sft_parser.add_argument(
        "--dry-run", action="store_true", help="Print config and exit"
    )

    args = parser.parse_args()

    cmd_sft(args)

if __name__ == "__main__":
    main()
