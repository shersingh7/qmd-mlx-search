# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "trl>=1.17.1,<1.18",
#     "peft>=0.21.3,<0.22",
#     "transformers>=5.22.2,<5.23",
#     "accelerate>=1.15.0,<1.16",
#     "huggingface_hub>=2.4.0,<3",
#     "datasets>=5.2.0,<5.3",
#     "bitsandbytes>=0.50.2; sys_platform == \"linux\"",
#     "torch>=2.13.0,<3",
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
"""
GRPO training for QMD query expansion (Qwen3-1.7B).

Experimental recipe run on top of merged SFT weights. Self-contained runner:
    uv run experiments/grpo/grpo.py

(If using HF Jobs, run this script as the job entrypoint.)
"""

import argparse
import os
import sys

# --- Config (inlined from experiments/grpo/grpo.yaml) ---
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
OUTPUT_MODEL = "tobil/qmd-query-expansion-1.7B-grpo"
DATASET = "tobil/qmd-query-expansion-train"


def build_config(**overrides):
    from trl import GRPOConfig

    kwargs = dict(
        output_dir="qmd-query-expansion-1.7B-grpo",
        push_to_hub=True, hub_model_id=OUTPUT_MODEL,
        num_generations=4, max_completion_length=200, beta=0.04,
        num_train_epochs=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=5e-7,
        max_grad_norm=0.5, max_steps=200,
        logging_steps=10, save_strategy="epoch", bf16=True, report_to="none",
    )
    kwargs.update(overrides)
    return GRPOConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without authentication, downloads, or training")
    args = parser.parse_args()
    if args.dry_run:
        print({"base_model": BASE_MODEL, "sft_model": SFT_MODEL, "output_model": OUTPUT_MODEL, "dataset": DATASET, "stage": "grpo"})
        return

    import torch
    from datasets import load_dataset
    from huggingface_hub import login
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer
    from pathlib import Path

    # Prefer the repository helper. Download only in standalone HF Jobs,
    # never during module import, --help, or --dry-run.
    helper_dir = Path(__file__).resolve().parent
    parents = Path(__file__).resolve().parents
    if len(parents) > 2 and (parents[2] / "jobs" / "eval_common.py").exists():
        helper_dir = parents[2] / "jobs"
    helper = helper_dir / "eval_common.py"
    if not helper.exists():
        import urllib.request
        opener = urllib.request.build_opener()
        token = os.environ.get("HF_TOKEN", "")
        if token:
            opener.addheaders = [("Authorization", f"Bearer {token}")]
        url = "https://huggingface.co/datasets/tobil/hf-cli-jobs-uv-run-scripts/resolve/main/eval_common.py"
        with opener.open(url, timeout=30) as response:
            helper.write_bytes(response.read())
    sys.path.insert(0, str(helper_dir))
    from eval_common import QMDRewardFunction, run_eval

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    print(f"Loading dataset: {DATASET}...")
    dataset = load_dataset(DATASET, split="train")

    def extract_prompt(example):
        content = example["messages"][0]["content"]
        messages = [{"role": "user", "content": content}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": formatted}

    dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
    print(f"Using {len(dataset)} prompts for GRPO")

    # Load base model, merge SFT adapter
    print(f"Loading base model {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Merging SFT adapter {SFT_MODEL}...")
    model = PeftModel.from_pretrained(base_model, SFT_MODEL)
    model = model.merge_and_unload()
    print("SFT adapter merged.")

    # Fresh LoRA for GRPO (small: rank 4, q/v only)
    grpo_lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, grpo_lora)
    model.print_trainable_parameters()

    config = build_config()

    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[QMDRewardFunction()],
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Pushing to Hub...")
    trainer.push_to_hub()
    print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")

    # --- Automatic evaluation ---
    print("\nStarting automatic evaluation...")
    trainer.model.eval()
    run_eval(trainer.model, tokenizer, "grpo")


if __name__ == "__main__":
    main()
