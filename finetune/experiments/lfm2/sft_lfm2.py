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
"""Self-contained LiquidAI LFM2-1.2B SFT runner for Hugging Face Jobs.

    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h experiments/lfm2/sft_lfm2.py
    python experiments/lfm2/sft_lfm2.py --dry-run
"""
import argparse
import os

BASE_MODEL = "LiquidAI/LFM2-1.2B"
OUTPUT_MODEL = "tobil/qmd-query-expansion-lfm2-sft"
DATASET = "tobil/qmd-query-expansion-train"


def build_config(**overrides):
    """Construct the real TRL configuration; CPU tests override device flags."""
    from trl import SFTConfig

    kwargs = dict(
        output_dir="qmd-query-expansion-lfm2-sft",
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL,
        hub_strategy="every_save",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=512,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        report_to="none",
    )
    kwargs.update(overrides)
    return SFTConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without authentication, downloads, or training")
    args = parser.parse_args()
    if args.dry_run:
        print({"base_model": BASE_MODEL, "output_model": OUTPUT_MODEL, "dataset": DATASET, "stage": "sft"})
        return

    from datasets import load_dataset
    from huggingface_hub import login
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTTrainer

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    dataset = load_dataset(DATASET, split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")
    # LFM2 is hybrid: attention out_proj/in_proj and SwiGLU w1/w2/w3.
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainer = SFTTrainer(
        model=BASE_MODEL, train_dataset=split["train"], eval_dataset=split["test"],
        args=build_config(), peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    trainer.push_to_hub()
    print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
