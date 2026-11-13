#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers>=5.22.2,<5.23",
#     "pydantic>=2.12.3,<3",
#     "jinja2",
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
"""Prepare QMD query expansion data for LFM2.5-1.2B-Instruct training.

LFM2.5 uses ChatML format:
  <|startoftext|><|im_start|>user
  Expand this search query: {query}<|im_end|>
  <|im_start|>assistant
  {output}<|im_end|>

No /no_think needed (that's Qwen3-specific).
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.schema import normalize_output_items, output_items_to_text

from transformers import AutoTokenizer


def format_for_training(query_text: str, output_items: list[list[str]], tokenizer) -> dict:
    """Format a single example for SFT training using LFM2.5 chat format."""
    output_text = output_items_to_text(output_items)

    messages = [
        {"role": "user", "content": f"Expand this search query: {query_text}"},
        {"role": "assistant", "content": output_text},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return {"text": text}


def main():
    input_path = Path("data/qmd_expansion_v2.jsonl")
    output_dir = Path("data/train-lfm2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LFM2.5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "LiquidAI/LFM2.5-1.2B-Instruct", trust_remote_code=True
    )

    examples = []
    with open(input_path) as f:
        for line in f:
            row = json.loads(line)
            items = normalize_output_items(row["output"])
            example = format_for_training(row["query"], items, tokenizer)
            examples.append(example)

    # Shuffle and split
    random.seed(42)
    random.shuffle(examples)

    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    val = examples[split_idx:]

    # Write as JSONL
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Written {len(train)} train, {len(val)} val examples to {output_dir}")
    print(f"\nSample formatted text:")
    print(train[0]["text"][:500])


if __name__ == "__main__":
    main()
