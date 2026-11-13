# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers>=5.22.2,<5.23",
#     "peft>=0.21.3,<0.22",
#     "torch>=2.13.0,<3",
#     "accelerate>=1.15.0,<1.16",
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
Minimal QMD query expansion evaluator.

Usage:
    uv run eval.py ./outputs/sft
    uv run eval.py ./outputs/sft --queries evals/queries.txt

By default, query file defaults to evals/queries.txt and runs all queries unless --max-queries is set.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Import reward scoring
sys.path.insert(0, str(Path(__file__).parent))
from reward import score_expansion_detailed



DEFAULT_QUERY_FILE = Path(__file__).parent / "evals" / "queries.txt"


def load_model(model_path: str):
    """Load model (adapter or merged)."""
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"

    # Get base model from adapter config or default
    base_model = "Qwen/Qwen3-1.7B"
    if adapter_config.exists():
        with open(adapter_config) as f:
            cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path", base_model)

    print(f"Loading base: {base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(base_model)
    config.tie_word_embeddings = False
    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map={"": 0 if use_cuda else "cpu"}, config=config,
    )
    if model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    # Load adapter if present
    if adapter_config.exists():
        print(f"Loading adapter: {model_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, str(model_path))

    model.eval()
    return model, tokenizer


def generate_batch(
    model, tokenizer, queries: list[str], max_new_tokens: int, max_time: float | None
) -> list[str]:
    """Generate expansions for a batch of queries."""
    import torch

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"/no_think Expand this search query: {q}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in queries
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if max_time and max_time > 0:
        generate_kwargs["max_time"] = max_time

    with torch.inference_mode():
        out = model.generate(**inputs, **generate_kwargs)

    outputs = []
    for i in range(len(queries)):
        gen_tokens = out[i][input_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        outputs.append(text.strip())

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate QMD model")
    parser.add_argument("model", help="Model path (local or HF)")
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERY_FILE),
        help="Queries file (one per line) [default: evals/queries.txt]",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Maximum new tokens to generate (default: 400)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=0,
        help="Max seconds per batch generation (0 disables)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for generation (default: 2)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Limit number of queries (0 disables)",
    )
    args = parser.parse_args()

    # Load queries (default to full evals/queries.txt)
    query_file = Path(args.queries)
    if not query_file.exists():
        raise FileNotFoundError(f"Queries file not found: {query_file}")
    with query_file.open(encoding="utf-8") as f:
        queries = [
            l.strip() for l in f if l.strip() and not l.strip().startswith("#")
        ]

    if args.max_queries and args.max_queries > 0:
        queries = queries[: args.max_queries]

    # Load model
    model, tokenizer = load_model(args.model)

    # Run eval
    scores = []
    batch_size = max(1, args.batch_size)
    total = len(queries)
    for start in range(0, total, batch_size):
        batch = queries[start : start + batch_size]
        batch_outputs = generate_batch(
            model, tokenizer, batch, args.max_new_tokens, args.max_time
        )
        for i, (query, expansion) in enumerate(zip(batch, batch_outputs), start + 1):
            print(f"\n[{i}/{total}] {query}")
            print("-" * 50)
            result = score_expansion_detailed(query, expansion)
            print(expansion[:300] + ("..." if len(expansion) > 300 else ""))
            print(f"Score: {result['percentage']:.0f}% ({result['rating']})")
            scores.append(result["percentage"])

    # Summary
    avg = sum(scores) / len(scores)
    print(f"\n{'=' * 50}")
    print(f"Average: {avg:.1f}%  |  Model: {args.model}")
    print(f"{'=' * 50}")

    return 0 if avg >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
