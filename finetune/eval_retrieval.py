# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers>=5.22.2,<5.23",
#     "peft>=0.21.3,<0.22",
#     "torch>=2.13.0,<3",
#     "accelerate==1.15.0+qmd.2",
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
QMD Retrieval-Based Evaluation with Precision & Recall

Evaluates model outputs against golden data (training set).
Measures how well the model reproduces the expected expansions.

Metrics:
- Precision: Of model-generated expansions, how many match golden?
- Recall: Of golden expansions, how many did the model generate?
- F1: Harmonic mean of precision and recall

Matching is done via token overlap (Jaccard similarity) with a threshold.

Usage:
    uv run eval_retrieval.py ./outputs/sft
    uv run eval_retrieval.py tobil/qmd-query-expansion-1.7B --golden data/qmd_expansion_v3_structured.jsonl
    uv run eval_retrieval.py ./outputs/sft --threshold 0.5 --sample 100
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# =============================================================================
# Matching Functions
# =============================================================================

def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word set, removing stopwords."""
    stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 
                 'or', 'it', 'this', 'that', 'be', 'with', 'as', 'on', 'by',
                 'how', 'what', 'do', 'does', 'can', 'you', 'your', 'i'}
    words = re.findall(r'\b\w+\b', text.lower())
    return {w for w in words if w not in stopwords and len(w) > 1}


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings based on token overlap."""
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def find_best_match(pred: str, golden_list: list[str], threshold: float) -> tuple[str | None, float]:
    """Find best matching golden expansion for a prediction."""
    best_match = None
    best_score = 0.0
    for golden in golden_list:
        score = jaccard_similarity(pred, golden)
        if score > best_score:
            best_score = score
            best_match = golden
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


# =============================================================================
# Parsing
# =============================================================================

def parse_model_output(text: str) -> dict[str, list[str]]:
    """Parse model output into {lex: [...], vec: [...], hyde: [...]}."""
    # Clean thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<|im_end|>', '').strip()
    
    result = {"lex": [], "vec": [], "hyde": []}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            result["lex"].append(line[4:].strip())
        elif line.startswith("vec:"):
            result["vec"].append(line[4:].strip())
        elif line.startswith("hyde:"):
            result["hyde"].append(line[5:].strip())
    return result


def parse_golden_data(searches: list[dict] | str) -> dict[str, list[str]]:
    """Parse golden data format into {lex: [...], vec: [...], hyde: [...]}."""
    # If it's a string (from messages format), parse it
    if isinstance(searches, str):
        return parse_model_output(searches)
    
    # Otherwise it's the structured format [{type, query}, ...]
    result = {"lex": [], "vec": [], "hyde": []}
    for item in searches:
        exp_type = item.get("type", "")
        value = item.get("query", "") or item.get("value", "")
        if exp_type in result:
            result[exp_type].append(value)
    return result


def load_golden_data(filepath: Path) -> list[dict]:
    """Load golden data from JSONL, supporting both structured and messages formats."""
    data = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Structured format: {query, searches}
            if "query" in item and "searches" in item:
                data.append({
                    "query": item["query"],
                    "searches": item["searches"]
                })
            # Messages format: {messages: [{role, content}, ...]}
            elif "messages" in item:
                messages = item["messages"]
                query = None
                searches = None
                for msg in messages:
                    if msg["role"] == "user":
                        # Extract query from "/no_think Expand this search query: ..."
                        content = msg["content"]
                        if "Expand this search query:" in content:
                            query = content.split("Expand this search query:")[-1].strip()
                        else:
                            query = content.strip()
                    elif msg["role"] == "assistant":
                        # The assistant content IS the expected output
                        searches = msg["content"]
                if query and searches:
                    data.append({
                        "query": query,
                        "searches": searches  # Will be parsed as string
                    })
    return data


# =============================================================================
# Metrics Calculation
# =============================================================================

# Different thresholds by type - lex needs strict matching, hyde is more flexible
DEFAULT_THRESHOLDS = {
    "lex": 0.5,   # Keywords should overlap well
    "vec": 0.35,  # Semantic sentences have more variation
    "hyde": 0.25, # Passages have the most variation
}


def calculate_metrics(
    predictions: dict[str, list[str]], 
    golden: dict[str, list[str]], 
    threshold: float | dict[str, float] = 0.4,
    return_mismatches: bool = False
) -> dict:
    """Calculate precision, recall, F1 per type and overall.
    
    Args:
        threshold: Either a single float, or dict mapping type -> threshold
        return_mismatches: If True, include lists of unmatched predictions/golden
    """
    if isinstance(threshold, (int, float)):
        thresholds = {"lex": threshold, "vec": threshold, "hyde": threshold}
    else:
        thresholds = threshold
    
    metrics = {}
    mismatches = {}
    total_tp = 0
    total_pred = 0
    total_golden = 0
    
    for exp_type in ["lex", "vec", "hyde"]:
        preds = predictions.get(exp_type, [])
        golds = golden.get(exp_type, [])
        type_threshold = thresholds.get(exp_type, 0.4)
        
        if not preds and not golds:
            continue
        
        # Track which golden items were matched
        matched_golden = set()
        unmatched_preds = []
        tp = 0
        
        for pred in preds:
            match, score = find_best_match(pred, golds, type_threshold)
            if match is not None:
                tp += 1
                matched_golden.add(match)
            else:
                unmatched_preds.append((pred, score))
        
        unmatched_golden = [g for g in golds if g not in matched_golden]
        
        precision = tp / len(preds) if preds else 0.0
        recall = len(matched_golden) / len(golds) if golds else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[exp_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_count": len(preds),
            "golden_count": len(golds),
            "matched": tp,
        }
        
        if return_mismatches:
            mismatches[exp_type] = {
                "unmatched_preds": unmatched_preds,
                "unmatched_golden": unmatched_golden,
            }
        
        total_tp += tp
        total_pred += len(preds)
        total_golden += len(golds)
    
    # Overall metrics (micro-averaged)
    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_golden if total_golden > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    metrics["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "pred_count": total_pred,
        "golden_count": total_golden,
        "matched": total_tp,
    }
    
    if return_mismatches:
        metrics["_mismatches"] = mismatches
    
    return metrics


# =============================================================================
# Model Loading and Generation
# =============================================================================

def _model_sources(model_path: str) -> tuple[str, dict | None, str]:
    """Keep routing aligned with eval.py without requiring sibling CLI files."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    local = Path(model_path)
    if local.is_dir():
        adapter_file = local / "adapter_config.json"
        adapter_file = adapter_file if adapter_file.is_file() else None
        has_tokenizer = (local / "tokenizer_config.json").is_file()
    else:
        # Only a definite missing *file* means this is not an adapter. A bad
        # repo ID, private/gated repo, offline cache miss or network failure
        # must remain visible, not silently evaluate an unrelated base model.
        try:
            adapter_file = Path(hf_hub_download(model_path, "adapter_config.json", repo_type="model"))
        except RemoteEntryNotFoundError:
            adapter_file = None
        has_tokenizer = False
        if adapter_file is not None:
            try:
                hf_hub_download(model_path, "tokenizer_config.json", repo_type="model")
                has_tokenizer = True
            except RemoteEntryNotFoundError:
                pass

    if adapter_file is None:
        return model_path, None, model_path
    with adapter_file.open(encoding="utf-8") as stream:
        adapter = json.load(stream)
    if not isinstance(adapter, dict):
        raise ValueError(f"Adapter {model_path!r} has invalid adapter_config.json metadata")
    base_model = adapter.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError(f"Adapter {model_path!r} has no valid base_model_name_or_path")
    # Adapter tokenizers can include added tokens/chat-template changes. Use
    # the base tokenizer only when metadata explicitly shows it is absent.
    return base_model, adapter, model_path if has_tokenizer else base_model


def _load_adapter(model, model_path: str):
    """Separate explicitly saved raw layers before PEFT copies either tensor."""
    import torch
    from peft import PeftModel

    separated_head = []

    def separate_raw_layers(module, state_dict, prefix, *unused):
        # PEFT has already installed/remapped modules_to_save wrappers here.
        # Its active copies are independent, but save_embedding_layers can also
        # include raw original_module backups still sharing the base Parameter.
        inputs = module.get_input_embeddings()
        outputs = module.get_output_embeddings()
        if inputs is None or outputs is None:
            return
        inputs = getattr(inputs, "original_module", inputs)
        outputs = getattr(outputs, "original_module", outputs)
        # LoRA can itself wrap an embedding/head. The raw saved weight belongs
        # to its base layer, not to the wrapper's read-only weight property.
        inputs = inputs.get_base_layer() if hasattr(inputs, "get_base_layer") else inputs
        outputs = outputs.get_base_layer() if hasattr(outputs, "get_base_layer") else outputs
        weight = getattr(outputs, "weight", None)
        if weight is None or weight is not getattr(inputs, "weight", None):
            return
        raw_keys = (
            prefix + name + ".weight"
            for name, layer in module.named_modules(remove_duplicate=False)
            if layer is inputs or layer is outputs
        )
        if not any(key in state_dict for key in raw_keys):
            return
        # Even a single saved layer must not overwrite the unsaved counterpart.
        # No tensor comparison or second adapter read is needed, and ordinary
        # LoRA never pays for an embedding-sized clone.
        outputs.weight = torch.nn.Parameter(weight.detach().clone(), requires_grad=weight.requires_grad)
        separated_head.append((outputs, weight))

    hook = None
    # PEFT also supports modules without Transformers embedding accessors.
    get_inputs = getattr(model, "get_input_embeddings", None)
    get_outputs = getattr(model, "get_output_embeddings", None)
    if callable(get_inputs) and callable(get_outputs):
        inputs, outputs = get_inputs(), get_outputs()
        weight = getattr(outputs, "weight", None)
        if weight is not None and weight is getattr(inputs, "weight", None):
            hook = model.register_load_state_dict_pre_hook(separate_raw_layers)
    try:
        return PeftModel.from_pretrained(model, model_path)
    except BaseException:
        # PEFT may modify its input in place before failing. Restore any tying
        # changed by this helper; other PEFT weight changes are not transactional.
        for outputs, weight in separated_head:
            outputs.weight = weight
        raise
    finally:
        if hook is not None:
            hook.remove()


def load_model(model_path: str):
    """Load the exact local/Hub full model, or its declared PEFT base + adapter."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base_model, adapter, tokenizer_source = _model_sources(model_path)
    print(f"Loading base: {base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Preserve tying while loading: tied checkpoints can omit the output head.
    # Only separate layers later if the adapter actually supplies raw weights.
    config = AutoConfig.from_pretrained(base_model)
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

    # Load an explicitly identified adapter, whether local or on the Hub.
    if adapter is not None:
        print(f"Loading adapter: {model_path}", file=sys.stderr)
        model = _load_adapter(model, model_path)

    model.eval()
    return model, tokenizer


def generate_expansion(model, tokenizer, query: str, max_new_tokens: int = 400) -> str:
    """Generate expansion for a single query."""
    import torch
    
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"/no_think Expand this search query: {query}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    gen_tokens = out[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="QMD Retrieval-Based Evaluation")
    parser.add_argument("model", help="Model path (local or HF)")
    parser.add_argument("--golden", default="data/qmd_expansion_v3_structured.jsonl",
                        help="Golden data JSONL file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Jaccard similarity threshold for all types (overrides --type-thresholds)")
    parser.add_argument("--type-thresholds", action="store_true",
                        help="Use type-specific thresholds (lex=0.5, vec=0.35, hyde=0.25)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample N queries (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--max-new-tokens", type=int, default=400,
                        help="Max new tokens to generate")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-query details")
    parser.add_argument("--show-mismatches", action="store_true",
                        help="Show examples of mismatched predictions")
    args = parser.parse_args()
    
    # Determine thresholds
    if args.threshold is not None:
        thresholds = args.threshold
    elif args.type_thresholds:
        thresholds = DEFAULT_THRESHOLDS.copy()
    else:
        thresholds = 0.4  # Default single threshold

    # Load golden data
    golden_path = Path(args.golden)
    if not golden_path.exists():
        # Try relative to script directory
        golden_path = Path(__file__).parent / args.golden
    
    if not golden_path.exists():
        print(f"Error: Golden data file not found: {args.golden}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading golden data from {golden_path}...", file=sys.stderr)
    golden_data = load_golden_data(golden_path)
    print(f"Loaded {len(golden_data)} golden examples", file=sys.stderr)

    # Sample if requested
    if args.sample > 0 and args.sample < len(golden_data):
        random.seed(args.seed)
        golden_data = random.sample(golden_data, args.sample)
        print(f"Sampled {len(golden_data)} examples", file=sys.stderr)

    # Load model
    model, tokenizer = load_model(args.model)

    # Evaluate
    all_metrics = []
    all_mismatches = []
    type_aggregates = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
    
    threshold_desc = thresholds if isinstance(thresholds, (int, float)) else f"lex={thresholds['lex']}, vec={thresholds['vec']}, hyde={thresholds['hyde']}"
    print(f"\nEvaluating {len(golden_data)} queries (thresholds: {threshold_desc})...\n")
    
    for i, item in enumerate(golden_data, 1):
        query = item["query"]
        golden_parsed = parse_golden_data(item["searches"])
        
        # Generate model output
        output = generate_expansion(model, tokenizer, query, args.max_new_tokens)
        pred_parsed = parse_model_output(output)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_parsed, golden_parsed, thresholds, return_mismatches=args.show_mismatches)
        all_metrics.append({"query": query, "metrics": metrics, "pred": pred_parsed, "golden": golden_parsed})
        
        if args.show_mismatches and "_mismatches" in metrics:
            all_mismatches.append({"query": query, "mismatches": metrics.pop("_mismatches")})
        
        # Aggregate by type
        for exp_type in ["lex", "vec", "hyde", "overall"]:
            if exp_type in metrics:
                type_aggregates[exp_type]["precision"].append(metrics[exp_type]["precision"])
                type_aggregates[exp_type]["recall"].append(metrics[exp_type]["recall"])
                type_aggregates[exp_type]["f1"].append(metrics[exp_type]["f1"])
        
        # Progress
        overall = metrics.get("overall", {})
        p = overall.get("precision", 0) * 100
        r = overall.get("recall", 0) * 100
        f = overall.get("f1", 0) * 100
        
        if args.verbose:
            print(f"[{i:3d}/{len(golden_data)}] P={p:5.1f}% R={r:5.1f}% F1={f:5.1f}%  {query[:50]}")
        elif i % 50 == 0 or i == len(golden_data):
            print(f"  Processed {i}/{len(golden_data)}...", file=sys.stderr)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"Threshold: {args.threshold} | Samples: {len(golden_data)}")
    print()
    
    print(f"{'Type':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)
    
    for exp_type in ["lex", "vec", "hyde", "overall"]:
        if exp_type in type_aggregates:
            agg = type_aggregates[exp_type]
            avg_p = sum(agg["precision"]) / len(agg["precision"]) * 100 if agg["precision"] else 0
            avg_r = sum(agg["recall"]) / len(agg["recall"]) * 100 if agg["recall"] else 0
            avg_f = sum(agg["f1"]) / len(agg["f1"]) * 100 if agg["f1"] else 0
            label = exp_type.upper() if exp_type != "overall" else "OVERALL"
            print(f"{label:<10} {avg_p:>9.1f}% {avg_r:>9.1f}% {avg_f:>9.1f}%")
    
    print(f"{'='*60}")
    
    # Show worst examples
    print("\nBottom 5 by F1:")
    sorted_by_f1 = sorted(all_metrics, key=lambda x: x["metrics"].get("overall", {}).get("f1", 0))
    for item in sorted_by_f1[:5]:
        f1 = item["metrics"].get("overall", {}).get("f1", 0) * 100
        print(f"  {f1:5.1f}%  {item['query'][:60]}")
    
    # Show mismatches if requested
    if args.show_mismatches and all_mismatches:
        print(f"\n{'='*60}")
        print("MISMATCH EXAMPLES")
        print(f"{'='*60}")
        
        # Group by type and show up to 3 examples per type
        for exp_type in ["lex", "vec", "hyde"]:
            type_mismatches = []
            for item in all_mismatches:
                if exp_type in item["mismatches"]:
                    mm = item["mismatches"][exp_type]
                    if mm["unmatched_preds"] or mm["unmatched_golden"]:
                        type_mismatches.append({
                            "query": item["query"],
                            **mm
                        })
            
            if type_mismatches:
                print(f"\n--- {exp_type.upper()} mismatches ({len(type_mismatches)} queries) ---")
                for example in type_mismatches[:3]:
                    print(f"\nQuery: {example['query'][:60]}")
                    if example["unmatched_preds"]:
                        print(f"  Unmatched predictions:")
                        for pred, score in example["unmatched_preds"][:2]:
                            print(f"    - [{score:.2f}] {pred[:80]}{'...' if len(pred) > 80 else ''}")
                    if example["unmatched_golden"]:
                        print(f"  Missing golden:")
                        for g in example["unmatched_golden"][:2]:
                            print(f"    - {g[:80]}{'...' if len(g) > 80 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
