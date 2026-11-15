#!/usr/bin/env python3
"""Compare saved real BERT embeddings with the same local HF model on CPU.

This is a same-model numerical reference gate, not a retrieval-quality benchmark.
Weights must already exist locally. Outputs are never downloaded or fabricated.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pooling', choices=['mean', 'cls'], default='mean')
    args = parser.parse_args()
    if not args.model.is_absolute() or not (args.model/'config.json').is_file():
        parser.error('model must be an existing absolute local directory')
    if not args.vectors.is_file() or args.vectors.stat().st_size > 10 * 1024**2:
        parser.error('vectors must be an existing bounded NPZ fixture')
    if args.output.exists():
        parser.error('output already exists; preserve old evidence and use a new path')
    for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']:
        os.environ[key] = '1'
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    from bench_apple_silicon import workload

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False).eval()
    batch = tokenizer(workload(), padding=True, truncation=False, return_tensors='pt')
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state.float()
        mask = batch['attention_mask'].unsqueeze(-1).float()
        pooled = hidden[:, 0] if args.pooling == 'cls' else (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        reference = torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()
    with np.load(args.vectors, allow_pickle=False) as values:
        actual = values['mixed']
    if actual.shape != reference.shape:
        raise ValueError(f'Reference shape {reference.shape} != candidate {actual.shape}')
    cosines = (reference * actual).sum(1)
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    # A dot product is a cosine only when both inputs are normalized.
    normalized = bool(np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-4))
    report = {
        'reference': 'Hugging Face AutoModel, same local weights, CPU float32',
        'model': str(args.model.resolve()), 'candidate_vectors': str(args.vectors.resolve()),
        'pooling': args.pooling, 'token_counts': batch['attention_mask'].sum(1).tolist(),
        'finite': finite, 'normalized': normalized, 'cosines': cosines.tolist(),
        'min_cosine': float(cosines.min()), 'max_abs': float(np.max(np.abs(reference - actual))),
        'passed': finite and normalized and bool(cosines.min() > 0.9999),
        'scope': 'Authored four-text numerical fixture only; not corpus retrieval quality',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
