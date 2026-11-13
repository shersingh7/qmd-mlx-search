# Apple Silicon encoder measurements

## Scope and provenance

Measured on the audit's **Apple M2 Pro, 32 GiB** host using existing local MiniLM and BGE-small checkpoints, offline, in serialized processes. The eight runtime runs cover five configurations; five first-run vector artifacts were independently compared with the same Hugging Face model on CPU. Repeats compare against their already-reference-qualified vector artifacts. This is an encoder benchmark, not whole-search latency or a representative retrieval-quality evaluation.

- Python source fingerprint: `282941577d8d37d53259b8a721063c0bf5c290a01140a135812220a3bf29de93`
- Benchmark fingerprint: `8ec241c1e1d07635009ca434a5a9808b1a487bd0bcd936c75b23acaaf32a217c`
- Dependency versions captured by the benchmark: `{"mlx": "0.32.4", "mlx-lm": "0.32.2", "numpy": "2.5.4", "transformers": "5.22.2"}`
- Harness limits: `{"metal_cache_bytes": 536870912, "metal_memory_bytes": 2147483648}`
- Raw reports/vectors: `rc-core-*.json`, `rc-core-*.npz` and the corresponding `*-reference.json` files in this directory.
- Machine-readable aggregation: [rc-summary.json](rc-summary.json); spreadsheet-friendly data: [rc-summary.csv](rc-summary.csv).

## Results

| Encoder | Compute / bulk target | Runs | 64-text batch p50, ms | Active Metal, MiB | Peak active Metal, MiB | Min reference cosine |
|---|---|---:|---:|---:|---:|---:|
| minilm | float32 / 512 | 2 | 85.6–85.8 | 86.7 | 177.6 | 0.999999881 |
| minilm | float16 / 512 | 2 | 77.0–77.8 | 43.3 | 130.0 | 0.999999046 |
| minilm | float16 / 4096 | 2 | 61.6–61.9 | 43.3 | 379.7 | 0.999999046 |
| minilm | bfloat16 / 512 | 1 | 86.5 (one run) | 43.3 | 130.0 | 0.999947309 |
| bge | float16 / 512 | 1 | 131.9 (one run) | 63.6 | 190.9 | 0.999999285 |

The MiniLM configurations were repeated in reverse order. Ranges show the two run medians, not confidence intervals. BF16 and BGE have one runtime run each.

For this fixture, the explicit **FP16 / 4096-token bulk profile** delivered **1.39× throughput** relative to the corrected FP32 / 512-token profile (ratio of median run latencies), or **27.9% lower batch latency**. FP16 reduced retained active Metal allocations by **50.0%**. This compares precision/batching configurations, not a claimed code-only speedup or MLX-versus-GGUF comparison.

**Tradeoff:** the larger bulk target increases peak active allocations and lengthens non-preemptible forwards. The default remains 512 rather than assuming these small-encoder results justify the same budget for large models. Cache allocation and process RSS are separate from active Metal; inspect the raw reports rather than treating this table as total machine memory usage. Single-query timings vary with warm state and are not used to claim an interactive speedup.

## Correctness and teardown

Every run passed its finite-value, normalization, mixed/singleton agreement and owned-executor cleanup gates. All final active Metal **and allocator-cache** readings were **zero bytes** after device synchronization, object collection and cache clearing. Immediate pre-barrier readings are retained separately. This does not prove the absence of every Python/native allocation leak or long-run memory growth.

The original BERT path omitted learned segment-zero embeddings and failed the independent numerical reference. Those old timings are retained as historical diagnostics, **not** correctness-preserving performance evidence. The measurements above use the repaired segment inputs, compute-compatible attention masks and float32 pooling/normalization.

## Opt-in MiniLM bulk profile

After verifying the checkpoint is cached and rebuilding a separate compatible index:

```sh
python scripts/mlx_embed_server.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --dtype float16 \
  --max-bulk-microbatch-tokens 4096 \
  --preload
```

This is a configuration example, not an installation or production cutover performed by the audit. Precision changes alter embedding-space identity. Do not reuse or relabel old vectors to bypass a mismatch.

## Explicit limits

- No real 4B embedding, generative-query-expansion or reranker model qualification was performed.
- No cross-M-series, GGUF comparative, full-corpus relevance or extended-soak conclusion follows from these fixtures.
- Recovery/qualification unit tests using authored one-hot vectors validate orchestration only; they are not included as model-quality or performance measurements.
- HTTP/SDK/MCP integration and installed-package acceptance are separate gates recorded in the overall audit report.
