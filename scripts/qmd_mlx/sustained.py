"""
sustained.py — Bounded Sustained MLX Embedding Qualification Harness

Reuses SmokeRunner / MLXWatchdog owned process composition to execute
bounded pilot (<=30 requests, <=120s) and bounded soak (<=100 requests, <=180s)
benchmarks across four verified token strata:
  - Short:  5 – 25 tokens (interactive queries)
  - Medium: 50 – 200 tokens (passages)
  - Long:   400 – 1500 tokens (documents)
  - Code:   50 – 500 tokens (syntax blocks)
  - 2048 boundary policy verification

Guarantees:
  1. Cold startup measured separately; warmup excluded from metrics.
  2. Latency percentiles computed via explicit linear interpolation (small samples marked descriptive).
  3. Wire end-to-end vs forward pass distinction.
  4. Raw-input tokens/s uses verified tokenizer counts; unavailable otherwise (including synthetic rehearsals).
  5. Process RSS/swap sampled by the watchdog; allocator snapshots must be complete to qualify.
  6. Conservative batch progression (1 -> 2 -> 4) and interactive interleaving.
  7. Absolute monotonic wall-clock deadlines and guaranteed watchdog teardown.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Optional
import numpy as np

from .adapters.embedding import (
    estimate_model_memory_mb,
    infer_model_params_b,
)
from .smoke import (
    NORM_TOLERANCE,
    SmokeHttpClient,
    check_dimensions,
    check_finite,
    check_l2_normalization,
    inspect_model_metadata,
)
from .watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    WatchdogCheckResult,
    is_numeric_loopback,
)


# --- Public Public Fixtures ---

SHORT_FIXTURES = [
    "Fast vector search with MLX on Apple Silicon Metal.",
    "What is the latency of semantic retrieval on unified memory?",
    "Locate the definition of EmbeddingDescriptor in contract module.",
    "How does BM25 and vector hybrid search rank documents?",
    "High-bandwidth shared unified memory across CPU and GPU cores.",
]

MEDIUM_FIXTURES = [
    (
        "Apple Silicon Metal unified memory architecture provides high-bandwidth shared memory access "
        "across CPU and GPU execution contexts. In contrast to discrete PCIe-attached accelerators with explicit DMA copies, "
        "Apple Silicon unified memory allows host tokenization buffers and Metal command encoders to operate on shared physical "
        "memory pools without PCIe bus serialization bottlenecks or data transfer overhead."
    ),
    (
        "Matryoshka Representation Learning enables embedding models to produce flexible representation dimensions within a single forward pass. "
        "By optimizing loss across truncated prefixes of the hidden embedding vector (such as 512, 1024, or 2560 dimensions), "
        "downstream vector indices can trade off storage footprint and cosine similarity precision without retraining or maintaining multiple separate models."
    ),
    (
        "Durable index recovery guarantees that interrupted bulk indexing operations resume cleanly with zero duplicated vector rows "
        "and deterministic document count reconciliation across all collections. Checkpoint transactions record document content hashes, chunk boundaries, "
        "and vector IDs in SQLite before committing updates to the primary full-text search and vector indexes on Apple Silicon hardware architecture."
    ),
]

LONG_DOC_BASE = (
    "Apple Silicon unified memory architecture integrates CPU, GPU, and Neural Engine cores onto a single system-on-chip (SoC) "
    "with a unified memory fabric. This physical topology eliminates the traditional memory separation between system host RAM and GPU VRAM. "
    "In machine learning inference workloads, this design allows large weight tensors to remain resident in physical RAM while Metal compute "
    "pipelines execute matrix multiplications directly in-place. "
    "Furthermore, memory pressure management requires explicit accounting of resident set size (RSS), wired memory allocations, "
    "and inactive purgeable file cache pages to prevent unexpected swap thrashing or process termination under heavy multi-tasking workloads.\n\n"
    "When evaluating embedding performance, end-to-end latency must be decomposed into distinct execution phases: host CPU tokenization, "
    "tensor buffer allocation, Metal command buffer encoding, GPU backbone execution, pooling operations, and L2 normalization. "
    "In causal architectures such as Qwen, last-token pooling requires indexing the hidden state at the exact position of the last non-padding "
    "token for each sequence in the batch. Padded tokens introduced to uniformize batch dimensions must not contaminate the extracted vector. "
    "Casting intermediate activations to float32 before normalization preserves numerical stability across diverse sequence length distributions.\n\n"
)

# Repeat base text to build ~500-1000 token long documents
LONG_FIXTURES = [
    LONG_DOC_BASE * 3,
    LONG_DOC_BASE * 4,
]

CODE_FIXTURES = [
    (
        "```python\n"
        "def process_embedding_batch(texts: list[str], max_batch_tokens: int = 4096, dims: int = 2560) -> np.ndarray:\n"
        "    \"\"\"Plans micro-batches and executes forward pass with L2 normalization across Apple Silicon Metal unified memory.\"\"\"\n"
        "    batches: list[list[str]] = []\n"
        "    current_batch: list[str] = []\n"
        "    current_tokens: int = 0\n"
        "    for text in texts:\n"
        "        token_len = len(text.split())\n"
        "        if current_batch and current_tokens + token_len > max_batch_tokens:\n"
        "            batches.append(current_batch)\n"
        "            current_batch, current_tokens = [text], token_len\n"
        "        else:\n"
        "            current_batch.append(text)\n"
        "            current_tokens += token_len\n"
        "    if current_batch:\n"
        "        batches.append(current_batch)\n"
        "    return np.vstack([forward_model(b) for b in batches])\n"
        "```"
    ),
    (
        "```typescript\n"
        "export function verifyEmbeddingCompatibility(\n"
        "  expected: EmbeddingDescriptor,\n"
        "  actual: EmbeddingDescriptor,\n"
        "  context?: string\n"
        "): boolean {\n"
        "  if (expected.outputDimensions !== actual.outputDimensions) return false;\n"
        "  if (expected.normalized !== actual.normalized) return false;\n"
        "  if (expected.pooling !== actual.pooling) return false;\n"
        "  const idExpected = computeEmbeddingSpaceId(expected);\n"
        "  const idActual = computeEmbeddingSpaceId(actual);\n"
        "  if (idExpected !== idActual) {\n"
        "    throw new Error(`Embedding space mismatch: expected ${idExpected}, got ${idActual}`);\n"
        "  }\n"
        "  return true;\n"
        "}\n"
        "```"
    ),
]

def compute_code_manifest(repo_root: str) -> dict[str, str]:
    """
    Computes exact SHA256 fingerprints for all relevant qualification and runtime scripts,
    including untracked and tracked files.
    """
    manifest_files = [
        "scripts/qmd-mlx-sustained.py",
        "scripts/qmd_mlx/sustained.py",
        "scripts/qmd_mlx/server.py",
        "scripts/qmd_mlx/runtime.py",
        "scripts/qmd_mlx/watchdog.py",
        "scripts/qmd_mlx/executor.py",
        "scripts/qmd_mlx/batching.py",
        "scripts/qmd_mlx/model_manager.py",
        "scripts/qmd_mlx/adapters/embedding.py",
        "scripts/qmd_mlx/adapters/tokenization.py",
        "test/python/test_mlx_sustained.py",
        "test/python/test_audit_qualification.py",
    ]
    manifest: dict[str, str] = {}
    for rel_path in manifest_files:
        full_path = os.path.join(repo_root, rel_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "rb") as f:
                    manifest[rel_path] = hashlib.sha256(f.read()).hexdigest()
            except Exception as e:
                manifest[rel_path] = f"error: {e}"
    return manifest


# These are bounded qualification criteria, not production/retrieval release gates.
RUN_BOUNDS = {"pilot": (120.0, 30), "soak": (180.0, 100)}
MAX_COLD_METAL_GROWTH_MB = 500.0  # Existing provisional cold-allocation allowance.
MAX_WARM_METAL_GROWTH_MB = 0.0  # No retained growth after warmup.
MAX_QUALIFICATION_SWAP_GROWTH_MB = 0.0  # Stricter than watchdog termination budget.
MIN_FREE_MEMORY_PCT = 12.0
TELEMETRY_ENDPOINT_TOLERANCE_S = 0.05  # Persisted wall-summary precision, not a sampling-period allowance.
STRATUM_BOUNDS = {
    "short": (5, 25), "medium": (50, 200), "long": (400, 1500),
    "code": (50, 500), "boundary_2047": (2047, 2047),
    "boundary_2048": (2048, 2048),
}


def _finite_number(value: Any, minimum: Optional[float] = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and (minimum is None or value >= minimum)
    except (OverflowError, ValueError):
        return False


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_benchmark_bounds(mode: str, timeout_s: float, max_requests: int,
                              min_headroom_mb: float, *, pilot_only: bool = False) -> None:
    """Validate library and CLI inputs identically, before sampling or spawning."""
    if not isinstance(mode, str) or mode not in RUN_BOUNDS or (pilot_only and mode != "pilot"):
        raise ValueError("GGUF supports pilot mode only; no soak loop is implemented" if pilot_only
                         else "Mode must be 'pilot' or 'soak'")
    max_timeout, request_limit = RUN_BOUNDS[mode]
    if not _finite_number(timeout_s) or not 0 < timeout_s <= max_timeout:
        raise ValueError(f"{mode} timeout_s must be finite and in (0, {max_timeout}] seconds")
    if not _positive_int(max_requests) or max_requests > request_limit:
        raise ValueError(f"{mode} max_requests must be an integer in [1, {request_limit}]")
    if not _finite_number(min_headroom_mb) or min_headroom_mb <= 0:
        raise ValueError("min_headroom_mb must be finite and positive")


def _valid_counts(counts: Any, item_count: int) -> bool:
    return (_positive_int(item_count) and isinstance(counts, list) and len(counts) == item_count
            and all(_positive_int(count) for count in counts))


def fetch_token_counts(client: SmokeHttpClient, base_url: str, texts: list[str],
                       deadline: float) -> Optional[list[int]]:
    """Tokenize raw request texts; server-added instructions/padding are not counted."""
    try:
        status, payload = client.post_json(f"{base_url}/tokenize", {"texts": texts}, deadline=deadline)
    except Exception:
        return None
    counts = payload.get("counts") if isinstance(payload, dict) else None
    return counts if status == 200 and _valid_counts(counts, len(texts)) else None


def _rate(count: Any, wire_ms: Any) -> Optional[float]:
    if not _finite_number(count, 0) or not _finite_number(wire_ms) or wire_ms <= 0:
        return None
    result = count / (wire_ms / 1000.0)
    return round(result, 1) if _finite_number(result) else None


def format_metric(value: Any, precision: int = 1) -> str:
    return f"{value:.{precision}f}" if _finite_number(value) else "unavailable"


def request_token_metrics(texts: list[str], counts: Optional[list[int]], wire_ms: float,
                          *, synthetic: bool = False) -> dict[str, Any]:
    """Synthetic whitespace tokenizers exercise plumbing, not real model token throughput."""
    verified = not synthetic and isinstance(counts, list) and _valid_counts(counts, len(texts))
    total = sum(counts) if verified else None
    words = [len(text.split()) for text in texts]
    return {
        "token_counts": list(counts) if verified else None,
        "total_tokens": total,
        "token_count_source": "tokenizer" if verified else "unavailable",
        "token_count_scope": "raw_request_text; server-added instructions and padding excluded",
        "effective_tokens_per_sec": _rate(total, wire_ms),
        "word_counts": words, "total_words": sum(words),
        "effective_words_per_sec": _rate(sum(words), wire_ms),
    }


def _record_tokens(record: dict[str, Any]) -> Optional[int]:
    counts = record.get("token_counts")
    if (isinstance(counts, list) and record.get("token_count_source") == "tokenizer"
            and _valid_counts(counts, record.get("batch_size", 0))
            and all(count <= 2048 for count in counts)
            and _positive_int(record.get("total_tokens"))
            and record["total_tokens"] == sum(counts)):
        return record["total_tokens"]
    return None


def compute_explicit_percentiles(values: list[Any]) -> dict[str, Any]:
    """Linear interpolation, descriptive at N<=30; missing/invalid is never zero latency."""
    complete = bool(values) and all(_finite_number(value, 0) for value in values)
    result = {"count": len(values), "complete": complete,
              "descriptive_only": len(values) <= 30,
              **{name: None for name in ("min", "p50", "p95", "max", "mean")}}
    if complete:
        arr = np.array(values, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            computed = {"min": float(np.min(arr)), "max": float(np.max(arr)),
                        "p50": float(np.percentile(arr, 50, method="linear")),
                        "p95": float(np.percentile(arr, 95, method="linear")),
                        "mean": float(np.mean(arr))}
        if all(_finite_number(value) for value in computed.values()):
            result.update({name: round(value, 2) for name, value in computed.items()})
        else:
            result["complete"] = False
    return result


def calibrate_boundary_fixtures(
    client: SmokeHttpClient,
    base_url: str,
    deadline: float,
) -> dict[str, str]:
    """
    Calibrates exact 2047, 2048, and 2049 token texts by querying the active /tokenize endpoint.
    Guarantees:
      - boundary_2047 encodes to exactly 2047 tokens (including special tokens).
      - boundary_2048 encodes to exactly 2048 tokens (including special tokens).
      - boundary_2049 encodes to exactly 2049 tokens (including special tokens), rejected with HTTP 400.
    """
    def calibrate(target: int, words: list[str]) -> tuple[str, list[str]]:
        for _ in range(21):
            text = " ".join(words)
            counts = fetch_token_counts(client, base_url, [text], deadline)
            if counts is None:
                raise RuntimeError(f"Tokenizer evidence unavailable while calibrating boundary {target}")
            count = counts[0]
            if count == target:
                return text, words
            difference = target - count
            words = words + ["token"] * difference if difference > 0 else words[:max(1, len(words) + difference)]
        raise RuntimeError(f"Could not calibrate exact {target}-token boundary")

    text_2047, words = calibrate(2047, ["token"] * 2040)
    text_2048, words = calibrate(2048, words + ["token"])
    return {
        "boundary_2047": text_2047,
        "boundary_2048": text_2048,
        # The caller must verify the exact rejected length from the tokenizer error.
        "boundary_2049": " ".join(words + ["token"]),
    }


def extract_watchdog_breach_dict(b: WatchdogCheckResult) -> dict[str, Any]:
    """Formats a structured snapshot of a primary watchdog breach."""
    return {
        "breach_type": b.breach_type.value if b.breach_type else "unknown",
        "breach_reason": b.breach_reason or "Watchdog breach triggered",
        "timestamp": round(b.timestamp, 3),
        "metrics": b.metrics or {},
        "terminated_pid": b.terminated_pid,
        "sigkill_used": b.sigkill_used,
    }


def _tagged_requests(records: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Include every raw tagged request, even malformed attempts that cannot qualify."""
    return [row for row in records if isinstance(row.get("tag"), str) and row["tag"].startswith(prefix)]


def compute_metrics_summary(
    measured_records: list[dict[str, Any]],
    rejection_records: list[dict[str, Any]],
    concurrent_interleaving_records: list[dict[str, Any]],
    solo_baseline_percentiles: Optional[dict[str, Any]],
    total_benchmark_elapsed_s: float,
    partial_run: bool = False,
) -> dict[str, Any]:
    """
    Computes complete, typed, finite metrics aggregations across all attempted/measured requests.
    Supports partial runs on failure without fabricating data.
    """
    successful_records = [r for r in measured_records if r.get("rejection_verified") is not True]
    all_successful_latencies = [r.get("wire_ms") for r in successful_records]
    token_totals = [_record_tokens(record) for record in successful_records]
    tokens_complete = bool(token_totals) and all(value is not None for value in token_totals)
    total_tokens = sum(token_totals) if tokens_complete else None

    def sum_wire(records: list[dict[str, Any]]) -> Optional[float]:
        values = [record.get("wire_ms") for record in records]
        if not all(_finite_number(value, 0) for value in values):
            return None
        total = sum(values)
        return total if _finite_number(total) else None

    successful_wire_ms = sum_wire(successful_records)
    rejection_wire_ms = sum_wire(rejection_records)
    latency_complete = bool(successful_records) and all(
        _finite_number(value) and value > 0 for value in all_successful_latencies
    )
    word_totals = [record.get("total_words") for record in successful_records]
    words_complete = bool(word_totals) and all(_positive_int(value) for value in word_totals)

    by_stratum = {}
    for stratum in [*STRATUM_BOUNDS, "mixed"]:
        values = [record.get("wire_ms") for record in successful_records if record.get("stratum") == stratum]
        if values:
            by_stratum[stratum] = compute_explicit_percentiles(values)
    by_batch_size = {}
    for batch_size in (1, 2, 4):
        values = [record.get("wire_ms") for record in successful_records if record.get("batch_size") == batch_size]
        if values:
            by_batch_size[f"batch_{batch_size}"] = compute_explicit_percentiles(values)

    def concurrent_percentiles(key: str) -> dict[str, Any]:
        return compute_explicit_percentiles([record.get(key) for record in concurrent_interleaving_records])

    slowdowns = [record.get("slowdown_factor") for record in concurrent_interleaving_records]
    slowdown_mean = compute_explicit_percentiles(slowdowns)["mean"]
    return {
        "total_measured_requests": len(measured_records) + len(rejection_records),
        "successful_embedding_requests": len(successful_records),
        "rejected_boundary_requests": len(rejection_records),
        "total_successful_embedded_tokens": total_tokens,
        "known_successful_embedded_tokens": sum(value for value in token_totals if value is not None),
        "token_metrics_complete": tokens_complete,
        "token_counts_unavailable_requests": sum(value is None for value in token_totals),
        "latency_metrics_complete": latency_complete,
        "overall_effective_tokens_per_sec": _rate(total_tokens, successful_wire_ms) if latency_complete else None,
        "throughput_denominator": "sum_successful_request_wire_s; overlapping requests counted separately, not wall sustained throughput",
        "overall_effective_words_per_sec": _rate(sum(word_totals), successful_wire_ms) if words_complete and latency_complete else None,
        "partial_run": partial_run,
        "cumulative_wall_accounting": {
            "total_benchmark_elapsed_s": round(total_benchmark_elapsed_s, 2) if _finite_number(total_benchmark_elapsed_s, 0) else None,
            "sum_successful_request_wire_s": round(successful_wire_ms / 1000.0, 3) if successful_wire_ms is not None else None,
            "sum_rejection_request_wire_s": round(rejection_wire_ms / 1000.0, 3) if rejection_wire_ms is not None else None,
            "forwarded_vs_wire_note": "Client wire end-to-end latency; forward, tokenization and padding overhead unavailable separately.",
        },
        "overall_successful_latency_ms": compute_explicit_percentiles(all_successful_latencies),
        "expected_rejections_latency_ms": compute_explicit_percentiles([r.get("wire_ms") for r in rejection_records]),
        "by_stratum_latency_ms": by_stratum,
        "by_batch_size_latency_ms": by_batch_size,
        "concurrent_tail_latency_ms": {
            "solo_baseline_interactive": solo_baseline_percentiles or compute_explicit_percentiles([]),
            # Receipts prove overlap; they must never select the latency population.
            "concurrent_interactive_under_load": compute_explicit_percentiles([
                row.get("wire_ms") for row in _tagged_requests(measured_records, "concurrent_interactive_query_")]),
            "concurrent_bulk_batches": compute_explicit_percentiles([
                row.get("wire_ms") for row in _tagged_requests(measured_records, "concurrent_bulk_batch_")]),
            "estimated_queue_wait_ms": concurrent_percentiles("estimated_queue_wait_ms"),
            "mean_slowdown_factor": slowdown_mean,
        },
        "padding_overhead": "unavailable",
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _same_number(left: Any, right: Any, tolerance: float = 0.02) -> bool:
    return _finite_number(left) and _finite_number(right) and abs(left - right) <= tolerance


def _same_typed_evidence(actual: Any, expected: Any) -> bool:
    """JSON's bool/int equality must not make fabricated numeric metrics pass."""
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return _finite_number(actual) and actual == expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _same_typed_evidence(actual[key], value)
                                                for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_typed_evidence(left, right) for left, right in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def _valid_memory_samples(samples: Any) -> bool:
    if not isinstance(samples, list) or len(samples) < 2:
        return False
    keys = ("timestamp", "elapsed_s", "rss_mb", "swap_used_mb", "swap_growth_mb", "memory_free_pct")
    if not all(isinstance(sample, dict) and all(_finite_number(sample.get(key), 0) for key in keys)
               and sample["rss_mb"] > 0 and sample["memory_free_pct"] <= 100 for sample in samples):
        return False
    return all(after["elapsed_s"] > before["elapsed_s"] and after["timestamp"] > before["timestamp"]
               for before, after in zip(samples, samples[1:]))


def _memory_samples_cover_run(samples: Any, cold_elapsed_s: Any, elapsed_s: Any) -> bool:
    """Process observations must bracket startup completion and the measured finish."""
    return (_valid_memory_samples(samples) and _finite_number(cold_elapsed_s, 0)
            and _finite_number(elapsed_s) and elapsed_s > 0
            and cold_elapsed_s <= elapsed_s + TELEMETRY_ENDPOINT_TOLERANCE_S
            and samples[0]["elapsed_s"] <= cold_elapsed_s
            and elapsed_s - TELEMETRY_ENDPOINT_TOLERANCE_S <= samples[-1]["elapsed_s"]
            <= elapsed_s + TELEMETRY_ENDPOINT_TOLERANCE_S)


def _valid_memory_snapshot(payload: Any) -> bool:
    """An error body is not allocator evidence, even if it includes finite metrics."""
    return (isinstance(payload, dict) and payload.get("error") is None
            and _finite_number(payload.get("active_mb"), 0)
            and _finite_number(payload.get("peak_mb"), 0)
            and payload["peak_mb"] >= payload["active_mb"])


def compute_memory_assessment(memory: dict[str, Any], samples: list[dict[str, Any]],
                              baseline_swap_mb: Any) -> dict[str, Any]:
    """Never turn absent allocator/process samples into zero growth or a stable verdict."""
    stages = ("initial", "post_warmup", "final")
    snapshots = [_mapping(memory.get(key)) for key in stages]
    statuses = _mapping(memory.get("http_status"))
    snapshots_complete = all(type(statuses.get(key)) is int and statuses[key] == 200 for key in stages)
    snapshots_complete = snapshots_complete and all(_valid_memory_snapshot(item) for item in snapshots)
    snapshots_complete = snapshots_complete and all(after["peak_mb"] >= before["peak_mb"]
                                                    for before, after in zip(snapshots, snapshots[1:]))
    metal_available = snapshots_complete and memory.get("measurement_kind") == "metal_allocator"
    process_complete = _valid_memory_samples(samples) and _finite_number(baseline_swap_mb, 0)
    delta = snapshots[2]["active_mb"] - snapshots[0]["active_mb"] if metal_available else None
    warm_delta = snapshots[2]["active_mb"] - snapshots[1]["active_mb"] if metal_available else None
    peak = max(item["peak_mb"] for item in snapshots) if metal_available else None
    rss_growth = round(samples[-1]["rss_mb"] - samples[0]["rss_mb"], 2) if process_complete else None
    swap_growth = (max(max(sample["swap_growth_mb"], sample["swap_used_mb"] - baseline_swap_mb, 0.0)
                       for sample in samples) if process_complete else None)
    complete = metal_available and process_complete
    stable = (complete and delta <= MAX_COLD_METAL_GROWTH_MB
              and warm_delta <= MAX_WARM_METAL_GROWTH_MB
              and swap_growth <= MAX_QUALIFICATION_SWAP_GROWTH_MB)
    return {
        "delta_active_mb": delta, "delta_post_warmup_active_mb": warm_delta,
        "peak_active_mb": peak, "telemetry_complete": bool(complete),
        "provisional_stability_assessment": {
            "verdict": "provisional_stable" if stable else ("unstable" if complete else "unavailable"),
            "rss_growth_mb": rss_growth, "swap_growth_mb": swap_growth,
            "metal_delta_active_mb": delta, "metal_peak_active_mb": peak,
            "notes": "Bounded endpoint snapshots only; not a 1000-iteration leak or production release certificate.",
        },
    }


def _nonfinite_paths(value: Any, path: str = "report") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _nonfinite_paths(child, f"{path}.{key}")]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in _nonfinite_paths(child, f"{path}[{index}]")]
    return []


def evaluate_qualification_gates(report: dict[str, Any], config: SustainedRunnerConfig) -> dict[str, Any]:
    """Fail closed on incomplete, contradictory or nonfinite *bounded* qualification evidence."""
    report = _mapping(report)
    gates: dict[str, Any] = {}
    failure_reasons: list[str] = []

    def gate(name: str, passed: bool, reason: str, **evidence: Any) -> None:
        gates[name] = {"passed": bool(passed), **evidence}
        if not passed:
            failure_reasons.append(reason)

    mapping_keys = ("config", "preflight", "cold_startup", "warmup", "descriptor", "solo_baseline",
                    "strata_verification", "metrics_summary", "metal_memory_telemetry")
    list_keys = ("measured_requests", "rejections", "concurrent_load_interleaving", "telemetry_samples")
    nonfinite = _nonfinite_paths(report)
    schema_ok = (type(report.get("schema_version")) is int and report["schema_version"] == 2
                 and all(isinstance(report.get(key), dict) for key in mapping_keys)
                 and all(isinstance(report.get(key), list) and all(isinstance(row, dict) for row in report[key])
                         for key in list_keys) and not nonfinite)
    gate("evidence_schema", schema_ok, "Missing, malformed or nonfinite qualification evidence", nonfinite_paths=nonfinite)
    gate("completed_real_run", report.get("status") in ("completed", "passed")
         and report.get("synthetic") is False and not config.use_fake_child and not config.dry_run
         and not report.get("error") and not report.get("primary_failure"),
         "Only a completed real run can qualify; rehearsal, dry-run, partial and failed runs cannot")

    declared = _mapping(report.get("config"))
    bounds_ok = True
    try:
        validate_benchmark_bounds(config.mode, config.timeout_s, config.max_requests, config.min_headroom_mb,
                                  pilot_only=report.get("backend") == "gguf")
        validate_benchmark_bounds(declared.get("mode"), declared.get("timeout_s"), declared.get("max_requests"),
                                  config.min_headroom_mb, pilot_only=report.get("backend") == "gguf")
    except (ValueError, TypeError):
        bounds_ok = False
    bounds_ok = (bounds_ok and report.get("mode") == f"sustained_{config.mode}"
                 and declared.get("mode") == config.mode and declared.get("timeout_s") == config.timeout_s
                 and declared.get("max_requests") == config.max_requests
                 and declared.get("use_fake_child") is False)
    gate("execution_bounds", bounds_ok, "Reported mode/bounds do not match validated pilot/soak configuration")

    cold = _mapping(report.get("cold_startup"))
    cold_elapsed = cold.get("cold_startup_elapsed_s")
    gate("cold_startup", cold.get("ready") is True and _finite_number(cold_elapsed, 0) and cold_elapsed <= 30.0,
         "Cold startup failed or exceeded 30s limit", cold_startup_elapsed_s=cold_elapsed, limit_s=30.0)
    warm = _mapping(report.get("warmup"))
    gate("warmup_excluded", warm.get("status") == "completed" and warm.get("excluded_from_measured_metrics") is True,
         "Warmup must complete and be excluded from measured distributions")

    def rows(key: str) -> list[dict[str, Any]]:
        value = report.get(key)
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []

    measured, rejected, concurrent = rows("measured_requests"), rows("rejections"), rows("concurrent_load_interleaving")
    dimensions = _mapping(report.get("descriptor")).get("outputDimensions")
    numerical_ok = bool(measured) and _positive_int(dimensions) and all(
        row.get("finite") is True and row.get("l2_normalized") is True
        and _positive_int(row.get("batch_size")) and row["batch_size"] in (1, 2, 4)
        and isinstance(row.get("shape"), list) and all(_positive_int(value) for value in row["shape"])
        and row["shape"] == [row["batch_size"], dimensions]
        and _finite_number(row.get("wire_ms")) and row["wire_ms"] > 0 for row in measured
    )
    gate("numerical_correctness", numerical_ok, "Vector validation failed (missing, wrong shape, nonfinite or unnormalized)",
         validated_request_count=len(measured))
    indices = [row.get("request_idx") for row in measured + rejected]
    counts_ok = (bool(indices) and all(_positive_int(index) for index in indices)
                 and sorted(indices) == list(range(1, len(indices) + 1)) and len(indices) <= config.max_requests)
    gate("request_accounting", counts_ok, "Request IDs/counts are incomplete, duplicated or exceed the configured ceiling")
    tokens_ok = bool(measured) and all(_record_tokens(row) is not None
        and _same_number(row.get("effective_tokens_per_sec"), _rate(row.get("total_tokens"), row.get("wire_ms")), 0.11)
        for row in measured)
    gate("tokenizer_throughput", tokens_ok, "Actual tokenizer counts and token throughput are unavailable or inconsistent")

    solo = _mapping(report.get("solo_baseline"))
    solo_samples: Any = solo.get("samples_ms")
    solo_ok = (isinstance(solo_samples, list) and len(solo_samples) == 3
               and all(_finite_number(value) and value > 0 for value in solo_samples))
    solo_ok = solo_ok and _same_typed_evidence(solo.get("percentiles"), compute_explicit_percentiles(solo_samples))
    gate("solo_baseline", solo_ok, "A complete finite solo baseline with matching percentiles is required")

    strata = _mapping(report.get("strata_verification"))
    strata_ok = True
    for stratum, (lower, upper) in STRATUM_BOUNDS.items():
        evidence = _mapping(strata.get(stratum))
        counts = evidence.get("token_counts")
        valid = (_positive_int(evidence.get("item_count"))
                 and _valid_counts(counts, evidence.get("item_count", 0)))
        strata_ok = strata_ok and valid and all(lower <= count <= upper for count in counts)
        strata_ok = strata_ok and valid and evidence.get("min_tokens") == min(counts) and evidence.get("max_tokens") == max(counts)
        strata_ok = strata_ok and any(row.get("stratum") == stratum for row in measured)
    strata_ok = strata_ok and {1, 2, 4}.issubset({row.get("batch_size") for row in measured if _positive_int(row.get("batch_size"))})
    if tokens_ok:
        for row in measured:
            stratum = row.get("stratum")
            if not isinstance(stratum, str) or stratum not in (*STRATUM_BOUNDS, "mixed"):
                strata_ok = False
                continue
            lower, upper = STRATUM_BOUNDS.get(stratum, (1, 2048))
            strata_ok = strata_ok and all(lower <= count <= upper for count in row["token_counts"])
    gate("strata_and_batch_coverage", strata_ok, "All documented token strata, exact boundaries and batch sizes 1/2/4 must be verified")
    boundary = _mapping(strata.get("boundary_2049"))
    rejection_ok = (bool(rejected) and boundary.get("token_length") == 2049
                    and boundary.get("rejection_verified") is True and boundary.get("status_code") == 400
                    and all(row.get("stratum") == "boundary_2049" and row.get("rejection_verified") is True
                            and row.get("status_code") == 400 and row.get("token_counts") == [2049]
                            and _finite_number(row.get("wire_ms"), 0) for row in rejected))
    gate("boundary_rejection_policy", rejection_ok, "Exact boundary 2049 explicit HTTP 400 rejection not verified",
         boundary_2049_rejections_count=len(rejected))

    iterations = [row.get("iteration") for row in concurrent]
    iterations_ok = bool(concurrent) and all(type(index) is int and _finite_number(index, 0) for index in iterations)
    concurrent_requests = _tagged_requests(measured, "concurrent_")
    actual_tags = [row["tag"] for row in concurrent_requests]
    expected_tags = {f"{prefix}{index}" for index in iterations for prefix in
                     ("concurrent_bulk_batch_", "concurrent_interactive_query_")} if iterations_ok else set()
    # Reconcile both directions, not just each provided receipt's selected pair.
    # Exact tag strings reject malformed suffixes without normalizing identifiers.
    overlap_ok = (iterations_ok and len(set(iterations)) == len(iterations) and solo_ok
                  and len(actual_tags) == 2 * len(concurrent) and len(set(actual_tags)) == len(actual_tags)
                  and set(actual_tags) == expected_tags
                  and all("tag" not in row or isinstance(row["tag"], str) for row in measured + rejected)
                  and not _tagged_requests(rejected, "concurrent_")
                  and all(row.get("rejection_verified") is not True for row in concurrent_requests))
    requests_by_tag = {row["tag"]: row for row in concurrent_requests}
    for row in concurrent:
        times = [row.get(key) for key in ("bulk_start_mono", "bulk_end_mono", "query_start_mono", "query_end_mono")]
        valid = all(_finite_number(value, 0) for value in times)
        if valid:
            bulk_start, bulk_end, query_start, query_end = times
            valid = (bulk_start <= query_start < bulk_end and query_end > query_start
                     and _same_number((bulk_end - bulk_start) * 1000.0, row.get("bulk_wire_ms"))
                     and _same_number((query_end - query_start) * 1000.0, row.get("concurrent_interactive_wire_ms")))
        overlap_ok = overlap_ok and valid and row.get("overlap_verified") is True
        overlap_ok = overlap_ok and row.get("overlap_basis") == "client_request_intervals"
        overlap_ok = overlap_ok and all(_finite_number(row.get(key), 0) for key in
            ("solo_baseline_fixture_ms", "solo_baseline_p50_ms", "estimated_queue_wait_ms", "slowdown_factor"))
        if overlap_ok:
            baseline_ms = solo_samples[row["iteration"] % len(solo_samples)]
            overlap_ok = (_same_number(row["solo_baseline_fixture_ms"], baseline_ms)
                          and _same_number(row["solo_baseline_p50_ms"], solo["percentiles"]["p50"])
                          and _same_number(row["estimated_queue_wait_ms"], max(0.0, row["concurrent_interactive_wire_ms"] - baseline_ms))
                          and _same_number(row["slowdown_factor"], round(row["concurrent_interactive_wire_ms"] / max(1.0, baseline_ms), 2)))
        for tag, latency_key in (("concurrent_bulk_batch_", "bulk_wire_ms"),
                                 ("concurrent_interactive_query_", "concurrent_interactive_wire_ms")):
            request = requests_by_tag.get(f"{tag}{row['iteration']}") if iterations_ok else None
            overlap_ok = overlap_ok and request is not None and _same_number(request.get("wire_ms"), row.get(latency_key))
    gate("genuine_concurrent_overlap", overlap_ok, "Concurrent client request overlap must be explicit and verified for every sample",
         sample_count=len(concurrent), basis="client request intervals, not simultaneous Metal execution")

    summary = _mapping(report.get("metrics_summary"))
    tail = _mapping(summary.get("concurrent_tail_latency_ms"))
    interactive_requests = _tagged_requests(measured, "concurrent_interactive_query_")
    interactive = compute_explicit_percentiles([
        row.get("wire_ms") for row in interactive_requests])
    p95, count = interactive["p95"], interactive["count"]
    gate("concurrent_interactive_responsiveness", _same_typed_evidence(tail.get("concurrent_interactive_under_load"), interactive)
         and interactive["complete"] and _positive_int(count) and _finite_number(p95, 0) and p95 <= 200.0
         and all(_finite_number(row.get("wire_ms")) and row["wire_ms"] > 0 for row in interactive_requests),
         "Concurrent interactive query violated the <= 200.0ms target or has missing/nonfinite samples",
         target_p95_ms=200.0, measured_p95_ms=p95, sample_count=count,
         descriptive_only=interactive.get("descriptive_only"), mean_slowdown_factor=tail.get("mean_slowdown_factor"))
    wall = _mapping(summary.get("cumulative_wall_accounting"))
    elapsed: Any = wall.get("total_benchmark_elapsed_s")
    summary_ok = _finite_number(elapsed, 0) and elapsed <= config.timeout_s and summary.get("partial_run") is False
    if summary_ok:
        expected = compute_metrics_summary(measured, rejected, concurrent, tail.get("solo_baseline_interactive"), elapsed)
        keys = ("total_measured_requests", "successful_embedding_requests", "rejected_boundary_requests",
                "total_successful_embedded_tokens", "overall_effective_tokens_per_sec", "token_metrics_complete",
                "latency_metrics_complete", "overall_successful_latency_ms", "expected_rejections_latency_ms",
                "by_stratum_latency_ms", "by_batch_size_latency_ms", "concurrent_tail_latency_ms")
        summary_ok = all(_same_typed_evidence(summary.get(key), expected[key]) for key in keys)
        summary_ok = summary_ok and _same_typed_evidence(tail.get("solo_baseline_interactive"), solo.get("percentiles"))
        summary_ok = summary_ok and all(wall.get(key) == expected["cumulative_wall_accounting"][key]
                                       for key in ("sum_successful_request_wire_s", "sum_rejection_request_wire_s"))
    gate("consistent_metrics", summary_ok, "Aggregate counts/percentiles/throughput contradict raw evidence or execution bounds")

    memory = _mapping(report.get("metal_memory_telemetry"))
    samples = rows("telemetry_samples")
    preflight = _mapping(report.get("preflight"))
    baseline_swap = preflight.get("swap_used_mb")
    coverage_ok = _memory_samples_cover_run(samples, cold_elapsed, elapsed)
    gate("process_telemetry_coverage", coverage_ok,
         "Process samples must advance on both clocks and cover startup completion through measured finish",
         first_elapsed_s=samples[0].get("elapsed_s") if samples else None,
         last_elapsed_s=samples[-1].get("elapsed_s") if samples else None,
         endpoint_tolerance_s=TELEMETRY_ENDPOINT_TOLERANCE_S)
    expected_memory = compute_memory_assessment(memory, samples, baseline_swap)
    stability = _mapping(memory.get("provisional_stability_assessment"))
    expected_stability = expected_memory["provisional_stability_assessment"]
    memory_ok = (coverage_ok and memory.get("telemetry_complete") is True and expected_memory["telemetry_complete"]
                 and stability.get("verdict") == "provisional_stable"
                 and expected_stability["verdict"] == "provisional_stable"
                 and all(_same_number(memory.get(key), expected_memory[key]) for key in
                         ("delta_active_mb", "delta_post_warmup_active_mb", "peak_active_mb"))
                 and all(_same_number(stability.get(key), expected_stability[key]) for key in
                         ("rss_growth_mb", "swap_growth_mb", "metal_delta_active_mb", "metal_peak_active_mb")))
    gate("memory_stability", memory_ok, "Memory telemetry is missing, nonfinite, contradictory, unstable or exceeds growth limits",
         verdict=stability.get("verdict"), swap_growth_mb=expected_stability["swap_growth_mb"],
         max_allowed_swap_growth_mb=MAX_QUALIFICATION_SWAP_GROWTH_MB,
         max_cold_metal_growth_mb=MAX_COLD_METAL_GROWTH_MB, max_post_warmup_metal_growth_mb=MAX_WARM_METAL_GROWTH_MB)
    limits = _mapping(preflight.get("watchdog_limits"))
    headroom = preflight.get("headroom_mb")
    free_pct = preflight.get("free_memory_pct")
    installed = preflight.get("installed_ram_mb")
    rss_limit, swap_limit, free_limit = (limits.get(key) for key in
                                       ("max_rss_mb", "max_swap_growth_mb", "min_free_memory_pct"))
    resources_ok = (all(_finite_number(value, 0) for value in (headroom, free_pct, installed, baseline_swap, rss_limit, swap_limit, free_limit))
                    and headroom >= max(6000.0, config.min_headroom_mb) and installed >= headroom
                    and 12.0 <= free_limit <= free_pct <= 100.0 and rss_limit > 0 and swap_limit > 0
                    and rss_limit <= round(min(8192.0, max(32.0, headroom * 0.65), headroom * 0.70), 1)
                    and swap_limit <= round(min(2048.0, max(32.0, headroom * 0.25)), 1)
                    and coverage_ok)
    if resources_ok:
        resources_ok = all(sample["rss_mb"] <= rss_limit and sample["memory_free_pct"] >= free_limit
                           and sample["elapsed_s"] <= elapsed + TELEMETRY_ENDPOINT_TOLERANCE_S
                           and _same_number(sample["swap_growth_mb"], max(0.0, sample["swap_used_mb"] - baseline_swap))
                           and sample["swap_growth_mb"] <= MAX_QUALIFICATION_SWAP_GROWTH_MB for sample in samples) if _finite_number(elapsed, 0) else False
    gate("process_resource_limits", resources_ok, "Process RSS, system headroom/free memory or swap telemetry violates resource criteria")
    gate("watchdog_integrity", "watchdog_breach" not in report, "Watchdog breach triggered", breach=report.get("watchdog_breach"))
    cleanup = _mapping(report.get("child_cleanup"))
    supervisor = _mapping(report.get("supervisor_cleanup"))
    gate("lifecycle_cleanup", cleanup.get("cleaned") is True and supervisor.get("stopped") is True
         and not report.get("cleanup_errors"), "Child/supervisor cleanup must be verified before qualification")
    return {
        "overall_status": "passed" if not failure_reasons else "failed",
        "qualification_scope": "synthetic_rehearsal" if report.get("synthetic") is True else f"bounded_{config.mode}",
        "production_qualified": False,
        "limitations": ["No 1000-iteration leak certification, reference-cosine or held-out retrieval release gate is established by this harness."],
        "gates": gates, "failure_reasons": failure_reasons,
    }


def evaluate_pilot_acceptance(report: dict[str, Any], config: SustainedRunnerConfig) -> dict[str, Any]:
    """Recompute pilot evidence before escalation; never trust a historical passed label."""
    report = _mapping(report)
    declared = _mapping(report.get("config"))
    try:
        pilot_config = dataclasses.replace(config, mode="pilot", timeout_s=declared.get("timeout_s"),
                                           max_requests=declared.get("max_requests"), use_fake_child=False, dry_run=False)
    except (TypeError, ValueError):
        # Keep evaluating the bad evidence against safe defaults so callers receive a failure, not a crash.
        pilot_config = dataclasses.replace(config, mode="pilot", timeout_s=120.0, max_requests=30,
                                           use_fake_child=False, dry_run=False)
    result = evaluate_qualification_gates(report, pilot_config)
    model_matches = bool(config.model_path) and declared.get("model_path") == config.model_path
    if report.get("status") != "passed" or not model_matches or report.get("backend", "mlx") != "mlx":
        result["overall_status"] = "failed"
        result["failure_reasons"].append("Escalation requires a passed real MLX pilot for the exact requested model path")
    return result


@dataclasses.dataclass
class SustainedRunnerConfig:
    model_path: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8797
    control_port: int = 8798
    mode: str = "pilot"          # 'pilot' (<=30 reqs, <=120s) or 'soak' (<=100 reqs, <=180s)
    timeout_s: float = 120.0
    max_requests: int = 30
    min_headroom_mb: float = 6000.0
    use_fake_child: bool = False
    fake_dims: int = 2560
    dry_run: bool = False
    real_model_opt_in: bool = False
    output_file: Optional[str] = None

    def __post_init__(self) -> None:
        validate_benchmark_bounds(self.mode, self.timeout_s, self.max_requests, self.min_headroom_mb)


class SustainedRunner:
    """
    Executes bounded sustained qualification benchmarks for MLX embedding models
    reusing the owned process composition and active MLXWatchdog supervision.
    """

    def __init__(self, config: SustainedRunnerConfig, sampler: Optional[SystemMemorySampler] = None):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self._preflight_defaults: Optional[Any] = None
        self.last_report: dict[str, Any] = {}

    def validate_preflight(self) -> dict[str, Any]:
        """
        Enforces system memory headroom (>=6000 MB for real models), port distinctness/availability,
        and loopback policy before process spawn.
        """
        self.config.__post_init__()
        preflight_info: dict[str, Any] = {
            "timestamp": time.time(),
            "host": self.config.host,
            "port": self.config.port,
            "control_port": self.config.control_port,
            "mode": self.config.mode,
        }

        # 1. Host restriction
        if not is_numeric_loopback(self.config.host) or self.config.host != "127.0.0.1":
            raise ValueError(f"Host '{self.config.host}' must be numeric loopback '127.0.0.1'")

        # 2. Port distinctness and validity
        if not (1 <= self.config.port <= 65535) or not (1 <= self.config.control_port <= 65535):
            raise ValueError(f"Invalid ports: port={self.config.port}, control_port={self.config.control_port}")
        if self.config.port == self.config.control_port:
            raise ValueError("Inference and control ports must be distinct")
        if self.config.port == 8787 or self.config.control_port == 8787:
            raise ValueError("Port 8787 is reserved for live daemon. Select distinct qualification ports.")

        # 3. Port availability
        import socket
        for p_name, p_val in [("Inference port", self.config.port), ("Control port", self.config.control_port)]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.config.host, p_val))
                s.close()
            except OSError as e:
                raise RuntimeError(f"{p_name} {p_val} on {self.config.host} is already in use: {e}")

        # 4. Memory sampling
        installed_ram = self.sampler.get_installed_ram_mb()
        headroom = self.sampler.get_memory_headroom_mb()
        swap_used = self.sampler.get_swap_used_mb()
        free_pct = self.sampler.get_memory_free_pct()
        defaults = self.sampler.compute_conservative_defaults(headroom_mb=headroom)
        self._preflight_defaults = defaults

        preflight_info.update({
            "installed_ram_mb": installed_ram,
            "headroom_mb": headroom,
            "swap_used_mb": swap_used,
            "free_memory_pct": free_pct,
            "watchdog_limits": {
                "max_rss_mb": defaults.max_rss_mb,
                "max_swap_growth_mb": defaults.max_swap_growth_mb,
                "min_free_memory_pct": MIN_FREE_MEMORY_PCT,
            },
        })

        # 5. Model-aware preflight headroom check
        if not self.config.use_fake_child:
            if not self.config.real_model_opt_in:
                raise ValueError("Real model execution requires explicit opt-in (--real-model).")
            if not self.config.model_path:
                raise ValueError("Model path is required for real model execution.")

            meta = inspect_model_metadata(self.config.model_path)
            preflight_info["model_metadata"] = meta.to_dict()

            # Enforce >= 6000 MB available headroom for sustained real model testing
            required_headroom = max(self.config.min_headroom_mb, meta.conservative_required_headroom_mb, 6000.0)
            if headroom < required_headroom:
                raise RuntimeError(
                    f"Insufficient memory headroom for sustained {meta.params_b:.1f}B model qualification: "
                    f"measured {headroom:.1f} MB < required {required_headroom:.1f} MB. "
                    f"Refusing to spawn server child."
                )
        else:
            if headroom < min(self.config.min_headroom_mb, 256.0):
                raise RuntimeError(
                    f"Insufficient headroom for rehearsal: measured {headroom:.1f} MB < required {self.config.min_headroom_mb:.1f} MB"
                )

        return preflight_info

    def run(self) -> dict[str, Any]:
        """
        Executes bounded sustained embedding qualification under active MLXWatchdog supervision.
        """
        run_start_time = time.time()
        start_mono = time.monotonic()
        deadline = start_mono + self.config.timeout_s

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        report: dict[str, Any] = {
            "schema_version": 2,
            "status": "in_progress",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start_time)),
            "mode": f"sustained_{self.config.mode}",
            "synthetic": self.config.use_fake_child,
            "code_manifest_sha256": compute_code_manifest(repo_root),
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "control_port": self.config.control_port,
                "timeout_s": self.config.timeout_s,
                "max_requests": self.config.max_requests,
                "mode": self.config.mode,
                "use_fake_child": self.config.use_fake_child,
                "model_path": self.config.model_path,
            },
            "strata_verification": {},
            "cold_startup": {},
            "warmup": {},
            "solo_baseline": {},
            "measured_requests": [],
            "rejections": [],
            "concurrent_load_interleaving": [],
            "metrics_summary": {},
            "telemetry_samples": [],
            "metal_memory_telemetry": {"measurement_kind": "synthetic" if self.config.use_fake_child else "metal_allocator"},
            "captured_logs": "",
            "child_cleanup": {
                "cleaned": False,
                "pid": None,
                "exit_code": None,
            },
        }
        self.last_report = report

        # 1. Preflight validation
        preflight = self.validate_preflight()
        report["preflight"] = preflight

        if self.config.dry_run:
            report["status"] = "dry_run_completed"
            report["duration_s"] = time.time() - run_start_time
            self.last_report = report
            return report

        # 2. Spawn Child Server
        instance_token = uuid.uuid4().hex
        child_env = os.environ.copy()
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        child_env["HF_DATASETS_OFFLINE"] = "1"
        child_env["MLX_INSTANCE_TOKEN"] = instance_token
        child_env["MLX_EMBED_PORT"] = str(self.config.port)
        child_env["MLX_CONTROL_PORT"] = str(self.config.control_port)

        if "PYTHONPATH" in child_env:
            child_env["PYTHONPATH"] = f"{repo_root}:{child_env['PYTHONPATH']}"
        else:
            child_env["PYTHONPATH"] = repo_root

        server_script = os.path.join(repo_root, "scripts", "mlx_embed_server.py")
        if self.config.use_fake_child:
            model_spec = f"synthetic-qwen3-4b-{self.config.fake_dims}d" if self.config.fake_dims != 2560 else "synthetic-qwen3-4b"
        else:
            model_spec = str(self.config.model_path)

        cmd = [
            sys.executable,
            server_script,
            "--model", model_spec,
            "--port", str(self.config.port),
            "--control-port", str(self.config.control_port),
            "--host", self.config.host,
            "--no-warmup",  # Harness controls and isolates warmup measurement
        ]

        child: Optional[subprocess.Popen] = None
        log_file = None
        log_file_path = None
        client = SmokeHttpClient(default_timeout_s=10.0)
        t_supervisor: Optional[threading.Thread] = None
        stop_supervisor = threading.Event()
        breach_event = threading.Event()
        breach_result: list[Optional[WatchdogCheckResult]] = [None]
        telemetry_samples: list[dict[str, Any]] = []
        cleanup_errors: list[str] = []
        measured_records: list[dict[str, Any]] = []
        rejection_records: list[dict[str, Any]] = []
        concurrent_interleaving_records: list[dict[str, Any]] = []

        try:
            log_file = tempfile.NamedTemporaryFile(mode="w+", prefix="mlx_sustained_server_", suffix=".log", delete=False)
            log_file_path = log_file.name

            child = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            report["child_pid"] = child.pid
            report["instance_token"] = instance_token

            # 3. Instantiate Watchdog & Supervisor Thread
            watchdog_config = MLXWatchdogConfig(
                pid=child.pid,
                host=self.config.host,
                port=self.config.port,
                control_port=self.config.control_port,
                check_interval_s=0.25,
                startup_grace_period_s=min(25.0, self.config.timeout_s),
                stalled_inference_timeout_s=min(15.0, self.config.timeout_s),
                expected_cmd_pattern=r"(python|mlx|qmd|server)",
                instance_token=instance_token,
                allow_external_pid=True,
                dry_run=False,
            )
            watchdog = MLXWatchdog(
                config=watchdog_config,
                sampler=self.sampler,
                owned_child=child,
                baseline_swap_mb=preflight.get("swap_used_mb"),
                defaults=self._preflight_defaults,
            )

            watchdog_step_lock = threading.Lock()

            def _sample_watchdog() -> bool:
                # Serialize explicit endpoint observations with periodic checks.
                with watchdog_step_lock:
                    now_m = time.monotonic()
                    if now_m >= deadline:
                        reason = f"Absolute execution ceiling of {self.config.timeout_s:.1f}s exceeded"
                        term_ok, sigkill = watchdog.terminate_target_process(reason)
                        res = WatchdogCheckResult(
                            timestamp=time.time(),
                            healthy=False,
                            breach_type=BreachType.HEALTH_CHECK_FAILED,
                            breach_reason=reason,
                            metrics={"timeout_s": self.config.timeout_s},
                            terminated_pid=child.pid if term_ok else None,
                            sigkill_used=sigkill,
                        )
                        breach_result[0] = res
                        breach_event.set()
                        return False

                    check = watchdog.check_step()
                    if not check.healthy:
                        breach_result[0] = check
                        breach_event.set()
                        return False

                    if check.metrics.get("startup_grace") is True:
                        report.setdefault("startup_supervision_events", []).append(dict(check.metrics))
                    elif check.metrics:
                        telemetry_samples.append({
                            "timestamp": check.timestamp,
                            "elapsed_s": time.monotonic() - start_mono,
                            "rss_mb": check.metrics.get("rss_mb"),
                            "swap_used_mb": check.metrics.get("swap_used_mb"),
                            "swap_growth_mb": check.metrics.get("swap_growth_mb"),
                            "memory_free_pct": check.metrics.get("memory_free_pct"),
                        })
                    return True

            def _supervisor_loop():
                while not stop_supervisor.is_set():
                    if not _sample_watchdog():
                        break
                    stop_supervisor.wait(timeout=0.25)

            t_supervisor = threading.Thread(target=_supervisor_loop, daemon=True, name="Sustained-Watchdog-Supervisor")
            t_supervisor.start()

            # 4. Poll Control Port for Readiness (Cold Startup Timing)
            base_url = f"http://{self.config.host}:{self.config.port}"
            ctrl_url = f"http://{self.config.host}:{self.config.control_port}"

            ready = False
            startup_deadline = min(deadline, time.monotonic() + 30.0)
            while time.monotonic() < startup_deadline:
                if breach_event.is_set():
                    b_res = breach_result[0]
                    raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")

                if child.poll() is not None:
                    time.sleep(0.1)
                    if breach_event.is_set():
                        b_res = breach_result[0]
                        raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")
                    raise RuntimeError(f"Server child exited prematurely with code {child.returncode}")

                try:
                    status_code, data = client.get_json(f"{ctrl_url}/health", deadline=startup_deadline, timeout_s=1.0)
                    if status_code == 200 and isinstance(data, dict):
                        if data.get("ready") is True and data.get("instance_token") == instance_token and _sample_watchdog():
                            ready = True
                            report["cold_startup"] = {
                                "ready": True,
                                "cold_startup_elapsed_s": time.monotonic() - start_mono,
                                "status": "passed",
                            }
                            break
                except Exception:
                    pass
                time.sleep(0.1)

            if not ready:
                raise TimeoutError("Server failed to report ready within startup period")

            # 5. Fetch and Validate Descriptor
            st_desc, desc = client.get_json(f"{ctrl_url}/descriptor", deadline=deadline, timeout_s=2.0)
            if st_desc != 200 or not isinstance(desc, dict):
                raise RuntimeError(f"Failed to fetch descriptor: status {st_desc}, data {desc}")
            report["descriptor"] = desc
            expected_dims = desc.get("outputDimensions") or desc.get("nativeDimensions") or 2560

            # 6. Sample Initial Metal Memory
            def _sample_memory(stage: str) -> None:
                # Preserve the exact failed body/status separately from derived metrics.
                memory = report["metal_memory_telemetry"]
                try:
                    status, payload = client.get_json(f"{ctrl_url}/memory", deadline=deadline, timeout_s=2.0)
                except Exception as error:
                    memory[stage] = None
                    memory.setdefault("http_status", {})[stage] = None
                    raise RuntimeError(f"Required {stage} /memory telemetry failed: {error}") from error
                memory[stage] = payload
                memory.setdefault("http_status", {})[stage] = status
                if type(status) is not int or status != 200 or not _valid_memory_snapshot(payload):
                    raise RuntimeError(f"Required {stage} /memory telemetry failed: HTTP {status}; "
                                       "expected an error-free object with finite nonnegative active_mb and peak_mb >= active_mb")

            _sample_memory("initial")

            def _check_breach():
                if breach_event.is_set():
                    b = breach_result[0]
                    raise RuntimeError(f"Watchdog breach: {b.breach_reason if b else 'unknown breach'}")
                if child.poll() is not None:
                    time.sleep(0.05)
                    if breach_event.is_set():
                        b = breach_result[0]
                        raise RuntimeError(f"Watchdog breach: {b.breach_reason if b else 'unknown breach'}")
                    raise RuntimeError(f"Server child exited unexpectedly with code {child.returncode}")

            # 7. Calibrate & Verify Exact Boundary Fixtures & Token Strata via /tokenize
            _check_breach()
            calibrated_boundaries = calibrate_boundary_fixtures(client, base_url, deadline)
            b2047 = calibrated_boundaries["boundary_2047"]
            b2048 = calibrated_boundaries["boundary_2048"]
            b2049 = calibrated_boundaries["boundary_2049"]

            strata_candidates = {
                "short": SHORT_FIXTURES,
                "medium": MEDIUM_FIXTURES,
                "long": LONG_FIXTURES,
                "code": CODE_FIXTURES,
                "boundary_2047": [b2047],
                "boundary_2048": [b2048],
            }
            strata_verified: dict[str, Any] = {}

            for stratum_name, texts in strata_candidates.items():
                _check_breach()
                st_tok, r_tok = client.post_json(f"{base_url}/tokenize", {"texts": texts}, deadline=deadline)
                if st_tok != 200 or not isinstance(r_tok, dict) or "counts" not in r_tok:
                    raise RuntimeError(f"Tokenization failed for {stratum_name}: HTTP {st_tok}, payload {r_tok}")
                counts = r_tok["counts"]
                if not _valid_counts(counts, len(texts)):
                    raise RuntimeError(f"Malformed tokenizer counts for {stratum_name}: {counts}")
                strata_verified[stratum_name] = {
                    "item_count": len(counts),
                    "token_counts": counts,
                    "min_tokens": min(counts),
                    "max_tokens": max(counts),
                    "mean_tokens": round(float(np.mean(counts)), 1),
                }

            # Boundary 2049 Policy Check (Must reject with HTTP 400 and exact token count in error)
            _check_breach()
            st_b_over, r_b_over = client.post_json(f"{base_url}/tokenize", {"texts": [b2049]}, deadline=deadline)
            rejection_error_str = str(r_b_over)
            if st_b_over == 400:
                m_tokens = re.search(r"\((\d+)\)\s+exceeds\s+max_length\s+\((\d+)\)", rejection_error_str)
                if not m_tokens or int(m_tokens.group(1)) != 2049 or int(m_tokens.group(2)) != 2048:
                    raise RuntimeError("Boundary tokenizer rejection did not verify exact 2049/2048 lengths")
                parsed_len = int(m_tokens.group(1))
                strata_verified["boundary_2049"] = {
                    "policy": "explicit_rejection",
                    "status_code": 400,
                    "rejection_verified": True,
                    "token_length": parsed_len,
                    "message": rejection_error_str,
                }
            else:
                raise RuntimeError(f"Expected HTTP 400 rejection for boundary_2049, got HTTP {st_b_over}: {r_b_over}")

            # Verify strata compliance ranges strictly
            assert 5 <= strata_verified["short"]["min_tokens"] and strata_verified["short"]["max_tokens"] <= 25, f"Short stratum out of range: {strata_verified['short']}"
            assert 50 <= strata_verified["medium"]["min_tokens"] and strata_verified["medium"]["max_tokens"] <= 200, f"Medium stratum out of range: {strata_verified['medium']}"
            assert 400 <= strata_verified["long"]["min_tokens"] and strata_verified["long"]["max_tokens"] <= 1500, f"Long stratum out of range: {strata_verified['long']}"
            assert 50 <= strata_verified["code"]["min_tokens"] and strata_verified["code"]["max_tokens"] <= 500, f"Code stratum out of range: {strata_verified['code']}"
            assert strata_verified["boundary_2047"]["min_tokens"] == 2047, f"Boundary 2047 not exact: {strata_verified['boundary_2047']}"
            assert strata_verified["boundary_2048"]["min_tokens"] == 2048, f"Boundary 2048 not exact: {strata_verified['boundary_2048']}"
            assert strata_verified["boundary_2049"]["rejection_verified"] is True, f"Boundary 2049 not rejected: {strata_verified['boundary_2049']}"
            report["strata_verification"] = strata_verified

            # 8. Warmup Pass (Explicitly Excluded from Measured Metrics)
            _check_breach()
            t_warm_0 = time.monotonic()
            st_w1, _ = client.post_json(f"{base_url}/embed", {"texts": [SHORT_FIXTURES[0]], "is_query": True}, deadline=deadline)
            st_w2, _ = client.post_json(f"{base_url}/embed", {"texts": [SHORT_FIXTURES[0]] * 4}, deadline=deadline)
            warmup_ms = (time.monotonic() - t_warm_0) * 1000.0
            if st_w1 != 200 or st_w2 != 200:
                raise RuntimeError(f"Warmup pass failed: st_w1={st_w1}, st_w2={st_w2}")
            report["warmup"] = {
                "status": "completed",
                "warmup_elapsed_ms": round(warmup_ms, 2),
                "excluded_from_measured_metrics": True,
            }

            # Sample post-warmup memory
            _sample_memory("post_warmup")

            # 9. Solo Baseline Interactive Measurements (Under Idle Load)
            solo_baseline_latencies: list[float] = []
            for solo_idx in range(3):
                _check_breach()
                t0_s = time.monotonic()
                st_s, r_s = client.post_json(f"{base_url}/embed", {"texts": [SHORT_FIXTURES[solo_idx % len(SHORT_FIXTURES)]], "is_query": True}, deadline=deadline)
                solo_ms = (time.monotonic() - t0_s) * 1000.0
                if st_s != 200:
                    raise RuntimeError(f"Solo baseline request {solo_idx} failed: HTTP {st_s}")
                solo_baseline_latencies.append(solo_ms)

            # Derive from persisted precision so a serialized report recomputes identically.
            solo_baseline_latencies = [round(value, 2) for value in solo_baseline_latencies]
            solo_baseline_p50 = float(np.percentile(solo_baseline_latencies, 50))
            report["solo_baseline"] = {
                "samples_ms": solo_baseline_latencies,
                "percentiles": compute_explicit_percentiles(solo_baseline_latencies),
            }

            # 10. Workload Execution
            measured_records: list[dict[str, Any]] = []
            rejection_records: list[dict[str, Any]] = []
            concurrent_interleaving_records: list[dict[str, Any]] = []
            request_counter = 0

            def _execute_measured_request(
                texts: list[str],
                stratum: str,
                is_query: bool = False,
                tag: str = "standard",
            ) -> Optional[dict[str, Any]]:
                nonlocal request_counter
                request_counter += 1
                _check_breach()

                tok_counts: Optional[list[int]]
                if stratum == "boundary_2047":
                    tok_counts = [2047]
                elif stratum == "boundary_2048":
                    tok_counts = [2048]
                elif stratum == "boundary_2049":
                    tok_counts = [2049]
                else:
                    tok_counts = fetch_token_counts(client, base_url, texts, deadline)

                t0 = time.monotonic()
                st_emb, r_emb = client.post_json(
                    f"{base_url}/embed",
                    {"texts": texts, "is_query": is_query},
                    deadline=deadline,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000.0

                if stratum == "boundary_2049" and st_emb == 400:
                    rec_rej = {
                        "request_idx": request_counter,
                        "stratum": stratum,
                        "tag": tag,
                        "batch_size": len(texts),
                        "token_counts": tok_counts,
                        "is_query": is_query,
                        "wire_ms": round(elapsed_ms, 2),
                        "rejection_verified": True,
                        "status_code": 400,
                    }
                    rejection_records.append(rec_rej)
                    return rec_rej

                if st_emb != 200 or not isinstance(r_emb, dict):
                    time.sleep(0.05)
                    _check_breach()
                    raise RuntimeError(f"Request {request_counter} ({stratum}) failed with HTTP {st_emb}: {r_emb}")

                arr = np.array(r_emb["embeddings"], dtype=np.float32)
                fin_ok, fin_msg = check_finite(arr)
                dim_ok, dim_msg = check_dimensions(arr, expected_rows=len(texts), expected_dims=expected_dims)
                norm_ok, norms, norm_msg = check_l2_normalization(arr)

                if not (fin_ok and dim_ok and norm_ok):
                    raise ValueError(f"Vector validation failed for request {request_counter}: {dim_msg}, {norm_msg}, {fin_msg}")

                record = {
                    "request_idx": request_counter,
                    "stratum": stratum,
                    "tag": tag,
                    "batch_size": len(texts),
                    **request_token_metrics(texts, tok_counts, round(elapsed_ms, 2), synthetic=self.config.use_fake_child),
                    "is_query": is_query,
                    "wire_ms": round(elapsed_ms, 2),

                    "shape": list(arr.shape),
                    "finite": fin_ok,
                    "l2_normalized": norm_ok,
                    "mean_norm": round(float(np.mean(norms)), 6) if norms else 1.0,
                }
                measured_records.append(record)
                return record

            # Stage A: Singletons across all strata (Batch Size = 1)
            for s_idx, short_t in enumerate(SHORT_FIXTURES[:3]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([short_t], stratum="short", is_query=True, tag=f"singleton_short_{s_idx}")

            for m_idx, med_t in enumerate(MEDIUM_FIXTURES[:2]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([med_t], stratum="medium", is_query=False, tag=f"singleton_medium_{m_idx}")

            for l_idx, long_t in enumerate(LONG_FIXTURES[:2]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([long_t], stratum="long", is_query=False, tag=f"singleton_long_{l_idx}")

            for c_idx, code_t in enumerate(CODE_FIXTURES[:2]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([code_t], stratum="code", is_query=False, tag=f"singleton_code_{c_idx}")

            # Stage B: Exact 2047, 2048, 2049 Boundary Tests
            if request_counter < self.config.max_requests:
                _execute_measured_request([b2047], stratum="boundary_2047", tag="boundary_2047_pass")
            if request_counter < self.config.max_requests:
                _execute_measured_request([b2048], stratum="boundary_2048", tag="boundary_2048_pass")
            if request_counter < self.config.max_requests:
                _execute_measured_request([b2049], stratum="boundary_2049", tag="boundary_2049_reject_400")

            # Stage C: Batch Size = 2 Conservative Progression
            if request_counter < self.config.max_requests:
                _execute_measured_request([SHORT_FIXTURES[0], SHORT_FIXTURES[1]], stratum="short", tag="batch2_short")
            if request_counter < self.config.max_requests:
                _execute_measured_request([MEDIUM_FIXTURES[0], MEDIUM_FIXTURES[1]], stratum="medium", tag="batch2_medium")
            if request_counter < self.config.max_requests:
                _execute_measured_request([CODE_FIXTURES[0], CODE_FIXTURES[1]], stratum="code", tag="batch2_code")

            # Stage D: Batch Size = 4 Conservative Progression
            if request_counter < self.config.max_requests:
                _execute_measured_request(SHORT_FIXTURES[:4], stratum="short", tag="batch4_short")
            if request_counter < self.config.max_requests:
                _execute_measured_request([MEDIUM_FIXTURES[0], MEDIUM_FIXTURES[1]], stratum="medium", tag="batch4_medium")

            # Stage E: True Concurrent Interleaving (Barrier & Telemetry Handshake)
            num_concurrent_iters = min(3, max(1, (self.config.max_requests - request_counter) // 2))
            for c_iter in range(num_concurrent_iters):
                if request_counter + 2 > self.config.max_requests:
                    break

                _check_breach()
                bulk_texts = [LONG_FIXTURES[0], MEDIUM_FIXTURES[0], CODE_FIXTURES[0], SHORT_FIXTURES[0]]
                query_texts = [SHORT_FIXTURES[c_iter % len(SHORT_FIXTURES)]]
                # Tokenize before either timed request: no counting traffic perturbs the overlap.
                bulk_counts = fetch_token_counts(client, base_url, bulk_texts, deadline)
                query_counts = fetch_token_counts(client, base_url, query_texts, deadline)

                bulk_client = SmokeHttpClient(default_timeout_s=15.0)
                query_client = SmokeHttpClient(default_timeout_s=15.0)

                bulk_res: dict[str, Any] = {}
                query_res: dict[str, Any] = {}
                bulk_started = threading.Event()
                query_launched = threading.Event()

                def _bulk_worker():
                    try:
                        t0_b = time.monotonic()
                        bulk_started.set()
                        st_b, r_b = bulk_client.post_json(
                            f"{base_url}/embed",
                            {"texts": bulk_texts, "is_query": False},
                            deadline=deadline,
                        )
                        t1_b = time.monotonic()
                        bulk_res.update({
                            "status_code": st_b,
                            "response": r_b,
                            "start_mono": t0_b,
                            "end_mono": t1_b,
                            "wire_ms": round((t1_b - t0_b) * 1000.0, 2),
                        })
                    except Exception as exc:
                        bulk_res["error"] = str(exc)
                    finally:
                        bulk_client.close()

                def _query_worker():
                    try:
                        bulk_started.wait(timeout=2.0)
                        time.sleep(0.015)  # Scheduling opportunity only; recorded intervals must prove overlap.
                        t0_q = time.monotonic()
                        query_launched.set()
                        st_q, r_q = query_client.post_json(
                            f"{base_url}/embed",
                            {"texts": query_texts, "is_query": True},
                            deadline=deadline,
                        )
                        t1_q = time.monotonic()
                        query_res.update({
                            "status_code": st_q,
                            "response": r_q,
                            "start_mono": t0_q,
                            "end_mono": t1_q,
                            "wire_ms": round((t1_q - t0_q) * 1000.0, 2),
                        })
                    except Exception as exc:
                        query_res["error"] = str(exc)
                    finally:
                        query_client.close()

                t_b = threading.Thread(target=_bulk_worker, daemon=True, name=f"Bulk-Client-{c_iter}")
                t_q = threading.Thread(target=_query_worker, daemon=True, name=f"Interactive-Client-{c_iter}")

                t_b.start()
                t_q.start()
                t_b.join(timeout=30.0)
                t_q.join(timeout=30.0)

                _check_breach()

                if "error" in bulk_res or bulk_res.get("status_code") != 200:
                    time.sleep(0.05)
                    _check_breach()
                    raise RuntimeError(f"Concurrent bulk batch failed: {bulk_res}")
                if "error" in query_res or query_res.get("status_code") != 200:
                    time.sleep(0.05)
                    _check_breach()
                    raise RuntimeError(f"Concurrent interactive query failed: {query_res}")

                arr_b = np.array(bulk_res["response"]["embeddings"], dtype=np.float32)
                arr_q = np.array(query_res["response"]["embeddings"], dtype=np.float32)

                fin_b, _ = check_finite(arr_b)
                norm_b, _, _ = check_l2_normalization(arr_b)
                fin_q, _ = check_finite(arr_q)
                norm_q, _, _ = check_l2_normalization(arr_q)
                dim_b, _ = check_dimensions(arr_b, expected_rows=len(bulk_texts), expected_dims=expected_dims)
                dim_q, _ = check_dimensions(arr_q, expected_rows=len(query_texts), expected_dims=expected_dims)

                if not (fin_b and norm_b and fin_q and norm_q and dim_b and dim_q):
                    raise ValueError("Concurrent embeddings validation failed (finite or norm violation)")

                request_counter += 1
                rec_b = {
                    "request_idx": request_counter,
                    "stratum": "mixed",
                    "tag": f"concurrent_bulk_batch_{c_iter}",
                    "batch_size": len(bulk_texts),
                    **request_token_metrics(bulk_texts, bulk_counts, bulk_res["wire_ms"], synthetic=self.config.use_fake_child),
                    "is_query": False,
                    "wire_ms": bulk_res["wire_ms"],

                    "shape": list(arr_b.shape),
                    "finite": fin_b,
                    "l2_normalized": norm_b,
                }
                measured_records.append(rec_b)

                request_counter += 1
                rec_q = {
                    "request_idx": request_counter,
                    "stratum": "short",
                    "tag": f"concurrent_interactive_query_{c_iter}",
                    "batch_size": 1,
                    **request_token_metrics(query_texts, query_counts, query_res["wire_ms"], synthetic=self.config.use_fake_child),
                    "is_query": True,
                    "wire_ms": query_res["wire_ms"],

                    "shape": list(arr_q.shape),
                    "finite": fin_q,
                    "l2_normalized": norm_q,
                }
                measured_records.append(rec_q)

                overlap = bool(
                    query_res["start_mono"] >= bulk_res["start_mono"]
                    and query_res["start_mono"] < bulk_res["end_mono"]
                )
                baseline_fixture_ms = (
                    solo_baseline_latencies[c_iter % len(solo_baseline_latencies)]
                    if solo_baseline_latencies
                    else solo_baseline_p50
                )
                est_queue_wait = max(0.0, query_res["wire_ms"] - baseline_fixture_ms)
                slowdown_factor = round(query_res["wire_ms"] / max(1.0, baseline_fixture_ms), 2)

                interleaving_entry = {
                    "iteration": c_iter,
                    "fixture_index": c_iter % len(SHORT_FIXTURES),
                    "overlap_verified": overlap,
                    "overlap_basis": "client_request_intervals",
                    "bulk_start_mono": bulk_res["start_mono"],
                    "bulk_end_mono": bulk_res["end_mono"],
                    "query_start_mono": query_res["start_mono"],
                    "query_end_mono": query_res["end_mono"],
                    "bulk_wire_ms": bulk_res["wire_ms"],
                    "concurrent_interactive_wire_ms": query_res["wire_ms"],
                    "solo_baseline_fixture_ms": round(baseline_fixture_ms, 2),
                    "solo_baseline_p50_ms": round(solo_baseline_p50, 2),
                    "estimated_queue_wait_ms": round(est_queue_wait, 2),
                    "slowdown_factor": slowdown_factor,
                }
                concurrent_interleaving_records.append(interleaving_entry)

            report["concurrent_load_interleaving"] = concurrent_interleaving_records

            # Stage F: If in soak mode, continue bounded iterations up to max_requests
            if self.config.mode == "soak":
                soak_pool = [
                    ([SHORT_FIXTURES[0]], "short", True),
                    ([MEDIUM_FIXTURES[0]], "medium", False),
                    ([SHORT_FIXTURES[1], SHORT_FIXTURES[2]], "short", False),
                    ([CODE_FIXTURES[0]], "code", False),
                    ([MEDIUM_FIXTURES[1], CODE_FIXTURES[1]], "mixed", False),
                    ([SHORT_FIXTURES[0], SHORT_FIXTURES[1], SHORT_FIXTURES[2], SHORT_FIXTURES[3]], "short", False),
                ]
                soak_idx = 0
                while request_counter < self.config.max_requests and time.monotonic() < (deadline - 5.0):
                    item = soak_pool[soak_idx % len(soak_pool)]
                    _execute_measured_request(item[0], stratum=item[1], is_query=item[2], tag=f"soak_iter_{request_counter}")
                    soak_idx += 1

            # 11. Sample Final Metal Memory & Steady-State Memory Trend
            _sample_memory("final")

            # Freeze periodic evidence, then observe the live child after all measured work.
            # A short pilot must not depend on whether the 250ms loop happened to tick.
            stop_supervisor.set()
            t_supervisor.join(timeout=1.0)
            if t_supervisor.is_alive():
                raise RuntimeError("Watchdog supervisor did not stop before final process sampling")
            _sample_watchdog()
            _check_breach()
            benchmark_elapsed_s = time.monotonic() - start_mono

            # 12. Compute Metrics Aggregations (Successful Tokens ONLY, Separate Rejections)
            report["measured_requests"] = measured_records
            report["rejections"] = rejection_records
            report["concurrent_load_interleaving"] = concurrent_interleaving_records
            report["telemetry_samples"] = telemetry_samples

            report["metrics_summary"] = compute_metrics_summary(
                measured_records=measured_records,
                rejection_records=rejection_records,
                concurrent_interleaving_records=concurrent_interleaving_records,
                solo_baseline_percentiles=report.get("solo_baseline", {}).get("percentiles"),
                total_benchmark_elapsed_s=benchmark_elapsed_s,
                partial_run=False,
            )

            # Qualification is finalized only after cleanup has actually completed.
            report["status"] = "completed"

        except Exception as e:
            report["status"] = "failed"
            report["error"] = str(e)
            report["telemetry_samples"] = telemetry_samples
            report["measured_requests"] = measured_records
            report["rejections"] = rejection_records
            report["concurrent_load_interleaving"] = concurrent_interleaving_records

            # Check if supervisor captured a breach or if child terminated
            if not breach_event.is_set():
                breach_event.wait(timeout=0.1)

            # Primary failure classification vs derivative errors
            if breach_event.is_set() or (breach_result[0] is not None):
                b = breach_result[0]
                if b is not None:
                    report["watchdog_breach"] = extract_watchdog_breach_dict(b)
                    report["primary_failure"] = {
                        "category": "watchdog_breach",
                        "breach_type": b.breach_type.value if b.breach_type else "unknown",
                        "reason": b.breach_reason,
                    }
                else:
                    report["primary_failure"] = {
                        "category": "watchdog_breach",
                        "reason": "Watchdog breach event was signaled",
                    }
            elif child is not None and child.poll() is not None and child.returncode != 0:
                report["primary_failure"] = {
                    "category": "process_exit",
                    "reason": f"Server child process exited unexpectedly with code {child.returncode}",
                    "exit_code": child.returncode,
                }
            else:
                report["primary_failure"] = {
                    "category": "execution_error",
                    "reason": str(e),
                }

            # Always populate partial metrics summary with completed request counts
            solo_p = report.get("solo_baseline", {}).get("percentiles")
            report["metrics_summary"] = compute_metrics_summary(
                measured_records=measured_records,
                rejection_records=rejection_records,
                concurrent_interleaving_records=concurrent_interleaving_records,
                solo_baseline_percentiles=solo_p,
                total_benchmark_elapsed_s=time.monotonic() - start_mono,
                partial_run=True,
            )
            report["qualification_evaluation"] = evaluate_qualification_gates(report, self.config)
            self.last_report = report
            raise
        finally:
            # 12. Guaranteed Teardown and Cleanup
            if log_file_path and os.path.exists(log_file_path):
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                        report["captured_logs"] = f.read()[-8192:]
                except Exception as log_err:
                    report["captured_logs"] = f"Failed to capture server logs: {log_err}"

            stop_supervisor.set()
            if t_supervisor is not None and t_supervisor.is_alive():
                try:
                    t_supervisor.join(timeout=1.0)
                except Exception:
                    pass

            report["supervisor_cleanup"] = {"stopped": t_supervisor is None or not t_supervisor.is_alive()}
            client.close()

            sigkill_used = False
            if child is not None:
                try:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=1.0)
                            sigkill_used = True
                    report["child_cleanup"] = {
                        "cleaned": True,
                        "pid": child.pid,
                        "exit_code": child.returncode,
                        "sigkill_used": sigkill_used,
                    }
                except Exception as cleanup_err:
                    cleanup_errors.append(f"Child cleanup error: {cleanup_err}")
                    report["child_cleanup"] = {
                        "cleaned": False,
                        "pid": child.pid if child else None,
                        "error": str(cleanup_err),
                    }

            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            if log_file_path and os.path.exists(log_file_path):
                try:
                    os.unlink(log_file_path)
                except Exception as unlink_err:
                    cleanup_errors.append(f"Log unlink error: {unlink_err}")

            if cleanup_errors:
                report["cleanup_errors"] = cleanup_errors

            report["duration_s"] = round(time.time() - run_start_time, 2)
            if breach_result[0] is not None:
                report["watchdog_breach"] = extract_watchdog_breach_dict(breach_result[0])
            # Failed/partial runs retain raw telemetry but never zero-fill missing metrics.
            report["metal_memory_telemetry"].update(compute_memory_assessment(
                report["metal_memory_telemetry"], telemetry_samples, preflight.get("swap_used_mb"),
            ))
            report["qualification_evaluation"] = evaluate_qualification_gates(report, self.config)
            if report["status"] == "completed":
                if self.config.use_fake_child and not cleanup_errors and report["supervisor_cleanup"]["stopped"] and "watchdog_breach" not in report:
                    # A plumbing rehearsal is successful, but is never real-model qualification.
                    report["status"] = "rehearsal_completed"
                else:
                    report["status"] = report["qualification_evaluation"]["overall_status"]
                    if report["status"] != "passed":
                        report["error"] = "; ".join(report["qualification_evaluation"]["failure_reasons"])
            self.last_report = report

            if self.config.output_file:
                try:
                    out_dir = os.path.dirname(os.path.abspath(self.config.output_file))
                    os.makedirs(out_dir, exist_ok=True)
                    with open(self.config.output_file, "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2)
                except Exception as save_err:
                    print(f"[sustained] Failed to save report to {self.config.output_file}: {save_err}", file=sys.stderr)

        return report
