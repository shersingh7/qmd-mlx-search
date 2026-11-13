#!/usr/bin/env python3
"""Bounded real-MLX comparison on authored public text, never model downloads.

This measures the production Python runtime, not HTTP or end-to-end retrieval.
Run baseline and candidate sequentially using the same model, dtype and budget.
No output is a claim of general retrieval quality or long-run leak freedom.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time


def percentile(values: list[float], percent: float) -> float:
    if not values or not 0 <= percent <= 100:
        raise ValueError("Nonempty samples and percentile in [0,100] required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float], texts_per_request: int) -> dict:
    if not values or any(not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("Finite positive timing samples required")
    count = len(values) * texts_per_request
    return {"samples_ms": values, "requests": len(values), "texts_processed": count,
            "p50_ms": statistics.median(values), "p95_ms": percentile(values, 95),
            "texts_per_second": count / (sum(values) / 1000)}


def bounded_repetitions(value: str) -> int:
    result = int(value)
    if not 1 <= result <= 200:
        raise ValueError("repetitions must be between 1 and 200")
    return result


def validate_model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("model must be an existing absolute local model directory with config.json")
    return path.resolve()


def workload() -> list[str]:
    return [
        "SQLite transactions provide atomic writes and crash recovery.",
        "Apple Silicon shares physical memory between the CPU and Metal GPU. " * 5,
        "Connection pools reuse open database connections and bound concurrent work. " * 20,
        "Reciprocal rank fusion combines lexical and semantic retrieval results." * 3,
    ]


def current_rss_bytes() -> int | None:
    try:
        value = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                               check=True, text=True, capture_output=True, timeout=2).stdout.strip()
        return int(value) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def source_digest(repo: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((repo / "scripts" / "qmd_mlx").rglob("*.py")):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local weights directory; remote IDs rejected")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path, help="New JSON receipt path; will not overwrite")
    parser.add_argument("--label", required=True)
    parser.add_argument("--bulk-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--repetitions", type=bounded_repetitions, default=200)
    parser.add_argument("--reference-vectors", type=Path)
    args = parser.parse_args(argv)
    model = validate_model_path(args.model)
    repo = args.repo.resolve()
    if not (repo / "scripts/qmd_mlx/runtime.py").is_file():
        parser.error("repo must contain scripts/qmd_mlx/runtime.py")
    if not 1 <= args.bulk_tokens <= 65536:
        parser.error("bulk-tokens must be between 1 and 65536")
    output = args.output.resolve()
    vectors_path = output.with_suffix(".npz")
    if output.exists() or vectors_path.exists():
        parser.error("output receipt/vector artifact already exists; use a fresh label/path")
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not os.environ.get("TMPDIR") or not Path(os.environ["TMPDIR"]).is_dir():
        parser.error("set TMPDIR to an existing managed scratch directory before running")
    sys.path.insert(0, str(repo))
    # Only import GPU modules after validation/offline setup. CPU unit tests never import them.
    import numpy as np
    import mlx.core as mx
    from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime
    import inspect
    actual_runtime = Path(inspect.getfile(MLXEmbeddingRuntime)).resolve()
    if actual_runtime != repo / "scripts/qmd_mlx/runtime.py":
        raise RuntimeError(f"Wrong runtime imported: {actual_runtime}")
    report = {
        "label": args.label, "repo": str(repo), "source_sha256": source_digest(repo),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": str(model), "dtype_requested": args.dtype, "bulk_microbatch_tokens": args.bulk_tokens,
        "scope": "real MLX runtime; public synthetic workload; not HTTP/retrieval-quality/long-soak qualification",
        "platform": platform.platform(), "machine": platform.machine(),
        "versions": {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "numpy", "transformers"]},
        "metal_available": bool(mx.metal.is_available()), "passed": False, "samples": {},
    }
    runtime = None
    baseline_threads = set(threading.enumerate())

    def memory() -> dict:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"metal_active_bytes": mx.get_active_memory(), "metal_cache_bytes": mx.get_cache_memory(),
                "metal_peak_bytes": mx.get_peak_memory(), "current_rss_bytes": current_rss_bytes(),
                "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}

    try:
        if not report["metal_available"]:
            raise RuntimeError("Metal unavailable; this is not a CPU substitute benchmark")
        # Process-local caps for a small encoder; never change OS or another process's budget.
        mx.set_memory_limit(2 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)
        report["limits"] = {"metal_memory_bytes": 2 * 1024**3, "metal_cache_bytes": 512 * 1024**2}
        report["before_load_memory"] = memory()
        started = time.perf_counter()
        runtime = MLXEmbeddingRuntime(model_name=str(model), max_length=256, dtype_str=args.dtype)
        # Existing baseline and candidate both expose the planner; same budget in either arm.
        runtime.batch_planner.max_bulk_microbatch_tokens = args.bulk_tokens
        report["load_seconds"] = time.perf_counter() - started
        runtime.warmup()
        texts = workload()
        report["descriptor"] = runtime.get_descriptor()
        token_counts = [len(ids) for ids in runtime.tokenize(texts)]
        report["fixture_token_counts"] = token_counts
        mixed = runtime.submit_embed(texts, timeout=30)
        solo = np.concatenate([runtime.submit_embed([text], timeout=30) for text in texts], axis=0)
        cosine = float(np.min(np.sum(mixed * solo, axis=1)))
        report["mixed_singleton_min_cosine"] = cosine
        report["mixed_singleton_max_abs"] = float(np.max(np.abs(mixed - solo)))
        if cosine < 0.9999:
            raise AssertionError("Single/mixed batch parity failed")
        if args.reference_vectors:
            with np.load(args.reference_vectors, allow_pickle=False) as reference:
                expected = reference["mixed"]
                if expected.shape != mixed.shape:
                    raise AssertionError("Reference vector dimensions mismatch")
                reference_cosine = float(np.min(np.sum(expected * mixed, axis=1)))
                report["reference_min_cosine"] = reference_cosine
                report["reference_max_abs"] = float(np.max(np.abs(expected - mixed)))
                if reference_cosine < 0.9999:
                    raise AssertionError("Reference vector parity failed")
        with vectors_path.open("xb") as file:
            np.savez(file, mixed=mixed, singletons=solo)
        report["vectors_artifact"] = str(vectors_path)
        # Caller-owned exported arrays may share unified memory. Release them before
        # the shutdown-allocation check; retaining output is not a runtime leak.
        del mixed, solo
        expected = None

        def bench(name: str, batch: list[str], repetitions: int, tokens_per_request: int, is_query=False):
            assert runtime is not None
            samples = []
            runtime.submit_embed(batch, timeout=30, is_query=is_query)  # per-workload warmup, not a sample
            for _ in range(repetitions):
                started = time.perf_counter()
                result = runtime.submit_embed(batch, timeout=30, is_query=is_query)
                samples.append((time.perf_counter() - started) * 1000)
                if not np.isfinite(result).all() or np.max(np.abs(np.linalg.norm(result, axis=1) - 1)) > 1e-4:
                    raise AssertionError("Invalid embedding values or norms")
            values = summarize(samples, len(batch))
            values["actual_tokens_per_second"] = tokens_per_request * repetitions / (sum(samples) / 1000)
            values["memory"] = memory()
            report["samples"][name] = values

        bench("singleton", [texts[1]], 20, token_counts[1], is_query=True)
        bench("bulk_64_mixed", texts * 16, 8, sum(token_counts) * 16)
        report["repeat_start_memory"] = memory()
        bench("repeated_batch_4", texts, args.repetitions, sum(token_counts))
        report["repeat_end_memory"] = memory()
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            executor = runtime.executor if runtime is not None else None
            if runtime is not None:
                runtime.shutdown()
            runtime = None
            report["immediate_post_shutdown_memory"] = memory()
            # Owner completion and object release precede the device/allocator
            # observation barrier. Keep the immediate sample, and still require
            # exactly zero active bytes after synchronization and collection.
            mx.synchronize()
            gc.collect()
            mx.clear_cache()
            report["after_shutdown_memory"] = memory()
            new_threads = [t for t in threading.enumerate() if t not in baseline_threads]
            report["new_live_threads_after_shutdown"] = [{"name": t.name, "daemon": t.daemon} for t in new_threads]
            # Transformers/tqdm may start a process-global daemon monitor. Report
            # it, but do not falsely attribute that dependency thread to QMD.
            report["owned_executor_alive_after_shutdown"] = bool(executor and executor.is_worker_alive())
            if (report["owned_executor_alive_after_shutdown"] or any(not t.daemon for t in new_threads)
                    or report["after_shutdown_memory"]["metal_active_bytes"] != 0):
                report["passed"] = False
                report["cleanup_error"] = "Owned executor, non-daemon threads or active Metal allocations remain after shutdown"
        except Exception as exc:
            report["passed"] = False
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        if source_digest(repo) != report["source_sha256"]:
            report["passed"] = False
            report["source_changed_during_run"] = True
        with output.open("x") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")
        print(json.dumps({"passed": report["passed"], "receipt": str(output),
                          "error": report.get("error"), "cleanup_error": report.get("cleanup_error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
