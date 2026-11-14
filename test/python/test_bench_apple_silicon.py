"""CPU-only validation of the Apple Silicon comparison runner."""
import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench_apple_silicon.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("bench_apple_silicon", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentile_interpolates_and_rejects_empty():
    module = load_runner()
    assert module.percentile([10, 20, 30, 40], 50) == 25
    assert module.percentile([10, 20, 30, 40], 95) == pytest.approx(38.5)
    with pytest.raises(ValueError):
        module.percentile([], 95)


def test_summary_keeps_samples_and_counts_actual_texts():
    module = load_runner()
    result = module.summarize([100, 200], texts_per_request=4)
    assert result["samples_ms"] == [100, 200]
    assert result["requests"] == 2
    assert result["texts_processed"] == 8
    assert result["texts_per_second"] == pytest.approx(8 / 0.3)


def test_budget_parser_rejects_unbounded_or_nonpositive_runs():
    module = load_runner()
    for value in ["0", "-1", "201", "nan"]:
        with pytest.raises(Exception):
            module.bounded_repetitions(value)
    assert module.bounded_repetitions("200") == 200


def test_local_model_validation_never_treats_repo_id_as_download():
    module = load_runner()
    with pytest.raises(ValueError, match="local"):
        module.validate_model_path("sentence-transformers/all-MiniLM-L6-v2")


def test_workload_is_deterministic_and_no_empty_texts():
    module = load_runner()
    assert module.workload() == module.workload()
    assert len(module.workload()) == 4
    assert all(isinstance(t, str) and t.strip() for t in module.workload())
