"""W14 qualification contracts; all evidence is authored, offline fixture data."""

import copy
import importlib
import json
from types import SimpleNamespace

import pytest

from scripts.qmd_mlx import sustained
from scripts.qmd_gguf.benchmark import GGUFBenchmarkConfig, GGUFBenchmarkRunner


def complete_report():
    records = []
    for index, (stratum, tokens) in enumerate([
        ("short", [10]), ("medium", [100]), ("long", [800]),
        ("code", [150]), ("boundary_2047", [2047]),
        ("boundary_2048", [2048]), ("short", [10, 11]),
        ("mixed", [800, 100, 150, 10]), ("short", [10]),
    ], 1):
        records.append({
            "request_idx": index, "stratum": stratum, "batch_size": len(tokens),
            "token_counts": tokens, "total_tokens": sum(tokens),
            "token_count_source": "tokenizer", "wire_ms": 50.0,
            "effective_tokens_per_sec": sum(tokens) / 0.05,
            "shape": [len(tokens), 4], "finite": True, "l2_normalized": True,
        })
    records[-2]["tag"] = "concurrent_bulk_batch_0"
    records[-2]["wire_ms"] = 100.0
    records[-2]["effective_tokens_per_sec"] = sum(records[-2]["token_counts"]) / 0.1
    records[-1]["tag"] = "concurrent_interactive_query_0"
    rejection = {
        "request_idx": 10, "stratum": "boundary_2049", "batch_size": 1,
        "token_counts": [2049], "wire_ms": 5.0,
        "rejection_verified": True, "status_code": 400,
    }
    interleaving = [{
        "iteration": 0, "overlap_verified": True,
        "overlap_basis": "client_request_intervals",
        "bulk_start_mono": 10.0, "bulk_end_mono": 10.1,
        "query_start_mono": 10.02, "query_end_mono": 10.07,
        "bulk_wire_ms": 100.0, "concurrent_interactive_wire_ms": 50.0,
        "solo_baseline_fixture_ms": 25.0, "solo_baseline_p50_ms": 25.0,
        "estimated_queue_wait_ms": 25.0, "slowdown_factor": 2.0,
    }]
    report = {
        "schema_version": 2, "status": "completed", "mode": "sustained_pilot",
        "child_cleanup": {"cleaned": True}, "supervisor_cleanup": {"stopped": True},
        "synthetic": False, "descriptor": {"outputDimensions": 4},
        "config": {"mode": "pilot", "timeout_s": 120.0, "max_requests": 30,
                   "use_fake_child": False, "model_path": "/fixture/model"},
        "preflight": {
            "installed_ram_mb": 32768.0, "headroom_mb": 12000.0,
            "swap_used_mb": 100.0, "free_memory_pct": 75.0,
            "watchdog_limits": {"max_rss_mb": 7800.0,
                                "max_swap_growth_mb": 2048.0,
                                "min_free_memory_pct": 12.0},
        },
        "cold_startup": {"ready": True, "cold_startup_elapsed_s": 2.0},
        "warmup": {"status": "completed", "excluded_from_measured_metrics": True},
        "solo_baseline": {"samples_ms": [25.0, 25.0, 25.0],
                          "percentiles": sustained.compute_explicit_percentiles([25.0, 25.0, 25.0])},
        "measured_requests": records, "rejections": [rejection],
        "concurrent_load_interleaving": interleaving,
        "telemetry_samples": [
            {"timestamp": 1000.0, "elapsed_s": 1.0, "rss_mb": 1000.0,
             "swap_used_mb": 100.0, "swap_growth_mb": 0.0, "memory_free_pct": 75.0},
            {"timestamp": 1012.0, "elapsed_s": 13.0, "rss_mb": 1000.0,
             "swap_used_mb": 100.0, "swap_growth_mb": 0.0, "memory_free_pct": 75.0},
        ],
        "metal_memory_telemetry": {
            "measurement_kind": "metal_allocator",
            "http_status": {"initial": 200, "post_warmup": 200, "final": 200},
            "telemetry_complete": True,
            "initial": {"active_mb": 100.0, "peak_mb": 150.0},
            "post_warmup": {"active_mb": 150.0, "peak_mb": 200.0},
            "final": {"active_mb": 150.0, "peak_mb": 200.0},
            "delta_active_mb": 50.0, "delta_post_warmup_active_mb": 0.0,
            "peak_active_mb": 200.0,
            "provisional_stability_assessment": {
                "verdict": "provisional_stable", "rss_growth_mb": 0.0,
                "swap_growth_mb": 0.0, "metal_delta_active_mb": 50.0,
                "metal_peak_active_mb": 200.0,
            },
        },
        "strata_verification": {},
    }
    for stratum, counts in {
        "short": [10, 11], "medium": [100], "long": [800], "code": [150],
        "boundary_2047": [2047], "boundary_2048": [2048],
    }.items():
        report["strata_verification"][stratum] = {
            "item_count": len(counts), "token_counts": counts,
            "min_tokens": min(counts), "max_tokens": max(counts),
        }
    report["strata_verification"]["boundary_2049"] = {
        "policy": "explicit_rejection", "status_code": 400,
        "rejection_verified": True, "token_length": 2049,
    }
    report["metrics_summary"] = sustained.compute_metrics_summary(
        records, [rejection], interleaving,
        sustained.compute_explicit_percentiles([25.0, 25.0, 25.0]), 13.0,
    )
    return report


def evaluate(report):
    return sustained.evaluate_qualification_gates(report, sustained.SustainedRunnerConfig())


def set_path(report, path, value):
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_complete_valid_pilot_qualifies_with_original_latency_target():
    result = evaluate(complete_report())
    assert result["overall_status"] == "passed", result
    assert all(gate["passed"] is True for gate in result["gates"].values())
    assert result["gates"]["concurrent_interactive_responsiveness"]["target_p95_ms"] == 200.0
    assert result["qualification_scope"] == "bounded_pilot"
    assert result["production_qualified"] is False  # No 1000-iteration/retrieval release claim.


@pytest.mark.parametrize("path,value", [
    (("concurrent_load_interleaving", 0, "overlap_verified"), False),
    (("concurrent_load_interleaving", 0, "overlap_verified"), 1),
    (("concurrent_load_interleaving", 0, "query_start_mono"), 11.0),
    (("concurrent_load_interleaving", 0, "bulk_end_mono"), None),
    (("metal_memory_telemetry", "provisional_stability_assessment", "verdict"), "unstable"),
    (("metal_memory_telemetry", "provisional_stability_assessment", "swap_growth_mb"), 0.1),
    (("metal_memory_telemetry", "provisional_stability_assessment", "rss_growth_mb"), None),
    (("metal_memory_telemetry", "provisional_stability_assessment", "metal_delta_active_mb"), 501.0),
    (("metal_memory_telemetry", "final", "active_mb"), 601.0),
    (("metal_memory_telemetry", "final", "peak_mb"), 149.0),
    (("metal_memory_telemetry", "measurement_kind"), "process_rss_proxy"),
    (("telemetry_samples", 1, "rss_mb"), 7800.1),
    (("telemetry_samples", 1, "memory_free_pct"), 11.9),
    (("telemetry_samples", 1, "swap_used_mb"), 100.1),
    (("telemetry_samples", 1, "swap_growth_mb"), 0.1),
    (("telemetry_samples", 1, "timestamp"), float("inf")),
    (("telemetry_samples", 1, "elapsed_s"), float("nan")),
    (("cold_startup", "cold_startup_elapsed_s"), -1.0),
    (("cold_startup", "cold_startup_elapsed_s"), 30.1),
    (("cold_startup", "cold_startup_elapsed_s"), "2"),
    (("measured_requests", 0, "finite"), False),
    (("measured_requests", 0, "shape"), [2, 4]),
    (("measured_requests", 0, "total_tokens"), 11),
    (("measured_requests", 0, "token_count_source"), "whitespace"),
    (("measured_requests", 0, "wire_ms"), True),
    (("metrics_summary", "concurrent_tail_latency_ms", "concurrent_interactive_under_load", "p95"), 0.0),
    (("metrics_summary", "concurrent_tail_latency_ms", "concurrent_interactive_under_load", "count"), 2),
    (("metrics_summary", "partial_run"), True),
    (("metrics_summary", "total_measured_requests"), 999),
    (("warmup", "excluded_from_measured_metrics"), False),
    (("strata_verification", "medium", "max_tokens"), 201),
    (("strata_verification", "short", "max_tokens"), 26),
    (("strata_verification", "boundary_2049", "token_length"), 2050),
    (("rejections", 0, "token_counts"), [2050]),
    (("preflight", "headroom_mb"), 5999.0),
    (("config", "max_requests"), 100),
    (("status",), "failed"),
    (("status",), "in_progress"),
    (("synthetic",), True),
    (("schema_version",), 2.0),
    (("solo_baseline", "samples_ms"), []),
    (("solo_baseline", "samples_ms"), [25.0, None, 25.0]),
    (("metal_memory_telemetry", "telemetry_complete"), False),
    (("concurrent_load_interleaving", 0, "iteration"), True),
    (("concurrent_load_interleaving", 0, "estimated_queue_wait_ms"), None),
    (("metrics_summary", "rejected_boundary_requests"), True),
    (("child_cleanup", "cleaned"), False),
    (("supervisor_cleanup", "stopped"), False),
    (("measured_requests", 0, "stratum"), []),
    (("measured_requests", 0, "request_idx"), 99),
])
def test_each_independent_bad_evidence_fails_closed(path, value):
    report = complete_report()
    set_path(report, path, value)
    assert evaluate(report)["overall_status"] == "failed", (path, value)


@pytest.mark.parametrize("key", [
    "cold_startup", "measured_requests", "rejections", "metrics_summary",
    "concurrent_load_interleaving", "metal_memory_telemetry", "telemetry_samples",
    "preflight", "config", "warmup", "solo_baseline", "strata_verification", "status", "synthetic",
])
@pytest.mark.parametrize("replacement", [None, {}, []])
def test_missing_or_wrong_section_types_are_unqualified(key, replacement):
    report = complete_report()
    report[key] = replacement
    assert evaluate(report)["overall_status"] == "failed"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf"), True, "0"])
@pytest.mark.parametrize("path", [
    ("metal_memory_telemetry", "initial", "active_mb"),
    ("metal_memory_telemetry", "post_warmup", "peak_mb"),
    ("metal_memory_telemetry", "provisional_stability_assessment", "swap_growth_mb"),
    ("telemetry_samples", 1, "rss_mb"),
    ("telemetry_samples", 1, "swap_growth_mb"),
    ("telemetry_samples", 1, "memory_free_pct"),
    ("metrics_summary", "concurrent_tail_latency_ms", "concurrent_interactive_under_load", "p95"),
])
def test_required_numeric_evidence_is_typed_and_finite(path, bad):
    report = complete_report()
    set_path(report, path, bad)
    result = evaluate(report)
    assert result["overall_status"] == "failed", (path, bad)


def test_truthful_unstable_growth_cannot_pass_with_no_watchdog_breach():
    report = complete_report()
    memory = report["metal_memory_telemetry"]
    memory["final"] = {"active_mb": 650.1, "peak_mb": 700.0}
    memory.update(delta_active_mb=550.1, delta_post_warmup_active_mb=500.1, peak_active_mb=700.0)
    memory["provisional_stability_assessment"].update(
        verdict="unstable", metal_delta_active_mb=550.1, metal_peak_active_mb=700.0,
    )
    assert evaluate(report)["gates"]["memory_stability"]["passed"] is False


def test_post_warmup_growth_is_not_a_zero_growth_soak_certificate():
    report = complete_report()
    memory = report["metal_memory_telemetry"]
    memory["final"]["active_mb"] += 1.0
    memory["delta_active_mb"] += 1.0
    memory["delta_post_warmup_active_mb"] += 1.0
    memory["provisional_stability_assessment"]["metal_delta_active_mb"] += 1.0
    assert evaluate(report)["gates"]["memory_stability"]["passed"] is False


@pytest.mark.parametrize("record", [
    {}, {"total_tokens": None}, {"total_tokens": 0},
    {"total_tokens": 10, "token_count_source": "words"},
    {"total_tokens": float("nan"), "token_count_source": "tokenizer"},
])
def test_unavailable_tokens_propagate_to_aggregate_instead_of_zero(record):
    known = {"batch_size": 1, "token_counts": [10], "total_tokens": 10,
             "token_count_source": "tokenizer", "wire_ms": 100.0}
    unknown = {"batch_size": 1, "wire_ms": 100.0, **record}
    summary = sustained.compute_metrics_summary([known, unknown], [], [], None, 1.0)
    assert summary["total_successful_embedded_tokens"] is None
    assert summary["overall_effective_tokens_per_sec"] is None
    assert summary["token_metrics_complete"] is False
    json.dumps(summary, allow_nan=False)


def test_actual_tokenizer_counts_not_word_counts_for_every_request():
    class Client:
        def post_json(self, url, payload, **kwargs):
            assert url == "http://fixture/tokenize"
            assert payload == {"texts": ["one word", "another"]}
            return 200, {"counts": [7, 4]}

    counts = sustained.fetch_token_counts(Client(), "http://fixture", ["one word", "another"], 123.0)
    metrics = sustained.request_token_metrics(["one word", "another"], counts, 100.0)
    assert metrics["token_counts"] == [7, 4]
    assert metrics["total_tokens"] == 11
    assert metrics["effective_tokens_per_sec"] == 110.0
    assert metrics["total_words"] == 3


@pytest.mark.parametrize("payload", [{}, {"counts": []}, {"counts": [1]},
                                      {"counts": [True, 2]}, {"counts": [1.5, 2]},
                                      {"counts": [1, float("inf")]}, {"counts": [0, 2]}])
def test_malformed_token_counts_are_unavailable_not_guessed(payload):
    client = SimpleNamespace(post_json=lambda *args, **kwargs: (200, payload))
    assert sustained.fetch_token_counts(client, "http://fixture", ["a b", "c"], 123.0) is None


def test_synthetic_endpoint_counts_are_not_presented_as_real_model_tokens():
    metrics = sustained.request_token_metrics(["two words"], [900], 100.0, synthetic=True)
    assert metrics["total_tokens"] is None
    assert metrics["effective_tokens_per_sec"] is None
    assert metrics["effective_words_per_sec"] == 20.0


@pytest.mark.parametrize("config_type", [sustained.SustainedRunnerConfig, GGUFBenchmarkConfig])
@pytest.mark.parametrize("kwargs", [
    {"timeout_s": 120.1}, {"timeout_s": 0}, {"timeout_s": -1},
    {"timeout_s": float("nan")}, {"timeout_s": float("inf")},
    {"timeout_s": True}, {"max_requests": 31}, {"max_requests": 0},
    {"max_requests": -1}, {"max_requests": True}, {"max_requests": 1.5},
    {"min_headroom_mb": float("nan")},
])
def test_library_pilot_bounds_rejected_before_any_sampling(config_type, kwargs):
    with pytest.raises(ValueError):
        config_type(**kwargs)


def test_mlx_soak_bounds_are_distinct_and_validated():
    sustained.SustainedRunnerConfig(mode="soak", timeout_s=180.0, max_requests=100)
    for kwargs in [{"timeout_s": 180.1}, {"max_requests": 101}, {"mode": "unknown"}]:
        with pytest.raises(ValueError):
            sustained.SustainedRunnerConfig(**{"mode": "soak", **kwargs})


def test_gguf_has_no_soak_loop_and_cannot_claim_soak_support():
    with pytest.raises(ValueError, match="pilot"):
        GGUFBenchmarkConfig(mode="soak")


@pytest.mark.parametrize("module_name", ["scripts.qmd-mlx-sustained", "scripts.qmd-gguf-benchmark"])
@pytest.mark.parametrize("flags", [
    ["--timeout-s", "180"], ["--timeout-s", "nan"], ["--timeout-s", "0"],
    ["--max-requests", "100"], ["--max-requests", "0"],
    ["--max-requests", "-1"], ["--min-headroom-mb", "nan"],
])
def test_cli_invalid_pilot_bounds_do_not_construct_runner(monkeypatch, module_name, flags):
    module = importlib.import_module(module_name)
    name = "SustainedRunner" if "mlx" in module_name else "GGUFBenchmarkRunner"
    monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("must reject before runner construction"))
    assert module.main(["--dry-run", *flags]) == 2


@pytest.mark.parametrize("module_name", ["scripts.qmd-mlx-sustained", "scripts.qmd-gguf-benchmark"])
def test_human_cli_can_print_unavailable_token_and_memory_metrics(monkeypatch, capsys, module_name):
    module = importlib.import_module(module_name)
    report = {
        "status": "rehearsal_completed", "duration_s": 1.0,
        "metrics_summary": {"total_successful_embedded_tokens": None,
                            "overall_effective_tokens_per_sec": None},
        "metal_memory_telemetry": {"peak_active_mb": None, "delta_active_mb": None},
    }
    name = "SustainedRunner" if "mlx" in module_name else "GGUFBenchmarkRunner"
    monkeypatch.setattr(module, name, lambda config: SimpleNamespace(run=lambda: copy.deepcopy(report)))
    assert module.main(["--rehearsal"]) == 0
    assert "unavailable" in capsys.readouterr().out


def test_pilot_gate_recomputes_evidence_instead_of_trusting_passed_label(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.qmd-mlx-sustained")
    report = complete_report()
    report["status"] = "passed"
    report["qualification_evaluation"] = {"overall_status": "passed"}
    report["concurrent_load_interleaving"][0]["overlap_verified"] = False
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(module, "SustainedRunner", lambda *a, **k: pytest.fail("invalid pilot must not escalate"))
    assert module.main(["--real-model", "--model-path", "/fixture/model", "--mode", "soak",
                        "--pilot-report", str(path)]) == 2


def test_pilot_acceptance_requires_matching_model_and_real_pilot():
    report = complete_report()
    report["status"] = "passed"
    config = sustained.SustainedRunnerConfig(model_path="/fixture/model")
    assert sustained.evaluate_pilot_acceptance(report, config)["overall_status"] == "passed"
    report["config"]["model_path"] = "/different/model"
    assert sustained.evaluate_pilot_acceptance(report, config)["overall_status"] == "failed"


@pytest.mark.parametrize("bad_config", [{}, {"mode": "pilot", "timeout_s": float("nan"), "max_requests": 30}])
def test_malformed_pilot_bounds_return_failure_not_exception(bad_config):
    report = complete_report()
    report["status"] = "passed"
    report["config"] = bad_config
    result = sustained.evaluate_pilot_acceptance(report, sustained.SustainedRunnerConfig(model_path="/fixture/model"))
    assert result["overall_status"] == "failed"


@pytest.mark.parametrize("warm_growth", [0.0001, 500.0001])
def test_growth_thresholds_are_checked_before_rounding(warm_growth):
    report = complete_report()
    memory = report["metal_memory_telemetry"]
    memory["final"] = {"active_mb": 150.0 + warm_growth, "peak_mb": 1000.0}
    memory.update(sustained.compute_memory_assessment(memory, report["telemetry_samples"], 100.0))
    assert evaluate(report)["gates"]["memory_stability"]["passed"] is False


@pytest.mark.parametrize("module_name,runner_name,config_type", [
    ("scripts.qmd_mlx.sustained", "SustainedRunner", sustained.SustainedRunnerConfig),
    ("scripts.qmd_gguf.benchmark", "GGUFBenchmarkRunner", GGUFBenchmarkConfig),
])
@pytest.mark.parametrize("unavailable_concurrent", [False, True])
@pytest.mark.parametrize("frozen_process_clock", [False, True])
def test_serial_and_concurrent_runner_paths_use_tokenizer_evidence_only(
    monkeypatch, module_name, runner_name, config_type, unavailable_concurrent, frozen_process_clock,
):
    """Exercise both full runners with in-memory HTTP/process/watchdog fixtures, never a model."""
    import threading
    import time
    import numpy as np
    from scripts.qmd_mlx.watchdog import WatchdogCheckResult

    module = importlib.import_module(module_name)
    fixture_counts = {}
    for texts, start in [(sustained.SHORT_FIXTURES, 13), (sustained.MEDIUM_FIXTURES, 100),
                         (sustained.LONG_FIXTURES, 800), (sustained.CODE_FIXTURES, 150)]:
        fixture_counts.update((text, start + index) for index, text in enumerate(texts))
    instance = {}
    bulk_inflight = threading.Event()
    query_finished = threading.Event()
    concurrent_texts = [sustained.LONG_FIXTURES[0], sustained.MEDIUM_FIXTURES[0],
                        sustained.CODE_FIXTURES[0], sustained.SHORT_FIXTURES[0]]

    class Child:
        pid = 4321
        returncode = None

        def __init__(self, command, **kwargs):
            instance["token"] = kwargs["env"].get("MLX_INSTANCE_TOKEN") or kwargs["env"].get("GGUF_INSTANCE_TOKEN")

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        kill = terminate

        def wait(self, **kwargs):
            return self.returncode

    class Client:
        def __init__(self, **kwargs):
            pass

        def get_json(self, url, **kwargs):
            if url.endswith("/health"):
                return 200, {"ready": True, "instance_token": instance["token"]}
            if url.endswith("/descriptor"):
                return 200, {"outputDimensions": 4}
            assert url.endswith("/memory")
            return 200, {"active_mb": 100.0, "peak_mb": 100.0}

        def post_json(self, url, payload, **kwargs):
            texts = payload["texts"]
            counts = [fixture_counts.get(text, len(text.split())) for text in texts]
            if max(counts) > 2048:
                return 400, {"error": "Text token length (2049) exceeds max_length (2048)"}
            if url.endswith("/tokenize"):
                if unavailable_concurrent and texts == [sustained.LONG_FIXTURES[0], sustained.MEDIUM_FIXTURES[0],
                                                       sustained.CODE_FIXTURES[0], sustained.SHORT_FIXTURES[0]]:
                    return 503, {"error": "fixture tokenizer unavailable"}
                return 200, {"counts": counts}
            assert url.endswith("/embed")
            if texts == concurrent_texts:
                # Keep the mock bulk call open until its query has actually entered.
                # A scheduling delay must not turn a valid fixture into nonoverlap.
                query_finished.clear()
                bulk_inflight.set()
                assert query_finished.wait(timeout=5.0), "fixture concurrent query never entered"
                bulk_inflight.clear()
            elif bulk_inflight.is_set():
                time.sleep(0.002)
                query_finished.set()
            else:
                time.sleep(0.002)
            return 200, {"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in texts]}

        def close(self):
            pass

    class Watchdog:
        def __init__(self, **kwargs):
            self.frozen_timestamp = time.time()

        def check_step(self):
            return WatchdogCheckResult(timestamp=self.frozen_timestamp if frozen_process_clock else time.time(),
                                      healthy=True, metrics={
                "rss_mb": 1000.0, "swap_used_mb": 100.0,
                "swap_growth_mb": 0.0, "memory_free_pct": 75.0,
            })

    runner_type = getattr(module, runner_name)
    monkeypatch.setattr(module.subprocess, "Popen", Child)
    monkeypatch.setattr(module, "SmokeHttpClient", Client)
    monkeypatch.setattr(module, "MLXWatchdog", Watchdog)
    monkeypatch.setattr(runner_type, "validate_preflight", lambda self: complete_report()["preflight"])
    config = config_type(model_path="/fixture/model", real_model_opt_in=True, timeout_s=30.0, max_requests=25)
    report = runner_type(config, sampler=SimpleNamespace()).run()
    assert "primary_failure" not in report, report.get("error")
    assert report["child_cleanup"]["cleaned"] is True
    gates = report["qualification_evaluation"]["gates"]
    assert gates["genuine_concurrent_overlap"]["passed"] is True
    assert gates["process_telemetry_coverage"]["passed"] is (not frozen_process_clock)
    assert gates["process_resource_limits"]["passed"] is (not frozen_process_clock)
    if frozen_process_clock:
        assert report["status"] == "failed"
        assert gates["memory_stability"]["passed"] is False
        assert report["metal_memory_telemetry"]["telemetry_complete"] is False
    concurrent_bulk = [row for row in report["measured_requests"] if row["tag"].startswith("concurrent_bulk_batch_")]
    assert len(concurrent_bulk) == 3
    for row in concurrent_bulk:
        if unavailable_concurrent:
            assert row["token_counts"] is None
            assert row["effective_tokens_per_sec"] is None
        else:
            assert row["token_counts"] == [800, 100, 150, 13]
            assert row["total_tokens"] == 1063
    if unavailable_concurrent:
        assert report["metrics_summary"]["overall_effective_tokens_per_sec"] is None
        assert report["metrics_summary"]["total_successful_embedded_tokens"] is None
    else:
        assert report["metrics_summary"]["token_metrics_complete"] is True
        assert np.isfinite(report["metrics_summary"]["overall_effective_tokens_per_sec"])
        if module_name == "scripts.qmd_mlx.sustained" and not frozen_process_clock:
            # This is a mock acceptance contract, not real model or GPU evidence.
            assert report["status"] == "passed", report["qualification_evaluation"]
            persisted = json.loads(json.dumps(report, allow_nan=False))
            assert sustained.evaluate_pilot_acceptance(persisted, config)["overall_status"] == "passed"
            persisted["concurrent_load_interleaving"].pop()
            persisted["metrics_summary"] = sustained.compute_metrics_summary(
                persisted["measured_requests"], persisted["rejections"], persisted["concurrent_load_interleaving"],
                persisted["solo_baseline"]["percentiles"],
                persisted["metrics_summary"]["cumulative_wall_accounting"]["total_benchmark_elapsed_s"],
            )
            result = sustained.evaluate_pilot_acceptance(persisted, config)
            assert result["overall_status"] == "failed"
            assert result["gates"]["genuine_concurrent_overlap"]["passed"] is False
            assert result["gates"]["concurrent_interactive_responsiveness"]["sample_count"] == 3


def test_gguf_cli_refuses_soak_and_real_soak_cannot_skip_pilot(monkeypatch):
    gguf = importlib.import_module("scripts.qmd-gguf-benchmark")
    mlx = importlib.import_module("scripts.qmd-mlx-sustained")
    monkeypatch.setattr(gguf, "GGUFBenchmarkRunner", lambda *a, **k: pytest.fail("no spawn"))
    monkeypatch.setattr(mlx, "SustainedRunner", lambda *a, **k: pytest.fail("no spawn"))
    assert gguf.main(["--rehearsal", "--mode", "soak"]) == 2
    assert mlx.main(["--real-model", "--model-path", "/fixture/model", "--mode", "soak", "--skip-pilot-gate"]) == 2


@pytest.mark.parametrize("module_name,mode,timeout_s,requests", [
    ("scripts.qmd-mlx-sustained", "pilot", 120.0, 30),
    ("scripts.qmd-mlx-sustained", "soak", 180.0, 100),
    ("scripts.qmd-gguf-benchmark", "pilot", 120.0, 30),
])
def test_cli_mode_defaults_are_exact_and_no_silent_clamping(monkeypatch, module_name, mode, timeout_s, requests):
    module = importlib.import_module(module_name)
    runner_name = "SustainedRunner" if "mlx" in module_name else "GGUFBenchmarkRunner"

    def runner(config):
        assert config.mode == mode
        assert config.timeout_s == timeout_s
        assert config.max_requests == requests
        return SimpleNamespace(run=lambda: {"status": "dry_run_completed"})

    monkeypatch.setattr(module, runner_name, runner)
    assert module.main(["--dry-run", "--mode", mode, "--json"]) == 0


def test_percentile_overflow_is_unavailable_and_json_finite():
    result = sustained.compute_explicit_percentiles([1e308, 1e308])
    assert result["complete"] is False
    assert result["mean"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("idle,age,expected_state", [(True, None, "idle"), (False, 0.1, "ready")])
def test_gguf_legacy_health_adapter_preserves_truthful_ready_progress(monkeypatch, idle, age, expected_state):
    from scripts.qmd_gguf import benchmark

    payload = {"status": "ok", "ready": True, "backend": "gguf", "worker_alive": True,
               "worker_idle": idle, "completed_sequence": 2, "active_job_age_s": age,
               "pid": 4321, "instance_token": "fixture-instance"}
    monkeypatch.setattr(benchmark.MLXWatchdog, "_default_health_probe", staticmethod(lambda *a: (True, payload, 200)))
    healthy, adapted, status = benchmark.probe_gguf_health("127.0.0.1", 1234, 1.0)
    assert healthy is True and status == 200
    assert adapted == {**payload, "state": expected_state, "state_source": "legacy_gguf_status_adapter"}
    assert "state" not in payload


@pytest.mark.parametrize("override", [
    {"state": "failed"}, {"state": None}, {"status": "failed"}, {"ready": False},
    {"worker_alive": False}, {"worker_idle": "true"}, {"completed_sequence": True},
    {"completed_sequence": -1}, {"active_job_age_s": float("nan")},
    {"active_job_age_s": 0.1}, {"backend": "mlx"},
    {"worker_idle": False, "active_job_age_s": 10 ** 1000},
])
def test_gguf_health_adapter_does_not_repair_unhealthy_or_malformed_fields(monkeypatch, override):
    from scripts.qmd_gguf import benchmark

    payload = {"status": "ok", "ready": True, "backend": "gguf", "worker_alive": True,
               "worker_idle": True, "completed_sequence": 2, "active_job_age_s": None, **override}
    monkeypatch.setattr(benchmark.MLXWatchdog, "_default_health_probe", staticmethod(lambda *a: (True, payload, 200)))
    assert benchmark.probe_gguf_health("127.0.0.1", 1234, 1.0) == (True, payload, 200)


def test_gguf_legacy_starting_status_is_not_promoted_to_ready(monkeypatch):
    from scripts.qmd_gguf import benchmark

    payload = {"status": "starting", "ready": False, "backend": "gguf", "worker_alive": True,
               "worker_idle": True, "completed_sequence": 0, "active_job_age_s": None}
    monkeypatch.setattr(benchmark.MLXWatchdog, "_default_health_probe", staticmethod(lambda *a: (True, payload, 200)))
    _, adapted, _ = benchmark.probe_gguf_health("127.0.0.1", 1234, 1.0)
    assert adapted["state"] == "starting"
    assert adapted["ready"] is False
