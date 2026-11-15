"""Offline receipt/telemetry regressions; fixtures are not GPU qualification evidence."""

import copy

import pytest

from scripts.qmd_mlx import sustained
from test_audit_qualification import complete_report


def recompute_summary(report):
    report["metrics_summary"] = sustained.compute_metrics_summary(
        report["measured_requests"], report["rejections"],
        report["concurrent_load_interleaving"], report["solo_baseline"]["percentiles"], 13.0,
    )


def append_concurrent_request(report, role, iteration=1, wire_ms=50.0):
    source_tag = f"concurrent_{role}_0"
    record = copy.deepcopy(next(row for row in report["measured_requests"] if row.get("tag") == source_tag))
    record.update(
        request_idx=1 + max(row["request_idx"] for row in report["measured_requests"] + report["rejections"]),
        tag=f"concurrent_{role}_{iteration}", wire_ms=wire_ms,
        effective_tokens_per_sec=sustained._rate(record["total_tokens"], wire_ms),
    )
    report["measured_requests"].append(record)
    return record


def append_pair(report, query_ms=50.0, *, receipt=False):
    append_concurrent_request(report, "bulk_batch", wire_ms=100.0)
    append_concurrent_request(report, "interactive_query", wire_ms=query_ms)
    if receipt:
        row = copy.deepcopy(report["concurrent_load_interleaving"][0])
        row.update(
            iteration=1, bulk_start_mono=11.0, bulk_end_mono=11.1,
            query_start_mono=11.02, query_end_mono=11.02 + query_ms / 1000.0,
            concurrent_interactive_wire_ms=query_ms,
            estimated_queue_wait_ms=max(0.0, query_ms - 25.0),
            slowdown_factor=round(query_ms / 25.0, 2),
        )
        report["concurrent_load_interleaving"].append(row)
    recompute_summary(report)


def evaluate(report, *, pilot=False):
    config = sustained.SustainedRunnerConfig(model_path="/fixture/model")
    if pilot:
        report["status"] = "passed"
        # A historical passed label must not bypass recomputation.
        report["qualification_evaluation"] = {"overall_status": "passed"}
        return sustained.evaluate_pilot_acceptance(report, config)
    return sustained.evaluate_qualification_gates(report, config)


@pytest.mark.parametrize("pilot", [False, True])
def test_omitted_slow_pair_receipt_cannot_hide_query_or_qualify(pilot):
    report = complete_report()
    append_pair(report, query_ms=500.0)
    result = evaluate(report, pilot=pilot)
    assert result["overall_status"] == "failed", result
    assert result["gates"]["genuine_concurrent_overlap"]["passed"] is False
    gate = result["gates"]["concurrent_interactive_responsiveness"]
    assert gate["passed"] is False
    assert gate["sample_count"] == 2
    assert gate["measured_p95_ms"] == sustained.compute_explicit_percentiles([50.0, 500.0])["p95"]
    # Rebuilding a self-consistent summary cannot repair missing overlap evidence.
    assert result["gates"]["consistent_metrics"]["passed"] is True
    assert report["metrics_summary"]["concurrent_tail_latency_ms"]["concurrent_interactive_under_load"]["count"] == 2


@pytest.mark.parametrize("pilot", [False, True])
def test_complete_second_pair_with_truthful_receipt_still_qualifies(pilot):
    report = complete_report()
    append_pair(report, query_ms=75.0, receipt=True)
    result = evaluate(report, pilot=pilot)
    assert result["overall_status"] == "passed", result
    assert result["gates"]["genuine_concurrent_overlap"]["sample_count"] == 2
    assert result["gates"]["concurrent_interactive_responsiveness"]["sample_count"] == 2
    assert result["production_qualified"] is False


@pytest.mark.parametrize("role", ["bulk_batch", "interactive_query"])
@pytest.mark.parametrize("iteration", [0, 1])
def test_duplicate_or_orphan_raw_concurrent_request_fails(role, iteration):
    report = complete_report()
    append_concurrent_request(report, role, iteration=iteration)
    recompute_summary(report)
    result = evaluate(report)
    assert result["overall_status"] == "failed"
    assert result["gates"]["genuine_concurrent_overlap"]["passed"] is False


@pytest.mark.parametrize("iteration", [0, 1])
def test_duplicate_or_orphan_receipt_cannot_invent_latency_samples(iteration):
    report = complete_report()
    receipt = copy.deepcopy(report["concurrent_load_interleaving"][0])
    receipt["iteration"] = iteration
    report["concurrent_load_interleaving"].append(receipt)
    recompute_summary(report)
    result = evaluate(report)
    assert result["gates"]["genuine_concurrent_overlap"]["passed"] is False
    assert result["gates"]["concurrent_interactive_responsiveness"]["sample_count"] == 1
    tail = report["metrics_summary"]["concurrent_tail_latency_ms"]
    assert tail["concurrent_interactive_under_load"]["count"] == 1
    assert tail["concurrent_bulk_batches"]["count"] == 1


@pytest.mark.parametrize("tag", [
    "concurrent_interactive_query_01", "concurrent_interactive_query_-1",
    "concurrent_interactive_query_+1", "concurrent_interactive_query_1.0",
    "concurrent_interactive_query_", "concurrent_unknown_1",
    ["concurrent_interactive_query_1"], {"tag": "concurrent_interactive_query_1"},
])
def test_malformed_extra_concurrent_tag_cannot_escape_reconciliation(tag):
    report = complete_report()
    row = append_concurrent_request(report, "interactive_query")
    row["tag"] = tag
    recompute_summary(report)
    result = evaluate(report)
    assert result["overall_status"] == "failed"
    assert result["gates"]["genuine_concurrent_overlap"]["passed"] is False


def test_concurrent_tag_in_rejections_is_not_an_unaccounted_successful_pair():
    report = complete_report()
    report["rejections"][0]["tag"] = "concurrent_bulk_batch_1"
    assert evaluate(report)["gates"]["genuine_concurrent_overlap"]["passed"] is False


def test_measured_rejection_flag_cannot_remove_slow_concurrent_query_from_tail():
    report = complete_report()
    append_pair(report, query_ms=500.0)
    report["measured_requests"][-1]["rejection_verified"] = True
    recompute_summary(report)
    result = evaluate(report)
    assert result["overall_status"] == "failed"
    assert result["gates"]["genuine_concurrent_overlap"]["passed"] is False
    assert result["gates"]["concurrent_interactive_responsiveness"]["sample_count"] == 2
    assert result["gates"]["concurrent_interactive_responsiveness"]["passed"] is False


def test_receipt_latency_and_intervals_must_still_match_exact_raw_requests():
    report = complete_report()
    receipt = report["concurrent_load_interleaving"][0]
    receipt.update(query_end_mono=10.08, concurrent_interactive_wire_ms=60.0,
                   estimated_queue_wait_ms=35.0, slowdown_factor=2.4)
    recompute_summary(report)
    result = evaluate(report)
    assert result["gates"]["genuine_concurrent_overlap"]["passed"] is False
    assert result["gates"]["concurrent_interactive_responsiveness"]["measured_p95_ms"] == 50.0


def test_forged_receipt_selected_tail_cannot_override_raw_latency_gate():
    report = complete_report()
    old_tail = copy.deepcopy(report["metrics_summary"]["concurrent_tail_latency_ms"])
    append_pair(report, query_ms=500.0)
    report["metrics_summary"]["concurrent_tail_latency_ms"] = old_tail
    result = evaluate(report)
    assert result["gates"]["consistent_metrics"]["passed"] is False
    gate = result["gates"]["concurrent_interactive_responsiveness"]
    assert gate["passed"] is False
    assert gate["sample_count"] == 2
    assert gate["measured_p95_ms"] > 200.0


@pytest.mark.parametrize("wire_ms", [None, True, "50", float("nan"), float("inf"), -1.0])
def test_bad_raw_concurrent_latency_stays_in_sample_count_and_is_not_zero(wire_ms):
    report = complete_report()
    append_pair(report, query_ms=wire_ms)
    result = evaluate(report)
    gate = result["gates"]["concurrent_interactive_responsiveness"]
    assert gate["passed"] is False
    assert gate["sample_count"] == 2
    assert gate["measured_p95_ms"] is None
    tail = report["metrics_summary"]["concurrent_tail_latency_ms"]["concurrent_interactive_under_load"]
    assert tail["count"] == 2
    assert tail["complete"] is False
    assert tail["p95"] is None


def test_zero_raw_concurrent_latency_is_not_a_responsive_query():
    report = complete_report()
    append_pair(report, query_ms=0.0)
    assert evaluate(report)["gates"]["concurrent_interactive_responsiveness"]["passed"] is False


@pytest.mark.parametrize("iteration", [None, True, -1, 0.0, [], {}, 10 ** 1000])
def test_invalid_receipt_iteration_is_not_normalized_or_allowed_to_crash(iteration):
    report = complete_report()
    report["concurrent_load_interleaving"][0]["iteration"] = iteration
    assert evaluate(report)["gates"]["genuine_concurrent_overlap"]["passed"] is False


@pytest.mark.parametrize("freeze", ["both", "timestamp", "elapsed_s", "duplicate_interior"])
@pytest.mark.parametrize("pilot", [False, True])
def test_frozen_process_sampling_is_unavailable_even_after_reassessment(freeze, pilot):
    report = complete_report()
    samples = report["telemetry_samples"]
    if freeze == "duplicate_interior":
        samples.insert(1, copy.deepcopy(samples[0]))
    else:
        for key in (("timestamp", "elapsed_s") if freeze == "both" else (freeze,)):
            samples[1][key] = samples[0][key]
    memory = report["metal_memory_telemetry"]
    assessment = sustained.compute_memory_assessment(memory, samples, 100.0)
    assert assessment["telemetry_complete"] is False
    assert assessment["provisional_stability_assessment"]["verdict"] == "unavailable"
    memory.update(assessment)
    result = evaluate(report, pilot=pilot)
    assert result["overall_status"] == "failed"
    assert result["gates"]["memory_stability"]["passed"] is False
    assert result["gates"]["process_resource_limits"]["passed"] is False


@pytest.mark.parametrize("start,end", [
    (1.0, 1.5),       # All process samples precede the measured workload.
    (3.0, 13.0),      # No process sample by the end of cold startup.
    (1.0, 12.0),      # End of measured work is unobserved.
    (1.0, 13.1),      # Future sample outside reported execution bounds.
])
@pytest.mark.parametrize("pilot", [False, True])
def test_progressing_samples_must_bracket_startup_and_measured_finish(start, end, pilot):
    report = complete_report()
    samples = report["telemetry_samples"]
    for sample, elapsed in zip(samples, [start, end]):
        sample.update(elapsed_s=elapsed, timestamp=999.0 + elapsed)
    memory = report["metal_memory_telemetry"]
    memory.update(sustained.compute_memory_assessment(memory, samples, 100.0))
    result = evaluate(report, pilot=pilot)
    assert result["overall_status"] == "failed", result
    assert result["gates"]["memory_stability"]["passed"] is False
    assert result["gates"]["process_resource_limits"]["passed"] is False


def test_endpoint_rounding_tolerance_is_bounded_not_a_sampling_interval():
    report = complete_report()
    sample = report["telemetry_samples"][-1]
    sample.update(elapsed_s=12.96, timestamp=1011.96)
    assert evaluate(report)["overall_status"] == "passed"
    sample.update(elapsed_s=12.94, timestamp=1011.94)
    assert evaluate(report)["overall_status"] == "failed"


def test_subcentisecond_sample_progress_is_not_rounded_away():
    samples = [
        {"timestamp": 1000.0001, "elapsed_s": 0.0001, "rss_mb": 1000.0,
         "swap_used_mb": 100.0, "swap_growth_mb": 0.0, "memory_free_pct": 75.0},
        {"timestamp": 1000.0002, "elapsed_s": 0.0002, "rss_mb": 1000.0,
         "swap_used_mb": 100.0, "swap_growth_mb": 0.0, "memory_free_pct": 75.0},
    ]
    assert sustained._valid_memory_samples(samples) is True
    memory = complete_report()["metal_memory_telemetry"]
    assert sustained.compute_memory_assessment(memory, samples, 100.0)["telemetry_complete"] is True
