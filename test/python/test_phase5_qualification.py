"""
test_phase5_qualification.py — Unit & Offline Integration Tests for Phase 5 Supervisor
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from scripts.qmd_mlx.phase5_e2e_qualification import Phase5Supervisor, Phase5SupervisorConfig
from scripts.qmd_mlx.watchdog import SystemMemorySampler, SystemMetricsError


def run_isolated_ts(script, args):
    """Only local executables, scratch HOME/TMPDIR, no inherited QMD settings."""
    executable = shutil.which("bun")
    assert executable is not None, "Local Bun executable required by this fixture"
    with tempfile.TemporaryDirectory(prefix="qmd-phase5-cli-", dir=os.environ.get("TMPDIR")) as root:
        env = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL"}}
        for name in ("home", "tmp", "cache", "config"):
            os.makedirs(os.path.join(root, name))
        env.update({"HOME": os.path.join(root, "home"), "TMPDIR": os.path.join(root, "tmp"),
            "TMP": os.path.join(root, "tmp"), "TEMP": os.path.join(root, "tmp"),
            "XDG_CACHE_HOME": os.path.join(root, "cache"), "XDG_CONFIG_HOME": os.path.join(root, "config"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
        return subprocess.run([executable, script, *args], cwd=os.path.dirname(os.path.dirname(script)),
            env=env, capture_output=True, text=True, timeout=10)


def test_phase5_supervisor_config_defaults():
    config = Phase5SupervisorConfig()
    assert config.mlx_port == 8797
    assert config.mlx_control_port == 8798
    assert config.timeout_s == 120.0
    assert config.min_headroom_mb == 6000.0
    assert config.host == "127.0.0.1"
    assert config.skip_gguf is True
    assert "phase5-isolated-e2e.json" in config.output_json


def test_phase5_supervisor_preflight_sufficient_headroom():
    config = Phase5SupervisorConfig(min_headroom_mb=6000.0)
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 0.0
    mock_sampler.get_memory_free_pct.return_value = 65.0

    supervisor = Phase5Supervisor(config, sampler=mock_sampler)
    info = supervisor._check_preflight_headroom()
    assert info["installed_ram_mb"] == 32768.0
    assert info["headroom_mb"] == 16000.0
    assert info["swap_used_mb"] == 0.0
    assert info["memory_free_pct"] == 65.0


def test_phase5_supervisor_preflight_insufficient_headroom_fails_closed():
    config = Phase5SupervisorConfig(min_headroom_mb=6000.0)
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 4500.0
    mock_sampler.get_swap_used_mb.return_value = 120.0
    mock_sampler.get_memory_free_pct.return_value = 15.0

    supervisor = Phase5Supervisor(config, sampler=mock_sampler)
    with pytest.raises(RuntimeError, match="Insufficient memory headroom"):
        supervisor._check_preflight_headroom()


def test_phase5_supervisor_preflight_telemetry_error_fails_closed():
    config = Phase5SupervisorConfig(min_headroom_mb=6000.0)
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.side_effect = SystemMetricsError("vm_stat failed")

    supervisor = Phase5Supervisor(config, sampler=mock_sampler)
    with pytest.raises(SystemMetricsError, match="Preflight memory sampling failed"):
        supervisor._check_preflight_headroom()


def test_phase5_supervisor_rejection_of_live_daemon_port_8787():
    # Config pointing to production port 8787 must fail closed in supervisor preflight
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 0.0
    mock_sampler.get_memory_free_pct.return_value = 65.0

    config_bad_port = Phase5SupervisorConfig(mlx_port=8787, mlx_control_port=8798)
    supervisor = Phase5Supervisor(config_bad_port, sampler=mock_sampler)

    # SmokeStageSupervisor validate_preflight refuses port 8787
    with pytest.raises(ValueError, match="Refusing to use port 8787"):
        from scripts.qmd_mlx.supervisor import StageSupervisorConfig, SmokeStageSupervisor
        s_cfg = StageSupervisorConfig(
            cmd=["dummy"],
            port=config_bad_port.mlx_port,
            control_port=config_bad_port.mlx_control_port,
        )
        s = SmokeStageSupervisor(s_cfg, sampler=mock_sampler)
        s.validate_preflight()


def test_phase5_supervisor_missing_model_fails_closed_without_autodownload():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "report.json")
        config = Phase5SupervisorConfig(
            mlx_model="/nonexistent/model/path/qwen3-4b",
            output_json=out_json,
            min_headroom_mb=4000.0,
        )
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 16000.0
        mock_sampler.get_swap_used_mb.return_value = 0.0
        mock_sampler.get_memory_free_pct.return_value = 65.0

        supervisor = Phase5Supervisor(config, sampler=mock_sampler)
        report = supervisor.run()

        assert report["status"] == "failed"
        assert any("Explicit MLX model path missing" in err for err in report["errors"])
        assert os.path.exists(out_json)


def test_phase5_supervisor_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "dry_run_report.json")
        config = Phase5SupervisorConfig(
            dry_run=True,
            output_json=out_json,
            min_headroom_mb=4000.0,
        )
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 16000.0
        mock_sampler.get_swap_used_mb.return_value = 0.0
        mock_sampler.get_memory_free_pct.return_value = 65.0
        mock_sampler.compute_conservative_defaults.return_value = MagicMock()

        supervisor = Phase5Supervisor(config, sampler=mock_sampler)
        report = supervisor.run()

        assert report["status"] == "dry_run_completed"
        assert os.path.exists(out_json)


def test_phase5_ts_runner_launch_token_assertion():
    """Verifies that phase5_e2e_runner.ts rejects direct unmanaged CLI execution without launch token."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ts_script = os.path.join(repo_root, "scripts", "phase5_e2e_runner.ts")

    # Run without --launch-token and without QMD_PHASE5_LAUNCH_TOKEN
    proc = run_isolated_ts(ts_script, [])
    assert proc.returncode != 0
    assert "Direct TypeScript execution rejected" in proc.stderr


def test_phase5_recovery_worker_launch_token_assertion():
    """Verifies that phase5_recovery_worker.ts rejects direct unmanaged CLI execution without launch token."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    worker_script = os.path.join(repo_root, "scripts", "phase5_recovery_worker.ts")

    proc = run_isolated_ts(worker_script, [])
    assert proc.returncode != 0
    assert "Direct TypeScript recovery worker execution rejected" in proc.stderr


def test_subordinate_process_runner_success():
    """Verifies that run_subordinate_process completes successful commands and captures bounded logs."""
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    deadline = time.monotonic() + 5.0
    res = run_subordinate_process(
        cmd=[sys.executable, "-c", "print('hello subordinate')"],
        deadline=deadline,
    )
    assert res.exit_code == 0
    assert "hello subordinate" in res.captured_logs
    assert not res.timed_out
    assert not res.breached
    assert res.error is None


def test_subordinate_process_runner_timeout():
    """Verifies that run_subordinate_process terminates and reaps stalled child upon deadline breach."""
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    deadline = time.monotonic() + 0.2
    t0 = time.monotonic()
    res = run_subordinate_process(
        cmd=[sys.executable, "-c", "import time; time.sleep(10)"],
        deadline=deadline,
        poll_interval_s=0.02,
    )
    assert res.timed_out is True
    assert res.duration_s < 2.0
    assert "exceeded monotonic deadline" in (res.error or "")


def test_subordinate_process_runner_breach_detection():
    """Verifies that run_subordinate_process immediately kills child when breach callback fires."""
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    call_count = [0]
    def mock_check_breach(label: str):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("Synthetic watchdog breach")

    deadline = time.monotonic() + 5.0
    res = run_subordinate_process(
        cmd=[sys.executable, "-c", "import time; time.sleep(10)"],
        deadline=deadline,
        check_breach=mock_check_breach,
        poll_interval_s=0.02,
    )
    assert res.breached is True
    assert "Synthetic watchdog breach" in (res.error or "")


def test_subordinate_stall_fake_nonmodel_sentinel_survives():
    """
    Proves that when a subordinate child process stalls and is reaped on deadline timeout,
    an external/sentinel non-model background process survives intact without corruption or termination.
    """
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    # Spawn an independent background sentinel process
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        sentinel_pid = sentinel.pid
        assert sentinel.poll() is None, "Sentinel should be running"

        # Run subordinate runner on a stalling child with short deadline
        deadline = time.monotonic() + 0.2
        res = run_subordinate_process(
            cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
            deadline=deadline,
            poll_interval_s=0.02,
        )

        assert res.timed_out is True
        assert res.pid is not None
        assert res.pid != sentinel_pid

        # Assert sentinel survived intact
        assert sentinel.poll() is None, "Sentinel process must survive subordinate process timeout and reap"
    finally:
        if sentinel.poll() is None:
            sentinel.terminate()
            try:
                sentinel.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                sentinel.kill()
                sentinel.wait(timeout=1.0)


def test_preflight_config_validation_nan_inf_negative():
    """Verifies that StageSupervisorConfig and Phase5Supervisor reject NaN, inf, negative, and conflicting configs."""
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 0.0
    mock_sampler.get_memory_free_pct.return_value = 65.0

    # 1. NaN timeout
    cfg_nan = Phase5SupervisorConfig(timeout_s=float("nan"))
    sup_nan = Phase5Supervisor(cfg_nan, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must not be NaN or Inf"):
        sup_nan._check_preflight_headroom()

    # 2. Inf headroom
    cfg_inf = Phase5SupervisorConfig(min_headroom_mb=float("inf"))
    sup_inf = Phase5Supervisor(cfg_inf, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must not be NaN or Inf"):
        sup_inf._check_preflight_headroom()

    # 3. Negative timeout
    cfg_neg = Phase5SupervisorConfig(timeout_s=-10.0)
    sup_neg = Phase5Supervisor(cfg_neg, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must be >="):
        sup_neg._check_preflight_headroom()

    # 4. Same ports
    cfg_same_port = Phase5SupervisorConfig(mlx_port=8797, mlx_control_port=8797)
    sup_same_port = Phase5Supervisor(cfg_same_port, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must be distinct"):
        sup_same_port._check_preflight_headroom()


def test_preflight_failure_report_persisted():
    """Verifies that a preflight configuration or telemetry error persists a failure report JSON fail-closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "preflight_fail.json")
        cfg_bad = Phase5SupervisorConfig(
            timeout_s=-1.0,
            output_json=out_json,
        )
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 16000.0
        mock_sampler.get_swap_used_mb.return_value = 0.0
        mock_sampler.get_memory_free_pct.return_value = 65.0

        supervisor = Phase5Supervisor(cfg_bad, sampler=mock_sampler)
        report = supervisor.run()

        assert report["status"] == "failed"
        assert len(report["errors"]) > 0
        assert "Preflight" in report["errors"][0]
        assert os.path.exists(out_json)
        with open(out_json, "r") as f:
            saved = json.load(f)
            assert saved["status"] == "failed"


def recovery_receipt_fixture():
    # Synthetic report-schema fixture only. The TS audit separately executes
    # and reaps actual worker PIDs against a synthetic HTTP inference endpoint.
    return {
        "mode": "fresh_process_recovery", "freshProcessRecovery": "passed",
        "consecutiveSequenceVerified": True, "logicalStateInvariantOnMismatch": True,
        "workers": [
            {"pid": 101, "exitCode": None, "signal": "SIGKILL", "timedOut": False},
            {"pid": 102, "exitCode": 0, "signal": None, "timedOut": False},
        ],
    }


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(workers=[]),
    lambda r: r["workers"][1].update(pid=101),
    lambda r: r["workers"][0].update(pid=True),
    lambda r: r["workers"][0].update(signal=None, exitCode=0),
    lambda r: r["workers"][1].update(timedOut=True),
    lambda r: r["workers"][1].update(exitCode=1),
    lambda r: r["workers"][0].pop("exitCode"),
    lambda r: r.update(freshProcessRecovery="pending"),
    lambda r: r.update(logicalStateInvariantOnMismatch=False),
])
def test_recovery_receipts_refuse_missing_or_false_fresh_process_evidence(mutation):
    from scripts.qmd_mlx.phase5_e2e_qualification import fresh_process_recovery_verified
    receipt = recovery_receipt_fixture()
    assert fresh_process_recovery_verified(receipt)
    mutation(receipt)
    assert not fresh_process_recovery_verified(receipt)


@pytest.fixture
def phase5_composition(tmp_path, monkeypatch):
    """Authored composition receipts only: no process, listener, or model is launched."""
    import time
    from contextlib import contextmanager
    from types import SimpleNamespace
    from scripts.qmd_mlx import phase5_e2e_qualification as phase5
    from scripts.qmd_mlx.supervisor import SubordinateProcessResult

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("NODE_OPTIONS", "do-not-inherit")
    monkeypatch.setenv("QMD_MLX_EMBED_URL", "http://127.0.0.1:8787")
    sampler = MagicMock(spec=SystemMemorySampler)
    sampler.get_installed_ram_mb.return_value = 32768.0
    sampler.get_memory_headroom_mb.return_value = 16000.0
    sampler.get_swap_used_mb.return_value = 0.0
    sampler.get_memory_free_pct.return_value = 65.0
    responses = {
        "/descriptor": (200, {"model": "synthetic", "outputDimensions": 4}),
        "/memory": (200, {"active_mb": 100.0, "peak_mb": 150.0, "model_mb": 80.0}),
        "/stats": (200, {"total_requests": 10, "avg_ms": 1.5, "compiled_shapes": 1, "uptime_sec": 2.0}),
    }
    health = {"status": "ok", "state": "ready", "ready": True, "pid": 900,
        "worker_alive": True, "worker_idle": True, "queue_depth": 0, "in_flight_requests": 0,
        "active_job_age_s": None, "completed_sequence": 10, "uptime_s": 2.0}
    context = SimpleNamespace(
        base_url="http://127.0.0.1:8797", ctrl_url="http://127.0.0.1:8798", child=SimpleNamespace(pid=900),
        instance_token=None, deadline=time.monotonic() + 20, check_breach=lambda _: None,
        report={"status": "in_progress", "errors": []},
    )
    context.add_error = context.report["errors"].append
    captured = {"health_calls": 0}
    options = {"child_fails": False, "subordinate_error": None, "stage_cleanup": {}, "mutate_report": lambda _: None,
        "final_health": {}, "initial_health": {}}

    def get_json(url, **_):
        route = "/" + url.rsplit("/", 1)[-1]
        if route == "/health":
            captured["health_calls"] += 1
            overrides = options["initial_health"] if captured["health_calls"] == 1 else options["final_health"]
            return 200, {**health, "instance_token": context.instance_token, **overrides}
        return responses[route]

    context.client = SimpleNamespace(get_json=get_json)

    @contextmanager
    def managed_stage():
        captured["stage_entries"] = captured.get("stage_entries", 0) + 1
        try:
            yield context
            context.report["status"] = "passed" if not context.report["errors"] else "failed"
        finally:
            context.report["child_cleanup"] = {"cleaned": True, "pid": 900, "exit_code": 0}
            context.report.update(options["stage_cleanup"])

    def stage(config, **_):
        context.instance_token = config.instance_token
        return SimpleNamespace(managed_stage=managed_stage)

    monkeypatch.setattr(phase5, "SmokeStageSupervisor", stage)

    def subordinate(**kwargs):
        captured.update(kwargs)
        command, env = kwargs["cmd"], kwargs["env"]
        assert os.path.isabs(command[0])
        assert command[command.index("--mlx-model-path") + 1] == "/synthetic-model-not-loaded"
        assert 0 < int(command[command.index("--worker-timeout-ms") + 1]) <= 20_000
        assert "NODE_OPTIONS" not in env
        assert env["QMD_MLX_EMBED_URL"] == context.base_url
        assert env["HOME"].startswith(str(tmp_path))
        assert os.path.isdir(env["TMPDIR"])
        captured["scratch_root"] = os.path.dirname(env["HOME"])
        data = {"status": "failed" if options["child_fails"] else "passed", "errors": [],
            "checks": {key: True for key in ["shadowTargetSafety", "durableIndexingCancellation",
                "durableIndexingResumption", "vectorDeduplicationReconciled", "fingerprintMismatchRefused",
                "sqliteOnlineBackupRestored", "restoredRetrievalIdentical", "mcpTransportProbePassed", "mlx4bRetrievalComplete"]},
            "recovery": recovery_receipt_fixture()}
        if options["child_fails"]:
            data["errors"] = ["synthetic child qualification failure"]
        options["mutate_report"](data)
        output = command[command.index("--output-json") + 1]
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(data, stream)
        return SubordinateProcessResult(exit_code=1 if options["child_fails"] else 0,
            captured_logs="synthetic child fixture", duration_s=0.1, pid=100, error=options["subordinate_error"])

    monkeypatch.setattr(phase5, "run_subordinate_process", subordinate)
    supervisor = Phase5Supervisor(Phase5SupervisorConfig(
        mlx_model="/synthetic-model-not-loaded", use_fake_child=True,
        output_json=str(tmp_path / "report.json")), sampler=sampler)

    def run(**kwargs):
        options.update(kwargs)
        report = supervisor.run()
        if "scratch_root" in captured:
            assert not os.path.exists(captured["scratch_root"])
        with open(tmp_path / "report.json", encoding="utf-8") as stream:
            assert json.load(stream) == report
        return report

    return SimpleNamespace(run=run, responses=responses, captured=captured, sampler=sampler,
                           context=context, supervisor=supervisor)


@pytest.mark.parametrize("child_fails", [False, True])
def test_phase5_report_consolidates_child_receipts_and_cleans_owned_scratch(phase5_composition, child_fails):
    report = phase5_composition.run(child_fails=child_fails)
    assert report["status"] == ("failed" if child_fails else "passed")
    assert report["checks"]["fresh_process_recovery"] is True
    assert not any("Fresh-process crash recovery" in gate for gate in report["unfulfilled_release_gates"])
    if child_fails:
        assert "synthetic child qualification failure" in report["errors"]
    else:
        assert report["errors"] == []
        assert report["memory_metrics"] == phase5_composition.responses["/memory"][1]
        assert report["stats_metrics"] == phase5_composition.responses["/stats"][1]
        assert phase5_composition.captured["health_calls"] == 2


@pytest.mark.parametrize("field", ["installed_ram_mb", "memory_headroom_mb", "swap_used_mb", "memory_free_pct"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None, True, "16000", -1.0])
def test_phase5_preflight_refuses_invalid_samples_before_stage_or_runner(phase5_composition, field, value):
    getattr(phase5_composition.sampler, "get_" + field).return_value = value
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any("Preflight" in error for error in report["errors"])
    assert "cmd" not in phase5_composition.captured
    assert phase5_composition.captured["health_calls"] == 0


@pytest.mark.parametrize("field,value", [("installed_ram_mb", 0.0), ("memory_free_pct", 100.1), ("memory_headroom_mb", 32768.1)])
def test_phase5_preflight_refuses_impossible_memory_ranges(phase5_composition, field, value):
    getattr(phase5_composition.sampler, "get_" + field).return_value = value
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert report["errors"]
    assert "cmd" not in phase5_composition.captured


@pytest.mark.parametrize("error", ["Subordinate cleanup failed: owned group did not terminate", "log cleanup failed", ""])
def test_phase5_subordinate_error_fails_even_with_exit_zero_and_green_report(phase5_composition, error):
    report = phase5_composition.run(subordinate_error=error)
    assert report["status"] == "failed"
    assert any("TypeScript" in message for message in report["errors"])
    assert error in "\n".join(report["errors"])


@pytest.mark.parametrize("route", ["/memory", "/stats"])
@pytest.mark.parametrize("response", [(503, {"error": "unavailable"}), (200, None), (200, []), (200, {}), (200, "malformed")])
def test_phase5_final_telemetry_is_required(phase5_composition, route, response):
    phase5_composition.responses[route] = response
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any(route in error for error in report["errors"])


@pytest.mark.parametrize("route,field", [
    ("/memory", "active_mb"), ("/memory", "peak_mb"), ("/memory", "model_mb"),
    ("/stats", "total_requests"), ("/stats", "avg_ms"), ("/stats", "compiled_shapes"), ("/stats", "uptime_sec"),
])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), True, "100", -1.0])
def test_phase5_final_telemetry_refuses_malformed_required_numbers(phase5_composition, route, field, value):
    phase5_composition.responses[route][1][field] = value
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any(field in error for error in report["errors"])


@pytest.mark.parametrize("route,field", [
    ("/memory", "active_mb"), ("/memory", "peak_mb"), ("/memory", "model_mb"),
    ("/stats", "total_requests"), ("/stats", "avg_ms"), ("/stats", "compiled_shapes"), ("/stats", "uptime_sec"),
])
def test_phase5_final_telemetry_refuses_missing_required_numbers(phase5_composition, route, field):
    del phase5_composition.responses[route][1][field]
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any(field in error for error in report["errors"])


@pytest.mark.parametrize("route,overrides", [
    ("/memory", {"peak_mb": 99.9}), ("/stats", {"total_requests": 0}),
    ("/stats", {"total_requests": 1.5}), ("/stats", {"compiled_shapes": 1.5}),
])
def test_phase5_final_telemetry_refuses_inconsistent_or_inactive_state(phase5_composition, route, overrides):
    phase5_composition.responses[route][1].update(overrides)
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any(route in error for error in report["errors"])


@pytest.mark.parametrize("overrides", [
    {"pid": 901}, {"pid": True}, {"instance_token": "wrong-instance"}, {"instance_token": None},
    {"state": "stopping"}, {"ready": False}, {"ready": 1}, {"worker_alive": False},
    {"worker_idle": False}, {"queue_depth": 1}, {"queue_depth": False}, {"in_flight_requests": 1},
    {"completed_sequence": float("nan")}, {"active_job_age_s": 1.0}, {"status": "degraded"},
])
def test_phase5_requires_same_instance_and_idle_final_health(phase5_composition, overrides):
    report = phase5_composition.run(final_health=overrides)
    assert report["status"] == "failed"
    assert any("health" in error.lower() for error in report["errors"])


@pytest.mark.parametrize("overrides", [{"pid": 901}, {"instance_token": "wrong-instance"}, {"state": "failed"}])
def test_phase5_initial_health_refusal_precedes_runner(phase5_composition, overrides):
    report = phase5_composition.run(initial_health=overrides)
    assert report["status"] == "failed"
    assert "cmd" not in phase5_composition.captured


@pytest.mark.parametrize("cleanup", [
    {"cleanup_errors": ["owned group cleanup failed"]},
    {"status": "failed"}, {"status": "in_progress"},
    {"child_cleanup": {"cleaned": False, "pid": 900}},
    {"child_cleanup": {"cleaned": True, "pid": 901}},
    {"child_cleanup": {}},
])
def test_phase5_requires_successful_supervisor_final_state_after_context_exit(phase5_composition, cleanup):
    report = phase5_composition.run(stage_cleanup=cleanup)
    assert report["status"] == "failed"
    assert any("supervis" in error.lower() for error in report["errors"])


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("errors"), lambda r: r.update(errors=None), lambda r: r.update(errors=""),
    lambda r: r.update(errors=[None]), lambda r: r.update(checks=[]),
    lambda r: r["checks"].update(shadowTargetSafety=1), lambda r: r.update(status="failed"),
])
def test_phase5_refuses_malformed_or_failed_ts_report_with_actionable_error(phase5_composition, mutation):
    report = phase5_composition.run(mutate_report=mutation)
    assert report["status"] == "failed"
    assert report["errors"]
