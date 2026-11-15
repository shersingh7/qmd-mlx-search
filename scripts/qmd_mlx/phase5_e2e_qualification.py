#!/usr/bin/env python3
"""
phase5_e2e_qualification.py — Supervised End-to-End Public Retrieval & Recovery Qualification

Orchestrates sequential, isolated end-to-end qualification composed with SmokeStageSupervisor:
1. Production Indexing & Durable Recovery (cancellation, resumption, deduplication, fingerprint safety, SQLite online backup/restore).
2. Supervised MLX-only stage invocation with TS runner (GGUF stage explicitly BLOCKED pending supervised integration).
3. Integrated MCP Streamable HTTP transport retrieval probe.
4. Active MLXWatchdog supervision with continuous memory, swap, and headroom tracking.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir)) if os.path.basename(current_dir) == "qmd_mlx" else current_dir
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.qmd_mlx.supervisor import (
    SmokeHttpClient,
    SmokeStageSupervisor,
    StageSupervisorConfig,
    SubordinateProcessResult,
    run_subordinate_process,
)
from scripts.qmd_mlx.watchdog import SystemMemorySampler, SystemMetricsError


def node_typescript_command(script: str, *, deadline: float, env: dict[str, str]) -> list[str]:
    """Use the supported HTTP runtime and local public loader, never the tsx CLI/Bun."""
    required = "Phase 5 MCP HTTP requires local Node.js >=22 and the local tsx loader; no Bun, installation or remote fallback is allowed"
    node = shutil.which("node", path=env.get("PATH", os.defpath))
    if not node:
        raise RuntimeError(required)
    node = os.path.abspath(node)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError(f"{required}: execution deadline expired before runtime validation")
    package_root = Path(repo_root, "node_modules", "tsx")
    if not package_root.is_dir():
        raise RuntimeError(f"{required}: repository-local tsx development dependency is missing")
    # Resolve the package's public export from this repository, not the caller's
    # cwd. Importing this absolute loader avoids tsx CLI IPC sockets under TMPDIR.
    probe = """
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire(pathToFileURL(process.cwd() + '/package.json'));
if (process.versions.bun || Number(process.versions.node?.split('.')[0]) < 22) {
  throw new Error('Node.js >=22 is required');
}
process.stdout.write(JSON.stringify({node: process.versions.node, bun: process.versions.bun ?? null,
  loader: require.resolve('tsx')}));
"""
    try:
        result = subprocess.run([node, "--input-type=module", "-e", probe], cwd=repo_root, env=env,
                                capture_output=True, text=True, timeout=min(2.0, remaining), check=True)
        data = json.loads(result.stdout)
        version = data.get("node") if isinstance(data, dict) else None
        if (not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version)
                or int(version.split(".")[0]) < 22 or data.get("bun") is not None):
            raise ValueError("unsupported or unverified Node runtime")
        loader = Path(data["loader"])
        if (not loader.is_absolute() or not loader.is_file()
                or not loader.resolve().is_relative_to(package_root.resolve())):
            raise ValueError("local public tsx loader is missing or not an absolute file")
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"{required}: {error}") from error
    return [node, "--import", loader.as_uri(), os.path.abspath(script)]


@dataclasses.dataclass
class Phase5SupervisorConfig:
    mlx_model: str = os.path.expanduser("~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine")
    gguf_model: str = os.path.expanduser("~/.cache/qmd/models/hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf")
    host: str = "127.0.0.1"
    mlx_port: int = 8797
    mlx_control_port: int = 8798
    timeout_s: float = 120.0
    min_headroom_mb: float = 6000.0
    output_json: str = "docs/reviews/artifacts/phase5-isolated-e2e.json"
    dry_run: bool = False
    use_fake_child: bool = False
    skip_mlx: bool = False
    skip_gguf: bool = True


def _telemetry_number(value: Any, field: str, *, minimum: float = 0.0,
                      maximum: Optional[float] = None, integer: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)
            or (integer and not isinstance(value, int))):
        kind = "integer" if integer else "number"
        raise ValueError(f"{field} must be a finite {kind} >= {minimum}"
                         + (f" and <= {maximum}" if maximum is not None else ""))


def fresh_process_recovery_verified(recovery: Any) -> bool:
    """Require receipts for an interrupted child and a distinct completed child."""
    if not isinstance(recovery, dict) or recovery.get("mode") != "fresh_process_recovery":
        return False
    workers = recovery.get("workers")
    if not isinstance(workers, list) or len(workers) != 2 or not all(isinstance(w, dict) for w in workers):
        return False
    first, resumed = workers
    if any(not {"pid", "exitCode", "signal", "timedOut"}.issubset(worker) for worker in workers):
        return False
    pids = [worker.get("pid") for worker in workers]
    return (
        all(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 for pid in pids)
        and pids[0] != pids[1]
        and first.get("exitCode") is None and first.get("signal") == "SIGKILL"
        and first.get("timedOut") is False
        and type(resumed.get("exitCode")) is int and resumed["exitCode"] == 0
        and resumed.get("signal") is None and resumed.get("timedOut") is False
        and recovery.get("freshProcessRecovery") == "passed"
        and recovery.get("consecutiveSequenceVerified") is True
        and recovery.get("logicalStateInvariantOnMismatch") is True
    )


class Phase5Supervisor:
    """
    Supervises Phase 5 End-to-End qualification by composing the existing
    SmokeStageSupervisor with active MLXWatchdog monitoring, launch token authorization,
    bounded execution deadlines, and guaranteed fail-closed teardown.
    """

    def __init__(
        self,
        config: Phase5SupervisorConfig,
        sampler: Optional[SystemMemorySampler] = None,
    ):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self.last_report: dict[str, Any] = {}

    def _check_preflight_headroom(self) -> dict[str, Any]:
        """Validates configuration sanity, system telemetry, and headroom >= 6000MB, failing closed on any error."""
        # Validate numeric configuration parameters
        for f_name, val, min_v in [
            ("timeout_s", self.config.timeout_s, 0.001),
            ("min_headroom_mb", self.config.min_headroom_mb, 0.0),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Config field '{f_name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Config field '{f_name}' must not be NaN or Inf, got {val}")
            if val < min_v:
                raise ValueError(f"Config field '{f_name}' must be >= {min_v}, got {val}")

        for p_name, p_val in [("mlx_port", self.config.mlx_port), ("mlx_control_port", self.config.mlx_control_port)]:
            if isinstance(p_val, bool) or not isinstance(p_val, int):
                raise ValueError(f"Config field '{p_name}' must be an integer, got {type(p_val).__name__}")
            if not (1 <= p_val <= 65535):
                raise ValueError(f"Invalid {p_name}: {p_val} (must be between 1 and 65535)")

        if self.config.mlx_port == self.config.mlx_control_port:
            raise ValueError(f"Inference port ({self.config.mlx_port}) and control port ({self.config.mlx_control_port}) must be distinct")

        try:
            installed_ram = self.sampler.get_installed_ram_mb()
            headroom = self.sampler.get_memory_headroom_mb()
            swap_used = self.sampler.get_swap_used_mb()
            free_pct = self.sampler.get_memory_free_pct()
            _telemetry_number(installed_ram, "installed_ram_mb", minimum=0.001)
            _telemetry_number(headroom, "headroom_mb", maximum=installed_ram)
            _telemetry_number(swap_used, "swap_used_mb")
            _telemetry_number(free_pct, "memory_free_pct", maximum=100.0)
        except (SystemMetricsError, ValueError) as e:
            raise SystemMetricsError(f"Preflight memory sampling failed: {e}")

        print(
            f"[Phase 5 Supervisor] Preflight: Installed RAM={installed_ram:.1f}MB, "
            f"Headroom={headroom:.1f}MB, Swap={swap_used:.1f}MB, Free={free_pct:.1f}%"
        )

        if headroom < self.config.min_headroom_mb:
            raise RuntimeError(
                f"Insufficient memory headroom for isolated qualification: "
                f"{headroom:.1f} MB available, required >= {self.config.min_headroom_mb:.1f} MB."
            )

        return {
            "installed_ram_mb": installed_ram,
            "headroom_mb": round(headroom, 1),
            "swap_used_mb": round(swap_used, 1),
            "memory_free_pct": round(free_pct, 1),
        }

    def _save_report(self, report: dict[str, Any]):
        out_file = os.path.abspath(self.config.output_json)
        try:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            report["status"] = "failed"
            report.setdefault("errors", []).append(f"Failed to write report to {out_file}: {e}")

    def _required_json(self, ctx: Any, route: str) -> dict[str, Any]:
        status, data = ctx.client.get_json(f"{ctx.ctrl_url}{route}", deadline=ctx.deadline, timeout_s=2.0)
        if type(status) is not int or status != 200 or not isinstance(data, dict):
            raise RuntimeError(f"Required telemetry {route} failed: HTTP {status}; expected a JSON object")
        if data.get("error") is not None:
            raise RuntimeError(f"Required telemetry {route} failed: HTTP {status}; payload contains an error")
        return data

    def _check_health(self, ctx: Any, *, final: bool) -> dict[str, Any]:
        health = self._required_json(ctx, "/health")
        label = "Final /health" if final else "Initial /health"
        pid = health.get("pid")
        if (type(pid) is not int or pid <= 0 or pid != ctx.child.pid
                or not ctx.instance_token or health.get("instance_token") != ctx.instance_token):
            raise RuntimeError(f"{label} owned server identity mismatch")
        if (health.get("status") != "ok" or health.get("state") != "ready"
                or health.get("ready") is not True or health.get("worker_alive") is not True
                or health.get("error") is not None):
            raise RuntimeError(f"{label} server must be ready with a live inference worker and no error")
        for field in ("queue_depth", "in_flight_requests", "completed_sequence"):
            _telemetry_number(health.get(field), f"{label}.{field}", integer=True)
        _telemetry_number(health.get("uptime_s"), f"{label}.uptime_s")
        if final and (health.get("worker_idle") is not True or health["queue_depth"] != 0
                      or health["in_flight_requests"] != 0 or health["completed_sequence"] < 1
                      or "active_job_age_s" not in health or health["active_job_age_s"] is not None
                      or health.get("active_job_description") is not None):
            raise RuntimeError(f"{label} must show completed work and an idle, drained inference worker")
        return health

    def _final_telemetry(self, ctx: Any) -> dict[str, Any]:
        memory = self._required_json(ctx, "/memory")
        for field in ("active_mb", "peak_mb", "model_mb"):
            _telemetry_number(memory.get(field), f"/memory.{field}")
        if memory["peak_mb"] < memory["active_mb"]:
            raise ValueError("/memory.peak_mb must be >= active_mb")
        stats = self._required_json(ctx, "/stats")
        _telemetry_number(stats.get("total_requests"), "/stats.total_requests", minimum=1, integer=True)
        _telemetry_number(stats.get("compiled_shapes"), "/stats.compiled_shapes", integer=True)
        for field in ("avg_ms", "uptime_sec"):
            _telemetry_number(stats.get(field), f"/stats.{field}")
        # /memory and /stats do not carry identity. Bracket this run with
        # control /health receipts from the exact supervisor-owned instance.
        health = self._check_health(ctx, final=True)
        return {"memory_metrics": memory, "stats_metrics": stats, "final_health": health}

    def run(self) -> dict[str, Any]:
        overall_t0 = time.time()
        start_mono = time.monotonic()
        launch_token = f"phase5-token-{uuid.uuid4().hex[:16]}"
        errors: list[str] = []

        # 1. Preflight Telemetry Check (min 6000MB, persisting failure report fail-closed)
        try:
            preflight_info = self._check_preflight_headroom()
        except Exception as preflight_err:
            failed_report = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "duration_s": round(time.monotonic() - start_mono, 3),
                "status": "failed",
                "errors": [f"Preflight configuration/telemetry failed: {preflight_err}"],
                "checks": {},
            }
            self._save_report(failed_report)
            self.last_report = failed_report
            return failed_report

        # 2. Explicit Model Path Check (no remote fallback or autodownload)
        expanded_mlx_path = os.path.expanduser(self.config.mlx_model)
        if not self.config.dry_run and not self.config.use_fake_child and not self.config.skip_mlx:
            if not os.path.exists(expanded_mlx_path):
                blocked_report = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "duration_s": round(time.monotonic() - start_mono, 3),
                    "status": "failed",
                    "preflight": preflight_info,
                    "errors": [
                        f"Explicit MLX model path missing: {expanded_mlx_path}. Network autodownloads are strictly prohibited."
                    ],
                    "checks": {
                        "model_weights_available": False,
                    },
                    "unfulfilled_release_gates": [
                        "MLX model weights not present in local cache directory; qualification blocked offline."
                    ],
                }
                self._save_report(blocked_report)
                self.last_report = blocked_report
                return blocked_report

        server_script = os.path.join(repo_root, "scripts", "mlx_embed_server.py")
        server_cmd = [
            sys.executable,
            server_script,
            "--model",
            expanded_mlx_path,
            "--port",
            str(self.config.mlx_port),
            "--control-port",
            str(self.config.mlx_control_port),
            "--host",
            self.config.host,
            "--preload",
        ]

        supervisor_config = StageSupervisorConfig(
            cmd=server_cmd,
            host=self.config.host,
            port=self.config.mlx_port,
            control_port=self.config.mlx_control_port,
            timeout_s=self.config.timeout_s,
            min_headroom_mb=self.config.min_headroom_mb,
            stage_name="phase5_qualification",
            model_identifier=self.config.mlx_model,
            instance_token=launch_token,
            output_file=self.config.output_json,
            dry_run=self.config.dry_run,
        )

        supervisor = SmokeStageSupervisor(config=supervisor_config, sampler=self.sampler)

        if self.config.dry_run:
            with supervisor.managed_stage() as ctx:
                ctx.record_fixture("dry_run", {"status": "skipped", "reason": "dry-run requested"})
                report = ctx.report
                report["status"] = "dry_run_completed"
                self.last_report = report
                return report

        report: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "in_progress",
            "preflight": preflight_info,
            "checks": {},
            "errors": [],
            "retrieval_comparison": {
                "gguf_06b": {
                    "status": "blocked",
                    "reason": "GGUF stage explicitly BLOCKED pending supervised integration instead of unmanaged subprocess.run",
                },
                "mlx_4b": {
                    "status": "not_run" if self.config.skip_mlx else "in_progress",
                },
            },
            "unfulfilled_release_gates": [
                "Representative large-model interactive latency under bulk load is not established by this small public-fixture smoke.",
                "1000-batch sustained load memory test unfulfilled for live promotion.",
                "Full-corpus evaluation (MS MARCO/BEIR) pending; small public fixture results labeled as Preliminary Release Smoke.",
                "GGUF stage explicitly BLOCKED pending supervised integration.",
                "Fresh-process crash recovery across OS boundaries marked pending; only in-process checkpoint recovery verified.",
            ],
        }

        ctx = None
        ts_temp_dir = None
        try:
            # Validate the host runtime before managed_stage can spawn/preload any model.
            # The resolver itself has a bounded (at most two-second) process probe.
            ts_runner_script = os.path.join(repo_root, "scripts", "phase5_e2e_runner.ts")
            ts_temp_dir = tempfile.TemporaryDirectory(prefix="qmd-phase5-ts-", dir=os.environ.get("TMPDIR"))
            tmp_ts_report = os.path.join(ts_temp_dir.name, "report.json")
            for name in ("home", "tmp", "cache", "config", "hermes"):
                os.makedirs(os.path.join(ts_temp_dir.name, name))
            ts_env = {key: value for key, value in os.environ.items() if key in {
                "PATH", "LANG", "LC_ALL", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH",
            }}
            ts_env.update({
                "HOME": os.path.join(ts_temp_dir.name, "home"),
                "TMPDIR": os.path.join(ts_temp_dir.name, "tmp"),
                "TMP": os.path.join(ts_temp_dir.name, "tmp"),
                "TEMP": os.path.join(ts_temp_dir.name, "tmp"),
                "XDG_CACHE_HOME": os.path.join(ts_temp_dir.name, "cache"),
                "XDG_CONFIG_HOME": os.path.join(ts_temp_dir.name, "config"),
                "HERMES_HOME": os.path.join(ts_temp_dir.name, "hermes"),
                "CI": "true",
            })
            ts_env["QMD_PHASE5_LAUNCH_TOKEN"] = launch_token
            ts_env["QMD_EMBED_BACKEND"] = "mlx"
            ts_env["QMD_MLX_FALLBACK"] = "0"
            ts_env["HF_HUB_OFFLINE"] = "1"
            ts_env["TRANSFORMERS_OFFLINE"] = "1"
            ts_env["HF_DATASETS_OFFLINE"] = "1"

            ts_cmd = node_typescript_command(ts_runner_script, deadline=time.monotonic() + self.config.timeout_s, env=ts_env)
            with supervisor.managed_stage() as ctx:
                # 3. Check supervisor health before invoking TS child
                ctx.check_breach("server_ready_preflight")
                report["initial_health"] = self._check_health(ctx, final=False)

                # Verify descriptor
                desc_status, desc = ctx.client.get_json(
                    f"{ctx.base_url}/descriptor",
                    deadline=ctx.deadline,
                    timeout_s=2.0,
                )
                if desc_status != 200 or not isinstance(desc, dict):
                    raise RuntimeError(f"Descriptor check on {ctx.ctrl_url}/descriptor failed: HTTP {desc_status} {desc}")

                ts_env["QMD_MLX_EMBED_URL"] = ctx.base_url
                # Leave room for worker-group teardown after runtime validation.
                worker_budget_s = min(self.config.timeout_s, ctx.deadline - time.monotonic()) - 1.0
                if worker_budget_s <= 0:
                    raise RuntimeError("Insufficient remaining execution budget for TypeScript recovery and bounded teardown")
                ts_cmd.extend([
                    "--launch-token", launch_token,
                    "--skip-gguf",  # GGUF stage explicitly BLOCKED pending supervised integration
                    "--mlx-port", str(self.config.mlx_port),
                    "--mlx-model-path", expanded_mlx_path,
                    "--worker-timeout-ms", str(max(1, int(worker_budget_s * 1000))),
                    "--output-json", tmp_ts_report,
                ])

                ts_res = run_subordinate_process(
                    cmd=ts_cmd,
                    deadline=ctx.deadline,
                    check_breach=ctx.check_breach,
                    cwd=repo_root,
                    env=ts_env,
                    capture_log_prefix="phase5_ts_",
                )

                ctx.check_breach("ts_runner_completed")

                # Cleanup/log errors are failures even when the process exited
                # zero and its own qualification report says passed.
                if ts_res.error is not None:
                    err_msg = f"TypeScript Phase 5 runner error: {ts_res.error or 'unspecified subordinate error'}"
                    errors.append(err_msg)
                    ctx.add_error(err_msg)
                if ts_res.timed_out:
                    err_msg = f"TypeScript Phase 5 runner timed out after {ts_res.duration_s}s (PID {ts_res.pid})"
                    errors.append(err_msg)
                    ctx.add_error(err_msg)
                elif ts_res.breached:
                    err_msg = f"Supervisor watchdog breach during TypeScript runner execution: {ts_res.error}"
                    errors.append(err_msg)
                    ctx.add_error(err_msg)
                elif type(ts_res.exit_code) is not int or ts_res.exit_code != 0:
                    err_msg = f"TypeScript Phase 5 runner failed with exit code {ts_res.exit_code}:\n{ts_res.captured_logs}"
                    errors.append(err_msg)
                    ctx.add_error(err_msg)

                ts_report_data: dict[str, Any] = {}
                if not os.path.exists(tmp_ts_report):
                    raise RuntimeError("TypeScript Phase 5 runner produced no report")
                else:
                    try:
                        with open(tmp_ts_report, "r", encoding="utf-8") as f:
                            ts_report_data = json.load(f)
                    finally:
                        try:
                            os.unlink(tmp_ts_report)
                        except Exception:
                            pass

                # Consolidate TS checks and recovery results
                if not isinstance(ts_report_data, dict) or not isinstance(ts_report_data.get("checks"), dict):
                    raise ValueError("TypeScript Phase 5 report/checks must be JSON objects")
                ts_errors = ts_report_data.get("errors")
                if not isinstance(ts_errors, list) or not all(isinstance(error, str) and error for error in ts_errors):
                    raise ValueError("TypeScript Phase 5 report.errors must be an array of nonempty strings")
                ts_checks = ts_report_data["checks"]
                report["checks"] = {
                    "shadow_target_safety": ts_checks.get("shadowTargetSafety", False),
                    "durable_indexing_cancellation": ts_checks.get("durableIndexingCancellation", False),
                    "durable_indexing_resumption": ts_checks.get("durableIndexingResumption", False),
                    "vector_deduplication_reconciled": ts_checks.get("vectorDeduplicationReconciled", False),
                    "fingerprint_mismatch_refused": ts_checks.get("fingerprintMismatchRefused", False),
                    "sqlite_online_backup_restored": ts_checks.get("sqliteOnlineBackupRestored", False),
                    "restored_retrieval_identical": ts_checks.get("restoredRetrievalIdentical", False),
                    "mcp_transport_probe_passed": ts_checks.get("mcpTransportProbePassed", False),
                    "mlx4b_retrieval_complete": ts_checks.get("mlx4bRetrievalComplete", False),
                    "gguf06b_stage_blocked": True,
                    "fresh_process_recovery": fresh_process_recovery_verified(ts_report_data.get("recovery")),
                }
                report["corpus"] = ts_report_data.get("corpus", {})
                report["recovery"] = ts_report_data.get("recovery", {})
                if report["checks"]["fresh_process_recovery"]:
                    report["unfulfilled_release_gates"] = [gate for gate in report["unfulfilled_release_gates"]
                        if not gate.startswith("Fresh-process crash recovery")]
                errors.extend(ts_errors)
                failed_checks = [name for name, passed in report["checks"].items() if passed is not True]
                if failed_checks:
                    errors.append(f"TypeScript Phase 5 required checks failed: {', '.join(failed_checks)}")
                if ts_report_data.get("status") != "passed":
                    errors.append(f"TypeScript Phase 5 report status is not passed: {ts_report_data.get('status')}")
                report["retrieval_comparison"]["mlx_4b"] = ts_report_data.get("retrieval", {}).get("mlx4b", {
                    "status": "completed" if ts_checks.get("mlx4bRetrievalComplete") else "failed",
                })
                report["mcp_probe"] = ts_report_data.get("mcpProbe", {})

                # Final telemetry is an acceptance gate, not optional context.
                report.update(self._final_telemetry(ctx))
                ctx.check_breach("phase5_final_telemetry")

                all_passed = (
                    all(value is True for value in report["checks"].values())
                    and ts_report_data.get("status") == "passed"
                    and ts_res.error is None
                    and ts_res.exit_code == 0
                    and not ts_res.timed_out
                    and not ts_res.breached
                    and len(errors) == 0
                )
                report["status"] = "passed" if all_passed else "failed"

        except Exception as e:
            report["status"] = "failed"
            errors.append(str(e))
            report["errors"] = errors
        finally:
            if ts_temp_dir is not None:
                try:
                    ts_temp_dir.cleanup()
                except Exception as cleanup_error:
                    errors.append(f"TypeScript scratch cleanup failed: {cleanup_error}")
            report["duration_s"] = round(time.time() - overall_t0, 3)
            report["errors"] = errors
            if ctx is not None:
                report["supervision"] = ctx.report
                # managed_stage finalizes status and teardown AFTER the body.
                # A green TS report cannot override its failed/missing cleanup.
                for field in ("errors", "cleanup_errors"):
                    stage_errors = ctx.report.get(field, [])
                    if not isinstance(stage_errors, list):
                        errors.append(f"Stage supervision {field} is malformed")
                    else:
                        for error in stage_errors:
                            message = f"Stage supervision {field}: {error}"
                            if error not in errors:
                                errors.append(message)
                cleanup = ctx.report.get("child_cleanup")
                if (not isinstance(cleanup, dict) or cleanup.get("cleaned") is not True
                        or type(cleanup.get("pid")) is not int or cleanup["pid"] != ctx.child.pid):
                    errors.append("Stage supervision did not verify cleanup of the exact owned child")
                if ctx.report.get("status") != "passed":
                    errors.append(f"Stage supervision final status is not passed: {ctx.report.get('status')}")
            if errors:
                report["status"] = "failed"
            self._save_report(report)
            self.last_report = report

        return report


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Isolated End-to-End Qualification Supervisor")
    parser.add_argument("--output-json", default="docs/reviews/artifacts/phase5-isolated-e2e.json")
    parser.add_argument("--mlx-port", type=int, default=8797)
    parser.add_argument("--mlx-control-port", type=int, default=8798)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--min-headroom-mb", type=float, default=6000.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = Phase5SupervisorConfig(
        mlx_port=args.mlx_port,
        mlx_control_port=args.mlx_control_port,
        timeout_s=args.timeout_s,
        min_headroom_mb=args.min_headroom_mb,
        output_json=args.output_json,
        dry_run=args.dry_run,
    )

    supervisor = Phase5Supervisor(config)
    report = supervisor.run()
    sys.exit(0 if report.get("status") == "passed" else 1)


if __name__ == "__main__":
    main()
