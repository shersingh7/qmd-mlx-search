"""
supervisor.py — Reusable Owned-Process Supervisor for Bounded MLX Stage Qualification

Guarantees & Invariants:
1. Active MLXWatchdog Lifecycle: Server child process is continuously supervised by
   an instantiated MLXWatchdog in a dedicated background thread.
2. TokenPID & Endpoint Binding: Enforces unique MLX_INSTANCE_TOKEN, strict numeric loopback (127.0.0.1),
   and matches PID and instance token on control /health probes.
3. Continuous Measured Telemetry: Continuously samples RSS, swap usage, swap growth, and memory
   pressure throughout execution (never fabricated data).
4. Wall-Clock Deadline Bounding: Enforces monotonic startup/probe deadlines and separate bounded teardown.
5. Structured Breach Handling & Process Isolation: Watchdog breaches or deadline overruns terminate ONLY
   the owned isolated process group. Sentinel / parent processes survive intact.
6. Guaranteed Teardown & Saved JSON: In finally blocks, captures bounded logs, stops supervisor thread,
   reaps child, unlinks log file, and returns a complete, structured JSON report even on exceptions or breaches.
7. Isolated HTTP Transport: Probes use SmokeHttpClient (trust_env=False, allow_redirects=False,
   monotonic deadline enforcement, bounded streaming response bodies).
"""

from __future__ import annotations

import contextlib
import dataclasses
import http.client
import json
import math
import os
import selectors
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Generator, Optional
from urllib.parse import urlsplit

import requests
from requests.structures import CaseInsensitiveDict

from .watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    WatchdogCheckResult,
    is_numeric_loopback,
    spawn_owned_process,
    terminate_owned_process,
)


def _read_log_tail(path: str, max_bytes: int) -> str:
    """Read bytes, not the whole file (nor a character-count approximation)."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("Log tail byte limit must be a non-negative integer")
    if max_bytes == 0:
        return ""
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - max_bytes))
        tail = stream.read(max_bytes)
    # Replacement characters can expand invalid UTF-8; keep the returned UTF-8
    # representation within the same byte cap, without cutting a character.
    encoded = tail.decode("utf-8", errors="replace").encode("utf-8")
    return encoded[-max_bytes:].decode("utf-8", errors="ignore")


def _startup_failure_message(reason: str, log_path: Optional[str]) -> str:
    """Keep bounded child stderr visible when a readiness/breach check fails."""
    if log_path is None:
        return reason
    try:
        logs = _read_log_tail(log_path, 2000)
    except OSError as error:
        logs = f"Could not read child logs: {error}"
    return f"{reason}. Server logs:\n{logs}"


class _BoundedLogCapture:
    """Drain a pipe in fixed chunks; only this owner writes the capped disk tail.

    Handing children a regular log file cannot cap disk growth: truncation races
    their inherited file offsets. A bounded pipe supplies backpressure instead.
    """

    def __init__(self, prefix: str, max_bytes: int, barrier: Optional[str] = None):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_log_bytes must be a non-negative integer")
        if barrier is not None and not isinstance(barrier, str):
            raise ValueError("barrier_sentinel must be a string or None")
        self.barrier = barrier.encode("utf-8") if barrier is not None else None
        self.max_bytes = max_bytes
        self.file = tempfile.NamedTemporaryFile(mode="w+b", prefix=prefix, suffix=".log", delete=False)
        self.path = self.file.name
        self.barrier_reached = threading.Event()
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.pipe = None
        self.error: Optional[str] = None

    def start(self, child: subprocess.Popen):
        self.pipe = child.stdout
        if self.pipe is None:
            raise ValueError("Bounded log capture requires stdout=PIPE")
        pipe = self.pipe
        os.set_blocking(pipe.fileno(), False)

        def drain():
            tail = bytearray()
            barrier_tail = b""
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(pipe, selectors.EVENT_READ)
                    while not self.stop.is_set():
                        if not selector.select(timeout=0.05):
                            continue
                        chunk = os.read(pipe.fileno(), 8192)
                        if not chunk:
                            break
                        if self.barrier is not None and not self.barrier_reached.is_set():
                            candidate = barrier_tail + chunk
                            if self.barrier in candidate:
                                self.barrier_reached.set()
                            keep = max(0, len(self.barrier) - 1)
                            barrier_tail = candidate[-keep:] if keep else b""
                        if self.max_bytes:
                            tail.extend(chunk)
                            del tail[:-self.max_bytes]
                            self.file.seek(0)
                            self.file.write(tail)
                            self.file.truncate(len(tail))
                            self.file.flush()
            except Exception as e:
                self.error = str(e)

        self.thread = threading.Thread(target=drain, name="Smoke-LogCapture", daemon=True)
        self.thread.start()

    def finish(self):
        if self.thread is not None:
            # Normal EOF drains the final child output. A descendant holding the
            # pipe open cannot leave a permanent reader thread behind.
            self.thread.join(timeout=0.25)
            self.stop.set()
            self.thread.join(timeout=0.25)
            if self.thread.is_alive():
                raise RuntimeError("Bounded log reader failed to stop")
        if self.pipe is not None:
            self.pipe.close()
        self.file.flush()

    def close(self):
        try:
            self.finish()
        finally:
            self.file.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.path)


@dataclasses.dataclass
class SubordinateProcessResult:
    exit_code: Optional[int]
    captured_logs: str
    duration_s: float
    timed_out: bool = False
    breached: bool = False
    pid: Optional[int] = None
    error: Optional[str] = None
    sigkill_used: bool = False
    barrier_reached: bool = False


def run_subordinate_process(
    cmd: list[str],
    deadline: float,
    check_breach: Optional[Callable[[str], None]] = None,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    poll_interval_s: float = 0.05,
    capture_log_prefix: str = "subordinate_",
    max_log_bytes: int = 65536,
    barrier_sentinel: Optional[str] = None,
    kill_on_barrier: bool = False,
) -> SubordinateProcessResult:
    """Run a bounded child, cap its disk log, and clean its entire owned group."""
    t0 = time.monotonic()
    if not math.isfinite(deadline):
        raise ValueError("Subordinate deadline must be finite")
    if t0 >= deadline:
        raise TimeoutError("Absolute execution deadline exceeded before initiating subordinate process")
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be finite and positive")

    capture = _BoundedLogCapture(capture_log_prefix, max_log_bytes, barrier_sentinel)
    proc: Optional[subprocess.Popen] = None
    timed_out = breached = barrier_reached = sigkill_used = False
    err_msg: Optional[str] = None
    captured_logs = ""
    try:
        proc = spawn_owned_process(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        capture.start(proc)
        while True:
            if time.monotonic() >= deadline:
                timed_out = True
                err_msg = f"Subordinate process (PID {proc.pid}) exceeded monotonic deadline"
                break
            if check_breach is not None:
                try:
                    check_breach("subordinate_process_wait")
                except Exception as b_err:
                    breached = True
                    err_msg = str(b_err)
                    break
            if capture.error:
                raise RuntimeError(f"Subordinate log capture failed: {capture.error}")
            barrier_reached = capture.barrier_reached.is_set()
            if barrier_reached and kill_on_barrier:
                break
            if proc.poll() is not None:
                break
            time.sleep(min(poll_interval_s, max(0, deadline - time.monotonic())))
    except Exception as e:
        err_msg = str(e)
    finally:
        if proc is not None:
            try:
                cleaned, sigkill_used = terminate_owned_process(proc)
                if not cleaned:
                    raise RuntimeError("Owned process group did not terminate")
            except Exception as e:
                err_msg = f"{err_msg + '; ' if err_msg else ''}Subordinate cleanup failed: {e}"
        try:
            capture.finish()
            barrier_reached = capture.barrier_reached.is_set()
            captured_logs = _read_log_tail(capture.path, max_log_bytes)
            if capture.error:
                raise RuntimeError(capture.error)
        except Exception as e:
            err_msg = f"{err_msg + '; ' if err_msg else ''}Subordinate log capture failed: {e}"
        finally:
            try:
                capture.close()
            except Exception as e:
                err_msg = f"{err_msg + '; ' if err_msg else ''}Log cleanup failed: {e}"

    return SubordinateProcessResult(
        exit_code=proc.returncode if proc is not None else None,
        captured_logs=captured_logs,
        duration_s=round(time.monotonic() - t0, 3),
        timed_out=timed_out, breached=breached,
        pid=proc.pid if proc else None, error=err_msg,
        sigkill_used=sigkill_used, barrier_reached=barrier_reached,
    )



class _DeadlineSocket(socket.socket):
    """Re-arm *every* underlying read, including buffered HTTP header reads.

    requests/urllib3 inactivity timeouts and iter_content checks do not bound a
    tiny-drip response. SocketIO calls recv_into for each actual read, so a
    buffering loop cannot bypass this deadline. No timer/worker thread is needed.
    """

    def __init__(self, deadline: float):
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        self.deadline = deadline

    def _remaining_timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP request wall-clock deadline exceeded")
        self.settimeout(remaining)

    def connect(self, address):
        self._remaining_timeout()
        return super().connect(address)

    def sendall(self, data, flags=0):
        self._remaining_timeout()
        return super().sendall(data, flags)

    def recv_into(self, buffer, nbytes=0, flags=0):
        self._remaining_timeout()
        return super().recv_into(buffer, nbytes, flags)


class SmokeHttpClient:
    """
    Isolated, wall-clock deadline-bounded HTTP client for smoke probes.
    Enforces:
    - trust_env=False (no system HTTP_PROXY / HTTPS_PROXY interference).
    - allow_redirects=False (no redirection on loopback endpoints).
    - Monotonic wall-clock deadline bounding across connect, send, and receive.
    - Maximum response body bytes cap to prevent memory exhaustion.
    """

    def __init__(self, default_timeout_s: float = 10.0, max_response_bytes: int = 10 * 1024 * 1024):
        if isinstance(default_timeout_s, bool) or not isinstance(default_timeout_s, (int, float)) or not math.isfinite(default_timeout_s) or default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be finite and positive")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes < 0:
            raise ValueError("max_response_bytes must be a non-negative integer")
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.max_redirects = 0
        self.default_timeout_s = default_timeout_s
        self.max_response_bytes = max_response_bytes

    def request(
        self,
        method: str,
        url: str,
        json_data: Optional[dict[str, Any]] = None,
        deadline: Optional[float] = None,
        per_request_timeout_s: Optional[float] = None,
    ) -> requests.Response:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise TimeoutError("Absolute execution deadline exceeded before initiating HTTP request")

        timeout_s = self.default_timeout_s if per_request_timeout_s is None else per_request_timeout_s
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("per_request_timeout_s must be finite and positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("HTTP deadline must be finite")
        total_deadline = min(deadline, now + timeout_s) if deadline is not None else now + timeout_s
        target = urlsplit(url)
        host = target.hostname
        if target.scheme != "http" or host is None or not is_numeric_loopback(host) or target.username is not None or target.password is not None:
            raise ValueError("HTTP smoke probes require a numeric IPv4 loopback http:// URL")
        # inet_pton rejects nonliteral spellings accepted by getaddrinfo, avoiding
        # DNS work that a socket timeout cannot independently interrupt.
        socket.inet_pton(socket.AF_INET, host)
        port = target.port if target.port is not None else 80
        if not 1 <= port <= 65535:
            raise ValueError("HTTP probe port must be between 1 and 65535")
        conn = http.client.HTTPConnection(host, port)
        conn.auto_open = False
        sock = _DeadlineSocket(total_deadline)
        raw = None
        resp = None
        try:
            prepared = self.session.prepare_request(requests.Request(method, url, json=json_data))
            prepared.headers.update({"Connection": "close", "Accept-Encoding": "identity"})
            sock.connect((host, port))
            conn.sock = sock
            conn.request(prepared.method, prepared.path_url, body=prepared.body, headers=dict(prepared.headers))
            raw = conn.getresponse()
            resp = requests.Response()
            resp.status_code = raw.status
            resp.reason = raw.reason
            resp.url = url
            resp.request = prepared
            resp.headers = CaseInsensitiveDict(raw.getheaders())
            resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)
            resp.raw = raw
            # We explicitly request identity, not decompression with an unbounded
            # expansion ratio. A server that ignores it fails closed.
            if raw.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("HTTP probe received unsupported Content-Encoding")
            content = bytearray()
            while True:
                if time.monotonic() >= total_deadline:
                    raise TimeoutError("HTTP request wall-clock deadline exceeded")
                chunk = raw.read1(min(8192, self.max_response_bytes - len(content) + 1))
                if not chunk:
                    if raw.length not in (None, 0):
                        raise RuntimeError("HTTP probe response ended before its Content-Length")
                    break
                content.extend(chunk)
                if len(content) > self.max_response_bytes:
                    raise ValueError(f"HTTP response exceeded byte limit of {self.max_response_bytes} bytes")
            if time.monotonic() >= total_deadline:
                raise TimeoutError("HTTP request wall-clock deadline exceeded")
            resp._content = bytes(content)
            resp._content_consumed = True
            return resp
        except (socket.timeout, TimeoutError) as e:
            raise TimeoutError("HTTP request wall-clock deadline exceeded") from e
        except (requests.exceptions.RequestException, http.client.HTTPException, OSError) as e:
            raise RuntimeError(f"HTTP probe request failed: {e}") from e
        finally:
            # Also close partial headers, oversized bodies and parse failures.
            # Returned Responses retain their bounded, already-buffered content.
            try:
                if resp is not None:
                    resp.close()
            finally:
                try:
                    if raw is not None:
                        raw.close()
                finally:
                    conn.close()
                    sock.close()

    def get_json(
        self,
        url: str,
        deadline: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[int, Any]:
        resp = self.request("GET", url, deadline=deadline, per_request_timeout_s=timeout_s)
        try:
            return resp.status_code, resp.json()
        except Exception as e:
            return resp.status_code, {"raw": resp.text, "error": str(e)}

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        deadline: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[int, Any]:
        resp = self.request("POST", url, json_data=payload, deadline=deadline, per_request_timeout_s=timeout_s)
        try:
            return resp.status_code, resp.json()
        except Exception as e:
            return resp.status_code, {"raw": resp.text, "error": str(e)}

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


@dataclasses.dataclass
class StageSupervisorConfig:
    cmd: list[str]
    host: str = "127.0.0.1"
    port: int = 8797
    control_port: int = 8798
    timeout_s: float = 60.0
    min_headroom_mb: float = 2048.0
    max_rss_mb: Optional[float] = None
    max_swap_growth_mb: Optional[float] = None
    min_free_memory_pct: float = 12.0
    check_interval_s: float = 0.25
    startup_grace_period_s: float = 30.0
    stalled_inference_timeout_s: float = 10.0
    stage_name: str = "stage"
    model_identifier: Optional[str] = None
    instance_token: Optional[str] = None
    extra_env: Optional[dict[str, str]] = None
    output_file: Optional[str] = None
    dry_run: bool = False
    expected_cmd_pattern: str = r"(python|mlx|qmd|server)"
    max_log_bytes: int = 8192


@dataclasses.dataclass
class SmokeStageContext:
    supervisor: SmokeStageSupervisor
    client: SmokeHttpClient
    child: subprocess.Popen
    watchdog: MLXWatchdog
    instance_token: str
    base_url: str
    ctrl_url: str
    start_mono: float
    deadline: float
    report: dict[str, Any]

    def check_breach(self, operation_label: str = "operation"):
        """Verifies that no watchdog breach or unexpected child exit occurred before proceeding."""
        self.supervisor.check_breach(operation_label)

    def record_check(self, key: str, value: bool):
        self.report.setdefault("checks", {})[key] = value

    def record_metric(self, key: str, value: Any):
        self.report.setdefault("metrics", {})[key] = value

    def record_fixture(self, key: str, value: Any):
        self.report.setdefault("fixtures", {})[key] = value

    def add_error(self, err: str):
        self.report.setdefault("errors", []).append(err)


class SmokeStageSupervisor:
    """
    Supervises a spawned MLX server child process under active MLXWatchdog monitoring.
    Enforces strict lifecycle guarantees, non-blocking telemetry, bounded execution,
    isolated child termination on breaches, and complete saved JSON output.
    """

    def __init__(
        self,
        config: StageSupervisorConfig,
        sampler: Optional[SystemMemorySampler] = None,
    ):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self._preflight_defaults: Optional[Any] = None
        self._preflight_info: dict[str, Any] = {}
        self.breach_event = threading.Event()
        self.breach_result: list[Optional[WatchdogCheckResult]] = [None]
        self.telemetry_samples: list[dict[str, Any]] = []
        self.watchdog: Optional[MLXWatchdog] = None
        self.child: Optional[subprocess.Popen] = None
        self.last_report: dict[str, Any] = {}
        self._log_capture: Optional[_BoundedLogCapture] = None

    def validate_preflight(self) -> dict[str, Any]:
        """
        Validates system telemetry, headroom, loopback policy, and port availability.
        Fails closed on any error before process spawning.
        """
        # 0. Numeric sanity checks (fail closed on NaN, inf, negative or wrong types)
        for field_name, val, min_val in [
            ("timeout_s", self.config.timeout_s, 0.001),
            ("min_headroom_mb", self.config.min_headroom_mb, 0.0),
            ("check_interval_s", self.config.check_interval_s, 0.001),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Config field '{field_name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Config field '{field_name}' must not be NaN or Inf, got {val}")
            if val < min_val:
                raise ValueError(f"Config field '{field_name}' must be >= {min_val}, got {val}")

        for port_name, port_val in [("port", self.config.port), ("control_port", self.config.control_port)]:
            if isinstance(port_val, bool) or not isinstance(port_val, int):
                raise ValueError(f"Config field '{port_name}' must be an integer, got {type(port_val).__name__}")
            if not (1 <= port_val <= 65535):
                raise ValueError(f"Invalid {port_name}: {port_val} (must be between 1 and 65535)")

        if isinstance(self.config.max_log_bytes, bool) or not isinstance(self.config.max_log_bytes, int) or self.config.max_log_bytes < 0:
            raise ValueError("max_log_bytes must be a non-negative integer")

        if self.config.max_rss_mb is not None:
            v = self.config.max_rss_mb
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) or v <= 0:
                raise ValueError(f"Config field 'max_rss_mb' must be positive number, got {v}")

        if self.config.max_swap_growth_mb is not None:
            v = self.config.max_swap_growth_mb
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) or v < 0:
                raise ValueError(f"Config field 'max_swap_growth_mb' must be non-negative number, got {v}")

        preflight_info: dict[str, Any] = {
            "timestamp": time.time(),
            "host": self.config.host,
            "port": self.config.port,
            "control_port": self.config.control_port,
            "stage": self.config.stage_name,
            "model": self.config.model_identifier,
        }

        # 1. Host restriction (strict numeric IPv4 loopback)
        if not is_numeric_loopback(self.config.host) or self.config.host != "127.0.0.1":
            raise ValueError(f"Host '{self.config.host}' must be numeric loopback '127.0.0.1'")

        # 2. Port distinctness and validity
        if self.config.port == self.config.control_port:
            raise ValueError(
                f"Inference port ({self.config.port}) and control port ({self.config.control_port}) must be distinct"
            )

        # 3. Prevent collision with live production daemon (8787)
        if self.config.port == 8787 or self.config.control_port == 8787:
            raise ValueError(
                "Refusing to use port 8787; port 8787 is reserved for live daemon. Choose distinct qualification ports."
            )

        # 4. Port availability check
        for p_name, p_val in [("Inference port", self.config.port), ("Control port", self.config.control_port)]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.config.host, p_val))
                s.close()
            except OSError as e:
                raise RuntimeError(f"{p_name} {p_val} on {self.config.host} is already in use: {e}")

        # 5. Memory telemetry & headroom check
        try:
            installed_ram = self.sampler.get_installed_ram_mb()
            headroom = self.sampler.get_memory_headroom_mb()
            swap_used = self.sampler.get_swap_used_mb()
            free_pct = self.sampler.get_memory_free_pct()
            defaults = self.sampler.compute_conservative_defaults(headroom_mb=headroom)
            self._preflight_defaults = defaults
            defaults_dict = (
                dataclasses.asdict(defaults)
                if dataclasses.is_dataclass(defaults)
                else (defaults.__dict__ if hasattr(defaults, "__dict__") else {})
            )
            preflight_info.update({
                "installed_ram_mb": installed_ram,
                "headroom_mb": round(headroom, 1),
                "swap_used_mb": round(swap_used, 1),
                "free_memory_pct": round(free_pct, 1),
                "watchdog_defaults": defaults_dict,
            })
        except SystemMetricsError as e:
            raise SystemMetricsError(f"Preflight memory sampling failed: {e}")

        # 6. Conservative headroom enforcement
        if headroom < self.config.min_headroom_mb:
            raise RuntimeError(
                f"Insufficient memory headroom for {self.config.stage_name} stage: "
                f"measured {headroom:.1f} MB < required {self.config.min_headroom_mb:.1f} MB. "
                f"Refusing to execute."
            )

        self._preflight_info = preflight_info
        return preflight_info

    def check_breach(self, operation_label: str = "operation"):
        """Checks if a watchdog breach has fired or if child process died."""
        if self._log_capture is not None and self._log_capture.error:
            raise RuntimeError(f"Server log capture failed: {self._log_capture.error}")
        if self.breach_event.is_set():
            b_res = self.breach_result[0]
            reason = b_res.breach_reason if b_res else "unknown breach"
            raise RuntimeError(f"Watchdog breach during {operation_label}: {reason}")

        if self.watchdog and self.watchdog.last_check_result and not self.watchdog.last_check_result.healthy:
            raise RuntimeError(f"Watchdog breach during {operation_label}: {self.watchdog.last_check_result.breach_reason}")

        if self.child is not None and self.child.poll() is not None:
            for _ in range(5):
                if self.breach_event.is_set() or (self.watchdog and self.watchdog.last_check_result and not self.watchdog.last_check_result.healthy):
                    break
                time.sleep(0.05)
            if self.breach_event.is_set():
                b_res = self.breach_result[0]
                reason = b_res.breach_reason if b_res else "unknown breach"
                raise RuntimeError(f"Watchdog breach during {operation_label}: {reason}")
            if self.watchdog and self.watchdog.last_check_result and not self.watchdog.last_check_result.healthy:
                raise RuntimeError(f"Watchdog breach during {operation_label}: {self.watchdog.last_check_result.breach_reason}")
            raise RuntimeError(f"Server child exited unexpectedly with code {self.child.returncode} during {operation_label}")

    @contextlib.contextmanager
    def managed_stage(self) -> Generator[SmokeStageContext, None, None]:
        """
        Context manager that handles child spawning, watchdog supervision, readiness polling,
        and guaranteed cleanup on all exit paths.
        """
        run_start_time = time.time()
        start_mono = time.monotonic()
        deadline = start_mono + self.config.timeout_s

        instance_token = self.config.instance_token or f"stage-{uuid.uuid4().hex[:12]}"

        report: dict[str, Any] = {
            "instance_token": instance_token,
            "stage": self.config.stage_name,
            "model": self.config.model_identifier,
            "port": self.config.port,
            "control_port": self.config.control_port,
            "timeout_s": self.config.timeout_s,
            "status": "in_progress",
            "start_time": run_start_time,
            "checks": {},
            "metrics": {},
            "fixtures": {},
            "errors": [],
            "telemetry_samples": [],
            "captured_logs": "",
            "child_cleanup": {
                "cleaned": False,
                "pid": None,
                "exit_code": None,
            },
        }
        self.last_report = report

        # Step 1: Preflight (persisting failure report fail-closed if preflight fails)
        try:
            preflight = self.validate_preflight()
            report["preflight"] = preflight
            report["metrics"]["preflight_headroom_mb"] = preflight.get("headroom_mb")
        except Exception as preflight_err:
            report["status"] = "failed"
            report["duration_s"] = round(time.time() - run_start_time, 3)
            err_msg = f"Preflight validation failed: {preflight_err}"
            report.setdefault("errors", []).append(err_msg)
            self.last_report = report
            self._save_report(report)
            raise

        if self.config.dry_run:
            ctx = SmokeStageContext(
                supervisor=self,
                client=SmokeHttpClient(),
                child=None,  # type: ignore
                watchdog=None,  # type: ignore
                instance_token=instance_token,
                base_url=f"http://{self.config.host}:{self.config.port}",
                ctrl_url=f"http://{self.config.host}:{self.config.control_port}",
                start_mono=start_mono,
                deadline=deadline,
                report=report,
            )
            try:
                report["status"] = "dry_run_completed"
                yield ctx
            except Exception as e:
                report["status"] = "failed"
                err_str = str(e)
                if err_str not in report.get("errors", []):
                    report.setdefault("errors", []).append(err_str)
                raise
            finally:
                ctx.client.close()
                report["duration_s"] = round(time.time() - run_start_time, 3)
                self.last_report = report
                self._save_report(report)
            return

        # Step 2: Prepare Child Environment
        child_env = os.environ.copy()
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        child_env["HF_DATASETS_OFFLINE"] = "1"
        child_env["MLX_INSTANCE_TOKEN"] = instance_token
        child_env["MLX_EMBED_PORT"] = str(self.config.port)
        child_env["MLX_CONTROL_PORT"] = str(self.config.control_port)

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if "PYTHONPATH" in child_env:
            child_env["PYTHONPATH"] = f"{repo_root}:{child_env['PYTHONPATH']}"
        else:
            child_env["PYTHONPATH"] = repo_root

        if self.config.extra_env:
            child_env.update(self.config.extra_env)

        child: Optional[subprocess.Popen] = None
        capture: Optional[_BoundedLogCapture] = None
        client = SmokeHttpClient(default_timeout_s=5.0)
        t_supervisor: Optional[threading.Thread] = None
        stop_supervisor = threading.Event()
        self.breach_event.clear()
        self.breach_result = [None]
        self.telemetry_samples = []
        cleanup_errors: list[str] = []

        try:
            capture = _BoundedLogCapture(f"mlx_smoke_{self.config.stage_name}_", self.config.max_log_bytes)
            self._log_capture = capture
            child = spawn_owned_process(
                self.config.cmd,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.child = child
            capture.start(child)
            report["child_pid"] = child.pid

            # Step 3: Instantiate MLXWatchdog & Background Supervisor Thread
            watchdog_config = MLXWatchdogConfig(
                pid=child.pid,
                host=self.config.host,
                port=self.config.port,
                control_port=self.config.control_port,
                max_rss_mb=self.config.max_rss_mb,
                max_swap_growth_mb=self.config.max_swap_growth_mb,
                min_free_memory_pct=self.config.min_free_memory_pct,
                check_interval_s=self.config.check_interval_s,
                startup_grace_period_s=min(self.config.startup_grace_period_s, self.config.timeout_s),
                stalled_inference_timeout_s=min(self.config.stalled_inference_timeout_s, self.config.timeout_s),
                expected_cmd_pattern=self.config.expected_cmd_pattern,
                instance_token=instance_token,
                health_timeout_s=min(1.0, self.config.timeout_s),
                grace_period_s=0.2,
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
            self.watchdog = watchdog

            def _supervisor_loop():
                while not stop_supervisor.is_set():
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
                        self.breach_result[0] = res
                        self.breach_event.set()
                        break

                    try:
                        if capture.error:
                            raise RuntimeError(f"Server log capture failed: {capture.error}")
                        check = watchdog.check_step()
                    except Exception as e:
                        check = WatchdogCheckResult(
                            timestamp=time.time(), healthy=False,
                            breach_type=BreachType.TELEMETRY_UNAVAILABLE,
                            breach_reason=f"Supervisor check failed closed: {e}",
                        )
                    if not check.healthy:
                        self.breach_result[0] = check
                        self.breach_event.set()
                        break

                    if check.metrics:
                        self.telemetry_samples.append({
                            "timestamp": round(check.timestamp, 2),
                            "elapsed_s": round(time.monotonic() - start_mono, 2),
                            "rss_mb": check.metrics.get("rss_mb"),
                            "swap_used_mb": check.metrics.get("swap_used_mb"),
                            "swap_growth_mb": check.metrics.get("swap_growth_mb"),
                            "memory_free_pct": check.metrics.get("memory_free_pct"),
                        })

                    stop_supervisor.wait(timeout=self.config.check_interval_s)

            t_supervisor = threading.Thread(
                target=_supervisor_loop,
                daemon=True,
                name=f"Smoke-Watchdog-{self.config.stage_name}",
            )
            t_supervisor.start()

            # Step 4: Poll Control Port for Readiness
            base_url = f"http://{self.config.host}:{self.config.port}"
            ctrl_url = f"http://{self.config.host}:{self.config.control_port}"

            ready = False
            startup_deadline = min(deadline, time.monotonic() + self.config.startup_grace_period_s)
            while time.monotonic() < startup_deadline:
                if self.breach_event.is_set():
                    b_res = self.breach_result[0]
                    raise RuntimeError(_startup_failure_message(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}", capture.path))

                if child.poll() is not None:
                    for _ in range(5):
                        if self.breach_event.is_set() or (watchdog.last_check_result and not watchdog.last_check_result.healthy):
                            break
                        time.sleep(0.05)
                    if self.breach_event.is_set():
                        b_res = self.breach_result[0]
                        raise RuntimeError(_startup_failure_message(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}", capture.path))
                    if watchdog.last_check_result and not watchdog.last_check_result.healthy:
                        raise RuntimeError(_startup_failure_message(f"Watchdog breach during server startup: {watchdog.last_check_result.breach_reason}", capture.path))

                    log_tail = ""
                    try:
                        log_tail = _read_log_tail(capture.path, 2000)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Server child exited prematurely with code {child.returncode}. Server logs:\n{log_tail}"
                    )

                try:
                    status_code, data = client.get_json(f"{ctrl_url}/health", deadline=startup_deadline, timeout_s=1.0)
                    if status_code == 200 and isinstance(data, dict):
                        if (
                            data.get("ready") is True
                            and data.get("instance_token") == instance_token
                            and data.get("pid") == child.pid
                        ):
                            ready = True
                            report["metrics"]["startup_time_s"] = round(time.monotonic() - start_mono, 3)
                            report["checks"]["server_ready"] = True
                            break
                    elif status_code == 503:
                        pass
                except Exception:
                    pass
                time.sleep(0.1)

            if not ready:
                if self.breach_event.is_set():
                    b_res = self.breach_result[0]
                    raise RuntimeError(_startup_failure_message(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}", capture.path))
                if watchdog.last_check_result and not watchdog.last_check_result.healthy:
                    raise RuntimeError(_startup_failure_message(f"Watchdog breach during server startup: {watchdog.last_check_result.breach_reason}", capture.path))
                raise TimeoutError(_startup_failure_message(f"Server child failed to report ready on {ctrl_url}/health within startup grace period", capture.path))

            ctx = SmokeStageContext(
                supervisor=self,
                client=client,
                child=child,
                watchdog=watchdog,
                instance_token=instance_token,
                base_url=base_url,
                ctrl_url=ctrl_url,
                start_mono=start_mono,
                deadline=deadline,
                report=report,
            )
            yield ctx

            if time.monotonic() >= deadline:
                raise TimeoutError("Absolute execution deadline exceeded during stage execution")
            self.check_breach("stage completion")
            if report.get("errors"):
                report["status"] = "failed"
            elif report.get("status") == "in_progress":
                report["status"] = "passed"

        except Exception as e:
            report["status"] = "failed"
            err_str = str(e)
            if err_str not in report.get("errors", []):
                report.setdefault("errors", []).append(err_str)
            raise
        finally:
            # Stop monitoring before intentional teardown, then clean even an
            # already-exited leader's group before draining and unlinking logs.
            stop_supervisor.set()
            if t_supervisor is not None:
                t_supervisor.join(timeout=1.5)
                if t_supervisor.is_alive():
                    cleanup_errors.append("Watchdog supervisor thread did not stop within teardown budget")
            client.close()

            if child is not None:
                try:
                    cleaned, sigkill_used = terminate_owned_process(child)
                    report["child_cleanup"] = {
                        "cleaned": cleaned, "pid": child.pid, "exit_code": child.returncode,
                        "sigkill_used": sigkill_used,
                    }
                    if not cleaned:
                        cleanup_errors.append("Owned process group failed to terminate")
                except Exception as cleanup_err:
                    cleanup_errors.append(f"Child cleanup error: {cleanup_err}")
                    report["child_cleanup"] = {
                        "cleaned": False, "pid": child.pid, "error": str(cleanup_err),
                    }

            if capture is not None:
                try:
                    capture.finish()
                    report["captured_logs"] = _read_log_tail(capture.path, self.config.max_log_bytes)
                    if capture.error:
                        cleanup_errors.append(f"Log capture error: {capture.error}")
                except Exception as log_err:
                    cleanup_errors.append(f"Log capture error: {log_err}")
                finally:
                    try:
                        capture.close()
                    except Exception as log_err:
                        cleanup_errors.append(f"Log cleanup error: {log_err}")
            self._log_capture = None

            if self.breach_event.is_set():
                report["status"] = "failed"
                breach = self.breach_result[0]
                reason = breach.breach_reason if breach else "Unknown watchdog breach"
                if reason not in report["errors"]:
                    report["errors"].append(reason)
            if cleanup_errors:
                report["status"] = "failed"
                report["cleanup_errors"] = cleanup_errors

            report["telemetry_samples"] = self.telemetry_samples
            report["duration_s"] = round(time.time() - run_start_time, 3)
            self.last_report = report
            self._save_report(report)

    def _save_report(self, report: dict[str, Any]):
        if self.config.output_file:
            try:
                out_dir = os.path.dirname(os.path.abspath(self.config.output_file))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(self.config.output_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                report.setdefault("errors", []).append(f"Failed to write output JSON to {self.config.output_file}: {e}")
