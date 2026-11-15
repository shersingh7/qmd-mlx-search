"""W07 regressions: real loopback drips and disposable, isolated process trees only."""

import contextlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from unittest.mock import MagicMock

import pytest

from scripts.qmd_mlx import supervisor
from scripts.qmd_mlx.watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    TargetProcessValidator,
    WatchdogDefaults,
)


@contextlib.contextmanager
def loopback_response(mode, delay_s=0.015):
    """A finite server: tests cannot pass by waiting for an endless fixture to end."""
    stop = threading.Event()
    started = threading.Event()
    peer_closed = threading.Event()
    stats = {"drip_writes": 0, "error": None}
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.1)
    port = listener.getsockname()[1]

    def serve():
        client = None
        try:
            while not stop.is_set():
                try:
                    client, _ = listener.accept()
                    break
                except socket.timeout:
                    continue
            if client is None:
                return
            client.settimeout(0.5)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            request = bytearray()
            while b"\r\n\r\n" not in request:
                data = client.recv(4096)
                if not data:
                    return
                request.extend(data)
            headers, _, received_body = bytes(request).partition(b"\r\n\r\n")
            content_length = 0
            for line in headers.split(b"\r\n")[1:]:
                name, _, value = line.partition(b":")
                if name.lower() == b"content-length":
                    content_length = int(value.strip())
            while len(received_body) < content_length:
                data = client.recv(min(4096, content_length - len(received_body)))
                if not data:
                    return
                received_body += data
            stats["request_body_bytes"] = len(received_body)
            started.set()
            body = b'{"ok":true}' if mode == "success" else b"x" * 50
            header = (
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            )
            if mode == "header":
                for byte in header:
                    if stop.wait(delay_s):
                        return
                    client.sendall(bytes([byte]))
                    stats["drip_writes"] += 1
            else:
                client.sendall(header)
            if mode == "body":
                for byte in body:
                    if stop.wait(delay_s):
                        return
                    client.sendall(bytes([byte]))
                    stats["drip_writes"] += 1
            else:
                client.sendall(body)
            # The response has a known length. Closing must not await server EOF.
            if client.recv(1) == b"":
                peer_closed.set()
        except (OSError, ConnectionError) as error:
            stats["error"] = repr(error)
            peer_closed.set()
        finally:
            if client is not None:
                client.close()

    thread = threading.Thread(target=serve, name="W07-finite-loopback")
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/health", started, peer_closed, stats
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=1)
        assert not thread.is_alive(), "Owned HTTP fixture failed to stop"


@pytest.mark.parametrize("mode", ["header", "body"])
@pytest.mark.parametrize("budget_source", ["absolute", "default", "per_request"])
def test_smoke_http_total_deadline_during_tiny_drips(mode, budget_source, monkeypatch):
    # Leave enough scheduling room for several actual drip writes on loaded Macs;
    # a fast timeout before any bytes arrive is not deadline-bypass evidence.
    budget = 0.5
    closed = []
    original_close = http.client.HTTPResponse.close

    def record_close(response):
        closed.append(response)
        return original_close(response)

    monkeypatch.setattr(http.client.HTTPResponse, "close", record_close)
    with loopback_response(mode) as (url, started, _, stats):
        client = supervisor.SmokeHttpClient(default_timeout_s=budget if budget_source == "default" else 1)
        before_threads = {t.ident for t in threading.enumerate()}
        start = time.monotonic()
        kwargs = {}
        if budget_source == "absolute":
            kwargs["deadline"] = start + budget
        elif budget_source == "per_request":
            kwargs["per_request_timeout_s"] = budget
        try:
            with pytest.raises(TimeoutError):
                client.request("GET", url, **kwargs)
            elapsed = time.monotonic() - start
            assert started.is_set(), "Probe did not reach the synthetic HTTP server"
            assert stats["drip_writes"] >= 3, f"Not a real tiny-drip probe: {stats}"
            assert elapsed < 0.85, f"Total HTTP budget {budget}s returned after {elapsed:.3f}s"
            assert closed and all(response.closed for response in closed)
            assert {t.ident for t in threading.enumerate()} <= before_threads
        finally:
            client.close()


@pytest.mark.parametrize("mode,limit", [("success", 1024), ("oversize", 8)])
def test_smoke_http_closes_response_on_success_and_size_error(mode, limit, monkeypatch):
    closed = []
    original_close = http.client.HTTPResponse.close

    def record_close(response):
        closed.append(response)
        return original_close(response)

    monkeypatch.setattr(http.client.HTTPResponse, "close", record_close)
    with loopback_response(mode) as (url, _, peer_closed, _):
        client = supervisor.SmokeHttpClient(max_response_bytes=limit)
        try:
            if mode == "success":
                response = client.request("GET", url)
                assert response.json() == {"ok": True}
                assert response.raw.closed
            else:
                with pytest.raises(ValueError, match="byte limit"):
                    client.request("GET", url)
            assert peer_closed.wait(0.3), "Client kept the response socket open"
            assert closed and all(response.closed for response in closed)
        finally:
            client.close()


def synthetic_sampler():
    sampler = MagicMock(spec=SystemMemorySampler)
    sampler.get_process_start_time.side_effect = lambda pid: f"owned synthetic pid={pid}"
    sampler.get_process_cmdline.return_value = "python synthetic-server"
    sampler.get_installed_ram_mb.return_value = 32768.0
    sampler.get_memory_headroom_mb.return_value = 16000.0
    sampler.get_swap_used_mb.return_value = 0.0
    sampler.get_memory_free_pct.return_value = 50.0
    sampler.get_process_rss_mb.return_value = 16.0
    sampler.compute_conservative_defaults.return_value = WatchdogDefaults(
        installed_ram_mb=32768, headroom_mb=16000, max_rss_mb=8000, max_swap_growth_mb=500,
    )
    return sampler


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_smoke_http_json_helpers_preserve_buffered_response_api(method):
    with loopback_response("success") as (url, started, peer_closed, stats):
        client = supervisor.SmokeHttpClient()
        try:
            result = client.get_json(url) if method == "GET" else client.post_json(url, {"probe": "synthetic"})
            assert result == (200, {"ok": True})
            assert started.is_set() and peer_closed.wait(0.5)
            assert stats["request_body_bytes"] > 0 if method == "POST" else stats["request_body_bytes"] == 0
        finally:
            client.close()


@contextlib.contextmanager
def disposable_process():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=1)


def owned_tree_code(pid_path, parent_exits=False):
    """Grandchild ignores TERM so cleanup must escalate after leader exits."""
    child_code = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pid_path)!r},'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    return (
        "import os,subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
        f"while not os.path.exists({str(pid_path)!r}): time.sleep(.005)\n"
        "print('TREE_READY',flush=True)\n"
        + ("sys.exit(0)\n" if parent_exits else "time.sleep(30)\n")
    )


def wait_file(path, timeout=1):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.01)
    pytest.fail(f"Owned process did not publish PID fixture {path}")


def assert_not_executing(pid):
    end = time.monotonic() + 0.7
    while time.monotonic() < end:
        if not TargetProcessValidator.is_pid_alive(pid):
            return
        time.sleep(0.02)
    pytest.fail(f"Owned descendant {pid} still executes after cleanup")


def cleanup_fixture_pid(path):
    # Failing-baseline tests must not themselves leak the synthetic grandchild.
    if path.exists():
        text = path.read_text()
        if text and TargetProcessValidator.is_pid_alive(int(text)):
            os.kill(int(text), signal.SIGKILL)


@pytest.mark.parametrize("exit_mode", ["timeout", "breach", "parent_exit", "barrier"])
def test_subordinate_cleans_owned_tree_and_preserves_sentinel(tmp_path, exit_mode):
    pid_path = tmp_path / "grandchild.pid"
    with disposable_process() as sentinel:
        try:
            def breach(_):
                if pid_path.exists() and pid_path.read_text():
                    raise RuntimeError("synthetic watchdog breach")

            result = supervisor.run_subordinate_process(
                [sys.executable, "-c", owned_tree_code(pid_path, exit_mode == "parent_exit")],
                deadline=time.monotonic() + (0.25 if exit_mode == "timeout" else 2),
                check_breach=breach if exit_mode == "breach" else None,
                barrier_sentinel="TREE_READY" if exit_mode == "barrier" else None,
                kill_on_barrier=exit_mode == "barrier",
                poll_interval_s=0.01,
            )
            assert result.timed_out is (exit_mode == "timeout")
            assert result.breached is (exit_mode == "breach")
            if exit_mode == "parent_exit":
                assert result.exit_code == 0 and result.error is None
            if exit_mode == "barrier":
                assert result.barrier_reached
            assert_not_executing(wait_file(pid_path))
            assert sentinel.poll() is None
        finally:
            cleanup_fixture_pid(pid_path)


def test_subordinate_log_capture_memory_and_disk_are_bounded(monkeypatch):
    paths = []
    original_temp = tempfile.NamedTemporaryFile

    def record_temp(*args, **kwargs):
        file = original_temp(*args, **kwargs)
        paths.append(Path(file.name))
        return file

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", record_temp)
    observed_sizes = []

    def observe(_):
        observed_sizes.extend(path.stat().st_size for path in paths if path.exists())

    tracemalloc.start()
    try:
        result = supervisor.run_subordinate_process(
            [sys.executable, "-c", "import os,time; os.write(1,b'x'*2097152+b'TAIL'); time.sleep(.1)"],
            deadline=time.monotonic() + 3,
            max_log_bytes=128,
            check_breach=observe,
            poll_interval_s=0.01,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.exit_code == 0 and result.error is None
    assert result.captured_logs.endswith("TAIL")
    assert len(result.captured_logs.encode()) <= 128
    assert observed_sizes and max(observed_sizes) <= 128
    assert peak < 1024 * 1024, f"Reading a 2MiB child log allocated {peak} bytes"
    assert all(not path.exists() for path in paths)


def test_subordinate_barrier_is_detected_across_log_chunks():
    result = supervisor.run_subordinate_process(
        [sys.executable, "-c", "import os,time; os.write(1,b'prefix-TREE_'); time.sleep(.08); os.write(1,b'READY-tail'); time.sleep(2)"],
        deadline=time.monotonic() + 1,
        max_log_bytes=8,
        barrier_sentinel="TREE_READY",
        kill_on_barrier=True,
        poll_interval_s=0.01,
    )
    assert result.barrier_reached and not result.timed_out
    assert len(result.captured_logs.encode()) <= 8


@pytest.mark.parametrize("exit_mode", ["exception", "parent_exit", "timeout", "breach"])
def test_stage_cleans_owned_tree_and_reports_failure(tmp_path, monkeypatch, exit_mode):
    pid_path = tmp_path / "stage-grandchild.pid"
    sampler = synthetic_sampler()
    script = owned_tree_code(pid_path)
    # Readiness is synthetic; no model or production service is involved.
    monkeypatch.setattr(supervisor.SmokeHttpClient, "get_json", lambda *a, **kw: (
        200, {"ready": True, "pid": sup.child.pid, "instance_token": "owned-stage"},
    ))
    monkeypatch.setattr(MLXWatchdog, "_default_health_probe", staticmethod(lambda *a, **kw: (
        True, {"state": "idle", "ready": True, "worker_alive": True, "worker_idle": True,
               "completed_sequence": 0, "pid": sup.child.pid, "instance_token": "owned-stage"}, 200,
    )))
    cfg = supervisor.StageSupervisorConfig(
        cmd=[sys.executable, "-c", script], instance_token="owned-stage",
        timeout_s=0.4 if exit_mode == "timeout" else 3, startup_grace_period_s=0.2,
        check_interval_s=0.02, output_file=str(tmp_path / "report.json"),
    )
    sup = supervisor.SmokeStageSupervisor(cfg, sampler=sampler)
    with disposable_process() as sentinel:
        try:
            with pytest.raises((RuntimeError, ValueError, TimeoutError)):
                with sup.managed_stage() as ctx:
                    grandchild_pid = wait_file(pid_path)
                    assert os.getpgid(ctx.child.pid) == ctx.child.pid
                    # Publishing a grandchild PID precedes the parent's print.
                    # Wait for the separate log handshake before terminating it.
                    log_deadline = time.monotonic() + 0.3
                    while "TREE_READY" not in supervisor._read_log_tail(sup._log_capture.path, 128):
                        assert time.monotonic() < log_deadline, "Owned parent did not publish its log handshake"
                        time.sleep(0.005)
                    if exit_mode == "parent_exit":
                        ctx.child.terminate()
                        ctx.child.wait(timeout=1)
                        ctx.check_breach("owned parent exit")
                    elif exit_mode == "timeout":
                        time.sleep(0.6)
                    elif exit_mode == "breach":
                        sampler.get_process_rss_mb.return_value = 9000
                        assert sup.breach_event.wait(1), "Synthetic RSS breach was not observed"
                    else:
                        raise ValueError("synthetic stage exception")
            assert_not_executing(grandchild_pid)
            assert sentinel.poll() is None
            report = json.loads((tmp_path / "report.json").read_text())
            assert report["status"] == "failed"
            assert report["child_cleanup"]["cleaned"] is True
            assert report["child_cleanup"]["sigkill_used"] is True
            assert "TREE_READY" in report["captured_logs"]
            assert not any(t.name == "Smoke-Watchdog-stage" for t in threading.enumerate())
        finally:
            cleanup_fixture_pid(pid_path)


def test_watchdog_exited_target_is_not_reported_healthy():
    with disposable_process() as proc:
        watchdog = MLXWatchdog(
            MLXWatchdogConfig(pid=proc.pid, instance_token="owned", dry_run=True),
            sampler=synthetic_sampler(), owned_child=proc,
        )
        proc.terminate()
        proc.wait(timeout=1)
        result = watchdog.check_step()
        assert result.breach_type == BreachType.TARGET_EXITED
        assert not result.healthy
        assert watchdog.run_loop() == 0


def test_watchdog_cleans_group_when_owned_leader_has_exited(tmp_path):
    from scripts.qmd_mlx.watchdog import spawn_owned_process, terminate_owned_process

    pid_path = tmp_path / "watchdog-descendant.pid"
    child = spawn_owned_process(
        [sys.executable, "-c", owned_tree_code(pid_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        watchdog = MLXWatchdog(
            MLXWatchdogConfig(pid=child.pid, instance_token="owned", grace_period_s=0.1),
            sampler=synthetic_sampler(), owned_child=child,
        )
        descendant = wait_file(pid_path)
        child.terminate()
        child.wait(timeout=1)
        result = watchdog.check_step()
        assert result.breach_type == BreachType.TARGET_EXITED and not result.healthy
        assert result.sigkill_used
        assert_not_executing(descendant)
    finally:
        terminate_owned_process(child)
        cleanup_fixture_pid(pid_path)


def test_log_tail_read_is_bounded_for_large_sparse_file(tmp_path):
    path = tmp_path / "sparse-owned.log"
    with path.open("wb") as stream:
        stream.seek(64 * 1024 * 1024)
        stream.write(b"tail-\xff-" + "\u20ac".encode() * 20)
    tracemalloc.start()
    try:
        tail = supervisor._read_log_tail(str(path), 32)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert tail.endswith("\u20ac" * 10)
    assert len(tail.encode()) <= 32
    assert peak < 65536
    assert supervisor._read_log_tail(str(path), 0) == ""


def test_owned_group_probe_distinguishes_darwin_zombies_from_permission_failure(monkeypatch):
    from scripts.qmd_mlx.watchdog import spawn_owned_process, terminate_owned_process

    child = spawn_owned_process([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL)
    child.wait(timeout=1)
    group = getattr(child, "_qmd_owned_group")
    real_members = group._has_live_members
    observations = []
    attempted = []

    def leader_exits_after_live_readback():
        observations.append(True)
        # Simulate a live ps snapshot just before the already pinned leader
        # exits. Subsequent observations inspect the real zombie-only group.
        return True if len(observations) == 1 else real_members()

    def zombie_probe(pgid, sig):
        assert pgid == child.pid
        attempted.append(sig)
        raise PermissionError("synthetic Darwin zombie-only group")

    monkeypatch.setattr(os, "killpg", zombie_probe)
    monkeypatch.setattr(group, "_has_live_members", leader_exits_after_live_readback)
    assert terminate_owned_process(child) == (True, False)
    assert attempted == [signal.SIGTERM]
    assert len(observations) >= 2
    with pytest.raises(ChildProcessError):
        os.waitpid(child.pid, os.WNOHANG)


def test_stage_disk_and_report_logs_are_bounded(tmp_path, monkeypatch):
    paths = []
    original_temp = tempfile.NamedTemporaryFile

    def record_temp(*args, **kwargs):
        file = original_temp(*args, **kwargs)
        paths.append(Path(file.name))
        return file

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", record_temp)
    monkeypatch.setattr(supervisor.SmokeHttpClient, "get_json", lambda *a, **kw: (
        200, {"ready": True, "pid": sup.child.pid, "instance_token": "bounded-log"},
    ))
    monkeypatch.setattr(MLXWatchdog, "_default_health_probe", staticmethod(lambda *a, **kw: (
        True, {"state": "idle", "ready": True, "worker_alive": True, "worker_idle": True,
               "completed_sequence": 0, "pid": sup.child.pid, "instance_token": "bounded-log"}, 200,
    )))
    sup = supervisor.SmokeStageSupervisor(supervisor.StageSupervisorConfig(
        cmd=[sys.executable, "-c", "import os,time; os.write(1,b'x'*1048576+b'TAIL'); time.sleep(30)"],
        max_log_bytes=128, instance_token="bounded-log", startup_grace_period_s=0.5,
        timeout_s=3, output_file=str(tmp_path / "bounded-report.json"),
    ), sampler=synthetic_sampler())
    sizes = []
    with sup.managed_stage():
        end = time.monotonic() + 1
        while time.monotonic() < end:
            sizes.extend(path.stat().st_size for path in paths)
            if paths and supervisor._read_log_tail(str(paths[0]), 128).endswith("TAIL"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("Owned stage did not finish its synthetic log flood")
    report = json.loads((tmp_path / "bounded-report.json").read_text())
    assert report["status"] == "passed" and report["child_cleanup"]["cleaned"]
    assert report["captured_logs"].endswith("TAIL")
    assert len(report["captured_logs"].encode()) <= 128
    assert sizes and max(sizes) <= 128
    assert all(not path.exists() for path in paths)


def test_owned_group_does_not_hide_real_signal_permission_failure(monkeypatch):
    from scripts.qmd_mlx.watchdog import spawn_owned_process, terminate_owned_process

    child = spawn_owned_process(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdout=subprocess.DEVNULL,
    )
    try:
        with monkeypatch.context() as patch:
            def denied(pgid, sig):
                assert pgid == child.pid
                raise PermissionError("synthetic real signal permission failure")

            patch.setattr(os, "killpg", denied)
            with pytest.raises(PermissionError, match="signal permission failure"):
                terminate_owned_process(child)
            assert child.poll() is None
    finally:
        assert terminate_owned_process(child)[0]


def test_watchdog_revalidates_owned_group_identity_before_sigkill(monkeypatch):
    from scripts.qmd_mlx.watchdog import spawn_owned_process, terminate_owned_process

    child = spawn_owned_process(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdout=subprocess.DEVNULL,
    )
    try:
        sampler = synthetic_sampler()
        watchdog = MLXWatchdog(
            MLXWatchdogConfig(pid=child.pid, instance_token="owned", grace_period_s=0.01),
            sampler=sampler, owned_child=child,
        )
        sampler.get_process_start_time.side_effect = [watchdog.expected_start_time, "recycled identity"]
        delivered = []
        with monkeypatch.context() as patch:
            def record_signal(pgid, sig):
                assert pgid == child.pid
                if sig:
                    delivered.append(sig)

            patch.setattr(os, "killpg", record_signal)
            cleaned, sigkill_used = watchdog.terminate_target_process("synthetic group identity change")
        assert not cleaned and not sigkill_used
        assert delivered == [signal.SIGTERM]
        assert child.poll() is None
    finally:
        assert terminate_owned_process(child)[0]
