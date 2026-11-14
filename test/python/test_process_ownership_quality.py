"""Negative process-ownership probes; PID reuse is simulated, never attempted."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from scripts.qmd_mlx import watchdog as watchdog_module
from scripts.qmd_mlx import process_anchor
from scripts.qmd_mlx.process_anchor import wait_child_without_reaping
from scripts.qmd_mlx.watchdog import (
    TargetValidationError,
    spawn_owned_process,
    terminate_owned_process,
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="Owned groups require POSIX")


def reap_exact_child_if_waitable(child):
    """Deliberately bypass the handle, emulating a competing application reaper."""
    try:
        pid, status = os.waitpid(child.pid, os.WNOHANG)
    except ChildProcessError:
        return
    assert pid == child.pid, "Fixture leader must already have exited"
    child.returncode = os.waitstatus_to_exitcode(status)


@pytest.mark.parametrize("reuse_at", ["before_cleanup", "after_term"])
def test_reaped_leader_never_authorizes_signals_to_recycled_numeric_group(monkeypatch, reuse_at):
    child = spawn_owned_process(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child.wait(timeout=2)
    group = getattr(child, "_qmd_owned_group")
    if reuse_at == "before_cleanup":
        reap_exact_child_if_waitable(child)
    delivered = []

    def recycled_group_signal(pgid, sig):
        assert pgid == child.pid
        if sig:
            delivered.append(sig)
        if sig == signal.SIGTERM and reuse_at == "after_term":
            # No real signals are delivered. The original leader is removed
            # from the kernel here, and subsequent numerical lookups lie.
            reap_exact_child_if_waitable(child)

    def recycled_group_telemetry(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=f"{child.pid} S\n", stderr="")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", recycled_group_signal)
            patch.setattr(subprocess, "run", recycled_group_telemetry)
            with pytest.raises(TargetValidationError, match="(?i)(anchor|ownership|reap)"):
                terminate_owned_process(child, grace_s=0.01)
        assert delivered == ([] if reuse_at == "before_cleanup" else [signal.SIGTERM])
        assert group.cleaned is False
        # Loss of the anchor is permanent, not a retryable PID/start-time lookup.
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", recycled_group_signal)
            with pytest.raises(TargetValidationError, match="(?i)(anchor|ownership|reap)"):
                terminate_owned_process(child, grace_s=0)
        assert delivered == ([] if reuse_at == "before_cleanup" else [signal.SIGTERM])
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
    finally:
        # Even an unexpectedly failing negative probe must release its original
        # zombie. Never try to reacquire the numerical group's lost capability.
        reap_exact_child_if_waitable(child)


@pytest.mark.parametrize("observer", ["poll", "wait", "communicate", "threaded_wait"])
def test_public_handle_exit_observers_retain_anchor_until_cleanup(observer, monkeypatch):
    child = spawn_owned_process(
        [sys.executable, "-c", "raise SystemExit(7)"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        if observer == "poll":
            deadline = time.monotonic() + 2
            while child.poll() is None:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        elif observer == "wait":
            assert child.wait(timeout=2) == 7
        elif observer == "communicate":
            assert child.communicate(timeout=2) == (b"", None)
        else:
            observed = []
            threads = [threading.Thread(target=lambda: observed.append(child.wait(timeout=2))) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
                assert not thread.is_alive()
            assert observed == [7, 7]
        assert child.returncode == 7
        # Both calls query the kernel, rather than the handle's cached status.
        assert wait_child_without_reaping(child.pid) == 7
        assert wait_child_without_reaping(child.pid) == 7
        assert terminate_owned_process(child, grace_s=0) == (True, False)
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
        # Once consumed, the capability never looks up or signals its old PGID.
        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", lambda *a: pytest.fail("Consumed group was signalled again"))
            patch.setattr(watchdog_module, "wait_child_without_reaping", lambda *a: pytest.fail("Consumed anchor was inspected again"))
            assert terminate_owned_process(child) == (True, False)
    finally:
        terminate_owned_process(child)
        if child.stdout is not None:
            child.stdout.close()


def test_successful_signal_syscalls_are_not_a_cleanup_readback(monkeypatch):
    child = spawn_owned_process(
        [sys.executable, "-c", "import time; time.sleep(15)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    delivered = []
    try:
        with monkeypatch.context() as patch:
            # Simulate accepted TERM/KILL without delivering either. Real ps
            # readbacks must still see the executing, exact owned child.
            patch.setattr(os, "killpg", lambda pgid, sig: delivered.append((pgid, sig)))
            cleaned, sigkill_used = terminate_owned_process(child, grace_s=0)
        assert delivered == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]
        assert sigkill_used and not cleaned
        assert not getattr(child, "_qmd_owned_group").cleaned
        assert child.poll() is None
    finally:
        assert terminate_owned_process(child)[0]
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)


@pytest.mark.parametrize("failure_at", ["before_term", "after_term"])
def test_anchor_syscall_failure_refuses_unverifiable_signalling(monkeypatch, failure_at):
    child = spawn_owned_process(
        [sys.executable, "-c", "import time; time.sleep(15)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    real_wait = watchdog_module.wait_child_without_reaping
    delivered = []

    def unavailable_anchor(pid):
        if failure_at == "before_term" or delivered:
            raise PermissionError("injected anchor syscall denial")
        return real_wait(pid)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(watchdog_module, "wait_child_without_reaping", unavailable_anchor)
            patch.setattr(os, "killpg", lambda pgid, sig: delivered.append(sig))
            with pytest.raises(PermissionError, match="anchor syscall denial"):
                terminate_owned_process(child, grace_s=0)
        assert delivered == ([] if failure_at == "before_term" else [signal.SIGTERM])
        assert not getattr(child, "_qmd_owned_group").cleaned
        assert child.poll() is None
    finally:
        assert terminate_owned_process(child)[0]
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)


def test_empty_no_match_readback_without_diagnostics_can_finish_cleanup(monkeypatch):
    child = spawn_owned_process(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        assert child.wait(timeout=2) == 0
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr=""))
            assert terminate_owned_process(child, grace_s=0) == (True, False)
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
    finally:
        terminate_owned_process(child)


CLI_DENIAL_RUNNER = r'''
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from scripts.qmd_mlx.watchdog import SystemMemorySampler, TargetProcessValidator, terminate_owned_process

root = Path(sys.argv[1])
cli = importlib.import_module("scripts.qmd-mlx-watchdog")
owned = []
real_spawn = cli.spawn_owned_process
real_killpg = os.killpg
sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class MemoryOnlySampler(SystemMemorySampler):
    def get_installed_ram_mb(self): return 32768.0
    def get_memory_headroom_mb(self): return 16000.0
    def get_swap_used_mb(self): return 0.0
    def get_memory_free_pct(self): return 50.0
    def get_process_rss_mb(self, pid): return 16.0
    def get_process_start_time(self, pid):
        end = time.monotonic() + 2
        while not (root / "child-ready").exists():
            if time.monotonic() >= end: raise TimeoutError("CPU child not ready")
            time.sleep(.01)
        return super().get_process_start_time(pid)

def remember_spawn(*args, **kwargs):
    child = real_spawn(*args, **kwargs)
    owned.append(child)
    return child

def denied(pgid, sig):
    assert pgid == owned[0].pid
    if sig: raise PermissionError("standalone owned signal denial")
    return real_killpg(pgid, sig)

cli.SystemMemorySampler = MemoryOnlySampler
cli.spawn_owned_process = remember_spawn
evidence = {}
try:
    with socket.socket() as inference, socket.socket() as control:
        inference.bind(("127.0.0.1", 0)); control.bind(("127.0.0.1", 0))
        ports = inference.getsockname()[1], control.getsockname()[1]
    os.killpg = denied
    code = cli.main([
        "--port", str(ports[0]), "--control-port", str(ports[1]),
        "--once", "--json", "--grace-period-s", "0.01", "--health-timeout-s", "0.05",
        "--launch", sys.executable, "-c",
        "import pathlib,sys,time; print('disposable child stdout log',flush=True); "
        "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(8)", str(root / "child-ready"),
    ])
    evidence.update(cli_exit_code=code, child_pid=owned[0].pid,
        child_executing_after_cli=TargetProcessValidator.is_pid_alive(owned[0].pid),
        claimed_cleaned=owned[0]._qmd_owned_group.cleaned, sentinel_alive_after_cli=sentinel.poll() is None)
finally:
    os.killpg = real_killpg
    try:
        for child in owned:
            evidence["outer_cleanup"] = terminate_owned_process(child, grace_s=.02)[0]
            evidence["child_executing_after_outer_cleanup"] = TargetProcessValidator.is_pid_alive(child.pid)
            try:
                os.waitpid(child.pid, os.WNOHANG)
                evidence["kernel_reaped"] = False
            except ChildProcessError:
                evidence["kernel_reaped"] = True
        evidence["sentinel_alive_after_owned_cleanup"] = sentinel.poll() is None
    finally:
        if sentinel.poll() is None: sentinel.kill()
        sentinel.wait(timeout=1)
        (root / "standalone-denial-evidence.json").write_text(json.dumps(evidence))
raise SystemExit(code)
'''


def test_standalone_cli_exit_and_json_are_finalized_after_failed_cleanup(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", CLI_DENIAL_RUNNER, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, timeout=12,
    )
    report = json.loads(result.stdout)
    assert result.returncode == report["exit_code"] == 2, result.stderr
    assert report["healthy"] is False and report["breach_type"] == "cleanup_failed"
    assert report["terminated_pid"] is None
    assert report["child_cleanup"]["cleaned"] is False
    assert "standalone owned signal denial" in report["child_cleanup"]["error"]
    assert "disposable child stdout log" in result.stderr
    evidence = json.loads((tmp_path / "standalone-denial-evidence.json").read_text())
    assert evidence["child_pid"] == report["child_cleanup"]["pid"] == report["metrics"]["target_pid"]
    assert evidence["child_executing_after_cli"] and not evidence["claimed_cleaned"]
    assert evidence["outer_cleanup"] and evidence["kernel_reaped"]
    assert not evidence["child_executing_after_outer_cleanup"]
    assert evidence["sentinel_alive_after_cli"] and evidence["sentinel_alive_after_owned_cleanup"]


@pytest.mark.parametrize("event,expected", [(None, None), ((1, 7), 7), ((2, 15), -15), ((3, 9), -9)])
def test_native_waitid_adapter_retains_the_nonconsuming_flags(monkeypatch, event, expected):
    observed = []
    pid = 43210  # Only an argument to a fake syscall; never a signal target.

    def fake_waitid(kind, target, flags):
        observed.append((kind, target, flags))
        return None if event is None else SimpleNamespace(
            si_pid=pid, si_code=event[0], si_status=event[1], si_signo=signal.SIGCHLD,
        )

    with monkeypatch.context() as patch:
        patch.setattr(os, "waitid", fake_waitid, raising=False)
        for name, value in (("CLD_EXITED", 1), ("CLD_KILLED", 2), ("CLD_DUMPED", 3)):
            patch.setattr(os, name, value, raising=False)
        assert process_anchor.wait_child_without_reaping(pid) == expected
    assert observed == [(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)]


def test_platform_without_nonconsuming_wait_is_refused_before_spawn(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(process_anchor.sys, "platform", "unsupported-posix")
        patch.setattr(os, "waitid", None, raising=False)
        patch.setattr(watchdog_module, "_AnchoredPopen", lambda *a, **k: pytest.fail("Spawn on unsupported platform"))
        with pytest.raises(RuntimeError, match="waitid"):
            spawn_owned_process([sys.executable, "-c", "pass"])


def test_application_sigchld_reaper_is_refused_before_spawn(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(signal, "getsignal", lambda sig: signal.SIG_IGN)
        patch.setattr(watchdog_module, "_AnchoredPopen", lambda *a, **k: pytest.fail("Spawn with an automatic child reaper"))
        with pytest.raises(TargetValidationError, match="SIGCHLD"):
            spawn_owned_process([sys.executable, "-c", "pass"])
