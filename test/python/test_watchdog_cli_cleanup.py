"""Public CLI lifecycle regressions using only disposable CPU process trees.

Memory readings are synthetic; CLI parsing, identity checks, HTTP probes,
process spawning, signals, escalation, and reaping run through the real code.
"""

import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts.qmd_mlx.watchdog import (
    SystemMemorySampler,
    TargetProcessValidator,
    TargetValidationError,
    terminate_owned_process,
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="Owned process groups require POSIX")


CHILD_SCRIPT = r'''
import http.server
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import sys
import time

role, directory, mode = sys.argv[1:]
root = Path(directory)
if role == "grandchild":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "grandchild.json").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp()}))
    time.sleep(60)
    raise SystemExit(0)

subprocess.Popen([sys.executable, __file__, "grandchild", directory, mode])
while not (root / "grandchild.json").exists():
    time.sleep(.005)

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        payload = {
            "pid": os.getpid(), "instance_token": os.environ["MLX_INSTANCE_TOKEN"],
            "state": "idle", "ready": True, "worker_alive": True,
            "worker_idle": True, "completed_sequence": 0, "active_job_age_s": None,
        }
        if mode == "pid_mismatch":
            payload["pid"] = -1
        if mode == "token_mismatch":
            payload["instance_token"] = "wrong-fixture-token"
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

port = int(os.environ["MLX_CONTROL_PORT"])
if mode == "health_timeout":
    server = socket.socket()
    server.bind(("127.0.0.1", port))
    server.listen(1)
elif not mode.startswith("leader_exit"):
    # TCPServer runs the real HTTP handler without HTTPServer's unbounded
    # getfqdn() call during binding. This fixture needs no reverse DNS.
    server = socketserver.TCPServer(("127.0.0.1", port), Handler)
(root / "leader.json").write_text(json.dumps({
    "pid": os.getpid(), "pgid": os.getpgrp(),
    "instance_token": os.environ["MLX_INSTANCE_TOKEN"],
    "port": int(os.environ["MLX_EMBED_PORT"]), "control_port": port,
}))
if mode.startswith("leader_exit"):
    while not (root / "exit-leader").exists():
        time.sleep(.005)
    raise SystemExit(0)
if mode == "health_timeout":
    peer, _ = server.accept()
    peer.recv(4096)
    (root / "health-request").write_text("received")
    time.sleep(60)
else:
    server.serve_forever()
'''


def read_json_when_ready(path, timeout_s=3):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
    diagnostics = ""
    log_path = path.with_name("child.log")
    if log_path.exists():
        with log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 4096))
            diagnostics = "\nChild stdout/stderr:\n" + log.read(4096).decode("utf-8", errors="replace")
    raise TimeoutError(f"Disposable CLI process did not publish {path.name}{diagnostics}")


def assert_not_executing(pid):
    end = time.monotonic() + 1
    while time.monotonic() < end:
        if not TargetProcessValidator.is_pid_alive(pid):
            return
        time.sleep(0.01)
    sampler = SystemMemorySampler()
    pytest.fail(
        f"Owned fixture PID {pid} still executes after CLI exit; "
        f"start={sampler.get_process_start_time(pid)!r}, command={sampler.get_process_cmdline(pid)!r}"
    )


def save_process_evidence(tree, child, descendant_pid):
    evidence = {
        "mode": tree.mode, "leader_pid": child.pid, "grandchild_pid": descendant_pid,
        "sentinel_pid": tree.sentinel.pid, "leader_returncode": child.returncode,
        "leader_executing": TargetProcessValidator.is_pid_alive(child.pid),
        "grandchild_executing": TargetProcessValidator.is_pid_alive(descendant_pid),
        "sentinel_executing": tree.sentinel.poll() is None,
    }
    (tree.root / "cleanup-evidence.json").write_text(json.dumps(evidence))


class SyntheticMemorySampler(SystemMemorySampler):
    """Replace memory telemetry, not real PID/start identity or command checks."""

    def __init__(self, tree):
        super().__init__()
        self.tree = tree

    def get_installed_ram_mb(self):
        return 32768.0

    def get_memory_headroom_mb(self):
        return 16000.0

    def get_swap_used_mb(self):
        return 0.0

    def get_memory_free_pct(self):
        return 50.0

    def get_process_start_time(self, pid):
        # Do not trigger a breach or validation failure before the grandchild
        # has installed SIG_IGN and the leader has published its ready barrier.
        ready = read_json_when_ready(self.tree.root / "leader.json")
        assert ready["pid"] == pid
        descendant = read_json_when_ready(self.tree.root / "grandchild.json")
        if self.tree.descendant_start_time is None:
            self.tree.descendant_start_time = super().get_process_start_time(descendant["pid"])
            assert self.tree.descendant_start_time, "Could not pin disposable descendant identity"
        return super().get_process_start_time(pid)

    def get_process_rss_mb(self, pid):
        if self.tree.mode.startswith("leader_exit"):
            (self.tree.root / "exit-leader").write_text("exit normally")
            assert_not_executing(pid)
            return None
        if self.tree.mode == "interrupt":
            raise KeyboardInterrupt
        if self.tree.mode == "exception":
            raise RuntimeError("synthetic CLI sampler exception")
        return 9000.0 if self.tree.mode in ("rss", "rss_dry_run", "external") else 16.0


@pytest.fixture
def cli_tree(tmp_path, monkeypatch, mode):
    cli = importlib.import_module("scripts.qmd-mlx-watchdog")
    # Hold both sockets while choosing distinct ephemeral fixture-only ports.
    with socket.socket() as inference, socket.socket() as control:
        inference.bind(("127.0.0.1", 0))
        control.bind(("127.0.0.1", 0))
        ports = inference.getsockname()[1], control.getsockname()[1]
    script = tmp_path / "owned_watchdog_child.py"
    script.write_text(CHILD_SCRIPT)
    tree = SimpleNamespace(
        root=tmp_path, mode=mode, cli=cli, children=[], ports=ports,
        command=[sys.executable, str(script), "leader", str(tmp_path), mode],
        descendant_start_time=None,
    )
    child_log = (tmp_path / "child.log").open("wb")
    original_popen = subprocess.Popen
    sentinel = original_popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    tree.sentinel = sentinel

    def record_real_spawn(command, *args, **kwargs):
        if command == tree.command:
            kwargs.update(stdout=child_log, stderr=subprocess.STDOUT)
        child = original_popen(command, *args, **kwargs)
        if command == tree.command:
            tree.children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", record_real_spawn)
    original_owned_spawn = cli.spawn_owned_process

    def record_owned_spawn(command, **kwargs):
        if command == tree.command:
            kwargs.update(stdout=child_log, stderr=subprocess.STDOUT)
        child = original_owned_spawn(command, **kwargs)
        if command == tree.command:
            tree.children.append(child)
        return child

    monkeypatch.setattr(cli, "spawn_owned_process", record_owned_spawn)
    monkeypatch.setattr(cli, "SystemMemorySampler", lambda: SyntheticMemorySampler(tree))
    tree.options = [
        "--port", str(ports[0]), "--control-port", str(ports[1]),
        "--max-rss-mb", "100", "--max-swap-growth-mb", "100",
        "--grace-period-s", "0.05", "--check-interval-s", "0.02",
        "--startup-grace-period-s", "0", "--health-timeout-s", "0.15",
        "--consecutive-failures", "1", "--json",
    ]
    try:
        yield tree
    finally:
        # Baseline/failing probes must not leak fixtures. Only exact children
        # and the recorded descendant of this unique script may be signalled.
        try:
            for child in tree.children:
                if getattr(child, "_qmd_owned_group", None) is not None:
                    assert terminate_owned_process(child, grace_s=0.05)[0]
                    with pytest.raises(ChildProcessError):
                        os.waitpid(child.pid, os.WNOHANG)
                else:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=1)
        finally:
            try:
                descendant_path = tmp_path / "grandchild.json"
                if descendant_path.exists():
                    descendant = read_json_when_ready(descendant_path)["pid"]
                    sampler = SystemMemorySampler()
                    command = sampler.get_process_cmdline(descendant)
                    start = sampler.get_process_start_time(descendant)
                    if (start and start == tree.descendant_start_time and command
                            and str(script) in command and "grandchild" in command):
                        try:
                            os.kill(descendant, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    assert_not_executing(descendant)
            finally:
                if sentinel.poll() is None:
                    sentinel.kill()
                sentinel.wait(timeout=1)
                child_log.close()


@pytest.mark.parametrize("mode,expected_code,expected_breach", [
    ("rss", 1, "rss_exceeded"),
    ("rss_dry_run", 1, "rss_exceeded"),
    ("health_timeout", 1, "health_check_failed"),
    ("startup_failure", 2, None),
    ("leader_exit", 0, None),
    ("leader_exit_dry_run", 0, None),
    ("once", 0, None),
    ("interrupt", 0, None),
    ("exception", 2, None),
    ("pid_mismatch", 1, "identity_mismatch"),
    ("token_mismatch", 1, "identity_mismatch"),
])
def test_public_cli_cleans_owned_tree_and_preserves_sentinel(
    cli_tree, mode, expected_code, expected_breach, capsys,
):
    tree = cli_tree
    options = list(tree.options)
    if not mode.startswith("leader_exit"):
        options.append("--once")
    if mode.endswith("dry_run"):
        options.append("--dry-run")
    if mode == "startup_failure":
        options.extend(["--expected-cmd", "^not-the-disposable-python-process$"])

    started = time.monotonic()
    code = tree.cli.main(options + ["--launch"] + tree.command)
    elapsed = time.monotonic() - started
    captured = capsys.readouterr()
    assert code == expected_code, captured.err
    leader = read_json_when_ready(tree.root / "leader.json")
    descendant = read_json_when_ready(tree.root / "grandchild.json")
    assert len(tree.children) == 1
    child = tree.children[0]
    assert child.returncode is not None, "CLI did not reap its direct child"
    with pytest.raises(ChildProcessError):
        os.waitpid(child.pid, os.WNOHANG)
    assert_not_executing(child.pid)
    assert_not_executing(descendant["pid"])
    assert leader["pid"] == child.pid == leader["pgid"] == descendant["pgid"]
    assert tree.sentinel.poll() is None
    assert os.getpgid(tree.sentinel.pid) not in (leader["pgid"], os.getpgrp())
    assert leader["instance_token"]
    assert (leader["port"], leader["control_port"]) == tree.ports

    result = json.loads(captured.out)
    assert result["exit_code"] == expected_code
    assert result["child_cleanup"]["required"] is True
    assert result["child_cleanup"]["cleaned"] is True
    assert result["child_cleanup"]["pid"] == child.pid
    assert result["child_cleanup"]["exit_code"] == child.returncode
    if expected_breach is not None or mode == "once":
        assert result["breach_type"] == expected_breach
        assert result["healthy"] is (mode == "once")
        if mode != "once" and not mode.endswith("dry_run"):
            assert result["terminated_pid"] == child.pid
            assert result["sigkill_used"] is True
        elif mode.endswith("dry_run"):
            assert result["terminated_pid"] is None
            assert result["sigkill_used"] is False
    if mode == "health_timeout":
        assert (tree.root / "health-request").read_text() == "received"
        assert elapsed < 2, f"Bounded HTTP timeout returned after {elapsed:.3f}s"
    if mode == "startup_failure":
        assert "safety validation failed" in captured.err
    if mode == "exception":
        assert "synthetic CLI sampler exception" in captured.err
    if mode.startswith("leader_exit"):
        assert child.returncode == 0

    # Retain PID-specific evidence inside the fresh pytest temporary fixture.
    save_process_evidence(tree, child, descendant["pid"])


@pytest.mark.parametrize("mode", ["external"])
def test_public_cli_external_attach_never_acquires_group_cleanup(cli_tree, monkeypatch, capsys):
    tree = cli_tree
    env = os.environ.copy()
    env.update(
        MLX_INSTANCE_TOKEN="external-disposable-cli-token",
        MLX_EMBED_PORT=str(tree.ports[0]), MLX_CONTROL_PORT=str(tree.ports[1]),
    )
    # This is deliberately a raw Popen handle, NOT spawn_owned_process.
    child = subprocess.Popen(tree.command, env=env, start_new_session=True)
    leader = read_json_when_ready(tree.root / "leader.json")
    descendant = read_json_when_ready(tree.root / "grandchild.json")
    group_signals = []

    def forbid_group_signal(pgid, sig):
        group_signals.append((pgid, sig))
        raise AssertionError("External/raw Popen targets must never receive group signals")

    monkeypatch.setattr(os, "killpg", forbid_group_signal)
    with pytest.raises(TargetValidationError, match="no owned process-group capability"):
        terminate_owned_process(child)
    for gate in ([], ["--allow-external-pid"]):
        assert tree.cli.main(["--pid", str(child.pid)] + gate) == 2
        assert child.poll() is None
    capsys.readouterr()
    code = tree.cli.main(tree.options + [
        "--pid", str(child.pid), "--allow-external-pid",
        "--instance-token", leader["instance_token"], "--once",
    ])
    result = json.loads(capsys.readouterr().out)
    assert code == 1 and result["breach_type"] == "rss_exceeded"
    assert result["terminated_pid"] == child.pid
    assert result["sigkill_used"] is False
    assert child.returncode is None, "CLI tried to reap an externally attached PID"
    child.wait(timeout=1)
    assert_not_executing(child.pid)
    # External attach remains leader-only; it cannot claim ownership of a tree.
    assert TargetProcessValidator.is_pid_alive(descendant["pid"])
    assert tree.sentinel.poll() is None
    assert not group_signals
    save_process_evidence(tree, child, descendant["pid"])


@pytest.mark.parametrize("mode", ["once"])
@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
def test_cli_cleanup_signal_denial_is_nonzero_machine_readable_failure(
    cli_tree, monkeypatch, capsys, denied_signal,
):
    tree = cli_tree
    real_killpg = os.killpg
    denied = []

    def deny_owned_signal(pgid, sig):
        assert tree.children and pgid == tree.children[0].pid
        if sig == denied_signal:
            denied.append(sig)
            raise PermissionError("injected owned cleanup signal denial")
        return real_killpg(pgid, sig)

    # Restore real syscalls before the fixture's finally cleanup. A failed
    # product cleanup must not be mistaken for that outer cleanup.
    with monkeypatch.context() as patch:
        patch.setattr(os, "killpg", deny_owned_signal)
        code = tree.cli.main(tree.options + ["--once", "--launch"] + tree.command)
    captured = capsys.readouterr()
    child = tree.children[0]
    descendant = read_json_when_ready(tree.root / "grandchild.json")["pid"]
    assert denied == [denied_signal]
    assert TargetProcessValidator.is_pid_alive(descendant)
    assert not child._qmd_owned_group.cleaned
    assert tree.sentinel.poll() is None
    result = json.loads(captured.out)
    assert code == result["exit_code"] == 2
    assert result["healthy"] is False
    assert result["breach_type"] == "cleanup_failed"
    assert result["terminated_pid"] is None
    assert result["child_cleanup"]["cleaned"] is False
    assert "signal denial" in result["child_cleanup"]["error"]
    assert "signal denial" in captured.err


@pytest.mark.parametrize("mode", ["leader_exit"])
@pytest.mark.parametrize("returncode,stdout_template,stderr", [
    (1, "", "ps: sysctl: Operation not permitted"),
    (0, "{pid} Z\n", "ps: incomplete process snapshot"),
    (1, "{pid} Z\n", ""),
])
def test_cli_group_readback_failure_retains_anchor_and_fails_closed(
    cli_tree, monkeypatch, capsys, returncode, stdout_template, stderr,
):
    from scripts.qmd_mlx.process_anchor import wait_child_without_reaping

    tree = cli_tree
    real_run = subprocess.run
    observations = []

    def failed_group_readback(command, *args, **kwargs):
        if command[:2] == ["ps", "-g"]:
            assert tree.children and int(command[2]) == tree.children[0].pid
            observations.append(command)
            return SimpleNamespace(
                returncode=returncode,
                stdout=stdout_template.format(pid=tree.children[0].pid), stderr=stderr,
            )
        return real_run(command, *args, **kwargs)

    # Only completion telemetry is injected: process identity, exited leader,
    # TERM-ignoring grandchild, sentinel and final fixture cleanup are real.
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", failed_group_readback)
        code = tree.cli.main(tree.options + ["--launch"] + tree.command)
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    child = tree.children[0]
    descendant = read_json_when_ready(tree.root / "grandchild.json")["pid"]
    assert observations
    assert code == report["exit_code"] == 2
    assert report["healthy"] is False
    assert report["breach_type"] == "cleanup_failed"
    assert report["child_cleanup"]["cleaned"] is False
    assert "Could not inspect owned process group" in report["child_cleanup"]["error"]
    assert not child._qmd_owned_group.cleaned
    assert wait_child_without_reaping(child.pid) == 0, "Unknown readback must retain the anchor"
    assert TargetProcessValidator.is_pid_alive(descendant)
    assert tree.sentinel.poll() is None


@pytest.mark.parametrize("mode", ["once"])
def test_cli_incomplete_cleanup_return_is_also_a_structured_failure(cli_tree, monkeypatch, capsys):
    tree = cli_tree
    with monkeypatch.context() as patch:
        patch.setattr(tree.cli, "terminate_owned_process", lambda *a, **k: (False, False))
        code = tree.cli.main(tree.options + ["--once", "--launch"] + tree.command)
    report = json.loads(capsys.readouterr().out)
    assert code == report["exit_code"] == 2
    assert not report["healthy"] and report["breach_type"] == "cleanup_failed"
    assert not report["child_cleanup"]["cleaned"]
    assert "did not complete" in report["child_cleanup"]["error"]
    assert tree.children[0].poll() is None
    assert not getattr(tree.children[0], "_qmd_owned_group").cleaned
    assert tree.sentinel.poll() is None
