"""CPU-only CI portability contracts; no model assets or Metal execution."""

import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from scripts.qmd_mlx.watchdog import SystemMemorySampler, SystemMetricsError
from test_watchdog_cli_cleanup import CHILD_SCRIPT, read_json_when_ready


MEMINFO = """MemTotal:       16777216 kB
MemFree:          262144 kB
MemAvailable:    4194304 kB
Buffers:          131072 kB
Cached:          524288 kB
SwapTotal:       2097152 kB
SwapFree:        1572864 kB
"""


def linux_sampler(tmp_path, text=MEMINFO):
    path = tmp_path / "meminfo"
    path.write_text(text)

    def forbid_darwin_commands(command, timeout_s):
        raise AssertionError(f"Linux telemetry must not run Darwin commands: {command}")

    return SystemMemorySampler(
        cmd_runner=forbid_darwin_commands, platform_name="linux", proc_meminfo_path=str(path),
    )


def test_linux_memory_uses_measured_memavailable_and_swap(tmp_path):
    sampler = linux_sampler(tmp_path)
    assert sampler.get_installed_ram_mb() == 16384.0
    # MemAvailable already accounts for reclaimable memory; never add cache/free.
    assert sampler.get_memory_headroom_mb() == 4096.0
    assert sampler.get_memory_free_pct() == 25.0
    assert sampler.get_swap_used_mb() == 512.0
    defaults = sampler.compute_conservative_defaults()
    assert defaults.installed_ram_mb == 16384.0
    assert defaults.headroom_mb == 4096.0
    assert defaults.max_rss_mb == 2662.4
    assert defaults.max_swap_growth_mb == 1024.0


def test_linux_zero_available_and_no_swap_are_real_readings(tmp_path):
    sampler = linux_sampler(tmp_path, MEMINFO.replace("4194304 kB", "0 kB").replace("2097152 kB", "0 kB").replace("1572864 kB", "0 kB"))
    assert sampler.get_memory_headroom_mb() == 0.0
    assert sampler.get_memory_free_pct() == 0.0
    assert sampler.get_swap_used_mb() == 0.0
    assert sampler.compute_conservative_defaults().max_rss_mb == 0.0


@pytest.mark.parametrize("text", [
    "", MEMINFO.replace("MemAvailable:    4194304 kB\n", ""),
    MEMINFO.replace("MemTotal:       16777216 kB", "MemTotal:       0 kB"),
    MEMINFO.replace("4194304 kB", "99999999 kB"),
    MEMINFO.replace("4194304 kB", "-1 kB"),
    MEMINFO.replace("4194304 kB", "NaN kB"),
    MEMINFO.replace("4194304 kB", "4194304 MB"),
    MEMINFO.replace("4194304 kB", "4194304 kB trailing"),
    MEMINFO + "MemAvailable: 1 kB\n",
    MEMINFO.replace("1572864 kB", "9999999 kB"),
    MEMINFO.replace("SwapFree:        1572864 kB\n", ""),
    MEMINFO + ("x" * 65536),
], ids=["empty", "missing-available", "zero-total", "available-over-total", "negative", "nan", "wrong-unit", "trailing", "duplicate", "swap-over-total", "missing-swap", "oversized"])
@pytest.mark.parametrize("method", ["get_installed_ram_mb", "get_memory_headroom_mb", "get_memory_free_pct", "get_swap_used_mb"])
def test_linux_malformed_telemetry_fails_closed(tmp_path, text, method):
    sampler = linux_sampler(tmp_path, text)
    with pytest.raises(SystemMetricsError):
        getattr(sampler, method)()


def test_linux_unreadable_meminfo_fails_closed(tmp_path):
    sampler = SystemMemorySampler(platform_name="linux", proc_meminfo_path=str(tmp_path / "missing"))
    with pytest.raises(SystemMetricsError, match="meminfo"):
        sampler.compute_conservative_defaults()


@pytest.mark.parametrize("method", ["get_installed_ram_mb", "get_memory_headroom_mb", "get_memory_free_pct", "get_swap_used_mb"])
def test_unsupported_memory_platform_fails_closed(method):
    sampler = SystemMemorySampler(platform_name="unknown-os")
    with pytest.raises(SystemMetricsError, match="Unsupported"):
        getattr(sampler, method)()


def test_darwin_telemetry_still_uses_darwin_commands():
    readings = {
        ("sysctl", "-n", "hw.memsize"): "17179869184\n",
        ("vm_stat",): "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 1000.\nPages inactive: 2000.\nPages speculative: 500.\nPages purgeable: 50000.\n",
        ("memory_pressure", "-Q"): "System-wide memory free percentage: 25%\n",
        ("sysctl", "vm.swapusage"): "total = 1024.00M used = 256.00M free = 768.00M\n",
    }
    sampler = SystemMemorySampler(cmd_runner=lambda cmd, timeout: (0, readings[tuple(cmd)], ""), platform_name="darwin")
    assert sampler.get_installed_ram_mb() == 16384.0
    assert sampler.get_memory_headroom_mb() == 13.671875
    assert sampler.get_memory_free_pct() == 25.0
    assert sampler.get_swap_used_mb() == 256.0


@pytest.mark.parametrize("output", [
    "not vm_stat", "Pages free: 100.\nPages inactive: 200.\nPages speculative: 10.\n",
    "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.\n",
])
def test_darwin_incomplete_page_telemetry_fails_closed(output):
    sampler = SystemMemorySampler(cmd_runner=lambda cmd, timeout: (0, output, ""), platform_name="darwin")
    with pytest.raises(SystemMetricsError):
        sampler.get_memory_headroom_mb()


def test_process_cmdline_requests_untruncated_ps_output():
    command = "/fixture/" + "long-path/" * 30 + "owned_watchdog_child.py grandchild"

    def ps(command_args, timeout):
        assert command_args == ["ps", "-ww", "-o", "command=", "-p", "12345"]
        return 0, command + "\n", ""

    assert SystemMemorySampler(cmd_runner=ps).get_process_cmdline(12345) == command


@pytest.fixture
def no_reverse_dns(tmp_path):
    hook = tmp_path / "python-startup"
    hook.mkdir()
    # Child stack evidence makes a missed barrier distinguishable from an
    # interpreter/import failure. Nothing relies on host resolver configuration.
    (hook / "sitecustomize.py").write_text('''import socket, traceback

def forbid_reverse_dns(*args, **kwargs):
    traceback.print_stack()
    raise RuntimeError("CI fixture forbids reverse DNS for numeric loopback")

socket.getfqdn = forbid_reverse_dns
''')
    return hook


def test_cli_process_fixture_binds_numeric_loopback_without_dns(tmp_path, no_reverse_dns):
    from scripts.qmd_mlx.watchdog import spawn_owned_process, terminate_owned_process

    script = tmp_path / "owned_watchdog_child.py"
    script.write_text(CHILD_SCRIPT)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(no_reverse_dns), MLX_CONTROL_PORT=str(port), MLX_EMBED_PORT=str(port + 1), MLX_INSTANCE_TOKEN="fixture-token")
    log = tmp_path / "child.stderr"
    with log.open("w+") as output:
        child = spawn_owned_process([sys.executable, str(script), "leader", str(tmp_path), "once"], env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            try:
                read_json_when_ready(tmp_path / "leader.json")
            except TimeoutError:
                output.flush()
                pytest.fail(log.read_text())
            leader = json.loads((tmp_path / "leader.json").read_text())
            assert leader["pid"] == child.pid
        finally:
            assert terminate_owned_process(child, grace_s=0.05)[0]
    assert "reverse DNS" not in log.read_text()


@pytest.mark.parametrize("class_name", ["ThreadedMLXServer", "MLXControlServer"])
def test_server_bind_failure_preserves_error_and_closes_its_socket(class_name):
    import errno
    from scripts.qmd_mlx import server as module

    cls = getattr(module, class_name)
    server = cls.__new__(cls)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        kwargs = {"mlx_server": None} if class_name == "MLXControlServer" else {}
        try:
            with pytest.raises(OSError) as caught:
                cls.__init__(server, occupied.getsockname(), module.http.server.BaseHTTPRequestHandler, **kwargs)
            assert caught.value.errno == errno.EADDRINUSE
            assert server.socket.fileno() == -1
            assert server._stopped_event.is_set()
        finally:
            # Failure-only cleanup; the assertions above must pass without it.
            if getattr(server, "socket", None) is not None:
                server.socket.close()


@pytest.mark.parametrize("module_name,class_name", [
    ("scripts.qmd_mlx.server", "ThreadedMLXServer"),
    ("scripts.qmd_mlx.server", "MLXControlServer"),
])
def test_real_server_bind_avoids_reverse_dns(monkeypatch, module_name, class_name):
    """Real bind path, without constructing adapters, workers or model state."""
    import socketserver

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    monkeypatch.setattr(socket, "getfqdn", lambda *a: pytest.fail("Numeric-loopback binding must not perform reverse DNS"))
    server = cls.__new__(cls)
    # Exercise inherited __init__/server_bind/activation on a real ephemeral
    # socket while deliberately bypassing stage initialization and all models.
    socketserver.TCPServer.__init__(server, ("127.0.0.1", 0), socketserver.BaseRequestHandler, bind_and_activate=False)
    try:
        server.server_bind()
        server.server_activate()
        assert server.server_name == "127.0.0.1"
        assert server.server_port == server.server_address[1] > 0
    finally:
        socketserver.TCPServer.server_close(server)


def test_synthetic_smoke_child_startup_does_not_need_reverse_dns(monkeypatch, no_reverse_dns):
    from scripts.qmd_mlx.smoke import SmokeRunner, SmokeRunnerConfig

    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        port, control_port = first.getsockname()[1], second.getsockname()[1]
    monkeypatch.setenv("PYTHONPATH", str(no_reverse_dns) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    runner = SmokeRunner(SmokeRunnerConfig(use_fake_child=True, port=port, control_port=control_port, timeout_s=30))
    report = runner.run()
    assert report["status"] == "passed"
    assert report["descriptor"]["synthetic"] is True
    assert report["child_cleanup"]["cleaned"] is True
    assert "reverse DNS" not in report["captured_logs"]


def test_python_launcher_path_retains_virtualenv_prefix(tmp_path):
    """Never replace a venv/framework launcher by its canonical target."""
    import venv

    root = tmp_path / "venv with spaces"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    interpreter = root / "bin" / "python"
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "HOME", "TMPDIR", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"}}
    result = subprocess.run([str(interpreter), "-c", "import json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix}))"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["executable"] == str(interpreter)
    assert Path(info["prefix"]) == root


@pytest.mark.parametrize("runner_kind", ["smoke", "stage"])
def test_startup_failure_exposes_bounded_child_stderr_and_cleans_up(monkeypatch, runner_kind):
    from scripts.qmd_mlx.smoke import SmokeRunner, SmokeRunnerConfig
    from scripts.qmd_mlx.supervisor import SmokeStageSupervisor, StageSupervisorConfig
    from scripts.qmd_mlx.watchdog import TargetProcessValidator

    class FixtureMemory(SystemMemorySampler):
        """Synthetic memory only; child lifecycle and PID identity remain real."""

        def get_installed_ram_mb(self):
            return 32768.0

        def get_memory_headroom_mb(self):
            return 16000.0

        def get_swap_used_mb(self):
            return 0.0

        def get_memory_free_pct(self):
            return 50.0

    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        port, control_port = first.getsockname()[1], second.getsockname()[1]
    marker = "disposable startup stderr diagnostic"
    command = [sys.executable, "-c", f"import sys,time; print({marker!r},file=sys.stderr,flush=True); time.sleep(0.5); raise SystemExit(23)"]
    if runner_kind == "smoke":
        real_popen = subprocess.Popen

        def substitute_server(cmd, **kwargs):
            if len(cmd) > 1 and cmd[1].endswith("mlx_embed_server.py"):
                cmd = command
            return real_popen(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", substitute_server)
        runner = SmokeRunner(SmokeRunnerConfig(use_fake_child=True, port=port, control_port=control_port, timeout_s=5), sampler=FixtureMemory())
        with pytest.raises(RuntimeError, match=marker):
            runner.run()
    else:
        runner = SmokeStageSupervisor(StageSupervisorConfig(cmd=command, port=port, control_port=control_port, timeout_s=5), sampler=FixtureMemory())
        with pytest.raises(RuntimeError, match=marker):
            with runner.managed_stage():
                pytest.fail("A failed startup must not yield a ready stage")
    report = runner.last_report
    assert report["status"] == "failed"
    assert marker in report["captured_logs"]
    assert len(report["captured_logs"].encode("utf-8")) <= 8192
    assert report["child_cleanup"]["cleaned"] is True
    assert not TargetProcessValidator.is_pid_alive(report["child_pid"])
