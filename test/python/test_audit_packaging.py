"""Packaging/launchd regressions. All services and paths are disposable fixtures."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_daemon():
    path = ROOT / "scripts/launchd/qmd_mlx_daemon.py"
    spec = importlib.util.spec_from_file_location("audit_packaging_daemon", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture_environment(tmp_path):
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL"}}
    for key, directory in {
        "HOME": "home & <fixture>", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config", "HERMES_HOME": "hermes",
    }.items():
        path = tmp_path / directory
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MLX_PYTHON=sys.executable)
    # Shadow the real command in every subprocess fixture, including failures.
    stub_dir = tmp_path / "fixture-bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "launchctl"
    stub.write_text("#!/bin/sh\nprintf 'unexpected fixture launchctl call\\n' >&2\nexit 97\n")
    stub.chmod(0o755)
    env["PATH"] = str(stub_dir) + os.pathsep + env["PATH"]
    return env


def local_model(tmp_path, name="model & <fixture>"):
    model = tmp_path / name
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture, not real weights")
    (model / "tokenizer.json").write_text("{}")
    return model


def test_real_bin_launcher_matrix():
    result = subprocess.run(["bash", str(ROOT / "test/launcher-detection.test.sh")], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "argv" in result.stdout, "the launcher test must execute the real launcher and assert argument forwarding"


def test_real_bin_launcher_matrix_with_symlinked_temp_root(tmp_path):
    env = fixture_environment(tmp_path)
    alias = tmp_path / "temporary directory alias"
    alias.symlink_to(Path(env["TMPDIR"]), target_is_directory=True)
    env["TMPDIR"] = str(alias)
    result = subprocess.run(["bash", str(ROOT / "test/launcher-detection.test.sh")],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("launcher checks passed")


def test_install_and_pack_build_source_without_automatic_hook_mutation():
    manifest = json.loads((ROOT / "package.json").read_text())
    assert "prepare" in manifest["scripts"] and "prepack" in manifest["scripts"]
    assert "|| true" not in manifest["scripts"]["prepare"]
    assert "install-hooks" not in manifest["scripts"]["prepare"]
    assert manifest["allowScripts"] == {"better-sqlite3": True}
    assert manifest["devDependencies"]["typescript"] == "5.9.3"


def test_retired_patcher_never_touches_installed_home(tmp_path):
    env = fixture_environment(tmp_path)
    target = Path(env["HOME"]) / ".hermes/node/lib/node_modules/@tobilu/qmd/dist/store.js"
    target.parent.mkdir(parents=True)
    target.write_text("immutable fixture")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/apply-fork-dist-patches.py")], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "is retired" in result.stderr.lower()
    assert target.read_text() == "immutable fixture"


def test_plist_structural_render_preserves_paths_and_settings(tmp_path):
    daemon = load_daemon()
    env = fixture_environment(tmp_path)
    env.update(MLX_EMBED_MODEL=str(local_model(tmp_path)), MLX_PORT="19871", MLX_CONTROL_PORT="19879", MLX_EMBED_DTYPE="float16", MLX_MAX_BULK_MICROBATCH_TOKENS="2048", MLX_RESIDENCY_BUDGET_MB="3072")
    config = daemon.configuration(ROOT, env)
    rendered = daemon.render_plist(config, "fixture-instance")
    payload = plistlib.loads(rendered)
    assert payload["WorkingDirectory"] == str(ROOT)
    assert payload["EnvironmentVariables"]["MLX_PORT"] == "19871"
    assert payload["EnvironmentVariables"]["MLX_CONTROL_PORT"] == "19879"
    assert payload["EnvironmentVariables"]["MLX_EMBED_MODEL"] == env["MLX_EMBED_MODEL"]
    assert payload["EnvironmentVariables"]["MLX_PYTHON"] == sys.executable
    assert payload["EnvironmentVariables"]["MLX_EMBED_DTYPE"] == "float16"
    assert payload["EnvironmentVariables"]["MLX_MAX_BULK_MICROBATCH_TOKENS"] == "2048"
    assert payload["EnvironmentVariables"]["MLX_RESIDENCY_BUDGET_MB"] == "3072"
    assert payload["EnvironmentVariables"]["MLX_INSTANCE_TOKEN"] == "fixture-instance"
    assert payload["EnvironmentVariables"]["HF_HUB_OFFLINE"] == "1"
    assert payload["StandardOutPath"] == str(Path(env["XDG_CACHE_HOME"]) / "qmd/mlx-daemon.log")
    assert payload["StandardOutPath"] != "/Users/shersingh/.cache/qmd/mlx-daemon.log"
    assert "__REPO__" not in rendered.decode()


@pytest.mark.parametrize("settings", [{"MLX_PORT": "0"}, {"MLX_PORT": "65536"}, {"MLX_PORT": "8787", "MLX_CONTROL_PORT": "8787"}, {"MLX_START_TIMEOUT": "nan"}, {"MLX_RESIDENCY_BUDGET_MB": "inf"}, {"MLX_EMBED_DTYPE": "invalid"}, {"MLX_MAX_BULK_MICROBATCH_TOKENS": "1.5"}])
def test_daemon_rejects_invalid_settings(tmp_path, settings):
    daemon = load_daemon()
    env = fixture_environment(tmp_path)
    env.update(settings)
    with pytest.raises(daemon.DaemonError):
        daemon.configuration(ROOT, env)


def test_daemon_preflight_missing_model_before_launchctl(tmp_path):
    env = fixture_environment(tmp_path)
    env["MLX_EMBED_MODEL"] = str(tmp_path / "missing-model")
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    stub = stub_dir / "launchctl"
    stub.write_text("#!/bin/sh\nprintf 'unexpected launchctl call\\n' >&2\nexit 97\n")
    stub.chmod(0o755)
    env["PATH"] = str(stub_dir) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", str(ROOT / "scripts/qmd-mlx-daemon.sh"), "start"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "model" in result.stderr.lower() and "missing" in result.stderr.lower()
    assert not (Path(env["HOME"]) / "Library/LaunchAgents/com.qmd.mlxd.plist").exists()


def test_daemon_owned_health_requires_pid_token_and_model(tmp_path):
    daemon = load_daemon()
    env = fixture_environment(tmp_path)
    env["MLX_EMBED_MODEL"] = str(local_model(tmp_path))
    config = daemon.configuration(ROOT, env)
    healthy = dict(ready=True, worker_alive=True, pid=1234, instance_token="expected-token", model=config.env["MLX_EMBED_MODEL"], rerank_model=None, generate_model=None)
    assert daemon.health_matches(healthy, config, "expected-token", 1234)
    for key, wrong in [("pid", 4321), ("instance_token", "foreign-token"), ("model", "/another/model"), ("ready", False), ("worker_alive", False)]:
        assert not daemon.health_matches({**healthy, key: wrong}, config, "expected-token", 1234), key


def test_daemon_preflight_refuses_occupied_ports_without_launchctl(tmp_path, monkeypatch):
    import socket
    daemon = load_daemon()
    env = fixture_environment(tmp_path)
    env["MLX_EMBED_MODEL"] = str(local_model(tmp_path))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        env["MLX_PORT"] = str(listener.getsockname()[1])
        env["MLX_CONTROL_PORT"] = "19879" if env["MLX_PORT"] != "19879" else "19878"
        config = daemon.configuration(ROOT, env)
        monkeypatch.setattr(daemon, "check_python", lambda config: None)
        with pytest.raises(daemon.DaemonError, match="occupied"):
            daemon.check_prerequisites(config)


def test_daemon_run_preserves_actual_argv_and_environment(tmp_path):
    env = fixture_environment(tmp_path)
    model = local_model(tmp_path)
    python_stub = tmp_path / "python with spaces"
    python_stub.write_text("#!" + sys.executable + "\nimport json,os,sys\nif '-c' in sys.argv: sys.exit(0)\nprint(json.dumps({'argv':sys.argv[1:],'token':os.environ.get('MLX_INSTANCE_TOKEN'),'offline':os.environ.get('HF_HUB_OFFLINE')}))\n")
    python_stub.chmod(0o755)
    env.update(MLX_PYTHON=str(python_stub), MLX_EMBED_MODEL=str(model), MLX_PORT="19871", MLX_CONTROL_PORT="19879", MLX_INSTANCE_TOKEN="fixture-token", MLX_MAX_BULK_MICROBATCH_TOKENS="1024")
    result = subprocess.run(["bash", str(ROOT / "scripts/qmd-mlx-daemon.sh"), "run"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["argv"][0] == str(ROOT / "scripts/mlx_embed_server.py")
    assert data["argv"][data["argv"].index("--model") + 1] == str(model)
    assert data["argv"][data["argv"].index("--port") + 1] == "19871"
    assert data["argv"][data["argv"].index("--control-port") + 1] == "19879"
    assert data["argv"][data["argv"].index("--max-bulk-microbatch-tokens") + 1] == "1024"
    assert data["token"] == "fixture-token" and data["offline"] == "1"


def control_fixture(tmp_path):
    env = fixture_environment(tmp_path)
    env.update(MLX_EMBED_MODEL=str(local_model(tmp_path)), MLX_START_TIMEOUT="0.4", MLX_STOP_TIMEOUT="1")
    python_stub = tmp_path / "python-prerequisite-stub"
    python_stub.write_text("#!/bin/sh\nexit 0\n")
    python_stub.chmod(0o755)
    env["MLX_PYTHON"] = str(python_stub)
    env["AUDIT_CONTROL_STATE"] = str(tmp_path / "control-state.json")
    env["AUDIT_CONTROL_LOG"] = str(tmp_path / "control-calls.jsonl")
    env["AUDIT_CONTROL_PID"] = str(os.getpid())
    stub = Path(env["PATH"].split(os.pathsep)[0]) / "launchctl"
    stub.write_text("#!" + sys.executable + "\n" + '''import json,os,sys
from pathlib import Path
state=Path(os.environ['AUDIT_CONTROL_STATE'])
with open(os.environ['AUDIT_CONTROL_LOG'],'a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')
command=sys.argv[1]
if command=='print':
    if not state.exists(): sys.exit(113)
    saved=json.loads(state.read_text())
    print('path = '+saved['plist'])
    print('state = '+os.environ.get('AUDIT_CONTROL_RUNNING','running'))
    print('pid = '+os.environ['AUDIT_CONTROL_PID'])
elif command=='bootstrap':
    if os.environ.get('AUDIT_CONTROL_FAIL')=='1':
        print('fixture bootstrap failure',file=sys.stderr); sys.exit(5)
    state.write_text(json.dumps({'plist':sys.argv[3]}))
elif command=='bootout':
    if state.exists(): state.unlink()
else: sys.exit(97)
''')
    stub.chmod(0o755)
    # Reserve distinct dynamically allocated test ports, never fixed live ports.
    import socket
    with socket.socket() as one, socket.socket() as two:
        one.bind(("127.0.0.1", 0))
        two.bind(("127.0.0.1", 0))
        env["MLX_PORT"] = str(one.getsockname()[1])
        env["MLX_CONTROL_PORT"] = str(two.getsockname()[1])
    return env


def call_control(env, command):
    return subprocess.run(["bash", str(ROOT / "scripts/qmd-mlx-daemon.sh"), command], env=env, capture_output=True, text=True, timeout=10)


def control_calls(env):
    log = Path(env["AUDIT_CONTROL_LOG"])
    return [json.loads(line)[0] for line in log.read_text().splitlines()] if log.exists() else []


def test_actual_start_bootstrap_failure_is_nonzero(tmp_path):
    env = control_fixture(tmp_path)
    env["AUDIT_CONTROL_FAIL"] = "1"
    result = call_control(env, "start")
    assert result.returncode != 0 and "bootstrap failed" in result.stderr
    assert "started " not in result.stdout
    assert control_calls(env) == ["print", "bootstrap"]
    plist = Path(env["HOME"]) / "Library/LaunchAgents/com.qmd.mlxd.plist"
    assert plist.exists()
    assert (Path(env["XDG_CACHE_HOME"]) / "qmd").is_dir()
    with plist.open("rb") as stream:
        payload = plistlib.load(stream)
    assert payload["EnvironmentVariables"]["MLX_PORT"] == env["MLX_PORT"]


def test_actual_start_not_running_never_reports_started_and_cleans_owned_job(tmp_path):
    env = control_fixture(tmp_path)
    env["AUDIT_CONTROL_RUNNING"] = "waiting"
    result = call_control(env, "start")
    assert result.returncode != 0 and "startup failed" in result.stderr
    assert "started " not in result.stdout
    assert "bootstrap" in control_calls(env) and "bootout" in control_calls(env)
    assert not Path(env["AUDIT_CONTROL_STATE"]).exists()


@pytest.mark.parametrize("command", ["start", "stop"])
def test_actual_control_refuses_foreign_registration(tmp_path, command):
    env = control_fixture(tmp_path)
    Path(env["AUDIT_CONTROL_STATE"]).write_text(json.dumps({"plist": str(tmp_path / "foreign.plist")}))
    result = call_control(env, command)
    assert result.returncode != 0 and "not owned" in result.stderr
    assert control_calls(env) == ["print"]
    assert Path(env["AUDIT_CONTROL_STATE"]).exists()


def test_actual_start_rejects_port_owner_without_bootout(tmp_path):
    import socket
    env = control_fixture(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        env["MLX_PORT"] = str(listener.getsockname()[1])
        result = call_control(env, "start")
    assert result.returncode != 0 and "occupied" in result.stderr
    assert control_calls(env) == ["print"]
    assert not (Path(env["HOME"]) / "Library/LaunchAgents/com.qmd.mlxd.plist").exists()


@pytest.mark.parametrize("bad_health", [False, True])
def test_actual_start_status_idempotence_and_stop_with_owned_health(tmp_path, bad_health):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    import time
    env = control_fixture(tmp_path)
    env["MLX_START_TIMEOUT"] = "1" if bad_health else "3"
    state = Path(env["AUDIT_CONTROL_STATE"])
    stop_event = threading.Event()
    servers = []
    threads = []
    served_tokens = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            saved = json.loads(state.read_text())
            with open(saved["plist"], "rb") as stream:
                selected = plistlib.load(stream)["EnvironmentVariables"]
            data: dict[str, object] = dict(ready=True, worker_alive=True, pid=os.getpid(), instance_token=selected["MLX_INSTANCE_TOKEN"], model=selected["MLX_EMBED_MODEL"], rerank_model=None, generate_model=None)
            if bad_health:
                data["instance_token"] = "unrelated-listener-token"
            served_tokens.append(data["instance_token"])
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def serve_after_bootstrap(port):
        while not state.exists() and not stop_event.wait(0.01):
            pass
        if stop_event.is_set():
            return
        server = HTTPServer(("127.0.0.1", port), Handler)
        servers.append(server)
        server.timeout = 0.05
        while not stop_event.is_set():
            server.handle_request()
        server.server_close()

    for port in (int(env["MLX_PORT"]), int(env["MLX_CONTROL_PORT"])):
        thread = threading.Thread(target=serve_after_bootstrap, args=(port,))
        threads.append(thread)
        thread.start()
    try:
        first = call_control(env, "start")
        if bad_health:
            assert first.returncode != 0 and "startup failed" in first.stderr
            # A final deadline may supersede the mismatch wording. Prove the
            # unrelated listener actually answered and was still refused.
            assert "unrelated-listener-token" in served_tokens
            assert "started " not in first.stdout
            assert "bootout" in control_calls(env) and not state.exists()
            return
        assert first.returncode == 0, first.stdout + first.stderr
        assert "started com.qmd.mlxd" in first.stdout
        second = call_control(env, "start")
        assert second.returncode == 0 and "already started" in second.stdout, second.stderr
        assert control_calls(env).count("bootstrap") == 1
        status = call_control(env, "status")
        assert status.returncode == 0 and "ready com.qmd.mlxd" in status.stdout, status.stderr
        changed = call_control({**env, "MLX_EMBED_DTYPE": "float16"}, "start")
        assert changed.returncode != 0 and "configuration differs" in changed.stderr
        stopped = call_control(env, "stop")
        assert stopped.returncode == 0 and "stopped com.qmd.mlxd" in stopped.stdout, stopped.stderr
        assert not state.exists()
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=3)
            assert not thread.is_alive()


def test_build_failure_propagates_and_removes_stale_dist(tmp_path):
    env = fixture_environment(tmp_path)
    snapshot = tmp_path / "build-fixture"
    compiler = snapshot / "node_modules/typescript/bin/tsc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("process.exit(41);\n")
    (snapshot / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/build.mjs", snapshot / "scripts/build.mjs")
    (snapshot / "dist").mkdir()
    (snapshot / "dist/stale.js").write_text("stale")
    result = subprocess.run(["node", str(snapshot / "scripts/build.mjs")], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 41
    assert not (snapshot / "dist").exists()


def test_prepare_packaged_dist_does_not_require_compiler_or_git(tmp_path):
    env = fixture_environment(tmp_path)
    snapshot = tmp_path / "package-fixture"
    (snapshot / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/prepare.mjs", snapshot / "scripts/prepare.mjs")
    for artifact in ("dist/index.js", "dist/index.d.ts", "dist/cli/qmd.js"):
        target = snapshot / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture")
    result = subprocess.run(["node", str(snapshot / "scripts/prepare.mjs")], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    (snapshot / "dist/index.js").unlink()
    result = subprocess.run(["node", str(snapshot / "scripts/prepare.mjs")], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Incomplete package" in result.stderr


def test_constraints_match_current_portable_dependencies():
    import importlib.metadata
    constraints = {}
    for line in (ROOT / "scripts/mlx_server_constraints.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            constraints[name] = version
    for name in ("numpy", "pytest", "requests"):
        assert importlib.metadata.version(name) == constraints[name]


@pytest.mark.parametrize("shape", ["array", "npm12"])
def test_pack_json_parser_supports_old_and_current_npm(tmp_path, shape):
    env = fixture_environment(tmp_path)
    artifact = {"filename": "tobilu-qmd-2.1.0.tgz"}
    value = [artifact] if shape == "array" else {"@tobilu/qmd": artifact}
    fixture = tmp_path / "npm-pack.json"
    fixture.write_text(json.dumps(value))
    result = subprocess.run(["node", str(ROOT / "scripts/pack-result.mjs"), str(fixture)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == artifact["filename"], result.stderr
    fixture.write_text(json.dumps([{**artifact, "filename": "../outside.tgz"}]))
    result = subprocess.run(["node", str(ROOT / "scripts/pack-result.mjs"), str(fixture)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Invalid npm pack artifact filename" in result.stderr


def test_explicit_hook_install_preserves_existing_hook_and_is_idempotent(tmp_path):
    env = fixture_environment(tmp_path)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    checkout = tmp_path / "hook-fixture"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/install-hooks.sh", scripts / "install-hooks.sh")
    (scripts / "pre-push").write_text("#!/bin/sh\nexit 0\n")
    subprocess.run(["git", "init", str(checkout)], env=env, capture_output=True, check=True)
    hook = checkout / ".git/hooks/pre-push"
    hook.write_text("existing user hook")
    result = subprocess.run(["bash", str(scripts / "install-hooks.sh")], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Refusing to replace" in result.stderr
    assert hook.read_text() == "existing user hook"
    hook.unlink()
    for _ in range(2):
        result = subprocess.run(["bash", str(scripts / "install-hooks.sh")], env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert hook.read_bytes() == (scripts / "pre-push").read_bytes()


def test_health_probe_bounds_slow_drip_headers(tmp_path):
    import socket
    import threading
    import time
    daemon = load_daemon()
    stop_event = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(1)

    def drip():
        try:
            client, _ = listener.accept()
            with client:
                client.recv(1024)
                for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n":
                    if stop_event.wait(0.02):
                        return
                    try:
                        client.sendall(bytes([byte]))
                    except OSError:
                        return
        finally:
            listener.close()

    port = listener.getsockname()[1]
    thread = threading.Thread(target=drip)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises((OSError, daemon.DaemonError)):
            daemon.probe(port, 0.15)
        assert time.monotonic() - started < 0.6
    finally:
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_offline_runner_scrubs_environment_and_preserves_failure(tmp_path):
    env = fixture_environment(tmp_path)
    env.update(PYTHONPATH="/unrelated-pollution", QMD_EMBED_MODEL="unrelated-live-model", UNRELATED_SECRET="not-for-child")
    code = "import json,os,sys; print(json.dumps(dict(os.environ))); sys.exit(23)"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run-offline.py"), sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 23
    child = json.loads(result.stdout.splitlines()[-1])
    assert child["HOME"] != env["HOME"] and child["HOME"].startswith(env["TMPDIR"])
    assert child["PYTHONPATH"] == str(ROOT)
    assert child["HF_HUB_OFFLINE"] == "1" and child["QMD_LLAMA_GPU"] == "false"
    assert "QMD_EMBED_MODEL" not in child and "UNRELATED_SECRET" not in child
