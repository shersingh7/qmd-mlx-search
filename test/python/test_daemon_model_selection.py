"""Offline cross-entrypoint model selection; never invoke real launchctl/MLX."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def daemon_module():
    spec = importlib.util.spec_from_file_location("daemon_selection_fixture", ROOT / "scripts/launchd/qmd_mlx_daemon.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def environment(tmp_path):
    env = {"PATH": os.environ["PATH"], "MLX_PYTHON": sys.executable}
    for key in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        path = tmp_path / key.lower()
        path.mkdir()
        env[key] = str(path)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "launchctl"
    stub.write_text("#!/bin/sh\nprintf '%s\\n' \"$1\" >> \"$HOME/control-calls\"\nexit 113\n")
    stub.chmod(0o755)
    env["PATH"] = str(stub_dir) + os.pathsep + env["PATH"]
    return env


def call(env, command):
    return subprocess.run(["bash", str(ROOT / "scripts/qmd-mlx-daemon.sh"), command], env=env,
                          capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("command", ["start", "run"])
@pytest.mark.parametrize("selection", [None, "", "   "])
def test_daemon_requires_explicit_selection_even_with_legacy_cache(tmp_path, command, selection):
    env = environment(tmp_path)
    legacy = Path(env["XDG_CACHE_HOME"]) / "qmd/models/qwen3-embedding-4b-mlx-4bit-affine"
    legacy.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (legacy / name).write_text("fixture")
    if selection is not None:
        env["MLX_EMBED_MODEL"] = selection
    result = call(env, command)
    assert result.returncode != 0
    assert "MLX_EMBED_MODEL" in result.stderr and "explicit" in result.stderr.lower()
    assert "local" in result.stderr.lower() and "cached" in result.stderr.lower()
    assert not (Path(env["HOME"]) / "control-calls").exists()
    assert not (Path(env["HOME"]) / "Library/LaunchAgents/com.qmd.mlxd.plist").exists()


def test_configuration_has_no_implicit_model_and_controls_need_no_model(tmp_path):
    env = environment(tmp_path)
    daemon = daemon_module()
    assert daemon.configuration(ROOT, env).env["MLX_EMBED_MODEL"] == ""
    stopped = call(env, "stop")
    assert stopped.returncode == 0 and "not started" in stopped.stdout
    status = call(env, "status")
    assert status.returncode != 0 and "not running" in status.stderr
    assert (Path(env["HOME"]) / "control-calls").read_text().splitlines() == ["print", "print"]


def python_cli(monkeypatch, model=None, quantization="auto"):
    captured = {}

    class Captured(Exception):
        pass

    def start_server(**kwargs):
        captured.update(kwargs)
        raise Captured

    fake = types.ModuleType("qmd_mlx.server")
    setattr(fake, "start_server", start_server)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "qmd_mlx.server", fake)
    monkeypatch.delenv("MLX_EMBED_MODEL", raising=False)
    for key in tuple(os.environ):
        if key.startswith("MLX_"):
            monkeypatch.delenv(key, raising=False)
    if model is not None:
        monkeypatch.setenv("MLX_EMBED_MODEL", model)
    monkeypatch.setenv("MLX_EMBED_QUANT", quantization)
    monkeypatch.setattr(sys, "argv", [str(ROOT / "scripts/mlx_embed_server.py")])
    with pytest.raises(Captured):
        runpy.run_path(str(ROOT / "scripts/mlx_embed_server.py"), run_name="__main__")
    return captured


def test_interactive_python_cli_and_server_keep_minilm_default(monkeypatch):
    assert python_cli(monkeypatch)["model_name"] == MINILM
    # Inspect the library default without importing the real runtime or MLX.
    tree = ast.parse((ROOT / "scripts/qmd_mlx/server.py").read_text())
    start = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "start_server")
    defaults = dict(zip([arg.arg for arg in start.args.args][-len(start.args.defaults):], start.args.defaults))
    assert ast.literal_eval(defaults["model_name"]) == MINILM


@pytest.mark.parametrize("quantization", ["auto", "none", "4bit", "8bit"])
def test_explicit_cached_minilm_selection_and_quantization_match_cli_run_and_plist(tmp_path, monkeypatch, quantization):
    env = environment(tmp_path)
    cache = tmp_path / "hf-cache"
    commit = "a" * 40
    hub = cache / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = hub / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (hub / "refs").mkdir()
    (hub / "refs/main").write_text(commit)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (snapshot / name).write_text("fixture, not loaded")
    python_stub = tmp_path / "python-stub"
    python_stub.write_text("#!" + sys.executable + "\nimport json,os,sys\nif '-c' in sys.argv: sys.exit(0)\nprint(json.dumps({'argv':sys.argv[1:],'env':dict(os.environ)}))\n")
    python_stub.chmod(0o755)
    env.update(MLX_EMBED_MODEL=MINILM, MLX_EMBED_QUANT=quantization,
               HF_HUB_CACHE=str(cache), MLX_PYTHON=str(python_stub))
    daemon = daemon_module()
    config = daemon.configuration(ROOT, env)
    daemon.select_local_models(config)
    import plistlib
    selected = plistlib.loads(daemon.render_plist(config, "fixture-token"))["EnvironmentVariables"]
    assert selected["MLX_EMBED_MODEL"] == str(snapshot)
    assert selected["MLX_EMBED_QUANT"] == quantization
    result = call(env, "run")
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    argv = actual["argv"]
    assert argv[argv.index("--model") + 1] == str(snapshot)
    assert argv[argv.index("--quantization") + 1] == quantization
    assert actual["env"]["HF_HUB_OFFLINE"] == actual["env"]["TRANSFORMERS_OFFLINE"] == "1"
    direct = python_cli(monkeypatch, str(snapshot), quantization)
    assert direct["model_name"] == str(snapshot)
    assert direct["quantization"] == (None if quantization == "auto" else quantization)
    assert not (Path(env["HOME"]) / "control-calls").exists()


def test_help_explains_required_offline_selection(tmp_path):
    env = environment(tmp_path)
    help_text = call(env, "--help")
    assert help_text.returncode == 0
    assert "MLX_EMBED_MODEL" in help_text.stdout and "required" in help_text.stdout.lower()
    assert "no default" in help_text.stdout.lower() and "downloads are disabled" in help_text.stdout.lower()
