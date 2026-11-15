#!/usr/bin/env python3
"""Safe, offline launchd control. This module never imports MLX or opens a model.

Start is idempotent only for the identical owned configuration. Reconfiguration
requires an explicit stop first; an arbitrary listener/job is never evicted.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import plistlib
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

LABEL = "com.qmd.mlxd"
MAX_PROBE_BYTES = 64 * 1024


class DaemonError(RuntimeError):
    pass


@dataclass(frozen=True)
class Configuration:
    repo: Path
    home: Path
    plist: Path
    log_dir: Path
    env: dict[str, str]
    start_timeout: float
    stop_timeout: float

    @property
    def ports(self) -> tuple[int, int]:
        return int(self.env["MLX_PORT"]), int(self.env["MLX_CONTROL_PORT"])


def integer(env: dict[str, str], key: str, default: int, minimum=1, maximum=65535) -> str:
    value = env.get(key, str(default))
    if not re.fullmatch(r"[0-9]+", value) or not minimum <= int(value) <= maximum:
        raise DaemonError(f"{key} must be an integer in {minimum}..{maximum}")
    return str(int(value))


def finite(env: dict[str, str], key: str, default: float, minimum=0.001, maximum=600.0) -> str:
    value = env.get(key, str(default))
    try:
        number = float(value)
    except ValueError as exc:
        raise DaemonError(f"{key} must be a finite number") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise DaemonError(f"{key} must be finite and in {minimum}..{maximum}")
    return value


def absolute(value: str, key: str) -> Path:
    # Interpret ~ against the explicitly selected HOME, not a second ambient HOME.
    path = Path(value)
    if not path.is_absolute():
        raise DaemonError(f"{key} must be an absolute path: {value}")
    return path


def configuration(repo: Path, source: dict[str, str]) -> Configuration:
    repo = repo.resolve()
    home = absolute(source.get("HOME", ""), "HOME")
    cache = absolute(source.get("XDG_CACHE_HOME", str(home / ".cache")), "XDG_CACHE_HOME")
    config_home = absolute(source.get("XDG_CONFIG_HOME", str(home / ".config")), "XDG_CONFIG_HOME")
    log_dir = absolute(source.get("MLX_LOG_DIR", str(cache / "qmd")), "MLX_LOG_DIR")
    python = absolute(source.get("MLX_PYTHON", str(repo / ".venv/bin/python")), "MLX_PYTHON")
    port = integer(source, "MLX_PORT", int(integer(source, "MLX_EMBED_PORT", 8787)))
    control = integer(source, "MLX_CONTROL_PORT", int(port) + 1)
    if port == control:
        raise DaemonError("MLX_PORT and MLX_CONTROL_PORT must differ")
    selected = {
        "HOME": str(home), "PATH": source.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "XDG_CACHE_HOME": str(cache), "XDG_CONFIG_HOME": str(config_home),
        "MLX_PYTHON": str(python), "MLX_PORT": port, "MLX_EMBED_PORT": port,
        "MLX_CONTROL_PORT": control, "MLX_LOG_DIR": str(log_dir),
        # A persistent offline service requires an explicit model selection,
        # unlike the interactive Python CLI's MiniLM convenience default.
        "MLX_EMBED_MODEL": source.get("MLX_EMBED_MODEL", ""),
        "MLX_RERANK_MODEL": source.get("MLX_RERANK_MODEL", ""),
        "MLX_GENERATE_MODEL": source.get("MLX_GENERATE_MODEL", ""),
        "MLX_EMBED_MAX_LENGTH": integer(source, "MLX_EMBED_MAX_LENGTH", 2048, maximum=1_000_000),
        "MLX_MAX_BATCH_TOKENS": integer(source, "MLX_MAX_BATCH_TOKENS", 0, minimum=0, maximum=100_000_000),
        "MLX_MAX_BULK_MICROBATCH_TOKENS": integer(source, "MLX_MAX_BULK_MICROBATCH_TOKENS", 512, maximum=100_000_000),
        "MLX_EMBED_DTYPE": source.get("MLX_EMBED_DTYPE", "float32"),
        "MLX_EMBED_QUANT": source.get("MLX_EMBED_QUANT", "auto"),
        "MLX_IDLE_UNLOAD_S": finite(source, "MLX_IDLE_UNLOAD_S", 300, minimum=0, maximum=86400 * 365),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1", "QMD_DAEMON_OWNER": str(repo),
    }
    if selected["MLX_EMBED_DTYPE"] not in {"float32", "float16", "bfloat16"}:
        raise DaemonError("MLX_EMBED_DTYPE must be float32, float16 or bfloat16")
    if selected["MLX_EMBED_QUANT"] not in {"auto", "none", "4bit", "8bit"}:
        raise DaemonError("MLX_EMBED_QUANT must be auto, none, 4bit or 8bit; use MLX_EMBED_DTYPE for floating precision")
    if "MLX_RESIDENCY_BUDGET_MB" in source:
        selected["MLX_RESIDENCY_BUDGET_MB"] = finite(source, "MLX_RESIDENCY_BUDGET_MB", 1, maximum=10_000_000)
    # Persist only selected operational settings, never arbitrary shell secrets.
    for key in ("HF_HOME", "HF_HUB_CACHE", "LANG", "LC_ALL"):
        if key in source:
            selected[key] = source[key]
    for key in ("MLX_EMBED_MODEL", "MLX_RERANK_MODEL", "MLX_GENERATE_MODEL"):
        if selected[key].startswith("~/"):
            selected[key] = str(home / selected[key][2:])
    return Configuration(repo, home, home / "Library/LaunchAgents" / (LABEL + ".plist"), log_dir, selected,
                         float(finite(source, "MLX_START_TIMEOUT", 60)), float(finite(source, "MLX_STOP_TIMEOUT", 15)))


def resolve_local_model(value: str, env: dict[str, str]) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_dir() and re.fullmatch(r"[\w.-]+/[\w.-]+", value):
        hf_home = Path(env.get("HF_HOME", str(Path(env["XDG_CACHE_HOME"]) / "huggingface")))
        cache = Path(env.get("HF_HUB_CACHE", str(hf_home / "hub")))
        hub = cache / ("models--" + value.replace("/", "--"))
        revision = hub / "refs/main"
        if revision.is_file():
            commit = revision.read_text().strip()
            if re.fullmatch(r"[a-fA-F0-9]{40,64}", commit):
                path = hub / "snapshots" / commit
    if not path.is_dir() or not (path / "config.json").is_file():
        raise DaemonError(f"model missing or not cached: {value}. Select a local model directory; download explicitly before starting.")
    if not any(candidate.is_file() for pattern in ("*.safetensors", "*.npz") for candidate in path.glob(pattern)):
        raise DaemonError(f"model weights missing: {path}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.txt", "sentencepiece.bpe.model")):
        raise DaemonError(f"model tokenizer missing: {path}")
    return str(path.resolve())


def select_local_models(config: Configuration) -> None:
    if not config.env["MLX_EMBED_MODEL"].strip():
        raise DaemonError("MLX_EMBED_MODEL is required: explicitly select a local model directory or cached Hugging Face model ID before start/run. There is no default; download models explicitly first (downloads are disabled).")
    for key in ("MLX_EMBED_MODEL", "MLX_RERANK_MODEL", "MLX_GENERATE_MODEL"):
        config.env[key] = resolve_local_model(config.env[key], config.env)


def check_python(config: Configuration) -> None:
    python = config.env["MLX_PYTHON"]
    if not os.access(python, os.X_OK) or not Path(python).is_file():
        raise DaemonError(f"Python missing or not executable: {python}; create the venv or set MLX_PYTHON")
    # find_spec checks installation only: no MLX import, allocation or network I/O.
    code = "import importlib.util,sys; names=['numpy','mlx','mlx_lm','mlx_embedding_models','transformers']; missing=[n for n in names if importlib.util.find_spec(n) is None]; print(', '.join(missing)); sys.exit(bool(missing))"
    result = subprocess.run([python, "-c", code], env=config.env, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise DaemonError("Python dependencies missing: " + (result.stdout.strip() or result.stderr.strip()) + "; install scripts/mlx_server_requirements.txt in the selected venv")


def check_prerequisites(config: Configuration) -> None:
    if not (config.repo / "scripts/mlx_embed_server.py").is_file():
        raise DaemonError("Missing packaged scripts/mlx_embed_server.py")
    select_local_models(config)
    check_python(config)
    # Keep both sockets alive for the whole check. This is a preflight, not a
    # guarantee against an external process binding between check and bootstrap.
    sockets = []
    try:
        for port in config.ports:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(listener)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as exc:
                raise DaemonError(f"port {port} is occupied or unavailable; refusing to stop its owner") from exc
    finally:
        for listener in sockets:
            listener.close()


def render_plist(config: Configuration, token: str) -> bytes:
    with (config.repo / "scripts/launchd/com.qmd.mlxd.plist").open("rb") as stream:
        payload = plistlib.load(stream)
    payload.update(Label=LABEL, ProgramArguments=[str(config.repo / "scripts/qmd-mlx-daemon.sh"), "run"],
                   WorkingDirectory=str(config.repo), EnvironmentVariables={**config.env, "MLX_INSTANCE_TOKEN": token},
                   StandardOutPath=str(config.log_dir / "mlx-daemon.log"), StandardErrorPath=str(config.log_dir / "mlx-daemon.log"))
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def write_plist(config: Configuration, token: str) -> None:
    config.plist.parent.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + LABEL + "-", suffix=".plist", dir=config.plist.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(render_plist(config, token))
        os.replace(temporary, config.plist)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Launchctl:
    def __init__(self):
        self.binary = shutil.which("launchctl")
        if not self.binary:
            raise DaemonError("launchctl is required (daemon start/stop/status are macOS only)")
        self.job_path: str | None = None
        self.domain = f"gui/{os.getuid()}"
        self.target = self.domain + "/" + LABEL

    def call(self, *args: str) -> subprocess.CompletedProcess:
        assert self.binary is not None
        return subprocess.run([self.binary, *args], capture_output=True, text=True, timeout=15)

    def state(self) -> tuple[bool, int | None]:
        result = self.call("print", self.target)
        if result.returncode:
            # 113 is launchctl's missing-service status. A real control failure
            # must not be mistaken for absence and followed by a bootstrap.
            if result.returncode not in (3, 113):
                raise DaemonError("launchctl print failed: " + result.stderr.strip())
            return False, None
        path = re.search(r"^\s*path = (.+)\s*$", result.stdout, re.MULTILINE)
        self.job_path = path.group(1).strip() if path else None
        match = re.search(r"^\s*pid = (\d+)\s*$", result.stdout, re.MULTILINE)
        running = bool(re.search(r"^\s*state = running\s*$", result.stdout, re.MULTILINE))
        return True, int(match.group(1)) if match and running else None


def owned_plist(config: Configuration, launchctl: Launchctl | None = None) -> dict:
    if launchctl is not None and launchctl.job_path != str(config.plist):
        raise DaemonError("Registered job path is not owned by this checkout; refusing to control it")
    try:
        with config.plist.open("rb") as stream:
            payload = plistlib.load(stream)
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        raise DaemonError("Cannot verify owned launchd plist; refusing to control this job") from exc
    env = payload.get("EnvironmentVariables", {})
    if payload.get("Label") != LABEL or env.get("QMD_DAEMON_OWNER") != str(config.repo) or not env.get("MLX_INSTANCE_TOKEN"):
        raise DaemonError("Existing job/plist is not owned by this checkout; refusing to replace or stop it")
    return payload


def health_matches(data: dict, config: Configuration, token: str, pid: int) -> bool:
    return (data.get("ready") is True and data.get("worker_alive") is True
            and type(data.get("pid")) is int and data["pid"] == pid
            and data.get("instance_token") == token
            and data.get("model") == config.env["MLX_EMBED_MODEL"]
            and (data.get("rerank_model") or "") == config.env["MLX_RERANK_MODEL"]
            and (data.get("generate_model") or "") == config.env["MLX_GENERATE_MODEL"])


def probe(port: int, timeout: float) -> dict:
    # Direct sockets ignore proxies. Absolute deadlines and header/body byte
    # limits apply even to a peer slowly dripping headers or response bytes.
    deadline = time.monotonic() + timeout
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as connection:
        connection.sendall(f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode("ascii"))
        response = bytearray()
        header_end = -1
        content_length = None
        while content_length is None or len(response) < header_end + 4 + content_length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DaemonError("health probe deadline exceeded")
            connection.settimeout(remaining)
            data = connection.recv(4096)
            if not data:
                raise DaemonError("health response truncated")
            response.extend(data)
            if len(response) > MAX_PROBE_BYTES + 8192:
                raise DaemonError("health response exceeds byte limit")
            if header_end < 0:
                header_end = response.find(b"\r\n\r\n")
                if header_end < 0:
                    if len(response) > 8192:
                        raise DaemonError("health headers exceed byte limit")
                    continue
                if header_end > 8192:
                    raise DaemonError("health headers exceed byte limit")
                headers = response[:header_end].decode("ascii").split("\r\n")
                if not re.fullmatch(r"HTTP/1\.[01] 200(?: .*)?", headers[0]):
                    raise DaemonError("health endpoint did not return HTTP 200")
                lengths = [line.split(":", 1)[1].strip() for line in headers[1:] if line.lower().startswith("content-length:")]
                if len(lengths) != 1 or not lengths[0].isdigit() or not 0 < int(lengths[0]) <= MAX_PROBE_BYTES:
                    raise DaemonError("health requires one bounded Content-Length")
                content_length = int(lengths[0])
        result = json.loads(response[header_end + 4:header_end + 4 + content_length])
        if not isinstance(result, dict):
            raise DaemonError("health must be an object")
        return result


def wait_ready(config: Configuration, launchctl: Launchctl, token: str) -> int:
    deadline = time.monotonic() + config.start_timeout
    error = "job is not running"
    while time.monotonic() < deadline:
        present, pid = launchctl.state()
        if not present:
            raise DaemonError("launchd job disappeared before readiness")
        if owned_plist(config, launchctl)["EnvironmentVariables"].get("MLX_INSTANCE_TOKEN") != token:
            raise DaemonError("launchd instance changed before readiness")
        if pid is not None:
            try:
                # Verify both listeners belong to this PID/token, not merely
                # that /ready on an unrelated listener returns a 200.
                if all(health_matches(probe(port, min(1.0, max(0.001, deadline - time.monotonic()))), config, token, pid)
                       for port in config.ports):
                    return pid
                error = "health identity/readiness mismatch"
            except (OSError, ValueError, DaemonError) as exc:
                error = str(exc)
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise DaemonError(f"startup failed: {error}; check {config.log_dir / 'mlx-daemon.log'}")


def start(config: Configuration, launchctl: Launchctl) -> None:
    present, _ = launchctl.state()
    if present:
        payload = owned_plist(config, launchctl)
        saved = payload["EnvironmentVariables"]
        # Canonicalize selected model paths the same way as first start.
        select_local_models(config)
        if {key: saved.get(key) for key in config.env} != config.env:
            raise DaemonError("job configuration differs; explicitly stop it before reconfiguring")
        pid = wait_ready(config, launchctl, saved["MLX_INSTANCE_TOKEN"])
        print(f"already started {LABEL} (pid {pid}, ports {config.ports[0]}/{config.ports[1]})")
        return
    # Complete validation before touching any launchd/plist state.
    check_prerequisites(config)
    if config.plist.exists():
        owned_plist(config)
    token = uuid.uuid4().hex
    write_plist(config, token)
    result = launchctl.call("bootstrap", launchctl.domain, str(config.plist))
    if result.returncode:
        raise DaemonError("launchd bootstrap failed: " + (result.stderr.strip() or result.stdout.strip()))
    try:
        pid = wait_ready(config, launchctl, token)
    except BaseException as startup_error:
        # Re-read exact job ownership before rolling back. Never remove a job
        # whose registration or instance changed during our readiness wait.
        try:
            if launchctl.state()[0]:
                saved = owned_plist(config, launchctl)["EnvironmentVariables"]
                if saved.get("MLX_INSTANCE_TOKEN") != token:
                    raise DaemonError("instance changed; refusing cleanup")
                stop(config, launchctl)
        except (DaemonError, OSError, subprocess.SubprocessError) as cleanup_error:
            raise DaemonError(f"{startup_error}; rollback failed: {cleanup_error}") from startup_error
        raise
    print(f"started {LABEL} (pid {pid}, ports {config.ports[0]}/{config.ports[1]})")


def stop(config: Configuration, launchctl: Launchctl) -> None:
    present, _ = launchctl.state()
    if not present:
        print(f"not started: {LABEL}")
        return
    owned_plist(config, launchctl)
    result = launchctl.call("bootout", launchctl.target)
    if result.returncode:
        raise DaemonError("launchd bootout failed: " + result.stderr.strip())
    deadline = time.monotonic() + config.stop_timeout
    while time.monotonic() < deadline:
        if not launchctl.state()[0]:
            print(f"stopped {LABEL}")
            return
        time.sleep(0.1)
    raise DaemonError("launchd job did not stop before deadline")


def run(config: Configuration) -> None:
    # launchd owns the foreground process; no daemonization or shell evaluation.
    # Bind races are left to the server to reject, not fixed by killing listeners.
    select_local_models(config)
    check_python(config)
    args = [config.env["MLX_PYTHON"], str(config.repo / "scripts/mlx_embed_server.py"),
            "--model", config.env["MLX_EMBED_MODEL"], "--rerank-model", config.env["MLX_RERANK_MODEL"],
            "--generate-model", config.env["MLX_GENERATE_MODEL"], "--port", config.env["MLX_PORT"],
            "--control-port", config.env["MLX_CONTROL_PORT"], "--host", "127.0.0.1", "--dtype", config.env["MLX_EMBED_DTYPE"],
            "--quantization", config.env["MLX_EMBED_QUANT"], "--max-length", config.env["MLX_EMBED_MAX_LENGTH"],
            "--max-batch-tokens", config.env["MLX_MAX_BATCH_TOKENS"],
            "--max-bulk-microbatch-tokens", config.env["MLX_MAX_BULK_MICROBATCH_TOKENS"]]
    env = {**config.env, "MLX_INSTANCE_TOKEN": os.environ.get("MLX_INSTANCE_TOKEN") or uuid.uuid4().hex}
    os.execve(args[0], args, env)


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["--help"], ["-h"]):
        print("usage: qmd-mlx-daemon.sh {run|start|stop|status}\nMLX_EMBED_MODEL is required for start/run (no default): explicitly select a local model directory or cached Hugging Face model ID. Models must exist locally/cached; downloads are disabled.\nOptional: MLX_PYTHON, MLX_RERANK_MODEL, MLX_GENERATE_MODEL, MLX_PORT, MLX_CONTROL_PORT, MLX_LOG_DIR, MLX_START_TIMEOUT.")
        return 0
    if len(arguments) > 1 or (arguments and arguments[0] not in {"run", "start", "stop", "status"}):
        raise DaemonError("usage: qmd-mlx-daemon.sh {run|start|stop|status}")
    command = arguments[0] if arguments else "status"
    config = configuration(Path(__file__).resolve().parents[2], dict(os.environ))
    if command == "run":
        run(config)
        return 0
    # Validate models before even discovering launchctl for a new start.
    if command == "start":
        select_local_models(config)
    launchctl = Launchctl()
    if command == "status":
        present, pid = launchctl.state()
        if not present or not pid:
            raise DaemonError(f"not running: {LABEL}")
        saved = owned_plist(config, launchctl)["EnvironmentVariables"]
        active = configuration(config.repo, saved)
        if not all(health_matches(probe(port, 2), active, saved["MLX_INSTANCE_TOKEN"], pid) for port in active.ports):
            raise DaemonError("job is running but its health identity/readiness does not match")
        print(f"ready {LABEL} (pid {pid}, ports {active.ports[0]}/{active.ports[1]})")
        return 0
    config.plist.parent.mkdir(parents=True, exist_ok=True)
    # Serialize only our launcher operations; never lock the user's whole HOME.
    lock_path = config.plist.parent / ("." + LABEL + ".lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DaemonError("another daemon start/stop is in progress") from exc
        if command == "start":
            start(config, launchctl)
        else:
            stop(config, launchctl)
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DaemonError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
