"""CPU-only qualification regressions; synthetic receipts are not model/GPU evidence."""

import copy
import importlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time
from types import SimpleNamespace
from urllib.parse import quote, urlparse

import pytest

from scripts.qmd_mlx import phase5_e2e_qualification as phase5
from scripts.qmd_mlx import sustained
from scripts.qmd_mlx.watchdog import WatchdogCheckResult
from test_audit_qualification import complete_report
from test_phase5_qualification import phase5_composition  # noqa: F401 — shared offline fixture


STAGES = ("initial", "post_warmup", "final")
VALID_MEMORY = {"active_mb": 100.0, "peak_mb": 150.0}
BAD_MEMORY_RESPONSES = [
    pytest.param(503, {"error": "held-out telemetry unavailable"}, id="503-error"),
    pytest.param(503, VALID_MEMORY, id="503-numbers"),
    pytest.param(503, {**VALID_MEMORY, "error": "held-out telemetry unavailable"}, id="503-numbers-error"),
    pytest.param(200, {**VALID_MEMORY, "error": "held-out telemetry unavailable"}, id="200-numbers-error"),
    pytest.param(200, {**VALID_MEMORY, "error": ""}, id="200-empty-error"),
    pytest.param(200, {**VALID_MEMORY, "error": False}, id="200-false-error"),
    pytest.param(200, None, id="null"),
    pytest.param(200, [], id="array"),
    pytest.param(200, {}, id="empty-object"),
    pytest.param(200, "malformed", id="string"),
    pytest.param(200, {"active_mb": None, "peak_mb": 150.0}, id="null-number"),
    pytest.param(200, {"active_mb": True, "peak_mb": 150.0}, id="bool-number"),
    pytest.param(200, {"active_mb": "100", "peak_mb": 150.0}, id="string-number"),
    pytest.param(200, {"active_mb": float("nan"), "peak_mb": 150.0}, id="nan"),
    pytest.param(200, {"active_mb": 100.0, "peak_mb": float("inf")}, id="infinite"),
    pytest.param(200, {"active_mb": 100.0, "peak_mb": 99.0}, id="peak-below-active"),
]


@pytest.fixture
def sustained_http(monkeypatch, tmp_path):
    """Exercise the actual sustained runner/CLI, mocking only its external services."""
    cli = importlib.import_module("scripts.qmd-mlx-sustained")
    state = SimpleNamespace(responses={stage: (200, copy.deepcopy(VALID_MEMORY)) for stage in STAGES},
                            memory_calls=[], children=[], clients=[], log_paths=[], raise_at=None)
    counts = {}
    for texts, start in [(sustained.SHORT_FIXTURES, 13), (sustained.MEDIUM_FIXTURES, 100),
                         (sustained.LONG_FIXTURES, 800), (sustained.CODE_FIXTURES, 150)]:
        counts.update((text, start + index) for index, text in enumerate(texts))
    bulk_inflight = threading.Event()
    query_finished = threading.Event()
    concurrent_texts = [sustained.LONG_FIXTURES[0], sustained.MEDIUM_FIXTURES[0],
                        sustained.CODE_FIXTURES[0], sustained.SHORT_FIXTURES[0]]

    class Child:
        pid = 4321
        returncode = None
        terminated = False
        waited = False

        def __init__(self, command, **kwargs):
            state.instance_token = kwargs["env"]["MLX_INSTANCE_TOKEN"]
            state.children.append(self)
            state.log_paths.append(kwargs["stdout"].name)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        kill = terminate

        def wait(self, **kwargs):
            self.waited = True
            return self.returncode

    class Client:
        closed = False

        def __init__(self, **kwargs):
            state.clients.append(self)

        def get_json(self, url, **kwargs):
            if url.endswith("/health"):
                return 200, {"ready": True, "instance_token": state.instance_token}
            if url.endswith("/descriptor"):
                return 200, {"outputDimensions": 4}
            assert url.endswith("/memory")
            stage = STAGES[len(state.memory_calls)]
            state.memory_calls.append(stage)
            if state.raise_at == stage:
                raise TimeoutError("fixture memory request timed out")
            return copy.deepcopy(state.responses[stage])

        def post_json(self, url, payload, **kwargs):
            texts = payload["texts"]
            lengths = [counts.get(text, len(text.split())) for text in texts]
            if max(lengths) > 2048:
                return 400, {"error": "Text token length (2049) exceeds max_length (2048)"}
            if url.endswith("/tokenize"):
                return 200, {"counts": lengths}
            assert url.endswith("/embed")
            if texts == concurrent_texts:
                query_finished.clear()
                bulk_inflight.set()
                assert query_finished.wait(timeout=5.0), "fixture query never entered"
                bulk_inflight.clear()
            elif bulk_inflight.is_set():
                time.sleep(0.002)
                query_finished.set()
            else:
                time.sleep(0.002)
            return 200, {"embeddings": [[1.0, 0.0, 0.0, 0.0] for _ in texts]}

        def close(self):
            self.closed = True

    class Watchdog:
        def __init__(self, **kwargs):
            pass

        def check_step(self):
            return WatchdogCheckResult(timestamp=time.time(), healthy=True, metrics={
                "rss_mb": 1000.0, "swap_used_mb": 100.0,
                "swap_growth_mb": 0.0, "memory_free_pct": 75.0,
            })

    monkeypatch.setattr(sustained.subprocess, "Popen", Child)
    monkeypatch.setattr(sustained, "SmokeHttpClient", Client)
    monkeypatch.setattr(sustained, "MLXWatchdog", Watchdog)
    monkeypatch.setattr(sustained.SustainedRunner, "validate_preflight", lambda _: complete_report()["preflight"])

    def run():
        path = tmp_path / "sustained.json"
        code = cli.main(["--real-model", "--model-path", "/fixture/model", "--timeout-s", "30",
                         "--max-requests", "25", "--output-file", str(path), "--json"])
        report = json.loads(path.read_text())
        assert report["child_cleanup"]["cleaned"] is True
        assert report["supervisor_cleanup"]["stopped"] is True
        assert all(child.terminated and child.waited for child in state.children)
        assert all(client.closed for client in state.clients)
        assert all(not Path(log).exists() for log in state.log_paths)
        return code, report

    state.run = run
    return state


def test_sustained_all_200_memory_snapshots_qualify_and_cli_exits_zero(sustained_http):
    code, report = sustained_http.run()
    assert code == 0
    assert report["status"] == "passed", report["qualification_evaluation"]
    assert sustained_http.memory_calls == list(STAGES)
    memory = report["metal_memory_telemetry"]
    assert memory["http_status"] == dict.fromkeys(STAGES, 200)
    assert memory["telemetry_complete"] is True
    assert memory["provisional_stability_assessment"]["verdict"] == "provisional_stable"
    assert sustained.evaluate_pilot_acceptance(report, sustained.SustainedRunnerConfig(model_path="/fixture/model"))["overall_status"] == "passed"


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("status,payload", BAD_MEMORY_RESPONSES)
def test_sustained_memory_failure_at_each_stage_aborts_and_cli_exits_nonzero(sustained_http, stage, status, payload):
    sustained_http.responses[stage] = (status, payload)
    code, report = sustained_http.run()
    assert code == 1
    assert report["status"] == "failed"
    assert stage in report["error"] and "/memory" in report["error"]
    assert sustained_http.memory_calls == list(STAGES[:STAGES.index(stage) + 1])
    memory = report["metal_memory_telemetry"]
    assert memory["http_status"][stage] == status
    # Failed payloads (including numeric/error bodies) remain forensic raw data.
    assert json.dumps(memory[stage], sort_keys=True) == json.dumps(payload, sort_keys=True)
    assert memory["telemetry_complete"] is False
    assert memory["delta_active_mb"] is None
    assert memory["delta_post_warmup_active_mb"] is None
    assert memory["peak_active_mb"] is None
    assert memory["provisional_stability_assessment"]["verdict"] == "unavailable"
    assert report["qualification_evaluation"]["overall_status"] == "failed"
    assert report["qualification_evaluation"]["gates"]["memory_stability"]["passed"] is False
    assert report["qualification_evaluation"]["gates"]["lifecycle_cleanup"]["passed"] is True
    assert report["metrics_summary"]["partial_run"] is True


@pytest.mark.parametrize("stage", STAGES)
def test_sustained_memory_request_failure_has_no_status_or_fabricated_metrics(sustained_http, stage):
    sustained_http.raise_at = stage
    code, report = sustained_http.run()
    assert code == 1 and report["status"] == "failed"
    memory = report["metal_memory_telemetry"]
    assert memory["http_status"][stage] is None and memory[stage] is None
    assert memory["telemetry_complete"] is False and memory["peak_active_mb"] is None
    assert stage in report["error"] and "timed out" in report["error"]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("status", [None, 503, 200.0, True, "200"])
def test_memory_assessment_requires_explicit_success_for_every_snapshot(stage, status):
    report = complete_report()
    memory = report["metal_memory_telemetry"]
    memory["http_status"] = dict.fromkeys(STAGES, 200)
    if status is None:
        del memory["http_status"][stage]
    else:
        memory["http_status"][stage] = status
    result = sustained.compute_memory_assessment(memory, report["telemetry_samples"], 100.0)
    assert result["telemetry_complete"] is False
    assert result["delta_active_mb"] is None
    assert result["provisional_stability_assessment"]["verdict"] == "unavailable"
    assert sustained.evaluate_qualification_gates(report, sustained.SustainedRunnerConfig())["gates"]["memory_stability"]["passed"] is False


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("error", ["unavailable", "", False, 0, [], {}])
def test_memory_assessment_rejects_error_bearing_numeric_payloads(stage, error):
    report = complete_report()
    memory = report["metal_memory_telemetry"]
    memory["http_status"] = dict.fromkeys(STAGES, 200)
    memory[stage]["error"] = error
    result = sustained.compute_memory_assessment(memory, report["telemetry_samples"], 100.0)
    assert result["telemetry_complete"] is False
    assert result["delta_active_mb"] is None
    assert result["provisional_stability_assessment"]["verdict"] == "unavailable"


@pytest.mark.parametrize("error", ["unavailable", "", False])
def test_phase5_refuses_error_bearing_numeric_memory_payload(phase5_composition, error):
    phase5_composition.responses["/memory"][1]["error"] = error
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any("/memory" in message for message in report["errors"])


def test_phase5_canonical_command_uses_absolute_public_tsx_loader_on_node(phase5_composition):
    report = phase5_composition.run()
    assert report["status"] == "passed", report["errors"]
    command = phase5_composition.captured["cmd"]
    assert Path(command[0]).name == "node"
    assert command[1] == "--import"
    assert urlparse(command[2]).scheme == "file" and urlparse(command[2]).path.startswith("/")
    assert Path(command[3]).name == "phase5_e2e_runner.ts"
    assert phase5_composition.captured["cwd"] == phase5.repo_root


@pytest.mark.parametrize("version", [None, "20.19.0", "21.7.3", "garbled", "22"])
def test_phase5_unsupported_node_fails_closed_without_bun_fallback(phase5_composition, monkeypatch, version):
    def probe(command, **kwargs):
        assert Path(command[0]).name == "node"
        assert "NODE_OPTIONS" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, json.dumps({"node": version, "bun": None,
            "loader": str(Path(phase5.repo_root) / "node_modules/tsx/dist/loader.mjs")}), "")

    monkeypatch.setattr(phase5.subprocess, "run", probe)
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any("Node.js >=22" in message for message in report["errors"])
    assert "cmd" not in phase5_composition.captured
    assert phase5_composition.captured.get("stage_entries", 0) == 0
    assert phase5_composition.captured["health_calls"] == 0
    assert "supervision" not in report


def test_phase5_missing_node_fails_closed_without_trying_bun(phase5_composition, monkeypatch):
    calls = []

    def which(name, **kwargs):
        calls.append(name)
        return None

    monkeypatch.setattr(phase5.shutil, "which", which)
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert calls == ["node"]
    assert any("Node.js >=22" in message for message in report["errors"])
    assert "cmd" not in phase5_composition.captured
    assert phase5_composition.captured.get("stage_entries", 0) == 0
    assert phase5_composition.captured["health_calls"] == 0
    assert "supervision" not in report


def test_phase5_missing_local_tsx_fails_without_install_or_bun(phase5_composition, tmp_path, monkeypatch):
    empty_repo = tmp_path / "no-development-dependencies"
    empty_repo.mkdir()
    monkeypatch.setattr(phase5, "repo_root", str(empty_repo))
    monkeypatch.setattr(phase5.subprocess, "run", lambda *a, **k: pytest.fail("must not install or run a fallback"))
    report = phase5_composition.run()
    assert report["status"] == "failed"
    assert any("repository-local tsx development dependency is missing" in message for message in report["errors"])
    assert "cmd" not in phase5_composition.captured
    assert phase5_composition.captured.get("stage_entries", 0) == 0
    assert phase5_composition.captured["health_calls"] == 0
    assert "supervision" not in report
    assert not list(tmp_path.glob("qmd-phase5-ts-*"))


def test_phase5_full_node_recovery_restore_and_http_on_public_cpu_fixture(phase5_composition, tmp_path, monkeypatch):
    """Real canonical Node runner/SQLite/recovery/MCP; authored Python stage and one-hot HTTP fixture.

    These known-fixture vectors intentionally encode the public judgments. They
    prove orchestration, NOT held-out retrieval quality, model speed or GPU safety.
    """
    from scripts.qmd_mlx.supervisor import run_subordinate_process

    root = tmp_path / ("long managed source qualification " + "x" * 70)
    root.mkdir()
    for name in ("home", "tmp", "cache", "config", "hermes", "fixture-model", "unrelated-cwd"):
        (root / name).mkdir()
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL"}}
    env.update({"HOME": str(root / "home"), "TMPDIR": str(root / "tmp"), "TMP": str(root / "tmp"),
                "TEMP": str(root / "tmp"), "XDG_CACHE_HOME": str(root / "cache"),
                "XDG_CONFIG_HOME": str(root / "config"), "HERMES_HOME": str(root / "hermes"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "TSX_DISABLE_CACHE": "1"})
    public_fixtures = Path(phase5.repo_root, "test/fixtures/public_eval_dataset.ts")
    command = phase5.node_typescript_command(str(public_fixtures), deadline=time.monotonic() + 10, env=env)
    fixtures = subprocess.run([*command[:3], "--input-type=module", "-e",
        f"import {{PUBLIC_CORPUS_DOCS, PUBLIC_HELD_OUT_QUERIES}} from {json.dumps(public_fixtures.as_uri())};"
        "process.stdout.write(JSON.stringify({docs:PUBLIC_CORPUS_DOCS,queries:PUBLIC_HELD_OUT_QUERIES}));"],
        cwd=root / "unrelated-cwd", env=env, capture_output=True, text=True, timeout=10, check=True)
    dataset = json.loads(fixtures.stdout)
    docs, queries = dataset["docs"], dataset["queries"]
    dimensions = len(docs)
    doc_indices = {doc["id"]: index for index, doc in enumerate(docs)}
    model = str(root / "fixture-model")
    descriptor = {"version": 1, "backend": "mlx", "model": model, "pooling": "mean",
                  "nativeDimensions": dimensions, "outputDimensions": dimensions,
                  "maxTokens": 2048, "normalized": True}
    requests, fixture_errors = [], []

    def embedding(text):
        index = next((doc_indices[query["relevantDocIds"][0]] for query in queries if query["query"] in text), None)
        if index is None:
            index = next((i for i, doc in enumerate(docs) if doc["content"] in text), None)
        if index is None:
            raise ValueError(f"Unknown synthetic fixture input: {text[:120]!r}")
        return [float(i == index) for i in range(dimensions)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_body(self, payload, *, status=200, binary=False):
            body = payload if binary else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream" if binary else "application/json")
            self.send_header("Content-Length", str(len(body)))
            if binary:
                self.send_header("X-QMD-Embedding-Descriptor", quote(json.dumps(descriptor), safe=""))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            requests.append(self.path)
            if self.path == "/health":
                self.send_body({"status": "ok", "ready": True, "model": model, "dims": dimensions, "descriptor": descriptor})
            elif self.path == "/descriptor":
                self.send_body(descriptor)
            else:
                self.send_body({"error": "unexpected fixture route"}, status=404)

        def do_POST(self):
            requests.append(self.path)
            try:
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                texts = data["texts"]
                if self.path == "/tokenize":
                    self.send_body({"counts": [max(1, (len(text) + 3) // 4) for text in texts]})
                elif self.path in ("/embed", "/embed-bin"):
                    vectors = [embedding(text) for text in texts]
                    if self.path == "/embed-bin":
                        flat = [value for row in vectors for value in row]
                        self.send_body(struct.pack("<ii", len(vectors), dimensions)
                                       + struct.pack(f"<{len(flat)}f", *flat), binary=True)
                    else:
                        self.send_body({"embeddings": vectors, "model": model, "dims": dimensions, "descriptor": descriptor})
                else:
                    raise ValueError(f"Unexpected fixture route {self.path}")
            except Exception as error:
                fixture_errors.append(str(error))
                self.send_body({"error": str(error)}, status=500)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    port = server.server_port
    assert port != 8787
    phase5_composition.context.base_url = f"http://127.0.0.1:{port}"
    phase5_composition.context.deadline = time.monotonic() + 60
    phase5_composition.supervisor.config.mlx_port = port
    phase5_composition.supervisor.config.mlx_model = model
    phase5_composition.supervisor.config.timeout_s = 60
    monkeypatch.setenv("TMPDIR", str(root / "tmp"))
    monkeypatch.chdir(root / "unrelated-cwd")
    results = []

    def execute(**kwargs):
        phase5_composition.captured.update(kwargs)
        phase5_composition.captured["scratch_root"] = str(Path(kwargs["env"]["HOME"]).parent)
        assert kwargs["cmd"][:3] == command[:3]
        assert "NODE_OPTIONS" not in kwargs["env"]
        result = run_subordinate_process(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(phase5, "run_subordinate_process", execute)
    try:
        report = phase5_composition.run()
        assert report["status"] == "passed", {"errors": report["errors"], "logs": [r.captured_logs for r in results]}
        assert report["errors"] == [] and all(report["checks"].values())
        assert results[0].exit_code == 0 and results[0].error is None
        assert report["checks"]["fresh_process_recovery"] is True
        assert report["checks"]["sqlite_online_backup_restored"] is True
        assert report["checks"]["restored_retrieval_identical"] is True
        assert report["checks"]["mcp_transport_probe_passed"] is True
        assert report["corpus"]["totalDocs"] == len(docs)
        assert report["recovery"]["backupDocCount"] == len(docs)
        assert report["mcp_probe"]["toolsCount"] >= 4
        assert report["mcp_probe"]["getContentLength"] > 50
        for worker in report["recovery"]["workers"]:
            with pytest.raises(ProcessLookupError):
                os.kill(worker["pid"], 0)
        with pytest.raises(ProcessLookupError):
            os.kill(results[0].pid, 0)
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", report["mcp_probe"]["port"])) != 0
        assert {"/descriptor", "/tokenize", "/embed"}.issubset(requests)
        assert fixture_errors == []
        assert not list((root / "tmp").iterdir())
        assert not list((root / "home").iterdir())
        print(json.dumps({"evidence": "Python-Phase5-Node-public-CPU-fixture", "node": command[0],
                          "status": report["status"], "workers": report["recovery"]["workers"],
                          "restoredDocs": report["recovery"]["backupDocCount"], "mcp": report["mcp_probe"]}))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0
