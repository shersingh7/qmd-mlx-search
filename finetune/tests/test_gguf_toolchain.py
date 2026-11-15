"""GGUF cold-start contracts; process outputs here are explicit routing fakes.

Real offline synthetic-model conversion is qualified separately. The real GGUF
writer supplies tiny zero-block fixtures, not usable converted model weights.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

import gguf_toolchain as tools
from test_gguf_structure import fake_gguf_bytes

ROOT = Path(__file__).resolve().parents[1]
FAKE_GGUF = fake_gguf_bytes()
STAGES = ("init", "fetch", "checkout", "revision", "probe", "configure", "build",
          "converter", "q4_k_m", "q6_k", "q8_0")


def fake_processes(monkeypatch, *, failure=None, partial=False):
    events = []
    monkeypatch.setattr(tools, "check_prerequisites", lambda **kwargs: {
        "git": "git", "cmake": "cmake", "cc": "cc", "c++": "c++", "make": "make",
    })

    def run(argv, **kwargs):
        argv = list(map(str, argv))
        artifact = None
        stdout = ""
        if argv[0] == "git":
            if "init" in argv:
                stage = "init"
            elif "fetch" in argv:
                stage = "fetch"
                assert argv[-2:] == [tools.SOURCE_URL, tools.SOURCE_REVISION]
            elif "checkout" in argv:
                stage = "checkout"
                assert argv[-1] == tools.SOURCE_REVISION
                source = Path(argv[argv.index("-C") + 1])
                (source / "convert_hf_to_gguf.py").write_text("EXPLICIT FAKE; never executed")
            else:
                assert argv[-2:] == ["rev-parse", "HEAD"]
                stage = "revision"
                stdout = tools.SOURCE_REVISION + "\n"
            assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
            assert kwargs["env"]["GIT_CONFIG_GLOBAL"] == tools.os.devnull
        elif "convert_hf_to_gguf.py" in " ".join(argv):
            assert argv[:2] == [sys.executable, "-s"]
            assert "NO_LOCAL_GGUF" not in kwargs["env"]
            assert "PYTHONPATH" not in kwargs["env"]
            stage = "probe" if argv[-1] == "--help" else "converter"
            if stage == "converter":
                artifact = Path(argv[argv.index("--outfile") + 1])
        elif argv[:2] == ["cmake", "-B"]:
            stage = "configure"
            assert "-DGGML_CUDA=OFF" in argv and "-DGGML_METAL=OFF" in argv
            assert "-DLLAMA_BUILD_SERVER=OFF" in argv and "-DLLAMA_BUILD_APP=OFF" in argv
            assert "-DLLAMA_OPENSSL=OFF" in argv
        elif argv[:2] == ["cmake", "--build"]:
            stage = "build"
            assert argv[-2:] == ["-j", "2"]
            binary = Path(argv[2]) / "bin/llama-quantize"
            binary.parent.mkdir(parents=True)
            binary.write_text("EXPLICIT FAKE; never executed")
        elif argv[0].endswith("llama-quantize"):
            stage = argv[-1].lower()
            assert Path(argv[1]).read_bytes() == FAKE_GGUF
            artifact = Path(argv[2])
        else:
            raise AssertionError(f"Unexpected external command {argv}")
        events.append(stage)
        code = 17 if stage == failure else 0
        if artifact is not None and (not code or partial):
            artifact.write_bytes(fake_gguf_bytes("f16" if stage == "converter" else stage))
        return NS(returncode=code, stdout=stdout, stderr="EXPLICIT FAKE process failure" if code else "")

    monkeypatch.setattr(subprocess, "run", run)
    return events


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("partial", [False, True])
def test_every_toolchain_stage_is_authoritative_before_publication(tmp_path, monkeypatch, stage, partial):
    events = fake_processes(monkeypatch, failure=stage, partial=partial)
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(RuntimeError, match="GGUF export failed:.*17"):
        toolchain = tools.prepare_toolchain(tmp_path / "owned-tools")
        tools.convert_model(toolchain, merged, "fixture")
    assert events == list(STAGES[:STAGES.index(stage) + 1])


def test_no_pip_install_mutable_head_or_inherited_python_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted"))
    monkeypatch.setenv("NO_LOCAL_GGUF", "1")
    events = fake_processes(monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()
    toolchain = tools.prepare_toolchain(tmp_path / "owned-tools")
    files = tools.convert_model(toolchain, merged, "fixture")
    assert events == list(STAGES)
    assert [path.name for path in files] == [f"fixture-{q.lower()}.gguf" for q in tools.QUANTIZATIONS]
    assert all(path.read_bytes() == fake_gguf_bytes(q.lower()) for path, q in zip(files, tools.QUANTIZATIONS, strict=True))
    assert not (toolchain.work / "gguf/fixture-f16.gguf").exists()


def test_wrong_source_identity_cannot_reach_converter_or_build(tmp_path, monkeypatch):
    events = fake_processes(monkeypatch)
    original = subprocess.run

    def wrong_revision(argv, **kwargs):
        result = original(argv, **kwargs)
        if "rev-parse" in argv:
            result.stdout = "0" * 40 + "\n"
        return result

    monkeypatch.setattr(subprocess, "run", wrong_revision)
    with pytest.raises(RuntimeError, match="source revision"):
        tools.prepare_toolchain(tmp_path / "owned-tools")
    assert events == list(STAGES[:4])


def test_skip_quantize_needs_no_native_build_and_keeps_f16(tmp_path, monkeypatch):
    events = fake_processes(monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()
    toolchain = tools.prepare_toolchain(tmp_path / "owned-tools", quantize=False)
    files = tools.convert_model(toolchain, merged, "fixture", quantize=False, keep_f16=True)
    assert events == ["init", "fetch", "checkout", "revision", "probe", "converter"]
    assert [path.name for path in files] == ["fixture-f16.gguf"]


def test_existing_tool_directory_is_not_reused(tmp_path, monkeypatch):
    fake_processes(monkeypatch)
    owned = tmp_path / "owned-tools"
    owned.mkdir()
    with pytest.raises(FileExistsError):
        tools.prepare_toolchain(owned)


@pytest.mark.parametrize("missing", ["sentencepiece.sentencepiece_model_pb2", "numpy"])
def test_missing_python_dependency_fails_without_install_or_model_work(monkeypatch, missing):
    real_import = tools.importlib.import_module

    def import_module(name):
        if name == missing:
            raise ImportError("EXPLICIT missing package")
        return real_import(name)

    monkeypatch.setattr(tools, "importlib", NS(import_module=import_module))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not run an installer"))
    with pytest.raises(RuntimeError, match="GGUF.*prerequisite"):
        tools.check_prerequisites()


@pytest.mark.parametrize("missing", ["git", "cmake", "cc", "c++", "make"])
def test_missing_native_prerequisite_fails_without_installer(monkeypatch, missing):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None if name == missing else "/EXPLICIT/" + name)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not install system packages"))
    with pytest.raises(RuntimeError, match="GGUF.*prerequisite"):
        tools.check_prerequisites()


def test_missing_prerequisite_is_before_native_model_load(tmp_path, monkeypatch):
    import transformers
    failure = RuntimeError("GGUF prerequisite missing: EXPLICIT FAKE cmake")
    monkeypatch.setattr(tools, "check_prerequisites", lambda **k: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("Tokenizer was loaded"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("Model was loaded"))
    spec = importlib.util.spec_from_file_location("cold_native", ROOT / "train_unsloth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["train_unsloth.py", "--model", "0.8B", "--device", "cpu"])
    with pytest.raises(RuntimeError) as caught:
        module.main()
    assert caught.value is failure


def test_missing_prerequisite_is_before_yaml_trainer_or_dataset(tmp_path, monkeypatch):
    import train
    import datasets
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: {base: EXPLICIT-FAKE, output: outputs/fixture}\n")
    failure = RuntimeError("GGUF prerequisite missing: EXPLICIT FAKE cmake")
    monkeypatch.setattr(tools, "check_prerequisites", lambda **k: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: pytest.fail("Dataset was loaded"))
    with pytest.raises(RuntimeError) as caught:
        train.cmd_sft(NS(config=str(cfg), dry_run=False))
    assert caught.value is failure


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("partial", [False, True])
def test_standalone_failure_never_uploads_reports_success_or_replaces_old_outputs(
    tmp_path, monkeypatch, capsys, stage, partial
):
    import convert_gguf as cli
    events = fake_processes(monkeypatch, failure=stage, partial=partial)
    model_events = []

    def save(path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)

    model = NS(save_pretrained=save)
    model.merge_and_unload = lambda: model
    tokenizer = NS(save_pretrained=save)
    monkeypatch.setattr(cli.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: (model_events.append("model") or model))
    monkeypatch.setattr(cli.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(cli.PeftModel, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(cli, "HfApi", lambda: pytest.fail("Must not upload any artifact after a failed process"))
    monkeypatch.setattr(cli, "login", lambda **k: pytest.fail("Must not authenticate in offline test"))
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["convert_gguf.py", "--size", "1.7B", "--output", "fixture/NEVER-WRITTEN-gguf"])
    old = tmp_path / "gguf_output"
    old.mkdir()
    for quant in ("f16", "q4_k_m", "q6_k", "q8_0"):
        (old / f"NEVER-WRITTEN-{quant}.gguf").write_bytes(b"EXPLICIT STALE output; must survive failure")
    with pytest.raises(RuntimeError, match="GGUF export failed:.*17"):
        cli.main()
    assert events == list(STAGES[:STAGES.index(stage) + 1])
    assert model_events == ([] if STAGES.index(stage) < 7 else ["model"])
    assert all(path.read_bytes() == b"EXPLICIT STALE output; must survive failure" for path in old.iterdir())
    assert not list(tmp_path.glob("qmd-gguf-convert-*"))
    assert "Done!" not in capsys.readouterr().out


@pytest.mark.parametrize("skip", [False, True])
def test_standalone_success_uploads_only_complete_current_q4_q6_q8_or_f16(tmp_path, monkeypatch, skip):
    import convert_gguf as cli
    events = fake_processes(monkeypatch)
    uploads = []

    def save(path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)

    model = NS(save_pretrained=save)
    model.merge_and_unload = lambda: model
    tokenizer = NS(save_pretrained=save)
    monkeypatch.setattr(cli.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(cli.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(cli.PeftModel, "from_pretrained", lambda *a, **k: model)

    class FakeApi:
        def create_repo(self, **kwargs):
            assert kwargs["repo_id"] == "fixture/NEVER-WRITTEN-gguf" and kwargs["repo_type"] == "model"

        def upload_file(self, **kwargs):
            if kwargs["path_in_repo"] != "README.md":
                quant = Path(kwargs["path_in_repo"]).stem.rsplit("-", 1)[1]
                assert Path(kwargs["path_or_fileobj"]).read_bytes() == fake_gguf_bytes(quant)
                uploads.append(kwargs["path_in_repo"])

    monkeypatch.setattr(cli, "HfApi", FakeApi)
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["convert_gguf.py", "--size", "1.7B", "--output", "fixture/NEVER-WRITTEN-gguf",
                                      *(["--skip-quantize"] if skip else [])])
    cli.main()
    quants = ("f16",) if skip else ("f16", "q4_k_m", "q6_k", "q8_0")
    assert uploads == [f"NEVER-WRITTEN-{q}.gguf" for q in quants]
    assert not list(tmp_path.glob("qmd-gguf-convert-*"))
    assert events == (["init", "fetch", "checkout", "revision", "probe", "converter"] if skip else list(STAGES))


@pytest.mark.parametrize("stage", ["converter", "q4_k_m", "q6_k", "q8_0"])
@pytest.mark.parametrize("contents", [None, b"", b"BAD not GGUF"])
def test_zero_exit_missing_empty_or_invalid_artifacts_are_not_success(tmp_path, monkeypatch, stage, contents):
    fake_processes(monkeypatch)
    original = subprocess.run

    def invalid_output(argv, **kwargs):
        proc = original(argv, **kwargs)
        argv = list(map(str, argv))
        artifact = None
        if stage == "converter" and "--outfile" in argv:
            artifact = Path(argv[argv.index("--outfile") + 1])
        elif argv[0].endswith("llama-quantize") and argv[-1].lower() == stage:
            artifact = Path(argv[2])
        if artifact is not None:
            if contents is None:
                artifact.unlink()
            else:
                artifact.write_bytes(contents)
        return proc

    monkeypatch.setattr(subprocess, "run", invalid_output)
    toolchain = tools.prepare_toolchain(tmp_path / "owned-tools")
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(RuntimeError, match="GGUF export failed:"):
        tools.convert_model(toolchain, merged, "fixture")


@pytest.mark.parametrize("arch,mtp,expected", [
    ("Qwen3ForCausalLM", False, []),
    ("Qwen3_5ForCausalLM", False, ["--no-mtp"]),
    ("Qwen3_5ForConditionalGeneration", True, []),
])
def test_qwen35_saved_without_mtp_uses_upstream_target_only_api(tmp_path, arch, mtp, expected):
    """Genuine tiny tensor metadata, not a converter/model-success simulation."""
    import json
    import torch
    from safetensors.torch import save_file
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    name = "mtp.layers.0.fixture.weight" if mtp else "model.layers.0.fixture.weight"
    save_file({name: torch.zeros((2, 2))}, tmp_path / "model.safetensors")
    assert tools.conversion_flags(tmp_path) == expected
