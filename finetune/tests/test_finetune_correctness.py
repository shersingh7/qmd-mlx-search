"""CPU PEFT reload parity and actual GGUF caller failure-routing regressions.

Models/tokenizers are tiny locally created fixtures; no training or downloads.
All external commands, training and Hub writes below are explicit fakes. The
GGUF writer zero-block fixtures prove structure/routing, not usable model weights.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from test_gguf_structure import fake_gguf_bytes

ROOT = Path(__file__).resolve().parents[1]
QUANTS = ("q4_k_m", "q6_k", "q8_0")
FAKE_GGUF = fake_gguf_bytes()
FAKE_PARTIAL = b"GGUF\x00EXPLICIT FAKE partial output from a FAILED process"
FAKE_STALE = b"GGUF\x00EXPLICIT FAKE previous export; must survive failed export"
STAGES = ("init", "fetch", "checkout", "revision", "probe", "configure", "build", "converter", *QUANTS)


def load_module(name):
    spec = importlib.util.spec_from_file_location("correctness_" + name.replace(".", "_"), ROOT / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("kind", ["tied-lora", "untied-lora", "independent-saved-heads"])
def test_actual_peft_adapter_preserves_base_weights_and_saved_heads(
    tmp_path, monkeypatch, tiny_model, tiny_tokenizer, evaluator, kind
):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if kind == "untied-lora":
        tiny_model.config.tie_word_embeddings = False
        tiny_model.lm_head.weight = torch.nn.Parameter(tiny_model.lm_head.weight.detach().clone() + 0.13)
    base = tmp_path / "declared-base"
    tiny_model.save_pretrained(base)
    tiny_tokenizer.save_pretrained(base)
    base_tied = tiny_model.config.tie_word_embeddings
    with safe_open(base / "model.safetensors", framework="pt") as saved:
        # Tied checkpoints deliberately omit a duplicate output-head tensor.
        assert ("lm_head.weight" not in saved.keys()) == base_tied
    tiny_model.name_or_path = str(base)
    modules_to_save = None
    if kind == "independent-saved-heads":
        # The base checkpoint remains tied; this adapter owns distinct heads.
        tiny_model.config.tie_word_embeddings = False
        tiny_model.lm_head.weight = torch.nn.Parameter(tiny_model.lm_head.weight.detach().clone() + 0.13)
        with torch.no_grad():
            tiny_model.model.embed_tokens.weight.add_(0.07)
        modules_to_save = ["embed_tokens", "lm_head"]
    reference = get_peft_model(tiny_model, LoraConfig(
        r=2, lora_alpha=4, target_modules=["v_proj"],
        modules_to_save=modules_to_save, task_type="CAUSAL_LM",
    ))
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.15)
    adapter = tmp_path / "requested-adapter"
    reference.save_pretrained(adapter)
    tiny_tokenizer.chat_template = "ADAPTER OWNED {{ messages[0]['content'] }}"
    tiny_tokenizer.save_pretrained(adapter)
    reference.eval()
    loaded, tokenizer = load_module(evaluator).load_model(str(adapter))
    ids = torch.tensor([[6, 7, 8, 9], [9, 8, 7, 6]])
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits
        actual = loaded(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert loaded.config.tie_word_embeddings is base_tied
    expected_tied = kind == "tied-lora"
    assert (loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight) is expected_tied
    torch.testing.assert_close(loaded.get_input_embeddings().weight, reference.get_input_embeddings().weight, rtol=0, atol=0)
    torch.testing.assert_close(loaded.get_output_embeddings().weight, reference.get_output_embeddings().weight, rtol=0, atol=0)
    assert loaded.device.type == "cpu" and loaded.dtype == torch.float32
    assert tokenizer.chat_template == tiny_tokenizer.chat_template


def fake_native_export(tmp_path, monkeypatch, *, failed_stage=None, partial=False, process_error=None):
    """Exercise real main/helper/exporter with every external action faked."""
    import datasets
    import huggingface_hub as hub
    import torch
    import train
    import transformers as tf
    import trl

    import gguf_toolchain as tools
    events = []
    llama = quantize = None
    original_upload = inspect.signature(hub.HfApi.upload_file)
    monkeypatch.setattr(tools, "check_prerequisites", lambda **kwargs: {
        "git": "git", "cmake": "cmake", "cc": "cc", "c++": "c++", "make": "make",
    })

    def process(argv, **kwargs):
        nonlocal llama, quantize
        argv = list(map(str, argv))
        artifact = None
        stdout = ""
        if argv[0] == "git":
            if "init" in argv:
                stage = "init"
                llama = Path(argv[-1])
                assert llama.is_dir() and llama.parent.name == "tools"
            elif "fetch" in argv:
                stage = "fetch"
                assert argv[-4:] == ["--depth", "1", tools.SOURCE_URL, tools.SOURCE_REVISION]
            elif "checkout" in argv:
                stage = "checkout"
                assert argv[-2:] == ["--detach", tools.SOURCE_REVISION]
                (llama / "convert_hf_to_gguf.py").write_text("EXPLICIT FAKE; never executed")
            else:
                assert argv[-2:] == ["rev-parse", "HEAD"]
                stage = "revision"
                stdout = tools.SOURCE_REVISION + "\n"
        elif argv[:2] == ["cmake", "-B"]:
            stage = "configure"
            assert "-DGGML_CUDA=OFF" in argv and "-DGGML_METAL=OFF" in argv
            quantize = Path(argv[2]) / "bin/llama-quantize"
        elif argv[:2] == ["cmake", "--build"]:
            assert argv[-4:] == ["--target", "llama-quantize", "-j", "2"]
            stage = "build"
            quantize.parent.mkdir(parents=True, exist_ok=True)
            quantize.write_text("EXPLICIT FAKE executable placeholder; NEVER EXECUTED")
        elif llama is not None and len(argv) > 2 and argv[2] == str(llama / "convert_hf_to_gguf.py"):
            assert argv[:2] == [sys.executable, "-s"]
            if argv[-1] == "--help":
                stage = "probe"
            else:
                assert Path(argv[3]).name == "merged" and Path(argv[3]).is_dir()
                assert Path(kwargs["stdout"].name).name == "converter.stdout"
                assert Path(kwargs["stderr"].name).name == "converter.stderr"
                stage = "converter"
                artifact = Path(argv[argv.index("--outfile") + 1])
        elif quantize is not None and argv[0] == str(quantize):
            assert argv[-1].lower() in QUANTS and Path(argv[1]).read_bytes() == FAKE_GGUF
            stage = argv[-1].lower()
            artifact = Path(argv[2])
        elif len(argv) > 1 and argv[1] == str(ROOT / "eval.py"):
            assert kwargs == {"check": True}
            events.append(("FAKE evaluation", argv))
            return NS(returncode=0)
        else:
            raise AssertionError(f"Forbidden unexpected external command: {argv}")
        rc = 17 if stage == failed_stage else 0
        events.append(("FAKE process", stage, rc, argv))
        if artifact is not None and (not rc or partial):
            artifact.write_bytes(FAKE_PARTIAL if rc else fake_gguf_bytes("f16" if stage == "converter" else stage))
        if stage == failed_stage and process_error is not None:
            raise process_error
        # Explicit returncode checks must work even with a process fake that
        # does not implement subprocess.run(check=True) on the caller's behalf.
        return NS(returncode=rc, stdout=stdout, stderr="EXPLICIT FAKE process failure" if rc else "")

    def save(where, **kwargs):
        events.append(("FAKE model/tokenizer save", str(where)))
        Path(where).mkdir(parents=True, exist_ok=True)

    model = NS(save_pretrained=save)
    model.merge_and_unload = lambda: (events.append(("FAKE merge",)) or model)
    model.push_to_hub = lambda repo: events.append(("FAKE adapter upload", repo))
    tokenizer = NS(pad_token=None, eos_token="</s>", save_pretrained=save)
    tokenizer.push_to_hub = lambda repo: events.append(("FAKE tokenizer upload", repo))

    class FakeDataset:
        def shuffle(self, **kwargs):
            return self

        def train_test_split(self, **kwargs):
            return {"train": self, "test": self}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = model

        def train(self):
            events.append(("FAKE training; NO optimizer step",))

        def save_model(self, where):
            save(where)

    class FakeApi:
        def upload_file(self, **kwargs):
            original_upload.bind(self, **kwargs)
            assert kwargs["repo_type"] == "model" and kwargs["repo_id"] == "fixture/NEVER-WRITTEN"
            quant = Path(kwargs["path_in_repo"]).stem.rsplit("-", 1)[1]
            assert Path(kwargs["path_or_fileobj"]).read_bytes() == fake_gguf_bytes(quant)
            events.append(("FAKE GGUF upload", kwargs["path_in_repo"]))
            return NS(oid="EXPLICIT-FAKE-no-remote-commit")

    monkeypatch.setattr(subprocess, "run", process)
    monkeypatch.setattr(tf.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(tf.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: FakeDataset())
    monkeypatch.setattr(trl, "SFTTrainer", FakeTrainer)
    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return events


def run_native(tmp_path, monkeypatch, extra=()):
    monkeypatch.setattr(sys, "argv", [
        "train_unsloth.py", "--model", "0.8B", "--device", "cpu",
        "--data", str(tmp_path / "EXPLICIT-FAKE-never-read-input.jsonl"),
        "--output", str(tmp_path / "adapter"), *extra,
    ])
    load_module("train_unsloth.py").main()


@pytest.mark.parametrize("stage,partial", [
    *((stage, False) for stage in STAGES[:7]),
    *((stage, partial) for stage in STAGES[7:] for partial in (False, True)),
])
def test_failed_export_process_never_moves_uploads_evaluates_or_reports_done(
    tmp_path, monkeypatch, capsys, stage, partial
):
    events = fake_native_export(tmp_path, monkeypatch, failed_stage=stage, partial=partial)
    destination = tmp_path / "adapter/gguf"
    destination.mkdir(parents=True)
    for quant in QUANTS:
        (destination / f"adapter-{quant}.gguf").write_bytes(FAKE_STALE)
    with pytest.raises(RuntimeError, match="GGUF.*fail.*17"):
        run_native(tmp_path, monkeypatch, ["--push-hub", "fixture/NEVER-WRITTEN"])
    processes = [event[1] for event in events if event[0] == "FAKE process"]
    assert processes == list(STAGES[:STAGES.index(stage) + 1])
    assert ("FAKE adapter upload", "fixture/NEVER-WRITTEN") in events
    assert ("FAKE tokenizer upload", "fixture/NEVER-WRITTEN") in events
    assert not any(event[0] in ("FAKE GGUF upload", "FAKE evaluation") for event in events)
    assert all((destination / f"adapter-{quant}.gguf").read_bytes() == FAKE_STALE for quant in QUANTS)
    assert not list((tmp_path / "adapter").glob(".gguf-export-*"))
    output = capsys.readouterr().out
    assert "Done!" not in output and "GGUF files saved to:" not in output


@pytest.mark.parametrize("stage", STAGES)
def test_export_process_exceptions_propagate_before_publication(tmp_path, monkeypatch, capsys, stage):
    failure = OSError(f"EXPLICIT FAKE process launch/I/O error at {stage}")
    events = fake_native_export(tmp_path, monkeypatch, failed_stage=stage, partial=True, process_error=failure)
    with pytest.raises(OSError) as caught:
        run_native(tmp_path, monkeypatch, ["--push-hub", "fixture/NEVER-WRITTEN"])
    assert caught.value is failure
    assert not any(event[0] in ("FAKE GGUF upload", "FAKE evaluation") for event in events)
    assert not (tmp_path / "adapter/gguf").exists()
    assert not list((tmp_path / "adapter").glob(".gguf-export-*"))
    assert "Done!" not in capsys.readouterr().out


@pytest.mark.parametrize("push", [False, True])
def test_successful_real_exporter_flow_keeps_fresh_all_quantizations(tmp_path, monkeypatch, capsys, push):
    events = fake_native_export(tmp_path, monkeypatch)
    destination = tmp_path / "adapter/gguf"
    destination.mkdir(parents=True)
    for quant in QUANTS:
        (destination / f"adapter-{quant}.gguf").write_bytes(FAKE_STALE)
    run_native(tmp_path, monkeypatch, ["--push-hub", "fixture/NEVER-WRITTEN"] if push else [])
    assert [event[1] for event in events if event[0] == "FAKE process"] == list(STAGES)
    assert all((destination / f"adapter-{quant}.gguf").read_bytes() == fake_gguf_bytes(quant) for quant in QUANTS)
    assert not (destination / "adapter-f16.gguf").exists()
    uploads = [event[1] for event in events if event[0] == "FAKE GGUF upload"]
    assert uploads == ([f"adapter-{quant}.gguf" for quant in QUANTS] if push else [])
    tags = [event[0] for event in events]
    assert tags.index("FAKE merge") < tags.index("FAKE process") < tags.index("FAKE evaluation")
    if push:
        assert tags.index("FAKE adapter upload") < tags.index("FAKE process")
        assert tags.index("FAKE tokenizer upload") < tags.index("FAKE process")
        assert max(i for i, tag in enumerate(tags) if tag == "FAKE process") < tags.index("FAKE GGUF upload") < tags.index("FAKE evaluation")
    else:
        assert not any("upload" in tag for tag in tags)
    assert not list((tmp_path / "adapter").glob(".gguf-export-*"))
    assert "Done!" in capsys.readouterr().out


@pytest.mark.parametrize("no_gguf,no_eval", [(True, False), (False, True), (True, True)])
def test_independent_skip_flags_preserve_adapter_and_tokenizer_publication(
    tmp_path, monkeypatch, capsys, no_gguf, no_eval
):
    events = fake_native_export(tmp_path, monkeypatch)
    flags = ["--push-hub", "fixture/NEVER-WRITTEN"]
    if no_gguf:
        flags.append("--no-gguf")
    if no_eval:
        flags.append("--no-eval")
    run_native(tmp_path, monkeypatch, flags)
    assert ("FAKE adapter upload", "fixture/NEVER-WRITTEN") in events
    assert ("FAKE tokenizer upload", "fixture/NEVER-WRITTEN") in events
    tags = [event[0] for event in events]
    assert ("FAKE evaluation" not in tags) is no_eval
    assert ("FAKE merge" not in tags) is no_gguf
    assert ("FAKE process" not in tags) is no_gguf
    assert [event[1] for event in events if event[0] == "FAKE GGUF upload"] == ([] if no_gguf else [f"adapter-{quant}.gguf" for quant in QUANTS])
    assert "Done!" in capsys.readouterr().out
