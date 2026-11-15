"""Reviewed compatibility regressions, offline and without real training/Hub I/O.

CLI tests invoke the actual main() and real CPU configuration constructors.
Training, downloads, GGUF conversion and uploads are explicitly labelled fakes;
FAKE GGUF bytes are routing fixtures, not usable exported models. Numerical
loader tests use only locally saved, randomly initialized tiny CPU models.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUANTS = ("q4_k_m", "q6_k", "q8_0")
FAKE_GGUF = b"GGUF\x00FAKE export/upload routing fixture; not model weights"


def load_module(relative):
    spec = importlib.util.spec_from_file_location("review_" + relative.replace(".", "_"), ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_hub_error(error_class, message):
    # Only error-constructor metadata is faked; no HTTP request is performed.
    return error_class(message, response=NS(headers={}, request=None))


def native_fakes(monkeypatch, tmp_path, export_mode="complete", upload_error=None):
    """No real training, conversion, process launches, authentication or uploads."""
    import datasets
    import huggingface_hub
    import peft
    import torch
    import train
    import transformers
    import trl

    events, trainers = [], []
    model = NS()
    model.merge_and_unload = lambda: (events.append(("FAKE merge",)) or model)
    model.push_to_hub = lambda repo: events.append(("FAKE adapter upload", repo))
    tokenizer = NS(pad_token=None, eos_token="</s>")
    tokenizer.save_pretrained = lambda path: events.append(("FAKE tokenizer save", path))
    tokenizer.push_to_hub = lambda repo: events.append(("FAKE tokenizer upload", repo))

    def model_load(*args, **kwargs):
        events.append(("FAKE model load", args, kwargs))
        return model

    def tokenizer_load(*args, **kwargs):
        events.append(("FAKE tokenizer load", args, kwargs))
        return tokenizer

    class FakeData:
        def shuffle(self, **kwargs):
            return self

        def train_test_split(self, **kwargs):
            return {"train": self, "test": self}

    def dataset_load(*args, **kwargs):
        events.append(("FAKE dataset load", args, kwargs))
        return FakeData()

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = model
            trainers.append(self)

        def train(self):
            events.append(("FAKE training",))

        def save_model(self, path):
            events.append(("FAKE adapter save", path))
            Path(path).mkdir(parents=True, exist_ok=True)

    def fake_export(merged, tok, directory, name):
        assert merged is model and tok is tokenizer
        events.append(("FAKE GGUF export", directory, name))
        if export_mode == "raise":
            raise RuntimeError("FAKE GGUF export failure")
        destination = Path(directory) / "gguf"
        destination.mkdir()
        for quant in QUANTS:
            if export_mode == "missing" or (export_mode == "partial" and quant == "q8_0"):
                continue
            data = b"" if export_mode == "empty" else b"bad-format" if export_mode == "invalid" else FAKE_GGUF
            (destination / f"{name}-{quant}.gguf").write_bytes(data)

    class FakeApi:
        def upload_file(self, **kwargs):
            # Bind the real maintained API signature, but never call its body.
            inspect.signature(original_upload).bind(self, **kwargs)
            assert Path(kwargs["path_or_fileobj"]).read_bytes() == FAKE_GGUF
            events.append(("FAKE GGUF upload", kwargs))
            if upload_error is not None:
                raise upload_error
            return NS(oid="FAKE-not-a-real-Hub-commit")

    def fake_eval(*args, **kwargs):
        events.append(("FAKE evaluation subprocess", args, kwargs))
        return NS(returncode=0)

    original_upload = huggingface_hub.HfApi.upload_file
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(datasets, "load_dataset", dataset_load)
    monkeypatch.setattr(trl, "SFTTrainer", FakeTrainer)
    monkeypatch.setattr(train, "export_gguf", fake_export)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(subprocess, "run", fake_eval)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return NS(events=events, trainers=trainers, model=model, tokenizer=tokenizer)


def run_native(monkeypatch, tmp_path, extra=()):
    module = load_module("train_unsloth.py")
    monkeypatch.setattr(sys, "argv", [
        "train_unsloth.py", "--model", "2B", "--epochs", "7", "--batch-size", "3",
        "--grad-accum", "5", "--lr", "0.0006", "--max-seq-len", "256",
        "--lora-rank", "8", "--data", str(tmp_path / "FAKE-input.jsonl"),
        "--output", str(tmp_path / "adapter"), "--device", "cpu", *extra,
    ])
    module.main()
    return module


def test_native_cli_publishes_fresh_q4_q6_q8_after_export_with_explicit_fakes(monkeypatch, tmp_path):
    surfaces = native_fakes(monkeypatch, tmp_path)
    run_native(monkeypatch, tmp_path, ["--push-hub", "fixture/repo"])
    events = surfaces.events
    uploads = [event[1] for event in events if event[0] == "FAKE GGUF upload"]
    assert {payload["path_in_repo"] for payload in uploads} == {f"adapter-{quant}.gguf" for quant in QUANTS}
    assert all(payload["repo_id"] == "fixture/repo" and payload["repo_type"] == "model" for payload in uploads)
    assert ("FAKE adapter upload", "fixture/repo") in events
    assert ("FAKE tokenizer upload", "fixture/repo") in events
    names = [event[0] for event in events]
    assert names.index("FAKE training") < names.index("FAKE adapter upload") < names.index("FAKE merge")
    assert names.index("FAKE GGUF export") < names.index("FAKE GGUF upload") < names.index("FAKE evaluation subprocess")
    assert not list((tmp_path / "adapter").glob(".gguf-export-*"))
    config = surfaces.trainers[0].kwargs["args"]
    assert config.num_train_epochs == 7 and config.per_device_train_batch_size == 3
    assert config.gradient_accumulation_steps == 5 and config.learning_rate == 0.0006
    assert config.max_length == 256 and config.get_warmup_steps(100) == 3
    assert config.optim.value == "adamw_torch" and config.use_cpu and not config.bf16
    lora = surfaces.trainers[0].kwargs["peft_config"]
    assert lora.r == 8 and lora.lora_alpha == 8
    child = next(event for event in events if event[0] == "FAKE evaluation subprocess")
    assert child[1][0] == [sys.executable, str(ROOT / "eval.py"), str(tmp_path / "adapter")]
    assert child[2] == {"check": True}


def test_native_cli_skip_flags_keep_adapter_publication_without_gguf_or_eval(monkeypatch, tmp_path):
    surfaces = native_fakes(monkeypatch, tmp_path)
    run_native(monkeypatch, tmp_path, ["--push-hub", "fixture/repo", "--no-gguf", "--no-eval"])
    assert ("FAKE adapter upload", "fixture/repo") in surfaces.events
    assert not any(event[0] in {"FAKE merge", "FAKE GGUF export", "FAKE GGUF upload", "FAKE evaluation subprocess"}
                   for event in surfaces.events)


def test_native_cli_without_push_keeps_local_gguf_and_does_not_upload(monkeypatch, tmp_path):
    surfaces = native_fakes(monkeypatch, tmp_path)
    run_native(monkeypatch, tmp_path, ["--no-eval"])
    assert {path.name for path in (tmp_path / "adapter" / "gguf").glob("*.gguf")} == {
        f"adapter-{quant}.gguf" for quant in QUANTS}
    assert not any("upload" in event[0] for event in surfaces.events)


@pytest.mark.parametrize("export_mode", ["missing", "partial", "empty", "invalid", "raise"])
def test_native_cli_export_failure_never_uploads_stale_ggufs(monkeypatch, tmp_path, capsys, export_mode):
    surfaces = native_fakes(monkeypatch, tmp_path, export_mode=export_mode)
    stale = tmp_path / "adapter" / "gguf"
    stale.mkdir(parents=True)
    for quant in QUANTS:
        (stale / f"adapter-{quant}.gguf").write_bytes(FAKE_GGUF)
    with pytest.raises(RuntimeError, match="GGUF export fail"):
        run_native(monkeypatch, tmp_path, ["--push-hub", "fixture/repo"])
    assert not any(event[0] in {"FAKE GGUF upload", "FAKE evaluation subprocess"} for event in surfaces.events)
    assert "Done!" not in capsys.readouterr().out


def test_native_cli_gguf_upload_failure_is_explicit(monkeypatch, tmp_path, capsys):
    failure = OSError("FAKE upload network failure")
    surfaces = native_fakes(monkeypatch, tmp_path, upload_error=failure)
    with pytest.raises(RuntimeError, match="GGUF upload failed for adapter-q4_k_m.gguf to fixture/repo") as caught:
        run_native(monkeypatch, tmp_path, ["--push-hub", "fixture/repo"])
    assert caught.value.__cause__ is failure
    assert not any(event[0] == "FAKE evaluation subprocess" for event in surfaces.events)
    assert "Done!" not in capsys.readouterr().out


@pytest.mark.parametrize("device", ["cuda", "auto"])
def test_native_missing_8bit_optimizer_fails_before_any_download_or_load(monkeypatch, tmp_path, capsys, device):
    import torch
    import transformers.utils
    import trl

    surfaces = native_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers.utils, "is_bitsandbytes_available", lambda: False)
    # Also safe as a negative control against the old code: never initialize
    # CUDA if a broken implementation wrongly gets as far as configuration.
    monkeypatch.setattr(trl, "SFTConfig", lambda **kwargs: NS(**kwargs))
    with pytest.raises(SystemExit) as caught:
        run_native(monkeypatch, tmp_path, ["--device", device, "--no-gguf", "--no-eval"])
    assert caught.value.code == 2
    assert surfaces.events == [] and surfaces.trainers == []
    assert "adamw_8bit requires bitsandbytes" in capsys.readouterr().err


def test_native_available_cuda_retains_8bit_optimizer_configuration_with_fakes(monkeypatch, tmp_path):
    import torch
    import transformers.utils
    import trl

    surfaces = native_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformers.utils, "is_bitsandbytes_available", lambda: True)
    # CUDA configuration mapping only: never initialize CUDA or a real trainer.
    monkeypatch.setattr(trl, "SFTConfig", lambda **kwargs: NS(**kwargs))
    run_native(monkeypatch, tmp_path, ["--device", "cuda", "--no-gguf", "--no-eval"])
    config = surfaces.trainers[0].kwargs["args"]
    assert config.optim == "adamw_8bit" and config.bf16 and not config.use_cpu
    model_call = next(event for event in surfaces.events if event[0] == "FAKE model load")
    assert model_call[2]["dtype"] == torch.bfloat16


def evaluator_fakes(monkeypatch):
    import peft
    import torch
    import transformers

    calls = []
    model = NS(generation_config=NS(do_sample=True, temperature=0.7, top_p=0.9, top_k=50))
    model.eval = lambda: calls.append(("eval",))
    tokenizer = NS(pad_token=None, eos_token="</s>")
    config = NS(tie_word_embeddings=True)

    def record(kind, result):
        def call(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return result
        return call

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", record("config", config))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", record("model", model))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", record("tokenizer", tokenizer))
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", record("adapter", model))
    return NS(calls=calls, model=model, tokenizer=tokenizer, config=config)


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("route", ["local-full", "remote-full", "local-adapter", "remote-adapter"])
@pytest.mark.parametrize("own_tokenizer", [False, True])
def test_evaluator_routes_exact_requested_model_and_metadata(monkeypatch, tmp_path, evaluator, route, own_tokenizer):
    import huggingface_hub
    import torch
    from huggingface_hub.errors import RemoteEntryNotFoundError

    surfaces = evaluator_fakes(monkeypatch)
    is_adapter = route.endswith("adapter")
    requested = "fixture/requested-model"
    directory = tmp_path / "requested-model"
    directory.mkdir()
    if route.startswith("local"):
        # Preserve the caller's spelling, including '/./', in loader arguments.
        requested = str(tmp_path) + "/./requested-model"
    adapter_file = directory / "adapter_config.json"
    if is_adapter:
        adapter_file.write_text(json.dumps({"base_model_name_or_path": "fixture/declared-base"}))
    if own_tokenizer:
        (directory / "tokenizer_config.json").write_text("{}")

    def metadata(repo, filename, **kwargs):
        assert route.startswith("remote") and repo == requested and kwargs == {"repo_type": "model"}
        if filename == "adapter_config.json" and is_adapter:
            return str(adapter_file)
        if filename == "tokenizer_config.json" and own_tokenizer:
            return str(directory / filename)
        raise fake_hub_error(RemoteEntryNotFoundError, "FAKE confirmed missing metadata file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", metadata)
    loaded, tokenizer = load_module(evaluator).load_model(requested)
    expected_model = "fixture/declared-base" if is_adapter else requested
    expected_tokenizer = requested if not is_adapter or own_tokenizer else expected_model
    model_call = next(call for call in surfaces.calls if call[0] == "model")
    assert model_call[1] == (expected_model,)
    assert model_call[2]["dtype"] == torch.float32 and model_call[2]["device_map"] == {"": "cpu"}
    assert next(call for call in surfaces.calls if call[0] == "tokenizer")[1] == (expected_tokenizer,)
    adapter_calls = [call for call in surfaces.calls if call[0] == "adapter"]
    assert len(adapter_calls) == int(is_adapter)
    if is_adapter:
        assert adapter_calls[0][1] == (surfaces.model, requested)
    assert model_call[2]["config"] is surfaces.config
    assert surfaces.config.tie_word_embeddings is True
    assert tokenizer.padding_side == "left" and tokenizer.pad_token == tokenizer.eos_token
    assert loaded is surfaces.model and not loaded.generation_config.do_sample


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("failure_kind", ["auth", "gated", "offline", "network"])
@pytest.mark.parametrize("metadata_file", ["adapter_config.json", "tokenizer_config.json"])
def test_evaluator_metadata_failures_propagate_without_default_model(
    monkeypatch, tmp_path, evaluator, failure_kind, metadata_file
):
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError

    failures = {
        "auth": fake_hub_error(RepositoryNotFoundError, "FAKE private repository 401"),
        "gated": fake_hub_error(GatedRepoError, "FAKE gated repository 403"),
        "offline": LocalEntryNotFoundError("FAKE offline cache miss"),
        "network": OSError("FAKE network connection failure"),
    }
    failure = failures[failure_kind]
    adapter_file = tmp_path / "adapter_config.json"
    adapter_file.write_text(json.dumps({"base_model_name_or_path": "fixture/base"}))
    surfaces = evaluator_fakes(monkeypatch)

    def metadata(repo, filename, **kwargs):
        if filename == metadata_file:
            raise failure
        return str(adapter_file)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", metadata)
    with pytest.raises(type(failure)) as caught:
        load_module(evaluator).load_model("fixture/requested-model")
    assert caught.value is failure and surfaces.calls == []


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("base", [None, "", "  ", 1])
def test_evaluator_invalid_adapter_base_never_guesses_qwen(monkeypatch, tmp_path, evaluator, base):
    surfaces = evaluator_fakes(monkeypatch)
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        load_module(evaluator).load_model(str(tmp_path))
    assert surfaces.calls == []


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("model_kind", ["full", "adapter", "merged"])
def test_evaluator_real_local_tiny_cpu_model_reload_parity(
    monkeypatch, tmp_path, tiny_model, tiny_tokenizer, evaluator, model_kind
):
    """No optimizer/training: numerical reload compatibility, not model quality."""
    import torch
    from peft import LoraConfig, get_peft_model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = tmp_path / "base"
    if model_kind == "adapter":
        # Exercise the evaluator's inherited independent adapter-head loading.
        tiny_model.config.tie_word_embeddings = False
        tiny_model.lm_head.weight = torch.nn.Parameter(tiny_model.lm_head.weight.detach().clone())
    tiny_model.save_pretrained(base)
    tiny_tokenizer.save_pretrained(base)
    requested = base
    reference = tiny_model
    if model_kind != "full":
        tiny_model.name_or_path = str(base)
        reference = get_peft_model(tiny_model, LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        # Make the adapter observable without any optimizer or training step.
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.1)
        requested = tmp_path / model_kind
        if model_kind == "merged":
            reference = reference.merge_and_unload()
        reference.save_pretrained(requested)
        tiny_tokenizer.chat_template = "OWN SAVED TOKENIZER {{ messages[0]['content'] }}"
        tiny_tokenizer.save_pretrained(requested)
    reference.eval()
    loaded, tokenizer = load_module(evaluator).load_model(str(requested))
    ids = torch.tensor([[6, 7, 8]])
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits
        actual = loaded(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert loaded.device.type == "cpu" and loaded.dtype == torch.float32
    assert tokenizer.chat_template == tiny_tokenizer.chat_template
    if model_kind != "adapter":
        assert loaded.config.tie_word_embeddings
        assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
