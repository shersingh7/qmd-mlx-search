"""Real PEFT saved-layer reload contracts using local tiny CPU fixtures only.

No training/optimizer, downloads, GPU work, processes, or external writes. Raw
single-layer checkpoints use PEFT's public state_dict/save_embedding_layers
save API; wrappers and both serialization formats use its actual save/load path.
"""
from __future__ import annotations

import gc
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING = "base_model.model.model.embed_tokens.weight"
HEAD = "base_model.model.lm_head.weight"
KINDS = (
    "ordinary", "raw-both", "raw-embedding-only", "raw-head-only",
    "modules-both", "modules-embedding-only", "modules-head-only",
)


@pytest.fixture(autouse=True)
def cpu_only_adapter_loads(monkeypatch):
    import torch

    # PEFT infers its checkpoint tensor device independently of the base's
    # device_map. ACCELERATE_USE_CPU alone does not disable its MPS branch.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Match the real cached function API, which torch._dynamo introspects.
    @lru_cache
    def no_mps():
        return False
    monkeypatch.setattr(torch.backends.mps, "is_available", no_mps)


def load_evaluator(filename):
    spec = importlib.util.spec_from_file_location("raw_layers_" + filename.replace(".", "_"), ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_adapter(tmp_path, model, tokenizer, *, kind, base_tied, serialization, raw_backups=False):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    if not base_tied:
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone() + 0.13)
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    base_embedding = model.get_input_embeddings().weight.detach().clone()
    base_head = model.get_output_embeddings().weight.detach().clone()
    model.name_or_path = str(base)
    if kind != "ordinary" and base_tied:
        # Keep the saved base tied, but make the reference's saved layers
        # observably independent, including single-layer-only snapshots.
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    selected = {
        "ordinary": (), "raw-both": (EMBEDDING, HEAD),
        "raw-embedding-only": (EMBEDDING,), "raw-head-only": (HEAD,),
        "modules-both": (EMBEDDING, HEAD),
        "modules-embedding-only": (EMBEDDING,), "modules-head-only": (HEAD,),
    }[kind]
    modules = None
    if kind.startswith("modules-"):
        modules = ["embed_tokens" if key == EMBEDDING else "lm_head" for key in selected]
    reference = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, target_modules=["v_proj"],
        modules_to_save=modules, task_type="CAUSAL_LM",
    ))
    with torch.no_grad():
        if EMBEDDING in selected or raw_backups:
            reference.get_input_embeddings().weight.add_(torch.randn_like(base_embedding) * 0.07)
        if HEAD in selected or raw_backups:
            reference.get_output_embeddings().weight.add_(torch.randn_like(base_head) * 0.11)
        for name, parameter in reference.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.15)
    reference.eval()
    adapter = tmp_path / "adapter"
    state = reference.state_dict()
    if kind.startswith("raw-"):
        # Save only the requested raw layers through the maintained public API,
        # not by hand-writing a made-up adapter serialization.
        for key in (EMBEDDING, HEAD):
            if key not in selected:
                del state[key]
    reference.save_pretrained(
        adapter, state_dict=state, save_embedding_layers=raw_backups or kind.startswith("raw-"),
        safe_serialization=serialization == "safetensors",
    )
    tokenizer.chat_template = "RAW LAYER ADAPTER TOKENIZER {{ messages[0]['content'] }}"
    tokenizer.save_pretrained(adapter)
    if serialization == "safetensors":
        saved = load_file(adapter / "adapter_model.safetensors", device="cpu")
    else:
        saved = torch.load(adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
    actual_layers = set(saved).intersection((EMBEDDING, HEAD))
    assert actual_layers == ({EMBEDDING, HEAD} if raw_backups else set(selected))
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert (config["modules_to_save"] is None) == (modules is None)
    assert config["base_model_name_or_path"] == str(base)
    return NS(base=base, adapter=adapter, reference=reference, tokenizer=tokenizer,
              base_embedding=base_embedding, base_head=base_head)


def observe_reloads(monkeypatch, *, fail_at=None, failure=None):
    """Observe real public loaders/hooks; inject only the specified failure."""
    import peft.peft_model as peft_model
    import torch
    from transformers import AutoModelForCausalLM

    existing_hook_calls = []
    trace = NS(bases=[], hooks=[], reads=[], split_before_failure=False,
               existing_hook_calls=existing_hook_calls)
    real_base_load = AutoModelForCausalLM.from_pretrained
    real_adapter_read = peft_model.load_peft_weights
    real_register = torch.nn.Module.register_load_state_dict_pre_hook

    def existing_hook(module, state_dict, prefix, *unused):
        existing_hook_calls.append(prefix)

    def base_load(*args, **kwargs):
        model = real_base_load(*args, **kwargs)
        # A preexisting caller-owned hook must survive success and failure.
        real_register(model, existing_hook)
        trace.bases.append(NS(
            model=model, config=model.config, requested_config=kwargs["config"],
            input_weight=model.get_input_embeddings().weight,
            output_weight=model.get_output_embeddings().weight,
            hooks=dict(model._load_state_dict_pre_hooks),
        ))
        return model

    def adapter_read(*args, **kwargs):
        trace.reads.append((args, kwargs))
        if fail_at == "before-state-load":
            assert failure is not None
            raise failure
        return real_adapter_read(*args, **kwargs)

    def register(module, hook):
        if any(item.model is module for item in trace.bases):
            trace.hooks.append(weakref.ref(hook))
            if fail_at == "before-raw-copy":
                def fail_after_separation(*args):
                    hook(*args)
                    trace.split_before_failure = (
                        module.get_input_embeddings().weight is not module.get_output_embeddings().weight
                    )
                    assert failure is not None
                    raise failure
                return real_register(module, fail_after_separation)
        return real_register(module, hook)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", base_load)
    monkeypatch.setattr(peft_model, "load_peft_weights", adapter_read)
    monkeypatch.setattr(torch.nn.Module, "register_load_state_dict_pre_hook", register)
    return trace


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("serialization", ["safetensors", "bin"])
@pytest.mark.parametrize("base_tied", [True, False], ids=["tied-base", "untied-base"])
@pytest.mark.parametrize("kind", KINDS)
def test_actual_saved_layer_forms_preserve_exact_weights_and_logits(
    tmp_path, monkeypatch, tiny_model, tiny_tokenizer, record_property,
    evaluator, serialization, base_tied, kind,
):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fixture = saved_adapter(tmp_path, tiny_model, tiny_tokenizer,
                            kind=kind, base_tied=base_tied, serialization=serialization)
    trace = observe_reloads(monkeypatch)
    loaded, tokenizer = load_evaluator(evaluator).load_model(str(fixture.adapter))
    assert len(trace.bases) == 1 and len(trace.reads) == 1
    assert trace.reads[0][1]["device"] == "cpu"
    initial = trace.bases[0]
    assert initial.config is loaded.config
    assert initial.requested_config.tie_word_embeddings is base_tied
    assert loaded.config.tie_word_embeddings is base_tied
    assert (initial.input_weight is initial.output_weight) is base_tied
    if kind == "ordinary":
        # Ordinary adapters must neither allocate a replacement head nor break
        # base tying; tied base checkpoints omit the duplicate saved head.
        assert loaded.get_input_embeddings().weight is initial.input_weight
        assert loaded.get_output_embeddings().weight is initial.output_weight
    elif kind.startswith("raw-") and not base_tied:
        # An already independent head requires no extra embedding-sized clone.
        assert loaded.get_output_embeddings().weight is initial.output_weight
    assert dict(initial.model._load_state_dict_pre_hooks) == initial.hooks
    assert len(trace.existing_hook_calls) == 1
    assert all(ref() is None for ref in trace.hooks)
    ids = torch.tensor([[6, 7, 8, 9], [9, 8, 7, 6]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected = fixture.reference(input_ids=ids, attention_mask=mask).logits
        actual = loaded(input_ids=ids, attention_mask=mask).logits
    error = float((actual - expected).abs().max())
    record_property("max_abs_logit_error", error)
    record_property("embedding_max_abs_error", float((loaded.get_input_embeddings().weight.detach() - fixture.reference.get_input_embeddings().weight.detach()).abs().max()))
    record_property("head_max_abs_error", float((loaded.get_output_embeddings().weight.detach() - fixture.reference.get_output_embeddings().weight.detach()).abs().max()))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loaded.get_input_embeddings().weight,
                               fixture.reference.get_input_embeddings().weight, rtol=0, atol=0)
    torch.testing.assert_close(loaded.get_output_embeddings().weight,
                               fixture.reference.get_output_embeddings().weight, rtol=0, atol=0)
    assert (loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight) is (
        base_tied and kind == "ordinary"
    )
    assert loaded.device.type == "cpu" and loaded.dtype == torch.float32
    assert not loaded.training and not loaded.generation_config.do_sample
    assert tokenizer.chat_template == fixture.tokenizer.chat_template
    assert tokenizer.padding_side == "left"


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("fail_at", ["before-state-load", "before-raw-copy", "after-raw-copy"])
def test_failed_real_adapter_load_removes_temporary_hooks_and_restores_base_tying(
    tmp_path, monkeypatch, tiny_model, tiny_tokenizer, evaluator, fail_at,
):
    import torch
    from transformers import Qwen3ForCausalLM

    fixture = saved_adapter(tmp_path, tiny_model, tiny_tokenizer,
                            kind="raw-both", base_tied=True, serialization="safetensors")
    failure = RuntimeError("EXPLICIT injected adapter-load failure: " + fail_at)
    trace = observe_reloads(monkeypatch, fail_at=fail_at, failure=failure)
    real_post_load = Qwen3ForCausalLM._load_from_state_dict

    def post_copy(module, state_dict, prefix, *args):
        # Run the actual head/input copies before failing. A child post-hook
        # raises at the end of this base's recursive torch load_state_dict.
        result = real_post_load(module, state_dict, prefix, *args)
        if fail_at == "after-raw-copy":
            def fail_after_copy(head, incompatible_keys):
                trace.split_before_failure = (
                    module.get_input_embeddings().weight is not module.get_output_embeddings().weight
                )
                torch.testing.assert_close(head.weight, fixture.reference.get_output_embeddings().weight, rtol=0, atol=0)
                raise failure
            module.get_output_embeddings().register_load_state_dict_post_hook(fail_after_copy)
        return result

    monkeypatch.setattr(Qwen3ForCausalLM, "_load_from_state_dict", post_copy)
    with pytest.raises(RuntimeError) as caught:
        load_evaluator(evaluator).load_model(str(fixture.adapter))
    assert caught.value is failure
    assert len(trace.bases) == 1 and len(trace.reads) == 1
    initial = trace.bases[0]
    assert initial.model.config.tie_word_embeddings is True
    assert initial.model.get_input_embeddings().weight is initial.input_weight
    assert initial.model.get_output_embeddings().weight is initial.output_weight
    assert initial.input_weight is initial.output_weight
    assert dict(initial.model._load_state_dict_pre_hooks) == initial.hooks
    assert len(trace.existing_hook_calls) == int(fail_at != "before-state-load")
    if fail_at != "before-state-load":
        assert trace.split_before_failure
    # Release both the exception's and pytest's retained traceback, then collect
    # torch's failed recursive-load closure cycle (GC is test-only). The base
    # model remains alive; it must not retain the temporary callback.
    caught.value.__traceback__ = None
    del caught
    gc.collect()
    assert all(ref() is None for ref in trace.hooks)


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("targets", [("embed_tokens",), ("lm_head",), ("embed_tokens", "lm_head")])
def test_raw_layers_inside_actual_lora_wrappers_restore_base_layer_parameters(
    tmp_path, monkeypatch, tiny_model, tiny_tokenizer, record_property, evaluator, targets,
):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    base = tmp_path / "base"
    tiny_model.save_pretrained(base)
    tiny_tokenizer.save_pretrained(base)
    tiny_model.name_or_path = str(base)
    tiny_model.config.tie_word_embeddings = False
    tiny_model.lm_head.weight = torch.nn.Parameter(tiny_model.lm_head.weight.detach().clone())
    reference = get_peft_model(tiny_model, LoraConfig(
        r=2, lora_alpha=4, target_modules=["v_proj", *targets], task_type="CAUSAL_LM",
    ))
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_B" in name or "lora_embedding_A" in name:
                parameter.fill_(0.09)
        if "embed_tokens" in targets:
            reference.get_input_embeddings().weight.add_(torch.randn_like(reference.get_input_embeddings().weight) * 0.07)
        if "lm_head" in targets:
            reference.get_output_embeddings().weight.add_(torch.randn_like(reference.get_output_embeddings().weight) * 0.11)
    reference.eval()
    adapter = tmp_path / "adapter"
    reference.save_pretrained(adapter, save_embedding_layers=True)
    tiny_tokenizer.save_pretrained(adapter)
    saved = load_file(adapter / "adapter_model.safetensors", device="cpu")
    for target, key in (("embed_tokens", EMBEDDING), ("lm_head", HEAD)):
        assert (key.removesuffix("weight") + "base_layer.weight" in saved) == (target in targets)
        assert key not in saved
    trace = observe_reloads(monkeypatch)
    loaded, _ = load_evaluator(evaluator).load_model(str(adapter))
    assert len(trace.reads) == 1 and len(trace.bases) == 1
    assert trace.reads[0][1]["device"] == "cpu"
    assert loaded.get_input_embeddings().weight is not loaded.get_output_embeddings().weight
    assert loaded.config.tie_word_embeddings is True
    assert dict(trace.bases[0].model._load_state_dict_pre_hooks) == trace.bases[0].hooks
    assert all(ref() is None for ref in trace.hooks)
    torch.testing.assert_close(loaded.get_input_embeddings().weight, reference.get_input_embeddings().weight, rtol=0, atol=0)
    torch.testing.assert_close(loaded.get_output_embeddings().weight, reference.get_output_embeddings().weight, rtol=0, atol=0)
    ids = torch.tensor([[6, 7, 8, 9], [9, 8, 7, 6]])
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits
        actual = loaded(input_ids=ids).logits
    record_property("max_abs_logit_error", float((actual - expected).abs().max()))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("evaluator", ["eval.py", "eval_retrieval.py"])
@pytest.mark.parametrize("serialization", ["safetensors", "bin"])
@pytest.mark.parametrize("base_tied", [True, False], ids=["tied-base", "untied-base"])
@pytest.mark.parametrize("kind", ["modules-both", "modules-embedding-only", "modules-head-only"])
def test_actual_modules_to_save_with_raw_backups_preserves_every_saved_layer(
    tmp_path, monkeypatch, tiny_model, tiny_tokenizer, record_property,
    evaluator, serialization, base_tied, kind,
):
    import torch

    fixture = saved_adapter(tmp_path, tiny_model, tiny_tokenizer,
                            kind=kind, base_tied=base_tied, serialization=serialization,
                            raw_backups=True)
    trace = observe_reloads(monkeypatch)
    loaded, tokenizer = load_evaluator(evaluator).load_model(str(fixture.adapter))
    assert len(trace.reads) == 1 and trace.reads[0][1]["device"] == "cpu"
    assert loaded.config.tie_word_embeddings is base_tied
    assert tokenizer.chat_template == fixture.tokenizer.chat_template
    assert dict(trace.bases[0].model._load_state_dict_pre_hooks) == trace.bases[0].hooks
    assert all(ref() is None for ref in trace.hooks)
    ids = torch.tensor([[6, 7, 8, 9], [9, 8, 7, 6]])
    with torch.inference_mode():
        expected = fixture.reference(input_ids=ids).logits
        actual = loaded(input_ids=ids).logits
    record_property("max_abs_logit_error", float((actual - expected).abs().max()))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        actual_layer = getattr(loaded, getter)()
        expected_layer = getattr(fixture.reference, getter)()
        torch.testing.assert_close(actual_layer.weight, expected_layer.weight, rtol=0, atol=0)
        if hasattr(expected_layer, "original_module"):
            # PEFT can save original_module raw alongside the active wrapper.
            # It must not overwrite the other layer through a tied base alias.
            torch.testing.assert_close(actual_layer.original_module.weight,
                                       expected_layer.original_module.weight, rtol=0, atol=0)
