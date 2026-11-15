"""Review regressions: inspect loaded layers, not quantizer/family labels."""

from __future__ import annotations

import gc
import json
import sys
import weakref

import numpy as np
import pytest

from scripts.qmd_mlx.adapters.embedding import QwenEmbeddingAdapter
from scripts.qmd_mlx.protocol import InvalidInputError, ModelUnavailableError
from test_audit_embedding_adapters import ModelSpy, batch, dependencies  # noqa: F401


FAMILY = "Qwen/Qwen3-Embedding-0.6B"


def qwen_config(dependencies, **overrides):
    dependencies.update_config(
        model_type="qwen3", architectures=["Qwen3ForCausalLM"], **overrides,
    )


def loaded_model_change(monkeypatch, change):
    original = sys.modules["mlx_lm"].load

    def load(*args, **kwargs):
        model, tokenizer, config = original(*args, **kwargs)
        change(model, config)
        return model, tokenizer, config

    monkeypatch.setattr(sys.modules["mlx_lm"], "load", load)


def assert_unpublished(adapter):
    assert not adapter.is_loaded()
    assert adapter.model is adapter.tokenizer is adapter.raw_hf_tokenizer is None
    descriptor = adapter.get_descriptor()
    assert descriptor["revision"] == descriptor["dtype"] == descriptor["quantization"] == "unknown"
    assert descriptor["nativeDimensions"] == 0
    assert not adapter.supports_mrl


@pytest.mark.parametrize("actual", [
    {"bits": 8, "group_size": 32, "mode": "affine"},
    {"bits": 8, "group_size": 64, "mode": "affine"},
    {"bits": 4, "group_size": 32, "mode": "affine"},
    {"bits": 4, "group_size": 64, "mode": "mxfp4"},
    {"bits": 4, "group_size": 64, "mode": "unknown"},
    {"bits": 4, "group_size": 64},
    {"bits": 4, "group_size": True, "mode": "affine"},
    {"bits": "4", "group_size": 64, "mode": "affine"},
    {},
])
def test_stale_config_cannot_relabel_actual_quantized_layer(dependencies, monkeypatch, actual):
    qwen_config(dependencies, quantization={"bits": 4, "group_size": 64})
    loaded_model_change(
        monkeypatch, lambda model, config: model.quantization_metadata.update(projection=actual),
    )
    adapter = QwenEmbeddingAdapter(FAMILY + "-4bit", quantization="4bit")
    with pytest.raises(ModelUnavailableError, match="quantization"):
        adapter.load()
    assert_unpublished(adapter)


@pytest.mark.parametrize("field,value", [("bits", 8), ("group_size", 32), ("mode", "mxfp4")])
def test_every_quantized_layer_must_match_uniform_policy(dependencies, monkeypatch, field, value):
    qwen_config(dependencies, quantization={"bits": 4, "group_size": 64})

    def mixed(model, config):
        model.weights["late_projection"] = {
            key: array.copy() for key, array in model.weights["projection"].items()
        }
        model.quantization_metadata["late_projection"] = dict(
            model.quantization_metadata["projection"], **{field: value},
        )

    loaded_model_change(monkeypatch, mixed)
    adapter = QwenEmbeddingAdapter(FAMILY)
    with pytest.raises(ModelUnavailableError, match="quantization"):
        adapter.load()
    assert_unpublished(adapter)


def test_uninspectable_packed_weight_module_fails_closed(dependencies, monkeypatch):
    qwen_config(dependencies, quantization={"bits": 4, "group_size": 64})
    monkeypatch.setattr(ModelSpy, "named_modules", None)
    adapter = QwenEmbeddingAdapter(FAMILY)
    with pytest.raises(ModelUnavailableError, match="quantization"):
        adapter.load()
    assert_unpublished(adapter)


def test_all_packed_weights_must_have_corresponding_layer_metadata(dependencies, monkeypatch):
    qwen_config(dependencies, quantization={"bits": 4, "group_size": 64})
    original = ModelSpy.named_modules
    monkeypatch.setattr(ModelSpy, "named_modules", lambda model: original(model)[:1])
    adapter = QwenEmbeddingAdapter(FAMILY)
    with pytest.raises(ModelUnavailableError, match="quantization"):
        adapter.load()
    assert_unpublished(adapter)


@pytest.mark.parametrize("field,value", [("group_size", 32), ("mode", "mxfp4")])
def test_loader_policy_must_agree_beyond_bit_count(dependencies, monkeypatch, field, value):
    qwen_config(dependencies, quantization={"bits": 4, "group_size": 64})
    loaded_model_change(
        monkeypatch,
        lambda model, config: config.update(quantization=dict(config["quantization"], **{field: value})),
    )
    adapter = QwenEmbeddingAdapter(FAMILY)
    with pytest.raises(ModelUnavailableError, match="quantization"):
        adapter.load()
    assert_unpublished(adapter)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_matching_loaded_uniform_affine_policy_is_published(dependencies, monkeypatch, bits, group_size):
    qwen_config(dependencies, quantization={"bits": bits, "group_size": group_size})

    def another_layer(model, config):
        model.weights["other_projection"] = {
            key: array.copy() for key, array in model.weights["projection"].items()
        }
        model.quantization_metadata["other_projection"] = dict(model.quantization_metadata["projection"])
        # Affine quantization still legitimately contains floating norms/heads.
        model.weights["norm"] = {"weight": np.ones((4,), dtype=np.float16)}

    loaded_model_change(monkeypatch, another_layer)
    adapter = QwenEmbeddingAdapter(FAMILY, quantization=f"{bits}bit")
    adapter.load()
    assert adapter.is_loaded()
    assert adapter.get_descriptor()["quantization"] == f"{bits}bit"
    assert adapter.get_descriptor()["dtype"] == "float32"
    assert dependencies.models[-1].weights["projection"]["weight"].dtype == np.uint32


def test_quantization_verification_error_does_not_retain_failed_model(dependencies, monkeypatch):
    qwen_config(dependencies, quantization={"bits": 4, "group_size": 64})
    loaded_model_change(
        monkeypatch,
        lambda model, config: model.quantization_metadata["projection"].update(bits=8, group_size=32),
    )
    adapter = QwenEmbeddingAdapter(FAMILY)
    with pytest.raises(ModelUnavailableError, match="quantization") as error:
        adapter.load()
    model_ref = weakref.ref(dependencies.models[-1])
    dependencies.models.clear()
    gc.collect()
    assert error.value.__cause__ is error.value.__context__ is None
    assert model_ref() is None
    assert_unpublished(adapter)


def test_family_name_does_not_grant_mrl_before_verifying_assets():
    assert not QwenEmbeddingAdapter(FAMILY).supports_mrl


@pytest.mark.parametrize("model_type,architectures", [
    ("qwen2", ["Qwen2ForCausalLM"]),
    ("qwen", ["QWenLMHeadModel"]),
    ("qwen3", ["Qwen2ForCausalLM"]),
    ("qwen3", []),
    ("qwen3", None),
    ("qwen3", "Qwen3ForCausalLM"),
    ("qwen3", ["Qwen3ForCausalLM", "UnqualifiedModel"]),
])
def test_embedding_family_label_cannot_grant_mrl_for_other_architecture(
    dependencies, model_type, architectures,
):
    dependencies.update_config(model_type=model_type, architectures=architectures)
    adapter = QwenEmbeddingAdapter(FAMILY)
    adapter.load()
    assert not adapter.supports_mrl
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.get_descriptor(2)
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.forward_batch(batch([[1, 2]]), requested_dims=2)
    assert adapter.forward_batch(batch([[1, 2]]), requested_dims=4).shape == (1, 4)


@pytest.mark.parametrize("metadata", [
    {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
    {"architectures": []},
    {"_name_or_path": "Qwen/Qwen3-0.6B"},
])
def test_mrl_must_also_match_loaded_configuration(dependencies, monkeypatch, metadata):
    qwen_config(dependencies)
    loaded_model_change(monkeypatch, lambda model, config: config.update(metadata))
    adapter = QwenEmbeddingAdapter(FAMILY)
    adapter.load()
    assert not adapter.supports_mrl
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.get_descriptor(2)


def test_mrl_must_match_actual_loaded_module_architecture(dependencies, monkeypatch):
    qwen_config(dependencies)
    loaded_model_change(monkeypatch, lambda model, config: setattr(model, "model_type", "qwen2"))
    adapter = QwenEmbeddingAdapter(FAMILY)
    adapter.load()
    assert not adapter.supports_mrl
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.get_descriptor(2)


@pytest.mark.parametrize("asset_name", ["Qwen/Qwen3-0.6B", "unqualified-embedding-model"])
def test_conflicting_asset_provenance_prevents_mrl(dependencies, asset_name):
    qwen_config(dependencies, _name_or_path=asset_name)
    adapter = QwenEmbeddingAdapter(FAMILY)
    adapter.load()
    assert not adapter.supports_mrl
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.get_descriptor(2)


@pytest.mark.parametrize("name", [
    "Qwen/Qwen3-0.6B", "someone/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-Unqualified",
])
def test_unknown_remote_family_has_no_mrl_without_verified_provenance(dependencies, name):
    qwen_config(dependencies)
    adapter = QwenEmbeddingAdapter(name)
    adapter.load()
    assert not adapter.supports_mrl
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.get_descriptor(2)


@pytest.mark.parametrize("asset_name,qualified", [(None, False), (FAMILY, True), ("Qwen/Qwen3-0.6B", False)])
def test_local_directory_name_is_not_embedding_training_metadata(dependencies, tmp_path, asset_name, qualified):
    qwen_config(dependencies)
    config = dict(dependencies.config)
    if asset_name is not None:
        config["_name_or_path"] = asset_name
    local = tmp_path / "Qwen3-Embedding-0.6B"
    local.mkdir()
    (local / "config.json").write_text(json.dumps(config))
    (local / "model.safetensors").write_bytes(b"fixture only")
    # The fixture loader returns the config it was given, like the real loader.
    dependencies.config.clear()
    dependencies.config.update(config)
    adapter = QwenEmbeddingAdapter(str(local))
    adapter.load()
    assert adapter.supports_mrl is qualified
    if qualified:
        assert adapter.get_descriptor(2)["outputDimensions"] == 2
    else:
        with pytest.raises(InvalidInputError, match="Matryoshka"):
            adapter.get_descriptor(2)


@pytest.mark.parametrize("name", [FAMILY, "mlx-community/Qwen3-Embedding-0.6B-4bit"])
def test_verified_qwen3_embedding_mrl_retained_across_idle_eviction(dependencies, name):
    qwen_config(dependencies)
    adapter = QwenEmbeddingAdapter(name)
    adapter.load()
    assert adapter.supports_mrl
    descriptor = adapter.get_descriptor(2)
    adapter.unload()
    assert adapter.get_descriptor(2) == descriptor
    adapter.load()
    assert adapter.supports_mrl
    result = adapter.forward_batch(batch([[1, 2], [3]]), requested_dims=2)
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), [1, 1], atol=1e-7)


def test_failed_reload_clears_previously_verified_mrl(dependencies, monkeypatch):
    qwen_config(dependencies)
    adapter = QwenEmbeddingAdapter(FAMILY)
    adapter.load()
    assert adapter.supports_mrl
    adapter.unload()
    monkeypatch.setattr(ModelSpy, "set_dtype", lambda *args: (_ for _ in ()).throw(RuntimeError("cast failed")))
    with pytest.raises(ModelUnavailableError, match="cast failed"):
        adapter.load()
    assert_unpublished(adapter)
