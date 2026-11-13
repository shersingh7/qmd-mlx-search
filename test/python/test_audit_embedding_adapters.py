"""W04: offline adapter/tokenizer contracts, using CPU dependency spies only."""

from __future__ import annotations

import gc
import json
import sys
import types
import weakref
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.qmd_mlx.adapters import embedding
from scripts.qmd_mlx.adapters.embedding import (
    BertEmbeddingAdapter,
    NomicEmbeddingAdapter,
    QwenEmbeddingAdapter,
    SyntheticEmbeddingAdapter,
    resolve_embedding_adapter,
)
from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch, TokenizerHelper
from scripts.qmd_mlx.protocol import InvalidInputError, ModelUnavailableError, UnsupportedModelError


class NumpyMLX:
    """No Metal operations; bf16 is a distinct spy target, not a precision model."""

    float32 = np.float32
    float16 = np.float16
    bfloat16 = np.float64
    int32 = np.int32
    uint32 = np.uint32
    floating = np.floating
    array = staticmethod(np.array)
    zeros = staticmethod(np.zeros)
    arange = staticmethod(np.arange)
    sqrt = staticmethod(np.sqrt)
    maximum = staticmethod(np.maximum)
    expand_dims = staticmethod(np.expand_dims)
    issubdtype = staticmethod(np.issubdtype)
    eval = staticmethod(lambda *args: None)
    get_active_memory = staticmethod(lambda: 0)

    def __init__(self):
        self.reduction_dtypes = []

    def sum(self, value, **kwargs):
        self.reduction_dtypes.append(value.dtype)
        return np.sum(value, **kwargs)


class TokenizerSpy:
    pad_token_id = 0
    eos_token_id = 99
    padding_side = "left"
    model_max_length = 512

    def __init__(self):
        self.batch_calls = []
        self.encode_calls = []

    @staticmethod
    def ids(text):
        return [101] + [ord(word[0]) for word in text.split()] + [102]

    def __call__(self, texts, **kwargs) -> Any:
        self.batch_calls.append((list(texts), kwargs))
        return {"input_ids": [self.ids(text) for text in texts]}

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return self.ids(text)


class ModelSpy:
    def __init__(self, config, *, quantized=False):
        self.args = SimpleNamespace(hidden_size=config["hidden_size"], model_type=config["model_type"])
        self.model_type = config["model_type"]
        self.weights: dict[str, dict[str, np.ndarray]] = {"projection": {"weight": np.ones((4, 4), dtype=np.float16)}}
        self.quantization_metadata = {}
        if quantized:
            info = config["quantization"]
            self.quantization_metadata["projection"] = {
                "bits": info["bits"], "group_size": info["group_size"], "mode": info.get("mode", "affine"),
            }
            self.weights["projection"] = {
                "weight": np.ones((4, info["group_size"] * info["bits"] // 32), dtype=np.uint32),
                "scales": np.ones((4, 1), dtype=np.float16),
                "biases": np.ones((4, 1), dtype=np.float16),
            }
        self.dtype_calls = []
        self.forward_calls = []
        self.eval_calls = 0
        self.model = self.backbone
        self.hidden = None

    def parameters(self):
        return self.weights

    def named_modules(self):
        # These fields belong to the loaded layers, independently of config.
        return [("", self)] + [
            (name, SimpleNamespace(**values, **self.quantization_metadata.get(name, {})))
            for name, values in self.weights.items()
        ]

    def set_dtype(self, dtype):
        self.dtype_calls.append(dtype)
        for module in self.weights.values():
            for name, value in module.items():
                if np.issubdtype(value.dtype, np.floating):
                    module[name] = value.astype(dtype)

    def eval(self):
        self.eval_calls += 1

    def backbone(self, ids):
        self.forward_calls.append((ids.copy(), None))
        if self.hidden is not None:
            return self.hidden[: len(ids), : ids.shape[1]].copy()
        return np.stack([ids, ids + 1, ids + 2, ids + 3], axis=-1).astype(np.float16)

    def __call__(self, ids, attention_mask=None, token_type_ids=None):
        assert token_type_ids is not None
        np.testing.assert_array_equal(token_type_ids, np.zeros_like(ids))
        hidden = self.backbone(ids)
        self.forward_calls[-1] = (ids.copy(), attention_mask.copy() if attention_mask is not None else None)
        return hidden, None


@pytest.fixture
def dependencies(monkeypatch, tmp_path):
    """Replace loaders before calling any adapter; never import/load model libs."""
    fake_mx = NumpyMLX()
    monkeypatch.setattr(embedding, "mx", fake_mx, raising=False)
    monkeypatch.setattr(embedding, "_MLX_AVAILABLE", True)
    config = {
        "model_type": "bert",
        "architectures": ["BertModel"],
        "hidden_size": 4,
        "max_position_embeddings": 512,
        "torch_dtype": "float16",
    }
    snapshot = tmp_path / "hub" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "model.safetensors").write_bytes(b"fixture weights, never loaded")
    (snapshot / "tokenizer.json").write_text('{"fixture": true}')
    calls = SimpleNamespace(snapshot=[], tokenizer=[], bert=[], qwen=[])
    tokenizer = TokenizerSpy()
    models = []

    def snapshot_download(**kwargs):
        calls.snapshot.append(kwargs)
        return str(snapshot)

    def from_tokenizer(path, **kwargs):
        calls.tokenizer.append((path, kwargs))
        return tokenizer

    def from_bert(path):
        calls.bert.append(path)
        model = ModelSpy(config)
        models.append(model)
        return model

    def load_qwen(path, **kwargs):
        calls.qwen.append((path, kwargs))
        model = ModelSpy(config, quantized=bool(config.get("quantization")))
        models.append(model)
        return model, SimpleNamespace(_tokenizer=tokenizer), dict(config)

    modules = {
        "huggingface_hub": {"snapshot_download": snapshot_download},
        "transformers": {"AutoTokenizer": SimpleNamespace(from_pretrained=from_tokenizer)},
        "mlx_embedding_models": {},
        "mlx_embedding_models.model": {"Bert": SimpleNamespace(from_pretrained=from_bert)},
        "mlx_embedding_models.nomic_model": {"NomicBert": SimpleNamespace(from_pretrained=from_bert)},
        "mlx_lm": {"load": load_qwen},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def update_config(**kwargs):
        config.update(kwargs)
        (snapshot / "config.json").write_text(json.dumps(config))

    return SimpleNamespace(
        mx=fake_mx, config=config, snapshot=snapshot, calls=calls,
        tokenizer=tokenizer, models=models, update_config=update_config,
    )


def batch(ids):
    return TokenizedBatch(ids, list(map(len, ids)), list(range(len(ids))), 0)


def test_bert_resolves_tokenizer_and_weights_to_one_snapshot(dependencies):
    adapter = BertEmbeddingAdapter("sentence-transformers/all-MiniLM-L6-v2", revision="release")
    adapter.load()
    assert dependencies.calls.snapshot[0]["revision"] == "release"
    path = str(dependencies.snapshot)
    assert dependencies.calls.bert == [path]
    assert dependencies.calls.tokenizer[0][0] == path
    assert dependencies.calls.tokenizer[0][1]["local_files_only"] is True
    assert dependencies.calls.tokenizer[0][1]["trust_remote_code"] is False
    assert dependencies.models[0].forward_calls == []  # No dummy tokenization/inference.
    assert dependencies.tokenizer.batch_calls == dependencies.tokenizer.encode_calls == []
    descriptor = adapter.get_descriptor()
    assert descriptor["revision"] == "a" * 40
    assert descriptor["dtype"] == "float32"
    assert descriptor["quantization"] == "none"
    assert descriptor["maxTokens"] == 512


def test_qwen_resolves_once_and_passes_explicit_trust(dependencies):
    dependencies.update_config(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    adapter = QwenEmbeddingAdapter("Qwen/Qwen3-Embedding-0.6B", revision="release")
    adapter.load()
    path, options = dependencies.calls.qwen[0]
    assert path == str(dependencies.snapshot)
    assert len(dependencies.calls.snapshot) == 1
    assert options["trust_remote_code"] is False
    assert options["tokenizer_config"]["trust_remote_code"] is False
    assert options["tokenizer_config"]["local_files_only"] is True
    assert adapter.get_descriptor()["revision"] == "a" * 40
    assert dependencies.tokenizer.padding_side == "right"


@pytest.mark.parametrize("adapter_type", [BertEmbeddingAdapter, QwenEmbeddingAdapter])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_actual_floating_dtype_applied_and_reported(dependencies, adapter_type, dtype):
    name = "sentence-transformers/all-MiniLM-L6-v2"
    if adapter_type is QwenEmbeddingAdapter:
        dependencies.update_config(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
        name = "Qwen/Qwen3-Embedding-0.6B"
    adapter = adapter_type(name, dtype_str=dtype)
    adapter.load()
    model = dependencies.models[-1]
    assert model.dtype_calls == [getattr(dependencies.mx, dtype)]
    assert model.eval_calls == 1
    assert adapter.get_descriptor()["dtype"] == dtype
    assert adapter.get_descriptor()["quantization"] == "none"  # Not stale torch_dtype.


def test_unloaded_descriptor_never_echoes_unverified_requests():
    adapter = BertEmbeddingAdapter("bert", dtype_str="float16", revision="moving-branch", quantization="4bit")
    descriptor = adapter.get_descriptor()
    assert descriptor["revision"] == "unknown"
    assert descriptor["quantization"] == "unknown"
    assert descriptor["dtype"] == "unknown"


@pytest.mark.parametrize("dtype", ["int8", "", "float64"])
def test_unsupported_dtype_fails_before_loading(dependencies, dtype):
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2", dtype_str=dtype)
    with pytest.raises(ModelUnavailableError, match="Unsupported compute dtype"):
        adapter.load()
    assert not dependencies.calls.bert
    assert not adapter.is_loaded()


@pytest.mark.parametrize("quantization", ["4bit", "8bit", "fp16", "arbitrary"])
def test_bert_rejects_quantization_instead_of_relabeling(dependencies, quantization):
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2", quantization=quantization)
    with pytest.raises(UnsupportedModelError, match="quantization"):
        adapter.load()
    assert not dependencies.calls.bert
    assert not adapter.is_loaded()


def test_qwen_prequantized_weights_match_request_and_keep_integer_storage(dependencies):
    dependencies.update_config(
        model_type="qwen3", architectures=["Qwen3ForCausalLM"],
        quantization={"bits": 4, "group_size": 64},
    )
    adapter = QwenEmbeddingAdapter("Qwen/Qwen3-Embedding-0.6B-4bit", quantization="4bit")
    adapter.load()
    assert dependencies.models[-1].weights["projection"]["weight"].dtype == np.uint32
    assert dependencies.models[-1].weights["projection"]["scales"].dtype == np.float32
    assert adapter.get_descriptor()["quantization"] == "4bit"
    assert adapter.get_descriptor()["dtype"] == "float32"


@pytest.mark.parametrize("quantization", ["8bit", "none", "unsupported"])
def test_qwen_rejects_unapplied_or_unsupported_quantization(dependencies, quantization):
    dependencies.update_config(
        model_type="qwen3", architectures=["Qwen3ForCausalLM"],
        quantization={"bits": 4, "group_size": 64},
    )
    adapter = QwenEmbeddingAdapter("Qwen/Qwen3-Embedding-0.6B-4bit", quantization=quantization)
    with pytest.raises(UnsupportedModelError, match="quantization"):
        adapter.load()
    assert not adapter.is_loaded()
    assert adapter.raw_hf_tokenizer is adapter.tokenizer is adapter.model is None


def test_failed_dtype_application_does_not_publish_partial_state(dependencies, monkeypatch):
    def fail(self, dtype):
        raise RuntimeError("cast failed")

    monkeypatch.setattr(ModelSpy, "set_dtype", fail)
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2")
    with pytest.raises(ModelUnavailableError, match="cast failed"):
        adapter.load()
    assert not adapter.is_loaded()
    assert adapter.raw_hf_tokenizer is adapter.tokenizer is adapter.model is None
    assert adapter.native_dims == 0
    assert adapter.get_descriptor()["revision"] == "unknown"


@pytest.mark.parametrize("adapter_type", [BertEmbeddingAdapter, QwenEmbeddingAdapter])
def test_retained_load_error_cannot_retain_failed_weights(dependencies, monkeypatch, adapter_type):
    name = "all-MiniLM-L6-v2"
    if adapter_type is QwenEmbeddingAdapter:
        dependencies.update_config(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
        name = "Qwen/Qwen3-Embedding-0.6B"
    adapter = adapter_type(name)
    monkeypatch.setattr(ModelSpy, "set_dtype", lambda self, dtype: (_ for _ in ()).throw(RuntimeError("dtype failure")))
    with pytest.raises(ModelUnavailableError, match="dtype failure") as retained:
        adapter.load()
    model_ref = weakref.ref(dependencies.models[0])
    dependencies.models.clear()
    gc.collect()
    assert retained.value.__context__ is None and retained.value.__cause__ is None
    assert model_ref() is None


def test_noop_dtype_cast_is_detected(dependencies, monkeypatch):
    monkeypatch.setattr(ModelSpy, "set_dtype", lambda self, dtype: None)
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2", dtype_str="float32")
    with pytest.raises(ModelUnavailableError, match="dtype"):
        adapter.load()
    assert not adapter.is_loaded()


def test_failure_retry_reloads_and_unload_retains_only_verified_metadata(dependencies, monkeypatch):
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2")
    original = ModelSpy.eval
    monkeypatch.setattr(ModelSpy, "eval", lambda self: (_ for _ in ()).throw(RuntimeError("eval failed")))
    with pytest.raises(ModelUnavailableError, match="eval failed"):
        adapter.load()
    assert adapter.model is adapter.raw_hf_tokenizer is None
    monkeypatch.setattr(ModelSpy, "eval", original)
    adapter.load()
    assert adapter.is_loaded()
    assert len(dependencies.calls.bert) == 2
    descriptor = adapter.get_descriptor()
    adapter.unload()
    assert adapter.model is adapter.raw_hf_tokenizer is adapter.tokenizer is None
    assert not adapter.is_loaded()
    assert adapter.get_descriptor() == descriptor
    adapter.load()
    assert adapter.get_descriptor() == descriptor
    assert len(dependencies.calls.bert) == 3


def test_local_identity_changes_with_weights_and_tokenizer(dependencies, tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()
    (local / "config.json").write_text(json.dumps(dependencies.config))
    weights = local / "model.safetensors"
    weights.write_bytes(b"weights A")
    tokenizer = local / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer A")
    revisions = []
    for mutate in (None, weights, tokenizer):
        if mutate:
            mutate.write_bytes(mutate.read_bytes() + b"changed")
        adapter = BertEmbeddingAdapter(str(local))
        adapter.load()
        revisions.append(adapter.get_descriptor()["revision"])
    assert len(set(revisions)) == 3
    assert all(revision.startswith("local-sha256:") for revision in revisions)
    assert not dependencies.calls.snapshot
    assert dependencies.calls.bert == [str(local)] * 3


def test_local_path_cannot_silently_ignore_requested_revision(dependencies, tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()
    (local / "config.json").write_text(json.dumps(dependencies.config))
    adapter = BertEmbeddingAdapter(str(local), revision="another-revision")
    with pytest.raises(ModelUnavailableError, match="revision"):
        adapter.load()
    assert not dependencies.calls.bert


def test_remote_resolution_requires_immutable_snapshot(dependencies, monkeypatch, tmp_path):
    mutable = tmp_path / "main"
    mutable.mkdir()
    (mutable / "config.json").write_text(json.dumps(dependencies.config))
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", lambda **kwargs: str(mutable))
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2", revision="main")
    with pytest.raises(ModelUnavailableError, match="immutable"):
        adapter.load()
    assert not dependencies.calls.bert


def test_qualified_bge_uses_cls_not_mean(dependencies):
    adapter = resolve_embedding_adapter("BAAI/bge-small-en-v1.5")
    assert isinstance(adapter, BertEmbeddingAdapter)
    adapter.load()
    assert adapter.get_descriptor()["pooling"] == "cls"
    result = adapter.forward_batch(batch([[1, 2, 3], [4]]))
    expected = np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    assert dependencies.models[-1].forward_calls[-1][1].tolist() == [[1, 1, 1], [1, 0, 0]]


@pytest.mark.parametrize("name,model_type", [
    ("BAAI/bge-m3", "xlm-roberta"),
    ("BAAI/bge-small-en-v1.5", "qwen3"),
    ("sentence-transformers/all-MiniLM-L6-v2", "nomic_bert"),
])
def test_incompatible_encoder_architecture_fails_before_weights(dependencies, name, model_type):
    dependencies.update_config(model_type=model_type)
    adapter = resolve_embedding_adapter(name)
    with pytest.raises(UnsupportedModelError, match="architecture"):
        adapter.load()
    assert not dependencies.calls.bert


@pytest.mark.parametrize("name", ["google/embeddinggemma-300m", "google/gemma-3-1b-it", "nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v2-moe"])
def test_unqualified_gemma_and_nomic_fail_explicitly(name):
    with pytest.raises(UnsupportedModelError):
        resolve_embedding_adapter(name)


def test_direct_nomic_adapter_does_not_bypass_qualification(dependencies):
    with pytest.raises(UnsupportedModelError, match="Nomic"):
        NomicEmbeddingAdapter("nomic-ai/nomic-embed-text-v1.5").load()


def test_qwen_cannot_bypass_architecture_check(dependencies):
    dependencies.update_config(model_type="gemma3", architectures=["Gemma3ForCausalLM"])
    adapter = QwenEmbeddingAdapter("Qwen/Qwen3-Embedding-0.6B")
    with pytest.raises(UnsupportedModelError, match="architecture"):
        adapter.load()
    assert not dependencies.calls.qwen


def test_minilm_float32_mean_pooling_padding_order_and_owned_output(dependencies):
    adapter = BertEmbeddingAdapter("sentence-transformers/all-MiniLM-L6-v2", dtype_str="float16")
    adapter.load()
    first = adapter.forward_batch(batch([[1, 2, 3], [4]]))
    expected = np.array([[2, 3, 4, 5], [4, 5, 6, 7]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(first, expected, atol=1e-7)
    assert first.dtype == np.float32
    assert first.flags.owndata and first.flags.c_contiguous and first.flags.writeable
    snapshot = first.copy()
    single = adapter.forward_batch(batch([[4]]))
    np.testing.assert_allclose(single[0], first[1], atol=1e-7)
    adapter.unload()
    np.testing.assert_array_equal(first, snapshot)
    first[0, 0] = 0  # Owned, mutable CPU storage survives model unload.
    assert all(dtype == np.float32 for dtype in dependencies.mx.reduction_dtypes)


def test_minilm_float32_path_is_bitwise_equal_to_baseline_pooling(dependencies):
    adapter = BertEmbeddingAdapter("sentence-transformers/all-MiniLM-L6-v2", dtype_str="float32")
    adapter.load()
    hidden = np.random.default_rng(42).standard_normal((2, 5, 4)).astype(np.float32)
    dependencies.models[-1].hidden = hidden
    result = adapter.forward_batch(batch([[1, 2, 3, 4, 5], [6, 7]]))
    # Freeze the original MiniLM mean+norm order for float32, independently of
    # the adapter helper. FP16 receives the intentional accumulation correction.
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=np.float32)[..., None]
    pooled = np.sum(hidden * mask, axis=1) / np.maximum(np.sum(mask, axis=1), 1e-9)
    expected = pooled / np.maximum(np.sqrt(np.sum(pooled ** 2, axis=-1, keepdims=True)), 1e-12)
    np.testing.assert_array_equal(result, expected)


def test_mean_reduction_casts_before_half_precision_overflow(dependencies):
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2", dtype_str="float16")
    adapter.load()
    hidden = np.array([[[40000, 20000, 10000, 5000]] * 3], dtype=np.float16)
    dependencies.models[-1].hidden = hidden
    result = adapter.forward_batch(batch([[1, 2, 3]]))
    expected = hidden.astype(np.float32).mean(axis=1)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_qwen_exact_last_nonpad_and_known_mrl_reduction(dependencies):
    dependencies.update_config(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    adapter = QwenEmbeddingAdapter("Qwen/Qwen3-Embedding-0.6B")
    adapter.load()
    result = adapter.forward_batch(batch([[1, 2, 3], [4]]), requested_dims=2)
    expected = np.array([[3, 4], [4, 5]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    assert adapter.get_descriptor(2)["outputDimensions"] == 2
    assert adapter.forward_batch(batch([]), requested_dims=2).shape == (0, 2)


@pytest.mark.parametrize("dimensions", [0, -1, True, False, 1.5, float("nan"), float("inf"), 5, "2"])
def test_invalid_dimensions_never_silently_clamp_or_slice(dependencies, dimensions):
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2")
    adapter.load()
    with pytest.raises(InvalidInputError, match="dimensions"):
        adapter.get_descriptor(dimensions)
    with pytest.raises(InvalidInputError, match="dimensions"):
        adapter.forward_batch(batch([[1]]), dimensions)
    with pytest.raises(InvalidInputError, match="dimensions"):
        adapter.forward_batch(batch([]), dimensions)


@pytest.mark.parametrize("name", ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"])
def test_non_mrl_encoder_dimension_reduction_refused(dependencies, name):
    adapter = resolve_embedding_adapter(name)
    adapter.load()
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.forward_batch(batch([[1]]), requested_dims=2)
    assert adapter.forward_batch(batch([[1]]), requested_dims=4).shape == (1, 4)


def test_generic_qwen_has_no_unqualified_mrl_support(dependencies):
    dependencies.update_config(model_type="qwen2", architectures=["Qwen2ForCausalLM"])
    adapter = QwenEmbeddingAdapter("Qwen/Qwen2.5-0.5B")
    adapter.load()
    with pytest.raises(InvalidInputError, match="Matryoshka"):
        adapter.get_descriptor(2)


def test_synthetic_mrl_behavior_remains_explicit_and_validated():
    adapter = SyntheticEmbeddingAdapter("synthetic-qwen3-4b", dims=4)
    adapter.load()
    result = adapter.forward_batch(batch([[1, 2]]), requested_dims=2)
    assert result.shape == (1, 2)
    assert adapter.get_descriptor(2)["synthetic"] is True
    with pytest.raises(InvalidInputError, match="dimensions"):
        adapter.forward_batch(batch([[1]]), requested_dims=-1)


def test_batched_tokenizer_is_called_once_with_exact_ids_order_lengths():
    tokenizer = TokenizerSpy()
    texts = ["first text", "second", "first text"]
    result = TokenizerHelper.tokenize_once(texts, tokenizer, max_length=20)
    assert result.token_ids == [tokenizer.ids(text) for text in texts]
    assert result.lengths == [4, 3, 4]
    assert result.original_indices == [0, 1, 2]
    assert len(tokenizer.batch_calls) == 1 and not tokenizer.encode_calls
    options = tokenizer.batch_calls[0][1]
    assert options["padding"] is False and options["truncation"] is False
    assert options["add_special_tokens"] is True
    reordered = result.slice([1, 2, 0])
    assert reordered.original_indices == [1, 2, 0]
    assert reordered.token_ids == [result.token_ids[index] for index in [1, 2, 0]]
    texts[0] = "mutated caller list"
    assert result.texts[0] == "first text"


def test_encode_only_fallback_no_keyword_retry_or_retokenization():
    calls = []

    class EncodeOnly:
        eos_token_id = 99

        def encode(self, text):
            calls.append(text)
            return [len(text), 99]

    result = TokenizerHelper.tokenize_once(["one", "two words"], EncodeOnly())
    assert calls == ["one", "two words"]
    assert result.token_ids == [[3, 99], [9, 99]]
    assert result.pad_token_id == 99


def test_batch_encode_plus_fast_path_without_callable_interface():
    calls = []

    class BatchOnly:
        def batch_encode_plus(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return {"input_ids": [[index + 1] for index, _ in enumerate(texts)]}

    result = TokenizerHelper.tokenize_once(["one", "two"], BatchOnly())
    assert result.token_ids == [[1], [2]]
    assert len(calls) == 1
    assert calls[0][1]["truncation"] is False


def test_tokenizer_typeerror_never_causes_a_second_encoding():
    calls = []

    class BrokenEncode:
        def encode(self, text, add_special_tokens=True):
            calls.append(text)
            raise TypeError("failure after encoding")

    with pytest.raises(InvalidInputError, match="[Tt]okeniz"):
        TokenizerHelper.tokenize_once(["one"], BrokenEncode())
    assert calls == ["one"]


@pytest.mark.parametrize("texts", [["ok", ""], ["ok", "  "], ["ok", 42], [None]])
def test_validate_all_texts_before_tokenizing(texts):
    tokenizer = TokenizerSpy()
    with pytest.raises(InvalidInputError):
        TokenizerHelper.tokenize_once(texts, tokenizer)
    assert tokenizer.batch_calls == tokenizer.encode_calls == []


@pytest.mark.parametrize("ids", [[], [[1]], [[1], []], [[1], [True]], [[1], [2.5]], [[1], [-1]], [[1], [2**31]], [1, 2]])
def test_invalid_batch_tokenizer_output_fails_without_fallback(ids):
    tokenizer = TokenizerSpy()
    tokenizer.ids = lambda text: []

    class Malformed(TokenizerSpy):
        def __call__(self, texts, **kwargs):
            self.batch_calls.append((texts, kwargs))
            return {"input_ids": ids}

    malformed = Malformed()
    with pytest.raises(InvalidInputError, match="token"):
        TokenizerHelper.tokenize_once(["one", "two"], malformed)
    assert len(malformed.batch_calls) == 1
    assert not malformed.encode_calls


def test_no_silent_truncation_on_batch_fast_path():
    tokenizer = TokenizerSpy()
    with pytest.raises(InvalidInputError, match="exceeds max_length"):
        TokenizerHelper.tokenize_once(["one two three", "four"], tokenizer, max_length=4)
    assert len(tokenizer.batch_calls) == 1
    assert tokenizer.batch_calls[0][1]["truncation"] is False
    assert not tokenizer.encode_calls


def test_empty_tokenizer_input_never_invokes_encoding():
    tokenizer = TokenizerSpy()
    result = TokenizerHelper.tokenize_once([], tokenizer)
    assert len(result) == 0
    assert result.original_indices == result.lengths == []
    assert tokenizer.batch_calls == tokenizer.encode_calls == []


def test_numpy_batch_ids_are_copied_and_not_aliased():
    ids = np.array([[1, 2], [3, 4]], dtype=np.int64)

    class ArrayTokenizer(TokenizerSpy):
        def __call__(self, texts, **kwargs):
            return {"input_ids": ids}

    result = TokenizerHelper.tokenize_once(["one", "two"], ArrayTokenizer())
    ids[0, 0] = 999
    assert result.token_ids == [[1, 2], [3, 4]]
    assert result.lengths == [2, 2]


def test_batch_tokenizer_typeerror_never_retries_encode():
    class BrokenBatch(TokenizerSpy):
        def __call__(self, texts, **kwargs):
            self.batch_calls.append((texts, kwargs))
            raise TypeError("batch failed after doing work")

    tokenizer = BrokenBatch()
    with pytest.raises(InvalidInputError, match="[Tt]okeniz"):
        TokenizerHelper.tokenize_once(["one", "two"], tokenizer)
    assert len(tokenizer.batch_calls) == 1
    assert not tokenizer.encode_calls


@pytest.mark.parametrize("output", [{}, {"input_ids": [[1], [2]], "attention_mask": [[1], [0]]}])
def test_missing_ids_and_unexpected_padding_are_rejected(output):
    class BadBatch(TokenizerSpy):
        def __call__(self, texts, **kwargs):
            return output

    with pytest.raises(InvalidInputError, match="tokeniz"):
        TokenizerHelper.tokenize_once(["one", "two"], BadBatch())


@pytest.mark.parametrize("metadata", [
    {"bits": 3, "group_size": 64},
    {"bits": 4, "group_size": 64, "mode": "mxfp4"},
    {"bits": 4, "group_size": 64, "layer.0": {"bits": 8}},
    {"bits": 4, "group_size": True},
    {"quant_type": "4bit"},
])
def test_unqualified_quantization_modes_fail_before_weights(dependencies, metadata):
    dependencies.update_config(model_type="qwen3", quantization=metadata)
    adapter = QwenEmbeddingAdapter("Qwen/Qwen3-Embedding-0.6B")
    with pytest.raises(UnsupportedModelError, match="quantization"):
        adapter.load()
    assert not dependencies.calls.qwen


def test_quantization_config_must_agree_with_actual_storage(dependencies, monkeypatch):
    dependencies.update_config(model_type="qwen3", quantization={"bits": 4, "group_size": 64})
    original = ModelSpy.__init__

    def float_only(self, config, **kwargs):
        original(self, config, quantized=False)

    monkeypatch.setattr(ModelSpy, "__init__", float_only)
    adapter = QwenEmbeddingAdapter("Qwen/Qwen3-Embedding-0.6B-4bit")
    with pytest.raises(ModelUnavailableError, match="storage"):
        adapter.load()
    assert adapter.get_descriptor()["quantization"] == "unknown"
    assert adapter.model is adapter.tokenizer is None


def test_explicit_commit_mismatch_is_never_accepted(dependencies):
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2", revision="b" * 40)
    with pytest.raises(ModelUnavailableError, match="revision"):
        adapter.load()
    assert not dependencies.calls.bert


def test_local_mutation_during_load_rolls_back(dependencies, monkeypatch, tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()
    (local / "config.json").write_text(json.dumps(dependencies.config))
    weights = local / "model.safetensors"
    weights.write_bytes(b"original")
    original = sys.modules["mlx_embedding_models.model"].Bert.from_pretrained

    def mutate(path):
        weights.write_bytes(b"changed weights during load")
        return original(path)

    monkeypatch.setattr(sys.modules["mlx_embedding_models.model"].Bert, "from_pretrained", mutate)
    adapter = BertEmbeddingAdapter(str(local))
    with pytest.raises(ModelUnavailableError, match="changed while loading"):
        adapter.load()
    assert adapter.model is adapter.raw_hf_tokenizer is None
    assert adapter.get_descriptor()["revision"] == "unknown"


def test_repeated_successful_load_is_idempotent(dependencies):
    adapter = BertEmbeddingAdapter("all-MiniLM-L6-v2")
    adapter.load()
    adapter.load()
    assert len(dependencies.calls.snapshot) == 1
    assert len(dependencies.calls.bert) == 1


@pytest.mark.parametrize("length", [0, -1, True, 1.5, float("nan")])
def test_invalid_context_lengths_are_rejected(length):
    with pytest.raises(InvalidInputError, match="max_length"):
        BertEmbeddingAdapter("all-MiniLM-L6-v2", max_length=length)
    with pytest.raises(InvalidInputError, match="max_length"):
        TokenizerHelper.tokenize_once(["one"], TokenizerSpy(), max_length=length)
