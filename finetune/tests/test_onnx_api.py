"""Native ONNX export, dynamic empty/present caches and quantization smoke tests.

Install the optional `onnx` dependency group to run these checks. No model or
weights are downloaded: the fixture is a randomly initialized one-layer Qwen3.
"""
from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx", reason="requires uv sync --group onnx")
ort = pytest.importorskip("onnxruntime", reason="requires uv sync --group onnx")


def run_cache_inference(session, model, batch=1, length=4, past_length=0):
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache

    ids = torch.ones((batch, length), dtype=torch.long)
    positions = torch.arange(past_length, past_length + length).unsqueeze(0).expand(batch, -1)
    attention = torch.ones((batch, past_length + length), dtype=torch.long)
    past = tuple(torch.zeros((batch, 1, past_length, 8)) for _ in range(2))
    cache = DynamicCache(ddp_cache_data=tuple(zip(past[::2], past[1::2])), config=model.config)
    with torch.inference_mode():
        expected = model(input_ids=ids, attention_mask=attention, position_ids=positions, past_key_values=cache, use_cache=True).logits.numpy()
    feed = {"input_ids": ids.numpy(), "attention_mask": attention.numpy(), "position_ids": positions.numpy(),
            "past_key_values.0.key": past[0].numpy(), "past_key_values.0.value": past[1].numpy()}
    actual = session.run(None, feed)
    assert actual[0].shape == expected.shape
    assert actual[1].shape == (batch, 1, past_length + length, 8)
    assert np.isfinite(actual[0]).all()
    return actual, expected


def test_onnx_dynamic_prefill_and_cached_decode(tiny_model, tiny_tokenizer, tmp_path):
    import numpy as np
    from convert_onnx import export_onnx

    export_onnx(tiny_model, tiny_tokenizer, str(tmp_path))
    path = tmp_path / "model.onnx"
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == ["logits", "present.0.key", "present.0.value"]
    for batch, length, past in [(1, 4, 0), (2, 3, 0), (1, 1, 4)]:
        actual, expected = run_cache_inference(session, tiny_model, batch, length, past)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-4, atol=1e-5)
    assert (tmp_path / "tokenizer.json").is_file()
    assert (tmp_path / "config.json").is_file()


@pytest.mark.parametrize("quantization", ["q4", "q8", "fp16"])
def test_onnx_quantization_apis(quantization, tiny_model, tiny_tokenizer, tmp_path):
    from convert_onnx import export_onnx, quantize_onnx

    export_onnx(tiny_model, tiny_tokenizer, str(tmp_path))
    quantize_onnx(str(tmp_path), quantization)
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    actual, _ = run_cache_inference(session, tiny_model)
    assert actual[0].shape == (1, 4, tiny_model.config.vocab_size)
