"""BERT segment embeddings must match the trained single-sequence forward."""
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.qmd_mlx.adapters import embedding
from scripts.qmd_mlx.adapters.embedding import BertEmbeddingAdapter
from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_single_sequence_bert_explicitly_supplies_zero_segment_ids(monkeypatch, dtype):
    mx = SimpleNamespace(
        array=np.array, int32=np.int32, float32=np.float32, float16=np.float16,
        bfloat16=np.float64, zeros=np.zeros,
        expand_dims=np.expand_dims, sum=np.sum, maximum=np.maximum,
        sqrt=np.sqrt, eval=lambda *args: None,
    )
    monkeypatch.setattr(embedding, "mx", mx, raising=False)
    adapter = BertEmbeddingAdapter("sentence-transformers/all-MiniLM-L6-v2", dtype_str=dtype)
    adapter.native_dims = 3
    calls = []

    def trained_forward(ids, *, attention_mask, token_type_ids=None):
        # mlx_embedding_models skips learned segment embeddings when this is
        # None; HF BERT instead defaults to segment zero for a single sequence.
        assert token_type_ids is not None, "Missing learned segment-zero embedding"
        # MLX attention rejects a float32 additive mask for float16 outputs;
        # the external BERT layer logs this mask before calling fused attention.
        assert attention_mask.dtype == np.dtype(getattr(mx, dtype))
        np.testing.assert_array_equal(token_type_ids, np.zeros_like(ids))
        calls.append((ids.copy(), attention_mask.copy(), token_type_ids.copy()))
        hidden = np.stack([ids + 1, ids + 2, ids + 3], axis=-1).astype(np.float32)
        return hidden, None

    adapter.model = trained_forward
    batch = TokenizedBatch([[101, 42, 102], [101, 102]], [3, 2], [0, 1], 0)
    result = adapter.forward_batch(batch)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1, 1], atol=1e-6)
    np.testing.assert_array_equal(calls[0][1], [[1, 1, 1], [1, 1, 0]])
    assert calls[0][2].dtype == np.int32
