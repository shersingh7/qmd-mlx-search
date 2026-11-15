"""All tests use public synthetic data, empty caches and CPU-only models."""
from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile

# Establish isolation before importing any HF/Trackio/PyTorch modules.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="qmd-finetune-tests-"))
atexit.register(shutil.rmtree, _TEST_HOME, ignore_errors=True)
for key, value in {
    "HOME": str(_TEST_HOME),
    "HF_HOME": str(_TEST_HOME / "hf"),
    "HF_HUB_CACHE": str(_TEST_HOME / "hf" / "hub"),
    "HF_DATASETS_CACHE": str(_TEST_HOME / "hf" / "datasets"),
    "XDG_CACHE_HOME": str(_TEST_HOME / "cache"),
    "XDG_DATA_HOME": str(_TEST_HOME / "data"),
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "ACCELERATE_USE_CPU": "true",
    "CUDA_VISIBLE_DEVICES": "",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
}.items():
    os.environ[key] = value
for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TRACKIO_SPACE_ID", "TRACKIO_BUCKET_ID", "TRACKIO_SERVER_URL"):
    os.environ.pop(key, None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original_connect = socket.socket.connect
    def deny(*args, **kwargs):
        raise AssertionError("Network is forbidden in portable fine-tuning tests")
    def local_ipc_only(sock, address):
        if sock.family == socket.AF_UNIX:
            return original_connect(sock, address)
        return deny()
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", local_ipc_only)


@pytest.fixture
def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<pad>", "</s>", "user", "assistant", ":", "fixture", "query", "lex", "vec", "hyde", "Expand", "this", "search", "no_think", "tokens", "synthetic", "test", "document", "answer"]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    fast.model_input_names = ["input_ids", "attention_mask"]
    fast.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    return fast


@pytest.fixture
def tiny_model(tiny_tokenizer):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = Qwen3Config(
        vocab_size=len(tiny_tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=128,
        pad_token_id=tiny_tokenizer.pad_token_id, bos_token_id=None,
        eos_token_id=tiny_tokenizer.eos_token_id, tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).to("cpu")
    model.set_attn_implementation("eager")
    return model
