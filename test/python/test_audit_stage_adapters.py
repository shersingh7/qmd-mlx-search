"""CPU-only stage adapter regressions; no weights, downloads, or Metal calls."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.qmd_mlx import generate, rerank
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    ModelUnavailableError,
    RequestCancelledError,
)


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    """Replace every adapter MLX entry point, including memory accounting."""
    fake_mx = SimpleNamespace(
        array=np.array, arange=np.arange, int32=np.int32, float32=np.float32,
        floating=np.floating, issubdtype=np.issubdtype, matmul=np.matmul,
        eval=lambda *args: None, get_active_memory=lambda: 0,
    )
    for module in (rerank, generate):
        monkeypatch.setattr(module, "_MLX_AVAILABLE", True)
        monkeypatch.setattr(module, "mx", fake_mx, raising=False)
        monkeypatch.setattr(module, "mlx_lm", SimpleNamespace(load=Mock()), raising=False)
    monkeypatch.setattr(rerank, "nn", SimpleNamespace(QuantizedLinear=type("QuantizedLinear", (), {})), raising=False)
    monkeypatch.setattr(generate, "make_sampler", lambda **kwargs: None, raising=False)
    generate.mlx_lm.stream_generate = Mock(return_value=iter([SimpleNamespace(text="ok")]))


class PairTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    vocab_size = 8

    def __init__(self):
        self.threads = []

    def encode(self, text, **kwargs):
        self.threads.append(threading.get_ident())
        suffix = rerank.MLXRerankAdapter.THINK_SUFFIX
        if text == suffix:
            return [1]
        if text == suffix + "yes" or text == "yes":
            return [1, 2] if text.startswith(suffix) else [2]
        if text == suffix + "no" or text == "no":
            return [1, 3] if text.startswith(suffix) else [3]
        # Variable lengths and terminal IDs make padding/ordering observable.
        document = text.split("<Document>: ")[-1].split(suffix)[0]
        value = int(document[-1]) if document[-1:].isdigit() else 1
        return [1] * value + [value]


class TiedModel:
    def __init__(self, before_forward=None):
        self.shapes = []
        self.full_calls = 0
        self.before_forward = before_forward
        self.weights = np.arange(24, dtype=np.float32).reshape(8, 3) / 16
        self.args = SimpleNamespace(tie_word_embeddings=True, vocab_size=8)
        self.model = self.Backbone(self)

    class Backbone:
        def __init__(self, parent):
            self.parent = parent
            self.embed_tokens = SimpleNamespace(as_linear=parent.project)

        def __call__(self, ids):
            if self.parent.before_forward:
                self.parent.before_forward(ids)
            return np.stack((ids, ids + 1, ids * 2), axis=-1).astype(np.float32)

    def project(self, hidden):
        self.shapes.append(hidden.shape)
        return hidden @ self.weights.T

    def __call__(self, ids):
        self.full_calls += 1
        return self.project(self.model(ids))


def loaded_reranker(**kwargs):
    adapter = rerank.MLXRerankAdapter(model_name="fixture-rerank", lazy_load=True, **kwargs)
    adapter.model = TiedModel()
    adapter.raw_hf_tokenizer = PairTokenizer()
    adapter.tokenizer = adapter.raw_hf_tokenizer
    adapter.yes_token_id, adapter.no_token_id = 2, 3
    return adapter


def test_rerank_load_is_transactional_and_retryable():
    adapter = rerank.MLXRerankAdapter(model_name="fixture-rerank", lazy_load=True)
    bad = PairTokenizer()
    bad.encode = lambda *args, **kwargs: []
    rerank.mlx_lm.load.side_effect = [(TiedModel(), bad), (TiedModel(), PairTokenizer())]
    with pytest.raises(rerank.RerankError):
        adapter.load()
    assert adapter.model is None
    assert adapter.tokenizer is None
    assert adapter.raw_hf_tokenizer is None
    assert adapter.yes_token_id is adapter.no_token_id is None
    assert adapter.model_memory_mb == 0
    adapter.load()
    assert adapter.is_loaded()
    assert rerank.mlx_lm.load.call_count == 2


def test_rerank_does_not_publish_during_token_resolution():
    adapter = rerank.MLXRerankAdapter(model_name="fixture-rerank", lazy_load=True)
    tokenizer = PairTokenizer()
    encode = tokenizer.encode

    def inspect(text, **kwargs):
        assert adapter.model is adapter.tokenizer is adapter.raw_hf_tokenizer is None
        assert not adapter.is_loaded()
        return encode(text, **kwargs)

    tokenizer.encode = inspect
    rerank.mlx_lm.load.return_value = TiedModel(), tokenizer
    adapter.load()
    assert adapter.is_loaded()


@pytest.mark.parametrize("invalid", [-1, True, 1.25, "2", 2**31])
def test_rerank_rejects_invalid_resolved_token_ids(invalid):
    adapter = rerank.MLXRerankAdapter(model_name="fixture-rerank", lazy_load=True)
    tok = PairTokenizer()
    tok.encode = lambda text, **kwargs: [1, invalid] if text.endswith("yes") else ([1, 3] if text.endswith("no") else [1])
    rerank.mlx_lm.load.return_value = TiedModel(), tok
    with pytest.raises(rerank.RerankError, match="token"):
        adapter.load()
    assert not adapter.is_loaded()
    assert adapter.model is None


def test_rerank_rejects_multitoken_yes_no_fallback():
    adapter = rerank.MLXRerankAdapter(model_name="fixture-rerank", lazy_load=True)
    tok = PairTokenizer()
    tok.encode = lambda text, **kwargs: [2, 4] if text.endswith("yes") else [3, 5]
    rerank.mlx_lm.load.return_value = TiedModel(), tok
    with pytest.raises(rerank.RerankError, match="token"):
        adapter.load()


def test_rerank_unload_clears_all_initialization_state():
    adapter = loaded_reranker()
    adapter.unload()
    assert not adapter.is_loaded()
    assert adapter.model is adapter.tokenizer is adapter.raw_hf_tokenizer is None
    assert adapter.yes_token_id is adapter.no_token_id is None


def test_tied_head_only_projects_last_nonpad_token_with_reference_parity():
    adapter = loaded_reranker()
    docs = ["doc4", "doc1", "doc3", "doc2", "doc5"]
    tokens = [adapter.raw_hf_tokenizer.encode(adapter._format_pair("q", d)) for d in docs]
    last = np.array([s[-1] for s in tokens], dtype=np.float32)
    hidden = np.stack((last, last + 1, last * 2), axis=-1)
    logits = hidden @ adapter.model.weights.T
    diff = (logits[:, 2] - logits[:, 3]).astype(np.float32).astype(np.float64)
    expected = [float(1 / (1 + np.exp(-d))) for d in diff]
    actual = adapter.score_pairs("q", docs, batch_size=2)
    np.testing.assert_array_equal(actual, expected)
    assert adapter.model.full_calls == 0
    assert adapter.model.shapes == [(2, 3), (2, 3), (1, 3)]
    assert adapter.total_requests == 1 and adapter.total_pairs_scored == len(docs)


@pytest.mark.parametrize("tokens", [[], [-1], [True], [1.5], [2**31]])
def test_rerank_rejects_invalid_encoded_sequences(tokens):
    adapter = loaded_reranker()
    adapter.raw_hf_tokenizer.encode = lambda *args, **kwargs: tokens
    with pytest.raises(rerank.RerankError):
        adapter.score_pairs("q", ["doc"])
    assert adapter.model.shapes == []


def test_rerank_oversize_is_validated_before_any_forward():
    adapter = loaded_reranker(max_length=3)
    with pytest.raises(rerank.RerankError, match="exceeds max safe budget"):
        adapter.score_pairs("q", ["doc1", "doc5"], batch_size=1)
    assert adapter.model.shapes == []


class LeaseManager:
    def __init__(self, stage="rerank"):
        self.stage = stage
        self.leases = 0
        self.lock = threading.Lock()

    def acquire_lease(self, stage, timeout=None):
        assert stage == self.stage
        with self.lock:
            self.leases += 1
        return True

    def release_lease(self, stage):
        with self.lock:
            self.leases -= 1

    def ensure_loaded(self, stage, **kwargs):
        assert self.leases > 0

    def touch(self, stage):
        pass


def test_rerank_microbatches_interleave_and_tokenization_is_off_owner():
    executor = GPUExecutor()
    first, release = threading.Event(), threading.Event()
    order = []
    manager = LeaseManager()
    adapter = loaded_reranker(executor=executor, model_manager=manager)
    caller_thread = []

    def forward(ids):
        assert executor.is_owner_thread()
        assert manager.leases > 0
        order.append(f"batch{len(order)}")
        if not first.is_set():
            first.set()
            assert release.wait(2)

    adapter.model.before_forward = forward

    def request():
        caller_thread.append(threading.get_ident())
        return adapter.score_pairs("q", ["doc1", "doc2", "doc3"], batch_size=1)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(request)
            assert first.wait(2)
            other, _, _ = executor.submit_async(lambda: order.append("interactive"), priority=0)
            release.set()
            scores = result.result(timeout=3)
            other.result(timeout=2)
        assert len(scores) == 3
        assert order[1] == "interactive"
        assert set(adapter.raw_hf_tokenizer.threads) == set(caller_thread)
        assert manager.leases == 0
    finally:
        release.set()
        executor.shutdown()


def test_rerank_timeout_keeps_forward_lease_until_gpu_job_really_returns():
    executor = GPUExecutor()
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    manager = LeaseManager()
    adapter = loaded_reranker(executor=executor, model_manager=manager)

    def forward(ids):
        entered.set()
        assert release.wait(2)
        assert manager.leases > 0
        exited.set()

    adapter.model.before_forward = forward
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(adapter.score_pairs, "q", ["doc1", "doc2"], timeout_s=0.1)
            assert entered.wait(2)
            with pytest.raises(DeadlineExceededError):
                result.result(timeout=2)
            assert manager.leases > 0
            release.set()
            assert exited.wait(2)
            executor.submit(lambda: None, timeout_s=2)
        assert manager.leases == 0
        assert len(adapter.model.shapes) <= 1
        assert adapter.total_requests == 0
    finally:
        release.set()
        executor.shutdown()


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_rerank_checks_after_each_forward_and_does_not_score_remaining_docs(mode):
    adapter = loaded_reranker()
    cancel = threading.Event()
    deadline = time.monotonic() + 2
    clock = [time.monotonic()]
    original = rerank.time.monotonic

    def forward(ids):
        if mode == "cancel":
            cancel.set()
        else:
            clock[0] = deadline + 1

    adapter.model.before_forward = forward
    error = RequestCancelledError if mode == "cancel" else DeadlineExceededError
    from unittest.mock import patch
    with patch.object(rerank.time, "monotonic", side_effect=lambda: clock[0] if mode == "deadline" else original()):
        with pytest.raises(error):
            adapter.score_pairs("q", ["doc1", "doc2"], batch_size=1, deadline=deadline, cancel_event=cancel)
    assert len(adapter.model.shapes) <= 1
    assert adapter.total_requests == 0


class GenerationTokenizer:
    def __init__(self, count):
        self.count = count

    def apply_chat_template(self, *args, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "formatted prompt"

    def encode(self, text, **kwargs):
        return [1] * self.count if text == "formatted prompt" else [2]


def loaded_generator(count=100, **kwargs):
    adapter = generate.MLXGenerateAdapter(model_name="fixture-generate", lazy_load=True, **kwargs)
    adapter.model = object()
    adapter.tokenizer = adapter.raw_hf_tokenizer = GenerationTokenizer(count)
    return adapter


def test_generate_rejects_actual_requested_completion_over_context():
    adapter = loaded_generator(count=100)
    with pytest.raises(generate.GenerateError, match="context"):
        adapter.submit_generate("prompt", max_tokens=4096)
    generate.mlx_lm.stream_generate.assert_not_called()


@pytest.mark.parametrize("count,max_tokens", [(3500, 1), (4095, 1), (100, 3996)])
def test_generate_allows_actual_requested_completion_within_context(count, max_tokens):
    adapter = loaded_generator(count=count)
    assert adapter.submit_generate("prompt", max_tokens=max_tokens) == "ok"
    kwargs = generate.mlx_lm.stream_generate.call_args.kwargs
    assert kwargs["max_tokens"] == max_tokens
    # Forward the exact validated IDs; do not let the decoder retokenize differently.
    assert kwargs["prompt"] == [1] * count


@pytest.mark.parametrize("max_tokens", [True, False, 0, -1, 1.5, 4097])
def test_generate_validates_completion_on_sync_and_queued_paths(max_tokens):
    adapter = loaded_generator()
    with pytest.raises(generate.GenerateError):
        adapter._generate_sync("prompt", max_tokens=max_tokens, temperature=0)
    with pytest.raises(generate.GenerateError):
        adapter.submit_generate("prompt", max_tokens=max_tokens)
    generate.mlx_lm.stream_generate.assert_not_called()


@pytest.mark.parametrize("temperature", [True, -1, float("nan"), float("inf"), "0"])
def test_generate_validates_temperature(temperature):
    adapter = loaded_generator()
    with pytest.raises(generate.GenerateError):
        adapter.submit_generate("prompt", temperature=temperature)
    generate.mlx_lm.stream_generate.assert_not_called()


@pytest.mark.parametrize("tokens", [[], [-1], [True], [1.5], [2**31]])
def test_generate_rejects_invalid_formatted_token_ids(tokens):
    adapter = loaded_generator()
    adapter.raw_hf_tokenizer.encode = lambda text, **kwargs: tokens
    with pytest.raises(generate.GenerateError):
        adapter.submit_generate("prompt")
    generate.mlx_lm.stream_generate.assert_not_called()


def test_generation_load_rolls_back_partial_initialization_then_retries():
    adapter = generate.MLXGenerateAdapter(model_name="fixture-generate", lazy_load=True)

    class BrokenWrapper:
        @property
        def _tokenizer(self):
            assert adapter.model is None
            raise ValueError("broken tokenizer")

    generate.mlx_lm.load.side_effect = [(object(), BrokenWrapper()), (object(), GenerationTokenizer(100))]
    with pytest.raises(ModelUnavailableError):
        adapter.load()
    assert adapter.model is adapter.tokenizer is adapter.raw_hf_tokenizer is None
    assert adapter.model_memory_mb == 0
    adapter.load()
    assert adapter.is_loaded()
    assert generate.mlx_lm.load.call_count == 2
    adapter.unload()
    assert adapter.model is adapter.tokenizer is adapter.raw_hf_tokenizer is None


@pytest.mark.parametrize("module,kwargs,error", [
    (rerank, {"max_length": True}, rerank.RerankError),
    (rerank, {"max_length": 0}, rerank.RerankError),
    (generate, {"max_context": 0}, generate.GenerateError),
    (generate, {"max_new_tokens": True}, generate.GenerateError),
])
def test_adapter_constructor_rejects_invalid_token_limits(module, kwargs, error):
    cls = rerank.MLXRerankAdapter if module is rerank else generate.MLXGenerateAdapter
    with pytest.raises(error):
        cls(model_name="fixture-model", lazy_load=True, **kwargs)


@pytest.mark.parametrize("kind", ["float", "quantized", "callable", "fallback"])
def test_rerank_projection_paths_preserve_reference_scores_and_single_pass_tokenization(kind):
    adapter = loaded_reranker()
    tied = adapter.model
    weight = tied.weights
    bias = np.arange(8, dtype=np.float32) / 8
    shapes = []

    def project(hidden):
        shapes.append(hidden.shape)
        return hidden @ weight.T

    if kind == "fallback":
        adapter.model = lambda ids: project(tied.model(ids))
    else:
        if kind == "float":
            class FloatHead:
                def __init__(self):
                    self.weight = weight
                    self.bias = bias

                def __call__(self, hidden):
                    return project(hidden) + self.bias
            head = FloatHead()
        elif kind == "quantized":
            class QuantizedHead(rerank.nn.QuantizedLinear):
                weight = np.zeros((8, 1), dtype=np.uint32)

                def __call__(self, hidden):
                    return project(hidden)
            head = QuantizedHead()
        else:
            head = project
        adapter.model = SimpleNamespace(model=tied.model, lm_head=head, args=SimpleNamespace(tie_word_embeddings=False))
    docs = ["doc2", "doc1", "doc4"]
    last = np.array([2, 1, 4], dtype=np.float32)
    hidden = np.stack((last, last + 1, last * 2), axis=-1)
    logits = hidden @ weight.T
    if kind == "float":
        logits = logits + bias
    diff = logits[:, 2] - logits[:, 3]
    expected = [float(1 / (1 + np.exp(-d))) for d in diff.astype(np.float32).astype(np.float64)]
    actual = adapter.score_pairs_sync("q", docs, batch_size=2)
    np.testing.assert_array_equal(actual, expected)
    assert len(adapter.raw_hf_tokenizer.threads) == len(docs)
    if kind in ("float", "quantized", "callable"):
        assert shapes == [(2, 3), (1, 3)]


def test_generation_matches_decoder_bos_handling_before_context_check():
    adapter = loaded_generator(max_context=4)
    adapter.tokenizer.bos_token = "<bos>"
    adapter.raw_hf_tokenizer.apply_chat_template = lambda *args, **kwargs: "<bos>prompt"
    seen = []

    def encode(text, **kwargs):
        seen.append(kwargs)
        return [1, 2, 3] if kwargs.get("add_special_tokens") is False else [0, 1, 2, 3]

    adapter.raw_hf_tokenizer.encode = encode
    assert adapter.submit_generate("prompt", max_tokens=1) == "ok"
    assert seen == [{"add_special_tokens": False}]
    assert generate.mlx_lm.stream_generate.call_args.kwargs["prompt"] == [1, 2, 3]


def test_generation_closes_decode_iterator_under_lease_on_cancellation():
    manager = LeaseManager("generate")
    adapter = loaded_generator(model_manager=manager)
    cancel = threading.Event()
    closed = []

    class Decode:
        def __iter__(self):
            return self

        def __next__(self):
            cancel.set()
            return SimpleNamespace(text="ignored")

        def close(self):
            assert manager.leases > 0
            closed.append(True)

    generate.mlx_lm.stream_generate.return_value = Decode()
    with pytest.raises(RequestCancelledError):
        adapter.submit_generate("prompt", max_tokens=1, cancel_event=cancel)
    assert closed == [True]
    assert manager.leases == 0


def test_rerank_idle_eviction_respects_cpu_tokenization_and_forward_leases(monkeypatch):
    from scripts.qmd_mlx import model_manager as mm

    monkeypatch.setattr(mm, "_MLX_AVAILABLE", False)
    executor = GPUExecutor()
    manager = mm.ModelResidencyManager(executor, idle_unload_s=1, residency_budget_mb=10)
    adapter = loaded_reranker(executor=executor, model_manager=manager)
    adapter.unload()
    adapter.estimated_memory_mb = 1
    rerank.mlx_lm.load.return_value = TiedModel(), PairTokenizer()
    manager.register_adapter("rerank", adapter)
    manager.ensure_loaded("rerank", timeout_s=2)
    encode = adapter.raw_hf_tokenizer.encode
    probes = []

    def evict_probe():
        with manager._lock:
            manager._last_active["rerank"] = time.monotonic() - 10
        manager._check_idle_unloads()
        assert adapter.is_loaded()
        probes.append(True)

    def encode_with_eviction(text, **kwargs):
        assert not executor.is_owner_thread()
        executor.submit(evict_probe, timeout_s=2)
        return encode(text, **kwargs)

    adapter.raw_hf_tokenizer.encode = encode_with_eviction
    adapter.model.before_forward = lambda ids: evict_probe()
    try:
        assert len(adapter.score_pairs("q", ["doc1", "doc2"], batch_size=1)) == 2
        assert len(probes) == 4
        assert manager._leases == {}
        with manager._lock:
            manager._last_active["rerank"] = time.monotonic() - 10
        executor.submit(manager._check_idle_unloads, timeout_s=2)
        assert not adapter.is_loaded()
    finally:
        executor.shutdown()


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "later"])
def test_sync_adapters_reject_invalid_absolute_deadlines(deadline):
    adapter = loaded_reranker()
    with pytest.raises(rerank.RerankError):
        adapter.score_pairs_sync("q", ["doc1"], deadline=deadline)
    generator = loaded_generator()
    with pytest.raises(generate.GenerateError):
        generator._generate_sync("prompt", 1, 0, deadline=deadline)


def test_generation_timeout_pins_model_until_decode_cleanup_really_finishes():
    executor = GPUExecutor()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    manager = LeaseManager("generate")
    adapter = loaded_generator(executor=executor, model_manager=manager)

    def decode(*args, **kwargs):
        try:
            assert executor.is_owner_thread()
            entered.set()
            assert release.wait(2)
            assert manager.leases > 0
            yield SimpleNamespace(text="too late")
        finally:
            assert manager.leases > 0
            closed.set()

    generate.mlx_lm.stream_generate.side_effect = decode
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(adapter.submit_generate, "prompt", max_tokens=1, timeout=0.1)
            assert entered.wait(2)
            with pytest.raises(DeadlineExceededError):
                result.result(timeout=2)
            assert manager.leases > 0
            release.set()
            assert closed.wait(2)
            executor.submit(lambda: None, timeout_s=2)
        assert manager.leases == 0
        assert adapter.total_requests == 0
    finally:
        release.set()
        executor.shutdown()


def test_resolved_ids_outside_model_vocabulary_fail_before_load_publication():
    adapter = rerank.MLXRerankAdapter(model_name="fixture-rerank", lazy_load=True)
    tok = PairTokenizer()
    encode = tok.encode
    tok.encode = lambda text, **kwargs: [1, 9] if text.endswith("yes") else encode(text, **kwargs)
    rerank.mlx_lm.load.return_value = TiedModel(), tok
    with pytest.raises(rerank.RerankError, match="vocabulary"):
        adapter.load()
    assert adapter.model is None and not adapter.is_loaded()


def test_rerank_rejects_input_and_padding_ids_outside_known_model_vocabulary():
    adapter = loaded_reranker()
    with pytest.raises(rerank.RerankError, match="vocabulary"):
        adapter.score_pairs("q", ["doc9"])
    adapter.raw_hf_tokenizer.pad_token_id = 9
    with pytest.raises(rerank.RerankError, match="vocabulary"):
        adapter.score_pairs("q", ["doc1"])
    assert adapter.model.shapes == []


def test_generate_rejects_ids_outside_known_model_vocabulary():
    adapter = loaded_generator()
    adapter.model = SimpleNamespace(args=SimpleNamespace(vocab_size=8))
    adapter.raw_hf_tokenizer.encode = lambda *args, **kwargs: [9]
    with pytest.raises(generate.GenerateError, match="vocabulary"):
        adapter.submit_generate("prompt", max_tokens=1)
    generate.mlx_lm.stream_generate.assert_not_called()


def test_explicit_untied_model_without_exposed_head_keeps_full_model_fallback():
    adapter = loaded_reranker()
    model = adapter.model
    model.args.tie_word_embeddings = False
    # This model's public forward has custom output semantics, so its input
    # embedding must not be substituted for an unexposed, untied output head.
    def public_forward(ids):
        model.full_calls += 1
        shape = (*ids.shape, 8)
        logits = np.zeros(shape, dtype=np.float32)
        logits[:, :, 2] = 1
        return logits

    class Untied:
        def __init__(self):
            self.args = model.args
            self.model = model.model

        def __call__(self, ids):
            return public_forward(ids)

    adapter.model = Untied()
    scores = adapter.score_pairs("q", ["doc1"])
    assert scores == [float(1 / (1 + np.exp(-np.float64(1))))]
    assert model.full_calls == 1
