"""Independent CPU-only acceptance tests for stage lifecycle and head semantics."""

from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import time
from types import SimpleNamespace
from typing import Any
import weakref

import numpy as np
import pytest

from scripts.qmd_mlx import generate, rerank
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.protocol import DeadlineExceededError, ModelUnavailableError, RequestCancelledError


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("A CPU stage fixture must explicitly supply its loader")

    backend = SimpleNamespace(
        array=np.array, arange=np.arange, int32=np.int32, float32=np.float32,
        eval=lambda *args: None, get_active_memory=lambda: 0,
        floating=np.floating, issubdtype=np.issubdtype, matmul=np.matmul,
    )
    for module in (generate, rerank):
        monkeypatch.setattr(module, "_MLX_AVAILABLE", True)
        monkeypatch.setattr(module, "mx", backend, raising=False)
        monkeypatch.setattr(module, "mlx_lm", SimpleNamespace(load=forbidden_load), raising=False)
    monkeypatch.setattr(generate, "make_sampler", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(generate.mlx_lm, "stream_generate", lambda *args, **kwargs: iter([SimpleNamespace(text="ok")]), raising=False)
    monkeypatch.setattr(rerank, "nn", SimpleNamespace(QuantizedLinear=type("QuantizedLinear", (), {})), raising=False)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, **kwargs):
        suffix = rerank.MLXRerankAdapter.THINK_SUFFIX
        if text in (suffix + "yes", "yes"):
            return [1, 2] if text.startswith(suffix) else [2]
        if text in (suffix + "no", "no"):
            return [1, 3] if text.startswith(suffix) else [3]
        return [1]


class ProjectionModel:
    """Its public full forward, not a weight-difference shortcut, is the oracle."""

    def __init__(self, dtype=np.float16, base=2048, kind="float", bias=False):
        self.dtype = dtype
        self.forward_calls = 0
        self.projected_shapes = []
        self.args = SimpleNamespace(vocab_size=4, tie_word_embeddings=kind == "tied")
        weight = np.array([[0, 0], [0, 0], [base, 1], [base, 0]], dtype=dtype)
        head_bias = np.array([0, 0, 1, 0], dtype=dtype) if bias else None
        parent = self

        class Head:
            def __init__(self):
                self.weight = weight
                self.bias = head_bias

            def __call__(self, hidden):
                parent.projected_shapes.append(hidden.shape)
                logits = hidden @ weight.T
                return logits + head_bias if head_bias is not None else logits

        if kind == "quantized":
            class QuantizedHead(rerank.nn.QuantizedLinear, Head):
                def __init__(self):
                    super().__init__()
                    self.weight = np.zeros((4, 1), dtype=np.uint32)
            self.head = QuantizedHead()
        else:
            self.head = Head()

        class Backbone:
            def __init__(self):
                self.embed_tokens = SimpleNamespace(as_linear=parent.head)

            def __call__(self, ids):
                parent.forward_calls += 1
                return np.stack((ids, np.ones_like(ids)), axis=-1).astype(dtype)

        self.model = Backbone()
        if kind != "tied":
            self.lm_head = self.head

    def __call__(self, ids):
        return self.head(self.model(ids))


def adapter_for(stage, **kwargs) -> Any:
    if stage == "rerank":
        adapter = rerank.MLXRerankAdapter(model_name="cpu-review-rerank", lazy_load=True, **kwargs)
        adapter.yes_token_id, adapter.no_token_id = 2, 3
    else:
        adapter = generate.MLXGenerateAdapter(model_name="cpu-review-generate", lazy_load=True, **kwargs)
    adapter.model = ProjectionModel()
    adapter.tokenizer = adapter.raw_hf_tokenizer = Tokenizer()
    return adapter


def request_for(stage, adapter, **kwargs):
    if stage == "rerank":
        return adapter.score_pairs("query", ["document"], **kwargs)
    return adapter.submit_generate("prompt", max_tokens=3, **kwargs)


@pytest.mark.parametrize("stage", ["rerank", "generate"])
@pytest.mark.parametrize("failure", ["loader", "unwrap", "validation"])
def test_retained_load_error_does_not_keep_failed_model_or_tokenizer_alive(stage, failure, monkeypatch):
    adapter = adapter_for(stage)
    adapter.unload()
    module = rerank if stage == "rerank" else generate
    refs = []

    class Weights:
        def __init__(self):
            self.tensor = np.ones((4, 4), dtype=np.float16)
            refs.extend((weakref.ref(self), weakref.ref(self.tensor)))

    class Raw(Tokenizer):
        def __init__(self, model):
            self.model = model
            refs.append(weakref.ref(self))

        def encode(self, *args, **kwargs):
            raise rerank.RerankError("invalid resolved tokens")

    class Wrapper:
        def __init__(self, model):
            self.model = model
            self.raw = Raw(model)
            refs.append(weakref.ref(self))

        @property
        def _tokenizer(self):
            if failure == "unwrap":
                raise ValueError("cannot unwrap tokenizer")
            if stage == "generate" and failure == "validation":
                # Trigger generation's encode-callability validation.
                setattr(self.raw, "encode", None)
            return self.raw

    def loader(*args, **kwargs):
        model = Weights()
        wrapper = Wrapper(model)
        if failure == "loader":
            raise ValueError("loader failed after allocating weights")
        return model, wrapper

    monkeypatch.setattr(module.mlx_lm, "load", loader)
    error_type = rerank.RerankError if stage == "rerank" and failure == "validation" else ModelUnavailableError
    with pytest.raises(error_type) as captured:
        adapter.load()
    retained_error = captured.value
    assert not adapter.is_loaded()
    assert adapter.model is adapter.tokenizer is adapter.raw_hf_tokenizer is None
    assert adapter.model_memory_mb == 0
    gc.collect()
    assert refs and all(ref() is None for ref in refs), "Retained load traceback still owns failed tensors/tokenizers"
    assert retained_error.__cause__ is retained_error.__context__ is None
    # Retaining the failed operation must not poison a subsequent load.
    monkeypatch.setattr(module.mlx_lm, "load", lambda *args, **kwargs: (ProjectionModel(), Tokenizer()))
    adapter.load()
    assert adapter.is_loaded()


@pytest.mark.parametrize("dtype,base", [(np.float16, 2048), (np.float32, 16777216)])
@pytest.mark.parametrize("kind", ["float", "tied", "quantized"])
@pytest.mark.parametrize("bias", [False, True])
def test_last_token_projection_matches_full_forward_native_rounding(dtype, base, kind, bias):
    adapter = adapter_for("rerank")
    model = adapter.model = ProjectionModel(dtype=dtype, base=base, kind=kind, bias=bias)
    batch = [[1], [1, 2]]
    padded = np.array([[1, 0], [1, 2]], dtype=np.int32)
    full_logits = model(padded)
    last_logits = full_logits[np.arange(2), [0, 1]]
    diffs = (last_logits[:, 2] - last_logits[:, 3]).astype(np.float32).astype(np.float64)
    expected = (1 / (1 + np.exp(-diffs))).tolist()
    # Both projected logits round to the same value in this fixture. Moving
    # subtraction before projection incorrectly produces sigmoid(1) (or (2)).
    assert expected == [0.5, 0.5]
    model.forward_calls = 0
    model.projected_shapes.clear()
    assert adapter._score_batch_sync(batch, 0, None, None) == expected
    assert model.forward_calls == 1
    assert model.projected_shapes == [(2, 2)]  # [batch, hidden], never [batch, seq, hidden]


class AcquisitionGate:
    """Pause the request immediately before its first real lock acquisition."""

    def __init__(self):
        self.lock = threading.RLock()
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.request_thread: int | None = None
        self.armed = True

    def __enter__(self):
        if threading.get_ident() == self.request_thread and self.armed:
            self.armed = False
            self.waiting.set()
            assert self.release.wait(3)
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


@pytest.mark.parametrize("stage", ["rerank", "generate"])
@pytest.mark.parametrize("action", ["unload", "cancel"])
def test_standalone_rechecks_readiness_and_cancel_after_acquiring_lock(stage, action, monkeypatch):
    adapter = adapter_for(stage)
    old_model = adapter.model
    replacement = ProjectionModel()
    module = rerank if stage == "rerank" else generate
    monkeypatch.setattr(module.mlx_lm, "load", lambda *args, **kwargs: (replacement, Tokenizer()))
    decode_calls = []
    if stage == "generate":
        def decode(*args, **kwargs):
            decode_calls.append(args[0])
            return iter([SimpleNamespace(text="ok")])
        monkeypatch.setattr(generate.mlx_lm, "stream_generate", decode)
    gate = adapter._init_lock = AcquisitionGate()
    cancel = threading.Event()

    def request():
        gate.request_thread = threading.get_ident()
        return request_for(stage, adapter, cancel_event=cancel)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(request)
            assert gate.waiting.wait(2)
            if action == "unload":
                adapter.unload()
            else:
                cancel.set()
            gate.release.set()
            if action == "cancel":
                with pytest.raises(RequestCancelledError):
                    result.result(timeout=3)
                assert old_model.forward_calls == 0
                assert decode_calls == []
                assert adapter.total_requests == 0
            else:
                assert result.result(timeout=3)
                assert adapter.is_loaded()
                assert old_model.forward_calls == 0
                assert replacement.forward_calls == (1 if stage == "rerank" else 0)
                assert decode_calls == ([replacement] if stage == "generate" else [])
    finally:
        gate.release.set()


@pytest.mark.parametrize("stage", ["rerank", "generate"])
def test_standalone_preparation_pins_bundle_until_request_finishes(stage):
    adapter = adapter_for(stage)
    entered, release, unload_started, unloaded = [threading.Event() for _ in range(4)]
    encode = adapter.raw_hf_tokenizer.encode
    lock_was_busy = []

    def blocking_encode(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return encode(*args, **kwargs)

    adapter.raw_hf_tokenizer.encode = blocking_encode

    def unload():
        acquired = adapter._init_lock.acquire(blocking=False)
        lock_was_busy.append(not acquired)
        if acquired:
            adapter._init_lock.release()
        unload_started.set()
        adapter.unload()
        unloaded.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = pool.submit(request_for, stage, adapter)
            assert entered.wait(2)
            eviction = pool.submit(unload)
            assert unload_started.wait(2)
            try:
                assert lock_was_busy == [True], "preparation did not own the bundle lock"
                assert not unloaded.is_set()
            finally:
                release.set()
            assert result.result(timeout=3)
            eviction.result(timeout=3)
        assert unloaded.is_set() and not adapter.is_loaded()
    finally:
        release.set()


@pytest.mark.parametrize("action", ["cancel", "deadline", "unload"])
def test_batch_rechecks_request_and_readiness_at_the_forward_lock(action, monkeypatch):
    adapter = adapter_for("rerank")
    old_model = adapter.model
    gate = adapter._init_lock = AcquisitionGate()
    cancel = threading.Event()
    now = time.monotonic()
    clock = [now]
    deadline = now + 10
    monkeypatch.setattr(rerank.time, "monotonic", lambda: clock[0])
    replacement = ProjectionModel()
    monkeypatch.setattr(rerank.mlx_lm, "load", lambda *args, **kwargs: (replacement, Tokenizer()))

    def request():
        gate.request_thread = threading.get_ident()
        return adapter._score_batch_sync([[1]], 0, deadline, cancel)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(request)
            assert gate.waiting.wait(2)
            if action == "cancel":
                cancel.set()
            elif action == "deadline":
                clock[0] = deadline + 1
            else:
                adapter.unload()
            gate.release.set()
            if action == "unload":
                assert result.result(timeout=3) == [0.5]
                assert replacement.forward_calls == 1
            else:
                error = RequestCancelledError if action == "cancel" else DeadlineExceededError
                with pytest.raises(error):
                    result.result(timeout=3)
                assert replacement.forward_calls == 0
            assert old_model.forward_calls == 0
    finally:
        gate.release.set()


@pytest.mark.parametrize("stage", ["rerank", "generate"])
def test_manager_retained_failed_operation_releases_loader_tensors(stage, monkeypatch):
    from scripts.qmd_mlx import model_manager as mm

    monkeypatch.setattr(mm, "_MLX_AVAILABLE", False)
    executor = GPUExecutor()
    manager = mm.ModelResidencyManager(executor, idle_unload_s=0, residency_budget_mb=10)
    adapter = adapter_for(stage, executor=executor, model_manager=manager)
    adapter.unload()
    adapter.estimated_memory_mb = 1
    manager.register_adapter(stage, adapter)
    module = rerank if stage == "rerank" else generate
    refs = []

    def loader(*args, **kwargs):
        weights = np.ones((4, 4), dtype=np.float16)
        refs.append(weakref.ref(weights))
        raise ValueError("manager load failed after tensor allocation")

    monkeypatch.setattr(module.mlx_lm, "load", loader)
    try:
        with pytest.raises(ModelUnavailableError) as captured:
            manager.ensure_loaded(stage, timeout_s=2)
        retained_error = captured.value
        gc.collect()
        assert refs and all(ref() is None for ref in refs)
        assert retained_error.__cause__ is retained_error.__context__ is None
        assert manager.get_state(stage) == mm.ModelState.FAILED
        assert manager.get_memory_info()["model_mb"] == 0
        assert not adapter.is_loaded()
    finally:
        executor.shutdown()


@pytest.mark.parametrize("managed", [False, True])
def test_queued_rerank_rejects_a_bundle_replaced_during_caller_preparation(managed, monkeypatch):
    class Manager:
        leases = 0

        def acquire_lease(self, stage, timeout=None):
            self.leases += 1

        def release_lease(self, stage):
            self.leases -= 1

        def ensure_loaded(self, stage, **kwargs):
            assert self.leases > 0

        def touch(self, stage):
            pass

    executor = GPUExecutor()
    manager = Manager() if managed else None
    adapter = adapter_for("rerank", executor=executor, model_manager=manager)
    old_model = adapter.model
    replacement = ProjectionModel()
    monkeypatch.setattr(rerank.mlx_lm, "load", lambda *args, **kwargs: (replacement, Tokenizer()))
    encode = adapter.raw_hf_tokenizer.encode

    def replace_bundle():
        assert executor.is_owner_thread()
        adapter.unload()
        adapter.load()

    def encode_with_replacement(*args, **kwargs):
        assert not executor.is_owner_thread()
        # This must not deadlock the execution owner behind the caller's lock.
        executor.submit(replace_bundle, timeout_s=2)
        return encode(*args, **kwargs)

    adapter.raw_hf_tokenizer.encode = encode_with_replacement
    try:
        with pytest.raises(ModelUnavailableError, match="bundle changed"):
            adapter.score_pairs("query", ["document"])
        assert old_model.forward_calls == replacement.forward_calls == 0
        assert adapter.total_requests == 0
        if manager is not None:
            assert manager.leases == 0
    finally:
        executor.shutdown()


def test_executor_only_rerank_never_reloads_on_caller_after_warm_readiness_race(monkeypatch):
    executor = GPUExecutor()
    adapter = adapter_for("rerank", executor=executor)
    is_loaded = adapter.is_loaded
    first_check = [True]
    loads = []

    def observed_readiness():
        ready = is_loaded()
        if first_check[0] and not executor.is_owner_thread():
            first_check[0] = False
            executor.submit(adapter.unload, timeout_s=2)
        return ready

    def loader(*args, **kwargs):
        loads.append(threading.get_ident())
        assert executor.is_owner_thread()
        return ProjectionModel(), Tokenizer()

    monkeypatch.setattr(adapter, "is_loaded", observed_readiness)
    monkeypatch.setattr(rerank.mlx_lm, "load", loader)
    try:
        with pytest.raises(ModelUnavailableError, match="unloaded before tokenization"):
            adapter.score_pairs("query", ["document"])
        assert loads == []
        assert adapter.total_requests == 0
    finally:
        executor.shutdown()


def test_default_generation_event_reaches_decode_on_executor_shutdown(monkeypatch):
    executor = GPUExecutor()
    adapter = adapter_for("generate", executor=executor)
    entered, release, closed = [threading.Event() for _ in range(3)]
    chunks = []

    def decode(*args, **kwargs):
        try:
            for index in range(kwargs["max_tokens"]):
                if index == 0:
                    entered.set()
                    assert release.wait(3)
                chunks.append(index)
                yield SimpleNamespace(text="ignored")
        finally:
            closed.set()

    monkeypatch.setattr(generate.mlx_lm, "stream_generate", decode)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(adapter.submit_generate, "prompt", max_tokens=3)
            assert entered.wait(2)
            executor.shutdown(timeout=0)
            with pytest.raises(RequestCancelledError):
                result.result(timeout=2)
            release.set()
            assert closed.wait(2)
            executor.join_worker(timeout=2)
        assert not executor.is_worker_alive()
        assert chunks == [0]
        assert adapter.total_requests == adapter.total_tokens_generated == 0
    finally:
        release.set()
        executor.shutdown()
