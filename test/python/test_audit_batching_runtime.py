"""CPU-only regression coverage for runtime ownership and batch memory flow."""

import gc
import threading
import time
import weakref

import numpy as np
import pytest

from scripts.qmd_mlx import batching, runtime as runtime_module
from scripts.qmd_mlx.adapters.embedding import BaseEmbeddingAdapter
from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch
from scripts.qmd_mlx.batching import BatchPlanner
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.model_manager import ModelResidencyManager
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    MLXRuntimeError,
    ModelUnavailableError,
    OutOfMemoryError,
    RequestCancelledError,
    UnsupportedModelError,
    WorkerUnavailableError,
    WorkClass,
)


@pytest.fixture(autouse=True)
def no_metal_telemetry(monkeypatch):
    # Real executor threads are useful here; real Metal work is not.
    monkeypatch.setattr("scripts.qmd_mlx.model_manager._MLX_AVAILABLE", False)


def make_batch(lengths):
    return TokenizedBatch(
        token_ids=[[i + 1] * length for i, length in enumerate(lengths)],
        lengths=list(lengths),
        original_indices=list(range(len(lengths))),
        pad_token_id=0,
    )


class CPUTokenizer:
    pad_token_id = 0

    def __init__(self, length):
        self.length = length

    def encode(self, text, **kwargs):
        return [1] * self.length


class CPUAdapter(BaseEmbeddingAdapter):
    synthetic = True
    model_name = "synthetic/audit"
    model_params_b = 0.6
    model_memory_mb = 1.0
    native_dims = 2
    pooling_strategy = "mean"

    def __init__(self, *, fail_load=False, length=3):
        super().__init__("synthetic/audit")
        self.model_memory_mb = self.estimated_memory_mb = 1.0
        self.loaded = False
        self.fail_load = fail_load
        self.length = length
        self.load_threads = []
        self.unload_threads = []
        self.forward_sizes = []
        self.forward_active = False

    def is_loaded(self):
        return self.loaded

    def load(self):
        self.load_threads.append(threading.get_ident())
        self.loaded = True  # Exercise cleanup of a partial publication.
        self.model = object()
        if self.fail_load:
            raise RuntimeError("intentional eager load failure")
        tokenizer = CPUTokenizer(self.length)
        self._publish_loaded(
            self.model, tokenizer, tokenizer, {"hidden_size": 2},
            "fixture-audit", "float32", "none", 0,
        )
        self.model_memory_mb = 1.0

    def unload(self):
        assert not self.forward_active, "unloaded while a forward was active"
        self.unload_threads.append(threading.get_ident())
        self.loaded = False
        super().unload()

    def forward_batch(self, tokenized_batch, requested_dims=None):
        batch = tokenized_batch
        self.forward_sizes.append(len(batch))
        return np.asarray([[i, i + 1] for i in batch.original_indices], dtype=np.float32)


def track_owned_executors(monkeypatch):
    created = []

    def create(*args, **kwargs):
        executor = GPUExecutor(*args, **kwargs)
        created.append(executor)
        return executor

    monkeypatch.setattr(runtime_module, "GPUExecutor", create)
    return created


@pytest.mark.parametrize("failure", ["unsupported", "missing_mlx", "invalid_budget"])
def test_constructor_checks_prerequisites_before_allocating_owner(monkeypatch, failure):
    created = track_owned_executors(monkeypatch)
    adapter = CPUAdapter()
    kwargs = {}
    if failure == "unsupported":
        def resolve(**kwargs):
            raise UnsupportedModelError("unsupported fixture")
        monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", resolve)
        expected = UnsupportedModelError
    else:
        monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
        if failure == "missing_mlx":
            adapter.synthetic = False
            monkeypatch.setattr(runtime_module, "_MLX_AVAILABLE", False)
            expected = MLXRuntimeError
        else:
            kwargs["max_batch_tokens"] = -1
            expected = ValueError
    try:
        with pytest.raises(expected):
            runtime_module.MLXEmbeddingRuntime("synthetic/audit", **kwargs)
        assert created == [], "invalid construction allocated an executor thread"
    finally:
        for executor in created:
            executor.shutdown()


@pytest.mark.parametrize("failure", ["manager", "registration", "eager_load", "retune"])
def test_failed_construction_stops_owned_executor(monkeypatch, failure):
    created = track_owned_executors(monkeypatch)
    adapter = CPUAdapter(fail_load=failure == "eager_load")
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    if failure == "manager":
        def bad_manager(executor):
            raise RuntimeError("intentional manager failure")
        monkeypatch.setattr(runtime_module, "ModelResidencyManager", bad_manager)
    elif failure == "registration":
        class BadRegistrationManager(ModelResidencyManager):
            def register_adapter(self, stage, adapter):
                super().register_adapter(stage, adapter)
                raise RuntimeError("intentional registration failure")
        monkeypatch.setattr(runtime_module, "ModelResidencyManager", BadRegistrationManager)
    elif failure == "retune":
        calls = []
        original = BatchPlanner.tune_for_model

        def bad_retune(self, params):
            calls.append(params)
            if len(calls) == 2:
                raise RuntimeError("intentional post-load tuning failure")
            return original(self, params)
        monkeypatch.setattr(BatchPlanner, "tune_for_model", bad_retune)
    try:
        with pytest.raises(Exception, match="intentional"):
            runtime_module.MLXEmbeddingRuntime("synthetic/audit")
        assert len(created) == 1
        assert not created[0].is_worker_alive(), "constructor leaked its GPU owner"
        assert not adapter.is_loaded()
        if adapter.unload_threads:
            assert set(adapter.unload_threads) == {created[0]._worker_thread.ident}
    finally:
        for executor in created:
            executor.shutdown()


def test_failed_constructor_never_shuts_down_supplied_executor(monkeypatch):
    adapter = CPUAdapter(fail_load=True)
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor()
    try:
        with pytest.raises(Exception, match="intentional eager load failure"):
            runtime_module.MLXEmbeddingRuntime("synthetic/audit", executor=executor)
        assert executor.is_accepting() and executor.is_worker_alive()
        assert executor.submit(lambda: "still usable") == "still usable"
        assert not adapter.is_loaded()
    finally:
        executor.shutdown()


def test_manager_only_uses_its_executor_and_preserves_other_stages(monkeypatch):
    created = track_owned_executors(monkeypatch)
    adapter = CPUAdapter()
    other = CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor()
    manager = ModelResidencyManager(executor)
    manager.register_adapter("rerank", other)
    manager.ensure_loaded("rerank")
    rt = None
    try:
        rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit", model_manager=manager)
        assert created == [], "a supplied manager must not create an unrelated owner"
        assert rt.executor is executor
        rt.shutdown()
        assert executor.is_accepting() and executor.is_worker_alive()
        assert other.is_loaded()
        assert other.unload_threads == []
        assert not adapter.is_loaded()
    finally:
        if rt is not None:
            rt.shutdown()
        manager.unload_all()
        executor.shutdown()
        for extra in created:
            extra.shutdown()


def test_mismatched_executor_and_manager_are_rejected_without_mutation(monkeypatch):
    adapter = CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor()
    other_executor = GPUExecutor()
    manager = ModelResidencyManager(other_executor)
    rt = None
    try:
        with pytest.raises(ValueError, match="executor"):
            rt = runtime_module.MLXEmbeddingRuntime(
                "synthetic/audit", executor=executor, model_manager=manager, lazy_load=True,
            )
        assert manager.get_adapter("embed") is None
        assert executor.is_accepting() and other_executor.is_accepting()
    finally:
        if rt is not None:
            rt.shutdown()
        executor.shutdown()
        other_executor.shutdown()


def test_existing_shared_embedding_registration_is_not_replaced(monkeypatch):
    existing = CPUAdapter()
    replacement = CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: replacement)
    executor = GPUExecutor()
    manager = ModelResidencyManager(executor)
    manager.register_adapter("embed", existing)
    manager.ensure_loaded("embed")
    rt = None
    try:
        with pytest.raises(ValueError, match="embed"):
            rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit", model_manager=manager)
        assert manager.get_adapter("embed") is existing
        assert existing.is_loaded()
        assert existing.unload_threads == []
    finally:
        if rt is not None:
            rt.shutdown()
        manager.unload_all()
        executor.shutdown()


def test_owned_shutdown_unloads_before_stopping_owner_and_is_idempotent(monkeypatch):
    adapter = CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit")
    owner_ident = rt.executor._worker_thread.ident
    rt.shutdown()
    rt.shutdown()
    assert adapter.unload_threads == [owner_ident]
    assert not rt.executor.is_worker_alive()
    assert not adapter.is_loaded()


@pytest.mark.parametrize("shared", [False, True])
def test_shutdown_timeout_defers_cleanup_until_running_forward_finishes(monkeypatch, shared):
    adapter = CPUAdapter()
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def blocked_forward(tokenized_batch, requested_dims=None):
        batch = tokenized_batch
        adapter.forward_active = True
        entered.set()
        assert release.wait(3.0)
        adapter.forward_active = False
        return np.ones((len(batch), 2), dtype=np.float32)

    adapter.forward_batch = blocked_forward
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor() if shared else None
    rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit", executor=executor)
    owner_ident = rt.executor._worker_thread.ident

    def embed():
        try:
            rt.submit_embed(["blocking fixture"])
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=embed)
    thread.start()
    try:
        assert entered.wait(2.0)
        start = time.monotonic()
        with pytest.raises(DeadlineExceededError):
            rt.shutdown(timeout=0.02)
        assert time.monotonic() - start < 0.5
        assert adapter.unload_threads == []
        assert rt.executor.is_worker_alive()
        release.set()
        thread.join(2.0)
        rt.shutdown(timeout=2.0)
        assert adapter.unload_threads == [owner_ident]
        assert not adapter.is_loaded()
        if shared:
            assert executor is not None
            assert executor.is_accepting() and executor.is_worker_alive()
        else:
            assert not rt.executor.is_worker_alive()
    finally:
        release.set()
        thread.join(3.0)
        rt.shutdown(timeout=3.0)
        if executor is not None:
            executor.shutdown()


def test_shutdown_called_on_owner_is_deferred_without_self_wait(monkeypatch):
    adapter = CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit")
    owner_ident = rt.executor._worker_thread.ident
    try:
        rt.executor.submit(lambda: rt.shutdown(timeout=0.1), timeout_s=1.0)
        rt.shutdown(timeout=2.0)
        assert adapter.unload_threads == [owner_ident]
        assert not rt.executor.is_worker_alive()
    finally:
        rt.shutdown()


def test_borrowed_existing_registration_is_not_unloaded(monkeypatch):
    adapter = CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor()
    manager = ModelResidencyManager(executor)
    manager.register_adapter("embed", adapter)
    manager.ensure_loaded("embed")
    rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit", model_manager=manager)
    try:
        rt.shutdown()
        assert adapter.is_loaded()
        assert adapter.unload_threads == []
        assert executor.is_accepting()
    finally:
        manager.unload_all()
        executor.shutdown()


def test_already_admitted_cold_request_cannot_reload_after_shutdown(monkeypatch):
    adapter = CPUAdapter()
    adapter.raw_hf_tokenizer = None
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor()
    manager = ModelResidencyManager(executor)
    rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit", model_manager=manager, lazy_load=True)
    acquired = threading.Event()
    release = threading.Event()
    original_acquire = rt.acquire_admission_lease
    errors = []

    def delayed_acquire(*args, **kwargs):
        lease = original_acquire(*args, **kwargs)
        acquired.set()
        assert release.wait(2.0)
        return lease

    # Pause an admitted request before its model lease. Explicit manager unload
    # now correctly refuses a live lifecycle lease, so that is not a shutdown
    # completion barrier; this test isolates the late-request reload fence.
    monkeypatch.setattr(rt, "acquire_admission_lease", delayed_acquire)

    def embed():
        try:
            rt.submit_embed(["late cold request"])
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=embed)
    thread.start()
    try:
        assert acquired.wait(1.0)
        rt.shutdown()
        release.set()
        thread.join(2.0)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], WorkerUnavailableError)
        assert adapter.load_threads == []
        assert not adapter.is_loaded()
        assert rt.in_flight_requests == 0
        assert executor.is_accepting()
    finally:
        release.set()
        thread.join(3.0)
        rt.shutdown()
        executor.shutdown()


def test_shutdown_reports_failed_unload_without_stopping_shared_executor(monkeypatch):
    adapter = CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor()
    rt = runtime_module.MLXEmbeddingRuntime("synthetic/audit", executor=executor)
    original_unload = adapter.unload

    def fail_unload():
        raise RuntimeError("intentional unload failure")

    adapter.unload = fail_unload
    try:
        with pytest.raises(ModelUnavailableError, match="intentional unload failure"):
            rt.shutdown()
        assert adapter.is_loaded()
        assert executor.is_accepting() and executor.is_worker_alive()
    finally:
        adapter.unload = original_unload
        rt.model_manager.unload("embed")
        executor.shutdown()


def test_oom_retry_releases_failed_forward_arrays_before_bisection():
    planner = BatchPlanner(max_batch_tokens=4096)
    failed_arrays = []
    calls = []

    def forward(batch):
        gc.collect()
        assert all(ref() is None for ref in failed_arrays), "failed traceback retained its activation array"
        calls.append(list(batch.original_indices))
        if len(batch) > 2:
            activation = np.ones((128, 128), dtype=np.float32)
            failed_arrays.append(weakref.ref(activation))
            raise RuntimeError("[metal] out of memory")
        return np.asarray([[i, i + 10] for i in batch.original_indices], dtype=np.float32)

    result = planner.execute_sub_batch_with_retry(make_batch([8] * 8), forward)
    np.testing.assert_array_equal(result[:, 0], np.arange(8))
    assert result.shape == (8, 2)
    assert calls == [list(range(8)), list(range(4)), [0, 1], [2, 3], list(range(4, 8)), [4, 5], [6, 7]]


@pytest.mark.parametrize("error", [
    RuntimeError("[metal] invalid memory layout"),
    RuntimeError("[metal] allocation metadata mismatch"),
    RuntimeError("ordinary validation failure"),
    ValueError("MLX out of memory is an invalid argument"),
    RequestCancelledError("metal out of memory request cancelled"),
    DeadlineExceededError("metal out of memory deadline"),
])
def test_non_oom_errors_are_not_bisected_or_rebudgeted(error):
    planner = BatchPlanner(max_batch_tokens=4096)
    calls = []

    def forward(batch):
        calls.append(len(batch))
        raise error

    with pytest.raises(type(error)) as caught:
        planner.execute_sub_batch_with_retry(make_batch([3] * 8), forward)
    assert caught.value is error
    assert calls == [8]
    assert planner.max_batch_tokens == 4096


@pytest.mark.parametrize("message", [
    "Metal buffer allocation failed: out of memory",
    "[metal] Failed to allocate 1024 bytes",
    "[metal] Unable to allocate 1024 bytes",
    "MLX out of memory",
])
def test_genuine_native_oom_messages_are_retried(message):
    planner = BatchPlanner(max_batch_tokens=4096)
    calls = []

    def forward(batch):
        calls.append(len(batch))
        if len(batch) > 1:
            raise RuntimeError(message)
        return np.asarray([[batch.original_indices[0]]], dtype=np.float32)

    result = planner.execute_sub_batch_with_retry(make_batch([3] * 4), forward)
    assert calls == [4, 2, 1, 1, 2, 1, 1]
    np.testing.assert_array_equal(result[:, 0], np.arange(4))


def test_oom_retry_depth_is_bounded_and_discards_original_traceback():
    planner = BatchPlanner(max_batch_tokens=4096)
    refs = []
    sizes = []

    def forward(batch):
        sizes.append(len(batch))
        activation = np.ones((8, 8))
        refs.append(weakref.ref(activation))
        raise OutOfMemoryError("bounded allocation failure")

    with pytest.raises(OutOfMemoryError) as caught:
        planner.execute_sub_batch_with_retry(make_batch([3] * 32), forward)
    gc.collect()
    assert sizes == [32, 16, 8, 4]
    assert caught.value.__context__ is None
    assert all(ref() is None for ref in refs)


def test_small_explicit_budget_is_not_increased_after_oom():
    planner = BatchPlanner(max_batch_tokens=10)

    def forward(batch):
        raise OutOfMemoryError("small budget fixture")

    with pytest.raises(OutOfMemoryError):
        planner.execute_sub_batch_with_retry(make_batch([3]), forward)
    assert 0 < planner.max_batch_tokens <= 10


@pytest.mark.parametrize("kwargs", [
    {"max_batch_tokens": -1}, {"max_batch_tokens": True},
    {"max_batch_tokens": 12.5}, {"max_batch_tokens": float("nan")},
    {"max_batch_tokens": float("inf")}, {"max_batch_tokens": "4096"},
    {"max_bulk_microbatch_tokens": 0}, {"max_bulk_microbatch_tokens": -1},
    {"max_bulk_microbatch_tokens": True}, {"max_bulk_microbatch_tokens": 12.5},
    {"max_bulk_microbatch_tokens": float("nan")},
    {"max_bulk_microbatch_tokens": float("inf")},
])
def test_invalid_batch_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BatchPlanner(**kwargs)


@pytest.mark.parametrize("bulk_target,expected_sizes", [
    (512, [1] * 8), (1800, [2] * 4), (7200, [8]), (100000, [8]),
])
def test_long_bulk_documents_use_explicit_target_without_losing_tokens(bulk_target, expected_sizes):
    batch = make_batch([900] * 8)
    planner = BatchPlanner(max_batch_tokens=7200, max_bulk_microbatch_tokens=bulk_target)
    batches, indices = planner.plan_micro_batches(batch, work_class=WorkClass.BULK)
    assert [len(part) for part in batches] == expected_sizes
    assert [i for part in indices for i in part] == list(range(8))
    assert sum(sum(part.lengths) for part in batches) == 7200
    assert planner.effective_bulk_microbatch_tokens == min(planner.max_batch_tokens, planner.max_bulk_microbatch_tokens)
    for part in batches:
        padded_tokens = len(part) * max(part.lengths)
        assert padded_tokens <= planner.effective_bulk_microbatch_tokens or len(part) == 1


def test_bulk_target_and_override_cannot_exceed_model_hardware_budget():
    planner = BatchPlanner(
        max_bulk_microbatch_tokens=65536,
        model_params_b=4.0,
        system_ram_gb_fn=lambda: 8.0,
    )
    assert planner.max_batch_tokens == 1024
    assert planner.max_bulk_microbatch_tokens == 65536
    assert planner.effective_bulk_microbatch_tokens == 1024
    batches, _ = planner.plan_micro_batches(make_batch([900] * 8), max_tokens_override=65536)
    assert [len(part) for part in batches] == [1] * 8


def test_runtime_exposes_bulk_target_separately_and_passes_it_to_planner(monkeypatch):
    adapter = CPUAdapter(length=900)
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    rt = runtime_module.MLXEmbeddingRuntime(
        "synthetic/audit", max_batch_tokens=7200, max_bulk_microbatch_tokens=7200,
    )
    try:
        result = rt.submit_embed([f"document {i}" for i in range(8)])
        assert result.shape == (8, 2)
        assert adapter.forward_sizes == [8]
        np.testing.assert_array_equal(result[:, 0], np.arange(8))
        stats = rt.get_stats_info()
        assert stats["max_batch_tokens"] == 7200
        assert stats["max_bulk_microbatch_tokens"] == 7200
        assert stats["effective_bulk_microbatch_tokens"] == 7200
    finally:
        rt.shutdown()


def test_restore_identity_single_batch_returns_original_without_allocation(monkeypatch):
    array = np.arange(12, dtype=np.float32).reshape(3, 4)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("identity restoration allocated another output")

    monkeypatch.setattr(batching.np, "empty", unexpected_allocation)
    restored = BatchPlanner.restore_ordering([array], [[0, 1, 2]], 3)
    assert restored is array


def test_restore_multiple_batches_allocates_once_and_scatter_preserves_dtype(monkeypatch):
    first = np.asarray([[20, 21], [0, 1]], dtype=np.float32)
    second = np.asarray([[30, 31], [10, 11]], dtype=np.float64)
    allocations = []
    original_empty = np.empty

    def allocate(*args, **kwargs):
        allocations.append(args[0])
        return original_empty(*args, **kwargs)

    def concatenate_is_forbidden(*args, **kwargs):
        pytest.fail("restoration concatenated before allocating its final output")

    monkeypatch.setattr(batching.np, "empty", allocate)
    monkeypatch.setattr(batching.np, "concatenate", concatenate_is_forbidden)
    result = BatchPlanner.restore_ordering([first, second], [[2, 0], [3, 1]], 4)
    assert allocations == [(4, 2)]
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[0, 1], [10, 11], [20, 21], [30, 31]])


@pytest.mark.parametrize("results,indices,total", [
    ([], [], 1),
    ([np.ones((1, 2))], [], 1),
    ([np.ones((2, 2))], [[0]], 1),
    ([np.ones((1, 2)), np.ones((1, 3))], [[0], [1]], 2),
    ([np.ones(2)], [[0]], 1),
    ([np.ones((2, 2))], [[0, 0]], 2),
    ([np.ones((1, 2))], [[-1]], 1),
    ([np.ones((1, 2))], [[1]], 1),
    ([np.ones((1, 2))], [[0.0]], 1),
    ([np.ones((1, 2))], [[True]], 1),
    ([np.ones((1, 2))], [[0]], 2),
    ([np.ones((1, 2))], [[0]], 0),
    ([], [], -1),
])
def test_restore_rejects_inconsistent_cardinality_shapes_and_indices(results, indices, total):
    with pytest.raises(ValueError):
        BatchPlanner.restore_ordering(results, indices, total)
