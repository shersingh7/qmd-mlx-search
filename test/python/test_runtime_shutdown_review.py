"""CPU-only shutdown obligations with a saturated or externally stopped owner."""

from concurrent.futures import Future
import threading
import time

import pytest

from scripts.qmd_mlx import runtime as runtime_module
from scripts.qmd_mlx.adapters.embedding import BaseEmbeddingAdapter
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError, ModelUnavailableError, OverloadedError, WorkerUnavailableError,
)


class ShutdownTokenizer:
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [1]


class ShutdownAdapter(BaseEmbeddingAdapter):
    synthetic = True
    model_name = "synthetic/shutdown-review"
    model_params_b = 0.6
    model_memory_mb = 1.0
    native_dims = 2
    pooling_strategy = "mean"

    def __init__(self):
        super().__init__("synthetic/shutdown-review")
        self.estimated_memory_mb = self.model_memory_mb = 1.0
        self.loaded = False
        self.active = threading.Event()
        self.unloaded = threading.Event()
        self.unload_threads = []

    def load(self):
        tokenizer = ShutdownTokenizer()
        self._publish_loaded(
            object(), tokenizer, tokenizer, {"hidden_size": 2},
            "fixture-shutdown", "float32", "none", 0,
        )
        self.model_memory_mb = 1.0
        self.loaded = True

    def is_loaded(self):
        return self.loaded

    def unload(self):
        assert not self.active.is_set(), "cleanup overlapped a live native operation"
        self.unload_threads.append(threading.get_ident())
        self.loaded = False
        super().unload()
        self.unloaded.set()


@pytest.fixture(autouse=True)
def no_metal_telemetry(monkeypatch):
    monkeypatch.setattr("scripts.qmd_mlx.model_manager._MLX_AVAILABLE", False)


def make_runtime(monkeypatch, shared, *, lazy_load=False):
    adapter = ShutdownAdapter()
    executor = GPUExecutor(max_queue_size=1)
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    monkeypatch.setattr(runtime_module, "GPUExecutor", lambda: executor)
    if shared:
        runtime = runtime_module.MLXEmbeddingRuntime(adapter.model_name, executor=executor, lazy_load=lazy_load)
    else:
        runtime = runtime_module.MLXEmbeddingRuntime(adapter.model_name, lazy_load=lazy_load)
    return runtime, executor, adapter


def block_owner(executor, adapter):
    entered, release = threading.Event(), threading.Event()

    def native_work():
        adapter.active.set()
        entered.set()
        try:
            assert release.wait(5.0), "test did not release its synthetic native work"
            return "native completed"
        finally:
            adapter.active.clear()

    future, _, _ = executor.submit_async(native_work, timeout_s=10.0)
    assert entered.wait(1.0)
    return release, future


def shutdown_helpers():
    return {thread for thread in threading.enumerate() if thread.name == "MLX-Runtime-Shutdown"}


@pytest.mark.parametrize("shared", [False, True])
def test_saturated_shutdown_remains_admitted_after_timeout_without_a_waiter_thread(monkeypatch, shared):
    runtime, executor, adapter = make_runtime(monkeypatch, shared)
    owner_ident = executor._worker_thread.ident
    release, active = block_owner(executor, adapter)
    pending, _, _ = executor.submit_async(lambda: "queued work", timeout_s=10.0)
    helpers_before = shutdown_helpers()
    try:
        start = time.monotonic()
        with pytest.raises(DeadlineExceededError):
            runtime.shutdown(timeout=0.02)
        assert time.monotonic() - start < 0.5
        completion = runtime._shutdown_future
        assert completion is not None and not completion.done()
        for _ in range(3):
            with pytest.raises(DeadlineExceededError):
                runtime.shutdown(timeout=0)
            assert runtime._shutdown_future is completion
        assert shutdown_helpers() == helpers_before, "shutdown retained a helper blocked on native work"
        assert adapter.unload_threads == []
        assert executor.is_worker_alive() and not active.done()
        with pytest.raises(WorkerUnavailableError):
            runtime.acquire_admission_lease()
        release.set()
        assert active.result(timeout=1.0) == "native completed"
        # No second shutdown call is needed to make the accepted cleanup happen.
        completion.result(timeout=2.0)
        if shared:
            assert pending.result(timeout=1.0) == "queued work"
            assert executor.submit(lambda: "shared still usable", timeout_s=1.0) == "shared still usable"
        else:
            executor.join_worker(timeout=1.0)
            assert not executor.is_worker_alive()
        runtime.shutdown(timeout=1.0)
        runtime.shutdown(timeout=1.0)
        assert adapter.unload_threads == [owner_ident]
        assert not adapter.is_loaded()
        assert executor.is_accepting() is shared
    finally:
        release.set()
        executor.shutdown(timeout=2.0)
        if not adapter.unloaded.is_set():
            runtime.shutdown(timeout=1.0)
        assert not executor.is_worker_alive()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("stop_first", [False, True])
def test_external_stop_cannot_strand_cleanup_before_or_after_shutdown_admission(monkeypatch, shared, stop_first):
    runtime, executor, adapter = make_runtime(monkeypatch, shared)
    release, active = block_owner(executor, adapter)
    owner_ident = executor._worker_thread.ident
    helpers_before = shutdown_helpers()
    try:
        if stop_first:
            executor.shutdown(timeout=0)
        with pytest.raises(DeadlineExceededError):
            runtime.shutdown(timeout=0.02)
        completion = runtime._shutdown_future
        assert completion is not None
        if not stop_first:
            executor.shutdown(timeout=0)
        for _ in range(2):
            with pytest.raises(DeadlineExceededError):
                runtime.shutdown(timeout=0.01)
        assert adapter.is_loaded() and not adapter.unload_threads
        assert executor.is_worker_alive() and not executor.is_accepting()
        assert shutdown_helpers() == helpers_before
        release.set()
        assert active.result(timeout=1.0) == "native completed"
        executor.join_worker(timeout=2.0)
        assert not executor.is_worker_alive()
        runtime.shutdown(timeout=1.0)
        runtime.shutdown(timeout=1.0)
        assert adapter.unload_threads == [owner_ident]
        assert not adapter.is_loaded()
        assert not executor.is_accepting()
        assert shutdown_helpers() == helpers_before
    finally:
        release.set()
        executor.shutdown(timeout=2.0)
        if not adapter.unloaded.is_set():
            runtime.shutdown(timeout=1.0)
        assert not executor.is_worker_alive()


@pytest.mark.parametrize("rejected_future", [None, "exception", "cancelled"])
def test_joined_owner_allows_safe_disposal_and_recovery_from_cached_rejection(monkeypatch, rejected_future):
    runtime, executor, adapter = make_runtime(monkeypatch, shared=True)
    executor.shutdown(timeout=1.0)
    assert not executor.is_worker_alive()
    if rejected_future:
        failed = Future()
        if rejected_future == "cancelled":
            assert failed.cancel()
        else:
            failed.set_exception(WorkerUnavailableError("Executor is shutting down"))
        runtime._shutdown_future = failed
    caller_ident = threading.get_ident()
    try:
        runtime.shutdown(timeout=1.0)
        completion = runtime._shutdown_future
        runtime.shutdown(timeout=1.0)
        assert runtime._shutdown_future is completion
        assert adapter.unload_threads == [caller_ident]
        assert not adapter.is_loaded()
        assert not executor.is_worker_alive()
    finally:
        if adapter.is_loaded():
            runtime.model_manager.unload("embed")


def test_exit_commit_race_defers_fallback_until_a_retry_confirms_owner_exit(monkeypatch):
    runtime, executor, adapter = make_runtime(monkeypatch, shared=True)
    exit_committed, release_exit = threading.Event(), threading.Event()
    original_settle = executor._settle_rejected_jobs

    def pause_before_exit(rejected):
        original_settle(rejected)
        if executor.is_owner_thread() and executor._worker_exiting:
            exit_committed.set()
            assert release_exit.wait(5.0)

    monkeypatch.setattr(executor, "_settle_rejected_jobs", pause_before_exit)
    try:
        executor.shutdown(timeout=0)
        assert exit_committed.wait(1.0)
        with pytest.raises(DeadlineExceededError):
            runtime.shutdown(timeout=0.02)
        completion = runtime._shutdown_future
        assert completion is not None and not completion.done()
        assert adapter.is_loaded() and not adapter.unload_threads
        assert executor.is_worker_alive()
        release_exit.set()
        executor.join_worker(timeout=1.0)
        assert not executor.is_worker_alive()
        runtime.shutdown(timeout=1.0)
        assert runtime._shutdown_future is completion
        assert adapter.unload_threads == [threading.get_ident()]
        assert not executor._cleanup_jobs
    finally:
        release_exit.set()
        executor.shutdown(timeout=2.0)
        if adapter.is_loaded():
            runtime.shutdown(timeout=1.0)


def test_concurrent_stopped_owner_disposal_has_one_claim_and_bounded_other_waiters(monkeypatch):
    runtime, executor, adapter = make_runtime(monkeypatch, shared=True)
    executor.shutdown(timeout=1.0)
    entered, release = threading.Event(), threading.Event()
    original_unload = adapter.unload
    outcomes = []

    def blocked_disposal():
        entered.set()
        assert release.wait(5.0)
        original_unload()

    monkeypatch.setattr(adapter, "unload", blocked_disposal)

    def dispose():
        try:
            outcomes.append(runtime.shutdown(timeout=1.0))
        except BaseException as exc:
            outcomes.append(exc)

    caller = threading.Thread(target=dispose, name="shutdown-disposal-caller")
    caller.start()
    try:
        assert entered.wait(1.0)
        completion = runtime._shutdown_future
        for _ in range(3):
            with pytest.raises(DeadlineExceededError):
                runtime.shutdown(timeout=0.01)
            assert runtime._shutdown_future is completion
        release.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        assert outcomes == [None]
        assert adapter.unload_threads == [caller.ident]
        runtime.shutdown(timeout=1.0)
    finally:
        release.set()
        caller.join(timeout=2.0)
        executor.shutdown(timeout=1.0)


def test_reserved_cleanup_is_bounded_idempotent_and_does_not_expand_request_capacity():
    executor = GPUExecutor(max_queue_size=1)
    adapter = ShutdownAdapter()
    release, active = block_owner(executor, adapter)
    cleanups = []
    try:
        for index in range(executor.MAX_CLEANUP_JOBS):
            cleanups.append(executor.reserve_cleanup(lambda index=index: index))
        with pytest.raises(OverloadedError, match="cleanup reservations"):
            executor.reserve_cleanup(lambda: None)
        pending, _, _ = executor.submit_async(lambda: "request")
        for cleanup in cleanups:
            assert not cleanup.cancel()
            assert executor.submit_cleanup(cleanup) is cleanup
            assert executor.submit_cleanup(cleanup) is cleanup
        assert executor.get_queue_depth() == 1
        assert executor.is_overloaded()
        with pytest.raises(OverloadedError):
            executor.submit_async(lambda: "too many requests")
        with executor._cv:
            assert len(executor._queue) == executor.MAX_CLEANUP_JOBS + 1
        release.set()
        active.result(timeout=1.0)
        assert pending.result(timeout=1.0) == "request"
        assert [cleanup.result(timeout=1.0) for cleanup in cleanups] == list(range(executor.MAX_CLEANUP_JOBS))
        executor.shutdown(timeout=1.0)
        assert not executor._cleanup_jobs
    finally:
        release.set()
        executor.shutdown(timeout=2.0)


def test_cleanup_selected_before_external_shutdown_is_not_cancelled():
    executor = GPUExecutor(max_queue_size=1)
    adapter = ShutdownAdapter()
    release, active = block_owner(executor, adapter)
    executed = []
    try:
        stale, stale_event, _ = executor.submit_async(lambda: pytest.fail("stale work ran"))
        cleanup = executor.reserve_cleanup(lambda: executed.append(executor.is_owner_thread()))
        executor.submit_cleanup(cleanup)
        stale.add_done_callback(lambda _: executor.shutdown(timeout=0))
        stale_event.set()
        release.set()
        active.result(timeout=1.0)
        cleanup.result(timeout=1.0)
        executor.join_worker(timeout=1.0)
        assert executed == [True]
        assert not executor.is_worker_alive()
    finally:
        release.set()
        executor.shutdown(timeout=2.0)


def test_cleanup_reservation_failure_precedes_model_registration_or_loading(monkeypatch):
    adapter = ShutdownAdapter()
    executor = GPUExecutor()
    cleanups = [executor.reserve_cleanup(lambda: None) for _ in range(executor.MAX_CLEANUP_JOBS)]
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    try:
        with pytest.raises(OverloadedError, match="cleanup reservations"):
            runtime_module.MLXEmbeddingRuntime(adapter.model_name, executor=executor)
        assert not adapter.is_loaded() and not adapter.unload_threads
        assert executor.is_accepting() and executor.is_worker_alive()
        assert executor.submit(lambda: "still usable", timeout_s=1.0) == "still usable"
    finally:
        for cleanup in cleanups:
            executor.submit_cleanup(cleanup)
        executor.shutdown(timeout=2.0)
        assert not executor.is_worker_alive()


@pytest.mark.parametrize("shared", [False, True])
def test_concurrent_shutdown_callers_share_one_saturated_cleanup(monkeypatch, shared):
    runtime, executor, adapter = make_runtime(monkeypatch, shared)
    release, active = block_owner(executor, adapter)
    executor.submit_async(lambda: None, timeout_s=10.0)
    barrier = threading.Barrier(5)
    outcomes, futures = [], []

    def call_shutdown():
        barrier.wait(timeout=2.0)
        try:
            runtime.shutdown(timeout=0.02)
        except BaseException as exc:
            outcomes.append(exc)
        futures.append(runtime._shutdown_future)

    callers = [threading.Thread(target=call_shutdown) for _ in range(4)]
    for caller in callers:
        caller.start()
    try:
        barrier.wait(timeout=2.0)
        for caller in callers:
            caller.join(timeout=1.0)
            assert not caller.is_alive()
        assert len(outcomes) == 4 and all(isinstance(exc, DeadlineExceededError) for exc in outcomes)
        assert len(futures) == 4 and all(future is runtime._shutdown_future for future in futures)
        assert adapter.is_loaded() and not adapter.unload_threads
        with executor._cv:
            assert sum(job.cleanup for job in executor._queue) == 1
        release.set()
        active.result(timeout=1.0)
        runtime.shutdown(timeout=1.0)
        assert adapter.unload_threads == [executor._worker_thread.ident]
    finally:
        release.set()
        for caller in callers:
            caller.join(timeout=2.0)
        executor.shutdown(timeout=2.0)
        if adapter.is_loaded():
            runtime.shutdown(timeout=1.0)


def test_owned_unload_failure_joins_owner_before_propagating(monkeypatch):
    runtime, executor, adapter = make_runtime(monkeypatch, shared=False)
    original_unload = adapter.unload

    def failed_unload():
        raise RuntimeError("intentional disposal failure")

    monkeypatch.setattr(adapter, "unload", failed_unload)
    try:
        with pytest.raises(ModelUnavailableError, match="intentional disposal failure"):
            runtime.shutdown(timeout=1.0)
        assert not executor.is_worker_alive()
        assert not executor.is_accepting()
        assert adapter.is_loaded()
    finally:
        monkeypatch.setattr(adapter, "unload", original_unload)
        runtime.model_manager.unload("embed")
        executor.shutdown(timeout=1.0)


@pytest.mark.parametrize("shared", [False, True])
def test_finished_native_load_releases_lease_before_its_cpu_waiter_resumes(monkeypatch, shared):
    runtime, executor, adapter = make_runtime(monkeypatch, shared, lazy_load=True)
    adapter.raw_hf_tokenizer = None
    waiter_returning, release_waiter = threading.Event(), threading.Event()
    original_submit = executor.submit
    outcomes = []

    def delayed_submit(*args, **kwargs):
        result = original_submit(*args, **kwargs)
        if kwargs.get("description") == "Initialize embedding tokenizer":
            waiter_returning.set()
            assert release_waiter.wait(5.0)
        return result

    monkeypatch.setattr(executor, "submit", delayed_submit)

    def tokenize():
        try:
            outcomes.append(runtime.tokenize(["cold initialization"]))
        except BaseException as exc:
            outcomes.append(exc)

    caller = threading.Thread(target=tokenize)
    caller.start()
    try:
        assert waiter_returning.wait(1.0), repr(outcomes)
        assert adapter.is_loaded()
        with runtime.model_manager._lock:
            assert runtime.model_manager._leases.get("embed", 0) == 0
        # A CPU waiter may be delayed indefinitely after native initialization.
        # It must not retain a lease that makes owner teardown refuse to unload.
        runtime.shutdown(timeout=1.0)
        assert adapter.unload_threads == [executor._worker_thread.ident]
        assert not adapter.is_loaded()
        assert caller.is_alive()
        release_waiter.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        assert len(outcomes) == 1 and isinstance(outcomes[0], WorkerUnavailableError)
    finally:
        release_waiter.set()
        caller.join(timeout=2.0)
        executor.shutdown(timeout=2.0)
        if adapter.is_loaded():
            runtime.shutdown(timeout=1.0)
