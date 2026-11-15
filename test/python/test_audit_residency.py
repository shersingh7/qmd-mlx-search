"""Synthetic, offline regression coverage for transactional model residency."""

import threading
import time

import pytest

from scripts.qmd_mlx import model_manager as manager_module
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.model_manager import ModelResidencyManager, ModelState
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    ModelUnavailableError,
    OutOfMemoryError,
    OverloadedError,
    RequestCancelledError,
)


@pytest.fixture(autouse=True)
def no_metal_calls(monkeypatch):
    # These tests exercise lifecycle bookkeeping only, never native GPU APIs.
    monkeypatch.setattr(manager_module, "_MLX_AVAILABLE", False)


@pytest.fixture
def residency():
    executor = GPUExecutor(max_queue_size=20)
    manager = ModelResidencyManager(executor, idle_unload_s=0, residency_budget_mb=100)
    try:
        yield executor, manager
    finally:
        executor.shutdown(timeout=2)
        assert not executor.is_worker_alive()


class TransactionalAdapter:
    """A reranker-shaped adapter whose readiness deliberately underchecks fields."""

    def __init__(self, memory_mb=60, fail_phase=None, silent_phase=None):
        self.model_name = "synthetic-residency-only"
        self.model_memory_mb = memory_mb
        self.model = None
        self.tokenizer = None
        self.raw_hf_tokenizer = None
        self.yes_token_id = None
        self.no_token_id = None
        self.fail_phase = fail_phase
        self.silent_phase = silent_phase
        self.fail_unload = False
        self.ignore_unload = False
        self.load_count = 0
        self.unload_count = 0
        self.entered = threading.Event()
        self.release: threading.Event | None = None

    def is_loaded(self):
        # Matches the weak production reranker predicate, so the manager must
        # also reject missing tokenizers / the second resolved token ID.
        return self.model is not None and self.yes_token_id is not None

    def load(self):
        self.load_count += 1
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(2), "test did not release synthetic loader"
        phases = (
            ("before_model", None, None),
            ("model", "model", object()),
            ("tokenizer", "tokenizer", object()),
            ("raw_tokenizer", "raw_hf_tokenizer", object()),
            ("yes", "yes_token_id", 1),
            ("no", "no_token_id", 2),
        )
        for phase, field, value in phases:
            if self.silent_phase == phase:
                return
            if field is not None:
                setattr(self, field, value)
            if self.fail_phase == phase:
                raise RuntimeError(f"injected {phase} initialization failure")

    def unload(self):
        self.unload_count += 1
        if self.fail_unload:
            raise RuntimeError("injected unload failure before reference release")
        if self.ignore_unload:
            return
        self.model = None
        self.tokenizer = None
        self.raw_hf_tokenizer = None
        self.yes_token_id = None
        self.no_token_id = None


def launch_load(manager, name, timeout, outcomes, key, cancel_event=None):
    def caller():
        try:
            manager.ensure_loaded(name, timeout_s=timeout, cancel_event=cancel_event)
            outcomes[key] = "ready"
        except Exception as exc:
            outcomes[key] = exc

    thread = threading.Thread(target=caller)
    thread.start()
    return thread


def test_initiator_deadline_does_not_finish_running_shared_load(residency):
    _, manager = residency
    adapter = TransactionalAdapter()
    adapter.release = threading.Event()
    manager.register_adapter("embed", adapter)
    outcomes = {}
    short = launch_load(manager, "embed", 0.15, outcomes, "short")
    long = None
    try:
        assert adapter.entered.wait(1)
        with manager._lock:
            completion = manager._loading_events["embed"]
        long = launch_load(manager, "embed", 1.5, outcomes, "long")
        short.join(1)
        assert isinstance(outcomes.get("short"), DeadlineExceededError)
        assert manager.get_state("embed") == ModelState.LOADING
        assert not completion.is_set()
        assert "long" not in outcomes
        assert long.is_alive()
        adapter.release.set()
        long.join(1)
        assert outcomes.get("long") == "ready"
        assert completion.is_set()
        assert adapter.load_count == 1
        assert manager.get_state("embed") == ModelState.READY
    finally:
        adapter.release.set()
        short.join(2)
        if long is not None:
            long.join(2)


def test_cancelled_waiter_returns_without_aborting_shared_load(residency):
    _, manager = residency
    adapter = TransactionalAdapter()
    adapter.release = threading.Event()
    manager.register_adapter("embed", adapter)
    outcomes = {}
    cancel = threading.Event()
    initiator = launch_load(manager, "embed", 1.5, outcomes, "initiator", cancel)
    waiter = None
    try:
        assert adapter.entered.wait(1)
        waiter = launch_load(manager, "embed", 1.5, outcomes, "waiter")
        cancel.set()
        initiator.join(0.5)
        assert isinstance(outcomes.get("initiator"), RequestCancelledError)
        assert manager.get_state("embed") == ModelState.LOADING
        assert "waiter" not in outcomes
        adapter.release.set()
        waiter.join(1)
        assert outcomes.get("waiter") == "ready"
        assert adapter.load_count == 1
    finally:
        adapter.release.set()
        initiator.join(2)
        if waiter is not None:
            waiter.join(2)


@pytest.mark.parametrize("phase", ["before_model", "model", "tokenizer", "raw_tokenizer", "yes", "no"])
def test_each_partial_load_failure_rolls_back_and_retries(residency, phase):
    _, manager = residency
    adapter = TransactionalAdapter(fail_phase=phase)
    manager.register_adapter("rerank", adapter)
    with pytest.raises(ModelUnavailableError, match=f"injected {phase}"):
        manager.ensure_loaded("rerank")
    assert adapter.unload_count == 1
    assert adapter.model is None
    assert manager.get_state("rerank") == ModelState.FAILED
    assert manager.get_memory_info()["model_mb"] == 0
    assert "rerank" not in manager._loading_events
    adapter.fail_phase = None
    manager.ensure_loaded("rerank")
    assert adapter.load_count == 2
    assert manager.get_state("rerank") == ModelState.READY
    assert manager.get_error("rerank") is None
    assert manager.get_memory_info()["model_mb"] == 60


@pytest.mark.parametrize("phase", ["model", "tokenizer", "raw_tokenizer", "yes", "no"])
def test_load_return_without_all_required_fields_is_not_ready(residency, phase):
    _, manager = residency
    adapter = TransactionalAdapter(silent_phase=phase)
    manager.register_adapter("rerank", adapter)
    with pytest.raises(ModelUnavailableError, match="not ready"):
        manager.ensure_loaded("rerank")
    assert manager.get_state("rerank") == ModelState.FAILED
    assert adapter.unload_count == 1
    assert adapter.model is None
    assert manager.get_memory_info()["model_mb"] == 0
    adapter.silent_phase = None
    manager.ensure_loaded("rerank")
    assert manager.get_state("rerank") == ModelState.READY


def test_failed_partial_rollback_remains_accounted_until_cleanup_succeeds(residency):
    _, manager = residency
    broken = TransactionalAdapter(fail_phase="model")
    broken.fail_unload = True
    incoming = TransactionalAdapter()
    manager.register_adapter("embed", broken)
    manager.register_adapter("rerank", incoming)
    with pytest.raises(ModelUnavailableError, match="initialization failure"):
        manager.ensure_loaded("embed")
    assert broken.model is not None
    assert not broken.is_loaded()
    assert broken.unload_count == 1
    assert "unload failure" in manager.get_error("embed")
    assert manager.get_memory_info()["model_mb"] == 60
    with pytest.raises(OutOfMemoryError):
        manager.ensure_loaded("rerank")
    assert incoming.load_count == 0
    assert manager.get_memory_info()["model_mb"] == 60
    broken.fail_phase = None
    with pytest.raises(ModelUnavailableError, match="unload failure"):
        manager.ensure_loaded("embed")
    assert broken.load_count == 1  # Never layer a new load over leaked weights.
    broken.fail_unload = False
    manager.ensure_loaded("embed")
    assert broken.load_count == 2
    assert manager.get_state("embed") == ModelState.READY
    assert manager.get_memory_info()["model_mb"] == 60


@pytest.mark.parametrize("path", ["explicit", "idle", "lru"])
@pytest.mark.parametrize("failure", ["raise", "no_release"])
def test_failed_evictions_retain_accounting_and_block_admission(residency, path, failure):
    executor, manager = residency
    resident = TransactionalAdapter()
    incoming = TransactionalAdapter()
    manager.register_adapter("embed", resident)
    manager.register_adapter("rerank", incoming)
    manager.ensure_loaded("embed")
    resident.fail_unload = failure == "raise"
    resident.ignore_unload = failure == "no_release"
    if path == "explicit":
        manager.unload("embed")
    elif path == "idle":
        manager.idle_unload_s = 1
        with manager._lock:
            manager._last_active["embed"] = time.monotonic() - 10
        executor.submit(manager._check_idle_unloads, timeout_s=1)
        manager.idle_unload_s = 0
    else:
        with pytest.raises(OutOfMemoryError):
            manager.ensure_loaded("rerank")
    assert resident.is_loaded()
    assert manager.get_state("embed") == ModelState.FAILED
    assert manager.get_error("embed")
    assert "embed" not in manager._evicting_events
    assert manager.get_memory_info()["model_mb"] == 60
    with pytest.raises(OutOfMemoryError):
        manager.ensure_loaded("rerank")
    assert incoming.load_count == 0
    assert manager.get_memory_info()["model_mb"] == 60
    resident.fail_unload = False
    resident.ignore_unload = False
    manager.ensure_loaded("rerank")
    assert not resident.is_loaded()
    assert incoming.is_loaded()
    assert manager.get_state("embed") == ModelState.UNLOADED
    assert manager.get_error("embed") is None
    assert manager.get_memory_info()["model_mb"] == 60


def test_post_load_reconciliation_failure_rolls_back_new_weights(residency):
    _, manager = residency
    resident = TransactionalAdapter()

    class ExpandingAdapter(TransactionalAdapter):
        def load(self):
            super().load()
            self.model_memory_mb = 80

    incoming = ExpandingAdapter(memory_mb=40)
    manager.register_adapter("embed", resident)
    manager.register_adapter("rerank", incoming)
    manager.ensure_loaded("embed")
    resident.fail_unload = True
    with pytest.raises(OutOfMemoryError):
        manager.ensure_loaded("rerank")
    assert resident.is_loaded()
    assert not incoming.is_loaded()
    assert incoming.unload_count == 1
    assert manager.get_state("rerank") == ModelState.FAILED
    assert manager.get_memory_info()["model_mb"] == 60
    resident.fail_unload = False
    manager.ensure_loaded("rerank")
    assert manager.get_state("rerank") == ModelState.READY
    assert manager.get_memory_info()["model_mb"] == 80


def test_rejected_submission_settles_generation_and_allows_retry(residency, monkeypatch):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    original = executor.submit_async

    def reject(*args, **kwargs):
        raise OverloadedError("injected queue rejection")

    monkeypatch.setattr(executor, "submit_async", reject)
    with pytest.raises(OverloadedError, match="injected queue rejection"):
        manager.ensure_loaded("embed")
    assert manager.get_state("embed") == ModelState.FAILED
    assert "embed" not in manager._loading_events
    assert manager.get_memory_info()["model_mb"] == 0
    monkeypatch.setattr(executor, "submit_async", original)
    manager.ensure_loaded("embed")
    assert manager.get_state("embed") == ModelState.READY
    assert adapter.load_count == 1


def test_queued_expiration_settles_without_executing_or_poisoning_retry(residency):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    entered, release = threading.Event(), threading.Event()
    hold, _, _ = executor.submit_async(lambda: (entered.set(), release.wait(2)), timeout_s=2)
    assert entered.wait(1)
    try:
        with pytest.raises(DeadlineExceededError):
            manager.ensure_loaded("embed", timeout_s=0.05)
        release.set()
        hold.result(timeout=1)
        executor.submit(lambda: None, timeout_s=1)
        assert manager.get_state("embed") == ModelState.FAILED
        assert "embed" not in manager._loading_events
        assert adapter.load_count == 0
        manager.ensure_loaded("embed")
        assert adapter.load_count == 1
        assert manager.get_state("embed") == ModelState.READY
    finally:
        release.set()


@pytest.mark.parametrize("field", ["tokenizer", "raw_hf_tokenizer", "yes_token_id", "no_token_id"])
def test_incomplete_companions_cannot_hide_behind_a_weak_ready_predicate(residency, field):
    _, manager = residency

    class MissingCompanionAdapter(TransactionalAdapter):
        def load(self):
            super().load()
            setattr(self, field, None)

    adapter = MissingCompanionAdapter()
    manager.register_adapter("rerank", adapter)
    with pytest.raises(ModelUnavailableError, match="not ready"):
        manager.ensure_loaded("rerank")
    assert adapter.unload_count == 1
    assert manager.get_state("rerank") == ModelState.FAILED
    assert manager.get_memory_info()["model_mb"] == 0


def test_post_load_measurement_failure_cannot_prevent_rollback(residency, monkeypatch):
    _, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    estimate = manager._estimate_adapter_mb

    def broken_measurement(ad):
        if ad.model is not None:
            raise RuntimeError("injected memory measurement failure")
        return estimate(ad)

    monkeypatch.setattr(manager, "_estimate_adapter_mb", broken_measurement)
    with pytest.raises(ModelUnavailableError, match="memory measurement failure"):
        manager.ensure_loaded("embed")
    assert adapter.unload_count == 1
    assert adapter.model is None
    assert manager.get_state("embed") == ModelState.FAILED
    assert manager.get_memory_info()["model_mb"] == 0
    monkeypatch.setattr(manager, "_estimate_adapter_mb", estimate)
    manager.ensure_loaded("embed")
    assert adapter.load_count == 2
    assert manager.get_state("embed") == ModelState.READY


def test_old_queued_completion_cannot_finish_newer_inline_generation(residency, monkeypatch):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    owner_entered = threading.Event()
    take_over = threading.Event()
    reload = threading.Event()
    second_release = threading.Event()
    queued = threading.Event()
    queued_futures = []
    original_submit = executor.submit_async

    def observe_submission(fn, *args, **kwargs):
        result = original_submit(fn, *args, **kwargs)
        if kwargs.get("description") == "Load embed":
            queued_futures.append(result[0])
            queued.set()
        return result

    monkeypatch.setattr(executor, "submit_async", observe_submission)

    def owner_job():
        owner_entered.set()
        assert take_over.wait(2)
        manager.ensure_loaded("embed", timeout_s=1.5)
        assert reload.wait(2)
        manager.unload("embed")
        adapter.entered.clear()
        adapter.release = second_release
        manager.ensure_loaded("embed", timeout_s=1.5)

    owner_future, _, _ = executor.submit_async(owner_job, timeout_s=5)
    caller = None
    outcomes = {}
    try:
        assert owner_entered.wait(1)
        caller = launch_load(manager, "embed", 1.5, outcomes, "external")
        assert queued.wait(1)
        with manager._lock:
            first_event = manager._loading_events["embed"]
        take_over.set()
        caller.join(1)
        assert outcomes.get("external") == "ready"
        assert first_event.is_set()
        # The original physical queue entry has not run: the owner fulfilled its
        # generation inline and is still inside owner_job.
        assert not queued_futures[0].done()
        adapter.entered.clear()
        reload.set()
        assert adapter.entered.wait(1)
        with manager._lock:
            second_event = manager._loading_events["embed"]
        assert second_event is not first_event
        assert not second_event.is_set()
        # Model queued cancellation/expiry while another owner job is running.
        queued_futures[0].set_exception(DeadlineExceededError("old queued generation expired"))
        assert not second_event.is_set()
        assert manager.get_state("embed") == ModelState.LOADING
        assert manager.get_error("embed") is None
        second_release.set()
        owner_future.result(timeout=1)
        executor.submit(lambda: None, timeout_s=1)  # Drain the stale queued closure.
        assert second_event.is_set()
        assert adapter.load_count == 2
        assert adapter.unload_count == 1
        assert manager.get_state("embed") == ModelState.READY
        assert manager.get_memory_info()["model_mb"] == 60
    finally:
        take_over.set()
        reload.set()
        second_release.set()
        if caller is not None:
            caller.join(2)


@pytest.mark.parametrize("abandon", ["deadline", "cancel"])
def test_queued_shared_load_outlives_initiators_wait_budget(residency, monkeypatch, abandon):
    """The queue entry must belong to the generation, not its first waiter."""
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    entered, release = threading.Event(), threading.Event()
    hold, _, _ = executor.submit_async(lambda: (entered.set(), release.wait(3)), timeout_s=4)
    assert entered.wait(1)
    outcomes = {}
    cancel = threading.Event() if abandon == "cancel" else None
    attached = threading.Event()
    original_check = manager._check_load_wait_budget
    checks = {}

    def observe_wait(stage, dl, event):
        original_check(stage, dl, event)
        name = threading.current_thread().name
        checks[name] = checks.get(name, 0) + 1
        if name == "long-residency-waiter" and checks[name] >= 2:
            attached.set()

    monkeypatch.setattr(manager, "_check_load_wait_budget", observe_wait)
    short = launch_load(manager, "embed", 0.2 if abandon == "deadline" else 2, outcomes, "short", cancel)
    long = None
    try:
        deadline = time.monotonic() + 1
        while executor.get_queue_depth() != 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert executor.get_queue_depth() == 1
        with manager._lock:
            operation = manager._load_operations["embed"]
            assert not operation.started

        def long_caller():
            try:
                manager.ensure_loaded("embed", timeout_s=2)
                outcomes["long"] = "ready"
            except Exception as exc:
                outcomes["long"] = exc

        long = threading.Thread(target=long_caller, name="long-residency-waiter")
        long.start()
        assert attached.wait(1)
        if cancel is not None:
            cancel.set()
        short.join(1)
        assert not short.is_alive()
        expected = DeadlineExceededError if abandon == "deadline" else RequestCancelledError
        assert isinstance(outcomes.get("short"), expected)
        assert long.is_alive() and not operation.completed
        release.set()
        hold.result(timeout=1)
        long.join(1)
        assert not long.is_alive()
        assert outcomes.get("long") == "ready"
        assert adapter.load_count == 1
        assert manager.get_state("embed") == ModelState.READY
        assert operation.completed and operation.event.is_set()
    finally:
        release.set()
        short.join(2)
        if long is not None:
            long.join(2)


@pytest.mark.parametrize("abandon", ["deadline", "cancel"])
def test_last_waiter_cancels_abandoned_pending_load_before_owner_drain(residency, abandon):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    entered, release = threading.Event(), threading.Event()
    hold, _, _ = executor.submit_async(lambda: (entered.set(), release.wait(3)), timeout_s=4)
    assert entered.wait(1)
    outcomes = {}
    cancel = threading.Event() if abandon == "cancel" else None
    caller = launch_load(manager, "embed", 0.15 if abandon == "deadline" else 2, outcomes, "caller", cancel)
    try:
        deadline = time.monotonic() + 1
        while executor.get_queue_depth() != 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert executor.get_queue_depth() == 1
        with manager._lock:
            operation = manager._load_operations["embed"]
        if cancel is not None:
            cancel.set()
        caller.join(1)
        assert not caller.is_alive()
        expected = DeadlineExceededError if abandon == "deadline" else RequestCancelledError
        assert isinstance(outcomes.get("caller"), expected)
        # No owner execution or queue-inspection side effect is needed to settle
        # the abandoned generation and allow a fresh request to retry it.
        assert operation.completed and operation.event.is_set()
        assert manager.get_state("embed") == ModelState.FAILED
        assert "embed" not in manager._load_operations
        assert "embed" not in manager._loading_events
        assert adapter.load_count == 0
        assert executor.get_queue_depth() == 0
        release.set()
        hold.result(timeout=1)
        manager.ensure_loaded("embed", timeout_s=1)
        assert adapter.load_count == 1
        assert manager.get_state("embed") == ModelState.READY
    finally:
        release.set()
        caller.join(2)


@pytest.mark.parametrize("path", ["external", "owner", "all"])
def test_explicit_eviction_refuses_active_lease_without_changing_residency(residency, path):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    manager.ensure_loaded("embed")
    manager.acquire_lease("embed")
    try:
        with pytest.raises(ModelUnavailableError, match="lease"):
            if path == "external":
                manager.unload("embed")
            elif path == "owner":
                executor.submit(lambda: manager.unload("embed"), timeout_s=1)
            else:
                manager.unload_all()
        assert adapter.is_loaded()
        assert adapter.unload_count == 0
        assert manager.get_state("embed") == ModelState.READY
        assert manager.get_error("embed") is None
        assert manager.get_memory_info()["model_mb"] == 60
        assert "embed" not in manager._evicting_events
    finally:
        manager.release_lease("embed")
    manager.unload("embed")
    assert adapter.unload_count == 1
    assert manager.get_state("embed") == ModelState.UNLOADED


def test_only_last_of_multiple_waiters_cancels_pending_generation(residency):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    entered, release = threading.Event(), threading.Event()
    hold, _, _ = executor.submit_async(lambda: (entered.set(), release.wait(3)), timeout_s=4)
    assert entered.wait(1)
    outcomes = {}
    first_cancel, second_cancel = threading.Event(), threading.Event()
    first = launch_load(manager, "embed", 2, outcomes, "first", first_cancel)
    second = launch_load(manager, "embed", 2, outcomes, "second", second_cancel)
    try:
        deadline = time.monotonic() + 1
        operation = None
        while time.monotonic() < deadline:
            with manager._lock:
                operation = manager._load_operations.get("embed")
                if operation is not None and operation.waiters == 2 and operation.future is not None:
                    break
            time.sleep(0.002)
        assert operation is not None and operation.waiters == 2
        assert operation.cancel_event is not first_cancel
        assert operation.cancel_event is not second_cancel
        first_cancel.set()
        first.join(1)
        assert isinstance(outcomes.get("first"), RequestCancelledError)
        assert operation.waiters == 1
        assert not operation.cancel_event.is_set() and not operation.completed
        second_cancel.set()
        second.join(1)
        assert isinstance(outcomes.get("second"), RequestCancelledError)
        assert operation.waiters == 0
        assert operation.cancel_event.is_set() and operation.completed
        assert operation.future is not None
        assert operation.future.cancelled()
        release.set()
        hold.result(timeout=1)
        executor.submit(lambda: None, timeout_s=1)
        assert adapter.load_count == 0
        assert manager.get_memory_info()["model_mb"] == 0
    finally:
        release.set()
        first_cancel.set()
        second_cancel.set()
        first.join(2)
        second.join(2)


def test_intrinsic_load_deadline_is_bounded_independently_of_waiter(residency, monkeypatch):
    executor, manager = residency
    monkeypatch.setattr(manager_module, "_LOAD_OPERATION_TIMEOUT_S", 0.1)
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    entered, release = threading.Event(), threading.Event()
    hold, _, _ = executor.submit_async(lambda: (entered.set(), release.wait(3)), timeout_s=4)
    assert entered.wait(1)
    outcomes = {}
    caller = launch_load(manager, "embed", 2, outcomes, "caller")
    try:
        deadline = time.monotonic() + 1
        while executor.get_queue_depth() != 1 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert executor.get_queue_depth() == 1
        with manager._lock:
            operation = manager._load_operations["embed"]
        time.sleep(max(0, operation.deadline - time.monotonic()) + 0.01)
        assert executor.get_queue_depth() == 0  # Prune before the owner is free.
        caller.join(1)
        assert isinstance(outcomes.get("caller"), DeadlineExceededError)
        assert operation.completed and not operation.started
        assert adapter.load_count == 0
        release.set()
        hold.result(timeout=1)
        manager.ensure_loaded("embed", timeout_s=1)
        assert adapter.load_count == 1
        assert manager.get_state("embed") == ModelState.READY
    finally:
        release.set()
        caller.join(2)


def test_explicit_eviction_rechecks_lease_after_waiting_in_owner_queue(residency, monkeypatch):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    manager.ensure_loaded("embed")
    entered, release, queued = threading.Event(), threading.Event(), threading.Event()
    hold, _, _ = executor.submit_async(lambda: (entered.set(), release.wait(3)), timeout_s=4)
    assert entered.wait(1)
    original = executor.submit_async
    outcomes = {}

    def observe_submission(fn, *args, **kwargs):
        result = original(fn, *args, **kwargs)
        if kwargs.get("description") == "Unload embed":
            queued.set()
        return result

    monkeypatch.setattr(executor, "submit_async", observe_submission)

    def unloader():
        try:
            manager.unload("embed")
            outcomes["unload"] = "unloaded"
        except Exception as exc:
            outcomes["unload"] = exc

    caller = threading.Thread(target=unloader)
    caller.start()
    try:
        assert queued.wait(1)
        manager.acquire_lease("embed")
        release.set()
        hold.result(timeout=1)
        caller.join(1)
        assert not caller.is_alive()
        assert isinstance(outcomes.get("unload"), ModelUnavailableError)
        assert "lease" in str(outcomes["unload"])
        assert adapter.is_loaded() and adapter.unload_count == 0
        assert manager.get_state("embed") == ModelState.READY
        assert manager.get_memory_info()["model_mb"] == 60
    finally:
        release.set()
        manager.release_lease("embed")
        caller.join(2)
    manager.unload("embed")
    assert adapter.unload_count == 1


def test_terminal_disposal_ignores_stale_leases_only_after_owner_exit(residency):
    executor, manager = residency

    class TrackingAdapter(TransactionalAdapter):
        unload_thread = None

        def unload(self):
            self.unload_thread = threading.current_thread()
            super().unload()

    adapter = TrackingAdapter()
    manager.register_adapter("embed", adapter)
    manager.ensure_loaded("embed")
    manager.acquire_lease("embed")
    entered, release = threading.Event(), threading.Event()
    hold, _, _ = executor.submit_async(lambda: (entered.set(), release.wait(3)), timeout_s=4)
    assert entered.wait(1)
    try:
        executor.shutdown(timeout=0.01)
        assert not executor.is_accepting() and executor.is_worker_alive()
        with pytest.raises(ModelUnavailableError, match="lease"):
            manager.unload("embed")
        assert adapter.is_loaded() and adapter.unload_count == 0
        release.set()
        hold.result(timeout=1)
        executor.join_worker(timeout=1)
        assert not executor.is_worker_alive()
        manager.unload("embed")
        assert adapter.unload_count == 1
        assert adapter.unload_thread is threading.current_thread()
        assert manager.get_state("embed") == ModelState.UNLOADED
        assert manager.get_memory_info()["model_mb"] == 0
    finally:
        release.set()
        manager.release_lease("embed")


@pytest.mark.parametrize("reload", [False, True])
def test_late_load_submission_cannot_resurrect_explicitly_evicted_generation(residency, monkeypatch, reload):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    entered, evict = threading.Event(), threading.Event()
    submitting, allow_submission = threading.Event(), threading.Event()
    original = executor.submit_async
    submissions = []

    def delayed_submission(fn, *args, **kwargs):
        if kwargs.get("description") == "Load embed":
            submitting.set()
            assert allow_submission.wait(3)
            result = original(fn, *args, **kwargs)
            submissions.append(result[0])
            return result
        return original(fn, *args, **kwargs)

    monkeypatch.setattr(executor, "submit_async", delayed_submission)

    def owner_job():
        entered.set()
        assert evict.wait(3)
        manager.unload("embed")
        assert manager.get_state("embed") == ModelState.UNLOADED
        if reload:
            manager.ensure_loaded("embed", timeout_s=1)

    owner, _, _ = executor.submit_async(owner_job, timeout_s=4)
    assert entered.wait(1)
    outcomes = {}
    caller = launch_load(manager, "embed", 2, outcomes, "caller")
    try:
        assert submitting.wait(1)
        with manager._lock:
            abandoned = manager._load_operations["embed"]
        evict.set()
        owner.result(timeout=1)
        assert abandoned.completed and abandoned.cancel_event.is_set()
        expected_state = ModelState.READY if reload else ModelState.UNLOADED
        assert manager.get_state("embed") == expected_state
        allow_submission.set()
        caller.join(1)
        assert not caller.is_alive()
        assert isinstance(outcomes.get("caller"), RequestCancelledError)
        executor.submit(lambda: None, timeout_s=1)
        assert len(submissions) == 1 and submissions[0].done()
        assert abandoned.waiters == 0
        assert adapter.load_count == int(reload)
        assert adapter.unload_count == 1
        assert manager.get_state("embed") == expected_state
        assert manager.get_error("embed") is None
        assert "embed" not in manager._load_operations
        assert "embed" not in manager._loading_events
    finally:
        evict.set()
        allow_submission.set()
        caller.join(2)


def test_reentrant_explicit_eviction_cannot_invalidate_active_load(residency):
    _, manager = residency

    class ReentrantAdapter(TransactionalAdapter):
        def load(self):
            with pytest.raises(ModelUnavailableError, match="active load transaction"):
                manager.unload("embed")
            super().load()

    adapter = ReentrantAdapter()
    manager.register_adapter("embed", adapter)
    manager.ensure_loaded("embed")
    assert adapter.is_loaded() and adapter.load_count == 1
    assert adapter.unload_count == 0
    assert manager.get_state("embed") == ModelState.READY


def test_prestart_future_failure_cannot_make_inline_takeover_report_false_ready(residency, monkeypatch):
    executor, manager = residency
    adapter = TransactionalAdapter()
    manager.register_adapter("embed", adapter)
    entered, take_over = threading.Event(), threading.Event()
    original_load = manager._do_load_internal
    injected = threading.Event()

    def fail_before_inline_claim(stage, ad, operation):
        if executor.is_owner_thread() and not injected.is_set():
            injected.set()
            assert operation.future is not None
            operation.future.set_exception(DeadlineExceededError("prestart generation failure"))
        return original_load(stage, ad, operation)

    monkeypatch.setattr(manager, "_do_load_internal", fail_before_inline_claim)

    def owner_job():
        entered.set()
        assert take_over.wait(3)
        with pytest.raises(DeadlineExceededError, match="prestart generation failure"):
            manager.ensure_loaded("embed", timeout_s=1)
        assert adapter.load_count == 0
        manager.ensure_loaded("embed", timeout_s=1)

    owner, _, _ = executor.submit_async(owner_job, timeout_s=4)
    assert entered.wait(1)
    outcomes = {}
    caller = launch_load(manager, "embed", 2, outcomes, "caller")
    try:
        deadline = time.monotonic() + 1
        operation = None
        while time.monotonic() < deadline:
            with manager._lock:
                operation = manager._load_operations.get("embed")
                if operation is not None and operation.future is not None:
                    break
            time.sleep(0.002)
        assert operation is not None and operation.future is not None
        take_over.set()
        owner.result(timeout=1)
        caller.join(1)
        assert not caller.is_alive()
        assert isinstance(outcomes.get("caller"), DeadlineExceededError)
        executor.submit(lambda: None, timeout_s=1)
        assert adapter.load_count == 1
        assert manager.get_state("embed") == ModelState.READY
        assert manager.get_error("embed") is None
    finally:
        take_over.set()
        caller.join(2)
