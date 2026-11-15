"""Barrier-controlled executor regressions; no models or Metal operations."""

from concurrent.futures import CancelledError, wait
from contextlib import contextmanager
import threading
import time

import pytest

from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    OverloadedError,
    RequestCancelledError,
    WorkerUnavailableError,
)


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.005)
    return predicate()


@contextmanager
def blocked_owner(max_queue_size=2, max_consecutive_interactive=4):
    executor = GPUExecutor(max_queue_size, max_consecutive_interactive)
    entered = threading.Event()
    release = threading.Event()

    def native_work():
        entered.set()
        assert release.wait(5.0), "test did not release its synthetic native work"
        return "native completed"

    future, event, _ = executor.submit_async(native_work, description="native work")
    try:
        assert entered.wait(1.0)
        yield executor, release, future, event
    finally:
        release.set()
        executor.shutdown(timeout=2.0)
        assert not executor.is_worker_alive()


def start_caller(fn):
    finished = threading.Event()
    outcome = []

    def call():
        try:
            outcome.append(fn())
        except BaseException as exc:
            outcome.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    return thread, finished, outcome


def test_cancelled_pending_submit_returns_and_reclaims_capacity_before_owner_finishes():
    with blocked_owner(max_queue_size=1) as (executor, release, active, _):
        cancellation = threading.Event()
        pending_executed = threading.Event()
        caller, finished, outcome = start_caller(
            lambda: executor.submit(
                pending_executed.set, cancel_event=cancellation,
                timeout_s=4.0, description="cancelled pending",
            )
        )
        try:
            assert wait_until(lambda: executor.get_queue_depth() == 1)
            cancellation.set()
            assert finished.wait(0.4), "pending caller remained blocked behind native work"
            assert isinstance(outcome[0], RequestCancelledError)
            with executor._cv:
                assert not executor._queue, "cancelled waiter retained its pending job"
            assert executor.get_queue_depth() == 0
            fresh, _, _ = executor.submit_async(lambda: "fresh", timeout_s=2.0)
            assert not active.done()
            assert not pending_executed.is_set()
            release.set()
            assert active.result(timeout=1.0) == "native completed"
            assert fresh.result(timeout=1.0) == "fresh"
            assert not pending_executed.is_set()
        finally:
            release.set()
            caller.join(2.0)
        assert not caller.is_alive()


def test_cancelled_running_submit_releases_waiter_not_native_ownership():
    executor = GPUExecutor(max_queue_size=1)
    entered = threading.Event()
    release = threading.Event()
    cancellation = threading.Event()
    active = threading.Event()
    order = []

    def native_work():
        assert executor.is_owner_thread()
        active.set()
        entered.set()
        try:
            assert release.wait(5.0)
            order.append("native completed")
        finally:
            active.clear()

    def unload_work():
        assert executor.is_owner_thread()
        assert not active.is_set(), "unload overlapped a still-owned native operation"
        order.append("unload")

    caller, finished, outcome = start_caller(
        lambda: executor.submit(
            native_work, cancel_event=cancellation, timeout_s=4.0,
            description="active native work",
        )
    )
    try:
        assert entered.wait(1.0)
        cancellation.set()
        assert finished.wait(0.4), "running caller did not observe cancellation promptly"
        assert isinstance(outcome[0], RequestCancelledError)
        progress = executor.get_worker_progress()
        assert progress["worker_alive"] and progress["is_accepting"]
        assert progress["active_job"] and not progress["is_idle"]
        assert progress["active_job_description"] == "active native work"
        assert progress["active_job_age_s"] is not None
        assert progress["completed_sequence"] == 0
        unload, _, _ = executor.submit_async(unload_work, timeout_s=2.0)
        assert not unload.done()
        assert active.is_set()
        release.set()
        unload.result(timeout=1.0)
        assert wait_until(lambda: executor.get_worker_progress()["is_idle"])
        assert executor.get_worker_progress()["completed_sequence"] == 2
        assert order == ["native completed", "unload"]
    finally:
        release.set()
        caller.join(2.0)
        executor.shutdown(timeout=2.0)
    assert not caller.is_alive()


def test_timed_out_pending_submits_reclaim_all_queue_capacity():
    with blocked_owner(max_queue_size=2) as (executor, release, active, _):
        executed = []
        callers = [
            start_caller(lambda: executor.submit(lambda: executed.append("stale"), timeout_s=0.08))
            for _ in range(2)
        ]
        try:
            for _, finished, outcome in callers:
                assert finished.wait(0.5)
                assert isinstance(outcome[0], DeadlineExceededError)
            with executor._cv:
                assert not executor._queue, "expired waiters retained their pending jobs"
            assert executor.get_queue_depth() == 0
            fresh, _, _ = executor.submit_async(lambda: "fresh")
            assert not active.done()
            release.set()
            assert fresh.result(timeout=1.0) == "fresh"
            assert executed == []
        finally:
            release.set()
            for caller, _, _ in callers:
                caller.join(2.0)


@pytest.mark.parametrize("stale_kind", ["cancel_event", "cancel_future", "deadline"])
def test_admission_reclaims_stale_async_jobs_while_native_owner_is_busy(stale_kind):
    with blocked_owner(max_queue_size=1) as (executor, release, active, _):
        executed = threading.Event()
        stale, event, _ = executor.submit_async(
            executed.set, timeout_s=-1.0 if stale_kind == "deadline" else 3.0,
        )
        if stale_kind == "cancel_event":
            event.set()
        elif stale_kind == "cancel_future":
            assert stale.cancel()

        # Do not query queue status first: admission itself must reclaim capacity.
        fresh, _, _ = executor.submit_async(lambda: "fresh", timeout_s=2.0)
        error_type = {
            "cancel_event": RequestCancelledError,
            "cancel_future": CancelledError,
            "deadline": DeadlineExceededError,
        }[stale_kind]
        with pytest.raises(error_type):
            stale.result(timeout=0.2)
        done, not_done = wait([stale], timeout=0.2)
        assert done == {stale} and not not_done
        assert not active.done()
        assert not executed.is_set()
        assert executor.get_queue_depth() == 1
        assert executor.is_overloaded()
        with pytest.raises(OverloadedError):
            executor.submit_async(lambda: "still full")
        release.set()
        assert fresh.result(timeout=1.0) == "fresh"
        assert not executed.is_set()


@pytest.mark.parametrize("query", ["get_queue_depth", "is_overloaded", "get_worker_progress"])
def test_queue_status_excludes_cancelled_pending_work(query):
    with blocked_owner(max_queue_size=1) as (executor, _, active, _):
        future, cancellation, _ = executor.submit_async(lambda: pytest.fail("cancelled job ran"))
        cancellation.set()
        result = getattr(executor, query)()
        if query == "get_queue_depth":
            assert result == 0
        elif query == "is_overloaded":
            assert result is False
        else:
            assert result["queue_depth"] == 0
            assert result["active_job"]
            assert not result["is_idle"]
            assert result["completed_sequence"] == 0
        with pytest.raises(RequestCancelledError):
            future.result(timeout=0.2)
        assert not active.done()


def test_running_future_cannot_be_cancelled_or_release_owner_slot():
    with blocked_owner() as (executor, release, active, _):
        assert active.running()
        assert active.cancel() is False
        assert executor.get_worker_progress()["active_job"]
        next_job, _, _ = executor.submit_async(lambda: "next")
        assert not next_job.done()
        release.set()
        assert active.result(timeout=1.0) == "native completed"
        assert next_job.result(timeout=1.0) == "next"


def test_future_cancelled_in_queue_never_executes_even_without_fresh_admission():
    with blocked_owner() as (executor, release, _, _):
        executed = threading.Event()
        pending, _, _ = executor.submit_async(executed.set)
        assert pending.cancel()
        release.set()
        assert wait_until(lambda: executor.get_worker_progress()["is_idle"])
        assert not executed.is_set()
        assert executor.get_worker_progress()["completed_sequence"] == 1


def test_fairness_boundary_selects_oldest_bulk_not_first_heap_array_match():
    with blocked_owner(max_queue_size=3, max_consecutive_interactive=1) as (executor, release, _, _):
        order = []
        futures = [
            executor.submit_async(lambda name=name: order.append(name), priority=priority)[0]
            for name, priority in [("oldest bulk", 1), ("newer bulk", 1), ("interactive", 0)]
        ]
        # In this heap the newer bulk job sits before the oldest in array order.
        with executor._cv:
            bulk_sequences = [job.seq for job in executor._queue if job.priority > 0]
        assert bulk_sequences != sorted(bulk_sequences)
        release.set()
        for future in futures:
            future.result(timeout=1.0)
        assert order == ["oldest bulk", "interactive", "newer bulk"]


def test_fairness_preserves_priority_then_fifo_among_eligible_bulk_jobs():
    with blocked_owner(max_queue_size=4, max_consecutive_interactive=1) as (executor, release, _, _):
        order = []
        futures = [
            executor.submit_async(lambda name=name: order.append(name), priority=priority)[0]
            for name, priority in [
                ("lower priority bulk", 2), ("oldest eligible bulk", 1),
                ("newer eligible bulk", 1), ("interactive", 0),
            ]
        ]
        release.set()
        for future in futures:
            future.result(timeout=1.0)
        assert order == [
            "oldest eligible bulk", "interactive", "newer eligible bulk", "lower priority bulk",
        ]


def test_shutdown_wakes_pending_waits_rejects_admission_and_retains_active_progress():
    with blocked_owner(max_queue_size=2) as (executor, release, active, active_cancellation):
        pending_executed = threading.Event()
        caller, finished, outcome = start_caller(
            lambda: executor.submit(pending_executed.set, timeout_s=4.0)
        )
        try:
            assert wait_until(lambda: executor.get_queue_depth() == 1)
            pending, _, _ = executor.submit_async(pending_executed.set)
            executor.shutdown(timeout=0.01)
            assert finished.wait(0.4)
            assert isinstance(outcome[0], WorkerUnavailableError)
            with pytest.raises(WorkerUnavailableError):
                pending.result(timeout=0.2)
            for submit in [executor.submit, executor.submit_async]:
                with pytest.raises(WorkerUnavailableError):
                    submit(lambda: "rejected")
            assert active_cancellation.is_set()
            assert not active.done()
            assert not pending_executed.is_set()
            progress = executor.get_worker_progress()
            assert progress["worker_alive"] and progress["active_job"]
            assert not progress["is_accepting"] and not progress["is_idle"]
            assert progress["active_job_description"] == "native work"
            assert progress["active_job_age_s"] is not None
            assert progress["queue_depth"] == 0
            assert progress["completed_sequence"] == 0
            release.set()
            assert active.result(timeout=1.0) == "native completed"
            executor.join_worker(timeout=1.0)
            progress = executor.get_worker_progress()
            assert not progress["worker_alive"] and not progress["is_accepting"]
            assert progress["is_idle"] and not progress["active_job"]
            assert progress["active_job_description"] is None
            assert progress["active_job_age_s"] is None
            assert progress["completed_sequence"] == 1
            assert not pending_executed.is_set()
        finally:
            release.set()
            caller.join(2.0)
        assert not caller.is_alive()


def test_cancellation_after_selection_never_executes_or_counts_the_closure():
    with blocked_owner(max_queue_size=3) as (executor, release, _, _):
        stale, stale_event, _ = executor.submit_async(lambda: None, priority=1)
        executed = threading.Event()
        selected, cancellation, _ = executor.submit_async(executed.set, description="selected")
        after, _, _ = executor.submit_async(lambda: "after")
        observed = []

        def cancel_selected(_):
            # Rejected-future callbacks run between selecting the next job and
            # executing its callable, providing a deterministic pre-forward race.
            observed.append((executor.is_owner_thread(), executor.get_worker_progress()["active_job_description"]))
            cancellation.set()

        stale.add_done_callback(cancel_selected)
        stale_event.set()
        release.set()
        with pytest.raises(RequestCancelledError):
            selected.result(timeout=1.0)
        assert after.result(timeout=1.0) == "after"
        assert wait_until(lambda: executor.get_worker_progress()["is_idle"])
        assert observed == [(True, "selected")]
        assert not executed.is_set()
        assert executor.get_worker_progress()["completed_sequence"] == 2


@pytest.mark.parametrize("operation", ["shutdown", "reclaim"])
def test_rejected_future_callbacks_do_not_hold_the_queue_lock(operation):
    with blocked_owner(max_queue_size=1) as (executor, _, _, _):
        future, cancellation, _ = executor.submit_async(lambda: None)
        callback_observations = []
        callback_threads = []

        def read_progress(_):
            thread, finished, outcome = start_caller(executor.get_worker_progress)
            callback_threads.append(thread)
            callback_observations.append((finished.wait(0.4), outcome))

        future.add_done_callback(read_progress)
        try:
            if operation == "shutdown":
                executor.shutdown(timeout=0.01)
            else:
                cancellation.set()
                assert executor.get_queue_depth() == 0
            assert len(callback_observations) == 1
            finished, outcome = callback_observations[0]
            assert finished, "future callback blocked another queue-lock reader"
            assert isinstance(outcome[0], dict)
            assert outcome[0]["active_job"]
        finally:
            for thread in callback_threads:
                thread.join(1.0)


def test_shutdown_from_owner_does_not_attempt_to_join_itself():
    executor = GPUExecutor()
    try:
        future, _, _ = executor.submit_async(lambda: executor.shutdown(timeout=0.01))
        assert future.result(timeout=1.0) is None
        executor.join_worker(timeout=1.0)
        assert not executor.is_worker_alive()
    finally:
        executor.shutdown(timeout=1.0)


def test_callable_timeout_error_is_not_relabelled_as_wait_deadline():
    executor = GPUExecutor()
    expected = TimeoutError("synthetic native failure")

    def fail():
        raise expected

    try:
        with pytest.raises(TimeoutError) as raised:
            executor.submit(fail, timeout_s=2.0)
        assert raised.value is expected
    finally:
        executor.shutdown(timeout=1.0)
