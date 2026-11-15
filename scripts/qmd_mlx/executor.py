"""
executor.py — Single GPU Execution Owner with Priority Scheduling, Fair Interleaving, and Monotonic Deadlines
"""

from __future__ import annotations

import heapq
import sys
import threading
import time
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .protocol import (
    DeadlineExceededError,
    OverloadedError,
    RequestCancelledError,
    WorkerUnavailableError,
)


class _CleanupFuture(Future):
    def cancel(self) -> bool:
        # A caller abandoning its wait must not discard mandatory disposal.
        return False


@dataclass(order=True)
class ExecutionJob:
    priority: int  # 0 = interactive (rerank, generate, query embed), 1 = bulk embed
    enqueue_time: float = field(compare=False)
    seq: int = field(compare=True)  # FIFO tie-breaker for same priority
    deadline: float = field(compare=False)
    fn: Callable[..., Any] = field(compare=False)
    future: Future = field(compare=False)
    cancel_event: threading.Event = field(compare=False)
    description: str = field(compare=False, default="")
    cleanup: bool = field(compare=False, default=False)


class GPUExecutor:
    """
    Single GPU execution owner thread.
    Serializes all Metal/MLX model evaluation, loading, and unloading operations.
    Enforces strict monotonic deadlines, priority scheduling, and fair interleaving.
    """

    MAX_CLEANUP_JOBS = 32

    def __init__(self, max_queue_size: int = 64, max_consecutive_interactive: int = 4):
        self.max_queue_size = max_queue_size
        self.max_consecutive_interactive = max_consecutive_interactive
        self._queue: list[ExecutionJob] = []
        self._cv = threading.Condition()
        self._seq = 0
        self._running = True
        self._worker_exiting = False
        self._cleanup_jobs: dict[Future, ExecutionJob] = {}
        self._current_job: Optional[ExecutionJob] = None
        self._current_job_start_time: Optional[float] = None
        self._current_job_description: Optional[str] = None
        self._total_completed_jobs: int = 0
        self._idle_callbacks: list[Callable[[], None]] = []
        self._consecutive_interactive = 0

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="MLX-GPU-Executor")
        self._worker_thread.start()

    def register_idle_callback(self, cb: Callable[[], None]):
        """Registers a callback to be called on the owner thread when executor is idle."""
        with self._cv:
            if cb not in self._idle_callbacks and len(self._idle_callbacks) < 32:
                self._idle_callbacks.append(cb)

    def is_alive(self) -> bool:
        with self._cv:
            return self._running and self._worker_thread.is_alive()

    def is_accepting(self) -> bool:
        """Returns True if the executor is active and accepting new submissions."""
        with self._cv:
            return self._running

    def is_worker_alive(self) -> bool:
        """Returns True if the underlying OS worker thread is currently running."""
        return self._worker_thread.is_alive()

    def reserve_cleanup(self, fn: Callable[[], Any], description: str = "") -> Future:
        """Reserve bounded disposal capacity before publishing a resource.

        The returned non-cancellable handle is dormant until submit_cleanup().
        Cleanup reservations are separate from normal request queue capacity;
        acquiring one can fail, but activating an acquired slot cannot overload.
        """
        with self._cv:
            if len(self._cleanup_jobs) >= self.MAX_CLEANUP_JOBS:
                raise OverloadedError("GPU Executor cleanup reservations are full")
            future = _CleanupFuture()
            self._cleanup_jobs[future] = ExecutionJob(
                priority=0, enqueue_time=0.0, seq=0, deadline=float("inf"),
                fn=fn, future=future, cancel_event=threading.Event(),
                description=description, cleanup=True,
            )
        return future

    def submit_cleanup(self, future: Future) -> Future:
        """Activate a reserved cleanup once, even while normal work is stopping.

        Accepted cleanup survives shutdown and runs on the owner after active
        native work. Refusal means owner exit has already been committed under
        the same lock; callers may join then dispose safely without a helper.
        """
        with self._cv:
            job = self._cleanup_jobs.get(future)
            if job is None:
                if future.done():
                    return future
                raise ValueError("Unknown GPU Executor cleanup reservation")
            if job.seq:
                return future
            if self._worker_exiting or not self._worker_thread.is_alive():
                del self._cleanup_jobs[future]
                raise WorkerUnavailableError("GPU Executor owner has exited")
            self._seq += 1
            job.seq = self._seq
            job.enqueue_time = time.monotonic()
            heapq.heappush(self._queue, job)
            self._cv.notify()
        return future

    def _request_queue_depth_locked(self) -> int:
        return sum(not job.cleanup for job in self._queue)

    @staticmethod
    def _settle_rejected_jobs(rejected: Iterable[tuple[ExecutionJob, BaseException]]):
        """Settle removed jobs outside _cv: Future callbacks may reenter the executor."""
        for job, error in rejected:
            if not job.future.done():
                try:
                    job.future.set_exception(error)
                except InvalidStateError:
                    # Future.cancel() may win between done() and set_exception().
                    pass
            if job.future.cancelled():
                # Removed jobs will never reach the worker's normal claim path.
                # Notify concurrent.futures.wait()/as_completed() of cancellation.
                job.future.set_running_or_notify_cancel()
            job.cancel_event.set()

    def _prune_queue_locked(self) -> list[tuple[ExecutionJob, BaseException]]:
        """Reclaim only pending work; never change the live native execution owner."""
        now = time.monotonic()
        pending = []
        rejected = []
        for job in self._queue:
            if job.cleanup:
                pending.append(job)
                continue
            if job.future.done() or job.cancel_event.is_set():
                error = RequestCancelledError(f"Job '{job.description}' was cancelled")
            elif now >= job.deadline:
                error = DeadlineExceededError(f"Job '{job.description}' expired in queue")
            else:
                pending.append(job)
                continue
            rejected.append((job, error))
        if rejected:
            self._queue = pending
            heapq.heapify(self._queue)
            self._cv.notify_all()
        return rejected

    def _remove_pending_job(self, future: Future, error: BaseException):
        rejected = []
        with self._cv:
            for idx, job in enumerate(self._queue):
                if job.future is future:
                    self._queue.pop(idx)
                    heapq.heapify(self._queue)
                    rejected.append((job, error))
                    self._cv.notify_all()
                    break
        self._settle_rejected_jobs(rejected)

    def is_overloaded(self) -> bool:
        """Returns True if the live pending queue is at maximum capacity."""
        return self.get_queue_depth() >= self.max_queue_size

    def get_queue_depth(self) -> int:
        """Returns live pending requests; reserved cleanup uses separate capacity."""
        with self._cv:
            rejected = self._prune_queue_locked()
            depth = self._request_queue_depth_locked()
        self._settle_rejected_jobs(rejected)
        return depth

    def get_worker_progress(self) -> dict[str, Any]:
        """Returns a snapshot of live worker execution progress and job duration under lock."""
        with self._cv:
            rejected = self._prune_queue_locked()
            now = time.monotonic()
            active_job_age_s = (
                round(now - self._current_job_start_time, 3)
                if self._current_job_start_time is not None
                else None
            )
            progress = {
                "worker_alive": self._worker_thread.is_alive(),
                "is_accepting": self._running,
                "is_idle": self._current_job is None and len(self._queue) == 0,
                "active_job": self._current_job is not None,
                "active_job_description": self._current_job_description,
                "active_job_age_s": active_job_age_s,
                "completed_sequence": self._total_completed_jobs,
                "queue_depth": self._request_queue_depth_locked(),
            }
        self._settle_rejected_jobs(rejected)
        return progress

    def join_worker(self, timeout: float = 5.0):
        """Waits for the background worker thread to join."""
        if self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
            self._worker_thread.join(timeout=timeout)

    def is_owner_thread(self) -> bool:
        """Returns True if the current thread is the dedicated GPU worker thread."""
        return threading.current_thread() == self._worker_thread

    def submit(
        self,
        fn: Callable[..., Any],
        priority: int = 0,
        timeout_s: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
        description: str = "",
    ) -> Any:
        """
        Submits a callable to run on the GPU execution owner thread.
        Blocks until completion or deadline/cancellation.
        """
        fut, event, deadline = self.submit_async(
            fn, priority=priority, timeout_s=timeout_s,
            cancel_event=cancel_event, description=description,
        )
        try:
            while True:
                # A settled queue/shutdown error takes precedence over cancellation.
                # Read callable exceptions without confusing TimeoutError with a wait timeout.
                if fut.done():
                    result = fut.result()
                    if time.monotonic() >= deadline:
                        raise DeadlineExceededError(
                            f"Request '{description}' timed out after {timeout_s:.1f}s (deadline exceeded)"
                        )
                    if event.is_set():
                        raise RequestCancelledError(f"Request '{description}' was cancelled")
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeadlineExceededError(
                        f"Request '{description}' timed out after {timeout_s:.1f}s (deadline exceeded)"
                    )
                if event.is_set():
                    raise RequestCancelledError(f"Request '{description}' was cancelled")
                try:
                    # Event.set() cannot wake Future.result(), so bound each wait.
                    fut.result(timeout=min(0.05, remaining))
                except TimeoutError:
                    # The next done() check propagates a callable's own TimeoutError.
                    continue
        except BaseException as exc:
            event.set()
            # Abandon only queued work. A running native callable stays owned until
            # it returns; its future and live progress must not be settled here.
            self._remove_pending_job(fut, exc)
            raise

    def submit_async(
        self,
        fn: Callable[..., Any],
        priority: int = 0,
        timeout_s: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
        description: str = "",
    ) -> tuple[Future, threading.Event, float]:
        """Submits a job and returns (future, cancel_event, deadline)."""
        now = time.monotonic()
        deadline = now + timeout_s
        event = cancel_event or threading.Event()
        fut: Future = Future()

        rejected = []
        admission_error = None
        with self._cv:
            if not self._running:
                admission_error = WorkerUnavailableError("GPU Executor worker thread is not running")
            else:
                rejected = self._prune_queue_locked()
                depth = self._request_queue_depth_locked()
                if depth >= self.max_queue_size:
                    admission_error = OverloadedError(
                        f"MLX GPU Executor queue is full ({depth}/{self.max_queue_size} jobs)"
                    )
                else:
                    self._seq += 1
                    job = ExecutionJob(
                        priority=priority,
                        enqueue_time=now,
                        seq=self._seq,
                        deadline=deadline,
                        fn=fn,
                        future=fut,
                        cancel_event=event,
                        description=description,
                    )
                    heapq.heappush(self._queue, job)
                    self._cv.notify()

        self._settle_rejected_jobs(rejected)
        if admission_error is not None:
            raise admission_error
        return fut, event, deadline

    def _worker_loop(self):
        """Main execution loop running exclusively on the dedicated GPU thread."""
        while True:
            try:
                job: Optional[ExecutionJob] = None
                callbacks_to_run: list[Callable[[], None]] = []

                with self._cv:
                    rejected = self._prune_queue_locked()
                    while self._running and not self._queue and not rejected:
                        self._cv.wait(timeout=1.0)
                        rejected = self._prune_queue_locked()
                        if not self._queue and self._running:
                            callbacks_to_run = list(self._idle_callbacks)
                            break

                    should_stop = not self._running and not self._queue
                    if should_stop:
                        # Fence late cleanup activation before releasing _cv.
                        self._worker_exiting = True
                    if self._queue:
                        # A heap is only partially ordered: its first bulk array
                        # entry need not be the oldest eligible (priority, seq).
                        bulk_idx = None
                        if self._consecutive_interactive >= self.max_consecutive_interactive:
                            bulk_idx = min(
                                (idx for idx, queued in enumerate(self._queue) if queued.priority > 0),
                                key=lambda idx: (self._queue[idx].priority, self._queue[idx].seq),
                                default=None,
                            )
                        if bulk_idx is not None:
                            job = self._queue.pop(bulk_idx)
                            heapq.heapify(self._queue)
                        else:
                            job = heapq.heappop(self._queue)

                        # Atomically claim the Future before calling native work.
                        # Future.cancel() must never cancel an already running job.
                        if job.future.set_running_or_notify_cancel():
                            self._current_job = job
                            self._current_job_start_time = time.monotonic()
                            self._current_job_description = job.description
                        else:
                            job.cancel_event.set()
                            job = None

                self._settle_rejected_jobs(rejected)
                if should_stop:
                    break

                # Execute idle callbacks strictly OUTSIDE the condition variable lock
                if callbacks_to_run:
                    for cb in callbacks_to_run:
                        try:
                            cb()
                        except Exception as e:
                            print(f"[mlx-executor] Idle callback error: {e}", file=sys.stderr)

                if job is None:
                    continue

                executed = False
                try:
                    # Recheck after dequeue: cancellation/shutdown may race with
                    # selection, but must not run a cancelled pending closure.
                    if job.cancel_event.is_set():
                        raise RequestCancelledError(f"Job '{job.description}' was cancelled")
                    if time.monotonic() >= job.deadline:
                        job.cancel_event.set()
                        raise DeadlineExceededError(f"Job '{job.description}' expired in queue")

                    executed = True
                    result = job.fn()
                    if not job.future.done():
                        job.future.set_result(result)
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                finally:
                    with self._cv:
                        if job.cleanup:
                            self._cleanup_jobs.pop(job.future, None)
                        if executed:
                            self._total_completed_jobs += 1
                            if job.priority == 0:
                                self._consecutive_interactive += 1
                            else:
                                self._consecutive_interactive = 0
                        self._current_job = None
                        self._current_job_start_time = None
                        self._current_job_description = None
            except Exception as e:
                # Top-level guard prevents unhandled bookkeeping errors from silently terminating the worker
                print(f"[mlx-executor] Worker loop unhandled error: {e}", file=sys.stderr)
                with self._cv:
                    self._current_job = None
                    self._current_job_start_time = None
                    self._current_job_description = None
                time.sleep(0.01)

    def shutdown(self, timeout: float = 5.0):
        """Cancel pending requests, retain cleanup, and never detach native work."""
        with self._cv:
            self._running = False
            if self._current_job is not None and not self._current_job.cleanup:
                self._current_job.cancel_event.set()
            rejected = [
                (job, WorkerUnavailableError("Executor is shutting down"))
                for job in self._queue if not job.cleanup
            ]
            self._queue = [job for job in self._queue if job.cleanup]
            heapq.heapify(self._queue)
            self._cv.notify_all()

        self._settle_rejected_jobs(rejected)
        self.join_worker(timeout=timeout)
