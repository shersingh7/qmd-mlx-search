"""
runtime.py — Model-aware MLX embedding runtime behind single GPU execution owner
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from typing import Any, Optional
import numpy as np

from .adapters import (
    BaseEmbeddingAdapter,
    TokenizedBatch,
    resolve_embedding_adapter,
)
from .batching import BatchPlanner
from .adapters.embedding import EmbeddingPreparation
from .executor import GPUExecutor
from .model_manager import ModelResidencyManager, ModelState
from .protocol import (
    DeadlineExceededError,
    InvalidInputError,
    MLXRuntimeError,
    MLXServerError,
    ModelUnavailableError,
    OutOfMemoryError,
    OverloadedError,
    ProtocolError,
    RequestCancelledError,
    UnsupportedModelError,
    WorkerUnavailableError,
    WorkClass,
)

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


from dataclasses import dataclass


@dataclass
class AdmissionLease:
    text_count: int
    total_bytes: int
    acquired_time: float


@dataclass(frozen=True)
class EmbeddingResult:
    """Owned CPU output plus the immutable identity captured by its operation.

    Keep this additive to submit_embed's public ndarray contract. Each descriptor
    access returns a fresh dictionary, never a mutable global 'last descriptor'.
    """

    embeddings: np.ndarray
    descriptor_json: str

    @property
    def descriptor(self) -> dict[str, Any]:
        return json.loads(self.descriptor_json)


class _DummyWorkQueue:
    def __init__(self, executor: GPUExecutor):
        self._executor = executor

    def join(self):
        time.sleep(0.05)


class MLXEmbeddingRuntime:
    """
    Manages MLX embedding model lifecycle and execution behind a dedicated
    GPUExecutor owner thread and ModelResidencyManager.
    """

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        dtype_str: str = "float32",
        max_length: int = 2048,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        max_batch_tokens: int = 0,
        executor: Optional[GPUExecutor] = None,
        model_manager: Optional[ModelResidencyManager] = None,
        lazy_load: bool = False,
        max_bulk_microbatch_tokens: int = 512,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.dtype_str = dtype_str
        self.max_length = max_length
        self.revision = revision
        self.trust_remote_code = trust_remote_code

        # Resolve and validate everything that does not need an owner thread
        # before allocating one. Unsupported models and bad budgets must not leak
        # a live executor from an otherwise failed constructor.
        if model_manager is not None:
            if executor is not None and model_manager.executor is not executor:
                raise ValueError("model_manager and runtime must use the same executor")
            executor = model_manager.executor

        self.adapter: BaseEmbeddingAdapter = resolve_embedding_adapter(
            model_name=model_name,
            quantization=quantization,
            dtype_str=dtype_str,
            max_length=max_length,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        if not getattr(self.adapter, "synthetic", False) and not _MLX_AVAILABLE:
            raise MLXRuntimeError("MLX is not installed. Please install mlx and mlx-lm.")

        existing_adapter = model_manager.get_adapter("embed") if model_manager is not None else None
        if existing_adapter is not None and existing_adapter is not self.adapter:
            raise ValueError("model_manager already has an embed adapter")

        # Fairness target and model/hardware budget are deliberately independent.
        # The safe 512-token bulk default is unchanged; higher targets are opt-in.
        self.batch_planner = BatchPlanner(
            max_batch_tokens=max_batch_tokens,
            max_bulk_microbatch_tokens=max_bulk_microbatch_tokens,
        )
        self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))

        # Bounded admission leasing controls
        self.max_concurrent_admissions: int = 32
        self.max_in_flight_admission_bytes: int = 64 * 1024 * 1024  # 64 MB
        self._admission_lock = threading.Condition()
        self._current_admitted_requests: int = 0
        self._current_admitted_bytes: int = 0
        self._shutting_down: bool = False
        self._shutdown_future: Optional[Future] = None
        self._shutdown_cleanup: Optional[Future] = None
        self._shutdown_needs_disposal = False
        self._owns_adapter_registration = False
        self._owns_executor = executor is None

        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.total_latency_ms: float = 0.0
        self.compiled_shapes: set[tuple[int, ...]] = set()

        self.executor = executor if executor is not None else GPUExecutor()
        self.model_manager: ModelResidencyManager
        try:
            self.model_manager = model_manager if model_manager is not None else ModelResidencyManager(self.executor)
            if existing_adapter is None:
                # Acquire bounded teardown capacity before registering/loading
                # weights. Normal request saturation cannot strand this slot.
                self._shutdown_cleanup = self.executor.reserve_cleanup(
                    self._unload_owned_embedding, description="Shutdown embedding runtime",
                )
                # Set ownership before registration so a partially successful
                # register_adapter can also be rolled back safely.
                self._owns_adapter_registration = True
                self.model_manager.register_adapter("embed", self.adapter)
            if not lazy_load:
                self.model_manager.ensure_loaded("embed")
                self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))
        except BaseException:
            try:
                self.shutdown()
            except BaseException:
                # A timed-out native load cannot be interrupted safely. If a
                # teardown was queued it still owns cleanup after completion.
                # If setup failed before that was possible, stop only our owner.
                if self._owns_executor and self._shutdown_future is None:
                    self.executor.shutdown()
            raise

    @property
    def in_flight_requests(self) -> int:
        with self._admission_lock:
            return self._current_admitted_requests

    @property
    def in_flight_bytes(self) -> int:
        with self._admission_lock:
            return self._current_admitted_bytes

    def acquire_admission_lease(
        self,
        text_count: int = 0,
        total_bytes: int = 0,
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
        num_texts: Optional[int] = None,
        num_bytes: Optional[int] = None,
    ) -> AdmissionLease:
        """
        Acquires an in-flight admission lease before performing CPU tokenization or large allocations.
        Guarantees bounded request concurrency, byte caps, and executor queue capacity.
        """
        if num_texts is not None:
            text_count = num_texts
        if num_bytes is not None:
            total_bytes = num_bytes

        if text_count < 0 or total_bytes < 0:
            raise InvalidInputError("Invalid admission lease request: count and bytes must be non-negative")
        if text_count > 512:
            raise InvalidInputError(f"Request item count ({text_count}) exceeds maximum allowed limit (512)")
        if total_bytes > self.max_in_flight_admission_bytes:
            raise InvalidInputError(
                f"Request size ({total_bytes} bytes) exceeds maximum in-flight capacity ({self.max_in_flight_admission_bytes} bytes)"
            )

        if self._shutting_down or not self.executor.is_accepting() or not self.executor.is_worker_alive():
            raise WorkerUnavailableError("GPU Executor worker thread is not running")

        now = time.monotonic()
        dl = deadline if deadline is not None else (now + 300.0)

        with self._admission_lock:
            if self.executor.is_overloaded():
                raise OverloadedError(
                    f"MLX GPU Executor queue is full ({self.executor.get_queue_depth()}/{self.executor.max_queue_size} jobs)"
                )

            while (
                self._current_admitted_requests >= self.max_concurrent_admissions
                or (self._current_admitted_bytes + total_bytes > self.max_in_flight_admission_bytes)
            ):
                if self._shutting_down or not self.executor.is_accepting() or not self.executor.is_worker_alive():
                    raise WorkerUnavailableError("GPU Executor is shutting down")
                if cancel_event and cancel_event.is_set():
                    raise RequestCancelledError("Request cancelled while waiting for admission")
                rem = dl - time.monotonic()
                if rem <= 0:
                    raise OverloadedError("Server admission capacity saturated; request timed out waiting for admission slot")
                self._admission_lock.wait(timeout=min(0.2, max(0.01, rem)))

            if self._shutting_down or not self.executor.is_accepting() or not self.executor.is_worker_alive():
                raise WorkerUnavailableError("GPU Executor is shutting down")
            if cancel_event and cancel_event.is_set():
                raise RequestCancelledError("Request cancelled before admission")
            if time.monotonic() > dl:
                raise DeadlineExceededError("Request deadline exceeded before admission")

            self._current_admitted_requests += 1
            self._current_admitted_bytes += total_bytes
            return AdmissionLease(text_count=text_count, total_bytes=total_bytes, acquired_time=time.monotonic())

    def release_admission_lease(self, lease: AdmissionLease):
        """Releases an in-flight admission lease."""
        with self._admission_lock:
            self._current_admitted_requests = max(0, self._current_admitted_requests - 1)
            self._current_admitted_bytes = max(0, self._current_admitted_bytes - lease.total_bytes)
            self._admission_lock.notify_all()

    @property
    def native_dims(self) -> int:
        return self.adapter.native_dims

    @property
    def pooling_strategy(self) -> str:
        return self.adapter.pooling_strategy

    @property
    def idle_unload_s(self) -> float:
        return self.model_manager.idle_unload_s

    @idle_unload_s.setter
    def idle_unload_s(self, value: float):
        self.model_manager.idle_unload_s = float(value)

    @property
    def _weights_loaded(self) -> bool:
        return self.adapter.is_loaded()

    @property
    def _work_queue(self) -> _DummyWorkQueue:
        return _DummyWorkQueue(self.executor)

    def is_ready(self) -> bool:
        return self.adapter.is_loaded() and self.executor.is_alive()

    def _check_request_active(self, deadline: float, cancel_event: threading.Event):
        if self._shutting_down:
            raise WorkerUnavailableError("Embedding runtime is shutting down")
        if cancel_event.is_set():
            raise RequestCancelledError("Embedding request cancelled")
        if time.monotonic() > deadline:
            raise DeadlineExceededError("Embedding request deadline exceeded")

    def _prepare_request(
        self, deadline: float, cancel_event: threading.Event,
        requested_dims: Optional[int] = None,
    ) -> EmbeddingPreparation:
        """Snapshot ready state on CPU, otherwise prepare it on the owner."""
        self._check_request_active(deadline, cancel_event)

        def _prepare_under_owner():
            self._check_request_active(deadline, cancel_event)
            # The lease belongs to native initialization, not its CPU waiter.
            # Release it before the owner can select a following teardown; a
            # delayed waiter must not make mandatory cleanup refuse a stale pin.
            self.model_manager.acquire_lease("embed")
            try:
                self._check_request_active(deadline, cancel_event)
                self.model_manager.ensure_loaded("embed", deadline=deadline, cancel_event=cancel_event)
                self._check_request_active(deadline, cancel_event)
                preparation = self.adapter.capture_preparation(requested_dims=requested_dims)
                self._check_request_active(deadline, cancel_event)
                return preparation
            finally:
                self.model_manager.release_lease("embed")

        # A ready tokenizer needs no owner round trip: interactive callers must
        # be able to tokenize while a bulk forward runs, then queue their own
        # forward at the next microbatch boundary. Snapshot acquisition never
        # waits for the adapter's native load/disposal lock on a CPU caller.
        preparation = self.adapter.try_capture_preparation(requested_dims=requested_dims)
        if preparation is None:
            if self.executor.is_owner_thread():
                preparation = _prepare_under_owner()
            else:
                preparation = self.executor.submit(
                    _prepare_under_owner,
                    priority=0,
                    timeout_s=max(0.0, deadline - time.monotonic()),
                    cancel_event=cancel_event,
                    description="Initialize embedding tokenizer",
                )
        # A paused CPU waiter can resume after owner-side shutdown completed.
        self._check_request_active(deadline, cancel_event)
        return preparation

    def tokenize(
        self,
        texts: list[str],
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[list[int]]:
        """
        Tokenizes on CPU from an atomically captured, verified tokenizer snapshot.
        Cold initialization is owner-only; no weight lease spans CPU tokenization.
        """
        if not texts:
            return []

        if len(texts) > 512:
            raise InvalidInputError(f"Request batch size ({len(texts)}) exceeds maximum allowed texts (512)")

        total_bytes = 0
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise InvalidInputError(f"Text at index {i} is empty or whitespace")
            tb = len(t.encode("utf-8"))
            if tb > 256 * 1024:
                raise InvalidInputError(f"Text at index {i} exceeds max length of 256KB ({tb} bytes)")
            total_bytes += tb

        if total_bytes > 10 * 1024 * 1024:
            raise InvalidInputError(f"Total request text size ({total_bytes} bytes) exceeds 10MB limit")

        now_mono = time.monotonic()
        req_deadline = deadline if deadline is not None else (now_mono + 120.0)
        event = cancel_event or threading.Event()

        if event.is_set():
            raise RequestCancelledError("Tokenize request cancelled before start")
        if time.monotonic() > req_deadline:
            raise DeadlineExceededError("Tokenize deadline exceeded before start")

        lease = self.acquire_admission_lease(len(texts), total_bytes, deadline=req_deadline, cancel_event=event)
        try:
            preparation = self._prepare_request(req_deadline, event)
            self._check_request_active(req_deadline, event)
            batch = preparation.tokenize_texts(texts)
            self._check_request_active(req_deadline, event)

            return batch.token_ids
        finally:
            self.release_admission_lease(lease)

    def warmup(self):
        """Runs warmup inference passes on GPU."""
        print("[mlx-runtime] Running GPU warmup...")
        t0 = time.time()
        for count in (1, 4):
            sample_texts = ["Warmup embedding Metal compiler verification."] * count
            self.submit_embed(sample_texts, timeout=30.0, is_query=True)
        elapsed = time.time() - t0
        print(f"[mlx-runtime] GPU warmup complete in {elapsed:.2f}s")

    def embed_direct(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
    ) -> np.ndarray:
        return self.submit_embed(texts, requested_dims=requested_dims, is_query=is_query)

    def submit_embed(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
        timeout: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> np.ndarray:
        """Public ndarray interface; identity-aware consumers use the result API."""
        if not texts:
            return np.empty((0, self.adapter.native_dims or 0), dtype=np.float32)
        return self.submit_embed_result(
            texts, requested_dims=requested_dims, is_query=is_query, timeout=timeout,
            cancel_event=cancel_event, deadline=deadline,
        ).embeddings

    def submit_embed_result(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
        timeout: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> EmbeddingResult:
        """
        Submits an embedding request through the GPU execution owner.
        Enforces a single monotonic deadline budget across admission lease, tokenize, queue, and micro-batches.
        Supports micro-batch interleaving so interactive requests yield between bulk steps.
        """
        if not texts:
            raise InvalidInputError("Embedding result requires at least one text")

        # 1. Early bounded input validation
        if len(texts) > 512:
            raise InvalidInputError(f"Request exceeds maximum allowed texts ({len(texts)} > 512)")

        total_bytes = 0
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise InvalidInputError(f"Item at index {i} is not a string")
            if not t.strip():
                raise InvalidInputError(f"Item at index {i} is empty or whitespace")
            if len(t) > 256 * 1024:
                raise InvalidInputError(f"Item at index {i} exceeds max length 256KB")
            total_bytes += len(t.encode("utf-8", errors="ignore"))

        if total_bytes > 10 * 1024 * 1024:
            raise InvalidInputError(f"Total request text size ({total_bytes} bytes) exceeds 10MB limit")

        now_mono = time.monotonic()
        req_deadline = deadline if deadline is not None else (now_mono + timeout)
        event = cancel_event or threading.Event()

        if event.is_set():
            raise RequestCancelledError("Request cancelled before start")
        if time.monotonic() > req_deadline:
            raise DeadlineExceededError("Request deadline exceeded before start")

        t0 = time.time()

        # 2. Acquire bounded admission lease before heavy CPU tokenization
        lease = self.acquire_admission_lease(len(texts), total_bytes, deadline=req_deadline, cancel_event=event)
        try:
            # 3. Capture verified state; tokenize on CPU using ONLY that snapshot.
            preparation = self._prepare_request(req_deadline, event, requested_dims)
            self._check_request_active(req_deadline, event)
            tokenized_batch = preparation.tokenize_texts(texts)
            self._check_request_active(req_deadline, event)

            # Retune batch planner with measured model params BEFORE planning micro-batches!
            self.batch_planner.tune_for_model(preparation.model_params_b)

            # Interactive queries get priority 0, bulk document embeddings get priority 1
            work_class = WorkClass.INTERACTIVE if (is_query or len(texts) <= 2) else WorkClass.BULK
            priority = 0 if work_class == WorkClass.INTERACTIVE else 1

            # 4. Plan micro-batches for fair interleaving at micro-batch boundaries
            sub_batches, micro_batches_indices = self.batch_planner.plan_micro_batches(
                tokenized_batch,
                is_query=is_query,
                work_class=work_class,
            )

            results: list[np.ndarray] = []
            for batch_idx, sub_batch in enumerate(sub_batches):
                self._check_request_active(req_deadline, event)

                rem_timeout = max(0.0, req_deadline - time.monotonic())

                def _forward_sub_batch(sb=sub_batch) -> np.ndarray:
                    self._check_request_active(req_deadline, event)

                    def _run_forward(b: TokenizedBatch) -> np.ndarray:
                        self._check_request_active(req_deadline, event)
                        self.adapter.assert_preparation_current(preparation)
                        if len(b) > 0:
                            padded_shape = (len(b), max(b.lengths) if b.lengths else 1)
                            if len(self.compiled_shapes) >= 256:
                                self.compiled_shapes.clear()
                            self.compiled_shapes.add(padded_shape)
                        value = self.adapter.forward_batch(b, requested_dims=requested_dims)
                        self._check_request_active(req_deadline, event)
                        self.adapter.assert_preparation_current(preparation)
                        return value

                    # A short native-only lease protects against reentrant stage
                    # eviction, not against fair interleaving between owner jobs.
                    self.model_manager.acquire_lease("embed")
                    try:
                        self.adapter.assert_preparation_current(preparation)
                        return self.batch_planner.execute_sub_batch_with_retry(sb, _run_forward)
                    finally:
                        self.model_manager.release_lease("embed")

                batch_res = self.executor.submit(
                    _forward_sub_batch,
                    priority=priority,
                    timeout_s=rem_timeout,
                    cancel_event=event,
                    description=f"Embed batch {batch_idx+1}/{len(sub_batches)} ({'query' if is_query else 'doc'})",
                )
                results.append(batch_res)

            # 5. Restore original input ordering
            res = self.batch_planner.restore_ordering(results, micro_batches_indices, len(tokenized_batch))
            self._check_request_active(req_deadline, event)

            latency_ms = (time.time() - t0) * 1000
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.model_manager.touch("embed")
            self.model_manager.update_peak_memory()
            return EmbeddingResult(res, preparation.descriptor_json)
        except MLXServerError:
            raise
        except Exception as exc:
            raise MLXRuntimeError(f"Embedding execution failed: {exc}")
        finally:
            self.release_admission_lease(lease)

    def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
        return self.adapter.get_descriptor(requested_dims=requested_dims)

    def get_memory_info(self) -> dict[str, Any]:
        return self.model_manager.get_memory_info()

    def get_stats_info(self) -> dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_ms = round(self.total_latency_ms / self.total_requests, 2) if self.total_requests > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "avg_ms": avg_ms,
            "compiled_shapes": len(self.compiled_shapes),
            "uptime_sec": round(uptime, 1),
            "weights_loaded": self.adapter.is_loaded(),
            "idle_unload_s": self.model_manager.idle_unload_s,
            "max_batch_tokens": self.batch_planner.max_batch_tokens,
            "max_bulk_microbatch_tokens": self.batch_planner.max_bulk_microbatch_tokens,
            "effective_bulk_microbatch_tokens": self.batch_planner.effective_bulk_microbatch_tokens,
        }

    def stop_admission(self):
        """Stops accepting new admission leases and wakes waiting threads."""
        with self._admission_lock:
            self._shutting_down = True
            self._admission_lock.notify_all()

    def _unload_owned_embedding(self):
        """Unload only this runtime's registration, on owner or after owner exit."""
        manager = getattr(self, "model_manager", None)
        if (
            getattr(self, "_owns_adapter_registration", True)
            and manager is not None
            and manager.get_adapter("embed") is self.adapter
        ):
            manager.unload("embed")
            if manager.get_state("embed") != ModelState.UNLOADED:
                raise ModelUnavailableError(
                    manager.get_error("embed") or "Embedding adapter did not unload successfully"
                )

    def _finish_shutdown(self, unload_future: Future, completion: Future):
        """Done callback: stop an owned executor without allocating a waiter."""
        error: Optional[BaseException] = None
        try:
            unload_future.result()
        except BaseException as exc:
            error = exc
        finally:
            if self._owns_executor:
                try:
                    self.executor.shutdown(timeout=0)
                except BaseException as exc:
                    if error is None:
                        error = exc
            if error is None:
                completion.set_result(None)
            else:
                completion.set_exception(error)

    def shutdown(self, timeout: float = 5.0):
        """Fence new work, unload on its owner, then stop only an owned executor.

        The timeout bounds the caller's wait, not a running Metal kernel. Queued
        teardown remains live after timeout; a subsequent call waits on the same
        operation. No helper waits on a stuck owner. Calls on the owner initiate
        teardown without waiting on self. If owner exit already won admission,
        a caller joins within its budget, then disposes only after confirmed exit.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("shutdown timeout must be a finite non-negative number")
        deadline = time.monotonic() + timeout
        self.stop_admission()

        no_cleanup = False
        with self._admission_lock:
            previous = self._shutdown_future
            if previous is not None and previous.done() and not self.executor.is_accepting():
                # A rejected/cancelled old teardown is not a permanent terminal
                # result. Retry safely once the externally stopped owner exits.
                if previous.cancelled() or isinstance(
                    previous.exception(), (WorkerUnavailableError, RequestCancelledError),
                ):
                    self._shutdown_future = None
            if self._shutdown_future is None:
                completion: Future = Future()
                completion.set_running_or_notify_cancel()
                self._shutdown_future = completion
                cleanup = self._shutdown_cleanup
                if cleanup is None or cleanup.done():
                    cleanup = Future()
                    self._shutdown_cleanup = cleanup
                    self._shutdown_needs_disposal = self._owns_adapter_registration
                    no_cleanup = not self._shutdown_needs_disposal
                else:
                    try:
                        self.executor.submit_cleanup(cleanup)
                    except WorkerUnavailableError:
                        self._shutdown_needs_disposal = True
                cleanup.add_done_callback(lambda done: self._finish_shutdown(done, completion))
            future = self._shutdown_future
            cleanup = self._shutdown_cleanup
            needs_disposal = self._shutdown_needs_disposal

        assert future is not None and cleanup is not None
        if no_cleanup:
            cleanup.set_result(None)
        if self.executor.is_owner_thread():
            return
        if needs_disposal:
            self.executor.join_worker(timeout=max(0.0, deadline - time.monotonic()))
            if not self.executor.is_worker_alive():
                with self._admission_lock:
                    claim_disposal = self._shutdown_needs_disposal
                    self._shutdown_needs_disposal = False
                if claim_disposal:
                    # Only this caller disposes; other callers wait on completion.
                    try:
                        self._unload_owned_embedding()
                    except BaseException as exc:
                        cleanup.set_exception(exc)
                    else:
                        cleanup.set_result(None)
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeoutError:
            if future.done():
                # Do not relabel a TimeoutError raised by the unload itself.
                raise
            raise DeadlineExceededError(
                "Runtime shutdown timed out; retry after active owner work exits"
            ) from None
        finally:
            if self._owns_executor and future.done():
                # Join on both success and unload failure; the done callback can
                # stop the worker, but cannot join itself on the owner thread.
                self.executor.join_worker(timeout=max(0.0, deadline - time.monotonic()))
        if self._owns_executor and self.executor.is_worker_alive():
            raise DeadlineExceededError("Runtime shutdown timed out waiting for owner exit")
