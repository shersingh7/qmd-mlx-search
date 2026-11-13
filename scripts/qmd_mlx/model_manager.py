"""
model_manager.py — Model Residency Manager for MLX Stages (Embed, Rerank, Generate)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .adapters.embedding import estimate_model_memory_mb, infer_model_params_b
from .executor import GPUExecutor
from .protocol import (
    DeadlineExceededError,
    MLXServerError,
    ModelUnavailableError,
    OutOfMemoryError,
    RequestCancelledError,
    UnsupportedModelError,
)

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class ModelState:
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    EVICTING = "evicting"


def calculate_residency_budget_mb(ram_gb: float) -> float:
    """
    Calculates conservative shared resident weight budget (in MB) based on physical RAM.
    Resident weight budgets constrain resident model parameters in unified memory,
    leaving headroom for activation buffers and OS allocations.
    """
    if ram_gb <= 8.5:
        return 2048.0   # 2 GB for 8GB Mac
    elif ram_gb <= 16.5:
        return 6144.0   # 6 GB for 16GB Mac
    elif ram_gb <= 24.5:
        return 10240.0  # 10 GB for 24GB Mac
    elif ram_gb <= 36.5:
        return 16384.0  # 16 GB for 32GB/36GB Mac
    elif ram_gb <= 64.5:
        return 36864.0  # 36 GB for 64GB Mac
    else:
        return max(36864.0, ram_gb * 0.6 * 1024.0)


def _detect_system_ram_gb() -> float:
    try:
        if sys.platform == "darwin":
            import subprocess
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return int(res.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024 ** 3)
    except Exception:
        return 16.0


# A generation has its own bounded queue/admission lifetime. Request deadlines
# only bound individual waits; they must not shorten another waiter's load.
_LOAD_OPERATION_TIMEOUT_S = 120.0


@dataclass
class _LoadOperation:
    """One generation's completion, independent of any caller's waiting budget."""

    event: threading.Event = field(default_factory=threading.Event)
    deadline: float = field(default_factory=lambda: time.monotonic() + _LOAD_OPERATION_TIMEOUT_S)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    waiters: int = 0
    future: Optional[Future] = None
    started: bool = False
    completed: bool = False
    error: Optional[BaseException] = None


class ModelResidencyManager:
    """
    Manages loading, evaluation, single-flight admission, and LRU/idle eviction
    for all MLX model adapters behind the single GPUExecutor execution owner.
    """

    def __init__(
        self,
        executor: GPUExecutor,
        idle_unload_s: Optional[float] = None,
        residency_budget_mb: Optional[float] = None,
        system_ram_gb_fn: Optional[Callable[[], float]] = None,
    ):
        self.executor = executor
        if idle_unload_s is not None:
            self.idle_unload_s = float(idle_unload_s)
        else:
            self.idle_unload_s = float(os.getenv("MLX_IDLE_UNLOAD_S", "300.0"))

        self._ram_fn = system_ram_gb_fn or _detect_system_ram_gb
        if residency_budget_mb is not None:
            self.residency_budget_mb = float(residency_budget_mb)
        elif "MLX_RESIDENCY_BUDGET_MB" in os.environ:
            self.residency_budget_mb = float(os.environ["MLX_RESIDENCY_BUDGET_MB"])
        else:
            self.residency_budget_mb = calculate_residency_budget_mb(self._ram_fn())

        # Registered adapters: "embed", "rerank", "generate"
        self._adapters: Dict[str, Any] = {}
        self._states: Dict[str, str] = {}
        self._last_active: Dict[str, float] = {}
        self._errors: Dict[str, Optional[str]] = {}

        # Single-flight synchronization lock
        self._lock = threading.Lock()
        self._loading_events: Dict[str, threading.Event] = {}
        self._load_operations: Dict[str, _LoadOperation] = {}
        self._evicting_events: Dict[str, threading.Event] = {}
        self._leases: Dict[str, int] = {}
        # Conservative weight reservations survive incomplete readiness and failed
        # unloads. Only a verified release removes a stage from this ledger.
        self._resident_mb: Dict[str, float] = {}

        self.peak_memory_mb: float = 0.0

        # Register idle check with executor
        self.executor.register_idle_callback(self._check_idle_unloads)

    def register_adapter(self, stage: str, adapter: Any):
        """Registers an adapter instance (embed, rerank, generate)."""
        resident_mb = self._estimate_adapter_mb(adapter) if self._adapter_has_weights(adapter) else 0.0
        with self._lock:
            self._adapters[stage] = adapter
            self._states[stage] = ModelState.UNLOADED
            self._last_active[stage] = time.monotonic()
            self._errors[stage] = None
            if resident_mb:
                self._resident_mb[stage] = resident_mb
            else:
                self._resident_mb.pop(stage, None)

    def get_adapter(self, stage: str) -> Optional[Any]:
        with self._lock:
            return self._adapters.get(stage)

    def get_state(self, stage: str) -> str:
        with self._lock:
            return self._states.get(stage, ModelState.UNLOADED)

    def get_error(self, stage: str) -> Optional[str]:
        with self._lock:
            return self._errors.get(stage)

    def acquire_lease(self, stage: str, timeout: Optional[float] = None) -> bool:
        """
        Pins a stage against idle, LRU, and explicit eviction while its owner lives.
        If the stage is currently in the process of eviction (EVICTING),
        waits for eviction to complete before acquiring lease.
        """
        dl = (time.monotonic() + timeout) if timeout is not None else None
        while True:
            with self._lock:
                state = self._states.get(stage, ModelState.UNLOADED)
                if state != ModelState.EVICTING:
                    self._leases[stage] = self._leases.get(stage, 0) + 1
                    self._last_active[stage] = time.monotonic()
                    return True
                ev_event = self._evicting_events.get(stage)

            if ev_event is not None:
                rem = (dl - time.monotonic()) if dl is not None else 30.0
                if rem <= 0 or not ev_event.wait(timeout=min(0.5, max(0.01, rem))):
                    if dl is not None and time.monotonic() >= dl:
                        raise DeadlineExceededError(f"Timed out acquiring lease on evicting stage '{stage}'")
            else:
                time.sleep(0.01)
                if dl is not None and time.monotonic() >= dl:
                    raise DeadlineExceededError(f"Timed out acquiring lease on stage '{stage}'")

    def release_lease(self, stage: str):
        """Releases an active lifecycle lease on a stage."""
        with self._lock:
            cur = self._leases.get(stage, 0)
            if cur <= 1:
                self._leases.pop(stage, None)
            else:
                self._leases[stage] = cur - 1
            self._last_active[stage] = time.monotonic()

    def touch(self, stage: str):
        with self._lock:
            self._last_active[stage] = time.monotonic()

    def _get_active_memory_mb(self) -> float:
        try:
            if _MLX_AVAILABLE:
                if hasattr(mx, "get_active_memory"):
                    return mx.get_active_memory() / (1024 * 1024)
                if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                    return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _get_peak_memory_mb(self) -> float:
        try:
            if _MLX_AVAILABLE:
                if hasattr(mx, "get_peak_memory"):
                    return mx.get_peak_memory() / (1024 * 1024)
                if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
                    return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            pass
        return self.peak_memory_mb

    def update_peak_memory(self):
        cur = self._get_peak_memory_mb()
        if cur > self.peak_memory_mb:
            self.peak_memory_mb = cur

    def _get_resident_model_mb(self, exclude_stage: Optional[str] = None) -> float:
        """Returns reserved/resident weights, including failed partial loads.

        Callers hold _lock: readiness is not proof that weights were released.
        """
        return sum(mb for stage, mb in self._resident_mb.items() if stage != exclude_stage)

    @staticmethod
    def _adapter_has_weights(adapter: Any) -> bool:
        return (
            getattr(adapter, "model", None) is not None
            or bool(getattr(adapter, "is_loaded", lambda: False)())
        )

    @staticmethod
    def _adapter_is_ready(adapter: Any) -> bool:
        if not getattr(adapter, "is_loaded", lambda: False)():
            return False
        # Some adapter predicates only check the model and the first resolved
        # token ID. Never publish their half-initialized companions as READY.
        for name in ("model", "tokenizer", "raw_hf_tokenizer", "yes_token_id", "no_token_id"):
            if hasattr(adapter, name) and getattr(adapter, name) is None:
                return False
        return True

    def _estimate_adapter_mb(self, adapter: Any) -> float:
        if hasattr(adapter, "model_memory_mb") and adapter.model_memory_mb > 0:
            return float(adapter.model_memory_mb)
        if hasattr(adapter, "estimated_memory_mb") and adapter.estimated_memory_mb > 0:
            return float(adapter.estimated_memory_mb)
        name = getattr(adapter, "model_name", "")
        quant = getattr(adapter, "quantization", None)
        params_b = getattr(adapter, "model_params_b", None)
        return estimate_model_memory_mb(name, quant, params_b)

    def _is_owner_thread(self) -> bool:
        return hasattr(self.executor, "is_owner_thread") and self.executor.is_owner_thread()

    def _release_adapter_under_owner(self, stage: str, adapter: Any):
        """Attempt release even when partial initialization makes is_loaded false."""
        with self._lock:
            reserved_mb = self._resident_mb.get(stage, 0.0)
        # Remember measurements before unload can reset them and then fail.
        # Broken metadata must not prevent the unconditional rollback attempt;
        # the earlier reservation remains the conservative fallback in that case.
        try:
            if reserved_mb or self._adapter_has_weights(adapter):
                measured_mb = self._estimate_adapter_mb(adapter)
                with self._lock:
                    self._resident_mb[stage] = max(reserved_mb, measured_mb)
        except Exception:
            pass
        adapter.unload()
        if self._adapter_has_weights(adapter):
            raise ModelUnavailableError(f"Unload of '{stage}' returned without releasing model weights")
        with self._lock:
            self._resident_mb.pop(stage, None)
        if _MLX_AVAILABLE and hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def _finish_eviction_under_owner(self, stage: str, adapter: Any, event: threading.Event) -> bool:
        error = None
        try:
            self._release_adapter_under_owner(stage, adapter)
        except Exception as exc:
            error = exc
            print(f"[mlx-manager] Failed to unload '{stage}': {exc}", file=sys.stderr)
        finally:
            with self._lock:
                if self._evicting_events.get(stage) is event:
                    self._states[stage] = ModelState.FAILED if error else ModelState.UNLOADED
                    self._errors[stage] = str(error) if error else None
                    self._evicting_events.pop(stage, None)
                event.set()
        return error is None

    def _evict_for_budget_under_owner(self, stage: str, needed_mb: float):
        """Evict unleased LRU weights, crediting only verified releases.

        The target's reservation is excluded because needed_mb replaces it during
        actual-size reconciliation. Failed weights remain eviction candidates and
        remain in the budget even if their adapter no longer reports readiness.
        """
        if needed_mb > self.residency_budget_mb:
            raise OutOfMemoryError(
                f"Model '{stage}' memory requirement ({needed_mb:.1f}MB) exceeds total residency budget ({self.residency_budget_mb:.1f}MB)"
            )

        attempted = set()
        while True:
            with self._lock:
                current = self._get_resident_model_mb(exclude_stage=stage)
                if current + needed_mb <= self.residency_budget_mb:
                    return
                candidates = [
                    (self._last_active.get(st, 0.0), st, ad)
                    for st, ad in self._adapters.items()
                    if st != stage
                    and st not in attempted
                    and self._resident_mb.get(st, 0.0) > 0
                    and self._states.get(st) not in (ModelState.LOADING, ModelState.EVICTING)
                    and self._leases.get(st, 0) == 0
                ]
                if not candidates:
                    break
                _, cand_stage, cand_adapter = min(candidates, key=lambda candidate: candidate[0])
                attempted.add(cand_stage)
                self._states[cand_stage] = ModelState.EVICTING
                event = threading.Event()
                self._evicting_events[cand_stage] = event
            # Never hold the manager lock over adapter / executor callbacks.
            if self._finish_eviction_under_owner(cand_stage, cand_adapter, event):
                print(f"[mlx-manager] Evicted LRU stage '{cand_stage}' to free memory for '{stage}'")

        raise OutOfMemoryError(
            f"Insufficient residency headroom to load '{stage}' (needed: {needed_mb:.1f}MB, resident: {current:.1f}MB, budget: {self.residency_budget_mb:.1f}MB)"
        )

    def _finish_load(
        self,
        stage: str,
        operation: _LoadOperation,
        error: Optional[BaseException],
        only_if_not_started: bool = False,
    ):
        """Settle exactly this generation, never a caller's newer retry."""
        with self._lock:
            self._finish_load_locked(stage, operation, error, only_if_not_started)

    def _finish_load_locked(
        self,
        stage: str,
        operation: _LoadOperation,
        error: Optional[BaseException],
        only_if_not_started: bool = False,
    ):
        if operation.completed or (only_if_not_started and operation.started):
            return
        operation.completed = True
        operation.error = error
        if self._load_operations.get(stage) is operation:
            self._states[stage] = ModelState.FAILED if error else ModelState.READY
            self._errors[stage] = str(error) if error else None
            if error is None:
                self._last_active[stage] = time.monotonic()
            self._load_operations.pop(stage, None)
            if self._loading_events.get(stage) is operation.event:
                self._loading_events.pop(stage, None)
        operation.event.set()

    def _release_load_waiter(self, stage: str, operation: _LoadOperation):
        future = None
        with self._lock:
            operation.waiters -= 1
            if operation.waiters == 0 and not operation.started and not operation.completed:
                # Remove the generation atomically with its last waiter. A new
                # request gets a fresh operation, never this cancelled queue job.
                operation.cancel_event.set()
                self._finish_load_locked(
                    stage, operation,
                    RequestCancelledError(f"Load of '{stage}' abandoned with no remaining waiters"),
                )
                future = operation.future
        # Future callbacks may reenter this manager; never cancel under _lock.
        if future is not None:
            future.cancel()

    def _do_load_internal(
        self,
        stage: str,
        adapter: Any,
        operation: _LoadOperation,
    ):
        """Run one load transaction on the executor owner, including rollback."""
        with self._lock:
            if self._load_operations.get(stage) is not operation or operation.completed:
                # An inline owner takeover already completed this queued job, or
                # a newer load generation replaced it. This job is now a no-op.
                return
            if operation.started:
                raise ModelUnavailableError(f"Recursive load of '{stage}' cannot wait on its execution owner")
            operation.started = True

        load_attempted = False
        error = None
        try:
            if operation.cancel_event.is_set():
                raise RequestCancelledError(f"Load of '{stage}' cancelled")
            if time.monotonic() >= operation.deadline:
                raise DeadlineExceededError(f"Load of '{stage}' exceeded deadline")

            with self._lock:
                has_reservation = self._resident_mb.get(stage, 0.0) > 0
            # A prior failed rollback must finish before retrying adapter.load(),
            # whose own model-present fast path may otherwise accept partial state.
            if has_reservation or self._adapter_has_weights(adapter):
                self._release_adapter_under_owner(stage, adapter)

            needed_mb = self._estimate_adapter_mb(adapter)
            self._evict_for_budget_under_owner(stage, needed_mb)
            with self._lock:
                self._resident_mb[stage] = needed_mb
            load_attempted = True
            adapter.load()
            if not self._adapter_is_ready(adapter):
                raise ModelUnavailableError(f"Loaded '{stage}' model is not ready (incomplete adapter initialization)")

            actual_mb = self._estimate_adapter_mb(adapter)
            with self._lock:
                self._resident_mb[stage] = actual_mb
            self._evict_for_budget_under_owner(stage, actual_mb)
            self.update_peak_memory()
        except Exception as exc:
            rollback_error = None
            if load_attempted:
                try:
                    # Never gate rollback on is_loaded(): partial model/tokenizer
                    # initialization can own weights without being inference-ready.
                    self._release_adapter_under_owner(stage, adapter)
                except Exception as cleanup_exc:
                    rollback_error = cleanup_exc
            if rollback_error is not None:
                message = f"{exc}; rollback failed: {rollback_error}"
                error = type(exc)(message) if isinstance(exc, MLXServerError) else ModelUnavailableError(
                    f"Failed to load '{stage}' model: {message}"
                )
            else:
                error = exc if isinstance(exc, MLXServerError) else ModelUnavailableError(
                    f"Failed to load '{stage}' model: {exc}"
                )
        finally:
            # The actual operation owns completion even if all callers stopped
            # waiting while a native load was still running.
            self._finish_load(stage, operation, error)
        if error is not None:
            raise error

    @staticmethod
    def _check_load_wait_budget(stage: str, dl: float, cancel_event: Optional[threading.Event]):
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError(f"Load of '{stage}' cancelled")
        if time.monotonic() >= dl:
            raise DeadlineExceededError(f"Timed out waiting for load of '{stage}'")

    def ensure_loaded(
        self,
        stage: str,
        timeout_s: float = 120.0,
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Ensure residency with per-caller waits and operation-owned completion.

        Owner callers execute inline (including takeover of queued external work).
        External callers share one async job with its own deadline/cancellation.
        Only the last departing waiter cancels a pending generation. A running
        native load/rollback always retains its real completion ownership.
        """
        dl = deadline if deadline is not None else (time.monotonic() + timeout_s)
        is_owner = self._is_owner_thread()
        while True:
            self._check_load_wait_budget(stage, dl, cancel_event)
            with self._lock:
                adapter = self._adapters.get(stage)
                state = self._states.get(stage)
            if adapter is None:
                raise ModelUnavailableError(f"No adapter registered for stage '{stage}'")
            # Adapter callbacks never run while the manager lock is held.
            try:
                ready = state == ModelState.READY and self._adapter_is_ready(adapter)
            except Exception:
                ready = False
            with self._lock:
                if self._adapters.get(stage) is not adapter or self._states.get(stage) != state:
                    continue
                if ready:
                    self._last_active[stage] = time.monotonic()
                    return
                if state == ModelState.EVICTING:
                    ev_event = self._evicting_events.get(stage)
                else:
                    operation = self._load_operations.get(stage)
                    designated_loader = operation is None
                    if operation is None:
                        # Preserve a pending event when the execution owner
                        # takes over an externally announced load inline.
                        operation = _LoadOperation(self._loading_events.get(stage) or threading.Event())
                        self._load_operations[stage] = operation
                        self._loading_events[stage] = operation.event
                        self._states[stage] = ModelState.LOADING
                        self._errors[stage] = None
                    operation.waiters += 1
                    break
            if is_owner:
                raise ModelUnavailableError(f"Cannot wait for eviction of '{stage}' on its execution owner")
            rem = dl - time.monotonic()
            if ev_event is not None:
                ev_event.wait(timeout=max(0.0, min(0.05, rem)))
            else:
                time.sleep(max(0.0, min(0.01, rem)))

        try:
            if is_owner:
                self._do_load_internal(stage, adapter, operation)
            elif designated_loader:
                def _job_fn():
                    self._do_load_internal(stage, adapter, operation)

                def _job_done(future: Future):
                    # The Future settles pre-start failures only. Once started,
                    # the native transaction owns completion and rollback.
                    exc = RequestCancelledError(f"Load of '{stage}' cancelled") if future.cancelled() else future.exception()
                    if exc is None:
                        exc = ModelUnavailableError(f"Load of '{stage}' did not execute")
                    self._finish_load(stage, operation, exc, only_if_not_started=True)

                try:
                    future, _, _ = self.executor.submit_async(
                        _job_fn,
                        priority=0,
                        timeout_s=max(0.0, operation.deadline - time.monotonic()),
                        cancel_event=operation.cancel_event,
                        description=f"Load {stage}",
                    )
                    future.add_done_callback(_job_done)
                    with self._lock:
                        operation.future = future
                        cancel_pending = operation.completed and not operation.started
                    # Explicit eviction may have settled this pending generation
                    # before its initiating thread finished submitting the job.
                    if cancel_pending:
                        future.cancel()
                except Exception as exc:
                    self._finish_load(stage, operation, exc, only_if_not_started=True)

            if not is_owner:
                while True:
                    self._check_load_wait_budget(stage, dl, cancel_event)
                    if operation.event.wait(timeout=min(0.05, max(0.0, dl - time.monotonic()))):
                        break
            self._check_load_wait_budget(stage, dl, cancel_event)
            if operation.error is not None:
                # Preserve this generation's typed failure even if another caller
                # already began a retry and cleared the stage's current error.
                raise operation.error
            ready = self._adapter_is_ready(adapter)
            with self._lock:
                if self._states.get(stage) == ModelState.READY and ready:
                    self._last_active[stage] = time.monotonic()
                    return
            raise ModelUnavailableError(f"Loaded '{stage}' model is no longer resident")
        finally:
            self._release_load_waiter(stage, operation)

    def unload(self, stage: str):
        """Explicitly evict on the owner, refusing any live lifecycle lease.

        Terminal disposal after the executor thread has exited may clear weights
        despite stale leases: no kernel can still use them or be newly admitted.
        A pending load is cancelled so its late queue entry cannot resurrect the
        explicitly evicted model. A running load cannot be evicted reentrantly.
        """
        # If currently evicting, wait for eviction
        with self._lock:
            adapter = self._adapters.get(stage)
            if adapter is None:
                return
            if self._leases.get(stage, 0) > 0 and self.executor.is_worker_alive():
                raise ModelUnavailableError(f"Cannot unload '{stage}' while it has active lifecycle leases")
            ev_event = self._evicting_events.get(stage)

        if ev_event is not None:
            if self._is_owner_thread():
                return
            ev_event.wait(timeout=5.0)

        def _do_unload():
            pending_future = None
            with self._lock:
                if self._adapters.get(stage) is not adapter:
                    raise ModelUnavailableError(f"Adapter registration for '{stage}' changed before unload")
                # Check at execution, not just submission: a request may have
                # pinned the model while this unload waited behind owner work.
                if self._leases.get(stage, 0) > 0 and self.executor.is_worker_alive():
                    raise ModelUnavailableError(f"Cannot unload '{stage}' while it has active lifecycle leases")
                if self._states.get(stage) == ModelState.EVICTING:
                    return
                operation = self._load_operations.get(stage)
                if operation is not None:
                    if operation.started:
                        raise ModelUnavailableError(f"Cannot unload '{stage}' during its active load transaction")
                    operation.cancel_event.set()
                    self._finish_load_locked(
                        stage, operation, RequestCancelledError(f"Load of '{stage}' cancelled by explicit unload"),
                    )
                    pending_future = operation.future
                self._states[stage] = ModelState.EVICTING
                ev = threading.Event()
                self._evicting_events[stage] = ev
            if pending_future is not None:
                pending_future.cancel()
            if self._finish_eviction_under_owner(stage, adapter, ev):
                print(f"[mlx-manager] Unloaded stage '{stage}' (active: {self._get_active_memory_mb():.1f}MB)")

        if self._is_owner_thread():
            _do_unload()
        elif self.executor.is_worker_alive():
            # Submit to owner thread. If worker thread is alive, NEVER execute off-owner!
            try:
                self.executor.submit(_do_unload, priority=0, timeout_s=30.0, description=f"Unload {stage}")
            except ModelUnavailableError:
                # A lease / active-load refusal is not an executor rejection and
                # must reach the caller rather than pretending eviction worked.
                raise
            except Exception:
                # If submission was rejected (e.g. executor shutting down), wait for worker thread to stop/join
                if hasattr(self.executor, "join_worker"):
                    self.executor.join_worker(timeout=5.0)
                if not self.executor.is_worker_alive():
                    _do_unload()
                else:
                    raise
        else:
            # Executor worker thread is stopped and joined; safe to unload on caller thread
            _do_unload()

    def unload_all(self):
        """Unloads all registered stages safely under owner thread."""
        with self._lock:
            stages = list(self._adapters.keys())
        for stage in stages:
            self.unload(stage)

    def _check_idle_unloads(self):
        """Called on the executor thread when idle to evict expired models."""
        if self.idle_unload_s <= 0:
            return

        now = time.monotonic()
        stages_to_unload = []
        with self._lock:
            for stage, adapter in self._adapters.items():
                state = self._states.get(stage)
                last_act = self._last_active.get(stage, now)
                is_leased = self._leases.get(stage, 0) > 0

                if is_leased or state == ModelState.EVICTING:
                    continue

                if self._resident_mb.get(stage, 0.0) > 0 and state in (ModelState.READY, ModelState.FAILED):
                    idle_time = now - last_act
                    if idle_time >= self.idle_unload_s:
                        # Atomically commit EVICTING state under lock
                        self._states[stage] = ModelState.EVICTING
                        ev_event = threading.Event()
                        self._evicting_events[stage] = ev_event
                        stages_to_unload.append((stage, adapter, idle_time, ev_event))

        for stage, adapter, idle_time, ev_event in stages_to_unload:
            if self._finish_eviction_under_owner(stage, adapter, ev_event):
                print(
                    f"[mlx-manager] Stage '{stage}' idle for {idle_time:.0f}s — "
                    f"unloaded ({self._get_active_memory_mb():.0f}MB active Metal)"
                )

    def get_memory_info(self) -> dict[str, Any]:
        with self._lock:
            total_model_mb = self._get_resident_model_mb()
        return {
            "active_mb": round(self._get_active_memory_mb(), 1),
            "peak_mb": round(self._get_peak_memory_mb(), 1),
            "model_mb": round(total_model_mb, 1),
            "residency_budget_mb": round(self.residency_budget_mb, 1),
            "notes": "Resident weight budgets are not total activation memory guarantees.",
        }
