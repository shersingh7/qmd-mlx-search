"""
batching.py — Token-once batching, length bucketing, memory budgeting, and order restoration
"""

from __future__ import annotations

import os
import sys
import threading
import time
from numbers import Integral
from typing import Any, Callable, List, Optional
import numpy as np

from .adapters.tokenization import TokenizedBatch, TokenizerHelper
from .protocol import (
    DeadlineExceededError,
    OutOfMemoryError,
    RequestCancelledError,
    WorkClass,
)


def get_system_ram_gb() -> float:
    """Detects total physical system RAM in gigabytes."""
    try:
        if sys.platform == "darwin":
            import subprocess
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                bytes_mem = int(res.stdout.strip())
                return bytes_mem / (1024 ** 3)
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024 ** 3)
    except Exception:
        return 16.0  # Safe default


def calculate_default_max_batch_tokens(
    custom_budget: int = 0,
    model_params_b: float = 0.6,
    system_ram_gb_fn: Optional[Callable[[], float]] = None,
) -> int:
    """
    Calculates safe GPU token budget based on physical system RAM and model size.
    """
    if custom_budget > 0:
        return max(512, min(custom_budget, 65536))

    ram_fn = system_ram_gb_fn or get_system_ram_gb
    ram_gb = ram_fn()
    if ram_gb <= 8.5:
        base = 4096
    elif ram_gb <= 16.5:
        base = 8192
    elif ram_gb <= 24.5:
        base = 12288
    elif ram_gb <= 36.5:
        base = 16384
    else:
        base = 32768

    try:
        scale = 0.6 / float(model_params_b or 0.6)
    except (TypeError, ValueError):
        scale = 1.0
    scale = min(1.0, max(0.1, scale))
    return max(1024, int(base * scale))


class BatchPlanner:
    """
    Plans execution batches with single-pass tokenization, length-based sorting
    to minimize padding waste, and exact order restoration.
    """

    def __init__(
        self,
        max_batch_tokens: Optional[int] = None,
        max_bulk_microbatch_tokens: int = 512,
        model_params_b: float = 0.6,
        system_ram_gb_fn: Optional[Callable[[], float]] = None,
    ):
        self._explicit_budget = self._validate_token_budget(
            "max_batch_tokens", 0 if max_batch_tokens is None else max_batch_tokens,
            allow_auto=True,
        )
        self._ram_fn = system_ram_gb_fn
        self._oom_halvings: int = 0
        self._tuned_model_params_b: Optional[float] = None
        self.max_bulk_microbatch_tokens = max_bulk_microbatch_tokens
        self.max_batch_tokens = self._explicit_budget or calculate_default_max_batch_tokens(
            model_params_b=model_params_b,
            system_ram_gb_fn=self._ram_fn,
        )
        if not self._explicit_budget:
            self._tuned_model_params_b = float(model_params_b or 0.6)

    @staticmethod
    def _validate_token_budget(name: str, value: int, *, allow_auto: bool = False) -> int:
        minimum = 0 if allow_auto else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        # Preserve small explicit budgets used for tight scheduling; cap excessive
        # operator values at the same ceiling as the automatic budget helper.
        return min(int(value), 65536)

    @property
    def max_batch_tokens(self) -> int:
        """Validated model/hardware or explicit operator budget, adapted after OOM."""
        return self._max_batch_tokens

    @max_batch_tokens.setter
    def max_batch_tokens(self, value: int):
        self._max_batch_tokens = self._validate_token_budget("max_batch_tokens", value)

    @property
    def max_bulk_microbatch_tokens(self) -> int:
        """Requested bulk fairness target, independently of the memory budget."""
        return self._max_bulk_microbatch_tokens

    @max_bulk_microbatch_tokens.setter
    def max_bulk_microbatch_tokens(self, value: int):
        self._max_bulk_microbatch_tokens = self._validate_token_budget(
            "max_bulk_microbatch_tokens", value,
        )

    @property
    def effective_bulk_microbatch_tokens(self) -> int:
        """Bulk scheduling target clamped to the current validated batch budget."""
        return min(self.max_batch_tokens, self.max_bulk_microbatch_tokens)

    def tune_for_model(self, model_params_b: float) -> int:
        """
        Re-scale the token budget once the loaded model's size is known.
        Preserves any adaptive OOM reduction factor previously applied.
        Does NOT overwrite on repeated calls with unchanged model params.
        """
        if self._explicit_budget > 0:
            return self.max_batch_tokens

        pb = float(model_params_b or 0.6)
        if self._tuned_model_params_b is not None and abs(self._tuned_model_params_b - pb) < 1e-4:
            # Already tuned for this model, preserve adaptive OOM reduction
            return self.max_batch_tokens

        self._tuned_model_params_b = pb
        base_tokens = calculate_default_max_batch_tokens(
            model_params_b=pb,
            system_ram_gb_fn=self._ram_fn,
        )
        # Apply existing OOM halvings if any
        for _ in range(self._oom_halvings):
            base_tokens = max(512, base_tokens // 2)

        self.max_batch_tokens = base_tokens
        return self.max_batch_tokens

    def plan_micro_batches(
        self,
        tokenized_batch: TokenizedBatch,
        is_query: bool = False,
        work_class: Optional[WorkClass] = None,
        max_tokens_override: Optional[int] = None,
    ) -> tuple[list[TokenizedBatch], list[list[int]]]:
        """
        Groups sequences by length under the effective padded-token budget.
        The conservative default bulk target keeps nonpreemptible GPU intervals
        short. Higher targets are an explicit latency/memory tradeoff, not a
        guarantee of better throughput.
        A single sequence may exceed this scheduling target and is never truncated
        here. Tokenization must still enforce the adapter's supported context.
        Returns (list of TokenizedBatch sub-batches, list of original index positions).
        """
        if max_tokens_override is not None:
            max_tokens_override = self._validate_token_budget(
                "max_tokens_override", max_tokens_override, allow_auto=True,
            )
        total_items = len(tokenized_batch)
        if total_items == 0:
            return [], []

        if total_items == 1:
            return [tokenized_batch], [[0]]

        if max_tokens_override:
            effective_budget = min(self.max_batch_tokens, max_tokens_override)
        elif work_class == WorkClass.BULK or (work_class is None and not is_query):
            effective_budget = self.effective_bulk_microbatch_tokens
        else:
            effective_budget = self.max_batch_tokens

        is_bulk = (work_class == WorkClass.BULK) or (work_class is None and not is_query)
        max_items_per_batch = 16 if is_bulk else 64

        lengths = tokenized_batch.lengths
        sorted_pos = sorted(range(total_items), key=lambda i: lengths[i])
        sorted_lengths = [lengths[i] for i in sorted_pos]

        micro_batches_indices: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0

        for pos, token_len in zip(sorted_pos, sorted_lengths):
            cand_max_len = max(current_max_len, token_len)
            cand_tokens = (len(current_batch) + 1) * cand_max_len

            if current_batch and (cand_tokens > effective_budget or len(current_batch) >= max_items_per_batch):
                micro_batches_indices.append(current_batch)
                current_batch = [pos]
                current_max_len = token_len
            else:
                current_batch.append(pos)
                current_max_len = cand_max_len

        if current_batch:
            micro_batches_indices.append(current_batch)

        sub_batches = [tokenized_batch.slice(indices) for indices in micro_batches_indices]
        return sub_batches, micro_batches_indices

    @staticmethod
    def restore_ordering(
        results: list[np.ndarray],
        micro_batches_indices: list[list[int]],
        total_items: int,
    ) -> np.ndarray:
        """Validate a full permutation, then scatter once into the final output.

        No concatenated intermediate is needed. A correctly ordered single
        NumPy batch already is the requested output and can be returned as-is.
        """
        if isinstance(total_items, bool) or not isinstance(total_items, Integral) or total_items < 0:
            raise ValueError("total_items must be a non-negative integer")
        if len(results) != len(micro_batches_indices):
            raise ValueError("Result batch count does not match index batch count")
        if not results:
            if total_items:
                raise ValueError("Missing embedding results")
            return np.empty((0, 0), dtype=np.float32)

        dims: Optional[int] = None
        seen: set[int] = set()
        positions: list[list[int]] = []
        for batch_result, batch_indices in zip(results, micro_batches_indices):
            if not isinstance(batch_result, np.ndarray) or batch_result.ndim != 2:
                raise ValueError("Each embedding result must be a 2D NumPy array")
            if batch_result.shape[0] != len(batch_indices):
                raise ValueError("Embedding result row count does not match its indices")
            if dims is None:
                dims = batch_result.shape[1]
            elif batch_result.shape[1] != dims:
                raise ValueError("Embedding result dimensions differ between batches")

            batch_positions: list[int] = []
            for position in batch_indices:
                if isinstance(position, bool) or not isinstance(position, Integral):
                    raise ValueError("Embedding result indices must be integers")
                position = int(position)
                if not 0 <= position < total_items or position in seen:
                    raise ValueError("Embedding result indices must be a permutation of the input")
                seen.add(position)
                batch_positions.append(position)
            positions.append(batch_positions)

        if len(seen) != total_items:
            raise ValueError("Embedding results do not cover all input items")
        if len(results) == 1 and positions[0] == list(range(total_items)):
            return results[0]

        assert dims is not None  # Every nonempty results list was validated above.
        dtype = np.result_type(*(result.dtype for result in results))
        reordered = np.empty((int(total_items), dims), dtype=dtype)
        for batch_result, batch_positions in zip(results, positions):
            reordered[batch_positions] = batch_result
        return reordered

    def plan_and_execute_tokenized(
        self,
        tokenized_batch: TokenizedBatch,
        embed_fn: Callable[[TokenizedBatch], np.ndarray],
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
        is_query: bool = False,
        work_class: Optional[WorkClass] = None,
    ) -> np.ndarray:
        """
        Executes embedding directly from a pre-tokenized TokenizedBatch.
        Avoids all redundant text tokenization and checks deadline between micro-batches.
        """
        total_items = len(tokenized_batch)
        if total_items == 0:
            return np.empty((0, 0), dtype=np.float32)

        sub_batches, micro_batches_indices = self.plan_micro_batches(
            tokenized_batch,
            is_query=is_query,
            work_class=work_class,
        )

        results: List[np.ndarray] = []
        for batch_idx, sub_batch in enumerate(sub_batches):
            if cancel_event and cancel_event.is_set():
                raise RequestCancelledError("Request cancelled by client")
            if deadline and time.monotonic() > deadline:
                raise DeadlineExceededError(
                    f"Deadline exceeded after {batch_idx}/{len(sub_batches)} micro-batches"
                )

            batch_result = self.execute_sub_batch_with_retry(sub_batch, embed_fn)
            results.append(batch_result)

        return self.restore_ordering(results, micro_batches_indices, total_items)

    def plan_and_execute(
        self,
        texts: list[str],
        tokenizer: Any,
        embed_fn: Callable[[list[str]], np.ndarray],
        max_length: int = 2048,
    ) -> np.ndarray:
        """
        Backwards-compatible wrapper that tokenizes texts once into TokenizedBatch.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        raw_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        tokenized_batch = TokenizerHelper.tokenize_once(texts, raw_tokenizer, max_length=max_length)

        def _adapted_embed(sub_batch: TokenizedBatch) -> np.ndarray:
            sub_texts = [texts[i] for i in sub_batch.original_indices]
            return embed_fn(sub_texts)

        return self.plan_and_execute_tokenized(tokenized_batch, _adapted_embed)

    def execute_sub_batch_with_retry(
        self,
        batch: TokenizedBatch,
        embed_fn: Callable[[TokenizedBatch], np.ndarray],
        depth: int = 0,
    ) -> np.ndarray:
        """Bisect genuine allocation failures, after releasing failed forward frames.

        Retry depth is capped at three. Keeping only the error text across the
        handler boundary prevents the failed forward's activation arrays from
        staying resident through its traceback while the smaller forwards run.
        """
        try:
            return embed_fn(batch)
        except Exception as exc:
            if not self._is_allocation_failure(exc):
                raise
            error_text = str(exc)
            exc.__traceback__ = None

        # Do not move retries into the except block: sys.exception() itself
        # retains the handled exception (and thus its failed forward frames).
        self._oom_halvings += 1
        self.max_batch_tokens = min(self.max_batch_tokens, max(512, self.max_batch_tokens // 2))
        if len(batch) <= 1 or depth >= 3:
            raise OutOfMemoryError(
                f"Metal allocation failed during embedding forward pass: {error_text}"
            ) from None

        mid = len(batch) // 2
        first = batch.slice(list(range(mid)))
        second = batch.slice(list(range(mid, len(batch))))
        first_result = self.execute_sub_batch_with_retry(first, embed_fn, depth=depth + 1)
        second_result = self.execute_sub_batch_with_retry(second, embed_fn, depth=depth + 1)
        return np.concatenate([first_result, second_result], axis=0)

    @staticmethod
    def _is_allocation_failure(exc: Exception) -> bool:
        if isinstance(exc, (OutOfMemoryError, MemoryError)):
            return True
        # Native MLX allocation failures are RuntimeErrors. Validation, protocol,
        # cancellation and deadline errors must not become retries based on text.
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).lower()
        native_backend = "metal" in message or "mlx" in message
        return native_backend and (
            "out of memory" in message
            or "buffer allocation failed" in message
            or "failed to allocate" in message
            or "unable to allocate" in message
        )

    # Backwards compatible alias
    _execute_with_retry = execute_sub_batch_with_retry
