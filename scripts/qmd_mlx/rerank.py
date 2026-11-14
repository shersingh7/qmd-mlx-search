"""
rerank.py — MLX-native Reranker Adapter for Qwen3-Reranker Models
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from numbers import Integral, Real
from typing import Any, Optional
import numpy as np

from .protocol import (
    DeadlineExceededError,
    MLXServerError,
    ModelUnavailableError,
    RequestCancelledError,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def infer_quantization_and_dtype(
    model_name_or_path: str,
    default_quant: Optional[str] = None,
    default_dtype: Optional[str] = None,
) -> tuple[str, str]:
    """
    Infers exact quantization format and compute dtype from local config.json or model naming.
    Never silently assumes or misnames quantization.
    """
    expanded = os.path.expanduser(model_name_or_path)
    quant_str = default_quant
    dtype_str = default_dtype or "bfloat16"

    if os.path.isdir(expanded):
        cfg_file = os.path.join(expanded, "config.json")
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                quant = cfg.get("quantization") or cfg.get("quantization_config")
                if isinstance(quant, dict):
                    bits = quant.get("bits")
                    mode = quant.get("mode")
                    if bits == 4:
                        quant_str = f"4bit-{mode}" if mode else "4bit"
                    elif bits == 8:
                        quant_str = f"8bit-{mode}" if mode else "8bit"
                    elif quant.get("quant_type"):
                        quant_str = str(quant["quant_type"])
                    elif quant.get("bits"):
                        quant_str = f"{quant['bits']}bit"
                elif isinstance(quant, str):
                    quant_str = quant

                dtype_cfg = cfg.get("torch_dtype") or cfg.get("dtype")
                if dtype_cfg:
                    dtype_str = str(dtype_cfg)
            except Exception:
                pass

    if not quant_str:
        norm_name = os.path.basename(expanded).lower()
        if "4bit" in norm_name or "q4" in norm_name:
            quant_str = "4bit"
        elif "8bit" in norm_name or "q8" in norm_name:
            quant_str = "8bit"
        elif "mxfp8" in norm_name or "fp8" in norm_name:
            quant_str = "mxfp8"
        elif "fp16" in norm_name:
            quant_str = "fp16"
        elif "bf16" in norm_name:
            quant_str = "bf16"
        else:
            quant_str = "mxfp8"

    return quant_str, dtype_str


class RerankError(MLXServerError):
    """Raised when reranker input validation or inference fails."""
    status_code = 400
    error_type = "rerank_error"


class MLXRerankAdapter:
    """
    Reranker adapter for instruction-aware Qwen3 reranking models on Apple Silicon MLX.
    Evaluates P('yes') vs P('no') on the final generated token logits.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be 'yes' or 'no'."
    )
    DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

    # Official Qwen3-Reranker suffix
    THINK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n\n\n\n"

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-Reranker-4B-mxfp8",
        quantization: Optional[str] = None,
        dtype_str: Optional[str] = None,
        max_length: int = 2048,
        revision: Optional[str] = None,
        lazy_load: bool = False,
        executor: Optional[Any] = None,
        model_manager: Optional[Any] = None,
    ):
        if not _MLX_AVAILABLE:
            raise ModelUnavailableError("MLX or mlx-lm is not installed.")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
            raise RerankError("max_length must be a positive integer")

        self.model_name = model_name
        inferred_quant, inferred_dtype = infer_quantization_and_dtype(model_name, quantization, dtype_str)
        self.quantization = inferred_quant
        self.dtype_str = inferred_dtype
        self.max_length = max_length
        self.revision = revision
        self.executor = executor
        self.model_manager = model_manager
        self._init_lock = threading.RLock()

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.yes_token_id: Optional[int] = None
        self.no_token_id: Optional[int] = None

        self.model_memory_mb: float = 0.0
        self.total_requests: int = 0
        self.total_pairs_scored: int = 0
        self.total_latency_ms: float = 0.0

        if not lazy_load and self.model_manager is None:
            self.load()

    def load_via_manager(self, timeout_s: float = 120.0):
        if self.model_manager is not None:
            self.model_manager.ensure_loaded("rerank", timeout_s=timeout_s)
        else:
            self.load()

    def is_loaded(self) -> bool:
        return (
            self.model is not None and self.raw_hf_tokenizer is not None
            and self.yes_token_id is not None and self.no_token_id is not None
        )

    def _get_active_memory_mb(self) -> float:
        try:
            if hasattr(mx, "get_active_memory"):
                return mx.get_active_memory() / (1024 * 1024)
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def load(self):
        """Publish a complete model/tokenizer/token-ID bundle, or no state at all."""
        with self._init_lock:
            if self.is_loaded():
                return
            self.unload()  # Discard any incomplete state from an earlier initialization.
            t0 = time.time()
            mem_before = self._get_active_memory_mb()
            try:
                model, tokenizer_wrap = mlx_lm.load(self.model_name, revision=self.revision)
                tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)
                if model is None or not callable(getattr(tokenizer, "encode", None)):
                    raise RerankError("Loaded reranker has no model or usable tokenizer")
                yes_id, no_id = self._resolve_token_ids(tokenizer)
                self._validate_vocab([yes_id, no_id], model)
                memory_mb = max(0.0, self._get_active_memory_mb() - mem_before)
                self.tokenizer = tokenizer_wrap
                self.raw_hf_tokenizer = tokenizer
                self.yes_token_id, self.no_token_id = yes_id, no_id
                self.model_memory_mb = memory_mb
                self.model = model  # Readiness is published last.
                print(
                    f"[mlx-rerank] Loaded '{self.model_name}' ✓ (yes={yes_id}, no={no_id}, "
                    f"{memory_mb:.1f}MB Metal) in {time.time() - t0:.2f}s"
                )
                return
            except MLXServerError as exc:
                error_type, message = type(exc), str(exc)
            except Exception as exc:
                error_type = ModelUnavailableError
                message = f"Failed to load rerank model '{self.model_name}': {exc}"
            self.unload()
            model = tokenizer_wrap = tokenizer = None
            # Raise outside except: cached errors must not own failed model
            # locals or the loader/token-resolution traceback through context.
            raise error_type(message) from None

    @staticmethod
    def _validate_tokens(tokens: Any) -> list[int]:
        if isinstance(tokens, np.ndarray) and tokens.ndim != 1:
            raise RerankError("Tokenizer must return a one-dimensional token sequence")
        if not isinstance(tokens, (list, tuple, np.ndarray)) or len(tokens) == 0:
            raise RerankError("Empty or invalid token sequence after formatting")
        if any(not isinstance(t, Integral) or isinstance(t, (bool, np.bool_)) or not 0 <= int(t) < 2**31 for t in tokens):
            raise RerankError("Tokenizer must return non-negative int32 token IDs")
        return [int(t) for t in tokens]

    @staticmethod
    def _validate_vocab(tokens: list[int], model: Any):
        vocab_size = getattr(getattr(model, "args", None), "vocab_size", None)
        if isinstance(vocab_size, Integral) and not isinstance(vocab_size, bool) and int(vocab_size) > 0:
            if max(tokens) >= int(vocab_size):
                raise RerankError("Token IDs exceed the model vocabulary")

    def _resolve_token_ids(self, tokenizer: Optional[Any] = None) -> tuple[int, int]:
        """Resolve IDs locally during load; preserve the standalone helper API."""
        publish = tokenizer is None
        tokenizer = self.raw_hf_tokenizer if publish else tokenizer
        if tokenizer is None:
            raise RerankError("Tokenizer not loaded")
        suffix_tokens = self._validate_tokens(tokenizer.encode(self.THINK_SUFFIX, add_special_tokens=False))
        yes_context = self._validate_tokens(tokenizer.encode(self.THINK_SUFFIX + "yes", add_special_tokens=False))
        no_context = self._validate_tokens(tokenizer.encode(self.THINK_SUFFIX + "no", add_special_tokens=False))

        if (
            len(yes_context) == len(suffix_tokens) + 1
            and len(no_context) == len(suffix_tokens) + 1
            and yes_context[:-1] == suffix_tokens
            and no_context[:-1] == suffix_tokens
        ):
            yes_id = yes_context[-1]
            no_id = no_context[-1]
        else:
            yes_tokens = self._validate_tokens(tokenizer.encode("yes", add_special_tokens=False))
            no_tokens = self._validate_tokens(tokenizer.encode("no", add_special_tokens=False))
            if len(yes_tokens) != 1 or len(no_tokens) != 1:
                raise RerankError(f"Could not resolve single yes/no token IDs for model '{self.model_name}'")
            yes_id = yes_tokens[0]
            no_id = no_tokens[0]

        if yes_id == no_id:
            raise RerankError(f"Ambiguous or identical yes/no token IDs ({yes_id}) for model '{self.model_name}'")

        if publish:
            self.yes_token_id, self.no_token_id = yes_id, no_id
        return yes_id, no_id

    def unload(self):
        """Clear the entire initialization bundle, including failed partial loads."""
        with self._init_lock:
            self.model = None
            self.tokenizer = None
            self.raw_hf_tokenizer = None
            self.yes_token_id = None
            self.no_token_id = None
            self.model_memory_mb = 0.0

    def _format_pair(self, query: str, document: str) -> str:
        """Formats query-document pair with the official Qwen3-Reranker prompt."""
        if not isinstance(document, str) or not document.strip():
            raise RerankError("Document text is empty or whitespace.")
        return (
            f"<|im_start|>system\n{self.DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {self.DEFAULT_INSTRUCT}\n\n<Query>: {query}\n\n<Document>: {document}"
            f"{self.THINK_SUFFIX}"
        )

    @staticmethod
    def _check_request(deadline, cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Rerank request cancelled by client")
        if deadline is not None:
            if not isinstance(deadline, Real) or isinstance(deadline, bool) or not math.isfinite(deadline):
                raise RerankError("deadline must be finite")
            if time.monotonic() >= deadline:
                raise DeadlineExceededError("Rerank request deadline exceeded")

    def _ensure_loaded(self, deadline, cancel_event):
        self._check_request(deadline, cancel_event)
        if self.model_manager is not None:
            self.model_manager.ensure_loaded("rerank", deadline=deadline, cancel_event=cancel_event)
        elif not self.is_loaded():
            self.load()
        self._check_request(deadline, cancel_event)

    def _score_batch_sync(self, batch, pad_id, deadline, cancel_event, expected_bundle=None) -> list[float]:
        """One bounded owner job; its lease outlives a timed-out submitter."""
        self._check_request(deadline, cancel_event)
        manager = self.model_manager
        leased = False
        try:
            if manager is not None:
                manager.acquire_lease("rerank", timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
                leased = True
            if manager is not None:
                self._ensure_loaded(deadline, cancel_event)
            with self._init_lock:
                self._check_request(deadline, cancel_event)
                if expected_bundle is not None:
                    model, tokenizer, yes_id, no_id = expected_bundle
                    if (
                        not self.is_loaded() or self.model is not model
                        or self.raw_hf_tokenizer is not tokenizer
                        or (self.yes_token_id, self.no_token_id) != (yes_id, no_id)
                    ):
                        raise ModelUnavailableError("Rerank model/tokenizer bundle changed during the request")
                elif manager is None:
                    self._ensure_loaded(deadline, cancel_event)
                elif not self.is_loaded():
                    raise ModelUnavailableError("Rerank model was unloaded before scoring")
                self._check_request(deadline, cancel_event)
                width = max(len(s) for s in batch)
                padded = [s + [pad_id] * (width - len(s)) for s in batch]
                input_ids = mx.array(padded, dtype=mx.int32)
                rows = mx.arange(len(batch))
                cols = mx.array([len(s) - 1 for s in batch], dtype=mx.int32)
                backbone = getattr(self.model, "model", None)
                if backbone is None:
                    backbone = getattr(self.model, "transformer", None)
                lm_head = getattr(self.model, "lm_head", None)
                if lm_head is None:
                    lm_head = getattr(self.model, "head", None)
                tied = getattr(getattr(self.model, "args", None), "tie_word_embeddings", None)
                embedding = getattr(backbone, "embed_tokens", None)
                tied_head = getattr(embedding, "as_linear", None)
                use_tied = callable(tied_head) and (tied is True or (lm_head is None and tied is not False))

                if backbone is not None and (lm_head is not None or use_tied):
                    hidden_states = backbone(input_ids)
                    last_hidden = hidden_states[rows, cols]
                    # Preserve the actual head's projection/bias/rounding, then
                    # subtract logits. h @ (W_yes - W_no) changes FP semantics.
                    # Gather first to allocate [batch, vocabulary], not [B, S, V].
                    head = tied_head if use_tied else lm_head
                    if not callable(head):
                        raise RerankError("Reranker output head is not callable")
                    last_logits = head(last_hidden)
                    diffs = self._logit_diffs(last_logits, len(batch))
                else:
                    # Compatibility fallback for models without a separable backbone.
                    logits = self.model(input_ids)
                    last_logits = logits[rows, cols]
                    diffs = self._logit_diffs(last_logits, len(batch))
                diffs_f32 = diffs.astype(mx.float32)
                mx.eval(diffs_f32)
                diffs_np = np.array(diffs_f32, dtype=np.float64)
                if diffs_np.shape != (len(batch),) or not np.isfinite(diffs_np).all():
                    raise RerankError("Reranker returned invalid or non-finite score logits")
                self._check_request(deadline, cancel_event)
                with np.errstate(over="ignore"):
                    return [float(1.0 / (1.0 + np.exp(-d))) for d in diffs_np]
        finally:
            if leased and manager is not None:
                manager.release_lease("rerank")

    def _validate_output_ids(self, vocab_size):
        ids = self._validate_tokens([self.yes_token_id, self.no_token_id])
        if ids[0] == ids[1] or max(ids) >= vocab_size:
            raise RerankError("Resolved yes/no token IDs are invalid for the output vocabulary")

    def _logit_diffs(self, logits, batch_size):
        if len(logits.shape) != 2 or logits.shape[0] != batch_size:
            raise RerankError("Reranker head must return [batch, vocabulary] logits")
        self._validate_output_ids(logits.shape[-1])
        return logits[:, self.yes_token_id] - logits[:, self.no_token_id]

    def _score_pairs(self, query, documents, batch_size, deadline, cancel_event, schedule):
        if not isinstance(query, str) or not query.strip():
            raise RerankError("Query must be a non-empty string.")
        if not isinstance(documents, list):
            raise RerankError("Documents must be a list of strings.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 32:
            raise RerankError(f"batch_size must be an integer in [1, 32], got {batch_size}")
        for i, doc in enumerate(documents):
            if not isinstance(doc, str) or not doc.strip():
                raise RerankError(f"Document at index {i} must be a non-empty string")
        self._check_request(deadline, cancel_event)
        if not documents:
            return []

        if self.model_manager is None and not schedule:
            # Without a residency manager, a synchronous request owns its
            # complete bundle through preparation and all of its forwards.
            with self._init_lock:
                self._check_request(deadline, cancel_event)
                return self._score_pairs_request(query, documents, batch_size, deadline, cancel_event, schedule)
        return self._score_pairs_request(query, documents, batch_size, deadline, cancel_event, schedule)

    def _score_pairs_request(self, query, documents, batch_size, deadline, cancel_event, schedule):
        t0 = time.time()
        manager = self.model_manager
        leased = False
        try:
            if manager is not None:
                manager.acquire_lease("rerank", timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
                leased = True
            if schedule and manager is None:
                assert self.executor is not None and deadline is not None
                if not self.is_loaded():
                    self.executor.submit(
                        lambda: self._ensure_loaded(deadline, cancel_event), priority=0,
                        timeout_s=deadline - time.monotonic(), cancel_event=cancel_event,
                        description="Load rerank",
                    )
            else:
                self._ensure_loaded(deadline, cancel_event)

            # Pin the preparation bundle under lock, but do not hold this lock
            # over caller-side tokenization or executor submission in managed
            # mode: bounded microbatches must still interleave on the owner.
            with self._init_lock:
                self._check_request(deadline, cancel_event)
                if not self.is_loaded():
                    raise ModelUnavailableError("Rerank model was unloaded before tokenization")
                bundle = (self.model, self.raw_hf_tokenizer, self.yes_token_id, self.no_token_id)
            model, tokenizer, _, _ = bundle
            seqs = []
            for doc in documents:
                self._check_request(deadline, cancel_event)
                tokens = self._validate_tokens(tokenizer.encode(self._format_pair(query, doc)))
                self._validate_vocab(tokens, model)
                if len(tokens) > self.max_length:
                    raise RerankError(
                        f"Document token length ({len(tokens)}) exceeds max safe budget ({self.max_length}) for reranker"
                    )
                seqs.append(tokens)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is None:
                raise RerankError("Tokenizer has neither pad nor EOS token for batch padding")
            pad_id = self._validate_tokens([pad_id])[0]
            self._validate_vocab([pad_id], model)
            scores = []
            for start in range(0, len(seqs), batch_size):
                self._check_request(deadline, cancel_event)
                batch = seqs[start:start + batch_size]
                if schedule:
                    assert self.executor is not None and deadline is not None
                    scores.extend(self.executor.submit(
                        lambda batch=batch: self._score_batch_sync(batch, pad_id, deadline, cancel_event, bundle),
                        priority=0, timeout_s=deadline - time.monotonic(), cancel_event=cancel_event,
                        description=f"Rerank docs {start + 1}-{start + len(batch)}",
                    ))
                else:
                    scores.extend(self._score_batch_sync(batch, pad_id, deadline, cancel_event, bundle))
            self._check_request(deadline, cancel_event)
            with self._init_lock:
                self._check_request(deadline, cancel_event)
                self.total_requests += 1
                self.total_pairs_scored += len(documents)
                self.total_latency_ms += (time.time() - t0) * 1000
            if manager is not None:
                manager.touch("rerank")
            return scores
        finally:
            if leased and manager is not None:
                manager.release_lease("rerank")

    def score_pairs_sync(
        self, query: str, documents: list[str], batch_size: int = 1,
        deadline: Optional[float] = None, cancel_event: Optional[threading.Event] = None,
    ) -> list[float]:
        """Inline owner execution (without an executor, execution is synchronous)."""
        if self.executor is not None and not self.executor.is_owner_thread():
            return self.score_pairs(query, documents, batch_size=batch_size, deadline=deadline, cancel_event=cancel_event)
        return self._score_pairs(query, documents, batch_size, deadline, cancel_event, schedule=False)

    def score_pairs(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 1,
        timeout_s: float | None = 100.0,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> list[float]:
        """Tokenize on the caller, submitting one bounded GPU job per microbatch."""
        to = timeout_s if timeout_s is not None else 300.0
        if not isinstance(to, Real) or isinstance(to, bool) or not math.isfinite(to):
            raise RerankError("timeout_s must be finite")
        dl = deadline if deadline is not None else time.monotonic() + to
        if not isinstance(dl, Real) or isinstance(dl, bool) or not math.isfinite(dl):
            raise RerankError("deadline must be finite")
        event = cancel_event if cancel_event is not None else threading.Event()
        schedule = self.executor is not None and not self.executor.is_owner_thread()
        return self._score_pairs(query, documents, batch_size, dl, event, schedule)

    def warmup(self):
        print("[mlx-rerank] Running GPU warmup...")
        self.score_pairs("warmup query", ["warmup document"], timeout_s=30.0)
        print("[mlx-rerank] Warmup complete ✓")

    def get_descriptor(self) -> dict[str, Any]:
        return {
            "version": 1,
            "backend": "mlx",
            "model": self.model_name,
            "revision": self.revision or "",
            "quantization": self.quantization,
            "dtype": self.dtype_str,
            "maxTokens": self.max_length,
            "yesTokenId": self.yes_token_id,
            "noTokenId": self.no_token_id,
        }

    def shutdown(self):
        if self.executor and hasattr(self.executor, "is_owner_thread") and self.executor.is_owner_thread():
            self.unload()
        elif self.executor and hasattr(self.executor, "is_worker_alive") and self.executor.is_worker_alive():
            try:
                self.executor.submit(self.unload, priority=0, timeout_s=10.0, description="Unload rerank")
            except Exception:
                if hasattr(self.executor, "join_worker"):
                    self.executor.join_worker(timeout=5.0)
                if not self.executor.is_worker_alive():
                    self.unload()
        else:
            self.unload()
