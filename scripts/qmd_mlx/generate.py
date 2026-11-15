"""
generate.py — MLX-native Query Expansion (Generation) Adapter
"""

from __future__ import annotations

import math
import threading
import time
from numbers import Integral, Real
from typing import Any, Optional

from .protocol import (
    DeadlineExceededError,
    InvalidInputError,
    ModelUnavailableError,
    RequestCancelledError,
)

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class GenerateError(InvalidInputError):
    """Raised when generation input validation or inference fails."""
    pass


class MLXGenerateAdapter:
    """
    Query-expansion generation adapter on Apple Silicon MLX.
    Deterministic-by-default (temp=0) short completions.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-1.7B-4bit",
        revision: Optional[str] = None,
        max_context: int = 4096,
        max_new_tokens: int = 600,
        lazy_load: bool = False,
        executor: Optional[Any] = None,
        model_manager: Optional[Any] = None,
    ):
        if not _MLX_AVAILABLE:
            raise ModelUnavailableError("MLX or mlx-lm is not installed.")
        if not isinstance(max_context, int) or isinstance(max_context, bool) or max_context < 1:
            raise GenerateError("max_context must be a positive integer")
        self._validate_max_tokens(max_new_tokens)

        self.model_name = model_name
        self.revision = revision
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.executor = executor
        self.model_manager = model_manager
        self._init_lock = threading.RLock()

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.model_memory_mb: float = 0.0
        self.total_requests: int = 0
        self.total_tokens_generated: int = 0
        self.total_latency_ms: float = 0.0

        if not lazy_load and self.model_manager is None:
            self.load()

    def load_via_manager(self, timeout_s: float = 120.0):
        if self.model_manager is not None:
            self.model_manager.ensure_loaded("generate", timeout_s=timeout_s)
        else:
            self.load()

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.raw_hf_tokenizer is not None

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
        with self._init_lock:
            if self.is_loaded():
                return
            self.unload()
            t0 = time.time()
            mem_before = self._get_active_memory_mb()
            try:
                model, tokenizer_wrap = mlx_lm.load(self.model_name, revision=self.revision)
                tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)
                if model is None or not callable(getattr(tokenizer, "encode", None)):
                    raise ValueError("Loaded generator has no model or usable tokenizer")
                memory_mb = max(0.0, self._get_active_memory_mb() - mem_before)
                self.tokenizer = tokenizer_wrap
                self.raw_hf_tokenizer = tokenizer
                self.model_memory_mb = memory_mb
                self.model = model
                print(
                    f"[mlx-generate] Loaded '{self.model_name}' ✓ ({memory_mb:.1f}MB Metal) "
                    f"in {time.time() - t0:.2f}s"
                )
                return
            except Exception as exc:
                message = f"Failed to load generation model '{self.model_name}': {exc}"
            self.unload()
            model = tokenizer_wrap = tokenizer = None
            # A manager/future can retain this error. Neither its locals nor an
            # implicit exception context may keep the failed loader's tensors.
            raise ModelUnavailableError(message) from None

    def unload(self):
        with self._init_lock:
            self.model = None
            self.tokenizer = None
            self.raw_hf_tokenizer = None
            self.model_memory_mb = 0.0

    @staticmethod
    def _validate_max_tokens(max_tokens):
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 0 < max_tokens <= 4096:
            raise GenerateError(f"max_tokens must be an integer in (0, 4096], got {max_tokens}")

    @staticmethod
    def _validate_temperature(temperature):
        if not isinstance(temperature, Real) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 0:
            raise GenerateError("temperature must be a finite non-negative number")

    @staticmethod
    def _check_request(deadline, cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Generation request cancelled")
        if deadline is not None:
            if not isinstance(deadline, Real) or isinstance(deadline, bool) or not math.isfinite(deadline):
                raise GenerateError("deadline must be finite")
            if time.monotonic() >= deadline:
                raise DeadlineExceededError("Generation request deadline exceeded")

    def _generate_sync(
        self, prompt: str, max_tokens: int, temperature: float,
        deadline: Optional[float] = None, cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, int]:
        """Keep a lifecycle lease until the actual decode finishes, not submit's timeout."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise GenerateError("Prompt must be a non-empty string.")
        self._validate_max_tokens(max_tokens)
        self._validate_temperature(temperature)
        self._check_request(deadline, cancel_event)
        manager = self.model_manager
        leased = False
        try:
            if manager is not None:
                manager.acquire_lease("generate", timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
                leased = True
            if manager is not None:
                manager.ensure_loaded("generate", deadline=deadline, cancel_event=cancel_event)
            with self._init_lock:
                self._check_request(deadline, cancel_event)
                if not self.is_loaded():
                    if manager is not None:
                        raise ModelUnavailableError("Generation model was unloaded before decoding")
                    self.load()
                self._check_request(deadline, cancel_event)
                return self._decode_sync(prompt, max_tokens, temperature, deadline, cancel_event)
        finally:
            if leased and manager is not None:
                manager.release_lease("generate")

    def _decode_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, int]:
        self._check_request(deadline, cancel_event)

        formatted_prompt = prompt
        chat_formatted = False
        if hasattr(self.raw_hf_tokenizer, "apply_chat_template"):
            try:
                formatted_prompt = self.raw_hf_tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chat_formatted = True
            except Exception:
                chat_formatted = False
        if not chat_formatted:
            formatted_prompt = prompt
        if not isinstance(formatted_prompt, str) or not formatted_prompt.strip():
            raise GenerateError("Chat template must produce a non-empty prompt string")

        t0 = time.time()

        # Match mlx-lm's string-prompt BOS rule, then pass these exact IDs to
        # decoding so validation cannot count a different tokenization.
        bos = getattr(self.tokenizer, "bos_token", None)
        add_special_tokens = not isinstance(bos, str) or not formatted_prompt.startswith(bos)
        toks = self.raw_hf_tokenizer.encode(formatted_prompt, add_special_tokens=add_special_tokens)
        if not isinstance(toks, (list, tuple)) or not toks:
            raise GenerateError("Empty or invalid formatted prompt token sequence")
        if any(not isinstance(t, Integral) or isinstance(t, bool) or not 0 <= int(t) < 2**31 for t in toks):
            raise GenerateError("Tokenizer must return non-negative int32 token IDs")
        toks = [int(t) for t in toks]
        vocab_size = getattr(getattr(self.model, "args", None), "vocab_size", None)
        if isinstance(vocab_size, Integral) and not isinstance(vocab_size, bool) and int(vocab_size) > 0:
            if max(toks) >= int(vocab_size):
                raise GenerateError("Formatted prompt token IDs exceed the model vocabulary")
        if len(toks) + max_tokens > self.max_context:
            raise GenerateError(
                f"Formatted prompt ({len(toks)} tokens) plus requested completion ({max_tokens} tokens) "
                f"exceeds max context ({self.max_context} tokens)"
            )
        self._check_request(deadline, cancel_event)

        sampler = make_sampler(temp=temperature)
        self._check_request(deadline, cancel_event)
        pieces = []
        if hasattr(mlx_lm, "stream_generate"):
            stream = mlx_lm.stream_generate(
                self.model,
                self.tokenizer,
                prompt=toks,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            try:
                iterator = iter(stream)
                while True:
                    self._check_request(deadline, cancel_event)
                    try:
                        response = next(iterator)
                    except StopIteration:
                        break
                    self._check_request(deadline, cancel_event)
                    pieces.append(getattr(response, "text", str(response)))
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            text = "".join(pieces)
        else:
            text = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=toks,
                max_tokens=max_tokens,
                sampler=sampler,
            )
        self._check_request(deadline, cancel_event)
        text = text if isinstance(text, str) else str(text)

        OPEN = "<think>"
        CLOSE = "</think>"
        NL2 = "\n\n"
        if text.startswith(NL2):
            text = text[len(NL2):]
        if text.startswith(OPEN):
            end = text.find(CLOSE)
            if end != -1:
                text = text[end + len(CLOSE):]
            else:
                text = text[len(OPEN):]
        text = text.strip()

        # Count tokens accurately
        if pieces:
            num_tokens = len(pieces)
        elif self.raw_hf_tokenizer is not None:
            num_tokens = len(self.raw_hf_tokenizer.encode(text))
        else:
            num_tokens = max(1, len(text.split()))

        self._check_request(deadline, cancel_event)
        latency_ms = (time.time() - t0) * 1000
        self.total_requests += 1
        self.total_tokens_generated += num_tokens
        self.total_latency_ms += latency_ms

        if self.model_manager is not None:
            self.model_manager.touch("generate")

        return text, num_tokens

    def submit_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise GenerateError("Prompt must be a non-empty string.")
        mt = max_tokens if max_tokens is not None else self.max_new_tokens
        self._validate_max_tokens(mt)
        self._validate_temperature(temperature)
        if not isinstance(timeout, Real) or isinstance(timeout, bool) or not math.isfinite(timeout):
            raise GenerateError("timeout must be finite")

        now = time.monotonic()
        dl = deadline if deadline is not None else (now + timeout)
        event = cancel_event if cancel_event is not None else threading.Event()
        self._check_request(dl, event)
        rem = dl - time.monotonic()

        if self.executor is not None and not self.executor.is_owner_thread():
            return self.executor.submit(
                lambda: self._generate_sync(
                    prompt, mt, temperature,
                    deadline=dl,
                    cancel_event=event,
                )[0],
                priority=0,  # interactive priority
                timeout_s=rem,
                cancel_event=event,
                description="Generate expansion",
            )

        text, _ = self._generate_sync(prompt, mt, temperature, deadline=dl, cancel_event=event)
        return text

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.0) -> str:
        return self.submit_generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def warmup(self):
        self.submit_generate("/no_think warmup", max_tokens=4, timeout=30.0)
        print("[mlx-generate] Warmup complete ✓")

    def get_stats_info(self) -> dict[str, Any]:
        avg_ms = round(self.total_latency_ms / self.total_requests, 2) if self.total_requests > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_ms": avg_ms,
        }

    def get_descriptor(self) -> dict[str, Any]:
        return {
            "version": 1,
            "backend": "mlx",
            "kind": "generate",
            "model": self.model_name,
            "revision": self.revision or "",
            "maxContext": self.max_context,
            "maxNewTokens": self.max_new_tokens,
        }

    def shutdown(self):
        if self.executor and hasattr(self.executor, "is_owner_thread") and self.executor.is_owner_thread():
            self.unload()
        elif self.executor and hasattr(self.executor, "is_worker_alive") and self.executor.is_worker_alive():
            try:
                self.executor.submit(self.unload, priority=0, timeout_s=10.0, description="Unload generate")
            except Exception:
                if hasattr(self.executor, "join_worker"):
                    self.executor.join_worker(timeout=5.0)
                if not self.executor.is_worker_alive():
                    self.unload()
        else:
            self.unload()