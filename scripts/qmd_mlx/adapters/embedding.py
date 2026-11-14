"""Explicit, fail-closed MLX embedding adapters and serving-space identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional
import numpy as np

from ..protocol import InvalidInputError, ModelUnavailableError, UnsupportedModelError
from .tokenization import TokenizedBatch, TokenizerHelper

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def infer_model_params_b(model_name: str, config: Optional[dict] = None) -> float:
    """Estimate parameters, distinguishing e.g. 4B from 4bit."""
    cfg = config or {}
    if "num_parameters" in cfg and cfg["num_parameters"]:
        return float(cfg["num_parameters"]) / 1e9
    hidden = cfg.get("hidden_size") or cfg.get("d_model")
    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    if hidden and layers:
        if hidden >= 3500:
            return 7.0
        elif hidden >= 2000:
            return 4.0
        elif hidden >= 1500:
            return 1.7
        elif hidden >= 1000:
            return 0.6
    match = re.search(r'(?:^|[_\-/])(\d+(?:\.\d+)?)[bB](?:[_\-/]|$)', model_name)
    if match:
        try:
            value = float(match.group(1))
            if 0.01 <= value <= 1000.0:
                return value
        except ValueError:
            pass
    name = model_name.lower()
    if "minilm" in name:
        return 0.033
    if "nomic" in name:
        return 0.137
    if "bert" in name:
        return 0.110
    return 0.6


def estimate_model_memory_mb(model_name: str, quantization: Optional[str] = None, params_b: Optional[float] = None) -> float:
    """Conservative pre-load estimate; not a claim about loaded precision."""
    pb = params_b if params_b is not None else infer_model_params_b(model_name)
    quant = (quantization or "").lower()
    name = model_name.lower()
    if "4bit" in quant or "4bit" in name or "q4" in quant or "q4" in name or "int4" in quant or "dwq" in name:
        bits = 4.5
    elif "8bit" in quant or "8bit" in name or "q8" in quant or "q8" in name or "int8" in quant or "mxfp8" in name:
        bits = 8.5
    elif "16" in quant or "fp16" in name or "bf16" in name or "float16" in quant or "bfloat16" in quant:
        bits = 16.0
    elif "32" in quant or "fp32" in name or "float32" in quant:
        bits = 32.0
    else:
        bits = 4.5 if ("4b" in name and "4bit" in name) else 16.0
    return max(100.0, (pb * 1e9 * (bits / 8.0) / (1024 * 1024)) * 1.25)


_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# Avoid downloading alternate ONNX/OpenVINO/TF exports of the same weights.
_SNAPSHOT_FILES = [
    "config.json", "model*.safetensors", "pytorch_model*.bin",
    "*.safetensors.index.json", "*.bin.index.json", "tokenizer*",
    "special_tokens_map.json", "added_tokens.json", "vocab.*", "merges.txt",
    "sentencepiece*", "spiece.model", "sentence_bert_config.json",
]


def _parameter_arrays(tree: Any, prefix: str = ""):
    """Walk MLX's parameter tree without allocating/copying parameter arrays."""
    if isinstance(tree, dict):
        for name, value in tree.items():
            yield from _parameter_arrays(value, f"{prefix}.{name}" if prefix else str(name))
    elif isinstance(tree, (tuple, list)):
        for index, value in enumerate(tree):
            yield from _parameter_arrays(value, f"{prefix}.{index}")
    elif hasattr(tree, "dtype"):
        yield prefix, tree


def _local_identity(path: Path) -> tuple[str, list[tuple[Path, int, int]]]:
    """Content identity for local model/tokenizer assets, not a path/mtime alias.

    HF snapshots already have an immutable commit identity. For a plain local
    directory hash relevant assets once, streaming weights instead of retaining
    them in RAM. Stat checks detect ordinary mutation during hashing/loading.
    """
    suffixes = {".json", ".safetensors", ".bin", ".npz", ".txt", ".model", ".tiktoken", ".py"}
    assets = sorted(
        (file for file in path.rglob("*") if file.is_file() and file.suffix in suffixes
         and not any(part.startswith(".") for part in file.relative_to(path).parts)),
        key=lambda file: file.relative_to(path).as_posix(),
    )
    if not any(file.suffix in {".safetensors", ".bin", ".npz"} for file in assets):
        raise ModelUnavailableError(f"No supported local weight files in '{path}'")
    digest = hashlib.sha256()
    fingerprints = []
    for file in assets:
        before = file.stat()
        name = file.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(before.st_size.to_bytes(8, "big"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ModelUnavailableError("Local model assets changed while resolving identity")
        fingerprints.append((file, after.st_size, after.st_mtime_ns))
    return f"local-sha256:{digest.hexdigest()}", fingerprints


@dataclass(frozen=True)
class EmbeddingPreparation:
    """One verified generation's CPU tokenizer and value-only serving metadata.

    Never retain a model, adapter, bound adapter method, or residency lease here:
    a delayed CPU caller must not keep weights alive or block mandatory cleanup.
    The serialized descriptor is immutable, including any nested metadata.
    """

    tokenizer: Any
    max_length: int
    generation: int
    descriptor_json: str
    model_params_b: float
    requested_dims: Optional[int]

    def tokenize_texts(self, texts: list[str]) -> TokenizedBatch:
        # Deliberately independent of the adapter's mutable lifecycle state.
        # In particular, this CPU-only path can never call adapter.load().
        return TokenizerHelper.tokenize_once(texts, self.tokenizer, max_length=self.max_length)


class BaseEmbeddingAdapter:
    """Common identity, precision, dimension, and lifecycle controls."""

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        dtype_str: str = "float32",
        max_length: int = 2048,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.dtype_str = dtype_str
        if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length <= 0:
            raise InvalidInputError("max_length must be a positive integer")
        self.max_length = int(max_length)
        self._requested_max_length = int(max_length)
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.measured_quantization: Optional[str] = None
        self.measured_revision: Optional[str] = None
        self.measured_dtype: Optional[str] = None
        self._init_lock = threading.RLock()
        self._load_generation = 0
        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.native_dims = 0
        self.pooling_strategy = "mean"
        self.padding_side = "right"
        self.is_causal = False
        self.model_type = "base"
        self.supports_mrl = False
        self.model_params_b = infer_model_params_b(model_name)
        self.estimated_memory_mb = estimate_model_memory_mb(model_name, quantization, self.model_params_b)
        self.model_memory_mb = 0.0

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

    def is_loaded(self) -> bool:
        return self.model is not None and self.raw_hf_tokenizer is not None

    def load(self):
        raise NotImplementedError

    def _clear_loaded_state(self):
        self.model = self.tokenizer = self.raw_hf_tokenizer = None
        self.measured_revision = self.measured_quantization = self.measured_dtype = None
        self.native_dims = 0
        self.supports_mrl = False
        self.max_length = self._requested_max_length
        self.model_memory_mb = 0.0

    def unload(self):
        with self._init_lock:
            # Retain only the small, last-verified descriptor across idle
            # eviction; it is not a request-derived identity. A reload resolves
            # both assets anew and clears this metadata if loading fails.
            # Do not retain the CPU tokenizer implicitly.
            self.model = self.tokenizer = self.raw_hf_tokenizer = None
            self.model_memory_mb = 0.0

    def _compute_dtype(self):
        if self.dtype_str not in {"float32", "float16", "bfloat16"}:
            raise ModelUnavailableError(f"Unsupported compute dtype: {self.dtype_str}")
        if not _MLX_AVAILABLE:
            raise ModelUnavailableError("MLX is not installed.")
        return getattr(mx, self.dtype_str)

    def _resolve_source(self) -> tuple[str, str, dict, list]:
        candidate = Path(self.model_name).expanduser()
        is_local = candidate.is_dir()
        if is_local:
            path = candidate.resolve()
        else:
            from huggingface_hub import snapshot_download
            patterns = list(_SNAPSHOT_FILES)
            if self.trust_remote_code:
                patterns.append("*.py")
            path = Path(snapshot_download(
                repo_id=self.model_name, revision=self.revision, allow_patterns=patterns,
            )).resolve()
        snapshot_commit = path.name if path.parent.name == "snapshots" and _COMMIT.fullmatch(path.name) else None
        if not is_local and snapshot_commit is None:
            raise ModelUnavailableError("Model resolution did not return an immutable snapshot")
        if self.revision is not None:
            # A local directory has no branch/tag resolver. Never silently
            # ignore a revision supplied alongside a local path.
            if is_local and self.revision != snapshot_commit:
                raise ModelUnavailableError("Cannot apply requested revision to a local model path")
            if _COMMIT.fullmatch(self.revision) and snapshot_commit != self.revision:
                raise ModelUnavailableError("Resolved snapshot does not match requested revision")
        if snapshot_commit:
            revision, fingerprints = snapshot_commit, []
        else:
            revision, fingerprints = _local_identity(path)
        with (path / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise ModelUnavailableError("Model config must be a JSON object")
        return str(path), revision, config, fingerprints

    @staticmethod
    def _check_source_unchanged(fingerprints):
        for file, size, mtime_ns in fingerprints:
            current = file.stat()
            if (current.st_size, current.st_mtime_ns) != (size, mtime_ns):
                raise ModelUnavailableError("Local model assets changed while loading")

    @staticmethod
    def _quantization_policy(config: dict, *, allow_packed: bool) -> Optional[tuple[int, int, str]]:
        """Resolve only a supported uniform weight-only quantization policy."""
        info = config.get("quantization")
        # Other quantizers, per-layer/mixed modes, and activation quantization
        # require separate qualification; do not label them from a filename.
        if config.get("quantization_config") or config.get("quantize_activations"):
            raise UnsupportedModelError("Unsupported model quantization configuration")
        if info is None:
            return None
        if (not allow_packed or not isinstance(info, dict)
                or set(info) - {"bits", "group_size", "mode"}
                or not isinstance(info.get("bits"), Integral) or isinstance(info.get("bits"), bool)
                or info.get("bits") not in {4, 8} or info.get("mode", "affine") != "affine"
                or not isinstance(info.get("group_size"), Integral) or isinstance(info.get("group_size"), bool)
                or info.get("group_size") not in {32, 64, 128}):
            raise UnsupportedModelError("Unsupported model quantization configuration")
        return int(info["bits"]), int(info["group_size"]), info.get("mode", "affine")

    def _quantization_for(self, config: dict, *, allow_packed: bool) -> str:
        if self.quantization is not None and (not isinstance(self.quantization, str)
                                               or self.quantization not in {"none", "4bit", "8bit"}):
            raise UnsupportedModelError(f"Unsupported requested quantization: {self.quantization!r}")
        if not allow_packed and self.quantization not in {None, "none"}:
            raise UnsupportedModelError("BERT adapter does not implement requested quantization")
        policy = self._quantization_policy(config, allow_packed=allow_packed)
        actual = "none" if policy is None else f"{policy[0]}bit"
        if self.quantization is not None and self.quantization != actual:
            raise UnsupportedModelError(
                f"Requested quantization {self.quantization!r} does not match loaded snapshot quantization {actual!r}; "
                "on-load quantization is not implemented"
            )
        return actual

    def _apply_and_measure_dtype(
        self, model: Any, compute_dtype: Any, quantization: str, *,
        quantization_policy: Optional[tuple[int, int, str]] = None,
    ) -> str:
        # MLX Module.set_dtype casts floating leaves only, preserving packed
        # integer weights. Verify its effect rather than echoing the request.
        model.set_dtype(compute_dtype)
        # eval() is the inference-mode switch. Inspect after it as some modules
        # can change their weight representation when switching training modes.
        model.eval()
        parameters = list(_parameter_arrays(model.parameters()))
        floating = [array for _, array in parameters if mx.issubdtype(array.dtype, mx.floating)]
        if not floating or any(array.dtype != compute_dtype for array in floating):
            raise ModelUnavailableError("Model did not apply the requested floating compute dtype")
        packed = {name.rpartition(".")[0]: array for name, array in parameters
                  if name.rsplit(".", 1)[-1] == "weight" and not mx.issubdtype(array.dtype, mx.floating)}
        if bool(packed) != (quantization != "none"):
            raise ModelUnavailableError("Loaded weight storage does not match model quantization metadata")
        if packed:
            if quantization_policy is None or f"{quantization_policy[0]}bit" != quantization:
                raise ModelUnavailableError("Loaded quantization has no verified uniform policy")
            named_modules = getattr(model, "named_modules", None)
            if not callable(named_modules):
                raise ModelUnavailableError("Cannot inspect loaded quantization layer metadata")
            modules: dict[str, Any] = dict(model.named_modules())
            for name, weight in packed.items():
                layer = modules.get(name)
                if layer is None or getattr(layer, "weight", None) is not weight:
                    raise ModelUnavailableError(f"Missing loaded quantization layer for weight '{name}'")
                # The loader can infer each layer's packing from tensor shapes
                # while leaving config.json/returned config at stale defaults.
                # Inspect the fields used by that layer's actual forward call.
                info = {key: getattr(layer, key, None) for key in ("bits", "group_size", "mode")}
                try:
                    actual_policy = self._quantization_policy({"quantization": info}, allow_packed=True)
                except UnsupportedModelError:
                    raise ModelUnavailableError(f"Unsupported loaded quantization metadata for layer '{name}'") from None
                if actual_policy is None or actual_policy != quantization_policy:
                    raise ModelUnavailableError(
                        f"Loaded quantization for layer '{name}' {actual_policy!r} "
                        f"does not match resolved uniform policy {quantization_policy!r}"
                    )
                if weight.dtype != mx.uint32:
                    raise ModelUnavailableError(f"Unsupported loaded quantization weight storage for layer '{name}'")
                scales: Any = getattr(layer, "scales", None)
                biases: Any = getattr(layer, "biases", None)
                if (not hasattr(scales, "dtype") or not hasattr(biases, "dtype")
                        or scales.dtype != compute_dtype or biases.dtype != compute_dtype
                        or weight.ndim < 2 or scales.ndim != weight.ndim or biases.shape != scales.shape
                        or scales.shape[:-1] != weight.shape[:-1]
                        or weight.shape[-1] * 32 != scales.shape[-1] * actual_policy[1] * actual_policy[0]):
                    raise ModelUnavailableError(f"Inconsistent loaded quantization storage for layer '{name}'")
        # Materialize dtype casts before publishing READY/measuring residency;
        # otherwise the first forward still owns the old weights and cast graph.
        mx.eval(model.parameters())
        return self.dtype_str

    def _publish_loaded(
        self, model, tokenizer, raw_tokenizer, config, revision, dtype, quantization, mem_before, *,
        supports_mrl: bool = False,
    ):
        dimensions = config.get("hidden_size")
        if isinstance(dimensions, bool) or not isinstance(dimensions, Integral) or dimensions <= 0:
            raise ModelUnavailableError("Model config has invalid native dimensions")
        if raw_tokenizer is None or not (
            callable(getattr(raw_tokenizer, "encode", None)) or callable(raw_tokenizer)
            or callable(getattr(raw_tokenizer, "batch_encode_plus", None))
        ):
            raise ModelUnavailableError("Model has no supported tokenizer")
        context_limits = [self._requested_max_length]
        for limit in (config.get("max_position_embeddings"), getattr(raw_tokenizer, "model_max_length", None)):
            if isinstance(limit, Integral) and not isinstance(limit, bool) and limit > 0:
                context_limits.append(int(limit))
        if hasattr(raw_tokenizer, "padding_side"):
            raw_tokenizer.padding_side = "right"
        params_b = infer_model_params_b(self.model_name, config)
        mem_delta = self._get_active_memory_mb() - mem_before
        # Publish only after every validation succeeds.
        self.native_dims = int(dimensions)
        self.max_length = min(context_limits)
        self.model_params_b = params_b
        self.model_memory_mb = mem_delta if mem_delta > 10.0 else self.estimated_memory_mb
        self.measured_revision = revision
        self.measured_dtype = dtype
        self.measured_quantization = quantization
        self.supports_mrl = supports_mrl
        self.tokenizer = tokenizer
        self.raw_hf_tokenizer = raw_tokenizer
        self.model = model
        self._load_generation += 1

    def _serving_descriptor_json(self, requested_dims: Optional[int]) -> str:
        descriptor = self.get_descriptor(requested_dims=requested_dims)
        if any(not descriptor.get(key) or descriptor[key] == "unknown"
               for key in ("revision", "dtype", "quantization")):
            raise ModelUnavailableError("Embedding preparation requires a verified serving identity")
        return json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def capture_preparation(self, requested_dims: Optional[int] = None) -> EmbeddingPreparation:
        """Atomically capture published state without initializing any weights."""
        with self._init_lock:
            if not self.is_loaded() or self.raw_hf_tokenizer is None or self._load_generation <= 0:
                raise ModelUnavailableError("Embedding tokenizer/model is not loaded for preparation")
            return EmbeddingPreparation(
                tokenizer=self.raw_hf_tokenizer, max_length=self.max_length,
                generation=self._load_generation,
                descriptor_json=self._serving_descriptor_json(requested_dims),
                model_params_b=self.model_params_b, requested_dims=requested_dims,
            )

    def try_capture_preparation(self, requested_dims: Optional[int] = None) -> Optional[EmbeddingPreparation]:
        """Ready CPU fast path; never wait behind a native load or disposal.

        CPU tokenization can overlap an unrelated owner forward. If lifecycle
        state is busy/unloaded, the runtime queues bounded owner preparation
        instead of waiting on _init_lock outside the request deadline machinery.
        """
        if not self._init_lock.acquire(blocking=False):
            return None
        try:
            if not self.is_loaded():
                return None
            return self.capture_preparation(requested_dims=requested_dims)
        finally:
            self._init_lock.release()

    def assert_preparation_current(self, preparation: EmbeddingPreparation):
        """Fail closed, even for an identical-asset unload/reload generation.

        Requests do not transparently rebind or mix partial output with a new
        generation. A caller can retry the complete request with a new budget.
        This check runs on the owner before/after every forward, including OOM
        retry subdivisions; no model lease spans CPU preparation or queue waits.
        """
        with self._init_lock:
            if (not self.is_loaded() or self._load_generation != preparation.generation
                    or self.raw_hf_tokenizer is not preparation.tokenizer
                    or self._serving_descriptor_json(preparation.requested_dims) != preparation.descriptor_json):
                raise ModelUnavailableError(
                    "Embedding model generation changed during request; retry the complete request"
                )

    def tokenize_texts(self, texts: list[str]) -> TokenizedBatch:
        # Direct adapter users must load explicitly; runtime users receive an
        # owner-captured snapshot instead. Never initialize weights from CPU.
        return self.capture_preparation().tokenize_texts(texts)

    def _output_dimensions(self, requested_dims: Optional[int]) -> int:
        if requested_dims is None:
            return self.native_dims
        if (isinstance(requested_dims, bool) or not isinstance(requested_dims, Integral)
                or requested_dims <= 0 or requested_dims > self.native_dims):
            raise InvalidInputError(f"Requested dimensions must be an integer in [1, {self.native_dims}]")
        if requested_dims < self.native_dims and not self.supports_mrl:
            raise InvalidInputError("Matryoshka dimension reduction is not qualified for this model")
        return int(requested_dims)

    @staticmethod
    def _normalized_output(pooled, dimensions: int) -> np.ndarray:
        # Pooling callers also cast BEFORE any mean reduction. Norm and MRL
        # renormalization remain float32 for every compute/weight precision.
        pooled = pooled.astype(mx.float32)[:, :dimensions]
        norm = mx.sqrt(mx.sum(pooled ** 2, axis=-1, keepdims=True))
        normalized = pooled / mx.maximum(norm, 1e-12)
        mx.eval(normalized)
        # Exactly one owned CPU allocation. Do not return a read-only MLX view
        # whose allocator lifetime would leak into runtime/HTTP consumers.
        return np.array(normalized, dtype=np.float32, copy=True, order="C")

    def forward_batch(self, tokenized_batch: TokenizedBatch, requested_dims: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError

    def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
        return {
            "version": 1,
            "backend": "mlx",
            "model": self.model_name,
            "revision": self.measured_revision or "unknown",
            "pooling": self.pooling_strategy,
            "nativeDimensions": self.native_dims,
            "outputDimensions": self._output_dimensions(requested_dims),
            "maxTokens": self.max_length,
            "normalized": True,
            "dtype": self.measured_dtype or "unknown",
            "quantization": self.measured_quantization or "unknown",
        }


class QwenEmbeddingAdapter(BaseEmbeddingAdapter):
    """Qualified Qwen causal backbones, right-padding, last non-pad pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_strategy = "last_token"
        self.is_causal = True
        self.model_type = "qwen_causal"

    def _qualified_mrl(self, model: Any, config: dict, loaded_config: dict) -> bool:
        """Bind reduction to Qwen3 embedding assets AND the loaded architecture.

        Ordinary Qwen3 causal models share the same architecture but not the
        embedding training. Only known embedding-family sources (or explicit
        origin metadata for local/renamed conversions) qualify. A directory or
        arbitrary repo containing 'Qwen3-Embedding' is not training metadata.
        """
        for metadata in (config, loaded_config):
            if (metadata.get("model_type") != "qwen3"
                    or metadata.get("architectures") != ["Qwen3ForCausalLM"]
                    or metadata.get("model_file") or metadata.get("auto_map")):
                return False
        args = getattr(model, "args", None)
        if (getattr(model, "model_type", None) != "qwen3"
                or getattr(args, "model_type", None) != "qwen3"
                or loaded_config.get("hidden_size") != config.get("hidden_size")
                or getattr(args, "hidden_size", None) != config.get("hidden_size")):
            return False
        family = re.compile(
            r"(?:Qwen|mlx-community)/Qwen3-Embedding-(0\.6B|4B|8B)"
            r"(?:-(?:4bit|8bit|bf16|fp16)(?:-DWQ)?)?", re.IGNORECASE,
        )
        sizes = []
        if not Path(self.model_name).expanduser().is_dir():
            source = family.fullmatch(self.model_name)
            if source:
                sizes.append(source.group(1).lower())
        for metadata in (config, loaded_config):
            origin = metadata.get("_name_or_path")
            if origin is not None and origin != "":
                source = family.fullmatch(origin) if isinstance(origin, str) else None
                if source is None:
                    return False
                sizes.append(source.group(1).lower())
        return bool(sizes) and len(set(sizes)) == 1

    def load(self):
        with self._init_lock:
            if self.is_loaded():
                return
            self._clear_loaded_state()
            try:
                compute_dtype = self._compute_dtype()
                mem_before = self._get_active_memory_mb()
                path, revision, config, fingerprints = self._resolve_source()
                if config.get("model_type") not in {"qwen", "qwen2", "qwen3"}:
                    raise UnsupportedModelError("Unsupported Qwen embedding architecture")
                quantization = self._quantization_for(config, allow_packed=True)
                quantization_policy = self._quantization_policy(config, allow_packed=True)
                import mlx_lm
                # One resolved directory for weights AND tokenizer. No fallback
                # that drops revision/trust controls on older loader APIs.
                model, tokenizer, loaded_config = mlx_lm.load(
                    path, return_config=True, trust_remote_code=self.trust_remote_code,
                    tokenizer_config={"trust_remote_code": self.trust_remote_code, "local_files_only": True},
                )
                if not isinstance(loaded_config, dict):
                    raise ModelUnavailableError("Loader configuration does not match resolved model quantization")
                try:
                    loaded_policy = self._quantization_policy(loaded_config, allow_packed=True)
                except UnsupportedModelError:
                    raise ModelUnavailableError("Loader configuration has unsupported model quantization") from None
                if loaded_policy != quantization_policy:
                    raise ModelUnavailableError("Loader configuration does not match resolved model quantization")
                raw_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
                dtype = self._apply_and_measure_dtype(
                    model, compute_dtype, quantization, quantization_policy=quantization_policy,
                )
                supports_mrl = self._qualified_mrl(model, config, loaded_config)
                self._check_source_unchanged(fingerprints)
                self._publish_loaded(
                    model, tokenizer, raw_tokenizer, config, revision, dtype, quantization, mem_before,
                    supports_mrl=supports_mrl,
                )
                return
            except (UnsupportedModelError, ModelUnavailableError) as exc:
                error_type, message = type(exc), str(exc)
            except Exception as exc:
                error_type = ModelUnavailableError
                message = f"Failed to load Qwen embedding model '{self.model_name}': {exc}"
            self._clear_loaded_state()
            model = tokenizer = raw_tokenizer = None
            # Raise outside the except scope: a residency manager may retain
            # this error for waiters. Its traceback must not retain failed model
            # tensors through loader/set_dtype frames or exception chaining.
            raise error_type(message) from None

    def forward_batch(self, tokenized_batch: TokenizedBatch, requested_dims: Optional[int] = None) -> np.ndarray:
        dimensions = self._output_dimensions(requested_dims)
        if len(tokenized_batch) == 0:
            return np.empty((0, dimensions), dtype=np.float32)
        padded_ids, _, lengths = tokenized_batch.pad_micro_batch(padding_side="right")
        input_ids = mx.array(padded_ids, dtype=mx.int32)
        backbone = getattr(self.model, "model", getattr(self.model, "transformer", self.model))
        # Causal attention means trailing right-padding cannot affect any real
        # token; select each row's last real token, not the final padded column.
        hidden = backbone(input_ids)
        last_indices = mx.array([length - 1 for length in lengths], dtype=mx.int32)
        pooled = hidden[mx.arange(len(tokenized_batch)), last_indices]
        return self._normalized_output(pooled, dimensions)


class NomicEmbeddingAdapter(BaseEmbeddingAdapter):
    """Compatibility import only: no Nomic backend is numerically qualified."""

    def load(self):
        # mlx_embedding_models.nomic_model explicitly documents that its output
        # does not match sentence-transformers. In particular, v1.5 needs its
        # own rotary architecture and pre-MRL layer norm, not BERT mean pooling.
        raise UnsupportedModelError(
            "Nomic embedding architectures are not qualified by this MLX adapter; "
            "use a supported MiniLM/BGE/Qwen model instead"
        )


class BertEmbeddingAdapter(BaseEmbeddingAdapter):
    """Standard BERT: MiniLM mean pooling; qualified BGE v1/v1.5 CLS pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = self.model_name.lower()
        self._bge = "bge" in name
        self.pooling_strategy = "cls" if self._bge else "mean"
        self.model_type = "bert"

    def load(self):
        with self._init_lock:
            if self.is_loaded():
                return
            self._clear_loaded_state()
            try:
                compute_dtype = self._compute_dtype()
                # Fail unsupported *requested* quantization before weight load.
                self._quantization_for({}, allow_packed=False)
                mem_before = self._get_active_memory_mb()
                path, revision, config, fingerprints = self._resolve_source()
                if (config.get("model_type") != "bert" or config.get("is_decoder", False)
                        or config.get("add_cross_attention", False) or config.get("position_embedding_type", "absolute") != "absolute"
                        or config.get("hidden_act", "gelu") != "gelu"):
                    raise UnsupportedModelError("Unsupported BERT embedding architecture (standard BERT required)")
                if self._bge and not re.search(r"bge-(?:small|base|large)-(?:en|zh)(?:-v1\.5)?(?:[/_-]|$)", self.model_name.lower()):
                    raise UnsupportedModelError("Unsupported BGE embedding architecture/model family")
                quantization = self._quantization_for(config, allow_packed=False)
                from transformers import AutoTokenizer
                from mlx_embedding_models.model import Bert
                tokenizer = AutoTokenizer.from_pretrained(
                    path, local_files_only=True, trust_remote_code=self.trust_remote_code,
                )
                model = Bert.from_pretrained(path)
                dtype = self._apply_and_measure_dtype(model, compute_dtype, quantization)
                self._check_source_unchanged(fingerprints)
                self._publish_loaded(model, tokenizer, tokenizer, config, revision, dtype, quantization, mem_before)
                return
            except (UnsupportedModelError, ModelUnavailableError) as exc:
                error_type, message = type(exc), str(exc)
            except Exception as exc:
                error_type = ModelUnavailableError
                message = f"Failed to load BERT model '{self.model_name}': {exc}"
            self._clear_loaded_state()
            model = tokenizer = None
            # Do not retain the failed loader's tensor-owning traceback in a
            # cached load error. Publish a fresh error outside the except scope.
            raise error_type(message) from None

    def forward_batch(self, tokenized_batch: TokenizedBatch, requested_dims: Optional[int] = None) -> np.ndarray:
        dimensions = self._output_dimensions(requested_dims)
        if len(tokenized_batch) == 0:
            return np.empty((0, dimensions), dtype=np.float32)
        padded_ids, attn_mask, _ = tokenized_batch.pad_micro_batch(padding_side="right")
        input_ids = mx.array(padded_ids, dtype=mx.int32)
        # The encoder converts 0/1 to an additive log-mask. Keep that mask in
        # compute precision: MLX fused attention rejects a float32 mask for
        # float16/bfloat16 output. Pooling below still accumulates in float32.
        attention_mask = mx.array(attn_mask, dtype=getattr(mx, self.measured_dtype or self.dtype_str))
        # A single-sequence BERT input uses learned segment-zero embeddings.
        # mlx_embedding_models omits them entirely when token_type_ids is None,
        # unlike Hugging Face BERT's default; omitting these changes the space.
        token_type_ids = mx.zeros(input_ids.shape, dtype=mx.int32)
        hidden, _ = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.pooling_strategy == "cls":
            pooled = hidden[:, 0].astype(mx.float32)
        else:
            # Casting only after summation loses precision or overflows in fp16.
            hidden_f32 = hidden.astype(mx.float32)
            expanded_mask = mx.expand_dims(attention_mask.astype(mx.float32), -1)
            sum_hidden = mx.sum(hidden_f32 * expanded_mask, axis=1)
            sum_mask = mx.maximum(mx.sum(expanded_mask, axis=1), 1e-9)
            pooled = sum_hidden / sum_mask
        return self._normalized_output(pooled, dimensions)


class SyntheticTokenizer:
    """Lightweight deterministic CPU tokenizer for synthetic fixture rehearsals."""

    def __init__(self, max_length: int = 2048):
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = 151643
        self.padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not text:
            return []
        tokens = [int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], byteorder="big") % 100000 + 1
                  for word in text.split()]
        if not tokens:
            tokens = [1]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"tok_{token}" for token in token_ids if token not in {self.eos_token_id, self.pad_token_id})


class SyntheticEmbeddingAdapter(BaseEmbeddingAdapter):
    """Explicitly synthetic CPU-only adapter for offline pipeline rehearsals."""

    def __init__(
        self,
        model_name: str = "synthetic-qwen3-4b",
        quantization: Optional[str] = None,
        dtype_str: str = "float32",
        max_length: int = 2048,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        dims: int = 2560,
    ):
        super().__init__(model_name, quantization, dtype_str, max_length, revision, trust_remote_code)
        self.synthetic = True
        self.native_dims = dims
        self.pooling_strategy = "last_token"
        self.is_causal = True
        self.model_type = "synthetic_embedding"
        self.supports_mrl = True  # Fixture reductions are explicit, not a real-model claim.
        self.model_params_b = infer_model_params_b(model_name)
        if self.model_params_b == 0.6 and "4b" in model_name.lower():
            self.model_params_b = 4.0
        self.estimated_memory_mb = self.model_memory_mb = 128.0
        self._loaded = False
        self.hang_on_embed = os.environ.get("MLX_HANG_ON_EMBED") == "1" or "hang" in model_name.lower()
        self.hang_after_requests = int(os.environ.get("MLX_HANG_AFTER_REQUESTS", "0"))
        self._request_count = 0

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def load(self):
        with self._init_lock:
            if self._loaded:
                return
            self.raw_hf_tokenizer = SyntheticTokenizer(max_length=self.max_length)
            self.tokenizer = self.raw_hf_tokenizer
            self.model = object()
            self._loaded = True
            self._load_generation += 1

    def unload(self):
        with self._init_lock:
            self._loaded = False
            self.model = self.tokenizer = self.raw_hf_tokenizer = None

    def forward_batch(self, tokenized_batch: TokenizedBatch, requested_dims: Optional[int] = None) -> np.ndarray:
        dimensions = self._output_dimensions(requested_dims)
        if self.hang_on_embed:
            while True:
                time.sleep(0.5)
        if self.hang_after_requests > 0 and self._request_count >= self.hang_after_requests:
            while True:
                time.sleep(0.5)
        self._request_count += 1
        if len(tokenized_batch) == 0:
            return np.empty((0, dimensions), dtype=np.float32)
        count = len(tokenized_batch)
        result = np.empty((count, dimensions), dtype=np.float32)
        fail_consistency = os.environ.get("MLX_FAKE_FAIL_CONSISTENCY") == "1"
        for index, sequence in enumerate(tokenized_batch.token_ids):
            seed = int.from_bytes(hashlib.sha256(bytes(str(sequence), "utf-8")).digest()[:8], byteorder="big")
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.native_dims).astype(np.float32)[:dimensions]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            original = tokenized_batch.original_indices[index] if tokenized_batch.original_indices else index
            if fail_consistency and count > 1 and original == 0:
                perturb = np.zeros_like(vector)
                perturb[0] = 0.0293
                vector = vector + perturb
                vector = vector / np.linalg.norm(vector)
            result[index] = vector
        return result

    def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
        return {
            "version": 1,
            "backend": "mlx_synthetic",
            "model": self.model_name,
            "revision": "synthetic-rehearsal",
            "pooling": self.pooling_strategy,
            "nativeDimensions": self.native_dims,
            "outputDimensions": self._output_dimensions(requested_dims),
            "maxTokens": self.max_length,
            "normalized": True,
            "dtype": "float32",  # Actual NumPy output, independent of requested MLX dtype.
            "quantization": "synthetic-4bit",
            "synthetic": True,
        }


def resolve_embedding_adapter(
    model_name: str,
    quantization: Optional[str] = None,
    dtype_str: str = "float32",
    max_length: int = 2048,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
) -> BaseEmbeddingAdapter:
    """Resolve qualified families only; a model-name substring is not an architecture."""
    name = model_name.lower()
    options = dict(model_name=model_name, quantization=quantization, dtype_str=dtype_str,
                   max_length=max_length, revision=revision, trust_remote_code=trust_remote_code)
    if "synthetic" in name or "fake" in name or name.startswith("test-"):
        match = re.search(r'(\d+)d\b', name)
        dims = int(match.group(1)) if match else (384 if "minilm" in name or "384" in name else 2560)
        return SyntheticEmbeddingAdapter(**options, dims=dims)
    if "gemma" in name or "nomic" in name:
        raise UnsupportedModelError(f"Unqualified MLX embedding architecture: '{model_name}' (Gemma/Nomic require dedicated validated adapters)")
    if "qwen" in name:
        return QwenEmbeddingAdapter(**options)
    if "minilm" in name or "bert" in name or "bge" in name:
        return BertEmbeddingAdapter(**options)
    raise UnsupportedModelError(
        f"Unsupported MLX embedding model: '{model_name}'. "
        "Registered adapters: Synthetic/Fake, Qwen causal, standard BERT MiniLM/BGE."
    )
