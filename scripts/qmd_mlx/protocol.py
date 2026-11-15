"""
protocol.py — Framing, binary serialization, typed errors, and request/response validation
"""

from enum import Enum
import json
import math
import re
import socket
import struct
import time
import numpy as np
from typing import Any, Optional


class WorkClass(str, Enum):
    """Workload classification for priority scheduling and microbatch sizing."""
    INTERACTIVE = "interactive"
    BULK = "bulk"


class MLXRuntimeError(Exception):
    """Base runtime error for all MLX operations."""
    pass


class MLXServerError(MLXRuntimeError):
    """Base exception for MLX server errors with HTTP status mapping."""
    status_code: int = 500
    error_type: str = "server_error"

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class ProtocolError(MLXServerError, ValueError):
    """Raised when wire protocol serialization or payload validation fails."""
    status_code = 400
    error_type = "invalid_protocol"


class InvalidInputError(ProtocolError):
    """Raised on invalid or malformed input parameters."""
    status_code = 400
    error_type = "invalid_input"


class OverloadedError(MLXServerError):
    """Raised when the executor queue is full or server is overwhelmed."""
    status_code = 429
    error_type = "overloaded"


class DeadlineExceededError(MLXServerError, TimeoutError):
    """Raised when request execution exceeds the absolute deadline."""
    status_code = 504
    error_type = "deadline_exceeded"


class RequestCancelledError(MLXServerError):
    """Raised when client aborts/cancels the request before completion."""
    status_code = 400
    error_type = "request_cancelled"


class UnsupportedModelError(MLXServerError):
    """Raised when a requested model architecture or identifier is unsupported."""
    status_code = 400
    error_type = "unsupported_model"


class ModelUnavailableError(MLXServerError):
    """Raised when model loading fails or adapter is not configured."""
    status_code = 503
    error_type = "model_unavailable"


class OutOfMemoryError(MLXServerError):
    """Raised when GPU/Metal memory allocation fails after retry budget."""
    status_code = 503
    error_type = "out_of_memory"


class WorkerUnavailableError(MLXServerError):
    """Raised when the execution owner worker is stopped or crashed."""
    status_code = 503
    error_type = "worker_unavailable"


class RequestReadTimeoutError(ProtocolError):
    """An absolute HTTP header/upload deadline expired (not an inference timeout)."""
    status_code = 408
    error_type = "request_timeout"


class BoundedDeadlineReader:
    """Socket reader with per-phase absolute deadlines and consumed-byte bounds.

    Read-ahead is limited to one socket chunk and retained across header/body and
    keepalive boundaries. It does not charge a coalesced body to the header limit.
    Every recv recomputes the remaining wall-clock budget; socket inactivity
    timeouts alone cannot bound a client that regularly drips bytes.
    """

    def __init__(self, raw_socket: socket.socket, deadline: float, max_bytes: int = 16384):
        self._sock = raw_socket
        self._buf = bytearray()
        self.set_limits(deadline, max_bytes)

    def set_limits(self, deadline: float, max_bytes: int):
        self._deadline = deadline
        self._max_bytes = max_bytes
        self._bytes_read = 0

    def _check_deadline(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise RequestReadTimeoutError("HTTP request read deadline exceeded")
        self._sock.settimeout(remaining)

    def _receive(self):
        self._check_deadline()
        try:
            data = self._sock.recv(4096)
        except socket.timeout as exc:
            raise RequestReadTimeoutError("HTTP request read deadline exceeded") from exc
        self._buf.extend(data)
        return bool(data)

    def _take(self, size: int) -> bytes:
        self._check_deadline()
        if self._bytes_read + size > self._max_bytes:
            raise ProtocolError("HTTP request exceeded maximum byte budget", status_code=431)
        data = bytes(self._buf[:size])
        del self._buf[:size]
        self._bytes_read += size
        return data

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size > self._max_bytes - self._bytes_read:
            raise ProtocolError("HTTP request exceeded maximum byte budget")
        chunks = []
        remaining = size
        while remaining != 0:
            self._check_deadline()
            if self._buf:
                count = len(self._buf) if remaining < 0 else min(remaining, len(self._buf))
                chunks.append(self._take(count))
                if remaining > 0:
                    remaining -= count
            elif not self._receive():
                break
        return b"".join(chunks)

    def readline(self, limit: int = -1) -> bytes:
        while True:
            self._check_deadline()
            newline = self._buf.find(b"\n")
            size = newline + 1 if newline >= 0 else len(self._buf)
            if limit > 0:
                size = min(size, limit)
            if self._bytes_read + size > self._max_bytes:
                raise ProtocolError("HTTP request exceeded maximum byte budget", status_code=431)
            if newline >= 0 or (limit > 0 and len(self._buf) >= limit):
                return self._take(size)
            if not self._receive():
                return self._take(len(self._buf))

    def close(self):
        # The HTTP server owns the socket; closing a phase reader must not close it.
        pass


def parse_http_body_headers(headers: Any, max_body_bytes: int) -> int:
    """Require unambiguous length-delimited JSON before reading/allocating a body."""
    if headers.get_all("Transfer-Encoding"):
        raise ProtocolError("Transfer-Encoding is not supported")
    lengths = headers.get_all("Content-Length", [])
    if len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0].strip(" \t")):
        raise ProtocolError("Exactly one valid Content-Length header is required")
    # Compare decimal strings first, avoiding arbitrary-size integer parsing.
    digits = lengths[0].strip(" \t").lstrip("0") or "0"
    maximum = str(max_body_bytes)
    if len(digits) > len(maximum) or (len(digits) == len(maximum) and digits > maximum):
        raise ProtocolError(f"Request body exceeds {max_body_bytes} bytes", status_code=413)
    length = int(digits)
    if length <= 0:
        raise ProtocolError("Request body cannot be empty")
    types = headers.get_all("Content-Type", [])
    if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
        raise ProtocolError("Content-Type must be application/json", status_code=415)
    if len(headers.get_all("X-Request-Timeout", [])) > 1:
        raise ProtocolError("X-Request-Timeout must not be repeated")
    return length


def decode_json_request(body: bytes) -> dict[str, Any]:
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise InvalidInputError("JSON numbers must be finite")
        return number

    def reject_constant(value):
        raise InvalidInputError(f"Non-finite JSON constant is not allowed: {value}")

    try:
        payload = json.loads(body.decode("utf-8"), parse_float=finite_float, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidInputError("Invalid JSON in request body") from exc
    if not isinstance(payload, dict):
        raise InvalidInputError("Payload must be a JSON object")
    return payload


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(f"'{name}' must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise InvalidInputError(f"'{name}' must be a finite number") from exc
    if not math.isfinite(number):
        raise InvalidInputError(f"'{name}' must be a finite number")
    return number


def parse_timeout(
    payload: Optional[dict] = None,
    header_val: Optional[str] = None,
    default_timeout: float = 300.0,
    min_timeout: float = 0.1,
    max_timeout: float = 3600.0,
) -> float:
    """Validate provided timeouts, retaining body precedence and finite clamping."""
    timeout = default_timeout
    if header_val is not None:
        try:
            timeout = float(header_val)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError("X-Request-Timeout must be a finite positive number") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise InvalidInputError("X-Request-Timeout must be a finite positive number")
    if isinstance(payload, dict) and "timeout" in payload:
        timeout = _finite_number(payload["timeout"], "timeout")
        if timeout <= 0:
            raise InvalidInputError("'timeout' must be a finite positive number")
    return max(min_timeout, min(timeout, max_timeout))


def encode_binary_embeddings(arr: np.ndarray, count: int, dims: int) -> bytes:
    """
    Encodes float32 numpy array into binary wire format:
    Header: [int32 count (4B)][int32 dims (4B)] little-endian (<ii)
    Data: raw float32 bytes
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 2**31 - 1 for n in (count, dims)):
        raise ProtocolError("count and dims must be non-negative int32 integers")
    if not isinstance(arr, np.ndarray):
        raise ProtocolError("Expected a numpy array")
    if arr.ndim != 2:
        raise ProtocolError(f"Expected 2D array, got ndim={arr.ndim} with shape {arr.shape}")
    if arr.shape[0] != count or arr.shape[1] != dims:
        raise ProtocolError(f"Shape {arr.shape} does not match count={count}, dims={dims}")

    if count == 0 or dims == 0:
        if count != 0 or dims != 0:
            raise ProtocolError("Empty frames must have count=0 and dims=0")
        return struct.pack("<ii", 0, 0)

    if arr.dtype != np.dtype("<f4"):
        try:
            arr = arr.astype("<f4")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ProtocolError("Array cannot be encoded as float32") from exc
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    if not np.all(np.isfinite(arr)):
        raise ProtocolError("Array contains NaN or Infinite float values")

    header = struct.pack("<ii", count, dims)
    return header + arr.tobytes()


def decode_binary_embeddings(buffer: bytes) -> tuple[np.ndarray, int, int]:
    """
    Decodes binary payload into (np.ndarray, count, dims).
    """
    if len(buffer) < 8:
        raise ProtocolError(f"Buffer length {len(buffer)} too short for 8-byte header")

    count, dims = struct.unpack("<ii", buffer[:8])
    if count < 0 or dims < 0:
        raise ProtocolError(f"Invalid count={count} or dims={dims}")

    if count == 0 or dims == 0:
        if count != 0 or dims != 0 or len(buffer) != 8:
            raise ProtocolError("Empty frames require count=0, dims=0 and exactly 8 bytes")
        return np.empty((0, 0), dtype=np.float32), 0, 0

    expected_bytes = 8 + count * dims * 4
    if len(buffer) != expected_bytes:
        raise ProtocolError(
            f"Buffer length {len(buffer)} does not match expected {expected_bytes} for {count}x{dims} float32"
        )

    floats = np.frombuffer(buffer, dtype="<f4", offset=8).reshape((count, dims))
    if not np.all(np.isfinite(floats)):
        raise ProtocolError("Decoded array contains NaN or Infinite float values")

    return floats, count, dims


def validate_embed_request(
    payload: Any,
    max_texts: int = 512,
    max_text_len: int = 256 * 1024,
) -> tuple[list[str], Optional[int], bool]:
    """
    Validates JSON request payload for /embed and /embed-bin.
    Returns (texts, dims, is_query).
    """
    if not isinstance(payload, dict):
        raise InvalidInputError("Payload must be a JSON object")

    texts = payload.get("texts")
    if not isinstance(texts, list):
        raise InvalidInputError("'texts' must be a list of strings")
    if len(texts) == 0:
        raise InvalidInputError("'texts' list cannot be empty")
    if len(texts) > max_texts:
        raise InvalidInputError(f"Request exceeds maximum allowed texts ({len(texts)} > {max_texts})")

    cleaned_texts: list[str] = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise InvalidInputError(f"Item at index {i} is not a string (type={type(t).__name__})")
        if not t.strip():
            raise InvalidInputError(f"Item at index {i} is empty or whitespace")
        if len(t) > max_text_len:
            raise InvalidInputError(f"Item at index {i} exceeds max length ({len(t)} > {max_text_len})")
        cleaned_texts.append(t)

    dims = payload.get("dims")
    if dims is not None:
        if isinstance(dims, bool) or not isinstance(dims, int) or not 0 < dims <= 2**31 - 1:
            raise InvalidInputError(f"'dims' must be a positive integer, got: {dims}")

    is_query = payload.get("is_query", False)
    if not isinstance(is_query, bool):
        raise InvalidInputError("'is_query' must be a boolean")
    return cleaned_texts, dims, is_query


def validate_rerank_request(
    payload: Any,
    max_documents: int = 128,
    max_doc_len: int = 256 * 1024,
) -> tuple[str, list[str]]:
    """Validates a /rerank request body. Returns (query, documents)."""
    if not isinstance(payload, dict):
        raise InvalidInputError("Payload must be a JSON object")

    query = payload.get("query")
    documents = payload.get("documents")
    if not isinstance(query, str) or not query.strip():
        raise InvalidInputError("'query' must be a non-empty string")
    if len(query) > max_doc_len:
        raise InvalidInputError(f"'query' exceeds max length {max_doc_len}")
    if not isinstance(documents, list) or len(documents) == 0:
        raise InvalidInputError("'documents' must be a non-empty list of strings")
    if len(documents) > max_documents:
        raise InvalidInputError(f"'documents' exceeds per-request limit of {max_documents} (got {len(documents)})")
    for i, doc in enumerate(documents):
        if not isinstance(doc, str):
            raise InvalidInputError(f"document at index {i} is not a string (type={type(doc).__name__})")
        if not doc.strip():
            raise InvalidInputError(f"document at index {i} is empty or whitespace")
        if len(doc) > max_doc_len:
            raise InvalidInputError(f"document at index {i} exceeds max length {max_doc_len} bytes")
    return query, documents


def validate_generate_request(
    payload: Any,
    max_prompt_len: int = 128 * 1024,
) -> tuple[str, int, float]:
    """Validates a /generate request body. Returns (prompt, max_tokens, temperature)."""
    if not isinstance(payload, dict):
        raise InvalidInputError("Payload must be a JSON object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidInputError("'prompt' must be a non-empty string")
    if len(prompt) > max_prompt_len:
        raise InvalidInputError(f"'prompt' exceeds max length {max_prompt_len} bytes")
    max_tokens = payload.get("max_tokens", 600)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0 or max_tokens > 4096:
        raise InvalidInputError(f"'max_tokens' must be an integer in (0, 4096], got {max_tokens}")
    temperature = _finite_number(payload.get("temperature", 0.0), "temperature")
    if not 0.0 <= temperature <= 2.0:
        raise InvalidInputError(f"'temperature' must be a number in [0.0, 2.0], got {temperature}")
    return prompt, max_tokens, float(temperature)
