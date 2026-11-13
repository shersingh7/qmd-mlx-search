"""
server.py — Bounded HTTP Server with Dedicated Control Listener, Single GPU Owner, and Health Checks
"""

from __future__ import annotations

import http.server
import json
import math
import os
import select
import socket
import socketserver
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional
from urllib.parse import quote
import numpy as np

from .executor import GPUExecutor
from .model_manager import ModelResidencyManager, ModelState
from .protocol import (
    BoundedDeadlineReader,
    DeadlineExceededError,
    InvalidInputError,
    MLXServerError,
    ModelUnavailableError,
    OutOfMemoryError,
    OverloadedError,
    ProtocolError,
    RequestCancelledError,
    UnsupportedModelError,
    WorkerUnavailableError,
    decode_binary_embeddings,
    decode_json_request,
    encode_binary_embeddings,
    parse_http_body_headers,
    parse_timeout,
    validate_embed_request,
    validate_generate_request,
    validate_rerank_request,
)
from .runtime import EmbeddingResult, MLXEmbeddingRuntime


class _SocketAdmissionMixin:
    """Own accepted sockets and release exactly the permits they acquired."""

    def _init_socket_admission(self, max_threads: int):
        self.thread_limiter = threading.BoundedSemaphore(max_threads)
        self._active_sockets: set[socket.socket] = set()
        self._admitted_sockets: set[socket.socket] = set()
        self._sockets_lock = threading.Lock()

    def get_request(self):
        sock, address = super().get_request()
        with self._sockets_lock:
            if not self._stop_event.is_set():
                self._active_sockets.add(sock)
                return sock, address
        sock.close()
        raise OSError("Server is stopping")

    def _close_all_active_sockets(self):
        with self._sockets_lock:
            sockets = list(self._active_sockets)
            admitted = self._admitted_sockets.copy()
            self._active_sockets.clear()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            # shutdown wakes an admitted handler's read. Its finally owns close:
            # closing the fd here can strand a concurrent timed recv/poll on
            # Darwin (and allow fd reuse before that poll has returned).
            if sock not in admitted:
                try:
                    sock.close()
                except OSError:
                    pass

    def process_request(self, request, client_address):
        if self._stop_event.is_set():
            self.shutdown_request(request)
            return
        if not self.thread_limiter.acquire(blocking=False):
            body = json.dumps({"error": "Server overloaded with concurrent connections", "type": "overloaded"}).encode()
            try:
                # Rejection runs on the accept loop; never let a non-reading peer block it.
                request.settimeout(0.2)
                request.sendall(
                    b"HTTP/1.1 429 Too Many Requests\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode()
                    + b"Connection: close\r\n\r\n" + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        with self._sockets_lock:
            self._admitted_sockets.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.shutdown_request(request)
            raise

    def close_request(self, request):
        with self._sockets_lock:
            self._active_sockets.discard(request)
            admitted = request in self._admitted_sockets
            self._admitted_sockets.discard(request)
        try:
            super().close_request(request)
        finally:
            if admitted:
                self.thread_limiter.release()


class ServerState:
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"


def get_health_payload(srv: ThreadedMLXServer) -> dict[str, Any]:
    """Generates a complete, typed, finite JSON health dictionary from server and worker state."""
    has_any_stage = bool(
        srv.runtime is not None
        or srv.rerank_adapter is not None
        or srv.generate_adapter is not None
    )
    is_ready = bool(
        srv.state == ServerState.READY
        and has_any_stage
        and srv.executor is not None
        and srv.executor.is_alive()
    )
    desc = None
    if is_ready:
        if srv.runtime and hasattr(srv.runtime, "get_descriptor"):
            try:
                d = srv.runtime.get_descriptor()
                if isinstance(d, dict):
                    desc = d
            except Exception:
                pass
        elif srv.rerank_adapter and hasattr(srv.rerank_adapter, "get_descriptor"):
            try:
                d = srv.rerank_adapter.get_descriptor()
                if isinstance(d, dict):
                    desc = d
            except Exception:
                pass
        elif srv.generate_adapter and hasattr(srv.generate_adapter, "get_descriptor"):
            try:
                d = srv.generate_adapter.get_descriptor()
                if isinstance(d, dict):
                    desc = d
            except Exception:
                pass

    worker_progress = {}
    if srv.executor and hasattr(srv.executor, "get_worker_progress"):
        try:
            wp = srv.executor.get_worker_progress()
            if isinstance(wp, dict):
                worker_progress = wp
        except Exception:
            pass

    queue_depth = 0
    if "queue_depth" in worker_progress and isinstance(worker_progress["queue_depth"], int):
        queue_depth = worker_progress["queue_depth"]
    elif srv.executor and hasattr(srv.executor, "get_queue_depth"):
        try:
            qd = srv.executor.get_queue_depth()
            if isinstance(qd, int):
                queue_depth = qd
        except Exception:
            pass

    is_overloaded = False
    if srv.executor and hasattr(srv.executor, "is_overloaded"):
        try:
            ov = srv.executor.is_overloaded()
            if isinstance(ov, bool):
                is_overloaded = ov
        except Exception:
            pass

    in_flight = 0
    if srv.runtime and hasattr(srv.runtime, "in_flight_requests"):
        try:
            ifr = srv.runtime.in_flight_requests
            if isinstance(ifr, int):
                in_flight = ifr
        except Exception:
            pass

    worker_alive = False
    if "worker_alive" in worker_progress and isinstance(worker_progress["worker_alive"], bool):
        worker_alive = worker_progress["worker_alive"]
    elif srv.executor is not None and hasattr(srv.executor, "is_alive"):
        try:
            worker_alive = bool(srv.executor.is_alive())
        except Exception:
            pass

    worker_idle = bool(worker_progress.get("is_idle", True)) if "is_idle" in worker_progress else True

    active_job_age = worker_progress.get("active_job_age_s")
    if not (
        isinstance(active_job_age, (int, float))
        and not isinstance(active_job_age, bool)
        and math.isfinite(active_job_age)
        and active_job_age >= 0
    ):
        active_job_age = None

    active_job_desc = worker_progress.get("active_job_description")
    if not isinstance(active_job_desc, str):
        active_job_desc = None

    completed_seq = worker_progress.get("completed_sequence", 0)
    if not isinstance(completed_seq, int) or isinstance(completed_seq, bool) or completed_seq < 0:
        completed_seq = 0

    status_str = (
        "ok"
        if is_ready
        else ("starting" if srv.state in (ServerState.STARTING, ServerState.LOADING) else "degraded")
    )

    model_name = getattr(srv.runtime, "model_name", None) if srv.runtime else None
    if not isinstance(model_name, str):
        model_name = None

    native_dims = getattr(srv.runtime, "native_dims", None) if (srv.runtime and is_ready) else None
    if not isinstance(native_dims, int) or isinstance(native_dims, bool):
        native_dims = None

    rerank_name = getattr(srv.rerank_adapter, "model_name", None) if srv.rerank_adapter else None
    if not isinstance(rerank_name, str):
        rerank_name = None

    gen_name = getattr(srv.generate_adapter, "model_name", None) if srv.generate_adapter else None
    if not isinstance(gen_name, str):
        gen_name = None

    uptime_s = round(max(0.0, time.monotonic() - srv.start_time_monotonic), 2)

    return {
        "status": status_str,
        "state": srv.state,
        "pid": srv.server_pid,
        "instance_token": srv.instance_token,
        "uptime_s": uptime_s,
        "model": model_name,
        "dims": native_dims,
        "ready": is_ready,
        "queue_depth": queue_depth,
        "in_flight_requests": in_flight,
        "overloaded": is_overloaded,
        "worker_alive": worker_alive,
        "worker_idle": worker_idle,
        "active_job_age_s": active_job_age,
        "active_job_description": active_job_desc,
        "completed_sequence": completed_seq,
        "descriptor": desc,
        "rerank_model": rerank_name,
        "generate_model": gen_name,
        "error": srv.state_error,
    }


class ThreadedMLXServer(_SocketAdmissionMixin, socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        max_threads: int = 128,
        max_concurrent_inference: int = 32,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.state = ServerState.STARTING
        self.state_error: Optional[str] = None
        self.bind_host: str = server_address[0]
        # Use MLX_INSTANCE_TOKEN from env if present, else fresh uuid
        self.instance_token: str = os.getenv("MLX_INSTANCE_TOKEN") or uuid.uuid4().hex
        self.server_pid: int = os.getpid()
        self.start_time_monotonic: float = time.monotonic()
        self.start_time_epoch: float = time.time()
        self.max_body_bytes: int = 10 * 1024 * 1024  # 10 MB limit
        self.max_concurrent_inference = max_concurrent_inference
        self._init_socket_admission(max_threads)
        self.inference_limiter = threading.BoundedSemaphore(max_concurrent_inference)
        self.header_timeout_s = 15.0
        self.body_timeout_s = 15.0
        self.max_header_bytes = 16384
        self.executor: Optional[GPUExecutor] = None
        self.model_manager: Optional[ModelResidencyManager] = None
        self.runtime: Optional[MLXEmbeddingRuntime] = None
        self.rerank_adapter: Optional[Any] = None
        self.generate_adapter: Optional[Any] = None
        self.control_server: Optional[MLXControlServer] = None
        self.control_port: Optional[int] = None
        self.t_http: Optional[threading.Thread] = None
        self.t_ctrl: Optional[threading.Thread] = None
        self._serving_event = threading.Event()
        self._stop_event = threading.Event()
        self._stopped_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._stop_lock = self._lifecycle_lock
        self._before_serve_hook: Optional[Any] = None

    def stop(self, timeout: float = 10.0):
        """
        Coordinates graceful server shutdown lifecycle:
        1. Sets STOPPING state and sets stop event under lifecycle lock.
        2. Closes all owned active client connections on both inference and control listeners.
        3. Stops control server if attached and joins control serving thread.
        4. Stops runtime admission to reject new requests and wake waiting admission leases.
        5. Explicitly shuts down shared server.executor (cancelling queued work and joining worker).
        6. If worker thread has terminated, safely unloads all resident models and sets _stopped_event.
           If worker thread is still alive (e.g. blocked job beyond join budget), defers unload and _stopped_event,
           truthfully reporting pending state without unloading live kernel.
        """
        with self._lifecycle_lock:
            self.state = ServerState.STOPPING
            self._stop_event.set()

            # Close active client connections to unblock any pending handlers immediately
            self._close_all_active_sockets()

            if self.control_server is not None:
                try:
                    self.control_server._close_all_active_sockets()
                    self.control_server.stop()
                    self.control_server.server_close()
                except Exception:
                    pass

            if self.t_ctrl is not None and self.t_ctrl.is_alive():
                try:
                    self.t_ctrl.join(timeout=1.0)
                except Exception:
                    pass

            # Stop runtime admission first
            if self.runtime:
                try:
                    self.runtime.stop_admission()
                except Exception:
                    pass

            # Explicitly shut down shared executor regardless of runtime presence
            if self.executor:
                try:
                    self.executor.shutdown(timeout=timeout)
                except Exception:
                    pass

            # If executor worker thread is still running (e.g. noncooperative forward beyond join budget),
            # do NOT unload models or falsely claim stopped.
            if self.executor and self.executor.is_worker_alive():
                print("[mlx-server] Worker thread still active after join timeout; cleanup pending.", file=sys.stderr)
                return

            # Safe to unload models ONLY AFTER the execution owner thread has completely exited
            if self.model_manager:
                try:
                    self.model_manager.unload_all()
                except Exception:
                    pass
            elif self.runtime:
                try:
                    if hasattr(self.runtime, "adapter") and not (self.executor and self.executor.is_worker_alive()):
                        self.runtime.adapter.unload()
                except Exception:
                    pass

            self._stopped_event.set()

    def serve_forever(self, poll_interval: float = 0.2):
        """
        Lifecycle-safe managed request dispatching loop.
        Uses a short poll timeout (handle_request) and repeatedly inspects _stop_event,
        eliminating races between stop() and BaseServer.shutdown() / serve_forever() handshakes.
        """
        self.timeout = poll_interval
        self._serving_event.set()
        try:
            while not self._stop_event.is_set():
                try:
                    self.handle_request()
                except (KeyboardInterrupt, OSError, ValueError):
                    break
        finally:
            self._serving_event.clear()

    def shutdown(self):
        self.stop()

    def server_close(self):
        self.stop()
        self._close_all_active_sockets()
        try:
            super().server_close()
        except Exception:
            pass

_BoundedDeadlineRfile = BoundedDeadlineReader


class MLXControlServer(_SocketAdmissionMixin, socketserver.ThreadingMixIn, http.server.HTTPServer):
    """
    Dedicated loopback HTTP control listener providing isolated control-plane capacity.
    Operates on a separate port from the inference listener so health checks and diagnostics
    never block or time out during inference socket saturation.
    """
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        mlx_server: ThreadedMLXServer,
        max_threads: int = 16,
        request_timeout_s: float = 5.0,
        max_header_bytes: int = 16384,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.mlx_server = mlx_server
        self.bind_host: str = server_address[0]
        self.request_timeout_s = request_timeout_s
        self.max_header_bytes = max_header_bytes
        self._init_socket_admission(max_threads)
        self._serving_event = threading.Event()
        self._stop_event = threading.Event()
        self._stopped_event = threading.Event()
        self._stop_lock = threading.Lock()

    def stop(self):
        with self._stop_lock:
            self._stop_event.set()
            self._close_all_active_sockets()
            self._stopped_event.set()

    def shutdown(self):
        self.stop()

    def server_close(self):
        self.stop()
        self._close_all_active_sockets()
        try:
            super().server_close()
        except Exception:
            pass

    def serve_forever(self, poll_interval: float = 0.2):
        self.timeout = poll_interval
        self._serving_event.set()
        try:
            while not self._stop_event.is_set():
                try:
                    self.handle_request()
                except (KeyboardInterrupt, OSError, ValueError):
                    break
        finally:
            self._serving_event.clear()

class MLXControlRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Request handler for the dedicated loopback control listener.
    Serves /health, /ready, /descriptor, /memory, /stats.
    Rejects all POST / inference endpoints with 405 Method Not Allowed.
    Enforces absolute wall-clock request/header deadlines and byte bounds.
    Unconditionally closes connection after every response (no keepalive).
    """
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.close_connection = True
        self.rfile.close()
        deadline = time.monotonic() + self.control_server.request_timeout_s
        self.rfile = _BoundedDeadlineRfile(self.connection, deadline, self.control_server.max_header_bytes)  # type: ignore

    @property
    def control_server(self) -> MLXControlServer:
        return self.server  # type: ignore

    @property
    def mlx_server(self) -> ThreadedMLXServer:
        return self.control_server.mlx_server

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200, close_connection: bool = True):
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Unconditionally close control connection (no keepalive)
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _validate_origin_and_referer(self) -> bool:
        """Validates Host, Origin, and Referer headers to protect against DNS rebinding & CSRF."""
        allowed_hosts = {"127.0.0.1", "localhost", self.control_server.bind_host, ""}
        host = self.headers.get("Host", "").split(":")[0]
        if host not in allowed_hosts:
            return False

        origin = self.headers.get("Origin")
        if origin:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            orig_host = (parsed.hostname or "").lower()
            if orig_host not in allowed_hosts:
                return False

        referer = self.headers.get("Referer")
        if referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            ref_host = (parsed.hostname or "").lower()
            if ref_host not in allowed_hosts:
                return False

        return True

    def do_GET(self):
        if not self._validate_origin_and_referer():
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403, close_connection=True)
            return

        path = self.path.split("?")[0].rstrip("/")
        srv = self.mlx_server

        if path == "/health":
            self._send_json(get_health_payload(srv), 200)
        elif path == "/ready":
            has_any_stage = bool(
                srv.runtime is not None
                or srv.rerank_adapter is not None
                or srv.generate_adapter is not None
            )
            is_ready = bool(
                srv.state == ServerState.READY
                and has_any_stage
                and srv.executor is not None
                and srv.executor.is_alive()
            )
            if is_ready:
                self._send_json({"ready": True}, 200)
            else:
                self._send_json({"ready": False, "state": srv.state, "error": srv.state_error}, 503)
        elif path == "/descriptor":
            if srv.state != ServerState.READY:
                self._send_json({"error": "Server not ready"}, 503)
            elif srv.runtime:
                d = srv.runtime.get_descriptor()
                if srv.rerank_adapter:
                    d["rerank"] = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.rerank_adapter:
                d = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.generate_adapter:
                d = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            else:
                self._send_json({"error": "No model stages loaded"}, 503)
        elif path == "/memory":
            if srv.runtime:
                self._send_json(srv.runtime.get_memory_info(), 200)
            else:
                active_mb = 0.0
                try:
                    import mlx.core as mx
                    if hasattr(mx, "get_active_memory"):
                        active_mb = mx.get_active_memory() / (1024 * 1024)
                    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                        active_mb = mx.metal.get_active_memory() / (1024 * 1024)
                except Exception:
                    pass
                model_mb = 0.0
                if srv.rerank_adapter and hasattr(srv.rerank_adapter, "model_memory_mb"):
                    model_mb += srv.rerank_adapter.model_memory_mb
                if srv.generate_adapter and hasattr(srv.generate_adapter, "model_memory_mb"):
                    model_mb += srv.generate_adapter.model_memory_mb
                self._send_json({
                    "active_mb": round(active_mb, 1),
                    "peak_mb": round(max(active_mb, model_mb), 1),
                    "model_mb": round(model_mb, 1),
                }, 200)
        elif path == "/stats":
            if srv.runtime:
                self._send_json(srv.runtime.get_stats_info(), 200)
            elif srv.rerank_adapter:
                avg_ms = round(srv.rerank_adapter.total_latency_ms / max(1, srv.rerank_adapter.total_requests), 2)
                self._send_json({
                    "total_requests": srv.rerank_adapter.total_requests,
                    "total_pairs_scored": srv.rerank_adapter.total_pairs_scored,
                    "avg_ms": avg_ms,
                    "stage": "rerank",
                }, 200)
            elif srv.generate_adapter:
                self._send_json(srv.generate_adapter.get_stats_info(), 200)
            else:
                self._send_json({"total_requests": 0, "avg_ms": 0.0, "compiled_shapes": 0, "uptime_sec": 0}, 200)
        else:
            self._send_json({"error": f"Not found: {self.path}"}, 404)

    def do_POST(self):
        # Dedicated control listener NEVER serves inference requests
        self._send_json(
            {
                "error": "Inference requests are disabled on control port; use dedicated inference port",
                "type": "method_not_allowed",
            },
            405,
            close_connection=True,
        )

    def handle_one_request(self):
        self.request_version = "HTTP/1.1"
        self.requestline = ""
        self.command = ""
        try:
            super().handle_one_request()
        except MLXServerError as exc:
            self.close_connection = True
            try:
                self._send_json({"error": str(exc), "type": exc.error_type}, exc.status_code)
            except OSError:
                pass
        except (ConnectionError, OSError):
            self.close_connection = True


class MLXHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """
    Request handler for the primary inference HTTP server.
    Handles inference endpoints (/embed, /embed-bin, /tokenize, /rerank, /generate)
    as well as diagnostic GET endpoints for backward compatibility.
    """
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.rfile.close()
        self.rfile = _BoundedDeadlineRfile(self.connection, time.monotonic() + self.mlx_server.header_timeout_s, self.mlx_server.max_header_bytes)  # type: ignore
        self._request_cancel_event = None

    @property
    def mlx_server(self) -> ThreadedMLXServer:
        return self.server  # type: ignore

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data: dict, status: int = 200, close_connection: bool = False):
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if close_connection or self.close_connection:
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, data: bytes, status: int = 200, close_connection: bool = False, descriptor: Optional[dict] = None):
        descriptor_header = quote(json.dumps(descriptor, allow_nan=False), safe="") if descriptor is not None else None
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        if descriptor_header is not None:
            self.send_header("X-QMD-Embedding-Descriptor", descriptor_header)
        if close_connection or self.close_connection:
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(data)

    def _validate_origin_and_referer(self) -> bool:
        """Validates Host, Origin, and Referer headers to protect against DNS rebinding & CSRF."""
        allowed_hosts = {"127.0.0.1", "localhost", self.mlx_server.bind_host, ""}
        host = self.headers.get("Host", "").split(":")[0]
        if host not in allowed_hosts:
            return False

        origin = self.headers.get("Origin")
        if origin:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            orig_host = (parsed.hostname or "").lower()
            if orig_host not in allowed_hosts:
                return False

        referer = self.headers.get("Referer")
        if referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            ref_host = (parsed.hostname or "").lower()
            if ref_host not in allowed_hosts:
                return False

        return True

    @contextmanager
    def _request_cancellation(self, deadline: float):
        """Observe unambiguous peer aborts without consuming pipelined input.

        TCP FIN only closes the peer's write direction: it is indistinguishable
        from a valid HTTP client using shutdown(SHUT_WR) while awaiting a reply.
        Do not cancel on FIN alone. RST/socket errors, server stop and the
        absolute request deadline cancel cooperatively; GPU ownership stays with
        the executor until any already-running forward actually returns.
        """
        event = threading.Event()
        done = threading.Event()
        self._request_cancel_event = event

        def watch():
            write_half_closed = False
            while not done.is_set():
                remaining = deadline - time.monotonic()
                if self.mlx_server._stop_event.is_set() or remaining <= 0:
                    event.set()
                    return
                if event.is_set():
                    return
                interval = min(0.05, remaining)
                try:
                    if self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                        event.set()
                        return
                    if write_half_closed:
                        done.wait(interval)
                        continue
                    # Poll readiness without a blocking select: done.wait below
                    # wakes immediately on completion, so successful keepalive
                    # requests do not pay a polling-interval join penalty.
                    readable, _, exceptional = select.select([self.connection], [], [self.connection], 0)
                    if done.is_set():
                        return
                    if exceptional:
                        event.set()
                        return
                    if readable:
                        data = self.connection.recv(1, socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0))
                        if not data:
                            write_half_closed = True
                    # FIN and pipelined bytes stay readable: avoid a busy loop.
                    done.wait(interval)
                except BlockingIOError:
                    continue
                except (OSError, ValueError):
                    event.set()
                    return

        monitor = threading.Thread(target=watch, daemon=True, name=f"MLX-Disconnect-{uuid.uuid4().hex[:8]}")
        try:
            monitor.start()
            yield event
        except (ConnectionError, OSError):
            event.set()
            raise
        finally:
            done.set()
            if monitor.ident is not None:
                monitor.join(timeout=0.25)
            self._request_cancel_event = None

    def _handle_exception(self, exc: Exception):
        """Maps typed exceptions to appropriate HTTP error status codes."""
        if isinstance(exc, MLXServerError):
            self._send_json({"error": str(exc), "type": getattr(exc, "error_type", "error")}, exc.status_code)
        elif isinstance(exc, TimeoutError):
            self._send_json({"error": f"Deadline exceeded: {exc}", "type": "deadline_exceeded"}, 504)
        else:
            self._send_json({"error": f"Internal server error: {exc}", "type": "server_error"}, 500)

    def do_GET(self):
        if not self._validate_origin_and_referer():
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403, close_connection=True)
            return

        path = self.path.split("?")[0].rstrip("/")
        srv = self.mlx_server

        if path == "/health":
            self._send_json(get_health_payload(srv), 200)
        elif path == "/ready":
            has_any_stage = bool(
                srv.runtime is not None
                or srv.rerank_adapter is not None
                or srv.generate_adapter is not None
            )
            is_ready = bool(
                srv.state == ServerState.READY
                and has_any_stage
                and srv.executor is not None
                and srv.executor.is_alive()
            )
            if is_ready:
                self._send_json({"ready": True}, 200)
            else:
                self._send_json({"ready": False, "state": srv.state, "error": srv.state_error}, 503)
        elif path == "/descriptor":
            if srv.state != ServerState.READY:
                self._send_json({"error": "Server not ready"}, 503)
            elif srv.runtime:
                d = srv.runtime.get_descriptor()
                if srv.rerank_adapter:
                    d["rerank"] = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.rerank_adapter:
                d = srv.rerank_adapter.get_descriptor()
                if srv.generate_adapter:
                    d["generate"] = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            elif srv.generate_adapter:
                d = srv.generate_adapter.get_descriptor()
                self._send_json(d, 200)
            else:
                self._send_json({"error": "No model stages loaded"}, 503)
        elif path == "/memory":
            if srv.runtime:
                self._send_json(srv.runtime.get_memory_info(), 200)
            else:
                active_mb = 0.0
                try:
                    import mlx.core as mx
                    if hasattr(mx, "get_active_memory"):
                        active_mb = mx.get_active_memory() / (1024 * 1024)
                    elif hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                        active_mb = mx.metal.get_active_memory() / (1024 * 1024)
                except Exception:
                    pass
                model_mb = 0.0
                if srv.rerank_adapter and hasattr(srv.rerank_adapter, "model_memory_mb"):
                    model_mb += srv.rerank_adapter.model_memory_mb
                if srv.generate_adapter and hasattr(srv.generate_adapter, "model_memory_mb"):
                    model_mb += srv.generate_adapter.model_memory_mb
                self._send_json({
                    "active_mb": round(active_mb, 1),
                    "peak_mb": round(max(active_mb, model_mb), 1),
                    "model_mb": round(model_mb, 1),
                }, 200)
        elif path == "/stats":
            if srv.runtime:
                self._send_json(srv.runtime.get_stats_info(), 200)
            elif srv.rerank_adapter:
                avg_ms = round(srv.rerank_adapter.total_latency_ms / max(1, srv.rerank_adapter.total_requests), 2)
                self._send_json({
                    "total_requests": srv.rerank_adapter.total_requests,
                    "total_pairs_scored": srv.rerank_adapter.total_pairs_scored,
                    "avg_ms": avg_ms,
                    "stage": "rerank",
                }, 200)
            elif srv.generate_adapter:
                self._send_json(srv.generate_adapter.get_stats_info(), 200)
            else:
                self._send_json({"total_requests": 0, "avg_ms": 0.0, "compiled_shapes": 0, "uptime_sec": 0}, 200)
        else:
            self._send_json({"error": f"Not found: {self.path}"}, 404)

    def do_POST(self):
        if not self._validate_origin_and_referer():
            self._send_json({"error": "Forbidden: Invalid Host, Origin, or Referer header"}, 403, close_connection=True)
            return

        srv = self.mlx_server
        has_any_stage = bool(
            srv.runtime is not None
            or srv.rerank_adapter is not None
            or srv.generate_adapter is not None
        )
        if srv.state != ServerState.READY or not has_any_stage:
            self._send_json({"error": f"Server not ready (current state: {srv.state})"}, 503, close_connection=True)
            return

        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/embed", "/embed-bin", "/tokenize", "/rerank", "/generate"):
            self._send_json({"error": f"Not found: {self.path}"}, 404, close_connection=True)
            return

        # Dedicated control capacity: limit concurrent inference requests so control plane remains reachable
        if not srv.inference_limiter.acquire(blocking=False):
            # Must send Connection: close when rejecting before reading body to prevent HTTP protocol desync
            self._send_json(
                {
                    "error": "Inference capacity saturated: maximum concurrent inference requests reached",
                    "type": "overloaded",
                },
                429,
                close_connection=True,
            )
            return

        try:
            self._handle_post_inference(path)
        finally:
            try:
                srv.inference_limiter.release()
            except ValueError:
                pass

    def _handle_post_inference(self, path: str):
        srv = self.mlx_server
        deadline = None
        try:
            length = parse_http_body_headers(self.headers, srv.max_body_bytes)
            default_timeout = 300.0 if path in ("/embed", "/embed-bin") else 120.0
            timeout_header = self.headers.get("X-Request-Timeout")
            header_timeout = parse_timeout(header_val=timeout_header, default_timeout=default_timeout)
            # A known header budget applies BEFORE any upload read. A timeout
            # supplied only inside JSON is applied retroactively to the same
            # request start once decoded; it never restarts after upload.
            upload_deadline = min(
                self._request_started_at + header_timeout,
                time.monotonic() + srv.body_timeout_s,
            )
            self.rfile.set_limits(upload_deadline, length)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ProtocolError("Incomplete request body read")
            payload = decode_json_request(body)
            request_timeout = parse_timeout(payload, timeout_header, default_timeout=default_timeout)
            deadline = self._request_started_at + request_timeout
            if time.monotonic() >= deadline:
                raise DeadlineExceededError("Request deadline exceeded during upload")
            with self._request_cancellation(deadline) as cancel_event:
                self._dispatch_inference(path, payload, request_timeout, deadline, cancel_event)
        except MLXServerError as exc:
            # Validation/read failures must close rather than reinterpret an
            # unread body as the next keepalive request.
            self.close_connection = True
            if isinstance(exc, RequestCancelledError) and deadline is not None and time.monotonic() >= deadline:
                exc = DeadlineExceededError("Request deadline exceeded")
            self._handle_exception(exc)
        except TimeoutError as exc:
            self.close_connection = True
            self._handle_exception(exc)
        except OSError:
            self.close_connection = True
        except Exception as exc:
            self.close_connection = True
            self._handle_exception(exc)

    def _check_request_active(self, deadline: float, cancel_event: threading.Event):
        if time.monotonic() >= deadline:
            raise DeadlineExceededError("Request deadline exceeded")
        if cancel_event.is_set():
            raise RequestCancelledError("Request cancelled")

    def _dispatch_inference(self, path: str, payload: dict, request_timeout: float, deadline: float, cancel_event: threading.Event):
        srv = self.mlx_server
        self._check_request_active(deadline, cancel_event)
        if path == "/tokenize":
            if srv.runtime is None:
                self._send_json({"error": "Embedding / tokenize adapter not configured on this server"}, 501)
                return
            texts = payload.get("texts", [])
            if not isinstance(texts, list):
                raise InvalidInputError("'texts' must be a list of strings")
            if len(texts) > 512:
                raise InvalidInputError(f"'texts' exceeds per-request limit of 512 (got {len(texts)})")
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    raise InvalidInputError(f"texts[{i}] is not a string")
                if len(text) > 256 * 1024:
                    raise InvalidInputError(f"texts[{i}] exceeds 256KB")
            tokens = srv.runtime.tokenize(texts, deadline=deadline, cancel_event=cancel_event)
            self._check_request_active(deadline, cancel_event)
            self._send_json({"tokens": tokens, "counts": [len(t) for t in tokens]}, 200)
            return
        if path == "/rerank":
            if srv.rerank_adapter is None:
                self._send_json({"error": "Rerank adapter not configured on this server"}, 501)
                return
            query, documents = validate_rerank_request(payload)
            scores = srv.rerank_adapter.score_pairs(
                query, documents, timeout_s=request_timeout,
                cancel_event=cancel_event, deadline=deadline,
            )
            self._check_request_active(deadline, cancel_event)
            result = {"scores": scores, "model": srv.rerank_adapter.model_name, "count": len(scores)}
            if hasattr(srv.rerank_adapter, "get_descriptor"):
                descriptor = srv.rerank_adapter.get_descriptor()
                if isinstance(descriptor, dict):
                    result["descriptor"] = descriptor
                    if isinstance(descriptor.get("revision"), str):
                        result["revision"] = descriptor["revision"]
            self._check_request_active(deadline, cancel_event)
            self._send_json(result, 200)
            return
        if path == "/generate":
            if srv.generate_adapter is None:
                self._send_json({"error": "Generate adapter not configured on this server"}, 501)
                return
            prompt, max_tokens, temperature = validate_generate_request(payload)
            text = srv.generate_adapter.submit_generate(
                prompt, max_tokens=max_tokens, temperature=temperature,
                timeout=request_timeout, cancel_event=cancel_event, deadline=deadline,
            )
            self._check_request_active(deadline, cancel_event)
            result = {"text": text, "model": srv.generate_adapter.model_name}
            if hasattr(srv.generate_adapter, "get_descriptor"):
                descriptor = srv.generate_adapter.get_descriptor()
                if isinstance(descriptor, dict):
                    result["descriptor"] = descriptor
                    if isinstance(descriptor.get("revision"), str):
                        result["revision"] = descriptor["revision"]
            self._check_request_active(deadline, cancel_event)
            self._send_json(result, 200)
            return
        if srv.runtime is None:
            self._send_json({"error": "Embedding adapter not configured on this server"}, 501)
            return
        texts, requested_dims, is_query = validate_embed_request(payload)
        operation = srv.runtime.submit_embed_result(
            texts=texts, requested_dims=requested_dims, is_query=is_query,
            timeout=request_timeout, cancel_event=cancel_event, deadline=deadline,
        )
        self._check_request_active(deadline, cancel_event)
        if not isinstance(operation, EmbeddingResult):
            raise ProtocolError("Embedding runtime did not return an operation-bound result")
        embeddings = operation.embeddings
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] <= 0:
            raise ProtocolError("Embedding result shape does not match request")
        descriptor = operation.descriptor
        if (not isinstance(descriptor, dict)
                or type(descriptor.get("outputDimensions")) is not int
                or descriptor["outputDimensions"] != embeddings.shape[1]):
            raise ProtocolError("Embedding descriptor does not match output dimensions")
        if path == "/embed-bin":
            count, dims = embeddings.shape
            self._send_binary(encode_binary_embeddings(embeddings, count, dims), 200, descriptor=descriptor)
        else:
            if not np.all(np.isfinite(embeddings)):
                raise ProtocolError("Embeddings contain NaN or Infinite values")
            self._send_json({
                "embeddings": embeddings.tolist(), "model": descriptor["model"],
                "dims": embeddings.shape[1], "descriptor": descriptor,
            }, 200)

    def handle_one_request(self):
        self._request_started_at = time.monotonic()
        self.request_version = "HTTP/1.1"
        self.requestline = ""
        self.command = ""
        self.rfile.set_limits(self._request_started_at + self.mlx_server.header_timeout_s, self.mlx_server.max_header_bytes)
        try:
            super().handle_one_request()
        except MLXServerError as exc:
            self.close_connection = True
            try:
                self._handle_exception(exc)
            except OSError:
                pass
        except OSError:
            self.close_connection = True
            if self._request_cancel_event is not None:
                self._request_cancel_event.set()


def start_server(
    model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    port: int = 8787,
    bind_host: str = "127.0.0.1",
    control_port: Optional[int] = None,
    quantization: Optional[str] = None,
    dtype_str: str = "float32",
    max_length: int = 2048,
    preload: bool = True,
    warmup: bool = True,
    max_batch_tokens: int = 0,
    rerank_model: Optional[str] = None,
    generate_model: Optional[str] = None,
    max_threads: int = 128,
    max_concurrent_inference: int = 32,
    control_max_threads: int = 16,
    control_request_timeout_s: float = 5.0,
    control_max_header_bytes: int = 16384,
    _before_serve_hook: Optional[Any] = None,
    max_bulk_microbatch_tokens: int = 512,
) -> tuple[ThreadedMLXServer, threading.Thread]:
    """
    Initializes unified executor, residency manager, adapters, and starts both:
    1. Primary inference HTTP server on (bind_host, port).
    2. Dedicated loopback control HTTP server on (127.0.0.1, control_port).
    Serves HTTP /health immediately in STARTING/LOADING state on both listeners
    so health probes do not hang or fail during cold model init.
    """
    if isinstance(max_bulk_microbatch_tokens, bool) or not isinstance(max_bulk_microbatch_tokens, int) or max_bulk_microbatch_tokens <= 0:
        raise InvalidInputError("max_bulk_microbatch_tokens must be a positive integer")
    server = ThreadedMLXServer(
        (bind_host, port),
        MLXHTTPRequestHandler,
        max_threads=max_threads,
        max_concurrent_inference=max_concurrent_inference,
    )
    server._before_serve_hook = _before_serve_hook
    server.state = ServerState.STARTING

    # Determine control port
    if control_port is None:
        effective_control_port = 0 if port == 0 else (port + 1)
    else:
        effective_control_port = control_port

    try:
        control_server = MLXControlServer(
            ("127.0.0.1", effective_control_port),
            MLXControlRequestHandler,
            mlx_server=server,
            max_threads=control_max_threads,
            request_timeout_s=control_request_timeout_s,
            max_header_bytes=control_max_header_bytes,
        )
    except BaseException:
        server.server_close()
        raise
    server.control_server = control_server
    server.control_port = control_server.server_address[1]

    # Dedicated serving threads for both listeners
    t_http = threading.Thread(target=server.serve_forever, daemon=True, name="MLX-HTTP-Serve")
    t_ctrl = threading.Thread(target=control_server.serve_forever, daemon=True, name="MLX-Control-Serve")
    server.t_http = t_http
    server.t_ctrl = t_ctrl
    try:
        t_http.start()
        t_ctrl.start()
    except BaseException:
        server.server_close()
        if t_http.ident is not None:
            t_http.join(timeout=1.0)
        raise

    def _init_and_coordinate():
        try:
            with server._lifecycle_lock:
                if server._stop_event.is_set():
                    return

                executor = GPUExecutor()
                server.executor = executor

                model_manager = ModelResidencyManager(executor)
                server.model_manager = model_manager

                server.state = ServerState.LOADING if preload else ServerState.READY

            has_embed = bool(model_name and model_name.strip().lower() not in ("", "none", "null", "false"))
            has_rerank = bool(rerank_model and rerank_model.strip().lower() not in ("", "none", "null", "false"))
            has_generate = bool(generate_model and generate_model.strip().lower() not in ("", "none", "null", "false"))

            if not (has_embed or has_rerank or has_generate):
                raise ValueError("No model configured (specify model_name, rerank_model, or generate_model)")

            if has_embed:
                runtime = MLXEmbeddingRuntime(
                    model_name=model_name,
                    quantization=quantization,
                    dtype_str=dtype_str,
                    max_length=max_length,
                    max_batch_tokens=max_batch_tokens,
                    max_bulk_microbatch_tokens=max_bulk_microbatch_tokens,
                    executor=executor,
                    model_manager=model_manager,
                    lazy_load=not preload,
                )

                with server._lifecycle_lock:
                    if server._stop_event.is_set():
                        return
                    server.runtime = runtime

                if preload and warmup:
                    if server._stop_event.is_set():
                        return
                    runtime.warmup()

            if has_rerank:
                if server._stop_event.is_set():
                    return
                from .rerank import MLXRerankAdapter
                rerank_adapter = MLXRerankAdapter(
                    model_name=rerank_model,
                    max_length=2048,
                    lazy_load=not preload,
                    executor=executor,
                    model_manager=model_manager,
                )
                model_manager.register_adapter("rerank", rerank_adapter)
                if preload:
                    model_manager.ensure_loaded("rerank")
                    if warmup:
                        rerank_adapter.warmup()
                with server._lifecycle_lock:
                    if server._stop_event.is_set():
                        return
                    server.rerank_adapter = rerank_adapter
                print(f"[mlx-server] Rerank adapter ready: {rerank_model}")

            if has_generate:
                if server._stop_event.is_set():
                    return
                from .generate import MLXGenerateAdapter
                generate_adapter = MLXGenerateAdapter(
                    model_name=generate_model,
                    lazy_load=not preload,
                    executor=executor,
                    model_manager=model_manager,
                )
                model_manager.register_adapter("generate", generate_adapter)
                if preload:
                    model_manager.ensure_loaded("generate")
                    if warmup:
                        generate_adapter.warmup()
                with server._lifecycle_lock:
                    if server._stop_event.is_set():
                        return
                    server.generate_adapter = generate_adapter
                print(f"[mlx-server] Generate adapter ready: {generate_model}")

            with server._lifecycle_lock:
                if server._stop_event.is_set():
                    return
                server.state = ServerState.READY
                primary = model_name if has_embed else (rerank_model if has_rerank else generate_model)
                print(f"[mlx-server] Server ready (primary='{primary}') on http://{bind_host}:{port} (control: http://127.0.0.1:{server.control_port})")

        except Exception as e:
            with server._lifecycle_lock:
                if not server._stop_event.is_set():
                    server.state = ServerState.FAILED
                    server.state_error = str(e)
                    print(f"[mlx-server] Model loading failed: {e}", file=sys.stderr)

        if server._before_serve_hook:
            try:
                server._before_serve_hook()
            except Exception:
                pass

        # Keep coordinator alive until stopped
        while not server._stop_event.is_set():
            time.sleep(0.05)

        t_http.join(timeout=2.0)
        t_ctrl.join(timeout=2.0)
        server.stop()

    t = threading.Thread(target=_init_and_coordinate, daemon=True, name="MLX-Init-Coordinator")
    try:
        t.start()
    except BaseException:
        server.server_close()
        t_http.join(timeout=1.0)
        t_ctrl.join(timeout=1.0)
        raise
    return server, t
