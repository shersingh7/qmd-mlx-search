"""Offline loopback regressions for HTTP ownership, bounds and serving identity."""

import contextlib
import http.client
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import unquote
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.qmd_mlx.protocol import (
    InvalidInputError,
    ProtocolError,
    RequestCancelledError,
    decode_binary_embeddings,
    encode_binary_embeddings,
    parse_timeout,
    validate_embed_request,
    validate_generate_request,
)
from scripts.qmd_mlx.server import (
    MLXControlRequestHandler,
    MLXControlServer,
    MLXHTTPRequestHandler,
    ServerState,
    ThreadedMLXServer,
    start_server,
)
from scripts.qmd_mlx.runtime import EmbeddingResult


def wait_until(predicate, timeout=1.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class SyntheticRuntime:
    model_name = "synthetic/actual-model"
    native_dims = 4

    def __init__(self):
        self.calls = []
        self.descriptor_dims = []

    def get_descriptor(self, requested_dims=None):
        self.descriptor_dims.append(requested_dims)
        return {
            "version": 1,
            "backend": "mlx_fake",
            "model": self.model_name,
            "nativeDimensions": self.native_dims,
            "outputDimensions": min(requested_dims or self.native_dims, self.native_dims),
            "normalized": True,
            "maxSequenceLength": 64,
            "pooling": "mean",
            "synthetic": True,
        }

    def submit_embed(self, texts, requested_dims=None, **kwargs):
        self.calls.append(kwargs)
        dims = min(requested_dims or self.native_dims, self.native_dims)
        return np.ones((len(texts), dims), dtype=np.float32)

    def submit_embed_result(self, texts, requested_dims=None, **kwargs):
        descriptor_json = json.dumps(self.get_descriptor(requested_dims), allow_nan=False)
        return EmbeddingResult(self.submit_embed(texts, requested_dims=requested_dims, **kwargs), descriptor_json)

    def tokenize(self, texts, **kwargs):
        self.calls.append(kwargs)
        return [[1] for _ in texts]

    def stop_admission(self):
        pass


@contextlib.contextmanager
def listener(control=False, max_threads=1, runtime=None):
    main = ThreadedMLXServer(
        ("127.0.0.1", 0), MLXHTTPRequestHandler,
        max_threads=max_threads, max_concurrent_inference=1,
    )
    main.runtime = runtime or SyntheticRuntime()
    main.state = ServerState.READY
    if control:
        server = MLXControlServer(
            ("127.0.0.1", 0), MLXControlRequestHandler, main,
            max_threads=max_threads, request_timeout_s=1.0,
        )
    else:
        server = main
    # Test fixtures never own a model/backend, even on hosts with MLX installed.
    server.handle_error = lambda *args: None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, main
    finally:
        server.server_close()
        if server is not main:
            main.server_close()
        thread.join(1.0)
        assert not thread.is_alive()


def receive_all(client):
    chunks = []
    while True:
        try:
            chunk = client.recv(65536)
        except ConnectionResetError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def raw_post(body, path="/embed", headers=b"", close=True):
    return (
        f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + (b"Connection: close\r\n" if close else b"")
        + headers + b"\r\n" + body
    )


def request(server, path, payload):
    conn = http.client.HTTPConnection(*server.server_address, timeout=1.0)
    try:
        conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json", "Connection": "close"})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.mark.parametrize("control", [False, True])
def test_overload_deregisters_only_rejected_sockets_without_releasing_live_permits(control):
    with listener(control=control) as (server, _):
        with socket.create_connection(server.server_address, timeout=1.0) as held:
            held.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Hold: ")
            assert wait_until(lambda: server.thread_limiter._value == 0)
            for _ in range(12):
                with socket.create_connection(server.server_address, timeout=1.0) as rejected:
                    response = receive_all(rejected)
                assert response.startswith(b"HTTP/1.1 429")
                head, body = response.split(b"\r\n\r\n", 1)
                assert f"Content-Length: {len(body)}".encode() in head
                assert json.loads(body)["type"] == "overloaded"
                assert wait_until(lambda: len(server._active_sockets) == 1)
                assert server.thread_limiter._value == 0
            # An unrelated/double close must not release somebody else's permit.
            server.close_request(MagicMock())
            assert server.thread_limiter._value == 0
        assert wait_until(lambda: not server._active_sockets)
        assert wait_until(lambda: server.thread_limiter._value == 1)


@pytest.mark.parametrize("control", [False, True])
def test_stop_interrupts_blocked_header_reads_with_peer_eof_and_restored_capacity(control):
    with listener(control=control) as (server, _):
        with socket.create_connection(server.server_address, timeout=0.5) as held:
            held.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Hold: ")
            assert wait_until(lambda: server.thread_limiter._value == 0)
            server.stop()
            # A timeout is NOT acceptance of shutdown: the peer must see EOF/reset.
            assert receive_all(held) == b""
            assert not server._active_sockets
            assert wait_until(lambda: server.thread_limiter._value == 1, timeout=0.5)


@pytest.mark.parametrize("iteration", range(8))
def test_stop_interrupts_incomplete_upload_and_restores_inference_permit(iteration):
    with listener() as (server, _):
        with socket.create_connection(server.server_address, timeout=0.5) as held:
            held.sendall(raw_post(b"{" + b" " * 50)[:-50])
            assert wait_until(lambda: server.inference_limiter._value == 0)
            server.stop()
            assert receive_all(held) == b""
            if not wait_until(lambda: server.inference_limiter._value == 1, timeout=0.5):
                import traceback
                stacks = {thread.name: "".join(traceback.format_stack(sys._current_frames()[thread.ident]))
                          for thread in threading.enumerate()
                          if thread is not threading.current_thread() and thread.ident in sys._current_frames()}
                pytest.fail(f"Upload handler did not release admission: {stacks}")
            assert wait_until(lambda: server.thread_limiter._value == 1, timeout=0.5)


def test_thread_start_failure_releases_admitted_socket_exactly_once():
    with listener() as (server, _):
        # A socketpair avoids racing an actual accept.
        left, right = socket.socketpair()
        try:
            with server._sockets_lock:
                server._active_sockets.add(left)
            with patch("threading.Thread.start", side_effect=RuntimeError("synthetic start failure")):
                with pytest.raises(RuntimeError, match="synthetic start failure"):
                    server.process_request(left, ("127.0.0.1", 1))
            assert left.fileno() == -1
            assert left not in server._active_sockets
            assert wait_until(lambda: server.thread_limiter._value == 1)
            server.close_request(left)
            assert server.thread_limiter._value == 1
        finally:
            left.close()
            right.close()


@pytest.mark.parametrize("control", [False, True])
def test_absolute_header_deadline_ends_regular_slow_drip(control):
    with listener(control=control) as (server, _):
        server.request_timeout_s = 0.16
        server.header_timeout_s = 0.16
        with socket.create_connection(server.server_address, timeout=0.5) as client:
            client.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Hold: ")
            started = time.monotonic()
            for _ in range(12):
                time.sleep(0.025)
                try:
                    client.sendall(b"x")
                except OSError:
                    break
            response = receive_all(client)
            assert not response or b"408" in response.split(b"\r\n", 1)[0]
            assert time.monotonic() - started < 0.6
        assert wait_until(lambda: not server._active_sockets)
        assert wait_until(lambda: server.thread_limiter._value == 1)


@pytest.mark.parametrize("timeout_header", [b"X-Request-Timeout: 0.16\r\n", b""])
def test_absolute_upload_deadline_precedes_inference_even_with_regular_drip(timeout_header):
    with listener() as (server, _):
        server.body_timeout_s = 0.16
        body = json.dumps({"texts": ["synthetic-upload"]}).encode()
        with socket.create_connection(server.server_address, timeout=0.5) as client:
            wire = raw_post(body, headers=timeout_header)
            header, _ = wire.split(b"\r\n\r\n", 1)
            client.sendall(header + b"\r\n\r\n")
            for byte in body:
                try:
                    client.sendall(bytes([byte]))
                except OSError:
                    break
                time.sleep(0.015)
            response = receive_all(client)
            assert b"408" in response.split(b"\r\n", 1)[0]
            assert json.loads(response.split(b"\r\n\r\n", 1)[1])["type"] == "request_timeout"
        assert not server.runtime.calls
        assert wait_until(lambda: server.inference_limiter._value == 1)
        assert wait_until(lambda: server.thread_limiter._value == 1)


def test_body_timeout_cannot_restart_deadline_after_upload():
    with listener() as (server, _):
        body = b'{"texts":["synthetic"],"timeout":0.1}'
        with socket.create_connection(server.server_address, timeout=0.5) as client:
            wire = raw_post(body)
            client.sendall(wire[:-1])
            time.sleep(0.15)
            client.sendall(wire[-1:])
            response = receive_all(client)
            assert b"504" in response.split(b"\r\n", 1)[0]
            assert json.loads(response.split(b"\r\n\r\n", 1)[1])["type"] == "deadline_exceeded"
        assert not server.runtime.calls


def test_inference_receives_original_absolute_deadline_after_upload():
    with listener() as (server, _):
        body = b'{"texts":["synthetic"],"timeout":0.4}'
        started = time.monotonic()
        with socket.create_connection(server.server_address, timeout=1.0) as client:
            wire = raw_post(body)
            client.sendall(wire[:-1])
            time.sleep(0.15)
            client.sendall(wire[-1:])
            assert b"200" in receive_all(client).split(b"\r\n", 1)[0]
        assert server.runtime.calls[0]["deadline"] - started < 0.48


@pytest.mark.parametrize("path,payload", [
    ("/embed", {"texts": ["synthetic"]}),
    ("/tokenize", {"texts": ["synthetic"]}),
    ("/rerank", {"query": "synthetic", "documents": ["doc"]}),
    ("/generate", {"prompt": "synthetic", "max_tokens": 1}),
])
def test_reset_disconnect_cancels_every_stage_and_joins_monitor(path, payload):
    entered = threading.Event()
    observed = {}

    def blocked(*args, **kwargs):
        observed["event"] = kwargs["cancel_event"]
        entered.set()
        if kwargs["cancel_event"].wait(0.7):
            raise RequestCancelledError("synthetic peer abort")
        raise AssertionError("peer abort did not propagate")

    with listener() as (server, _):
        server.runtime.submit_embed = blocked
        server.runtime.tokenize = blocked
        server.rerank_adapter = MagicMock(score_pairs=blocked)
        server.generate_adapter = MagicMock(submit_generate=blocked)
        client = socket.create_connection(server.server_address, timeout=1.0)
        try:
            client.sendall(raw_post(json.dumps(payload).encode(), path))
            assert entered.wait(1.0)
            # TCP RST unambiguously aborts both directions. A FIN alone does not.
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            client.close()
            assert observed["event"].wait(0.5)
            assert wait_until(lambda: server.inference_limiter._value == 1)
            assert wait_until(lambda: server.thread_limiter._value == 1)
            assert wait_until(lambda: not server._active_sockets)
            assert wait_until(lambda: not any(t.name.startswith("MLX-Disconnect-") for t in threading.enumerate()))
        finally:
            client.close()
            observed.get("event", threading.Event()).set()


@pytest.mark.parametrize("half_close", [False, True])
def test_normal_half_close_and_pipelining_do_not_cancel_or_consume_next_request(half_close):
    entered, release = threading.Event(), threading.Event()
    observed = {}
    runtime = SyntheticRuntime()
    original = runtime.submit_embed

    def blocked(*args, **kwargs):
        observed["event"] = kwargs["cancel_event"]
        entered.set()
        assert release.wait(0.8)
        return original(*args, **kwargs)

    runtime.submit_embed = blocked
    with listener(runtime=runtime) as (server, _):
        with socket.create_connection(server.server_address, timeout=1.0) as client:
            client.sendall(raw_post(b'{"texts":["synthetic"]}', close=half_close))
            assert entered.wait(1.0)
            if half_close:
                client.shutdown(socket.SHUT_WR)
            else:
                client.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            time.sleep(0.1)
            try:
                assert not observed["event"].is_set()
            finally:
                release.set()
            response = receive_all(client)
            assert response.count(b"HTTP/1.1 200 OK") == (1 if half_close else 2)
        assert wait_until(lambda: not any(t.name.startswith("MLX-Disconnect-") for t in threading.enumerate()))


@pytest.mark.parametrize("field,value", [
    ("dims", float("nan")), ("dims", float("inf")), ("dims", 1.5),
    ("dims", 2.0), ("dims", True), ("dims", "2"), ("dims", 2**40),
    ("is_query", "false"), ("is_query", 1),
    ("timeout", float("nan")), ("timeout", float("inf")),
    ("timeout", 0), ("timeout", -1), ("timeout", True), ("timeout", "1"),
])
def test_invalid_numeric_and_boolean_options_are_typed_4xx(field, value):
    with listener() as (server, _):
        status, headers, body = request(server, "/embed", {"texts": ["synthetic"], field: value})
        assert status == 400
        assert json.loads(body)["type"] == "invalid_input"
        assert not server.runtime.calls


@pytest.mark.parametrize("header", ["NaN", "Infinity", "-1", "0", "nonsense"])
def test_invalid_timeout_header_is_rejected_before_upload(header):
    with listener() as (server, _):
        with socket.create_connection(server.server_address, timeout=0.5) as client:
            wire = raw_post(b"{" * 10, headers=f"X-Request-Timeout: {header}\r\n".encode())
            client.sendall(wire[:-10])
            response = receive_all(client)
            assert b"400" in response.split(b"\r\n", 1)[0]
            assert json.loads(response.split(b"\r\n\r\n", 1)[1])["type"] == "invalid_input"
        assert not server.runtime.calls


@pytest.mark.parametrize("framing,status", [
    (b"Content-Length: +2\r\n", 400),
    (b"Content-Length: -2\r\n", 400),
    (b"Content-Length: 1_0\r\n", 400),
    (b"Content-Length: 2, 2\r\n", 400),
    (b"Content-Length: 2\r\nContent-Length: 2\r\n", 400),
    (b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n", 400),
    (b"Transfer-Encoding: chunked\r\n", 400),
    (b"Content-Length: 999999999999999999999999999\r\n", 413),
])
def test_malformed_framing_fails_closed_before_reading_body(framing, status):
    with listener() as (server, _):
        with socket.create_connection(server.server_address, timeout=0.5) as client:
            client.sendall(
                b"POST /embed HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                + framing + b"\r\n"
            )
            response = receive_all(client)
            assert f" {status} ".encode() in response.split(b"\r\n", 1)[0]
            assert json.loads(response.split(b"\r\n\r\n", 1)[1])["type"] == "invalid_protocol"
            assert b"Connection: close" in response
        assert not server.runtime.calls


@pytest.mark.parametrize("body", [b"\xff", b"{broken", b"[]", b'{"texts":["x"],"ignored":NaN}', b'{"texts":["x"],"timeout":1e999}'])
def test_malformed_json_returns_typed_4xx(body):
    with listener() as (server, _):
        with socket.create_connection(server.server_address, timeout=0.5) as client:
            client.sendall(raw_post(body))
            response = receive_all(client)
            assert b"400" in response.split(b"\r\n", 1)[0]
            assert json.loads(response.split(b"\r\n\r\n", 1)[1])["type"] == "invalid_input"
        assert not server.runtime.calls


@pytest.mark.parametrize("binary", [False, True])
def test_embedding_envelopes_use_actual_runtime_descriptor_and_output_dimensions(binary):
    runtime = SyntheticRuntime()
    runtime.model_name = "synthetic/実際\r\nNot-A-Header"
    with listener(runtime=runtime) as (server, _):
        status, headers, body = request(server, "/embed-bin" if binary else "/embed", {
            "texts": ["synthetic"], "dims": 2, "model": "untrusted-request-label",
        })
        assert status == 200
        assert runtime.descriptor_dims == [2]
        expected = runtime.get_descriptor(2)
        if binary:
            header = headers["X-QMD-Embedding-Descriptor"]
            assert header.isascii() and "\r" not in header and "\n" not in header
            assert json.loads(unquote(header)) == expected
            assert body == struct.pack("<ii", 1, 2) + np.ones((1, 2), dtype="<f4").tobytes()
        else:
            data = json.loads(body)
            assert data["descriptor"] == expected
            assert data["dims"] == 2
            assert data["model"] == runtime.model_name


@pytest.mark.parametrize("binary", [False, True])
def test_embedding_response_refuses_uncaptured_ndarray_instead_of_guessing_identity(binary, monkeypatch):
    runtime = SyntheticRuntime()
    monkeypatch.setattr(runtime, "submit_embed_result", lambda **kw: np.ones((1, 4), dtype=np.float32))
    monkeypatch.setattr(runtime, "get_descriptor", lambda *args, **kw: pytest.fail("response looked up mutable global identity"))
    with listener(runtime=runtime) as (server, _):
        status, _, body = request(server, "/embed-bin" if binary else "/embed", {"texts": ["synthetic"]})
        assert status == 400
        assert json.loads(body)["type"] == "invalid_protocol"


@pytest.mark.parametrize("binary", [False, True])
def test_fake_server_preserves_synthetic_identity_in_result_envelopes(binary):
    from scripts.qmd_mlx.fake_server import FakeMLXInferenceHandler, FakeMLXInferenceServer

    server = FakeMLXInferenceServer(("127.0.0.1", 0), FakeMLXInferenceHandler, "synthetic-token", "synthetic-only", 4)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, body = request(server, "/embed-bin" if binary else "/embed", {
            "texts": ["synthetic"], "dims": 2, "model": "spoofed-real-model",
        })
        assert status == 200
        descriptor = json.loads(unquote(headers["X-QMD-Embedding-Descriptor"])) if binary else json.loads(body)["descriptor"]
        assert descriptor["backend"] == "mlx_fake"
        assert descriptor["model"] == "synthetic-only"
        assert descriptor["synthetic"] is True
        assert descriptor["nativeDimensions"] == 4
        assert descriptor["outputDimensions"] == 2
        if binary:
            _, count, dims = decode_binary_embeddings(body)
            assert (count, dims) == (1, 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1.0)
        assert not thread.is_alive()


@pytest.mark.parametrize("count,dims,trailing", [(0, 4, b""), (2, 0, b""), (0, 0, b"extra")])
def test_empty_binary_frames_require_canonical_shape_and_exact_length(count, dims, trailing):
    with pytest.raises(ProtocolError):
        decode_binary_embeddings(struct.pack("<ii", count, dims) + trailing)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.5, 2.0, True, "2", 2**40])
def test_protocol_dims_fail_with_typed_error(value):
    with pytest.raises(InvalidInputError):
        validate_embed_request({"texts": ["synthetic"], "dims": value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1", 0, -1, 10**1000])
def test_protocol_timeout_requires_finite_positive_numeric_value(value):
    with pytest.raises(InvalidInputError):
        parse_timeout({"timeout": value})


def test_protocol_temperature_overflow_is_typed():
    with pytest.raises(InvalidInputError):
        validate_generate_request({"prompt": "synthetic", "temperature": 10**1000})


def test_start_server_wires_bulk_microbatch_option_to_runtime():
    runtime = MagicMock()
    with patch("scripts.qmd_mlx.server.MLXEmbeddingRuntime", return_value=runtime) as constructor:
        server, coordinator = start_server("synthetic", port=0, preload=False, warmup=False, max_bulk_microbatch_tokens=2048)
        try:
            assert wait_until(lambda: server.state in (ServerState.READY, ServerState.FAILED))
            assert server.state == ServerState.READY, server.state_error
            assert constructor.call_args.kwargs["max_bulk_microbatch_tokens"] == 2048
        finally:
            server.server_close()
            coordinator.join(1.0)
            assert not coordinator.is_alive()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("inf")])
def test_start_server_rejects_invalid_bulk_budget_before_binding(value):
    with patch("scripts.qmd_mlx.server.ThreadedMLXServer") as constructor:
        with pytest.raises((ValueError, InvalidInputError), match="max_bulk_microbatch_tokens"):
            start_server("synthetic", port=0, max_bulk_microbatch_tokens=value)
        constructor.assert_not_called()


def test_cli_help_exposes_bulk_budget_without_starting_models():
    result = subprocess.run(
        [sys.executable, "scripts/mlx_embed_server.py", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-bulk-microbatch-tokens" in result.stdout


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan"])
def test_cli_rejects_invalid_bulk_budget_without_starting_server(value):
    from scripts.mlx_embed_server import main

    with patch("sys.argv", ["mlx_embed_server.py", "--max-bulk-microbatch-tokens", value]), \
         patch("scripts.mlx_embed_server.start_server") as start:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        start.assert_not_called()


def test_cli_forwards_valid_bulk_budget():
    from scripts.mlx_embed_server import main

    server, thread = MagicMock(), MagicMock()
    server._stopped_event.is_set.return_value = True
    thread.is_alive.return_value = False
    with patch("sys.argv", ["mlx_embed_server.py", "--max-bulk-microbatch-tokens", "1024"]), \
         patch("scripts.mlx_embed_server.start_server", return_value=(server, thread)) as start, \
         patch("signal.signal"):
        main()
    assert start.call_args.kwargs["max_bulk_microbatch_tokens"] == 1024


@pytest.mark.parametrize("control", [False, True])
def test_total_header_byte_budget_is_enforced_on_both_listeners(control):
    with listener(control=control) as (server, _):
        server.max_header_bytes = 160
        with socket.create_connection(server.server_address, timeout=0.5) as client:
            client.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Large: " + b"a" * 200 + b"\r\n\r\n")
            response = receive_all(client)
            assert b"431" in response.split(b"\r\n", 1)[0]
            assert json.loads(response.split(b"\r\n\r\n", 1)[1])["type"] == "invalid_protocol"
        assert wait_until(lambda: not server._active_sockets)
        assert wait_until(lambda: server.thread_limiter._value == 1)


def test_header_read_ahead_does_not_charge_coalesced_body_against_header_budget():
    with listener() as (server, _):
        server.max_header_bytes = 256
        status, _, body = request(server, "/embed", {"texts": ["a" * 8192]})
        assert status == 200, body
        assert len(server.runtime.calls) == 1


@pytest.mark.parametrize("binary", [False, True])
def test_mismatched_serving_descriptor_never_emits_success(binary):
    runtime = SyntheticRuntime()
    runtime.get_descriptor = lambda requested_dims=None: {"outputDimensions": 3, "model": "different-space"}
    with listener(runtime=runtime) as (server, _):
        status, _, body = request(server, "/embed-bin" if binary else "/embed", {"texts": ["synthetic"], "dims": 2})
        assert status == 400
        assert json.loads(body)["type"] == "invalid_protocol"


def test_binary_descriptor_serialization_failure_does_not_publish_a_200_prefix():
    runtime = SyntheticRuntime()
    descriptor = runtime.get_descriptor()
    descriptor["invalid"] = float("nan")
    runtime.get_descriptor = lambda requested_dims=None: descriptor
    with listener(runtime=runtime) as (server, _):
        with socket.create_connection(server.server_address, timeout=1.0) as client:
            client.sendall(raw_post(b'{"texts":["synthetic"]}', path="/embed-bin"))
            response = receive_all(client)
        assert response.startswith(b"HTTP/1.1 500")
        assert response.count(b"HTTP/1.1 ") == 1


def test_runtime_bulk_budget_integration_uses_real_runtime_without_loading_a_model():
    class Adapter:
        synthetic = True
        model = None
        model_name = "synthetic/config-only"
        model_params_b = 0.001
        model_memory_mb = 0.0
        native_dims = 4

        def is_loaded(self):
            return False

        def unload(self):
            pass

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=Adapter()):
        server, coordinator = start_server(
            "synthetic/config-only", port=0, preload=False, warmup=False,
            max_batch_tokens=4096, max_bulk_microbatch_tokens=2048,
        )
        try:
            assert wait_until(lambda: server.state in (ServerState.READY, ServerState.FAILED))
            assert server.state == ServerState.READY, server.state_error
            assert server.runtime.batch_planner.max_bulk_microbatch_tokens == 2048
            assert server.runtime.batch_planner.max_batch_tokens == 4096
        finally:
            server.server_close()
            coordinator.join(1.0)
            assert not coordinator.is_alive()


@pytest.mark.parametrize("failed_name", ["MLX-HTTP-Serve", "MLX-Control-Serve", "MLX-Init-Coordinator"])
def test_startup_thread_failure_closes_both_listeners_and_started_threads(failed_name):
    servers = []
    original_init = ThreadedMLXServer.__init__
    original_start = threading.Thread.start

    def remember(server, *args, **kwargs):
        original_init(server, *args, **kwargs)
        servers.append(server)

    def fail_one(thread):
        if thread.name == failed_name:
            raise RuntimeError("synthetic thread startup failure")
        return original_start(thread)

    with patch.object(ThreadedMLXServer, "__init__", remember), patch.object(threading.Thread, "start", fail_one):
        with pytest.raises(RuntimeError, match="synthetic thread startup failure"):
            start_server("synthetic", port=0, preload=False, warmup=False)
    server = servers[0]
    assert server.socket.fileno() == -1
    assert server.control_server.socket.fileno() == -1
    assert not server.t_http.is_alive()
    assert not server.t_ctrl.is_alive()


def test_control_bind_failure_closes_previously_bound_inference_socket():
    servers = []
    original_init = ThreadedMLXServer.__init__

    def remember(server, *args, **kwargs):
        original_init(server, *args, **kwargs)
        servers.append(server)

    with patch.object(ThreadedMLXServer, "__init__", remember), \
         patch("scripts.qmd_mlx.server.MLXControlServer", side_effect=OSError("synthetic bind failure")):
        with pytest.raises(OSError, match="synthetic bind failure"):
            start_server("synthetic", port=0, preload=False, warmup=False)
    assert servers[0].socket.fileno() == -1


def test_deadline_cancellation_remains_a_504_not_a_peer_abort_400():
    runtime = SyntheticRuntime()

    def blocked(*args, **kwargs):
        assert kwargs["cancel_event"].wait(0.5)
        raise RequestCancelledError("cooperative cancellation after monitor deadline")

    runtime.submit_embed = blocked
    with listener(runtime=runtime) as (server, _):
        status, _, body = request(server, "/embed", {"texts": ["synthetic"], "timeout": 0.1})
        assert status == 504
        assert json.loads(body)["type"] == "deadline_exceeded"
        assert wait_until(lambda: server.inference_limiter._value == 1)
        assert wait_until(lambda: not any(t.name.startswith("MLX-Disconnect-") for t in threading.enumerate()))


def test_completed_monitor_wakes_immediately_instead_of_waiting_for_poll_interval():
    with listener() as (server, _):
        # Keep both socketpair ends owned for the test, without an inference job.
        peer, remote = socket.socketpair()
        try:
            handler = MLXHTTPRequestHandler.__new__(MLXHTTPRequestHandler)
            handler.server = server
            handler.connection = peer
            started = time.monotonic()
            for _ in range(10):
                with handler._request_cancellation(time.monotonic() + 1.0):
                    pass
            assert time.monotonic() - started < 0.2
            assert not any(t.name.startswith("MLX-Disconnect-") for t in threading.enumerate())
        finally:
            peer.close()
            remote.close()


def test_no_model_startup_reports_failure_but_keeps_diagnostics_available():
    server, coordinator = start_server(None, port=0, preload=False, warmup=False, max_bulk_microbatch_tokens=1024)
    try:
        assert wait_until(lambda: server.state == ServerState.FAILED)
        assert "No model configured" in server.state_error
        assert server.runtime is None
        connection = http.client.HTTPConnection("127.0.0.1", server.control_port, timeout=1.0)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["state"] == ServerState.FAILED
        finally:
            connection.close()
    finally:
        server.server_close()
        coordinator.join(1.0)
        assert not coordinator.is_alive()
