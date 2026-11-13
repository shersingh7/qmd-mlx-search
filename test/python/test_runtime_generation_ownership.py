"""CPU-only positive contracts for request-bound tokenizer/model ownership."""

from __future__ import annotations

import gc
import http.client
import json
import threading
import time
import weakref
from urllib.parse import unquote

import numpy as np
import pytest

from scripts.qmd_mlx import runtime as runtime_module
from scripts.qmd_mlx.adapters.embedding import (
    BaseEmbeddingAdapter, BertEmbeddingAdapter, QwenEmbeddingAdapter, SyntheticEmbeddingAdapter,
)
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.model_manager import ModelState
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError, ModelUnavailableError, RequestCancelledError, WorkerUnavailableError,
    decode_binary_embeddings,
)
from scripts.qmd_mlx.server import MLXHTTPRequestHandler, ServerState, ThreadedMLXServer


class CPUWeights:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)


class CPUTokenizer:
    pad_token_id = 0

    def __init__(self, token_id=1, *, entered=None, resume=None):
        self.token_id = token_id
        self.entered = entered
        self.resume = resume
        self.threads = []

    def encode(self, text, **kwargs):
        self.threads.append(threading.get_ident())
        if self.entered is not None:
            self.entered.set()
            assert self.resume is not None
            assert self.resume.wait(3.0), "test did not release CPU tokenization"
        return [self.token_id, self.token_id + 1]


class CPUAdapter(BaseEmbeddingAdapter):
    """Real publication/snapshot contract, fictional CPU weights and assets."""

    synthetic = True

    def __init__(self, source=None):
        super().__init__(str(source or "synthetic/generation-contract"), max_length=32)
        self.source = source
        self.estimated_memory_mb = 1.0
        self.next_revision = "fixture-A"
        self.next_vector = [1, 0]
        self.next_token_id = 1
        self.tokenizer_events: tuple[threading.Event | None, threading.Event | None] = (None, None)
        self.load_threads = []
        self.unload_threads = []
        self.model_refs = []
        self.tokenizer_refs = []
        self.forward_calls = []

    def load(self):
        with self._init_lock:
            if self.is_loaded():
                return
            self.load_threads.append(threading.get_ident())
            if self.source is None:
                config = {"hidden_size": 2, "max_position_embeddings": 32}
                revision, vector, fingerprints = self.next_revision, self.next_vector, []
            else:
                path, revision, config, fingerprints = self._resolve_source()
                vector = np.frombuffer((self.source / "model.bin").read_bytes(), dtype=np.uint8)
            model = CPUWeights(vector)
            tokenizer = CPUTokenizer(
                self.next_token_id, entered=self.tokenizer_events[0], resume=self.tokenizer_events[1],
            )
            self._check_source_unchanged(fingerprints)
            self._publish_loaded(model, tokenizer, tokenizer, config, revision, "float32", "none", 0)
            self.model_memory_mb = 1.0
            self.model_refs.append(weakref.ref(model))
            self.tokenizer_refs.append(weakref.ref(tokenizer))

    def unload(self):
        self.unload_threads.append(threading.get_ident())
        super().unload()

    def forward_batch(self, tokenized_batch, requested_dims=None):
        batch = tokenized_batch
        self.forward_calls.append((self.measured_revision, [list(row) for row in batch.token_ids]))
        return np.broadcast_to(self.model.vector, (len(batch), 2)).copy()


@pytest.fixture(autouse=True)
def no_metal_telemetry(monkeypatch):
    monkeypatch.setattr("scripts.qmd_mlx.model_manager._MLX_AVAILABLE", False)
    monkeypatch.setattr("scripts.qmd_mlx.adapters.embedding._MLX_AVAILABLE", False)


def make_runtime(monkeypatch, adapter=None, *, shared=False, lazy_load=False, **kwargs):
    adapter = adapter or CPUAdapter()
    monkeypatch.setattr(runtime_module, "resolve_embedding_adapter", lambda **kw: adapter)
    executor = GPUExecutor() if shared else None
    runtime = runtime_module.MLXEmbeddingRuntime(
        adapter.model_name, executor=executor, lazy_load=lazy_load, **kwargs,
    )
    return runtime, adapter


def finish(runtime):
    runtime.shutdown(timeout=2.0)
    runtime.executor.shutdown(timeout=2.0)
    assert not runtime.executor.is_worker_alive()
    assert runtime.in_flight_requests == runtime.in_flight_bytes == 0


def start_call(function):
    outcomes = []

    def run():
        try:
            outcomes.append(function())
        except BaseException as exc:
            outcomes.append(exc)

    thread = threading.Thread(target=run, name="generation-contract-caller")
    thread.start()
    return thread, outcomes


@pytest.mark.parametrize("adapter_type", [BertEmbeddingAdapter, QwenEmbeddingAdapter, SyntheticEmbeddingAdapter])
def test_cpu_tokenization_never_implicitly_loads_weights(monkeypatch, adapter_type):
    adapter = adapter_type("synthetic/no-implicit-load")
    loads = []
    monkeypatch.setattr(adapter, "load", lambda: loads.append(threading.get_ident()))
    with pytest.raises(ModelUnavailableError, match="(?:loaded|prepar|tokenizer)"):
        adapter.tokenize_texts(["unprepared CPU input"])
    assert loads == []
    assert adapter.model is None


@pytest.mark.parametrize("missing", ["measured_revision", "measured_dtype", "measured_quantization"])
def test_ready_weights_cannot_supply_a_guessed_serving_identity(monkeypatch, missing):
    runtime, adapter = make_runtime(monkeypatch)
    setattr(adapter, missing, None)
    try:
        with pytest.raises(ModelUnavailableError, match="verified serving identity"):
            runtime.submit_embed(["metadata is part of readiness"])
        assert adapter.forward_calls == []
        assert adapter.raw_hf_tokenizer.threads == []
        assert len(adapter.load_threads) == 1
    finally:
        finish(runtime)


def test_snapshot_retains_only_cpu_tokenizer_across_owner_eviction(monkeypatch):
    runtime, adapter = make_runtime(monkeypatch)
    try:
        preparation = runtime.executor.submit(adapter.capture_preparation)
        runtime.model_manager.unload("embed")
        gc.collect()
        assert adapter.model_refs[0]() is None
        assert adapter.tokenizer_refs[0]() is preparation.tokenizer
        batch = preparation.tokenize_texts(["captured CPU state, no reload"])
        assert batch.token_ids == [[1, 2]]
        assert adapter.load_threads == adapter.unload_threads == [runtime.executor._worker_thread.ident]
        assert not adapter.is_loaded()
        assert runtime.get_memory_info()["model_mb"] == 0
        assert json.loads(preparation.descriptor_json)["revision"] == "fixture-A"
    finally:
        finish(runtime)


def test_ready_cpu_tokenization_can_overlap_a_busy_owner(monkeypatch):
    runtime, adapter = make_runtime(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def owner_work():
        entered.set()
        assert release.wait(3.0)

    active, _, _ = runtime.executor.submit_async(owner_work)
    assert entered.wait(1.0)
    caller, outcomes = start_call(lambda: runtime.tokenize(["ready CPU tokenizer"]))
    try:
        caller.join(0.5)
        assert not caller.is_alive(), "ready tokenization queued behind a native forward"
        assert outcomes == [[[1, 2]]]
        assert not active.done()
        assert adapter.raw_hf_tokenizer.threads == [caller.ident]
        with runtime.model_manager._lock:
            assert runtime.model_manager._leases.get("embed", 0) == 0
    finally:
        release.set()
        caller.join(2.0)
        active.result(timeout=1.0)
        finish(runtime)


def test_cpu_snapshot_never_waits_unboundedly_for_a_busy_lifecycle_lock(monkeypatch):
    runtime, adapter = make_runtime(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def owner_lifecycle():
        with adapter._init_lock:
            entered.set()
            assert release.wait(3.0)

    active, _, _ = runtime.executor.submit_async(owner_lifecycle)
    assert entered.wait(1.0)
    try:
        start = time.monotonic()
        with pytest.raises(DeadlineExceededError):
            runtime.tokenize(["busy lifecycle"], deadline=start + 0.03)
        assert time.monotonic() - start < 0.5
        assert not active.done()
        assert adapter.raw_hf_tokenizer.threads == []
        assert runtime.in_flight_requests == runtime.in_flight_bytes == 0
    finally:
        release.set()
        active.result(timeout=1.0)
        finish(runtime)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("method", ["tokenize", "submit_embed"])
@pytest.mark.parametrize("pause", ["after_snapshot", "during_tokenization"])
def test_shutdown_fences_cpu_preparation_without_a_weight_lease(monkeypatch, shared, method, pause):
    entered, resume = threading.Event(), threading.Event()
    adapter = CPUAdapter()
    if pause == "during_tokenization":
        adapter.tokenizer_events = (entered, resume)
    runtime, adapter = make_runtime(monkeypatch, adapter, shared=shared, lazy_load=True)
    original_submit = runtime.executor.submit

    def delayed_waiter(*args, **kwargs):
        value = original_submit(*args, **kwargs)
        if kwargs.get("description") == "Initialize embedding tokenizer" and pause == "after_snapshot":
            entered.set()
            assert resume.wait(3.0)
        return value

    monkeypatch.setattr(runtime.executor, "submit", delayed_waiter)
    caller, outcomes = start_call(lambda: getattr(runtime, method)(["admitted CPU preparation"]))
    try:
        assert entered.wait(1.0), repr(outcomes)
        with runtime.model_manager._lock:
            assert runtime.model_manager._leases.get("embed", 0) == 0
        runtime.shutdown(timeout=1.0)
        completion = runtime._shutdown_future
        assert completion is not None
        assert completion.done()
        assert not adapter.is_loaded()
        assert runtime.model_manager.get_state("embed") == ModelState.UNLOADED
        assert runtime.get_memory_info()["model_mb"] == 0
        gc.collect()
        assert adapter.model_refs[0]() is None, "CPU preparation retained GPU weights"
        assert adapter.tokenizer_refs[0]() is not None, "snapshot lost its CPU tokenizer"
        resume.set()
        caller.join(1.0)
        assert not caller.is_alive()
        assert len(outcomes) == 1 and isinstance(outcomes[0], WorkerUnavailableError)
        runtime.shutdown(timeout=1.0)
        assert runtime._shutdown_future is completion
        assert adapter.load_threads == adapter.unload_threads == [runtime.executor._worker_thread.ident]
        assert not adapter.is_loaded()
        if shared:
            assert runtime.executor.submit(lambda: "other stage remains usable") == "other stage remains usable"
    finally:
        resume.set()
        caller.join(2.0)
        finish(runtime)


@pytest.mark.parametrize("changed_identity", [False, True])
def test_eviction_during_cpu_tokenization_refuses_old_request_but_new_request_recovers(monkeypatch, changed_identity):
    entered, resume = threading.Event(), threading.Event()
    adapter = CPUAdapter()
    adapter.tokenizer_events = (entered, resume)
    runtime, adapter = make_runtime(monkeypatch, adapter)
    caller, outcomes = start_call(lambda: runtime.submit_embed(["old CPU tokenizer"]))
    try:
        assert entered.wait(1.0)
        # CPU preparation must leave the owner free for other stages/eviction.
        assert runtime.executor.submit(lambda: threading.get_ident(), timeout_s=0.3) == runtime.executor._worker_thread.ident
        runtime.model_manager.unload("embed")
        if changed_identity:
            adapter.next_revision, adapter.next_vector, adapter.next_token_id = "fixture-B", [0, 1], 7
        adapter.tokenizer_events = (None, None)
        runtime.model_manager.ensure_loaded("embed")
        resume.set()
        caller.join(1.0)
        assert not caller.is_alive()
        assert len(outcomes) == 1 and isinstance(outcomes[0], ModelUnavailableError)
        assert adapter.forward_calls == []
        assert adapter.tokenizer_refs[0]().threads == [caller.ident]
        result = runtime.submit_embed_result(["new complete request"])
        np.testing.assert_array_equal(result.embeddings, [adapter.next_vector])
        assert result.descriptor["revision"] == adapter.next_revision
        assert adapter.forward_calls == [(adapter.next_revision, [[adapter.next_token_id, adapter.next_token_id + 1]])]
        assert set(adapter.load_threads + adapter.unload_threads) == {runtime.executor._worker_thread.ident}
    finally:
        resume.set()
        caller.join(2.0)
        finish(runtime)


@pytest.mark.parametrize("reload", [False, True])
@pytest.mark.parametrize("changed_identity", [False, True])
def test_microbatches_never_combine_load_generations(monkeypatch, reload, changed_identity):
    runtime, adapter = make_runtime(monkeypatch, max_batch_tokens=2)
    original_submit = runtime.executor.submit

    def evict_between_batches(*args, **kwargs):
        value = original_submit(*args, **kwargs)
        if kwargs.get("description") == "Embed batch 1/3 (doc)":
            runtime.model_manager.unload("embed")
            if changed_identity:
                adapter.next_revision, adapter.next_vector = "fixture-B", [0, 1]
            if reload:
                runtime.model_manager.ensure_loaded("embed")
        return value

    monkeypatch.setattr(runtime.executor, "submit", evict_between_batches)
    try:
        with pytest.raises(ModelUnavailableError, match="generation"):
            runtime.submit_embed_result(["first", "second", "third"])
        assert adapter.forward_calls == [("fixture-A", [[1, 2]])]
        # Strict policy: even identical-asset reloads need a whole new request.
        result = runtime.submit_embed_result(["whole request retry"])
        np.testing.assert_array_equal(result.embeddings, [adapter.next_vector])
        assert result.descriptor["revision"] == adapter.next_revision
    finally:
        finish(runtime)


def test_real_local_asset_mutation_between_microbatches_fails_closed(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 2, "max_position_embeddings": 32}))
    (tmp_path / "model.bin").write_bytes(bytes([1, 0]))
    (tmp_path / "tokenizer.json").write_text('{"fictional_cpu_fixture": "A"}')
    runtime, adapter = make_runtime(monkeypatch, CPUAdapter(tmp_path), max_batch_tokens=2)
    first_revision = runtime.get_descriptor()["revision"]
    original_submit = runtime.executor.submit

    def mutate_source(*args, **kwargs):
        value = original_submit(*args, **kwargs)
        if kwargs.get("description") == "Embed batch 1/3 (doc)":
            (tmp_path / "model.bin").write_bytes(bytes([0, 1]))
            (tmp_path / "tokenizer.json").write_text('{"fictional_cpu_fixture": "B"}')
            runtime.model_manager.unload("embed")
            runtime.model_manager.ensure_loaded("embed")
        return value

    monkeypatch.setattr(runtime.executor, "submit", mutate_source)
    try:
        with pytest.raises(ModelUnavailableError, match="generation"):
            runtime.submit_embed(["first", "second", "third"])
        assert len(adapter.forward_calls) == 1
        result = runtime.submit_embed_result(["new source request"])
        assert first_revision.startswith("local-sha256:")
        assert result.descriptor["revision"] != first_revision
        np.testing.assert_array_equal(result.embeddings, [[0, 1]])
    finally:
        finish(runtime)


@pytest.mark.parametrize("change", ["reload", "descriptor", "tokenizer"])
def test_retry_subdivisions_revalidate_the_preparation(monkeypatch, change):
    runtime, adapter = make_runtime(monkeypatch, max_batch_tokens=32)
    calls = []
    original_forward = adapter.forward_batch

    def fail_first(batch, requested_dims=None):
        calls.append(len(batch))
        if len(calls) == 1:
            if change == "reload":
                adapter.unload()
                adapter.load()  # Test-only reentrant lifecycle on the real owner.
            elif change == "descriptor":
                adapter.measured_revision = "fixture-B"
            else:
                adapter.raw_hf_tokenizer = CPUTokenizer(9)
            raise RuntimeError("MLX out of memory")
        return original_forward(batch, requested_dims=requested_dims)

    monkeypatch.setattr(adapter, "forward_batch", fail_first)
    try:
        with pytest.raises(ModelUnavailableError, match="generation"):
            runtime.submit_embed(["first", "second", "third", "fourth"])
        assert calls == [4], "an OOM subdivision used a different preparation"
        assert not adapter.forward_calls
    finally:
        finish(runtime)


def test_successful_final_forward_cannot_change_generation_unnoticed(monkeypatch):
    runtime, adapter = make_runtime(monkeypatch)
    original_forward = adapter.forward_batch

    def changed_after_forward(batch, requested_dims=None):
        result = original_forward(batch, requested_dims=requested_dims)
        adapter.unload()
        adapter.next_revision, adapter.next_vector = "fixture-B", [0, 1]
        adapter.load()
        return result

    monkeypatch.setattr(adapter, "forward_batch", changed_after_forward)
    try:
        with pytest.raises(ModelUnavailableError, match="generation"):
            runtime.submit_embed_result(["final microbatch"])
    finally:
        finish(runtime)


@pytest.mark.parametrize("reason", ["cancel", "deadline", "shutdown"])
def test_preparation_completion_rechecks_original_budget_and_shutdown(monkeypatch, reason):
    runtime, adapter = make_runtime(monkeypatch)
    event = threading.Event()
    deadline = time.monotonic() + 1.0
    original_encode = adapter.raw_hf_tokenizer.encode

    def encode(text, **kwargs):
        if reason == "cancel":
            event.set()
        elif reason == "deadline":
            time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)
        else:
            runtime.shutdown(timeout=1.0)
        return original_encode(text, **kwargs)

    monkeypatch.setattr(adapter.raw_hf_tokenizer, "encode", encode)
    try:
        expected = {"cancel": RequestCancelledError, "deadline": DeadlineExceededError, "shutdown": WorkerUnavailableError}[reason]
        with pytest.raises(expected):
            runtime.submit_embed(["CPU completion fence"], deadline=deadline, cancel_event=event)
        assert adapter.forward_calls == []
        assert len(adapter.load_threads) == 1
    finally:
        finish(runtime)


def test_result_captures_identity_without_changing_public_ndarray_contract(monkeypatch):
    runtime, adapter = make_runtime(monkeypatch)
    try:
        vectors = runtime.submit_embed(["public ndarray"])
        assert type(vectors) is np.ndarray
        operation = runtime.submit_embed_result(["captured identity"])
        assert type(operation.embeddings) is np.ndarray
        descriptor = operation.descriptor
        descriptor["revision"] = "caller mutation"
        runtime.model_manager.unload("embed")
        adapter.next_revision, adapter.next_vector = "fixture-B", [0, 1]
        runtime.model_manager.ensure_loaded("embed")
        next_operation = runtime.submit_embed_result(["other request"])
        assert operation.descriptor["revision"] == "fixture-A"
        assert next_operation.descriptor["revision"] == "fixture-B"
        np.testing.assert_array_equal(operation.embeddings, [[1, 0]])
        np.testing.assert_array_equal(next_operation.embeddings, [[0, 1]])
    finally:
        finish(runtime)


@pytest.mark.parametrize("binary", [False, True])
def test_response_identity_survives_another_request_reload_before_serialization(monkeypatch, binary):
    runtime, adapter = make_runtime(monkeypatch, shared=True)
    server = ThreadedMLXServer(("127.0.0.1", 0), MLXHTTPRequestHandler)
    server.runtime, server.executor, server.model_manager = runtime, runtime.executor, runtime.model_manager
    server.state = ServerState.READY
    captured = []
    original_result = runtime.submit_embed_result

    def reload_after_inference(*args, **kwargs):
        operation = original_result(*args, **kwargs)
        captured.append(operation)
        runtime.model_manager.unload("embed")
        adapter.next_revision, adapter.next_vector = "fixture-B", [0, 1]
        runtime.model_manager.ensure_loaded("embed")
        # Complete another operation before the original HTTP handler resumes.
        captured.append(original_result(["competing request"]))
        return operation

    monkeypatch.setattr(runtime, "submit_embed_result", reload_after_inference)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2.0)
    try:
        connection.request("POST", "/embed-bin" if binary else "/embed", json.dumps({"texts": ["response A"]}), {
            "Content-Type": "application/json", "Connection": "close",
        })
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200, body
        if binary:
            header = response.getheader("X-QMD-Embedding-Descriptor")
            assert header is not None
            descriptor = json.loads(unquote(header))
            vectors, count, dims = decode_binary_embeddings(body)
            assert (count, dims) == (1, 2)
        else:
            payload = json.loads(body)
            descriptor, vectors = payload["descriptor"], payload["embeddings"]
            assert payload["model"] == descriptor["model"]
        assert descriptor == captured[0].descriptor
        assert descriptor["revision"] == "fixture-A"
        assert captured[1].descriptor["revision"] == "fixture-B"
        assert runtime.get_descriptor()["revision"] == "fixture-B"
        np.testing.assert_array_equal(vectors, [[1, 0]])
    finally:
        connection.close()
        server.server_close()
        serving.join(1.0)
        assert not serving.is_alive()
        finish(runtime)
