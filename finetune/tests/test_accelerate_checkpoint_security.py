"""Installed-wheel regression/integration tests; CPU-only, owned fixtures only.

Normal: python -m pytest -q tests/test_accelerate_checkpoint_security.py
Unmodified-wheel negative proof (intercepted before ANY shard load):
  QMD_ACCELERATE_BASELINE=1 python -m pytest -q tests/test_accelerate_checkpoint_security.py -k baseline

No validator is imported from the repository. All runtime guards are exercised
through the installed Accelerate public APIs, including retained imported aliases.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import errno
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import pytest
import torch
from safetensors.torch import save_file

import accelerate
import accelerate.big_modeling as big_modeling
import accelerate.utils as accelerate_utils
import accelerate.utils.modeling as modeling

# Capture by-value aliases BEFORE importing PEFT/TRL. No sitecustomize, runtime
# package monkeypatch, or application startup ordering is needed by the patch.
from accelerate import load_checkpoint_and_dispatch as retained_dispatch
from accelerate import load_checkpoint_in_model as retained_in_model
from accelerate.utils.modeling import load_state_dict as retained_state_dict

FINETUNE = Path(__file__).resolve().parents[1]
VENDOR = FINETUNE / "vendor" / "accelerate"
BASELINE = os.getenv("QMD_ACCELERATE_BASELINE") == "1"
ENTRIES = [
    ("top-in-model", retained_in_model),
    ("utils-in-model", accelerate_utils.load_checkpoint_in_model),
    ("modeling-in-model", modeling.load_checkpoint_in_model),
    ("big-modeling-in-model", big_modeling.load_checkpoint_in_model),
    ("top-dispatch", retained_dispatch),
    ("big-modeling-dispatch", big_modeling.load_checkpoint_and_dispatch),
]


@pytest.fixture(autouse=True)
def installed_wheel_contract(request):
    is_baseline = "baseline" in request.node.name
    if BASELINE != is_baseline:
        pytest.skip("separate unmodified-wheel baseline run" if is_baseline else "baseline-only run")
    expected = "1.15.0" if BASELINE else "1.15.0+qmd.2"
    assert accelerate.__version__ == expected
    assert importlib.metadata.version("accelerate") == expected
    assert "site-packages" in str(Path(accelerate.__file__).resolve())
    assert VENDOR not in Path(accelerate.__file__).resolve().parents
    torch.set_num_threads(1)


@pytest.fixture(params=ENTRIES, ids=[name for name, _ in ENTRIES])
def entry(request):
    return request.param[1]


def model():
    result = torch.nn.Linear(2, 2)
    with torch.no_grad():
        result.weight.fill_(-9)
        result.bias.fill_(-9)
    return result


def weights():
    return {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "bias": torch.tensor([5.0, 6.0])}


def write_weights(path, state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = weights() if state is None else state
    if str(path).endswith(".safetensors"):
        save_file(state, str(path), metadata={"format": "pt"})
    else:
        torch.save(state, path)


def index_at(root, mapping, *, wrapped=True):
    path = root / "model.safetensors.index.json"
    path.write_text(json.dumps({"metadata": {"total_size": 24}, "weight_map": mapping} if wrapped else mapping))
    return path


def invoke(entry, checkpoint, target=None, **kwargs):
    target = model() if target is None else target
    result = entry(target, checkpoint, device_map={"": "cpu"}, **kwargs)
    if "dispatch" in entry.__name__:
        assert result is target
    return target


def assert_weights(target, expected=None):
    expected = weights() if expected is None else expected
    for key, value in expected.items():
        torch.testing.assert_close(target.state_dict()[key], value)


def forbid_reads(monkeypatch, forbidden=()):
    """Catch forbidden data-open attempts before syscall, never open a FIFO.

    Linux O_PATH is metadata-only and allowed by this monitor. Directory opens
    needed to walk held capabilities are allowed. Device cases use synthetic
    stat metadata, never real host devices or elevated privileges.
    """
    real_open, real_builtin, real_stat = os.open, builtins.open, os.stat
    forbidden_ids = set()
    for path in forbidden:
        try:
            info = real_stat(path, follow_symlinks=False)
            forbidden_ids.add((info.st_dev, info.st_ino))
        except OSError:
            pass
    attempts = []

    def target_info(path, dir_fd=None):
        # Kernel FD aliases are symlinks in lstat, but the held inode is the
        # object being opened. Inspect it; never broadly exempt /proc paths.
        value = os.fspath(path)
        match = re.fullmatch(r"/(?:proc/self/fd|dev/fd)/([0-9]+)", value) if isinstance(value, str) else None
        if match is not None:
            return os.fstat(int(match[1]))
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=False)

    def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
        if not (flags & os.O_DIRECTORY) and not (flags & getattr(os, "O_PATH", 0)):
            try:
                info = target_info(path, dir_fd)
            except (OSError, TypeError):
                info = None
            if info is not None and (
                (not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode))
                or (info.st_dev, info.st_ino) in forbidden_ids
            ):
                attempts.append(str(path))
                raise AssertionError(f"Rejected shard reached a DATA open: {path}")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def guarded_builtin(path, *args, **kwargs):
        if not isinstance(path, int):
            try:
                info = target_info(path)
            except (OSError, TypeError):
                info = None
            if info is not None and (info.st_dev, info.st_ino) in forbidden_ids:
                attempts.append(str(path))
                raise AssertionError("Rejected shard reached builtins.open")
        return real_builtin(path, *args, **kwargs)

    def no_deserialize(*args, **kwargs):
        attempts.append("deserializer")
        raise AssertionError("A rejected checkpoint reached a tensor deserializer")

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin)
    monkeypatch.setattr(modeling.torch, "load", no_deserialize)
    monkeypatch.setattr(modeling, "safe_open", no_deserialize)
    monkeypatch.setattr(modeling, "safe_load_file", no_deserialize)
    return attempts


def security_error():
    from accelerate.utils.checkpoint_security import CheckpointSecurityError
    return CheckpointSecurityError


@pytest.mark.parametrize("direct_index", [False, True], ids=["directory", "explicit-index"])
@pytest.mark.parametrize("kind", ["traversal", "absolute", "fifo", "directory"])
def test_baseline_old_wheel_passes_unsafe_shards(entry, direct_index, kind, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    outside = tmp_path / "owned-not-private.bin"
    write_weights(outside)
    if kind == "traversal":
        name = "../" + outside.name
    elif kind == "absolute":
        name = str(outside)
    elif kind == "fifo":
        name = "never-open.fifo"
        os.mkfifo(root / name)
    else:
        name = "directory.bin"
        (root / name).mkdir()
    index = index_at(root, {"weight": name})
    calls = []
    class Intercepted(Exception):
        pass
    def intercept(path, **kwargs):
        calls.append(os.fspath(path))
        raise Intercepted("intercepted before any shard data read")
    monkeypatch.setattr(modeling, "load_state_dict", intercept)
    attempts = forbid_reads(monkeypatch, [outside])
    with pytest.raises(Intercepted):
        invoke(entry, index if direct_index else root)
    assert calls == [os.path.join(str(root), name)]
    assert attempts == []


BAD_NAMES = [
    "../owned.bin", "nested/../../owned.bin", "nested/../owned.bin", "./shard.bin",
    "", ".", "..", "nested//shard.bin", "shard.bin/", "/absolute/owned.bin",
    "//server/share/owned.bin", "C:\\owned.bin", "C:/owned.bin", "C:owned.bin",
    "\\rooted.bin", "\\\\server\\share\\owned.bin", "file:///owned.bin", "a\x00b.bin",
    "nested\\..\\owned.bin", "shard:stream.bin", "../checkpoint-prefix-sibling/owned.bin",
]


@pytest.mark.parametrize("name", BAD_NAMES)
@pytest.mark.parametrize("direct_index", [False, True])
def test_reject_path_forms_before_any_shard_read(entry, name, direct_index, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    index = index_at(root, {"weight": name})
    attempts = forbid_reads(monkeypatch)
    with pytest.raises(security_error()):
        invoke(entry, index if direct_index else root)
    assert attempts == []


BAD_SCHEMA = [
    [], None, "text", 7, {}, {"weight_map": []}, {"weight_map": None}, {"weight_map": {}},
    {"weight_map": {"weight": 12}}, {"weight_map": {"weight": False}},
    {"weight_map": {"weight": ["shard.bin"]}}, {"weight_map": {"weight": {"file": "shard.bin"}}},
    {"weight_map": {"weight": "ok.bin", "bias": 2}}, {"weight_map": {"": "shard.bin"}},
    {"weight_map": {"bad\x00key": "shard.bin"}}, {"weight_map": {"weight": "ok.bin"}, "metadata": []},
]


@pytest.mark.parametrize("schema", BAD_SCHEMA)
@pytest.mark.parametrize("direct_index", [False, True])
def test_reject_index_schema_before_any_shard_read(entry, schema, direct_index, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    index = root / "model.index.json"
    index.write_text(json.dumps(schema))
    attempts = forbid_reads(monkeypatch)
    with pytest.raises(security_error()):
        invoke(entry, index if direct_index else root)
    assert attempts == []


@pytest.mark.parametrize("text", ["not json", '{"weight_map":{"weight":"x","weight":"y"}}', '{"weight_map":{"weight":"x"},"weight_map":{"weight":"y"}}'])
def test_reject_malformed_or_duplicate_index(entry, text, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    index = root / "model.index.json"
    index.write_text(text)
    attempts = forbid_reads(monkeypatch)
    with pytest.raises(security_error()):
        invoke(entry, index)
    assert attempts == []


def socket_at(path):
    # Darwin limits sockaddr_un paths to 104 bytes; pytest node directories can
    # be longer. Bind an owned short path, then rename the socket fixture.
    resource = socket.socket(socket.AF_UNIX)
    with tempfile.TemporaryDirectory(prefix="qmd-s-") as directory:
        source = Path(directory) / "s"
        resource.bind(str(source))
        source.rename(path)
    return resource


def make_special(root, kind):
    path = root / "bad.bin"
    resource = None
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "socket":
        resource = socket_at(path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "broken-link":
        path.symlink_to(root / "does-not-exist")
    elif kind != "missing":
        path.write_bytes(b"owned simulated device placeholder")
    return path, resource


def simulate_device(monkeypatch, path, kind):
    if kind not in ("character", "block"):
        return
    real_stat = os.stat
    inode = real_stat(path)
    def changed_stat(candidate, *args, **kwargs):
        info = real_stat(candidate, *args, **kwargs)
        if (info.st_dev, info.st_ino) == (inode.st_dev, inode.st_ino):
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            values["st_mode"] = (stat.S_IFCHR if kind == "character" else stat.S_IFBLK) | 0o600
            return SimpleNamespace(**values)
        return info
    monkeypatch.setattr(os, "stat", changed_stat)


@pytest.mark.parametrize("kind", ["fifo", "socket", "directory", "missing", "broken-link", "character", "block"])
@pytest.mark.parametrize("direct_index", [False, True])
def test_reject_special_shards_before_open(entry, kind, direct_index, tmp_path, monkeypatch):
    root = tmp_path / "c"
    root.mkdir()
    path, resource = make_special(root, kind)
    index = index_at(root, {"weight": path.name})
    simulate_device(monkeypatch, path, kind)
    attempts = forbid_reads(monkeypatch, [path])
    try:
        with pytest.raises(security_error()):
            invoke(entry, index if direct_index else root)
        assert attempts == []
    finally:
        if resource is not None:
            resource.close()


@pytest.mark.parametrize("kind", ["fifo", "socket", "directory", "broken-link"])
@pytest.mark.parametrize("filename", ["pytorch_model.bin", "model.safetensors", "model.index.json"])
@pytest.mark.parametrize("direct", [False, True])
def test_special_whole_checkpoint_and_index_discovery(entry, kind, filename, direct, tmp_path, monkeypatch):
    root = tmp_path / "c"
    root.mkdir()
    path, resource = make_special(root, kind)
    destination = root / filename
    path.rename(destination)
    attempts = forbid_reads(monkeypatch, [destination])
    try:
        with pytest.raises(security_error()):
            invoke(entry, destination if direct else root)
        assert attempts == []
    finally:
        if resource is not None:
            resource.close()


@pytest.mark.parametrize("kind", ["fifo", "socket", "directory", "broken-link", "missing"])
def test_final_load_state_dict_boundary(kind, tmp_path, monkeypatch):
    path, resource = make_special(tmp_path, kind)
    attempts = forbid_reads(monkeypatch, [path])
    try:
        with pytest.raises(security_error()):
            retained_state_dict(path)
        assert attempts == []
    finally:
        if resource is not None:
            resource.close()


@pytest.mark.parametrize("escape", ["file-symlink", "directory-symlink", "prefix-sibling", "absolute-link"])
@pytest.mark.parametrize("direct_index", [False, True])
def test_reject_symlink_escapes(entry, escape, direct_index, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    outside = tmp_path / "checkpoint-prefix-sibling" / "owned.bin"
    write_weights(outside)
    if escape == "directory-symlink":
        (root / "nested").symlink_to(outside.parent, target_is_directory=True)
        name = "nested/owned.bin"
    else:
        link = root / "shard.bin"
        link.symlink_to(outside if escape == "absolute-link" else "../checkpoint-prefix-sibling/owned.bin")
        name = link.name
    index = index_at(root, {"weight": name})
    attempts = forbid_reads(monkeypatch, [outside])
    with pytest.raises(security_error()):
        invoke(entry, index if direct_index else root)
    assert attempts == []


@pytest.mark.parametrize("extension", ["bin", "safetensors"])
@pytest.mark.parametrize("direct_index", [False, True])
@pytest.mark.parametrize("layout", ["ordinary", "nested", "flat-index", "internal-symlink"])
def test_real_sharded_weights_compatibility(entry, extension, direct_index, layout, tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    state = weights()
    mapping = {}
    for key, value in state.items():
        relative = ("nested/" if layout in ("nested", "internal-symlink") else "") + key + "." + extension
        write_weights(root / relative, {key: value})
        if layout == "internal-symlink":
            linked = key + "." + extension
            (root / linked).symlink_to(relative)
            relative = linked
        mapping[key] = relative
    index = index_at(root, mapping, wrapped=layout != "flat-index")
    target = invoke(entry, index if direct_index else root, strict=True)
    assert_weights(target)


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "model.safetensors"])
@pytest.mark.parametrize("direct", [False, True])
def test_real_whole_checkpoint_compatibility(entry, filename, direct, tmp_path):
    path = tmp_path / filename
    write_weights(path)
    target = invoke(entry, path if direct else tmp_path, strict=True)
    assert_weights(target)


@pytest.mark.parametrize("extension", ["bin", "safetensors"])
@pytest.mark.parametrize("device_map", [None, {"": "cpu"}, {"weight": "cpu", "bias": "disk"}])
def test_final_load_state_dict_real_weights_and_devices(extension, device_map, tmp_path):
    path = tmp_path / ("weights." + extension)
    write_weights(path)
    actual = retained_state_dict(path, device_map=device_map)
    for key, value in weights().items():
        torch.testing.assert_close(actual[key], value)


def cache_fixture(tmp_path, monkeypatch, *, approved=True, nested=False):
    import huggingface_hub.constants
    configured = tmp_path / "approved-hub"
    configured.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(configured))
    hub = configured if approved else tmp_path / "fake-hub"
    repo = hub / "models--qmd--owned-fixture"
    snapshot = repo / "snapshots" / ("a" * 40)
    checkpoint = snapshot / "nested" if nested else snapshot
    checkpoint.mkdir(parents=True)
    (repo / "blobs").mkdir()
    return repo, snapshot, checkpoint


@pytest.mark.parametrize("direct_index", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("extension", ["bin", "safetensors"])
def test_explicitly_approved_hf_cache_blob_links(entry, direct_index, nested, extension, tmp_path, monkeypatch):
    repo, snapshot, root = cache_fixture(tmp_path, monkeypatch, nested=nested)
    blob = repo / "blobs" / ("b" * 64)
    # Blob filenames have no format suffix; the logical snapshot shard name
    # determines the same format as upstream.
    temporary = repo / ("owned." + extension)
    write_weights(temporary)
    temporary.rename(blob)
    link = root / ("shard." + extension)
    link.symlink_to(os.path.relpath(blob, root))
    index = index_at(root, {key: link.name for key in weights()})
    # Real hub layouts also symlink the index itself to an approved blob.
    index_blob = repo / "blobs" / ("c" * 40)
    index.rename(index_blob)
    index.symlink_to(os.path.relpath(index_blob, root))
    assert_weights(invoke(entry, index if direct_index else root, strict=True))


@pytest.mark.parametrize("kind", ["fake-cache", "blob-symlink", "blob-directory-symlink", "different-repo", "nonhash-blob", "external-file"])
def test_cache_exception_is_not_arbitrary_symlink_permission(entry, kind, tmp_path, monkeypatch):
    repo, snapshot, root = cache_fixture(tmp_path, monkeypatch, approved=kind != "fake-cache")
    outside = tmp_path / "owned-outside.bin"
    write_weights(outside)
    blob = repo / "blobs" / ("b" * 64)
    if kind in ("fake-cache", "external-file"):
        target = outside
    elif kind == "blob-symlink":
        blob.symlink_to(outside)
        target = blob
    elif kind == "blob-directory-symlink":
        (repo / "blobs").rmdir()
        (repo / "blobs").symlink_to(tmp_path, target_is_directory=True)
        target = blob
    elif kind == "different-repo":
        target = repo.parent / "models--other--fixture" / "blobs" / ("b" * 64)
        target.parent.mkdir(parents=True)
        target.symlink_to(outside)
    else:
        target = repo / "blobs" / "not-a-hash"
        write_weights(target)
    link = root / "shard.bin"
    link.symlink_to(os.path.relpath(target, root))
    index = index_at(root, {"weight": link.name})
    attempts = forbid_reads(monkeypatch, [outside, target])
    with pytest.raises(security_error()):
        invoke(entry, index)
    assert attempts == []


@pytest.mark.parametrize("kind", ["external-symlink", "fifo", "socket", "directory", "missing", "broken-link"])
@pytest.mark.parametrize("direct_index", [False, True])
def test_mutation_after_preflight_is_rechecked_at_final_boundary(entry, kind, direct_index, tmp_path, monkeypatch):
    import accelerate.utils.checkpoint_security as security
    root = tmp_path / "c"
    root.mkdir()
    shard = root / "shard.bin"
    write_weights(shard)
    outside = tmp_path / "owned-outside.bin"
    write_weights(outside)
    index = index_at(root, {key: shard.name for key in weights()})
    original = security._Root.preflight
    resources = []
    def replace_after_preflight(capability):
        original(capability)
        shard.unlink()
        if kind == "external-symlink":
            shard.symlink_to(outside)
        elif kind == "fifo":
            os.mkfifo(shard)
        elif kind == "socket":
            sock = socket_at(shard)
            resources.append(sock)
        elif kind == "directory":
            shard.mkdir()
        elif kind == "broken-link":
            shard.symlink_to(root / "missing")
    monkeypatch.setattr(security._Root, "preflight", replace_after_preflight)
    attempts = forbid_reads(monkeypatch, [outside])
    try:
        with pytest.raises(security_error()):
            invoke(entry, index if direct_index else root)
        assert attempts == []
    finally:
        for resource in resources:
            resource.close()


@pytest.mark.parametrize("kind", ["fifo", "external-symlink", "directory"])
def test_atomic_pin_race_does_not_data_open_rejected_inode(kind, tmp_path, monkeypatch):
    import accelerate.utils.checkpoint_security as security
    shard = tmp_path / "pytorch_model.bin"
    write_weights(shard)
    outside = tmp_path / "owned-external.bin"
    write_weights(outside)
    attempts = forbid_reads(monkeypatch, [outside])
    swapped = []
    def swap():
        shard.unlink()
        if kind == "fifo":
            os.mkfifo(shard)
        elif kind == "directory":
            shard.mkdir()
        else:
            shard.symlink_to(outside)
        swapped.append(kind)
    if sys.platform == "darwin":
        original = os.link
        def race(source, *args, **kwargs):
            if source == shard.name:
                swap()
            return original(source, *args, **kwargs)
        monkeypatch.setattr(security.os, "link", race)
    else:
        original = os.open
        def race(source, flags, *args, **kwargs):
            if source == shard.name and flags & os.O_PATH:
                swap()
            return original(source, flags, *args, **kwargs)
        monkeypatch.setattr(security.os, "open", race)
    with pytest.raises(security_error()):
        retained_state_dict(shard)
    assert swapped == [kind]
    assert attempts == []


@pytest.mark.parametrize("extension", ["bin", "safetensors"])
@pytest.mark.parametrize("replacement", ["fifo", "external-symlink"])
def test_held_handle_survives_path_replacement_after_pin(extension, replacement, tmp_path, monkeypatch):
    shard = tmp_path / ("shard." + extension)
    write_weights(shard)
    outside = tmp_path / ("owned-other." + extension)
    write_weights(outside, {key: value + 100 for key, value in weights().items()})
    swaps = []
    original = modeling.safe_open if extension == "safetensors" else modeling.torch.load
    def replace_then_deserialize(source, *args, **kwargs):
        if not swaps:
            shard.unlink()
            if replacement == "fifo":
                os.mkfifo(shard)
            else:
                shard.symlink_to(outside)
            swaps.append(str(source))
        if extension == "safetensors":
            assert str(source).startswith(("/dev/fd/", "/proc/self/fd/"))
        else:
            assert hasattr(source, "fileno") and stat.S_ISREG(os.fstat(source.fileno()).st_mode)
        return original(source, *args, **kwargs)
    if extension == "safetensors":
        monkeypatch.setattr(modeling, "safe_open", replace_then_deserialize)
    else:
        monkeypatch.setattr(modeling.torch, "load", replace_then_deserialize)
    actual = retained_state_dict(shard)
    for key, value in weights().items():
        torch.testing.assert_close(actual[key], value)
    assert len(swaps) == 1


def test_held_root_survives_ancestor_directory_replacement(entry, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    write_weights(root / "shard.bin")
    index = index_at(root, {key: "shard.bin" for key in weights()})
    outside = tmp_path / "owned-other-root"
    outside.mkdir()
    write_weights(outside / "shard.bin", {key: value + 100 for key, value in weights().items()})
    original = modeling.load_state_dict
    def replace_root(path, **kwargs):
        root.rename(tmp_path / "original-detached")
        root.symlink_to(outside, target_is_directory=True)
        return original(path, **kwargs)
    monkeypatch.setattr(modeling, "load_state_dict", replace_root)
    assert_weights(invoke(entry, index, strict=True))


def test_symlink_target_mutation_while_resolving_is_contained(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    write_weights(root / "inside.bin")
    outside = tmp_path / "owned-outside.bin"
    write_weights(outside)
    link = root / "link.bin"
    link.symlink_to("inside.bin")
    original = os.readlink
    replaced = []
    def replace_link(path, *args, **kwargs):
        if path == link.name and not replaced:
            link.unlink()
            link.symlink_to(outside)
            replaced.append(True)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "readlink", replace_link)
    attempts = forbid_reads(monkeypatch, [outside])
    with pytest.raises(security_error()):
        retained_state_dict(link)
    assert replaced and not attempts


def test_nested_directory_swap_during_walk_does_not_follow_escape(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    write_weights(root / "nested" / "shard.bin")
    outside = tmp_path / "owned-other"
    write_weights(outside / "shard.bin")
    original = os.open
    swapped = []
    def replace_directory(name, flags, *args, **kwargs):
        if name == "nested" and flags & os.O_DIRECTORY and not swapped:
            (root / "nested").rename(root / "original-nested")
            (root / "nested").symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return original(name, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", replace_directory)
    attempts = forbid_reads(monkeypatch, [outside / "shard.bin"])
    with pytest.raises(security_error()):
        retained_state_dict(root / "nested" / "shard.bin")
    assert swapped and not attempts


def test_root_swap_while_establishing_capability_is_rejected(entry, tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    write_weights(root / "pytorch_model.bin")
    outside = tmp_path / "owned-other"
    write_weights(outside / "pytorch_model.bin")
    original = os.path.realpath
    swapped = []
    def replace_root(path, *args, **kwargs):
        if os.fspath(path) == str(root) and not swapped:
            root.rename(tmp_path / "original-root")
            root.symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os.path, "realpath", replace_root)
    attempts = forbid_reads(monkeypatch, [outside / "pytorch_model.bin"])
    with pytest.raises(security_error(), match="root changed"):
        invoke(entry, root)
    assert swapped and not attempts


def test_macos_cross_device_failure_does_not_fallback_to_unsafe_open(tmp_path, monkeypatch):
    if sys.platform != "darwin":
        pytest.skip("Darwin-specific fail-closed staging contract")
    import errno
    shard = tmp_path / "shard.bin"
    write_weights(shard)
    def no_link(*args, **kwargs):
        raise OSError(errno.EXDEV, "simulated cross-device hardlink")
    monkeypatch.setattr(os, "link", no_link)
    attempts = forbid_reads(monkeypatch)
    with pytest.raises(security_error(), match="same-filesystem"):
        retained_state_dict(shard)
    assert not attempts


def test_unsupported_platform_fails_closed(tmp_path, monkeypatch):
    import accelerate.utils.checkpoint_security as security
    shard = tmp_path / "shard.bin"
    write_weights(shard)
    monkeypatch.setattr(security.sys, "platform", "win32")
    with pytest.raises(security_error(), match="requires Linux or macOS"):
        retained_state_dict(shard)


def test_installed_wheel_source_record_and_public_aliases():
    import accelerate.utils.checkpoint_security as security
    provenance = json.loads((VENDOR / "provenance.json").read_text())
    wheel = VENDOR / provenance["artifact"]["filename"]
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == provenance["artifact"]["sha256"]
    site = Path(accelerate.__file__).parent.parent
    for member in ["accelerate/__init__.py", "accelerate/utils/modeling.py", "accelerate/utils/checkpoint_security.py"]:
        assert hashlib.sha256((site / member).read_bytes()).hexdigest() == provenance["wheel_members"][member]
    assert security.__file__ == str(site / "accelerate/utils/checkpoint_security.py")
    assert all(entry is retained_in_model for _, entry in ENTRIES[:4])
    assert ENTRIES[4][1] is ENTRIES[5][1] is retained_dispatch
    assert list(inspect.signature(retained_state_dict).parameters) == ["checkpoint_file", "device_map"]
    with zipfile.ZipFile(VENDOR / "upstream" / "accelerate-1.15.0-py3-none-any.whl") as old, zipfile.ZipFile(wheel) as new:
        assert old.read("accelerate-1.15.0.dist-info/licenses/LICENSE") == new.read("accelerate-1.15.0+qmd.2.dist-info/licenses/LICENSE")
        changed = set(provenance["patched_upstream_source_members"])
        for name in old.namelist():
            if name.startswith("accelerate/") and name not in changed:
                assert old.read(name) == new.read(name), name


def test_builder_reproduces_exact_wheel_and_provenance(tmp_path):
    result = subprocess.run(
        [sys.executable, str(FINETUNE / "tools/build_patched_accelerate.py"), "--output-dir", str(tmp_path)],
        text=True, capture_output=True, env=dict(os.environ), timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    provenance = json.loads((VENDOR / "provenance.json").read_text())
    name = provenance["artifact"]["filename"]
    assert (tmp_path / name).read_bytes() == (VENDOR / name).read_bytes()
    assert (tmp_path / "provenance.json").read_bytes() == (VENDOR / "provenance.json").read_bytes()


def test_builder_rejects_wrong_upstream_hash(tmp_path):
    bad = tmp_path / "untrusted.whl"
    bad.write_bytes(b"not the pinned official wheel")
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [sys.executable, str(FINETUNE / "tools/build_patched_accelerate.py"), "--upstream", str(bad), "--output-dir", str(output)],
        capture_output=True, text=True, env=dict(os.environ), timeout=30,
    )
    assert result.returncode != 0 and "SHA256 mismatch" in result.stderr
    assert not output.exists()


def test_upstream_model_device_offload_loop_is_ast_identical():
    with zipfile.ZipFile(VENDOR / "upstream" / "accelerate-1.15.0-py3-none-any.whl") as archive:
        old_tree = ast.parse(archive.read("accelerate/utils/modeling.py"))
    new_tree = ast.parse(Path(modeling.__file__).read_text())
    old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint_in_model")
    new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == old.name)
    start = next(i for i, node in enumerate(old.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "offload_index" for target in node.targets))
    assert isinstance(new.body[-1], ast.With)
    assert ast.dump(ast.Module(body=old.body[start:], type_ignores=[])) == ast.dump(ast.Module(body=new.body[-1].body, type_ignores=[]))
    assert ast.dump(old.args) == ast.dump(new.args)


@pytest.mark.parametrize("direct_index", [False, True])
def test_nonreading_broadcast_rank_preserves_index_only_compatibility(entry, direct_index, tmp_path, monkeypatch):
    # CPU-only branch contract, NOT a claim of a real distributed/GPU run. The
    # real upstream loader delegates rank-0 broadcast to Torch; only that final
    # collective is intercepted here. Non-reading ranks need no shard files.
    import torch.distributed.checkpoint.state_dict as distributed_state
    root = tmp_path / "checkpoint"
    root.mkdir()
    index = index_at(root, {key: "unavailable-on-this-rank.bin" for key in weights()})
    calls = []
    monkeypatch.setattr(modeling.dist, "get_rank", lambda: 1)
    def capture(target, state, *, options):
        assert state == {} and options.broadcast_from_rank0
        calls.append(True)
    monkeypatch.setattr(distributed_state, "set_model_state_dict", capture)
    attempts = forbid_reads(monkeypatch)
    target = model()
    entry(target, index if direct_index else root, device_map=None, broadcast_from_rank0=True)
    assert calls == [True] and attempts == []


@pytest.mark.parametrize("extension", ["bin", "safetensors"])
def test_device_map_none_preserves_upstream_loading(entry, extension, tmp_path):
    checkpoint = tmp_path / ("weights." + extension)
    write_weights(checkpoint)
    target = model()
    entry(target, checkpoint, device_map=None, strict=True)
    assert_weights(target)


@pytest.mark.parametrize("extension", ["bin", "safetensors"])
def test_dtype_and_cpu_state_dict_offload_are_preserved(entry, extension, tmp_path):
    checkpoint = tmp_path / ("weights." + extension)
    write_weights(checkpoint)
    target = invoke(entry, checkpoint, dtype="float16", offload_state_dict=True, strict=True)
    assert_weights(target, {key: value.to(torch.float16) for key, value in weights().items()})


@pytest.mark.parametrize("kind", ["fifo", "directory", "external-symlink"])
def test_all_shards_validated_before_first_deserializer(entry, kind, tmp_path, monkeypatch):
    root = tmp_path / "c"
    root.mkdir()
    write_weights(root / "a-valid.bin", {"weight": weights()["weight"]})
    rejected = root / "z-invalid.bin"
    outside = tmp_path / "owned-outside.bin"
    write_weights(outside)
    if kind == "fifo":
        os.mkfifo(rejected)
    elif kind == "directory":
        rejected.mkdir()
    else:
        rejected.symlink_to(outside)
    index = index_at(root, {"weight": "a-valid.bin", "bias": rejected.name})
    attempts = forbid_reads(monkeypatch, [rejected, outside])
    with pytest.raises(security_error()):
        invoke(entry, index)
    assert attempts == []


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "model.safetensors"])
@pytest.mark.parametrize("direct", [False, True])
def test_approved_cache_whole_model_links(entry, filename, direct, tmp_path, monkeypatch):
    repo, snapshot, root = cache_fixture(tmp_path, monkeypatch)
    original = root / filename
    write_weights(original)
    blob = repo / "blobs" / ("d" * 64)
    original.rename(blob)
    original.symlink_to(os.path.relpath(blob, root))
    assert_weights(invoke(entry, original if direct else root, strict=True))


def test_safetensors_uses_held_descriptor_mmap_not_whole_file_copy(tmp_path, monkeypatch):
    import accelerate.utils.checkpoint_security as security
    checkpoint = tmp_path / "weights.safetensors"
    write_weights(checkpoint)
    original = security.os.fdopen
    class NoBufferedRead:
        def __init__(self, file):
            self.file = file
        def __getattr__(self, name):
            return getattr(self.file, name)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.file.__exit__(*args)
        def read(self, *args, **kwargs):
            raise AssertionError("Safetensors must mmap the pinned FD, not copy all weights into bytes")
        readinto = read
    def guarded_fdopen(*args, **kwargs):
        return NoBufferedRead(original(*args, **kwargs))
    monkeypatch.setattr(security.os, "fdopen", guarded_fdopen)
    actual = retained_state_dict(checkpoint)
    for key, value in weights().items():
        torch.testing.assert_close(actual[key], value)


def test_deserializer_oserror_not_mislabeled_as_security_error(entry, tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.bin"
    write_weights(checkpoint)
    expected = OSError("owned deserializer error sentinel")
    def fail(*args, **kwargs):
        raise expected
    monkeypatch.setattr(modeling.torch, "load", fail)
    with pytest.raises(OSError) as caught:
        invoke(entry, checkpoint)
    assert caught.value is expected


def test_handles_are_closed_after_success_and_rejection(tmp_path):
    descriptor_directory = "/proc/self/fd" if sys.platform == "linux" else "/dev/fd"
    checkpoint = tmp_path / "weights.bin"
    write_weights(checkpoint)
    rejected = tmp_path / "rejected.bin"
    os.mkfifo(rejected)
    retained_state_dict(checkpoint)  # Warm up normal Torch imports/caches.
    before = len(os.listdir(descriptor_directory))
    for _ in range(20):
        retained_state_dict(checkpoint)
        with pytest.raises(security_error()):
            retained_state_dict(rejected)
    assert len(os.listdir(descriptor_directory)) == before


def _live_descriptor_set():
    directory = "/proc/self/fd" if sys.platform == "linux" else "/dev/fd"
    result = set()
    for name in os.listdir(directory):
        try:
            descriptor = int(name)
            os.fstat(descriptor)
            result.add(descriptor)
        except (ValueError, OSError):
            pass  # The descriptor used by listdir may already have closed.
    return result


def test_read_monitor_checks_fd_target_identity_without_exempting_fd_aliases(tmp_path, monkeypatch):
    good, forbidden = tmp_path / "index.json", tmp_path / "owned-forbidden.bin"
    good.write_text("{}")
    forbidden.write_bytes(b"owned sentinel")
    with good.open("rb") as readable, forbidden.open("rb") as denied:
        attempts = forbid_reads(monkeypatch, [forbidden])
        prefix = "/proc/self/fd" if sys.platform == "linux" else "/dev/fd"
        with contextlib.closing(os.fdopen(os.open(f"{prefix}/{readable.fileno()}", os.O_RDONLY), "rb")) as file:
            assert file.read() == b"{}"
        with pytest.raises(AssertionError, match="Rejected shard reached a DATA open"):
            os.open(f"{prefix}/{denied.fileno()}", os.O_RDONLY)
        assert attempts == [f"{prefix}/{denied.fileno()}"]


@pytest.mark.parametrize("operation", ["public-load", "preflight", "validate", "locate", "open"])
def test_retired_capability_is_revoked_before_recycled_fd_access(tmp_path, monkeypatch, operation):
    security = importlib.import_module("accelerate.utils.checkpoint_security")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_weights(first / "pytorch_model.bin")
    write_weights(second / "pytorch_model.bin")
    with security.checkpoint_files(first, "pytorch_model.bin", "model.safetensors") as capabilities:
        capability = capabilities[0]
        old_fd = capability.root.fd
    owned = []
    try:
        while old_fd not in owned and len(owned) < 32:
            owned.append(os.open(second, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC))
        assert old_fd in owned, "Fixture must establish real descriptor recycling without dup2"
        attempts = forbid_reads(monkeypatch, [second / "pytorch_model.bin"])
        with pytest.raises(security.CheckpointSecurityError, match="closed|expired|revoked"):
            if operation == "public-load":
                retained_state_dict(capability)
            elif operation == "preflight":
                capability.root.preflight()
            elif operation == "validate":
                capability.root.validate(capability.name)
            elif operation == "locate":
                with capability.root.locate(capability.name):
                    pass
            else:
                with capability.root.open(capability.name):
                    pass
        assert attempts == []
        capability.root.close()  # Idempotence must not close the recycled owner.
        assert stat.S_ISDIR(os.fstat(old_fd).st_mode)
    finally:
        for descriptor in owned:
            with contextlib.suppress(OSError):
                os.close(descriptor)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS hardlink staging cleanup")
@pytest.mark.parametrize("stage", ["unlink", "temporary-directory-exit"])
def test_staging_cleanup_failure_retains_fd_ownership_until_transfer(tmp_path, monkeypatch, stage):
    security = importlib.import_module("accelerate.utils.checkpoint_security")
    checkpoint = tmp_path / "weights.bin"
    write_weights(checkpoint)
    retained_state_dict(checkpoint)  # Finish import/caching work before the snapshot.
    before = _live_descriptor_set()
    original_unlink = os.unlink
    original_cleanup = security.tempfile.TemporaryDirectory.cleanup
    injected = []
    def unlink(path, *args, **kwargs):
        if stage == "unlink" and path == "pinned" and kwargs.get("dir_fd") is not None and not injected:
            injected.append(stage)
            raise OSError(errno.EIO, "owned fixture staging unlink failure")
        return original_unlink(path, *args, **kwargs)
    def cleanup(directory):
        original_cleanup(directory)
        if stage == "temporary-directory-exit" and not injected:
            injected.append(stage)
            raise OSError(errno.EIO, "owned fixture staging exit failure")
    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", unlink)
        patch.setattr(security.tempfile.TemporaryDirectory, "cleanup", cleanup)
        with pytest.raises(security.CheckpointSecurityError, match="staging"):
            retained_state_dict(checkpoint)
    leaked = _live_descriptor_set() - before
    try:
        assert injected == [stage]
        assert not leaked, f"Staging failure leaked descriptors: {sorted(leaked)}"
    finally:
        # A red regression may leak an owned fixture FD. Never close an unrelated descriptor.
        expected = os.stat(checkpoint)
        for descriptor in leaked:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
                os.close(descriptor)


@pytest.mark.parametrize("stage", ["wrapper-construction", "directory-context-exit"])
def test_open_context_keeps_descriptor_ownership_on_setup_failure(tmp_path, monkeypatch, stage):
    security = importlib.import_module("accelerate.utils.checkpoint_security")
    checkpoint = tmp_path / "weights.bin"
    write_weights(checkpoint)
    root = security._Root(tmp_path)
    pinned = []
    original_pin, original_locate = security._pin_regular, root.locate
    before = _live_descriptor_set()
    def pin(*args):
        descriptor = original_pin(*args)
        pinned.append(descriptor)
        return descriptor
    @contextlib.contextmanager
    def locate(name):
        with original_locate(name) as target:
            yield target
        if pinned:
            raise OSError(errno.EIO, "owned fixture directory-context cleanup failure")
    def fdopen(*args, **kwargs):
        raise OSError(errno.EIO, "owned fixture file-wrapper construction failure")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(security, "_pin_regular", pin)
            if stage == "wrapper-construction":
                patch.setattr(os, "fdopen", fdopen)
            else:
                patch.setattr(root, "locate", locate)
            with pytest.raises(OSError, match="owned fixture"):
                with root.open(checkpoint.name):
                    pytest.fail("Setup failure must precede the caller's load")
        assert pinned
        assert _live_descriptor_set() == before
        for descriptor in pinned:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        # Clean up only these test-owned handles on an intentionally red run.
        expected = os.stat(checkpoint)
        for descriptor in pinned:
            with contextlib.suppress(OSError):
                info = os.fstat(descriptor)
                if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
                    os.close(descriptor)
        root.close()


def test_peft_trl_indirect_runtime_and_retained_aliases(tiny_model, tiny_tokenizer, tmp_path, monkeypatch):
    """Real PEFT reload and TRL construction, then both loaders on their models.

    These installed PEFT/TRL versions do not themselves invoke either vulnerable
    checkpoint API. Do not invent an indirect exploit path: check the real
    transitive runtime, and exercise retained guarded APIs with actual models
    produced by those libraries. Their independent adapter loaders are outside
    the Accelerate patch's security claim.
    """
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    clean_base = copy.deepcopy(tiny_model)
    calls = set()
    def profile(frame, event, arg):
        if event == "call" and "/site-packages/accelerate/" in frame.f_code.co_filename:
            calls.add(frame.f_code.co_name)
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        args = SFTConfig(
            output_dir=str(tmp_path / "adapter"), use_cpu=True, bf16=False, report_to="none",
            max_length=32, gradient_checkpointing=False, save_strategy="no", disable_tqdm=True,
            dataset_num_proc=None,
        )
        trainer = SFTTrainer(
            model=tiny_model, args=args, processing_class=tiny_tokenizer,
            train_dataset=Dataset.from_list([{"text": "user: fixture query\nassistant: fixture tokens"}]),
            peft_config=LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
        )
        trainer.save_model()
        reloaded = PeftModel.from_pretrained(clean_base, args.output_dir, local_files_only=True, torch_device="cpu")
    finally:
        sys.setprofile(previous)
    assert isinstance(trainer.model, PeftModel) and isinstance(reloaded, PeftModel)
    assert calls, "Expected actual indirect calls into the installed Accelerate package"
    assert "load_checkpoint_in_model" not in calls and "load_checkpoint_and_dispatch" not in calls
    # Both public APIs and the aliases retained before the transitive imports
    # work with real PEFT/TRL objects, without monkeypatching any package alias.
    for index, (entry, target) in enumerate([(retained_in_model, trainer.model), (retained_dispatch, reloaded)]):
        state = {key: value.detach().clone() for key, value in target.state_dict().items()}
        checkpoint = tmp_path / f"peft-trl-{index}.safetensors"
        write_weights(checkpoint, state)
        invoke(entry, checkpoint, target=target, strict=True)
        assert_weights(target, state)
    malformed = tmp_path / "malicious.index.json"
    malformed.write_text(json.dumps({"weight_map": {"weight": "../owned-only.bin"}}))
    attempts = forbid_reads(monkeypatch)
    for entry, target in [(retained_in_model, trainer.model), (retained_dispatch, reloaded)]:
        with pytest.raises(security_error()):
            invoke(entry, malformed, target=target)
    assert attempts == []
