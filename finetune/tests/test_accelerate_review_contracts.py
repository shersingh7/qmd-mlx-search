"""Maintained review-derived CPU contracts against the installed wheel only.

Derived from the retained independent qmd.1 review. The parent updated the
installed-version contract and FD-alias monitor; original evidence is immutable
under vendor/accelerate/evidence/. This file is not a new independent verdict.
"""
from __future__ import annotations
import builtins
import contextlib
import contextvars
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
import accelerate
import accelerate.big_modeling as big
import accelerate.utils as utils
import accelerate.utils.bnb as bnb
import accelerate.utils.modeling as modeling
import accelerate.utils.checkpoint_security as security
import huggingface_hub.constants as hub_constants

ENTRIES = [accelerate.load_checkpoint_in_model, accelerate.load_checkpoint_and_dispatch, bnb.load_checkpoint_in_model]
LOADERS = [modeling.load_state_dict, utils.load_state_dict]

@pytest.fixture(autouse=True)
def contract():
    assert accelerate.__version__ == '1.15.0+qmd.2'
    assert 'site-packages' in accelerate.__file__
    torch.set_num_threads(1)


def state(offset=0):
    return {'weight': torch.arange(4, dtype=torch.float32).reshape(2, 2) + offset, 'bias': torch.tensor([8., 9.]) + offset}


def save(path, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.safetensors':
        save_file(state(offset), str(path), metadata={'format': 'pt'})
    else:
        torch.save(state(offset), path)


def index(root, name='shard.bin'):
    out = root / 'weights.index.json'
    out.write_text(json.dumps({'weight_map': {key: name for key in state()}}))
    return out


def invoke(entry, path):
    result = torch.nn.Linear(2, 2)
    entry(result, str(path), device_map={'': 'cpu'}, strict=True)
    for key, value in state().items():
        torch.testing.assert_close(result.state_dict()[key], value)
    return result


def guard_no_data(monkeypatch, *paths):
    """Identify forbidden regular fixtures by inode, and reject all special reads."""
    real_open, real_builtin = os.open, builtins.open
    forbidden = {(s.st_dev, s.st_ino) for p in paths if p.exists() for s in [os.stat(p)]}
    attempts = []
    def check(path, dir_fd=None):
        try:
            value = os.fspath(path)
            match = re.fullmatch(r'/(?:proc/self/fd|dev/fd)/([0-9]+)', value) if isinstance(value, str) else None
            info = os.fstat(int(match[1])) if match else os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        except (OSError, TypeError):
            return
        if (info.st_dev, info.st_ino) in forbidden or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            attempts.append(str(path))
            raise AssertionError('forbidden fixture reached data-open: ' + str(path))
    def opened(path, flags, mode=0o777, *, dir_fd=None):
        if not flags & (os.O_DIRECTORY | getattr(os, 'O_PATH', 0)):
            check(path, dir_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)
    def builtin(path, *args, **kwargs):
        if not isinstance(path, int):
            check(path)
        return real_builtin(path, *args, **kwargs)
    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(builtins, 'open', builtin)
    return attempts


@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('direct', [False, True])
def test_bnb_alias_and_plain_strings_reject_traversal(entry, direct, tmp_path, monkeypatch):
    root = tmp_path / 'root'; root.mkdir()
    external = tmp_path / 'owned.bin'; save(external)
    pointer = index(root, '../owned.bin')
    attempts = guard_no_data(monkeypatch, external)
    with pytest.raises(security.CheckpointSecurityError):
        invoke(entry, pointer if direct else root)
    assert not attempts
    assert bnb.load_checkpoint_in_model is modeling.load_checkpoint_in_model


@pytest.mark.parametrize('loader', LOADERS)
@pytest.mark.parametrize('kind', ['fifo', 'external-symlink', 'broken-link'])
def test_final_plain_string_loader_aliases(loader, kind, tmp_path, monkeypatch):
    root = tmp_path / 'r'; root.mkdir()
    external = tmp_path / 'owned.bin'; save(external)
    leaf = root / 'shard.bin'
    if kind == 'fifo': os.mkfifo(leaf)
    elif kind == 'external-symlink': leaf.symlink_to(external)
    else: leaf.symlink_to('missing')
    attempts = guard_no_data(monkeypatch, external)
    with pytest.raises(security.CheckpointSecurityError): loader(str(leaf))
    assert not attempts


@pytest.mark.parametrize('entry', ENTRIES)
@pytest.mark.parametrize('filename', ['weights.index.json', 'pytorch_model.bin', 'model.safetensors'])
@pytest.mark.parametrize('direct', [False, True])
def test_external_index_or_whole_model_symlink_is_not_authority(entry, filename, direct, tmp_path, monkeypatch):
    root = tmp_path / 'root'; root.mkdir()
    outside = tmp_path / ('owned.' + ('safetensors' if filename.endswith('safetensors') else 'bin'))
    if filename.endswith('json'): outside.write_text('{"weight_map":{"weight":"shard.bin"}}')
    else: save(outside)
    target = root / filename; target.symlink_to(outside)
    attempts = guard_no_data(monkeypatch, outside)
    with pytest.raises(security.CheckpointSecurityError):
        invoke(entry, target if direct else root)
    assert not attempts


@pytest.mark.parametrize('revision', ['e'*40, 'e'*64])
@pytest.mark.parametrize('entry', ENTRIES)
def test_hub_64_revision_and_deep_subfolder(entry, revision, tmp_path, monkeypatch):
    hub = tmp_path / 'hub'; root = hub / 'models--owner--model' / 'snapshots' / revision / 'deep' / 'subdir'
    root.mkdir(parents=True)
    blob = hub / 'models--owner--model' / 'blobs' / ('f'*64)
    save(blob)
    (root / 'shard.bin').symlink_to(os.path.relpath(blob, root))
    monkeypatch.setattr(hub_constants, 'HF_HUB_CACHE', str(hub))
    invoke(entry, index(root))


@pytest.mark.parametrize('kind', ['uppercase-revision', 'short-revision', 'invalid-repo', 'hub-prefix-sibling', 'symlinked-repo', 'uppercase-blob', 'nested-blob'])
def test_hub_bounds_beyond_supplied_matrix(kind, tmp_path, monkeypatch):
    hub = tmp_path / 'hub'; hub.mkdir()
    actual = tmp_path / 'hub-sibling' if kind == 'hub-prefix-sibling' else hub
    revision = 'A'*40 if kind == 'uppercase-revision' else ('a'*39 if kind == 'short-revision' else 'a'*40)
    repo = actual / ('datasets--owner--model' if kind == 'invalid-repo' else 'models--owner--model')
    if kind == 'symlinked-repo':
        actual_repo = tmp_path / 'owned-external-repo'; actual_repo.mkdir()
        repo.symlink_to(actual_repo, target_is_directory=True)
    root = repo / 'snapshots' / revision; root.mkdir(parents=True)
    blob = repo / 'blobs' / ('B'*64 if kind == 'uppercase-blob' else ('nested/' if kind == 'nested-blob' else '')+'b'*64)
    save(blob)
    (root / 'shard.bin').symlink_to(os.path.relpath(blob, root))
    monkeypatch.setattr(hub_constants, 'HF_HUB_CACHE', str(hub))
    attempts = guard_no_data(monkeypatch, blob)
    with pytest.raises(security.CheckpointSecurityError): invoke(ENTRIES[0], index(root))
    assert not attempts


@pytest.mark.parametrize('extension', ['bin', 'safetensors'])
def test_internal_symlinked_directories_work_for_indexed_shards(extension, tmp_path):
    root = tmp_path / 'r'; root.mkdir()
    save(root / 'physical' / ('shard.'+extension))
    (root / 'logical').symlink_to('physical', target_is_directory=True)
    invoke(ENTRIES[0], index(root, 'logical/shard.'+extension))


@pytest.mark.parametrize('loader', LOADERS)
def test_cycle_and_dangling_symlink_never_opened(loader, tmp_path, monkeypatch):
    (tmp_path / 'a.bin').symlink_to('b.bin')
    (tmp_path / 'b.bin').symlink_to('a.bin')
    attempts = guard_no_data(monkeypatch)
    with pytest.raises(security.CheckpointSecurityError, match='too many'):
        loader(str(tmp_path / 'a.bin'))
    assert not attempts


@pytest.mark.parametrize('kind', ['character', 'block'])
@pytest.mark.parametrize('extension', ['bin', 'safetensors'])
def test_atomic_mac_pin_simulated_device_after_precheck(kind, extension, tmp_path, monkeypatch):
    """Never touch actual devices: change staging inode's synthetic stat type."""
    if sys.platform != 'darwin': pytest.skip('native macOS hardlink branch')
    leaf = tmp_path / ('shard.'+extension); save(leaf)
    real_link, real_stat, real_open = os.link, os.stat, os.open
    staged = []
    def link(*args, **kwargs):
        result = real_link(*args, **kwargs)
        staged.append(True)
        return result
    def info(path, *args, **kwargs):
        data = real_stat(path, *args, **kwargs)
        if str(path) == 'pinned' and staged:
            values = {n: getattr(data, n) for n in dir(data) if n.startswith('st_')}
            values['st_mode'] = (stat.S_IFCHR if kind == 'character' else stat.S_IFBLK) | 0o600
            return SimpleNamespace(**values)
        return data
    attempts = []
    def opened(path, flags, *args, **kwargs):
        if str(path) == 'pinned' and not flags & os.O_DIRECTORY:
            attempts.append(str(path)); raise AssertionError('simulated device was data-opened')
        return real_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'link', link); monkeypatch.setattr(os, 'stat', info); monkeypatch.setattr(os, 'open', opened)
    with pytest.raises(security.CheckpointSecurityError): modeling.load_state_dict(str(leaf))
    assert staged and not attempts


@pytest.mark.parametrize('extension', ['bin', 'safetensors'])
@pytest.mark.parametrize('device_map', [None, {'': 'cpu'}, {'weight':'cpu', 'bias':'disk'}])
def test_deserializer_exception_closes_all_root_and_file_fds(extension, device_map, tmp_path, monkeypatch):
    leaf = tmp_path / ('shard.'+extension); save(leaf)
    expected = RuntimeError('heldout deserializer sentinel')
    def fail(*args, **kwargs): raise expected
    before = len(os.listdir('/dev/fd' if sys.platform == 'darwin' else '/proc/self/fd'))
    if extension == 'bin': monkeypatch.setattr(torch, 'load', fail)
    else: monkeypatch.setattr(modeling, 'safe_open', fail)
    for _ in range(30):
        with pytest.raises(RuntimeError) as caught: modeling.load_state_dict(str(leaf), device_map=device_map)
        assert caught.value is expected
    after = len(os.listdir('/dev/fd' if sys.platform == 'darwin' else '/proc/self/fd'))
    assert after == before


@pytest.mark.parametrize('entry', ENTRIES)
def test_reentrant_unrelated_load_cannot_reuse_root_authority(entry, tmp_path, monkeypatch):
    """Exercise nested loader and copied Context state without reusing any validator."""
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir(); second.mkdir(); save(first / 'shard.bin'); save(second / 'shard.bin', 500)
    pointer = index(first)
    actual_load = modeling._load_state_dict_from_handle
    reentered = []
    def nested(opened, device_map=None):
        if not reentered:
            reentered.append(True)
            independent = contextvars.copy_context().run(utils.load_state_dict, str(second / 'shard.bin'))
            for key, value in state(500).items(): torch.testing.assert_close(independent[key], value)
        return actual_load(opened, device_map=device_map)
    monkeypatch.setattr(modeling, '_load_state_dict_from_handle', nested)
    invoke(entry, pointer)
    assert reentered


def test_schema_failure_and_rejection_do_not_leave_context_or_descriptors(tmp_path):
    root = tmp_path / 'r'; root.mkdir()
    bad = root / 'bad.index.json'
    bad.write_text('{"weight_map":{"weight":"x","weight":"y"}}')
    leaf = root / 'shard.bin'; save(leaf)
    before = len(os.listdir('/dev/fd' if sys.platform == 'darwin' else '/proc/self/fd'))
    for _ in range(30):
        with pytest.raises(security.CheckpointSecurityError): invoke(ENTRIES[0], bad)
        result = contextvars.copy_context().run(utils.load_state_dict, str(leaf))
        for key, value in state().items(): torch.testing.assert_close(result[key], value)
    assert len(os.listdir('/dev/fd' if sys.platform == 'darwin' else '/proc/self/fd')) == before
