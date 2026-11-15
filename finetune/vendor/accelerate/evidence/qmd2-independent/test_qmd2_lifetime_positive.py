"""Independent positive lifetime contracts; owned fixtures and injected errors only.

This is a NEW acceptance runner, not the historical qmd.1 defect reproducer.
The prior negative scripts/receipts are never run, edited, or relabeled green.
"""
from __future__ import annotations
import contextlib
import errno
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import stat
import sys
from unittest.mock import patch

# Refuse network even during imports; all checkpoint data is created below.
def no_network(*args, **kwargs):
    raise AssertionError('Independent lifetime probes must remain offline')
socket.create_connection = no_network
socket.socket.connect = no_network

import pytest
import torch
from safetensors.torch import save_file
import accelerate
import accelerate.utils.checkpoint_security as security
from accelerate.utils.modeling import load_state_dict

BASE = Path(__file__).resolve().parent
REAL_CLOSE = os.close
REAL_FSTAT = os.fstat
OBSERVATIONS = []


@pytest.fixture(scope='session', autouse=True)
def isolated_runtime(request):
    assert sys.platform == 'darwin', 'This receipt is a native macOS run, not a mocked platform'
    assert accelerate.__version__ == importlib.metadata.version('accelerate') == '1.15.0+qmd.2'
    assert Path(os.environ['HOME']).is_relative_to(BASE)
    assert os.environ['TMPDIR'] == '/Users/shersingh/.hermes/cache/scratch'
    for key, value in {'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '', 'ACCELERATE_USE_CPU': 'true'}.items():
        assert os.environ[key] == value, key
    torch.set_num_threads(1)
    yield
    (BASE / 'independent-probe-observations.json').write_text(json.dumps({
        'scope': 'qmd.2 positive descriptor-lifetime acceptance; owned synthetic fixtures only',
        'platform': sys.platform,
        'collected': request.session.testscollected,
        'failed': request.session.testsfailed,
        'observations': OBSERVATIONS,
    }, indent=2) + '\n')


def live_fds():
    # Metadata only: never read ambient descriptor contents or host devices.
    result = set()
    for name in os.listdir('/dev/fd'):
        try:
            fd = int(name)
            REAL_FSTAT(fd)
            result.add(fd)
        except (OSError, ValueError):
            pass  # listdir's temporary descriptor may already be closed.
    return result


def assert_closed(fd):
    with pytest.raises(OSError) as error:
        REAL_FSTAT(fd)
    assert error.value.errno == errno.EBADF


def inode(path):
    info = path.stat()
    return info.st_dev, info.st_ino


def close_if_fixture(fd, expected):
    # Failure cleanup may close ONLY a proven descriptor owned by this probe.
    try:
        info = REAL_FSTAT(fd)
    except OSError as error:
        assert error.errno == errno.EBADF
        return
    assert (info.st_dev, info.st_ino) == expected
    REAL_CLOSE(fd)


def write_weights(path, marker=17):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'marker': torch.tensor([float(marker)])}
    if path.suffix == '.safetensors':
        save_file(state, str(path), metadata={'format': 'pt'})
    else:
        torch.save(state, path)
    return state


def operation_for(capability, operation):
    root = capability.root
    if operation == 'public-load':
        return load_state_dict(capability)
    if operation == 'preflight':
        return root.preflight()
    if operation == 'validate':
        return root.validate(capability.name)
    if operation == 'locate':
        with root.locate(capability.name):
            pytest.fail('Expired root reached locate body')
    else:
        with root.open(capability.name):
            pytest.fail('Expired root reached open body')


@pytest.mark.parametrize('consume_preflight', [False, True], ids=['pending', 'consumed'])
@pytest.mark.parametrize('operation', ['public-load', 'preflight', 'validate', 'locate', 'open'])
def test_retired_capability_rejects_before_any_filesystem_call(tmp_path, operation, consume_preflight):
    first, second = tmp_path / 'first', tmp_path / 'second'
    write_weights(first / 'pytorch_model.bin', 1)
    write_weights(second / 'pytorch_model.bin', 999)
    before = live_fds()
    with security.checkpoint_files(first, 'pytorch_model.bin', 'model.safetensors') as capabilities:
        capability = capabilities[0]
        old_fd = capability.root.fd
        if consume_preflight:
            assert load_state_dict(capability)['marker'].item() == 1
            assert capability.root._pending_shards is None
        else:
            assert capability.root._pending_shards == ('pytorch_model.bin',)
    assert capability.root.fd == -1
    assert_closed(old_fd)
    owned = []
    calls = []
    def unexpected(*args, **kwargs):
        calls.append('filesystem-or-deserializer')
        raise AssertionError('A retired root must reject before I/O, not merely fail an OS operation')
    try:
        while old_fd not in owned and len(owned) < 64:
            owned.append(os.open(second, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC))
        assert old_fd in owned, 'Establish real descriptor reuse with ordinary opens, never dup2'
        recycled = REAL_FSTAT(old_fd)
        assert (recycled.st_dev, recycled.st_ino) == inode(second)
        with contextlib.ExitStack() as stack:
            for target in ['os.stat', 'os.open', 'os.fdopen', 'accelerate.utils.modeling.torch.load', 'accelerate.utils.modeling.safe_open', 'accelerate.utils.modeling.safe_load_file']:
                stack.enter_context(patch(target, unexpected))
            with pytest.raises(security.CheckpointSecurityError, match='closed'):
                operation_for(capability, operation)
        assert calls == []
        capability.root.close()
        capability.root.close()
        assert REAL_FSTAT(old_fd).st_ino == recycled.st_ino
    finally:
        for fd in owned:
            close_if_fixture(fd, inode(second))
    assert live_fds() == before
    OBSERVATIONS.append({'contract': 'R1-expired-root', 'operation': operation, 'preflight_consumed': consume_preflight, 'numeric_fd_recycled': old_fd, 'io_attempts': len(calls), 'closed_root_sentinel': capability.root.fd, 'fd_set_restored': True})


def test_root_revokes_before_real_close_and_reentrant_recycle(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    before = live_fds()
    root = security._Root(first)
    old_fd = root.fd
    recycled = []
    events = []
    def close(fd):
        assert fd == old_fd
        events.append(root.fd)
        assert root.fd == -1, 'Revocation must precede the OS close/recycle boundary'
        REAL_CLOSE(fd)
        # Earlier component-walk closes can leave a lower-numbered free slot.
        # Ordinary opens must fill those slots before reusing the retired FD.
        while old_fd not in recycled and len(recycled) < 64:
            recycled.append(os.open(second, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC))
        assert old_fd in recycled
        # Reentrant close after actual recycling must not close another owner.
        root.close()
        assert stat.S_ISDIR(REAL_FSTAT(old_fd).st_mode)
    try:
        with patch('os.close', close):
            root.close()
            root.close()
        assert events == [-1]
        assert root.fd == -1
    finally:
        root.close()
        for fd in recycled:
            close_if_fixture(fd, inode(second))
    assert live_fds() == before
    OBSERVATIONS.append({'contract': 'R1-revoke-before-OS-close', 'root_state_seen_by_close': events, 'reentrant_close_kept_recycled_owner_live': True, 'fd_set_restored': True})


@pytest.mark.parametrize('extension', ['bin', 'safetensors'])
@pytest.mark.parametrize('fault', ['unlink', 'stage-directory-close', 'temporary-directory-exit'])
def test_readable_owned_until_every_staging_cleanup_finishes(tmp_path, extension, fault):
    checkpoint = tmp_path / ('weights.' + extension)
    write_weights(checkpoint)
    assert load_state_dict(checkpoint)['marker'].item() == 17
    expected_inode = inode(checkpoint)
    before = live_fds()
    original_open, original_unlink = os.open, os.unlink
    original_directory = security._open_absolute_directory
    original_cleanup = security.tempfile.TemporaryDirectory.cleanup
    readable, stage_fds, injected, closes = [], [], [], []
    injected_error = OSError(errno.EIO, 'independent owned-fixture ' + fault)
    def open_file(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if path == 'pinned' and not flags & os.O_DIRECTORY:
            info = REAL_FSTAT(fd)
            assert (info.st_dev, info.st_ino) == expected_inode
            readable.append(fd)
        return fd
    def directory(path):
        fd = original_directory(path)
        if Path(path).name.startswith('accelerate-checkpoint-'):
            stage_fds.append(fd)
        return fd
    def unlink(path, *args, **kwargs):
        if fault == 'unlink' and path == 'pinned' and kwargs.get('dir_fd') in stage_fds and readable and not injected:
            injected.append(fault)
            raise injected_error
        return original_unlink(path, *args, **kwargs)
    def close(fd):
        if fd in readable:
            closes.append(fd)
        REAL_CLOSE(fd)
        # Model an error reported at directory teardown without deliberately
        # leaking a directory handle in the injection machinery itself.
        if fault == 'stage-directory-close' and fd in stage_fds and readable and not injected:
            injected.append(fault)
            raise injected_error
    def cleanup(staging):
        original_cleanup(staging)
        if fault == 'temporary-directory-exit' and Path(staging.name).name.startswith('accelerate-checkpoint-') and readable and not injected:
            injected.append(fault)
            raise injected_error
    try:
        with patch('os.open', open_file), patch('os.unlink', unlink), patch('os.close', close), patch.object(security, '_open_absolute_directory', directory), patch.object(security.tempfile.TemporaryDirectory, 'cleanup', cleanup):
            with pytest.raises(security.CheckpointSecurityError) as error:
                load_state_dict(checkpoint)
        assert error.value.__cause__ is injected_error
        assert injected == [fault]
        assert len(readable) == 1
        assert closes.count(readable[0]) == 1
        for fd in readable:
            assert_closed(fd)
        assert live_fds() == before
    finally:
        for fd in readable:
            close_if_fixture(fd, expected_inode)
    # Failure must not poison later legitimate loads or leave a staging link.
    assert load_state_dict(checkpoint)['marker'].item() == 17
    assert checkpoint.stat().st_nlink == 1
    assert live_fds() == before
    OBSERVATIONS.append({'contract': 'R2-staging-ownership', 'fault': fault, 'format': extension, 'opened_readable_descriptors': readable, 'raw_fd_closed_once': True, 'later_public_load_succeeded': True, 'fixture_link_count': 1, 'fd_set_restored': True})


@pytest.mark.parametrize('fault', ['directory-context-exit', 'wrapper-construction'])
def test_nested_directory_exit_and_wrapper_setup_keep_raw_fd_owner(tmp_path, fault):
    checkpoint = tmp_path / 'nested' / 'owned.bin'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'independent owned fixture')
    expected_inode = inode(checkpoint)
    root = security._Root(tmp_path)
    original_pin, original_locate, original_fdopen = security._pin_regular, root.locate, os.fdopen
    before = live_fds()
    pinned, closed, wrapped = [], [], []
    sentinel = OSError(errno.EIO, 'independent ' + fault)
    def pin(*args, **kwargs):
        fd = original_pin(*args, **kwargs)
        pinned.append(fd)
        return fd
    @contextlib.contextmanager
    def locate(name):
        with original_locate(name) as target:
            yield target
        assert pinned
        raise sentinel
    def fdopen(fd, *args, **kwargs):
        wrapped.append(fd)
        assert kwargs.get('closefd') is False
        if fault == 'wrapper-construction':
            raise sentinel
        return original_fdopen(fd, *args, **kwargs)
    def close(fd):
        if fd in pinned:
            closed.append(fd)
        REAL_CLOSE(fd)
    try:
        with patch.object(security, '_pin_regular', pin), patch('os.fdopen', fdopen), patch('os.close', close):
            with contextlib.ExitStack() as stack:
                if fault == 'directory-context-exit':
                    stack.enter_context(patch.object(root, 'locate', locate))
                with pytest.raises(OSError) as caught:
                    with root.open('nested/owned.bin'):
                        pytest.fail('Setup failure must precede delivery of the file to caller')
                assert caught.value is sentinel
        assert len(pinned) == 1
        assert closed == pinned
        assert wrapped == ([] if fault == 'directory-context-exit' else pinned)
        for fd in pinned:
            assert_closed(fd)
        assert live_fds() == before
    finally:
        for fd in pinned:
            close_if_fixture(fd, expected_inode)
        root.close()
    OBSERVATIONS.append({'contract': 'open-setup-owner', 'fault': fault, 'nested_directory_fixture': True, 'raw_fd_closed_once': True, 'fd_set_restored': True})


def test_wrapper_teardown_error_still_closes_raw_fd_once(tmp_path):
    checkpoint = tmp_path / 'owned.bin'
    checkpoint.write_bytes(b'raw owner remains live until wrapper teardown')
    expected_inode = inode(checkpoint)
    root = security._Root(tmp_path)
    before = live_fds()
    original_fdopen = os.fdopen
    descriptors, closes, wrappers = [], [], []
    sentinel = OSError(errno.EIO, 'independent wrapper teardown')
    class Wrapper:
        def __init__(self, file):
            self.file = file
        def __enter__(self):
            return self.file.__enter__()
        def __exit__(self, *args):
            self.file.__exit__(*args)
            assert self.file.closed
            # closefd=False leaves raw ownership with root.open, until finally.
            assert (REAL_FSTAT(descriptors[0]).st_dev, REAL_FSTAT(descriptors[0]).st_ino) == expected_inode
            raise sentinel
    def fdopen(fd, *args, **kwargs):
        assert kwargs.get('closefd') is False
        descriptors.append(fd)
        wrapper = Wrapper(original_fdopen(fd, *args, **kwargs))
        wrappers.append(wrapper)
        return wrapper
    def close(fd):
        if fd in descriptors:
            closes.append(fd)
        REAL_CLOSE(fd)
    try:
        with patch('os.fdopen', fdopen), patch('os.close', close):
            with pytest.raises(OSError) as caught:
                with root.open('owned.bin') as opened:
                    assert opened.file.read() == checkpoint.read_bytes()
            assert caught.value is sentinel
        assert len(descriptors) == 1
        assert closes == descriptors
        assert_closed(descriptors[0])
        assert live_fds() == before
    finally:
        for wrapper in wrappers:
            wrapper.file.close()
        for fd in descriptors:
            close_if_fixture(fd, expected_inode)
        root.close()
    OBSERVATIONS.append({'contract': 'wrapper-teardown-owner', 'original_error_preserved': True, 'raw_fd_closed_once': True, 'fd_set_restored': True})


@pytest.mark.parametrize('error_class', [RuntimeError, KeyboardInterrupt], ids=['exception', 'base-exception'])
def test_caller_error_releases_file_and_preserves_original_exception(tmp_path, error_class):
    checkpoint = tmp_path / 'owned.bin'
    checkpoint.write_bytes(b'independent caller-error fixture')
    root = security._Root(tmp_path)
    before = live_fds()
    expected_inode = inode(checkpoint)
    pinned = []
    sentinel = error_class('independent original caller error')
    original_pin = security._pin_regular
    def pin(*args, **kwargs):
        fd = original_pin(*args, **kwargs)
        pinned.append(fd)
        return fd
    try:
        with patch.object(security, '_pin_regular', pin):
            for _ in range(20):
                with pytest.raises(error_class) as caught:
                    with root.open('owned.bin') as opened:
                        assert opened.file.read() == b'independent caller-error fixture'
                        raise sentinel
                assert caught.value is sentinel
                assert opened.file.closed
                assert_closed(pinned[-1])
        assert live_fds() == before
    finally:
        for fd in set(pinned):
            close_if_fixture(fd, expected_inode)
        root.close()
    OBSERVATIONS.append({'contract': 'caller-error-owner', 'error_class': error_class.__name__, 'iterations': 20, 'original_error_preserved': True, 'fd_set_restored': True})


@pytest.mark.parametrize('fault', ['inode-mismatch', 'fstat-error'])
def test_root_construction_failure_closes_acquired_descriptor(tmp_path, fault):
    directory = tmp_path / 'owned-directory'
    directory.mkdir()
    original_directory, original_same = security._open_absolute_directory, security._same_inode
    opened = []
    before = live_fds()
    expected_inode = inode(directory)
    sentinel = OSError(errno.EIO, 'independent root construction metadata error')
    def open_directory(path):
        fd = original_directory(path)
        opened.append(fd)
        return fd
    def same(first, second):
        return False if fault == 'inode-mismatch' else original_same(first, second)
    def fstat(fd):
        if fault == 'fstat-error' and fd in opened:
            raise sentinel
        return REAL_FSTAT(fd)
    try:
        with patch.object(security, '_open_absolute_directory', open_directory), patch.object(security, '_same_inode', same), patch('os.fstat', fstat):
            if fault == 'inode-mismatch':
                with pytest.raises(security.CheckpointSecurityError, match='root changed'):
                    security._Root(directory)
            else:
                with pytest.raises(OSError) as caught:
                    security._Root(directory)
                assert caught.value is sentinel
        assert len(opened) == 1
        assert_closed(opened[0])
        assert live_fds() == before
    finally:
        for fd in opened:
            close_if_fixture(fd, expected_inode)
    OBSERVATIONS.append({'contract': 'root-construction-owner', 'fault': fault, 'fd_set_restored': True})
