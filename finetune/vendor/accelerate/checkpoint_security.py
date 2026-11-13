# Copyright 2026 QMD contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for details.
"""QMD checkpoint containment and regular-file capabilities (local patch).

Directory traversal uses held directory descriptors and never follows a symlink
in open(2). The index is untrusted; the caller's checkpoint directory and the
configured HF_HUB_CACHE are the only roots. A final load pins the inode, not a
previously checked pathname. Linux uses O_PATH + /proc/self/fd; Darwin uses a
private, same-filesystem hardlink before opening a checked regular file. Neither
strategy copies tensor data. Unsupported platforms fail closed.

This is not a sandbox for pickle/code, hostile same-UID processes, mount changes,
or in-place modification of an already authorized inode. See the patch README.
"""

from __future__ import annotations

import contextlib
import json
import ntpath
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, NoReturn

_DIR_FD_APIS = all(api in os.supports_dir_fd for api in (os.open, os.stat, os.readlink))


class CheckpointSecurityError(ValueError):
    """A checkpoint violates the local file-loading security policy."""


def _reject(message) -> NoReturn:
    raise CheckpointSecurityError("Unsafe checkpoint: " + message)


def _require_platform():
    if (
        os.name != "posix"
        or sys.platform not in ("linux", "darwin")
        or not hasattr(os, "O_NOFOLLOW")
        or not _DIR_FD_APIS
    ):
        _reject("secure checkpoint loading requires Linux or macOS descriptor-relative filesystem APIs")
    if sys.platform == "linux" and not hasattr(os, "O_PATH"):
        _reject("Linux checkpoint loading requires O_PATH and /proc/self/fd")


def _path(value):
    try:
        value = os.fspath(value)
    except TypeError:
        _reject("checkpoint must be a string or path-like object")
    if not isinstance(value, str) or not value or "\0" in value:
        _reject("checkpoint must be a nonempty text path without NUL")
    return os.path.abspath(value)


def _shard_name(name):
    # Reject foreign roots even on POSIX; do not silently repair attacker input.
    if (
        not isinstance(name, str)
        or not name
        or "\0" in name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or ntpath.splitdrive(name)[0]
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        _reject(f"invalid relative shard name {name!r}")
    return name


def _below(path, root):
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _open_absolute_directory(path):
    """Open each canonical component without following a mutable symlink."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for part in path.split("/"):
            if not part:
                continue
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _regular(info, label):
    if not stat.S_ISREG(info.st_mode):
        _reject(f"{label!r} is not a regular file")


def _same_inode(first, second):
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _pin_regular(parent_fd, name):
    """Return a readable FD only after pinning and checking the actual inode.

    No rejected special file is data-opened. On Linux O_PATH is a metadata-only
    capability (including during a concurrent regular -> special replacement).
    On Darwin linkat pins a directory entry without invoking the file's open
    operation; a private staging entry is checked before its first data open.
    O_NONBLOCK alone would NOT be sufficient for devices and is not our proof.
    """
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _regular(before, name)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    if sys.platform == "linux":
        pinned = os.open(name, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        try:
            info = os.fstat(pinned)
            _regular(info, name)
            # The procfs link is a capability to our held inode, not an attacker
            # pathname. O_NOFOLLOW must NOT be used for this kernel magic link.
            readable = os.open(f"/proc/self/fd/{pinned}", os.O_RDONLY | os.O_CLOEXEC)
            try:
                if not _same_inode(info, os.fstat(readable)):
                    _reject("descriptor reopening changed the checkpoint inode")
                return readable
            except BaseException:
                os.close(readable)
                raise
        finally:
            os.close(pinned)
    if sys.platform != "darwin":
        _reject("unsupported secure-opening platform")

    # A data open of an attacker-controlled pathname on Darwin could have
    # device side effects even with O_NONBLOCK. Hardlink first; inspect second.
    # EXDEV/EPERM deliberately fail closed rather than falling back to unsafe
    # open/copy. TMPDIR must be on the checkpoint filesystem and permit links.
    readable = None
    try:
        with tempfile.TemporaryDirectory(prefix="accelerate-checkpoint-") as staging:
            stage_fd = _open_absolute_directory(os.path.realpath(staging))
            try:
                stage_info = os.fstat(stage_fd)
                if stage_info.st_uid != os.geteuid() or stat.S_IMODE(stage_info.st_mode) != 0o700:
                    _reject("checkpoint staging directory is not private")
                try:
                    os.link(name, "pinned", src_dir_fd=parent_fd, dst_dir_fd=stage_fd, follow_symlinks=False)
                except OSError as exc:
                    _reject(
                        "macOS requires private same-filesystem hardlink staging; "
                        "set TMPDIR to a trusted directory on the checkpoint filesystem "
                        f"(link failed: {exc.strerror})"
                    )
                try:
                    info = os.stat("pinned", dir_fd=stage_fd, follow_symlinks=False)
                    _regular(info, name)
                    readable = os.open("pinned", flags, dir_fd=stage_fd)
                    if not _same_inode(info, os.fstat(readable)):
                        _reject("private staging changed the checkpoint inode")
                finally:
                    os.unlink("pinned", dir_fd=stage_fd)
            finally:
                os.close(stage_fd)
        # Transfer only after unlink, directory close AND TemporaryDirectory
        # cleanup succeed. Any preceding failure still belongs to this owner.
        if readable is None:
            _reject("staging completed without a readable checkpoint descriptor")
        result, readable = readable, None
        return result
    finally:
        if readable is not None:
            os.close(readable)


@dataclass(frozen=True)
class _CheckpointFile:
    root: "_Root"
    name: str


@dataclass
class _OpenedFile:
    name: str
    file: BinaryIO

    @property
    def mmap_path(self):
        # Both Torch's file object API and safetensors' mmap API consume the held
        # inode. Safetensors only accepts filenames, not Python file objects.
        prefix = "/proc/self/fd" if sys.platform == "linux" else "/dev/fd"
        return f"{prefix}/{self.file.fileno()}"


class _Root:
    def __init__(self, directory):
        _require_platform()
        self.fd = -1  # Closed sentinel must never mean process-relative dir_fd=None.
        directory = _path(directory)
        selected = os.stat(directory)
        if not stat.S_ISDIR(selected.st_mode):
            _reject("checkpoint root is not a directory")
        self.path = os.path.realpath(directory)
        self._hub = None
        self._pending_shards: tuple[str, ...] | None = None
        # A hub-looking directory is NOT authorization. The configured cache
        # root is operator input imported by huggingface_hub at process startup.
        from huggingface_hub.constants import HF_HUB_CACHE

        hub = os.path.realpath(_path(HF_HUB_CACHE))
        if _below(self.path, hub):
            relative = os.path.relpath(self.path, hub).split(os.sep)
            if (
                len(relative) >= 3
                and re.fullmatch(r"models--[A-Za-z0-9_.-]+", relative[0])
                and relative[1] == "snapshots"
                and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", relative[2])
            ):
                self._hub = (hub, relative[0])
        self.fd = _open_absolute_directory(self.path)
        try:
            if not _same_inode(selected, os.fstat(self.fd)):
                _reject("checkpoint root changed while establishing its directory capability")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd >= 0:
            descriptor, self.fd = self.fd, -1  # Revoke BEFORE the OS may recycle it.
            os.close(descriptor)

    def _require_open(self):
        if self.fd < 0:
            _reject("checkpoint directory capability is closed")

    @contextlib.contextmanager
    def locate(self, name):
        self._require_open()
        _shard_name(name)
        pending = name.split("/")
        links = 0
        # A restart after readlink re-walks every component relative to the held
        # root. No resolved string is ever later passed to a data-open syscall.
        while True:
            with contextlib.ExitStack() as stack:
                current = self.fd
                seen = []
                restart = False
                for position, part in enumerate(pending):
                    info = os.stat(part, dir_fd=current, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        links += 1
                        if links > 40:
                            _reject("too many checkpoint symlinks")
                        target = os.readlink(part, dir_fd=current)
                        if "\0" in target:
                            _reject("NUL in checkpoint symlink")
                        target = os.path.normpath(os.path.join(self.path, *seen, target))
                        tail = pending[position + 1 :]
                        if _below(target, self.path):
                            relative = os.path.relpath(target, self.path)
                            pending = ([] if relative == "." else relative.split(os.sep)) + tail
                            if not pending:
                                _reject("checkpoint shard resolves to its root directory")
                            restart = True
                            break
                        # Only a direct blob in the SAME, explicitly configured
                        # model cache is allowed to escape a snapshot. Neither
                        # the blob nor repo/blobs directories may be symlinks.
                        if self._hub is not None and not tail:
                            hub, repo = self._hub
                            blob_dir = os.path.join(hub, repo, "blobs")
                            if (
                                os.path.dirname(target) == blob_dir
                                and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", os.path.basename(target))
                            ):
                                blob_fd = _open_absolute_directory(blob_dir)
                                stack.callback(os.close, blob_fd)
                                blob_name = os.path.basename(target)
                                blob_info = os.stat(blob_name, dir_fd=blob_fd, follow_symlinks=False)
                                _regular(blob_info, name)
                                yield blob_fd, blob_name
                                return
                        _reject(f"shard {name!r} escapes the checkpoint root or approved cache blob root")
                    if position == len(pending) - 1:
                        _regular(info, name)
                        yield current, part
                        return
                    if not stat.S_ISDIR(info.st_mode):
                        _reject(f"non-directory component in shard {name!r}")
                    next_fd = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current
                    )
                    stack.callback(os.close, next_fd)
                    current = next_fd
                    seen.append(part)
                if not restart:
                    _reject("empty checkpoint path")

    def validate(self, name):
        with self.locate(name):
            pass

    def preflight(self):
        self._require_open()
        # Defer filesystem checks until this process actually reads a shard.
        # Upstream rank-0 broadcast legitimately needs only the index on other
        # ranks. Schema/path syntax is still validated during discovery there.
        # Before the FIRST shard data open, validate the complete shard set.
        if self._pending_shards is not None:
            for name in self._pending_shards:
                self.validate(name)
            self._pending_shards = None

    @contextlib.contextmanager
    def open(self, name):
        fd = None
        try:
            with self.locate(name) as (parent, leaf):
                fd = _pin_regular(parent, leaf)
            # Keep one explicit FD owner across directory-context cleanup,
            # file-wrapper construction, caller errors and wrapper teardown.
            with os.fdopen(fd, "rb", closefd=False) as file:
                yield _OpenedFile(os.path.join(self.path, name), file)
        finally:
            if fd is not None:
                os.close(fd)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _reject(f"duplicate checkpoint index key {key!r}")
        result[key] = value
    return result


def _index_shards(file):
    try:
        index = json.load(file, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        _reject(f"malformed checkpoint index ({exc})")
    if not isinstance(index, dict) or not index:
        _reject("checkpoint index must be a nonempty JSON object")
    if "weight_map" in index:
        if "metadata" in index and not isinstance(index["metadata"], dict):
            _reject("checkpoint index metadata must be an object")
        index = index["weight_map"]
    if not isinstance(index, dict) or not index:
        _reject("checkpoint weight_map must be a nonempty object")
    for key, name in index.items():
        if not isinstance(key, str) or not key or "\0" in key:
            _reject("checkpoint weight names must be nonempty strings without NUL")
        _shard_name(name)
    return sorted(set(index.values()))


@contextlib.contextmanager
def checkpoint_files(checkpoint, weights_name, safe_weights_name):
    """Validate discovery/schema; retain a root for all-shard preflight and loads."""
    _require_platform()
    path = _path(checkpoint)
    with contextlib.ExitStack() as stack:
        try:
            selected = os.stat(path)
            is_directory = stat.S_ISDIR(selected.st_mode)
            directory = path if is_directory else os.path.dirname(path)
            root = _Root(directory)
            stack.callback(root.close)
            if is_directory and not _same_inode(selected, os.fstat(root.fd)):
                _reject("checkpoint directory changed during discovery")
            if is_directory:
                names = os.listdir(root.fd)
                if weights_name in names:
                    shards, index = [weights_name], None
                elif safe_weights_name in names:
                    shards, index = [safe_weights_name], None
                else:
                    indexes = [name for name in names if name.endswith(".index.json")]
                    if len(indexes) != 1:
                        _reject("checkpoint folder must contain one index or a whole-model checkpoint")
                    shards, index = None, indexes[0]
            else:
                name = os.path.basename(path)
                shards, index = (None, name) if name.endswith(".json") else ([name], None)
            if index is not None:
                with root.open(index) as opened:
                    shards = _index_shards(opened.file)
            assert shards is not None
            # Syntax is unconditional. Filesystem preflight occurs before the
            # first actual shard read (not on non-reading broadcast ranks).
            # Final pinning, not this preflight, supplies the race guarantee.
            for name in shards:
                _shard_name(name)
            root._pending_shards = tuple(shards)
        except OSError as exc:
            raise CheckpointSecurityError(f"Unsafe checkpoint: filesystem validation failed ({exc})") from exc
        # Do not relabel exceptions raised by the caller's model/device/offload
        # code as security failures. Only filesystem setup above is translated.
        yield [_CheckpointFile(root, name) for name in shards]


@contextlib.contextmanager
def open_checkpoint_file(checkpoint_file):
    """Guard the final public load_state_dict boundary, including direct calls."""
    with contextlib.ExitStack() as stack:
        try:
            if isinstance(checkpoint_file, _CheckpointFile):
                root, name = checkpoint_file.root, checkpoint_file.name
                root.preflight()
            else:
                path = _path(checkpoint_file)
                root = _Root(os.path.dirname(path))
                stack.callback(root.close)
                name = os.path.basename(path)
            opened = stack.enter_context(root.open(name))
        except OSError as exc:
            raise CheckpointSecurityError(f"Unsafe checkpoint: final file validation failed ({exc})") from exc
        yield opened
