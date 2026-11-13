#!/usr/bin/env python3
"""Build the controlled Accelerate wheel using only the Python standard library.

The official input wheel is SHA256-pinned. This builder applies a strict source
patch, adds the reviewed capability module, updates the local version/RECORD,
and emits an order/timestamp/compression-independent (ZIP_STORED) wheel.
No package installation, build hooks, global environment changes, or subprocesses.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

UPSTREAM_VERSION = "1.15.0"
VERSION = "1.15.0+qmd.2"
UPSTREAM_NAME = "accelerate-1.15.0-py3-none-any.whl"
WHEEL_NAME = f"accelerate-{VERSION}-py3-none-any.whl"
UPSTREAM_SHA256 = "97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec"
UPSTREAM_URL = (
    "https://files.pythonhosted.org/packages/8a/4c/"
    "34f0450479d01195027260da68d8a3880683f1640c3ca5adf64acb3185f1/" + UPSTREAM_NAME
)
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "accelerate"
NOTICE = """QMD controlled local security patch: Accelerate 1.15.0+qmd.2

This is NOT an official Hugging Face release or an upstream advisory closure.
Upstream 1.15.0 remains vulnerable to GHSA-4j2p-28q2-5m79 / CVE-2026-69112.

Modified: accelerate/__init__.py and accelerate/utils/modeling.py.
Added: accelerate/utils/checkpoint_security.py (Apache-2.0).
Changed package version and wheel RECORD; original Apache-2.0 LICENSE preserved.
The source patch, builder, provenance, tests and limitations accompany this wheel
in qmd-mlx-search/finetune/vendor/accelerate. Read that README before deployment.
The protection covers Accelerate checkpoint index/discovery and final shard
loads, not other libraries' independent loaders or arbitrary remote Python code.
"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def apply_patch(files, patch):
    """Apply unified diffs with exact context; refuse fuzz, offsets or deletions."""
    lines = patch.splitlines(keepends=True)
    i = 0
    changed = []
    while i < len(lines):
        if not lines[i].startswith("--- a/"):
            raise ValueError(f"Unexpected patch header: {lines[i]!r}")
        old_path = lines[i][6:].rstrip("\r\n")
        i += 1
        if i >= len(lines) or not lines[i].startswith("+++ b/"):
            raise ValueError("Missing patch target")
        path = lines[i][6:].rstrip("\r\n")
        i += 1
        if old_path != path or path not in files or path in changed:
            raise ValueError(f"Unexpected patch path: {path}")
        source = files[path].decode("utf-8").splitlines(keepends=True)
        output = []
        cursor = 0
        while i < len(lines) and lines[i].startswith("@@ "):
            match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", lines[i])
            if match is None:
                raise ValueError("Malformed patch hunk")
            old_start, old_count, new_start, new_count = match.groups()
            old_start, old_count = int(old_start), int(old_count or 1)
            new_start, new_count = int(new_start), int(new_count or 1)
            start = old_start - 1 if old_start else 0
            if start < cursor or start > len(source):
                raise ValueError("Invalid patch offset")
            output.extend(source[cursor:start])
            cursor = start
            if len(output) != (new_start - 1 if new_start else 0):
                raise ValueError("Invalid new-source offset")
            i += 1
            read_count = write_count = 0
            while i < len(lines) and lines[i][:1] in (" ", "+", "-"):
                # A next-file header is not a hunk member once its counts match.
                if read_count == old_count and write_count == new_count:
                    break
                kind, value = lines[i][0], lines[i][1:]
                if kind in (" ", "-"):
                    if cursor >= len(source) or source[cursor] != value:
                        raise ValueError(f"Patch context mismatch in {path}:{cursor + 1}")
                    cursor += 1
                    read_count += 1
                if kind in (" ", "+"):
                    output.append(value)
                    write_count += 1
                i += 1
            if (read_count, write_count) != (old_count, new_count):
                raise ValueError("Patch hunk count mismatch")
        output.extend(source[cursor:])
        files[path] = "".join(output).encode("utf-8")
        changed.append(path)
    if set(changed) != {"accelerate/__init__.py", "accelerate/utils/modeling.py"}:
        raise ValueError(f"Unexpected patched member set: {changed}")
    return sorted(changed)


def build(upstream: Path, output: Path):
    payload = upstream.read_bytes()
    if sha256(payload) != UPSTREAM_SHA256:
        raise ValueError("Official upstream wheel SHA256 mismatch; refusing to build")
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("Duplicate wheel members")
        files = {name: source.read(name) for name in source.namelist() if not name.endswith("/")}
    original = dict(files)
    patch_path = VENDOR / "source.patch"
    module_path = VENDOR / "checkpoint_security.py"
    changed = apply_patch(files, patch_path.read_text())
    module = module_path.read_bytes()
    compile(module, "checkpoint_security.py", "exec")
    files["accelerate/utils/checkpoint_security.py"] = module
    for name in changed:
        compile(files[name], name, "exec")

    old_dist = f"accelerate-{UPSTREAM_VERSION}.dist-info/"
    new_dist = f"accelerate-{VERSION}.dist-info/"
    files = {(new_dist + name[len(old_dist):] if name.startswith(old_dist) else name): value for name, value in files.items()}
    metadata = files[new_dist + "METADATA"]
    old_line = f"Version: {UPSTREAM_VERSION}\n".encode()
    if metadata.count(old_line) != 1:
        raise ValueError("Unexpected upstream version metadata")
    files[new_dist + "METADATA"] = metadata.replace(old_line, f"Version: {VERSION}\n".encode())
    files[new_dist + "QMD_PATCH_NOTICE.txt"] = NOTICE.encode()
    license_name = new_dist + "licenses/LICENSE"
    if files[license_name] != original[old_dist + "licenses/LICENSE"]:
        raise ValueError("Upstream license must remain byte-identical")
    files.pop(new_dist + "RECORD")
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name in sorted(files):
        data = files[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((new_dist + "RECORD", "", ""))
    files[new_dist + "RECORD"] = record.getvalue().encode()

    output.mkdir(parents=True, exist_ok=True)
    wheel = output / WHEEL_NAME
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, files[name])
    result = {
        "artifact": {"filename": WHEEL_NAME, "sha256": sha256(wheel.read_bytes()), "size": wheel.stat().st_size, "version": VERSION},
        "upstream": {"filename": UPSTREAM_NAME, "sha256": UPSTREAM_SHA256, "url": UPSTREAM_URL, "version": UPSTREAM_VERSION},
        "inputs": {
            "build_patched_accelerate.py": sha256(Path(__file__).read_bytes()),
            "source.patch": sha256(patch_path.read_bytes()),
            "checkpoint_security.py": sha256(module),
        },
        "license": {"identifier": "Apache-2.0", "wheel_path": license_name, "sha256": sha256(files[license_name]), "preserved_verbatim": True},
        "patched_upstream_source_members": changed,
        "added_source_members": ["accelerate/utils/checkpoint_security.py"],
        "wheel_members": {name: sha256(data) for name, data in sorted(files.items())},
        "reproducibility": "sorted ZIP_STORED members, fixed 1980 timestamps and Unix 0644 attributes; no clock/network/build hooks in the build step",
        "advisory": "GHSA-4j2p-28q2-5m79 / CVE-2026-69112; controlled local fix, not upstream closure",
    }
    (output / "provenance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wheel": str(wheel), **result["artifact"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=VENDOR / "upstream" / UPSTREAM_NAME)
    parser.add_argument("--output-dir", type=Path, default=VENDOR)
    parser.add_argument("--download-upstream", action="store_true", help="Fetch only the immutable SHA256-pinned official input wheel")
    args = parser.parse_args()
    if args.download_upstream:
        data = urllib.request.urlopen(UPSTREAM_URL, timeout=60).read()
        if sha256(data) != UPSTREAM_SHA256:
            raise ValueError("Downloaded wheel SHA256 mismatch")
        args.upstream.parent.mkdir(parents=True, exist_ok=True)
        args.upstream.write_bytes(data)
    build(args.upstream, args.output_dir)


if __name__ == "__main__":
    main()
