#!/usr/bin/env python3
"""Run a model-free command with a disposable HOME and offline model caches."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    command = sys.argv[1:]
    if not command:
        print("usage: python scripts/run-offline.py COMMAND [ARG ...]", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[1]
    workspace = Path(tempfile.mkdtemp(prefix="qmd-offline-"))
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "SYSTEMROOT"}}
    for key, folder in {"HOME": "home", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config", "HERMES_HOME": "hermes", "TMPDIR": "tmp", "HF_HOME": "hf"}.items():
        path = workspace / folder
        path.mkdir()
        env[key] = str(path)
    env.update(TMP=env["TMPDIR"], TEMP=env["TMPDIR"], CI="true", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(root), QMD_LLAMA_GPU="false")
    print(f"Offline fixture: {workspace}", flush=True)
    return subprocess.run(command, cwd=root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
