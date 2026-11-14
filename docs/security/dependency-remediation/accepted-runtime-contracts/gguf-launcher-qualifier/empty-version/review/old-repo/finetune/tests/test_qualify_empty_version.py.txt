"""Real CLI negative controls: blank prerequisite versions refuse before building."""
from pathlib import Path
import os
import shlex
import subprocess
import sys

import pytest

PROBE = Path(__file__).resolve().parents[1] / "tools" / "qualify_gguf_image.py"
FULL_TOOLS = ("git", "cc", "c++", "make", "uv")
QUERIED_TOOLS = [(False, name) for name in FULL_TOOLS] + [
    (True, name) for name in FULL_TOOLS if name != "uv"
]


@pytest.mark.parametrize("native_only,empty_tool", QUERIED_TOOLS)
@pytest.mark.parametrize("empty_output", ["", "\n", " \t\r\n"], ids=["empty", "newline", "whitespace"])
def test_blank_version_is_clean_refusal_before_build(tmp_path, native_only, empty_tool, empty_output):
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls.log"
    attempted_build = tmp_path / "unexpected-build"
    for name in FULL_TOOLS:
        output = empty_output if name == empty_tool else (
            "uv 0.11.18 (x86_64-unknown-linux-musl)\n" if name == "uv" else "authored version-only fixture\n"
        )
        script = (
            "#!/bin/sh\n"
            + "printf '%s\\n' " + shlex.quote(name) + '" $*" >> ' + shlex.quote(str(calls)) + "\n"
            + 'if [ "$#" -ne 1 ] || [ "$1" != "--version" ]; then\n'
            + "  printf '%s' unexpected > " + shlex.quote(str(attempted_build)) + "\n"
            + "  exit 97\nfi\n"
            + "printf '%s' " + shlex.quote(output) + "\n"
        )
        tool = tools / name
        tool.write_text(script)
        tool.chmod(0o700)

    env = {
        "PATH": str(tools) + os.pathsep + "/usr/bin:/bin",
        "CI": "1", "QMD_LLAMA_GPU": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
    }
    for key, folder in {
        "HOME": "home", "HERMES_HOME": "hermes", "HERMES_DIR": "hermes",
        "HERMES_KNOWLEDGE_DIR": "knowledge", "KNOWLEDGE_DIR": "knowledge",
        "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
        "HF_HOME": "hf", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    }.items():
        directory = tmp_path / folder
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)

    result = subprocess.run(
        [sys.executable, "-I", "-B", "-S", str(PROBE), *(["--native-only"] if native_only else [])],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15,
    )
    observed = calls.read_text().splitlines()
    expected_tools = FULL_TOOLS[:-1] if native_only else FULL_TOOLS
    assert observed == [name + " --version" for name in expected_tools[:expected_tools.index(empty_tool) + 1]]
    assert not attempted_build.exists()
    assert not list((tmp_path / "tmp").iterdir())
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert result.stderr == f"Native prerequisite probe failed: empty version output from native prerequisite: {empty_tool}\n"
    assert "Traceback" not in result.stderr
