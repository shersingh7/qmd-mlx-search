"""Pinned llama.cpp tooling; never install into the caller's Python environment.

The project/PEP 723 lock provisions the converter's Python dependencies. Upstream
requirements pin an older training stack and MUST NOT be installed at runtime.
Only an immutable, fresh, call-owned source checkout is executed or compiled.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_URL = "https://github.com/ggml-org/llama.cpp.git"
SOURCE_REVISION = "5193ef1d4f7c95b006ab3b9f857ed27be9a63235"
QUANTIZATIONS = ("Q4_K_M", "Q6_K", "Q8_0")


def check_prerequisites(*, quantize: bool = True) -> dict[str, str]:
    """Fail before model/dataset loading, without an installer or environment edits."""
    for name in ("torch", "transformers", "numpy", "gguf", "yaml", "tqdm",
                 "safetensors", "sentencepiece", "sentencepiece.sentencepiece_model_pb2"):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(
                f"GGUF export prerequisite missing: {name}. Provision the reviewed "
                "project/script lock before starting; runtime package installation is disabled."
            ) from exc
    required = ("git", "cmake", "cc", "c++", "make") if quantize else ("git",)
    tools = {}
    for name in required:
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(
                f"GGUF export prerequisite missing: {name}. Provision native build tools "
                "(and the lock's cmake package) before starting; no apt/pip bootstrap is performed."
            )
        tools[name] = path
    return tools


def run_checked(command, stage, **kwargs):
    """Status is authoritative even if a failed process leaves an output file."""
    result = subprocess.run(command, **kwargs)
    if result.returncode != 0:
        detail = getattr(result, "stderr", "") or getattr(result, "stdout", "") or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(f"GGUF export failed: {stage} exited with status {result.returncode}\n{detail[-2000:]}")
    return result


@dataclass(frozen=True)
class Toolchain:
    work: Path
    source: Path
    quantizer: Path | None
    env: dict[str, str]


def prepare_toolchain(work: Path, *, quantize: bool = True) -> Toolchain:
    """Fetch a pinned revision and build CPU-only tools in a new owned directory.

    The caller owns/cleans `work`; it must not already exist. In particular, no
    shared /tmp/llama.cpp checkout, unverified binary or mutable branch is reused.
    """
    commands = check_prerequisites(quantize=quantize)
    work = Path(work)
    work.mkdir()
    source = work / "llama.cpp"
    source.mkdir()
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    git_env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    git = [commands["git"], "-c", f"core.hooksPath={os.devnull}", "-c", "credential.helper="]
    common = dict(capture_output=True, text=True, env=git_env, timeout=600)
    run_checked([*git, "init", str(source)], "source initialization", **common)
    run_checked([*git, "-C", str(source), "fetch", "--depth", "1", SOURCE_URL, SOURCE_REVISION],
                "pinned source fetch", **common)
    run_checked([*git, "-C", str(source), "checkout", "--detach", SOURCE_REVISION],
                "pinned source checkout", **common)
    revision = run_checked([*git, "-C", str(source), "rev-parse", "HEAD"],
                           "source revision verification", **common)
    if revision.stdout.strip() != SOURCE_REVISION:
        raise RuntimeError("GGUF export failed: source revision does not match the reviewed pin")

    # Force the pinned checkout's matching gguf-py, not a caller override or an
    # unrelated installed version. Do not load user-site modules in the child.
    env = {key: value for key, value in os.environ.items()
           if key not in {"NO_LOCAL_GGUF", "PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    run_checked([sys.executable, "-s", str(source / "convert_hf_to_gguf.py"), "--help"],
                "converter import/API preflight", capture_output=True, text=True, env=env, timeout=180)
    quantizer = None
    if quantize:
        build = work / "build"
        run_checked([
            commands["cmake"], "-B", str(build), "-S", str(source), "-G", "Unix Makefiles",
            f"-DCMAKE_C_COMPILER={commands['cc']}", f"-DCMAKE_CXX_COMPILER={commands['c++']}",
            f"-DCMAKE_MAKE_PROGRAM={commands['make']}",
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
            "-DGGML_OPENMP=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_CONNECT=OFF",
        ], "llama-quantize CMake configuration", capture_output=True, text=True, env=env, timeout=300)
        run_checked([commands["cmake"], "--build", str(build), "--target", "llama-quantize", "-j", "2"],
                    "llama-quantize build", capture_output=True, text=True, env=env, timeout=1800)
        quantizer = build / "bin" / "llama-quantize"
        if not quantizer.is_file():
            raise RuntimeError("GGUF export failed: build did not produce llama-quantize")
    return Toolchain(work, source, quantizer, env)


def validate_gguf(path: Path) -> None:
    """Parse metadata/tensor ranges with gguf's maintained, mmap-backed reader.

    GGUF 0.19 and the pinned upstream reader have no close/context API. Allocate
    before calling __init__ so its mapping is reachable even when parsing fails;
    closing it in finally must not depend on exception destruction or cyclic GC.
    """
    from gguf import GGUFReader

    if not path.is_file() or path.stat().st_size <= 4:
        raise RuntimeError(f"GGUF export failed: missing or empty artifact {path.name}")
    reader = GGUFReader.__new__(GGUFReader)
    tensor = None
    error = None
    try:
        reader.__init__(path, mode="r")
        size = int(reader.data.nbytes)
        if not 0 <= int(reader.data_offset) <= size:
            raise ValueError("tensor data section extends beyond the file")
        for tensor in reader.tensors:
            start, length = int(tensor.data_offset), int(tensor.n_bytes)
            if (start < int(reader.data_offset) or length < 0 or start + length > size
                    or int(tensor.data.nbytes) != length):
                raise ValueError(f"tensor {tensor.name!r} extends beyond its declared data range")
    except Exception as exc:
        # Keep only text, not parse tracebacks whose locals may contain mmap
        # views. Raise outside this handler so callers retaining the diagnostic
        # do not retain an exception context with views into a closed mapping.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        data = getattr(reader, "data", None)
        mapping = getattr(data, "_mmap", None)
        reader = tensor = data = None
        if mapping is not None:
            mapping.close()
    if error is not None:
        raise RuntimeError(f"GGUF export failed: invalid GGUF artifact {path.name}: {error}")


def conversion_flags(merged: Path) -> list[str]:
    """Use the pinned converter's target-only API for Qwen3.5 without MTP.

    Transformers' text CausalLM save may contain no speculative MTP tensors.
    The pinned upstream otherwise asserts when inferring zero MTP layers. Read
    only safetensors key metadata; never invent MTP weights or discard real ones.
    """
    import json
    from safetensors import safe_open

    config_path = merged / "config.json"
    if not config_path.is_file():
        return []  # The converter remains authoritative for an invalid model.
    config = json.loads(config_path.read_text())
    if not set(config.get("architectures", ())) & {"Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"}:
        return []
    shards = list(merged.glob("*.safetensors"))
    if not shards:
        return []
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            # Match base.Model.filter_tensors at SOURCE_REVISION: upstream
            # removes every language_model. occurrence before Qwen's MTP test.
            # Normalize only this classification, never the saved identifiers.
            if any(name.replace("language_model.", "").startswith(("mtp.", "model.mtp."))
                   for name in tensors.keys()):
                return []
    print("No Qwen3.5 MTP tensors saved; exporting the standard text target (--no-mtp).")
    return ["--no-mtp"]


def convert_model(toolchain: Toolchain, merged: Path, model_name: str, *,
                  quantize: bool = True, keep_f16: bool = False) -> list[Path]:
    """Return only a complete fresh artifact set; never publish partial outputs."""
    if Path(model_name).name != model_name or model_name in {"", ".", ".."}:
        raise ValueError("GGUF model_name must be a filename component")
    output = toolchain.work / "gguf"
    output.mkdir()
    fp16 = output / f"{model_name}-f16.gguf"
    log_out, log_err = toolchain.work / "converter.stdout", toolchain.work / "converter.stderr"
    try:
        with log_out.open("w") as out, log_err.open("w") as err:
            run_checked([sys.executable, "-s", str(toolchain.source / "convert_hf_to_gguf.py"),
                         str(merged), "--outfile", str(fp16), "--outtype", "f16", *conversion_flags(merged)],
                        "HF-to-GGUF conversion", stdout=out, stderr=err, text=True,
                        env=toolchain.env)
    except RuntimeError as exc:
        # Logs are call-owned and may be cleaned on exception; preserve a bounded
        # diagnostic in the exception rather than pointing at deleted log files.
        with log_err.open("rb") as stream:
            stream.seek(max(0, log_err.stat().st_size - 2000))
            detail = stream.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc}\n{detail}") from exc
    validate_gguf(fp16)
    files = [fp16] if keep_f16 or not quantize else []
    if quantize:
        if toolchain.quantizer is None:
            raise RuntimeError("GGUF export failed: quantizer was not prepared")
        for quant in QUANTIZATIONS:
            target = output / f"{model_name}-{quant.lower()}.gguf"
            run_checked([str(toolchain.quantizer), str(fp16), str(target), quant],
                        f"{quant} quantization", capture_output=True, text=True, env=toolchain.env)
            validate_gguf(target)
            files.append(target)
    if fp16 not in files:
        fp16.unlink()
    return files
