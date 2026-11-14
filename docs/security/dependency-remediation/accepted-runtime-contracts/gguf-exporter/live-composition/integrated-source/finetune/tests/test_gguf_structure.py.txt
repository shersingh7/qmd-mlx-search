"""Portable GGUF structure, MTP metadata and publication contracts.

The maintained writer creates small real GGUF structures containing zero tensor
blocks. They are deliberately NOT usable models or converter qualifications.
External conversion/build/training/Hub operations remain explicit process fakes.
"""
from __future__ import annotations

import functools
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np
import pytest

import gguf_toolchain as tools


@functools.lru_cache()
def fake_gguf_bytes(kind="f16", label="current"):
    """Real writer output, not a magic-only stand-in for successful conversion."""
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFWriter, LlamaFileType

    tensor_type, file_type = {
        "f16": (GGMLQuantizationType.F16, LlamaFileType.MOSTLY_F16),
        "q4_k_m": (GGMLQuantizationType.Q4_K, LlamaFileType.MOSTLY_Q4_K_M),
        "q6_k": (GGMLQuantizationType.Q6_K, LlamaFileType.MOSTLY_Q6_K),
        "q8_0": (GGMLQuantizationType.Q8_0, LlamaFileType.MOSTLY_Q8_0),
    }[kind]
    with tempfile.TemporaryDirectory(prefix="tiny-gguf-writer-") as directory:
        path = Path(directory) / "EXPLICIT-FAKE-not-a-model.gguf"
        writer = GGUFWriter(path, "qwen3")
        try:
            writer.add_name(f"EXPLICIT FAKE {label} zero blocks; not a usable model")
            writer.add_file_type(file_type)
            if kind == "f16":
                writer.add_tensor("fixture.weight", np.zeros((16, 256), dtype=np.float16))
            else:
                # Zero scales and packed values are valid raw zero blocks. Sixteen
                # rows make every fixture end on actual tensor data, not padding.
                _, type_size = GGML_QUANT_SIZES[tensor_type]
                writer.add_tensor("fixture.weight", np.zeros((16, type_size), dtype=np.uint8),
                                  raw_dtype=tensor_type)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        return path.read_bytes()


def inspect_gguf(path):
    """Return scalar metadata only, with cleanup even if the real reader fails."""
    from gguf import GGUFReader
    reader = GGUFReader.__new__(GGUFReader)
    try:
        reader.__init__(path, mode="r")
        return {
            "tensor_types": [tensor.tensor_type.name for tensor in reader.tensors],
            "tensor_count": len(reader.tensors),
            "tensor_end": max((tensor.data_offset + tensor.n_bytes for tensor in reader.tensors), default=0),
            "file_type": int(reader.get_field("general.file_type").parts[-1][0]),
        }
    finally:
        data = getattr(reader, "data", None)
        if data is not None:
            data._mmap.close()


@pytest.mark.parametrize("kind,tensor_type", [
    ("f16", "F16"), ("q4_k_m", "Q4_K"), ("q6_k", "Q6_K"), ("q8_0", "Q8_0"),
])
def test_writer_fixtures_have_real_structure_and_quantization_types(tmp_path, kind, tensor_type):
    path = tmp_path / (kind + ".gguf")
    content = fake_gguf_bytes(kind)
    path.write_bytes(content)
    metadata = inspect_gguf(path)
    assert metadata["tensor_count"] == 1
    assert metadata["tensor_types"] == [tensor_type]
    assert metadata["tensor_end"] == len(content)
    tools.validate_gguf(path)
    assert path.read_bytes() == content


@pytest.mark.parametrize("kind", ["five-byte-magic", "truncated-header", "missing-final-tensor-byte"])
def test_structural_validation_rejects_exact_truncations(tmp_path, kind):
    reference = fake_gguf_bytes("q8_0")
    damaged = {"five-byte-magic": b"GGUF\0", "truncated-header": reference[:16],
               "missing-final-tensor-byte": reference[:-1]}[kind]
    path = tmp_path / "damaged.gguf"
    path.write_bytes(damaged)
    with pytest.raises(Exception):
        inspect_gguf(path)
    with pytest.raises(RuntimeError, match="GGUF export failed:"):
        tools.validate_gguf(path)
    assert path.read_bytes() == damaged


def fd_count():
    for directory in ("/proc/self/fd", "/dev/fd"):
        if Path(directory).is_dir():
            return len(os.listdir(directory))
    return None


@pytest.mark.parametrize("kind", ["valid", "five-byte-magic", "truncated-header", "missing-final-tensor-byte"])
def test_validation_closes_actual_reader_mappings_even_with_retained_errors(tmp_path, monkeypatch, kind):
    """No production GC, no dependence on traceback destruction or finalizers."""
    reader_module = importlib.import_module("gguf.gguf_reader")
    real_numpy = reader_module.np
    mappings = []

    class ObservedNumpy:
        def __getattr__(self, name):
            return getattr(real_numpy, name)

        def memmap(self, *args, **kwargs):
            data = real_numpy.memmap(*args, **kwargs)
            mappings.append(data._mmap)
            return data

    monkeypatch.setattr(reader_module, "np", ObservedNumpy())
    reference = fake_gguf_bytes("q8_0")
    path = tmp_path / "resource-probe.gguf"
    path.write_bytes({"valid": reference, "five-byte-magic": b"GGUF\0",
                      "truncated-header": reference[:16],
                      "missing-final-tensor-byte": reference[:-1]}[kind])
    # Warm imports before counting descriptors. Every iteration retains its
    # outward exception; disabling cyclic GC exposes constructor-failure leaks.
    retained = []
    before = fd_count()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(24):
            try:
                tools.validate_gguf(path)
            except RuntimeError as exc:
                assert kind != "valid"
                retained.append(exc)
            else:
                assert kind == "valid"
            assert len(mappings) == _ + 1
            assert all(mapping.closed for mapping in mappings)
        assert len(retained) == (0 if kind == "valid" else 24)
        # Parse tracebacks must not survive on the caller's retained exception:
        # their locals can include ndarray views into the now-closed mapping.
        assert all(exc.__cause__ is None and exc.__context__ is None for exc in retained)
        if before is not None:
            assert fd_count() == before
    finally:
        if gc_was_enabled:
            gc.enable()


@pytest.mark.parametrize("arch", ["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"])
@pytest.mark.parametrize("namespace", [
    "mtp.", "model.mtp.", "language_model.mtp.", "model.language_model.mtp.",
    "language_model.model.mtp.", "model.language_model.language_model.mtp.",
])
@pytest.mark.parametrize("mtp_shard", [0, 1, 2])
def test_all_upstream_normalized_mtp_namespaces_in_any_shard_are_preserved(tmp_path, arch, namespace, mtp_shard):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    originals = {}
    for shard in range(3):
        name = (namespace + "layers.0.self_attn.q_proj.weight" if shard == mtp_shard
                else f"model.layers.{shard}.fixture.weight")
        path = tmp_path / f"model-{shard:05}.safetensors"
        save_file({name: torch.zeros((2, 2), device="cpu")}, path)
        originals[path] = path.read_bytes()
    assert tools.conversion_flags(tmp_path) == []
    for path, content in originals.items():
        assert path.read_bytes() == content  # Classification must not rewrite saved keys.
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if path.name == f"model-{mtp_shard:05}.safetensors":
                assert tensors.keys() == [namespace + "layers.0.self_attn.q_proj.weight"]


@pytest.mark.parametrize("keys,expected", [
    (None, []), ([], ["--no-mtp"]),
    (["model.layers.0.fixture.weight", "model.language_model.layers.0.fixture.weight"], ["--no-mtp"]),
    (["not_mtp.layers.0.weight", "model.mtp_other.weight", "other.mtp.weight"], ["--no-mtp"]),
])
def test_target_only_flag_requires_saved_shards_without_mtp(tmp_path, keys, expected):
    import torch
    from safetensors.torch import save_file
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}))
    if keys is not None:
        save_file({key: torch.zeros((2, 2), device="cpu") for key in keys}, tmp_path / "model.safetensors")
    assert tools.conversion_flags(tmp_path) == expected


@pytest.mark.parametrize("namespace", ["mtp.", "model.language_model.mtp.", "model.layers."])
def test_plain_qwen3_never_receives_qwen35_flag(tmp_path, namespace):
    import torch
    from safetensors.torch import save_file
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    save_file({namespace + "fixture.weight": torch.zeros((2, 2), device="cpu")}, tmp_path / "model.safetensors")
    assert tools.conversion_flags(tmp_path) == []


def truncate_zero_status_output(monkeypatch, stage, kind):
    original = subprocess.run

    def run(argv, **kwargs):
        result = original(argv, **kwargs)
        argv = list(map(str, argv))
        path = None
        if stage == "converter" and "--outfile" in argv:
            path = Path(argv[argv.index("--outfile") + 1])
        elif argv[0].endswith("llama-quantize") and argv[-1].lower() == stage:
            path = Path(argv[2])
        if path is not None:
            assert result.returncode == 0
            content = path.read_bytes()
            path.write_bytes({"five-byte-magic": b"GGUF\0", "truncated-header": content[:16],
                              "missing-final-tensor-byte": content[:-1]}[kind])
        return result

    monkeypatch.setattr(subprocess, "run", run)


@pytest.mark.parametrize("stage", ["converter", "q4_k_m", "q6_k", "q8_0"])
@pytest.mark.parametrize("kind", ["five-byte-magic", "truncated-header", "missing-final-tensor-byte"])
def test_zero_status_truncation_in_native_flow_preserves_every_previous_export(tmp_path, monkeypatch, capsys, stage, kind):
    from test_finetune_correctness import QUANTS, fake_native_export, run_native
    events = fake_native_export(tmp_path, monkeypatch)
    truncate_zero_status_output(monkeypatch, stage, kind)
    destination = tmp_path / "adapter/gguf"
    destination.mkdir(parents=True)
    previous = {quant: fake_gguf_bytes(quant, "previous") for quant in QUANTS}
    for quant, content in previous.items():
        (destination / f"adapter-{quant}.gguf").write_bytes(content)
    with pytest.raises(RuntimeError, match="GGUF export failed:"):
        run_native(tmp_path, monkeypatch, ["--push-hub", "fixture/NEVER-WRITTEN"])
    assert all((destination / f"adapter-{quant}.gguf").read_bytes() == content for quant, content in previous.items())
    assert not any(event[0] in ("FAKE GGUF upload", "FAKE evaluation") for event in events)
    assert not list((tmp_path / "adapter").glob(".gguf-export-*"))
    output = capsys.readouterr().out
    assert "Done!" not in output and "GGUF files saved to:" not in output


@pytest.mark.parametrize("stage", ["converter", "q4_k_m", "q6_k", "q8_0"])
@pytest.mark.parametrize("kind", ["five-byte-magic", "truncated-header", "missing-final-tensor-byte"])
def test_zero_status_truncation_in_standalone_never_publishes_or_uploads(tmp_path, monkeypatch, capsys, stage, kind):
    import convert_gguf as cli
    from test_gguf_toolchain import fake_processes
    fake_processes(monkeypatch)
    truncate_zero_status_output(monkeypatch, stage, kind)

    def save(path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)

    model = NS(save_pretrained=save)
    model.merge_and_unload = lambda: model
    monkeypatch.setattr(cli.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(cli.AutoTokenizer, "from_pretrained", lambda *a, **k: NS(save_pretrained=save))
    monkeypatch.setattr(cli.PeftModel, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(cli, "HfApi", lambda: pytest.fail("Must not upload refused GGUF artifacts"))
    monkeypatch.setattr(cli, "login", lambda **k: pytest.fail("Must not authenticate in an offline test"))
    monkeypatch.setattr(cli.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["convert_gguf.py", "--size", "1.7B", "--output", "fixture/NEVER-WRITTEN-gguf"])
    previous = tmp_path / "qmd-gguf-output-previous"
    previous.mkdir()
    for quant in ("q4_k_m", "q6_k", "q8_0"):
        (previous / (quant + ".gguf")).write_bytes(fake_gguf_bytes(quant, "previous"))
    with pytest.raises(RuntimeError, match="GGUF export failed:"):
        cli.main()
    assert list(tmp_path.glob("qmd-gguf-output-*")) == [previous]
    assert all((previous / (quant + ".gguf")).read_bytes() == fake_gguf_bytes(quant, "previous") for quant in ("q4_k_m", "q6_k", "q8_0"))
    assert not list(tmp_path.glob("qmd-gguf-convert-*"))
    output = capsys.readouterr().out
    assert "Done!" not in output and "GGUF files saved to:" not in output
