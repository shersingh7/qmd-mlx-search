"""Offline launcher contracts. All code/lock/input fixtures here are synthetic."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "tools" / "submit_locked_job.py"
SPEC = importlib.util.spec_from_file_location("qmd_locked_job_submission", LAUNCHER)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


@pytest.fixture
def synthetic_repo(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-checkout"
    root.mkdir()
    version, digest, commit = "1.15.0+qmd.2", "a" * 64, "b" * 40
    wheel = f"finetune/vendor/accelerate/accelerate-{version}-py3-none-any.whl"
    url = f"https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/{commit}/{wheel.replace('+', '%2B')}"
    policy = {
        "schema_version": 1,
        "accelerate": {
            "advisory": "GHSA-4j2p-28q2-5m79", "version": version,
            "sha256": digest, "commit": commit, "wheel_path": wheel,
            "immutable_url": f"{url}#sha256={digest}",
            "runner_paths": list(launcher.CODE_CLOSURE),
        },
    }
    policy_path = root / launcher.POLICY_PATH
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps(policy))
    header = (
        '# /// script\n# requires-python = ">=3.12"\n'
        f'# dependencies = ["accelerate=={version}"]\n'
        '# [tool.uv]\n'
        f'# constraint-dependencies = ["accelerate=={version}"]\n'
        '# [tool.uv.sources]\n'
        f'# accelerate = {{ url = "{url}" }}\n# ///\n'
        'print("SYNTHETIC runner: not a production training script")\n'
    )
    lock = (
        '# SYNTHETIC unit-test lock; NOT a production dependency graph.\n'
        'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n'
        '[[package]]\nname = "accelerate"\n'
        f'version = "{version}"\nsource = {{ url = "{url}" }}\n'
        f'wheels = [{{ url = "{url}", hash = "sha256:{digest}" }}]\n'
        '[manifest]\n'
        f'requirements = [{{ name = "accelerate", url = "{url}" }}]\n'
        f'constraints = [{{ name = "accelerate", url = "{url}" }}]\n'
    )
    paths = set(launcher.CODE_CLOSURE)
    for helpers in launcher.CODE_CLOSURE.values():
        paths.update(helpers)
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header if name in launcher.CODE_CLOSURE else "# Synthetic reviewed helper/asset\n")
    for name in launcher.CODE_CLOSURE:
        (root / (name + ".lock")).write_text(lock)
    # These MUST NOT be discovered or uploaded, even when passed as script args.
    for name in [".env", "finetune/data/private.jsonl", "finetune/unreviewed.py", "finetune/.cache/token"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC PRIVATE SENTINEL: do not include\n")
    monkeypatch.setattr(launcher, "ROOT", root)
    return root


def argv(runner="finetune/jobs/sft.py", *extra):
    return ["--namespace", "synthetic-owner", "--flavor", "cpu-basic", "--timeout", "5m",
            "--runner", runner, *extra]


class FakeVolume(NS):
    pass


class FakeApi:
    """Records local SDK composition only; never talks to HF."""
    def __init__(self, *, private=True, mismatch=False):
        self.private = private
        self.mismatch = mismatch
        self.calls = []
        self.uploaded = {}

    def create_bucket(self, **kwargs):
        self.calls.append(("create_bucket", kwargs))

    def bucket_info(self, **kwargs):
        self.calls.append(("bucket_info", kwargs))
        return NS(id=kwargs["bucket_id"], private=self.private)

    def sync_job_volume(self, source, mount_path, **kwargs):
        self.calls.append(("sync_job_volume", {"mount_path": mount_path, **kwargs}))
        source = Path(source)
        self.uploaded = {p.relative_to(source).as_posix(): p.read_bytes()
                         for p in source.rglob("*") if p.is_file()}
        return FakeVolume(type="bucket", source=kwargs["namespace"] + "/jobs-artifacts",
                          mount_path=mount_path, path=kwargs["remote_name"], read_only=True)

    def run_job(self, **kwargs):
        self.calls.append(("run_job", kwargs))
        self.job_spec = kwargs
        return NS(id="synthetic-job-id")

    def list_bucket_tree(self, **kwargs):
        self.calls.append(("list_bucket_tree", kwargs))
        return [NS(path=kwargs["prefix"] + path, size=len(content))
                for path, content in self.uploaded.items()]

    def inspect_job(self, **kwargs):
        self.calls.append(("inspect_job", kwargs))
        return NS(id=kwargs["job_id"], command=[] if self.mismatch else self.job_spec["command"],
                  docker_image=self.job_spec["image"], flavor=self.job_spec["flavor"],
                  url="https://example.invalid/synthetic-job-id")


def fake_sdk(monkeypatch, api):
    monkeypatch.setattr(launcher, "load_sdk", lambda: (api, FakeVolume))


def test_default_dry_run_stages_exact_closure_without_sdk(synthetic_repo, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("dry-run imported/initialized SDK"))
    seen = {}
    original = launcher.stage_files

    def inspect_stage(files, directory):
        original(files, directory)
        seen.update({p.relative_to(directory).as_posix(): p.read_bytes()
                     for p in directory.rglob("*") if p.is_file()})

    monkeypatch.setattr(launcher, "stage_files", inspect_stage)
    assert launcher.main(argv()) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["mode"] == "dry-run"
    assert "job_id" not in manifest
    assert set(seen) == {"finetune/jobs/sft.py", "finetune/jobs/sft.py.lock", "finetune/jobs/eval_common.py"}
    assert {entry["path"] for entry in manifest["files"]} == set(seen)
    assert not any(b"PRIVATE SENTINEL" in content for content in seen.values())
    import hashlib
    for entry in manifest["files"]:
        assert entry["sha256"] == hashlib.sha256(seen[entry["path"]]).hexdigest()
        assert entry["size"] == len(seen[entry["path"]])


@pytest.mark.parametrize("runner", list(launcher.CODE_CLOSURE))
def test_all_supported_runner_closures_have_adjacent_locks(synthetic_repo, runner):
    plan, files = launcher.prepare(launcher.parse_args(argv(runner)))
    assert runner in files and runner + ".lock" in files
    expected = {runner, *launcher.CODE_CLOSURE[runner]}
    expected.update(p + ".lock" for p in expected.copy() if p in launcher.CODE_CLOSURE)
    assert set(files) == expected
    uv_command = plan["command"][5:]
    assert uv_command[:5] == ["uv", "run", "--locked", "--no-project", "--script"]
    assert uv_command[5] == runner.removeprefix("finetune/")


def test_native_training_stages_its_actual_shared_exporter(synthetic_repo):
    _, files = launcher.prepare(launcher.parse_args(argv("finetune/train_unsloth.py")))
    # The real native entrypoint imports export_gguf from train at runtime.
    assert "finetune/train.py" in files
    assert "finetune/train.py.lock" in files


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, KeyboardInterrupt])
def test_wrapper_construction_failure_releases_owned_descriptor(tmp_path, monkeypatch, error_type):
    import errno
    source = tmp_path / "synthetic-source.txt"
    source.write_bytes(b"synthetic fixture")
    opened = []

    def fail_wrapper(fd, *args, **kwargs):
        opened.append(fd)
        raise error_type("synthetic wrapper construction failure")

    local_os = NS(**vars(os))
    local_os.fdopen = fail_wrapper
    monkeypatch.setattr(launcher, "os", local_os)
    with pytest.raises((ValueError, error_type)):
        launcher.read_regular(source)
    assert len(opened) == 1
    try:
        os.fstat(opened[0])
    except OSError as exc:
        assert exc.errno == errno.EBADF
    else:
        os.close(opened[0])  # Clean up the deliberately exercised buggy baseline.
        pytest.fail("read_regular leaked the descriptor when fdopen raised")


def test_default_image_matches_the_qualified_uv_runtime():
    assert launcher.DEFAULT_IMAGE == (
        "ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie-slim@sha256:"
        "9f9a7c30d604c8297ac14321b91b42f8516ff207f42b76b1bf29974fad523f6e"
    )


def test_missing_helper_lock_and_bad_policy_fail_before_sdk(synthetic_repo, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("preflight initialized SDK"))
    (synthetic_repo / "finetune/eval.py.lock").unlink()
    assert launcher.main(argv("finetune/train.py", "--submit")) == 2
    assert "eval.py.lock" in capsys.readouterr().err
    policy_path = synthetic_repo / launcher.POLICY_PATH
    policy = json.loads(policy_path.read_text())
    policy["accelerate"]["sha256"] = "c" * 64
    policy_path.write_text(json.dumps(policy))
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit")) == 2
    assert "policy" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "fifo", "bad-toml", "wrong-hash", "wrong-url", "duplicate-package"])
def test_invalid_lock_blocked_before_network(synthetic_repo, monkeypatch, capsys, kind):
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("invalid lock reached SDK"))
    lock = synthetic_repo / "finetune/jobs/sft.py.lock"
    original = lock.read_text()
    if kind in {"missing", "directory", "symlink", "fifo"}:
        lock.unlink()
        if kind == "directory":
            lock.mkdir()
        elif kind == "symlink":
            lock.symlink_to(synthetic_repo / ".env")
        elif kind == "fifo":
            os.mkfifo(lock)
    elif kind == "bad-toml":
        lock.write_text("not = [valid")
    elif kind == "wrong-hash":
        lock.write_text(original.replace("sha256:" + "a" * 64, "sha256:" + "c" * 64))
    elif kind == "wrong-url":
        lock.write_text(original.replace("b" * 40, "c" * 40))
    else:
        lock.write_text(original + '\n[[package]]\nname="accelerate"\nversion="0.0.0"\n')
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit")) == 2
    assert capsys.readouterr().err


@pytest.mark.parametrize("runner", ["finetune/unreviewed.py", "finetune/../finetune/jobs/sft.py", "/finetune/jobs/sft.py", "finetune/jobs/*.py"])
def test_unlisted_or_nonliteral_runner_refused(synthetic_repo, runner):
    with pytest.raises(ValueError):
        launcher.prepare(launcher.parse_args(argv(runner)))


def test_script_and_ancestor_symlinks_and_hardlinks_refused(synthetic_repo):
    helper = synthetic_repo / "finetune/jobs/eval_common.py"
    helper.unlink()
    helper.symlink_to(synthetic_repo / ".env")
    with pytest.raises(ValueError, match="[Ss]ymlink|regular|safe"):
        launcher.prepare(launcher.parse_args(argv()))
    helper.unlink()
    os.link(synthetic_repo / ".env", helper)
    with pytest.raises(ValueError, match="hardlink|regular|safe"):
        launcher.prepare(launcher.parse_args(argv()))
    helper.unlink()
    helper.write_text("# Synthetic helper\n")
    jobs = helper.parent
    moved = jobs.with_name("real-jobs")
    jobs.rename(moved)
    jobs.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="[Ss]ymlink|regular|safe"):
        launcher.prepare(launcher.parse_args(argv()))


def test_script_args_preserved_literally_and_not_auto_uploaded(synthetic_repo, monkeypatch):
    unrelated = synthetic_repo / "finetune/data/private.jsonl"
    arguments = ["--help", "--", str(unrelated), "--value=with spaces", "$(touch unwanted)", "*.jsonl", "", "--submit"]
    monkeypatch.chdir(synthetic_repo.parent)
    plan, files = launcher.prepare(launcher.parse_args([*argv(), "--", *arguments]))
    assert plan["command"][11:] == arguments
    assert "finetune/data/private.jsonl" not in files


def test_explicit_inputs_require_consent_and_keep_paths(synthetic_repo, tmp_path):
    first = tmp_path / "first/data.jsonl"
    second = tmp_path / "second/data.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("SYNTHETIC explicitly opted-in first input\n")
    second.write_text("SYNTHETIC explicitly opted-in second input\n")
    options = ["--input", f"{first}=data/first.jsonl", "--input", f"{second}=data/second.jsonl"]
    with pytest.raises(ValueError, match="allow-input-upload"):
        launcher.prepare(launcher.parse_args(argv("finetune/jobs/sft.py", *options)))
    plan, files = launcher.prepare(launcher.parse_args(argv("finetune/jobs/sft.py", *options, "--allow-input-upload")))
    assert files["finetune/data/first.jsonl"] == first.read_bytes()
    assert files["finetune/data/second.jsonl"] == second.read_bytes()
    assert [f["path"] for f in plan["files"] if f["kind"] == "input"] == ["finetune/data/first.jsonl", "finetune/data/second.jsonl"]


@pytest.mark.parametrize("destination", ["../escape", "/escape", "jobs/sft.py", "jobs/sft.py.lock", ".env", "data/../x", "data/*.jsonl", "data\\escape", "jobs/sft.py/child", "data/x.py", "pyproject.toml", "uv.toml"])
def test_input_traversal_code_and_config_overrides_refused(synthetic_repo, tmp_path, destination):
    source = tmp_path / "synthetic.jsonl"
    source.write_text("SYNTHETIC\n")
    with pytest.raises(ValueError):
        launcher.prepare(launcher.parse_args(argv("finetune/jobs/sft.py", "--allow-input-upload", "--input", f"{source}={destination}")))


def test_inputs_reject_collision_directories_links_and_credential_names(synthetic_repo, tmp_path):
    source = tmp_path / "synthetic.jsonl"
    source.write_text("SYNTHETIC\n")
    base = argv("finetune/jobs/sft.py", "--allow-input-upload")
    with pytest.raises(ValueError, match="collision"):
        launcher.prepare(launcher.parse_args([*base, "--input", f"{source}=data/x.jsonl", "--input", f"{source}=data/x.jsonl"]))
    for path in [source.parent, synthetic_repo / ".env"]:
        with pytest.raises(ValueError):
            launcher.prepare(launcher.parse_args([*base, "--input", f"{path}=data/x.jsonl"]))
    link = tmp_path / "link.jsonl"
    link.symlink_to(source)
    with pytest.raises(ValueError):
        launcher.prepare(launcher.parse_args([*base, "--input", f"{link}=data/x.jsonl"]))
    cached = tmp_path / ".cache/synthetic/input.jsonl"
    cached.parent.mkdir(parents=True)
    cached.write_text("SYNTHETIC cache sentinel: do not upload\n")
    with pytest.raises(ValueError):
        launcher.prepare(launcher.parse_args([*base, "--input", f"{cached}=data/x.jsonl"]))


def test_mocked_submission_composes_exact_upload_run_and_readback(synthetic_repo, monkeypatch, capsys):
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    monkeypatch.setenv("QMD_SYNTHETIC_SECRET", "synthetic-secret-not-for-manifest")
    options = ["--submit", "--env", "EXPERIMENT=synthetic", "--secret-env", "QMD_SYNTHETIC_SECRET",
               "--volume", "dataset=synthetic-owner/input=/inputs/train",
               "--writable-volume", "bucket=synthetic-owner/output=/outputs/run"]
    assert launcher.main(argv("finetune/jobs/sft.py", *options, "--", "--dry-run")) == 0
    output = capsys.readouterr().out
    manifest = json.loads(output)
    assert manifest["mode"] == "submitted"
    assert manifest["job_id"] == "synthetic-job-id"
    assert "synthetic-secret-not-for-manifest" not in output
    assert [name for name, _ in api.calls] == ["create_bucket", "bucket_info", "sync_job_volume", "list_bucket_tree", "run_job", "inspect_job"]
    assert api.calls[0][1] == {"bucket_id": "synthetic-owner/jobs-artifacts", "private": True, "exist_ok": True}
    assert api.calls[2][1]["read_only"] is True
    assert api.calls[2][1]["namespace"] == "synthetic-owner"
    assert set(api.uploaded) == {"finetune/jobs/sft.py", "finetune/jobs/sft.py.lock", "finetune/jobs/eval_common.py"}
    assert api.job_spec["command"] == manifest["command"]
    assert api.job_spec["attempts"] == 1
    assert api.job_spec["namespace"] == "synthetic-owner"
    assert api.job_spec["flavor"] == "cpu-basic"
    assert api.job_spec["timeout"] == "5m"
    assert api.job_spec["secrets"] == {"QMD_SYNTHETIC_SECRET": "synthetic-secret-not-for-manifest"}
    assert [(v.mount_path, v.read_only) for v in api.job_spec["volumes"]] == [("/qmd-code", True), ("/inputs/train", True), ("/outputs/run", False)]


def test_public_bucket_refused_before_any_upload(synthetic_repo, monkeypatch, capsys):
    api = FakeApi(private=False)
    fake_sdk(monkeypatch, api)
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit")) == 2
    assert "private" in capsys.readouterr().err
    assert [name for name, _ in api.calls] == ["create_bucket", "bucket_info"]
    assert not api.uploaded


def test_opted_in_input_upload_does_not_sweep_its_parent(synthetic_repo, tmp_path, monkeypatch, capsys):
    source = tmp_path / "input-folder/reviewed.jsonl"
    source.parent.mkdir()
    source.write_text("SYNTHETIC consented input bytes\n")
    (source.parent / ".env").write_text("SYNTHETIC credential sentinel: must stay local\n")
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit", "--allow-input-upload",
                              "--input", f"{source}=data/reviewed.jsonl")) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "submitted"
    assert api.uploaded["finetune/data/reviewed.jsonl"] == source.read_bytes()
    assert set(api.uploaded) == {"finetune/jobs/sft.py", "finetune/jobs/sft.py.lock",
                                 "finetune/jobs/eval_common.py", "finetune/data/reviewed.jsonl"}
    assert not any(b"credential sentinel" in content for content in api.uploaded.values())


def test_job_readback_mismatch_is_not_reported_as_success(synthetic_repo, monkeypatch, capsys):
    api = FakeApi(mismatch=True)
    fake_sdk(monkeypatch, api)
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit")) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "synthetic-job-id" in captured.err
    assert "readback" in captured.err


def test_missing_secret_fails_before_network_and_dry_run_does_not_read_it(synthetic_repo, monkeypatch, capsys):
    monkeypatch.delenv("QMD_SYNTHETIC_ABSENT", raising=False)
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    options = argv("finetune/jobs/sft.py", "--secret-env", "QMD_SYNTHETIC_ABSENT")
    assert launcher.main(options) == 0
    assert "QMD_SYNTHETIC_ABSENT" in capsys.readouterr().out
    assert launcher.main([*options, "--submit"]) == 2
    assert not api.calls


@pytest.mark.parametrize("options", [
    ["--env", "UV_NO_SOURCES=1"], ["--secret-env", "PYTHONPATH"],
    ["--volume", "dataset=synthetic-owner/x=/qmd-code"],
    ["--volume", "dataset=synthetic-owner/x=/inputs", "--volume", "dataset=synthetic-owner/y=/inputs/child"],
    ["--env", "X=1", "--env", "X=2"], ["--env", "X=1", "--secret-env", "X"],
    ["--env", "HF_TOKEN=synthetic-must-not-be-printed"],
])
def test_invalid_options_rejected_before_sdk(synthetic_repo, monkeypatch, options):
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("bad options reached SDK"))
    assert launcher.main(argv("finetune/jobs/sft.py", *options, "--submit")) == 2


def test_real_cli_is_stdlib_only_and_independent_of_cwd(synthetic_repo, tmp_path):
    copied_launcher = synthetic_repo / "finetune/tools/submit_locked_job.py"
    copied_launcher.parent.mkdir()
    copied_launcher.write_bytes(LAUNCHER.read_bytes())
    result = subprocess.run([sys.executable, "-B", "-I", "-S", str(copied_launcher), *argv(), "--", "--help"],
                            cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["mode"] == "dry-run"
    assert manifest["command"][-1] == "--help"


def test_actual_bootstrap_preserves_argv_and_writable_helper_layout(synthetic_repo, tmp_path):
    """Execute the real shell wrapper with an explicit fake uv, not a cloud job."""
    arguments = ["--", "a b", "$(touch MUST_NOT_EXIST)", "*.jsonl", "", "--help"]
    manifest, files = launcher.prepare(launcher.parse_args([*argv("finetune/train.py"), "--", *arguments]))
    staging = tmp_path / "synthetic read-only mount"
    staging.mkdir()
    launcher.stage_files(files, staging)
    fake_uv = tmp_path / "synthetic-uv"
    fake_uv.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "p=pathlib.Path.cwd()\n"
        "(p/'SYNTHETIC_OUTPUT.txt').write_text('synthetic output')\n"
        "print(json.dumps({'argv':sys.argv[1:], 'cwd':str(p), "
        "'files':sorted(str(f.relative_to(p.parent)) for f in p.rglob('*') if f.is_file())}))\n"
    )
    fake_uv.chmod(0o700)
    command = manifest["command"].copy()
    command[4] = str(staging)  # Mount simulation; shell text stays byte-for-byte unchanged.
    command[5] = str(fake_uv)  # Mock only uv execution, so no install/model/training.
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ, "TMPDIR": str(tmp_path)},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["argv"] == ["run", "--locked", "--no-project", "--script", "train.py", *arguments]
    assert set(files).issubset(receipt["files"])
    assert Path(receipt["cwd"]).name == "finetune"
    assert Path(receipt["cwd"]).parent.is_relative_to(tmp_path)
    assert not (tmp_path / "MUST_NOT_EXIST").exists()
    assert not (staging / "finetune/SYNTHETIC_OUTPUT.txt").exists()


def test_data_only_graph_does_not_gain_accelerate(synthetic_repo):
    script = synthetic_repo / "finetune/dataset/validate_schema.py"
    script.write_text(script.read_text().replace('# dependencies = ["accelerate==1.15.0+qmd.2"]', '# dependencies = []', 1))
    lock = script.with_suffix(".py.lock")
    text = lock.read_text()
    text = text[:text.index("[[package]]")] + text[text.index("[manifest]"):]
    text = re.sub(r"(?m)^requirements = .*", "requirements = []", text)
    lock.write_text(text)
    _, files = launcher.prepare(launcher.parse_args(argv("finetune/dataset/validate_schema.py")))
    assert set(files) == {"finetune/dataset/validate_schema.py", "finetune/dataset/validate_schema.py.lock", "finetune/dataset/schema.py"}
    assert b"[[package]]" not in files["finetune/dataset/validate_schema.py.lock"]


def test_url_fragment_in_script_source_is_not_treated_as_integrity(synthetic_repo):
    script = synthetic_repo / "finetune/jobs/sft.py"
    text = script.read_text()
    script.write_text(text.replace('-py3-none-any.whl"', '-py3-none-any.whl#sha256=' + 'a' * 64 + '"'))
    with pytest.raises(ValueError, match="policy"):
        launcher.prepare(launcher.parse_args(argv()))


def test_oversized_input_is_refused_without_reading_contents(synthetic_repo, tmp_path):
    large = tmp_path / "SYNTHETIC_sparse.jsonl"
    with large.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="size limit"):
        launcher.prepare(launcher.parse_args(argv("finetune/jobs/sft.py", "--allow-input-upload", "--input", f"{large}=data/large.jsonl")))


def test_artifact_readback_mismatch_blocks_paid_compute(synthetic_repo, monkeypatch, capsys):
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    monkeypatch.setattr(api, "list_bucket_tree", lambda **kwargs: [])
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit")) == 2
    assert "readback" in capsys.readouterr().err
    assert not any(name == "run_job" for name, _ in api.calls)


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, OSError])
def test_sdk_exception_details_and_secret_values_are_not_echoed(synthetic_repo, monkeypatch, capsys, error_type):
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    monkeypatch.setenv("QMD_SYNTHETIC_SECRET", "SYNTHETIC_SECRET_DO_NOT_ECHO")
    def fail(**kwargs):
        raise error_type("SYNTHETIC_SECRET_DO_NOT_ECHO")
    monkeypatch.setattr(api, "create_bucket", fail)
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit", "--secret-env", "QMD_SYNTHETIC_SECRET")) == 2
    captured = capsys.readouterr()
    assert "SYNTHETIC_SECRET_DO_NOT_ECHO" not in captured.out + captured.err
    assert "RuntimeError" in captured.err


@pytest.mark.parametrize("submit", [False, True])
@pytest.mark.parametrize("literal", [
    "HF_TOKEN=SYNTHETIC_SECRET_NEVER_ECHO",
    "HF_TOKEN SYNTHETIC_SECRET_NEVER_ECHO",
    "HF_TOKEN\nSYNTHETIC_SECRET_NEVER_ECHO",
])
def test_malformed_secret_name_is_refused_without_echo(synthetic_repo, monkeypatch, capsys, submit, literal):
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("invalid secret option reached SDK"))
    options = ["--secret-env", literal]
    if submit:
        options.insert(0, "--submit")
    assert launcher.main(argv("finetune/jobs/sft.py", *options)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unsupported environment name" in captured.err
    assert "SYNTHETIC_SECRET_NEVER_ECHO" not in captured.out + captured.err
    assert literal not in captured.out + captured.err


def test_sdk_public_bucket_warning_is_fatal(synthetic_repo, monkeypatch, capsys):
    import warnings
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    def became_public(*args, **kwargs):
        warnings.warn("Bucket 'synthetic-owner/jobs-artifacts' already exists and is public: data synced for the Job will be publicly accessible.", UserWarning)
        pytest.fail("warning should have aborted before upload")
    monkeypatch.setattr(api, "sync_job_volume", became_public)
    assert launcher.main(argv("finetune/jobs/sft.py", "--submit")) == 2
    assert not api.uploaded
    assert "no success confirmed" in capsys.readouterr().err


# Independent acceptance expectations, not a second traversal of CODE_CLOSURE.
GGUF_RUNNERS = ("finetune/convert_gguf.py", "finetune/train.py", "finetune/train_unsloth.py")
GGUF_HELPER = "finetune/gguf_toolchain.py"
GGUF_IMAGE = ("ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie@sha256:"
              "c8fe61165767c52118d7d3e67db5d0be5454c2e0da1f9fe15c697ebdd72f304e")
UNCHANGED_HELPERS = {
    "finetune/convert_onnx.py": (),
    "finetune/dataset/analyze_data.py": ("finetune/dataset/schema.py",),
    "finetune/dataset/prepare_data.py": ("finetune/dataset/schema.py",),
    "finetune/dataset/prepare_data_lfm2.py": ("finetune/dataset/schema.py",),
    "finetune/dataset/score_data.py": ("finetune/dataset/schema.py", "finetune/reward.py"),
    "finetune/dataset/validate_schema.py": ("finetune/dataset/schema.py",),
    "finetune/eval.py": ("finetune/reward.py", "finetune/evals/queries.txt"),
    "finetune/eval_retrieval.py": (),
    "finetune/experiments/grpo/grpo.py": ("finetune/jobs/eval_common.py",),
    "finetune/experiments/lfm2/sft_lfm2.py": (),
    "finetune/jobs/eval.py": (),
    "finetune/jobs/sft.py": ("finetune/jobs/eval_common.py",),
}


def expected_gguf_paths(runner):
    paths = {runner, runner + ".lock", GGUF_HELPER}
    if runner != "finetune/convert_gguf.py":
        paths.update({"finetune/eval.py", "finetune/eval.py.lock", "finetune/reward.py",
                      "finetune/evals/queries.txt"})
    if runner == "finetune/train_unsloth.py":
        paths.update({"finetune/train.py", "finetune/train.py.lock"})
    return paths


@pytest.fixture
def real_export_sources(synthetic_repo):
    """Stage actual source/lock bytes as DATA; never import/execute ML runners.

    This intentionally requires the integrated GGUF bodies and helper. A launcher
    proposal alone cannot qualify a future helper's bytes or export behavior.
    """
    source_root = LAUNCHER.parents[2]
    paths = {launcher.POLICY_PATH}
    for runner in GGUF_RUNNERS:
        paths.update(expected_gguf_paths(runner))
    for relative in paths:
        destination = synthetic_repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    return synthetic_repo


@pytest.mark.parametrize("runner", GGUF_RUNNERS)
def test_gguf_closure_follows_real_imports_and_export_calls(real_export_sources, runner):
    import ast
    root = real_export_sources
    plan, files = launcher.prepare(launcher.parse_args(argv(runner)))
    assert set(files) == expected_gguf_paths(runner)
    trees = {path: ast.parse(content) for path, content in files.items() if path.endswith(".py")}
    # Walk real (including deferred) local imports, independent of the upload list.
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            candidate = Path(path).parent / (node.module.replace(".", "/") + ".py")
            if (root / candidate).is_file():
                assert candidate.as_posix() in files, (path, node.lineno, node.module)
    imports = [node for node in ast.walk(trees[runner])
               if isinstance(node, ast.ImportFrom) and node.module == "gguf_toolchain"]
    assert imports, "Every GGUF-capable entrypoint must stage its actual prerequisite import"
    helper_definitions = {node.name for node in trees[GGUF_HELPER].body
                          if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "gguf_toolchain":
                assert {name.name for name in node.names} <= helper_definitions
    exporter_tree = trees["finetune/convert_gguf.py" if runner == "finetune/convert_gguf.py" else "finetune/train.py"]
    calls = {node.func.id for node in ast.walk(exporter_tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"prepare_toolchain", "convert_model"} <= calls
    if runner != "finetune/convert_gguf.py":
        run_eval = next(node for node in trees["finetune/train.py"].body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_eval")
        eval_path = ast.dump(ast.parse('Path(__file__).parent / "eval.py"', mode="eval").body)
        assert any(isinstance(node, ast.Assign) and ast.dump(node.value) == eval_path
                   for node in ast.walk(run_eval))
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
                   and node.func.attr == "run" for node in ast.walk(run_eval))
        query_path = ast.dump(ast.parse('Path(__file__).parent / "evals" / "queries.txt"', mode="eval").body)
        assert any(isinstance(node, ast.Assign) and ast.dump(node.value) == query_path
                   for node in trees["finetune/eval.py"].body)
    if runner == "finetune/train_unsloth.py":
        assert any(isinstance(node, ast.ImportFrom) and node.module == "train"
                   and any(name.name == "export_gguf" for name in node.names)
                   for node in ast.walk(trees[runner]))
    assert plan["image"] == GGUF_IMAGE


@pytest.mark.parametrize("runner", GGUF_RUNNERS)
def test_public_gguf_preview_stages_exact_helper_hash_and_preserves_argv(real_export_sources, monkeypatch, capsys, runner):
    import hashlib
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("preview reached SDK"))
    arguments = [str(real_export_sources / ".env"), "--image", "literal-script-value", "$(touch forbidden)", "*.jsonl", "", "--"]
    seen = {}
    original = launcher.stage_files

    def inspect_stage(files, directory):
        original(files, directory)
        seen.update({p.relative_to(directory).as_posix(): p.read_bytes()
                     for p in directory.rglob("*") if p.is_file()})

    monkeypatch.setattr(launcher, "stage_files", inspect_stage)
    assert launcher.main([*argv(runner), "--", *arguments]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["mode"] == "dry-run" and manifest["image"] == GGUF_IMAGE
    assert manifest["command"][5:11] == ["uv", "run", "--locked", "--no-project", "--script", runner.removeprefix("finetune/")]
    assert manifest["command"][11:] == arguments
    assert set(seen) == expected_gguf_paths(runner)
    assert seen[GGUF_HELPER] == (real_export_sources / GGUF_HELPER).read_bytes()
    assert {entry["path"] for entry in manifest["files"]} == set(seen)
    for entry in manifest["files"]:
        assert entry["sha256"] == hashlib.sha256(seen[entry["path"]]).hexdigest()
        assert entry["size"] == len(seen[entry["path"]])
    assert not any(b"PRIVATE SENTINEL" in content for content in seen.values())


@pytest.mark.parametrize("runner", GGUF_RUNNERS)
@pytest.mark.parametrize("change", ["missing", "directory", "symlink", "hardlink", "fifo"])
def test_changed_or_missing_gguf_helper_refused_before_sdk(synthetic_repo, monkeypatch, capsys, runner, change):
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("unsafe helper reached SDK"))
    helper = synthetic_repo / GGUF_HELPER
    if helper.exists():
        helper.unlink()
    if change == "directory":
        helper.mkdir()
    elif change == "symlink":
        helper.symlink_to(synthetic_repo / ".env")
    elif change == "hardlink":
        os.link(synthetic_repo / ".env", helper)
    elif change == "fifo":
        os.mkfifo(helper)
    assert launcher.main(argv(runner, "--submit")) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "gguf_toolchain.py" in captured.err


@pytest.mark.parametrize("runner,helpers", list(UNCHANGED_HELPERS.items()))
def test_unrelated_runners_keep_lightweight_image_and_exact_closure(synthetic_repo, runner, helpers):
    plan, files = launcher.prepare(launcher.parse_args(argv(runner)))
    assert plan["image"] == launcher.DEFAULT_IMAGE
    assert set(files) == {runner, runner + ".lock", *helpers}
    assert GGUF_HELPER not in files


@pytest.mark.parametrize("runner", [*GGUF_RUNNERS, "finetune/dataset/validate_schema.py", "finetune/eval.py"])
@pytest.mark.parametrize("image", ["example.invalid/custom/image:literal@sha256:" + "d" * 64, "", " invalid image "])
def test_explicit_image_is_literal_not_a_default_fallback(synthetic_repo, monkeypatch, capsys, runner, image):
    monkeypatch.setattr(launcher, "load_sdk", lambda: pytest.fail("explicit-image preview reached SDK"))
    args = launcher.parse_args(argv(runner, "--image", image))
    assert args.image == image
    assert launcher.prepare(args)[0]["image"] == image
    assert launcher.main(argv(runner, "--image", image)) == 0
    assert json.loads(capsys.readouterr().out)["image"] == image


@pytest.mark.parametrize("runner", GGUF_RUNNERS)
@pytest.mark.parametrize("image", [None, "example.invalid/explicit:literal"])
def test_mocked_sdk_receives_and_reads_back_selected_image(synthetic_repo, monkeypatch, capsys, runner, image):
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    options = [] if image is None else ["--image", image]
    assert launcher.main(argv(runner, "--submit", *options)) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert api.job_spec["image"] == manifest["image"] == (GGUF_IMAGE if image is None else image)
    assert set(api.uploaded) == expected_gguf_paths(runner)
    assert api.uploaded[GGUF_HELPER] == (synthetic_repo / GGUF_HELPER).read_bytes()


@pytest.mark.parametrize("runner", [*GGUF_RUNNERS, "finetune/dataset/validate_schema.py"])
@pytest.mark.parametrize("override", [False, True])
def test_real_cli_image_selection_is_stdlib_only(synthetic_repo, tmp_path, runner, override):
    copied = synthetic_repo / "finetune/tools/submit_locked_job.py"
    copied.parent.mkdir(exist_ok=True)
    copied.write_bytes(LAUNCHER.read_bytes())
    options = ["--image", "example.invalid/cli:literal"] if override else []
    result = subprocess.run([sys.executable, "-I", "-B", "-S", str(copied),
                             *argv(runner, *options), "--", "--image", "script-literal"],
                            cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["image"] == ("example.invalid/cli:literal" if override else
                                  GGUF_IMAGE if runner in GGUF_RUNNERS else launcher.DEFAULT_IMAGE)
    assert manifest["command"][-2:] == ["--image", "script-literal"]


@pytest.mark.parametrize("runner", GGUF_RUNNERS)
def test_mocked_image_readback_change_is_not_success(synthetic_repo, monkeypatch, capsys, runner):
    api = FakeApi()
    fake_sdk(monkeypatch, api)
    original = api.inspect_job

    def wrong_image(**kwargs):
        result = original(**kwargs)
        result.docker_image = "example.invalid/unrequested:image"
        return result

    monkeypatch.setattr(api, "inspect_job", wrong_image)
    assert launcher.main(argv(runner, "--submit", "--image", "example.invalid/requested:literal")) == 2
    captured = capsys.readouterr()
    assert not captured.out and "readback failed" in captured.err
    assert [name for name, _ in api.calls].count("run_job") == 1


@pytest.mark.parametrize("missing", ["git", "cc", "c++", "make", "uv"])
def test_native_image_probe_missing_prerequisite_is_bounded_and_offline(tmp_path, missing):
    probe = LAUNCHER.parent / "qualify_gguf_image.py"
    tools = tmp_path / "synthetic-tools"
    tools.mkdir()
    for name in ("git", "cc", "c++", "make", "uv"):
        if name != missing:
            path = tools / name
            path.write_text("#!/bin/sh\nprintf '%s\\n' 'SYNTHETIC tool version'\n")
            path.chmod(0o700)
    result = subprocess.run([sys.executable, "-I", "-B", "-S", str(probe)],
                            env={**os.environ, "PATH": str(tools)}, cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and not result.stdout
    assert "missing native prerequisite: " + missing in result.stderr
    assert set(tmp_path.iterdir()) == {tools}


def test_native_image_probe_wrong_uv_version_fails_before_build(tmp_path):
    probe = LAUNCHER.parent / "qualify_gguf_image.py"
    tools = tmp_path / "synthetic-tools"
    tools.mkdir()
    for name in ("git", "cc", "c++", "make", "uv"):
        path = tools / name
        path.write_text("#!/bin/sh\nprintf '%s\\n' 'uv 0.0.0'\n")
        path.chmod(0o700)
    result = subprocess.run([sys.executable, "-I", "-B", "-S", str(probe)],
                            env={**os.environ, "PATH": str(tools), "TMPDIR": str(tmp_path)}, cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and not result.stdout
    assert "requires uv 0.11.18" in result.stderr
    assert set(tmp_path.iterdir()) == {tools}


@pytest.mark.parametrize("banner", [
    "uv 0.11.18",
    "uv 0.11.18 (x86_64-unknown-linux-musl)",
    "uv 0.11.18 (e32666915 2026-06-01 x86_64-unknown-linux-musl)",
])
def test_native_image_probe_accepts_exact_release_with_build_metadata(tmp_path, banner):
    """Synthetic uv display, REAL host compilation; not image qualification."""
    import shlex

    probe = LAUNCHER.parent / "qualify_gguf_image.py"
    tools = tmp_path / "synthetic-version-only"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(banner) + "\n")
    uv.chmod(0o700)
    env = {"PATH": str(tools) + os.pathsep + os.environ.get("PATH", os.defpath),
           "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "QMD_LLAMA_GPU": "false", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    result = subprocess.run([sys.executable, "-I", "-B", "-S", str(probe)], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["versions"]["uv"] == banner
    assert receipt["native_only"] is False
    assert receipt["outputs"] == ["qmd-c-ok", "qmd-cxx-ok"]
    assert receipt["locked_python_converter_training_gpu_cloud_qualified"] is False
    assert set(tmp_path.iterdir()) == {tools}


@pytest.mark.parametrize("banner", [
    "uv 0.11.17 (x86_64-unknown-linux-musl)",
    "uv 0.11.180 (x86_64-unknown-linux-musl)",
    "uv 0.11.18+1 (e32666915 2026-06-01 x86_64-unknown-linux-musl)",
    "uv 0.11.18rc1 (x86_64-unknown-linux-musl)",
    "uv 0.11.18.dev1 (x86_64-unknown-linux-musl)",
    "uv 0.11.18 trailing",
    "other 0.11.18",
    "uv 0.11.18 ()",
    "uv 0.11.18 (unterminated",
    "uv 0.11.18 (x86_64-unknown-linux-musl) trailing",
])
def test_native_image_probe_does_not_normalize_or_prefix_match_release(tmp_path, banner):
    import shlex

    probe = LAUNCHER.parent / "qualify_gguf_image.py"
    tools = tmp_path / "synthetic-tools"
    tools.mkdir()
    for name in ("git", "cc", "c++", "make", "uv"):
        executable = tools / name
        executable.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(banner) + "\n")
        executable.chmod(0o700)
    env = {"PATH": str(tools), "HOME": str(tmp_path), "TMPDIR": str(tmp_path),
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "QMD_LLAMA_GPU": "false"}
    result = subprocess.run([sys.executable, "-I", "-B", "-S", str(probe)], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and not result.stdout
    assert "requires uv 0.11.18" in result.stderr
    assert set(tmp_path.iterdir()) == {tools}
