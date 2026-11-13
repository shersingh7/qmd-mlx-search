#!/usr/bin/env python3
"""Dry-run by default: stage an allowlisted runner, helpers and reviewed uv locks.

Only --submit initializes the Hugging Face client or performs network operations.
No script argument is interpreted as a local file. See finetune/LOCKED-JOBS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import tomllib
from urllib.parse import quote
import uuid
import warnings

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = "docs/security/dependency-remediation/patched-dependencies.json"
# Explicit reviewed runtime closure, NOT a glob or a recursive checkout upload.
CODE_CLOSURE = {
    "finetune/convert_gguf.py": (),
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
    "finetune/train.py": ("finetune/eval.py", "finetune/reward.py", "finetune/evals/queries.txt"),
    "finetune/train_unsloth.py": ("finetune/train.py", "finetune/eval.py", "finetune/reward.py", "finetune/evals/queries.txt"),
}
# Registry-verified multi-architecture digest; uv matches the lock controls.
# Registry metadata was read, but this image has not been pulled or executed.
DEFAULT_IMAGE = ("ghcr.io/astral-sh/uv:0.11.18-python3.12-trixie-slim@sha256:"
                 "9f9a7c30d604c8297ac14321b91b42f8516ff207f42b76b1bf29974fad523f6e")
MOUNT = "/qmd-code"
# Fixed shell text only. Script args are passed as "$@", never interpolated.
# A writable copy preserves relative inputs/outputs and __file__ helper paths.
BOOTSTRAP = ('source="$1"; shift; work="$(mktemp -d "${TMPDIR:-/tmp}/qmd-locked.XXXXXXXXXX")" && cp -R "$source/." "$work/" '
             '&& cd "$work/finetune" && exec "$@"')
SCRIPT_BLOCK = re.compile(r"(?m)^# /// script\n(?P<body>(?:#.*\n)*?)# ///\n")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INPUT_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".tsv", ".parquet"}
PRIVATE_NAMES = {"token", "tokens", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
PRIVATE_DIRS = {".git", ".ssh", ".aws", ".azure", ".kube", ".gnupg", ".venv",
                ".cache", ".config", ".local", ".huggingface", "__pycache__"}


class SubmissionRefused(ValueError):
    """A launcher-authored, safe-to-print refusal, not an SDK error body."""


def read_regular(path: Path, limit: int = 16 * 1024 * 1024) -> bytes:
    """Read a bounded, singly linked regular file without following ANY symlink.

    Directory descriptors keep the ancestry check and the open on the same path.
    O_NONBLOCK prevents an unexpected FIFO/device from blocking before fstat.
    """
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a safe absolute file path: {path}")
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            # The raw descriptor has one owner even if wrapper construction,
            # __enter__, reading or teardown raises (including BaseException).
            with os.fdopen(fd, "rb", closefd=False) as stream:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError(f"Expected a regular file without hardlinks: {path}")
                if info.st_size > limit:
                    raise ValueError(f"File exceeds staging size limit: {path}")
                content = stream.read(limit + 1)
                if len(content) > limit:
                    raise ValueError(f"File exceeds staging size limit: {path}")
                return content
        finally:
            os.close(fd)
    except OSError as exc:
        raise ValueError(f"Missing or unsafe file (symlink/nonregular/unreadable): {path}") from exc
    finally:
        os.close(directory)


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate publication policy field")
        result[key] = value
    return result


def load_policy(root: Path) -> dict:
    raw = read_regular(root / POLICY_PATH, 1024 * 1024)
    def invalid_constant(_):
        raise ValueError("Invalid publication policy numeric constant")
    policy = json.loads(raw, object_pairs_hook=unique_fields, parse_constant=invalid_constant)
    fields = {"advisory", "version", "wheel_path", "sha256", "commit", "immutable_url", "runner_paths"}
    if (not isinstance(policy, dict) or set(policy) != {"schema_version", "accelerate"}
            or type(policy["schema_version"]) is not int or policy["schema_version"] != 1
            or not isinstance(policy["accelerate"], dict) or set(policy["accelerate"]) != fields):
        raise ValueError("Invalid publication policy schema")
    item = policy["accelerate"]
    for field, pattern in [("version", r"[0-9]+\.[0-9]+\.[0-9]+\+[a-z0-9]+(?:\.[a-z0-9]+)*"),
                           ("sha256", r"[0-9a-f]{64}"), ("commit", r"[0-9a-f]{40}")]:
        if not isinstance(item[field], str) or not re.fullmatch(pattern, item[field]):
            raise ValueError("Invalid publication policy identity")
    wheel = f"finetune/vendor/accelerate/accelerate-{item['version']}-py3-none-any.whl"
    url = (f"https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/{item['commit']}/"
           f"{quote(wheel, safe='/')}#sha256={item['sha256']}")
    runners = item["runner_paths"]
    if (item["advisory"] != "GHSA-4j2p-28q2-5m79" or item["wheel_path"] != wheel
            or item["immutable_url"] != url or not isinstance(runners, list)
            or not all(isinstance(p, str) for p in runners)
            or len(runners) != len(set(runners)) or set(runners) != set(CODE_CLOSURE)):
        raise ValueError("Publication policy and reviewed runner closure disagree")
    return item


def name_of(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", re.split(r"[\[<>=!~; @]", requirement, maxsplit=1)[0]).lower()


def check_script_lock(script: bytes, lock: bytes, policy: dict, path: str) -> None:
    """Check source/hash binding locally; uv checks graph freshness remotely.

    This does not regenerate a missing lock, approve a patch, or resolve packages.
    """
    matches = list(SCRIPT_BLOCK.finditer(script.decode("utf-8")))
    if len(matches) != 1:
        raise ValueError(f"Expected one PEP 723 block: {path}")
    metadata = tomllib.loads("\n".join(line.removeprefix("# ").removeprefix("#")
                                     for line in matches[0]["body"].splitlines()))
    uv = metadata.get("tool", {}).get("uv", {})
    url = policy["immutable_url"].split("#", 1)[0]
    pin = "accelerate==" + policy["version"]
    dependencies = metadata.get("dependencies", [])
    constraints = uv.get("constraint-dependencies", [])
    if (not isinstance(dependencies, list) or not all(isinstance(r, str) for r in dependencies)
            or not isinstance(constraints, list) or not all(isinstance(r, str) for r in constraints)
            or [r for r in constraints if name_of(r) == "accelerate"] != [pin]
            or uv.get("sources", {}).get("accelerate") != {"url": url}):
        raise ValueError(f"Script source/constraint does not match publication policy: {path}")
    direct = [r for r in dependencies if name_of(r) == "accelerate"]
    if direct and direct != [pin]:
        raise ValueError(f"Script Accelerate dependency is not the policy pin: {path}")
    data = tomllib.loads(lock.decode("utf-8"))
    manifest = data.get("manifest", {})
    locked_requirements = [p for p in manifest.get("requirements", []) if p.get("name") == "accelerate"]
    locked_constraints = [p for p in manifest.get("constraints", []) if p.get("name") == "accelerate"]
    packages = data.get("package", [])
    instances = [p for p in packages if p.get("name") == "accelerate"]
    needs_accelerate = bool(direct) or any(
        dep.get("name") == "accelerate" for p in packages for dep in p.get("dependencies", []))
    if (data.get("version") != 1 or not isinstance(packages, list)
            or data.get("requires-python") != metadata.get("requires-python")
            or locked_requirements != ([{"name": "accelerate", "url": url}] if direct else [])
            or locked_constraints != [{"name": "accelerate", "url": url}]
            or len(instances) > 1 or (needs_accelerate and not instances)):
        raise ValueError(f"Invalid or unbound script lock: {path}.lock")
    if instances:
        package = instances[0]
        wheels = package.get("wheels", [])
        if (package.get("version") != policy["version"] or package.get("source") != {"url": url}
                or package.get("sdist") is not None or len(wheels) != 1
                or wheels[0].get("url") != url or wheels[0].get("hash") != "sha256:" + policy["sha256"]):
            raise ValueError(f"Lock wheel URL/hash does not match publication policy: {path}.lock")


def safe_relative(value: str) -> PurePosixPath:
    if (not value or "\\" in value or any(c in value for c in "*?[]")
            or any(ord(c) < 32 for c in value)):
        raise ValueError("Expected a literal, safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} or part.startswith(".") for part in value.split("/")):
        raise ValueError("Expected a literal, safe relative path")
    return path


def check_env_name(name: str) -> None:
    if (not ENV_NAME.fullmatch(name) or name.startswith(("UV_", "PYTHON", "LD_"))
            or name in {"PATH", "VIRTUAL_ENV", "BASH_ENV", "ENV"}):
        raise ValueError(f"Unsupported environment name: {name}")


def remote_volumes(args) -> list[dict]:
    volumes = []
    for specs, read_only in [(args.volume, True), (args.writable_volume, False)]:
        for spec in specs:
            parts = spec.split("=")
            if len(parts) != 3 or parts[0] not in {"bucket", "dataset", "model", "space"}:
                raise ValueError("Volume syntax: TYPE=NAMESPACE/NAME=/inputs/NAME (or /outputs/NAME)")
            kind, source, mount = parts
            if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", source)
                    or not mount.startswith("/") or "//" in mount):
                raise ValueError("Invalid remote volume source or mount")
            safe_relative(mount[1:])
            if PurePosixPath(mount).parts[1] not in {"inputs", "outputs"}:
                raise ValueError("User volumes must mount under /inputs or /outputs, not reserved code/runtime paths")
            for existing in volumes:
                other = existing["mount_path"]
                if mount == other or mount.startswith(other + "/") or other.startswith(mount + "/"):
                    raise ValueError("Overlapping remote volume mounts")
            volumes.append({"type": kind, "source": source, "mount_path": mount, "read_only": read_only})
    return volumes


def prepare(args) -> tuple[dict, dict[str, bytes]]:
    """Complete local preflight and immutable byte snapshot before any SDK call."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.namespace):
        raise ValueError("Specify a literal HF namespace")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.flavor):
        raise ValueError("Specify a literal HF hardware flavor")
    if not re.fullmatch(r"[1-9][0-9]*(?:s|m|h|d)?", args.timeout):
        raise ValueError("Specify a positive timeout, e.g. 30m")
    policy = load_policy(ROOT)
    safe_relative(args.runner)
    if args.runner not in CODE_CLOSURE:
        raise ValueError("Runner is not in the reviewed publication policy/closure")
    paths = {args.runner, *CODE_CLOSURE[args.runner]}
    paths.update(p + ".lock" for p in paths.copy() if p in CODE_CLOSURE)
    files = {path: read_regular(ROOT / path) for path in sorted(paths)}
    for path in sorted(paths & CODE_CLOSURE.keys()):
        check_script_lock(files[path], files[path + ".lock"], policy, path)
    kinds = {path: "reviewed-code-or-lock" for path in files}
    if args.input and not args.allow_input_upload:
        raise ValueError("Explicit input upload requires --allow-input-upload; review every input first")
    for spec in args.input:
        source, separator, destination = spec.partition("=")
        if not separator:
            raise ValueError("Input syntax: /absolute/local/file=relative/remote/file")
        target = "finetune/" + str(safe_relative(destination))
        if (PurePosixPath(destination).suffix.lower() not in INPUT_SUFFIXES
                or any(part.lower() in PRIVATE_NAMES for part in PurePosixPath(destination).parts)):
            raise ValueError("Inputs may only be data/config/text files, not code, uv config, weights or credentials")
        if any(target == p or target.startswith(p + "/") or p.startswith(target + "/") for p in files):
            raise ValueError("Staging path collision")
        local = Path(source)
        if (not local.is_absolute() or ".." in local.parts or any(c in source for c in "*?[]")
                or any(ord(c) < 32 for c in source)
                or local.name.startswith(".") or local.name.lower() in PRIVATE_NAMES
                or any(part.lower() in PRIVATE_DIRS for part in local.parts)):
            raise ValueError("Input must be an absolute regular non-credential file, without glob/traversal/symlink")
        files[target] = read_regular(local, 64 * 1024 * 1024)
        kinds[target] = "input"
    if sum(map(len, files.values())) > 128 * 1024 * 1024:
        raise ValueError("Staging exceeds 128 MiB; explicitly mount a remote volume for large data")
    env = {"UV_NO_CONFIG": "1"}
    for spec in args.env:
        name, separator, value = spec.partition("=")
        check_env_name(name)
        if re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|PRIVATE_KEY", name, re.IGNORECASE):
            raise ValueError("Sensitive environment names require --secret-env, not --env")
        if not separator or name in env:
            raise ValueError("Environment syntax/duplicate: NAME=VALUE")
        env[name] = value
    for name in args.secret_env:
        check_env_name(name)
        if name in env or args.secret_env.count(name) > 1:
            raise ValueError("Duplicate environment/secret name")
    manifest = {
        "mode": "dry-run", "namespace": args.namespace, "flavor": args.flavor,
        "timeout": args.timeout, "attempts": 1, "image": DEFAULT_IMAGE,
        "resource_group_id": args.resource_group_id,
        "command": ["sh", "-c", BOOTSTRAP, "qmd-locked-job", MOUNT, "uv", "run", "--locked", "--no-project", "--script",
                    args.runner.removeprefix("finetune/"), *args.script_args],
        "env": env, "secret_env_names": list(args.secret_env), "volumes": remote_volumes(args),
        "artifacts": {"bucket_id": args.namespace + "/jobs-artifacts", "remote_name": "qmd-locked-" + uuid.uuid4().hex,
                      "mount_path": MOUNT, "read_only": True},
        "files": [{"path": p, "kind": kinds[p], "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for p, content in sorted(files.items())],
        "local_staging": "temporary; deleted when this command exits",
    }
    return manifest, files


def stage_files(files: dict[str, bytes], directory: Path) -> None:
    for relative, content in files.items():
        destination = directory / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)


def load_sdk():
    # Deliberately absent from import/help/preflight/dry-run paths.
    from huggingface_hub import HfApi, Volume
    api = HfApi()
    if not all(callable(getattr(api, name, None)) for name in (
            "create_bucket", "bucket_info", "sync_job_volume", "list_bucket_tree", "run_job", "inspect_job")):
        raise SubmissionRefused("Submission requires huggingface_hub with the public Jobs/Buckets APIs (2.4 or newer)")
    return api, Volume


def submit(manifest: dict, directory: Path) -> dict:
    secrets = {}
    for name in manifest["secret_env_names"]:
        value = os.environ.get(name)
        if not value:
            raise SubmissionRefused(f"Required secret environment variable is unset: {name}")
        secrets[name] = value
    api, volume_type = load_sdk()
    artifacts = manifest["artifacts"]
    bucket_id = artifacts["bucket_id"]
    api.create_bucket(bucket_id=bucket_id, private=True, exist_ok=True)
    info = api.bucket_info(bucket_id=bucket_id)
    if info.id != bucket_id or info.private is not True:
        raise SubmissionRefused("Refusing upload: jobs-artifacts bucket must be verified private")
    with warnings.catch_warnings():
        # The SDK otherwise only WARNS and uploads if an existing bucket is public.
        warnings.filterwarnings("error", message=r"Bucket .*already exists and is public:.*", category=UserWarning)
        volume = api.sync_job_volume(directory, MOUNT, remote_name=artifacts["remote_name"],
                                     read_only=True, namespace=manifest["namespace"])
    if (volume.source != bucket_id or volume.path != artifacts["remote_name"]
            or volume.mount_path != MOUNT or volume.read_only is not True):
        raise SubmissionRefused("Uploaded volume does not match the requested private staging target")
    # Read back the exact remote directory metadata before starting paid compute.
    expected = {artifacts["remote_name"] + "/" + item["path"]: item["size"] for item in manifest["files"]}
    actual = {item.path: item.size for item in api.list_bucket_tree(
        bucket_id=bucket_id, prefix=artifacts["remote_name"] + "/", recursive=True) if hasattr(item, "size")}
    if actual != expected:
        raise SubmissionRefused("Uploaded artifact directory readback does not match staged paths/sizes")
    volumes = [volume] + [volume_type(**spec) for spec in manifest["volumes"]]
    job = api.run_job(image=manifest["image"], command=manifest["command"],
                      env=manifest["env"], secrets=secrets, flavor=manifest["flavor"],
                      timeout=manifest["timeout"], attempts=1, namespace=manifest["namespace"],
                      volumes=volumes, resource_group_id=manifest["resource_group_id"],
                      labels={}, expose=[], expose_public=[], ssh=False)
    try:
        confirmed = api.inspect_job(job_id=job.id, namespace=manifest["namespace"])
        if (confirmed.id != job.id or confirmed.command != manifest["command"]
                or confirmed.docker_image != manifest["image"] or confirmed.flavor != manifest["flavor"]):
            raise ValueError("Job spec mismatch")
    except Exception:
        raise SubmissionRefused(f"Job {job.id} was created but readback failed; inspect/cancel it before retrying") from None
    return {**manifest, "mode": "submitted", "job_id": confirmed.id, "job_url": confirmed.url}


def parse_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Exact repository-relative policy-listed runner")
    parser.add_argument("--namespace", required=True, help="Explicit HF user/organization namespace")
    parser.add_argument("--flavor", required=True, help="Explicit HF hardware flavor; may incur charges")
    parser.add_argument("--timeout", required=True, help="Explicit job deadline, e.g. 30m")
    parser.add_argument("--submit", action="store_true", help="OPT IN to private uploads and a paid remote job")
    parser.add_argument("--input", action="append", default=[], metavar="LOCAL=REMOTE", help="One explicit absolute input file; no directories/globs")
    parser.add_argument("--allow-input-upload", action="store_true", help="Consent to uploading every --input file")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="Explicit non-secret remote environment value")
    parser.add_argument("--secret-env", action="append", default=[], metavar="NAME", help="Forward this environment variable as a remote secret only with --submit")
    parser.add_argument("--volume", action="append", default=[], metavar="TYPE=SOURCE=MOUNT", help="Explicit read-only remote volume under /inputs or /outputs")
    parser.add_argument("--writable-volume", action="append", default=[], metavar="TYPE=SOURCE=MOUNT", help="Explicit read-write remote volume under /inputs or /outputs")
    parser.add_argument("--resource-group-id", default=None, help="Optional explicit HF resource group")
    args = parser.parse_args(arguments[:separator])
    args.script_args = arguments[separator + 1:] if separator < len(arguments) else []
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        manifest, files = prepare(args)
        with tempfile.TemporaryDirectory(prefix="qmd-locked-job-") as temporary:
            directory = Path(temporary) / "upload"
            directory.mkdir(mode=0o700)
            stage_files(files, directory)
            if args.submit:
                try:
                    manifest = submit(manifest, directory)
                except SubmissionRefused:
                    raise
                except Exception:
                    # Even SDK ValueError/OSError bodies can contain auth values.
                    raise RuntimeError("HF SDK operation failed") from None
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Locked job refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # SDK errors can contain auth/request details: do not echo them or secrets.
        print(f"Locked job failed ({type(exc).__name__}); no success confirmed. Check the private artifact bucket and Jobs before retrying.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
