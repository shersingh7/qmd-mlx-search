"""Security floors apply to every locked variant and standalone script."""
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_the_linux_quantized_optimizer():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    optimizer = [value for value in requirements if canonicalize_name(value.name) == "bitsandbytes"]
    assert len(optimizer) == 1, "Native CUDA's adamw_8bit must exist in the project install"
    assert optimizer[0].marker is not None
    assert optimizer[0].marker.evaluate({"sys_platform": "linux"})
    assert not optimizer[0].marker.evaluate({"sys_platform": "darwin"})
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = [value["version"] for value in lock["package"] if value["name"] == "bitsandbytes"]
    assert versions and all(optimizer[0].specifier.contains(value) for value in versions)


def test_lock_and_every_script_share_the_policy():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert project["project"]["requires-python"] == lock["requires-python"] == ">=3.11"
    constraints = [Requirement(req) for req in project["tool"]["uv"]["constraint-dependencies"]]
    packages = {}
    for package in lock["package"]:
        if "version" in package:
            packages.setdefault(canonicalize_name(package["name"]), []).append(Version(package["version"]))
    assert len(constraints) == 23
    for constraint in constraints:
        for version in packages.get(canonicalize_name(constraint.name), []):
            assert constraint.specifier.contains(version), (constraint.name, version, constraint.specifier)
    # Monitoring remains a project dependency, not removed to shrink the audit.
    assert "trackio" in packages
    assert "torch" in packages and "trl" in packages and "peft" in packages
    check = subprocess.run([sys.executable, "tools/sync_script_dependencies.py", "--check"], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stdout + check.stderr


def test_all_standalone_lock_graphs_preserve_floors_and_reviewed_wheel_hashes():
    policy = json.loads((ROOT.parent / 'docs/security/dependency-remediation/patched-dependencies.json').read_text())['accelerate']
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    constraints = [Requirement(value) for value in project['tool']['uv']['constraint-dependencies']]
    url = policy['immutable_url'].partition('#')[0]
    assert len(policy['runner_paths']) == len(set(policy['runner_paths'])) == 15
    data_only = {'finetune/dataset/' + name + '.py' for name in (
        'analyze_data', 'prepare_data', 'prepare_data_lfm2', 'score_data', 'validate_schema')}
    accelerate_instances = 0
    for runner in policy['runner_paths']:
        source = ROOT.parent / (runner + '.lock')
        lock = tomllib.loads(source.read_text())
        assert lock['version'] == 1 and lock['revision'] == 3
        assert lock['requires-python'] == project['project']['requires-python']
        packages = {}
        for package in lock['package']:
            name = canonicalize_name(package['name'])
            packages.setdefault(name, []).append(package)
            if name == 'accelerate':
                accelerate_instances += 1
                assert package['version'] == policy['version'], runner
                assert package['source'] == {'url': url}, runner
                assert package['wheels'] == [{'url': url, 'hash': 'sha256:' + policy['sha256']}], runner
                assert 'sdist' not in package
        for constraint in constraints:
            for package in packages.get(canonicalize_name(constraint.name), []):
                assert constraint.specifier.contains(package['version']), (runner, constraint.name, package['version'])
        if runner in data_only:
            assert not {'accelerate', 'torch'} & packages.keys(), runner
    assert accelerate_instances == 10
