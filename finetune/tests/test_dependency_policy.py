"""Security floors apply to every locked variant and standalone script."""
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


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
