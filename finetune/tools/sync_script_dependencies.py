#!/usr/bin/env python3
"""Keep standalone uv runners on the same audited policy as the project.

PEP 723 scripts resolve independently of finetune/uv.lock. Run this after
changing project requirements; --check is read-only and suitable for CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_BLOCK = re.compile(r"(?m)^# /// script\n(?P<body>(?:#.*\n)*?)# ///\n")


def metadata(text: str) -> dict | None:
    match = SCRIPT_BLOCK.search(text)
    if match is None:
        return None
    return tomllib.loads("\n".join(line.removeprefix("# ").removeprefix("#") for line in match["body"].splitlines()))


def package_name(requirement: str) -> str:
    original_name = re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", original_name).lower()


def render_block(old: dict, project: dict) -> str:
    project_requirements = {package_name(req): req for req in project["project"]["dependencies"]}
    requirements = [project_requirements.get(package_name(req), req) for req in old["dependencies"]]
    # Quantized optimizers are CUDA/Linux-only; retain them for GPU jobs rather
    # than making portable evaluation/dry-runs install a CUDA-only extension.
    requirements = [
        'bitsandbytes>=0.50.2; sys_platform == "linux"' if package_name(req) == "bitsandbytes" else req
        for req in requirements
    ]
    lines = ["# /// script", f'# requires-python = "{project["project"]["requires-python"]}"', "# dependencies = ["]
    # JSON quoting is valid TOML and escapes platform-marker quotes correctly.
    import json
    lines.extend(f"#     {json.dumps(req)}," for req in requirements)
    lines.extend(["# ]", "# [tool.uv]", "# constraint-dependencies = ["])
    lines.extend(f"#     {json.dumps(req)}," for req in project["tool"]["uv"]["constraint-dependencies"])
    lines.extend(["# ]", "# ///", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    changed = []
    count = 0
    for path in sorted(ROOT.rglob("*.py")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text()
        old = metadata(text)
        if old is None and re.search(r"(?m)^# /// script$", text):
            raise ValueError(f"Malformed PEP 723 block: {path.relative_to(ROOT)}")
        if old is None or not old.get("dependencies"):
            continue
        count += 1
        block = render_block(old, project)
        updated = SCRIPT_BLOCK.sub(lambda _: block, text, count=1)
        if updated != text:
            changed.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(updated)
    if args.check and changed:
        print("Stale PEP 723 policy: " + ", ".join(changed))
        return 1
    print(f"{'Checked' if args.check else 'Synchronized'} {count} dependency-bearing PEP 723 runners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
