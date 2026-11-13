#!/usr/bin/env python3
"""Keep standalone uv runners on the same audited policy as the project.

PEP 723 scripts resolve independently of finetune/uv.lock. Run this after
changing project requirements; --check is read-only and suitable for CI.
A reviewed local wheel is translated to its immutable publication URL, never
to a runner-relative path. uv enforces hashes from checked-in script locks,
not from a URL fragment. This formatter does not grant source approval;
scripts/check_dependency_security.py verifies the complete binding separately.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import tomllib
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_BLOCK = re.compile(r"(?m)^# /// script\n(?P<body>(?:#.*\n)*?)# ///\n")
POLICY_PATH = 'docs/security/dependency-remediation/patched-dependencies.json'


def metadata(text: str) -> dict | None:
    matches = list(SCRIPT_BLOCK.finditer(text))
    if len(matches) > 1:
        raise ValueError('Duplicate PEP 723 script blocks')
    if not matches:
        return None
    return tomllib.loads("\n".join(
        line.removeprefix("# ").removeprefix("#") for line in matches[0]["body"].splitlines()
    ))


def package_name(requirement: str) -> str:
    original_name = re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", original_name).lower()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate publication policy field')
        result[key] = value
    return result


def _invalid_constant(_):
    raise ValueError('Invalid publication policy numeric constant')


def script_sources(project: dict, root: Path) -> dict:
    """Translate the supported project wheel source, without approving it."""
    uv = project['tool']['uv']
    sources = uv.get('sources', {})
    if not sources:
        return {}
    if not isinstance(sources, dict) or set(sources) != {'accelerate'}:
        raise ValueError('Unsupported project source: an explicit publication policy is required')
    try:
        with (root.parent / POLICY_PATH).open('rb') as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError('Publication policy exceeds the read budget')
        policy = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique, parse_constant=_invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('A readable publication policy is required before source propagation') from exc
    if (not isinstance(policy, dict) or set(policy) != {'schema_version', 'accelerate'}
            or type(policy['schema_version']) is not int or policy['schema_version'] != 1):
        raise ValueError('Unsupported publication policy schema')
    item = policy['accelerate']
    fields = {'advisory', 'version', 'wheel_path', 'sha256', 'commit', 'immutable_url', 'runner_paths'}
    if not isinstance(item, dict) or set(item) != fields:
        raise ValueError('Incomplete publication policy')
    version, digest, commit = item['version'], item['sha256'], item['commit']
    if (item['advisory'] != 'GHSA-4j2p-28q2-5m79'
            or not isinstance(version, str)
            or not re.fullmatch(r'(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\+[a-z0-9]+(?:\.[a-z0-9]+)*', version)
            or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
            or not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit)):
        raise ValueError('Invalid reviewed publication identity')
    wheel = f'finetune/vendor/accelerate/accelerate-{version}-py3-none-any.whl'
    url = (f'https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/{commit}/'
           f'{quote(wheel, safe="/")}#sha256={digest}')
    if (item['wheel_path'] != wheel or item['immutable_url'] != url
            or sources['accelerate'] != {'path': wheel.removeprefix('finetune/')}):
        raise ValueError('Project source and immutable publication policy disagree')
    for requirements in (project['project']['dependencies'], uv['constraint-dependencies']):
        pins = [requirement for requirement in requirements if package_name(requirement) == 'accelerate']
        if pins != [f'accelerate=={version}']:
            raise ValueError('Project and constraint pins must exactly match the publication policy')
    # The policy retains the full hash-bound identity. uv's sources schema
    # rejects fragments, and PEP 508 URL fragments are not install-time hash
    # enforcement for uv run. Checked-in <script>.py.lock wheels bind SHA-256.
    return {'accelerate': {'url': url.partition('#')[0]}}


def _key(key: str) -> str:
    if not isinstance(key, str):
        raise ValueError('Metadata keys must be strings')
    return key if re.fullmatch(r'[A-Za-z0-9_-]+', key) else json.dumps(key)


def _literal(value) -> str:
    if isinstance(value, (str, bool, int, float)):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, list):
        return '[' + ', '.join(_literal(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{ ' + ', '.join(f'{_key(key)} = {_literal(item)}' for key, item in value.items()) + ' }'
    raise ValueError('Unsupported metadata value; refusing to silently discard it')


def _table_lines(table: dict, prefix=()):
    if prefix:
        yield '[' + '.'.join(_key(key) for key in prefix) + ']'
    for key, value in table.items():
        if isinstance(value, dict):
            continue
        if isinstance(value, list):
            yield f'{_key(key)} = ['
            yield from ('    ' + _literal(item) + ',' for item in value)
            yield ']'
        else:
            yield f'{_key(key)} = {_literal(value)}'
    for key, value in table.items():
        if isinstance(value, dict):
            yield from _table_lines(value, (*prefix, key))


def render_block(old: dict, project: dict, *, sources: dict | None = None) -> str:
    project_requirements = {package_name(req): req for req in project['project']['dependencies']}
    requirements = [project_requirements.get(package_name(req), req) for req in old['dependencies']]
    if project['tool']['uv'].get('sources') and not sources:
        raise ValueError('Local project sources need a reviewed standalone publication mapping')
    # Preserve unrelated tool/options/source metadata rather than deleting it
    # while updating the audited requirements. Only Accelerate is rebound.
    data = {'requires-python': project['project']['requires-python'], 'dependencies': requirements,
            **{key: deepcopy(value) for key, value in old.items() if key not in {'requires-python', 'dependencies'}}}
    uv = data.setdefault('tool', {}).setdefault('uv', {})
    uv['constraint-dependencies'] = deepcopy(project['tool']['uv']['constraint-dependencies'])
    if sources:
        uv.setdefault('sources', {}).update(deepcopy(sources))
    body = '\n'.join(_table_lines(data))
    if tomllib.loads(body) != data:
        raise ValueError('Metadata did not round-trip without modification')
    return '# /// script\n' + ''.join('# ' + line + '\n' for line in body.splitlines()) + '# ///\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    sources = script_sources(project, ROOT)
    changed = []
    count = 0
    for path in sorted(ROOT.rglob('*.py')):
        if any(part.startswith('.') or part in {'vendor', '__pycache__', 'node_modules'}
               for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text()
        old = metadata(text)
        if old is None and re.search(r'(?m)^# /// script$', text):
            raise ValueError(f'Malformed PEP 723 block: {path.relative_to(ROOT)}')
        if old is None or not old.get('dependencies'):
            continue
        count += 1
        block = render_block(old, project, sources=sources)
        updated = SCRIPT_BLOCK.sub(lambda _: block, text, count=1)
        if updated != text:
            changed.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(updated)
    if args.check and changed:
        print('Stale PEP 723 policy: ' + ', '.join(changed))
        return 1
    print(f"{'Checked' if args.check else 'Synchronized'} {count} dependency-bearing PEP 723 runners")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
