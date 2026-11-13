#!/usr/bin/env python3
"""Check every locked version against retained advisories and, optionally, OSV.

No dependency installation, advisory dismissal, model access or live-index access.
Unknown/malformed inputs and unavailable scans fail closed, not as zero findings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import tomllib

import json5
from packaging.specifiers import SpecifierSet
from packaging.version import Version as PipVersion
import requests
from semantic_version import NpmSpec, Version as NpmVersion
import yaml


class UniqueYamlLoader(yaml.SafeLoader):
    pass


def unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate mapping key: {key!r}')
        result[key] = value
    return result


def yaml_mapping(loader, node):
    loader.flatten_mapping(node)
    return unique_mapping((loader.construct_object(k), loader.construct_object(v)) for k, v in node.value)


UniqueYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, yaml_mapping)


def canonical(ecosystem: str, name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError('Package name must be nonempty text')
    if ecosystem == 'pip':
        return re.sub(r'[-_.]+', '-', name).lower()
    if ecosystem == 'npm':
        return name
    raise ValueError(f'Unsupported ecosystem: {ecosystem}')


def affected(ecosystem: str, version: str, constraint: str) -> bool:
    if not isinstance(constraint, str) or not constraint.strip():
        raise ValueError('Missing advisory range')
    if ecosystem == 'pip':
        return SpecifierSet(constraint).contains(PipVersion(version), prereleases=True)
    if ecosystem == 'npm':
        parsed = NpmVersion(version)
        if parsed.prerelease:
            raise ValueError('Prerelease npm versions require an explicit advisory review')
        # Translate only comparator formatting; retain the original range in receipts.
        # GitHub permits commas and spaces after operators; NpmSpec does not.
        translated = re.sub(r'([<>]=?|[=~^])\s+', r'\1', constraint.replace(',', ' '))
        return NpmSpec(' '.join(translated.split())).match(parsed)
    raise ValueError(f'Unsupported ecosystem: {ecosystem}')


def npm_locator(locator: str) -> tuple[str, str]:
    if not isinstance(locator, str):
        raise ValueError('Invalid npm locator')
    name, separator, version = locator.split('(', 1)[0].rpartition('@')
    if not separator or not name:
        raise ValueError(f'Unsupported npm locator: {locator}')
    NpmVersion(version)
    return name, version


def validate_alerts(alerts):
    if not isinstance(alerts, list) or not alerts:
        raise ValueError('Advisory baseline must be a nonempty list')
    seen = set()
    for alert in alerts:
        number = alert['number']
        if type(number) is not int or number <= 0 or number in seen:
            raise ValueError('Duplicate or invalid alert number')
        seen.add(number)
        package = alert['dependency']['package']
        canonical(package['ecosystem'], package['name'])
        constraint = alert['security_vulnerability']['vulnerable_version_range']
        # Validate even when an upgraded graph legitimately removed the package.
        affected(package['ecosystem'], '0.0.0', constraint)
    return alerts


def collect(root: Path, alerts) -> list[dict]:
    validate_alerts(alerts)
    records = []

    def add(ecosystem, name, version, source, locator):
        if not isinstance(version, str):
            raise ValueError('Locked version must be text')
        (PipVersion if ecosystem == 'pip' else NpmVersion)(version)
        records.append({'ecosystem': ecosystem, 'name': canonical(ecosystem, name),
                        'original_name': name, 'version': version, 'source': source, 'locator': locator})

    # This subclass retains SafeLoader tag restrictions; it only rejects duplicate maps.
    pnpm = yaml.load((root / 'pnpm-lock.yaml').read_text(), Loader=UniqueYamlLoader)
    if not isinstance(pnpm, dict) or str(pnpm.get('lockfileVersion')) not in {'9.0', '9'}:
        raise ValueError('Unsupported pnpm lock schema')
    if not isinstance(pnpm.get('packages'), dict) or not pnpm['packages']:
        raise ValueError('Missing pnpm package graph')
    for locator in pnpm['packages']:
        name, version = npm_locator(locator)
        add('npm', name, version, 'pnpm-lock.yaml', locator)

    bun = json5.loads((root / 'bun.lock').read_text(), allow_duplicate_keys=False)
    if not isinstance(bun, dict) or bun.get('lockfileVersion') != 1:
        raise ValueError('Unsupported Bun lock schema')
    if not isinstance(bun.get('packages'), dict) or not bun['packages']:
        raise ValueError('Missing Bun package graph')
    for locator, record in bun['packages'].items():
        if not isinstance(record, list) or not record:
            raise ValueError(f'Invalid Bun package entry: {locator}')
        name, version = npm_locator(record[0])
        add('npm', name, version, 'bun.lock', locator)

    project = json.loads((root / 'package.json').read_text(), object_pairs_hook=unique_mapping)
    affected_npm_names = {a['dependency']['package']['name'] for a in alerts
                          if a['dependency']['package']['ecosystem'] == 'npm'}
    for section in ('dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies'):
        for name, version in project.get(section, {}).items():
            try:
                NpmVersion(version)
            except (ValueError, TypeError):
                if name in affected_npm_names:
                    raise ValueError(f'Affected direct requirement must be an exact reviewed pin: {name}')
                continue  # Actual resolved versions are still scanned in both locks.
            add('npm', name, version, 'package.json', section + ':' + name)

    uv = tomllib.loads((root / 'finetune/uv.lock').read_text())
    if uv.get('version') != 1 or not isinstance(uv.get('package'), list) or not uv['package']:
        raise ValueError('Missing/unsupported uv package graph')
    # Deliberately do not evaluate platform/Python markers: scan every resolution.
    for index, record in enumerate(uv['package']):
        add('pip', record['name'], record['version'], 'finetune/uv.lock', str(index))

    for source in ('scripts/mlx_server_constraints.txt', 'scripts/security_audit_requirements.txt'):
        for line in (root / source).read_text().splitlines():
            line = line.partition('#')[0].strip()
            if not line or line.startswith('-c '):
                continue
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^;\s]+)', line)
            if not match:
                raise ValueError(f'Expected an exact audited requirement in {source}')
            add('pip', match[1], match[2], source, match[1])
    return records


def assess(alerts, records) -> dict:
    validate_alerts(alerts)
    if not records:
        raise ValueError('An empty dependency graph cannot certify remediation')
    rows = []
    for alert in sorted(alerts, key=lambda item: item['number']):
        package = alert['dependency']['package']
        vulnerability = alert['security_vulnerability']
        ecosystem = package['ecosystem']
        name = canonical(ecosystem, package['name'])
        instances = [r for r in records if r['ecosystem'] == ecosystem and canonical(ecosystem, r['name']) == name]
        bad = [r for r in instances if affected(ecosystem, r['version'], vulnerability['vulnerable_version_range'])]
        rows.append({'number': alert['number'], 'ghsa': alert['security_advisory']['ghsa_id'],
                     'package_original': package['name'], 'ecosystem': ecosystem,
                     'severity': vulnerability['severity'], 'vulnerable_range': vulnerability['vulnerable_version_range'],
                     'first_patched_version': (vulnerability.get('first_patched_version') or {}).get('identifier'),
                     'instances': instances, 'vulnerable_instances': bad,
                     'status': 'vulnerable' if bad else 'outside_affected_range' if instances else 'not_in_graph'})
    return {'passed': not any(r['vulnerable_instances'] for r in rows), 'alerts_checked': len(rows),
            'affected_alerts': sum(bool(r['vulnerable_instances']) for r in rows), 'alerts': rows}


def request_osv(queries):
    for attempt in range(3):
        try:
            response = requests.post('https://api.osv.dev/v1/querybatch', json={'queries': queries}, timeout=(10, 60))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('Unreachable')


def scan_osv(records):
    unique = sorted({('PyPI' if r['ecosystem'] == 'pip' else 'npm', r['name'],
                      PipVersion(r['version']).public if r['ecosystem'] == 'pip' else r['version']) for r in records})
    if not unique:
        raise ValueError('Cannot audit an empty graph')
    queries = [{'package': {'ecosystem': eco, 'name': name}, 'version': version} for eco, name, version in unique]
    findings = []
    for start in range(0, len(queries), 100):
        batch = queries[start:start + 100]
        payload = request_osv(batch)
        if not isinstance(payload, dict) or payload.get('error') or not isinstance(payload.get('results'), list) or len(payload['results']) != len(batch):
            raise ValueError('Incomplete/unavailable OSV batch response')
        for query, result in zip(batch, payload['results'], strict=True):
            if not isinstance(result, dict) or result.get('error') or result.get('next_page_token'):
                raise ValueError('Unavailable/paginated OSV result cannot certify a clean scan')
            vulns = result.get('vulns', [])
            if not isinstance(vulns, list) or any(not isinstance(v, dict) or not isinstance(v.get('id'), str) for v in vulns):
                raise ValueError('Malformed OSV vulnerabilities')
            if vulns:
                findings.append({'query': query, 'advisories': [v['id'] for v in vulns]})
    return {'passed': not findings, 'unique_queries': len(queries), 'affected': findings}


def write_report(path: Path, report):
    # Exclusive creation preserves prior receipts and refuses existing link aliases.
    with path.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--alerts', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--online', action='store_true', help='Also scan every unique resolved package/version through OSV')
    args = parser.parse_args(argv)
    report = {'passed': False, 'created_at': datetime.now(timezone.utc).isoformat()}
    try:
        source = args.alerts or args.repo / 'docs/security/dependency-remediation/alerts-before.json'
        alerts = json.loads(source.read_text(), object_pairs_hook=unique_mapping)
        records = collect(args.repo, alerts)
        report.update(assess(alerts, records))
        report['graph_instances'] = len(records)
        if args.online:
            report['osv'] = scan_osv(records)
            report['passed'] = report['passed'] and report['osv']['passed']
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, requests.RequestException) as error:
        report.update(passed=False, error=f'{type(error).__name__}: {error}')
    if args.output:
        write_report(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'alerts', 'osv'}}, indent=2))
    if 'osv' in report:
        print(json.dumps({'osv': report['osv']}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
