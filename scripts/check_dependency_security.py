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


def assess(alerts, records, *, verified_patches=frozenset()) -> dict:
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
        ghsa = alert['security_advisory']['ghsa_id']
        effective_range = vulnerability['vulnerable_version_range']
        source_review = None
        # Source/wheel inspection found gaps in two advisory bounds. A clean
        # registry range response must not override these retained findings.
        if ghsa == 'GHSA-x9r9-c232-4q39':
            if (ecosystem, name) != ('pip', 'transformers'):
                raise ValueError('Source-reviewed advisory/package mismatch')
            effective_range = '>=4.49.0, <5.17.0'
            source_review = 'Transformers 5.17.0 is the verified safe floor; the recorded 5.8.1 upper bound is insufficient.'
        bad = [r for r in instances if affected(ecosystem, r['version'], effective_range)]
        status = 'vulnerable' if bad else 'outside_affected_range' if instances else 'not_in_graph'
        if ghsa == 'GHSA-4j2p-28q2-5m79':
            if (ecosystem, name) != ('pip', 'accelerate'):
                raise ValueError('Source-reviewed advisory/package mismatch')
            source_review = 'Official Accelerate 1.15.0 remains vulnerable; a separately verified source-bound patch is mandatory.'
            bad = [r for r in instances if (r['version'], r['source'], r.get('locator')) not in verified_patches]
            status = ('unverified_source_mitigation' if bad else
                      'verified_patched_artifact' if instances else 'not_in_graph')
        rows.append({'number': alert['number'], 'ghsa': ghsa,
                     'package_original': package['name'], 'ecosystem': ecosystem,
                     'severity': vulnerability['severity'], 'vulnerable_range': vulnerability['vulnerable_version_range'],
                     'effective_range': effective_range, 'source_review': source_review,
                     'first_patched_version': (vulnerability.get('first_patched_version') or {}).get('identifier'),
                     'instances': instances, 'vulnerable_instances': bad, 'status': status})
    return {'passed': not any(r['vulnerable_instances'] for r in rows), 'alerts_checked': len(rows),
            'affected_alerts': sum(bool(r['vulnerable_instances']) for r in rows), 'alerts': rows}


OSV_URL = 'https://api.osv.dev/v1/querybatch'
NPM_AUDIT_URL = 'https://registry.npmjs.org/-/npm/v1/security/advisories/bulk'
MAX_AUDIT_RESPONSE_BYTES = 8 * 1024 * 1024


def invalid_json_constant(value):
    raise ValueError(f'Non-JSON numeric constant: {value}')


def request_audit_json(url, payload):
    """Bound and strictly decode the actual response; never trust CLI exit 0."""
    for attempt in range(3):
        try:
            with requests.post(url, json=payload, timeout=(10, 60), stream=True,
                               allow_redirects=False) as response:
                if response.status_code != 200:
                    raise requests.HTTPError(f'Audit HTTP status {response.status_code}')
                media_type = response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                # npm's bulk endpoint can omit Content-Type; the body/schema
                # remain mandatory. An explicitly non-JSON media type is rejected.
                if media_type and media_type != 'application/json' and not (
                    media_type.startswith('application/') and media_type.endswith('+json')
                ):
                    raise ValueError('Audit response is not declared JSON')
                body = bytearray()
                for chunk in response.iter_content(65536):
                    if len(body) + len(chunk) > MAX_AUDIT_RESPONSE_BYTES:
                        raise ValueError('Audit response exceeds the bounded response budget')
                    body.extend(chunk)
                # Empty bodies, duplicate keys and NaN/Infinity must not become {}.
                return json.loads(body, object_pairs_hook=unique_mapping,
                                  parse_constant=invalid_json_constant)
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('Unreachable')


def request_osv(queries):
    return request_audit_json(OSV_URL, {'queries': queries})


def request_npm(packages):
    return request_audit_json(NPM_AUDIT_URL, packages)


def scan_npm(records):
    """Scan every locked npm version against the documented bulk advisory API."""
    packages = {}
    for record in records:
        if record['ecosystem'] == 'npm':
            NpmVersion(record['version'])
            packages.setdefault(canonical('npm', record['name']), set()).add(record['version'])
    if not packages:
        raise ValueError('Cannot audit an empty npm graph')
    query = {name: sorted(versions) for name, versions in sorted(packages.items())}
    payload = request_npm(query)
    if not isinstance(payload, dict) or 'error' in payload or set(payload) - set(query):
        raise ValueError('Unavailable/invalid npm bulk advisory response')
    findings = []
    # The live bulk API binds names through the map key and may omit the
    # redundant advisory.name, despite the CLI documentation listing it.
    required = {'url', 'id', 'severity', 'vulnerable_versions', 'title'}
    for name, advisories in payload.items():
        if not isinstance(advisories, list):
            raise ValueError('Malformed npm advisory list')
        for advisory in advisories:
            if (not isinstance(advisory, dict) or not required <= set(advisory)
                    or 'error' in advisory or ('name' in advisory and advisory['name'] != name)
                    or type(advisory['id']) is not int or advisory['id'] <= 0
                    or any(not isinstance(advisory[key], str) or not advisory[key]
                           for key in ('url', 'title', 'severity', 'vulnerable_versions'))
                    or advisory['severity'] not in {'info', 'low', 'moderate', 'medium', 'high', 'critical'}):
                raise ValueError('Malformed npm bulk advisory')
            versions = [version for version in query[name]
                        if affected('npm', version, advisory['vulnerable_versions'])]
            if versions:
                findings.append({'package': name, 'versions': versions,
                                 'id': advisory['id'], 'severity': advisory['severity'],
                                 'range': advisory['vulnerable_versions'], 'url': advisory['url']})
    return {'passed': not findings, 'unique_packages': len(query),
            'unique_versions': sum(map(len, query.values())), 'affected': findings}


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
        if (not isinstance(payload, dict) or set(payload) != {'results'}
                or not isinstance(payload['results'], list) or len(payload['results']) != len(batch)):
            raise ValueError('Incomplete/unavailable OSV batch response')
        for query, result in zip(batch, payload['results'], strict=True):
            # {} is OSV's documented clean result. Arbitrary objects/errors are not.
            if not isinstance(result, dict) or set(result) - {'vulns', 'next_page_token'}:
                raise ValueError('Invalid OSV result schema')
            if 'next_page_token' in result and (
                not isinstance(result['next_page_token'], str) or result['next_page_token']
            ):
                raise ValueError('Unavailable/paginated OSV result cannot certify a clean scan')
            vulns = result.get('vulns', [])
            if (not isinstance(vulns, list) or any(
                not isinstance(v, dict) or set(v) - {'id', 'modified'}
                or not isinstance(v.get('id'), str) or not v['id'].strip()
                or ('modified' in v and (not isinstance(v['modified'], str) or not v['modified']))
                for v in vulns
            )):
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
    parser.add_argument('--npm-online', action='store_true', help='Also validate a fresh npm bulk advisory response for all locked npm versions')
    args = parser.parse_args(argv)
    report = {'passed': False, 'created_at': datetime.now(timezone.utc).isoformat()}
    try:
        source = args.alerts or args.repo / 'docs/security/dependency-remediation/alerts-current.json'
        alerts = json.loads(source.read_text(), object_pairs_hook=unique_mapping)
        records = collect(args.repo, alerts)
        report.update(assess(alerts, records))
        report['graph_instances'] = len(records)
        if args.online:
            report['osv'] = scan_osv(records)
            report['passed'] = report['passed'] and report['osv']['passed']
        if args.npm_online:
            report['npm'] = scan_npm(records)
            report['passed'] = report['passed'] and report['npm']['passed']
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
