"""Regression contracts for the complete dependency/advisory graph checker."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('dependency_security', ROOT / 'scripts/check_dependency_security.py')
assert spec is not None and spec.loader is not None
security = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = security
spec.loader.exec_module(security)


def alert(number: object = 1, package='example', ecosystem='npm', vulnerable='< 2.0.0'):
    return {'number': number, 'dependency': {'package': {'name': package, 'ecosystem': ecosystem},
            'manifest_path': 'pnpm-lock.yaml' if ecosystem == 'npm' else 'finetune/uv.lock'},
            'security_vulnerability': {'severity': 'critical', 'vulnerable_version_range': vulnerable,
                                       'first_patched_version': None},
            'security_advisory': {'ghsa_id': 'GHSA-fixture-only'}}


class DependencySecurityTests(unittest.TestCase):
    def test_npm_boundaries_and_comma_ranges(self):
        self.assertTrue(security.affected('npm', '1.9.9', '>= 1.0.0, < 2.0.0'))
        self.assertFalse(security.affected('npm', '2.0.0', '>= 1.0.0, < 2.0.0'))
        self.assertTrue(security.affected('npm', '3.0.0', '< 2.0.0 || >= 3.0.0, < 3.1.0'))

    def test_pip_aliases_and_local_versions(self):
        self.assertEqual(security.canonical('pip', 'Some_Package.Name'), 'some-package-name')
        self.assertEqual(security.canonical('pip', 'Pillow'), 'pillow')
        self.assertTrue(security.affected('pip', '2.8.0+cu128', '< 2.13.0'))
        self.assertFalse(security.affected('pip', '2.15.0', '< 2.13.0'))

    def test_missing_first_patch_does_not_skip_advisory(self):
        result = security.assess([alert()], [{'ecosystem': 'npm', 'name': 'example', 'version': '1.0.0', 'source': 'fixture'}])
        self.assertFalse(result['passed'])
        self.assertEqual(result['alerts'][0]['number'], 1)
        self.assertIsNone(result['alerts'][0]['first_patched_version'])

    def test_every_version_and_both_lock_graphs_are_checked(self):
        rows = [{'ecosystem': 'npm', 'name': 'example', 'version': version, 'source': source}
                for source, version in [('pnpm-lock.yaml', '2.0.0'), ('bun.lock', '1.0.0')]]
        result = security.assess([alert()], rows)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['alerts'][0]['instances']), 2)
        self.assertEqual(result['alerts'][0]['vulnerable_instances'][0]['source'], 'bun.lock')

    def test_removal_is_distinct_from_version_upgrade(self):
        removed = security.assess([alert()], [{'ecosystem': 'npm', 'name': 'other', 'version': '1.0.0', 'source': 'fixture'}])
        upgraded = security.assess([alert()], [{'ecosystem': 'npm', 'name': 'example', 'version': '2.0.0', 'source': 'fixture'}])
        self.assertEqual(removed['alerts'][0]['status'], 'not_in_graph')
        self.assertEqual(upgraded['alerts'][0]['status'], 'outside_affected_range')

    def test_duplicate_or_invalid_alert_ids_fail_closed(self):
        for rows in ([alert(), alert()], [alert(True)], [alert('001')], []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                security.assess(rows, [])

    def test_empty_graph_and_unsafe_yaml_are_not_accepted(self):
        with self.assertRaises(ValueError):
            security.assess([alert()], [])
        with self.assertRaises(security.yaml.YAMLError):
            security.yaml.load('x: !!python/object/apply:builtins.str [fixture]', Loader=security.UniqueYamlLoader)
        with self.assertRaises(ValueError):
            security.yaml.load('packages: {}\npackages: {}\n', Loader=security.UniqueYamlLoader)

    def test_unknown_ecosystem_or_invalid_range_fails_closed(self):
        for eco, version, constraint in [('maven', '1.0.0', '<2'), ('npm', 'not-a-version', '<2'),
                                         ('pip', '1.0', 'broken'), ('npm', '1.0.0', '')]:
            with self.subTest(eco=eco, version=version), self.assertRaises(ValueError):
                security.affected(eco, version, constraint)

    def test_unknown_prerelease_is_not_certified_by_stable_ranges(self):
        with self.assertRaises(ValueError):
            security.affected('npm', '2.0.0-beta.1', '< 2.0.0')

    def test_graph_collection_keeps_all_uv_platform_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'finetune').mkdir()
            (root / 'scripts').mkdir()
            (root / 'package.json').write_text(json.dumps({'dependencies': {'example': '2.0.0'}}))
            (root / 'pnpm-lock.yaml').write_text("lockfileVersion: '9.0'\npackages:\n  example@2.0.0: {}\n  '@scope/pkg@1.0.0': {}\n")
            (root / 'bun.lock').write_text('{"lockfileVersion": 1, // comment\n"packages": {"example": ["example@2.0.0", "", {}, "hash"],},}')
            (root / 'finetune/uv.lock').write_text('version = 1\n[[package]]\nname = "Pillow"\nversion = "12.3.0"\n[[package]]\nname = "pillow"\nversion = "13.0.0"\n')
            (root / 'scripts/mlx_server_constraints.txt').write_text('requests==2.34.2\n')
            (root / 'scripts/security_audit_requirements.txt').write_text('-c mlx_server_constraints.txt\njson5==0.17.3\n')
            rows = security.collect(root, [alert(package='Pillow', ecosystem='pip', vulnerable='<13.0.0')])
            matches = [r for r in rows if r['name'] == 'pillow']
            self.assertEqual({r['version'] for r in matches}, {'12.3.0', '13.0.0'})
            self.assertFalse(security.assess([alert(package='Pillow', ecosystem='pip', vulnerable='<13.0.0')], rows)['passed'])
            (root / 'bun.lock').write_text('{"lockfileVersion":1,"packages":{}}')
            with self.assertRaises(ValueError):
                security.collect(root, [alert()])

    def test_output_is_exclusive_preserving_prior_receipts_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'receipt.json'
            security.write_report(target, {'passed': True})
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                security.write_report(target, {'passed': False})
            self.assertEqual(target.read_bytes(), original)
            link = Path(directory) / 'link.json'
            link.symlink_to(target)
            with self.assertRaises(FileExistsError):
                security.write_report(link, {'passed': False})
            self.assertEqual(target.read_bytes(), original)

    def test_osv_error_or_incomplete_response_is_not_clean(self):
        package = [{'ecosystem': 'npm', 'name': 'example', 'version': '2.0.0', 'source': 'fixture'}]
        for payload in ({'error': 'unavailable'}, {'results': []}, {'results': [{'error': 'bad query'}]},
                        {'results': [{'next_page_token': 'more'}]}, {'results': [None]}):
            with self.subTest(payload=payload), patch.object(security, 'request_osv', return_value=payload), self.assertRaises(ValueError):
                security.scan_osv(package)

    def test_osv_queries_every_unique_instance_and_public_pip_version(self):
        rows = [{'ecosystem': 'pip', 'name': 'torch', 'version': '2.15.0+cpu', 'source': 'a'},
                {'ecosystem': 'npm', 'name': 'example', 'version': '2.0.0', 'source': 'b'},
                {'ecosystem': 'npm', 'name': 'example', 'version': '1.0.0', 'source': 'c'}]
        captured = []
        def response(queries):
            captured.extend(queries)
            return {'results': [{'vulns': [{'id': 'GHSA-fixture-only'}]} if q['version'] == '1.0.0' else {} for q in queries]}
        with patch.object(security, 'request_osv', side_effect=response):
            result = security.scan_osv(rows)
        self.assertEqual(len(captured), 3)
        self.assertIn('2.15.0', [q['version'] for q in captured])
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['affected']), 1)


if __name__ == '__main__':
    unittest.main()
