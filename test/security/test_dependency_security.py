"""Regression contracts for the complete dependency/advisory graph checker."""
import importlib.util
import json
import os
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

    def test_source_review_rejects_advisory_bound_blind_spots(self):
        transformer = alert(package='transformers', ecosystem='pip', vulnerable='>=4.49.0, <=5.8.1')
        transformer['security_advisory']['ghsa_id'] = 'GHSA-x9r9-c232-4q39'
        row = {'ecosystem': 'pip', 'name': 'transformers', 'version': '5.16.0', 'source': 'fixture'}
        self.assertFalse(security.assess([transformer], [row])['passed'])
        row['version'] = '5.17.0'
        self.assertTrue(security.assess([transformer], [row])['passed'])
        accelerate = alert(package='accelerate', ecosystem='pip', vulnerable='<=1.14.0')
        accelerate['security_advisory']['ghsa_id'] = 'GHSA-4j2p-28q2-5m79'
        for version in ('1.14.0', '1.15.0', '1.15.0+qmd.1'):
            with self.subTest(version=version):
                row = {'ecosystem': 'pip', 'name': 'accelerate', 'version': version,
                       'source': 'finetune/uv.lock', 'locator': '0'}
                result = security.assess([accelerate], [row])
                self.assertFalse(result['passed'], 'A version label alone is not a source mitigation')
                self.assertEqual(result['alerts'][0]['status'], 'unverified_source_mitigation')
        proof = {('1.15.0+qmd.1', 'finetune/uv.lock', '0')}
        self.assertTrue(security.assess([accelerate], [row], verified_patches=proof)['passed'])
        self.assertFalse(security.assess([accelerate], [row, {**row, 'locator': '1'}], verified_patches=proof)['passed'])

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

    def test_configured_script_locks_are_collected_without_platform_filtering_or_fifo_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'finetune/dataset').mkdir(parents=True)
            (root / 'scripts').mkdir()
            (root / 'docs/security/dependency-remediation').mkdir(parents=True)
            (root / 'package.json').write_text(json.dumps({'dependencies': {'example': '2.0.0'}}))
            (root / 'pnpm-lock.yaml').write_text("lockfileVersion: '9.0'\npackages:\n  example@2.0.0: {}\n")
            (root / 'bun.lock').write_text('{"lockfileVersion":1,"packages":{"example":["example@2.0.0"]}}')
            (root / 'finetune/uv.lock').write_text('version = 1\n[[package]]\nname = "accelerate"\nversion = "0.0.0+qmd.synthetic1"\n')
            (root / 'scripts/mlx_server_constraints.txt').write_text('requests==2.34.2\n')
            (root / 'scripts/security_audit_requirements.txt').write_text('json5==0.17.3\n')
            runners = ['finetune/train.py', 'finetune/dataset/data.py']
            wheel = 'finetune/vendor/accelerate/accelerate-0.0.0+qmd.synthetic1-py3-none-any.whl'
            synthetic = {'advisory': 'GHSA-4j2p-28q2-5m79', 'version': '0.0.0+qmd.synthetic1',
                         'wheel_path': wheel, 'commit': 'd' * 40, 'sha256': 'a' * 64,
                         'immutable_url': 'https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/'
                         + 'd' * 40 + '/' + wheel.replace('+', '%2B') + '#sha256=' + 'a' * 64,
                         'runner_paths': runners}
            policy_path = root / security._binding.POLICY_PATH
            policy_path.write_text(json.dumps({'schema_version': 1, 'accelerate': synthetic}))
            training = root / (runners[0] + '.lock')
            training.write_text('version = 1\n[[package]]\nname = "accelerate"\nversion = "0.0.0+qmd.synthetic1"\n'
                                '[[package]]\nname = "Pillow"\nversion = "12.3.0"\n'
                                '[[package]]\nname = "pillow"\nversion = "13.0.0"\n')
            data = root / (runners[1] + '.lock')
            data.write_text('version = 1\n[[package]]\nname = "pydantic"\nversion = "2.0.0"\n')
            baseline = [alert(package='Pillow', ecosystem='pip', vulnerable='<13.0.0')]
            rows = security.collect(root, baseline)
            self.assertEqual({row['source'] for row in rows if row['ecosystem'] == 'pip' and row['source'].endswith('.lock')},
                             {'finetune/uv.lock', *(runner + '.lock' for runner in runners)})
            self.assertEqual({row['version'] for row in rows if row['name'] == 'pillow'}, {'12.3.0', '13.0.0'})
            self.assertEqual(len([row for row in rows if row['name'] == 'accelerate']), 2)
            self.assertFalse(security.assess(baseline, rows)['passed'])
            data.unlink()
            with self.assertRaises(ValueError):
                security.collect(root, baseline)
            os.mkfifo(data)
            original_open = os.open
            def no_fifo_open(filename, *args, **kwargs):
                self.assertNotEqual(filename, data.name, 'Reject known FIFO before open')
                return original_open(filename, *args, **kwargs)
            with patch.object(security._binding.os, 'open', side_effect=no_fifo_open), self.assertRaises(ValueError):
                security.collect(root, baseline)
            # Removing the optional policy restores the pre-policy snapshot
            # contract: no standalone files are read or demanded.
            policy_path.unlink()
            baseline_rows = security.collect(root, baseline)
            self.assertEqual({row['source'] for row in baseline_rows if row['source'].endswith('.lock')}, {'finetune/uv.lock', 'bun.lock'})

    def test_osv_error_or_incomplete_response_is_not_clean(self):
        package = [{'ecosystem': 'npm', 'name': 'example', 'version': '2.0.0', 'source': 'fixture'}]
        for payload in ({'error': 'unavailable'}, {'results': []}, {'results': [{'error': 'bad query'}]},
                        {'results': [{'next_page_token': 'more'}]}, {'results': [None]},
                        {'error': {}, 'results': [{}]}, {'error': None, 'results': [{}]},
                        {'results': [{'error': {}}]}, {'results': [{'error': None}]},
                        {'results': [{'unexpected': 'not-an-osv-query-result'}]},
                        {'results': [{'vulns': None}]}, {'results': [{'next_page_token': False}]},
                        {'results': [{}], 'unexpected': 'not-an-osv-batch'}):
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


class AuditProviderTests(unittest.TestCase):
    rows = [{'ecosystem': 'npm', 'name': 'example', 'version': '2.0.0', 'source': 'fixture'}]

    def test_npm_schema_failures_are_not_clean(self):
        for payload in (None, [], {'error': {}}, {'error': None}, {'unexpected': []},
                        {'example': {}}, {'example': [None]}, {'example': [{'id': 1}]}):
            with self.subTest(payload=payload), patch.object(security, 'request_npm', return_value=payload), self.assertRaises(ValueError):
                security.scan_npm(self.rows)

    def test_npm_deduplicates_both_locks_but_keeps_every_version(self):
        rows = self.rows + [{'ecosystem': 'npm', 'name': 'example', 'version': '1.0.0', 'source': source}
                            for source in ('pnpm-lock.yaml', 'bun.lock')]
        advisory = {'name': 'example', 'id': 1, 'title': 'Synthetic regression fixture',
                    'url': 'https://example.invalid/advisory', 'severity': 'low',
                    'vulnerable_versions': '< 2.0.0'}
        with patch.object(security, 'request_npm', return_value={'example': [advisory]}) as request:
            result = security.scan_npm(rows)
        request.assert_called_once_with({'example': ['1.0.0', '2.0.0']})
        self.assertFalse(result['passed'])
        self.assertEqual(result['affected'][0]['versions'], ['1.0.0'])
        self.assertEqual(result['unique_versions'], 2)
        with patch.object(security, 'request_npm', return_value={}):
            self.assertTrue(security.scan_npm(self.rows)['passed'])
        with self.assertRaises(ValueError):
            security.scan_npm([])

    def test_npm_accepts_map_bound_names_but_not_mismatched_names(self):
        advisory = {'id': 1, 'title': 'Synthetic fixture matching the live API shape',
                    'url': 'https://example.invalid/advisory', 'severity': 'high',
                    'vulnerable_versions': '<3.0.0'}
        with patch.object(security, 'request_npm', return_value={'example': [advisory]}):
            self.assertFalse(security.scan_npm(self.rows)['passed'])
        advisory['name'] = 'not-the-requested-package'
        with patch.object(security, 'request_npm', return_value={'example': [advisory]}), self.assertRaises(ValueError):
            security.scan_npm(self.rows)

    def test_real_http_empty_malformed_error_and_duplicate_bodies_fail_closed(self):
        # Real HTTP responses, not a mocked CLI success code or invented service result.
        import http.server
        import socketserver
        import threading

        class NumericServer(http.server.HTTPServer):
            def server_bind(self):
                socketserver.TCPServer.server_bind(self)
                self.server_name = str(self.server_address[0])
                self.server_port = self.server_address[1]

        cases = [(200, b'', False), (200, b'not-json', False),
                 (200, b'{"error":{}}', False), (200, b'{"error":"unavailable"}', False),
                 (200, b'{"example":[],"example":[]}', False),
                 (200, b'{"example":NaN}', False), (503, b'{}', False), (200, b'{}', True)]
        for status, body, valid in cases:
            captured = []
            class Handler(http.server.BaseHTTPRequestHandler):
                def do_POST(self):
                    self.connection.settimeout(2)
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    captured.append((self.path, request))
                    self.send_response(status)
                    # The live npm bulk endpoint also omits Content-Type.
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def log_message(self, format, *args):
                    pass
            with self.subTest(status=status, body=body), NumericServer(('127.0.0.1', 0), Handler) as server:
                thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
                thread.start()
                try:
                    url = f'http://127.0.0.1:{server.server_port}/-/npm/v1/security/advisories/bulk'
                    with patch.object(security, 'NPM_AUDIT_URL', url), patch.object(security.time, 'sleep'):
                        if valid:
                            self.assertTrue(security.scan_npm(self.rows)['passed'])
                        else:
                            with self.assertRaises((ValueError, security.requests.RequestException)):
                                security.scan_npm(self.rows)
                    self.assertGreaterEqual(len(captured), 1)
                    self.assertLessEqual(len(captured), 3)
                    self.assertTrue(all(path == '/-/npm/v1/security/advisories/bulk' and
                                        payload == {'example': ['2.0.0']} for path, payload in captured))
                finally:
                    server.shutdown()
                    thread.join(timeout=3)
                    self.assertFalse(thread.is_alive())

    def test_cli_unavailable_providers_write_failed_receipts(self):
        import contextlib
        import io
        for flag, method, response in (
            ('--online', 'request_osv', {'error': {}, 'results': [{}]}),
            ('--online', 'request_osv', {'results': [{'unexpected': 'wrong schema'}]}),
            ('--npm-online', 'request_npm', {'error': {}}),
        ):
            with self.subTest(flag=flag, response=response), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                alerts = root / 'synthetic-alerts.json'
                alerts.write_text(json.dumps([alert()]))
                output = root / 'receipt.json'
                with patch.object(security, 'collect', return_value=self.rows), patch.object(security, method, return_value=response), contextlib.redirect_stdout(io.StringIO()):
                    code = security.main(['--repo', str(root), '--alerts', str(alerts), flag, '--output', str(output)])
                self.assertEqual(code, 1)
                self.assertFalse(json.loads(output.read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
