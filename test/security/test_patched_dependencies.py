"""Source-binding regression tests; every wheel/commit below is SYNTHETIC.

These minimal ZIPs are not the reviewed Accelerate artifact and do not exercise
or approve its patch. The isolated real-uv probe below installs only a tiny
synthetic wheel over loopback; product runners and models are never executed.
"""
from __future__ import annotations

from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tomllib
from urllib.parse import quote
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'patched_dependencies_under_test', ROOT / 'scripts/verify_patched_dependencies.py'
)
assert SPEC is not None and SPEC.loader is not None
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)

# Deliberately not the candidate's version, commit, hash or wheel bytes.
VERSION = '0.0.0+qmd.synthetic1'
COMMIT = 'd' * 40
POLICY = 'docs/security/dependency-remediation/patched-dependencies.json'
WHEEL = f'finetune/vendor/accelerate/accelerate-{VERSION}-py3-none-any.whl'
LOCAL_WHEEL = WHEEL.removeprefix('finetune/')
RUNNERS = [
    'finetune/train.py',
    'finetune/dataset/prepare.py',
    'finetune/experiments/demo/train.py',
    'finetune/jobs/eval.py',
]
SCRIPT_PROOFS = {(VERSION, runner + '.lock', '1') for runner in RUNNERS
                 if runner != 'finetune/dataset/prepare.py'}
BASE_PROOFS = SCRIPT_PROOFS | {(VERSION, 'finetune/uv.lock', '1')}


def synthetic_wheel(*, name='accelerate', version=VERSION, extra_headers='',
                    metadata_path=None, extra_entries=()) -> bytes:
    """Return an intentionally minimal, non-installable synthetic ZIP fixture."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            metadata_path or f'accelerate-{VERSION}.dist-info/METADATA',
            f'Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n'
            f'Summary: SYNTHETIC SOURCE-BINDING TEST ONLY\n{extra_headers}\n',
        )
        archive.writestr(
            'accelerate/__init__.py',
            "raise RuntimeError('SYNTHETIC WHEEL MUST NEVER BE EXECUTED')\n",
        )
        for entry_name, value in extra_entries:
            archive.writestr(entry_name, value)
    return output.getvalue()


class SyntheticGraph:
    def __init__(self, root):
        self.root = root
        self.policy = {'schema_version': 1, 'accelerate': {
            'advisory': 'GHSA-4j2p-28q2-5m79', 'version': VERSION,
            'wheel_path': WHEEL, 'sha256': '', 'commit': COMMIT,
            'immutable_url': '', 'runner_paths': RUNNERS.copy(),
        }}
        self.wheel = root / WHEEL
        self.wheel.parent.mkdir(parents=True)
        self.wheel.write_bytes(synthetic_wheel())
        self.refresh_hash()
        self.write_project()
        self.write_lock()
        for runner in RUNNERS:
            self.write_runner(runner, direct=runner != 'finetune/dataset/prepare.py')
            self.write_script_lock(runner, direct=runner != 'finetune/dataset/prepare.py')
        self.write('finetune/reward.py', '# /// script\n# dependencies = []\n# ///\n')
        self.write_policy()

    @property
    def patch(self):
        return self.policy['accelerate']

    @property
    def url(self):
        return self.patch['immutable_url']

    @property
    def installer_url(self):
        return self.url.partition('#')[0]

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def refresh_hash(self):
        self.patch['sha256'] = sha256(self.wheel.read_bytes()).hexdigest()
        self.patch['immutable_url'] = (
            f'https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/{COMMIT}/'
            f'{quote(WHEEL, safe="/")}#sha256={self.patch["sha256"]}'
        )

    def write_policy(self):
        self.write(POLICY, json.dumps(self.policy))

    def rebind_changed_wheel(self):
        """Update every synthetic binding so only wheel content is under test."""
        self.refresh_hash()
        self.write_policy()
        self.write_lock()
        for runner in RUNNERS:
            self.write_runner(runner, direct=runner != 'finetune/dataset/prepare.py')
            self.write_script_lock(runner, direct=runner != 'finetune/dataset/prepare.py')

    def write_project(self, *, dependency=None, constraint=None, source=None):
        self.write('finetune/pyproject.toml',
                   '[project]\nname = "synthetic-project"\nversion = "0.0.0"\n'
                   f'dependencies = [{json.dumps(dependency or f"accelerate=={VERSION}")}]\n'
                   '[tool.uv]\n'
                   f'constraint-dependencies = [{json.dumps(constraint or f"accelerate=={VERSION}")}]\n'
                   '[tool.uv.sources]\n'
                   f'accelerate = {source or ("{ path = " + json.dumps(LOCAL_WHEEL) + " }")}\n')

    def lock_entry(self, *, version=VERSION, source=None, digest=None):
        return ('\n[[package]]\nname = "accelerate"\n'
                f'version = {json.dumps(version)}\n'
                f'source = {source or ("{ path = " + json.dumps(LOCAL_WHEEL) + " }")}\n'
                'wheels = [{ filename = '
                f'{json.dumps(self.wheel.name)}, hash = "sha256:{digest or self.patch["sha256"]}"' + ' }]\n')

    def write_lock(self, extra=''):
        self.write('finetune/uv.lock',
                   'version = 1\n[[package]]\nname = "pydantic"\nversion = "2.0.0"\n'
                   + self.lock_entry() + extra)

    def write_runner(self, path, *, direct=True, dependency=None, constraint=None, source=None):
        requirements = ([dependency or f'accelerate=={VERSION}'] if direct else ['pydantic>=2'])
        body = ('requires-python = ">=3.11"\n'
                f'dependencies = {json.dumps(requirements)}\n'
                '[tool.uv]\n'
                f'constraint-dependencies = [{json.dumps(constraint or f"accelerate=={VERSION}")}]\n'
                '[tool.uv.sources]\n'
                f'accelerate = {source or ("{ url = " + json.dumps(self.installer_url) + " }")}\n')
        self.write(path, '# /// script\n' + ''.join('# ' + line + '\n' for line in body.splitlines())
                   + '# ///\nraise RuntimeError("SYNTHETIC RUNNER MUST NOT EXECUTE")\n')

    def write_script_lock(self, runner, *, direct=True, transitive=False):
        requirement = ('{ name = "accelerate", url = ' + json.dumps(self.installer_url) + ' }'
                       if direct else '{ name = "pydantic", specifier = ">=2" }')
        text = ('version = 1\nrevision = 3\nrequires-python = ">=3.11"\n[manifest]\n'
                f'requirements = [{requirement}]\n'
                'constraints = [{ name = "accelerate", url = '
                + json.dumps(self.installer_url) + ' }]\n'
                '[[package]]\nname = "pydantic"\nversion = "2.0.0"\n')
        if transitive:
            text += 'dependencies = [{ name = "accelerate" }]\n'
        if direct or transitive:
            text += ('[[package]]\nname = "accelerate"\n'
                     f'version = {json.dumps(VERSION)}\n'
                     'source = { url = ' + json.dumps(self.installer_url) + ' }\n'
                     'wheels = [{ url = ' + json.dumps(self.installer_url)
                     + ', hash = "sha256:' + self.patch['sha256'] + '" }]\n')
        self.write(runner + '.lock', text)

    def records(self):
        records = []
        for source in [*(runner + '.lock' for runner in RUNNERS), 'finetune/uv.lock']:
            packages = tomllib.loads((self.root / source).read_text())['package']
            records.extend({'ecosystem': 'pip', 'name': package['name'], 'original_name': package['name'],
                            'version': package['version'], 'source': source, 'locator': str(index)}
                           for index, package in enumerate(packages))
        return records

    def verify(self, records=None):
        return binding.verify_accelerate_binding(self.root, self.records() if records is None else records)


@pytest.fixture
def graph(tmp_path):
    return SyntheticGraph(tmp_path / 'synthetic-repository')


def test_missing_policy_is_not_success_even_without_a_project(tmp_path):
    proofs, receipt = binding.verify_accelerate_binding(tmp_path, [])
    assert proofs == set()
    assert receipt['status'] == 'not_configured'
    assert receipt.get('passed') is not True


def test_valid_source_graph_grants_only_exact_uv_record_keys(graph):
    proofs, receipt = graph.verify()
    assert proofs == BASE_PROOFS
    assert receipt['status'] == 'verified_source_binding'
    assert receipt['runner_paths'] == sorted(RUNNERS)
    assert receipt['resolved_instances'] == 4
    assert receipt['script_lock_paths'] == sorted(runner + '.lock' for runner in RUNNERS)
    assert receipt['scope'] == 'source_binding_only'
    assert 'http' not in json.dumps(receipt)
    data_only = (graph.root / 'finetune/dataset/prepare.py').read_text()
    metadata = binding._script_metadata(data_only)
    assert metadata['dependencies'] == ['pydantic>=2']
    assert not any('torch' in item for item in metadata['dependencies'])


def test_all_platform_instances_must_match_and_get_distinct_proofs(graph):
    graph.write_lock(graph.lock_entry())
    proofs, receipt = graph.verify()
    assert proofs == BASE_PROOFS | {(VERSION, 'finetune/uv.lock', '2')}
    assert receipt['resolved_instances'] == 5


@pytest.mark.parametrize('field,value', [
    ('advisory', 'GHSA-wrong'), ('advisory', None),
    ('version', '1.15.0'), ('version', 'v1.15.0+qmd.1'),
    ('version', '01.15.0+qmd.1'), ('version', '1.15.0+QMD.1'),
    ('version', '1.15.0rc1+qmd.1'), ('version', 115),
    ('sha256', 'a' * 63), ('sha256', 'A' * 64), ('sha256', 3),
    ('commit', 'main'), ('commit', 'a' * 39), ('commit', 'A' * 40),
    ('wheel_path', '../outside.whl'), ('wheel_path', '/' + WHEEL),
    ('wheel_path', './' + WHEEL), ('wheel_path', WHEEL.replace('/vendor/', '//vendor/')),
    ('wheel_path', WHEEL.replace('/vendor/', '/other/')),
    ('runner_paths', []), ('runner_paths', [RUNNERS[0], RUNNERS[0]]),
    ('runner_paths', [42]), ('runner_paths', ['../outside.py']),
    ('runner_paths', ['./' + RUNNERS[0]]), ('runner_paths', ['/absolute.py']),
    ('runner_paths', ['finetune/.hidden/run.py']),
])
def test_invalid_policy_fields_fail_closed(graph, field, value):
    graph.patch[field] = value
    graph.write_policy()
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('value', [None, [], True, 1, {'schema_version': True, 'accelerate': {}},
                                 {'schema_version': 2, 'accelerate': {}},
                                 {'schema_version': 1, 'accelerate': []}])
def test_policy_schema_must_be_exact(graph, value):
    graph.write(POLICY, json.dumps(value))
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('mutate', [
    lambda policy: policy.pop('schema_version'),
    lambda policy: policy.update(unreviewed=True),
    lambda policy: policy['accelerate'].pop('sha256'),
    lambda policy: policy['accelerate'].update(unreviewed=True),
])
def test_missing_and_unknown_policy_fields_fail_closed(graph, mutate):
    mutate(graph.policy)
    graph.write_policy()
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('text', [
    '{"schema_version":1,"schema_version":1,"accelerate":{}}',
    '{"schema_version":1,"accelerate":{"version":"a","version":"b"}}',
    '{"schema_version":NaN,"accelerate":{}}', '{incomplete',
])
def test_duplicate_or_malformed_json_is_never_silently_normalized(graph, text):
    graph.write(POLICY, text)
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('mutate', [
    lambda url: url.partition('#')[0],
    lambda url: url + '&token=synthetic-secret',
    lambda url: url.replace('#sha256=', '?token=synthetic-secret#sha256='),
    lambda url: url.replace('https://', 'http://'),
    lambda url: url.replace('raw.githubusercontent.com', 'raw.githubusercontent.com.evil.invalid'),
    lambda url: url.replace('raw.githubusercontent.com', 'user:synthetic-secret@raw.githubusercontent.com'),
    lambda url: url.replace(COMMIT, 'main'),
    lambda url: url.replace(COMMIT, 'e' * 40),
    lambda url: url.replace('%2B', '+'),
    lambda url: url.replace('%2B', '%2b'),
    lambda url: url.replace('/shersingh7/', '/other-owner/'),
    lambda url: url + '\n',
])
def test_immutable_url_is_exact_and_errors_do_not_leak_input(graph, mutate):
    graph.patch['immutable_url'] = mutate(graph.url)
    graph.write_policy()
    with pytest.raises(ValueError) as error:
        graph.verify()
    assert 'synthetic-secret' not in str(error.value)
    assert 'http' not in str(error.value)


def test_changed_wheel_bytes_fail_even_when_version_label_is_unchanged(graph):
    graph.wheel.write_bytes(graph.wheel.read_bytes() + b'SYNTHETIC TAMPERING')
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('changes', [
    {'name': 'not-accelerate'}, {'version': '0.0.0+wrong'},
    {'extra_headers': 'Name: accelerate\n'}, {'extra_headers': f'Version: {VERSION}\n'},
    {'metadata_path': f'other-{VERSION}.dist-info/METADATA'},
    {'extra_entries': [(f'other-{VERSION}.dist-info/METADATA', 'Name: other\nVersion: 1\n')]},
])
def test_actual_wheel_metadata_not_just_filename_must_match(graph, changes):
    graph.wheel.write_bytes(synthetic_wheel(**changes))
    graph.rebind_changed_wheel()
    with pytest.raises(ValueError):
        graph.verify()


def test_invalid_zip_with_matching_hash_and_all_sources_is_rejected(graph):
    graph.wheel.write_bytes(b'SYNTHETIC invalid wheel, deliberately not a ZIP')
    graph.rebind_changed_wheel()
    with pytest.raises(ValueError, match='wheel ZIP'):
        graph.verify()


def test_duplicate_zip_names_are_rejected_even_when_metadata_matches(graph):
    with pytest.warns(UserWarning, match='Duplicate name'):
        graph.wheel.write_bytes(synthetic_wheel(extra_entries=[('accelerate/__init__.py', 'synthetic duplicate')]))
    graph.rebind_changed_wheel()
    with pytest.raises(ValueError, match='duplicate entries'):
        graph.verify()


def test_compressed_metadata_read_is_bounded_independently_of_wheel_size(graph):
    graph.wheel.write_bytes(synthetic_wheel(extra_headers='X-Synthetic: ' + 'a' * (1024 * 1024)))
    graph.rebind_changed_wheel()
    with pytest.raises(ValueError, match='bounded regular'):
        graph.verify()


@pytest.mark.parametrize('object_kind', ['directory', 'fifo', 'symlink', 'dangling_symlink', 'hardlink'])
@pytest.mark.parametrize('target', [POLICY, WHEEL, RUNNERS[0], 'finetune/uv.lock', RUNNERS[0] + '.lock'])
def test_nonregular_and_alias_paths_are_rejected_without_opening_objects(graph, object_kind, target):
    path = graph.root / target
    original = path.read_bytes()
    path.unlink()
    if object_kind == 'directory':
        path.mkdir()
    elif object_kind == 'fifo':
        os.mkfifo(path)
    elif object_kind in {'symlink', 'hardlink'}:
        other = graph.root / 'synthetic-alias-target'
        other.write_bytes(original)
        if object_kind == 'symlink':
            path.symlink_to(other)
        else:
            os.link(other, path)
    else:
        path.symlink_to(graph.root / 'nonexistent')
    with pytest.raises(ValueError):
        graph.verify(records=[{'ecosystem': 'pip', 'name': 'accelerate', 'original_name': 'accelerate',
                               'version': VERSION, 'source': 'finetune/uv.lock', 'locator': '1'}])


def test_symlinked_wheel_parent_is_rejected(graph):
    parent = graph.wheel.parent
    moved = graph.root / 'wheel-alias'
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        graph.verify()


def test_wheel_read_is_bounded(graph):
    with graph.wheel.open('r+b') as stream:
        stream.truncate(25 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='bounded read budget'):
        graph.verify()


@pytest.mark.parametrize('requirement', [
    'accelerate>=0', 'accelerate==0.0.0', f'accelerate=={VERSION}; sys_platform == "linux"',
    f'accelerate=={VERSION}.*', f'accelerate[extra]=={VERSION}',
    'accelerate @ https://example.invalid/unreviewed.whl',
])
@pytest.mark.parametrize('location', ['project_dependency', 'project_constraint', 'runner_dependency', 'runner_constraint'])
def test_every_requirement_is_an_unconditional_exact_pin(graph, requirement, location):
    if location == 'project_dependency':
        graph.write_project(dependency=requirement)
    elif location == 'project_constraint':
        graph.write_project(constraint=requirement)
    elif location == 'runner_dependency':
        graph.write_runner(RUNNERS[-1], dependency=requirement)
    else:
        graph.write_runner(RUNNERS[-1], constraint=requirement)
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('location', ['project', 'runner'])
@pytest.mark.parametrize('kind', ['missing', 'duplicate_dependency', 'duplicate_constraint'])
def test_requirement_presence_and_uniqueness(graph, location, kind):
    relative = 'finetune/pyproject.toml' if location == 'project' else RUNNERS[-1]
    path = graph.root / relative
    text = path.read_text()
    if kind == 'missing':
        text = text.replace(f'"accelerate=={VERSION}"', '"pydantic>=2"')
    else:
        prefix = '' if location == 'project' else '# '
        key = 'dependencies' if kind == 'duplicate_dependency' else 'constraint-dependencies'
        text = text.replace(f'{prefix}{key} = [', f'{prefix}{key} = ["accelerate=={VERSION}", ')
    path.write_text(text)
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('location', ['project', 'runner'])
def test_constraint_cannot_disappear_while_direct_requirement_stays_pinned(graph, location):
    relative = 'finetune/pyproject.toml' if location == 'project' else RUNNERS[-1]
    path = graph.root / relative
    path.write_text(path.read_text().replace(
        f'constraint-dependencies = ["accelerate=={VERSION}"]', 'constraint-dependencies = []'))
    with pytest.raises(ValueError, match='missing or duplicated'):
        graph.verify()


@pytest.mark.parametrize('location', ['project', 'runner'])
def test_source_table_cannot_disappear(graph, location):
    relative = 'finetune/pyproject.toml' if location == 'project' else RUNNERS[-1]
    path = graph.root / relative
    path.write_text(path.read_text().replace('[tool.uv.sources]', '[tool.uv.unrelated]'))
    with pytest.raises(ValueError, match='sources'):
        graph.verify()


@pytest.mark.parametrize('location', ['project', 'runner'])
def test_source_aliases_cannot_hide_duplicate_package_sources(graph, location):
    relative = 'finetune/pyproject.toml' if location == 'project' else RUNNERS[-1]
    path = graph.root / relative
    prefix = '' if location == 'project' else '# '
    source = ('{ path = ' + json.dumps(LOCAL_WHEEL) + ' }' if location == 'project'
              else '{ url = ' + json.dumps(graph.url) + ' }')
    path.write_text(path.read_text().replace(
        prefix + '[tool.uv.sources]', prefix + '[tool.uv.sources]\n' + prefix + 'Accelerate = ' + source))
    with pytest.raises(ValueError, match='ambiguous'):
        graph.verify()


@pytest.mark.parametrize('location', ['project', 'runner'])
def test_uv_dependency_override_cannot_bypass_the_reviewed_pin(graph, location):
    relative = 'finetune/pyproject.toml' if location == 'project' else RUNNERS[-1]
    path = graph.root / relative
    prefix = '' if location == 'project' else '# '
    path.write_text(path.read_text().replace(
        prefix + '[tool.uv]', prefix + '[tool.uv]\n' + prefix + 'override-dependencies = ["accelerate==0.0.0"]'))
    with pytest.raises(ValueError, match='exact reviewed pin'):
        graph.verify()


@pytest.mark.parametrize('source', [
    '{ path = "../vendor/unsafe.whl" }', '{ path = "./' + LOCAL_WHEEL + '" }',
    '{ registry = "https://pypi.org/simple" }',
    '{ path = "' + LOCAL_WHEEL + '", marker = "sys_platform == \'linux\'" }',
    '[{ path = "' + LOCAL_WHEEL + '" }]',
])
def test_project_source_cannot_be_registry_or_marker_selected_alias(graph, source):
    graph.write_project(source=source)
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('change', ['missing', 'fragment', 'wrong_hash', 'wrong_url', 'local_path', 'marker', 'alias'])
def test_every_script_has_the_exact_uv_supported_immutable_source(graph, change):
    if change == 'missing':
        path = graph.root / RUNNERS[-1]
        path.write_text(path.read_text().replace('# accelerate = {', '# other = {'))
    else:
        sources = {
            'fragment': '{ url = ' + json.dumps(graph.url) + ' }',
            'wrong_hash': '{ url = ' + json.dumps(graph.url[:-1] + ('0' if graph.url[-1] != '0' else '1')) + ' }',
            'wrong_url': '{ url = "https://example.invalid/unsafe.whl#sha256=' + graph.patch['sha256'] + '" }',
            'local_path': '{ path = ' + json.dumps(LOCAL_WHEEL) + ' }',
            'marker': '{ url = ' + json.dumps(graph.url) + ', marker = "sys_platform == \'linux\'" }',
            'alias': '[{ url = ' + json.dumps(graph.url) + ' }]',
        }
        graph.write_runner(RUNNERS[-1], source=sources[change])
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('change', ['no_source', 'registry', 'alias', 'wrong_version', 'no_hash', 'wrong_hash', 'wrong_filename', 'sdist'])
def test_all_uv_instances_are_bound_by_source_filename_version_and_hash(graph, change):
    path = graph.root / 'finetune/uv.lock'
    text = path.read_text()
    if change == 'no_source':
        text = text.replace('source = { path = ' + json.dumps(LOCAL_WHEEL) + ' }\n', '')
    elif change == 'registry':
        text = text.replace('path = ' + json.dumps(LOCAL_WHEEL), 'registry = "https://pypi.org/simple"')
    elif change == 'alias':
        text = text.replace(LOCAL_WHEEL, './' + LOCAL_WHEEL)
    elif change == 'wrong_version':
        text = text.replace(f'version = "{VERSION}"', 'version = "0.0.0"')
    elif change == 'no_hash':
        text = text.replace(', hash = "sha256:' + graph.patch['sha256'] + '"', '')
    elif change == 'wrong_hash':
        text = text.replace(graph.patch['sha256'], '0' * 64)
    elif change == 'wrong_filename':
        text = text.replace('filename = ' + json.dumps(graph.wheel.name), 'filename = "other.whl"')
    else:
        text += 'sdist = { url = "https://example.invalid/unreviewed.tar.gz" }\n'
    path.write_text(text)
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('change', ['extra', 'removed', 'emptied', 'policy_omits', 'policy_adds'])
def test_real_runner_inventory_equals_reviewed_policy(graph, change):
    if change == 'extra':
        graph.write_runner('finetune/dataset/new_runner.py', direct=False)
    elif change == 'removed':
        (graph.root / RUNNERS[-1]).unlink()
    elif change == 'emptied':
        graph.write(RUNNERS[-1], '# /// script\n# dependencies = []\n# ///\n')
    elif change == 'policy_omits':
        graph.patch['runner_paths'].pop()
        graph.write_policy()
    else:
        graph.patch['runner_paths'].append('finetune/not_present.py')
        graph.write_policy()
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('text', [
    '# /// script\n# dependencies = ["pydantic"]\n',
    '# /// script\ndependencies = ["pydantic"]\n# ///\n',
    '# /// script\n#dependencies = ["pydantic"]\n# ///\n',
    '# /// script \n# dependencies = ["pydantic"]\n# ///\n',
    ' # /// script\n# dependencies = ["pydantic"]\n# ///\n',
    '# /// script\n# dependencies = ["pydantic"]\n# ///\n' * 2,
    '# /// script\n# dependencies = ["pydantic"]\n# dependencies = []\n# ///\n',
    '# /// script\n# dependencies = "pydantic"\n# ///\n',
    '# /// script\n# dependencies = [1]\n# ///\n',
    '# ///\n',
])
def test_malformed_or_duplicate_metadata_blocks_never_hide_a_runner(graph, text):
    graph.write('finetune/dataset/new_runner.py', text)
    with pytest.raises(ValueError):
        graph.verify()


@pytest.mark.parametrize('separator', ['\r', '\v', '\f', '\x85', '\u2028', '\u2029'])
def test_non_pep_line_separators_cannot_change_which_metadata_is_parsed(graph, separator):
    path = graph.root / RUNNERS[-1]
    path.write_text(path.read_text().replace('\n', separator))
    with pytest.raises(ValueError, match='line separators'):
        graph.verify()


def test_crlf_metadata_is_supported_without_parser_disagreement(graph):
    path = graph.root / RUNNERS[-1]
    path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    assert graph.verify()[0] == BASE_PROOFS


def test_walk_prunes_generated_hidden_vendor_and_cache_trees(graph):
    for directory in ('.venv', 'vendor', '__pycache__', 'cache', 'node_modules'):
        graph.write(f'finetune/{directory}/untrusted.py', '# /// script\nmalformed\n')
    proofs, _ = graph.verify()
    assert proofs == BASE_PROOFS


def test_nonignored_directory_symlink_cannot_hide_additional_runners(graph):
    (graph.root / 'finetune/aliased-directory').symlink_to(graph.root / 'finetune/jobs', target_is_directory=True)
    with pytest.raises(ValueError):
        graph.verify()


def test_toml_parser_errors_do_not_reflect_secret_source_urls(graph):
    path = graph.root / RUNNERS[-1]
    path.write_text(path.read_text().replace('# [tool.uv.sources]',
                    '# broken = "https://user:synthetic-secret@example.invalid/\n# [tool.uv.sources]'))
    with pytest.raises(ValueError, match='Malformed TOML') as error:
        graph.verify()
    assert 'synthetic-secret' not in str(error.value)
    assert 'http' not in str(error.value)


def test_file_growth_during_a_descriptor_read_is_rejected(graph, monkeypatch):
    original_read = os.read
    modified = False
    def mutate_during_read(descriptor, count):
        nonlocal modified
        chunk = original_read(descriptor, count)
        if chunk.startswith(b'PK') and not modified:
            modified = True
            with graph.wheel.open('ab') as stream:
                stream.write(b'SYNTHETIC concurrent mutation')
        return chunk
    monkeypatch.setattr(binding.os, 'read', mutate_during_read)
    with pytest.raises(ValueError, match='changed while being read'):
        graph.verify()
    assert modified


@pytest.mark.parametrize('case', ['unsafe_extra_lock', 'safe_extra_lock_after_collection', 'extra_record', 'version', 'locator', 'duplicate', 'original_name'])
def test_late_or_mismatched_resolved_instances_fail_closed(graph, case):
    records = graph.records()
    if case == 'unsafe_extra_lock':
        graph.write_lock(graph.lock_entry(version='0.0.0'))
    elif case == 'safe_extra_lock_after_collection':
        graph.write_lock(graph.lock_entry())
    elif case == 'extra_record':
        records.append({**records[-1], 'locator': '2'})
    elif case == 'version':
        records[-1]['version'] = '0.0.0'
    elif case == 'locator':
        records[-1]['locator'] = '01'
    elif case == 'duplicate':
        records.append(records[-1].copy())
    else:
        records[-1]['original_name'] = 'other'
    with pytest.raises(ValueError):
        graph.verify(records)


def test_other_source_and_path_alias_instances_receive_no_proof(graph):
    records = graph.records()
    extras = [{**records[-1], 'source': source} for source in (
        'other/uv.lock', './finetune/uv.lock', 'https://user:synthetic-secret@example.invalid/lock',
    )]
    with pytest.raises(ValueError, match='unbound') as error:
        graph.verify(records + extras)
    assert 'synthetic-secret' not in str(error.value)
    assert 'http' not in str(error.value)


def test_missing_resolved_record_does_not_create_proof_from_lock_alone(graph):
    with pytest.raises(ValueError):
        graph.verify(graph.records()[:-1])


def test_verifier_is_read_only_and_does_not_execute_install_or_connect(graph, monkeypatch, tmp_path):
    before = {str(path.relative_to(graph.root)): path.read_bytes()
              for path in graph.root.rglob('*') if path.is_file()}
    home = tmp_path / 'synthetic-home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(home)
    def forbidden(*args, **kwargs):
        raise AssertionError('No subprocess, artifact execution or network is permitted')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    graph.verify()
    after = {str(path.relative_to(graph.root)): path.read_bytes()
             for path in graph.root.rglob('*') if path.is_file()}
    assert before == after
    assert list(home.iterdir()) == []


@pytest.mark.parametrize('state', [
    'valid', 'missing_policy', 'changed_runner', 'changed_wheel', 'extra_instance',
    'missing_script_lock', 'malformed_script_lock', 'stale_script_lock', 'wrong_script_hash',
])
def test_real_audit_cli_uses_whole_source_binding_before_granting_proof(graph, state):
    # All records/artifacts here remain explicitly synthetic. The actual CLI,
    # parser, graph collector, verifier and assessment compose without mocks.
    graph.write('package.json', json.dumps({'dependencies': {'example': '2.0.0'}}))
    graph.write('pnpm-lock.yaml', "lockfileVersion: '9.0'\npackages:\n  example@2.0.0: {}\n")
    graph.write('bun.lock', '{"lockfileVersion":1,"packages":{"example":["example@2.0.0","",{},"synthetic"]}}')
    graph.write('scripts/mlx_server_constraints.txt', 'requests==2.34.2\n')
    graph.write('scripts/security_audit_requirements.txt', '-c mlx_server_constraints.txt\njson5==0.17.3\n')
    graph.write('synthetic-alerts.json', json.dumps([{
        'number': 1,
        'dependency': {'package': {'ecosystem': 'pip', 'name': 'accelerate'}, 'manifest_path': 'finetune/uv.lock'},
        'security_vulnerability': {'severity': 'critical', 'vulnerable_version_range': '<=1.14.0', 'first_patched_version': None},
        'security_advisory': {'ghsa_id': 'GHSA-4j2p-28q2-5m79'},
    }]))
    if state == 'missing_policy':
        (graph.root / POLICY).unlink()
    elif state == 'changed_runner':
        graph.write_runner(RUNNERS[-1], source='{ url = "https://example.invalid/unreviewed.whl" }')
    elif state == 'changed_wheel':
        graph.wheel.write_bytes(graph.wheel.read_bytes() + b'SYNTHETIC MODIFICATION')
    elif state == 'extra_instance':
        graph.write_lock(graph.lock_entry(source='{ registry = "https://pypi.org/simple" }'))
    elif state.endswith('script_lock') or state == 'wrong_script_hash':
        lock_path = graph.root / (RUNNERS[-1] + '.lock')
        if state == 'missing_script_lock':
            lock_path.unlink()
        elif state == 'malformed_script_lock':
            lock_path.write_text('bad = [ TOML')
        elif state == 'stale_script_lock':
            lock_path.write_text(lock_path.read_text().replace('>=3.11', '>=3.12'))
        else:
            lock_path.write_text(lock_path.read_text().replace(graph.patch['sha256'], '0' * 64))
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/check_dependency_security.py'), '--repo', str(graph.root),
         '--alerts', str(graph.root / 'synthetic-alerts.json'), '--output', str(graph.root / 'synthetic-report.json')],
        capture_output=True, text=True, timeout=20,
    )
    report = json.loads((graph.root / 'synthetic-report.json').read_text())
    assert report['passed'] is (state == 'valid'), (result.stdout, result.stderr)
    assert result.returncode == (0 if state == 'valid' else 1)
    if state == 'valid':
        assert report['source_binding']['status'] == 'verified_source_binding'
        assert report['source_binding']['resolved_instances'] == 4
        assert report['alerts'][0]['status'] == 'verified_patched_artifact'
    elif state == 'missing_policy':
        assert report['source_binding']['status'] == 'not_configured'
        assert report['alerts'][0]['status'] == 'unverified_source_mitigation'


@pytest.mark.parametrize('change', [
    'missing', 'malformed', 'version', 'revision', 'empty_graph', 'missing_source',
    'registry', 'wrong_url', 'fragment_url', 'wrong_version', 'missing_hash',
    'wrong_hash', 'extra_wheel', 'sdist', 'missing_accelerate', 'oversized',
])
def test_each_script_lock_is_required_and_every_accelerate_artifact_is_bound(graph, change):
    records = graph.records()
    path = graph.root / (RUNNERS[-1] + '.lock')
    text = path.read_text()
    if change == 'missing':
        path.unlink()
    elif change == 'oversized':
        with path.open('ab') as stream:
            stream.truncate(binding.MAX_LOCK_BYTES + 1)
    else:
        changes = {
            'malformed': 'not = [ valid TOML',
            'version': text.replace('version = 1\n', 'version = 2\n', 1),
            'revision': text.replace('revision = 3', 'revision = 999'),
            'empty_graph': text.partition('[[package]]')[0],
            'missing_source': text.replace('source = { url = ' + json.dumps(graph.installer_url) + ' }\n', ''),
            'registry': text.replace('source = { url = ' + json.dumps(graph.installer_url) + ' }',
                                     'source = { registry = "https://pypi.org/simple" }'),
            'wrong_url': text.replace(graph.installer_url, graph.installer_url.replace(COMMIT, 'e' * 40)),
            'fragment_url': text.replace(graph.installer_url, graph.url),
            'wrong_version': text.replace(f'version = "{VERSION}"', 'version = "0.0.0"'),
            'missing_hash': text.replace(', hash = "sha256:' + graph.patch['sha256'] + '"', ''),
            'wrong_hash': text.replace(graph.patch['sha256'], '0' * 64),
            'extra_wheel': text.replace('wheels = [', 'wheels = [{ url = "https://example.invalid/other.whl" }, '),
            'sdist': text + 'sdist = { url = "https://example.invalid/unsafe.tar.gz" }\n',
            'missing_accelerate': text.partition('[[package]]\nname = "accelerate"')[0],
        }
        path.write_text(changes[change])
    with pytest.raises(ValueError):
        graph.verify(records)


@pytest.mark.parametrize('change', [
    'python', 'missing_manifest', 'missing_requirements', 'missing_constraints',
    'requirements', 'constraints', 'overrides', 'unknown_field', 'bad_requirement',
    'duplicate_requirement', 'changed_script_dependency', 'changed_script_python',
])
def test_script_lock_manifest_must_match_current_metadata(graph, change):
    path = graph.root / (RUNNERS[-1] + '.lock')
    text = path.read_text()
    if change.startswith('changed_script_'):
        runner = graph.root / RUNNERS[-1]
        before = runner.read_text()
        if change == 'changed_script_dependency':
            runner.write_text(before.replace('# dependencies = [', '# dependencies = ["pydantic>=2", '))
        else:
            runner.write_text(before.replace('>=3.11', '>=3.12'))
    else:
        changes = {
            'python': text.replace('>=3.11', '>=3.12'),
            'missing_manifest': text.replace('[manifest]', '[not_manifest]'),
            'missing_requirements': text.replace('requirements =', 'not_requirements ='),
            'missing_constraints': text.replace('constraints =', 'not_constraints ='),
            'requirements': text.replace('requirements = [{ name = "accelerate", url = ' + json.dumps(graph.installer_url) + ' }]',
                                         'requirements = [{ name = "pydantic", specifier = ">=2" }]'),
            'constraints': text.replace('constraints = [{ name = "accelerate", url = ' + json.dumps(graph.installer_url) + ' }]',
                                        'constraints = []'),
            'overrides': text.replace('[manifest]', '[manifest]\noverrides = [{ name = "accelerate", specifier = "==0.0.0" }]'),
            'unknown_field': text.replace('[manifest]', '[manifest]\nunknown = []'),
            'bad_requirement': text.replace('requirements = [', 'requirements = ["not a requirement", '),
            'duplicate_requirement': text.replace('requirements = [', 'requirements = [{ name = "accelerate", url = '
                                                   + json.dumps(graph.installer_url) + ' }, '),
        }
        path.write_text(changes[change])
    with pytest.raises(ValueError, match='manifest|metadata|TOML'):
        graph.verify()


def test_transitive_script_instance_is_proved_without_adding_a_direct_accelerate_requirement(graph):
    runner = 'finetune/dataset/prepare.py'
    graph.write_script_lock(runner, direct=False, transitive=True)
    proofs, receipt = graph.verify()
    assert proofs == BASE_PROOFS | {(VERSION, runner + '.lock', '1')}
    assert receipt['resolved_instances'] == 5
    assert binding._script_metadata((graph.root / runner).read_text())['dependencies'] == ['pydantic>=2']


@pytest.mark.parametrize('case', ['omitted', 'duplicate', 'wrong_locator', 'other_source'])
def test_script_proofs_are_instance_specific_and_not_optional(graph, case):
    records = graph.records()
    target = next(row for row in records if row['name'] == 'accelerate' and row['source'] != 'finetune/uv.lock')
    if case == 'omitted':
        records.remove(target)
    elif case == 'duplicate':
        records.append(target.copy())
    elif case == 'wrong_locator':
        target['locator'] = '0'
    else:
        target['source'] = 'finetune/other.py.lock'
    with pytest.raises(ValueError):
        graph.verify(records)


def test_special_script_lock_is_rejected_before_open(graph, monkeypatch):
    records = graph.records()
    relative = RUNNERS[-1] + '.lock'
    path = graph.root / relative
    path.unlink()
    os.mkfifo(path)
    original_open = os.open
    def no_fifo_open(filename, *args, **kwargs):
        assert filename != path.name, 'A known FIFO must not be opened, even nonblocking'
        return original_open(filename, *args, **kwargs)
    monkeypatch.setattr(binding.os, 'open', no_fifo_open)
    with pytest.raises(ValueError):
        graph.verify(records)


def test_real_uv_supported_source_and_locked_hash_enforcement(tmp_path):
    """Real uv, real HTTP and install; only a model-free SYNTHETIC tiny wheel.

    CI must install uv explicitly. Missing uv is a failure, not a skipped proof.
    Each runtime control gets a cold cache so a previous install cannot bypass
    the download/hash check. No product dependency or runner is installed/run.
    """
    import functools
    import http.server
    import shutil
    import socketserver
    import threading

    uv = shutil.which('uv')
    assert uv, 'The real uv executable is required for the hash enforcement regression'
    wheel = tmp_path / 'qmd_synthetic_lock_probe-0.0.0-py3-none-any.whl'
    dist = 'qmd_synthetic_lock_probe-0.0.0.dist-info'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('qmd_synthetic_lock_probe.py', '__version__ = "0.0.0"\n')
        archive.writestr(dist + '/METADATA', 'Metadata-Version: 2.4\nName: qmd-synthetic-lock-probe\nVersion: 0.0.0\n\nSYNTHETIC TEST ONLY\n')
        archive.writestr(dist + '/WHEEL', 'Wheel-Version: 1.0\nGenerator: synthetic-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr(dist + '/RECORD', ''.join(name + ',,\n' for name in [
            'qmd_synthetic_lock_probe.py', dist + '/METADATA', dist + '/WHEEL', dist + '/RECORD']))
    digest = sha256(wheel.read_bytes()).hexdigest()
    requested = []
    class NumericServer(http.server.HTTPServer):
        def server_bind(self):
            socketserver.TCPServer.server_bind(self)
            self.server_name = str(self.server_address[0])
            self.server_port = self.server_address[1]
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.path)
            return super().do_GET()
        def log_message(self, format, *args):
            pass
    home = tmp_path / 'home'
    home.mkdir()
    env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(home),
           'TMPDIR': str(tmp_path), 'UV_CACHE_DIR': str(tmp_path / 'resolve-cache'),
           'UV_PYTHON_DOWNLOADS': 'never', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    with NumericServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(tmp_path))) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/{wheel.name}'
            script = tmp_path / 'probe.py'
            source = ('# /// script\n# requires-python = ">=3.11"\n'
                      '# dependencies = ["qmd-synthetic-lock-probe==0.0.0"]\n'
                      '# [tool.uv.sources]\n# qmd-synthetic-lock-probe = { url = ' + json.dumps(url) + ' }\n'
                      '# ///\nimport qmd_synthetic_lock_probe\nprint("SYNTHETIC_UV_LOCK_PROBE_RAN")\n')
            script.write_text(source)
            command = [uv, 'lock', '--script', str(script), '--python', sys.executable,
                       '--no-build', '--no-index', '--no-config']
            resolved = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=45)
            assert resolved.returncode == 0, resolved.stderr
            lock = Path(str(script) + '.lock')
            good = lock.read_text()
            package = tomllib.loads(good)['package'][0]
            assert package['source'] == {'url': url}
            assert package['wheels'] == [{'url': url, 'hash': 'sha256:' + digest}]
            checked = subprocess.run([*command, '--check'], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=45)
            assert checked.returncode == 0, checked.stderr
            assert lock.read_text() == good
            runtime = [uv, 'run', '--locked', '--no-build', '--no-index', '--no-config',
                       '--python', sys.executable, str(script)]
            installed = subprocess.run(runtime, env={**env, 'UV_CACHE_DIR': str(tmp_path / 'good-cache')},
                                       cwd=tmp_path, capture_output=True, text=True, timeout=45)
            assert installed.returncode == 0, installed.stderr
            assert installed.stdout.strip() == 'SYNTHETIC_UV_LOCK_PROBE_RAN'
            lock.write_text(good.replace('sha256:' + digest, 'sha256:' + '0' * 64))
            rejected = subprocess.run(runtime, env={**env, 'UV_CACHE_DIR': str(tmp_path / 'bad-cache')},
                                      cwd=tmp_path, capture_output=True, text=True, timeout=45)
            assert rejected.returncode != 0
            assert 'hash mismatch' in rejected.stderr.lower(), rejected.stderr
            assert digest in rejected.stderr and '0' * 64 in rejected.stderr
            assert 'SYNTHETIC_UV_LOCK_PROBE_RAN' not in rejected.stdout
            lock.write_text(good)
            script.write_text(source.replace('requires-python = ">=3.11"', 'requires-python = ">=3.12"'))
            stale = subprocess.run([*command, '--check'], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=45)
            assert stale.returncode != 0 and 'needs to be updated' in stale.stderr, stale.stderr
            assert lock.read_text() == good
            script.write_text(source.replace(url, url + '#sha256=' + digest))
            fragment = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=45)
            assert fragment.returncode != 0 and 'fragment' in fragment.stderr.lower(), fragment.stderr
            assert requested and set(requested) == {'/' + wheel.name}
            # Retain real outputs, including the deliberate failed operations,
            # separately from the passing pytest assertion summary.
            with (tmp_path / 'uv-runtime-receipt.json').open('x') as receipt:
                json.dump({'fixture': 'SYNTHETIC MODEL-FREE WHEEL ONLY', 'wheel_sha256': digest,
                           'resolve_exit_code': resolved.returncode, 'check_exit_code': checked.returncode,
                           'good_runtime_exit_code': installed.returncode,
                           'wrong_hash_exit_code': rejected.returncode, 'wrong_hash_stderr': rejected.stderr,
                           'stale_metadata_exit_code': stale.returncode, 'stale_metadata_stderr': stale.stderr,
                           'unsupported_fragment_exit_code': fragment.returncode,
                           'unsupported_fragment_stderr': fragment.stderr}, receipt, indent=2)
        finally:
            server.shutdown()
            thread.join(timeout=3)
            assert not thread.is_alive()
