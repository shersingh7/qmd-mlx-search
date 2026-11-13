"""Model-free contracts for standalone metadata propagation; sources are synthetic."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'qmd_script_source_sync', Path(__file__).resolve().parents[1] / 'tools/sync_script_dependencies.py'
)
assert SPEC and SPEC.loader
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)

VERSION = '0.0.0+qmd.synthetic1'
COMMIT = 'd' * 40
DIGEST = 'a' * 64
WHEEL = f'finetune/vendor/accelerate/accelerate-{VERSION}-py3-none-any.whl'
IDENTITY_URL = (f'https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/{COMMIT}/'
                f'finetune/vendor/accelerate/accelerate-0.0.0%2Bqmd.synthetic1-py3-none-any.whl#sha256={DIGEST}')
# uv rejects fragments in tool.uv.sources. Integrity belongs to the script lock.
URL = IDENTITY_URL.partition('#')[0]


def project():
    return {
        'project': {'requires-python': '>=3.11', 'dependencies': [
            f'accelerate=={VERSION}', 'pydantic>=2.12.3,<3',
            'bitsandbytes>=0.50.2,<0.51; sys_platform == "linux"',
        ]},
        'tool': {'uv': {
            'constraint-dependencies': [f'accelerate=={VERSION}', 'transformers>=5.17.0'],
            'sources': {'accelerate': {'path': WHEEL.removeprefix('finetune/')}},
        }},
    }


def policy():
    return {'schema_version': 1, 'accelerate': {
        'advisory': 'GHSA-4j2p-28q2-5m79', 'version': VERSION,
        'wheel_path': WHEEL, 'sha256': DIGEST, 'commit': COMMIT,
        'immutable_url': IDENTITY_URL, 'runner_paths': ['finetune/train.py'],
    }}


def write_policy(root, value):
    path = root.parent / 'docs/security/dependency-remediation/patched-dependencies.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_project_source_requires_an_explicit_publication_policy(tmp_path):
    root = tmp_path / 'finetune'
    root.mkdir()
    with pytest.raises(ValueError, match='policy'):
        sync.script_sources(project(), root)
    write_policy(root, policy())
    assert sync.script_sources(project(), root) == {'accelerate': {'url': URL}}


@pytest.mark.parametrize('field,value', [
    ('immutable_url', IDENTITY_URL.replace(COMMIT, 'main')),
    ('immutable_url', IDENTITY_URL.replace(DIGEST, 'b' * 64)),
    ('immutable_url', IDENTITY_URL.replace('raw.githubusercontent.com', 'example.invalid')),
    ('version', '1.15.0'), ('version', '0.0.0+QMD.synthetic1'),
    ('commit', 'D' * 40), ('sha256', 'A' * 64),
    ('wheel_path', '../outside.whl'), ('advisory', 'not-reviewed'),
])
def test_invalid_publication_mapping_is_not_propagated(tmp_path, field, value):
    root = tmp_path / 'finetune'
    root.mkdir()
    value_policy = policy()
    value_policy['accelerate'][field] = value
    write_policy(root, value_policy)
    with pytest.raises(ValueError):
        sync.script_sources(project(), root)


def test_duplicate_policy_fields_fail(tmp_path):
    root = tmp_path / 'finetune'
    root.mkdir()
    path = write_policy(root, policy())
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError):
        sync.script_sources(project(), root)


@pytest.mark.parametrize('location', ['dependencies', 'constraint-dependencies', 'sources'])
def test_project_declaration_must_match_publication(tmp_path, location):
    root = tmp_path / 'finetune'
    root.mkdir()
    write_policy(root, policy())
    value = project()
    if location == 'dependencies':
        value['project']['dependencies'][0] = 'accelerate>=1.15.0'
    elif location == 'constraint-dependencies':
        value['tool']['uv'][location][0] = 'accelerate>=1.15.0'
    else:
        value['tool']['uv'][location]['accelerate']['path'] = 'vendor/other.whl'
    with pytest.raises(ValueError):
        sync.script_sources(value, root)


def test_data_only_script_does_not_gain_accelerate_or_torch():
    old = {'requires-python': '>=3.10', 'dependencies': ['pydantic>=2']}
    rendered = sync.render_block(old, project(), sources={'accelerate': {'url': URL}})
    result = sync.metadata(rendered)
    assert result['dependencies'] == ['pydantic>=2.12.3,<3']
    assert result['tool']['uv']['sources']['accelerate'] == {'url': URL}
    assert f'accelerate=={VERSION}' in result['tool']['uv']['constraint-dependencies']
    assert sync.render_block(result, project(), sources={'accelerate': {'url': URL}}) == rendered


def test_renderer_preserves_unrelated_metadata_and_uses_project_optimizer_policy():
    old = {'dependencies': ['bitsandbytes>=0.48'], 'tool': {
        'example': {'literal': 'quote"\nnew line', 'enabled': True, 'depth': 3},
        'uv': {'override-dependencies': ['example==1.0'], 'sources': {
            'example': {'git': 'https://example.invalid/example.git', 'rev': 'synthetic'},
            'accelerate': {'url': 'https://example.invalid/obsolete.whl'},
        }},
    }}
    original = deepcopy(old)
    result = sync.metadata(sync.render_block(old, project(), sources={'accelerate': {'url': URL}}))
    assert old == original
    assert result['dependencies'] == ['bitsandbytes>=0.50.2,<0.51; sys_platform == "linux"']
    assert result['tool']['example'] == old['tool']['example']
    assert result['tool']['uv']['override-dependencies'] == ['example==1.0']
    assert result['tool']['uv']['sources']['example'] == old['tool']['uv']['sources']['example']
    assert result['tool']['uv']['sources']['accelerate'] == {'url': URL}


def test_local_wheel_path_never_leaks_into_standalone_metadata():
    with pytest.raises(ValueError):
        sync.render_block({'dependencies': ['pydantic>=2']}, project())


def test_duplicate_script_blocks_are_rejected():
    block = '# /// script\n# dependencies = []\n# ///\n'
    with pytest.raises(ValueError):
        sync.metadata(block + block)


def test_registry_only_project_does_not_require_patch_policy(tmp_path):
    value = project()
    del value['tool']['uv']['sources']
    assert sync.script_sources(value, tmp_path / 'finetune') == {}
