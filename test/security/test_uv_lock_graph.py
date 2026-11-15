"""Synthetic contracts for the bounded uv graph/marker proof; no resolver I/O."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('qmd_lock_graph_under_test', ROOT / 'scripts/uv_lock_graph.py')
assert SPEC and SPEC.loader
uv_graph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uv_graph)
REGISTRY = {'registry': 'https://pypi.org/simple'}


def package(name='base', version='1.0.0', /, **fields: Any):
    return {'name': name, 'version': version, 'source': REGISTRY.copy(), **fields}


def graph(*packages, dependencies=('base>=1',), python: Any='>=3.11', constraints=(), **fields: Any):
    lock = {'version': 1, 'revision': 3, 'requires-python': python,
            'package': list(packages) or [package()], **fields}
    metadata = {'requires-python': python, 'dependencies': list(dependencies),
                'tool': {'uv': {'constraint-dependencies': list(constraints)}}}
    return lock, metadata


def validate(pair, **kwargs):
    return uv_graph.validate_lock_graph(*pair, **kwargs)


def masks(*texts, python='>=3.11'):
    space = uv_graph.MarkerSpace()
    expressions = [space.marker(text) for text in texts]
    allowed = space.python(python)
    space.seal()
    return space, space.mask(allowed), [space.mask(expr) & space.mask(allowed) for expr in expressions]


@pytest.mark.parametrize('bound', ['0', '3.12', '3.12.0', '3.98.123', '99.345.67890'])
@pytest.mark.parametrize('left,right', [('<', '>='), ('<=', '>'), ('==', '!=')])
def test_numeric_complements_cover_the_entire_unbounded_domain(bound, left, right):
    space, domain, (a, b) = masks(f'python_full_version {left} "{bound}"',
                                 f'python_full_version {right} "{bound}"', python='>=0')
    assert a | b == domain == space.all
    assert not a & b


def test_python_projection_is_correlated_not_an_independent_environment_axis():
    _, domain, (minor, full, impossible) = masks(
        'python_version == "3.12"', 'python_full_version == "3.12.*"',
        'python_version < "3.12" and python_full_version >= "3.12"')
    assert minor == full != 0
    assert impossible == 0
    assert minor != domain


def test_all_python_micro_releases_and_future_major_releases_are_covered_symbolically():
    _, domain, (low, middle, high) = masks('python_full_version < "3.12"',
        'python_full_version == "3.12.*"', 'python_full_version >= "3.13"')
    assert low | middle | high == domain
    assert not (low & middle or middle & high or low & high)
    _, domain, (below, above, gap) = masks('python_full_version < "3.98.123"',
        'python_full_version >= "3.98.124"', 'python_full_version == "3.98.123"')
    assert domain & ~(below | above) == gap != 0


def test_wildcard_compatible_and_requires_python_exclusions_are_exact():
    _, domain, (compatible, prefix, excluded) = masks('python_full_version ~= "3.12.1"',
        'python_full_version == "3.12.*" and python_full_version >= "3.12.1"',
        'python_full_version == "3.12.123"', python='>=3.12,!=3.12.123,<3.13')
    assert compatible == prefix != 0
    assert excluded == 0
    assert compatible != domain  # 3.12.0 remains a real release cell.


def test_categorical_other_class_is_cofinite_and_machine_spellings_are_distinct():
    known = ['AMD64', 'ARM64', 'aarch64', 'amd64', 'arm64', 'x86_64']
    union = ' or '.join(f'platform_machine == "{name}"' for name in known)
    other = ' and '.join(f'platform_machine != "{name}"' for name in known)
    _, domain, (a, b, upper, lower) = masks(union, other,
        'platform_machine == "AMD64"', 'platform_machine == "amd64"')
    assert a | b == domain
    assert b and not a & b
    assert upper and lower and not upper & lower


def test_boolean_precedence_parentheses_reverse_comparisons_and_string_complements():
    _, domain, (a, b, eq, neq) = masks(
        '"3.12" <= python_full_version and (sys_platform == "win32" or sys_platform == "emscripten")',
        '(python_full_version >= "3.12" and sys_platform == "win32") or '
        '(python_full_version >= "3.12" and sys_platform == "emscripten")',
        '"linux" == sys_platform', 'sys_platform != "linux"')
    assert a == b != 0
    assert eq | neq == domain and not eq & neq


@pytest.mark.parametrize('marker', [
    'implementation_version >= "3.12"', 'platform_release == "unix"',
    'sys_platform in "linux win32"', 'sys_platform not in "win32"',
    'platform_machine >= "aarch64"', 'python_full_version == "3.12rc1"',
    'python_full_version >= "3.12.dev1"', 'python_full_version == "3.12.post1"',
    'python_full_version == "3.12+local"', 'python_full_version == "3.12.1.0"',
    'python_full_version === "3.12"', 'python_full_version == "3.12.0.*"',
    'python_full_version >= "3.12.*"', 'python_version < "3.12.1"',
    'python_version == "3"', 'python_full_version == sys_platform',
    '"linux" == "linux"', 'platform_machine == "1.0"',
    'sys_platform in extras', 'sys_platform ==',
])
def test_unsupported_or_ambiguous_marker_forms_fail_explicitly(marker):
    with pytest.raises(ValueError, match='[Uu]nsupported|[Mm]alformed|[Vv]ersion-like'):
        masks(marker)


@pytest.mark.parametrize('variable', ['python_version', 'python_full_version'])
@pytest.mark.parametrize('operator', ['==', '!='])
@pytest.mark.parametrize('literal', ['3.*', '3.12.*'])
def test_literal_left_python_wildcards_are_not_silently_reversed(variable, operator, literal):
    # Prefix comparisons are not symmetric in real uv/packaging. Refuse this
    # spelling instead of proving the different, variable-left proposition.
    with pytest.raises(ValueError, match='Unsupported literal-left Python wildcard'):
        masks(f'"{literal}" {operator} {variable}')


@pytest.mark.parametrize('python', ['>=3.12rc1', '===3.12', '>=3.12.0.1', '==3.12.0.*'])
def test_unsupported_python_domains_never_get_approximated(python):
    with pytest.raises(ValueError):
        masks(python=python)


def test_extra_normalization_is_contextual_and_cannot_define_resolution_domains():
    space = uv_graph.MarkerSpace()
    expr = space.marker('extra == "Some_Feature"')
    space.seal()
    assert space.mask(expr) == 0
    assert space.mask(expr, 'some-feature') == space.all
    with pytest.raises(ValueError, match='Extras cannot'):
        space.marker('extra == "Some_Feature"', resolution=True)


def test_partition_and_syntax_budgets_fail_closed_not_truncate(monkeypatch):
    monkeypatch.setattr(uv_graph, 'MAX_CELLS', 3)
    with pytest.raises(ValueError, match='partition budget'):
        masks('sys_platform == "linux" or sys_platform == "win32" or sys_platform == "emscripten"')
    monkeypatch.setattr(uv_graph, 'MAX_MARKER_LENGTH', 8)
    with pytest.raises(ValueError, match='syntax budget'):
        masks('sys_platform == "linux"')


def test_predicate_budget_and_sealed_registration_are_enforced(monkeypatch):
    monkeypatch.setattr(uv_graph, 'MAX_ATOMS', 1)
    with pytest.raises(ValueError, match='predicate budget'):
        masks('sys_platform == "linux"', 'sys_platform == "win32"')
    space = uv_graph.MarkerSpace()
    space.marker('sys_platform == "linux"')
    space.seal()
    with pytest.raises(ValueError, match='sealed'):
        space.marker('sys_platform == "win32"')


def test_direct_requirement_and_recorded_transitive_edge_must_both_exist():
    pair = graph(package(dependencies=[{'name': 'child'}]), package('child'))
    validate(pair)
    pair[0]['package'].pop()
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)
    pair = graph(package('unrelated'))
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)


@pytest.mark.parametrize('change', ['version', 'source', 'selector_version', 'selector_source'])
def test_incompatible_direct_and_explicit_transitive_selectors_are_rejected(change):
    pair = graph(package())
    if change == 'version':
        pair[0]['package'][0]['version'] = '0.1.0'
    elif change == 'source':
        pair[0]['package'][0]['source'] = {'url': 'https://example.invalid/unbound.whl'}
    else:
        pair[0]['package'].append(package('child', '2.0.0'))
        pair[0]['package'][0]['dependencies'] = [{'name': 'child', 'version': '2.0.0', 'source': REGISTRY.copy()}]
        edge = pair[0]['package'][0]['dependencies'][0]
        edge['version' if change == 'selector_version' else 'source'] = (
            '1.0.0' if change == 'selector_version' else {'registry': 'https://other.invalid/simple'})
    with pytest.raises(ValueError, match='Incompatible'):
        validate(pair)


def test_unqualified_transitive_edge_does_not_authorize_an_arbitrary_url_source():
    pair = graph(package(dependencies=[{'name': 'child'}]),
                 package('child', source={'url': 'https://example.invalid/child.whl'}))
    with pytest.raises(ValueError, match='Incompatible'):
        validate(pair)
    pair[1]['tool']['uv']['sources'] = {'child': {'url': 'https://example.invalid/child.whl'}}
    validate(pair)


def test_constraints_do_not_add_absent_packages_but_bind_every_present_branch():
    pair = graph(package(), constraints=('accelerate==0.0.0+synthetic', 'base>=1'))
    validate(pair)
    assert {record['name'] for record in pair[0]['package']} == {'base'}
    pair[1]['tool']['uv']['constraint-dependencies'].append('base>=2; sys_platform == "win32"')
    with pytest.raises(ValueError, match='constraint'):
        validate(pair)
    pair = graph(package(), constraints=('base[fast]>=1',))
    with pytest.raises(ValueError, match='extras in uv graph constraints'):
        validate(pair)


def test_applicable_requirement_coverage_is_relative_to_requires_python_not_the_host():
    pair = graph(package(**{'resolution-markers': ['python_full_version >= "3.98.123"']}),
        dependencies=('base>=1; python_full_version >= "3.98.123"',))
    validate(pair)
    pair[1]['dependencies'] = ['base>=1']
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)
    pair[1]['requires-python'] = pair[0]['requires-python'] = '>=3.98.123'
    validate(pair)


def test_missing_one_direct_or_transitive_universal_branch_cannot_be_hidden_by_other_version():
    low, high = 'python_full_version < "3.12"', 'python_full_version >= "3.12"'
    pair = graph(package('base', '1.0.0', **{'resolution-markers': [low]}),
                 package('base', '2.0.0', **{'resolution-markers': [high]}),
                 **{'resolution-markers': [low, high]})
    validate(pair)
    pair[0]['package'].pop()
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)
    pair = graph(package(dependencies=[{'name': 'child'}]),
                 package('child', **{'resolution-markers': ['sys_platform != "win32"']}))
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)


def test_top_level_resolution_markers_must_partition_the_complete_python_domain():
    pair = graph(package(), **{'resolution-markers': ['sys_platform != "win32"']})
    with pytest.raises(ValueError, match='Incomplete universal'):
        validate(pair)
    pair[0]['resolution-markers'].append('sys_platform == "win32"')
    validate(pair)
    pair[0]['resolution-markers'].append('sys_platform == "linux"')
    with pytest.raises(ValueError, match='overlapping'):
        validate(pair)


def test_duplicate_and_overlapping_package_instances_are_ambiguous_even_without_incoming_edges():
    pair = graph(package(), package('orphan'), package('orphan'))
    with pytest.raises(ValueError, match='Ambiguous duplicate'):
        validate(pair)
    pair[0]['package'][-1]['version'] = '2.0.0'
    with pytest.raises(ValueError, match='Ambiguous duplicate'):
        validate(pair)
    pair[0]['package'][-2]['resolution-markers'] = ['sys_platform != "win32"']
    pair[0]['package'][-1]['resolution-markers'] = ['sys_platform == "win32"']
    validate(pair)


def test_platform_inactive_orphan_versions_are_retained_without_filtering_the_input():
    pair = graph(package(), package('orphan', '0.1.0', **{
        'resolution-markers': ['python_full_version < "3.10" and sys_platform == "win32"']}))
    before = deepcopy(pair)
    validate(pair)
    assert pair == before
    assert [record['version'] for record in pair[0]['package']] == ['1.0.0', '0.1.0']


def test_selected_extras_propagate_dependencies_but_unused_extras_do_not_force_packages():
    base = package(**{'optional-dependencies': {'fast': [{'name': 'child'}], 'unused': [{'name': 'absent'}]}})
    pair = graph(base, package('child'), dependencies=('base[fast]>=1',))
    validate(pair)
    pair[0]['package'].pop()
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)
    pair[1]['dependencies'] = ['base>=1']
    validate(pair)
    pair[1]['dependencies'] = ['base[unknown]>=1']
    with pytest.raises(ValueError, match='extra is absent'):
        validate(pair)


def test_conditional_extra_only_requires_its_child_in_the_selected_branch():
    pair = graph(package(**{'optional-dependencies': {'fast': [{'name': 'child'}]}}),
                 package('child', **{'resolution-markers': ['sys_platform == "win32"']}),
                 dependencies=('base[fast]>=1; sys_platform == "win32"', 'base>=1; sys_platform != "win32"'))
    validate(pair)
    pair[1]['dependencies'] = ['base[fast]>=1']
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)


def test_extra_cycles_terminate_and_propagate_missing_descendants():
    pair = graph(package(**{'optional-dependencies': {'fast': [{'name': 'child', 'extra': ['fast']}]}}),
                 package('child', **{'optional-dependencies': {'fast': [{'name': 'base', 'extra': ['fast']}]}}),
                 dependencies=('base[fast]>=1',))
    validate(pair)
    pair[0]['package'][1]['optional-dependencies']['fast'].append({'name': 'missing'})
    with pytest.raises(ValueError, match='Missing package'):
        validate(pair)


def test_selected_extra_metadata_cannot_be_satisfied_by_an_orphan_without_an_edge():
    pair = graph(package(metadata={'provides-extras': ['fast'], 'requires-dist': [
        {'name': 'child', 'specifier': '>=1', 'marker': 'extra == "fast"'}]}),
        package('child'), dependencies=('base[fast]>=1',))
    with pytest.raises(ValueError, match='absent from uv graph edges'):
        validate(pair)
    pair[0]['package'][0]['optional-dependencies'] = {'fast': [{'name': 'child'}]}
    validate(pair)


def project_graph():
    lock, script_metadata = graph(package())
    lock['package'].append(package('synthetic-project', '0.0.0', source={'virtual': '.'},
        dependencies=[{'name': 'base'}], **{
            'dev-dependencies': {'dev': [{'name': 'dev-only'}]},
            'optional-dependencies': {'feature': [{'name': 'extra-only'}]}}))
    lock['package'].extend([package('dev-only'), package('extra-only')])
    metadata = {'project': {'name': 'synthetic-project', 'version': '0.0.0',
        'requires-python': '>=3.11', 'dependencies': ['base>=1'],
        'optional-dependencies': {'feature': ['extra-only>=1']}},
        'dependency-groups': {'dev': ['dev-only>=1']}, 'tool': script_metadata['tool']}
    return lock, metadata


def test_root_project_and_all_optional_dev_profiles_are_bound_to_actual_records_and_edges():
    pair = project_graph()
    validate(pair, project=True)
    for name in ['synthetic-project', 'base', 'dev-only', 'extra-only']:
        changed = deepcopy(pair)
        changed[0]['package'] = [record for record in changed[0]['package'] if record['name'] != name]
        with pytest.raises(ValueError):
            validate(changed, project=True)
    for table in ['dependencies', 'optional-dependencies', 'dev-dependencies']:
        changed = deepcopy(pair)
        changed[0]['package'][1].pop(table)
        with pytest.raises(ValueError):
            validate(changed, project=True)


def test_unknown_group_includes_workspace_dynamic_and_resolver_forms_fail_closed():
    pair = project_graph()
    pair[1]['dependency-groups']['dev'] = [{'include-group': 'other'}]
    with pytest.raises(ValueError):
        validate(pair, project=True)
    pair = project_graph()
    pair[1]['project']['dynamic'] = ['dependencies']
    with pytest.raises(ValueError, match='dynamic/workspace'):
        validate(pair, project=True)
    pair = project_graph()
    pair[1]['tool']['uv']['environments'] = ['sys_platform == "linux"']
    with pytest.raises(ValueError, match='resolver/source'):
        validate(pair, project=True)


@pytest.mark.parametrize('entry', [
    {'name': 'child', 'version': '1.0.0'}, {'name': 'child', 'source': REGISTRY},
    {'name': 'child', 'extras': ['fast']}, {'name': 'child', 'extra': 'fast'},
    {'name': 'child', 'extra': ['fast', 'Fast']}, {'name': 'child', 'marker': ''},
    {'name': 'child', 'marker': 'implementation_version >= "3.12"'}, {}, [], 'child', None,
])
def test_malformed_edges_are_refused_even_when_the_parent_is_an_orphan(entry):
    pair = graph(package(), package('orphan', dependencies=[entry]))
    with pytest.raises(ValueError):
        validate(pair)


@pytest.mark.parametrize('source', [None, {}, [], {'url': 'https://example.invalid/a', 'registry': 'https://pypi.org/simple'},
    {'git': 'https://example.invalid/unproven'}, {'editable': '.'}, {'path': '../outside.whl'},
    {'path': './alias.whl'}, {'url': 'file:///synthetic.whl'}, {'url': 'https://example.invalid/a#hash=unbound'},
    {'registry': 'https://user:synthetic-secret@example.invalid/simple'}])
def test_unproven_source_forms_fail_without_echoing_source_secrets(source):
    with pytest.raises(ValueError) as error:
        validate(graph(package(source=source)))
    assert 'synthetic-secret' not in str(error.value)
    assert 'https://' not in str(error.value)


def test_package_resolution_markers_cannot_depend_on_unbound_extras():
    with pytest.raises(ValueError, match='Extras cannot'):
        validate(graph(package(**{'resolution-markers': ['extra != "fast"']})))


def test_nonredundant_overrides_are_not_used_to_silently_bypass_current_requirements():
    pair = graph(package(), dependencies=('base>=2',))
    pair[1]['tool']['uv']['override-dependencies'] = ['base==1.0.0']
    with pytest.raises(ValueError, match='Incompatible'):
        validate(pair)
    pair[1]['dependencies'] = ['base>=1']
    validate(pair)
    pair[1]['tool']['uv']['override-dependencies'] = ['base==1.0.0; sys_platform == "linux"']
    with pytest.raises(ValueError, match='conditional'):
        validate(pair)


def test_unbound_extra_markers_cannot_hide_direct_requirements_or_constraints():
    pair = graph(package('unrelated'), dependencies=('base; extra == "fast"',))
    with pytest.raises(ValueError, match='Unbound extra'):
        validate(pair)
    pair = graph(package(), constraints=('base>=2; extra == "fast"',))
    with pytest.raises(ValueError, match='Unbound extra'):
        validate(pair)


def test_extra_markers_on_resolved_edges_are_explicitly_unsupported_not_ignored():
    pair = graph(package(metadata={'provides-extras': ['fast']},
        dependencies=[{'name': 'missing', 'marker': 'extra == "fast"'}]),
        dependencies=('base[fast]>=1',))
    with pytest.raises(ValueError, match='extra-conditioned'):
        validate(pair)
