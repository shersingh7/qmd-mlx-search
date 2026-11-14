#!/usr/bin/env python3
"""Read-only source binding for an explicitly reviewed Accelerate patch policy.

The optional policy is docs/security/dependency-remediation/patched-dependencies.json:
``schema_version: 1`` and exactly one ``accelerate`` object with ``advisory``,
``version``, ``wheel_path``, ``sha256``, ``commit``, ``immutable_url`` and
``runner_paths``. Both path fields are canonical, repository-relative POSIX
paths. runner_paths lists every dependency-bearing finetune PEP 723 script;
empty-dependency scripts do not install Accelerate and are not forced to do so.

A version must be a canonical stable three-component PEP 440 local version.
Project sources use the wheel path relative to finetune; script sources use the
exact commit-pinned publication URL without the policy's identity fragment.
Every script has a checked-in .py.lock with matching metadata and wheel hashes;
the adjacent static uv graph contract also proves requirement/edge coverage
through every admitted universal marker branch (not just the current host).
uv run --locked enforces those hashes, not URL fragments. This module never
resolves dependencies, imports applications, runs wheel code,
installs anything, accesses the network, or writes an approval policy.

This proves source binding, NOT that a patch fixes an advisory, that a review
approved it, or that native Linux CI passed. Policy publication is a separate
approval gate. Missing policy returns no proof and status ``not_configured``;
any present but invalid policy or incomplete binding raises ValueError.
"""
from __future__ import annotations

from email import policy as email_policy
from email.parser import BytesParser
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import tomllib
from typing import Literal, overload
from urllib.parse import quote
import zipfile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

# Resolve our exact adjacent helper, not an ambient package or a mutable
# sys.path entry. Both the public CLI and read-only spec imports use this path.
_graph_spec = importlib.util.spec_from_file_location(
    '_qmd_uv_lock_graph', Path(__file__).with_name('uv_lock_graph.py')
)
if _graph_spec is None or _graph_spec.loader is None:
    raise ImportError('The required uv graph verifier is unavailable')
_graph = importlib.util.module_from_spec(_graph_spec)
_graph_spec.loader.exec_module(_graph)

POLICY_PATH = 'docs/security/dependency-remediation/patched-dependencies.json'
UV_LOCK_PATH = 'finetune/uv.lock'
ADVISORY = 'GHSA-4j2p-28q2-5m79'
MAX_WHEEL_BYTES = 25 * 1024 * 1024
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_LOCK_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_INVENTORY_ENTRIES = 20000
PRUNED_DIRECTORIES = frozenset({'vendor', '__pycache__', 'cache', 'caches', 'node_modules'})
PATCH_FIELDS = frozenset({
    'advisory', 'version', 'wheel_path', 'sha256', 'commit', 'immutable_url', 'runner_paths',
})


def _relative(value: object) -> str:
    # Validate before Path/PurePath can silently normalize aliases such as a//b.
    if (not isinstance(value, str) or not value or '\\' in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or any(part in {'', '.', '..'} for part in value.split('/'))):
        raise ValueError('Expected a canonical confined repository-relative path')
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?', value):
        raise ValueError('Invalid dependency name in source-binding inputs')
    return re.sub(r'[-_.]+', '-', value).lower()


class _RepositoryReader:
    """Anchor reads at a directory FD; refuse links, special files and growth.

    O_NOFOLLOW applies to every relative component, O_NONBLOCK avoids FIFO
    hangs, and fstat validates the descriptor actually read. Files with multiple
    hardlinks are rejected too. These are bounded per-file reads, not an atomic
    snapshot of a concurrently changing repository.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve(strict=True)
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.close(self.fd)

    def directory(self, relative: str) -> int:
        descriptor = os.dup(self.fd)
        try:
            for part in _relative(relative).split('/'):
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @overload
    def read(self, relative: str, limit: int, *, missing_ok: Literal[False] = False) -> bytes: ...

    @overload
    def read(self, relative: str, limit: int, *, missing_ok: Literal[True]) -> bytes | None: ...

    def read(self, relative: str, limit: int, *, missing_ok=False) -> bytes | None:
        parts = _relative(relative).split('/')
        directory = None
        descriptor = None
        try:
            directory = self.directory('/'.join(parts[:-1])) if len(parts) > 1 else os.dup(self.fd)
            # Refuse known FIFOs/devices before open, then verify the opened FD
            # as well: a replacement between stat/open must not hang or pass.
            metadata = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError('Source-binding input must be a regular file without link aliases')
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                 dir_fd=directory)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError('Source-binding input must be a regular file without link aliases')
            if before.st_size > limit:
                raise ValueError('Source-binding input exceeds the bounded read budget')
            body = bytearray()
            while True:
                chunk = os.read(descriptor, min(65536, limit + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > limit:
                    raise ValueError('Source-binding input exceeds the bounded read budget')
            after = os.fstat(descriptor)
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                                     item.st_ctime_ns, item.st_nlink)
            if identity(before) != identity(after) or len(body) != before.st_size:
                raise ValueError('Source-binding input changed while being read')
            return bytes(body)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise ValueError('Required source-binding input is missing') from None
        except OSError:
            raise ValueError('Cannot safely read a source-binding input') from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if directory is not None:
                os.close(directory)

    def python_files(self):
        pending = ['finetune']
        inspected = 0
        while pending:
            relative = pending.pop()
            descriptor = self.directory(relative)
            try:
                children = []
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        inspected += 1
                        if inspected > MAX_INVENTORY_ENTRIES:
                            raise ValueError('Runner inventory exceeds the bounded traversal budget')
                        if entry.name.startswith('.') or entry.name in PRUNED_DIRECTORIES:
                            continue
                        children.append((entry.name, entry.stat(follow_symlinks=False)))
                children.sort(key=lambda item: item[0])
            finally:
                os.close(descriptor)
            for name, metadata in children:
                child = _relative(relative + '/' + name)
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError('Runner inventory may not contain link aliases')
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(child)
                elif name.endswith('.py'):
                    yield child


def _unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key in patch policy')
        result[key] = value
    return result


def _invalid_constant(_):
    raise ValueError('Non-JSON numeric constant in patch policy')


def _patch_policy(raw: bytes) -> dict:
    try:
        document = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_mapping,
                              parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError('Malformed patch policy JSON') from None
    if (not isinstance(document, dict) or set(document) != {'schema_version', 'accelerate'}
            or type(document['schema_version']) is not int or document['schema_version'] != 1):
        raise ValueError('Unsupported patch policy schema')
    patch = document['accelerate']
    if not isinstance(patch, dict) or set(patch) != PATCH_FIELDS:
        raise ValueError('Incomplete or unsupported Accelerate patch policy')
    if patch['advisory'] != ADVISORY:
        raise ValueError('Accelerate patch policy advisory mismatch')
    version = patch['version']
    if not isinstance(version, str) or not re.fullmatch(
        r'(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\+[a-z0-9]+(?:\.[a-z0-9]+)*',
        version,
    ):
        raise ValueError('Patch policy requires a canonical stable local version')
    try:
        if str(Version(version)) != version:
            raise ValueError('Patch policy version is not canonical')
    except InvalidVersion:
        raise ValueError('Patch policy version is invalid') from None
    for field, length in (('sha256', 64), ('commit', 40)):
        if not isinstance(patch[field], str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', patch[field]):
            raise ValueError('Patch policy requires lowercase full-length digest and commit values')
    wheel_path = _relative(patch['wheel_path'])
    if wheel_path != f'finetune/vendor/accelerate/accelerate-{version}-py3-none-any.whl':
        raise ValueError('Patch policy wheel path does not match the reviewed package version')
    expected_url = (
        'https://raw.githubusercontent.com/shersingh7/qmd-mlx-search/'
        f'{patch["commit"]}/{quote(wheel_path, safe="/")}#sha256={patch["sha256"]}'
    )
    if not isinstance(patch['immutable_url'], str) or patch['immutable_url'] != expected_url:
        raise ValueError('Patch policy publication must be the exact immutable hash-bound source')
    runners = patch['runner_paths']
    if not isinstance(runners, list) or not runners:
        raise ValueError('Patch policy must inventory every dependency-bearing runner')
    seen = set()
    for runner in runners:
        path = _relative(runner)
        parts = path.split('/')
        if (not path.startswith('finetune/') or not path.endswith('.py')
                or any(part.startswith('.') or part in PRUNED_DIRECTORIES for part in parts)
                or path in seen):
            raise ValueError('Patch policy contains an invalid or duplicate runner path')
        seen.add(path)
    return patch


def _toml(raw: bytes) -> dict:
    try:
        return tomllib.loads(raw.decode('utf-8'))
    except (UnicodeError, ValueError, RecursionError):
        # Parser excerpts can contain credentials in malicious source URLs.
        raise ValueError('Malformed TOML in source-binding input') from None


def _table(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f'Missing or malformed {label} table')
    return value


def _pins(requirements, version: str, *, required: bool) -> None:
    if not isinstance(requirements, list):
        raise ValueError('Dependency declarations must be requirement lists')
    count = 0
    for text in requirements:
        if not isinstance(text, str):
            raise ValueError('Dependency declarations must contain requirement strings')
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            raise ValueError('Malformed dependency requirement in source-binding input') from None
        if _name(requirement.name) != 'accelerate':
            continue
        count += 1
        pins = list(requirement.specifier)
        if (requirement.marker is not None or requirement.url is not None or requirement.extras
                or len(pins) != 1 or pins[0].operator != '==' or pins[0].version != version):
            raise ValueError('Accelerate requires an unconditional exact reviewed pin')
    if count > 1 or (required and count != 1):
        raise ValueError('Accelerate pin is missing or duplicated')


def _uv_binding(document: dict, patch: dict, *, project=False) -> None:
    version = patch['version']
    dependencies = _table(document.get('project'), 'project').get('dependencies') if project else document.get('dependencies')
    _pins(dependencies, version, required=project)
    uv = _table(_table(document.get('tool'), 'tool').get('uv'), 'tool.uv')
    _pins(uv.get('constraint-dependencies'), version, required=True)
    _pins(uv.get('override-dependencies', []), version, required=False)
    sources = _table(uv.get('sources'), 'tool.uv.sources')
    keys = [key for key in sources if _name(key) == 'accelerate']
    expected = ({'path': patch['wheel_path'].removeprefix('finetune/')} if project
                else {'url': patch['immutable_url'].partition('#')[0]})
    if keys != ['accelerate'] or sources['accelerate'] != expected:
        raise ValueError('Accelerate source is missing, ambiguous or not the exact reviewed binding')


def _script_metadata(text: str) -> dict | None:
    """Strictly parse comment metadata only; never import a runner or its tools."""
    text = text.replace('\r\n', '\n')
    if any(char in text for char in '\r\v\f\x1c\x1d\x1e\x85\u2028\u2029'):
        raise ValueError('Ambiguous line separators in PEP 723 runner source')
    body = []
    active = False
    complete = False
    for line in text.split('\n'):
        if re.match(r'^\s*#\s*///', line):
            if line == '# /// script' and not active and not complete:
                active = True
            elif line == '# ///' and active:
                active = False
                complete = True
            else:
                raise ValueError('Malformed, duplicate or unsupported PEP 723 metadata block')
        elif active:
            if line == '#':
                body.append('')
            elif line.startswith('# '):
                body.append(line[2:])
            else:
                raise ValueError('Malformed PEP 723 comment metadata')
    if active:
        raise ValueError('Unterminated PEP 723 metadata block')
    if not complete:
        return None
    document = _toml('\n'.join(body).encode('utf-8'))
    dependencies = document.get('dependencies', [])
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise ValueError('PEP 723 dependencies must be a list of requirement strings')
    return document


def _wheel_binding(raw: bytes, patch: dict) -> None:
    if sha256(raw).hexdigest() != patch['sha256']:
        raise ValueError('Reviewed wheel SHA-256 does not match the actual artifact')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(entries) > 10000 or len(set(names)) != len(names):
                raise ValueError('Wheel ZIP contains duplicate entries or exceeds its entry budget')
            metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
            expected = f'accelerate-{patch["version"]}.dist-info/METADATA'
            if metadata_names != [expected]:
                raise ValueError('Wheel must contain exactly the expected Accelerate METADATA')
            entry = archive.getinfo(expected)
            file_type = stat.S_IFMT(entry.external_attr >> 16)
            if (entry.is_dir() or entry.flag_bits & 1 or file_type not in {0, stat.S_IFREG}
                    or entry.file_size > MAX_METADATA_BYTES):
                raise ValueError('Wheel METADATA must be a bounded regular unencrypted file')
            with archive.open(entry) as source:
                metadata_bytes = source.read(MAX_METADATA_BYTES + 1)
            if len(metadata_bytes) > MAX_METADATA_BYTES:
                raise ValueError('Wheel METADATA exceeds the bounded read budget')
            metadata_bytes.decode('utf-8')
            metadata = BytesParser(policy=email_policy.default).parsebytes(metadata_bytes)
            if (metadata.defects or metadata.get_all('Name') != ['accelerate']
                    or metadata.get_all('Version') != [patch['version']]):
                raise ValueError('Actual wheel METADATA name or version does not match policy')
    except (zipfile.BadZipFile, UnicodeError, RuntimeError, NotImplementedError, OSError):
        raise ValueError('Reviewed wheel is not a readable bounded wheel ZIP') from None


def _lock_packages(document: dict) -> list[dict]:
    packages = document.get('package')
    if (type(document.get('version')) is not int or document['version'] != 1
            or not isinstance(packages, list) or not packages):
        raise ValueError('Missing or unsupported uv package graph')
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError('Malformed uv package entry')
        _name(package.get('name'))
        try:
            if not isinstance(package.get('version'), str):
                raise ValueError('Missing uv package version')
            Version(package['version'])
        except InvalidVersion:
            raise ValueError('Malformed uv package version') from None
    return packages


def read_dependency_locks(root: Path) -> dict[str, dict]:
    """Collect all policy-owned uv graphs through the bounded no-follow reader.

    Old --repo snapshots without a policy still need only finetune/uv.lock.
    A configured policy makes every listed script lock mandatory, even when a
    data-only graph has no Accelerate instance. No application is imported.
    """
    with _RepositoryReader(root) as reader:
        raw = reader.read(POLICY_PATH, MAX_TEXT_BYTES, missing_ok=True)
        paths = [UV_LOCK_PATH]
        if raw is not None:
            patch = _patch_policy(raw)
            paths.extend(runner + '.lock' for runner in patch['runner_paths'])
        graphs = {}
        for path in paths:
            graph = _toml(reader.read(path, MAX_LOCK_BYTES))
            _lock_packages(graph)
            graphs[path] = graph
        return graphs


def _requirement_key(requirement: Requirement) -> tuple:
    return (_name(requirement.name), tuple(sorted(_name(extra) for extra in requirement.extras)),
            str(requirement.marker) if requirement.marker else '', requirement.url or '',
            str(requirement.specifier))


def _manifest_key(entry: object) -> tuple:
    if (not isinstance(entry, dict) or not {'name'} <= set(entry)
            or set(entry) - {'name', 'extras', 'marker', 'url', 'specifier'}):
        raise ValueError('Malformed standalone lock manifest requirement')
    name = _name(entry['name'])
    extras = entry.get('extras', [])
    if (not isinstance(extras, list) or any(not isinstance(extra, str) for extra in extras)
            or len({_name(extra) for extra in extras}) != len(extras)):
        raise ValueError('Malformed standalone lock manifest extras')
    if any(not isinstance(entry[key], str) or not entry[key] for key in entry if key not in {'name', 'extras'}):
        raise ValueError('Malformed standalone lock manifest requirement fields')
    if 'url' in entry and 'specifier' in entry:
        raise ValueError('Ambiguous standalone lock manifest source')
    text = name + ('[' + ','.join(extras) + ']' if extras else '')
    text += ' @ ' + entry['url'] if 'url' in entry else entry.get('specifier', '')
    text += ' ; ' + entry['marker'] if 'marker' in entry else ''
    try:
        return _requirement_key(Requirement(text))
    except InvalidRequirement:
        raise ValueError('Malformed standalone lock manifest requirement') from None


def _script_lock_metadata(document: dict, metadata: dict, patch: dict) -> None:
    """Check the input manifest, not just whether an old graph has a good wheel.

    Compare parsed requirements (including extras/markers/constraints/overrides)
    because uv normalizes names and specifier ordering in its manifest. Unknown
    standalone resolver/source forms fail closed rather than approximating uv.
    CI must additionally run real `uv lock --script ... --check --no-build`.
    Neither metadata equality nor uv --check proves graph completeness: the
    independent static graph contract below must pass before any patch proof.
    """
    if (type(document.get('revision')) is not int or document['revision'] != 3
            or not isinstance(metadata.get('requires-python'), str)
            or document.get('requires-python') != metadata['requires-python']):
        raise ValueError('Standalone lock revision or Python metadata is stale or unsupported')
    try:
        SpecifierSet(metadata['requires-python'])
    except ValueError:
        raise ValueError('Malformed standalone Python metadata') from None
    manifest = _table(document.get('manifest'), 'standalone lock manifest')
    if set(manifest) - {'requirements', 'constraints', 'overrides'}:
        raise ValueError('Unsupported standalone lock manifest fields')
    uv = metadata['tool']['uv']
    if (set(uv) - {'constraint-dependencies', 'override-dependencies', 'sources'}
            or set(uv['sources']) != {'accelerate'}):
        raise ValueError('Unsupported standalone resolver/source metadata')
    pairs = (('requirements', metadata['dependencies']),
             ('constraints', uv['constraint-dependencies']),
             ('overrides', uv.get('override-dependencies', [])))
    url = patch['immutable_url'].partition('#')[0]
    for field, requirements in pairs:
        entries = manifest.get(field, [])
        if not isinstance(entries, list):
            raise ValueError('Malformed standalone lock manifest list')
        actual = [_manifest_key(entry) for entry in entries]
        expected = []
        for text in requirements:
            try:
                requirement = Requirement(text)
                # uv substitutes tool.uv.sources in both direct and constraint
                # entries. _pins already checked the original exact version.
                if _name(requirement.name) == 'accelerate':
                    requirement = Requirement('accelerate @ ' + url)
                expected.append(_requirement_key(requirement))
            except InvalidRequirement:
                raise ValueError('Malformed standalone requirement metadata') from None
        if (len(set(actual)) != len(actual) or len(set(expected)) != len(expected)
                or sorted(actual) != sorted(expected)):
            raise ValueError('Standalone lock manifest does not match current script metadata')


def _lock_bindings(document: dict, patch: dict, *, script=False, required=True) -> dict[str, dict]:
    packages = _lock_packages(document)
    matches = {}
    url = patch['immutable_url'].partition('#')[0]
    expected_source = ({'url': url} if script
                       else {'path': patch['wheel_path'].removeprefix('finetune/')})
    artifact = ({'url': url} if script else {'filename': patch['wheel_path'].rsplit('/', 1)[1]})
    expected_wheels = [{**artifact, 'hash': 'sha256:' + patch['sha256']}]
    for index, package in enumerate(packages):
        dependencies = package.get('dependencies', [])
        if not isinstance(dependencies, list):
            raise ValueError('Malformed uv dependency edges')
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                raise ValueError('Malformed uv dependency edge')
            if _name(dependency.get('name')) == 'accelerate':
                required = True
        if _name(package['name']) != 'accelerate':
            continue
        if (package['version'] != patch['version'] or package.get('source') != expected_source
                or package.get('wheels') != expected_wheels or 'sdist' in package):
            raise ValueError('Every resolved Accelerate instance must use the reviewed wheel and hash')
        matches[str(index)] = package
    if required and not matches:
        raise ValueError('Reviewed Accelerate artifact is not in the uv graph')
    return matches


def _record_proofs(records: list, packages: dict[tuple[str, str], dict], patch: dict):
    if not isinstance(records, list):
        raise ValueError('Resolved dependency records must be a list')
    matched = set()
    sources = {source for source, _ in packages}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Malformed resolved dependency record')
        if record.get('ecosystem') != 'pip':
            continue
        if _name(record.get('name')) != 'accelerate':
            continue
        if (_name(record.get('original_name')) != 'accelerate'
                or any(not isinstance(record.get(key), str) for key in ('version', 'source', 'locator'))):
            raise ValueError('Malformed resolved Accelerate record')
        key = (record['source'], record['locator'])
        if record['source'] not in sources:
            raise ValueError('Resolved Accelerate record has an unbound source')
        if (not re.fullmatch(r'0|[1-9][0-9]*', record['locator']) or key not in packages
                or key in matched or record['version'] != patch['version']
                or record['original_name'] != packages[key]['name']):
            raise ValueError('Resolved Accelerate records do not exactly match the inspected uv graphs')
        matched.add(key)
    if matched != set(packages):
        raise ValueError('Resolved records omit an inspected Accelerate instance')
    return {(patch['version'], source, locator) for source, locator in matched}


def verify_accelerate_binding(root: Path, records: list) -> tuple[set[tuple[str, str, str]], dict]:
    """Return instance-specific proof keys and a URL-free source-binding receipt.

    No proof is returned until the entire configured project/runner/lock closure
    passes. Every root/script instance must be collected exactly once; extra
    unbound instances fail the entire closure without reflecting source URLs.
    """
    receipt = {'status': 'not_configured', 'scope': 'source_binding_only',
               'policy_path': POLICY_PATH, 'advisory': ADVISORY, 'resolved_instances': 0}
    try:
        with _RepositoryReader(root) as reader:
            raw_policy = reader.read(POLICY_PATH, MAX_TEXT_BYTES, missing_ok=True)
            if raw_policy is None:
                return set(), receipt
            patch = _patch_policy(raw_policy)
            _wheel_binding(reader.read(patch['wheel_path'], MAX_WHEEL_BYTES), patch)
            project = _toml(reader.read('finetune/pyproject.toml', MAX_TEXT_BYTES))
            _uv_binding(project, patch, project=True)
            project_lock = _toml(reader.read(UV_LOCK_PATH, MAX_LOCK_BYTES))
            _graph.validate_lock_graph(project_lock, project, project=True)
            packages = {(UV_LOCK_PATH, locator): package for locator, package in
                        _lock_bindings(project_lock, patch).items()}
            runners = {}
            for path in reader.python_files():
                text = reader.read(path, MAX_TEXT_BYTES).decode('utf-8')
                metadata = _script_metadata(text)
                if metadata is not None and metadata.get('dependencies'):
                    runners[path] = metadata
            if set(runners) != set(patch['runner_paths']):
                raise ValueError('Actual dependency-bearing runner inventory differs from patch policy')
            lock_hashes = {}
            for path, metadata in runners.items():
                _uv_binding(metadata, patch)
                lock_path = path + '.lock'
                raw_lock = reader.read(lock_path, MAX_LOCK_BYTES)
                lock = _toml(raw_lock)
                _script_lock_metadata(lock, metadata, patch)
                _graph.validate_lock_graph(lock, metadata)
                direct = any(_name(Requirement(text).name) == 'accelerate' for text in metadata['dependencies'])
                packages.update({(lock_path, locator): package for locator, package in
                                 _lock_bindings(lock, patch, script=True, required=direct).items()})
                lock_hashes[lock_path] = sha256(raw_lock).hexdigest()
            proofs = _record_proofs(records, packages, patch)
    except (OSError, UnicodeError, RecursionError):
        raise ValueError('Cannot safely inspect the complete source-binding closure') from None
    return proofs, {**receipt, 'status': 'verified_source_binding', 'version': patch['version'],
                    'wheel_path': patch['wheel_path'], 'sha256': patch['sha256'], 'commit': patch['commit'],
                    'runner_paths': sorted(runners), 'script_lock_paths': sorted(lock_hashes),
                    'script_lock_sha256': dict(sorted(lock_hashes.items())),
                    'resolved_instances': len(proofs), 'unbound_instances': 0}
