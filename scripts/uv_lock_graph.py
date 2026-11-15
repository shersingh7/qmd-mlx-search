"""Bounded, read-only consistency proof for uv v1/revision-3 lock graphs.

This is NOT a resolver or a replacement for uv's freshness/artifact checks. It
connects current declarations to recorded packages and checks the recorded edge
closure, including all universal branches. It cannot recover upstream metadata
which a lock does not contain, prove wheel tag availability, or authenticate an
ordinary registry artifact. The separate reviewed-wheel verifier owns that proof.

Admitted marker language: and/or/parentheses; final Python language releases
(major, minor, micro are unbounded nonnegative integers), numeric comparisons,
prefix ==/!= and ~=; categorical ==/!= against non-version strings; and named
extras in package metadata/project optional declarations (resolved edges use
uv's explicit extra list and optional-dependencies tables, not extra markers).
python_version is the *same* language version projected to major/minor,
not an independent variable. Pre/dev/post/local Python marker bounds, string
ordering/membership, implementation_version, variable-to-variable comparisons,
and unknown resolver/source forms are explicitly unsupported. This is the final
release domain used by the admitted universal uv forks, not a claim about
pre-release interpreters. Categorical variables are independent: we conservatively
prove even combinations a real OS might not produce, never assume host facts.

The finite bitset is an EXACT quotient, not sampled Python/OS environments:
every numeric predicate is constant on intervals split at *all input bounds*,
including a final unbounded interval; every string equality is constant on each
mentioned value and one cofinite OTHER class. Thus set difference proves full
coverage for the admitted language. Complexity budgets refuse, never truncate.

All records (including inactive/orphan records) stay in the advisory collector.
Here every record's base edges are checked in its advertised resolution domain;
extras are activated only by actual requirements/edges, while every project
optional/dev group is checked as an installable profile. Constraints do not
create requirements. Unknown extras fail rather than silently discarding them.
"""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import product
from math import prod
import re
from urllib.parse import urlsplit

from packaging._parser import Op, Value, Variable
from packaging.markers import InvalidMarker, Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

MAX_CELLS = 65536
MAX_PACKAGES = 20000
MAX_EDGES = 100000
MAX_MARKERS = 4096
MAX_MARKER_LENGTH = 4096
MAX_MARKER_NODES = 512
MAX_ATOMS = 512
CATEGORICAL = frozenset({'sys_platform', 'os_name', 'platform_machine',
                         'platform_system', 'platform_python_implementation',
                         'implementation_name'})
PYTHON = frozenset({'python_version', 'python_full_version'})
ALL = ('all',)
ORIGIN = (0, 0, 0)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?', value):
        raise ValueError('Malformed uv graph dependency name')
    return re.sub(r'[-_.]+', '-', value).lower()


def _text(value):
    if (not isinstance(value, str) or not value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('Malformed uv graph text field')
    return value


def _list(value):
    if not isinstance(value, list):
        raise ValueError('Malformed uv graph requirement/edge list')
    return value


def _table(value):
    if not isinstance(value, dict):
        raise ValueError('Malformed uv graph table')
    return value


def _extras(values):
    result = tuple(sorted(_name(value) for value in _list(values)))
    if len(set(result)) != len(result):
        raise ValueError('Duplicate uv graph extra')
    return result


def _version(value):
    try:
        return Version(_text(value))
    except InvalidVersion:
        raise ValueError('Malformed uv graph package version') from None


def _source(value, *, virtual=False):
    value = _table(value)
    if len(value) != 1:
        raise ValueError('Missing or ambiguous uv graph source')
    key, text = next(iter(value.items()))
    text = _text(text)
    if key in {'registry', 'url'}:
        try:
            parsed = urlsplit(text)
            if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.fragment):
                raise ValueError('Unsupported uv graph source URL')
            parsed.port
        except ValueError:
            raise ValueError('Unsupported uv graph source URL') from None
    elif key == 'path':
        if ('\\' in text or text.startswith('/')
                or any(part in {'', '.', '..'} for part in text.split('/'))):
            raise ValueError('Unsupported uv graph source path')
    elif not (virtual and key == 'virtual' and text == '.'):
        raise ValueError('Unsupported uv graph source form')
    return (key, text)


def _python_ranges(variable, operator, value):
    """Return disjoint half-open intervals of final language-version triples."""
    match = re.fullmatch(r'(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?(?:\.(0|[1-9][0-9]*))?(\.\*)?', value)
    if not match:
        raise ValueError('Unsupported Python marker/specifier bound')
    parts = tuple(int(part) for part in match.groups()[:3] if part is not None)
    wildcard = match.group(4) is not None
    if variable == 'python_version' and (len(parts) > 2 or (len(parts) != 2 and not wildcard)):
        raise ValueError('Unsupported python_version projection bound')
    if wildcard and (operator not in {'==', '!='} or len(parts) > 2):
        raise ValueError('Unsupported Python wildcard marker')
    lower = parts + (0,) * (3 - len(parts))
    # A python_version equality denotes a whole minor, full-version equality
    # denotes one final micro release. No upper cap is placed on any component.
    width = len(parts) if wildcard else (2 if variable == 'python_version' else 3)
    upper = (*lower[:width - 1], lower[width - 1] + 1, *((0,) * (3 - width)))
    if operator == '==':
        return ((lower, upper),)
    if operator == '!=':
        return ((ORIGIN, lower), (upper, None))
    if operator == '<':
        return ((ORIGIN, lower),)
    if operator == '>=':
        return ((lower, None),)
    if operator == '<=':
        return ((ORIGIN, upper),)
    if operator == '>':
        return ((upper, None),)
    if operator == '~=' and len(parts) >= 2:
        compatible = (parts[0] + 1, 0, 0) if len(parts) == 2 else (parts[0], parts[1] + 1, 0)
        return ((lower, compatible),)
    raise ValueError('Unsupported Python marker operator')


class MarkerSpace:
    """Register predicates, seal the exact finite partition, then use bitsets."""

    def __init__(self):
        self.expressions = {}
        self.atoms = {}
        self.bounds: set[tuple[int, ...]] = {ORIGIN}
        self.categories = defaultdict(set)
        self.masks = {}
        self.cache = {}
        self.sealed = False

    def _atom(self, variable, operator, value):
        if self.sealed:
            raise ValueError('Cannot extend a sealed uv marker proof')
        if variable in PYTHON:
            definition = _python_ranges(variable, operator, value)
            self.bounds.update(point for interval in definition for point in interval if point is not None)
        elif variable in CATEGORICAL or variable == 'extra':
            if operator not in {'==', '!='}:
                raise ValueError('Unsupported categorical uv marker operator')
            # packaging switches numeric-looking strings to PEP 440 semantics.
            # Do not pretend those predicates are ordinary string equalities.
            try:
                Version(value)
            except InvalidVersion:
                pass
            else:
                raise ValueError('Version-like categorical uv markers are unsupported')
            _text(value) if variable != 'extra' else _name(value)
            if variable == 'extra':
                value = _name(value)
            else:
                self.categories[variable].add(value)
            definition = None
        else:
            raise ValueError('Unsupported uv marker variable')
        atom = (variable, operator, value)
        self.atoms[atom] = definition
        if len(self.atoms) > MAX_ATOMS:
            raise ValueError('uv marker proof exceeds its bounded predicate budget')
        return ('atom', atom)

    def marker(self, text=None, *, resolution=False):
        if text is None:
            return ALL
        _text(text)
        if text in self.expressions:
            expression = self.expressions[text]
            if resolution and self.has_extra(expression):
                raise ValueError('Extras cannot select uv graph resolution domains')
            return expression
        if len(text) > MAX_MARKER_LENGTH or len(self.expressions) >= MAX_MARKERS:
            raise ValueError('uv marker proof exceeds its bounded syntax budget')
        remaining = MAX_MARKER_NODES
        def parse(node):
            nonlocal remaining
            remaining -= 1
            if remaining < 0:
                raise ValueError('uv marker proof exceeds its bounded syntax budget')
            if isinstance(node, tuple) and len(node) == 3:
                left, op, right = node
                if not isinstance(op, Op):
                    raise ValueError('Unsupported uv marker parser form')
                operator = op.value
                if isinstance(left, Value) and isinstance(right, Variable):
                    # Wildcards on the literal-left side are not symmetric:
                    # swapping them changes uv's requirement activation domain.
                    # Refuse this spelling rather than certify a different atom.
                    if right.value in PYTHON and '*' in left.value:
                        raise ValueError('Unsupported literal-left Python wildcard marker')
                    left, right = right, left
                    operator = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '==': '==', '!=': '!='}.get(operator)
                if not (isinstance(left, Variable) and isinstance(right, Value) and operator is not None):
                    raise ValueError('Unsupported uv marker operands')
                return self._atom(left.value, operator, right.value)
            if not isinstance(node, list) or not node or len(node) % 2 != 1:
                raise ValueError('Unsupported uv marker parser form')
            groups = [[]]
            for index, item in enumerate(node):
                if index % 2 == 0:
                    groups[-1].append(parse(item))
                elif item == 'or':
                    groups.append([])
                elif item != 'and':
                    raise ValueError('Unsupported uv marker boolean operator')
            return ('or', tuple(('and', tuple(group)) for group in groups))
        try:
            # packaging has no public AST API. Admit its checked, typed AST
            # shape only; any future incompatible parser shape fails closed.
            expression = parse(Marker(text)._markers)
        except (InvalidMarker, RecursionError):
            raise ValueError('Malformed or unsupported uv graph marker') from None
        self.expressions[text] = expression
        if resolution and self.has_extra(expression):
            raise ValueError('Extras cannot select uv graph resolution domains')
        return expression

    def has_extra(self, expression):
        if expression[0] == 'atom':
            return expression[1][0] == 'extra'
        return expression[0] != 'all' and any(self.has_extra(child) for child in expression[1])

    def python(self, text):
        try:
            specs = SpecifierSet(_text(text))
        except InvalidSpecifier:
            raise ValueError('Malformed uv graph Python requirement') from None
        return ('and', tuple(self._atom('python_full_version', item.operator, item.version) for item in specs))

    def seal(self):
        if self.sealed:
            raise ValueError('uv marker proof already sealed')
        bounds = sorted(self.bounds)
        axes = sorted(self.categories)
        values = [tuple(sorted(self.categories[axis])) + (None,) for axis in axes]
        count = len(bounds) * prod(len(axis) for axis in values)
        if count > MAX_CELLS:
            raise ValueError('uv marker proof exceeds its bounded partition budget')
        self.all = (1 << count) - 1
        for atom, intervals in self.atoms.items():
            variable, operator, value = atom
            if variable == 'extra':
                continue
            mask = 0
            for index, cell in enumerate(product(bounds, *values)):
                if intervals is not None:
                    active = any(low <= cell[0] and (high is None or cell[0] < high) for low, high in intervals)
                else:
                    equal = cell[axes.index(variable) + 1] == value
                    active = equal if operator == '==' else not equal
                if active:
                    mask |= 1 << index
            self.masks[atom] = mask
        self.sealed = True

    def mask(self, expression, extra=''):
        if not self.sealed:
            raise ValueError('uv marker proof must be sealed before evaluation')
        key = (expression, extra)
        if key not in self.cache:
            kind = expression[0]
            if kind == 'all':
                result = self.all
            elif kind == 'atom':
                atom = expression[1]
                if atom[0] == 'extra':
                    equal = extra == atom[2]
                    result = self.all if equal == (atom[1] == '==') else 0
                else:
                    result = self.masks[atom]
            else:
                result = self.all if kind == 'and' else 0
                for child in expression[1]:
                    if kind == 'and':
                        result &= self.mask(child, extra)
                    else:
                        result |= self.mask(child, extra)
            self.cache[key] = result
        return self.cache[key]


class _Requirement:
    def __init__(self, name, specifier, source, extras, marker):
        self.name, self.specifier, self.source = name, specifier, source
        self.extras, self.marker = extras, marker


def _declaration(text, space, sources):
    try:
        req = Requirement(_text(text))
    except InvalidRequirement:
        raise ValueError('Malformed declared uv graph requirement') from None
    name = _name(req.name)
    source = _source({'url': req.url}) if req.url is not None else sources.get(name)
    if req.url is not None and name in sources and source != sources[name]:
        raise ValueError('Conflicting declared uv graph sources')
    return _Requirement(name, req.specifier, source, tuple(sorted(_name(extra) for extra in req.extras)),
                        space.marker(str(req.marker)) if req.marker is not None else ALL)


def _metadata_requirement(entry, space):
    entry = _table(entry)
    if not {'name'} <= set(entry) or set(entry) - {'name', 'specifier', 'url', 'path', 'extras', 'marker'}:
        raise ValueError('Unsupported uv graph metadata requirement')
    if len(set(entry) & {'url', 'path', 'specifier'}) > 1:
        raise ValueError('Ambiguous uv graph metadata requirement')
    try:
        spec = SpecifierSet(_text(entry['specifier'])) if 'specifier' in entry else SpecifierSet()
    except InvalidSpecifier:
        raise ValueError('Malformed uv graph metadata specifier') from None
    source = _source({key: entry[key]}) if (key := next((key for key in ('url', 'path') if key in entry), None)) else None
    return _Requirement(_name(entry.get('name')), spec, source, _extras(entry.get('extras', [])),
                        space.marker(entry.get('marker')))


def _groups(value, parse):
    groups = {}
    for name, entries in _table(value).items():
        key = _name(name)
        if key in groups:
            raise ValueError('Ambiguous uv graph group/extra aliases')
        groups[key] = [parse(entry) for entry in _list(entries)]
    return groups


class _Graph:
    def __init__(self, document, metadata, project):
        self.space = MarkerSpace()
        self.project = project
        self.pending_extras = deque()
        self.extra_domains = defaultdict(int)
        self.packages = []
        self.by_name = defaultdict(list)
        self.edge_count = 0
        if (set(document) - {'version', 'revision', 'requires-python', 'resolution-markers', 'manifest', 'package'}
                or type(document.get('version')) is not int or document['version'] != 1
                or type(document.get('revision')) is not int or document['revision'] != 3):
            raise ValueError('Unsupported uv graph schema/resolver options')
        project_data = _table(metadata.get('project')) if project else metadata
        if document.get('requires-python') != project_data.get('requires-python'):
            raise ValueError('uv graph Python metadata does not match current declarations')
        self.python = self.space.python(document.get('requires-python'))
        self.resolutions = [self.space.marker(_text(value), resolution=True)
                            for value in _list(document.get('resolution-markers', []))]
        if 'resolution-markers' in document and not self.resolutions:
            raise ValueError('Empty uv graph resolution partition')
        uv = _table(_table(metadata.get('tool', {})).get('uv', {}))
        if set(uv) - {'sources', 'constraint-dependencies', 'override-dependencies'}:
            raise ValueError('Unsupported uv graph resolver/source configuration')
        sources = {}
        for key, source in _table(uv.get('sources', {})).items():
            name = _name(key)
            if name in sources:
                raise ValueError('Ambiguous declared uv graph source aliases')
            sources[name] = _source(source)
        self.sources = sources
        parse = lambda text: _declaration(text, self.space, sources)
        self.direct = [parse(text) for text in _list(project_data.get('dependencies', []))]
        self.constraints = [parse(text) for text in _list(uv.get('constraint-dependencies', []))]
        if any(item.extras for item in self.constraints):
            raise ValueError('Unsupported extras in uv graph constraints')
        # Conditional/replacing resolver overrides are deliberately not
        # approximated. A redundant unconditional override is harmless; below
        # it must agree with every applicable declaration/edge requirement.
        self.overrides = [parse(text) for text in _list(uv.get('override-dependencies', []))]
        if any(item.marker != ALL or item.extras for item in self.overrides):
            raise ValueError('Unsupported conditional/extras uv graph override')
        if len({item.name for item in self.overrides}) != len(self.overrides):
            raise ValueError('Ambiguous uv graph overrides')
        self.constraints.extend(self.overrides)
        self.optional = _groups(project_data.get('optional-dependencies', {}), parse) if project else {}
        self.dev = _groups(metadata.get('dependency-groups', {}), parse) if project else {}
        unbound = self.direct + self.constraints + [item for group in self.dev.values() for item in group]
        if any(self.space.has_extra(item.marker) for item in unbound):
            raise ValueError('Unbound extra marker in uv graph declaration/constraint')
        if project and ('dynamic' in project_data or 'workspace' in uv):
            raise ValueError('Unsupported dynamic/workspace uv graph declarations')
        packages = _list(document.get('package'))
        if not packages or len(packages) > MAX_PACKAGES:
            raise ValueError('uv graph exceeds its bounded package budget or is empty')
        for record in packages:
            record = _table(record)
            if set(record) - {'name', 'version', 'source', 'resolution-markers', 'dependencies',
                              'optional-dependencies', 'dev-dependencies', 'metadata', 'wheels', 'sdist'}:
                raise ValueError('Unsupported uv graph package fields')
            package = {'name': _name(record.get('name')), 'version': _version(record.get('version')),
                       'source': _source(record.get('source'), virtual=project),
                       'resolution': [self.space.marker(_text(value), resolution=True)
                                      for value in _list(record.get('resolution-markers', []))]}
            if 'resolution-markers' in record and not package['resolution']:
                raise ValueError('Empty uv graph package resolution domain')
            package['edges'] = [self.edge(entry) for entry in _list(record.get('dependencies', []))]
            package['optional'] = _groups(record.get('optional-dependencies', {}), self.edge)
            package['dev'] = _groups(record.get('dev-dependencies', {}), self.edge)
            details = _table(record.get('metadata', {}))
            if set(details) - {'requires-dist', 'provides-extras', 'requires-dev'}:
                raise ValueError('Unsupported uv graph package metadata fields')
            parse_metadata = lambda entry: _metadata_requirement(entry, self.space)
            package['requirements'] = [parse_metadata(entry) for entry in _list(details.get('requires-dist', []))]
            package['requires_dev'] = _groups(details.get('requires-dev', {}), parse_metadata)
            if any(self.space.has_extra(item.marker) for group in package['requires_dev'].values() for item in group):
                raise ValueError('Unbound extra marker in uv graph dev metadata')
            package['extras'] = set(package['optional']) | set(_extras(details.get('provides-extras', [])))
            self.by_name[package['name']].append(len(self.packages))
            self.packages.append(package)
        self.space.seal()
        self.domain = self.space.mask(self.python)
        if not self.domain:
            raise ValueError('uv graph Python requirement has no admitted final release')
        seen = 0
        for expression in self.resolutions:
            mask = self.space.mask(expression) & self.domain
            if seen & mask:
                raise ValueError('Ambiguous overlapping uv graph resolution branches')
            seen |= mask
        if self.resolutions and seen != self.domain:
            raise ValueError('Incomplete universal uv graph resolution partition')
        for package in self.packages:
            domain = self.domain if not package['resolution'] else 0
            for expression in package['resolution']:
                domain |= self.space.mask(expression) & self.domain
            package['domain'] = domain
        for indices in self.by_name.values():
            seen = 0
            identities = set()
            for index in indices:
                package = self.packages[index]
                identity = (package['version'], package['source'], tuple(package['resolution']))
                if identity in identities or seen & package['domain']:
                    raise ValueError('Ambiguous duplicate/overlapping uv graph package records')
                identities.add(identity)
                seen |= package['domain']

    def edge(self, entry):
        self.edge_count += 1
        if self.edge_count > MAX_EDGES:
            raise ValueError('uv graph exceeds its bounded edge budget')
        entry = _table(entry)
        if not {'name'} <= set(entry) or set(entry) - {'name', 'version', 'source', 'marker', 'extra'}:
            raise ValueError('Malformed or unsupported uv graph edge')
        # uv's version/source selector pair identifies a fork exactly. Partial
        # selectors have no admitted meaning, even if a host resolver guesses.
        if ('version' in entry) != ('source' in entry):
            raise ValueError('Ambiguous partial uv graph edge selector')
        version = _version(entry['version']) if 'version' in entry else None
        marker = self.space.marker(entry.get('marker'))
        if self.space.has_extra(marker):
            raise ValueError('Unsupported extra-conditioned uv graph edge')
        return {'name': _name(entry.get('name')), 'version': version,
                'source': _source(entry['source']) if 'source' in entry else None,
                'extras': _extras(entry.get('extra', [])), 'marker': marker}

    def activate(self, index, extra, domain):
        if not domain:
            return
        if extra not in self.packages[index]['extras']:
            raise ValueError('Required uv graph extra is absent or cannot be proven')
        key = (index, extra)
        added = domain & ~self.extra_domains[key]
        if added:
            self.extra_domains[key] |= added
            self.pending_extras.append((index, extra, added))

    def targets(self, name, domain, *, specifier=None, source=None, version=None, extras=()):
        covered = 0
        for index in self.by_name.get(name, []):
            package = self.packages[index]
            applicable = package['domain'] & domain
            if not applicable:
                continue
            if (package['source'][0] == 'virtual'
                    or (version is not None and package['version'] != version)
                    or (specifier is not None and not specifier.contains(package['version'], prereleases=True))
                    or (source is not None and package['source'] != source)
                    or (source is None and specifier is not None and package['source'][0] != 'registry')):
                raise ValueError('Incompatible version/source in required uv graph branch')
            covered |= applicable
            for extra in extras:
                self.activate(index, extra, applicable)
        if domain & ~covered:
            raise ValueError('Missing package/marker coverage in required uv graph branch')

    def require(self, requirement, domain, *, extra='', edges=None):
        applicable = domain & self.space.mask(requirement.marker, extra)
        self.targets(requirement.name, applicable, specifier=requirement.specifier,
                     source=requirement.source, extras=requirement.extras)
        if edges is not None:
            covered = 0
            for edge in edges:
                if edge['name'] != requirement.name or not set(requirement.extras) <= set(edge['extras']):
                    continue
                covered |= self.space.mask(edge['marker'], extra)
            if applicable & ~covered:
                raise ValueError('Declared requirement is absent from uv graph edges')

    def follow(self, edge, domain, *, extra=''):
        # An omitted selector means the declared uv source for that name, or a
        # registry package, not permission to substitute an arbitrary URL/path.
        source = edge['source'] if edge['source'] is not None else self.sources.get(edge['name'])
        self.targets(edge['name'], domain & self.space.mask(edge['marker'], extra),
                     source=source, specifier=SpecifierSet(), version=edge['version'], extras=edge['extras'])

    def validate(self, metadata):
        root = None
        if self.project:
            project = metadata['project']
            candidates = [package for package in self.packages if package['source'] == ('virtual', '.')]
            if (len(candidates) != 1 or candidates[0]['name'] != _name(project.get('name'))
                    or candidates[0]['version'] != _version(project.get('version'))
                    or candidates[0]['domain'] != self.domain):
                raise ValueError('Missing/incompatible uv graph root project record')
            root = candidates[0]
            if set(root['dev']) != set(self.dev) or set(root['optional']) != set(self.optional):
                raise ValueError('uv graph project optional/dev groups do not match declarations')
        for requirement in self.direct:
            self.require(requirement, self.domain, edges=root['edges'] if root else None)
        for group, requirements in self.dev.items():
            assert root is not None  # dev groups are admitted only for projects
            for requirement in requirements:
                self.require(requirement, self.domain, edges=root['dev'][group])
        for group, requirements in self.optional.items():
            assert root is not None
            for requirement in requirements:
                self.require(requirement, self.domain, extra=group, edges=root['optional'][group])
        for index, package in enumerate(self.packages):
            domain = package['domain']
            # No reachability pruning: an orphan record is still part of the
            # certified lock, and every recorded base edge must be well formed.
            for edge in package['edges']:
                self.follow(edge, domain)
            for edges in package['dev'].values():
                for edge in edges:
                    self.follow(edge, domain)
            for requirement in package['requirements']:
                self.require(requirement, domain, edges=package['edges'])
            for group, requirements in package['requires_dev'].items():
                if group not in package['dev']:
                    raise ValueError('Missing uv graph metadata dev group')
                for requirement in requirements:
                    self.require(requirement, domain, edges=package['dev'][group])
            for constraint in self.constraints:
                if constraint.name != package['name']:
                    continue
                applicable = domain & self.space.mask(constraint.marker)
                if applicable and (not constraint.specifier.contains(package['version'], prereleases=True)
                        or (constraint.source is not None and constraint.source != package['source'])):
                    raise ValueError('Resolved uv graph package violates an applicable constraint/override')
            if package is root:
                for extra in self.optional:
                    self.activate(index, extra, domain)
        # Monotone bitset propagation terminates even for cycles/self-extras:
        # each (record, extra, partition-cell) is added at most once.
        while self.pending_extras:
            index, extra, domain = self.pending_extras.popleft()
            package = self.packages[index]
            edges = package['optional'].get(extra, [])
            for edge in edges:
                self.follow(edge, domain, extra=extra)
            for requirement in package['requirements']:
                self.require(requirement, domain, extra=extra, edges=package['edges'] + edges)


def validate_lock_graph(document: dict, metadata: dict, *, project=False) -> None:
    """Refuse any unproven requirement/edge branch without altering the graph.

    Inputs must already have been read with the caller's bounded safe reader.
    The caller separately compares stored standalone manifests to current inputs
    and verifies the exact reviewed wheel/source/hash policy. No patch proof may
    be granted until this graph contract passes for *every* configured lock.
    """
    try:
        graph = _Graph(_table(document), _table(metadata), project)
        graph.validate(metadata)
    except (RecursionError, OverflowError):
        raise ValueError('uv graph proof exceeds its bounded validation budget') from None
