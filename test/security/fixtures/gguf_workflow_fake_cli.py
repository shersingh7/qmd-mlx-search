"""Credential-free SYNTHETIC CLI/lifetime fixture. Never executes a real CLI/probe."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

if sys.argv[1] == '--worker':
    root = Path(sys.argv[2])
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    fd = os.open(root / 'lifetime.fifo', os.O_WRONLY)
    os.write(fd, b'alive')
    (root / 'ready.json').write_text(json.dumps({
        'pid': os.getpid(), 'pgid': os.getpgrp(), 'leader_pid': os.getppid(),
        'started': time.monotonic(), 'bounded_seconds': 2.0, 'synthetic': True,
    }))
    time.sleep(2.0)
    (root / 'late.json').write_text(json.dumps({'completed': time.monotonic(), 'synthetic': True}))
    os.close(fd)
    raise SystemExit(0)

if sys.argv[1] == '--sentinel':
    root = Path(sys.argv[2])
    (root / 'sentinel-ready').write_text('ready')
    time.sleep(4.0)
    (root / 'sentinel-finished').write_text('naturally completed')
    raise SystemExit(0)

config = Path(sys.argv[1])
cfg = json.loads(config.read_text())
root = config.parent
argv = sys.argv[2:]
assert argv and argv[0] in ('git', 'docker'), 'fail closed: unexpected destination'
with (root / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps({'argv': argv, 'synthetic': True}) + '\n')
if argv[0] == 'git':
    assert argv[1:] == ['-C', cfg['workspace'], 'rev-parse', '--verify', 'HEAD']
    print('1' * 40)
    raise SystemExit(0)
assert argv[1] == '--config' and argv[3:5] == ['--host', 'unix:///var/run/docker.sock']
assert json.loads((Path(argv[2]) / 'config.json').read_text()) == {'auths': {}}
args = argv[5:]
state_path = root / 'state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {'present': False, 'removed': []}
cid, sentinel = 'a' * 64, 'b' * 64

def finish(value=None, code=0):
    state_path.write_text(json.dumps(state))
    if value is not None:
        print(json.dumps(value), flush=True)
    raise SystemExit(code)

if args[0] == 'version':
    finish({'Server': {'Version': 'SYNTHETIC', 'Os': 'linux', 'Arch': 'amd64'}})
elif args[0] == 'pull':
    assert args[1:3] == ['--platform', 'linux/amd64']
    state['image'] = args[3]
    finish()
elif args[:2] == ['image', 'inspect']:
    assert args[-1] == state['image']
    tagged, digest = state['image'].split('@')
    finish({'Id': 'sha256:' + 'c' * 64, 'Os': 'linux', 'Architecture': 'amd64',
            'RepoDigests': [tagged.rsplit(':', 1)[0] + '@' + digest]})
elif args[0] == 'run':
    assert args[-5:] == [state['image'], '-I', '-B', '-S', '/probe.py']
    assert '--native-only' not in args
    cidfile = Path(args[args.index('--cidfile') + 1])
    cidfile.write_text(cid + '\n')
    state.update(label=args[args.index('--label') + 1], present=cfg.get('descendant', False))
    state_path.write_text(json.dumps(state))
    if cfg.get('descendant'):
        # This real owned Python child stays in the real command/anchor group.
        subprocess.Popen([sys.executable, '-I', '-B', '-S', __file__, '--worker', str(root)])
        until = time.monotonic() + 1.0
        while not (root / 'ready.json').exists():
            assert time.monotonic() < until, 'fixture worker startup failed'
            time.sleep(0.005)
        if not cfg.get('early_exit'):
            time.sleep(4.0)
    if cfg.get('raw_signal'):
        signal.raise_signal(signal.SIGTERM)
    source = args[args.index('--mount') + 1].removeprefix('type=bind,src=').removesuffix(',dst=/probe.py,readonly')
    finish({'status': 'pass', 'native_only': False, 'platform': 'Linux/x86_64', 'python': '3.12.13',
            'versions': {tool: 'SYNTHETIC' for tool in ('git', 'cc', 'c++', 'make', 'uv')},
            'outputs': ['qmd-c-ok', 'qmd-cxx-ok'], 'probe_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'fixture_notice': 'SYNTHETIC: no Docker, probe or compiler execution'}, cfg.get('probe_exit', 0))
elif args[:2] == ['container', 'ls']:
    if cfg.get('storage_failure'):
        (Path(argv[2]).parent / 'receipt.json').mkdir(exist_ok=True)
    state_path.write_text(json.dumps(state))
    if cfg.get('cleanup_failure'):
        raise SystemExit(29)
    print(json.dumps(sentinel))
    if state['present']:
        print(json.dumps(cid))
    raise SystemExit(0)
elif args[:2] == ['container', 'inspect']:
    assert args[-1] == cid
    key, value = state['label'].split('=', 1)
    finish({'Id': cid, 'Labels': {key: value}})
elif args[:2] == ['container', 'rm']:
    assert args[2:] == ['--force', cid] and sentinel not in args
    state['removed'].append(cid)
    state['present'] = False
    finish()
raise AssertionError('fail closed: unknown synthetic CLI argv')
