"""Portable offline GGUF workflow regressions (stdlib unittest, POSIX only).

Run: python -m unittest discover -s test/security -p test_gguf_workflow_controller.py
The source-relative workflow/launcher are read, never imported. Only the exact
inline controller and its generated owner run; every CLI destination is replaced
at the actual command boundary by an owned Python fixture, with no fallback.
No real Docker, qualifier, launcher, compiler, registry or network execution.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import platform
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/gguf-native-image-qualification.yml'
FIXTURE = Path(__file__).parent / 'fixtures/gguf_workflow_fake_cli.py'


def controller_source(path):
    text = path.read_text()
    marker = '        run: |\n'
    assert text.count(marker) == 1
    block = text.split(marker, 1)[1]
    assert all(not line or line.startswith('          ') for line in block.splitlines())
    script = '\n'.join(line[10:] if line else '' for line in block.splitlines()) + '\n'
    return script.split("<<'PY'\n", 1)[1].rsplit('\nPY\n', 1)[0]


def isolated_env(root):
    env = {'PATH': str(root / 'denied'), 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'QMD_LLAMA_GPU': 'false', 'CUDA_VISIBLE_DEVICES': ''}
    for key, leaf in {'HOME': 'home', 'HERMES_HOME': 'hermes', 'HERMES_DIR': 'hermes',
                      'HERMES_KNOWLEDGE_DIR': 'knowledge', 'KNOWLEDGE_DIR': 'knowledge',
                      'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data',
                      'HF_HOME': 'hf', 'TMPDIR': 'tmp'}.items():
        path = root / leaf
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    return env


def fixture_driver():
    """Exec verbatim controller; only transport/platform are synthetic."""
    workflow, case = map(Path, sys.argv[2:4])
    cfg_path = case / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    source = controller_source(workflow)
    platform.system = lambda: 'Linux'
    platform.machine = lambda: 'x86_64'
    popen, killpg, selecting = subprocess.Popen, os.killpg, select.select
    audit, owners, signaled = [], {}, set()
    signal_requested = False

    def save(event):
        audit.append(event)
        (case / 'transport.json').write_text(json.dumps(audit) + '\n')

    def transport(argv, **kwargs):
        # No pass-through branch: a missed seam cannot reach Docker or git.
        assert argv[:4] == [sys.executable, '-I', '-B', '-S']
        assert Path(argv[4]) == case / 'owned/command-owner.py'
        assert kwargs['start_new_session'] is True
        assert kwargs['pass_fds'] == (int(argv[5]), int(argv[6]))
        original = argv[8:]
        assert original[0] in ('git', 'docker')
        probe = original[0] == 'docker' and original[5] == 'run'
        adapted = list(argv[:8])
        # This is the old suite's synthetic shortened wait at the new boundary.
        # Subtract the original duration, not the elapsed time: never restart it.
        if probe and cfg.get('short_budget'):
            adapted[7] = str(float(adapted[7]) - 180 + cfg['short_budget'])
        if probe and cfg.get('admission_delay'):
            time.sleep(cfg['admission_delay'])
        child = popen(adapted + [sys.executable, '-I', '-B', '-S', str(case / 'fake.py'), str(cfg_path), *original], **kwargs)
        owners[child.pid] = probe
        save({'start': original, 'anchor': child.pid, 'deadline': float(adapted[7])})
        wait = child.wait

        def guarded_wait(timeout=None):
            assert child.pid in signaled, 'anchor reaped before final group signal'
            result = wait(timeout=timeout)
            save({'reaped': child.pid, 'exit': result})
            if probe and cfg.get('reap_failure'):
                # Diagnostic seam: actual owned process already reaped, but
                # controller must fail closed if its wait reports incomplete.
                raise subprocess.TimeoutExpired(argv, timeout)
            return result

        child.wait = guarded_wait
        child.poll = lambda: (_ for _ in ()).throw(AssertionError('anchor polled before final group signal'))
        if probe and cfg.get('signal_during_registration'):
            until = time.monotonic() + 1
            while not (case / 'ready.json').exists():
                assert time.monotonic() < until
                time.sleep(0.005)
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGTERM)
        return child

    def group_signal(pgid, sig):
        assert pgid in owners and sig == signal.SIGKILL
        assert pgid not in signaled, 'no post-reap/repeated numeric group signal'
        signaled.add(pgid)
        save({'final_group_signal': pgid, 'signal': sig})
        if owners[pgid] and cfg.get('signal_failure'):
            raise PermissionError('SYNTHETIC signal delivery failure')
        return killpg(pgid, sig)

    def waiting(read, write, error, timeout=None):
        nonlocal signal_requested
        if cfg.get('interrupt') and not signal_requested and (case / 'ready.json').exists():
            signal_requested = True
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGINT)
        return selecting(read, write, error, timeout)

    subprocess.Popen, os.killpg, select.select = transport, group_signal, waiting
    sys.argv = ['-', str(case / 'owned')]
    save({'controller_body_sha256': hashlib.sha256(source.encode()).hexdigest()})
    exec(compile(source, str(workflow) + ':inline-python', 'exec'), {'__name__': '__main__'})


@unittest.skipUnless(os.name == 'posix', 'workflow requires POSIX process groups')
class WorkflowControllerTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get('GGUF_TEST_ARTIFACTS')
        if retained:
            base = Path(retained)
            base.mkdir(parents=True, exist_ok=True)
            self.case = base / self._testMethodName
            self.case.mkdir()
        else:
            temporary = tempfile.TemporaryDirectory(prefix='gguf-workflow-')
            self.addCleanup(temporary.cleanup)
            self.case = Path(temporary.name)
        self.env = isolated_env(self.case)
        tools = self.case / 'workspace/finetune/tools'
        tools.mkdir(parents=True)
        self.launcher = (ROOT / 'finetune/tools/submit_locked_job.py').read_bytes()
        (tools / 'submit_locked_job.py').write_bytes(self.launcher)
        (tools / 'qualify_gguf_image.py').write_bytes(b'# SYNTHETIC public probe data, never executed\n')
        self.tools = tools
        owned = self.case / 'owned'
        owned.mkdir()
        (owned / 'docker').mkdir()
        shutil.copyfile(FIXTURE, self.case / 'fake.py')
        self.env.update(QMD_WORKSPACE=str(tools.parent.parent), QMD_EVENT='pull_request',
                        QMD_CONTEXT_SHA='3' * 40, QMD_EVENT_HEAD_SHA='2' * 40,
                        QMD_RUNNER_OS='Linux', QMD_RUNNER_ARCH='X64', QMD_RUNNER_NAME='SYNTHETIC')

    def run_controller(self, cfg=None):
        cfg = dict(cfg or {}, workspace=str(self.tools.parent.parent))
        (self.case / 'config.json').write_text(json.dumps(cfg))
        # Even a missed transport cannot find git/docker: PATH is an empty owned
        # directory and transport rejects every unrecognized spawn before Popen.
        (self.case / 'denied').mkdir(exist_ok=True)
        return subprocess.Popen([sys.executable, '-I', '-B', '-S', str(Path(__file__).resolve()),
                                 '--fixture-driver', str(WORKFLOW), str(self.case)],
                                env=self.env, cwd=self.case, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def receipt(self, child, expected):
        stdout, stderr = child.communicate(timeout=10)
        self.returned = time.monotonic()
        (self.case / 'stdout.log').write_text(stdout)
        (self.case / 'stderr.log').write_text(stderr)
        self.assertEqual(child.returncode, expected, stderr)
        records = [json.loads(line) for line in stdout.splitlines()]
        receipts = [record['qualification_receipt'] for record in records if 'qualification_receipt' in record]
        self.assertEqual(len(receipts), 1)
        result = receipts[0]
        self.assertEqual(result['final_exit'], child.returncode)
        self.assertEqual(result['status'] == 'qualified-native-amd64-prerequisites', expected == 0)
        self.assertEqual(stderr, '')
        saved = self.case / 'owned/receipt.json'
        if result['receipt_persistence']['status'] == 'written':
            self.assertEqual(json.loads(saved.read_text()), result)
        else:
            self.assertTrue(saved.is_dir(), 'required receipt failure must remain observable')
        audit = json.loads((self.case / 'transport.json').read_text())
        calls = [json.loads(line)['argv'] for line in (self.case / 'calls.jsonl').read_text().splitlines()]
        starts = [event['start'] for event in audit if 'start' in event]
        cfg = json.loads((self.case / 'config.json').read_text())
        if cfg.get('admission_delay'):
            # This one command must NOT cross the actual execution boundary:
            # the helper receives an already-expired, not reset, deadline.
            self.assertEqual(starts[:-1], calls)
            self.assertEqual(starts[-1][0], 'docker')
            self.assertEqual(starts[-1][5], 'run')
        else:
            self.assertEqual(starts, calls, 'fake seam must be hit for every admitted CLI')
        self.assertEqual(audit[0]['controller_body_sha256'], hashlib.sha256(controller_source(WORKFLOW).encode()).hexdigest())
        return result, calls

    def binding_case(self, pattern, refused):
        with (self.tools / 'submit_locked_job.py').open('ab') as stream:
            stream.write(('\nmatch None:\n    case ' + pattern + ':\n        pass\n').encode())
        result, calls = self.receipt(self.run_controller(), 1 if refused else 0)
        if refused:
            self.assertIn('ambiguous/missing', result['error'])
            self.assertFalse(any(call[0] == 'docker' for call in calls), 'reject before pull')

    def test_receipt_success(self):
        result, _ = self.receipt(self.run_controller(), 0)
        self.assertEqual(result['primary_exit'], 0)

    def test_receipt_storage_failure(self):
        result, _ = self.receipt(self.run_controller({'storage_failure': True}), 1)
        self.assertEqual(result['primary_exit'], 0)
        self.assertEqual(result['receipt_persistence']['status'], 'failed')
        self.assertEqual([path.name for path in (self.case / 'owned').glob('receipt.*')], ['receipt.json'])

    def test_receipt_primary_and_storage_failure(self):
        result, _ = self.receipt(self.run_controller({'probe_exit': 37, 'storage_failure': True}), 37)
        self.assertEqual(result['primary_exit'], 37)
        self.assertEqual(result['receipt_persistence']['status'], 'failed')

    def test_receipt_raw_signal_and_storage_failure(self):
        result, _ = self.receipt(self.run_controller({'raw_signal': True, 'storage_failure': True}), 143)
        probe = next(item for item in result['commands'] if item['name'] == 'probe')
        self.assertEqual(probe['command_exit'], -signal.SIGTERM)
        self.assertEqual(probe['exit'], 143)
        self.assertEqual(result['primary_exit'], 143)

    def lifetime_case(self, cfg, expected, abrupt=False):
        os.mkfifo(self.case / 'lifetime.fifo')
        reader = os.open(self.case / 'lifetime.fifo', os.O_RDONLY | os.O_NONBLOCK)
        self.addCleanup(os.close, reader)
        sentinel = subprocess.Popen([sys.executable, '-I', '-B', '-S', str(self.case / 'fake.py'), '--sentinel', str(self.case)], env=self.env, cwd=self.case)
        # Direct owned Popen handle only, not a sampled PID; naturally drains.
        self.addCleanup(lambda: sentinel.wait(timeout=6))
        child = self.run_controller(dict({'descendant': True, 'short_budget': 0.3}, **cfg))
        if abrupt:
            until = time.monotonic() + 2
            while not (self.case / 'ready.json').exists():
                self.assertLess(time.monotonic(), until)
                time.sleep(0.005)
            child.kill()  # exact owned handle; helper must observe persistent EOF
            child.communicate(timeout=3)
            self.returned = time.monotonic()
            self.assertEqual(child.returncode, -signal.SIGKILL)
            result = None
        else:
            result, _ = self.receipt(child, expected)
        returned = self.returned
        ready = json.loads((self.case / 'ready.json').read_text())
        audit = json.loads((self.case / 'transport.json').read_text())
        anchor = next(item['anchor'] for item in audit if 'start' in item and item['start'][0] == 'docker' and item['start'][5] == 'run')
        self.assertEqual(ready['pgid'], anchor)
        self.assertNotEqual(ready['pid'], anchor)
        self.assertNotEqual(ready['leader_pid'], anchor, 'exercise separate CLI/anchor lifetimes')
        self.assertEqual(os.read(reader, 1024), b'alive', 'real worker must have held lifetime FD')
        try:
            terminated_at_return = os.read(reader, 1) == b''
        except BlockingIOError:
            terminated_at_return = False
        snapshot = {'ready': ready, 'returned': returned, 'terminated_at_return': terminated_at_return,
                    'late_at_return': (self.case / 'late.json').exists(), 'sentinel_alive_at_return': sentinel.poll() is None,
                    'sentinel_ready_at_return': (self.case / 'sentinel-ready').exists(),
                    'fixture_cleanup_started': False, 'descendant_reap_certified': False}
        (self.case / 'return-state.json').write_text(json.dumps(snapshot, indent=2))
        self.assertFalse(snapshot['late_at_return'])
        self.assertTrue(snapshot['sentinel_alive_at_return'])
        self.assertTrue(snapshot['sentinel_ready_at_return'])
        # Parent-approved positive oracle: do not require a SIGKILLed worker's
        # normal-completion marker. Observe the original 3s window before cleanup.
        until = returned + 3.0
        terminated = terminated_at_return
        while time.monotonic() < until:
            self.assertFalse((self.case / 'late.json').exists(), 'owned descendant performed late work')
            if not terminated:
                try:
                    terminated = os.read(reader, 1) == b''
                except BlockingIOError:
                    pass
            time.sleep(0.02)
        self.assertTrue(terminated, 'owned worker lifetime FD did not close')
        if result is not None:
            probe = next(item for item in result['commands'] if item['name'] == 'probe')
            self.assertTrue(probe['lifetime']['anchor_reaped'])
            if not (probe['lifetime']['signal_delivered'] or probe['lifetime'].get('group_absent_at_signal')):
                # Darwin can report EPERM for the already self-KILLed group.
                # Signal delivery is not invented from observed FD closure:
                # require an explicit failure and nonqualification instead.
                self.assertIn('error', probe['lifetime'])
                self.assertFalse(result['command_cleanup_ok'])
                self.assertNotEqual(result['final_exit'], 0)
            self.assertFalse(probe['lifetime']['descendant_reap_certified'])
            self.assertEqual(result['cleanup']['absence_verified'], not cfg.get('cleanup_failure', False))
        sentinel.wait(timeout=3)
        self.assertEqual(sentinel.returncode, 0)
        self.assertTrue((self.case / 'sentinel-finished').exists())
        (self.case / 'observation.json').write_text(json.dumps({'no_late_work': True, 'worker_terminated': terminated,
            'sentinel_exit': sentinel.returncode, 'observation_seconds': time.monotonic() - returned,
            'fixture_cleanup_started': False, 'all_descendants_reaped_claimed': False}, indent=2))

    def test_timeout_descendant(self):
        self.lifetime_case({}, 124)

    def test_early_success_descendant(self):
        self.lifetime_case({'early_exit': True}, 0)

    def test_early_failure_descendant(self):
        self.lifetime_case({'early_exit': True, 'probe_exit': 37}, 37)

    def test_interrupt_descendant(self):
        self.lifetime_case({'interrupt': True, 'short_budget': 0.8}, 143)

    def test_repeated_signals_during_registration(self):
        self.lifetime_case({'signal_during_registration': True, 'short_budget': 0.8}, 143)

    def test_timeout_with_failed_docker_cleanup(self):
        self.lifetime_case({'cleanup_failure': True}, 124)

    def test_controller_killed_pipe_revocation(self):
        self.lifetime_case({}, None, abrupt=True)

    def test_reap_failure_is_nonqualification(self):
        result, _ = self.receipt(self.run_controller({'reap_failure': True}), 1)
        self.assertFalse(result['command_cleanup_ok'])
        self.assertIn('anchor reap timed out', next(item for item in result['commands'] if item['name'] == 'probe')['lifetime']['error'])

    def test_signal_failure_is_nonqualification(self):
        result, _ = self.receipt(self.run_controller({'signal_failure': True, 'short_budget': 0.3}), 1)
        self.assertFalse(result['command_cleanup_ok'])
        self.assertIn('PermissionError', next(item for item in result['commands'] if item['name'] == 'probe')['lifetime']['error'])

    def test_startup_consumes_original_budget(self):
        result, calls = self.receipt(self.run_controller({'short_budget': 0.3, 'admission_delay': 0.5}), 124)
        self.assertFalse(any(call[0] == 'docker' and call[5] == 'run' for call in calls))
        self.assertEqual(result['probe_exit'], 124)
        self.assertFalse(result['cleanup']['absence_verified'])


# Individual unittest cases, not hidden subtest counts. ast.walk must cover nested
# patterns and every alternative; unrelated binders/literal keys stay harmless.
PATTERNS = {
    'capture': ('GGUF_IMAGE', True),
    'mapping_capture': ('{"image": GGUF_IMAGE}', True),
    'star_capture': ('[*GGUF_IMAGE]', True),
    'mapping_rest': ('{**GGUF_IMAGE}', True),
    'nested_capture': ('{"image": [GGUF_IMAGE]}', True),
    'nested_star': ('{"image": [*GGUF_IMAGE]}', True),
    'nested_rest': ('[{"image": _, **GGUF_IMAGE}]', True),
    'or_as_capture': ('(1 | 2) as GGUF_IMAGE', True),
    'or_nested_capture': ('[GGUF_IMAGE] | {"image": GGUF_IMAGE}', True),
    'unrelated_capture': ('other', False),
    'unrelated_star_rest': ('{"GGUF_IMAGE": [*other], **rest}', False),
    'unrelated_nested_or': ('[other] | {"image": other}', False),
}
for name, (pattern, refused) in PATTERNS.items():
    def _pattern_case(self, pattern=pattern, refused=refused):
        self.binding_case(pattern, refused)
    setattr(WorkflowControllerTests, 'test_pattern_' + name, _pattern_case)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--fixture-driver':
        fixture_driver()
    else:
        unittest.main()
