#!/usr/bin/env python3
"""Run a bounded, real local MLX/SDK/MCP smoke; never touch a live index.

Requires an already-cached qualified model and the repository's Node deps.
No synthetic inference or network model downloads are substituted on failure.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def integration_digest(root):
    files = [*root.joinpath('src').rglob('*.ts'), root/'package.json',
             root/'scripts/verify_local_stack.py', root/'scripts/verify_local_stack.ts']
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.relative_to(root).as_posix().encode() + b'\0')
        digest.update(file.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    args = parser.parse_args()
    if not args.model.is_absolute() or not (args.model/'config.json').is_file():
        parser.error('model must be an existing absolute local weights directory')
    output = args.output.resolve()
    sdk_output = output.with_name(output.stem + '-sdk.json')
    if output.exists() or sdk_output.exists():
        parser.error('Use a fresh output path; old evidence is preserved')
    temporary = os.environ.get('TMPDIR')
    if not temporary or not Path(temporary).is_dir():
        parser.error('TMPDIR must point to an existing managed scratch directory')
    fixture = Path(tempfile.mkdtemp(prefix='qmd-real-', dir=temporary))
    for name in ('home', 'cache', 'config'):
        (fixture/name).mkdir()
    for key in list(os.environ):
        if key.startswith(('QMD_', 'MLX_')):
            del os.environ[key]
    os.environ.update({'HOME': str(fixture/'home'), 'XDG_CACHE_HOME': str(fixture/'cache'),
                       'XDG_CONFIG_HOME': str(fixture/'config'), 'HF_HUB_OFFLINE': '1',
                       'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                       'TOKENIZERS_PARALLELISM': 'false'})
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import mlx.core as mx
    from scripts.qmd_mlx.server import start_server, ServerState
    from scripts.qmd_mlx.watchdog import spawn_owned_process, terminate_owned_process
    from bench_apple_silicon import source_digest
    digest = source_digest(root)
    integration_sha = integration_digest(root)
    report = {'passed': False, 'scope': 'Real cached MLX model -> JSON/binary -> SDK SQLite -> REST/MCP, authored public fixture only',
              'model': str(args.model), 'dtype_requested': args.dtype, 'source_sha256': digest, 'integration_source_sha256': integration_sha,
              'fixture': str(fixture), 'sdk_report': str(sdk_output)}
    server = child = init_thread = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not mx.metal.is_available():
            raise RuntimeError('Metal unavailable')
        mx.set_memory_limit(2 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)
        started = time.monotonic()
        server, init_thread = start_server(model_name=str(args.model), port=0, control_port=0,
                                           quantization=None, dtype_str=args.dtype, max_length=256)
        while server.state not in (ServerState.READY, ServerState.FAILED):
            if time.monotonic() - started > 30:
                raise TimeoutError('Model readiness exceeded 30 seconds')
            time.sleep(.05)
        if server.state != ServerState.READY:
            raise RuntimeError(server.state_error or 'Model initialization failed')
        report['ready_seconds'] = time.monotonic() - started
        url = f'http://127.0.0.1:{server.server_address[1]}'
        report['url'] = url
        report['control_port'] = server.control_port
        report['instance_token'] = server.instance_token
        node = shutil.which('node')
        if not node:
            raise RuntimeError('Node executable missing')
        command = [node, '--import', 'tsx', str(root/'scripts/verify_local_stack.ts'),
                   '--url', url, '--root', str(fixture), '--output', str(sdk_output)]
        report['command'] = command
        with (fixture/'node.log').open('w') as log:
            child = spawn_owned_process(command, cwd=root, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            report['child_pid'] = child.pid
            try:
                status = child.wait(timeout=90)
            finally:
                cleaned, sigkill_used = terminate_owned_process(child, grace_s=.2)
                report['child_cleanup'] = {'cleaned': cleaned, 'sigkill_used': sigkill_used}
                report['child_exit_code'] = child.returncode
                if not cleaned:
                    raise RuntimeError('Owned SDK worker group did not finish cleanup')
            report['node_log'] = str(fixture/'node.log')
        if status:
            raise RuntimeError(f'Actual SDK/MCP runner failed with exit {status}; see {fixture / "node.log"}')
        sdk_report = json.loads(sdk_output.read_text())
        if sdk_report.get('passed') is not True:
            raise AssertionError('SDK report did not pass')
        report['checks'] = {key: sdk_report.get(key) for key in ['documents','jsonBinaryMaxAbs','restoredRankingsIdentical','mcpGetVerified','sseStreamVerified','foreignOriginRejected','noopEmbed']}
        assert server.runtime is not None
        report['memory_before_stop'] = server.runtime.get_memory_info()
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if server is not None:
                server.stop()
                server.server_close()
                # start_server returns a coordinator that joins the HTTP loops;
                # its caller owns joining it before judging completed teardown.
                if init_thread is not None:
                    init_thread.join(timeout=5)
                report['server_stopped'] = server._stopped_event.is_set()
                report['worker_alive_after_stop'] = bool(server.executor and server.executor.is_worker_alive())
                report['http_threads_alive'] = any(thread is not None and thread.is_alive() for thread in (server.t_http, server.t_ctrl))
                if not report['server_stopped'] or report['worker_alive_after_stop'] or report['http_threads_alive']:
                    raise RuntimeError('Server did not fully stop')
            report['metal_active_immediate_after_stop'] = mx.get_active_memory()
            mx.synchronize()
            gc.collect()
            mx.clear_cache()
            report['metal_active_after_stop'] = mx.get_active_memory()
            report['metal_cache_after_stop'] = mx.get_cache_memory()
            if report['metal_active_after_stop'] != 0:
                raise RuntimeError('Active Metal allocations remain after stop')
            if source_digest(root) != digest or integration_digest(root) != integration_sha:
                raise RuntimeError('Source changed during verification; rerun settled code')
        except Exception as exc:
            report['passed'] = False
            report['cleanup_error'] = f'{type(exc).__name__}: {exc}'
        with output.open('x') as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write('\n')
        print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
