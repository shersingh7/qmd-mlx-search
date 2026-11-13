// Native-runtime MCP stdio integration against a quarantined installed tarball.
import assert from 'node:assert/strict';
import { realpathSync, writeFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import path from 'node:path';

const installed = realpathSync(process.argv[2]);
const output = process.argv[3];
const workspace = path.dirname(realpathSync(process.env.HOME));
assert(output && process.env.CI === 'true' && path.basename(workspace).startsWith('qmd-package-smoke-'));
assert(installed.startsWith(workspace + path.sep));
const runtime = process.execPath;
const child = spawn(runtime, [path.join(installed, 'dist/cli/qmd.js'), 'mcp'], {
  cwd: path.dirname(path.dirname(path.dirname(installed))), env: process.env,
  stdio: ['pipe', 'pipe', 'pipe'], detached: process.platform !== 'win32',
});
const report = { ok: false, runtime: globalThis.Bun ? `bun ${Bun.version}` : `node ${process.version}`, pid: child.pid, checks: [] };
const pending = new Map();
let stderr = '', stdoutBytes = 0;
child.stderr.on('data', data => { stderr = (stderr + data.toString()).slice(-8192); });
child.stdout.on('data', data => {
  stdoutBytes += data.length;
  if (stdoutBytes > 1024 * 1024) child.kill('SIGKILL');
});
const lines = createInterface({ input: child.stdout });
lines.on('line', line => {
  try {
    const body = JSON.parse(line); const waiter = pending.get(body.id);
    if (waiter) { pending.delete(body.id); clearTimeout(waiter.timer); waiter.resolve(body); }
  } catch (error) {
    for (const waiter of pending.values()) { clearTimeout(waiter.timer); waiter.reject(error); }
    pending.clear();
  }
});
child.on('error', error => { for (const waiter of pending.values()) { clearTimeout(waiter.timer); waiter.reject(error); } pending.clear(); });
child.on('exit', (code, signal) => {
  for (const waiter of pending.values()) { clearTimeout(waiter.timer); waiter.reject(new Error(`MCP exited ${code}/${signal}: ${stderr}`)); }
  pending.clear();
});
async function rpc(id, method, params = {}) {
  const result = new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`MCP ${method} timed out: ${stderr}`)); }, 4000);
    pending.set(id, { resolve, reject, timer });
  });
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  const response = await result; assert(!response.error, JSON.stringify(response.error)); return response.result;
}
const exited = new Promise(resolve => child.once('exit', resolve));
try {
  const init = await rpc(1, 'initialize', { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'native-stdio-smoke', version: '1' } });
  assert(init.serverInfo);
  child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');
  const result = await rpc(2, 'tools/list');
  report.tools = result.tools.map(tool => tool.name);
  assert(report.tools.includes('query') && report.tools.includes('get') && report.tools.includes('status'));
  const status = await rpc(3, 'tools/call', { name: 'status', arguments: {} });
  assert(!status.isError);
  report.checks.push('installed native MCP stdio initialize, tools/list and status');
  report.ok = true;
} catch (error) { report.error = error?.stack || String(error); }
finally {
  child.stdin.end();
  // The test owns this newly spawned process. Close it as an MCP client does;
  // never use a persistent PID file or target any unrelated process.
  let timer;
  await Promise.race([exited, new Promise(resolve => { timer = setTimeout(resolve, 250); })]);
  clearTimeout(timer);
  if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
  await Promise.race([exited, new Promise(resolve => { timer = setTimeout(resolve, 1500); })]);
  clearTimeout(timer);
  if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
  await Promise.race([exited, new Promise(resolve => { timer = setTimeout(resolve, 1500); })]);
  clearTimeout(timer);
  lines.close();
  report.childExited = child.exitCode !== null || child.signalCode !== null;
  if (!report.childExited) report.ok = false;
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(report, null, 2));
}
if (!report.ok) process.exitCode = 1;
