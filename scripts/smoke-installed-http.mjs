// Native runtime smoke: deliberately no Vitest/Bun-test compatibility layer.
import assert from 'node:assert/strict';
import { mkdirSync, realpathSync, writeFileSync } from 'node:fs';
import { connect } from 'node:net';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const installed = realpathSync(process.argv[2]);
const output = process.argv[3];
assert(output && process.env.CI === 'true', 'Run through the isolated package smoke');
const workspace = path.dirname(realpathSync(process.env.HOME));
assert(path.basename(workspace).startsWith('qmd-package-smoke-') && installed.startsWith(workspace + path.sep), 'Only a disposable installed package may be tested');
const { createStore } = await import(pathToFileURL(path.join(installed, 'dist/index.js')).href);
const { startMcpHttpServer } = await import(pathToFileURL(path.join(installed, 'dist/mcp/server.js')).href);
const docs = path.join(workspace, 'http-docs');
mkdirSync(docs);
writeFileSync(path.join(docs, 'transactions.md'), '# Transactions\n\nAtomicity ensures that a database transaction commits all its writes or none.\n');
writeFileSync(path.join(docs, 'gardening.md'), '# Gardening\n\nBasil needs sunlight and well-drained soil.\n');
const report = { ok: false, runtime: globalThis.Bun ? `bun ${Bun.version}` : `node ${process.version}`, checks: [] };
let store, server;
try {
  store = await createStore({ dbPath: path.join(workspace, 'http-shadow.sqlite'), config: { collections: { fixture: { path: docs, pattern: '**/*.md' } } } });
  const update = await store.update();
  assert.equal(update.indexed, 2);
  const lexical = await store.search({ queries: [{ type: 'lex', query: 'atomicity' }], rerank: false });
  assert(lexical.some(row => row.displayPath.endsWith('transactions.md')));
  report.checks.push('installed SDK/SQLite indexing and lexical retrieval');
  if (globalThis.Bun) {
    await assert.rejects(() => startMcpHttpServer(0, { store, quiet: true }), /requires Node\.js >=22/);
    report.httpSupported = false;
    report.checks.push("native Bun HTTP fails closed with explicit Node.js remedy (not HTTP feature acceptance)");
    report.ok = true;
  } else {
  report.httpSupported = true;
  server = await startMcpHttpServer(0, { store, quiet: true, bodyTimeoutMs: 80, requestTimeoutMs: 2000, sseKeepAliveMs: 25 });
  const base = `http://localhost:${server.port}`;
  const searchBody = JSON.stringify({ searches: [{ type: 'lex', query: 'atomicity' }], rerank: false });
  const rest = await fetch(base + '/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: searchBody, signal: AbortSignal.timeout(3000) });
  assert(rest.ok); assert(JSON.stringify(await rest.json()).includes('transactions.md'));
  const denied = await fetch(base + '/query', { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://untrusted.example' }, body: searchBody, signal: AbortSignal.timeout(3000) });
  assert.equal(denied.status, 403); await denied.text();
  report.checks.push('installed REST retrieval and foreign-origin rejection');

  const headers = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
  const init = await fetch(base + '/mcp', { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'native-package-smoke', version: '1' } } }), signal: AbortSignal.timeout(3000) });
  assert(init.ok); assert((await init.json()).result);
  headers['mcp-session-id'] = init.headers.get('mcp-session-id'); assert(headers['mcp-session-id']);
  const notification = await fetch(base + '/mcp', { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }), signal: AbortSignal.timeout(3000) });
  assert(notification.ok); await notification.text();
  const get = await fetch(base + '/mcp', { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'get', arguments: { file: 'qmd://fixture/transactions.md' } } }), signal: AbortSignal.timeout(3000) });
  assert(get.ok); const document = await get.json();
  assert(!document.error && !document.result?.isError && JSON.stringify(document).includes('Atomicity'));
  report.checks.push('installed MCP initialize and document retrieval');
  const controller = new AbortController();
  const sseDeadline = setTimeout(() => controller.abort(), 2000);
  try {
    const sse = await fetch(base + '/mcp', { headers: { ...headers, Accept: 'text/event-stream' }, signal: controller.signal });
    assert(sse.ok && sse.headers.get('content-type').includes('text/event-stream'));
    const reader = sse.body.getReader(); const first = await reader.read();
    assert(!first.done && first.value.length > 0); await reader.cancel();
  } finally { clearTimeout(sseDeadline); controller.abort(); }
  report.checks.push('native fetch consumes SSE without complete-body buffering');

  // A client that never finishes its body must not hold the connection forever.
  const started = performance.now();
  await new Promise((resolve, reject) => {
    const socket = connect({ host: 'localhost', port: server.port });
    const timer = setTimeout(() => { socket.destroy(); reject(new Error('Partial upload did not terminate within 1500ms')); }, 1500);
    let text = '';
    socket.on('data', chunk => { text += chunk.toString(); if (text.length > 8192) { socket.destroy(); reject(new Error('Unexpected unbounded timeout response')); } });
    socket.once('connect', () => socket.write(`POST /query HTTP/1.1\r\nHost: localhost:${server.port}\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{`));
    socket.once('error', error => { if (error.code !== 'ECONNRESET') { clearTimeout(timer); reject(error); } });
    socket.once('close', () => {
      clearTimeout(timer);
      try { assert(text === '' || /^HTTP\/1\.1 408\b/.test(text), text); resolve(); }
      catch (error) { reject(error); }
    });
  });
  report.partialUploadMs = performance.now() - started;
  report.checks.push('partial upload bounded; 408 or fail-closed connection termination');
  report.ok = true;
  }
} catch (error) {
  report.error = error?.stack || String(error);
} finally {
  const errors = [];
  try { await server?.stop(); } catch (error) { errors.push(String(error)); }
  try { await store?.close(); } catch (error) { errors.push(String(error)); }
  if (errors.length) { report.ok = false; report.cleanupErrors = errors; }
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(report, null, 2));
}
if (!report.ok) process.exitCode = 1;
