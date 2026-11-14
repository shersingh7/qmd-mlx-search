#!/usr/bin/env -S node --import tsx
/** Real local MLX -> SDK/SQLite -> REST/MCP smoke on authored public documents. */
import assert from 'node:assert/strict';
import { mkdirSync, realpathSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { createStore, MlxEmbedClient } from '../src/index.js';
import { startMcpHttpServer } from '../src/mcp/server.js';

const { values } = parseArgs({ options: { url: { type: 'string' }, root: { type: 'string' }, output: { type: 'string' }, help: { type: 'boolean' } } });
if (values.help) {
  console.log('verify_local_stack.ts --url http://127.0.0.1:PORT --root OWNED_TMPDIR/qmd-real-* --output REPORT.json');
  process.exit(0);
}
assert(values.url && values.root && values.output, 'url, root and output are required');
const endpoint = new URL(values.url);
assert(endpoint.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(endpoint.hostname), 'Only owned loopback inference is allowed');
const root = realpathSync(values.root);
const temporary = realpathSync(tmpdir());
assert(root.startsWith(temporary + path.sep) && path.basename(root).startsWith('qmd-real-'), 'Root must be a dedicated managed temporary fixture');
const docs = path.join(root, 'docs');
mkdirSync(docs);
const bodies: Record<string, string> = {
  'raft.md': '# Raft consensus\n\nRaft elects a leader and replicates a log across servers. A majority quorum agrees on committed entries in a distributed consensus cluster.\n',
  'transactions.md': '# SQLite transactions\n\nA database transaction groups writes atomically. Readers must not see uncommitted changes. Commit makes changes durable; rollback restores the previous logical state.\n',
  'pooling.md': '# Connection pooling\n\nA connection pool reuses database connections rather than opening a new connection for every request. A bounded pool limits simultaneous database work.\n',
  'origins.md': '# Browser origin protection\n\nA localhost API checks Host and Origin headers to block requests from untrusted web pages. Foreign browser origins must not read local documents.\n',
  'deadlines.md': '# Request deadlines\n\nAn absolute request deadline stops slow network operations, even when a peer sends occasional bytes. Cancelling a request releases pending work without pretending that a running native kernel has finished.\n',
  'eviction.md': '# Model weight eviction\n\nUnload unused MLX model weights after idle time to free unified memory. Keep operation leases until active inference completes so a model is not disposed while the GPU uses it.\n',
  'spaces.md': '# Embedding identity\n\nIndex and query vectors must use the same model, revision, pooling and dimensions. Different models can have equally sized vectors but incompatible semantic spaces. Reject identity changes rather than mixing vectors.\n',
  'gardening.md': '# Balcony gardening\n\nWater basil and mint regularly. Provide sunlight and soil drainage. Inspect the leaves for pests and harvest herbs for vegetarian cooking.\n',
};
for (const [file, body] of Object.entries(bodies)) writeFileSync(path.join(docs, file), body);
const config = { models: { embedBackend: 'mlx' as const, mlxUrl: endpoint.origin, mlxFallback: false }, collections: { public: { path: docs, pattern: '**/*.md' } } };
const client = new MlxEmbedClient({ url: endpoint.origin, timeoutMs: 15000, binary: true });
const report: Record<string, any> = { scope: 'Real local MiniLM inference and public authored fixture; no representative corpus-quality or long-run leak claim', passed: false, documents: Object.keys(bodies).length };
let store: Awaited<ReturnType<typeof createStore>> | undefined;
let restored: Awaited<ReturnType<typeof createStore>> | undefined;
let mcp: Awaited<ReturnType<typeof startMcpHttpServer>> | undefined;
try {
  const descriptor = await client.getDescriptor();
  assert(descriptor?.backend === 'mlx', 'A real MLX descriptor is required, not a fake adapter');
  report.descriptor = descriptor;
  const inputs = Object.values(bodies);
  const binary = await client.embedBatch(inputs);
  assert(binary.length === inputs.length && binary.every(v => v && v.embedding.length === descriptor.outputDimensions));
  const jsonClient = new MlxEmbedClient({ url: endpoint.origin, timeoutMs: 15000, binary: false });
  try {
    const json = await jsonClient.embedBatch(inputs);
    let maxDelta = 0;
    for (let i = 0; i < binary.length; i++) {
      assert(json[i]);
      for (let j = 0; j < descriptor.outputDimensions; j++) maxDelta = Math.max(maxDelta, Math.abs(binary[i]!.embedding[j]! - json[i]!.embedding[j]!));
    }
    assert(maxDelta < 1e-5, 'JSON/binary embeddings diverged');
    report.jsonBinaryMaxAbs = maxDelta;
  } finally { jsonClient.dispose(); }
  store = await createStore({ dbPath: path.join(root, 'shadow.sqlite'), config });
  report.update = await store.update();
  report.embed = await store.embed({ maxDocsPerBatch: 3 });
  assert(report.embed.status === 'complete' && report.embed.chunksCommitted >= inputs.length);
  report.noopEmbed = await store.embed();
  assert(report.noopEmbed.docsSelected === 0 && report.noopEmbed.chunksAttempted === 0, 'Unchanged corpus was selected again');
  const queries = [
    { query: 'distributed consensus chooses a leader and replicates logs', expected: 'raft.md' },
    { query: 'reuse database connections instead of opening them repeatedly', expected: 'pooling.md' },
    { query: 'block untrusted browser origins from localhost APIs', expected: 'origins.md' },
  ];
  report.queries = [];
  for (const query of queries) {
    const results = await store.searchVector(query.query, { limit: 3 });
    assert(results.length && results.every(r => Number.isFinite(r.score)));
    assert(results.some(r => r.displayPath.endsWith(query.expected)), `Expected ${query.expected} absent from top three`);
    report.queries.push({ ...query, results: results.map(r => ({ file: r.displayPath, score: r.score })) });
  }
  const hybrid = await store.search({ queries: [{ type: 'lex', query: 'Raft' }, { type: 'vec', query: queries[0]!.query }], rerank: false, limit: 3 });
  assert(hybrid.some(r => r.displayPath.endsWith('raft.md')));
  report.hybrid = hybrid.map(r => ({ file: r.displayPath, score: r.score }));
  const backup = path.join(root, 'restored.sqlite');
  store.internal.db.prepare('VACUUM INTO ?').run(backup);
  restored = await createStore({ dbPath: backup, config });
  for (let i = 0; i < queries.length; i++) {
    const after: { displayPath: string; score: number }[] = await restored.searchVector(queries[i]!.query, { limit: 3 });
    assert.deepEqual(after.map(r => ({ file: r.displayPath, score: r.score })), report.queries[i].results);
  }
  report.restoredRankingsIdentical = true;
  mcp = await startMcpHttpServer(0, { store, quiet: true, sseKeepAliveMs: 25 });
  const url = `http://localhost:${mcp.port}/mcp`;
  const headers: Record<string, string> = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
  const init = await fetch(url, { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'local-acceptance', version: '1' } } }), signal: AbortSignal.timeout(10000) });
  assert(init.ok); const initBody = await init.json() as any; assert(initBody.result);
  headers['mcp-session-id'] = init.headers.get('mcp-session-id')!; assert(headers['mcp-session-id']);
  const notify = await fetch(url, { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }), signal: AbortSignal.timeout(10000) });
  assert(notify.ok); await notify.text();
  async function rpc(id: number, method: string, params: unknown = {}) {
    const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', id, method, params }), signal: AbortSignal.timeout(15000) });
    assert(response.ok); const body = await response.json() as any; assert(!body.error, JSON.stringify(body.error)); return body.result;
  }
  const tools = await rpc(2, 'tools/list'); report.mcpTools = tools.tools.map((tool: any) => tool.name);
  assert(report.mcpTools.includes('query') && report.mcpTools.includes('get'));
  const result = await rpc(3, 'tools/call', { name: 'query', arguments: { searches: [{ type: 'vec', query: queries[0]!.query }], rerank: false, limit: 3 } });
  assert(!result.isError && JSON.stringify(result).includes('raft.md')); report.mcpQuery = result;
  const get = await rpc(4, 'tools/call', { name: 'get', arguments: { file: 'qmd://public/raft.md' } });
  assert(!get.isError && JSON.stringify(get).includes('majority quorum')); report.mcpGetVerified = true;
  const abort = new AbortController();
  const deadline = setTimeout(() => abort.abort(), 3000);
  try {
    const sse = await fetch(url, { headers: { ...headers, Accept: 'text/event-stream' }, signal: abort.signal });
    assert(sse.ok && sse.headers.get('content-type')?.includes('text/event-stream'));
    const reader = sse.body!.getReader();
    const frame = await reader.read(); assert(!frame.done && frame.value.length > 0);
    await reader.cancel(); report.sseStreamVerified = true;
  } finally { clearTimeout(deadline); abort.abort(); }
  const denied = await fetch(`http://localhost:${mcp.port}/query`, { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://untrusted.example' }, body: JSON.stringify({ searches: [{ type: 'lex', query: 'Raft' }] }), signal: AbortSignal.timeout(5000) });
  assert.equal(denied.status, 403); await denied.text(); report.foreignOriginRejected = true;
  report.passed = true;
} catch (error) {
  report.error = error instanceof Error ? error.stack : String(error);
} finally {
  const cleanupErrors: string[] = [];
  try { await mcp?.stop(); } catch (error) { cleanupErrors.push(String(error)); }
  for (const resource of [restored, store]) {
    try { await resource?.close(); } catch (error) { cleanupErrors.push(String(error)); }
  }
  try { client.dispose(); } catch (error) { cleanupErrors.push(String(error)); }
  if (cleanupErrors.length) { report.passed = false; report.cleanupErrors = cleanupErrors; }
  writeFileSync(values.output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ passed: report.passed, output: values.output, error: report.error }));
}
if (!report.passed) process.exitCode = 1;
