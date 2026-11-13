// Read-only held-out cancellation probes; owned synthetic listeners only.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { writeFileSync } from 'node:fs';
import { startMcpHttpServer } from '/Users/shersingh/github/qmd-mlx-search/src/mcp/server.ts';

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const headers = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
const report = [];
async function until(check, ms = 1500) {
  const deadline = performance.now() + ms;
  while (!check() && performance.now() < deadline) await delay(5);
  assert(check(), 'fixture did not reach expected state');
}

async function scenario(kind) {
  const operations = [];
  const cleanup = new AbortController();
  let backendRequests = 0, backendClosed = 0;
  const backend = createServer((_req, res) => {
    backendRequests++;
    res.on('close', () => backendClosed++);
  });
  await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve));
  const backendPort = backend.address().port;
  const store = {
    getDefaultCollectionNames: async () => [],
    getStatus: async () => ({ totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] }),
    listContexts: async () => [], getGlobalContext: async () => undefined,
    search: async ({ signal }) => {
      assert(signal, 'missing cancellation signal');
      const operation = { signal, settled: false };
      operations.push(operation);
      try {
        const response = await fetch(`http://127.0.0.1:${backendPort}/owned-work`, { signal: AbortSignal.any([signal, cleanup.signal]) });
        await response.text();
        return [];
      } finally { operation.settled = true; }
    },
    close: async () => {},
  };
  let handle;
  const requests = [];
  try {
    handle = await startMcpHttpServer(0, { store, quiet: true, requestTimeoutMs: 3000 });
    const base = `http://localhost:${handle.port}`;
    const init = await fetch(base + '/mcp', { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', id: 'init', method: 'initialize', params: { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'quality-heldout', version: '1' } } }) });
    assert.equal(init.status, 200); await init.text();
    const session = { ...headers, 'mcp-session-id': init.headers.get('mcp-session-id') };
    function startWork(id, query) {
      const request = fetch(base + '/mcp', { method: 'POST', headers: session, body: JSON.stringify({ jsonrpc: '2.0', id, method: 'tools/call', params: { name: 'query', arguments: { searches: [{ type: 'lex', query }], rerank: false } } }), signal: AbortSignal.timeout(5000) }).then(async response => ({ status: response.status, body: await response.text() }), error => ({ error: error.message }));
      requests.push(request);
    }
    if (kind === 'valid-id-zero-notification-cancel') {
      startWork(0, 'synthetic');
      await until(() => backendRequests === 1);
      const cancel = await fetch(base + '/mcp', { method: 'POST', headers: session, body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/cancelled', params: { requestId: 0, reason: 'quality held-out' } }) });
      assert.equal(cancel.status, 202); await cancel.text();
      await delay(200);
      report.push({ kind, cancellationStatus: cancel.status, signalAbortedBeforeStop: operations[0].signal.aborted, operationSettledBeforeStop: operations[0].settled, backendRequests, backendClosedBeforeStop: backendClosed });
    } else {
      startWork('duplicate', 'synthetic first');
      await until(() => backendRequests === 1);
      startWork('duplicate', 'synthetic second');
      await until(() => backendRequests === 2);
      await handle.stop();
      await delay(100);
      report.push({ kind, signalsAbortedAfterStop: operations.map(operation => operation.signal.aborted), operationsSettledAfterStop: operations.map(operation => operation.settled), backendRequests, backendClosedAfterStop: backendClosed });
    }
  } finally {
    await handle?.stop();
    cleanup.abort();
    await Promise.all(requests);
    await until(() => operations.every(operation => operation.settled));
    const closed = new Promise(resolve => backend.close(resolve));
    backend.closeAllConnections();
    await closed;
  }
}
await scenario('valid-id-zero-notification-cancel');
await scenario('duplicate-active-ids-stop');
writeFileSync('/Users/shersingh/.hermes/cache/scratch/qmd-audit/quality-mcp-heldout-cancel-result.json', JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify(report, null, 2));
if (!report[0].signalAbortedBeforeStop || report[1].signalsAbortedAfterStop.some(aborted => !aborted)) process.exitCode = 1;
