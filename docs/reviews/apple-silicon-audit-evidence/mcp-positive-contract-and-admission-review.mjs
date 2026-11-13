// Contract-correct independent review; only owned CPU loopback fixtures.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { connect } from 'node:net';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { startMcpHttpServer } from '/Users/shersingh/github/qmd-mlx-search/src/mcp/server.ts';

const source = '/Users/shersingh/github/qmd-mlx-search/src/mcp/server.ts';
const test = '/Users/shersingh/github/qmd-mlx-search/test/audit-mcp-request-ownership.test.ts';
const originalRed = '/Users/shersingh/.hermes/cache/scratch/qmd-audit/quality-mcp-heldout-cancel.mjs';
const reportPath = '/Users/shersingh/.hermes/cache/scratch/qmd-audit/mcp-positive-contract-and-admission-review-result.json';
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const before = Object.fromEntries([source, test, originalRed].map(path => [path, hash(path)]));
const headers = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, message, ms = 1500) {
  const end = performance.now() + ms;
  while (!check() && performance.now() < end) await delay(5);
  assert(check(), message);
}
const query = (id, text = 'synthetic') => ({ jsonrpc: '2.0', id, method: 'tools/call', params: {
  name: 'query', arguments: { searches: [{ type: 'lex', query: text }], rerank: false },
} });
async function fixture(options = {}) {
  const operations = [], responses = [], sockets = [], jobs = [];
  const teardown = new AbortController();
  let backendClosed = 0, storeClosed = 0;
  const backend = createServer((_req, res) => {
    responses.push(res);
    res.once('close', () => backendClosed++);
  });
  let handle;
  try {
    await new Promise((resolve, reject) => {
      backend.once('error', reject); backend.listen(0, '127.0.0.1', resolve);
    });
    const backendPort = backend.address().port;
    const store = {
      getDefaultCollectionNames: async () => [],
      getStatus: async () => ({ totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] }),
      listContexts: async () => [], getGlobalContext: async () => undefined,
      search: async ({ signal }) => {
        assert(signal, 'store search must receive an owned AbortSignal');
        const operation = { signal, settled: false };
        operations.push(operation);
        try {
          const response = await fetch(`http://127.0.0.1:${backendPort}/owned-search`, {
            signal: AbortSignal.any([signal, teardown.signal]),
          });
          await response.text();
          return [];
        } finally { operation.settled = true; }
      },
      close: async () => { storeClosed++; },
    };
    handle = await startMcpHttpServer(0, { store, quiet: true, requestTimeoutMs: 4000, ...options });
    const base = `http://localhost:${handle.port}`;
    function post(body, sid) {
      const job = (async () => {
        const response = await fetch(base + '/mcp', { method: 'POST', headers: {
          ...headers, ...(sid ? { 'mcp-session-id': sid } : {}),
        }, body: JSON.stringify(body), signal: AbortSignal.any([teardown.signal, AbortSignal.timeout(5000)]) });
        const text = await response.text();
        return { status: response.status, sid: response.headers.get('mcp-session-id'), body: text ? JSON.parse(text) : undefined };
      })();
      void job.catch(() => {}); jobs.push(job); return job;
    }
    async function init() {
      return post({ jsonrpc: '2.0', id: 'init', method: 'initialize', params: {
        protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'positive-contract-review', version: '1' },
      } });
    }
    async function initSid() {
      const result = await init(); assert.equal(result.status, 200); assert(result.sid); return result.sid;
    }
    function wire(body, sid, incomplete = false) {
      const text = typeof body === 'string' ? body : JSON.stringify(body);
      const length = incomplete ? 65536 : Buffer.byteLength(text);
      let received = '', error;
      const socket = connect(handle.port, 'localhost', () => socket.write(
        `POST /mcp HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nmcp-session-id: ${sid}\r\nContent-Length: ${length}\r\nConnection: close\r\n\r\n${text}`,
      ));
      sockets.push(socket);
      socket.on('data', chunk => { received += chunk.toString(); });
      socket.on('error', failure => { error = failure; });
      const closed = new Promise((resolve, reject) => {
        const timer = setTimeout(() => socket.destroy(new Error('Original socket failed to settle on its own')), 2500);
        socket.once('close', () => {
          clearTimeout(timer);
          if (error && error.code !== 'ECONNRESET') reject(error); else resolve(received);
        });
      });
      void closed.catch(() => {}); jobs.push(closed);
      return { socket, closed, received: () => received };
    }
    async function cancel(sid, id) {
      const result = await post({ jsonrpc: '2.0', method: 'notifications/cancelled', params: {
        requestId: id, reason: 'positive held-out contract',
      } }, sid);
      assert.equal(result.status, 202);
    }
    return { handle, post, init, initSid, wire, cancel, operations, responses,
      backendClosed: () => backendClosed,
      cleanup: async () => {
        await handle.stop();
        teardown.abort();
        for (const socket of sockets) socket.destroy();
        await Promise.allSettled(jobs);
        await until(() => operations.every(op => op.settled), 'teardown left unsettled owned operations');
        await new Promise(resolve => { backend.close(resolve); backend.closeAllConnections(); });
        assert.equal(storeClosed, 0, 'injected store must stay caller-owned');
        assert.equal(handle.httpServer.listening, false);
      },
    };
  } catch (error) {
    teardown.abort(); await handle?.stop();
    for (const socket of sockets) socket.destroy();
    await new Promise(resolve => { backend.close(resolve); backend.closeAllConnections(); });
    throw error;
  }
}

const report = { before, positiveContract: [], defects: [] };
try {
  for (const id of [0, '', 1, '1']) {
    const f = await fixture();
    try {
      const sid = await f.initSid();
      const original = f.wire(query(id, 'legitimate original'), sid);
      await until(() => f.responses.length === 1, 'original did not start its downstream HTTP request');
      const duplicate = await f.post(query(id, 'duplicate must not start'), sid);
      assert.equal(duplicate.status, 409);
      assert.equal(duplicate.body.error.code, -32600);
      assert.equal(f.operations.length, 1);
      assert.equal(f.operations[0].signal.aborted, false);
      await f.cancel(sid, id);
      await until(() => f.operations[0].settled && f.backendClosed() === 1, 'original did not abort and settle downstream');
      assert.equal(f.operations[0].signal.aborted, true);
      const wire = await original.closed;
      assert.match(wire, /^HTTP\/1\.1 200 /);
      assert(wire.includes(`"id":${JSON.stringify(id)}`));
      assert(wire.includes('"isError":true'));
      const reused = f.post(query(id, 'reused only after completion'), sid);
      await until(() => f.responses.length === 2, 'completed ID could not be reused');
      f.responses[1].end('ok');
      const reusedResult = await reused;
      assert.equal(reusedResult.status, 200);
      assert.equal(reusedResult.body.id, id);
      assert.notEqual(reusedResult.body.result.isError, true);
      report.positiveContract.push({ id, duplicateStatus: 409, operationsBeforeCancellation: 1,
        originalSignalAborted: true, originalOperationSettled: true, downstreamSocketClosed: true,
        originalSocketClosedWithoutClientTimeout: true, reusedStatus: 200 });
    } finally { await f.cleanup(); }
  }

  {
    const f = await fixture({ maxSessions: 1, maxActiveRequestsPerSession: 1, maxActiveRequests: 1,
      maxBodyBytes: 65536, bodyTimeoutMs: 3000 });
    try {
      const sid = await f.initSid();
      const original = f.wire(query(0), sid);
      await until(() => f.responses.length === 1, 'admitted operation did not fill the sole active slot');
      const pending = [];
      const observe = req => {
        const item = { req, bytes: 0 };
        pending.push(item); req.on('data', chunk => { item.bytes += chunk.length; });
      };
      f.handle.httpServer.on('request', observe);
      const partials = Array.from({ length: 16 }, () => f.wire(' '.repeat(32768), sid, true));
      await until(() => pending.length === 16 && pending.every(p => p.bytes === 32768), 'partial bodies were not all read');
      await delay(150);
      const retained = pending.filter(p => !p.req.complete && !p.req.destroyed).length;
      const earlyResponses = partials.filter(p => p.received().length > 0).length;
      assert.equal(retained, 16);
      assert.equal(earlyResponses, 0);
      assert.equal(f.operations.length, 1);
      assert.equal(f.operations[0].signal.aborted, false);
      report.defects.push({ contract: 'pending HTTP upload admission is bounded before body buffering',
        maxActiveRequests: 1, maxSessions: 1, admittedBlockedOperations: 1,
        simultaneousAdditionalUnfinishedUploadsRetained: retained,
        bytesReadAcrossUnfinishedUploads: pending.reduce((total, p) => total + p.bytes, 0),
        overflowResponsesBeforeBodyDeadline: earlyResponses,
        finding: 'Every partial MCP upload reached collectBody and retained bytes; neither session nor active-ID admission bounds pending uploads.' });
      f.handle.httpServer.off('request', observe);
      await f.cancel(sid, 0);
      await until(() => f.operations[0].settled && f.backendClosed() === 1, 'active owner was lost during upload overload');
      await original.closed;
    } finally { await f.cleanup(); }
  }

  {
    const f = await fixture({ maxSessions: 2, maxActiveRequestsPerSession: 1, maxActiveRequests: 1 });
    try {
      const sid = await f.initSid();
      const unrelatedIdleSid = await f.initSid();
      const original = f.wire(query(0), sid);
      await until(() => f.responses.length === 1, 'active global slot did not fill');
      const rejected = await f.init();
      assert.equal(rejected.status, 503);
      assert.equal(rejected.body.error.message, 'Server active request capacity reached; retry later');
      const unrelated = await f.post({ jsonrpc: '2.0', id: 'status', method: 'tools/call', params: {
        name: 'status', arguments: {},
      } }, unrelatedIdleSid);
      assert.equal(unrelated.status, 404);
      assert.equal(f.operations.length, 1);
      assert.equal(f.operations[0].signal.aborted, false);
      report.defects.push({ contract: 'global overflow rejected before allocation or mutation of existing sessions',
        maxSessions: 2, maxActiveRequests: 1, rejectedInitializeStatus: rejected.status,
        unrelatedPreviouslyValidIdleSessionStatusAfterRejectedInitialize: unrelated.status,
        finding: 'reserveSession evicts the unrelated idle session before ownership.reserve rejects global overflow.' });
      await f.cancel(sid, 0);
      await until(() => f.operations[0].settled && f.backendClosed() === 1, 'first owner lost after rejected initialization');
      await original.closed;
    } finally { await f.cleanup(); }
  }
} catch (error) {
  report.runnerError = { name: error.name, message: error.message, stack: error.stack };
} finally {
  report.after = Object.fromEntries([source, test, originalRed].map(path => [path, hash(path)]));
  report.sourceAndOriginalRedUnchanged = JSON.stringify(before) === JSON.stringify(report.after);
  report.overall = !report.runnerError && report.defects.length === 0 && report.sourceAndOriginalRedUnchanged ? 'PASS' : 'FAIL';
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ ...report, reportPath }, null, 2));
}
if (report.overall !== 'PASS') process.exitCode = 1;
