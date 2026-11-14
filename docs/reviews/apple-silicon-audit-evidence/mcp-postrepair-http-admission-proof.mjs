// Post-repair positive contracts; preserve the earlier negative runner/results.
// Owned CPU-only loopback fixtures, injected synthetic stores, public APIs only.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { connect } from 'node:net';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { startMcpHttpServer } from '/Users/shersingh/github/qmd-mlx-search/src/mcp/server.ts';

const root = '/Users/shersingh/github/qmd-mlx-search';
const audit = '/Users/shersingh/.hermes/cache/scratch/qmd-audit';
const reportPath = audit + '/mcp-postrepair-http-admission-proof-result.json';
const preserved = [
  root + '/src/mcp/server.ts', root + '/test/audit-mcp-admission.test.ts',
  root + '/test/audit-mcp-request-ownership.test.ts', root + '/test/audit-mcp-http.test.ts',
  root + '/test/audit-mcp-runtime-review.test.ts',
  audit + '/mcp-positive-contract-and-admission-review.mjs',
  audit + '/mcp-positive-contract-and-admission-review-result.json',
  audit + '/quality-mcp-heldout-cancel.mjs',
];
const hash = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const snapshot = () => Object.fromEntries(preserved.map(path => [path, hash(path)]));
const before = snapshot();
const report = { purpose: 'Final narrow pre-body HTTP bounds and rejected-initialization non-eviction closure',
  runtime: process.version, before, cases: [] };
const headers = { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' };
const status = { totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] };
const search = { searches: [{ type: 'lex', query: 'owned synthetic proof' }], rerank: false };
const query = id => ({ jsonrpc: '2.0', id, method: 'tools/call', params: { name: 'query', arguments: search } });
const cancel = requestId => ({ jsonrpc: '2.0', method: 'notifications/cancelled', params: { requestId } });
const statusTool = id => ({ jsonrpc: '2.0', id, method: 'tools/call', params: { name: 'status', arguments: {} } });
const init = { jsonrpc: '2.0', id: 'init', method: 'initialize', params: {
  protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'postrepair-heldout-proof', version: '1' },
} };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, description, ms = 1500) {
  const end = performance.now() + ms;
  while (!check() && performance.now() < end) await delay(5);
  assert(check(), description);
}
async function fixture(options = {}) {
  const operations = [], responses = [], sockets = [], jobs = [], records = [];
  const calls = { defaults: 0, status: 0, contexts: 0, global: 0, storeClose: 0 };
  const hooks = { status: undefined };
  const teardown = new AbortController();
  let handle;
  const backend = createServer((req, res) => {
    const index = Number(req.url.slice(1));
    responses[index] = res;
    res.once('close', () => { operations[index].downstreamClosed = true; });
  });
  function track(job) { void job.catch(() => {}); jobs.push(job); return job; }
  try {
    await new Promise((resolve, reject) => {
      backend.once('error', reject); backend.listen(0, '127.0.0.1', resolve);
    });
    const store = {
      getDefaultCollectionNames: async () => { calls.defaults++; return []; },
      getStatus: async () => { calls.status++; return hooks.status ? hooks.status() : status; },
      listContexts: async () => { calls.contexts++; return []; },
      getGlobalContext: async () => { calls.global++; return undefined; },
      search: async ({ signal }) => {
        assert(signal, 'missing real operation AbortSignal');
        const index = operations.length;
        const operation = { signal, settled: false, downstreamClosed: false };
        operations.push(operation);
        try {
          const response = await fetch(`http://127.0.0.1:${backend.address().port}/${index}`, {
            signal: AbortSignal.any([signal, teardown.signal]),
          });
          await response.text(); return [];
        } finally { operation.settled = true; }
      },
      close: async () => { calls.storeClose++; },
    };
    handle = await startMcpHttpServer(0, { store, quiet: true, requestTimeoutMs: 6000,
      bodyTimeoutMs: 3000, ...options });
    const base = `http://localhost:${handle.port}`;
    handle.httpServer.on('request', (req, res) => {
      // The app handler ran synchronously up to its first await. Never attach a
      // data listener to a rejected request: that would manufacture body reads.
      const record = { req, res, tag: req.headers['x-proof'], collectors: req.listenerCount('data'), bytes: 0 };
      records.push(record);
      if (record.collectors) req.on('data', chunk => { record.bytes += chunk.length; });
    });
    function wire(frame, watchdogMs = 7000) {
      let received = '', error;
      const started = performance.now();
      const socket = connect(handle.port, 'localhost', () => socket.write(frame));
      sockets.push(socket);
      socket.on('data', chunk => { received += chunk.toString(); });
      socket.on('error', failure => { error = failure; });
      const closed = track(new Promise((resolve, reject) => {
        const timer = setTimeout(() => socket.destroy(new Error('Owned wire watchdog fired; not a successful server close')), watchdogMs);
        socket.once('close', () => {
          clearTimeout(timer);
          if (error && error.code !== 'ECONNRESET') reject(error);
          else resolve({ wire: received, elapsedMs: performance.now() - started });
        });
      }));
      return { socket, closed, received: () => received };
    }
    function upload(path, body, length, sid, tag, extra = '') {
      const head = `POST ${path} HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n${sid ? `mcp-session-id: ${sid}\r\n` : ''}${tag ? `X-Proof: ${tag}\r\n` : ''}Content-Length: ${length}\r\nConnection: close\r\n${extra}\r\n`;
      return wire(Buffer.concat([Buffer.from(head), Buffer.from(body)]));
    }
    function rpcWire(body, sid, tag) {
      const text = JSON.stringify(body);
      return upload('/mcp', text, Buffer.byteLength(text), sid, tag);
    }
    function post(path, body, sid) {
      return track((async () => {
        const response = await fetch(base + path, { method: 'POST', headers: {
          ...headers, ...(sid ? { 'mcp-session-id': sid } : {}),
        }, body: JSON.stringify(body), signal: AbortSignal.any([teardown.signal, AbortSignal.timeout(6500)]) });
        const text = await response.text();
        return { status: response.status, sid: response.headers.get('mcp-session-id'), body: text ? JSON.parse(text) : undefined };
      })());
    }
    async function initialize() {
      const response = await post('/mcp', init);
      assert.equal(response.status, 200); assert(response.sid); return response.sid;
    }
    async function health(expected = 200) {
      const response = await fetch(base + '/health', { signal: AbortSignal.timeout(1000) });
      const body = await response.json(); assert.equal(response.status, expected);
      if (expected === 200) assert.equal(body.status, 'ok'); return response.status;
    }
    async function sse(sid, expected = 200) {
      const response = await fetch(base + '/mcp', { headers: { Accept: 'text/event-stream', 'mcp-session-id': sid }, signal: AbortSignal.timeout(1000) });
      assert.equal(response.status, expected); await response.body?.cancel(); return response.status;
    }
    async function remove(sid, expected = 200) {
      const response = await fetch(base + '/mcp', { method: 'DELETE', headers: { 'mcp-session-id': sid }, signal: AbortSignal.timeout(1000) });
      assert.equal(response.status, expected); await response.text(); return response.status;
    }
    async function finish(index) {
      await until(() => !!responses[index], `operation ${index} did not reach owned backend`);
      responses[index].end('owned normal completion');
    }
    async function restSuccess(path = '/query') {
      const index = operations.length;
      const work = post(path, search);
      await finish(index); const result = await work;
      assert.equal(result.status, 200); assert.deepEqual(result.body, { results: [] });
      assert.equal(operations[index].signal.aborted, false); return result;
    }
    return { handle, base, calls, hooks, operations, responses, records, wire, upload, rpcWire, post,
      initialize, health, sse, remove, finish, restSuccess,
      record: tag => records.find(record => record.tag === tag),
      cleanup: async () => {
        await handle.stop(); teardown.abort();
        for (const socket of sockets) socket.destroy();
        await Promise.allSettled(jobs);
        await until(() => operations.every(op => op.settled), 'teardown left an owned operation running');
        await new Promise(resolve => { backend.close(resolve); backend.closeAllConnections(); });
        assert.equal(calls.storeClose, 0, 'injected store is caller-owned');
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
async function run(name, fn) {
  const started = performance.now();
  try {
    const evidence = await fn();
    report.cases.push({ name, status: 'PASS', elapsedMs: performance.now() - started, evidence });
  } catch (error) {
    report.cases.push({ name, status: 'FAIL', error: { name: error.name, message: error.message, stack: error.stack } });
  }
}

await run('MCP and both REST aliases share a pre-body upload bound; refusal never installs a collector', async () => {
  const f = await fixture({ maxPendingUploads: 2, maxHttpRequests: 8, maxControlRequests: 2,
    maxActiveRequests: 1, maxSessions: 1, maxBodyBytes: 65536 });
  try {
    const sid = await f.initialize();
    const original = f.rpcWire(query(0), sid, 'legitimate');
    await until(() => !!f.responses[0], 'original did not start');
    const partials = [f.upload('/mcp', ' '.repeat(16384), 65536, sid, 'partial-mcp'),
      f.upload('/search', ' '.repeat(16384), 65536, undefined, 'partial-rest')];
    await until(() => ['partial-mcp', 'partial-rest'].every(tag => f.record(tag)?.bytes === 16384), 'bounded partials were not collected');
    const refusals = [];
    for (const [i, path] of ['/query', '/search', '/mcp'].entries()) {
      const tag = 'overflow-' + i;
      const refused = await f.upload(path, ' '.repeat(16384), 65536, sid, tag).closed;
      assert.match(refused.wire, /^HTTP\/1\.1 503 /); assert(refused.elapsedMs < 800);
      const record = f.record(tag); assert.equal(record.collectors, 0); assert.equal(record.bytes, 0);
      await until(() => record.req.destroyed, 'refused unread request was not destroyed');
      refusals.push({ path, status: 503, collectors: record.collectors, applicationBodyBytes: record.bytes });
    }
    assert.equal(f.records.filter(r => r.collectors && !r.req.complete && !r.req.destroyed).length, 2);
    assert(partials.every(partial => !partial.received()));
    assert.equal(f.operations.length, 1); assert.equal(f.operations[0].signal.aborted, false);
    await f.health();
    assert.equal((await f.post('/mcp', cancel(0), sid)).status, 202);
    const completed = await original.closed;
    assert.match(completed.wire, /^HTTP\/1\.1 200 /); assert(completed.wire.includes('"id":0'));
    assert(completed.wire.includes('"isError":true'));
    await until(() => f.operations[0].settled && f.operations[0].downstreamClosed, 'real cancelled downstream did not close');
    await f.remove(sid);
    const stopStarted = performance.now();
    await Promise.all([f.handle.stop(), f.handle.stop()]);
    await Promise.all(partials.map(partial => partial.closed));
    assert(performance.now() - stopStarted < 1000);
    return { pendingUploads: 2, applicationBytesRetained: 32768, refusals, health: 200,
      cancel: 202, delete: 200, originalSettledAndDownstreamClosed: true, stopClosedPartialUploads: true };
  } finally { await f.cleanup(); }
});

await run('The bounded control reserve refuses saturation honestly and recovers health, DELETE, and cancellation', async () => {
  const f = await fixture({ maxPendingUploads: 1, maxHttpRequests: 1, maxControlRequests: 1, maxActiveRequests: 1 });
  try {
    const sid = await f.initialize();
    const original = f.rpcWire(query(''), sid, 'control-original');
    await until(() => !!f.responses[0], 'active HTTP slot did not fill');
    assert.equal((await f.post('/mcp', query(''), sid)).status, 409);
    assert.equal((await f.post('/mcp', { jsonrpc: '2.0', method: 'notifications/initialized' }, sid)).status, 503);
    const blocked = f.upload('/mcp', '{', 128, sid, 'control-partial');
    await until(() => f.record('control-partial')?.bytes === 1, 'control reserve did not admit its bounded upload');
    assert.equal((await f.post('/mcp', cancel(''), sid)).status, 503);
    await f.health(503); await f.remove(sid, 503);
    assert.equal(f.operations.length, 1); assert.equal(f.operations[0].signal.aborted, false);
    blocked.socket.destroy(); await blocked.closed;
    await until(() => f.record('control-partial').req.destroyed, 'aborted control upload retained its slot');
    await f.health();
    assert.equal((await f.post('/mcp', cancel(''), sid)).status, 202);
    const completed = await original.closed;
    assert.match(completed.wire, /^HTTP\/1\.1 200 /); assert(completed.wire.includes('"id":""'));
    assert(completed.wire.includes('"isError":true'));
    await until(() => f.operations[0].settled && f.operations[0].downstreamClosed, 'control cancellation did not terminate real downstream');
    // DELETE itself makes progress while the single DATA slot is occupied.
    const next = f.rpcWire(query(0), sid, 'delete-original');
    await until(() => !!f.responses[1], 'fresh legitimate owner was not admitted');
    await f.remove(sid);
    await next.closed;
    await until(() => f.operations[1].settled && f.operations[1].downstreamClosed, 'DELETE did not cancel actual active work');
    await f.restSuccess();
    return { duplicate: 409, nonControlFallback: 503, saturatedCancel: 503, saturatedHealth: 503,
      saturatedDelete: 503, recoveredHealth: 200, recoveredCancel: 202, deleteUnderDataSaturation: 200,
      literalIds: ['', 0], downstreamsClosed: 2, freshRest: 200 };
  } finally { await f.cleanup(); }
});

await run('Chunked fallback cancellation uploads have a 4 KiB ceiling and no owner/control leak after overflow', async () => {
  const f = await fixture({ maxHttpRequests: 1, maxControlRequests: 1, maxPendingUploads: 1, maxActiveRequests: 1 });
  try {
    const sid = await f.initialize();
    const original = f.rpcWire(query(0), sid);
    await until(() => !!f.responses[0], 'active owner missing');
    const bytes = ' '.repeat(4097);
    const head = `POST /mcp HTTP/1.1\r\nHost: localhost:${f.handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nmcp-session-id: ${sid}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n`;
    const result = await f.wire(head + bytes.length.toString(16) + '\r\n' + bytes + '\r\n').closed;
    assert.match(result.wire, /^HTTP\/1\.1 413 /);
    assert.equal(f.operations.length, 1); assert.equal(f.operations[0].signal.aborted, false);
    await f.health(); assert.equal((await f.post('/mcp', cancel(0), sid)).status, 202);
    assert((await original.closed).wire.includes('"isError":true'));
    await until(() => f.operations[0].settled && f.operations[0].downstreamClosed, 'overflow damaged cancellation progress');
    return { chunkedBytes: 4097, ceiling: 4096, status: 413, activeOwnerUntouchedUntilRealCancel: true, health: 200, cancel: 202 };
  } finally { await f.cleanup(); }
});

await run('Receipt releases upload slots during long legitimate MCP/REST work; shared HTTP processing capacity remains exact', async () => {
  const f = await fixture({ maxPendingUploads: 1, maxHttpRequests: 3, maxActiveRequests: 1 });
  try {
    const sid = await f.initialize();
    const mcp = f.rpcWire(query(0), sid, 'long-mcp');
    await until(() => !!f.responses[0], 'MCP did not start');
    const restA = f.post('/query', search);
    await until(() => !!f.responses[1], 'complete MCP upload retained the sole upload slot');
    const restB = f.post('/search', search);
    await until(() => !!f.responses[2], 'complete REST upload retained the sole upload slot');
    const rejected = await f.upload('/query', '', 8192, undefined, 'http-overflow').closed;
    assert.match(rejected.wire, /^HTTP\/1\.1 503 /);
    assert.equal(f.record('http-overflow').collectors, 0);
    assert.equal(f.operations.length, 3); assert(f.operations.every(op => !op.signal.aborted));
    await f.health(); assert.equal((await f.post('/mcp', cancel(0), sid)).status, 202);
    assert((await mcp.closed).wire.includes('"isError":true'));
    assert.equal(f.operations[1].signal.aborted, false); assert.equal(f.operations[2].signal.aborted, false);
    await f.finish(1); await f.finish(2);
    assert.equal((await restA).status, 200); assert.equal((await restB).status, 200);
    const reused = f.post('/mcp', query(0), sid); await f.finish(3);
    const result = await reused; assert.equal(result.status, 200); assert.equal(result.body.id, 0);
    assert.notEqual(result.body.result.isError, true);
    return { concurrentCompletedUploadsStillProcessing: 3, maxPendingUploads: 1,
      refusedFourthHttpHandler: 503, refusedCollector: 0, untouchedRestOperations: 2, normalRestCompletions: 2, reusedLiteralZero: 200 };
  } finally { await f.cleanup(); }
});

for (const path of ['/mcp', '/query', '/search']) {
  for (const cause of ['disconnect', 'body deadline', 'processing deadline', 'invalid JSON', 'invalid UTF-8']) {
    await run(`${path}: upload/HTTP slots recover after ${cause} without underflow`, async () => {
      const f = await fixture({ maxPendingUploads: 1, maxHttpRequests: 2, maxBodyBytes: 2048,
        bodyTimeoutMs: cause === 'body deadline' ? 180 : 1500,
        requestTimeoutMs: cause === 'processing deadline' ? 180 : 2500 });
      try {
        const bytes = cause === 'invalid UTF-8' ? Buffer.from([255]) : Buffer.from('{');
        const full = cause.startsWith('invalid');
        const first = f.upload(path, full ? bytes : '', full ? bytes.length : 128, undefined, 'release-original');
        await until(() => !!f.record('release-original'), 'initial release-test upload not observed');
        assert.equal(f.record('release-original').collectors, 1);
        if (cause === 'disconnect') first.socket.destroy();
        const result = await first.closed;
        const expected = cause === 'body deadline' ? 408 : cause === 'processing deadline' ? 504 : 400;
        if (cause !== 'disconnect') assert.match(result.wire, new RegExp(`^HTTP/1\\.1 ${expected} `));
        await until(() => f.record('release-original').req.destroyed, 'failed upload retained the request');
        assert.equal(f.operations.length, 0);
        // Refilling exactly ONE slot plus another refusal checks both leaks and
        // accidental double release/negative counts after the prior failure.
        const refill = f.upload('/query', '', 128, undefined, 'refill');
        await until(() => !!f.record('refill'), 'released capacity was not reusable');
        assert.equal(f.record('refill').collectors, 1);
        const overflow = await f.upload('/search', '', 128, undefined, 'refill-overflow').closed;
        assert.match(overflow.wire, /^HTTP\/1\.1 503 /);
        assert.equal(f.record('refill-overflow').collectors, 0);
        refill.socket.destroy(); await refill.closed;
        await until(() => f.record('refill').req.destroyed, 'refill abort did not settle');
        await f.restSuccess('/search');
        return { cause, initialStatus: cause === 'disconnect' ? 'client disconnect' : expected,
          refillCollectors: 1, exactCapacityOverflowStatus: 503, overflowCollectors: 0, recoveredRest: 200 };
      } finally { await f.cleanup(); }
    });
  }
}

await run('Raw socket count, absolute incomplete-header deadline, and parser header bytes are bounded before application requests', async () => {
  const f = await fixture({ maxConnections: 3, maxHeaderBytes: 1024, headersTimeoutMs: 150 });
  let trickle;
  try {
    const raw = [f.wire(''), f.wire(`GET /health HTTP/1.1\r\nHost: localhost:${f.handle.port}\r\nX-Trickle: `),
      f.wire('GET /health HTTP/1.1\r\n')];
    await until(() => raw.every(item => !item.socket.connecting), 'raw fixture connections not established');
    const full = `GET /health HTTP/1.1\r\nHost: localhost:${f.handle.port}\r\nConnection: close\r\n\r\n`;
    const dropped = await f.wire(full).closed;
    assert.equal(dropped.wire, ''); assert.equal(f.records.length, 0);
    trickle = setInterval(() => { if (!raw[1].socket.destroyed) raw[1].socket.write('x'); }, 10);
    const timedOut = await Promise.all(raw.map(item => item.closed));
    clearInterval(trickle); trickle = undefined;
    assert.match(timedOut[1].wire, /^HTTP\/1\.1 408 /); assert(timedOut.every(item => item.elapsedMs < 900));
    assert.equal(f.records.length, 0);
    const oversized = await f.wire(`GET /health HTTP/1.1\r\nHost: localhost:${f.handle.port}\r\nX-Pad: ${'x'.repeat(2048)}\r\n\r\n`).closed;
    assert.match(oversized.wire, /^HTTP\/1\.1 431 /); assert.equal(f.records.length, 0);
    assert.deepEqual(f.calls, { defaults: 1, status: 0, contexts: 0, global: 0, storeClose: 0 });
    const recovered = await f.wire(full).closed; assert.match(recovered.wire, /^HTTP\/1\.1 200 /);
    assert.equal(f.records.length, 1);
    return { maxConnections: 3, overcapWireBytes: 0, absoluteHeaderDeadlineMs: 150, trickleStatus: 408,
      maxHeaderBytes: 1024, oversizedStatus: 431, applicationRequestsBeforeRecovery: 0, recoveredHealth: 200 };
  } finally { clearInterval(trickle); await f.cleanup(); }
});

await run('Rejected global initialization leaves idle B valid, active A untouched, and later legitimate initialization available', async () => {
  const f = await fixture({ maxSessions: 2, maxActiveRequests: 1, maxActiveRequestsPerSession: 1 });
  try {
    const a = await f.initialize(), b = await f.initialize();
    const original = f.rpcWire(query(0), a, 'no-evict-active');
    await until(() => !!f.responses[0], 'active A missing');
    const initialCalls = { ...f.calls };
    const rejectedStatuses = [];
    for (let i = 0; i < 7; i++) {
      const refused = await f.post('/mcp', init);
      assert.equal(refused.status, 503);
      assert.equal(refused.body.error.message, 'Server active request capacity reached; retry later');
      rejectedStatuses.push(refused.status);
      await f.sse(b); // A real existing-session operation, no global ID needed.
    }
    assert.deepEqual(f.calls, initialCalls, 'refused initialization performed allocation-time store reads');
    assert.equal((await f.post('/mcp', query(0), a)).status, 409);
    assert.equal(f.operations.length, 1); assert.equal(f.operations[0].signal.aborted, false);
    assert.equal((await f.post('/mcp', cancel(0), a)).status, 202);
    const completed = await original.closed;
    assert.match(completed.wire, /^HTTP\/1\.1 200 /); assert(completed.wire.includes('"isError":true'));
    await until(() => f.operations[0].settled && f.operations[0].downstreamClosed, 'A was not normally cancellable');
    assert.equal((await f.post('/mcp', statusTool('B-after-refusals'), b)).status, 200);
    // Now a REAL admitted init may intentionally evict idle A. Do not freeze
    // state across this later permitted mutation or forbid own-record cleanup.
    const c = await f.initialize();
    assert.equal((await f.post('/mcp', statusTool('B-after-admitted-C'), b)).status, 200);
    assert.equal((await f.post('/mcp', statusTool('C-valid'), c)).status, 200);
    await f.sse(a, 404);
    return { refusedInitializations: rejectedStatuses, BDuringEveryRefusal: 200, allocationReadsOnRefusal: 0,
      duplicateA: 409, ASignalUnaffectedUntilCancel: true, ACancel: 202, BAfterRelease: 200,
      admittedC: 200, BAfterAdmittedC: 200, CValid: 200, intentionallyEvictedIdleA: 404 };
  } finally { await f.cleanup(); }
});

await run('An asynchronous initialization holds actual global capacity before bystander eviction can be attempted', async () => {
  const f = await fixture({ maxSessions: 2, maxActiveRequests: 1 });
  let release;
  try {
    const b = await f.initialize();
    f.hooks.status = () => { f.hooks.status = undefined; return new Promise(resolve => { release = () => resolve(status); }); };
    const pending = f.post('/mcp', init);
    await until(() => !!release, 'init did not hold across asynchronous instruction reads');
    for (let i = 0; i < 4; i++) {
      const refused = await f.post('/mcp', init);
      assert.equal(refused.status, 503);
      assert.equal(refused.body.error.message, 'Server active request capacity reached; retry later');
      await f.sse(b);
    }
    release(); const admitted = await pending;
    assert.equal(admitted.status, 200); assert(admitted.sid);
    assert.equal((await f.post('/mcp', statusTool('B-survived-async'), b)).status, 200);
    assert.equal((await f.post('/mcp', statusTool('new-owner-valid'), admitted.sid)).status, 200);
    return { heldAcrossAwait: true, refusedCompetitors: 4, refusedStatus: 503, idleBSurvived: 200,
      originalInitializationCompleted: 200, initializedOwnerValid: 200 };
  } finally { release?.(); await f.cleanup(); }
});

await run('Session-slot refusal releases the pre-reserved global lease once, with neither leaks nor counter underflow', async () => {
  const f = await fixture({ maxSessions: 1, maxActiveRequests: 2, maxActiveRequestsPerSession: 3 });
  try {
    const a = await f.initialize();
    const original = f.rpcWire(query(0), a);
    await until(() => !!f.responses[0], 'first global owner missing');
    for (let i = 0; i < 9; i++) {
      const refused = await f.post('/mcp', init);
      assert.equal(refused.status, 503); assert.equal(refused.body.error.message, 'Session capacity reached; retry later');
    }
    const second = f.post('/mcp', query(''), a);
    await until(() => !!f.responses[1], 'session refusal leaked global capacity');
    const overflow = await f.post('/mcp', query('third owner forbidden'), a);
    assert.equal(overflow.status, 503);
    assert.equal(overflow.body.error.message, 'Server active request capacity reached; retry later');
    assert.equal(f.operations.length, 2); assert(f.operations.every(op => !op.signal.aborted));
    assert.equal((await f.post('/mcp', query(0), a)).status, 409);
    await f.finish(0); await f.finish(1);
    const firstResult = await original.closed; assert.match(firstResult.wire, /^HTTP\/1\.1 200 /);
    assert(!firstResult.wire.includes('"isError":true'));
    assert.equal((await second).status, 200);
    const fresh = await f.initialize(); assert(fresh);
    return { rejectedSessionReservations: 9, firstOwnerPreserved: true, remainingGlobalSlotUsable: true,
      exactGlobalOwners: 2, refusedThird: 503, duplicatePrecedence: 409, normalCompletions: 2, freshInitialization: 200 };
  } finally { await f.cleanup(); }
});

await run('A real initialization store failure returns all leases, and a subsequent owner still sees the exact global cap', async () => {
  const f = await fixture({ maxSessions: 1, maxActiveRequests: 1, maxActiveRequestsPerSession: 2 });
  try {
    f.hooks.status = async () => { f.hooks.status = undefined; throw new Error('owned synthetic instructions failure'); };
    assert.equal((await f.post('/mcp', init)).status, 500);
    const a = await f.initialize();
    const original = f.rpcWire(query(0), a);
    await until(() => !!f.responses[0], 'failed initialization leaked capacity');
    assert.equal((await f.post('/mcp', query('overflow'), a)).status, 503);
    assert.equal(f.operations.length, 1); assert.equal(f.operations[0].signal.aborted, false);
    await f.finish(0); const completed = await original.closed;
    assert.match(completed.wire, /^HTTP\/1\.1 200 /); assert(!completed.wire.includes('"isError":true'));
    const b = await f.initialize(); assert(b);
    return { injectedInitializationFailure: 500, nextInitialization: 200, actualOwnersAtCap: 1,
      refusedAdditionalOwner: 503, normalCompletion: 200, laterInitialization: 200 };
  } finally { await f.cleanup(); }
});

report.after = snapshot();
report.sourceTestsAndNegativeEvidenceUnchanged = JSON.stringify(before) === JSON.stringify(report.after);
report.passed = report.cases.filter(entry => entry.status === 'PASS').length;
report.failed = report.cases.filter(entry => entry.status === 'FAIL').length;
report.total = report.cases.length;
report.overall = report.failed === 0 && report.sourceTestsAndNegativeEvidenceUnchanged ? 'PASS' : 'FAIL';
writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ overall: report.overall, total: report.total, passed: report.passed, failed: report.failed,
  sourceTestsAndNegativeEvidenceUnchanged: report.sourceTestsAndNegativeEvidenceUnchanged, reportPath,
  failures: report.cases.filter(entry => entry.status === 'FAIL') }, null, 2));
if (report.overall !== 'PASS') process.exitCode = 1;
