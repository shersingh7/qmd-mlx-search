import { afterEach, describe, expect, test, vi } from "vitest";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { connect, type Socket } from "node:net";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { QMDStore } from "../src/index.js";
import { startMcpHttpServer, type HttpServerHandle, type HttpServerOptions } from "../src/mcp/server.js";

// Owned loopback sockets and synthetic stores only; no models or live services.
const handles: HttpServerHandle[] = [];
const backends: Server[] = [];
const sockets: Socket[] = [];
const status = { totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] };
const headers = { "Content-Type": "application/json", Accept: "application/json, text/event-stream" };
const search = { searches: [{ type: "lex", query: "synthetic" }], rerank: false };
const init = { jsonrpc: "2.0", id: "init", method: "initialize", params: {
  protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "admission-audit", version: "1" },
} };
const query = (id: number | string) => ({ jsonrpc: "2.0", id, method: "tools/call", params: { name: "query", arguments: search } });
const cancel = (requestId: number | string) => ({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId } });
const toolStatus = { jsonrpc: "2.0", id: "status", method: "tools/call", params: { name: "status", arguments: {} } };
const paths = ["/mcp", "/query", "/search"] as const;
const optionNames = ["maxPendingUploads", "maxHttpRequests", "maxControlRequests", "maxConnections", "maxHeaderBytes", "headersTimeoutMs"] as const;
const releaseCauses = ["complete", "invalid JSON", "invalid UTF-8", "oversize", "disconnect", "body deadline", "processing deadline"] as const;
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
async function until(check: () => boolean, timeout = 1000) {
  const end = performance.now() + timeout;
  while (!check() && performance.now() < end) await delay(5);
  expect(check()).toBe(true);
}
function fixtureStore(): QMDStore {
  return {
    getDefaultCollectionNames: vi.fn(async () => []), getStatus: vi.fn(async () => status),
    listContexts: vi.fn(async () => []), getGlobalContext: vi.fn(async () => undefined),
    search: vi.fn(async () => []), close: vi.fn(async () => {}),
  } as unknown as QMDStore;
}
async function start(options: HttpServerOptions = {}, store = fixtureStore()) {
  const handle = await startMcpHttpServer(0, { quiet: true, store, requestTimeoutMs: 3000, ...options });
  handles.push(handle);
  return { handle, store, base: `http://localhost:${handle.port}` };
}
function post(base: string, path: string, body: unknown, sid?: string) {
  const job = (async () => {
    const response = await fetch(base + path, { method: "POST", headers: {
      ...headers, ...(sid ? { "mcp-session-id": sid } : {}),
    }, body: JSON.stringify(body), signal: AbortSignal.timeout(2000) });
    const text = await response.text();
    return { response, body: text ? JSON.parse(text) : undefined };
  })();
  void job.catch(() => {});
  return job;
}
async function initialize(base: string) {
  const { response, body } = await post(base, "/mcp", init);
  expect(response.status).toBe(200);
  expect(body.result.serverInfo.name).toBe("qmd");
  const sid = response.headers.get("mcp-session-id");
  expect(sid).toBeTruthy();
  return sid!;
}
function wire(handle: HttpServerHandle, frame: string) {
  let received = "";
  let error: NodeJS.ErrnoException | undefined;
  const socket = connect(handle.port, "localhost", () => socket.write(frame));
  sockets.push(socket);
  socket.on("data", chunk => { received += chunk.toString(); });
  socket.on("error", failure => { error = failure; });
  const closed = new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => socket.destroy(new Error("Server did not close the owned socket")), 1500);
    socket.once("close", () => {
      clearTimeout(timer);
      if (error && error.code !== "ECONNRESET") reject(error);
      else resolve(received);
    });
  });
  void closed.catch(() => {});
  return { socket, closed, received: () => received };
}
function upload(handle: HttpServerHandle, path: string, body: string, length: number, sid?: string, extra = "") {
  return wire(handle, `POST ${path} HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n${sid ? `mcp-session-id: ${sid}\r\n` : ""}Content-Length: ${length}\r\nConnection: close\r\n${extra}\r\n${body}`);
}
function watchRequests(handle: HttpServerHandle) {
  const requests: { req: IncomingMessage; collectors: number; bytes: number }[] = [];
  // The application handler has already run synchronously up to its first await.
  // Never attach a data listener to a refused upload: that would drain it and
  // manufacture a different contract than the real client experiences.
  handle.httpServer.on("request", req => {
    const item = { req, collectors: req.listenerCount("data"), bytes: 0 };
    requests.push(item);
    if (item.collectors) req.on("data", chunk => { item.bytes += chunk.length; });
  });
  return requests;
}
async function backendFixture() {
  const responses: ServerResponse[] = [];
  const operations: { signal: AbortSignal; settled: boolean }[] = [];
  const state = { closed: 0 };
  const backend = createServer((_req, res) => {
    responses.push(res);
    res.once("close", () => { state.closed++; });
  });
  backends.push(backend);
  await new Promise<void>(resolve => backend.listen(0, "127.0.0.1", resolve));
  const port = (backend.address() as import("node:net").AddressInfo).port;
  const store = fixtureStore();
  vi.spyOn(store, "search").mockImplementation(async ({ signal }) => {
    if (!signal) throw new Error("Missing real search cancellation signal");
    const operation = { signal, settled: false };
    operations.push(operation);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/owned`, { signal });
      await response.text();
      return [];
    } finally { operation.settled = true; }
  });
  return { store, operations, responses, state };
}
async function getSse(base: string, sid: string) {
  const response = await fetch(base + "/mcp", { headers: { Accept: "text/event-stream", "mcp-session-id": sid }, signal: AbortSignal.timeout(1000) });
  expect(response.status).toBe(200);
  await response.body?.cancel();
}
async function health(base: string) {
  const response = await fetch(base + "/health", { signal: AbortSignal.timeout(1000) });
  expect(response.status).toBe(200);
  expect((await response.json() as any).status).toBe("ok");
}

afterEach(async () => {
  for (const socket of sockets.splice(0)) socket.destroy();
  await Promise.all(handles.splice(0).map(handle => handle.stop()));
  await Promise.all(backends.splice(0).map(backend => new Promise<void>(resolve => {
    backend.close(() => resolve()); backend.closeAllConnections();
  })));
  vi.restoreAllMocks();
});

if (process.versions.bun) {
  test("Bun entrypoint executes all pre-body admission assertions on the supported Node HTTP runtime", async () => {
    const { stdout } = await promisify(execFile)("node", [
      "node_modules/vitest/vitest.mjs", "run", "--pool=threads", "--maxWorkers=1",
      "--cache=false", "--reporter=json", "test/audit-mcp-admission.test.ts",
    ], { cwd: fileURLToPath(new URL("../", import.meta.url)), env: { ...process.env, CI: "true", QMD_LLAMA_GPU: "false" },
      timeout: 30000, killSignal: "SIGKILL", maxBuffer: 4 * 1024 * 1024 });
    const report = JSON.parse(stdout);
    expect(report.numTotalTests).toBe(optionNames.length + paths.length + releaseCauses.length + 13);
    expect(report.numPassedTests).toBe(report.numTotalTests);
    expect(report.numFailedTests).toBe(0);
    expect(report.numPendingTests).toBe(0);
  }, 35000);
} else describe("pre-body HTTP admission and initialization isolation", () => {
  test.each(optionNames)("validates %s before store/listener allocation", async name => {
    const store = fixtureStore();
    for (const value of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER, ...(name === "headersTimeoutMs" ? [300001] : [])]) {
      // Retain an accidentally started listener for teardown if validation regresses.
      const attempt = startMcpHttpServer(0, { store, quiet: true, [name]: value }).then(handle => { handles.push(handle); });
      await expect(attempt).rejects.toThrow(new RegExp(name));
    }
    expect(store.getDefaultCollectionNames).not.toHaveBeenCalled();
  });

  test.each(paths)("%s partial uploads are refused before collection, without losing the active owner or control plane", async path => {
    const fixture = await backendFixture();
    const { handle, base } = await start({ maxPendingUploads: 2, maxHttpRequests: 4, maxControlRequests: 2,
      maxSessions: 1, maxActiveRequests: 1, maxActiveRequestsPerSession: 1, bodyTimeoutMs: 1200, maxBodyBytes: 65536 }, fixture.store);
    const sid = await initialize(base);
    const original = post(base, "/mcp", query(0), sid);
    await until(() => fixture.responses.length === 1);
    const requests = watchRequests(handle);
    const partials = [upload(handle, path, " ".repeat(32768), 65536, sid), upload(handle, path, " ".repeat(32768), 65536, sid)];
    await until(() => requests.length === 2 && requests.every(item => item.bytes === 32768));
    const started = performance.now();
    const refused = upload(handle, path, " ".repeat(32768), 65536, sid);
    const refusedWire = await refused.closed;
    expect(performance.now() - started).toBeLessThan(500);
    expect(refusedWire).toMatch(/^HTTP\/1\.1 503 /);
    expect(requests.map(item => item.collectors)).toEqual([1, 1, 0]);
    // Application rejection allows a 50ms error-flush window before explicitly
    // destroying the unread IncomingMessage, even if the TCP EOF arrived first.
    await until(() => requests[2]!.req.destroyed);
    expect(requests.filter(item => !item.req.complete && !item.req.destroyed)).toHaveLength(2);
    expect(partials.every(item => !item.received())).toBe(true);
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    await health(base);
    // Small cancellation uploads have a separate, explicitly bounded reserve.
    expect((await post(base, "/mcp", cancel(0), sid)).response.status).toBe(202);
    expect((await original).body.result.isError).toBe(true);
    await until(() => fixture.operations[0]!.settled && fixture.state.closed === 1);
    const deleted = await fetch(base + "/mcp", { method: "DELETE", headers: { "mcp-session-id": sid }, signal: AbortSignal.timeout(1000) });
    expect(deleted.status).toBe(200); await deleted.text();
    const before = performance.now();
    await Promise.all([handle.stop(), handle.stop()]);
    await Promise.all(partials.map(item => item.closed));
    expect(performance.now() - before).toBeLessThan(1000);
    expect(handle.httpServer.listening).toBe(false);
    expect(fixture.store.close).not.toHaveBeenCalled();
  });

  test.each(releaseCauses)("upload slot recovers after %s", async cause => {
    const { handle, base, store } = await start({ maxPendingUploads: 1, maxHttpRequests: 3, maxBodyBytes: 512,
      bodyTimeoutMs: cause === "body deadline" ? 100 : 1000, requestTimeoutMs: cause === "processing deadline" ? 100 : 2000 });
    const requests = watchRequests(handle);
    const text = JSON.stringify(search);
    const bytes = cause === "invalid UTF-8" ? Buffer.from([0xff]) : Buffer.from(cause === "invalid JSON" ? "{" : text);
    // Use a binary write for UTF-8 rather than re-encoding invalid bytes as text.
    const partial = upload(handle, "/query", "", cause === "oversize" ? 513 : bytes.length);
    await until(() => requests.length === 1);
    if (cause === "disconnect") partial.socket.destroy();
    else if (cause !== "body deadline" && cause !== "processing deadline" && cause !== "oversize") partial.socket.write(bytes);
    const result = await partial.closed;
    if (cause === "complete") expect(result).toMatch(/^HTTP\/1\.1 200 /);
    else if (cause === "oversize") expect(result).toMatch(/^HTTP\/1\.1 413 /);
    else if (cause === "body deadline") expect(result).toMatch(/^HTTP\/1\.1 408 /);
    else if (cause === "processing deadline") expect(result).toMatch(/^HTTP\/1\.1 504 /);
    else if (cause !== "disconnect") expect(result).toMatch(/^HTTP\/1\.1 400 /);
    expect((await post(base, "/search", search)).response.status).toBe(200);
    expect(store.search).toHaveBeenCalledTimes(cause === "complete" ? 2 : 1);
  });

  test("completed uploads release their slot while a REST query is still running, but the HTTP processing cap applies to both aliases", async () => {
    const fixture = await backendFixture();
    const { handle, base } = await start({ maxPendingUploads: 1, maxHttpRequests: 2 }, fixture.store);
    const first = post(base, "/query", search);
    await until(() => fixture.responses.length === 1);
    const second = post(base, "/search", search);
    await until(() => fixture.responses.length === 2);
    const requests = watchRequests(handle);
    const third = upload(handle, "/query", "", 100);
    expect(await third.closed).toMatch(/^HTTP\/1\.1 503 /);
    expect(requests[0]!.collectors).toBe(0);
    await health(base);
    expect(fixture.operations.every(item => !item.signal.aborted)).toBe(true);
    fixture.responses.forEach(response => response.end("ok"));
    expect((await first).response.status).toBe(200);
    expect((await second).response.status).toBe(200);
    const recovered = post(base, "/search", search);
    await until(() => fixture.responses.length === 3);
    fixture.responses[2]!.end("recovered");
    expect((await recovered).response.status).toBe(200);
  });

  test("HTTP data saturation preserves duplicate 409, refuses non-control fallback work, and still dispatches real cancellation", async () => {
    const fixture = await backendFixture();
    const { base } = await start({ maxHttpRequests: 1, maxPendingUploads: 1, maxControlRequests: 1, maxActiveRequests: 1 }, fixture.store);
    const sid = await initialize(base);
    const original = post(base, "/mcp", query(0), sid);
    await until(() => fixture.responses.length === 1);
    expect((await post(base, "/mcp", query(0), sid)).response.status).toBe(409);
    expect((await post(base, "/mcp", query("not a control message"), sid)).response.status).toBe(503);
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    await health(base);
    expect((await post(base, "/mcp", cancel(0), sid)).response.status).toBe(202);
    expect((await original).body.result.isError).toBe(true);
    expect((await post(base, "/mcp", toolStatus, sid)).response.status).toBe(200);
  });

  test("control uploads are themselves bounded and an overloaded cancellation is refused, never falsely acknowledged", async () => {
    const fixture = await backendFixture();
    const { handle, base } = await start({ maxHttpRequests: 1, maxPendingUploads: 1, maxControlRequests: 1, maxActiveRequests: 1 }, fixture.store);
    const sid = await initialize(base);
    const original = post(base, "/mcp", query(0), sid);
    await until(() => fixture.responses.length === 1);
    const requests = watchRequests(handle);
    const control = upload(handle, "/mcp", "{", 100, sid);
    await until(() => requests.length === 1 && requests[0]!.bytes === 1);
    const refused = await post(base, "/mcp", cancel(0), sid);
    expect(refused.response.status).toBe(503);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    control.socket.destroy(); await control.closed;
    await until(() => requests[0]!.req.destroyed);
    expect((await post(base, "/mcp", cancel(0), sid)).response.status).toBe(202);
    expect((await original).body.result.isError).toBe(true);
  });

  test("chunked control overflow has a hard 4 KiB ceiling and releases its reserve without touching the active owner", async () => {
    const fixture = await backendFixture();
    const { handle, base } = await start({ maxHttpRequests: 1, maxPendingUploads: 1, maxControlRequests: 1 }, fixture.store);
    const sid = await initialize(base);
    const original = post(base, "/mcp", query(0), sid);
    await until(() => fixture.responses.length === 1);
    const bytes = " ".repeat(4097);
    const oversized = wire(handle, `POST /mcp HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nmcp-session-id: ${sid}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n${bytes.length.toString(16)}\r\n${bytes}\r\n`);
    expect(await oversized.closed).toMatch(/^HTTP\/1\.1 413 /);
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    expect((await post(base, "/mcp", cancel(0), sid)).response.status).toBe(202);
    expect((await original).body.result.isError).toBe(true);
  });

  test.each(["session IDs", "global IDs", "session slots"])("fully received %s overflow on the original TCP connection preserves its active owner", async cause => {
    const fixture = await backendFixture();
    const { handle, base } = await start({ maxSessions: 1,
      maxActiveRequests: cause === "global IDs" ? 1 : 2,
      maxActiveRequestsPerSession: cause === "session IDs" ? 1 : 2 }, fixture.store);
    const sid = await initialize(base);
    const frame = (body: unknown, sessionId: string | undefined, close: boolean) => {
      const text = JSON.stringify(body);
      return `POST /mcp HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n${sessionId ? `mcp-session-id: ${sessionId}\r\n` : ""}Content-Length: ${Buffer.byteLength(text)}\r\nConnection: ${close ? "close" : "keep-alive"}\r\n\r\n${text}`;
    };
    const original = wire(handle, frame(query(0), sid, false));
    await until(() => fixture.responses.length === 1);
    original.socket.write(frame(cause === "session slots" ? init : query("overflow"), cause === "session slots" ? undefined : sid, true));
    await delay(80); // Beyond the unread-body error path's socket-destruction window.
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    fixture.responses[0]!.end("original completed normally");
    const result = await original.closed;
    expect(result.match(/HTTP\/1\.1 \d{3} /g)).toEqual(["HTTP/1.1 200 ", cause === "session IDs" ? "HTTP/1.1 429 " : "HTTP/1.1 503 "]);
    expect(result).toContain('"id":0');
  });

  test("global rejected initialization cannot evict an idle session or disturb another session's active owner", async () => {
    const fixture = await backendFixture();
    const connects = vi.spyOn(McpServer.prototype, "connect");
    const { base } = await start({ maxSessions: 2, maxActiveRequests: 1, maxActiveRequestsPerSession: 1 }, fixture.store);
    const a = await initialize(base);
    const b = await initialize(base);
    const original = post(base, "/mcp", query(0), a);
    await until(() => fixture.responses.length === 1);
    for (let i = 0; i < 4; i++) {
      const rejected = await post(base, "/mcp", init);
      expect(rejected.response.status).toBe(503);
      expect(rejected.body.error.message).toBe("Server active request capacity reached; retry later");
    }
    expect(connects).toHaveBeenCalledTimes(2);
    expect(fixture.store.getStatus).toHaveBeenCalledTimes(2);
    await getSse(base, b); // HTTP 200 even while all ID-bearing work is at capacity.
    expect((await post(base, "/mcp", query(0), a)).response.status).toBe(409);
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    expect((await post(base, "/mcp", cancel(0), a)).response.status).toBe(202);
    expect((await original).body.result.isError).toBe(true);
    expect((await post(base, "/mcp", toolStatus, b)).response.status).toBe(200);
    const c = await initialize(base); // Real admission and idle eviction still work.
    expect((await post(base, "/mcp", toolStatus, b)).response.status).toBe(200);
    expect((await post(base, "/mcp", toolStatus, c)).response.status).toBe(200);
  });

  test("an initializing owner holds real global capacity across asynchronous store reads, without evicting a bystander on the next refusal", async () => {
    const { base, store } = await start({ maxSessions: 2, maxActiveRequests: 1 });
    const b = await initialize(base);
    let release!: (value: typeof status) => void;
    vi.spyOn(store, "getStatus").mockImplementationOnce(() => new Promise(resolve => { release = resolve; }));
    const pending = post(base, "/mcp", init);
    try {
      await until(() => !!release);
      expect((await post(base, "/mcp", init)).response.status).toBe(503);
      await getSse(base, b);
    } finally { release(status); }
    expect((await pending).response.status).toBe(200);
    expect((await post(base, "/mcp", toolStatus, b)).response.status).toBe(200);
  });

  test("failure to obtain a session slot releases the already-reserved global owner exactly once", async () => {
    const fixture = await backendFixture();
    const { base } = await start({ maxSessions: 1, maxActiveRequests: 2, maxActiveRequestsPerSession: 3 }, fixture.store);
    const a = await initialize(base);
    const original = post(base, "/mcp", query(0), a);
    await until(() => fixture.responses.length === 1);
    for (let i = 0; i < 3; i++) {
      const rejected = await post(base, "/mcp", init);
      expect(rejected.response.status).toBe(503);
      expect(rejected.body.error.message).toBe("Session capacity reached; retry later");
    }
    const concurrent = post(base, "/mcp", query("remaining real global slot"), a);
    await until(() => fixture.responses.length === 2);
    expect((await post(base, "/mcp", query("overflow"), a)).response.status).toBe(503);
    expect(fixture.operations.every(item => !item.signal.aborted)).toBe(true);
    fixture.responses.forEach(response => response.end("ok"));
    expect((await original).response.status).toBe(200);
    expect((await concurrent).response.status).toBe(200);
    await initialize(base);
  });

  test("malformed initialization is rejected before idle-session eviction", async () => {
    const connects = vi.spyOn(McpServer.prototype, "connect");
    const { base } = await start({ maxSessions: 1, maxActiveRequests: 1 });
    const sid = await initialize(base);
    for (const invalid of [{ ...init, params: {} }, [init, init], { ...init, params: { ...init.params, clientInfo: null } }]) {
      expect((await post(base, "/mcp", invalid)).response.status).toBe(400);
      expect((await post(base, "/mcp", toolStatus, sid)).response.status).toBe(200);
    }
    expect(connects).toHaveBeenCalledTimes(1);
  });

  test("raw connections, incomplete headers and oversized headers are bounded before any application request", async () => {
    const { handle, base, store } = await start({ maxConnections: 2, maxHeaderBytes: 512, headersTimeoutMs: 100 });
    const requests = watchRequests(handle);
    const empty = wire(handle, "");
    const partial = wire(handle, `GET /health HTTP/1.1\r\nHost: localhost:${handle.port}\r\nX-Trickle: `);
    await until(() => !empty.socket.connecting && !partial.socket.connecting);
    const dropped = wire(handle, `GET /health HTTP/1.1\r\nHost: localhost:${handle.port}\r\n\r\n`);
    expect(await dropped.closed).toBe("");
    expect(requests).toHaveLength(0);
    const trickle = setInterval(() => { if (!partial.socket.destroyed) partial.socket.write("x"); }, 10);
    try {
      expect(await partial.closed).toMatch(/^HTTP\/1\.1 408 /);
      await empty.closed;
    } finally { clearInterval(trickle); }
    expect(requests).toHaveLength(0);
    const oversized = wire(handle, `GET /health HTTP/1.1\r\nHost: localhost:${handle.port}\r\nX-Pad: ${"x".repeat(1024)}\r\n\r\n`);
    expect(await oversized.closed).toMatch(/^HTTP\/1\.1 431 /);
    expect(requests).toHaveLength(0);
    await health(base);
    expect(store.getStatus).not.toHaveBeenCalled();
    expect(store.search).not.toHaveBeenCalled();
  });

  test("Origin rejection still precedes upload/HTTP overload admission", async () => {
    const fixture = await backendFixture();
    const { handle, base } = await start({ maxHttpRequests: 1, maxPendingUploads: 1 }, fixture.store);
    const original = post(base, "/query", search);
    await until(() => fixture.responses.length === 1);
    const requests = watchRequests(handle);
    const rejected = upload(handle, "/mcp", "{", 65536, undefined, "Origin: null\r\n");
    expect(await rejected.closed).toMatch(/^HTTP\/1\.1 403 /);
    expect(requests[0]!.collectors).toBe(0);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    fixture.responses[0]!.end("ok");
    expect((await original).response.status).toBe(200);
  });
});
