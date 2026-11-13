import { afterEach, describe, expect, test, vi } from "vitest";
import { createServer, type Server, type ServerResponse } from "node:http";
import { connect, type Socket } from "node:net";
import { PassThrough } from "node:stream";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import * as index from "../src/index.js";
import type { QMDStore } from "../src/index.js";
import { startMcpHttpServer, startMcpServer, type HttpServerHandle, type HttpServerOptions } from "../src/mcp/server.js";

// All sockets and stores belong to these model-free fixtures. Run the HTTP
// contract on Node even when Bun invokes the suite; never skip its assertions.
const handles: HttpServerHandle[] = [];
const backends: Server[] = [];
const sockets: Socket[] = [];
const stdioServers: McpServer[] = [];
const headers = { "Content-Type": "application/json", Accept: "application/json, text/event-stream" };
const status = { totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] };
const ids = [0, "", 1, "1"] as const;
const causes = ["disconnect", "DELETE", "deadline", "stop"] as const;

function query(id: number | string, text = "synthetic") {
  return { jsonrpc: "2.0", id, method: "tools/call", params: {
    name: "query", arguments: { searches: [{ type: "lex", query: text }], rerank: false },
  } };
}
function cancellation(requestId: number | string) {
  return { jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId, reason: "owned cancellation audit" } };
}
function fixtureStore(): QMDStore {
  return {
    getDefaultCollectionNames: vi.fn(async () => []), getStatus: vi.fn(async () => status),
    listContexts: vi.fn(async () => []), getGlobalContext: vi.fn(async () => undefined),
    search: vi.fn(async () => []), close: vi.fn(async () => {}),
  } as unknown as QMDStore;
}
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
async function until(check: () => boolean, timeout = 1000) {
  const deadline = performance.now() + timeout;
  while (!check() && performance.now() < deadline) await delay(5);
  expect(check()).toBe(true);
}
function post(base: string, body: unknown, sid?: string) {
  const work = (async () => {
    const response = await fetch(base + "/mcp", {
      method: "POST", headers: { ...headers, ...(sid ? { "mcp-session-id": sid } : {}) },
      body: JSON.stringify(body), signal: AbortSignal.timeout(2000),
    });
    const text = await response.text();
    return { response, body: text ? JSON.parse(text) : undefined };
  })();
  // A failed assertion may leave this deliberately blocked call pending until
  // teardown; attach rejection now, while retaining the rejecting await result.
  void work.catch(() => {});
  return work;
}
async function initialize(base: string) {
  const { response } = await post(base, {
    jsonrpc: "2.0", id: "initialize", method: "initialize", params: {
      protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "ownership-audit", version: "1" },
    },
  });
  expect(response.status).toBe(200);
  const sid = response.headers.get("mcp-session-id");
  expect(sid).toBeTruthy();
  return sid!;
}

// Unlike a fetch timeout, this proves the server finishes/closes the original
// HTTP socket on its own. A fixture watchdog is a failure, not successful EOF.
function wirePost(handle: HttpServerHandle, sid: string, body: unknown) {
  const text = JSON.stringify(body);
  let received = "";
  let error: NodeJS.ErrnoException | undefined;
  const socket = connect(handle.port, "localhost", () => socket.write(
    `POST /mcp HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nmcp-session-id: ${sid}\r\nContent-Length: ${Buffer.byteLength(text)}\r\nConnection: close\r\n\r\n${text}`,
  ));
  sockets.push(socket);
  socket.on("data", chunk => { received += chunk.toString(); });
  socket.on("error", e => { error = e; });
  const closed = new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => socket.destroy(new Error("Original MCP socket did not settle")), 1500);
    socket.once("close", () => {
      clearTimeout(timer);
      if (error && error.code !== "ECONNRESET") reject(error);
      else resolve(received);
    });
  });
  void closed.catch(() => {});
  return { socket, closed };
}

async function backendFixture() {
  const responses: ServerResponse[] = [];
  const operations: { signal: AbortSignal; settled: boolean; text: string }[] = [];
  const state = { closed: 0 };
  const backend = createServer((_req, res) => {
    responses.push(res);
    res.once("close", () => { state.closed++; });
  });
  backends.push(backend);
  await new Promise<void>(resolve => backend.listen(0, "127.0.0.1", resolve));
  const port = (backend.address() as import("node:net").AddressInfo).port;
  const store = fixtureStore();
  let cleanupGate: Promise<void> | undefined;
  vi.spyOn(store, "search").mockImplementation(async ({ signal, queries }) => {
    if (!signal) throw new Error("Missing real request cancellation signal");
    const operation = { signal, settled: false, text: queries![0]!.query };
    operations.push(operation);
    try {
      const response = await fetch(`http://127.0.0.1:${port}/owned-search`, { signal });
      await response.text();
      return [];
    } finally {
      if (cleanupGate) await cleanupGate;
      operation.settled = true;
    }
  });
  return { store, responses, operations, state, holdCleanup(gate: Promise<void>) { cleanupGate = gate; } };
}
async function start(fixture: Awaited<ReturnType<typeof backendFixture>>, requestTimeoutMs = 3000, options: HttpServerOptions = {}) {
  const handle = await startMcpHttpServer(0, { quiet: true, store: fixture.store, requestTimeoutMs, ...options });
  handles.push(handle);
  const base = `http://localhost:${handle.port}`;
  return { handle, base, sid: await initialize(base) };
}
async function rejectedDuplicate(base: string, sid: string, body: unknown) {
  const duplicate = await post(base, body, sid);
  expect(duplicate.response.status).toBe(409);
  expect(duplicate.body).toEqual({
    jsonrpc: "2.0", id: null, error: { code: -32600, message: "Duplicate active JSON-RPC request ID" },
  });
}
async function assertCancelled(fixture: Awaited<ReturnType<typeof backendFixture>>, count: number) {
  await until(() => fixture.operations.every(op => op.settled) && fixture.state.closed === count);
  expect(fixture.operations).toHaveLength(count);
  expect(fixture.operations.every(op => op.signal.aborted)).toBe(true);
}

async function stdioFixture(store: QMDStore) {
  const input = new PassThrough();
  const output = new PassThrough();
  const transport = new StdioServerTransport(input, output);
  const messages: any[] = [];
  let buffer = "";
  output.on("data", chunk => {
    buffer += chunk.toString();
    let end: number;
    while ((end = buffer.indexOf("\n")) >= 0) {
      messages.push(JSON.parse(buffer.slice(0, end))); buffer = buffer.slice(end + 1);
    }
  });
  vi.spyOn(index, "createStore").mockResolvedValue(store);
  vi.spyOn(index, "getDefaultDbPath").mockReturnValue(":memory:");
  const originalConnect = McpServer.prototype.connect;
  vi.spyOn(McpServer.prototype, "connect").mockImplementation(function(this: McpServer, wrapped) {
    stdioServers.push(this);
    // Keep the application's ownership decorator and real SDK framing. Replace
    // only the process stdin/stdout endpoints with this test's controlled streams.
    const start = StdioServerTransport.prototype.start;
    const send = StdioServerTransport.prototype.send;
    const close = StdioServerTransport.prototype.close;
    vi.spyOn(StdioServerTransport.prototype, "start").mockImplementation(function(this: StdioServerTransport) {
      if (this === transport) return start.call(this);
      transport.onmessage = message => this.onmessage?.(message);
      transport.onclose = () => this.onclose?.();
      transport.onerror = error => this.onerror?.(error);
      return start.call(transport);
    });
    vi.spyOn(StdioServerTransport.prototype, "send").mockImplementation(message => send.call(transport, message));
    vi.spyOn(StdioServerTransport.prototype, "close").mockImplementation(() => close.call(transport));
    return originalConnect.call(this, wrapped);
  });
  await startMcpServer();
  return { messages, write(message: unknown) { input.write(JSON.stringify(message) + "\n"); },
    close() { input.destroy(); output.destroy(); } };
}

afterEach(async () => {
  for (const socket of sockets.splice(0)) socket.destroy();
  await Promise.all(handles.splice(0).map(handle => handle.stop()));
  await Promise.all(stdioServers.splice(0).map(server => server.close()));
  await Promise.all(backends.splice(0).map(backend => new Promise<void>(resolve => {
    backend.close(() => resolve()); backend.closeAllConnections();
  })));
  vi.restoreAllMocks();
});

if (process.versions.bun) {
  test("Bun entrypoint executes every request-ownership assertion on the supported Node HTTP runtime", async () => {
    const { stdout } = await promisify(execFile)("node", [
      "node_modules/vitest/vitest.mjs", "run", "--pool=threads", "--maxWorkers=1", "--minWorkers=1",
      "--cache=false", "--reporter=json", "test/audit-mcp-request-ownership.test.ts",
    ], { cwd: fileURLToPath(new URL("../", import.meta.url)), env: { ...process.env, CI: "true", QMD_LLAMA_GPU: "false" },
      timeout: 30000, killSignal: "SIGKILL", maxBuffer: 4 * 1024 * 1024 });
    const report = JSON.parse(stdout);
    expect(report.numTotalTests).toBe(32);
    expect(report.numPassedTests).toBe(report.numTotalTests);
    expect(report.numFailedTests).toBe(0);
    expect(report.numPendingTests).toBe(0);
  }, 35000);

  test.each([0, ""])("Bun native stdio cancels literal ID %j with the shared factory", async id => {
    const store = fixtureStore();
    let signal: AbortSignal | undefined;
    let settled = false;
    vi.spyOn(store, "search").mockImplementation(async options => {
      signal = options.signal;
      if (!signal) throw new Error("Missing native Bun stdio cancellation signal");
      const ownedSignal = signal;
      try {
        await new Promise<void>((_resolve, reject) => {
          ownedSignal.addEventListener("abort", () => reject(ownedSignal.reason), { once: true });
          ownedSignal.throwIfAborted();
        });
        return [];
      } finally { settled = true; }
    });
    const stdio = await stdioFixture(store);
    stdio.write(query(id));
    await until(() => !!signal);
    stdio.write(cancellation(id));
    await until(() => settled && stdio.messages.some(message => message.id === id && message.result?.isError));
    expect(signal!.aborted).toBe(true);
    stdio.close();
  });
} else describe("MCP request ID ownership through the real SDK", () => {
  test.each(ids)("explicit cancellation preserves literal ID %j and closes both owned sockets", async id => {
    const fixture = await backendFixture();
    const { handle, base, sid } = await start(fixture);
    const request = wirePost(handle, sid, query(id));
    await until(() => fixture.responses.length === 1);
    expect((await post(base, cancellation(id), sid)).response.status).toBe(202);
    await assertCancelled(fixture, 1);
    expect(await request.closed).toMatch(/^HTTP\/1\.1 200 /);
    // A cancelled QMD tool returns its normal SDK isError result; its session
    // stays live, and the ID is reusable only after work and response cleanup.
    const reused = post(base, query(id), sid);
    await until(() => fixture.responses.length === 2);
    fixture.responses[1]!.end("ok");
    const result = await reused;
    expect(result.response.status).toBe(200);
    expect(result.body.id).toBe(id);
    expect(result.body.result.isError).not.toBe(true);
  });

  test.each(ids)("two concurrent HTTP calls cannot overwrite the active owner of %j", async id => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture);
    const original = post(base, query(id, "original"), sid);
    await until(() => fixture.responses.length === 1);
    await rejectedDuplicate(base, sid, query(id, "must not start"));
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    fixture.responses[0]!.end("complete original");
    const result = await original;
    expect(result.body.id).toBe(id);
    expect(result.body.result.isError).not.toBe(true);
    const reused = post(base, query(id, "reuse"), sid);
    await until(() => fixture.responses.length === 2);
    fixture.responses[1]!.end("complete reuse");
    expect((await reused).body.id).toBe(id);
  });

  test.each(ids)("prevalidates a complete batch with duplicate ID %j before starting any member", async id => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture);
    await rejectedDuplicate(base, sid, [query("unique before collision"), query(id), query(id)]);
    expect(fixture.operations).toHaveLength(0);
    // No partial reservations survive rejection either.
    const accepted = post(base, query("unique before collision"), sid);
    await until(() => fixture.responses.length === 1);
    fixture.responses[0]!.end("ok");
    expect((await accepted).response.status).toBe(200);
  });

  test("a duplicate pipelined on the original TCP connection cannot destroy the original owner", async () => {
    const fixture = await backendFixture();
    const { handle, sid } = await start(fixture);
    const frame = (text: string, close: boolean) => {
      const body = JSON.stringify(query(0, text));
      return `POST /mcp HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nmcp-session-id: ${sid}\r\nContent-Length: ${Buffer.byteLength(body)}\r\nConnection: ${close ? "close" : "keep-alive"}\r\n\r\n${body}`;
    };
    const socket = connect(handle.port, "localhost", () => socket.write(frame("original", false)));
    sockets.push(socket);
    let wire = "";
    socket.on("data", chunk => { wire += chunk.toString(); });
    const closed = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => socket.destroy(new Error("Pipelined MCP fixture did not settle")), 1500);
      socket.once("close", () => { clearTimeout(timer); resolve(); });
      socket.once("error", reject);
    });
    void closed.catch(() => {});
    await until(() => fixture.responses.length === 1);
    socket.write(frame("duplicate must not start", true));
    await delay(100); // Past the untrusted-body error socket-destruction window.
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    fixture.responses[0]!.end("complete original");
    await closed;
    expect(wire.match(/HTTP\/1\.1 \d{3} /g)).toEqual(["HTTP/1.1 200 ", "HTTP/1.1 409 "]);
    expect(wire).toContain('"id":0');
    expect(wire).toContain('"code":-32600');
  });

  test("a batch colliding with an active HTTP call has no partial dispatch or partial reservations", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture);
    const original = post(base, query(0), sid);
    await until(() => fixture.responses.length === 1);
    await rejectedDuplicate(base, sid, [cancellation(0), query("unreserved"), query(0)]);
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    const accepted = post(base, query("unreserved"), sid);
    await until(() => fixture.responses.length === 2);
    fixture.responses.forEach(res => res.end("ok"));
    expect((await original).body.id).toBe(0);
    expect((await accepted).body.id).toBe("unreserved");
  });

  test("0, empty string, numeric 1 and string '1' are independent owners in one batch", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture);
    const batch = post(base, ids.map((id, i) => query(id, `operation ${i}`)), sid);
    await until(() => fixture.responses.length === ids.length);
    await post(base, cancellation(1), sid);
    await until(() => fixture.operations[2]!.settled);
    expect(fixture.operations.map(op => op.signal.aborted)).toEqual([false, false, true, false]);
    fixture.responses.forEach(res => { if (!res.destroyed) res.end("ok"); });
    const result = await batch;
    expect(result.response.status).toBe(200);
    expect(result.body.map((message: { id: number | string }) => message.id)).toEqual(ids);
    expect(result.body[2].result.isError).toBe(true);
    expect(result.body[3].result.isError).not.toBe(true);
  });

  test("same-batch zero-ID cancellation before the query callback starts loses neither owner nor response", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture);
    const result = await post(base, [query(0), cancellation(0)], sid);
    expect(result.response.status).toBe(200);
    expect(result.body.id).toBe(0);
    expect(result.body.result.isError).toBe(true);
    expect(fixture.operations).toHaveLength(0);
  });

  test("a fast batch member stays reserved until the whole HTTP response is cleaned up", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture);
    const batch = post(base, [query(0), query("")], sid);
    await until(() => fixture.responses.length === 2);
    fixture.responses[0]!.end("first done");
    await until(() => fixture.operations[0]!.settled);
    await rejectedDuplicate(base, sid, query(0));
    expect(fixture.operations).toHaveLength(2);
    fixture.responses[1]!.end("second done");
    expect((await batch).response.status).toBe(200);
    const reused = post(base, query(0), sid);
    await until(() => fixture.responses.length === 3);
    fixture.responses[2]!.end("reused");
    expect((await reused).body.id).toBe(0);
  });

  test("explicit cancellation does not release an ID while its real operation is still cleaning up", async () => {
    const fixture = await backendFixture();
    let release!: () => void;
    fixture.holdCleanup(new Promise<void>(resolve => { release = resolve; }));
    try {
      const { base, sid } = await start(fixture);
      const original = post(base, query(0), sid);
      await until(() => fixture.responses.length === 1);
      await post(base, cancellation(0), sid);
      await until(() => fixture.state.closed === 1);
      expect(fixture.operations[0]!.settled).toBe(false);
      await rejectedDuplicate(base, sid, query(0));
      release();
      const result = await original;
      expect(result.body.id).toBe(0);
      expect(result.body.result.isError).toBe(true);
    } finally { release(); }
  });

  test.each(causes)("%s after a rejected collision aborts every accepted batch operation and closes sockets", async cause => {
    const fixture = await backendFixture();
    const { handle, base, sid } = await start(fixture, cause === "deadline" ? 250 : 3000, {
      maxActiveRequestsPerSession: 2, maxActiveRequests: 2,
    });
    const original = wirePost(handle, sid, [query(0), query("")]);
    await until(() => fixture.responses.length === 2);
    await rejectedDuplicate(base, sid, query(0));
    expect(fixture.operations).toHaveLength(2);
    expect(fixture.operations.every(op => !op.signal.aborted)).toBe(true);
    if (cause === "disconnect") original.socket.destroy();
    else if (cause === "DELETE") {
      const deleted = await fetch(base + "/mcp", { method: "DELETE", headers: { "mcp-session-id": sid }, signal: AbortSignal.timeout(1000) });
      expect(deleted.status).toBe(200); await deleted.text();
    } else if (cause === "stop") {
      const before = performance.now();
      await Promise.all([handle.stop(), handle.stop()]);
      expect(performance.now() - before).toBeLessThan(1000);
    }
    const wire = await original.closed;
    if (cause === "deadline") expect(wire).toMatch(/^HTTP\/1\.1 504 /);
    await assertCancelled(fixture, 2);
    expect(fixture.store.close).not.toHaveBeenCalled();
    if (cause !== "stop") {
      const freshSid = await initialize(base); // Closed owners returned global capacity.
      const reused = post(base, query(0), freshSid);
      await until(() => fixture.responses.length === 3);
      fixture.responses[2]!.end("fresh session");
      expect((await reused).response.status).toBe(200);
    }
  });

  test("per-session admission counts existing owners plus the complete incoming batch", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture, 3000, { maxActiveRequestsPerSession: 3, maxActiveRequests: 10 });
    const original = post(base, query(0), sid);
    await until(() => fixture.responses.length === 1);
    const rejected = await post(base, [query("a"), query("b"), query("c")], sid);
    expect(rejected.response.status).toBe(429);
    expect(rejected.body).toEqual({ jsonrpc: "2.0", id: null, error: {
      code: -32000, message: "Session active request capacity reached; retry later",
    } });
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    const batch = post(base, [query("a"), query("b")], sid);
    await until(() => fixture.responses.length === 3);
    expect((await post(base, query("c"), sid)).response.status).toBe(429);
    expect(fixture.operations).toHaveLength(3);
    fixture.responses.forEach(res => res.end("completed"));
    expect((await original).response.status).toBe(200);
    expect((await batch).response.status).toBe(200);
    const reused = post(base, query("c"), sid);
    await until(() => fixture.responses.length === 4);
    fixture.responses[3]!.end("reused capacity");
    expect((await reused).response.status).toBe(200);
  });

  test("an oversized batch consumes no per-session capacity and starts no work", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture, 3000, { maxActiveRequestsPerSession: 1 });
    expect((await post(base, [query(0), query("")], sid)).response.status).toBe(429);
    expect(fixture.operations).toHaveLength(0);
    const accepted = post(base, query(0), sid);
    await until(() => fixture.responses.length === 1);
    fixture.responses[0]!.end("ok");
    expect((await accepted).response.status).toBe(200);
  });

  test("global admission bounds different sessions without merging their literal IDs", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture, 3000, { maxActiveRequests: 2 });
    const secondSid = await initialize(base);
    const thirdSid = await initialize(base);
    const first = post(base, query(0), sid);
    await until(() => fixture.responses.length === 1);
    const second = post(base, query(0), secondSid);
    await until(() => fixture.responses.length === 2);
    const rejected = await post(base, [query(0), query("")], thirdSid);
    expect(rejected.response.status).toBe(503);
    expect(rejected.body.error).toEqual({ code: -32000, message: "Server active request capacity reached; retry later" });
    expect(fixture.operations).toHaveLength(2);
    expect(fixture.operations.every(op => !op.signal.aborted)).toBe(true);
    await post(base, cancellation(0), sid); // Notifications remain usable at capacity.
    expect((await first).body.result.isError).toBe(true);
    const third = post(base, query(0), thirdSid);
    await until(() => fixture.responses.length === 3);
    expect(fixture.operations[1]!.signal.aborted).toBe(false);
    fixture.responses[1]!.end("second"); fixture.responses[2]!.end("third");
    expect((await second).body.id).toBe(0);
    expect((await third).body.id).toBe(0);
  });

  test("global overflow during initialization releases its session reservation without touching active work", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture, 3000, { maxActiveRequests: 1, maxSessions: 2 });
    const original = post(base, query(0), sid);
    await until(() => fixture.responses.length === 1);
    for (let i = 0; i < 3; i++) {
      const result = await post(base, {
        jsonrpc: "2.0", id: "init overflow", method: "initialize", params: {
          protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "overflow", version: "1" },
        },
      });
      expect(result.response.status).toBe(503);
      expect(result.body.error.message).toBe("Server active request capacity reached; retry later");
    }
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    fixture.responses[0]!.end("ok");
    expect((await original).response.status).toBe(200);
    await initialize(base);
  });

  test("SDK rejection, invalid batches and transport validation return all reserved capacity", async () => {
    const fixture = await backendFixture();
    const { base, sid } = await start(fixture, 3000, { maxActiveRequestsPerSession: 1, maxActiveRequests: 1 });
    const unknown = await post(base, { jsonrpc: "2.0", id: 0, method: "not/a/method" }, sid);
    expect(unknown.response.status).toBe(200);
    expect(unknown.body.error.code).toBe(-32601);
    const invalid = await post(base, [query(0), { jsonrpc: "1.0" }], sid);
    expect(invalid.response.status).toBe(400);
    const badVersion = await fetch(base + "/mcp", { method: "POST", headers: {
      ...headers, "mcp-session-id": sid, "mcp-protocol-version": "invalid",
    }, body: JSON.stringify(query(0)), signal: AbortSignal.timeout(1000) });
    expect(badVersion.status).toBe(400); await badVersion.text();
    expect(fixture.operations).toHaveLength(0);
    const accepted = post(base, query(0), sid);
    await until(() => fixture.responses.length === 1);
    fixture.responses[0]!.end("after errors");
    expect((await accepted).response.status).toBe(200);
  });

  test("failed SDK startup returns the initial request's global capacity", async () => {
    const fixture = await backendFixture();
    const handle = await startMcpHttpServer(0, { store: fixture.store, quiet: true, maxActiveRequests: 1, maxSessions: 1 });
    handles.push(handle);
    const base = `http://localhost:${handle.port}`;
    vi.spyOn(WebStandardStreamableHTTPServerTransport.prototype, "start").mockRejectedValueOnce(new Error("owned connect failure"));
    await expect(initialize(base)).rejects.toThrow();
    await initialize(base);
  });

  test("the same literal request ID is valid in different sessions and cancellation stays session-local", async () => {
    const fixture = await backendFixture();
    const { handle, base, sid } = await start(fixture);
    const otherSid = await initialize(base);
    const first = wirePost(handle, sid, query(0));
    await until(() => fixture.responses.length === 1);
    const second = wirePost(handle, otherSid, query(0));
    await until(() => fixture.responses.length === 2);
    await post(base, cancellation(0), sid);
    await until(() => fixture.operations[0]!.settled);
    expect(fixture.operations[1]!.signal.aborted).toBe(false);
    expect(await first.closed).toMatch(/^HTTP\/1\.1 200 /);
    await handle.stop();
    await assertCancelled(fixture, 2);
    await second.closed;
  });

  test.each([0, ""])("shared stdio factory cancels ID %j without the SDK falsy-ID check", async id => {
    const fixture = await backendFixture();
    const stdio = await stdioFixture(fixture.store);
    const { messages } = stdio;
    stdio.write(query(id));
    await until(() => fixture.responses.length === 1);
    stdio.write(query(id, "duplicate must not start"));
    await until(() => messages.some(message => message.error?.code === -32600));
    expect(messages.find(message => message.error?.code === -32600)).toEqual({
      jsonrpc: "2.0", error: { code: -32600, message: "Duplicate active JSON-RPC request ID" },
    });
    expect(fixture.operations).toHaveLength(1);
    expect(fixture.operations[0]!.signal.aborted).toBe(false);
    stdio.write(cancellation(id));
    await assertCancelled(fixture, 1);
    await until(() => messages.some(message => message.id === id && message.result?.isError === true));
    stdio.write(query(id));
    await until(() => fixture.responses.length === 2);
    fixture.responses[1]!.end("ok");
    await until(() => messages.some(message => message.id === id && message.result && !message.result.isError));
    stdio.close();
  });

  test("stdio active-ID retention is bounded and transport close aborts every admitted owner", async () => {
    const store = fixtureStore();
    const signals: AbortSignal[] = [];
    let settled = 0;
    vi.spyOn(store, "search").mockImplementation(async ({ signal }) => {
      if (!signal) throw new Error("Missing bounded stdio cancellation signal");
      signals.push(signal);
      try {
        await new Promise<void>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), { once: true });
          signal.throwIfAborted();
        });
        return [];
      } finally { settled++; }
    });
    const stdio = await stdioFixture(store);
    for (let id = 0; id < 129; id++) stdio.write(query(id));
    await until(() => signals.length === 128 && stdio.messages.some(message => message.error?.code === -32000));
    expect(stdio.messages.find(message => message.error?.code === -32000)).toEqual({ jsonrpc: "2.0", error: {
      code: -32000, message: "Session active request capacity reached; retry later",
    } });
    expect(signals.every(signal => !signal.aborted)).toBe(true);
    await stdioServers[0]!.close();
    await until(() => settled === 128);
    expect(signals.every(signal => signal.aborted)).toBe(true);
    expect(stdioServers[0]!.server.transport).toBeUndefined();
    stdio.close();
  });
});
