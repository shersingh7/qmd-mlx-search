import { afterEach, describe, expect, test, vi } from "vitest";
import { createServer, type Server } from "node:http";
import { connect, type Socket } from "node:net";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import type { QMDStore } from "../src/index.js";
import { startMcpHttpServer, type HttpServerHandle, type HttpServerOptions } from "../src/mcp/server.js";

// Model-free, owned loopback listeners only. Cancellation must reach a real
// downstream HTTP operation, not just stop awaiting a manually released mock.
const handles: HttpServerHandle[] = [];
const backends: Server[] = [];
const sockets: Socket[] = [];
const status = { totalDocuments: 0, needsEmbedding: 0, hasVectorIndex: false, collections: [] };
const searchBody = { searches: [{ type: "lex", query: "synthetic" }], rerank: false };
const headers = { "Content-Type": "application/json", Accept: "application/json, text/event-stream" };
const cancellationCases = ["/query", "/search", "/mcp"].flatMap(path =>
  ["disconnect", "deadline", "stop"].map(cause => ({ path, cause })),
);

function fixtureStore(): QMDStore {
  return {
    getDefaultCollectionNames: vi.fn(async () => []),
    getStatus: vi.fn(async () => status),
    listContexts: vi.fn(async () => []),
    getGlobalContext: vi.fn(async () => undefined),
    search: vi.fn(async () => []),
    close: vi.fn(async () => {}),
  } as unknown as QMDStore;
}

async function start(store = fixtureStore(), options: HttpServerOptions = {}) {
  const handle = await startMcpHttpServer(0, { quiet: true, store, ...options });
  handles.push(handle);
  return { handle, base: `http://localhost:${handle.port}`, store };
}

async function until(check: () => boolean, timeoutMs = 1000) {
  const deadline = performance.now() + timeoutMs;
  while (!check() && performance.now() < deadline) await new Promise(resolve => setTimeout(resolve, 5));
  expect(check()).toBe(true);
}

async function post(base: string, body: unknown, sid?: string) {
  return fetch(base + "/mcp", {
    method: "POST", headers: { ...headers, ...(sid ? { "mcp-session-id": sid } : {}) },
    body: JSON.stringify(body), signal: AbortSignal.timeout(1500),
  });
}

async function initialize(base: string) {
  const response = await post(base, {
    jsonrpc: "2.0", id: 1, method: "initialize",
    params: { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "runtime-review", version: "1" } },
  });
  expect(response.status).toBe(200);
  await response.text();
  const sid = response.headers.get("mcp-session-id");
  expect(sid).toBeTruthy();
  return sid!;
}

function blockedPost(port: number, path: string, sid?: string) {
  const body = JSON.stringify(path === "/mcp" ? {
    jsonrpc: "2.0", id: 23, method: "tools/call", params: { name: "query", arguments: searchBody },
  } : searchBody);
  let received = "";
  let socketError: Error | undefined;
  const socket = connect(port, "localhost", () => {
    socket.write(`POST ${path} HTTP/1.1\r\nHost: localhost:${port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n${sid ? `mcp-session-id: ${sid}\r\n` : ""}Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
  });
  sockets.push(socket);
  socket.on("data", chunk => { received += chunk.toString(); });
  socket.on("error", error => { socketError = error; });
  const closed = new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => socket.destroy(new Error("Owned POST did not close within 1500ms")), 1500);
    socket.once("close", () => {
      clearTimeout(timer);
      if (socketError && (socketError as NodeJS.ErrnoException).code !== "ECONNRESET") reject(socketError);
      else resolve(received);
    });
  });
  // Attach rejection immediately, even while awaiting the backend to start.
  void closed.catch(() => {});
  return { socket, closed };
}

async function cancellableBackend() {
  const state = { requests: 0, closed: 0, settled: false, signal: undefined as AbortSignal | undefined };
  const backend = createServer((_req, res) => {
    state.requests++;
    res.once("close", () => { state.closed++; });
    // Deliberately never return headers/body: only cancellation can finish the
    // store's actual HTTP request. Teardown happens after all assertions.
  });
  backends.push(backend);
  await new Promise<void>((resolve, reject) => {
    backend.once("error", reject);
    backend.listen(0, "127.0.0.1", resolve);
  });
  const port = (backend.address() as import("node:net").AddressInfo).port;
  const store = fixtureStore();
  vi.spyOn(store, "search").mockImplementation(async ({ signal }) => {
    if (!signal) throw new Error("Missing request/session cancellation signal");
    state.signal = signal;
    try {
      const response = await fetch(`http://127.0.0.1:${port}/owned-search`, { signal });
      await response.text();
      return [];
    } finally { state.settled = true; }
  });
  return { store, state };
}

async function assertSearchCancelled(state: Awaited<ReturnType<typeof cancellableBackend>>["state"]) {
  await until(() => state.settled && state.closed === 1);
  expect(state.signal).toBeInstanceOf(AbortSignal);
  expect(state.signal!.aborted).toBe(true);
  expect(state.requests).toBe(1);
}

async function health(base: string) {
  const response = await fetch(base + "/health", { signal: AbortSignal.timeout(1000) });
  expect(response.status).toBe(200);
  return await response.json() as { status: string; uptime: number; pid: number; instanceToken: string };
}

function restoreToken(value: string | undefined) {
  if (value === undefined) delete process.env.QMD_MCP_INSTANCE_TOKEN;
  else process.env.QMD_MCP_INSTANCE_TOKEN = value;
}

afterEach(async () => {
  for (const socket of sockets.splice(0)) socket.destroy();
  await Promise.all(handles.splice(0).map(handle => handle.stop()));
  await Promise.all(backends.splice(0).map(backend => new Promise<void>((resolve, reject) => {
    backend.close(error => error ? reject(error) : resolve());
    backend.closeAllConnections();
  })));
  vi.restoreAllMocks();
});

if (process.versions.bun) {
  test("Bun entrypoint verifies cancellation and listener identity on the required Node HTTP runtime", async () => {
    const { stdout } = await promisify(execFile)("node", [
      "node_modules/vitest/vitest.mjs", "run", "--pool=threads", "--maxWorkers=1",
      "--cache=false", "--testTimeout=4000", "--reporter=json", "test/audit-mcp-runtime-review.test.ts",
    ], { cwd: fileURLToPath(new URL("../", import.meta.url)), env: { ...process.env, CI: "true", QMD_LLAMA_GPU: "false" }, timeout: 15000, killSignal: "SIGKILL", maxBuffer: 4 * 1024 * 1024 });
    const report = JSON.parse(stdout);
    expect(report.numTotalTests).toBe(cancellationCases.length + 5);
    expect(report.numPassedTests).toBe(report.numTotalTests);
    expect(report.numFailedTests).toBe(0);
    expect(report.numPendingTests).toBe(0);
    console.log(`Node cancellation/identity assertions: ${report.numPassedTests}/${report.numTotalTests} passed`);
  }, 20000);
} else describe("MCP HTTP runtime cancellation and ownership review", () => {
  test.each(cancellationCases)("$path $cause aborts the search signal and its owned downstream HTTP call", async ({ path, cause }) => {
    const { store, state } = await cancellableBackend();
    const { handle, base } = await start(store, { requestTimeoutMs: cause === "deadline" ? 200 : 2000 });
    const sid = path === "/mcp" ? await initialize(base) : undefined;
    const request = blockedPost(handle.port, path, sid);
    await until(() => state.requests === 1);
    expect(state.signal!.aborted).toBe(false);
    if (cause === "disconnect") request.socket.destroy();
    else if (cause === "stop") await handle.stop();
    else expect(await request.closed).toMatch(/^HTTP\/1\.1 504 /);
    await assertSearchCancelled(state);
    await request.closed;
    expect(store.close).not.toHaveBeenCalled(); // Injected store remains caller-owned.
    if (cause === "stop") expect(handle.httpServer.listening).toBe(false);
    else if (sid) {
      const stale = await post(base, { jsonrpc: "2.0", id: 24, method: "tools/call", params: { name: "status", arguments: {} } }, sid);
      expect(stale.status).toBe(404);
      await stale.text();
    }
  });

  test("SDK notifications/cancelled reaches the tool callback signal without closing its session", async () => {
    const { store, state } = await cancellableBackend();
    const { base, handle } = await start(store, { requestTimeoutMs: 2000 });
    const sid = await initialize(base);
    const request = blockedPost(handle.port, "/mcp", sid);
    await until(() => state.requests === 1);
    const cancelled = await post(base, {
      jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 23, reason: "offline cancellation audit" },
    }, sid);
    expect(cancelled.status).toBe(202);
    await cancelled.text();
    await assertSearchCancelled(state);
    const stillLive = await post(base, { jsonrpc: "2.0", id: 24, method: "tools/call", params: { name: "status", arguments: {} } }, sid);
    expect(stillLive.status).toBe(200);
    expect((await stillLive.json() as any).result.structuredContent.totalDocuments).toBe(0);
    request.socket.destroy(); // SDK suppresses replies to cancelled request IDs.
    await request.closed;
  });

  test("DELETE closes the session and cancels its actual downstream search HTTP call", async () => {
    const { store, state } = await cancellableBackend();
    const { base, handle } = await start(store);
    const sid = await initialize(base);
    const request = blockedPost(handle.port, "/mcp", sid);
    await until(() => state.requests === 1);
    const deleted = await fetch(base + "/mcp", { method: "DELETE", headers: { "mcp-session-id": sid }, signal: AbortSignal.timeout(1000) });
    expect(deleted.status).toBe(200);
    await deleted.text();
    await assertSearchCancelled(state);
    await request.closed;
  });

  test("health exposes the process PID and an explicit stable per-listener ownership token", async () => {
    const original = process.env.QMD_MCP_INSTANCE_TOKEN;
    try {
      process.env.QMD_MCP_INSTANCE_TOKEN = "environment-owner";
      const { base } = await start(fixtureStore(), { instanceToken: "explicit-owner" });
      for (const result of [await health(base), await health(base)]) {
        expect(result.status).toBe("ok");
        expect(result.pid).toBe(process.pid);
        expect(result.instanceToken).toBe("explicit-owner");
        expect(result.uptime).toBeGreaterThanOrEqual(0);
      }
    } finally { restoreToken(original); }
  });

  test("environment ownership token is captured once, not reread on each health request", async () => {
    const original = process.env.QMD_MCP_INSTANCE_TOKEN;
    try {
      process.env.QMD_MCP_INSTANCE_TOKEN = "first-owner";
      const a = await start();
      process.env.QMD_MCP_INSTANCE_TOKEN = "second-owner";
      const b = await start();
      expect((await health(a.base)).instanceToken).toBe("first-owner");
      expect((await health(b.base)).instanceToken).toBe("second-owner");
      expect((await health(a.base)).instanceToken).toBe("first-owner");
    } finally { restoreToken(original); }
  });

  test("listeners without a supplied token generate distinct stable UUID ownership markers", async () => {
    const original = process.env.QMD_MCP_INSTANCE_TOKEN;
    try {
      delete process.env.QMD_MCP_INSTANCE_TOKEN;
      const a = await start();
      const b = await start();
      const first = await health(a.base);
      const second = await health(b.base);
      expect(first.instanceToken).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
      expect(second.instanceToken).not.toBe(first.instanceToken);
      expect((await health(a.base)).instanceToken).toBe(first.instanceToken);
      expect((await health(b.base)).instanceToken).toBe(second.instanceToken);
      expect(first.pid).toBe(process.pid);
      expect(second.pid).toBe(process.pid);
    } finally { restoreToken(original); }
  });
});
