import { afterEach, describe, expect, test, vi } from "vitest";
import { request as httpRequest, type ClientRequest, type IncomingMessage } from "node:http";
import { connect as netConnect, type Socket } from "node:net";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { createStore, type QMDStore } from "../src/index.js";
import { startMcpHttpServer, type HttpServerHandle, type HttpServerOptions } from "../src/mcp/server.js";

// These tests use owned ephemeral listeners and synthetic data only. Never skip
// them in CI: no inference, model downloads, live collections or services.
const handles: HttpServerHandle[] = [];
const clients: ClientRequest[] = [];
const sockets: Socket[] = [];
const stores: QMDStore[] = [];
const status = { totalDocuments: 1, needsEmbedding: 1, hasVectorIndex: false, collections: [] };
const initializeBody = {
  jsonrpc: "2.0", id: 1, method: "initialize",
  params: { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "offline-audit", version: "1" } },
};
const searchBody = { searches: [{ type: "lex", query: "synthetic" }], rerank: false };
const mcpHeaders = { "Content-Type": "application/json", Accept: "application/json, text/event-stream" };

function fixtureStore(): QMDStore {
  return {
    getDefaultCollectionNames: vi.fn(async () => ["docs"]),
    getStatus: vi.fn(async () => status),
    listContexts: vi.fn(async () => []),
    getGlobalContext: vi.fn(async () => undefined),
    search: vi.fn(async () => []),
    get: vi.fn(async () => ({ error: "not_found", similarFiles: [] })),
    close: vi.fn(async () => {}),
  } as unknown as QMDStore;
}

function pendingSearch(store: QMDStore) {
  const state = { signal: undefined as AbortSignal | undefined, stopped: false };
  vi.spyOn(store, "search").mockImplementationOnce(({ signal }) => {
    if (!signal) throw new Error("Search must receive the real request/session signal");
    state.signal = signal;
    return new Promise((_resolve, reject) => {
      const onAbort = () => { state.stopped = true; reject(signal.reason); };
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) onAbort();
    });
  });
  return state;
}

async function start(options: Partial<HttpServerOptions> = {}, store = fixtureStore()) {
  const handle = await startMcpHttpServer(0, { quiet: true, store, ...options });
  handles.push(handle);
  return { handle, store, base: `http://localhost:${handle.port}` };
}

async function jsonPost(base: string, path: string, body: unknown, headers: Record<string, string> = {}) {
  const response = await fetch(base + path, {
    method: "POST", headers: { ...mcpHeaders, ...headers },
    body: JSON.stringify(body), signal: AbortSignal.timeout(2000),
  });
  return { response, body: await response.json() as any };
}

async function initialize(base: string) {
  const result = await jsonPost(base, "/mcp", initializeBody);
  expect(result.response.status).toBe(200);
  expect(result.response.headers.get("content-type")).toContain("application/json");
  expect(result.body.result.serverInfo.name).toBe("qmd");
  const sid = result.response.headers.get("mcp-session-id");
  expect(sid).toBeTruthy();
  return sid!;
}

async function tool(base: string, sid: string, name: string, args: object = {}) {
  return jsonPost(base, "/mcp", {
    jsonrpc: "2.0", id: 2, method: "tools/call", params: { name, arguments: args },
  }, { "mcp-session-id": sid });
}

// Use actual wire bytes: Bun's node:http client normalizes request targets and
// does not read a response to a partial upload until req.end(). Neither behavior
// can test a server's early rejection/absolute upload deadline.
function raw(base: string, path: string, options: {
  method?: string; headers?: Record<string, string>; body?: string | Buffer; partial?: boolean;
} = {}): Promise<{ status: number; body: string; elapsedMs: number }> {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const url = new URL(base);
    const body = Buffer.from(options.body ?? "");
    const headers: Record<string, string> = {
      Host: url.host, Connection: "keep-alive", ...mcpHeaders, ...options.headers,
    };
    const chunked = Object.entries(headers).some(([key, value]) => key.toLowerCase() === "transfer-encoding" && value === "chunked");
    if (!chunked && !Object.keys(headers).some(key => key.toLowerCase() === "content-length")) {
      headers["Content-Length"] = String(body.length);
    }
    const socket = netConnect(Number(url.port), url.hostname, () => {
      const head = `${options.method ?? "POST"} ${path} HTTP/1.1\r\n${Object.entries(headers).map(([key, value]) => `${key}: ${value}\r\n`).join("")}\r\n`;
      const payload: Buffer[] = [Buffer.from(head)];
      if (chunked) {
        if (body.length) payload.push(Buffer.from(`${body.length.toString(16)}\r\n`), body, Buffer.from("\r\n"));
        if (!options.partial) payload.push(Buffer.from("0\r\n\r\n"));
      } else if (body.length) payload.push(body);
      socket.write(Buffer.concat(payload));
    });
    sockets.push(socket);
    const chunks: Buffer[] = [];
    const timer = setTimeout(() => socket.destroy(new Error("Wire fixture did not receive EOF within 1500ms")), 1500);
    socket.on("data", chunk => chunks.push(chunk));
    let socketError: (Error & { code?: string }) | undefined;
    socket.on("error", error => { socketError = error; });
    socket.once("close", () => {
      clearTimeout(timer);
      try {
        // A parser rejection can reset after sending its complete 400 response.
        // A local watchdog firing is NEVER a successful server-side close.
        if (socketError && socketError.code !== "ECONNRESET") throw socketError;
        const wire = Buffer.concat(chunks).toString();
        const boundary = wire.indexOf("\r\n\r\n");
        const head = wire.slice(0, boundary);
        const code = head.match(/^HTTP\/1\.1 (\d{3}) /)?.[1];
        if (boundary < 0 || !code) throw new Error(`Invalid HTTP response: ${wire}`);
        let decoded = wire.slice(boundary + 4);
        if (/\r\ntransfer-encoding: chunked(?:\r\n|$)/i.test(head)) {
          let remaining = decoded;
          decoded = "";
          while (true) {
            const end = remaining.indexOf("\r\n");
            const size = Number.parseInt(remaining.slice(0, end), 16);
            if (end < 0 || !Number.isSafeInteger(size)) throw new Error("Invalid chunked response");
            if (size === 0) break;
            decoded += remaining.slice(end + 2, end + 2 + size);
            remaining = remaining.slice(end + 2 + size + 2);
          }
        }
        resolve({ status: Number(code), body: decoded, elapsedMs: performance.now() - started });
      } catch (error) { reject(error); }
    });
  });
}

function openSse(base: string, sid: string): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    const req = httpRequest(base + "/mcp", {
      headers: { Accept: "text/event-stream", "mcp-session-id": sid }, agent: false,
    }, res => {
      res.on("error", () => {}); // Expected when the fixture disconnects/stops.
      resolve(res);
    });
    clients.push(req);
    req.on("error", reject);
    req.setTimeout(1500, () => req.destroy(new Error("SSE headers were not flushed")));
    req.end();
  });
}

async function until(check: () => boolean, timeout = 1500) {
  await retryAssertion(() => expect(check()).toBe(true), timeout);
}

// vi.waitFor/vi.mocked are not implemented by Bun 1.3.8's Vitest compatibility.
// Keep the same assertions and deadline rather than skipping runtime coverage.
async function retryAssertion(check: () => void | Promise<void>, timeout = 1500, interval = 10) {
  const deadline = performance.now() + timeout;
  while (true) {
    try { await check(); return; }
    catch (error) {
      if (performance.now() >= deadline) throw error;
      await new Promise(resolve => setTimeout(resolve, interval));
    }
  }
}

afterEach(async () => {
  for (const client of clients.splice(0)) client.destroy();
  for (const socket of sockets.splice(0)) socket.destroy();
  await Promise.all(handles.splice(0).map(handle => handle.stop()));
  await Promise.all(stores.splice(0).map(store => store.close()));
  vi.restoreAllMocks();
});

if (process.versions.bun) {
  test("Bun rejects unsafe HTTP startup before store/listener side effects and names the Node route", async () => {
    const store = fixtureStore();
    const sigint = process.listenerCount("SIGINT");
    const sigterm = process.listenerCount("SIGTERM");
    const started = performance.now();
    await expect(startMcpHttpServer(0, { store, quiet: true, bodyTimeoutMs: 40 })).rejects.toThrow(/QMD_RUNTIME=node qmd mcp --http/);
    expect(performance.now() - started).toBeLessThan(250);
    expect(store.getDefaultCollectionNames).not.toHaveBeenCalled();
    expect(store.getStatus).not.toHaveBeenCalled();
    expect(store.search).not.toHaveBeenCalled();
    expect(store.close).not.toHaveBeenCalled();
    expect(process.listenerCount("SIGINT")).toBe(sigint);
    expect(process.listenerCount("SIGTERM")).toBe(sigterm);
  });

  test("Bun CI entrypoint runs every HTTP safety assertion on the required Node runtime", async () => {
    const { stdout } = await promisify(execFile)("node", [
      "node_modules/vitest/vitest.mjs", "run", "--pool=threads", "--maxWorkers=1",
      "--cache=false", "--testTimeout=4000", "--reporter=json", "test/audit-mcp-http.test.ts",
    ], { cwd: fileURLToPath(new URL("../", import.meta.url)), env: { ...process.env, CI: "true", QMD_LLAMA_GPU: "false" }, timeout: 15000, killSignal: "SIGKILL", maxBuffer: 4 * 1024 * 1024 });
    const report = JSON.parse(stdout);
    expect(report.numTotalTests).toBe(27);
    expect(report.numPassedTests).toBe(27);
    expect(report.numFailedTests).toBe(0);
    expect(report.numPendingTests).toBe(0);
    console.log(`Node HTTP safety assertions: ${report.numPassedTests}/${report.numTotalTests} passed`);
  }, 20000);
} else describe("offline MCP/REST HTTP safety audit", () => {
  test("real HTTP initialize, lex query, get and REST aliases use a synthetic SQLite store", async () => {
    const store = await createStore({ dbPath: ":memory:", config: {
      collections: { docs: { path: "/synthetic-audit-docs", pattern: "**/*.md" } },
    } });
    stores.push(store);
    const db = store.internal.db;
    db.prepare("INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)")
      .run("abcd1234567890", "# Synthetic\n\nSynthetic protocol audit document.", "2026-01-01");
    db.prepare(`INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
      VALUES ('docs', 'audit.md', 'Synthetic', 'abcd1234567890', '2026-01-01', '2026-01-01', 1)`).run();
    const { base } = await start({}, store);
    const sid = await initialize(base);
    const query = await tool(base, sid, "query", searchBody);
    expect(query.body.result.isError).not.toBe(true);
    expect(query.body.result.structuredContent.results[0].file).toBe("docs/audit.md");
    const doc = await tool(base, sid, "get", { file: "docs/audit.md", fromLine: 3, maxLines: 1 });
    expect(doc.body.result.isError).not.toBe(true);
    expect(doc.body.result.content[0].resource.text).toContain("Synthetic protocol audit document.");
    for (const path of ["/query", "/search"]) {
      const result = await jsonPost(base, path, searchBody);
      expect(result.response.status).toBe(200);
      expect(result.body.results[0].file).toBe("docs/audit.md");
    }
    const notification = await fetch(base + "/mcp", {
      method: "POST", headers: { ...mcpHeaders, "mcp-session-id": sid },
      body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
    });
    expect(notification.status).toBe(202);
    expect(await notification.text()).toBe("");
    const deleted = await fetch(base + "/mcp", { method: "DELETE", headers: { "mcp-session-id": sid } });
    expect(deleted.status).toBe(200);
    expect((await tool(base, sid, "status")).response.status).toBe(404);
  });

  test.each(["{", "", "null", "[]", "42", '"text"', '{"jsonrpc":"1.0"}'])("rejects malformed/invalid JSON body %j with 400, not 500", async body => {
    const { base, store } = await start();
    for (const path of ["/query", "/search", "/mcp"]) {
      const result = await raw(base, path, { body });
      expect(result.status).toBe(400);
      expect(JSON.parse(result.body)).toHaveProperty("error");
    }
    expect(store.search).not.toHaveBeenCalled();
    expect(store.getStatus).not.toHaveBeenCalled();
  });

  test("rejects declared and chunked oversize bodies before store/session work", async () => {
    const { base, store } = await start({ maxBodyBytes: 256 });
    for (const path of ["/query", "/mcp"]) {
      const declared = await raw(base, path, { headers: { "Content-Length": "10000" }, partial: true });
      expect(declared.status).toBe(413);
      const chunked = await raw(base, path, { headers: { "Transfer-Encoding": "chunked" }, body: "x".repeat(300) });
      expect(chunked.status).toBe(413);
      expect(JSON.parse(chunked.body)).toHaveProperty("error");
    }
    expect(store.search).not.toHaveBeenCalled();
    expect(store.getStatus).not.toHaveBeenCalled();
  });

  test("times out partial request bodies with a valid JSON 408", async () => {
    const { base, store } = await start({ bodyTimeoutMs: 40 });
    for (const path of ["/query", "/mcp"]) {
      const result = await raw(base, path, { headers: { "Content-Length": "100" }, body: "{", partial: true });
      expect(result.status).toBe(408);
      expect(JSON.parse(result.body)).toHaveProperty("error");
      expect(result.elapsedMs).toBeLessThan(500); // Includes server-side connection EOF, not just headers.
    }
    expect(store.search).not.toHaveBeenCalled();
  });

  test("requires JSON media type, rejects compression and malformed UTF-8", async () => {
    const { base } = await start();
    for (const path of ["/query", "/mcp"]) {
      for (const type of ["text/plain", "application/json.evil", "application/x-www-form-urlencoded"]) {
        expect((await raw(base, path, { headers: { "Content-Type": type }, body: "{}" })).status).toBe(415);
      }
      expect((await raw(base, path, { headers: { "Content-Encoding": "gzip" }, body: "{}" })).status).toBe(415);
      expect((await raw(base, path, { body: Buffer.from([0x7b, 0x22, 0xff, 0x22, 0x3a, 0x31, 0x7d]) })).status).toBe(400);
    }
  });

  test("REST schema and numeric ranges reject unsafe values without invoking search", async () => {
    const { base, store } = await start();
    const bad = [
      { searches: [] }, { searches: Array(11).fill({ type: "lex", query: "test" }) },
      { searches: [{ type: "unknown", query: "test" }] }, { searches: [{ type: "lex", query: 5 }] },
      { searches: [{ type: "lex", query: " " }] }, { searches: [{ type: "lex", query: "test\nnext" }] },
      { searches: [{ type: "lex", query: '"unclosed' }] }, { searches: [{ type: "vec", query: "test -excluded" }] },
      ...[0, -1, 1.5, 101, "10"].map(limit => ({ ...searchBody, limit })),
      ...[-1, 1.1, "0.5"].map(minScore => ({ ...searchBody, minScore })),
      { ...searchBody, collections: "docs" }, { ...searchBody, collections: [1] },
      { ...searchBody, candidateLimit: -1 }, { ...searchBody, rerank: "false" },
      { ...searchBody, intent: {} }, { searches: [{ type: "lex", query: "x".repeat(16385) }] },
    ];
    for (const body of bad) {
      const result = await jsonPost(base, "/query", body);
      expect(result.response.status, JSON.stringify(body).slice(0, 150)).toBe(400);
    }
    expect(store.search).not.toHaveBeenCalled();
    expect((await jsonPost(base, "/query", { ...searchBody, limit: 1, minScore: 1 })).response.status).toBe(200);
    expect(store.search).toHaveBeenCalledWith(expect.objectContaining({ rerank: false, limit: 1, minScore: 1 }));
  });

  test("MCP tools enforce the same query bounds and positive retrieval ranges", async () => {
    const { base, store } = await start();
    const sid = await initialize(base);
    for (const [name, args] of [
      ["query", { ...searchBody, limit: -1 }], ["query", { ...searchBody, minScore: 2 }],
      ["query", { ...searchBody, candidateLimit: 1.5 }], ["get", { file: "doc.md", fromLine: 0 }],
      ["get", { file: "doc.md", maxLines: -1 }], ["get", { file: "doc.md:0" }],
      ["get", { file: "doc.md:99999999999999999999999" }],
      ["multi_get", { pattern: "*", maxBytes: -1 }],
    ] as const) {
      const result = await tool(base, sid, name, args);
      expect(result.body.result.isError, JSON.stringify(args)).toBe(true);
    }
    expect(store.search).not.toHaveBeenCalled();
    expect(store.get).not.toHaveBeenCalled();
  });

  test("source-level Origin/Host guards protect REST, MCP and health before body/session work", async () => {
    const { base, handle, store } = await start({ bodyTimeoutMs: 1000 });
    const origins = ["https://attacker.example", "null", "", "http://localhost:0", "http://localhost",
      `http://localhost.evil:${handle.port}`, `http://localhost:${handle.port}/path`,
      `http://user@localhost:${handle.port}`, `http://127.0.0.2:${handle.port}`,
      `http://localhost:${handle.port}, https://attacker.example`];
    for (const path of ["/query", "/search", "/mcp", "/health"]) {
      for (const origin of origins) {
        const result = await raw(base, path, { method: path === "/health" ? "GET" : "POST",
          headers: { Origin: origin, "Content-Length": "9999999" }, partial: true });
        expect(result.status, `${path} ${origin}`).toBe(403);
      }
      for (const host of [`attacker.example:${handle.port}`, "localhost:0", `127.0.0.2:${handle.port}`]) {
        expect((await raw(base, path, { headers: { Host: host }, body: "{" })).status).toBe(403);
      }
    }
    expect(store.search).not.toHaveBeenCalled();
    expect(store.getStatus).not.toHaveBeenCalled();
    for (const host of ["localhost", "127.0.0.1", "[::1]"]) {
      expect((await jsonPost(base, "/query", searchBody, { Origin: `http://${host}:${handle.port}` })).response.status).toBe(200);
    }
    expect((await jsonPost(base, "/query", searchBody)).response.status).toBe(200); // Native CLI, no Origin.
    const sid = await initialize(base);
    for (const method of ["GET", "DELETE", "PUT"]) {
      const result = await fetch(base + "/mcp", { method, headers: {
        Origin: "null", "mcp-session-id": sid, Accept: "text/event-stream",
      } });
      expect(result.status).toBe(403);
    }
    expect((await tool(base, sid, "status")).response.status).toBe(200);
  });

  test("SSE flushes headers promptly, streams keepalives, and cancels on disconnect so GET can reconnect", async () => {
    const { base } = await start({ sseKeepAliveMs: 20 });
    const sid = await initialize(base);
    const first = await openSse(base, sid);
    expect(first.statusCode).toBe(200);
    expect(first.headers["content-type"]).toContain("text/event-stream");
    let text = "";
    first.on("data", chunk => { text += chunk.toString(); });
    await until(() => text.includes(": keepalive\n\n"));
    const conflict = await openSse(base, sid);
    expect(conflict.statusCode).toBe(409);
    conflict.resume();
    first.destroy();
    await retryAssertion(async () => {
      const reconnect = await openSse(base, sid);
      const code = reconnect.statusCode;
      reconnect.destroy();
      expect(code).toBe(200);
    }, 1000, 20);
  });

  test("rejects duplicate Host/Origin headers and accepts trusted MCP Origin at the actual port", async () => {
    const { base, handle, store } = await start();
    for (const extra of [`Host: attacker.example\r\n`, `Origin: ${base}\r\nOrigin: null\r\n`]) {
      const response = await new Promise<string>((resolve, reject) => {
        const socket = netConnect(handle.port, "localhost", () => {
          socket.write(`POST /query HTTP/1.1\r\nHost: localhost:${handle.port}\r\n${extra}Content-Length: 0\r\nConnection: close\r\n\r\n`);
        });
        let result = "";
        socket.on("data", chunk => { result += chunk.toString(); });
        socket.on("end", () => resolve(result));
        socket.on("error", reject);
        socket.setTimeout(1000, () => socket.destroy(new Error("Raw header fixture timed out")));
      });
      expect(response).toMatch(/^HTTP\/1\.1 403 /);
    }
    expect(store.getStatus).not.toHaveBeenCalled();
    const initialized = await jsonPost(base, "/mcp", initializeBody, { Origin: base });
    expect(initialized.response.status).toBe(200);
    expect(initialized.body.result.serverInfo.name).toBe("qmd");
  });

  test("malformed or foreign request targets are 400 without store work", async () => {
    const { base, store } = await start();
    for (const target of ["//attacker.example/query", "//[invalid/query", "http://attacker.example/query", "//localhost:0/mcp"]) {
      expect((await raw(base, target, { body: "{}" })).status).toBe(400);
    }
    expect(store.search).not.toHaveBeenCalled();
    expect(store.getStatus).not.toHaveBeenCalled();
  });

  test("rejects non-POST MCP payloads and invalid transport negotiation without allocating sessions", async () => {
    const { base, store } = await start();
    const connect = vi.spyOn(McpServer.prototype, "connect");
    expect((await jsonPost(base, "/mcp", initializeBody, { Accept: "application/json" })).response.status).toBe(406);
    expect(store.getStatus).not.toHaveBeenCalled();
    expect(connect).not.toHaveBeenCalled();
    const sid = await initialize(base);
    for (const method of ["GET", "DELETE"]) {
      const result = await raw(base, "/mcp", { method, headers: {
        "mcp-session-id": sid, "Content-Length": "2",
      }, body: "{}" });
      expect(result.status).toBe(400);
    }
    expect((await tool(base, sid, "status")).response.status).toBe(200);
    expect((await raw(base, "/mcp", { method: "PUT" })).status).toBe(405);
  });

  test("stream bridge applies backpressure and cancels an infinite producer instead of buffering it", async () => {
    let pulls = 0;
    let cancelled = false;
    const original = WebStandardStreamableHTTPServerTransport.prototype.handleRequest;
    vi.spyOn(WebStandardStreamableHTTPServerTransport.prototype, "handleRequest").mockImplementation(function(this: WebStandardStreamableHTTPServerTransport, req, options) {
      if (req.method !== "GET") return original.call(this, req, options);
      return Promise.resolve(new Response(new ReadableStream<Uint8Array>({
        pull(controller) { pulls++; controller.enqueue(new Uint8Array(65536)); },
        cancel() { cancelled = true; },
      }, { highWaterMark: 0 }), { headers: { "Content-Type": "text/event-stream" } }));
    });
    const { base, handle } = await start({ streamWriteTimeoutMs: 50 });
    const sid = await initialize(base);
    // Bun's node:http response.pause() pauses its JS facade, not its eager
    // fetch-backed reader. Stop a real TCP reader to exert real backpressure.
    const response = netConnect(handle.port, "localhost", () => {
      response.write(`GET /mcp HTTP/1.1\r\nHost: localhost:${handle.port}\r\nAccept: text/event-stream\r\nmcp-session-id: ${sid}\r\n\r\n`);
      response.pause();
    });
    sockets.push(response);
    response.on("error", () => {});
    await until(() => cancelled);
    expect(pulls).toBeGreaterThan(0);
    expect(pulls).toBeLessThan(256); // Not an unbounded arrayBuffer()/read-ahead loop.
    response.destroy();
  });

  test("session capacity evicts the least recently used idle session and closes both SDK layers", async () => {
    const serverClose = vi.spyOn(McpServer.prototype, "close");
    const transportClose = vi.spyOn(WebStandardStreamableHTTPServerTransport.prototype, "close");
    const connect = vi.spyOn(McpServer.prototype, "connect");
    const { base } = await start({ maxSessions: 2 });
    const a = await initialize(base);
    const b = await initialize(base);
    await tool(base, a, "status");
    const c = await initialize(base);
    expect((await tool(base, b, "status")).response.status).toBe(404);
    expect((await tool(base, a, "status")).response.status).toBe(200);
    expect((await tool(base, c, "status")).response.status).toBe(200);
    expect(serverClose).toHaveBeenCalled();
    expect(transportClose).toHaveBeenCalled();
    expect((connect.mock.contexts[1] as McpServer).server.transport).toBeUndefined(); // Preserve SDK onclose cleanup.
  });

  test("idle expiry also cancels an open SSE stream and disposes the session server", async () => {
    const close = vi.spyOn(McpServer.prototype, "close");
    const { base } = await start({ sessionIdleTimeoutMs: 80, sessionSweepIntervalMs: 10, sseKeepAliveMs: 10 });
    const sid = await initialize(base);
    const sse = await openSse(base, sid);
    sse.resume();
    await until(() => sse.destroyed);
    expect((await tool(base, sid, "status")).response.status).toBe(404);
    expect(close).toHaveBeenCalled();
  });

  test("failed connect disposes its server and transport and releases the reserved session slot", async () => {
    const close = vi.spyOn(McpServer.prototype, "close");
    const transportClose = vi.spyOn(WebStandardStreamableHTTPServerTransport.prototype, "close");
    vi.spyOn(WebStandardStreamableHTTPServerTransport.prototype, "start").mockRejectedValueOnce(new Error("Synthetic connect failure"));
    const { base } = await start({ maxSessions: 1 });
    expect((await jsonPost(base, "/mcp", initializeBody)).response.status).toBe(500);
    expect(close).toHaveBeenCalled();
    expect(transportClose).toHaveBeenCalled();
    await initialize(base);
  });

  test("pending initializations count toward capacity; timeout and late completion cannot resurrect a session", async () => {
    let release!: (value: typeof status) => void;
    const store = fixtureStore();
    vi.spyOn(store, "getStatus").mockImplementationOnce(() => new Promise(resolve => { release = resolve; }));
    const close = vi.spyOn(McpServer.prototype, "close");
    const { base } = await start({ maxSessions: 1, requestTimeoutMs: 100 }, store);
    const pending = jsonPost(base, "/mcp", initializeBody);
    await until(() => !!release);
    expect((await jsonPost(base, "/mcp", initializeBody)).response.status).toBe(503);
    expect((await pending).response.status).toBe(504);
    release(status);
    await until(() => close.mock.calls.length > 0);
    await initialize(base);
  });

  test("stop during initialization closes late SDK objects without reconnecting or resurrecting the session", async () => {
    let release!: (value: typeof status) => void;
    const store = fixtureStore();
    vi.spyOn(store, "getStatus").mockImplementationOnce(() => new Promise(resolve => { release = resolve; }));
    const close = vi.spyOn(McpServer.prototype, "close");
    const connect = vi.spyOn(McpServer.prototype, "connect");
    const transportClose = vi.spyOn(WebStandardStreamableHTTPServerTransport.prototype, "close");
    const { base, handle } = await start({}, store);
    // Attach both outcomes now: stop is allowed to close the socket before a 503 flushes.
    const pending = jsonPost(base, "/mcp", initializeBody).then(() => {}, () => {});
    await until(() => !!release);
    await handle.stop();
    expect(handle.httpServer.listening).toBe(false);
    expect(transportClose).toHaveBeenCalled();
    release(status);
    await until(() => close.mock.calls.length > 0);
    expect(connect).not.toHaveBeenCalled();
    await pending;
  });

  test("disconnect during a POST closes the session and aborts SDK bookkeeping", async () => {
    const store = fixtureStore();
    const search = pendingSearch(store);
    const close = vi.spyOn(McpServer.prototype, "close");
    const connect = vi.spyOn(McpServer.prototype, "connect");
    const { base, handle } = await start({}, store);
    const sid = await initialize(base);
    const body = JSON.stringify({ jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "query", arguments: searchBody } });
    const req = netConnect(handle.port, "localhost", () => {
      req.write(`POST /mcp HTTP/1.1\r\nHost: localhost:${handle.port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nmcp-session-id: ${sid}\r\nContent-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
    });
    req.on("error", () => {});
    sockets.push(req);
    await until(() => !!search.signal);
    req.destroy();
    await until(() => close.mock.calls.length > 0);
    await until(() => search.stopped);
    expect(search.signal!.aborted).toBe(true);
    expect((connect.mock.contexts[0] as McpServer).server.transport).toBeUndefined();
    expect((await tool(base, sid, "status")).response.status).toBe(404);
  });

  test("processing timeout closes a stuck MCP session without waiting for the store promise", async () => {
    const store = fixtureStore();
    const search = pendingSearch(store);
    const { base } = await start({ requestTimeoutMs: 50 }, store);
    const sid = await initialize(base);
    const result = await tool(base, sid, "query", searchBody);
    expect(result.response.status).toBe(504);
    await until(() => search.stopped);
    expect(search.signal!.aborted).toBe(true);
    expect((await tool(base, sid, "status")).response.status).toBe(404);
  });

  test("stop waits for listener closure, cancels SSE/partial bodies, is idempotent and removes signal handlers", async () => {
    const sigint = process.listenerCount("SIGINT");
    const sigterm = process.listenerCount("SIGTERM");
    const { base, handle, store } = await start();
    const sid = await initialize(base);
    const sse = await openSse(base, sid);
    sse.resume();
    const partial = httpRequest(base + "/query", { method: "POST", headers: { ...mcpHeaders, "Content-Length": "100" } });
    partial.on("error", () => {});
    clients.push(partial);
    partial.write("{");
    await Promise.all([handle.stop(), handle.stop()]);
    expect(handle.httpServer.listening).toBe(false);
    await until(() => sse.destroyed);
    expect(process.listenerCount("SIGINT")).toBe(sigint);
    expect(process.listenerCount("SIGTERM")).toBe(sigterm);
    expect(store.close).not.toHaveBeenCalled(); // Injected store is caller-owned.
  });
});
