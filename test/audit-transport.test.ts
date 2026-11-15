import { afterAll, afterEach, beforeAll, describe, expect, test, vi } from "vitest";
import http from "node:http";
import { getEventListeners } from "node:events";
import {
  MlxEmbedClient, embedBatchWithMlx, embedBatchConcurrent, embedWithMlx,
  generateWithMlx, rerankWithMlx, tokenizeWithMlx,
  mlxHealth, mlxMemory, mlxStats, getMlxDescriptor,
} from "../src/mlx.js";
import { decodeBinaryEmbeddings, decodeEmbeddingDescriptorHeader, encodeBinaryEmbeddings, validateJsonEmbedResponse } from "../src/embedding/protocol.js";
import { computeEmbeddingSpaceId, formatDocForDescriptor, formatQueryForDescriptor, type EmbeddingDescriptor } from "../src/embedding/contract.js";
import { resolveEmbeddingConfig } from "../src/embedding/config.js";
import { parseExpansionOutput } from "../src/expansion/protocol.js";

const descriptor: EmbeddingDescriptor = {
  version: 1, backend: "mlx", model: "fixture/encoder-A", pooling: "mean",
  nativeDimensions: 2, outputDimensions: 2, maxTokens: 512, normalized: true,
};
let server: http.Server;
let base: string;
const requests: string[] = [];
let active = 0;
let peak = 0;

beforeAll(async () => {
  server = http.createServer(async (req, res) => {
    const path = req.url!;
    requests.push(path);
    const endpoint = path.split("/").at(-1);
    if (path.startsWith("/headers/")) return; // Deliberately never send headers.
    if (path.startsWith("/body/") || (path.startsWith("/embed-body/") && endpoint?.startsWith("embed"))) {
      res.writeHead(200, { "Content-Type": endpoint === "embed-bin" ? "application/octet-stream" : "application/json" });
      res.write(endpoint === "embed-bin" ? Buffer.alloc(8) : "{");
      return; // Real headers followed by a permanently stalled body.
    }
    if (endpoint === "descriptor") {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify(path.startsWith("/bad-descriptor/") ? { ...descriptor, normalized: "true" } : descriptor));
      return;
    }
    if (["health", "memory", "stats"].includes(endpoint!)) {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify(endpoint === "health" ? { status: "ok", ready: true, descriptor }
        : endpoint === "memory" ? { active_mb: 1, peak_mb: 2, model_mb: 1 }
        : { total_requests: 1, avg_ms: 1, compiled_shapes: 1, uptime_sec: 1 }));
      return;
    }
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk);
    const payload = JSON.parse(Buffer.concat(chunks).toString() || "{}");
    if (endpoint === "embed" || endpoint === "embed-bin") {
      active++;
      peak = Math.max(peak, active);
      const delay = path.startsWith("/ordered/") ? (8 - Number(payload.texts[0])) * 3 : 0;
      const timer = setTimeout(() => {
        active--;
        const dims = path.startsWith("/wrong-dims/") ? 3 : (payload.dims ?? 2);
        const actual = { ...descriptor, outputDimensions: dims, nativeDimensions: Math.max(2, dims),
          model: path.startsWith("/drift/") ? "fixture/encoder-B" : descriptor.model };
        const vectors = payload.texts.map((text: string) => Array.from({ length: dims }, (_, i) => Number(text) || i + 1));
        if (endpoint === "embed-bin") {
          res.setHeader("Content-Type", "application/octet-stream");
          if (!path.startsWith("/missing/")) res.setHeader("X-QMD-Embedding-Descriptor", encodeURIComponent(JSON.stringify(actual)));
          res.end(encodeBinaryEmbeddings(vectors, dims));
        } else {
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ embeddings: vectors, dims, model: actual.model,
            ...(path.startsWith("/missing/") ? {} : { descriptor: actual }) }));
        }
      }, Math.max(0, delay));
      res.once("close", () => clearTimeout(timer));
      return;
    }
    if (endpoint === "rerank") res.end(JSON.stringify({ scores: payload.documents.map(() => 0.5), model: "fixture/ranker" }));
    else if (endpoint === "generate") res.end(JSON.stringify({ text: "vec: database indexing", model: "fixture/generator" }));
    else if (endpoint === "tokenize") res.end(JSON.stringify({ counts: payload.texts.map(() => 2) }));
    else { res.writeHead(404); res.end(); }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
});
afterAll(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

async function bounded<T>(work: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  try {
    return await Promise.race([work, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("Operation exceeded test watchdog")), 1000);
    })]);
  } finally { clearTimeout(timer!); }
}

describe("audit: complete deadlines and cancellation", () => {
  test.each(["headers", "body"])("aborts real stalled %s on JSON and binary requests", async (mode) => {
    for (const binary of [false, true]) {
      const start = performance.now();
      expect(await bounded(embedWithMlx("x", { url: `${base}/${mode}`, timeoutMs: 40, binary }))).toBeNull();
      expect(performance.now() - start).toBeGreaterThanOrEqual(20);
      expect(performance.now() - start).toBeLessThan(800);
    }
  });
  test("rerank/generate/tokenize and metadata never lengthen caller deadlines", async () => {
    const config = { url: `${base}/body`, timeoutMs: 35 };
    const calls = [rerankWithMlx("q", ["d"], config), generateWithMlx("p", config), tokenizeWithMlx(["t"], config),
      mlxHealth(config), mlxMemory(config), mlxStats(config), getMlxDescriptor(config)];
    expect(await bounded(Promise.all(calls))).toEqual(Array(7).fill(null));
  });
  test("caller abort cancels the fetch and rejects instead of becoming backend failure", async () => {
    const controller = new AbortController();
    const promise = embedWithMlx("x", { url: `${base}/body`, timeoutMs: 5000 }, { signal: controller.signal });
    setTimeout(() => controller.abort(), 25);
    await expect(bounded(promise)).rejects.toMatchObject({ name: "AbortError" });
    expect(getEventListeners(controller.signal, "abort")).toHaveLength(0);
  });
  test("deadline also bounds non-cooperative fetch implementations", async () => {
    vi.stubGlobal("fetch", () => new Promise(() => {}));
    expect(await bounded(embedWithMlx("x", { timeoutMs: 25 }))).toBeNull();
  });
  test("late headers after cancellation have their body cancelled", async () => {
    let publish!: (response: Response) => void;
    let cancelled = false;
    vi.stubGlobal("fetch", () => new Promise<Response>((resolve) => { publish = resolve; }));
    expect(await bounded(embedWithMlx("x", { timeoutMs: 25 }))).toBeNull();
    publish(new Response(new ReadableStream({ cancel() { cancelled = true; } })));
    await new Promise((resolve) => setImmediate(resolve));
    expect(cancelled).toBe(true);
  });
  test("completed requests remove listeners and leave no delayed abort", async () => {
    const controller = new AbortController();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    expect(await embedWithMlx("1", { url: base, timeoutMs: 60, signal: controller.signal })).not.toBeNull();
    const signal = fetchSpy.mock.calls[0]![1]!.signal!;
    expect(getEventListeners(controller.signal, "abort")).toHaveLength(0);
    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(signal.aborted).toBe(false);
  });
  test("batch cancellation stops dequeuing while preserving caller cancellation", async () => {
    const controller = new AbortController();
    const before = requests.length;
    const promise = embedBatchConcurrent(Array(20).fill("x"), { url: `${base}/body`, concurrency: 2, signal: controller.signal }, undefined, 1);
    setTimeout(() => controller.abort(), 25);
    await expect(bounded(promise)).rejects.toMatchObject({ name: "AbortError" });
    expect(requests.length - before).toBeLessThanOrEqual(2);
  });
});

describe("audit: byte caps and cleanup", () => {
  test("preserves starting health reports with null, not fabricated, identity", async () => {
    vi.stubGlobal("fetch", async () => Response.json({ status: "starting", ready: false, model: null, dims: null, descriptor: null }));
    expect(await mlxHealth()).toMatchObject({ status: "starting", ready: false, model: undefined, dims: undefined, descriptor: undefined });
  });
  test("bounds error-body consumption and log size, cancelling the stream", async () => {
    let pulled = 0;
    let cancelled = false;
    vi.stubGlobal("fetch", async () => new Response(new ReadableStream({
      pull(controller) { pulled++; controller.enqueue(new Uint8Array(1024).fill(120)); },
      cancel() { cancelled = true; },
    }), { status: 503 }));
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(await embedWithMlx("x")).toBeNull();
    expect(cancelled).toBe(true);
    expect(pulled).toBeLessThan(12);
    expect(log.mock.calls.flat().join(" ").length).toBeLessThan(2048);
  });
  test.each([mlxHealth, mlxMemory, mlxStats, getMlxDescriptor])("cancels failed GET bodies without reading them", async (call) => {
    let cancelled = false;
    vi.stubGlobal("fetch", async () => new Response(new ReadableStream({ cancel() { cancelled = true; } }), { status: 503 }));
    expect(await call({ timeoutMs: 40 })).toBeNull();
    expect(cancelled).toBe(true);
  });
  test.each([false, true])("caps embedding bytes by expected count/dimensions (binary=%s)", async (binary) => {
    let cancelled = false;
    let pulled = 0;
    vi.stubGlobal("fetch", async () => new Response(new ReadableStream({
      pull(controller) { pulled++; controller.enqueue(new Uint8Array(8192).fill(32)); },
      cancel() { cancelled = true; },
    })));
    vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(embedBatchWithMlx(["x"], { binary, expectedDescriptor: descriptor }, { dims: 2 })).rejects.toThrow();
    expect(cancelled).toBe(true);
    expect(pulled).toBeLessThan(30);
  });
  test("rejects declared oversized bodies without pulling or allocating them", async () => {
    let cancelled = false;
    vi.stubGlobal("fetch", async () => new Response(new ReadableStream({ cancel() { cancelled = true; } }),
      { headers: { "Content-Length": "9999999999999" } }));
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(await embedWithMlx("x", { binary: true }, { dims: 2 })).toBeNull();
    expect(cancelled).toBe(true);
  });
  test("a no-progress body cannot evade the cap with infinite empty chunks", async () => {
    let cancelled = false;
    vi.stubGlobal("fetch", async () => new Response(new ReadableStream({
      pull(controller) { controller.enqueue(new Uint8Array(0)); }, cancel() { cancelled = true; },
    })));
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(await bounded(embedWithMlx("x"))).toBeNull();
    expect(cancelled).toBe(true);
  });
});

describe("audit: negotiated serving identity and immutable client", () => {
  test.each([false, true])("requires response identity and rejects same-dimension daemon replacement (binary=%s)", async (binary) => {
    for (const mode of ["drift", "missing", "wrong-dims"]) {
      const client = new MlxEmbedClient({ url: `${base}/${mode}`, binary });
      expect(await client.probeDims()).toBe(2);
      await expect(client.embed("1", { model: "caller-must-not-label-serving-model" })).rejects.toThrow();
      client.dispose();
    }
  });
  test("validates descriptors at negotiation instead of trusting a cast", async () => {
    expect(await getMlxDescriptor({ url: `${base}/bad-descriptor` })).toBeNull();
    const client = new MlxEmbedClient({ url: `${base}/bad-descriptor` });
    await expect(client.probeDims()).rejects.toThrow();
    client.dispose();
  });
  test("single-flight negotiation pins a detached frozen descriptor and preserves binary:false", async () => {
    const before = requests.length;
    const supplied = { ...descriptor };
    const config = { url: base, binary: false, expectedDescriptor: supplied };
    const client = new MlxEmbedClient(config);
    config.binary = true;
    supplied.model = "mutated-after-construction";
    const [a, b] = await Promise.all([client.getDescriptor(), client.getDescriptor()]);
    expect(a).toBe(b);
    expect(Object.isFrozen(a)).toBe(true);
    expect(Object.isFrozen(client.config())).toBe(true);
    expect(client.config().binary).toBe(false);
    const result = await client.embedBatch(Array(16).fill("1"));
    expect(result.every((entry) => entry?.model === descriptor.model)).toBe(true);
    expect(requests.slice(before).filter((path) => path === "/descriptor")).toHaveLength(1);
    expect(requests.at(-1)).toBe("/embed");
    client.dispose();
  });
  test("dispose aborts in-flight work, is idempotent and forbids resurrection", async () => {
    const client = new MlxEmbedClient({ url: `${base}/body`, timeoutMs: 5000 });
    const pending = client.getDescriptor();
    setTimeout(() => client.dispose(), 25);
    await expect(bounded(pending)).rejects.toMatchObject({ name: "AbortError" });
    client.dispose();
    expect(client.descriptor).toBeNull();
    await expect(client.embed("x")).rejects.toMatchObject({ name: "AbortError" });
    await expect(client.getDescriptor()).rejects.toMatchObject({ name: "AbortError" });
  });
  test("late non-cooperative negotiation cannot publish after dispose", async () => {
    let publish!: (response: Response) => void;
    let cancelled = false;
    vi.stubGlobal("fetch", () => new Promise<Response>((resolve) => { publish = resolve; }));
    const external = new AbortController();
    const client = new MlxEmbedClient({ signal: external.signal });
    expect(getEventListeners(external.signal, "abort")).toHaveLength(1);
    const pending = client.getDescriptor();
    client.dispose();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    const bytes = new TextEncoder().encode(JSON.stringify(descriptor));
    publish(new Response(new ReadableStream({
      start(controller) { controller.enqueue(bytes); }, cancel() { cancelled = true; },
    })));
    await new Promise((resolve) => setImmediate(resolve));
    expect(client.descriptor).toBeNull();
    expect(cancelled).toBe(true);
    expect(getEventListeners(external.signal, "abort")).toHaveLength(0);
  });
  test("dispose also aborts an embedding after negotiation has completed", async () => {
    const client = new MlxEmbedClient({ url: `${base}/embed-body`, timeoutMs: 5000 });
    expect(await client.probeDims()).toBe(2);
    const pending = client.embed("x");
    setTimeout(() => client.dispose(), 25);
    await expect(bounded(pending)).rejects.toMatchObject({ name: "AbortError" });
    await expect(client.embedBatch([])).rejects.toMatchObject({ name: "AbortError" });
  });
  test("invalid per-call options are refused before negotiation dispatch", async () => {
    const mockFetch = vi.fn();
    vi.stubGlobal("fetch", mockFetch);
    const client = new MlxEmbedClient({ url: base });
    await expect(client.embed("x", { dims: NaN })).rejects.toThrow();
    await expect(client.embedBatchConcurrent(["x"], undefined, Infinity)).rejects.toThrow();
    expect(await client.embedBatch([])).toEqual([]);
    expect(mockFetch).not.toHaveBeenCalled();
    client.dispose();
  });
  test("one cancelled negotiation waiter does not cancel another caller", async () => {
    let publish!: (response: Response) => void;
    let requestSignal!: AbortSignal;
    const fetchMock = vi.fn((_url: string, init: RequestInit) => {
      requestSignal = init.signal!;
      return new Promise<Response>((resolve) => { publish = resolve; });
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new MlxEmbedClient();
    const controller = new AbortController();
    const first = client.getDescriptor(controller.signal);
    const second = client.getDescriptor();
    controller.abort();
    await expect(first).rejects.toMatchObject({ name: "AbortError" });
    expect(requestSignal.aborted).toBe(false);
    publish(Response.json(descriptor));
    expect(await second).toEqual(descriptor);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(getEventListeners(controller.signal, "abort")).toHaveLength(0);
    client.dispose();
  });
  test("binary result uses actual serving model, never the caller label", async () => {
    const result = await embedWithMlx("1", { url: base, binary: true, expectedDescriptor: descriptor }, { model: "wrong" });
    expect(result?.model).toBe(descriptor.model);
  });
  test("out-of-order completions retain values/cardinality and concurrency bounds", async () => {
    active = peak = 0;
    const texts = Array.from({ length: 8 }, (_, i) => String(i + 1));
    const result = await embedBatchConcurrent(texts, { url: `${base}/ordered`, concurrency: 3 }, undefined, 1);
    expect(result.map((entry) => entry?.embedding[0])).toEqual(texts.map(Number));
    expect(peak).toBe(3);
  });
});

describe("audit: strict scalar and wire contracts", () => {
  test.each([NaN, Infinity, 0, -1, 1.5])("rejects invalid batching/concurrency/timeout/dims: %s", async (value) => {
    await expect(embedBatchConcurrent(["a", "b"], undefined, undefined, value)).rejects.toThrow();
    await expect(embedBatchConcurrent(["a", "b"], { concurrency: value })).rejects.toThrow();
    await expect(embedBatchWithMlx(["a"], { timeoutMs: value })).rejects.toThrow();
    await expect(embedBatchWithMlx(["a"], undefined, { dims: value })).rejects.toThrow();
  });
  test.each([0, -1, 1.5, NaN, Infinity])("rejects invalid reported JSON dims: %s", (dims) => {
    expect(() => validateJsonEmbedResponse({ embeddings: [[1], [1, 2]], dims, model: "x" })).toThrow();
  });
  test("rejects zero-count trailing bytes and positive-count zero-dimensional binary frames", () => {
    expect(() => decodeBinaryEmbeddings(new ArrayBuffer(12))).toThrow();
    const frame = new ArrayBuffer(8);
    new DataView(frame).setInt32(0, 1, true);
    expect(() => decodeBinaryEmbeddings(frame)).toThrow();
    expect(() => encodeBinaryEmbeddings([[Number.MAX_VALUE]], 1)).toThrow();
  });
  test("enforces requested count/dimensions and finite values on both wire formats", () => {
    const payload = { embeddings: [[1, 2]], model: descriptor.model, dims: 2, descriptor };
    expect(() => validateJsonEmbedResponse(payload, { count: 2 })).toThrow();
    expect(() => validateJsonEmbedResponse(payload, { dims: 3 })).toThrow();
    expect(() => validateJsonEmbedResponse({ ...payload, embeddings: [[1, Infinity]] })).toThrow();
    const frame = encodeBinaryEmbeddings([[1, 2]], 2).buffer as ArrayBuffer;
    expect(() => decodeBinaryEmbeddings(frame, { count: 2 })).toThrow();
    expect(() => decodeBinaryEmbeddings(frame, { dims: 3 })).toThrow();
    new DataView(frame).setFloat32(8, Infinity, true);
    expect(() => decodeBinaryEmbeddings(frame)).toThrow();
  });
  test("decodes percent-encoded Unicode identity and rejects malformed header encodings", () => {
    const unicode = { ...descriptor, model: "fixture/数据库" };
    expect(decodeEmbeddingDescriptorHeader(encodeURIComponent(JSON.stringify(unicode)), unicode)?.model).toBe(unicode.model);
    for (const header of ["%ZZ", "%7Bnot-json%7D", encodeURIComponent("null"), encodeURIComponent(JSON.stringify({ ...descriptor, maxTokens: 0 }))]) {
      expect(() => decodeEmbeddingDescriptorHeader(header, descriptor)).toThrow();
    }
  });
  test("a fatal batch protocol error cancels siblings and stops dequeuing", async () => {
    const signals: AbortSignal[] = [];
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      signals.push(init.signal!);
      const { texts } = JSON.parse(init.body as string);
      if (texts[0] === "bad") return Response.json({ embeddings: [[1]], dims: 1, model: descriptor.model, descriptor });
      return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode("{")); } }));
    });
    vi.stubGlobal("fetch", fetchMock);
    await expect(bounded(embedBatchConcurrent(["bad", "waiting", "never"], { expectedDescriptor: descriptor, concurrency: 2 }, undefined, 1))).rejects.toThrow();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(signals[1]!.aborted).toBe(true);
  });
  test("maxTokens changes the embedding space fingerprint", () => {
    expect(computeEmbeddingSpaceId(descriptor)).not.toBe(computeEmbeddingSpaceId({ ...descriptor, maxTokens: 128 }));
  });
  test("unrecognized encoders receive raw text, not Gemma instructions", () => {
    expect(formatQueryForDescriptor("查询", { model: "other-encoder" })).toBe("查询");
    expect(formatDocForDescriptor("正文", "标题", { model: "other-encoder" })).toBe("标题\n正文");
  });
  test("resolved configuration is immutable and rejects malformed numeric environment suffixes", () => {
    vi.stubEnv("QMD_MLX_CONCURRENCY", "2workers");
    expect(() => resolveEmbeddingConfig()).toThrow();
    vi.stubEnv("QMD_MLX_CONCURRENCY", "2");
    expect(Object.isFrozen(resolveEmbeddingConfig())).toBe(true);
    vi.unstubAllEnvs();
  });
});

describe("audit: bounded Unicode-aware expansion parsing", () => {
  test("non-Latin queries do not bypass relevance checks", () => {
    expect(parseExpansionOutput("vec: chocolate cake recipe", { query: "数据库索引" }).usable).toBe(false);
    expect(parseExpansionOutput("vec: 数据库索引优化", { query: "数据库索引" }).usable).toBe(true);
    expect(parseExpansionOutput("vec: поиск базы данных", { query: "базы данных" }).usable).toBe(true);
    expect(parseExpansionOutput("vec: unrelated topic", { query: "école" }).usable).toBe(false);
  });
  test("rejects excessive bytes and line cardinality without retaining unbounded raw text", () => {
    const many = parseExpansionOutput(Array.from({ length: 1000 }, (_, i) => `vec: database ${i}`).join("\n"), { query: "database" });
    expect(many.usable).toBe(false);
    expect(many.queryables).toHaveLength(0);
    const large = parseExpansionOutput("vec: " + "数".repeat(50000));
    expect(large.usable).toBe(false);
    expect(Buffer.byteLength(large.rawText)).toBeLessThanOrEqual(65536);
    const thirteen = parseExpansionOutput(Array.from({ length: 13 }, (_, i) => `vec: database ${i}`).join("\n"), { query: "database" });
    expect(thirteen.usable).toBe(false);
    expect(thirteen.queryables).toHaveLength(0);
  });
});
