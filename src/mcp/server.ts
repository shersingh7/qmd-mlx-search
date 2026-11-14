/**
 * QMD MCP Server - Model Context Protocol server for QMD
 *
 * Exposes QMD search and document retrieval as MCP tools and resources.
 * Documents are accessible via qmd:// URIs.
 *
 * Follows MCP spec 2025-06-18 for proper response types.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "url";
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { WebStandardStreamableHTTPServerTransport }
  from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import {
  CancelledNotificationSchema, isInitializeRequest, isJSONRPCRequest,
  isJSONRPCResultResponse, isJSONRPCErrorResponse, JSONRPCMessageSchema,
  type JSONRPCMessage, type RequestId,
} from "@modelcontextprotocol/sdk/types.js";
import type { Transport, TransportSendOptions } from "@modelcontextprotocol/sdk/shared/transport.js";
import { z } from "zod";
import { existsSync } from "fs";
import {
  createStore,
  extractSnippet,
  addLineNumbers,
  getDefaultDbPath,
  DEFAULT_MULTI_GET_MAX_BYTES,
  type QMDStore,
  type ExpandedQuery,
  type IndexStatus,
} from "../index.js";
import { getConfigPath } from "../collections.js";
import { validateLexQuery, validateSemanticQuery } from "../store.js";

// Keep REST and MCP validation aligned. Limits bound work as well as JSON size;
// in particular, never coerce a client-supplied type, collection list or limit.
const subSearchSchema = z.object({
  type: z.enum(["lex", "vec", "hyde"]).describe(
    "lex = BM25 keywords (supports phrases and negation); vec = semantic question; hyde = hypothetical answer passage"
  ),
  query: z.string().min(1).max(16384).refine(value => value.trim().length > 0, "Query must not be blank"),
}).superRefine((search, ctx) => {
  const error = /[\r\n]/.test(search.query)
    ? "Queries must be single-line"
    : search.type === "lex" ? validateLexQuery(search.query) : validateSemanticQuery(search.query);
  if (error) ctx.addIssue({ code: "custom", path: ["query"], message: error });
});
const searchInputShape = {
  searches: z.array(subSearchSchema).min(1).max(10).describe("Typed sub-queries (lex/vec/hyde). First gets 2x weight."),
  limit: z.number().int().min(1).max(100).optional().default(10).describe("Max results, 1-100 (default: 10)"),
  minScore: z.number().min(0).max(1).optional().default(0).describe("Min relevance, 0-1 (default: 0)"),
  candidateLimit: z.number().int().min(1).max(1000).optional().describe("Maximum candidates to rerank"),
  collections: z.array(z.string().min(1).max(256)).max(100).optional().describe("Filter to collections (OR match)"),
  intent: z.string().max(16384).optional().describe("Background context to disambiguate the query and improve snippets"),
  rerank: z.boolean().optional().default(true).describe("Rerank with LLM; false avoids reranking (default: true)"),
};
const searchInputSchema = z.object(searchInputShape);
const documentPathSchema = z.string().min(1).max(4096).refine(file => {
  const suffix = file.match(/:(\d+)$/)?.[1];
  return suffix === undefined || (Number.isSafeInteger(Number(suffix)) && Number(suffix) >= 1);
}, "Line suffix must be a positive safe integer");
const maxLinesSchema = z.number().int().min(1).max(100000).optional();

// =============================================================================
// Types for structured content
// =============================================================================

type SearchResultItem = {
  docid: string;  // Short docid (#abc123) for quick reference
  file: string;
  title: string;
  score: number;
  context: string | null;
  snippet: string;
};

type StatusResult = {
  totalDocuments: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  collections: {
    name: string;
    path: string | null;
    pattern: string | null;
    documents: number;
    lastUpdated: string;
  }[];
};

// =============================================================================
// Helper functions
// =============================================================================

/**
 * Encode a path for use in qmd:// URIs.
 * Encodes special characters but preserves forward slashes for readability.
 */
function encodeQmdPath(path: string): string {
  // Encode each path segment separately to preserve slashes
  return path.split('/').map(segment => encodeURIComponent(segment)).join('/');
}

/**
 * Format search results as human-readable text summary
 */
function formatSearchSummary(results: SearchResultItem[], query: string): string {
  if (results.length === 0) {
    return `No results found for "${query}"`;
  }
  const lines = [`Found ${results.length} result${results.length === 1 ? '' : 's'} for "${query}":\n`];
  for (const r of results) {
    lines.push(`${r.docid} ${Math.round(r.score * 100)}% ${r.file} - ${r.title}`);
  }
  return lines.join('\n');
}

function getPackageVersion(): string {
  try {
    const pkgPath = join(dirname(fileURLToPath(import.meta.url)), "../../package.json");
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
    return pkg.version ?? "unknown";
  } catch {
    return "unknown";
  }
}

// =============================================================================
// MCP Server
// =============================================================================

/**
 * Build dynamic server instructions from actual index state.
 * Injected into the LLM's system prompt via MCP initialize response —
 * gives the LLM immediate context about what's searchable without a tool call.
 */
async function buildInstructions(store: QMDStore): Promise<string> {
  const status = await store.getStatus();
  const contexts = await store.listContexts();
  const globalCtx = await store.getGlobalContext();
  const lines: string[] = [];

  // --- What is this? ---
  lines.push(`QMD is your local search engine over ${status.totalDocuments} markdown documents.`);
  if (globalCtx) lines.push(`Context: ${globalCtx}`);

  // --- What's searchable? ---
  if (status.collections.length > 0) {
    lines.push("");
    lines.push("Collections (scope with `collection` parameter):");
    for (const col of status.collections) {
      // Find root context for this collection
      const rootCtx = contexts.find(c => c.collection === col.name && (c.path === "" || c.path === "/"));
      const desc = rootCtx ? ` — ${rootCtx.context}` : "";
      lines.push(`  - "${col.name}" (${col.documents} docs)${desc}`);
    }
  }

  // --- Capability gaps ---
  if (!status.hasVectorIndex) {
    lines.push("");
    lines.push("Note: No vector embeddings yet. Run `qmd embed` to enable semantic search (vec/hyde).");
  } else if (status.needsEmbedding > 0) {
    lines.push("");
    lines.push(`Note: ${status.needsEmbedding} documents need embedding. Run \`qmd embed\` to update.`);
  }

  // --- Search tool ---
  lines.push("");
  lines.push("Search: Use `query` with sub-queries (lex/vec/hyde):");
  lines.push("  - type:'lex' — BM25 keyword search (exact terms, fast)");
  lines.push("  - type:'vec' — semantic vector search (meaning-based)");
  lines.push("  - type:'hyde' — hypothetical document (write what the answer looks like)");
  lines.push("");
  lines.push("  Always provide `intent` on every search call to disambiguate and improve snippets.");
  lines.push("");
  lines.push("Examples:");
  lines.push("  Quick keyword lookup: [{type:'lex', query:'error handling'}]");
  lines.push("  Semantic search: [{type:'vec', query:'how to handle errors gracefully'}]");
  lines.push("  Best results: [{type:'lex', query:'error'}, {type:'vec', query:'error handling best practices'}]");
  lines.push("  With intent: searches=[{type:'lex', query:'performance'}], intent='web page load times'");

  // --- Retrieval workflow ---
  lines.push("");
  lines.push("Retrieval:");
  lines.push("  - `get` — single document by path or docid (#abc123). Supports line offset (`file.md:100`).");
  lines.push("  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`) or comma-separated list.");

  // --- Non-obvious things that prevent mistakes ---
  lines.push("");
  lines.push("Tips:");
  lines.push("  - File paths in results are relative to their collection.");
  lines.push("  - Use `minScore: 0.5` to filter low-confidence results.");
  lines.push("  - Results include a `context` field describing the content type.");

  return lines.join("\n");
}

/**
 * Create an MCP server with all QMD tools, resources, and prompts registered.
 * Shared by both stdio and HTTP transports.
 */
async function createMcpServer(store: QMDStore, ownership: McpRequestOwnership): Promise<McpServer> {
  // Resolve fallible store reads before allocating an SDK server.
  const instructions = await buildInstructions(store);
  const defaultCollectionNames = await store.getDefaultCollectionNames();
  const server = new McpServer(
    { name: "qmd", version: getPackageVersion() },
    { instructions },
  );
  // SDK 1.32.2's cancellation handler treats 0/"" as absent and suppresses
  // replies, leaving JSON HTTP requests pending. Use our per-operation owner
  // through the public notification API instead. QMD callbacks finish normally
  // (with an isError tool result on cancellation), so transport cleanup and ID
  // reuse wait for actual completion rather than merely signalling abort.
  server.server.setNotificationHandler(CancelledNotificationSchema, notification => {
    ownership.cancel(notification.params.requestId, notification.params.reason);
  });

  // ---------------------------------------------------------------------------
  // Resource: qmd://{path} - read-only access to documents by path
  // Note: No list() - documents are discovered via search tools
  // ---------------------------------------------------------------------------

  server.registerResource(
    "document",
    new ResourceTemplate("qmd://{+path}", { list: undefined }),
    {
      title: "QMD Document",
      description: "A markdown document from your QMD knowledge base. Use search tools to discover documents.",
      mimeType: "text/markdown",
    },
    async (uri, { path }, extra) => {
      const signal = ownership.signal(extra);
      signal.throwIfAborted();
      // Decode URL-encoded path (MCP clients send encoded URIs)
      const pathStr = Array.isArray(path) ? path.join('/') : (path || '');
      const decodedPath = decodeURIComponent(pathStr);

      // Use SDK to find document — findDocument handles collection/path resolution
      const result = await store.get(decodedPath, { includeBody: true });
      signal.throwIfAborted();

      if ("error" in result) {
        return { contents: [{ uri: uri.href, text: `Document not found: ${decodedPath}` }] };
      }

      let text = addLineNumbers(result.body || "");  // Default to line numbers
      if (result.context) {
        text = `<!-- Context: ${result.context} -->\n\n` + text;
      }

      return {
        contents: [{
          uri: uri.href,
          name: result.displayPath,
          title: result.title || result.displayPath,
          mimeType: "text/markdown",
          text,
        }],
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: query (Primary search tool)
  // ---------------------------------------------------------------------------


  server.registerTool(
    "query",
    {
      title: "Query",
      description: `Search the knowledge base using a query document — one or more typed sub-queries combined for best recall.

## Query Types

**lex** — BM25 keyword search. Fast, exact, no LLM needed.
Full lex syntax:
- \`term\` — prefix match ("perf" matches "performance")
- \`"exact phrase"\` — phrase must appear verbatim
- \`-term\` or \`-"phrase"\` — exclude documents containing this

Good lex examples:
- \`"connection pool" timeout -redis\`
- \`"machine learning" -sports -athlete\`
- \`handleError async typescript\`

**vec** — Semantic vector search. Write a natural language question. Finds documents by meaning, not exact words.
- \`how does the rate limiter handle burst traffic?\`
- \`what is the tradeoff between consistency and availability?\`

**hyde** — Hypothetical document. Write 50-100 words that look like the answer. Often the most powerful for nuanced topics.
- \`The rate limiter uses a token bucket algorithm. When a client exceeds 100 req/min, subsequent requests return 429 until the window resets.\`

## Strategy

Combine types for best results. First sub-query gets 2× weight — put your strongest signal first.

| Goal | Approach |
|------|----------|
| Know exact term/name | \`lex\` only |
| Concept search | \`vec\` only |
| Best recall | \`lex\` + \`vec\` |
| Complex/nuanced | \`lex\` + \`vec\` + \`hyde\` |
| Unknown vocabulary | Use a standalone natural-language query (no typed lines) so the server can auto-expand it |

## Examples

Simple lookup:
\`\`\`json
[{ "type": "lex", "query": "CAP theorem" }]
\`\`\`

Best recall on a technical topic:
\`\`\`json
[
  { "type": "lex", "query": "\\"connection pool\\" timeout -redis" },
  { "type": "vec", "query": "why do database connections time out under load" },
  { "type": "hyde", "query": "Connection pool exhaustion occurs when all connections are in use and new requests must wait. This typically happens under high concurrency when queries run longer than expected." }
]
\`\`\`

Intent-aware lex (C++ performance, not sports):
\`\`\`json
[
  { "type": "lex", "query": "\\"C++ performance\\" optimization -sports -athlete" },
  { "type": "vec", "query": "how to optimize C++ program performance" }
]
\`\`\``,
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: searchInputShape,
    },
    async ({ searches, limit, minScore, candidateLimit, collections, intent, rerank }, extra) => {
      const signal = ownership.signal(extra);
      signal.throwIfAborted();
      // Map to internal format
      const queries: ExpandedQuery[] = searches.map(s => ({
        type: s.type,
        query: s.query,
      }));

      // Use default collections if none specified
      const effectiveCollections = collections ?? defaultCollectionNames;

      const results = await store.search({
        queries,
        collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
        limit,
        minScore,
        rerank,
        intent,
        signal,
      });
      signal.throwIfAborted();

      // Use first lex or vec query for snippet extraction
      const primaryQuery = searches.find(s => s.type === 'lex')?.query
        || searches.find(s => s.type === 'vec')?.query
        || searches[0]?.query || "";

      const filtered: SearchResultItem[] = results.map(r => {
        const { line, snippet } = extractSnippet(r.bestChunk, primaryQuery, 300, undefined, undefined, intent);
        return {
          docid: `#${r.docid}`,
          file: r.displayPath,
          title: r.title,
          score: Math.round(r.score * 100) / 100,
          context: r.context,
          snippet: addLineNumbers(snippet, line),
        };
      });

      return {
        content: [{ type: "text", text: formatSearchSummary(filtered, primaryQuery) }],
        structuredContent: { results: filtered },
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_get (Retrieve document)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "get",
    {
      title: "Get Document",
      description: "Retrieve the full content of a document by its file path or docid. Use paths or docids (#abc123) from search results. Suggests similar files if not found.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        file: documentPathSchema.describe("File path or docid from search results (e.g., 'pages/meeting.md', '#abc123', or 'pages/meeting.md:100' to start at line 100)"),
        fromLine: z.number().int().min(1).max(Number.MAX_SAFE_INTEGER).optional().describe("Start from this line number (1-indexed)"),
        maxLines: maxLinesSchema.describe("Maximum number of lines to return (1-100000)"),
        lineNumbers: z.boolean().optional().default(false).describe("Add line numbers to output (format: 'N: content')"),
      },
    },
    async ({ file, fromLine, maxLines, lineNumbers }, extra) => {
      const signal = ownership.signal(extra);
      signal.throwIfAborted();
      // Support :line suffix in `file` (e.g. "foo.md:120") when fromLine isn't provided
      let parsedFromLine = fromLine;
      let lookup = file;
      const colonMatch = lookup.match(/:(\d+)$/);
      if (colonMatch && colonMatch[1] && parsedFromLine === undefined) {
        parsedFromLine = parseInt(colonMatch[1], 10);
        lookup = lookup.slice(0, -colonMatch[0].length);
      }

      const result = await store.get(lookup, { includeBody: false });
      signal.throwIfAborted();

      if ("error" in result) {
        let msg = `Document not found: ${file}`;
        if (result.similarFiles.length > 0) {
          msg += `\n\nDid you mean one of these?\n${result.similarFiles.map(s => `  - ${s}`).join('\n')}`;
        }
        return {
          content: [{ type: "text", text: msg }],
          isError: true,
        };
      }

      const body = await store.getDocumentBody(result.filepath, { fromLine: parsedFromLine, maxLines }) ?? "";
      signal.throwIfAborted();
      let text = body;
      if (lineNumbers) {
        const startLine = parsedFromLine || 1;
        text = addLineNumbers(text, startLine);
      }
      if (result.context) {
        text = `<!-- Context: ${result.context} -->\n\n` + text;
      }

      return {
        content: [{
          type: "resource",
          resource: {
            uri: `qmd://${encodeQmdPath(result.displayPath)}`,
            name: result.displayPath,
            title: result.title,
            mimeType: "text/markdown",
            text,
          },
        }],
      };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_multi_get (Retrieve multiple documents)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "multi_get",
    {
      title: "Multi-Get Documents",
      description: "Retrieve multiple documents by glob pattern (e.g., 'journals/2025-05*.md') or comma-separated list. Skips files larger than maxBytes.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {
        pattern: z.string().min(1).max(16384).describe("Glob pattern or comma-separated list of file paths"),
        maxLines: maxLinesSchema.describe("Maximum lines per file (1-100000)"),
        maxBytes: z.number().int().min(1).max(10485760).optional().default(10240).describe("Skip files larger than this (default: 10240 = 10KB, max: 10MB)"),
        lineNumbers: z.boolean().optional().default(false).describe("Add line numbers to output (format: 'N: content')"),
      },
    },
    async ({ pattern, maxLines, maxBytes, lineNumbers }, extra) => {
      const signal = ownership.signal(extra);
      signal.throwIfAborted();
      const { docs, errors } = await store.multiGet(pattern, { includeBody: true, maxBytes: maxBytes || DEFAULT_MULTI_GET_MAX_BYTES });
      signal.throwIfAborted();

      if (docs.length === 0 && errors.length === 0) {
        return {
          content: [{ type: "text", text: `No files matched pattern: ${pattern}` }],
          isError: true,
        };
      }

      const content: ({ type: "text"; text: string } | { type: "resource"; resource: { uri: string; name: string; title?: string; mimeType: string; text: string } })[] = [];

      if (errors.length > 0) {
        content.push({ type: "text", text: `Errors:\n${errors.join('\n')}` });
      }

      for (const result of docs) {
        if (result.skipped) {
          content.push({
            type: "text",
            text: `[SKIPPED: ${result.doc.displayPath} - ${result.skipReason}. Use 'qmd_get' with file="${result.doc.displayPath}" to retrieve.]`,
          });
          continue;
        }

        let text = result.doc.body || "";
        if (maxLines !== undefined) {
          const lines = text.split("\n");
          text = lines.slice(0, maxLines).join("\n");
          if (lines.length > maxLines) {
            text += `\n\n[... truncated ${lines.length - maxLines} more lines]`;
          }
        }
        if (lineNumbers) {
          text = addLineNumbers(text);
        }
        if (result.doc.context) {
          text = `<!-- Context: ${result.doc.context} -->\n\n` + text;
        }

        content.push({
          type: "resource",
          resource: {
            uri: `qmd://${encodeQmdPath(result.doc.displayPath)}`,
            name: result.doc.displayPath,
            title: result.doc.title,
            mimeType: "text/markdown",
            text,
          },
        });
      }

      return { content };
    }
  );

  // ---------------------------------------------------------------------------
  // Tool: qmd_status (Index status)
  // ---------------------------------------------------------------------------

  server.registerTool(
    "status",
    {
      title: "Index Status",
      description: "Show the status of the QMD index: collections, document counts, and health information.",
      annotations: { readOnlyHint: true, openWorldHint: false },
      inputSchema: {},
    },
    async (_args, extra) => {
      const signal = ownership.signal(extra);
      signal.throwIfAborted();
      const status: StatusResult = await store.getStatus();
      signal.throwIfAborted();

      const summary = [
        `QMD Index Status:`,
        `  Total documents: ${status.totalDocuments}`,
        `  Needs embedding: ${status.needsEmbedding}`,
        `  Vector index: ${status.hasVectorIndex ? 'yes' : 'no'}`,
        `  Collections: ${status.collections.length}`,
      ];

      for (const col of status.collections) {
        summary.push(`    - ${col.name}: ${col.path} (${col.documents} docs)`);
      }

      return {
        content: [{ type: "text", text: summary.join('\n') }],
        structuredContent: status,
      };
    }
  );

  return server;
}

// =============================================================================
// Transport: stdio (default)
// =============================================================================

export async function startMcpServer(): Promise<void> {
  const configPath = getConfigPath();
  const store = await createStore({
    dbPath: getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });
  const ownership = new McpRequestOwnership();
  const server = await createMcpServer(store, ownership);
  const transport = new StdioServerTransport();
  await server.connect(new OwnedMcpTransport(transport, ownership, false));
}

// =============================================================================
// Transport: Streamable HTTP
// =============================================================================

export type HttpServerHandle = {
  httpServer: import("http").Server;
  port: number;
  stop: () => Promise<void>;
};

/** Additive controls for embedders and short, deterministic offline tests. */
export type HttpServerOptions = {
  quiet?: boolean;
  /** Per-listener daemon ownership marker, NOT an authentication credential. */
  instanceToken?: string;
  store?: QMDStore;
  dbPath?: string;
  /** Maximum JSON body size, including chunked uploads (default: 1 MiB). */
  maxBodyBytes?: number;
  /** Concurrent data uploads, released as soon as receipt finishes (default: 32). */
  maxPendingUploads?: number;
  /** Data HTTP handlers, including REST processing and GET SSE (default: 256). */
  maxHttpRequests?: number;
  /** Separate slots for health, DELETE and small cancellation uploads (default: 16).
   * At data/upload capacity, existing-session MCP POSTs may use this reserve with
   * a body of at most 4 KiB, but ONLY cancellation notifications are dispatched.
   * The reserve is bounded too: overload returns 503, never a fake 202.
   */
  maxControlRequests?: number;
  /** All TCP connections, including idle sockets and incomplete headers (default: 512). */
  maxConnections?: number;
  /** Parser-level header byte ceiling, before application allocation (default: 16 KiB). */
  maxHeaderBytes?: number;
  /** Absolute header-receipt deadline, 1..300000 ms (default: 10s). */
  headersTimeoutMs?: number;
  /** Absolute body-receipt deadline, not reset by trickling bytes (default: 10s). */
  bodyTimeoutMs?: number;
  /** Processing deadline; does not limit an established GET SSE stream (default: 120s). */
  requestTimeoutMs?: number;
  /** Bound initialized AND initializing sessions (default: 100). */
  maxSessions?: number;
  /** Bound active JSON-RPC request IDs per session, including whole batches (default: 128). */
  maxActiveRequestsPerSession?: number;
  /** Bound active JSON-RPC request IDs across this listener (default: 1024). */
  maxActiveRequests?: number;
  /** Expire sessions without MCP requests, even with an open SSE stream (default: 30 min). */
  sessionIdleTimeoutMs?: number;
  sessionSweepIntervalMs?: number;
  /** SSE comment interval; zero disables keepalive (default: 15s). */
  sseKeepAliveMs?: number;
  /** Maximum wait for a backpressured client to drain (default: 10s). */
  streamWriteTimeoutMs?: number;
};

class HttpRequestError extends Error {
  constructor(readonly status: number, message: string, readonly rpcCode = -32000) {
    super(message);
  }
}

class RequestAdmissionError extends HttpRequestError {}

class RequestIdConflictError extends RequestAdmissionError {
  constructor() { super(409, "Duplicate active JSON-RPC request ID", -32600); }
}

/** Per-session owners: Map keys deliberately retain number/string identity. */
class McpRequestOwnership {
  private readonly active = new Map<RequestId, AbortController>();
  private closed = false;

  constructor(
    private readonly maxActive = 128,
    private readonly budget = { active: 0, maxActive: 1024 },
  ) {}

  /** Validate the ENTIRE batch, then reserve synchronously before SDK dispatch. */
  reserve(messages: JSONRPCMessage[]): () => void {
    if (this.closed) throw new HttpRequestError(404, "Session closed", -32001);
    const ids = new Set<RequestId>();
    for (const message of messages) {
      if (!isJSONRPCRequest(message)) continue;
      if (ids.has(message.id) || this.active.has(message.id)) throw new RequestIdConflictError();
      ids.add(message.id);
    }
    // Admission precedes controller allocation and is atomic across the entire
    // batch and the listener. Rejected work cannot consume or steal a slot.
    if (this.active.size + ids.size > this.maxActive) {
      throw new RequestAdmissionError(429, "Session active request capacity reached; retry later");
    }
    if (this.budget.active + ids.size > this.budget.maxActive) {
      throw new RequestAdmissionError(503, "Server active request capacity reached; retry later");
    }
    const reserved = new Map<RequestId, AbortController>();
    for (const id of ids) {
      const controller = new AbortController();
      reserved.set(id, controller);
      this.active.set(id, controller);
    }
    this.budget.active += reserved.size;
    return () => {
      for (const [id, controller] of reserved) {
        if (this.active.get(id) === controller) {
          this.active.delete(id);
          this.budget.active--;
        }
      }
    };
  }

  signal(extra: { requestId: RequestId; signal: AbortSignal }): AbortSignal {
    const controller = this.active.get(extra.requestId);
    if (!controller) throw new Error("MCP request has no cancellation owner");
    // SDK close/disconnect and explicit notifications both reach the real store
    // operation; neither depends on the SDK's private controller map.
    return AbortSignal.any([extra.signal, controller.signal]);
  }

  cancel(id: RequestId | undefined, reason?: string): void {
    if (id !== undefined) this.active.get(id)?.abort(new DOMException(reason ?? "Request cancelled", "AbortError"));
  }

  close(): void {
    this.closed = true;
    const controllers = [...this.active.values()];
    this.budget.active -= this.active.size;
    this.active.clear();
    for (const controller of controllers) controller.abort(new DOMException("Session closed", "AbortError"));
  }
}

/** Only public Transport callbacks; do not patch SDK methods or dependency bytes. */
class OwnedMcpTransport implements Transport {
  onmessage?: Transport["onmessage"];
  onclose?: Transport["onclose"];
  onerror?: Transport["onerror"];
  private readonly stdioReservations = new Map<RequestId, () => void>();

  constructor(
    private readonly delegate: Transport,
    private readonly ownership: McpRequestOwnership,
    private readonly httpReservations: boolean,
  ) {}

  get sessionId(): string | undefined { return this.delegate.sessionId; }
  setProtocolVersion(version: string): void { this.delegate.setProtocolVersion?.(version); }

  async start(): Promise<void> {
    this.delegate.onmessage = (message, extra) => {
      if (!this.httpReservations && isJSONRPCRequest(message)) {
        try { this.stdioReservations.set(message.id, this.ownership.reserve([message])); }
        catch (error) {
          if (!(error instanceof RequestAdmissionError)) throw error;
          // An uncorrelated MCP error (no id) cannot settle the original owner.
          void this.delegate.send({ jsonrpc: "2.0", error: {
            code: error.rpcCode, message: error.message,
          } }).catch(failure => {
            this.onerror?.(failure);
            void this.close().catch(error => this.onerror?.(error));
          });
          return;
        }
      }
      this.onmessage?.(message, extra);
    };
    this.delegate.onerror = error => this.onerror?.(error);
    this.delegate.onclose = () => {
      this.ownership.close();
      this.stdioReservations.clear();
      this.onclose?.(); // Preserve the SDK's abort/timer/handler cleanup.
    };
    await this.delegate.start();
  }

  async send(message: JSONRPCMessage, options?: TransportSendOptions): Promise<void> {
    const terminal = isJSONRPCResultResponse(message) || isJSONRPCErrorResponse(message);
    const id = terminal ? message.id : undefined;
    const release = !this.httpReservations && id !== undefined ? this.stdioReservations.get(id) : undefined;
    try { await this.delegate.send(message, options); }
    catch (error) { await this.close(); throw error; }
    if (release && id !== undefined) {
      release();
      this.stdioReservations.delete(id);
    }
    // HTTP reservations belong to the whole POST, not individual batch members.
    // The HTTP finally releases them only after response/stream cleanup.
  }

  async close(): Promise<void> {
    this.ownership.close();
    await this.delegate.close();
  }
}

function boundedOption(value: number | undefined, fallback: number, name: string, min = 1, max = 2147483647): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < min || result > max) {
    throw new RangeError(`${name} must be an integer in ${min}..${max}`);
  }
  return result;
}

/** Stop awaiting work on disconnect/deadline without leaving abort listeners behind. */
function abortable<T>(work: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const onAbort = () => { signal.removeEventListener("abort", onAbort); reject(signal.reason); };
    signal.addEventListener("abort", onAbort, { once: true });
    work.then(value => {
      signal.removeEventListener("abort", onAbort);
      resolve(value);
    }, error => {
      signal.removeEventListener("abort", onAbort);
      reject(error);
    });
    if (signal.aborted) onAbort();
  });
}

/** Event-based collection avoids async-iterator destruction before a 4xx can flush. */
function collectBody(req: IncomingMessage, maxBytes: number, timeoutMs: number, signal: AbortSignal): Promise<string> {
  const length = req.headers["content-length"];
  if (length !== undefined && Number(length) > maxBytes) {
    return Promise.reject(new HttpRequestError(413, "Request body too large"));
  }
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    const cleanup = () => {
      clearTimeout(timer);
      req.off("data", onData);
      req.off("end", onEnd);
      req.off("error", onError);
      signal.removeEventListener("abort", onAbort);
    };
    const fail = (error: unknown) => {
      cleanup();
      chunks.length = 0;
      req.pause();
      reject(error);
    };
    const onError = () => fail(new HttpRequestError(400, "Incomplete request body", -32700));
    const onAbort = () => fail(signal.reason);
    const onData = (chunk: Buffer) => {
      size += chunk.length;
      if (size > maxBytes) fail(new HttpRequestError(413, "Request body too large"));
      else chunks.push(chunk);
    };
    const onEnd = () => {
      cleanup();
      try {
        resolve(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks, size)));
      } catch {
        reject(new HttpRequestError(400, "Invalid UTF-8 JSON body", -32700));
      }
    };
    const timer = setTimeout(() => fail(new HttpRequestError(408, "Request body timed out")), timeoutMs);
    timer.unref();
    req.on("data", onData);
    req.once("end", onEnd);
    req.once("error", onError);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
}

function requireJson(req: IncomingMessage): void {
  if (req.headers["content-type"]?.split(";")[0]?.trim().toLowerCase() !== "application/json") {
    throw new HttpRequestError(415, "Content-Type must be application/json");
  }
  if (req.headers["content-encoding"] && req.headers["content-encoding"].toLowerCase() !== "identity") {
    throw new HttpRequestError(415, "Compressed request bodies are not supported");
  }
}

function parseJson(raw: string): unknown {
  try { return JSON.parse(raw); }
  catch { throw new HttpRequestError(400, "Invalid JSON body", -32700); }
}

function waitForDrain(res: ServerResponse, signal: AbortSignal, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      res.off("drain", onDrain);
      signal.removeEventListener("abort", onAbort);
    };
    const onDrain = () => { cleanup(); resolve(); };
    const onAbort = () => { cleanup(); reject(signal.reason); };
    const timer = setTimeout(() => {
      cleanup(); reject(new HttpRequestError(504, "Response client did not drain"));
    }, timeoutMs);
    timer.unref();
    res.once("drain", onDrain);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
}

/** One Web-stream chunk at a time; never buffer a whole (possibly infinite) response. */
async function writeWebResponse(response: Response, res: ServerResponse, signal: AbortSignal, timeoutMs: number): Promise<void> {
  signal.throwIfAborted();
  res.writeHead(response.status, Object.fromEntries(response.headers));
  res.flushHeaders(); // GET SSE must return headers before the first event/keepalive.
  const end = () => abortable(new Promise<void>(resolve => res.end(resolve)), signal);
  if (!response.body) { await end(); return; }
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel(signal.reason).catch(() => {}); };
  signal.addEventListener("abort", cancel, { once: true });
  let complete = false;
  try {
    if (signal.aborted) cancel();
    while (true) {
      const { done, value } = await abortable(reader.read(), signal);
      signal.throwIfAborted();
      if (done) { complete = true; break; }
      if (!res.write(value)) await waitForDrain(res, signal, timeoutMs);
    }
    await end();
  } finally {
    signal.removeEventListener("abort", cancel);
    if (!complete) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

type HttpSession = {
  transport: WebStandardStreamableHTTPServerTransport;
  ownership: McpRequestOwnership;
  server?: McpServer;
  id?: string;
  lastUsed: number;
  inFlight: number;
  closed: boolean;
  disposePromise?: Promise<void>;
  requests: Set<AbortController>;
  deleteRequest?: AbortController;
};

/**
 * Loopback-only MCP/REST listener. POST replies remain JSON; GET uses real SSE.
 * Browser Origins are restricted to this listener, not every localhost port.
 */
export async function startMcpHttpServer(port: number, options: HttpServerOptions = {}): Promise<HttpServerHandle> {
  // Bun 1.3.8's node:http compatibility does not reliably emit response/socket
  // close after a completed POST disconnects, and can retain incomplete uploads
  // despite Connection: close. Do not advertise bounded/cancellable HTTP on a
  // transport that cannot provide that contract. Stdio is unaffected.
  if (process.versions.bun) {
    throw new Error(
      "QMD MCP HTTP requires Node.js >=22: Bun's node:http cannot reliably cancel disconnected requests. " +
      "Run QMD_RUNTIME=node qmd mcp --http (source checkout: node --import tsx src/cli/qmd.ts mcp --http). " +
      "Bun remains supported for MCP stdio.",
    );
  }
  const maxBodyBytes = boundedOption(options.maxBodyBytes, 1048576, "maxBodyBytes");
  const maxPendingUploads = boundedOption(options.maxPendingUploads, 32, "maxPendingUploads");
  const maxHttpRequests = boundedOption(options.maxHttpRequests, 256, "maxHttpRequests");
  const maxControlRequests = boundedOption(options.maxControlRequests, 16, "maxControlRequests");
  const maxConnections = boundedOption(options.maxConnections, 512, "maxConnections");
  const maxHeaderBytes = boundedOption(options.maxHeaderBytes, 16384, "maxHeaderBytes");
  // Keep Node's default 300s request timeout as an additional parser safeguard.
  const headersTimeoutMs = boundedOption(options.headersTimeoutMs, 10000, "headersTimeoutMs", 1, 300000);
  const controlBodyBytes = Math.min(maxBodyBytes, 4096);
  const bodyTimeoutMs = boundedOption(options.bodyTimeoutMs, 10000, "bodyTimeoutMs");
  const requestTimeoutMs = boundedOption(options.requestTimeoutMs, 120000, "requestTimeoutMs");
  const maxSessions = boundedOption(options.maxSessions, 100, "maxSessions");
  const maxActivePerSession = boundedOption(options.maxActiveRequestsPerSession, 128, "maxActiveRequestsPerSession");
  const requestBudget = { active: 0, maxActive: boundedOption(options.maxActiveRequests, 1024, "maxActiveRequests") };
  const idleTimeoutMs = boundedOption(options.sessionIdleTimeoutMs, 1800000, "sessionIdleTimeoutMs");
  const sweepIntervalMs = boundedOption(options.sessionSweepIntervalMs, Math.min(60000, Math.max(1, Math.floor(idleTimeoutMs / 2))), "sessionSweepIntervalMs");
  const keepAliveMs = boundedOption(options.sseKeepAliveMs, 15000, "sseKeepAliveMs", 0);
  const writeTimeoutMs = boundedOption(options.streamWriteTimeoutMs, 10000, "streamWriteTimeoutMs");
  const instanceToken = options.instanceToken ?? process.env.QMD_MCP_INSTANCE_TOKEN ?? randomUUID();
  const configPath = getConfigPath();
  const isOwnedStore = !options.store;
  const store = options.store ?? await createStore({
    dbPath: options.dbPath ?? getDefaultDbPath(),
    ...(existsSync(configPath) ? { configPath } : {}),
  });
  let defaultCollectionNames: string[];
  try { defaultCollectionNames = await store.getDefaultCollectionNames(); }
  catch (error) { if (isOwnedStore) await store.close(); throw error; }

  const sessions = new Map<string, HttpSession>();
  const liveSessions = new Set<HttpSession>(); // Includes initialization reservations.
  const closingSessions = new Set<Promise<void>>();
  const requests = new Set<AbortController>();
  let activeHttpRequests = 0;
  let activeControlRequests = 0;
  let pendingUploads = 0;
  const startTime = Date.now();
  let actualPort = 0;
  let stopping = false;
  let stopPromise: Promise<void> | undefined;
  const log = (message: string) => { if (!options.quiet) console.error(message); };
  const allowedHosts = () => ["localhost", "127.0.0.1", "[::1]"].map(host => new URL(`http://${host}:${actualPort}`).host);
  const allowedOrigins = () => allowedHosts().map(host => `http://${host}`);

  function disposeSession(session: HttpSession, except?: AbortController): Promise<void> {
    if (session.disposePromise) return session.disposePromise;
    session.closed = true;
    liveSessions.delete(session);
    if (session.id) sessions.delete(session.id);
    for (const controller of session.requests) {
      if (controller !== except) controller.abort(new HttpRequestError(404, "Session closed", -32001));
    }
    session.ownership.close();
    // Defer until the promise is assigned: server.close() invokes onclose itself.
    session.disposePromise = Promise.resolve().then(async () => {
      try { await session.server?.close(); }
      finally { await session.transport.close(); }
    }).catch(error => { log(`Session cleanup error: ${String(error)}`); });
    closingSessions.add(session.disposePromise);
    void session.disposePromise.finally(() => closingSessions.delete(session.disposePromise!));
    return session.disposePromise;
  }

  function expireSessions(): void {
    const now = performance.now();
    for (const session of liveSessions) {
      if (session.inFlight === 0 && now - session.lastUsed >= idleTimeoutMs) void disposeSession(session);
    }
  }

  function reserveSession(ownership: McpRequestOwnership): HttpSession {
    expireSessions();
    if (liveSessions.size >= maxSessions) {
      let oldest: HttpSession | undefined;
      for (const candidate of liveSessions) {
        if (candidate.inFlight === 0 && (!oldest || candidate.lastUsed < oldest.lastUsed)) oldest = candidate;
      }
      if (!oldest) throw new RequestAdmissionError(503, "Session capacity reached; retry later");
      void disposeSession(oldest); // Removes the slot synchronously, before any await.
    }
    const transport = new WebStandardStreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(), enableJsonResponse: true,
      maxRequestBodySize: maxBodyBytes, keepAliveMs,
      enableDnsRebindingProtection: true, allowedHosts: allowedHosts(), allowedOrigins: allowedOrigins(),
      onsessioninitialized: (id: string) => {
        if (session.closed || stopping) throw new Error("Session initialization was cancelled");
        session.id = id;
        sessions.set(id, session);
      },
    });
    const session: HttpSession = {
      transport, ownership,
      lastUsed: performance.now(), inFlight: 0, closed: false, requests: new Set(),
    };
    liveSessions.add(session);
    return session;
  }

  async function connectSession(session: HttpSession, controller: AbortController): Promise<void> {
    try {
      session.server = await createMcpServer(store, session.ownership);
      // A deadline/stop may have disposed the reservation while building instructions.
      if (session.closed || stopping) {
        await session.server.close();
        throw new HttpRequestError(503, "Server/session closed during initialization");
      }
      // Do not overwrite transport.onclose after connect: the SDK owns that callback
      // and uses it to abort handlers and clear request/timer state.
      session.server.server.onclose = () => { void disposeSession(session, session.deleteRequest); };
      await session.server.connect(new OwnedMcpTransport(session.transport, session.ownership, true));
    } catch (error) {
      await disposeSession(session, controller);
      throw error;
    }
  }

  function guardLoopbackRequest(req: IncomingMessage): void {
    // Node otherwise merges duplicate Origin headers and silently picks one Host.
    for (const name of ["host", "origin"]) {
      let count = 0;
      for (let i = 0; i < req.rawHeaders.length; i += 2) {
        if (req.rawHeaders[i]?.toLowerCase() === name) count++;
      }
      if (count > 1) throw new HttpRequestError(403, `Duplicate ${name} header`);
    }
    if (!req.headers.host || !allowedHosts().includes(req.headers.host.toLowerCase())) {
      throw new HttpRequestError(403, "Untrusted Host header");
    }
    const origin = req.headers.origin;
    // Missing Origin is valid for native CLI clients; null, foreign, malformed,
    // and other loopback ports are NOT trusted browser origins.
    if (origin !== undefined && !allowedOrigins().includes(origin)) {
      throw new HttpRequestError(403, "Untrusted Origin header");
    }
  }

  const httpServer = createServer({
    maxHeaderSize: maxHeaderBytes, headersTimeout: headersTimeoutMs,
    connectionsCheckingInterval: Math.min(headersTimeoutMs, 1000),
  }, async (nodeReq: IncomingMessage, nodeRes: ServerResponse) => {
    let controller: AbortController | undefined;
    let deadline: ReturnType<typeof setTimeout> | undefined;
    let cleanupRequest: (() => void) | undefined;
    let releaseHttpRequest: (() => void) | undefined;
    let releaseUpload: (() => void) | undefined;
    let session: HttpSession | undefined;
    let created = false;
    let registered = false;
    let releaseOwnership: (() => void) | undefined;
    let pathname = "/";
    try {
      guardLoopbackRequest(nodeReq); // Before JSON, store calls, session allocation, or route dispatch.
      if (stopping) throw new HttpRequestError(503, "Server is stopping");
      if (!nodeReq.url?.startsWith("/")) throw new HttpRequestError(400, "Invalid request target");
      const listenerOrigin = `http://localhost:${actualPort}`;
      let requestUrl: URL;
      try { requestUrl = new URL(nodeReq.url, listenerOrigin); }
      catch { throw new HttpRequestError(400, "Invalid request target"); }
      if (requestUrl.origin !== new URL(listenerOrigin).origin) throw new HttpRequestError(400, "Foreign request target");
      pathname = requestUrl.pathname;
      const isSearchPost = (pathname === "/query" || pathname === "/search") && nodeReq.method === "POST";
      const isMcpPost = pathname === "/mcp" && nodeReq.method === "POST";
      const isUpload = isSearchPost || isMcpPost;
      if (isUpload) requireJson(nodeReq);
      if (pathname === "/mcp") {
        if (!["POST", "GET", "DELETE"].includes(nodeReq.method ?? "")) {
          nodeRes.setHeader("Allow", "GET, POST, DELETE");
          throw new HttpRequestError(405, "Method not allowed");
        }
        if (isMcpPost) {
          const accept = nodeReq.headers.accept;
          if (!accept?.includes("application/json") || !accept.includes("text/event-stream")) {
            throw new HttpRequestError(406, "Client must accept application/json and text/event-stream");
          }
        }
      }
      if ((pathname === "/mcp" || pathname === "/health") && ["GET", "DELETE"].includes(nodeReq.method ?? "")
        && (nodeReq.headers["transfer-encoding"] || Number(nodeReq.headers["content-length"] ?? 0) > 0)) {
        throw new HttpRequestError(400, "GET/DELETE request bodies are not supported");
      }
      const length = nodeReq.headers["content-length"];
      if (isUpload && length !== undefined && Number(length) > maxBodyBytes) {
        throw new HttpRequestError(413, "Request body too large");
      }
      const overload = activeHttpRequests >= maxHttpRequests
        ? "Server HTTP request capacity reached; retry later"
        : isUpload && pendingUploads >= maxPendingUploads ? "Pending upload capacity reached; retry later" : undefined;
      let controlOnly = false;
      let controlRequest = (pathname === "/health" && nodeReq.method === "GET")
        || (pathname === "/mcp" && nodeReq.method === "DELETE");
      if (!controlRequest && overload) {
        // A standard MCP cancellation is itself a POST: provide a small reserve
        // without trusting a header's claim that arbitrary work is control work.
        // Unknown/chunked lengths are allowed, but collected under the 4 KiB cap.
        const sid = nodeReq.headers["mcp-session-id"];
        if (isMcpPost && typeof sid === "string" && sessions.has(sid)
          && (length === undefined || Number(length) <= controlBodyBytes)) {
          controlRequest = controlOnly = true;
        } else throw new HttpRequestError(503, overload);
      }
      if (controlRequest) {
        if (activeControlRequests >= maxControlRequests) {
          throw new HttpRequestError(503, "Server control request capacity reached; retry later");
        }
        activeControlRequests++;
        releaseHttpRequest = () => { activeControlRequests--; };
      } else {
        activeHttpRequests++;
        releaseHttpRequest = () => { activeHttpRequests--; };
        if (isUpload) {
          pendingUploads++;
          let held = true;
          releaseUpload = () => { if (held) { held = false; pendingUploads--; } };
        }
      }
      // All counters are reserved synchronously BEFORE controllers, timers, body
      // chunks or SDK objects. REST and MCP share these pre-body HTTP bounds.
      const admittedController = new AbortController();
      controller = admittedController;
      const signal = admittedController.signal;
      requests.add(admittedController);
      const onDisconnect = () => admittedController.abort(new HttpRequestError(400, "Client disconnected"));
      const onClose = () => { if (!nodeRes.writableFinished) onDisconnect(); };
      nodeReq.once("aborted", onDisconnect);
      nodeRes.once("close", onClose);
      nodeRes.once("error", onDisconnect);
      deadline = setTimeout(() => admittedController.abort(new HttpRequestError(504, "Request processing timed out")), requestTimeoutMs);
      deadline.unref();
      cleanupRequest = () => {
        requests.delete(admittedController);
        nodeReq.off("aborted", onDisconnect);
        nodeRes.off("close", onClose);
        nodeRes.off("error", onDisconnect);
      };
      const receiveBody = async () => {
        try { return await collectBody(nodeReq, controlOnly ? controlBodyBytes : maxBodyBytes, bodyTimeoutMs, signal); }
        finally { releaseUpload?.(); } // Upload slots must not be held by long-running queries.
      };
      if (pathname === "/health" && nodeReq.method === "GET") {
        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(JSON.stringify({
          status: "ok", uptime: Math.floor((Date.now() - startTime) / 1000), pid: process.pid, instanceToken,
        }));
        return;
      }
      if (isSearchPost) {
        const body = parseJson(await receiveBody());
        const parsed = searchInputSchema.safeParse(body);
        if (!parsed.success) {
          throw new HttpRequestError(400, `Invalid search parameters: ${parsed.error.issues.slice(0, 5).map(issue => `${issue.path.join(".")}: ${issue.message}`).join("; ")}`);
        }
        const params = parsed.data;
        const effectiveCollections = params.collections ?? defaultCollectionNames;
        const results = await abortable(store.search({
          queries: params.searches,
          collections: effectiveCollections.length > 0 ? effectiveCollections : undefined,
          limit: params.limit, minScore: params.minScore, intent: params.intent, rerank: params.rerank,
          signal,
        }), signal);
        signal.throwIfAborted();
        const primaryQuery = params.searches.find(s => s.type === "lex")?.query
          || params.searches.find(s => s.type === "vec")?.query || params.searches[0]!.query;
        const formatted = results.map(r => {
          const { line, snippet } = extractSnippet(r.bestChunk, primaryQuery, 300, undefined, undefined, params.intent);
          return {
            docid: `#${r.docid}`, file: r.displayPath, title: r.title,
            score: Math.round(r.score * 100) / 100, context: r.context, snippet: addLineNumbers(snippet, line),
          };
        });
        nodeRes.writeHead(200, { "Content-Type": "application/json" });
        nodeRes.end(JSON.stringify({ results: formatted }));
        return;
      }
      if (pathname !== "/mcp") {
        nodeRes.writeHead(404); nodeRes.end("Not Found"); return;
      }
      let rawBody: string | undefined;
      let body: unknown;
      let messages: JSONRPCMessage[] = [];
      if (nodeReq.method === "POST") {
        rawBody = await receiveBody();
        body = parseJson(rawBody);
        const rawMessages = Array.isArray(body) ? body : [body];
        if (rawMessages.length === 0 || rawMessages.length > 100) {
          throw new HttpRequestError(400, "Invalid JSON-RPC message or batch", -32600);
        }
        const parsedMessages = rawMessages.map(message => JSONRPCMessageSchema.safeParse(message));
        if (parsedMessages.some(message => !message.success)) {
          throw new HttpRequestError(400, "Invalid JSON-RPC message or batch", -32600);
        }
        messages = parsedMessages.map(message => message.data!);
      }
      const sessionId = nodeReq.headers["mcp-session-id"];
      if (typeof sessionId === "string") {
        expireSessions();
        session = sessions.get(sessionId);
        if (!session) throw new HttpRequestError(404, "Session not found", -32001);
        if (nodeReq.method === "POST") releaseOwnership = session.ownership.reserve(messages);
      } else if (nodeReq.method === "POST" && isInitializeRequest(body)) {
        // Hold the ACTUAL global budget before any session allocation/eviction,
        // including across connectSession's asynchronous instruction reads.
        const ownership = new McpRequestOwnership(maxActivePerSession, requestBudget);
        releaseOwnership = ownership.reserve(messages);
        session = reserveSession(ownership);
        created = true;
      } else {
        throw new HttpRequestError(400, "Bad Request: Missing session ID");
      }
      // Check ID conflicts first so data saturation cannot turn an existing
      // owner's duplicate 409 into a 503 or accidentally dispatch its cancel.
      if (controlOnly && !messages.every(message => !isJSONRPCRequest(message)
        && CancelledNotificationSchema.safeParse(message).success)) {
        throw new RequestAdmissionError(503, overload!);
      }
      session.lastUsed = performance.now();
      session.requests.add(controller);
      registered = true;
      // Long-lived GET streams do not prevent idle expiry or capacity eviction.
      if (nodeReq.method !== "GET") session.inFlight++;
      if (created) await abortable(connectSession(session, controller), signal);
      signal.throwIfAborted();
      if (nodeReq.method === "DELETE") session.deleteRequest = controller;
      const headers = new Headers();
      for (const [key, value] of Object.entries(nodeReq.headers)) {
        if (typeof value === "string") headers.set(key, key === "host" ? value.toLowerCase() : value);
      }
      const request = new Request(requestUrl.href, {
        method: nodeReq.method, headers, signal, ...(rawBody !== undefined ? { body: rawBody } : {}),
      });
      const responseTask = session.transport.handleRequest(request, { parsedBody: body });
      // A late Response must not orphan an SSE reader after the HTTP client left.
      void responseTask.then(response => {
        if (signal.aborted) void response.body?.cancel().catch(() => {});
      }, () => {});
      const response = await abortable(responseTask, signal);
      if (created && (response.status >= 400 || !session.id)) await disposeSession(session, controller);
      if (nodeReq.method === "GET") clearTimeout(deadline);
      await writeWebResponse(response, nodeRes, signal, writeTimeoutMs);
      if (nodeReq.method === "DELETE" && response.status < 300) await disposeSession(session, controller);
    } catch (error) {
      // Rejected admission never reached the SDK and must not cancel the
      // original request, close its session, or release its reservation.
      if (session && (created || !(error instanceof RequestAdmissionError))
        && (created || nodeReq.method !== "GET" || !controller?.signal.aborted)) await disposeSession(session, controller);
      if (!nodeRes.destroyed) {
        if (nodeRes.headersSent) nodeRes.destroy();
        else {
          const failure = error instanceof HttpRequestError ? error : new HttpRequestError(500, "Internal Server Error");
          if (failure.status === 500) log(`HTTP handler error: ${String(error)}`);
          nodeReq.pause();
          // Admission errors have a fully received, validated body. Do not
          // destroy their socket: an earlier pipelined POST on that connection
          // may still own the original operation which we deliberately kept.
          const closeConnection = !(failure instanceof RequestAdmissionError);
          // Do not drain an untrusted/incomplete upload indefinitely. Closing after
          // the response lets clients receive a valid 4xx, unlike req.destroy().
          nodeRes.writeHead(failure.status, {
            "Content-Type": "application/json", ...(closeConnection ? { Connection: "close" } : {}),
          });
          nodeRes.end(JSON.stringify(pathname === "/mcp" ? {
            jsonrpc: "2.0", error: { code: failure.rpcCode, message: failure.message }, id: null,
          } : { error: failure.message }));
          // Give errors a bounded flush window, then close this connection even
          // if a peer stops reading or never finishes its untrusted upload.
          if (closeConnection) {
            const socket = nodeReq.socket;
            setTimeout(() => {
              nodeRes.destroy();
              socket.destroy();
              nodeReq.destroy();
            }, 50).unref();
          }
        }
      }
    } finally {
      releaseOwnership?.();
      releaseUpload?.();
      releaseHttpRequest?.();
      clearTimeout(deadline);
      cleanupRequest?.();
      if (registered && session && controller) {
        session.requests.delete(controller);
        if (nodeReq.method !== "GET") session.inFlight--;
        if (!session.closed) session.lastUsed = performance.now();
        if (session.deleteRequest === controller) session.deleteRequest = undefined;
      }
    }
  });
  httpServer.maxConnections = maxConnections;

  try {
    await new Promise<void>((resolve, reject) => {
      const onError = (error: Error) => reject(error);
      httpServer.once("error", onError);
      httpServer.listen(port, "localhost", () => { httpServer.off("error", onError); resolve(); });
    });
  } catch (error) {
    if (isOwnedStore) await store.close();
    throw error;
  }
  actualPort = (httpServer.address() as import("net").AddressInfo).port;
  const sweep = setInterval(expireSessions, sweepIntervalMs);
  sweep.unref();

  const stop = (): Promise<void> => {
    if (stopPromise) return stopPromise;
    stopping = true;
    clearInterval(sweep);
    process.off("SIGTERM", onSigterm);
    process.off("SIGINT", onSigint);
    stopPromise = (async () => {
      const closed = new Promise<void>((resolve, reject) => httpServer.close(error => error ? reject(error) : resolve()));
      for (const controller of requests) controller.abort(new HttpRequestError(503, "Server is stopping"));
      await Promise.all([...liveSessions].map(session => disposeSession(session)));
      await Promise.all(closingSessions); // Include evictions/expiry already removed from the live map.
      // close() alone waits forever for SSE/partial uploads. These are our own
      // listener's sockets, not a process-global service or injected store.
      httpServer.closeAllConnections();
      await closed;
      if (isOwnedStore) await store.close();
    })();
    return stopPromise;
  };
  function onSignal(): void {
    void stop().then(() => process.exit(0), error => { console.error(error); process.exit(1); });
  }
  const onSigterm = onSignal;
  const onSigint = onSignal;
  process.on("SIGTERM", onSigterm);
  process.on("SIGINT", onSigint);
  log(`QMD MCP server listening on http://localhost:${actualPort}/mcp`);
  return { httpServer, port: actualPort, stop };
}

// Run if this is the main module
if (fileURLToPath(import.meta.url) === process.argv[1] || process.argv[1]?.endsWith("/server.ts") || process.argv[1]?.endsWith("/server.js")) {
  startMcpServer().catch(console.error);
}
