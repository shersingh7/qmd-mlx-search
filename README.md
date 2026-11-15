# QMD — Query Markup Documents (MLX-Native)

> **Fork installation:** this repository is `shersingh7/qmd-mlx-search`. Build it from source or install a verified tarball from this checkout to use these MLX changes. The registry package `@tobilu/qmd` and its `npx` examples refer to the upstream release, not automatically to this fork. The package/import name remains unchanged for SDK compatibility; no registry publication is implied.

An on-device search engine for everything you need to remember. Index your markdown notes, meeting transcripts, documentation, and knowledge bases. Search with keywords or natural language.

QMD combines BM25 full-text search, vector semantic search, and LLM re-ranking — all running locally. Two embedding backends: **MLX-native** (Apple Silicon GPU; performance is model/workload dependent) or **GGUF** (node-llama-cpp, CPU/Metal).

![QMD Architecture](assets/qmd-architecture.png)

You can read more about QMD's progress in the [CHANGELOG](CHANGELOG.md).

---

## MLX-Native Embeddings (Apple Silicon)

On Apple Silicon (M-series MacBooks/Macs), QMD supports **native MLX embeddings** via a dedicated Python server powered by Apple's MLX framework. The MLX runtime runs models directly on the unified memory GPU with a single-flight execution owner, model-aware pooling, token-once batching, and bounded binary transport. The client validates the serving embedding descriptor on every production inference response. Binary typed-array views reduce decoding overhead, but transport and public `number[]` conversions are not end-to-end zero-copy.

### Architectural Highlights

| Feature | GGUF (node-llama-cpp) | MLX-Native (`scripts/mlx_embed_server.py`) |
|---|---|---|
| **GPU Execution** | Metal via llama.cpp | Direct MLX Metal Compute (Dedicated Worker Thread) |
| **Model Families** | GGUF quantized models | Explicit standard BERT MiniLM/BGE and supported Qwen causal adapters; unqualified Nomic/Gemma architectures fail closed |
| **Hidden State Extraction** | Native embedding GGUFs | Architectural hidden state extraction (Encoder BERT / Causal LM backbone) |
| **Pooling & Norm** | Model-specific GGUF | Model-specific mean / CLS / last-token pooling + float32 accumulation and L2 normalization |
| **Wire Protocol** | In-process native bindings | Bounded HTTP + Little-endian Binary Float32 (`/embed-bin`) |
| **Safety & Contracts** | GGUF model path checks | SHA-256 `EmbeddingDescriptor` space identity (fail-closed backend selection) |

### Quick Start (MLX Mode)

```bash
# 1. Install Python dependencies in a repo-local virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/mlx_server_requirements.txt

# 2. Start the MLX embedding server
python scripts/mlx_embed_server.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --dtype float32 \
  --preload \
  --port 8787

# 3. Configure QMD to use MLX backend
export QMD_EMBED_BACKEND=mlx
export QMD_MLX_EMBED_URL=http://127.0.0.1:8787

# 4. Use QMD as normal — embeddings use MLX Metal acceleration
qmd embed
qmd query "search query"
```

The MLX server exposes standard REST and binary endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/embed` | POST | JSON embedding request (array of texts) |
| `/embed-bin` | POST | High-throughput binary Float32 transport (`[count: i32][dims: i32][floats]`) |
| `/rerank` | POST | Rerank documents for a query (`{query, documents}` → `{scores}`) — requires `--rerank-model` |
| `/generate` | POST | Query-expansion completions (`{prompt, max_tokens, temperature}` → `{text}`) — requires `--generate-model` |
| `/tokenize` | POST | Token count validation and single-pass length profiling |
| `/descriptor` | GET | Model descriptor fingerprint (model, pooling, dims, max_tokens + rerank/generate sub-descriptors) |
| `/health` | GET | Process liveness check (includes rerank/generate model names) |
| `/ready` | GET | Readiness check (returns 200 once ALL configured models are loaded and warm, 503 during load) |
| `/memory` | GET | Active and peak Metal unified memory usage |
| `/stats` | GET | Total requests, average latency, and uptime |

### MLX Server Flags

```
--model PATH             Supported HF repo or local path (default: sentence-transformers/all-MiniLM-L6-v2)
--rerank-model PATH      Reranker model; enables /rerank (e.g. local Qwen3-Reranker-4B MLX 4-bit)
--generate-model PATH    Generation model; enables /generate (e.g. mlx-community/Qwen3-1.7B-4bit)
--port 8787              Server port (default: 8787)
--host 127.0.0.1         Bind address (default: 127.0.0.1 loopback)
--dtype float32          Floating compute precision: float32, float16, bfloat16
--quantization auto      Verify stored weights: auto, none, 4bit, 8bit; not on-load quantization
--max-batch-tokens N     Memory/model-aware padded-token budget (default: automatic)
--max-bulk-microbatch-tokens N  Bulk scheduling target (default: 512; larger is opt-in)
--preload                Preload and validate model weights at startup before serving
```

### HTTP host runtime

**MCP/REST HTTP hosting requires Node.js 22 or newer.** Bun 1.3.8's `node:http` compatibility layer did not reliably propagate request disconnects, so HTTP startup under Bun is explicitly refused rather than pretending that cancellation works. Bun remains usable for the CLI, SDK and MCP stdio. For HTTP, use a Node/npm installation (with a Node-compatible `better-sqlite3` addon) and `QMD_RUNTIME=node qmd mcp --http`.

Development CLI scripts use Node's `--import tsx` loader rather than a separate
`tsx` CLI wrapper, avoiding its Unix IPC-socket collisions under long temporary paths.

The full unit suite uses **Vitest**: run `bun run test` or `npm test`, not `bun test` against Vitest's APIs. CI separately installs the tarball and exercises native Node and Bun CLI/SQLite/SDK/MCP-stdio behavior; only Node receives HTTP/SSE feature acceptance, and Bun's safe HTTP refusal is tested explicitly.

### Daemon supervision

The persistent daemon requires an **explicit model selection**. The selected model must already exist locally (an absolute snapshot directory or a cached Hugging Face repository ID); startup never downloads weights. Unlike the interactive Python server's convenience default, no legacy Qwen checkpoint is silently selected.

```sh
scripts/qmd-mlx-daemon.sh status   # launchd state + /ready probe
MLX_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
  scripts/qmd-mlx-daemon.sh start  # install the explicitly selected cached model
scripts/qmd-mlx-daemon.sh stop     # bootout
QMD_EMBED_BACKEND=mlx qmd status   # includes an MLX Daemon section (embed/rerank/generate models)
```

The separate `qmd mcp --http --daemon` command also waits for the owned listener's health identity before reporting startup. Its PID record now includes the port and instance token; legacy integer-only or mismatched records are deliberately refused for process termination.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QMD_EMBED_BACKEND` | `gguf` | Active embedding backend: `mlx` or `gguf` |
| `QMD_MLX_EMBED_URL` | `http://127.0.0.1:8787` | MLX server URL |
| `QMD_MLX_CONCURRENCY` | `2` | Bounded concurrent in-flight batch requests |
| `QMD_MLX_FALLBACK` | `false` | Legacy configuration; embeddings remain fail-closed and never substitute a different GGUF space |
| `QMD_MLX_RERANK` | unset (GGUF) | Set to `1` to route reranking through the MLX daemon; qualify ranking quality and memory on your corpus before changing backends |
| `QMD_MLX_EXPAND` | unset (GGUF) | Set to `1` to route query expansion through the MLX daemon (opt-in pending Recall@10 eval) |
| `QMD_MLX_RERANK_FALLBACK` / `QMD_MLX_EXPAND_FALLBACK` | `1` | Set to `0` for fail-closed (error instead of GGUF fallback) |
| `MLX_RERANK_MODEL` / `MLX_GENERATE_MODEL` | unset | Daemon-side model selection (also `--rerank-model` / `--generate-model`) |
| `MLX_IDLE_UNLOAD_S` | `300` | Seconds of inactivity before weight eviction; reload cost depends on the model. `0` disables idle eviction |
| `XDG_CACHE_HOME` | `~/.cache` | Cache directory location |

Expansion and reranking cache keys include the verified serving-stage identity,
not just a model name. Fallback results must match their actual backend identity.
Embedding selection does not automatically move expansion or reranking to MLX;
the stage flags above are independent.

### Verification and performance scope

Use `scripts/bench_apple_silicon.py` for bounded, offline, real-Metal measurements from an existing local checkpoint. It records source identity, raw latency samples, actual token counts, current/peak RSS, active/cache/peak Metal memory and teardown state. `scripts/check_embedding_reference.py` compares its saved BERT vectors with the same local Hugging Face model on CPU.

The current audit found that the earlier MiniLM/BERT adapter omitted learned segment-zero embeddings: it was self-consistent but disagreed with the trained model. The repaired adapter passes the [same-model numerical reference fixture](docs/benchmarks/apple-silicon-audit/rc-core-minilm-float32-512-reference.json). Old timings from semantically wrong vectors are **not** valid speedup evidence. Historical Nomic throughput claims have therefore been removed rather than relabeled as current results.

The [repeated M2 Pro measurements](docs/benchmarks/apple-silicon-audit/RESULTS.md) show MiniLM 64-text batch medians of **85.6–85.8 ms** for corrected FP32/512 versus **61.6–61.9 ms** for the opt-in FP16/4096 profile. Active Metal allocations fell from **86.7 to 43.3 MiB**, while the larger batch raised peak active allocation to **379.7 MiB**. These are configuration-specific synthetic measurements, not a universal code-speedup claim.

Larger bulk targets trade latency and memory against throughput; they are opt-in, not a universal fast mode. A short small-encoder run does not establish 4B performance, representative retrieval quality, every M-series Mac or long-run leak freedom. See the [implementation plan](docs/plans/apple-silicon-audit-implementation.md) and [baseline limitations](docs/reviews/apple-silicon-audit-baseline.md).

### Compatibility and index safety

- The space fingerprint now includes context policy and validates actual inference identity. Old MLX descriptors, corrected BERT numerics, changed dtype/pooling/revision or chunk strategy require an **explicit rebuilt index**. Matching dimensions do not establish compatibility.
- Back up an existing index and validate a separate rebuilt index before switching. Do not silently relabel old vectors, delete a live index, or enable fallback to make a mismatch disappear.
- `--quantization` verifies checkpoint storage (`auto`, `none`, `4bit`, `8bit`); floating precision is selected with `--dtype`. Old `bf16`/`fp16`/`q4_0` quantization labels are not accepted as equivalent storage identities.
- Nomic and Gemma are not currently qualified by these adapters and are rejected explicitly. Model-family support is narrower than historical README examples.
- Creating a daemon with `scripts/qmd-mlx-daemon.sh start` is an explicit installation action, not part of testing this fork. The launcher validates local weights and persists selected settings; it does not download a model for you.

---

## Quick Start (GGUF / Default Mode)

```sh
# Install globally (Node or Bun)
npm install -g @tobilu/qmd
# or
bun install -g @tobilu/qmd

# Or run directly
npx @tobilu/qmd ...
bunx @tobilu/qmd ...

# Create collections for your notes, docs, and meeting transcripts
qmd collection add ~/notes --name notes
qmd collection add ~/Documents/meetings --name meetings
qmd collection add ~/work/docs --name docs

# Add context to help with search results, each piece of context will be returned when matching sub documents are returned. This works as a tree. This is the key feature of QMD as it allows LLMs to make much better contextual choices when selecting documents. Don't sleep on it!
qmd context add qmd://notes "Personal notes and ideas"
qmd context add qmd://meetings "Meeting transcripts and notes"
qmd context add qmd://docs "Work documentation"

# Generate embeddings for semantic search
qmd embed

# Search across everything
qmd search "project timeline"           # Fast keyword search
qmd vsearch "how to deploy"             # Semantic search
qmd query "quarterly planning process"  # Hybrid + reranking (best quality)

# Get a specific document
qmd get "meetings/2024-01-15.md"

# Get a document by docid (shown in search results)
qmd get "#abc123"

# Get multiple documents by glob pattern
qmd multi-get "journals/2025-05*.md"

# Search within a specific collection
qmd search "API" -c notes

# Export all matches for an agent
qmd search "API" --all --files --min-score 0.3
```

### Using with AI Agents

QMD's `--json` and `--files` output formats are designed for agentic workflows:

```sh
# Get structured results for an LLM
qmd search "authentication" --json -n 10

# List all relevant files above a threshold
qmd query "error handling" --all --files --min-score 0.4

# Retrieve full document content
qmd get "docs/api-reference.md" --full
```

### MCP Server

Although the tool works perfectly fine when you just tell your agent to use it on the command line, it also exposes an MCP (Model Context Protocol) server for tighter integration.

**Tools exposed:**
- `query` — Search with typed sub-queries (`lex`/`vec`/`hyde`), combined via RRF + reranking
- `get` — Retrieve a document by path or docid (with fuzzy matching suggestions)
- `multi_get` — Batch retrieve by glob pattern, comma-separated list, or docids
- `status` — Index health and collection info

**Claude Desktop configuration** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["mcp"]
    }
  }
}
```

**Claude Code** — Install the plugin (recommended):

```bash
claude plugin marketplace add tobi/qmd
claude plugin install qmd@qmd
```

Or configure MCP manually in `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["mcp"]
    }
  }
}
```

#### HTTP Transport

By default, QMD's MCP server uses stdio (launched as a subprocess by each client). For a shared, long-lived server that avoids repeated model loading, use the HTTP transport:

```sh
# Foreground (Ctrl-C to stop)
qmd mcp --http                    # localhost:8181
qmd mcp --http --port 8080        # custom port

# Background daemon
qmd mcp --http --daemon           # start, writes PID to ~/.cache/qmd/mcp.pid
qmd mcp stop                      # stop via PID file
qmd status                        # shows "MCP: running (PID ...)" when active
```

The HTTP server exposes two endpoints:
- `POST /mcp` — MCP Streamable HTTP (JSON responses, stateless)
- `GET /health` — liveness check with uptime

LLM models stay loaded in VRAM across requests. Embedding/reranking contexts are disposed after 5 min idle and transparently recreated on the next request (~1s penalty, models remain loaded).

Point any MCP client at `http://localhost:8181/mcp` to connect.

### SDK / Library Usage

Use QMD as a library in your own Node.js or Bun applications.

#### Installation

```sh
npm install @tobilu/qmd
```

#### Quick Start

```typescript
import { createStore } from '@tobilu/qmd'

const store = await createStore({
  dbPath: './my-index.sqlite',
  config: {
    collections: {
      docs: { path: '/path/to/docs', pattern: '**/*.md' },
    },
  },
})

const results = await store.search({ query: "authentication flow" })
console.log(results.map(r => `${r.title} (${Math.round(r.score * 100)}%)`))

await store.close()
```

### SDK: MLX Embedding Backend

Configure MLX embeddings programmatically:

```typescript
import { createStore } from '@tobilu/qmd'

const store = await createStore({
  dbPath: './my-index.sqlite',
  config: {
    collections: {
      docs: { path: '/path/to/docs', pattern: '**/*.md' },
    },
    models: {
      embedBackend: 'mlx',
      mlxUrl: 'http://127.0.0.1:8787',
      mlxConcurrency: 2,
      mlxFallback: false,    // never mix another backend's vectors into this index
    },
  },
})
```

Or use the MLX client directly for standalone embedding:

```typescript
import { MlxEmbedClient } from '@tobilu/qmd'

const client = new MlxEmbedClient({
  url: 'http://127.0.0.1:8787',
  timeoutMs: 30_000,
  concurrency: 2,
})
try {
  const descriptor = await client.getDescriptor()
  const result = await client.embed("what is the meaning of life?")
  // result is { embedding: number[], model: string } or null on failure.
  // The production wrapper rejects missing/mismatched serving identity.
  const batch = await client.embedBatchConcurrent(["first query", "second query"])
  console.log(descriptor, result, batch, await client.stats())
} finally {
  client.dispose()
}
```

#### Store Creation

`createStore()` accepts three modes:

```typescript
import { createStore } from '@tobilu/qmd'

// 1. Inline config — no files needed besides the DB
const store = await createStore({
  dbPath: './index.sqlite',
  config: {
    collections: {
      docs: { path: '/path/to/docs', pattern: '**/*.md' },
      notes: { path: '/path/to/notes' },
    },
  },
})

// 2. YAML config file — collections defined in a file
const store2 = await createStore({
  dbPath: './index.sqlite',
  configPath: './qmd.yml',
})

// 3. DB-only — reopen a previously configured store
const store3 = await createStore({ dbPath: './index.sqlite' })
```

#### Search

The unified `search()` method handles both simple queries and pre-expanded structured queries:

```typescript
// Simple query — auto-expanded via LLM, then BM25 + vector + reranking
const results = await store.search({ query: "authentication flow" })

// With options
const results2 = await store.search({
  query: "rate limiting",
  intent: "API throttling and abuse prevention",
  collection: "docs",
  limit: 5,
  minScore: 0.3,
  explain: true,
})

// Pre-expanded queries — skip auto-expansion, control each sub-query
const results3 = await store.search({
  queries: [
    { type: 'lex', query: '"connection pool" timeout -redis' },
    { type: 'vec', query: 'why do database connections time out under load' },
  ],
  collections: ["docs", "notes"],
})

// Skip reranking for faster results
const fast = await store.search({ query: "auth", rerank: false })
```

For direct backend access:

```typescript
// BM25 keyword search (fast, no LLM)
const lexResults = await store.searchLex("auth middleware", { limit: 10 })

// Vector similarity search (embedding model, no reranking)
const vecResults = await store.searchVector("how users log in", { limit: 10 })

// Manual query expansion for full control
const expanded = await store.expandQuery("auth flow", { intent: "user login" })
const results4 = await store.search({ queries: expanded })
```

#### Retrieval

```typescript
// Get a document by path or docid
const doc = await store.get("docs/readme.md")
const byId = await store.get("#abc123")

if (!("error" in doc)) {
  console.log(doc.title, doc.displayPath, doc.context)
}

// Get document body with line range
const body = await store.getDocumentBody("docs/readme.md", {
  fromLine: 50,
  maxLines: 100,
})

// Batch retrieve by glob or comma-separated list
const { docs, errors } = await store.multiGet("docs/**/*.md", {
  maxBytes: 20480,
})
```

#### Collections

```typescript
// Add a collection
await store.addCollection("myapp", {
  path: "/src/myapp",
  pattern: "**/*.ts",
  ignore: ["node_modules/**", "*.test.ts"],
})

// List collections with document stats
const collections = await store.listCollections()
// => [{ name, pwd, glob_pattern, doc_count, active_count, last_modified, includeByDefault }]

// Get names of collections included in queries by default
const defaults = await store.getDefaultCollectionNames()

// Remove / rename
await store.removeCollection("myapp")
await store.renameCollection("old-name", "new-name")
```

#### Context

Context adds descriptive metadata that improves search relevance and is returned alongside results:

```typescript
// Add context for a path within a collection
await store.addContext("docs", "/api", "REST API reference documentation")

// Set global context (applies to all collections)
await store.setGlobalContext("Internal engineering documentation")

// List all contexts
const contexts = await store.listContexts()
// => [{ collection, path, context }]

// Remove context
await store.removeContext("docs", "/api")
await store.setGlobalContext(undefined)  // clear global
```

#### Indexing

```typescript
// Re-index collections by scanning the filesystem
const result = await store.update({
  collections: ["docs"],  // optional — defaults to all
  onProgress: ({ collection, file, current, total }) => {
    console.log(`[${collection}] ${current}/${total} ${file}`)
  },
})
// => { collections, indexed, updated, unchanged, removed, needsEmbedding }

// Generate vector embeddings
const embedResult = await store.embed({
  force: false,           // true to re-embed everything
  chunkStrategy: "auto",  // "regex" (default) or "auto" (AST for code files)
  onProgress: ({ current, total, collection }) => {
    console.log(`Embedding ${current}/${total}`)
  },
})
```

#### Types

Key types exported for SDK consumers:

```typescript
import type {
  QMDStore,            // The store interface
  SearchOptions,       // Options for search()
  LexSearchOptions,    // Options for searchLex()
  VectorSearchOptions, // Options for searchVector()
  HybridQueryResult,   // Search result with score, snippet, context
  SearchResult,        // Result from searchLex/searchVector
  ExpandedQuery,       // Typed sub-query { type: 'lex'|'vec'|'hyde', query }
  DocumentResult,      // Document metadata + body
  DocumentNotFound,    // Error with similarFiles suggestions
  MultiGetResult,      // Batch retrieval result
  UpdateProgress,      // Progress callback info for update()
  UpdateResult,        // Aggregated update result
  EmbedProgress,       // Progress callback info for embed()
  EmbedResult,         // Embedding result
  StoreOptions,        // createStore() options
  CollectionConfig,    // Inline config shape
  IndexStatus,         // From getStatus()
  IndexHealthInfo,     // From getIndexHealth()
} from '@tobilu/qmd'
```

Utility exports:

```typescript
import {
  extractSnippet,              // Extract a relevant snippet from text
  addLineNumbers,              // Add line numbers to text
  DEFAULT_MULTI_GET_MAX_BYTES, // Default max file size for multiGet (10KB)
  Maintenance,                 // Database maintenance operations
} from '@tobilu/qmd'
```

#### Lifecycle

```typescript
// Close the store — disposes LLM models and DB connection
await store.close()
```

The SDK requires explicit `dbPath` — no defaults are assumed. This makes it safe to embed in any application without side effects.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QMD Hybrid Search Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   User Query    │
                              └────────┬────────┘
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
               ┌────────────────┐            ┌────────────────┐
               │ Query Expansion│            │  Original Query│
               │  (fine-tuned)  │            │   (×2 weight)  │
               └───────┬────────┘            └───────┬────────┘
                       │                             │
                       │ 2 alternative queries       │
                       └──────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
     │ Original Query  │     │ Expanded Query 1│     │ Expanded Query 2│
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                       │                       │
      ┌───────┴───────┐       ┌───────┴───────┐       ┌───────┴───────┐
      ▼               ▼       ▼               ▼       ▼               ▼
  ┌───────┐       ┌───────┐ ┌───────┐     ┌───────┐ ┌───────┐     ┌───────┐
  │ BM25  │       │Vector │ │ BM25  │     │Vector │ │ BM25  │     │Vector │
  │(FTS5) │       │Search │ │(FTS5) │     │Search │ │(FTS5) │     │Search │
  └───┬───┘       └───┬───┘ └───┬───┘     └───┬───┘ └───┬───┘     └───┬───┘
      │               │         │             │         │             │
      └───────┬───────┘         └──────┬──────┘         └──────┬──────┘
              │                        │                       │
              └────────────────────────┼───────────────────────┘
                                       │
                                       ▼
                          ┌───────────────────────┐
                          │   RRF Fusion + Bonus  │
                          │  Original query: ×2   │
                          │  Top-rank bonus: +0.05│
                          │     Top 30 Kept       │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │    LLM Re-ranking     │
                          │  (qwen3-reranker)     │
                          │  Yes/No + logprobs    │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  Position-Aware Blend │
                          │  Top 1-3:  75% RRF    │
                          │  Top 4-10: 60% RRF    │
                          │  Top 11+:  40% RRF    │
                          └───────────────────────┘

                        ┌──────────────────────────┐
                        │   Embedding Backends     │
                        ├────────────┬─────────────┤
                        │  MLX       │  GGUF       │
                        │  (GPU)     │  (CPU/Metal)│
                        │  1 owner   │  node-llama │
                        │  binary    │  native     │
                        │  dtype cfg │  quant cfg  │
                        └────────────┴─────────────┘
```

## Score Normalization & Fusion

### Search Backends

| Backend | Raw Score | Conversion | Range |
|---------|-----------|------------|-------|
| **FTS (BM25)** | SQLite FTS5 BM25 | `Math.abs(score)` | 0 to ~25+ |
| **Vector** | Cosine distance | `1 / (1 + distance)` | 0.0 to 1.0 |
| **Reranker** | LLM 0-10 rating | `score / 10` | 0.0 to 1.0 |

### Fusion Strategy

The `query` command uses **Reciprocal Rank Fusion (RRF)** with position-aware blending:

1. **Query Expansion**: Original query (×2 for weighting) + 1 LLM variation
2. **Parallel Retrieval**: Each query searches both FTS and vector indexes
3. **RRF Fusion**: Combine all result lists using `score = Σ(1/(k+rank+1))` where k=60
4. **Top-Rank Bonus**: Documents ranking #1 in any list get +0.05, #2-3 get +0.02
5. **Top-K Selection**: Take top 30 candidates for reranking
6. **Re-ranking**: LLM scores each document (yes/no with logprobs confidence)
7. **Position-Aware Blending**:
   - RRF rank 1-3: 75% retrieval, 25% reranker (preserves exact matches)
   - RRF rank 4-10: 60% retrieval, 40% reranker
   - RRF rank 11+: 40% retrieval, 60% reranker (trust reranker more)

**Why this approach**: Pure RRF can dilute exact matches when expanded queries don't match. The top-rank bonus preserves documents that score #1 for the original query. Position-aware blending prevents the reranker from destroying high-confidence retrieval results.

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 0.8 - 1.0 | Highly relevant |
| 0.5 - 0.8 | Moderately relevant |
| 0.2 - 0.5 | Somewhat relevant |
| 0.0 - 0.2 | Low relevance |

## Requirements

### System Requirements

- **Node.js** >= 22
- **Bun** >= 1.0.0
- **macOS**: Homebrew SQLite (for extension support)
  ```sh
  brew install sqlite
  ```

### MLX Mode (Apple Silicon only)

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Python** >= 3.10
- **MLX** and **mlx-lm** (install via `scripts/mlx_server_requirements.txt`)

### GGUF Models (via node-llama-cpp)

QMD uses three local GGUF models (auto-downloaded on first use):

| Model | Purpose | Size |
|-------|---------|------|
| `embeddinggemma-300M-Q8_0` | Vector embeddings (default) | ~300MB |
| `qwen3-reranker-0.6b-q8_0` | Re-ranking | ~640MB |
| `qmd-query-expansion-1.7B-q4_k_m` | Query expansion (fine-tuned) | ~1.1GB |

Models are downloaded from HuggingFace and cached in `~/.cache/qmd/models/`.

### Custom Embedding Model

Override the default embedding model via the `QMD_EMBED_MODEL` environment variable.
This is useful for multilingual corpora (e.g. Chinese, Japanese, Korean) where
`embeddinggemma-300M` has limited coverage.

```sh
# Use Qwen3-Embedding-0.6B for better multilingual (CJK) support
export QMD_EMBED_MODEL="hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"

# After changing the model, re-embed all collections:
qmd embed -f
```

Supported model families:
- **embeddinggemma** (default) — English-optimized, small footprint
- **Qwen3-Embedding** — Multilingual (119 languages including CJK), MTEB top-ranked

> **Note:** When switching embedding models, you must re-index with `qmd embed -f`
> since vectors are not cross-compatible between models. The prompt format is
> automatically adjusted for each model family.

## Installation

```sh
npm install -g @tobilu/qmd
# or
bun install -g @tobilu/qmd
```

### Development

```sh
git clone https://github.com/shersingh7/qmd-mlx-search
cd qmd-mlx-search
npm install
npm link
```

## Usage

### Collection Management

```sh
# Create a collection from current directory
qmd collection add . --name myproject

# Create a collection with explicit path and custom glob mask
qmd collection add ~/Documents/notes --name notes --mask "**/*.md"

# List all collections
qmd collection list

# Remove a collection
qmd collection remove myproject

# Rename a collection
qmd collection rename myproject my-project

# List files in a collection
qmd ls notes
qmd ls notes/subfolder
```

### Generate Vector Embeddings

```sh
# Embed all indexed documents (900 tokens/chunk, 15% overlap)
qmd embed

# Force re-embed everything
qmd embed -f

# Enable AST-aware chunking for code files (TS, JS, Python, Go, Rust)
qmd embed --chunk-strategy auto

# Also works with query for consistent chunk selection
qmd query "auth flow" --chunk-strategy auto
```

**AST-aware chunking** (`--chunk-strategy auto`) uses tree-sitter to chunk code
files at function, class, and import boundaries instead of arbitrary text
positions. This produces higher-quality chunks and better search results for
codebases. Markdown and other file types always use regex-based chunking
regardless of strategy.

The default is `regex` (existing behavior). Use `--chunk-strategy auto` to
opt in. Run `qmd status` to verify which grammars are available.

> **Note:** Tree-sitter grammars are optional dependencies. If they are not
> installed, `--chunk-strategy auto` falls back to regex-only chunking
> automatically. Tested on both Node.js and Bun.

### Context Management

Context adds descriptive metadata to collections and paths, helping search understand your content.

```sh
# Add context to a collection (using qmd:// virtual paths)
qmd context add qmd://notes "Personal notes and ideas"
qmd context add qmd://docs/api "API documentation"

# Add context from within a collection directory
cd ~/notes && qmd context add "Personal notes and ideas"
cd ~/notes/work && qmd context add "Work-related notes"

# Add global context (applies to all collections)
qmd context add / "Knowledge base for my projects"

# List all contexts
qmd context list

# Remove context
qmd context rm qmd://notes/old
```

### Search Commands

```
┌──────────────────────────────────────────────────────────────────┐
│                        Search Modes                              │
├──────────┬───────────────────────────────────────────────────────┤
│ search   │ BM25 full-text search only                           │
│ vsearch  │ Vector semantic search only                          │
│ query    │ Hybrid: FTS + Vector + Query Expansion + Re-ranking  │
└──────────┴───────────────────────────────────────────────────────┘
```

```sh
# Full-text search (fast, keyword-based)
qmd search "authentication flow"

# Vector search (semantic similarity)
qmd vsearch "how to login"

# Hybrid search with re-ranking (best quality)
qmd query "user authentication"
```

### Options

```sh
# Search options
-n <num>           # Number of results (default: 5, or 20 for --files/--json)
-c, --collection   # Restrict search to a specific collection
--all              # Return all matches (use with --min-score to filter)
--min-score <num>  # Minimum score threshold (default: 0)
--full             # Show full document content
--line-numbers     # Add line numbers to output
--explain          # Include retrieval score traces (query, JSON/CLI output)
--index <name>     # Use named index

# Output formats (for search and multi-get)
--files            # Output: docid,score,filepath,context
--json             # JSON output with snippets
--csv              # CSV output
--md               # Markdown output
--xml              # XML output

# Get options
qmd get <file>[:line]  # Get document, optionally starting at line
-l <num>               # Maximum lines to return
--from <num>           # Start from line number

# Multi-get options
-l <num>           # Maximum lines per file
--max-bytes <num>  # Skip files larger than N bytes (default: 10KB)
```

### Output Format

Default output is colorized CLI format (respects `NO_COLOR` env).

When stdout is a TTY, result paths are emitted as clickable terminal hyperlinks (OSC 8). Clicking a path opens the file in your editor using an editor URI template.

When stdout is not a TTY (for example piped to another command or redirected to a file), QMD emits plain text paths with no escape sequences.

TTY example:

```
docs/guide.md:42 #a1b2c3
Title: Software Craftsmanship
Context: Work documentation
Score: 93%

This section covers the **craftsmanship** of building
quality software with attention to detail.
See also: engineering principles


notes/meeting.md:15 #d4e5f6
Title: Q4 Planning
Context: Personal notes and ideas
Score: 67%

Discussion about code quality and craftsmanship
in the development process.
```

Configure the editor link target with `QMD_EDITOR_URI` (or `editor_uri` in config):

```sh
# VS Code (default)
export QMD_EDITOR_URI="vscode://file/{path}:{line}:{col}"

# Cursor
export QMD_EDITOR_URI="cursor://file/{path}:{line}:{col}"

# Zed
export QMD_EDITOR_URI="zed://file/{path}:{line}:{col}"

# Sublime Text
export QMD_EDITOR_URI="subl://open?url=file://{path}&line={line}"
```

Template placeholders:
- `{path}` absolute filesystem path (URI-encoded)
- `{line}` 1-based line number
- `{col}` or `{column}` 1-based column number

- **Path**: Collection-relative path (e.g., `docs/guide.md`)
- **Docid**: Short hash identifier (e.g., `#a1b2c3`) - use with `qmd get #a1b2c3`
- **Title**: Extracted from document (first heading or filename)
- **Context**: Path context if configured via `qmd context add`
- **Score**: Color-coded (green >70%, yellow >40%, dim otherwise)
- **Snippet**: Context around match with query terms highlighted

### Examples

```sh
# Get 10 results with minimum score 0.3
qmd query -n 10 --min-score 0.3 "API design patterns"

# Output as markdown for LLM context
qmd search --md --full "error handling"

# JSON output for scripting
qmd query --json "quarterly reports"

# Inspect how each result was scored (RRF + rerank blend)
qmd query --json --explain "quarterly reports"

# Use separate index for different knowledge base
qmd --index work search "quarterly reports"
```

### Index Maintenance

```sh
# Show index status and collections with contexts
qmd status

# Re-index all collections
qmd update

# Re-index with git pull first (for remote repos)
qmd update --pull

# Get document by filepath (with fuzzy matching suggestions)
qmd get notes/meeting.md

# Get document by docid (from search results)
qmd get "#abc123"

# Get document starting at line 50, max 100 lines
qmd get notes/meeting.md:50 -l 100

# Get multiple documents by glob pattern
qmd multi-get "journals/2025-05*.md"

# Get multiple documents by comma-separated list (supports docids)
qmd multi-get "doc1.md, doc2.md, #abc123"

# Limit multi-get to files under 20KB
qmd multi-get "docs/*.md" --max-bytes 20480

# Output multi-get as JSON for agent processing
qmd multi-get "docs/*.md" --json

# Clean up cache and orphaned data
qmd cleanup
```

## Data Storage

Index stored in: `~/.cache/qmd/index.sqlite`

### Schema

```sql
collections     -- Indexed directories with name and glob patterns
path_contexts   -- Context descriptions by virtual path (qmd://...)
documents       -- Markdown content with metadata and docid (6-char hash)
documents_fts   -- FTS5 full-text index
content_vectors -- Embedding chunks (hash, seq, pos, 900 tokens each)
vectors_vec     -- sqlite-vec vector index (hash_seq key)
llm_cache       -- Cached LLM responses (query expansion, rerank scores)
```

## How It Works

### Indexing Flow

```
Collection ──► Glob Pattern ──► Markdown Files ──► Parse Title ──► Hash Content
    │                                                   │              │
    │                                                   │              ▼
    │                                                   │         Generate docid
    │                                                   │         (6-char hash)
    │                                                   │              │
    └──────────────────────────────────────────────────►└──► Store in SQLite
                                                                       │
                                                                       ▼
                                                                  FTS5 Index
```

### Embedding Flow

Documents are chunked into ~900-token pieces with 15% overlap using smart boundary detection:

```
Document ──► Smart Chunk (~900 tokens) ──► Format each chunk ──► Backend ──► Store Vectors
                │                           "title | text"        MLX or
                │                                                 node-llama-cpp
                │
                └─► Chunks stored with:
                    - hash: document hash
                    - seq: chunk sequence (0, 1, 2...)
                    - pos: character position in original
```

### Smart Chunking

Instead of cutting at hard token boundaries, QMD uses a scoring algorithm to find natural markdown break points. This keeps semantic units (sections, paragraphs, code blocks) together.

**Break Point Scores:**

| Pattern | Score | Description |
|---------|-------|-------------|
| `# Heading` | 100 | H1 - major section |
| `## Heading` | 90 | H2 - subsection |
| `### Heading` | 80 | H3 |
| `#### Heading` | 70 | H4 |
| `##### Heading` | 60 | H5 |
| `###### Heading` | 50 | H6 |
| ` ``` ` | 80 | Code block boundary |
| `---` / `***` | 60 | Horizontal rule |
| Blank line | 20 | Paragraph boundary |
| `- item` / `1. item` | 5 | List item |
| Line break | 1 | Minimal break |

**Algorithm:**

1. Scan document for all break points with scores
2. When approaching the 900-token target, search a 200-token window before the cutoff
3. Score each break point: `finalScore = baseScore × (1 - (distance/window)² × 0.7)`
4. Cut at the highest-scoring break point

The squared distance decay means a heading 200 tokens back (score ~30) still beats a simple line break at the target (score 1), but a closer heading wins over a distant one.

**Code Fence Protection:** Break points inside code blocks are ignored—code stays together. If a code block exceeds the chunk size, it's kept whole when possible.

**AST-Aware Chunking (Code Files):**

For supported code files, QMD also parses the source with [tree-sitter](https://tree-sitter.github.io/) and adds AST-derived break points that are merged with the regex scores above:

| AST Node | Score | Languages |
|----------|-------|-----------|
| Class / interface / struct / impl / trait | 100 | All |
| Function / method | 90 | All |
| Type alias / enum | 80 | All |
| Import / use declaration | 60 | All |

Supported for `.ts`, `.tsx`, `.js`, `.jsx`, `.py`, `.go`, and `.rs` files. Enable with `--chunk-strategy auto`. Markdown and other file types always use regex chunking.

### Query Flow (Hybrid)

```
Query ──► LLM Expansion ──► [Original, Variant 1, Variant 2]
                │
      ┌─────────┴─────────┐
      ▼                   ▼
   For each query:     FTS (BM25)
      │                   │
      ▼                   ▼
   Vector Search      Ranked List
      │
      ▼
   Ranked List
      │
      └─────────┬─────────┘
                ▼
         RRF Fusion (k=60)
         Original query ×2 weight
         Top-rank bonus: +0.05/#1, +0.02/#2-3
                │
                ▼
         Top 30 candidates
                │
                ▼
         LLM Re-ranking
         (yes/no + logprob confidence)
                │
                ▼
         Position-Aware Blend
         Rank 1-3:  75% RRF / 25% reranker
         Rank 4-10: 60% RRF / 40% reranker
         Rank 11+:  40% RRF / 60% reranker
                │
                ▼
         Final Results
```

## Model Configuration

Models are configured in `src/llm.ts` as HuggingFace URIs:

```typescript
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";
```

### EmbeddingGemma Prompt Format

```
// For queries
"task: search result | query: {query}"

// For documents
"title: {title} | text: {content}"
```

### Qwen3-Reranker

Uses node-llama-cpp's `createRankingContext()` and `rankAndSort()` API for cross-encoder reranking. Returns documents sorted by relevance score (0.0 - 1.0).

### Qwen3 (Query Expansion)

Used for generating query variations via `LlamaChatSession`.

## License

MIT
