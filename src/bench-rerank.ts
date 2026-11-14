#!/usr/bin/env bun
/**
 * QMD Reranker Benchmark
 *
 * Measures reranking performance across different configurations.
 * Reports device, parallelism, memory, VRAM, and throughput.
 *
 * Usage:
 *   bun src/bench-rerank.ts              # full benchmark
 *   bun src/bench-rerank.ts --quick      # quick smoke test (10 docs, 1 iteration)
 *   bun src/bench-rerank.ts --docs 100   # custom doc count
 */

import {
  getLlama,
  resolveModelFile,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { inspect } from "node:util";
import { cpus } from "os";

// ============================================================================
// Config
// ============================================================================

const RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const MODEL_CACHE = join(homedir(), ".cache", "qmd", "models");
const CONTEXT_SIZE = 2048;

const args = process.argv.slice(2);
const quick = args.includes("--quick");
const docsIdx = args.indexOf("--docs");
const DOC_COUNT = docsIdx >= 0 ? parseInt(args[docsIdx + 1]!) : (quick ? 10 : 40);
const ITERATIONS = quick ? 1 : 3;
const PARALLEL_CONFIGS = quick ? [1, 4] : [1, 2, 4, 8];

// ============================================================================
// Test data — realistic-ish chunks of varying length
// ============================================================================

const QUERY = "How do AI agents work and what are their limitations?";

function generateDocs(n: number): string[] {
  const templates = [
    "Artificial intelligence agents are software systems that perceive their environment and take actions to achieve goals. They use techniques like reinforcement learning, planning, and natural language processing to operate autonomously.",
    "The transformer architecture, introduced in 2017, revolutionized natural language processing. Self-attention mechanisms allow models to weigh the importance of different parts of input sequences when generating outputs.",
    "Machine learning models require careful evaluation to avoid overfitting. Cross-validation, holdout sets, and metrics like precision, recall, and F1 score help assess generalization performance.",
    "Retrieval-augmented generation combines information retrieval with language models. Documents are embedded into vector spaces, retrieved based on query similarity, and used as context for generation.",
    "Neural network training involves forward propagation, loss computation, and backpropagation. Optimizers like Adam and SGD adjust weights to minimize the loss function over training iterations.",
    "Large language models exhibit emergent capabilities at scale, including few-shot learning, chain-of-thought reasoning, and instruction following. These properties were not explicitly trained for.",
    "Embedding models convert text into dense vector representations that capture semantic meaning. Similar texts produce similar vectors, enabling efficient similarity search and clustering.",
    "Autonomous agents face challenges including hallucination, lack of grounding, limited planning horizons, and difficulty with multi-step reasoning. Safety and alignment remain open research problems.",
    "The attention mechanism computes query-key-value interactions to determine which parts of the input are most relevant. Multi-head attention allows the model to attend to different representation subspaces.",
    "Fine-tuning adapts a pre-trained model to specific tasks using domain-specific data. Techniques like LoRA reduce the number of trainable parameters while maintaining performance.",
  ];
  return Array.from({ length: n }, (_, i) => templates[i % templates.length]!);
}

// ============================================================================
// Helpers
// ============================================================================

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getMemUsage(): { rss: number; heapUsed: number } {
  const m = process.memoryUsage();
  return { rss: m.rss, heapUsed: m.heapUsed };
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

// ============================================================================
// Benchmark runner
// ============================================================================

interface BenchResult {
  parallelism: number;
  contextSize: number;
  flashAttention: boolean | "auto";
  times: number[];       // ms per run
  medianMs: number;
  docsPerSec: number;
  vramPerContext: number; // bytes
  totalVram: number;      // bytes
  peakRss: number;        // bytes
}

// A failed release invalidates further measurements in this process.
class BenchmarkCleanupError extends AggregateError {}

async function disposeOwned(resources: readonly { dispose(): void | Promise<void> }[]): Promise<unknown[]> {
  const errors: unknown[] = [];
  for (const resource of resources) {
    try { await resource.dispose(); } catch (error) { errors.push(error); }
  }
  return errors;
}

async function benchmarkConfig(
  model: LlamaModel,
  llama: Llama,
  docs: string[],
  parallelism: number,
): Promise<BenchResult> {
  // Measure VRAM before
  const vramBefore = llama.gpu ? await llama.getVramState() : null;
  const rssBefore = getMemUsage().rss;

  // Create contexts. On CPU, split threads evenly across contexts.
  const cpuThreads = !llama.gpu ? Math.floor(llama.cpuMathCores / parallelism) : 0;
  const contexts: Awaited<ReturnType<LlamaModel["createRankingContext"]>>[] = [];
  const operationErrors: unknown[] = [];
  try {
    for (let i = 0; i < parallelism; i++) {
      try {
        contexts.push(await model.createRankingContext({
          contextSize: CONTEXT_SIZE,
          // Ranking contexts inherit the model's defaultContextFlashAttention;
          // node-llama-cpp 3.22.3 has no per-ranking-context flash option.
          ...(cpuThreads > 0 ? { threads: cpuThreads } : {}),
        }));
      } catch {
        if (contexts.length === 0) {
          // Retry the first context once with the same supported options.
          contexts.push(await model.createRankingContext({
            contextSize: CONTEXT_SIZE,
            ...(cpuThreads > 0 ? { threads: cpuThreads } : {}),
          }));
        }
        break;
      }
    }
    const actualParallelism = contexts.length;
    const firstContext = contexts[0];
    if (!firstContext) throw new Error("No ranking contexts created");

    // Measure VRAM after context creation
    const vramAfter = llama.gpu ? await llama.getVramState() : null;
    const vramUsed = vramBefore && vramAfter ? vramAfter.used - vramBefore.used : 0;
    const vramPerCtx = actualParallelism > 0 ? vramUsed / actualParallelism : 0;

    // Warm up
    await firstContext.rankAll(QUERY, docs.slice(0, 2));

    // Benchmark iterations
    const times: number[] = [];
    let peakRss = getMemUsage().rss;

    for (let iter = 0; iter < ITERATIONS; iter++) {
      const chunkSize = Math.ceil(docs.length / actualParallelism);

      const t0 = performance.now();
      // Async callbacks also capture synchronous refusals. Do not release any
      // context until every started ranking has settled, even if a sibling fails.
      const outcomes = await Promise.allSettled(
        contexts.map(async (context, i) => {
          const chunk = docs.slice(i * chunkSize, (i + 1) * chunkSize);
          return chunk.length > 0 ? context.rankAll(QUERY, chunk) : [];
        })
      );
      const elapsed = performance.now() - t0;
      times.push(elapsed);
      const failures = outcomes.filter((entry): entry is PromiseRejectedResult => entry.status === "rejected");
      if (failures.length === 1) throw failures[0]!.reason;
      if (failures.length > 1) throw new AggregateError(failures.map(entry => entry.reason), "Ranking operations failed");

      // Verify scores are valid
      const flat = outcomes.flatMap(entry => entry.status === "fulfilled" ? entry.value : []);
      if (flat.some(s => s < 0 || s > 1 || isNaN(s))) {
        throw new Error("Invalid scores detected");
      }

      const currentRss = getMemUsage().rss;
      if (currentRss > peakRss) peakRss = currentRss;
    }

    const med = median(times);
    return {
      parallelism: actualParallelism,
      contextSize: CONTEXT_SIZE,
      flashAttention: model.defaultContextFlashAttention,
      times,
      medianMs: med,
      docsPerSec: (docs.length / med) * 1000,
      vramPerContext: vramPerCtx,
      totalVram: vramUsed,
      peakRss,
    };
  } catch (error) {
    operationErrors.push(error);
    throw error;
  } finally {
    const cleanupErrors = await disposeOwned(contexts);
    if (cleanupErrors.length) throw new BenchmarkCleanupError(
      [...operationErrors, ...cleanupErrors], "Ranking context cleanup failed; benchmark cannot continue",
    );
  }
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("  QMD Reranker Benchmark");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const llama = await getLlama({
    // attempt to build
    build: "autoAttempt",
    logLevel: LlamaLogLevel.error
  });
  const results: BenchResult[] = [];
  let model: LlamaModel | undefined;
  const operationErrors: unknown[] = [];
  try {
    const gpuLabel = llama.gpu === false ? "cpu" : llama.gpu;

    // System info
    const cpuInfo = cpus();
    const cpuModel = cpuInfo[0]?.model || "unknown";
    const cpuCount = cpuInfo.length;

    console.log("System");
    console.log(`  CPU:       ${cpuModel}`);
    console.log(`  Cores:     ${cpuCount} (${llama.cpuMathCores} math)`);
    console.log(`  Device:    ${gpuLabel}`);

    if (llama.gpu) {
      const gpuNames = await llama.getGpuDeviceNames();
      const counts = new Map<string, number>();
      for (const name of gpuNames) counts.set(name, (counts.get(name) || 0) + 1);
      const devStr = Array.from(counts.entries())
        .map(([name, n]) => n > 1 ? `${n}× ${name}` : name).join(", ");
      console.log(`  GPU:       ${devStr}`);
      const vram = await llama.getVramState();
      console.log(`  VRAM:      ${formatBytes(vram.total)} total, ${formatBytes(vram.free)} free`);
    }

    console.log(`  RAM:       ${formatBytes(getMemUsage().rss)} RSS at start`);

    // Load model
    console.log(`\nModel`);
    console.log(`  URI:       ${RERANK_MODEL}`);
    const modelPath = await resolveModelFile(RERANK_MODEL, MODEL_CACHE);
    const vramPreModel = llama.gpu ? await llama.getVramState() : null;
    model = await llama.loadModel({ modelPath });
    const vramPostModel = llama.gpu ? await llama.getVramState() : null;
    const modelVram = vramPreModel && vramPostModel ? vramPostModel.used - vramPreModel.used : 0;
    console.log(`  Params:    ${model.trainContextSize} train ctx`);
    if (modelVram > 0) console.log(`  VRAM:      ${formatBytes(modelVram)} (model weights)`);

    // Generate test docs
    const docs = generateDocs(DOC_COUNT);
    console.log(`\nBenchmark`);
    console.log(`  Documents: ${DOC_COUNT}`);
    console.log(`  Ctx size:  ${CONTEXT_SIZE}`);
    console.log(`  Iterations:${ITERATIONS}`);
    console.log(`  Query:     "${QUERY.slice(0, 50)}..."`);

    // Run benchmarks
    const configErrors: unknown[] = [];
    for (const p of PARALLEL_CONFIGS) {
      if (!llama.gpu && p > 1) {
        // CPU: only test if we have enough cores (at least 4 per context)
        if (llama.cpuMathCores < p * 4) {
          console.log(`\n  [${p} ctx] skipped (need ${p * 4} cores, have ${llama.cpuMathCores})`);
          continue;
        }
      }

      // The public ranking API uses the model default; do not label an ignored
      // per-context option as a flash/no-flash comparison.
      process.stdout.write(`\n  [${p} ctx, flash=${model.defaultContextFlashAttention}] running...`);
      try {
        const r = await benchmarkConfig(model, llama, docs, p);
        results.push(r);
        process.stdout.write(` ${r.medianMs.toFixed(0)}ms (${r.docsPerSec.toFixed(1)} docs/s)\n`);
      } catch (error) {
        // Failed cleanup contaminates later configurations: stop, release the
        // parents, and propagate it instead of publishing a misleading summary.
        if (error instanceof BenchmarkCleanupError) throw error;
        configErrors.push(error);
        process.stdout.write(` failed: ${error instanceof Error ? error.message : String(error)}\n`);
      }
    }
    if (!results.length) throw new AggregateError(configErrors, "No benchmark configurations succeeded");
  } catch (error) {
    operationErrors.push(error);
    throw error;
  } finally {
    // Contexts have settled and had every release attempted by benchmarkConfig.
    // Always attempt both parent releases, in dependency order.
    const cleanupErrors = await disposeOwned([...(model ? [model] : []), llama]);
    if (cleanupErrors.length) throw new BenchmarkCleanupError(
      [...operationErrors, ...cleanupErrors], "Benchmark resource cleanup failed",
    );
  }

  // Publish the summary only after parent cleanup succeeds.

  // Summary table
  console.log("\n═══════════════════════════════════════════════════════════════");
  console.log("  Results");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const header = "  Ctx  Flash  Median    Docs/s   VRAM/ctx   Total VRAM  Peak RSS";
  const sep    = "  ───  ─────  ──────    ──────   ────────   ──────────  ────────";
  console.log(header);
  console.log(sep);

  const baseline = results[0]?.medianMs ?? 1;
  for (const r of results) {
    const speedup = baseline / r.medianMs;
    const speedupStr = r === results[0] ? "      " : `(${speedup.toFixed(1)}×)`;
    console.log(
      `  ${String(r.parallelism).padStart(3)}  ` +
      `${r.flashAttention === "auto" ? "auto " : r.flashAttention ? " yes " : "  no "}  ` +
      `${r.medianMs.toFixed(0).padStart(5)}ms  ` +
      `${r.docsPerSec.toFixed(1).padStart(6)}  ` +
      `${formatBytes(r.vramPerContext).padStart(8)}  ` +
      `${formatBytes(r.totalVram).padStart(10)}  ` +
      `${formatBytes(r.peakRss).padStart(8)}  ` +
      speedupStr
    );
  }

  // Best config
  if (results.length > 0) {
    const best = results.reduce((a, b) => a.docsPerSec > b.docsPerSec ? a : b);
    console.log(`\n  Best: ${best.parallelism} contexts, flash=${best.flashAttention}`);
    console.log(`        ${best.medianMs.toFixed(0)}ms for ${DOC_COUNT} docs (${best.docsPerSec.toFixed(1)} docs/s)`);
    if (best.totalVram > 0) console.log(`        ${formatBytes(best.totalVram)} VRAM`);
  }

  console.log("");
}

// Native flags short-circuit filesystem work. Older Node needs both paths
// resolved, including when --preserve-symlinks-main preserves the module URL.
if (import.meta.main ?? (() => {
  if (process.argv[1] === undefined) return false;
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    // Missing or unresolvable entry paths must not make imports fail or launch.
    return false;
  }
})()) {
  main().catch(error => {
    // Keep nested primary/cleanup failures visible on Node as well as Bun.
    console.error(inspect(error, { depth: null, colors: false }));
    process.exitCode = 1;
  });
}
