/** In-process TypeScript loading: no esbuild subprocess outside the worker sandbox. */
import { register, stripTypeScriptTypes } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { isMainThread } from "node:worker_threads";

if (isMainThread) register(import.meta.url);

export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    if (error.code !== "ERR_MODULE_NOT_FOUND" || !context.parentURL
        || !/^(\.\.?\/).*\.js$/.test(specifier)) throw error;
    const candidate = new URL(specifier.replace(/\.js$/, ".ts"), context.parentURL);
    if (candidate.protocol !== "file:" || !existsSync(candidate)) throw error;
    return nextResolve(candidate.href, context);
  }
}

export async function load(url, context, nextLoad) {
  if (!url.startsWith("file:") || !url.endsWith(".ts")) return nextLoad(url, context);
  return { format: "module", shortCircuit: true,
    source: stripTypeScriptTypes(readFileSync(fileURLToPath(url), "utf8"), { mode: "transform", sourceUrl: url }) };
}
