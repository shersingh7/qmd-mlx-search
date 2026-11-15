import { readFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// npm <=11 emits an array; npm 12 emits a package-name keyed object.
export function packResult(value) {
  const entries = Array.isArray(value) ? value : value && typeof value === "object" ? Object.values(value) : [];
  if (entries.length !== 1) throw new Error("Expected exactly one npm pack artifact");
  const result = entries[0];
  if (!result || typeof result.filename !== "string" || result.filename !== basename(result.filename) || !/^[A-Za-z0-9][A-Za-z0-9._-]*\.tgz$/.test(result.filename)) {
    throw new Error("Invalid npm pack artifact filename");
  }
  return result;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error("usage: node scripts/pack-result.mjs /path/to/npm-pack.json");
  console.log(packResult(JSON.parse(readFileSync(process.argv[2], "utf8"))).filename);
}
