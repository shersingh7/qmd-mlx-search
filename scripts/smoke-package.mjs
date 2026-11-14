import assert from "node:assert/strict";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { packResult } from "./pack-result.mjs";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
let runtime = "node";
let tarball;
const argv = process.argv.slice(2);
while (argv.length) {
  const option = argv.shift();
  if (option === "--runtime") runtime = argv.shift();
  else if (option === "--package") tarball = resolve(argv.shift());
  else throw new Error(`Unknown option: ${option}`);
}
assert(["node", "bun"].includes(runtime), "--runtime must be node or bun");
const workspace = mkdtempSync(join(tmpdir(), "qmd-package-smoke-"));
const env = Object.fromEntries(["PATH", "LANG", "LC_ALL", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"].filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
for (const [key, folder] of Object.entries({ HOME: "home", XDG_CACHE_HOME: "cache", XDG_CONFIG_HOME: "config", HERMES_HOME: "hermes", TMPDIR: "tmp" })) {
  env[key] = join(workspace, folder);
  mkdirSync(env[key], { recursive: true });
}
Object.assign(env, { HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", HF_DATASETS_OFFLINE: "1", CI: "true", QMD_LLAMA_GPU: "false" });
function run(command, args, cwd, capture = false) {
  const result = spawnSync(command, args, { cwd, env, encoding: "utf8", stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit", maxBuffer: 16 * 1024 * 1024, timeout: 240_000 });
  if (result.error || result.status !== 0) {
    if (capture) process.stderr.write((result.stdout ?? "") + (result.stderr ?? ""));
    throw new Error(`${command} ${args.join(" ")} failed (${result.status ?? result.error?.message})`);
  }
  return result.stdout;
}
try {
  if (!tarball) {
    // A fresh source snapshot proves nested src/modules and Python packages are
    // included. Never mutate dist, hooks, HOME or install state of the checkout.
    const snapshot = join(workspace, "source");
    mkdirSync(snapshot);
    for (const name of ["src", "scripts", "bin", "package.json", "tsconfig.json", "tsconfig.build.json", "LICENSE", "CHANGELOG.md", "README.md"]) {
      const source = join(root, name);
      if (existsSync(source)) cpSync(source, join(snapshot, name), { recursive: true, filter: path => !path.split(/[\\/]/).includes("__pycache__") && !/\.py[co]$/.test(path) });
    }
    symlinkSync(join(root, "node_modules"), join(snapshot, "node_modules"), "dir");
    // Exercise the source-install prepare lifecycle itself, not just a direct tsc.
    run("npm", ["run", "prepare"], snapshot);
    const packed = packResult(JSON.parse(run("npm", ["pack", "--json", "--pack-destination", workspace], snapshot, true)));
    assert.equal(packed.name, JSON.parse(readFileSync(join(snapshot, "package.json"), "utf8")).name);
    tarball = join(workspace, packed.filename);
  }
  assert(existsSync(tarball), `Tarball missing: ${tarball}`);
  const inventory = run("tar", ["-tzf", tarball], workspace, true).trim().split("\n");
  for (const path of ["bin/qmd", "dist/index.js", "dist/index.d.ts", "dist/cli/qmd.js", "dist/mcp/server.js", "scripts/mlx_embed_server.py", "scripts/mlx_server_requirements.txt", "scripts/mlx_server_constraints.txt", "scripts/qmd_mlx/server.py", "scripts/qmd_mlx/adapters/embedding.py", "scripts/qmd_mlx/adapters/tokenization.py", "scripts/qmd_gguf/benchmark.py", "scripts/launchd/qmd_mlx_daemon.py"]) {
    assert(inventory.includes(`package/${path}`), `Package omitted ${path}`);
  }
  assert(!inventory.some(path => /(^|\/)(__pycache__|\.venv|node_modules)(\/|$)|\.py[co]$/.test(path)), "Package contains local cache/dependency artifacts");
  const target = join(workspace, "installed");
  mkdirSync(target);
  writeFileSync(join(target, "package.json"), JSON.stringify({ name: "qmd-package-smoke-fixture", private: true, type: "module", allowScripts: { "better-sqlite3": true } }, null, 2));
  // No dependency postinstall is allowed implicitly. Approve/rebuild exactly the
  // Node SQLite addon; node-llama-cpp is not asked to fetch/build any model/runtime.
  if (runtime === "bun") {
    run("bun", ["add", "--ignore-scripts", tarball], target);
  } else {
    run("npm", ["install", "--ignore-scripts", "--no-audit", "--no-fund", tarball], target);
    run("npm", ["rebuild", "better-sqlite3", "--ignore-scripts=false"], target);
  }
  const installed = join(target, "node_modules/@tobilu/qmd");
  const launcher = join(installed, "bin/qmd");
  const help = run(launcher, ["--help"], target, true);
  assert.match(help, /Usage:|Commands:/);
  run(launcher, ["collection", "list"], target);
  run(launcher, ["status"], target);
  const script = `const {openDatabase,loadSqliteVec}=await import(${JSON.stringify(join(installed, "dist/db.js"))}); const db=openDatabase(':memory:'); loadSqliteVec(db); const row=db.prepare('select vec_version() as version').get(); if(!row?.version) throw Error('sqlite-vec unavailable'); console.log('sqlite-vec',row.version); db.close();`;
  run(runtime, [...(runtime === "node" ? ["--input-type=module"] : []), "-e", script], target);
  const stdioReceipt = join(workspace, `native-${runtime}-stdio.json`);
  run(runtime, [join(root, "scripts/smoke-installed-stdio.mjs"), installed, stdioReceipt], target);
  const stdioChecks = JSON.parse(readFileSync(stdioReceipt, "utf8"));
  assert.equal(stdioChecks.ok, true);
  const httpReceipt = join(workspace, `native-${runtime}-http.json`);
  run(runtime, [join(root, "scripts/smoke-installed-http.mjs"), installed, httpReceipt], target);
  const httpChecks = JSON.parse(readFileSync(httpReceipt, "utf8"));
  assert.equal(httpChecks.ok, true);
  const receipt = { ok: true, runtime, tarball, installed, workspace, packagedFiles: inventory.length, httpReceipt, stdioReceipt, httpSupported: httpChecks.httpSupported, checks: ["source prepare and prepack (unless --package supplied)", "recursive package inventory", "isolated local dependency install", "real launcher help/collection/status", "SQLite and sqlite-vec load", ...stdioChecks.checks, ...httpChecks.checks] };
  writeFileSync(join(workspace, "receipt.json"), JSON.stringify(receipt, null, 2) + "\n");
  console.log(JSON.stringify(receipt, null, 2));
} catch (error) {
  console.error(`Package smoke failed; retained fixture: ${workspace}`);
  throw error;
}
