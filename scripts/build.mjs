import { chmodSync, existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const compiler = join(root, "node_modules/typescript/bin/tsc");
if (!existsSync(compiler)) {
  console.error("TypeScript build dependency missing; install development dependencies before building from source.");
  process.exit(1);
}
// A failed compiler must not leave a plausible, partially updated package tree.
rmSync(join(root, "dist"), { recursive: true, force: true });
const result = spawnSync(process.execPath, [compiler, "-p", "tsconfig.build.json"], { cwd: root, stdio: "inherit" });
if (result.error || result.status !== 0) {
  rmSync(join(root, "dist"), { recursive: true, force: true });
  if (result.error) console.error(result.error.message);
  process.exit(result.status ?? 1);
}
for (const artifact of ["dist/index.js", "dist/index.d.ts", "dist/cli/qmd.js", "dist/mcp/server.js"]) {
  if (!existsSync(join(root, artifact))) throw new Error(`Build did not produce ${artifact}`);
}
const cli = join(root, "dist/cli/qmd.js");
const body = readFileSync(cli, "utf8");
writeFileSync(cli, body.startsWith("#!") ? body : `#!/usr/bin/env node\n${body}`);
chmodSync(cli, 0o755);
