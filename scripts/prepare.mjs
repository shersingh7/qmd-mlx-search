import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
// npm's Git/local source installs need dist. Published packages already include
// it and must not require a compiler or mutate the consumer's Git hooks.
if (existsSync(join(root, "src/cli/qmd.ts"))) {
  const result = spawnSync(process.execPath, [join(root, "scripts/build.mjs")], { cwd: root, stdio: "inherit" });
  if (result.error) console.error(result.error.message);
  process.exit(result.status ?? 1);
}
for (const artifact of ["dist/index.js", "dist/index.d.ts", "dist/cli/qmd.js"]) {
  if (!existsSync(join(root, artifact))) {
    console.error(`Incomplete package: missing ${artifact}; rebuild from a source checkout.`);
    process.exit(1);
  }
}
