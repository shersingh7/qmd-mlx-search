import { expect, test } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { sourceCliCommand } from "./helpers/source-cli.js";

test("development entrypoints avoid the tsx CLI IPC wrapper", async () => {
  const pkg = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
  for (const name of ["qmd", "index", "vector", "search", "vsearch", "rerank"]) {
    expect(pkg.scripts[name]).toMatch(/^node --import tsx src\/cli\/qmd\.ts(?: |$)/);
  }
  expect(pkg.scripts.inspector).toBe("npx @modelcontextprotocol/inspector node --import tsx src/cli/qmd.ts mcp");
});

test("parallel source CLIs work with long managed TMPDIR and unrelated cwd", async () => {
  const root = await mkdtemp(join(tmpdir(), "source-cli-"));
  const temp = join(root, "long-managed-temp-" + "x".repeat(120));
  const home = join(root, "home");
  const config = join(root, "config");
  const children: ChildProcess[] = [];
  const completions: Promise<{ code: number | null; signal: string | null; stdout: string; stderr: string }>[] = [];
  await Promise.all([mkdir(temp), mkdir(home), mkdir(config)]);
  expect(Buffer.byteLength(join(temp, "tsx-501", "99999.pipe"))).toBeGreaterThan(108);
  try {
    for (let index = 0; index < 2; index++) {
      const child = spawn(...sourceCliCommand(["--help"]), {
        cwd: root,
        env: { ...process.env, CI: "true", HOME: home, QMD_CONFIG_DIR: config,
          INDEX_PATH: join(root, `${index}.sqlite`), TMPDIR: temp, TMP: temp, TEMP: temp },
        stdio: ["ignore", "pipe", "pipe"],
      });
      children.push(child);
      completions.push(new Promise((resolve, reject) => {
        let stdout = "";
        let stderr = "";
        const timer = setTimeout(() => child.kill("SIGKILL"), 10000);
        child.stdout.on("data", chunk => { stdout += chunk.toString(); });
        child.stderr.on("data", chunk => { stderr += chunk.toString(); });
        child.once("error", error => { clearTimeout(timer); reject(error); });
        child.once("close", (code, signal) => { clearTimeout(timer); resolve({ code, signal, stdout, stderr }); });
      }));
    }
    for (const result of await Promise.all(completions)) {
      expect(result.code, result.stderr || `terminated by ${result.signal}`).toBe(0);
      expect(result.stdout).toMatch(/qmd|Usage/i);
      expect(result.stderr).not.toMatch(/EADDRINUSE|EEXIST|ENAMETOOLONG/);
    }
  } finally {
    for (const child of children) if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    await Promise.allSettled(completions);
    await rm(root, { recursive: true, force: true });
  }
});
