import { afterEach, describe, expect, test, vi } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const cli = join(root, "src/cli/qmd.ts");
const loader = join(root, "node_modules/tsx/dist/esm/index.mjs");
const fixtures: string[] = [];
const children: ChildProcess[] = [];
const receipts: string[] = [];
// Track only fixture-spawned children, never discover/stop any existing daemon.
const daemons: number[] = [];

function fixture() {
  const dir = mkdtempSync(join(tmpdir(), "qmd-daemon-entrypoints-"));
  fixtures.push(dir);
  const env: NodeJS.ProcessEnv = {
    PATH: process.env.PATH, HOME: join(dir, "home"), TMPDIR: join(dir, "tmp"),
    XDG_CACHE_HOME: join(dir, "cache"), XDG_CONFIG_HOME: join(dir, "config"),
    QMD_CONFIG_DIR: join(dir, "config/qmd"), INDEX_PATH: join(dir, "index.sqlite"),
    HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", QMD_LLAMA_GPU: "false", CI: "1",
    QMD_MCP_START_TIMEOUT_MS: "10000", QMD_MCP_STOP_TIMEOUT_MS: "3000",
  };
  for (const key of ["HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "QMD_CONFIG_DIR"]) mkdirSync(env[key]!, { recursive: true });
  const cache = join(env.XDG_CACHE_HOME!, "qmd");
  mkdirSync(cache);
  return { dir, env, cache, pidPath: join(cache, "mcp.pid") };
}

function alive(pid: number) {
  try { process.kill(pid, 0); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") return false; throw error; }
}

function run(env: NodeJS.ProcessEnv, args: string[] = [], timeout = 15000, onSpawn?: (child: ChildProcess) => void) {
  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolveResult, reject) => {
    const child = spawn(process.execPath, ["--import", loader, cli, "mcp", ...args], { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] });
    children.push(child);
    onSpawn?.(child);
    let stdout = "", stderr = "";
    child.stdout.on("data", data => { stdout += data; });
    child.stderr.on("data", data => { stderr += data; });
    const timer = setTimeout(() => { child.kill("SIGKILL"); reject(new Error(`CLI fixture timed out: ${stdout}\n${stderr}`)); }, timeout);
    child.once("error", error => { clearTimeout(timer); reject(error); });
    child.once("exit", code => { clearTimeout(timer); resolveResult({ code, stdout, stderr }); });
  });
}

async function freePort() {
  const listener = createServer();
  listener.listen(0, "localhost");
  await once(listener, "listening");
  const port = (listener.address() as import("node:net").AddressInfo).port;
  await new Promise<void>((resolveResult, reject) => listener.close(error => error ? reject(error) : resolveResult()));
  return port;
}

async function sentinel(env: NodeJS.ProcessEnv) {
  const child = spawn(process.execPath, ["--input-type=module", "-e", `
    import { createServer } from 'node:http';
    const server = createServer((req, res) => {
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({status: 'ok', pid: process.pid, instanceToken: 'foreign-listener'}));
    });
    server.listen(0, 'localhost', () => console.log(JSON.stringify({pid: process.pid, port: server.address().port})));
  `], { env, stdio: ["ignore", "pipe", "pipe"] });
  children.push(child);
  let output = "";
  const ready = new Promise<{ pid: number; port: number }>((resolveResult, reject) => {
    child.stdout!.on("data", data => { output += data; if (output.includes("\n")) resolveResult(JSON.parse(output.trim())); });
    child.once("error", reject);
    child.once("exit", () => reject(new Error("Sentinel exited before listening")));
  });
  return { child, ...await ready };
}

function health(port: number) {
  return new Promise<any>((resolveResult, reject) => {
    const req = request({ hostname: "localhost", port, path: "/health", agent: false }, res => {
      let body = "";
      res.on("data", data => { body += data; });
      res.once("end", () => { try { resolveResult(JSON.parse(body)); } catch (error) { reject(error); } });
      res.once("error", reject);
    });
    req.once("error", reject);
    req.setTimeout(1000, () => req.destroy(new Error("Fixture health timeout")));
    req.end();
  });
}

function instrumentChild(f: ReturnType<typeof fixture>, body = "") {
  const file = join(f.dir, "child-probe.mjs");
  const receipt = join(f.dir, "spawned-child.json");
  receipts.push(receipt);
  writeFileSync(file, `
    import { isMainThread } from 'node:worker_threads';
    // esbuild/tsx helpers inherit NODE_OPTIONS and the token, but are not the daemon.
    if (isMainThread && process.argv[1] === ${JSON.stringify(cli)} && process.env.QMD_MCP_INSTANCE_TOKEN) {
      const { writeFileSync } = await import('node:fs');
      writeFileSync(${JSON.stringify(receipt)}, JSON.stringify({pid: process.pid, instanceToken: process.env.QMD_MCP_INSTANCE_TOKEN, argv: process.argv}));
      ${body}
    }
  `);
  f.env.NODE_OPTIONS = `--import=${file}`;
  return receipt;
}

function trackReceipt(receipt: string) {
  expect(existsSync(receipt)).toBe(true);
  const data = JSON.parse(readFileSync(receipt, "utf8"));
  daemons.push(data.pid);
  return data;
}

afterEach(async () => {
  for (const child of children.splice(0)) {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGKILL");
      await once(child, "exit").catch(() => {});
    }
  }
  // Fallback cleanup only reads receipts created by our unique child preload,
  // never a pre-existing daemon record or a process/port discovery query.
  for (const receipt of receipts.splice(0)) {
    if (existsSync(receipt)) daemons.push(JSON.parse(readFileSync(receipt, "utf8")).pid);
  }
  for (const pid of new Set(daemons.splice(0))) {
    if (!Number.isSafeInteger(pid) || pid <= 1) throw new Error("Invalid fixture child PID");
    if (alive(pid)) process.kill(pid, "SIGKILL");
  }
  for (const dir of fixtures.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("quarantined real MCP daemon CLI", () => {
  test("daemon receipt instrumentation ignores token-bearing helper processes and threads", async () => {
    const f = fixture();
    const receipt = instrumentChild(f);
    const probe = join(f.dir, "unrelated-helper.mjs");
    writeFileSync(probe, `import { Worker } from 'node:worker_threads';
      const worker = new Worker('setImmediate(() => process.exit(0))', { eval: true });
      await new Promise((resolve, reject) => { worker.once('exit', resolve); worker.once('error', reject); });`);
    const child = spawn(process.execPath, [probe], {
      cwd: root, env: { ...f.env, QMD_MCP_INSTANCE_TOKEN: randomUUID() }, stdio: "ignore",
    });
    children.push(child);
    const [code] = await once(child, "exit");
    const incorrectlyRecorded = existsSync(receipt);
    // This process was already reaped; never let fallback cleanup signal its old PID.
    rmSync(receipt, { force: true });
    expect(code).toBe(0);
    expect(incorrectlyRecorded).toBe(false);
  });

  test("only reports Started after exact PID/token health; records identity and verifies stop", async () => {
    const f = fixture();
    const receipt = instrumentChild(f);
    const port = await freePort();
    const result = await run(f.env, ["--http", "--daemon", "--port", String(port), "--index", "fixture-index"]);
    const spawned = trackReceipt(receipt);
    expect(result.code, result.stderr).toBe(0);
    expect(result.stdout).toContain(`Started on http://localhost:${port}/mcp`);
    const record = JSON.parse(readFileSync(f.pidPath, "utf8"));
    expect(record).toEqual({ version: 1, pid: spawned.pid, port, instanceToken: spawned.instanceToken });
    expect(spawned.argv.slice(-2)).toEqual(["--index", "fixture-index"]);
    expect(record.instanceToken).toMatch(/^[0-9a-f-]{36}$/);
    const actual = await health(port);
    expect(actual).toMatchObject({ status: "ok", pid: record.pid, instanceToken: record.instanceToken });
    const status = await run(f.env, ["status"]);
    expect(status.code, status.stderr).toBe(0);
    expect(status.stdout).toContain(`running on http://localhost:${port}/mcp (PID ${record.pid})`);
    const second = await run(f.env, ["--http", "--daemon", "--port", String(port)]);
    expect(second.code).not.toBe(0);
    expect(second.stderr).toContain("Already running");
    expect(readFileSync(f.pidPath, "utf8")).toBe(JSON.stringify(record) + "\n");
    const stopped = await run(f.env, ["stop"]);
    expect(stopped.code, stopped.stderr).toBe(0);
    expect(stopped.stdout).toContain(`Stopped QMD MCP server (PID ${record.pid})`);
    expect(existsSync(f.pidPath)).toBe(false);
    expect(alive(record.pid)).toBe(false);
  });

  test.each(["stop", "start"])("refuses legacy raw PID on %s without killing or rewriting an unrelated sentinel", async command => {
    const f = fixture();
    const other = await sentinel(f.env);
    const legacy = String(other.pid) + "\n";
    writeFileSync(f.pidPath, legacy);
    const result = await run(f.env, command === "stop" ? ["stop"] : ["--http", "--daemon", "--port", String(other.port)]);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toMatch(/unverified|legacy/i);
    expect(result.stdout).not.toContain("Started");
    expect(readFileSync(f.pidPath, "utf8")).toBe(legacy);
    expect(alive(other.pid)).toBe(true);
    expect((await health(other.port)).pid).toBe(other.pid);
  });

  test("valid-shaped but foreign PID/token record cannot authorize SIGTERM", async () => {
    const f = fixture();
    const other = await sentinel(f.env);
    const record = JSON.stringify({ version: 1, pid: other.pid, port: other.port, instanceToken: randomUUID() });
    writeFileSync(f.pidPath, record);
    const result = await run(f.env, ["stop"]);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toMatch(/identity|unverified/i);
    expect(readFileSync(f.pidPath, "utf8")).toBe(record);
    expect(alive(other.pid)).toBe(true);
    expect((await health(other.port)).instanceToken).toBe("foreign-listener");
  });

  test("occupied listener's healthy response is not startup success; sentinel survives", async () => {
    const f = fixture();
    const other = await sentinel(f.env);
    const receipt = instrumentChild(f);
    const result = await run(f.env, ["--http", "--daemon", "--port", String(other.port)]);
    const spawned = trackReceipt(receipt);
    expect(result.code).not.toBe(0);
    expect(result.stdout).not.toContain("Started");
    expect(result.stderr).toMatch(/exited|startup|in use/i);
    expect(existsSync(f.pidPath)).toBe(false);
    expect(alive(spawned.pid)).toBe(false);
    expect(alive(other.pid)).toBe(true);
    expect((await health(other.port)).pid).toBe(other.pid);
  });

  test("real child initialization failure has nonzero exit, no record and no false Started", async () => {
    const f = fixture();
    const receipt = instrumentChild(f);
    mkdirSync(f.env.INDEX_PATH!); // SQLite refuses a directory, no models opened.
    const port = await freePort();
    const result = await run(f.env, ["--http", "--daemon", "--port", String(port)]);
    const spawned = trackReceipt(receipt);
    expect(result.code).not.toBe(0);
    expect(result.stdout).not.toContain("Started");
    expect(result.stderr).toMatch(/exited|startup/i);
    expect(existsSync(f.pidPath)).toBe(false);
    expect(alive(spawned.pid)).toBe(false);
    expect(existsSync(join(f.cache, "mcp.pid.lock"))).toBe(false);
  });

  test("bounded startup timeout cleans only its own delayed child, not unrelated sentinel", async () => {
    const f = fixture();
    const other = await sentinel(f.env);
    const receipt = instrumentChild(f, "await new Promise(resolve => setTimeout(resolve, 10000));");
    f.env.QMD_MCP_START_TIMEOUT_MS = "400";
    const before = performance.now();
    const result = await run(f.env, ["--http", "--daemon", "--port", String(await freePort())]);
    const elapsed = performance.now() - before;
    const spawned = trackReceipt(receipt);
    expect(result.code).not.toBe(0);
    expect(result.stdout).not.toContain("Started");
    expect(result.stderr).toMatch(/deadline|timeout/i);
    expect(elapsed).toBeLessThan(8000);
    expect(existsSync(f.pidPath)).toBe(false);
    expect(alive(spawned.pid)).toBe(false);
    expect(alive(other.pid)).toBe(true);
    expect((await health(other.port)).pid).toBe(other.pid);
  });

  test("symlink PID records are refused without changing the target or signaling its PID", async () => {
    const f = fixture();
    const other = await sentinel(f.env);
    const target = join(f.dir, "unrelated-record");
    const record = JSON.stringify({ version: 1, pid: other.pid, port: other.port, instanceToken: randomUUID() });
    writeFileSync(target, record);
    symlinkSync(target, f.pidPath);
    const result = await run(f.env, ["stop"]);
    expect(result.code).not.toBe(0);
    expect(readFileSync(target, "utf8")).toBe(record);
    expect(alive(other.pid)).toBe(true);
  });

  test("serialized starts cannot overwrite an owned live record or leave extra children", async () => {
    const f = fixture();
    const receipt = instrumentChild(f);
    const port = await freePort();
    const results = await Promise.all([run(f.env, ["--http", "--daemon", "--port", String(port)]), run(f.env, ["--http", "--daemon", "--port", String(port)])]);
    const spawned = trackReceipt(receipt);
    expect(results.filter(result => result.code === 0)).toHaveLength(1);
    expect(results.filter(result => result.stdout.includes("Started"))).toHaveLength(1);
    const record = JSON.parse(readFileSync(f.pidPath, "utf8"));
    expect(record.pid).toBe(spawned.pid);
    expect(await health(port)).toMatchObject({ pid: record.pid, instanceToken: record.instanceToken });
    const stopped = await run(f.env, ["stop"]);
    expect(stopped.code, stopped.stderr).toBe(0);
    expect(alive(record.pid)).toBe(false);
  });

  test("record publication is no-clobber and rolls back only its child on a collision", async () => {
    const f = fixture();
    const other = await sentinel(f.env);
    const receipt = instrumentChild(f, "await new Promise(resolve => setTimeout(resolve, 500));");
    const pending = run(f.env, ["--http", "--daemon", "--port", String(await freePort())]);
    await vi.waitFor(() => expect(existsSync(receipt)).toBe(true), { interval: 10, timeout: 5000 });
    const spawned = trackReceipt(receipt);
    const foreign = JSON.stringify({ version: 1, pid: other.pid, port: other.port, instanceToken: randomUUID() });
    writeFileSync(f.pidPath, foreign);
    const result = await pending;
    expect(result.code).not.toBe(0);
    expect(result.stdout).not.toContain("Started");
    expect(readFileSync(f.pidPath, "utf8")).toBe(foreign);
    expect(alive(spawned.pid)).toBe(false);
    expect(alive(other.pid)).toBe(true);
    expect((await health(other.port)).pid).toBe(other.pid);
  });

  test("interrupting startup rolls back the owned child and releases the operation lock", async () => {
    const f = fixture();
    const other = await sentinel(f.env);
    const receipt = instrumentChild(f, "await new Promise(resolve => setTimeout(resolve, 10000));");
    let starter: ChildProcess | undefined;
    const pending = run(f.env, ["--http", "--daemon", "--port", String(await freePort())], 15000, child => { starter = child; });
    await vi.waitFor(() => expect(existsSync(receipt)).toBe(true), { interval: 10, timeout: 5000 });
    const spawned = trackReceipt(receipt);
    starter!.kill("SIGTERM");
    const result = await pending;
    expect(result.code).toBe(143);
    expect(result.stdout).not.toContain("Started");
    expect(existsSync(f.pidPath)).toBe(false);
    expect(existsSync(join(f.cache, "mcp.pid.lock"))).toBe(false);
    expect(alive(spawned.pid)).toBe(false);
    expect(alive(other.pid)).toBe(true);
  });

  test.each(["0", "-1", "65536", "not-a-port"])("refuses invalid daemon port %s before spawning", async port => {
    const f = fixture();
    const receipt = instrumentChild(f);
    const result = await run(f.env, ["--http", "--daemon", "--port", port]);
    expect(result.code).not.toBe(0);
    expect(result.stderr).toContain("--port");
    expect(result.stdout).not.toContain("Started");
    expect(existsSync(receipt)).toBe(false);
    expect(existsSync(f.pidPath)).toBe(false);
  });

  test("normal foreground HTTP and stdio initialize still work without daemon records", async () => {
    const f = fixture();
    const port = await freePort();
    let foreground: ChildProcess | undefined;
    const pending = run(f.env, ["--http", "--port", String(port)], 15000, child => { foreground = child; });
    await vi.waitFor(async () => expect((await health(port)).pid).toBe(foreground!.pid), { interval: 20, timeout: 5000 });
    expect(existsSync(f.pidPath)).toBe(false);
    foreground!.kill("SIGTERM");
    expect((await pending).code).toBe(0);

    const stdio = spawn(process.execPath, ["--import", loader, cli, "mcp"], { cwd: root, env: f.env, stdio: ["pipe", "pipe", "pipe"] });
    children.push(stdio);
    const initialized = new Promise<any>((resolveResult, reject) => {
      let output = "";
      stdio.stdout.on("data", data => {
        output += data;
        if (output.includes("\n")) {
          try { resolveResult(JSON.parse(output.split("\n")[0]!)); } catch (error) { reject(error); }
        }
      });
      stdio.once("error", reject);
      stdio.once("exit", () => reject(new Error("stdio exited before initialize response")));
    });
    stdio.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: {
      protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "quarantined-cli", version: "1" },
    } }) + "\n");
    expect((await initialized).result.serverInfo.name).toBe("qmd");
    expect(existsSync(f.pidPath)).toBe(false);
    stdio.kill("SIGTERM");
    await once(stdio, "exit");
  });
});
