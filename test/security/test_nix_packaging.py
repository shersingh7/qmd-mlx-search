"""Nix packaging contracts and an offline installed CLI/SDK/HTTP smoke.

Without arguments this runs stdlib-only source regression tests. The flake's
check-installed app passes --installed and its own exact Node executable; no
consumer package manager, models, MLX runtime, GPU or real index is used.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_SYSTEMS = {"x86_64-linux", "aarch64-darwin"}
REQUIRED_FILES = (
    "package.json", "bin/qmd", "LICENSE", "CHANGELOG.md",
    "dist/index.js", "dist/index.d.ts", "dist/cli/qmd.js",
    "dist/mcp/server.js", "dist/db.js", "dist/ast.js", "dist/mlx.js",
    "dist/embedding/config.js", "dist/indexing/job.js",
    "scripts/mlx_embed_server.py", "scripts/mlx_server_requirements.txt",
    "scripts/mlx_server_constraints.txt", "scripts/qmd_mlx/server.py",
    "scripts/qmd_mlx/adapters/embedding.py", "scripts/qmd_mlx/adapters/tokenization.py",
    "scripts/qmd_gguf/benchmark.py", "scripts/launchd/qmd_mlx_daemon.py",
    "node_modules/better-sqlite3/build/Release/better_sqlite3.node",
)


class NixPackagingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flake = (ROOT / "flake.nix").read_text()
        cls.workflow = (ROOT / ".github/workflows/nix.yml").read_text()

    def test_only_explicit_ci_hosts_are_advertised(self):
        systems = re.search(r"supportedSystems\s*=\s*\[([^]]+)\]", self.flake)
        self.assertIsNotNone(systems, "Do not advertise untested flake-utils default hosts")
        assert systems is not None
        self.assertEqual(set(re.findall(r'"([^\"]+)"', systems[1])), SUPPORTED_SYSTEMS)
        self.assertIn("flake-utils.lib.eachSystem supportedSystems", self.flake)
        self.assertIn("platforms = supportedSystems", self.flake)
        for system in SUPPORTED_SYSTEMS:
            self.assertIn(f"system: {system}", self.workflow)
        self.assertNotIn("platforms = platforms.unix", self.flake)

    def test_dependency_fetch_remains_fixed_output_and_script_free(self):
        for contract in (
            "--frozen-lockfile", "--ignore-scripts", "--backend copyfile",
            'outputHashAlgo = "sha256"', 'outputHashMode = "recursive"',
            "outputHash = nodeModulesHashes.${system}",
            "./package.json ./bun.lock", "node_modules-build", "--production",
        ):
            self.assertIn(contract, self.flake)
        self.assertNotIn("--no-verify", self.flake)
        self.assertNotIn("__noChroot", self.flake)
        self.assertNotRegex(self.workflow, r"(?m)^\s*(?:run:\s*)?nix[^\n]*--impure\b")

    def test_compiler_headers_and_wrapper_share_one_node(self):
        self.assertIn("nodejs = pkgs.nodejs_22", self.flake)
        # nixpkgs node-gyp hardcodes both its interpreter and npm_config_nodedir.
        # Merely putting Node 22 on PATH / passing --nodedir cannot override it.
        self.assertIn("nodeGyp = (pkgs.node-gyp.override { inherit nodejs; }).overrideAttrs", self.flake)
        self.assertIn("--set npm_config_nodedir ${pkgs.lib.getDev nodejs}", self.flake)
        self.assertRegex(self.flake, r"nativeBuildInputs = \[[^]]*\bnodeGyp\b")
        self.assertIn("--nodedir=${pkgs.lib.getDev nodejs}", self.flake)
        self.assertIn("node-gyp rebuild --release", self.flake)
        self.assertIn("${nodejs}/bin/node scripts/build.mjs", self.flake)
        self.assertIn("makeWrapper ${nodejs}/bin/node $out/bin/qmd", self.flake)
        self.assertIn('$out/lib/qmd/dist/cli/qmd.js', self.flake)
        self.assertNotIn("makeWrapper ${pkgs.bun}", self.flake)
        self.assertNotIn("DYLD_LIBRARY_PATH", self.flake)
        self.assertNotIn("LD_LIBRARY_PATH", self.flake)

    def test_install_includes_compiled_sdk_and_recursive_python_assets(self):
        self.assertIn("cp -R dist scripts bin $out/lib/qmd/", self.flake)
        self.assertIn("cp package.json LICENSE CHANGELOG.md $out/lib/qmd/", self.flake)
        self.assertIn('"__pycache__"', self.flake)
        self.assertIn('".venv"', self.flake)
        self.assertNotIn("cp -r src $out", self.flake)
        self.assertIn("doInstallCheck = true", self.flake)

    def test_ci_exercises_installed_package_not_a_source_launcher(self):
        self.assertIn("nix run .#check-installed", self.workflow)
        self.assertIn("./test/security/test_nix_packaging.py", self.flake)
        self.assertIn('--node ${nodejs}/bin/node', self.flake)
        self.assertIn("permissions:", self.workflow)
        self.assertIn("contents: read", self.workflow)
        self.assertIn("timeout-minutes:", self.workflow)

    def test_installed_probe_cannot_inherit_private_index_or_runtime_overrides(self):
        with tempfile.TemporaryDirectory(prefix="qmd-nix-env-test-", dir=os.environ["TMPDIR"]) as directory:
            workspace = Path(directory)
            with patch.dict(os.environ, {
                "HOME": "/fixture-only-do-not-use", "INDEX_PATH": "/fixture-only-private.sqlite",
                "QMD_RUNTIME": "bun", "QMD_EMBED_BACKEND": "mlx", "NODE_OPTIONS": "--invalid-fixture-option",
                "PYTHONPATH": "/fixture-only", "GITHUB_TOKEN": "fixture-not-a-secret",
            }):
                env = isolated_env(workspace)
                self.assertEqual(os.environ["HOME"], "/fixture-only-do-not-use")
            self.assertEqual(env["HOME"], str(workspace / "home"))
            self.assertEqual(env["INDEX_PATH"], str(workspace / "index.sqlite"))
            self.assertEqual(env["QMD_EMBED_BACKEND"], "gguf")
            for name in ("QMD_RUNTIME", "NODE_OPTIONS", "PYTHONPATH", "GITHUB_TOKEN"):
                self.assertNotIn(name, env)

    def test_incomplete_installed_output_fails_before_any_process_launch(self):
        with tempfile.TemporaryDirectory(prefix="qmd-nix-missing-test-", dir=os.environ["TMPDIR"]) as directory:
            with patch("subprocess.run") as run, self.assertRaisesRegex(AssertionError, "omitted package.json"):
                check_installed(Path(directory), Path("/unused-fixture-node"))
            run.assert_not_called()


# Keep this independent of the checkout's JS test runner and source imports.
# All imports below resolve from the installed output. The ephemeral HTTP
# listener and every document/database belong to this one invocation.
INSTALLED_SMOKE = r"""
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const installed = process.env.QMD_NIX_PACKAGE;
const workspace = process.env.QMD_NIX_FIXTURE;
assert(!globalThis.Bun && Number(process.versions.node.split('.')[0]) >= 22);
const load = name => import(pathToFileURL(path.join(installed, 'dist', name)).href);
const { openDatabase, loadSqliteVec, isBun } = await load('db.js');
assert.equal(isBun, false);
const db = openDatabase(':memory:');
let sqliteVersion, vecVersion;
try {
  loadSqliteVec(db);
  sqliteVersion = db.prepare('select sqlite_version() as version').get().version;
  vecVersion = db.prepare('select vec_version() as version').get().version;
  assert(sqliteVersion && vecVersion);
} finally { db.close(); }
const { getASTBreakPoints } = await load('ast.js');
const points = await getASTBreakPoints('def transaction():\n    return 1\n', 'fixture.py');
assert(points.length > 0, 'installed Python tree-sitter WASM grammar did not load');
const docs = path.join(workspace, 'docs');
mkdirSync(docs);
writeFileSync(path.join(docs, 'transactions.md'), '# Transactions\n\nAtomicity ensures all writes commit or none do.\n');
writeFileSync(path.join(docs, 'gardening.md'), '# Gardening\n\nBasil needs sunlight and well-drained soil.\n');
const { createStore } = await load('index.js');
const { startMcpHttpServer } = await load('mcp/server.js');
let store, server;
const checks = [];
try {
  store = await createStore({ dbPath: process.env.INDEX_PATH, config: {
    collections: { fixture: { path: docs, pattern: '**/*.md' } },
  } });
  assert.equal((await store.update()).indexed, 2);
  const rows = await store.search({ queries: [{ type: 'lex', query: 'atomicity' }], rerank: false });
  assert(rows.some(row => row.displayPath.endsWith('transactions.md')));
  assert.equal((await store.getStatus()).totalDocuments, 2);
  checks.push('compiled SDK indexes and retrieves two disposable documents without models');
  server = await startMcpHttpServer(0, { store, quiet: true, requestTimeoutMs: 2000 });
  const base = `http://localhost:${server.port}`;
  const request = (route, options = {}) => fetch(base + route, { ...options, signal: AbortSignal.timeout(5000) });
  const body = JSON.stringify({ searches: [{ type: 'lex', query: 'atomicity' }], rerank: false });
  const headers = { 'Content-Type': 'application/json' };
  const rest = await request('/query', { method: 'POST', headers, body });
  assert.equal(rest.status, 200);
  assert(JSON.stringify(await rest.json()).includes('transactions.md'));
  const denied = await request('/query', { method: 'POST', headers: { ...headers, Origin: 'https://untrusted.example' }, body });
  assert.equal(denied.status, 403); await denied.text();
  checks.push('installed HTTP REST lexical retrieval and foreign-origin rejection');
  const rpcHeaders = { ...headers, Accept: 'application/json, text/event-stream' };
  const init = await request('/mcp', { method: 'POST', headers: rpcHeaders, body: JSON.stringify({
    jsonrpc: '2.0', id: 1, method: 'initialize', params: {
      protocolVersion: '2025-03-26', capabilities: {},
      clientInfo: { name: 'nix-installed-smoke', version: '1' },
    },
  }) });
  assert.equal(init.status, 200);
  assert((await init.json()).result?.serverInfo);
  rpcHeaders['mcp-session-id'] = init.headers.get('mcp-session-id');
  assert(rpcHeaders['mcp-session-id']);
  const ready = await request('/mcp', { method: 'POST', headers: rpcHeaders, body: JSON.stringify({
    jsonrpc: '2.0', method: 'notifications/initialized',
  }) });
  assert.equal(ready.status, 202); await ready.text();
  const get = await request('/mcp', { method: 'POST', headers: rpcHeaders, body: JSON.stringify({
    jsonrpc: '2.0', id: 2, method: 'tools/call', params: {
      name: 'get', arguments: { file: 'qmd://fixture/transactions.md' },
    },
  }) });
  assert.equal(get.status, 200);
  const document = await get.json();
  assert(!document.error && !document.result?.isError && JSON.stringify(document).includes('Atomicity'));
  checks.push('installed MCP HTTP initialize and document readback');
} finally {
  try { await server?.stop(); } finally { await store?.close(); }
}
const pkg = JSON.parse(readFileSync(path.join(installed, 'package.json'), 'utf8'));
writeFileSync(process.env.QMD_NIX_NODE_RECEIPT, JSON.stringify({
  ok: true, version: pkg.version, node: process.version, nodeABI: process.versions.modules,
  sqliteVersion, vecVersion, astBreakPoints: points.length, checks,
}), { flag: 'wx' });
"""


def isolated_env(workspace: Path) -> dict[str, str]:
    # No inherited QMD/MLX/index overrides, Node options, Python paths, home,
    # proxies or tokens. Preserve only executable lookup and locale basics.
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT") if key in os.environ}
    for key, folder in {
        "HOME": "home", "XDG_CONFIG_HOME": "config", "XDG_CACHE_HOME": "cache",
        "XDG_DATA_HOME": "data", "HERMES_HOME": "hermes",
    }.items():
        directory = workspace / folder
        directory.mkdir()
        env[key] = str(directory)
    # Keep all children in the caller's explicit managed scratch/build directory.
    env["TMPDIR"] = os.environ["TMPDIR"]
    env.update({
        "INDEX_PATH": str(workspace / "index.sqlite"),
        "QMD_CONFIG_DIR": str(workspace / "config"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "QMD_LLAMA_GPU": "false", "CI": "true", "NO_COLOR": "1", "QMD_EMBED_BACKEND": "gguf",
    })
    return env


def run_checked(argv: list[str], workspace: Path, env: dict[str, str]) -> str:
    result = subprocess.run(argv, cwd=workspace, env=env, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{argv!r} exited {result.returncode}\n{result.stdout[-8192:]}\n{result.stderr[-8192:]}")
    return result.stdout


def check_installed(output: Path, node: Path, receipt: Path | None = None) -> dict:
    if not os.environ.get("TMPDIR"):
        raise ValueError("An explicit managed TMPDIR is required")
    output = output.resolve(strict=True)
    installed = output / "lib/qmd"
    for name in REQUIRED_FILES:
        if not (installed / name).is_file():
            raise AssertionError(f"Installed package omitted {name}")
    if not (output / "bin/qmd").is_file():
        raise AssertionError("Installed qmd wrapper missing")
    if (installed / "src").exists():
        raise AssertionError("Nix should install compiled JS, not launch source TypeScript")
    helpers = list((installed / "scripts").rglob("*.py"))
    for helper in helpers:
        ast.parse(helper.read_text(), filename=str(helper))
    for path in (installed / "scripts").rglob("*"):
        if path.name in {"__pycache__", ".venv", "node_modules"} or path.suffix in {".pyc", ".pyo"}:
            raise AssertionError(f"Local cache/dependency artifact was packaged: {path}")
    with tempfile.TemporaryDirectory(prefix="qmd-nix-smoke-", dir=os.environ["TMPDIR"]) as directory:
        workspace = Path(directory)
        env = isolated_env(workspace)
        env.update({"QMD_NIX_PACKAGE": str(installed), "QMD_NIX_FIXTURE": str(workspace),
                    "QMD_NIX_NODE_RECEIPT": str(workspace / "node-receipt.json")})
        help_text = run_checked([str(output / "bin/qmd"), "--help"], workspace, env)
        if not re.search(r"Usage:|Commands:", help_text):
            raise AssertionError("Installed wrapper did not display CLI help")
        run_checked([str(output / "bin/qmd"), "collection", "list"], workspace, env)
        run_checked([str(node), "--input-type=module", "--eval", INSTALLED_SMOKE], workspace, env)
        report = json.loads((workspace / "node-receipt.json").read_text())
        # A real wrapper-to-database query catches accidental Bun/source launch
        # and ABI drift which merely importing package metadata cannot catch.
        rows = json.loads(run_checked([str(output / "bin/qmd"), "search", "atomicity", "--json"], workspace, env))
        if "transactions.md" not in json.dumps(rows):
            raise AssertionError("Installed CLI did not retrieve the SDK's committed fixture")
        report["checks"].extend([
            "installed pinned-Node CLI help, collection listing and committed lexical readback",
            "native Node SQLite addon and sqlite-vec extension loaded",
            "installed Python tree-sitter WASM grammar loaded",
            "recursive Python helper inventory and syntax (not MLX runtime execution)",
        ])
        report.update({"output": str(output), "pythonHelpers": len(helpers)})
    if receipt:
        # Receipts must not overwrite a previous result or follow a link.
        with receipt.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--receipt", type=Path)
    args, rest = parser.parse_known_args()
    if args.installed is not None:
        if args.node is None or rest:
            parser.error("--installed requires --node and accepts no test-runner arguments")
        print(json.dumps(check_installed(args.installed, args.node, args.receipt), indent=2))
    else:
        if args.node or args.receipt:
            parser.error("--node/--receipt require --installed")
        unittest.main(argv=[__file__, *rest])
