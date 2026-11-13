import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";
import ts from "typescript";
import { describe, expect, test } from "vitest";

const root = fileURLToPath(new URL("../", import.meta.url));
const manifest = JSON.parse(readFileSync(`${root}/package.json`, "utf8"));
const pnpm = parse(readFileSync(`${root}/pnpm-lock.yaml`, "utf8"));
// Bun's text lock is JSONC, not JSON: use a real parser for comments and trailing commas.
const bunResult = ts.parseConfigFileTextToJson("bun.lock", readFileSync(`${root}/bun.lock`, "utf8"));
if (bunResult.error) throw new Error(ts.flattenDiagnosticMessageText(bunResult.error.messageText, "\n"));
const bun = bunResult.config;

type Inventory = Map<string, Set<string>>;
function inventory(identifiers: string[]): Inventory {
  const result: Inventory = new Map();
  for (const identifier of identifiers) {
    const match = /^(.+)@(\d+\.\d+\.\d+(?:-[^()]+)?)(?:\(.*\))?$/.exec(identifier);
    if (!match) throw new Error(`Unrecognized registry package in lock: ${identifier}`);
    const [, name, version] = match;
    if (!result.has(name!)) result.set(name!, new Set());
    result.get(name!)!.add(version!);
  }
  return result;
}

const locks = {
  pnpm: inventory(Object.keys(pnpm.packages)),
  bun: inventory(Object.values(bun.packages).map((entry: any) => entry[0])),
};

// Floors from the dependency-security remediation advisories. Check every instance,
// including nested versions and optional dependencies on platforms other than this host.
const floors: Record<string, string> = {
  "@hono/node-server": "2.1.3", // 1.x backports do not fix GHSA-rmxm-3fg6-px4f.
  "@vitest/mocker": "4.1.12",
  "body-parser": "2.4.0",
  esbuild: "0.28.1",
  "fast-uri": "3.1.9",
  hono: "4.13.16",
  "ip-address": "10.7.4",
  postcss: "8.5.23",
  "proxy-addr": "2.0.8",
  qs: "6.16.1",
  "simple-git": "4.0.1",
  "source-map-js": "1.2.2",
  tar: "7.5.21",
  tinypool: "2.1.2",
  vite: "7.3.7",
  vitest: "4.1.12",
};
function atLeast(version: string, minimum: string): boolean {
  if (!/^\d+\.\d+\.\d+$/.test(version)) return false;
  const actual = version.split(".").map(Number);
  const expected = minimum.split(".").map(Number);
  for (let index = 0; index < 3; index++) {
    if (actual[index] !== expected[index]) return actual[index]! > expected[index]!;
  }
  return true;
}
function minimumFor(name: string, version: string): string | undefined {
  // Nano ID has separate supported CJS and ESM security branches; never force 3 -> 5.
  if (name === "nanoid") return Number(version.split(".")[0]) <= 3 ? "3.3.12" : "5.1.16";
  return floors[name];
}

describe("registry-resolved dependency locks", () => {
  test.each(Object.entries(locks))("%s has no vulnerable locked instances", (_manager, packages) => {
    const vulnerable: string[] = [];
    for (const [name, versions] of packages) {
      for (const version of versions) {
        const minimum = minimumFor(name, version);
        if (minimum && !atLeast(version, minimum)) vulnerable.push(`${name}@${version} < ${minimum}`);
      }
    }
    expect(vulnerable).toEqual([]);
    // Ensure malformed/empty inventories cannot turn the security scan into a no-op.
    for (const name of ["@modelcontextprotocol/sdk", "hono", "node-llama-cpp", "simple-git", "vitest", "@vitest/mocker", "vite", "esbuild"]) {
      expect(packages.has(name), `${name} missing from lock`).toBe(true);
    }
  });

  test("both locks match every direct runtime, development, optional and peer requirement", () => {
    for (const section of ["dependencies", "devDependencies", "optionalDependencies", "peerDependencies"]) {
      const requested = manifest[section] ?? {};
      expect(bun.workspaces[""][section] ?? {}).toEqual(requested);
      if (section === "peerDependencies") continue; // pnpm represents peers through importer resolution.
      expect(Object.keys(pnpm.importers["."][section] ?? {}).sort()).toEqual(Object.keys(requested).sort());
      for (const [name, specifier] of Object.entries(requested)) {
        expect(pnpm.importers["."][section][name].specifier, name).toBe(specifier);
        for (const packages of Object.values(locks)) {
          expect(packages.get(name)?.has(specifier as string), `${name}@${specifier} missing`).toBe(true);
        }
      }
    }
  });

  test("both managers resolve the same complete package/version inventory", () => {
    const identifiers = (packages: Inventory) => [...packages]
      .flatMap(([name, versions]) => [...versions].map(version => `${name}@${version}`))
      .sort();
    expect(identifiers(locks.bun)).toEqual(identifiers(locks.pnpm));
  });

  test("both managers agree on every security-sensitive package version", () => {
    for (const name of [...Object.keys(floors), "nanoid"]) {
      expect([...(locks.bun.get(name) ?? [])].sort(), name).toEqual([...(locks.pnpm.get(name) ?? [])].sort());
    }
  });
});
