/** Tests for legacy config paths and isolated SDK config scopes. */
import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { join } from "path";
import { homedir, tmpdir } from "os";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import YAML from "yaml";
import {
  addCollection, addContext, configExists, findContextForPath, getCollection,
  getConfigPath, loadConfig, saveConfig, setConfigIndexName, setConfigSource,
  setGlobalContext, withConfigSource, type CollectionConfig, type ConfigSource,
} from "../src/collections.js";

let savedEnv: Record<string, string | undefined>;
let fixture: string;

beforeEach(() => {
  savedEnv = {
    HOME: process.env.HOME,
    QMD_CONFIG_DIR: process.env.QMD_CONFIG_DIR,
    XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME,
  };
  fixture = mkdtempSync(join(tmpdir(), "qmd-collections-config-"));
  process.env.HOME = join(fixture, "home");
  setConfigIndexName("index");
  setConfigSource(undefined);
});

afterEach(() => {
  setConfigIndexName("index");
  setConfigSource(undefined);
  for (const [key, val] of Object.entries(savedEnv)) {
    if (val === undefined) delete process.env[key];
    else process.env[key] = val;
  }
  rmSync(fixture, { recursive: true, force: true });
});

describe("getConfigDir via getConfigPath", () => {
  test("defaults to ~/.config/qmd when no env vars are set", () => {
    delete process.env.QMD_CONFIG_DIR;
    delete process.env.XDG_CONFIG_HOME;
    expect(getConfigPath()).toBe(join(homedir(), ".config", "qmd", "index.yml"));
  });

  test("QMD_CONFIG_DIR takes highest priority", () => {
    process.env.QMD_CONFIG_DIR = "/custom/qmd-config";
    process.env.XDG_CONFIG_HOME = "/xdg/config";
    expect(getConfigPath()).toBe(join("/custom/qmd-config", "index.yml"));
  });

  test("XDG_CONFIG_HOME is used when QMD_CONFIG_DIR is not set", () => {
    delete process.env.QMD_CONFIG_DIR;
    process.env.XDG_CONFIG_HOME = "/xdg/config";
    expect(getConfigPath()).toBe(join("/xdg/config", "qmd", "index.yml"));
  });

  test("XDG_CONFIG_HOME appends qmd subdirectory", () => {
    delete process.env.QMD_CONFIG_DIR;
    process.env.XDG_CONFIG_HOME = "/home/agent/.config";
    expect(getConfigPath()).toBe(join("/home/agent/.config", "qmd", "index.yml"));
  });

  test("QMD_CONFIG_DIR overrides XDG_CONFIG_HOME", () => {
    process.env.QMD_CONFIG_DIR = "/override";
    process.env.XDG_CONFIG_HOME = "/should-not-use";
    expect(getConfigPath()).toBe(join("/override", "index.yml"));
  });

  test("respects custom index name", () => {
    delete process.env.QMD_CONFIG_DIR;
    process.env.XDG_CONFIG_HOME = "/xdg/config";
    setConfigIndexName("myindex");
    expect(getConfigPath()).toBe(join("/xdg/config", "qmd", "myindex.yml"));
  });
});

describe("scoped configuration sources", () => {
  test("nested synchronous scopes restore the parent and legacy source after throw", () => {
    const cli: CollectionConfig = { collections: {} };
    const outer: CollectionConfig = { collections: {} };
    const inner: CollectionConfig = { collections: {} };
    setConfigSource({ config: cli });
    withConfigSource({ config: outer }, () => {
      addCollection("outer", join(fixture, "outer"));
      expect(getConfigPath()).toBe("<inline>");
      expect(configExists()).toBe(true);
      expect(() => withConfigSource({ config: inner }, () => {
        addCollection("inner", join(fixture, "inner"));
        throw new Error("synthetic nested failure");
      })).toThrow("synthetic nested failure");
      expect(loadConfig()).toBe(outer);
    });
    expect(loadConfig()).toBe(cli);
    expect(Object.keys(cli.collections)).toEqual([]);
    expect(Object.keys(outer.collections)).toEqual(["outer"]);
    expect(Object.keys(inner.collections)).toEqual(["inner"]);
  });

  test("overlapping async scopes keep inline/file targets despite CLI source changes", async () => {
    process.env.QMD_CONFIG_DIR = join(fixture, "default-config");
    const inline: CollectionConfig = { collections: {} };
    const cli: CollectionConfig = { collections: {} };
    const path = join(fixture, "scoped", "config.yml");
    let release!: () => void;
    const ready = new Promise<void>(resolve => { release = resolve; });
    const memoryWork = withConfigSource({ config: inline }, async () => {
      await ready;
      addCollection("inline", join(fixture, "inline"));
      addContext("inline", "/", "Inline context");
      await Promise.resolve();
      expect(findContextForPath("inline", "doc.md")).toBe("Inline context");
      expect(getCollection("file")).toBeNull();
      expect(loadConfig()).toBe(inline);
    });
    const fileWork = withConfigSource({ configPath: path }, async () => {
      expect(configExists()).toBe(false);
      await ready;
      addCollection("file", join(fixture, "file"), "**/*.txt", ["drafts/**"]);
      await Promise.resolve();
      setGlobalContext("File global");
      expect(getConfigPath()).toBe(path);
      expect(configExists()).toBe(true);
      expect(getCollection("inline")).toBeNull();
    });
    setConfigSource({ config: cli });
    release();
    await Promise.all([memoryWork, fileWork]);
    expect(Object.keys(inline.collections)).toEqual(["inline"]);
    expect(YAML.parse(readFileSync(path, "utf8"))).toEqual({
      collections: { file: { path: join(fixture, "file"), pattern: "**/*.txt", ignore: ["drafts/**"] } },
      global_context: "File global",
    });
    expect(loadConfig()).toBe(cli);
    expect(cli).toEqual({ collections: {} });
    expect(existsSync(join(fixture, "default-config", "index.yml"))).toBe(false);
  });

  test("an async rejection restores its parent without resetting another source", async () => {
    const cli: CollectionConfig = { collections: {} };
    const parent: CollectionConfig = { collections: {} };
    setConfigSource({ config: cli });
    await withConfigSource({ config: parent }, async () => {
      await expect(withConfigSource({ config: { collections: {} } }, async () => {
        await Promise.resolve();
        throw new Error("synthetic async failure");
      })).rejects.toThrow("synthetic async failure");
      expect(loadConfig()).toBe(parent);
      setGlobalContext("Parent still active");
    });
    expect(parent.global_context).toBe("Parent still active");
    expect(loadConfig()).toBe(cli);
  });

  test("reusing a scoped source retains inline saveConfig replacements", () => {
    const source: ConfigSource = { config: { collections: {} } };
    const replacement: CollectionConfig = { collections: {}, global_context: "Replacement" };
    withConfigSource(source, () => saveConfig(replacement));
    withConfigSource(source, () => {
      expect(loadConfig()).toBe(replacement);
      addCollection("saved", join(fixture, "saved"));
    });
    expect(source.config?.collections.saved?.path).toBe(join(fixture, "saved"));
  });
});
