import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

type FailureStage = "init" | "grammar" | "constructor" | "setLanguage" | "parse"
  | "query" | "rootNode" | "captures" | "capture";

type FakeOptions = {
  failAt?: FailureStage;
  error?: unknown;
  nullTree?: boolean;
  treeDeleteError?: Error;
  parserDeleteError?: Error;
};

// Replace only the dependency boundary, then import the actual source module.
// Each returned resource has its own spy, so repeated calls cannot hide a leak
// or a double release behind an aggregate counter.
async function loadWithFakes(options: FakeOptions = {}) {
  const events: string[] = [];
  const grammar = {};
  const rootNode = {};
  const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
  const fail = (stage: FailureStage) => {
    if (options.failAt === stage) {
      throw ("error" in options ? options.error : new Error(`${stage} failed`));
    }
  };
  const makeTree = () => ({
    get rootNode() {
      events.push("rootNode");
      fail("rootNode");
      return rootNode;
    },
    delete: vi.fn(() => {
      events.push("tree.delete");
      if (options.treeDeleteError) throw options.treeDeleteError;
    }),
  });
  const trees: ReturnType<typeof makeTree>[] = [];
  const parsers: FakeParser[] = [];
  const queries: FakeQuery[] = [];
  const queryArguments: { grammar: unknown; source: string }[] = [];

  class FakeParser {
    static init = vi.fn(async () => {
      events.push("init");
      fail("init");
    });

    constructor() {
      events.push("parser");
      fail("constructor");
      parsers.push(this);
    }

    setLanguage = vi.fn((_grammar: unknown) => {
      events.push("setLanguage");
      fail("setLanguage");
    });

    parse = vi.fn((_content: string) => {
      events.push("parse");
      fail("parse");
      if (options.nullTree) return null;
      const tree = makeTree();
      trees.push(tree);
      return tree;
    });

    delete = vi.fn(() => {
      events.push("parser.delete");
      if (options.parserDeleteError) throw options.parserDeleteError;
    });
  }

  class FakeQuery {
    constructor(grammar: unknown, source: string) {
      events.push("query");
      queryArguments.push({ grammar, source });
      fail("query");
      queries.push(this);
    }

    captures = vi.fn((_node: unknown) => {
      events.push("captures");
      fail("captures");
      if (options.failAt === "capture") {
        return [{ name: "func", node: { get startIndex(): number {
          events.push("capture");
          fail("capture");
          return 0;
        } } }];
      }
      return [
        { name: "func", node: { startIndex: 24 } },
        { name: "func", node: { startIndex: 8 } },
        { name: "class", node: { startIndex: 8 } },
        { name: "unknown", node: { startIndex: 18 } },
      ];
    });
  }

  const load = vi.fn(async (_path: string) => {
    events.push("grammar");
    fail("grammar");
    return grammar;
  });
  vi.doMock("web-tree-sitter", () => ({
    Parser: FakeParser,
    Language: { load },
    Query: FakeQuery,
  }));
  const ast = await import("../src/ast.js");
  return { ast, options, events, grammar, rootNode, warning, parsers, trees,
    queries, queryArguments, load, init: FakeParser.init };
}

type Fixture = Awaited<ReturnType<typeof loadWithFakes>>;
const expectedPoints = [
  { pos: 8, score: 100, type: "ast:class" },
  { pos: 18, score: 20, type: "ast:unknown" },
  { pos: 24, score: 90, type: "ast:func" },
];

function expectReleased(fixture: Fixture, parserCount: number, treeCount: number) {
  expect(fixture.parsers).toHaveLength(parserCount);
  expect(fixture.trees).toHaveLength(treeCount);
  for (const parser of fixture.parsers) expect(parser.delete).toHaveBeenCalledTimes(1);
  for (const tree of fixture.trees) expect(tree.delete).toHaveBeenCalledTimes(1);
}

function expectWarning(fixture: Fixture, ...messages: string[]) {
  expect(fixture.warning).toHaveBeenCalledTimes(1);
  const message = fixture.warning.mock.calls[0]![0];
  expect(message).toContain("AST parse failed for fixture.ts, falling back to regex:");
  for (const expected of messages) expect(message).toContain(expected);
}

beforeEach(() => vi.resetModules());
afterEach(() => {
  vi.doUnmock("web-tree-sitter");
  vi.resetModules();
  vi.restoreAllMocks();
});

describe("getASTBreakPoints resource ownership", () => {
  test("unsupported files acquire nothing and do not initialize WASM", async () => {
    const fixture = await loadWithFakes({ failAt: "init" });
    await expect(fixture.ast.getASTBreakPoints("# Plain text", "fixture.md")).resolves.toEqual([]);
    expect(fixture.events).toEqual([]);
    expectReleased(fixture, 0, 0);
    expect(fixture.warning).not.toHaveBeenCalled();
  });

  test.each(["init", "grammar", "constructor"] as const)(
    "%s failure returns fallback without releasing an unacquired resource", async (failAt) => {
      const fixture = await loadWithFakes({ failAt });
      await expect(fixture.ast.getASTBreakPoints("function f() {}", "fixture.ts")).resolves.toEqual([]);
      expectReleased(fixture, 0, 0);
      expect(fixture.events).toEqual(failAt === "init" ? ["init"]
        : failAt === "grammar" ? ["init", "grammar"] : ["init", "grammar", "parser"]);
      expect(fixture.warning).toHaveBeenCalledTimes(1);
      expect(fixture.warning.mock.calls[0]![0]).toContain(`${failAt} failed`);
    },
  );

  test.each(["setLanguage", "parse", "query", "rootNode", "captures", "capture"] as const)(
    "%s failure releases every acquired resource exactly once", async (failAt) => {
      const fixture = await loadWithFakes({ failAt });
      await expect(fixture.ast.getASTBreakPoints("function f() {}", "fixture.ts")).resolves.toEqual([]);
      const hasTree = failAt !== "setLanguage" && failAt !== "parse";
      expectReleased(fixture, 1, hasTree ? 1 : 0);
      expect(fixture.events.slice(hasTree ? -2 : -1)).toEqual(
        hasTree ? ["tree.delete", "parser.delete"] : ["parser.delete"],
      );
      expectWarning(fixture, `${failAt} failed`);
    },
  );

  test("null parse releases only the acquired parser", async () => {
    const fixture = await loadWithFakes({ nullTree: true });
    await expect(fixture.ast.getASTBreakPoints("", "fixture.ts")).resolves.toEqual([]);
    expectReleased(fixture, 1, 0);
    expect(fixture.events).toEqual(["init", "grammar", "parser", "setLanguage", "parse", "parser.delete"]);
    expect(fixture.queryArguments).toEqual([]);
    expect(fixture.warning).not.toHaveBeenCalled();
  });

  test("success preserves sorted highest-score breakpoints and releases tree before parser", async () => {
    const fixture = await loadWithFakes();
    await expect(fixture.ast.getASTBreakPoints("source", "fixture.ts")).resolves.toEqual(expectedPoints);
    expectReleased(fixture, 1, 1);
    expect(fixture.parsers[0]!.setLanguage).toHaveBeenCalledWith(fixture.grammar);
    expect(fixture.parsers[0]!.parse).toHaveBeenCalledWith("source");
    expect(fixture.queries[0]!.captures).toHaveBeenCalledWith(fixture.rootNode);
    expect(fixture.events).toEqual(["init", "grammar", "parser", "setLanguage", "parse",
      "query", "rootNode", "captures", "tree.delete", "parser.delete"]);
    expect(fixture.warning).not.toHaveBeenCalled();
  });

  test("repeated calls cache grammars and per-language queries, not parsers or trees", async () => {
    const fixture = await loadWithFakes();
    for (const extension of ["ts", "js", "tsx", "ts", "js", "tsx"]) {
      await expect(fixture.ast.getASTBreakPoints("source", `fixture.${extension}`)).resolves.toEqual(expectedPoints);
    }
    expectReleased(fixture, 6, 6);
    expect(fixture.init).toHaveBeenCalledTimes(1);
    expect(fixture.load).toHaveBeenCalledTimes(2);
    expect(fixture.load.mock.calls.map(([path]) => path.split("/").pop())).toEqual([
      "tree-sitter-typescript.wasm", "tree-sitter-tsx.wasm",
    ]);
    expect(fixture.queries).toHaveLength(3);
    expect(fixture.queryArguments.map(({ source }) => source.includes("interface_declaration")))
      .toEqual([true, false, true]);
    expect(fixture.warning).not.toHaveBeenCalled();
  });

  test("a failed query is not cached and a later call can succeed", async () => {
    const fixture = await loadWithFakes({ failAt: "query" });
    await expect(fixture.ast.getASTBreakPoints("source", "fixture.ts")).resolves.toEqual([]);
    fixture.options.failAt = undefined;
    await expect(fixture.ast.getASTBreakPoints("source", "fixture.ts")).resolves.toEqual(expectedPoints);
    expectReleased(fixture, 2, 2);
    expect(fixture.queryArguments).toHaveLength(2);
    expect(fixture.queries).toHaveLength(1);
    expect(fixture.load).toHaveBeenCalledTimes(1);
    expectWarning(fixture, "query failed");
  });

  const cleanupCases = [
    { name: "tree", treeDeleteError: new Error("tree cleanup failed"), parserDeleteError: undefined },
    { name: "parser", treeDeleteError: undefined, parserDeleteError: new Error("parser cleanup failed") },
    { name: "both", treeDeleteError: new Error("tree cleanup failed"), parserDeleteError: new Error("parser cleanup failed") },
  ];

  test.each(cleanupCases)("$name cleanup failure still attempts both releases and returns fallback", async (options) => {
    const fixture = await loadWithFakes(options);
    await expect(fixture.ast.getASTBreakPoints("source", "fixture.ts")).resolves.toEqual([]);
    expectReleased(fixture, 1, 1);
    expect(fixture.events.slice(-2)).toEqual(["tree.delete", "parser.delete"]);
    expectWarning(fixture, ...[options.treeDeleteError, options.parserDeleteError]
      .filter((error): error is Error => error !== undefined).map(error => error.message));
  });

  test.each(cleanupCases)("capture failure retains its context with $name cleanup failure", async (options) => {
    const fixture = await loadWithFakes({ ...options, failAt: "capture" });
    await expect(fixture.ast.getASTBreakPoints("source", "fixture.ts")).resolves.toEqual([]);
    expectReleased(fixture, 1, 1);
    expect(fixture.events.slice(-2)).toEqual(["tree.delete", "parser.delete"]);
    expectWarning(fixture, "capture failed", ...[options.treeDeleteError, options.parserDeleteError]
      .filter((error): error is Error => error !== undefined).map(error => error.message));
    expect(fixture.warning.mock.calls[0]![0].indexOf("capture failed"))
      .toBeLessThan(fixture.warning.mock.calls[0]![0].indexOf("cleanup failed"));
  });

  test("setLanguage failure is not replaced by parser cleanup failure", async () => {
    const fixture = await loadWithFakes({ failAt: "setLanguage", parserDeleteError: new Error("parser cleanup failed") });
    await expect(fixture.ast.getASTBreakPoints("source", "fixture.ts")).resolves.toEqual([]);
    expectReleased(fixture, 1, 0);
    expectWarning(fixture, "setLanguage failed", "parser cleanup failed");
  });

  test("null parse with parser cleanup failure does not retry deletion", async () => {
    const fixture = await loadWithFakes({ nullTree: true, parserDeleteError: new Error("parser cleanup failed") });
    await expect(fixture.ast.getASTBreakPoints("", "fixture.ts")).resolves.toEqual([]);
    expectReleased(fixture, 1, 0);
    expectWarning(fixture, "parser cleanup failed");
  });

  test.each([undefined, "non-Error capture failure"])(
    "preserves a non-Error operation failure (%s) when both releases fail", async (error) => {
      const fixture = await loadWithFakes({ failAt: "capture", error,
        treeDeleteError: new Error("tree cleanup failed"), parserDeleteError: new Error("parser cleanup failed") });
      await expect(fixture.ast.getASTBreakPoints("source", "fixture.ts")).resolves.toEqual([]);
      expectReleased(fixture, 1, 1);
      expectWarning(fixture, String(error), "tree cleanup failed", "parser cleanup failed");
    },
  );
});
