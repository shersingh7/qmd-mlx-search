# AST error-path ownership follow-up

## Why this is being checked

The integrated `getASTBreakPoints` implementation allocates a tree-sitter parser and potentially a tree, then deletes them on successful completion. Its outer error handler returns the documented regex fallback, but exceptions in language selection, parsing, query construction or capture processing may bypass those deletions. This is a concrete source-level ownership concern; a failing runtime regression must establish it before repair.

An additional full native-Bun Vitest invocation exited with SIGTRAP. Its local crash report identifies allocator failure during a Wasm helper thread's teardown. The same real AST product call and the two isolated AST suites subsequently passed under Bun. That crash is separate, not proof of the ownership concern, and must not be claimed fixed by this follow-up.

## Scope

- Modify only `src/ast.ts` and one new focused lifecycle test file.
- Preserve the public API, successful breakpoints, caching, grammar selection and error-to-regex fallback.
- Do not change dependencies, runtime pins, compiler roots, existing assertions, deadlines, server behavior or other accepted changes.
- Implementation stays in an owned derivative; the parent integrates after independent review.

## Implementation steps

1. Bind the exact input and copy only public source into an owned test workspace with isolated caches and HOME.
2. Add deterministic resource-ownership tests using explicit fake tree-sitter objects at the real module boundary. Reproduce parser/tree retention before changing source.
3. Repair structured cleanup so every acquired parser/tree receives exactly one release attempt, including early return and exceptions. One failing release must not prevent the other attempt. Keep failure behavior honest and retain the original operation error where cleanup also fails.
4. Run the new regressions plus the unchanged AST suites on actual Node and Bun. Retain valid real-WASM positive controls and the previous native-Bun failure separately.
5. Deliver a two-path patch, RED/GREEN receipts, source manifests and explicit limitations. Do not self-approve or edit the live checkout.

## Acceptance gates

- Reproduced REDs become GREEN without weakening assertions.
- Successful and unsupported-language behavior remains unchanged.
- Allocation, operation and disposal failure paths are covered; no extra model/network/index work occurs.
- Existing AST suites remain intact and pass on both installed runtimes.
- Independent review, parent patch replay and final checkout tests precede release.
