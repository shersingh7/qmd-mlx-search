import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Match CI: process-lifecycle fixtures have real short deadlines. Extra
    // workers oversubscribe the host without speeding the long CLI test file.
    maxWorkers: 2,
    testTimeout: 30000,
    include: ["test/**/*.test.ts"],
    // Bun's ESM namespace membership trips Vitest's CJS default interop.
    // Vitest's string matcher covers Zod-prefixed packages on that test host;
    // leave native product imports alone.
    server: {
      deps: { inline: process.versions.bun ? ["zod"] : [] },
    },
  },
});
