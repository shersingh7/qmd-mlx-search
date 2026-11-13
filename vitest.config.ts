import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Match CI: process-lifecycle fixtures have real short deadlines. Extra
    // workers oversubscribe the host without speeding the long CLI test file.
    maxWorkers: 2,
    testTimeout: 30000,
    include: ["test/**/*.test.ts"],
  },
});
