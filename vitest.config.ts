import { configDefaults, defineConfig } from "vitest/config";

const recoveryTests = [
  "test/audit-recovery*.test.ts",
  "test/audit-native-bun-recovery.test.ts",
];

export default defineConfig({
  test: {
    // Preserve parallelism for ordinary application/CLI tests. Process-lifecycle
    // tests use real 200ms deadlines and must not compete with those workloads.
    maxWorkers: 2,
    testTimeout: 30000,
    // Bun's ESM namespace membership trips Vitest's CJS default interop.
    // Vitest's string matcher covers Zod-prefixed packages on that test host;
    // leave native product imports alone.
    server: {
      deps: { inline: process.versions.bun ? ["zod"] : [] },
    },
    projects: [
      {
        extends: true,
        test: {
          name: "recovery",
          include: recoveryTests,
          fileParallelism: false,
          // Zero is Vitest 4.1's default serial bucket (scheduled last).
          sequence: { groupOrder: 1 },
        },
      },
      {
        extends: true,
        test: {
          name: "application",
          include: ["test/**/*.test.ts"],
          exclude: [...configDefaults.exclude, ...recoveryTests],
          fileParallelism: true,
          sequence: { groupOrder: 2 },
        },
      },
    ],
  },
});
