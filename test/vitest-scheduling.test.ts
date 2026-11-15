import { describe, expect, it } from "vitest";
import { globSync } from "fast-glob";
import picomatch from "picomatch";
import { fileURLToPath } from "node:url";
import config from "../vitest.config.js";

type Project = {
  extends: boolean;
  test: {
    name: string;
    include: string[];
    exclude?: string[];
    fileParallelism: boolean;
    sequence: { groupOrder: number };
  };
};

const projects = () => (config.test?.projects ?? []) as Project[];
const matches = (project: Project, file: string) =>
  picomatch(project.test.include)(file) && !picomatch(project.test.exclude ?? [])(file);

describe("canonical test scheduling", () => {
  it("assigns every canonical test file to exactly one project", () => {
    const configured = projects();
    expect(configured).toHaveLength(2);
    const root = fileURLToPath(new URL("../", import.meta.url));
    const files = globSync("test/**/*.test.ts", { cwd: root });
    expect(files.length).toBeGreaterThan(0);
    for (const file of files) {
      expect(configured.filter(project => matches(project, file)), file).toHaveLength(1);
    }
  });

  it("isolates real recovery deadlines while preserving other file parallelism", () => {
    const configured = projects();
    const recovery = configured.find(project => project.test.name === "recovery");
    const application = configured.find(project => project.test.name === "application");
    expect(recovery).toBeDefined();
    expect(application).toBeDefined();
    expect(recovery!.extends).toBe(true);
    expect(application!.extends).toBe(true);
    expect(recovery!.test.fileParallelism).toBe(false);
    expect(application!.test.fileParallelism).toBe(true);
    // Vitest 4.1 treats zero as its default serial bucket, which runs last.
    // Use explicit nonzero groups so the installed runner respects recovery-first ordering.
    expect(recovery!.test.sequence.groupOrder).toBeGreaterThan(0);
    expect(recovery!.test.sequence.groupOrder).toBeLessThan(application!.test.sequence.groupOrder);
    expect(matches(recovery!, "test/audit-recovery-transition.test.ts")).toBe(true);
    expect(matches(recovery!, "test/audit-native-bun-recovery.test.ts")).toBe(true);
    expect(matches(application!, "test/cli.test.ts")).toBe(true);
    expect(matches(recovery!, "test/cli.test.ts")).toBe(false);
    expect(config.test?.maxWorkers).toBe(2);
    expect(config.test?.testTimeout).toBe(30_000);
  });
});
