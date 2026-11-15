import { fstatSync } from "node:fs";

export type PipeSnapshot = Map<number, string>;

// Observation only: this identity must never authorize signalling or closing.
export function pipeSnapshot(): PipeSnapshot {
  const result: PipeSnapshot = new Map();
  for (let fd = 3; fd < 512; fd++) {
    try {
      const stat = fstatSync(fd, { bigint: true });
      if (stat.isFIFO() || stat.isSocket()) result.set(fd, `${stat.dev}:${stat.ino}`);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EBADF") throw error;
    }
  }
  return result;
}

export function changedPipes(before: PipeSnapshot, after: PipeSnapshot): number[] {
  return [...after].filter(([fd, identity]) => before.get(fd) !== identity).map(([fd]) => fd);
}
