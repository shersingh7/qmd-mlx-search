import { describe, expect, it } from "vitest";
import { changedPipes } from "./helpers/recovery-fd-observation.js";

describe("descriptor observation across asynchronous native release", () => {
  it("detects a new pipe at the same descriptor number as a baseline pipe", () => {
    expect(changedPipes(new Map([[6, "1:90"]]), new Map([[6, "1:91"]]))).toEqual([6]);
  });
  it("excludes unchanged resources and closed descriptors", () => {
    expect(changedPipes(new Map([[6, "1:90"], [7, "1:91"]]), new Map([[6, "1:90"]]))).toEqual([]);
  });
  it("detects newly allocated descriptors without conflating device identities", () => {
    expect(changedPipes(new Map([[6, "1:90"]]), new Map([[6, "2:90"], [8, "1:92"]]))).toEqual([6, 8]);
  });
});
