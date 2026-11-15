/** Shared source-CLI invocation for subprocess tests. */
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const script = fileURLToPath(new URL("../../src/cli/qmd.ts", import.meta.url));

export function sourceCliCommand(args: string[]): [string, string[]] {
  // The tsx CLI creates a Unix IPC socket under TMPDIR. Long managed paths
  // can truncate/collide on macOS; importing its loader needs no CLI socket
  // or wrapper process. An absolute URL also works from fixture-only cwd.
  return [process.execPath, ["--import", pathToFileURL(require.resolve("tsx")).href, script, ...args]];
}
