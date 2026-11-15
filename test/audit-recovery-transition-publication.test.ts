import { describe, expect, it } from "vitest";
import { spawnSync } from "node:child_process";
import { recoveryTransitionWrapper } from "./helpers/recovery-transition-wrapper.js";

const script = `import ast, json, os, sys, tempfile, types
from pathlib import Path
wrapper = sys.argv[1]
with tempfile.TemporaryDirectory(prefix="qmd-stage-publication-") as root:
    target = Path(root) / 'stage.json'
    wrapper = wrapper.replace('STAGE_PATH', str(target))
    tree = ast.parse(wrapper)
    hold = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'hold')
    code = compile(ast.Module(body=[hold], type_ignores=[]), '<hold>', 'exec')
    pauses = []
    scope = dict(paused=False, os=os, time=types.SimpleNamespace(sleep=pauses.append))
    def interrupted_dump(value, stream):
        stream.write('{')
        stream.flush()
        raise InterruptedError('cancelled during publication')
    scope['json'] = types.SimpleNamespace(dump=interrupted_dump)
    exec(code, scope)
    try:
        scope['hold']('before-ready')
    except InterruptedError:
        pass
    else:
        raise AssertionError('The interrupted writer must propagate its error')
    assert not target.exists(), 'Partial JSON became a published stage receipt'
    assert pauses == []
    for item in Path(root).iterdir():
        item.unlink()
    scope['paused'] = False
    scope['json'] = json
    scope['hold']('before-ready')
    assert json.loads(target.read_text()) == dict(stage='before-ready', pid=os.getpid())
    assert pauses == [1.300]
    print('atomic-stage-publication-ok')
`;

describe.skipIf(process.platform === "win32")("transition receipt publication", () => {
  it("never exposes partial stage JSON or hides an interrupted writer", () => {
    const wrapper = recoveryTransitionWrapper({ source: "unused-source.py", stageFile: "STAGE_PATH",
      forkFile: "unused-forks.jsonl", stage: "before-ready", pause: "delay" });
    const result = spawnSync(process.env.QMD_RECOVERY_PYTHON ?? "/usr/bin/python3", ["-B", "-S", "-c", script, wrapper],
      { encoding: "utf8", timeout: 5_000 });
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe("atomic-stage-publication-ok");
  });
});
