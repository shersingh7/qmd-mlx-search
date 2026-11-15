export type RecoveryTransitionStage = "before-fork" | "fork-return" | "registered" | "before-ready" | "after-ready";
export type RecoveryTransitionPause = "delay" | "stop";

/** Generate the subprocess fixture without importing/registering a test suite. */
export function recoveryTransitionWrapper({ source, stageFile, forkFile, stage, pause }: {
  source: string; stageFile: string; forkFile: string; stage: RecoveryTransitionStage; pause: RecoveryTransitionPause;
}): string {
  // Fork-return and after-ready already hold in their hooks: no AST or tracer.
  const needsTrace = stage !== "fork-return" && stage !== "after-ready";
  const anchors = needsTrace ? `import ast
tree = ast.parse(code, source)
mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
if len(mains) != 1: raise ValueError('Expected one top-level supervisor main')
main = mains[0]
def statement_lists(node):
    # Stay in this function's lexical scope; nested shadow functions/classes
    # cannot supply its creation or readiness transition.
    for _, value in ast.iter_fields(node):
        children = value if isinstance(value, list) else [value]
        if any(isinstance(child, ast.stmt) for child in children): yield children
        for child in children:
            if isinstance(child, ast.AST) and not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                yield from statement_lists(child)
def worker_fork(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'worker'
            and isinstance(node.value, ast.Call) and not node.value.args and not node.value.keywords
            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'fork'
            and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'os')
forks = [(node, siblings) for siblings in statement_lists(main) for node in siblings if worker_fork(node)]
if len(forks) != 1: raise ValueError('Expected one real-main worker = os.fork()')
fork, siblings = forks[0]
fork_line = fork.lineno
# READY must be the unique subsequent sibling in the same startup block, not
# a nested ready() shadow or the later cleanup-ready call outside that block.
readies = [node for node in siblings[siblings.index(fork) + 1:] if isinstance(node, ast.Expr)
           and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
           and node.value.func.id == 'ready' and not node.value.args and not node.value.keywords]
if len(readies) != 1: raise ValueError('Expected one startup ready() after worker creation')
ready_line = readies[0].lineno
` : "";
  const binding = needsTrace ? `import types
# Coordinates select a constant from the SAME module code passed to exec; they
# are not evidence that the loaded callable retained that code or its globals.
first_line = min([main.lineno] + [node.lineno for node in main.decorator_list])
compiled_mains = [constant for constant in module_code.co_consts
                  if type(constant) is types.CodeType and constant.co_name == 'main'
                  and constant.co_firstlineno == first_line]
if len(compiled_mains) != 1: raise ValueError('Expected one compiled top-level supervisor main')
main_code = compiled_mains[0]
loaded_main = scope.get('main')
# Fixture integrity only, not a Python sandbox. Reject metadata shells before
# consulting attributes, and require identity rather than structural equality.
if (type(loaded_main) is not types.FunctionType or loaded_main.__code__ is not main_code
        or loaded_main.__globals__ is not scope):
    raise ValueError('Loaded main does not match the qualified source anchor')
` : "";
  const tracing = needsTrace ? `main_frame = None
def trace(frame, event, arg):
    global main_frame
    # The global tracer still receives CALL events, but returning None prevents
    # unrelated frames from generating LINE events. Identity survives fork.
    if frame.f_code is not main_code or frame.f_globals is not scope: return None
    if main_frame is None: main_frame = frame
    if frame is not main_frame: return None
    if not paused and event == 'line':
        if ${stage === "before-fork" ? "True" : "False"} and frame.f_lineno == fork_line: hold('before-fork')
        if ${stage === "registered" ? "True" : "False"} and frame.f_lineno > fork_line and frame.f_locals.get('worker', 0) > 0: hold('registered')
        if ${stage === "before-ready" ? "True" : "False"} and frame.f_lineno == ready_line: hold('before-ready')
    return trace
sys.settrace(trace)
` : "";
  // Python exec compiles the exact local public source; this is not shell exec.
  // Keep real source loading/imports and all holds inside the original API call.
  return `import json, os, signal, sys, time
source = ${JSON.stringify(source)}
code = open(source, 'rb').read()
${anchors}sys.path.insert(0, os.path.dirname(source))
scope = dict(__name__='semantic_transition_regression')
module_code = compile(code, source, 'exec')
exec(module_code, scope)
${binding}paused = False
def hold(label):
    global paused
    if paused: return
    paused = True
    with open(${JSON.stringify(stageFile)}, 'w') as record:
        json.dump(dict(stage=label, pid=os.getpid()), record)
    ${pause === "stop" ? "os.kill(os.getpid(), signal.SIGSTOP)" : "time.sleep(1.300)"}
original_send = scope['send']
def observed_send(message):
    original_send(message)
    if message['type'] == 'ready' and ${stage === "after-ready" ? "True" : "False"}: hold('after-ready')
scope['send'] = observed_send
original_fork = os.fork
def observed_fork():
    pid = original_fork()
    if pid:
        with open(${JSON.stringify(forkFile)}, 'a') as record:
            record.write(json.dumps(dict(parent=os.getpid(), pid=pid)) + '\\n')
        if ${stage === "fork-return" ? "True" : "False"}: hold('fork-return')
    return pid
os.fork = observed_fork
${tracing}scope['main']()
`;
}
