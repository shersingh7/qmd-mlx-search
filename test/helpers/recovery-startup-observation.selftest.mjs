// Supplemental controls only: importing the helper must not launch a probe.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, linkSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { LIMITS, actualHost, classifyObservation, createOwnedRun, makeSupervisorWrapper, parsePythonSidecar,
  readPythonSidecar, replaceSupervisorArgv, writeExclusiveJson, isolatedEnvironment } from "./recovery-startup-observation.ts";

const root = fileURLToPath(new URL("../../", import.meta.url));
const helper = fileURLToPath(new URL("./recovery-startup-observation.ts", import.meta.url));
const source = path.join(root, "scripts/phase5_recovery_process.py");
const fixture = () => mkdtempSync(path.join(tmpdir(), "startup-observation-selftest-"));
const row = (event, seq = 0, extra = {}) => ({ event, seq, pid: 12345, tMs: seq, ...extra });
const entry = row("python-entry", 0, { version: "3.9.6 (authored parser fixture)", executable: "/authored/python" });
const jsonl = rows => rows.map(value => JSON.stringify(value) + "\n").join("");
const sha = value => createHash("sha256").update(value).digest("hex");

test("import is inert; explicit CLI refuses missing opt-in and a false host enum before allocating output", () => {
  const directory = fixture();
  try {
    const env = isolatedEnvironment(path.join(directory, "environment"));
    const importOnly = spawnSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e", `await import(${JSON.stringify(helper)}); console.log('import-only');`],
      { cwd: root, env, timeout: 10_000, encoding: "utf8" });
    assert.equal(importOnly.status, 0, importOnly.stderr); assert.equal(importOnly.stdout.trim(), "import-only");
    const output = path.join(directory, "not-created");
    const args = ["--import", "tsx", helper, "--runtime", "node"];
    const disabled = spawnSync(process.execPath, args, { cwd: root, env: { ...env, QMD_RECOVERY_STARTUP_DIR: output }, timeout: 10_000, encoding: "utf8" });
    assert.equal(disabled.status, 2); assert.equal(existsSync(output), false);
    const mismatch = spawnSync(process.execPath, [...args.slice(0, -1), "bun"], { cwd: root,
      env: { ...env, QMD_RECOVERY_STARTUP_DIR: output, QMD_RECOVERY_STARTUP_OBSERVATION: "1" }, timeout: 10_000, encoding: "utf8" });
    assert.equal(mismatch.status, 2); assert.equal(existsSync(output), false);
    assert.ok(!disabled.stderr.includes("import-only"));
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("actual test host is Node, not the package launcher or the requested enum", () => {
  assert.equal(actualHost("node").runtime, "node");
  assert.equal(actualHost().bun, null); assert.equal(actualHost().executable, process.execPath);
  assert.throws(() => actualHost("bun"), /host mismatch/);
  assert.throws(() => actualHost("$(echo event-text)"), /runtime enum/);
});

test("absent and partial sidecars remain unknown/partial, never invented launch or completion proof", () => {
  const absent = parsePythonSidecar(null);
  assert.equal(absent.status, "absent"); assert.equal(absent.selectedInterpreter, null);
  assert.ok(Object.values(absent.milestones).every(value => value === null));
  assert.equal(parsePythonSidecar("").status, "partial");
  const truncated = parsePythonSidecar(jsonl([entry]) + '{"event":"module-loaded"');
  assert.equal(truncated.status, "partial"); assert.equal(truncated.records.length, 1);
  assert.equal(truncated.selectedInterpreter.executable, "/authored/python");
  assert.equal(truncated.milestones["module-loaded"], null);
  assert.equal(truncated.milestones["wrapper-return"], null);
  assert.equal(parsePythonSidecar(jsonl([entry, row("module-loaded", 1)])).status, "partial");
});

test("bounded sidecar parser rejects malformed, foreign-PID, nonmonotone, extra-field and oversized records", () => {
  for (const text of ["not-json\n", jsonl([entry, row("module-loaded", 1, { pid: 7 })]),
    jsonl([entry, row("module-loaded", 1, { tMs: -1 })]), jsonl([{ ...entry, credentials: "never-emit" }]),
    jsonl([entry, row("unknown-event", 1)]), jsonl([entry, row("module-loaded", 9)]),
    jsonl([entry, row("admission-read", 1, { budgetMs: 0 })])]) {
    const parsed = parsePythonSidecar(text);
    assert.equal(parsed.status, "malformed"); assert.ok(!JSON.stringify(parsed).includes("never-emit"));
  }
  assert.equal(parsePythonSidecar("x".repeat(LIMITS.sidecarBytes + 1)).status, "oversized");
  const complete = parsePythonSidecar(jsonl([entry, row("module-loaded", 1), row("wrapper-return", 2)]));
  assert.equal(complete.status, "complete"); assert.equal(complete.milestones["prepared-before-send"], null);
  const prefix = parsePythonSidecar(jsonl([entry]) + "malformed\n");
  assert.equal(prefix.status, "malformed"); assert.equal(prefix.records.length, 1);
});

test("sidecar file reader refuses links and nonregular objects, bounds bytes, and does not delete partial evidence", () => {
  const directory = fixture();
  try {
    const file = path.join(directory, "sidecar.jsonl");
    assert.equal(readPythonSidecar(file).status, "absent");
    writeFileSync(file, jsonl([entry]));
    assert.equal(readPythonSidecar(file).status, "partial");
    assert.equal(existsSync(file), true);
    const link = path.join(directory, "link"); symlinkSync(file, link);
    assert.equal(readPythonSidecar(link).status, "unsafe");
    const hard = path.join(directory, "hard"); linkSync(file, hard);
    assert.equal(readPythonSidecar(hard).status, "unsafe");
    assert.equal(readPythonSidecar(directory).status, "unsafe");
    const large = path.join(directory, "large"); writeFileSync(large, "x".repeat(LIMITS.sidecarBytes + 1));
    assert.equal(readPythonSidecar(large).status, "oversized");
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

// These child scripts import the exact helper. No private export or replacement observer.
const rawReaderScript = `
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {syncBuiltinESMExports} from 'node:module';
import path from 'node:path';
const {readPythonSidecar,LIMITS,actualHost}=await import(process.env.STARTUP_SELFTEST_HELPER);
const directory=process.env.STARTUP_SELFTEST_DIRECTORY;
const row=(event,seq,extra={})=>({event,seq,pid:123,tMs:seq,...extra});
const entry=row('python-entry',0,{version:'Python café Δ 🐍 �',executable:'/authored/解释器'});
const line=value=>JSON.stringify(value)+'\\n';
const terminal=row('wrapper-return',1);
const checks=[];
function check(name,bytes,status,prefix,trailing=0){
 const file=path.join(directory,name+'.jsonl');fs.writeFileSync(file,bytes,{flag:'wx',mode:0o600});
 const actual=readPythonSidecar(file);
 assert.equal(actual.status,status,name);assert.equal(actual.bytes,bytes.length,name);
 assert.equal(actual.records.length,prefix,name);assert.equal(actual.trailingBytes,trailing,name);
 assert.deepEqual(fs.readFileSync(file),bytes,'raw evidence retained: '+name);
 if(prefix===0)assert.equal(actual.selectedInterpreter,null,name);
 else assert.deepEqual(actual.selectedInterpreter,{version:entry.version,executable:entry.executable,pid:123},name);
 checks.push({name,status:actual.status,prefix:actual.records.length,bytes:actual.bytes,trailingBytes:actual.trailingBytes});
 return file;
}
const head=Buffer.from('{"event":"python-entry","seq":0,"pid":123,"tMs":0,"version":"');
const tail=Buffer.from('","executable":"/authored/python"}\\n'+line(terminal));
for(const [name,bad] of [['ff',[255]],['continuation',[128]],['overlong',[192,175]],
 ['bad-continuation',[226,40,161]],['surrogate',[237,160,128]],['above-unicode',[244,144,128,128]],['truncated-sequence',[226,130]]]){
 check(name,Buffer.concat([head,Buffer.from(bad),tail]),'malformed',0);
}
check('corrupt-after-valid-entry',Buffer.concat([Buffer.from(line(entry)+'{"event":"module-'),Buffer.from([255]),
 Buffer.from('loaded","seq":1,"pid":123,"tMs":1}\\n'+line(row('wrapper-return',2)))]),'malformed',1);
const valid=check('valid-unicode-and-literal-replacement',Buffer.from(line(entry)+line(terminal)),'complete',2);
const incomplete=Buffer.from([226,130]);
const partial=Buffer.concat([Buffer.from(line(entry)+'{"event":"'),incomplete]);
check('truncated-multibyte-tail',partial,'partial',1,Buffer.byteLength('{"event":"')+incomplete.length);
const atCap=Buffer.concat([Buffer.from(line(entry)),Buffer.alloc(LIMITS.sidecarBytes-Buffer.byteLength(line(entry)),120)]);
check('bounded-tail',atCap,'partial',1,atCap.length-Buffer.byteLength(line(entry)));
check('above-cap',Buffer.concat([atCap,Buffer.from('x')]),'oversized',0);
// Exact-reader FD failure injection is qualified only when the host applies it.
let opens=0,closes=0,reads=0;
const original={openSync:fs.openSync,closeSync:fs.closeSync,readSync:fs.readSync};
fs.openSync=function(...args){const fd=original.openSync(...args);opens++;return fd;};
fs.closeSync=function(...args){closes++;return original.closeSync(...args);};
fs.readSync=function(){reads++;throw Object.assign(new Error('authored read failure'),{code:'EIO'});};
syncBuiltinESMExports();
let fault;
try{
 const observed=readPythonSidecar(valid);
 if(process.versions.bun && reads===0){
  assert.equal(observed.status,'complete');assert.equal(opens,0);assert.equal(closes,0);
  fault={qualified:false,reason:'native Bun does not synchronize node:fs namespace mocks into existing named imports',opens,closes,reads};
 }else{
  assert.equal(observed.status,'io-error');assert.equal(opens,1);assert.equal(closes,1);assert.equal(reads,1);
  fault={qualified:true,status:observed.status,opens,closes,reads};
 }
}finally{Object.assign(fs,original);syncBuiltinESMExports();}
console.log(JSON.stringify({actualHost:actualHost(),checks,fault,fixtureCleanupStarted:false}));
`;

for (const runtime of ["node", "bun"]) test(`raw UTF-8 reader validates bytes without replacing corruption on actual ${runtime}`, () => {
  const directory = fixture();
  try {
    const env = { ...isolatedEnvironment(path.join(directory, "environment")), STARTUP_SELFTEST_HELPER: helper, STARTUP_SELFTEST_DIRECTORY: directory };
    const executable = runtime === "node" ? process.execPath : "bun";
    const args = runtime === "node" ? ["--import", "tsx", "--input-type=module", "-e", rawReaderScript] : ["-e", rawReaderScript];
    const child = spawnSync(executable, args, { cwd: root, env, encoding: "utf8", timeout: 10_000 });
    assert.equal(child.status, 0, `${runtime}: ${child.stderr || child.error}`);
    const receipt = JSON.parse(child.stdout);
    assert.equal(receipt.actualHost.runtime, runtime);
    assert.equal(receipt.checks.length, 12);
    assert.equal(receipt.fixtureCleanupStarted, false);
    if (runtime === "node") assert.equal(receipt.fault.qualified, true);
    // Keep unsupported Bun mock coverage explicit in TAP, never call it a passed injection.
    console.log(JSON.stringify({ evidence: "raw-reader-controls", ...receipt }));
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

const exceptionalCallerScript = `
import assert from 'node:assert/strict';
import {mock} from 'node:test';
import childProcess,{spawn as namedSpawn} from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
import {EventEmitter,errorMonitor} from 'node:events';
import {PassThrough} from 'node:stream';
import {readFileSync,existsSync} from 'node:fs';
import path from 'node:path';
const root=process.cwd(),source=path.join(root,'scripts/phase5_recovery_process.py');
const originalSpawn=childProcess.spawn,environment=process.env;
const result={launch:{state:'started',supervisorReaped:true},timedOut:false,exitCode:0,signal:null,
 cleanup:{containmentVerified:false,childReaped:true,groupTerminated:true,sessionTerminated:true,pipesClosed:true,forcedPipeClose:false}};
const failure=new Error('authored caller/finish failure');
let mode,calls,children;
const fakeSpawn=function(exe,args){
 assert.equal(exe,'/usr/bin/python3');assert.deepEqual(args.slice(0,2),['-S','-u']);assert.ok(args[2].endsWith('/self-observing-supervisor.py'));
 const child=new EventEmitter();Object.assign(child,{pid:123,exitCode:null,signalCode:null,stdout:new PassThrough(),stderr:new PassThrough()});
 child.stdio=[null,child.stdout,child.stderr,new PassThrough()];
 child.kill=()=>{throw Error('fixture must never signal subject');};
 const targets=[child,child.stdout,child.stderr,child.stdio[3]],baselines=[],removed=new Set();let removals=0,failSnapshot=false;
 Object.defineProperty(child,'_closesNeeded',{get(){if(failSnapshot)throw failure;return 4;}});
 for(const target of targets){
  for(const event of ['spawn','exit','close','end','data',errorMonitor])target.on(event,()=>{});
  const baseline=new Map(target.eventNames().map(event=>[event,target.rawListeners(event)]));baselines.push(baseline);
  const remove=target.removeListener;
  target.removeListener=function(event,listener){
   if(!(baseline.get(event)||[]).includes(listener)){removals++;assert.ok(!removed.has(listener),'observer released twice');removed.add(listener);}
   return remove.call(this,event,listener);
  };
 }
 const record={child,targets,baselines,removed,get removals(){return removals;},fail(){failSnapshot=true;},reset(){failSnapshot=false;}};
 children.push(record);return child;
};
childProcess.spawn=fakeSpawn;syncBuiltinESMExports();
const replacement=mock.module(new URL('./scripts/phase5_e2e_runner.ts', 'file://'+root+'/').href,{namedExports:{executeRecoveryWorker(){
 calls++;
 if(!mode.startsWith('no-observer'))namedSpawn('/usr/bin/python3',['-S','-u',source,JSON.stringify({timeoutMs:200,contained:false})],{detached:true,stdio:['pipe','pipe','pipe','pipe']});
 if(mode==='sync-throw')throw failure;
 if(mode==='reject'||mode==='no-observer-reject')return Promise.reject(failure);
 if(mode==='finish-throw')children.at(-1).fail();
 return Promise.resolve(mode==='no-observer'?{...result,launch:{state:'not-attempted',supervisorReaped:false}}:result);
}}});
const checks=[];
try{
 const {runStartupObservation}=await import(process.env.STARTUP_SELFTEST_HELPER);
 for(mode of ['reject','sync-throw','finish-throw','normal','no-observer','no-observer-reject']){
  calls=0;children=[];
  const observed=await runStartupObservation(process.env.STARTUP_SELFTEST_DIRECTORY,'node');
  assert.equal(process.env,environment);assert.equal(childProcess.spawn,fakeSpawn);assert.equal(namedSpawn,fakeSpawn);
  const failed=!['normal','no-observer'].includes(mode);
  assert.equal(calls,failed?1:2,mode);assert.equal(observed.report.cases.length,failed?0:2,mode);
  assert.deepEqual(observed.report.collectionErrors,failed?['observation-incomplete; inspect retained owned artifacts']:[],mode);
  assert.deepEqual(JSON.parse(readFileSync(path.join(observed.directory,'observation.json'),'utf8')),observed.report);
  if(failed)assert.equal(existsSync(path.join(observed.directory,'case-5000')),false);
  for(const record of children){
   record.reset();
   assert.equal(record.removals,13,mode+' must release exactly the 13 observer listeners before fixture cleanup');
   assert.equal(record.removed.size,13);
   for(const [i,target]of record.targets.entries()){
    assert.deepEqual(new Map(target.eventNames().map(event=>[event,target.rawListeners(event)])),record.baselines[i],mode+' must preserve caller/unrelated listeners');
    if(i>0)assert.equal(target.destroyed,false,'observer must not close subject streams');
   }
  }
  for(const item of observed.report.cases){
   assert.equal(item.invocation.restored,true);assert.equal(item.snapshot.subjectFixtureSignals,0);assert.equal(item.snapshot.fixtureCleanupStarted,false);
   if(mode==='no-observer')assert.equal(item.nodeOwner,null);else assert.equal(item.nodeOwner.restoration.listenersRemoved,true);
  }
  checks.push({mode,calls,observerRemovals:children.map(x=>x.removals),collectionErrors:observed.report.collectionErrors,environmentRestored:true,spawnRestored:true,fixtureCleanupStarted:false});
 }
 console.log(JSON.stringify({checks,realRecoveryCalls:0}));
}finally{replacement.restore();childProcess.spawn=originalSpawn;syncBuiltinESMExports();}
`;

test("exceptional caller paths release only observer listeners; normal/no-observer disposal is idempotent", () => {
  const directory = fixture();
  try {
    const env = { ...isolatedEnvironment(path.join(directory, "environment")), STARTUP_SELFTEST_HELPER: helper,
      STARTUP_SELFTEST_DIRECTORY: directory, QMD_RECOVERY_STARTUP_OBSERVATION: "1" };
    const child = spawnSync(process.execPath, ["--experimental-test-module-mocks", "--import", "tsx", "--input-type=module", "-e", exceptionalCallerScript],
      { cwd: root, env, encoding: "utf8", timeout: 10_000 });
    assert.equal(child.status, 0, child.stderr || String(child.error));
    const receipt = JSON.parse(child.stdout); assert.equal(receipt.realRecoveryCalls, 0); assert.equal(receipt.checks.length, 6);
    console.log(JSON.stringify({ evidence: "exceptional-caller-controls", ...receipt }));
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("exclusive publication and owned directories cannot replace existing files or follow an output alias", () => {
  const directory = fixture();
  try {
    const run = createOwnedRun(path.join(directory, "artifacts"), "node");
    assert.equal(lstatSync(run).isDirectory(), true);
    assert.equal(lstatSync(run).mode & 0o777, 0o700);
    const file = path.join(run, "result.json"); writeExclusiveJson(file, { public: true });
    const before = readFileSync(file);
    assert.throws(() => writeExclusiveJson(file, { public: false })); assert.deepEqual(readFileSync(file), before);
    const alias = path.join(directory, "alias"); symlinkSync(run, alias);
    assert.throws(() => createOwnedRun(alias, "node"), /directory|symlink/);
    const linked = path.join(run, "linked.json"); symlinkSync(file, linked);
    assert.throws(() => writeExclusiveJson(linked, { public: false })); assert.deepEqual(readFileSync(file), before);
    assert.throws(() => writeExclusiveJson(path.join(run, "too-large.json"), { text: "x".repeat(LIMITS.jsonBytes) }), /bounded/);
    assert.equal(existsSync(path.join(run, "too-large.json")), false);
    assert.throws(() => createOwnedRun("relative", "node"), /absolute/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("environment construction clears ambient context and separates every process's offline caches", () => {
  const directory = fixture();
  try {
    const env = isolatedEnvironment(path.join(directory, "one"));
    const second = isolatedEnvironment(path.join(directory, "two"));
    for (const key of ["OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "PYTHONPATH", "NODE_OPTIONS", "GITHUB_TOKEN"]) assert.equal(key in env, false);
    for (const key of ["HOME", "HERMES_HOME", "HERMES_DIR", "HERMES_KNOWLEDGE_DIR", "KNOWLEDGE_DIR", "XDG_CACHE_HOME", "HF_HOME", "UV_CACHE_DIR", "TMPDIR"]) {
      assert.ok(env[key].startsWith(directory)); assert.notEqual(env[key], second[key]);
    }
    for (const key of ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "UV_OFFLINE", "CI", "REQUIRE_BUN"]) assert.equal(env[key], "1");
    assert.equal(env.QMD_LLAMA_GPU, "false");
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("override changes exactly the script pathname, never interpreter, original flags, command bytes or budget", () => {
  const command = ["/bin/echo", "$(event-text); literal ' argv"];
  for (const timeoutMs of [200, 5000]) {
    const args = ["/usr/bin/python3", "-S", "-u", source, JSON.stringify({ command, timeoutMs, contained: false })];
    const before = structuredClone(args);
    const replaced = replaceSupervisorArgv(args, source, "/owned/wrapper.py");
    assert.deepEqual(args, before); assert.deepEqual(replaced, [...args.slice(0, 3), "/owned/wrapper.py", args[4]]);
    assert.equal(JSON.parse(replaced[4]).timeoutMs, timeoutMs); assert.deepEqual(JSON.parse(replaced[4]).command, command);
    for (const bad of [["/PATH/python", ...args.slice(1)], [args[0], "-B", ...args.slice(2)], [...args.slice(0, 3), "/other.py", args[4]]])
      assert.throws(() => replaceSupervisorArgv(bad, source, "/owned/wrapper.py"), /supervisor argv/);
  }
  const wrapper = makeSupervisorWrapper({ source, expectedSha256: sha(readFileSync(source)), sidecar: "/owned/sidecar.jsonl" });
  assert.ok(wrapper.includes("exec(compile(code, source, 'exec'), scope)"));
  assert.ok(wrapper.includes("original_read(limit)")); assert.ok(!wrapper.includes("os.read(0"));
  assert.ok(!wrapper.includes("S200")); assert.ok(!wrapper.includes("time.sleep"));
  assert.ok(!wrapper.includes("sys.settrace")); assert.ok(!wrapper.includes("time.monotonic ="));
});

test("classification keeps failure/no-work/incomplete cleanup distinct from real injected work, never product qualification", () => {
  const result = { launch: { state: "started", admissionGranted: true, supervisorReaped: true }, pid: 12345,
    timedOut: false, exitCode: 0, signal: null, report: { status: "completed", pid: 12345 },
    cleanup: { childReaped: true, groupTerminated: true, sessionTerminated: true, pipesClosed: true, forcedPipeClose: false, containmentVerified: false } };
  assert.equal(classifyObservation(result, true), "authored-work-observed-uncontained");
  assert.equal(classifyObservation({ ...result, timedOut: true }, true), "timed-out");
  assert.equal(classifyObservation({ ...result, cleanup: { ...result.cleanup, forcedPipeClose: true } }, true), "incomplete-cleanup");
  assert.equal(classifyObservation(result, false), "work-not-validated");
  assert.equal(classifyObservation({ ...result, launch: { state: "not-started" }, pid: undefined }, false), "not-started");
  assert.equal(classifyObservation({ ...result, cleanup: { ...result.cleanup, containmentVerified: true } }, true), "unexpected-containment-claim");
});

test("workflow leaves original canonical prefix and old observation upload intact and runs actual hosts only afterward", () => {
  const workflow = readFileSync(path.join(root, ".github/workflows/ci.yml"), "utf8");
  const canonicalEnd = workflow.indexOf("      # Explicit AFTER the canonical suite/smoke:");
  assert.ok(canonicalEnd > 0);
  // Hash of the original workflow through (and including) canonical smoke.
  assert.equal(sha(workflow.slice(0, canonicalEnd)), "d723590bbdca7e4e20a61eb6782d77f25defd3a633403c2c9d654ecfcfa00aa1");
  const upload = workflow.indexOf("      - name: Preserve first-invocation bootstrap observations even on failure");
  for (const name of ["After-canonical caller startup observation (actual Node)", "After-canonical caller startup observation (native Bun)", "After-canonical startup observation selftests"]) {
    const at = workflow.indexOf(name); assert.ok(at > canonicalEnd && at < upload);
  }
  assert.ok(workflow.includes('env QMD_RECOVERY_STARTUP_OBSERVATION=1'));
  assert.ok(workflow.includes('node --import tsx test/helpers/recovery-startup-observation.ts --runtime node'));
  assert.ok(workflow.includes('bun test/helpers/recovery-startup-observation.ts --runtime bun'));
  assert.ok(workflow.includes('${{ runner.temp }}/qmd-recovery-bootstrap/*.json'));
  assert.ok(workflow.includes('${{ runner.temp }}/qmd-recovery-bootstrap/*.tap'));
  assert.ok(!workflow.includes('continue-on-error: ${{'));
});

test("optional actual-host evidence requires both valid positive controls and preserves independent short outcomes", { skip: !process.env.QMD_RECOVERY_STARTUP_TEST_DIR }, () => {
  const directory = process.env.QMD_RECOVERY_STARTUP_TEST_DIR;
  const reports = readdirSync(directory).filter(name => /^(node|bun)-/.test(name)).map(name => JSON.parse(readFileSync(path.join(directory, name, "observation.json"), "utf8")));
  assert.equal(reports.length, 2);
  assert.deepEqual(reports.map(report => report.actualHost.runtime).sort(), ["bun", "node"]);
  for (const report of reports) {
    assert.equal(report.productQualified, false); assert.equal(report.coldCausality, "not-established");
    assert.equal(report.cases.length, 2);
    assert.deepEqual(report.cases.map(item => item.timeoutMs), [200, 5000]);
    for (const item of report.cases) {
      assert.equal(item.productQualified, false); assert.equal(item.result.cleanup.containmentVerified, false);
      assert.deepEqual(item.invocation.originalArgv.slice(0, 4), ["/usr/bin/python3", "-S", "-u", source]);
      assert.equal(JSON.parse(item.invocation.actualArgv[4]).timeoutMs, item.timeoutMs);
      assert.equal(item.unsupportedCallerMilestones.preparedReceiptMs, null);
      assert.equal(item.unsupportedCallerMilestones.remainingBudgetMs, null);
      for (const event of item.python.records.filter(record => record.event === "admission-read")) assert.ok(event.budgetMs > 0 && event.budgetMs <= item.timeoutMs);
      assert.equal(item.snapshot.fixtureCleanupStarted, false);
      assert.equal(item.snapshot.subjectFixtureSignals, 0);
      assert.equal(item.snapshot.supervisorPidPresent, false);
    }
    const positive = report.cases[1];
    assert.equal(positive.classification, "authored-work-observed-uncontained", JSON.stringify(positive));
    assert.equal(positive.workValidated, true);
    assert.equal(positive.python.status, "complete");
    assert.equal(positive.snapshot.workerPidPresent, false);
    assert.equal(positive.invocation.restored, true);
    if (report.actualHost.runtime === "bun") assert.equal(positive.nodeOwner, null);
  }
});
