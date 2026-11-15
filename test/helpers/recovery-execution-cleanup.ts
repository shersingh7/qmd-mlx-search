import type { ChildProcess } from "node:child_process";

type Release = { label: string; release: () => unknown | Promise<unknown> };
export type CleanupOutcome = { label: string; status: "completed" | "failed"; error?: unknown };
type Emitter = { on(event: string, listener: (...args: any[]) => void): unknown; once(event: string, listener: (...args: any[]) => void): unknown; removeListener(event: string, listener: (...args: any[]) => void): unknown };
type LocalPipe = Emitter & { closed?: boolean; destroyed?: boolean; destroy?: () => unknown };
type PipeOwner = { name: string; target: LocalPipe; terminal: TerminalObservation };
type LocalOwnership = { owner: RecoveryExecutionCleanup; pipes: PipeOwner[]; terminal: TerminalObservation; errors: unknown[] };
/** Private caller/reaper seam: observations are call-owned, never a registry. */
type OwnedProcessRelease = LocalOwnership & { deadline: number };

export class RecoveryExecutionCleanupError extends Error {
  constructor(readonly outcomes: readonly CleanupOutcome[]) {
    super("admission fixture cleanup failed; completion/custody is not certified");
    this.name = "RecoveryExecutionCleanupError";
  }
}

/** Call-owned error state. Never attach errors to an arbitrary thrown value. */
export class RecoveryExecutionCleanup {
  readonly outcomes: CleanupOutcome[] = [];
  readonly notes: Array<Record<string, unknown>> = [];
  private readonly releases: Release[] = [];
  private readonly failureObservers = new Set<(error: unknown) => void>();
  private hasPrimary = false;
  private primary: unknown;
  private inBody = false;
  report?: () => unknown | Promise<unknown>;

  defer(label: string, release: Release["release"]): void { this.releases.push({ label, release }); }
  note(value: Record<string, unknown>): void { this.notes.push(value); }
  failed(label: string, error: unknown): void { this.outcomes.push({ label, status: "failed", error }); }
  /** Capture at the actual synchronous body boundary, BEFORE async unwind.
   * A previously observed callback stays primary; opaque falsey values count. */
  capture(error: unknown): void {
    if (!this.hasPrimary) { this.hasPrimary = true; this.primary = error; }
    else if (!Object.is(error, this.primary)) this.failed("body-secondary", error);
  }
  /** Actual operation/event failure: admit only while this owner's body is
   * active. Shared teardown and pure failed() diagnostics never claim primary. */
  fail(error: unknown, label = "callback"): void {
    this.failed(label, error);
    if (!this.hasPrimary && this.inBody) { this.hasPrimary = true; this.primary = error; }
    if (this.inBody) for (const observer of this.failureObservers) observer(this.primary);
  }
  check(): void { if (this.hasPrimary) throw this.primary; }
  callback(listener: (...args: any[]) => void): (...args: any[]) => void {
    return (...args) => {
      // Protocol work stops when its actual body drains. Independent resource
      // observations remain live through pending terminal delivery.
      if (!this.inBody) return;
      try { listener(...args); } catch (error) { this.fail(error); }
    };
  }
  ownedHandle(child: ChildProcess, label: string, release: (observation: OwnedProcessRelease) => unknown | Promise<unknown>): void {
    const local = acquireLocal(this, child, label);
    // Retain the capability BEFORE fallible attachment. Every release shares
    // these acquisition observations and this one release/disposal deadline.
    this.defer(label, async () => {
      const deadline = performance.now() + 800;
      let failed = false, primary: unknown;
      try { await release({ ...local, deadline }); } catch (error) { failed = true; primary = error; }
      await disposePipes(this, local.pipes, deadline, false);
      local.terminal.handoff();
      if (failed) throw primary;
      if (local.errors.length) throw local.errors[0];
    });
    startLocal(local);
    this.check();
  }
  listen(target: Emitter, event: string, listener: (...args: any[]) => void, once = false): void {
    const guarded = this.callback(listener);
    this.defer("listener-" + event, () => target.removeListener(event, guarded));
    try { if (once) target.once(event, guarded); else target.on(event, guarded); }
    catch (error) { this.capture(error); throw error; }
  }
  diagnostics() {
    return { notes: this.notes.map(note => ({ ...note })), outcomes: this.outcomes.map(({ label, status, error }) => {
      let failure: string | undefined;
      if (status === "failed") { try { failure = String(error); } catch { failure = "unprintable cleanup failure"; } }
      return { label, status, ...(failure === undefined ? {} : { failure }) };
    }) };
  }
  async run<T>(body: () => T | Promise<T>): Promise<T> {
    let result: T | undefined;
    this.inBody = true;
    // Never race away work still using a resource/root. Local caller catches
    // capture synchronous failures; this await is only the fallback boundary.
    try { result = await body(); }
    catch (error) { this.capture(error); }
    this.inBody = false;
    this.failureObservers.clear();
    for (const { label, release } of this.releases.reverse()) await this.attempt(label, release);
    if (this.report) await this.attempt("report", this.report);
    if (this.hasPrimary) throw this.primary;
    if (this.outcomes.some(outcome => outcome.status === "failed")) throw new RecoveryExecutionCleanupError(this.outcomes);
    return result as T;
  }
  private async attempt(label: string, release: Release["release"]): Promise<void> {
    try { await release(); this.outcomes.push({ label, status: "completed" }); }
    catch (error) { this.failed(label, error); }
  }

  controller(child: ChildProcess): { closed: Promise<void>; refuse: (error: unknown) => void } {
    let resolveClosed!: () => void, rejectClosed!: (error: unknown) => void;
    const closed = new Promise<void>((resolve, reject) => { resolveClosed = resolve; rejectClosed = reject; });
    void closed.catch(() => {}); // Setup may throw before the caller awaits it.
    this.failureObservers.add(rejectClosed);
    const local = acquireLocal(this, child, "controller");
    void local.terminal.closed.then(resolveClosed);
    this.defer("controller", () => releaseRecoveryController(child, {
      drained: local.terminal.closed, isClosed: () => local.terminal.isClosed, local,
    }));
    startLocal(local);
    this.check();
    return { closed, refuse: error => this.fail(error, "controller-refusal") };
  }
}

/** Resource-local lifetime observation, shared by active work and release.
 * Flags diagnose acquisition only; ONLY this object's close callback is a
 * witness. Unknown historical terminal delivery remains pending/unqualified. */
class TerminalObservation {
  readonly closed: Promise<void>;
  isClosed = false;
  private started = false;
  private resolve!: () => void;
  private readonly state: Record<string, unknown>;
  private readonly onError: (error: unknown) => void;
  private readonly onClose: () => void;
  constructor(private readonly owner: RecoveryExecutionCleanup, private readonly target: Emitter, private readonly label: string) {
    this.closed = new Promise(resolve => { this.resolve = resolve; });
    this.state = { resource: label, state: "acquired", closeObserved: false,
      closedFlagAtAcquisition: (target as LocalPipe).closed === true, destroyedFlagAtAcquisition: (target as LocalPipe).destroyed === true };
    owner.note(this.state);
    this.onError = error => owner.fail(error, label + "-error");
    this.onClose = () => {
      this.isClosed = true; this.state.state = "closed"; this.state.closeObserved = true;
      this.resolve();
      for (const [event, listener] of [["error", this.onError], ["close", this.onClose]] as const) {
        try { target.removeListener(event, listener); }
        catch (error) { this.state.listenerRestoration = "unconfirmed"; owner.fail(error, label + "-remove-" + event); }
      }
    };
  }
  start(): unknown[] {
    if (this.started) return [];
    this.started = true;
    const errors: unknown[] = [];
    for (const [event, listener] of [["error", this.onError], ["close", this.onClose]] as const) {
      // Independent attempts also cover call-through-after-attachment refusal.
      // Capture immediately, before another attachment can dispatch a callback.
      try { this.target.on(event, listener); }
      catch (error) { errors.push(error); this.owner.fail(error, this.label + "-observe-" + event); this.state.observation = "unconfirmed"; }
      if (this.isClosed) {
        try { this.target.removeListener(event, listener); }
        catch (error) { errors.push(error); this.owner.fail(error, this.label + "-remove-" + event); this.state.listenerRestoration = "unconfirmed"; }
      }
    }
    return errors;
  }
  handoff(): void {
    if (!this.isClosed) { this.state.state = "pending"; this.state.qualification = "unqualified"; }
  }
}
function ownPipes(owner: RecoveryExecutionCleanup, child: ChildProcess, label: string): PipeOwner[] {
  const pipes: PipeOwner[] = [];
  for (const name of ["stdin", "stdout", "stderr"] as const) {
    const target = child[name];
    if (target) pipes.push({ name, target, terminal: new TerminalObservation(owner, target, label + "-local-" + name) });
  }
  return pipes;
}
function acquireLocal(owner: RecoveryExecutionCleanup, child: ChildProcess, label: string): LocalOwnership {
  return { owner, pipes: ownPipes(owner, child, label), terminal: new TerminalObservation(owner, child, label), errors: [] };
}
function startLocal(local: LocalOwnership): unknown[] {
  local.errors.push(...local.terminal.start());
  for (const pipe of local.pipes) local.errors.push(...pipe.terminal.start());
  return local.errors;
}
async function within(owner: RecoveryExecutionCleanup, deadline: number, promise: Promise<unknown>): Promise<boolean> {
  const remaining = deadline - performance.now();
  if (remaining <= 0) return false;
  let timer: ReturnType<typeof setTimeout> | undefined, failed = false;
  try { return await Promise.race([promise.then(() => true), new Promise<false>((resolve, reject) => {
    try { timer = setTimeout(() => resolve(false), remaining); }
    catch (error) { owner.fail(error, "deadline-timer"); reject(error); }
  })]); }
  catch (error) {
    failed = true;
    // Admit at the observed await boundary, before a finalizer can fail.
    // fail() preserves standalone-body precedence without promoting failures
    // from a shared teardown into body primaries.
    owner.fail(error, "deadline-wait");
    throw error;
  }
  finally {
    try { clearTimeout(timer); }
    catch (error) { owner.fail(error, "deadline-timer-dispose"); if (!failed) throw error; }
  }
}
async function disposePipes(owner: RecoveryExecutionCleanup, pipes: PipeOwner[], deadline: number, keepInput: boolean): Promise<void> {
  // Observations already exist. Start all independent disposal attempts before
  // waiting for any actual close; destroyed/closed do not settle that wait.
  for (const pipe of pipes) {
    try {
      if (!pipe.terminal.isClosed && !pipe.target.destroyed && !(keepInput && pipe.name === "stdin")) {
        owner.note({ resource: "local-" + pipe.name, disposal: "forced-local-only", naturalDrain: false, descendantReclamation: false });
        if (!pipe.target.destroy) throw new Error("owned local endpoint has no disposal capability; close is unconfirmed");
        pipe.target.destroy();
      }
    } catch (error) { owner.fail(error, "local-" + pipe.name + "-dispose"); }
  }
  const disposing = pipes.filter(pipe => !(keepInput && pipe.name === "stdin"));
  if (disposing.some(pipe => !pipe.terminal.isClosed)) {
    try { await within(owner, deadline, Promise.all(disposing.map(pipe => pipe.terminal.closed))); }
    catch (error) { owner.fail(error, "local-close-wait"); }
  }
  for (const pipe of pipes) if (!pipe.terminal.isClosed) {
    pipe.terminal.handoff();
    if (!(keepInput && pipe.name === "stdin")) owner.failed("local-" + pipe.name + "-close", new Error("owned local endpoint close remains unconfirmed"));
  }
}

/** Sidecars are observations ONLY: this function does not grant signal custody. */
export function recoveryForkRecords(text: string | null): { status: "missing" | "empty" | "valid"; records: Array<{ parent: number; pid: number }> } {
  if (text === null) return { status: "missing", records: [] };
  const lines = text.trim().split("\n").filter(Boolean);
  const records: Array<{ parent: number; pid: number }> = [];
  for (const line of lines) {
    const value: unknown = JSON.parse(line);
    if (!value || typeof value !== "object" || Array.isArray(value)
      || !Number.isSafeInteger((value as any).parent) || (value as any).parent <= 0
      || !Number.isSafeInteger((value as any).pid) || (value as any).pid <= 0) {
      throw new TypeError("invalid recovery fork observation; descendant custody remains unqualified");
    }
    records.push({ parent: (value as any).parent, pid: (value as any).pid });
  }
  return { status: lines.length ? "valid" : "empty", records };
}

/** Direct HANDLE ownership only; never call this for an admitted controller.
 * The optional private observation carries the original acquisition witnesses
 * and deadline. A standalone caller acquires its own, honestly later lifetime. */
export async function releaseOwnedProcess(child: ChildProcess, killed: number[], budget: number | OwnedProcessRelease = 800): Promise<void> {
  const shared = typeof budget === "number" ? undefined : budget;
  const cleanup = shared?.owner ?? new RecoveryExecutionCleanup();
  const deadline = shared?.deadline ?? performance.now() + (budget as number);
  const local = shared ?? acquireLocal(cleanup, child, "owned-handle");
  const release = async () => {
    try {
      const errors = shared ? [] : startLocal(local);
      if (child.exitCode !== null || child.signalCode !== null) {
        cleanup.note({ resource: "owned-handle", processExit: "observed", localPipeClose: "separate", custody: "unqualified" });
      } else {
        let resolve!: () => void;
        const exited = new Promise<void>(done => { resolve = done; });
        const onExit = () => resolve();
        // A release-local observer is removed independently even when attachment
        // throws; unlike lifetime containment it must not outlive this attempt.
        try {
          try { child.on("exit", onExit); } catch (error) { errors.push(error); cleanup.fail(error, "exit-observer"); }
          try {
            killed.push(child.pid!); // Diagnostic only. The handle is authority.
            if (!child.kill("SIGKILL")) throw new Error("owned handle refused kill; release is unconfirmed");
          } catch (error) { errors.push(error); cleanup.fail(error, "owned-kill"); }
          if (!errors.length && !await within(cleanup, deadline, exited)) {
            const error = new Error("owned handle exit timed out; release is unconfirmed");
            errors.push(error); cleanup.fail(error, "owned-exit");
          }
        } finally {
          try { child.removeListener("exit", onExit); } catch (error) { errors.push(error); cleanup.fail(error, "exit-listener"); }
        }
        cleanup.note({ resource: "owned-handle", processExit: errors.length ? "unconfirmed" : "observed", localPipeClose: "separate", custody: "unqualified" });
      }
      if (errors.length) throw errors[0];
    } catch (error) {
      cleanup.fail(error, "owned-release"); throw error;
    }
  };
  if (shared) await release(); // ownedHandle independently owns disposal/handoff.
  else await cleanup.run(async () => {
    cleanup.defer("terminal-handoff", () => local.terminal.handoff());
    cleanup.defer("local-endpoints", () => disposePipes(cleanup, local.pipes, deadline, false));
    await release();
  });
}

type ControllerObservation = { drained: Promise<void>; isClosed: () => boolean; local?: LocalOwnership };
/** EOF retains controller authority. 800ms is one total release/disposal budget,
 * never permission for a controller SIGKILL or numeric descendant rescue. */
export async function releaseRecoveryController(child: ChildProcess, observation: ControllerObservation, timeoutMs = 800): Promise<void> {
  const cleanup = observation.local?.owner ?? new RecoveryExecutionCleanup();
  const deadline = performance.now() + timeoutMs;
  const pipes = observation.local?.pipes ?? ownPipes(cleanup, child, "controller");
  const terminal = observation.local?.terminal ?? (typeof child.on === "function" ? new TerminalObservation(cleanup, child, "controller") : undefined);
  const release = async () => {
    const errors: unknown[] = [...(observation.local?.errors ?? [])];
    let keepInput = true;
    try {
      if (!observation.local) {
        errors.push(...(terminal?.start() ?? []));
        for (const pipe of pipes) errors.push(...pipe.terminal.start());
      }
      try { if (!observation.isClosed()) child.stdin?.end(); }
      catch (error) { errors.push(error); cleanup.fail(error, "controller-eof"); }
      if (!observation.isClosed() && !await within(cleanup, deadline, observation.drained)) {
        const error = new Error("controller cooperative drain timed out; descendant custody remains unqualified");
        errors.push(error); cleanup.fail(error, "controller-drain");
      }
      keepInput = !observation.isClosed();
    } catch (error) {
      // An unexpected release-body refusal is observed here, not after the
      // standalone run's async unwind. Shared teardown stays diagnostic-only.
      errors.push(error); cleanup.fail(error, "controller-release");
    }
    await disposePipes(cleanup, pipes, deadline, keepInput);
    terminal?.handoff();
    if (errors.length) throw errors[0];
  };
  if (observation.local) await release();
  else await cleanup.run(release);
}
