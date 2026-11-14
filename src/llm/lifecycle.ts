/** Per-owner leases and cancellation. Native work keeps its lease until it settles. */
export class LLMDisposedError extends Error {
  constructor() {
    super("LLM instance is closing or disposed");
    this.name = "LLMDisposedError";
  }
}

export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Operation aborted", "AbortError");
}

/** Unlike AbortSignal.any, these subscriptions can be removed at scope exit. */
export function linkSignals(signals: (AbortSignal | undefined)[]): { signal: AbortSignal; dispose: () => void } {
  const controller = new AbortController();
  const removers: (() => void)[] = [];
  for (const signal of new Set(signals)) {
    if (!signal) continue;
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    const onAbort = () => controller.abort(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    removers.push(() => signal.removeEventListener("abort", onAbort));
  }
  return { signal: controller.signal, dispose: () => { for (const remove of removers) remove(); } };
}

/** Reject the caller promptly, without pretending an uninterruptible native task has stopped. */
export function abortable<T>(work: Promise<T>, signal: AbortSignal, onLateValue?: (value: T) => void): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason ?? new DOMException("Operation aborted", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
    work.then(
      value => {
        signal.removeEventListener("abort", onAbort);
        if (signal.aborted) {
          try { onLateValue?.(value); } catch { /* cleanup must not leave an unhandled continuation */ }
          onAbort();
        } else resolve(value);
      },
      error => { signal.removeEventListener("abort", onAbort); reject(error); },
    );
    if (signal.aborted) {
      signal.removeEventListener("abort", onAbort);
      onAbort();
    }
  });
}

export class LLMLifecycle {
  private state: "active" | "closing" | "closed" = "active";
  private controller = new AbortController();
  private sessions = 0;
  private operations = 0;
  private pending = new Set<Promise<unknown>>();
  private cleanupError: Error | null = null;
  onActivity: () => void = () => {};

  get signal(): AbortSignal { return this.controller.signal; }
  get active(): boolean { return this.state === "active"; }
  get canUnload(): boolean { return this.active && !this.cleanupError && this.sessions === 0 && this.operations === 0 && this.pending.size === 0; }

  assertActive(): void {
    if (!this.active) throw new LLMDisposedError();
    if (this.cleanupError) throw this.cleanupError;
  }

  /** A failed release fences reuse, but must not prevent final close and drain. */
  setCleanupError(error: Error | null): void {
    this.cleanupError = error;
    this.onActivity();
  }

  acquireSession(): () => void { return this.acquire("sessions"); }
  acquireOperation(): () => void { return this.acquire("operations"); }

  private acquire(kind: "sessions" | "operations"): () => void {
    this.assertActive();
    this[kind]++;
    this.onActivity();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this[kind]--;
      this.onActivity();
    };
  }

  track<T>(work: Promise<T>): Promise<T> {
    this.pending.add(work);
    const settled = () => { this.pending.delete(work); this.onActivity(); };
    // Do not leave a rejecting `finally` promise detached from its consumer.
    work.then(settled, settled);
    return work;
  }

  beginClose(): void {
    if (!this.active) return;
    this.state = "closing";
    this.controller.abort(new LLMDisposedError());
    this.onActivity();
  }

  async drain(): Promise<void> {
    while (this.pending.size > 0) await Promise.allSettled([...this.pending]);
  }

  finishClose(): void { this.state = "closed"; }
}
