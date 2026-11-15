# Execution-lease repair — pre-edit design and ownership

Entry: nine live hashes matched the frozen writer receipt, plus read-only process_anchor.py/__init__.py. `entry/` is the immutable attribution baseline; no other writer's changes will be reverted. Original independent reproducer is copied byte-for-byte, and its relocated fresh RED executed exit 1 with actual-user-work after queued EOF. No production edit preceded this design.

## Mechanism

* S becomes one bounded ASCII frame `S<remaining integer milliseconds>\n`. Python records its monotonic anchor immediately before PREPARED. The caller computes remaining time from its ORIGINAL deadline only after receiving PREPARED. Python's deadline is anchor + remaining, capped by its original timeout. Anchoring earlier than the caller's subtraction conservatively charges transport/startup; there is no assumption of shared Node/Bun/Python clock epochs. The fork inherits that Python deadline. No reset after E or worker creation.
* E still acknowledges READY; it is NOT a durable execution capability. The worker retains a CLOEXEC duplicate of the caller's control read descriptor. Only after consuming the private E gate, establishing canonical containment, and validating the inherited deadline AND the still-live control channel does it admit exec. EOF persists across reads; an unexpected byte/cancellation/error fails closed. The child never consumes S/E from the caller stream before the supervisor has forwarded E.
* A child waiting on the private gate independently expires at the inherited deadline. Rejection must also wake a stalled/stopped owner: it blocks TERM/INT, kernel-joins the supervisor's existing group with setpgid(0, sessionId), and only AFTER successful same-session membership sends TERM then CONT to its OWN now-pinned group, then exits. It cannot create a group with another PID. Failure to join confers NO signal authority; no numerical PID/group fallback. Membership stays pinned through both signals and exit. This is the same kernel authentication principle as the existing joining anchors, without modifying their implementation or adding a watchdog process.
* The supervisor independently enforces the same absolute local deadline and parent EOF during execution and always cleans up retained direct-child/WNOWAIT anchors. All terminal control read/write errors become cancellation, not unwinds around cleanup. The existing runtime-owned TERM/CONT path is retained, including Bun's numeric signal through its native owned handle.

## State/ownership table

| State / transition | Execution authority | Cancellation owner / proof |
|---|---|---|
| Bootstrap, PREPARED, no S | none | Node/Bun exact native supervisor handle may SIGKILL; no worker exists |
| Valid bounded S; before/inside fork registration | gated creation only | Existing interrupting TERM/CONT; Python kernel waitid recovers an interrupted return registration; inherited deadline protects any later child |
| Worker created, READY not yet sent / E not consumed | none | Supervisor direct-child/WNOWAIT anchor; child's bounded gate timer independently refuses and authenticates owner-group wakeup |
| E consumed or forwarded but delayed | none until child admission | Same independent child lease and persistent parent EOF; refusal wakes interrupted/stopped supervisor without parent exit hooks |
| Child receives E / containment preparation | none | Same inherited lease and live caller channel, checked AFTER preparation; supervisor already independently monitors deadline/EOF |
| Last trusted child gate | execution admission linearizes at successful final live-channel/monotonic validation | Pending EOF, cancellation/data/error or expired lease denies; after admission, supervisor owns revocation/termination through retained direct child |
| Admitted child delayed before exec | admitted, revocable work | Scheduled supervisor detects original deadline/EOF and kills exact retained child; no fresh timeout is issued |
| Parent abrupt SIGKILL (no exit hook) | no new stale admission | Kernel EOF reaches child gate and supervisor; gated child can TERM/CONT its kernel-authenticated owner group |
| Cleanup / channels broken | no new admission | Supervisor retains WNOWAIT/group anchors until last signal/reap; proof transport failure cannot escape cleanup |

## Failure model and linearization limit

There is no POSIX primitive atomically combining pipe liveness, a deadline and execve. Admission is a revocable decision, not a promise that every user instruction precedes any later parent death. Parent death after the final child validation is concurrent with already-admitted execution; the independently scheduled supervisor owns cancellation. Pausing the supervisor across E is protected by the child, and pausing the child across/after its last gate is protected by the supervisor. Pausing ALL trusted execution/cancellation agents at once prevents hard-real-time detection/cleanup; no such guarantee is made. A permanently kernel-stopped supervisor after the Python gate has been replaced by arbitrary exec likewise cannot autonomously schedule itself; the existing live caller uses its owned CONT handle. This does NOT exclude actual abrupt parent death in the reproduced stale-E domain, which will be tested without exit-hook assistance and before fixture cleanup.

## Planned tests (before production edits)

Permanent focused regression: valid execution (separate 5s readiness control), EOF without E, raw E consumed + 1300ms hold with EOF, forwarding, child grant consumption, before and after final child validation, real SIGSTOP/resume, and real Node/Bun alive/graceful/hard-exit controls. Record actual shell side effects, supervisor/worker/sentinel state, helper receipts when present, original 200ms/800ms/1800ms/<1500ms live-caller bounds, and every failure-only fixture kill. Adapt old raw S transport only; preserve original evidence bytes and semantic assertions. Retain all original 70 focused tests, native FD reuse/GC, no-fork, preflight and honest forced-close contracts. No cold completion within 200ms requirement is inferred from the 5s positives. Parent still owns independent review, type-only rebase, canonical suite/build and the separate unresolved native Mac CI bootstrap/open-stream gate.
