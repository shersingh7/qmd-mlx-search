# W1 revision design (before implementation)

One cohesive lifetime owner, per invocation: a short, workflow-generated stdlib Python helper starts as a new session/group leader. It owns the real CLI child in that same group. It may reap that CLI, but stays alive as the group's anchor until the controller's final SIGKILL. The controller never calls wait/poll/communicate on its anchor before that last group signal. Even unexpected anchor exit leaves its PID kernel-reserved as an unreaped child until then. No post-reap signal, sampled/reconstructed PGID, external recovery lease or third source-checkout file is used.

The helper receives an absolute monotonic deadline formed by the controller before spawning it, using the same Python executable on the same host (Python >=3.10's system-wide monotonic clock). Startup/transport/command-creation time consumes, rather than resets, the budget. It watches the controller-only pipe for persistent EOF and the same deadline; on either it kills its own still-owned group. The controller also enforces that deadline. KILL, rather than TERM followed by a leader-dependent grace period, is the one final group signal. This deliberately gives up CLI TERM grace, not the bounded Docker ownership/readback cleanup. Deferred, first-signal-wins handlers avoid asynchronous exceptions during spawn/registration; repeat TERM/INT cannot tear down cleanup.

| State/event | Final group signal ownership | Result/evidence |
|---|---|---|
| CLI normal or early exit (including lingering descendants) | helper remains live, unreaped | raw CLI status sent through private result pipe; controller KILLs entire group before reaping helper |
| Timeout | controller/helper use original deadline; helper identity remains owned | public 124; final group KILL; bounded anchor wait; Docker cleanup still attempted |
| TERM/INT, including during spawn/repeated delivery | nonthrowing handler records first signal; owned anchor registration completes | public 128+signal; same KILL-before-reap and bounded cleanup |
| Helper launch/transport/final signal/reap failure | no guessed fallback identities and no post-reap signals | nonqualification; primary failure retained; signal delivery and anchor reap separately recorded |
| Controller exits abruptly | pipe writer is not inherited by CLI; EOF remains observable by helper | runnable helper KILLs its own group; no guarantee if sole surviving owner is itself unscheduled/killed |
| Required receipt write fails | storage resolved before final log | primary nonzero retained, otherwise final 1; not-qualified log with persistence error; required receipt is not deleted to conceal the failure |

`process_exit` remains the direct Popen child's raw OS status (now the anchor); separate `command_exit` retains the CLI's raw OS status when observed. A timeout/cancellation can leave CLI status unknown: don't fabricate it from delivered KILL. Group signal delivery plus anchor reap is not a certification that every non-child descendant was reaped, nor a proof against a descendant that deliberately escapes the process group. The receipt will state these boundaries. A bounded wait that cannot reap the anchor is truthful incomplete cleanup, never qualification.

S1 is a narrow AST extension for `MatchAs.name`, `MatchStar.name`, `MatchMapping.rest`, with `ast.walk` covering nested/OR patterns and harmless unrelated controls. It is direct syntactic binding validation, not proof of arbitrary dynamic Python immutability. The current frozen launcher and derived image pin stay unchanged.

S2 computes candidate final state, attempts required persistence, catches storage errors, then emits the sole final log. Persistent/logged success must agree. Negative raw CLI signal statuses use a public 128+signal representation while retaining raw status separately.

## Evidence/oracle boundaries

The original 74 suite and 34-characterization runner are retained byte-identically in `original/`, and fresh baseline runs precede edits. Parent explicitly authorized a separate S3 positive oracle because the original runner requires the killed descendant's normal-completion side-effect file. Its original RED remains RED evidence. The positive oracle must retain ready/group ownership data, take return-time state before any fixture cleanup, observe through the original two-second worker window (plus original three-second observation allowance), require no late effect, establish owned termination/reap or a truthful failure, and keep an unrelated sentinel alive. Other 74/34 assertions are preserved, except source-hash/path and actual command-boundary transport changes, each with a fail-closed/proven-hit seam.

No real Docker/daemon, network, registry, application launcher/probe/compiler/converter/model/GPU/cloud or CI execution. No acceptance claimed; parent commissions independent SPEC then QUALITY.
