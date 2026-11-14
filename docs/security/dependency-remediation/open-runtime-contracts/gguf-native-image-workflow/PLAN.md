# Isolated GGUF image workflow proposal

1. Authenticate the frozen launcher/probe/old recipe and retain their hashes. Do not edit or execute the launcher, original probe, parent-owned repair or any live checkout.
2. Propose only `.github/workflows/gguf-native-image-qualification.yml`: pre-merge PR/candidate-push events, exact AST-bound full uv default, source/event/checkout receipts, host+daemon+image amd64 gates and offline bounded non-root probe execution.
3. In the inline stdlib controller, retain command logs in CI, preserve primary failures, verify owned-CID identity before deletion and successful all-container readback before claiming absence. State hard-kill limitations.
4. Extract and syntax-check the actual generated Bash/Python; run the supplied checksum-bound official actionlint binary with owned HOME/cache/temp roots.
5. Exercise the generated control flow with owned subprocess Docker-CLI fixtures, including malformed selections, architecture/digest refusals, failed/timeout probes and cleanup failures. All simulated Docker/host/probe responses are explicitly synthetic, never image evidence.
6. Deliver the new file, new-file patch, hash manifest, raw validation/control-flow results and before/after unchanged-input proof. Parent owns independent review, qualifier handoff, publication and actual Linux amd64 CI/readback.
