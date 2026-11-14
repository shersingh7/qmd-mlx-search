# MLX transport deadlines — scoped local acceptance

**Accepted source scope:** `src/mlx.ts` and `test/audit-transport-review.test.ts`. Independent SPEC review passed before QUALITY review, with no findings. This does **not** approve the separate recovery optimization, installed package, remote CI, merge, or whole release.

## Repairs

- Classify an expired outer embedding operation by its absolute monotonic deadline, even when synchronous work delays timeout callbacks. A live-budget malformed descriptor remains a protocol error.
- Publish negotiated metadata only while an actual registered, non-aborted consumer still owns a valid deadline. A surviving consumer keeps its rights; an expired sole consumer cannot populate the cache.
- Release temporary deadline associations and waiter identities on success, cancellation, timeout and rejection. Stale completion cannot overwrite or clear a replacement negotiation.
- Anchor the shared-request regression to actual descriptor dispatch, not merely method entry. Preserve the real 80 ms deadline and 100 ms barrier, cancellation shapes, strict serving identity and complete fixture teardown.

## Verification

| Surface | Actual result |
|---|---|
| Independent source review | SPEC PASS → QUALITY PASS; original-source, boundary, admission, renewed-budget, cancellation and leak mutants retained |
| Fresh parent Node replay | **117 passed**, three files, including 30 independent held-outs |
| Fresh parent native-Bun replay | **75 passed, 20 filtered**, actual Bun launcher and workers; not the full native Vitest suite |
| Real-checkout canonical build | `pnpm run build`: exit 0, actual Node 22.22.3, active compiler guard |
| Real-checkout canonical tests | **1,554 passed, 66 skipped, 54 files**; `pnpm run test` with JSON reporting |
| Source/build binding | All **300** tracked runtime/qualification inputs unchanged during the combined run; five built artifact hashes equal the independently reviewed snapshot |

The first real-checkout test attempt failed (**1,414 passed, 124 failed, 82 skipped**) because the parent's added read-guard incorrectly derived dependency/source authority from each subprocess's working directory. CLI fixtures deliberately change directories. Five causal controls demonstrated the defect; the guard was pinned to the declared repository, without changing product assertions or granting additional filesystem access. All **172** cases in the four affected files then passed, followed by the full canonical run above. The original failed run is retained.

Three optional ancestor `package.json` probes were blocked before contents were read during the final suite. They were not silently permitted or reported as zero refusals. This is **not a zero-refusal workspace-layout qualification**. Instrumented JavaScript filesystem checks are not an OS sandbox and do not fully mediate Python or native Bun internals.

## Durable evidence

- [Acceptance summary](acceptance-summary.json): exact preimages/postimages, actual commands/counts, limits and storage hashes.
- [SPEC review](SPEC-REVIEW.md) and [QUALITY review](QUALITY-REVIEW.md): the full independent source verdicts.
- `evidence-index.json.gz`: maps **7,152 logical files** to their SHA-256 content objects, original modes and historical source paths.
- `evidence-objects.tar.xz`: stores **1,235 unique objects**. Every original member and every stored object was reopened and hash-verified. Deduplication preserves the original proposals, superseded witness, negative controls, review, parent replays, source snapshots and failed/successful integration attempts without committing repeated snapshots many times.

To inspect one indexed file without reconstructing the historical workspace:

```python
import gzip, hashlib, json, tarfile

with gzip.open("evidence-index.json.gz", "rt") as stream:
    index = json.load(stream)
row = index["groups"]["independent-review"]["files"]["SPEC-REVIEW.md"]
with tarfile.open(index["objectArchive"], "r:xz") as archive:
    data = archive.extractfile("objects/" + row["sha256"]).read()
assert hashlib.sha256(data).hexdigest() == row["sha256"]
print(data.decode())
```

For reconstruction, use only the relative group/member names inside a newly owned directory; restore modes from the index. **Do not write to the historical absolute `source` paths.** Dependency trees and disposable caches are intentionally excluded. Historical absolute commands require explicit relocation before replay.

## Remaining gates

A new installed-package qualification, exact published-head CI/readback, broader recovery/native-image/Nix gates, and merge/default-branch alert readback remain separate. The native-Bun gated-child ownership failure captured during the **unaccepted transition-fixture optimization** remains open; these Node passes do not erase it. No new real-model, GPU-throughput, large-model or private-corpus claim is made by this follow-up.
