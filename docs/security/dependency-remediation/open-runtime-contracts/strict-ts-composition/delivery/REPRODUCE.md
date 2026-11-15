# Reproduce from the owned composition root

Do not run in the live checkout. Commands preserve existing source/config; expected reds remain reds. Use fresh labels and optionally fresh JSON output paths for additional runs. Focused hosts use at most two workers; run listed workloads serially.

## types-node-final

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run types-rerun-types-node-final mirror /Users/shersingh/.hermes/node/bin/node node_modules/typescript/bin/tsc -p tsconfig.json --noEmit --pretty false
```

## types-bun-final

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run types-rerun-types-bun-final mirror /Users/shersingh/.bun/bin/bun --preload ../probes/compiler-read-guard.cjs node_modules/typescript/bin/tsc -p tsconfig.json --noEmit --pretty false
```

## build-node-checkpoint

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run build-rerun-build-node-checkpoint mirror /Users/shersingh/.hermes/node/bin/node scripts/build.mjs
```

## node-focused-final

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run rerun-node-focused-final mirror /Users/shersingh/.hermes/node/bin/node node_modules/vitest/vitest.mjs run --config ../mirror-isolated.config.mjs --no-cache --maxWorkers=2 --reporter=dot --reporter=json --outputFile.json=../evidence/rerun-node-focused-final-results.json test/audit-llm-cleanup-review.test.ts test/audit-llm-deadline-review.test.ts test/audit-llm-lifecycle.test.ts test/bench-rerank-entrypoint.test.ts test/bench-rerank-lifecycle.test.ts test/eval-bm25.test.ts test/eval.test.ts test/llm.test.ts test/multi-collection-filter.test.ts test/phase5-recovery.test.ts test/rrf-trace.test.ts test/store.helpers.unit.test.ts test/zod-import.test.ts
```

## bun-focused-final

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run rerun-bun-focused-final mirror /Users/shersingh/.bun/bin/bun --bun node_modules/vitest/vitest.mjs run --config ../mirror-isolated.config.mjs --no-cache --maxWorkers=2 --reporter=dot --reporter=json --outputFile.json=../evidence/rerun-bun-focused-final-results.json test/audit-llm-cleanup-review.test.ts test/audit-llm-deadline-review.test.ts test/audit-llm-lifecycle.test.ts test/bench-rerank-entrypoint.test.ts test/bench-rerank-lifecycle.test.ts test/eval-bm25.test.ts test/eval.test.ts test/llm.test.ts test/multi-collection-filter.test.ts test/phase5-recovery.test.ts test/rrf-trace.test.ts test/store.helpers.unit.test.ts test/zod-import.test.ts
```

## node-sdk

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run rerun-node-sdk sdk-probe /Users/shersingh/.hermes/node/bin/node ../mirror/node_modules/vitest/vitest.mjs run --config ../sdk-isolated.config.mjs --no-cache --maxWorkers=1 --reporter=dot --reporter=json --outputFile.json=../evidence/rerun-node-sdk-results.json test/sdk-inlined-zod.test.ts test/zod-prefix-scope.test.ts
```

## bun-sdk

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run rerun-bun-sdk sdk-probe /Users/shersingh/.bun/bin/bun --bun ../mirror/node_modules/vitest/vitest.mjs run --config ../sdk-isolated.config.mjs --no-cache --maxWorkers=1 --reporter=dot --reporter=json --outputFile.json=../evidence/rerun-bun-sdk-results.json test/sdk-inlined-zod.test.ts test/zod-prefix-scope.test.ts
```

## node-native-bun-wrappers

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run rerun-node-native-bun-wrappers mirror /Users/shersingh/.hermes/node/bin/node node_modules/vitest/vitest.mjs run --config ../mirror-isolated.config.mjs --no-cache --maxWorkers=1 --reporter=dot --reporter=json --outputFile.json=../evidence/rerun-node-native-bun-wrappers-results.json test/audit-native-bun-recovery.test.ts
```

## node-held-sparse

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run rerun-node-held-sparse held /Users/shersingh/.hermes/node/bin/node ../mirror/node_modules/vitest/vitest.mjs run --config ../held-isolated.config.mjs --no-cache --maxWorkers=1 --reporter=dot --reporter=json --outputFile.json=../evidence/rerun-node-held-sparse-results.json '--testNamePattern=refuses missing query slots' phase5-recovery.test.ts
```

## patch-replay-check

```sh
/usr/bin/env -i /bin/sh /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/run rerun-patch-replay-check baseline /usr/bin/git apply --check /Users/shersingh/.hermes/cache/scratch/qmd-types-compose-nd8v4x2p/parent-integration.partial.patch
```
