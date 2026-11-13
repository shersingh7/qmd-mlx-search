# Reproducing the local audit gates

Run from a fresh checkout with the pinned Node/Python development dependencies and a managed `TMPDIR`. These commands do not deploy a daemon or migrate a live index. `scripts/run-offline.py` creates isolated HOME/cache/config state and disables model downloads.

```sh
python3 scripts/run-offline.py pnpm run build
python3 scripts/run-offline.py pnpm run test
python3 scripts/run-offline.py .venv/bin/python -m pytest test/python -q -m 'not real_model'
python3 scripts/run-offline.py bash test/launcher-detection.test.sh
python3 scripts/run-offline.py bash test/smoke-install.sh --runtime node
python3 scripts/run-offline.py bash test/smoke-install.sh --runtime bun
```

The test script includes `test/`; appending filenames does not narrow that selection. Use `node node_modules/vitest/vitest.mjs run path/to/test.test.ts` for a focused run. Default and CI worker limits are two; the real process deadline assertions remain unchanged.

For hardware checks, set `CACHED_MINILM` to an existing absolute local snapshot path. Supply fresh output names—do not overwrite old evidence. Reference comparison also needs the Hugging Face/PyTorch CPU dependencies; missing dependencies are a blocker, not permission to substitute generated vectors.

```sh
.venv/bin/python scripts/bench_apple_silicon.py --repo . \
  --model "$CACHED_MINILM" --dtype float16 --bulk-tokens 4096 \
  --label local-repeat --output /absolute/managed/path/local-repeat.json
.venv/bin/python scripts/check_embedding_reference.py \
  --model "$CACHED_MINILM" --vectors /absolute/managed/path/local-repeat.npz \
  --pooling mean --output /absolute/managed/path/local-repeat-reference.json
.venv/bin/python scripts/verify_local_stack.py \
  --model "$CACHED_MINILM" --dtype float16 \
  --output /absolute/managed/path/local-stack.json
```

Use BGE-small with CLS pooling for its independent reference. The benchmark/runtime artifacts record exact source identity, effective configuration, limits and raw measurements. Public CPU qualification fixtures are orchestration tests, not model quality or GPU measurements.
