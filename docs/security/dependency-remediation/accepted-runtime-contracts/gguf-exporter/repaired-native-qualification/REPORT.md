# Fresh repaired GGUF exporter: native CPU qualification

**PASS within scope; no production integration or acceptance seal.**

- Actual composed `convert_gguf.main --no-upload` freshly produced F16, Q4_K_M, Q6_K and Q8_0 after two real nonzero SFT/GRPO merges.
- Actual composed `train.export_gguf` freshly produced Q4_K_M, Q6_K and Q8_0 with the same unmodified stock tokenizer.
- 22 real export commands returned zero: two new exact-pin Git source checkouts, two initial-lock CMake CPU-only `-j 2` builds, two converters, six quantizers, plus source/API/configure preflights.
- Installed GGUF 0.19.0 and the maintained reader retained from the actual new pinned checkout independently inspected all seven artifacts: `qwen35`, 25 tensors, exact full 248077-token vocabulary, 247587 merges, chat template, quantization tensor types, aligned nonoverlapping in-file bounds, complete artifact and tensor-payload hashes.
- All 25 F16 output tensors exactly match the independent checkpoint/arithmetic oracle. All 27 base tensors are accounted for, including V-head permutations and concatenations. The adapted V projection exactly matches both sequential nonzero merges and differs from the base.

## Source, headers, and initial runtime

- Exact repaired helper: `5c36cc3f73ec03cd7c89f7f8f7083987824f4b8bae19ee720a02177fa03ea028`.
- Own allowlisted mirror contains 36 public metadata inputs with the seven reviewed source files overlaid (40 resulting files). Three entrypoints replace only their PEP 723 block with the new metadata-preview block; every other reviewed byte is unchanged. The other four reviewed files match exactly.
- `source-header-environment-manifest.json` records original/reviewed/new whole-file hashes, old/new header hashes and unchanged body hashes. Actual imported paths/hashes are in each export receipt.
- Initial cold-installed Python 3.12.13, Torch 2.15.0, Transformers 5.22.2, Accelerate 1.15.0+qmd.2, GGUF 0.19.0, sentencepiece 0.2.3, protobuf 7.36.2 and CMake 4.4.4. No pip module; no installs, version changes, old CMake fallback or source repairs.
- Public source pin: `5193ef1d4f7c95b006ab3b9f857ed27be9a63235`. Both actual call-owned checkouts verify that commit. Exact source/module and built-quantizer hashes are recorded before normal call cleanup.

## New artifacts

| Entrypoint | Format | Bytes | Tensor types | SHA-256 |
|---|---|---:|---|---|
| standalone | f16 | 140593376 | {"F16": 14, "F32": 11} | `4aaf136b76ca60b91c3107db953b435a58e65c6a6ebe55ffc1c417866b0505ff` |
| standalone | q4_k_m | 63829280 | {"F32": 11, "Q4_K": 11, "Q6_K": 3} | `dba3d739e11e483e783e7baf2f1e643df6658deacff6b4fa2ac84666e54bf562` |
| standalone | q6_k | 64125504 | {"F32": 11, "Q6_K": 14} | `659a7ea01f8843c3c1abee6be703d6876a5f34e9f0e17871f94e084faeab3447` |
| standalone | q8_0 | 79824192 | {"F32": 11, "Q8_0": 14} | `e260da836b8e5df34bd662186f0807788e57a518972b848c2ddf4cc6f33c38d3` |
| native | q4_k_m | 63829280 | {"F32": 11, "Q4_K": 11, "Q6_K": 3} | `32a4f31df10fe1ccbf1bb5df5742ebdd5255f836ab1c2a80dce7b9ab1a5c40f7` |
| native | q6_k | 64125504 | {"F32": 11, "Q6_K": 14} | `8f485b8f4fce4b82361a1787c96c3dfb0977b991a928d415257c4460b09705ce` |
| native | q8_0 | 79824192 | {"F32": 11, "Q8_0": 14} | `adb5d3e358ba75c6132bd5738351e9c1738eef9cfb5cbcfaf580998c79be7464` |

Every absolute new artifact path is listed in `qualification-manifest.json`. Output hashes intentionally reproduce the prior deterministic random-fixture bytes; new paths, real new process receipts and fresh source/build identities establish freshness. Old GGUFs, checkouts and binaries were not copied or reused.

## CPU, resource, and preservation evidence

- Actual model parameters/buffers, 27 real checkpoint materializations per load, adapter reads/merges, and finite forward including maximum token ID 248076 are CPU. Test-only real-dispatch observation found only CPU/meta tensors and no accelerator operation. Meta is lazy model construction, not accelerator materialization.
- Fresh per-child allowlisted environments isolate HOME, Hermes knowledge/config, HF/XDG/cache/TMPDIR; HF offline and telemetry-disabled. No credentials or inherited Python/Git overrides. Non-Git child networking is OS-sandbox denied.
- Sampled owned-process-tree peaks (bytes): `{"fixture": 969637888, "native": 1686208512, "pinned": 1244708864, "standalone": 1823211520, "verify2": 1769488384}`. All stayed below the 2 GiB sampled guard; no unowned process arguments were read.
- Full environment stream-hash snapshots match the parent baseline and each other: 30255 regular files, 2537 directories, 3 symlinks (targets recorded, not followed); 1075927846 regular bytes. Canonical inventory digest: `dcf43d083786a5fcdd5276bbf01818a085bcc58832752b195f1f35ba657a4df8`.
- All 71 explicitly protected source/metadata/receipt files and the entire original qualification tree (109 files, 23 directories) are unchanged. This includes retained old negative evidence.

## Evidence and issues

- `*-launch.json`: actual runner argv, complete cleared env, exit, timeout, own-PID resource evidence. `standalone-receipt.json` and `native-receipt.json`: every real Git/CMake/converter/quantizer argv/env/exit plus retained stdout/stderr paths, source/build identities, actual imports, merges and reader calls.
- `verify2-receipt.json`: installed-reader metadata, full bounds/payload hashes, independent all-F16 numeric values. `pinned-receipt.json`: all seven second-reader readbacks from the actual new checkout package.
- `environment-*.json`, `preserved-originals-{before,after}.json` and `final-integrity-comparison.json`: complete before/after comparisons and final real Python/CMake/pip-absence probes.
- No new product or harness failure was encountered. No production source/environment repair was made. Prior verifier/controller negatives are neither overwritten nor counted as passes.

## Limits

- Random initialized CPU text-only Qwen3.5, two hybrid layers, hidden size 256; deterministic nontrained nonzero rank-2 SFT/GRPO adapters, not pretrained weights or trained model quality.
- No MTP-bearing native model conversion or model-quality qualification; target-only checkpoint correctly uses --no-mtp. No invented MTP tensors were added.
- macOS ARM64 native CPU only; not Linux/container-image/cloud, CUDA, MPS, inference quality, or production-scale resource qualification.
- The actual standalone caller requests trust_remote_code=True; the local hash-bound fixture has no Python code or auto_map and resolves to installed stock Transformers classes. Native/fixture stock loads explicitly use False.
- Torch availability overrides and real-dispatch/CPU checkpoint observers exist only in the qualification process, preserving cache APIs; production/installed package code is unchanged.
- Process-tree memory is sampled, not an OS hard limit. Converter/build/quantizer networking is OS-sandbox denied; Python network is refused. Only the actual exact-pin anonymous upstream Git fetch is allowed.
- This is evidence for the composed source/header binding in the already cold-installed initial project environment, not a new environment install, live metadata-graph acceptance, production integration or acceptance seal. Parent owns those gates.
