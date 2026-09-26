# KVarN: 4090 (sm_89) machine handoff

This branch (`wip/kvarn-cache`) is CPU-complete and needs a CUDA box for:
Triton parity acceptance, micro-KLD numbers, and any perf work. Everything
below was verified up to the hardware boundary on a GPU-less box.

## Clone

```sh
git clone https://github.com/aand18/exllamav3
cd exllamav3
git checkout wip/kvarn-cache
```

## Prerequisites

- Python 3.10–3.13, ~15 GB free (CUDA torch ~2 GB + build tree).
- CUDA toolkit 12.4+ (13.x works; cu130 torch + 13.4 toolkit built clean).
- Windows: MSVC 14.x + Windows 10/11 SDK ("Desktop development with C++",
  without the optional clang component — nvcc only accepts MSVC anyway).
- Target arch is sm_89 (RTX 4090) only; keep every other arch out for speed.

## Build the extension

Python files need no build; only `exllamav3/exllamav3_ext/` does. Build in
a venv, never in system Python:

```sh
python -m venv exl-cuda
exl-cuda/Scripts/activate            # or source exl-cuda/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install pydantic tokenizers safetensors numpy rich typing_extensions \
    pillow pyyaml marisa_trie llguidance pytest
```

Windows trap we hit: `VsDevCmd.bat` exports an environment that does NOT
reliably propagate to child shells (ninja/`cl`/`nvcc` then fail with
`CreateProcess` / `Cannot find compiler 'cl.exe'`). Bypass it — set the
toolchain paths manually before building:

```bat
set MSVC=C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.51.36231
set KIT=C:/Program Files (x86)/Windows Kits/10
set PATH=%MSVC%/bin/Hostx64/x64;%KIT%/bin/10.0.26100.0/x64;<cuda-bin>;<venv-Scripts>;%PATH%
set INCLUDE=%MSVC%/include;%KIT%/Include/10.0.26100.0/ucrt;%KIT%/Include/10.0.26100.0/um;%KIT%/Include/10.0.26100.0/shared;%KIT%/Include/10.0.26100.0/winrt;%KIT%/Include/10.0.26100.0/cppwinrt
set LIB=%MSVC%/lib/x64;%KIT%/Lib/10.0.26100.0/ucrt/x64;%KIT%/Lib/10.0.26100.0/um/x64
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=4
python setup.py build_ext --inplace
```

Notes: `TORCH_CUDA_ARCH_LIST` is space-separated (`8.6 9.0+PTX` style —
semicolons break torch's parser). `MAX_JOBS=4` avoids nvcc OOM on 16 GB
RAM; scale to the box. Adjust the MSVC/SDK version numbers to whatever is
installed. Linux: same torch install, then
`TORCH_CUDA_ARCH_LIST=8.9 python setup.py build_ext --inplace`
(or `pip install --no-build-isolation .`).

## Validate, in order

1. CPU-style suite (also runs on CUDA box, torch path is the default):
   `python -m pytest tests/test_kvarn_cpu.py tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py tests/test_kvarn_triton.py -q`
2. Triton acceptance (the kernel has NEVER launched — this is its first run):
   `EXL3_KVARN_TRITON=1 EXL3_KVARN_TRITON_PARITY=1 python -m pytest tests/test_kvarn_triton.py tests/test_kvarn_cpu.py -q`
   Parity mode runs torch + Triton side by side and asserts equality.
   Do not trust `EXL3_KVARN_TRITON=1` without parity passing first.
3. Micro-KLD (needs ~8 GB download, one-time):
   ```sh
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='turboderp/Qwen3.8-27B-exl3', revision='SC_1.40bpw_H3_V3', local_dir='models/Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3', allow_patterns=['model.safetensors','*.json','*.txt','merges.txt','vocab.json','*.jinja'])"
   python eval/kvarn_microkld.py -m models/Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3 -cq kvarn4 -ntok 200
   python eval/kvarn_microkld.py -m models/Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3 -cq kvarn5,kvarn4 -ntok 200
   ```
   Note: that checkpoint is Qwen3.5-dense (`head_dim` 256, 16 full-attention
   layers) — good KVarN coverage, but it exercises neither QSA nor MoE.

## Report back

Parity pass/fail (+ assertion text on failure), micro-KLD median/mean/max +
same-top % per preset, prefill tok/s for fp16 vs kvarn4 vs kvarn5,kvarn4,
and GPU model. Post results to the branch's draft PR (#2).

## Validation results (RTX 4090, sm_89, Windows native)

Harness: `eval/kvarn_microkld.py` (336+ past + 64 scored continuation
tokens at 400 tok; past + 64 scored at longer ctx). All numbers plain
decimals; KLD = per-position next-token KLD vs fp16 cache. This table is
the regression baseline — compare future runs (and BeeLlama reference
numbers) against it. `same-top` was 100.00% on every run at every length.

Status of the checklist above: (2) Triton acceptance PASSED on 4090
(sparse-path bug found and fixed along the way, `247a393`); kernels now
run under `EXL3_KVARN_TRITON=1` with `PARITY=1` asserting bit-exactness.
(3) Micro-KLD done at 400–32768 tokens (procedure now uses `-mcl`,
`--max_tokens`, `--chunk`, `--decode`; see `--help`).

### Qwen3.8-27B dense 1.40bpw (`SC_1.40bpw_H3_V3`, Qwen3_5, hd 256)

| ctx | preset | median | mean | max | p99 | p99.9 | fp16 pre | kvarn pre |
|-----|--------|--------|------|-----|-----|-------|----------|-----------|
| 8192 | kvarn4 | 0.000001 | 0.000013 | 0.000261 | 0.000179 | 0.000253 | 4.7s | 10.0s |
| 8192 | kvarn5,kvarn4 | 0.000001 | 0.000011 | 0.000144 | 0.000127 | 0.000142 | 4.7s | 10.0s |
| 16384 | kvarn4 | 0.000001 | 0.000010 | 0.000183 | 0.000168 | 0.000182 | 6.6s | 8.8s |
| 16384 | kvarn5,kvarn4 | 0.000001 | 0.000022 | 0.000788 | 0.000450 | 0.000754 | 6.5s | 8.8s |
| 32768 | kvarn4 | 0.000001 | 0.000019 | 0.000278 | 0.000230 | 0.000273 | 13.7s | 17.6s |
| 32768 | kvarn5,kvarn4 | 0.000001 | 0.000023 | 0.000356 | 0.000319 | 0.000352 | 13.6s | 17.6s |

Prefill history on this model @8192, kvarn4 (kvarn5,kvarn4 in
parentheses; same quality throughout — every step reproduced the KLD
digits exactly, same-top 100.00%):

| step | change (commit) | fp16 pre | kvarn pre | ratio |
|------|-----------------|----------|-----------|-------|
| 0 | baseline: per-group Python loops (~30 launches/group, ~160/group seal) | 4.7s | 134.6s (130.9s) | 28.6x |
| 1 | batched dequant across groups (`2345080`): 1 unpack + 1 dequant + 1 WHT per layer | 4.7s | 96.2s (95.8s) | 20.5x |
| 2 | batched seals (`21be28b`): 1 Sinkhorn+quant+pack per K/V for all sealable groups | 4.7s | 10.7s (10.8s) | 2.3x |
| 3 | incremental image (`839360c`): dirty-group refresh instead of O(n^2) rematerialization | 4.7s | 10.0s (10.0s) | 2.1x |
| 4 | chunk 512 -> 2048 (run flag, no code): fewer forwards amortize per-call fixed costs | 3.4s | 5.0s | 1.5x |
| 5 | chunk 4096 | 3.2s | 4.5s | 1.4x |
| 6 | chunk 8192 (single forward) | 3.2s | 4.3s (4.3s) | 1.3x |
| 7 | `EXL3_KVARN_TRITON=1` (fused dequant kernel, `e3294df`, parity-proven) | 3.2s | 4.2s | 1.3x |

Total: 134.6s -> 4.2s (32x). Step 7 confirms dequant is off the
critical path; the remaining ~1.1s is store-side row-WHT + Sinkhorn
seals + Python syncs per layer. Named next step: fuse the inverse WHT
into the Triton kernel (XOR-butterfly in-register, no extra traffic).
CPU suite time also improved with step 2 (8.8s -> 4.4s).

Note on BeeLlama comparison: Bee runs CPU/ggml with a fused
`flash_attn_ext_kvarn` kernel (online dequant, no fp16 materialization)
and single-threaded scalar store; there is no shared benchmark harness,
so timing is not directly comparable. The architectural lessons taken
are recorded above (fuse dequant into attention; batch, don't
thread); this table is the timing baseline for our own regressions.

Decode @8192, 1.40bpw 27B (64 greedy steps, warm inductor cache):
fp16 86.0-86.6 tok/s vs kvarn 7.8-19.5 tok/s depending on the step below
(current: 19.5 tok/s TRITON=1, fp16 86.3 same run).
Per-token profile (16 cached layers): store ~3.7ms/layer, serve
~1.1ms/layer, attention ~0.7ms/layer (fp16 step total 11.6ms).

Generation optimization history (all KLD-identical, same-top 100%):
- 8.9 tok/s baseline (clone-per-call design).
- In-place overlay + stash restore tried, measured 7.8-7.9 (clone was
  bandwidth-cheap; bookkeeping added latency) and reverted (`0158aa1`).
- Single-row store fast path (`9feae92`), `_touch_batch` vectorized
  (`36d420a`, was O(ctx) syncs/token), K+V stacked store WHT
  (`d4998a6`), exact-evict on 128-boundaries (`737976e`,
  provably final-state-equivalent): 11.1-11.3 tok/s (+27%).
- Ping-pong WHT (`294f238`) and Triton row-WHT kernel shared by
  dequant+store (`b2997ca`, parity-proven): ~zero end-to-end (cost is
  launch/sync count, not math). Now 12.1 tok/s (+36%).
- Fused exact-overlay Triton kernel (`44e0b39`, 1 launch zero syncs,
  `EXL3_KVARN_TRITON=1`, parity-proven): 13.0 tok/s.
- `_touch_batch` batch-vectorized (`9a8427a`, 3 syncs/entry -> 1
  whole-batch validation sync): 19.0 tok/s (+114% over baseline,
  TRITON=1, fp16 86.5 same run). KLD digits identical throughout.
- Phase 2b sync cuts (`39dcd7f` deferred `int(n_new)` past the fused
  store, `923f08c` count-guarded open-staging append instead of
  `bool(any())`, 2 syncs/layer saved): 19.5 tok/s (delta within run
  noise -- the remaining wall is launch count, not syncs, per Kineto).
  The unguarded append crashed the all-sealed refresh (empty cat
  through the WHT reshape, caught by the 8k KLD, not the suites);
  the `Gs_o.numel()` count guard is load-bearing.
- Triton FWHT exactness fixes (`aa88720`, same runs as the 19.0
  number): `tl.debug_barrier` does not sync warps (triton 3.8/sm_89,
  nondeterministic corruption at 1000+ rows) -> all FWHT launches
  single-warp; fused dequant+WHT transformed the wrong axis for K
  ([dim, token] tiles) -> K dequantized raw, transposed, slice-FWHT'd.
  Both were masked until PARITY=1 ran at scale; TRITON=1 prefill
  before this fix was nondeterministically corrupt. PARITY=1 suite
  green 3x, stock 82-test suite green.
- Kineto, 5 decode steps @8192: ~3000 aten calls/step, Self CPU
  112ms vs Self CUDA 19ms -- starved on the host. Top CPU: index
  29ms (448 calls x ~65us dispatch each), copy_ 18ms,
  nonzero/unique 18ms, to-casts 9ms, 109 DtoH syncs/step.
- torch.compile probe (`eval/kvarn_compile_probe.py`): 585 dynamo
  calls into 63 unique graphs, recompile limit hit on id-keyed
  `params['dev_cache']` -- fragmentation, not fusion. The
  compile-friendly backend (static structures replacing dicts +
  dynamic shapes + .item()) is confirmed as the necessary project;
  a flag flip cannot do it. Speculative (draft-target) decoding
  would multiply effective tok/s orthogonally.

### Qwen3.8-27B dense 5.00bpw (`SC_5.00bpw_H6`)

| ctx | preset | median | mean | max | fp16 pre | kvarn pre |
|-----|--------|--------|------|-----|----------|-----------|
| 400 | kvarn4 | 0.000043 | 0.006330 | 0.380000 | — | — |
| 400 | kvarn5,kvarn4 | 0.000039 | 0.000406 | 0.006960 | — | — |
| 8192 | kvarn4 | 0.000000 | 0.000011 | 0.000329 | 5.0s | 144.6s* |
| 8192 | kvarn5,kvarn4 | 0.000000 | 0.000005 | 0.000066 | 4.9s | 149.8s* |

\*: pre-optimization numbers (per-group Python loops); see 1.40bpw
history above for the optimized path. Quality digits match across
checkpoints.

### Qwen3.8-Flash-Next 3.05bpw (Qwen4Exp, MoE+QSA, `-mcl 40`)

| ctx | preset | median | mean | max |
|-----|--------|--------|------|-----|
| 400 | kvarn4 | 0.000131 | 0.000818 | 0.009026 |
| 400 | kvarn5,kvarn4 | 0.000121 | 0.001276 | 0.059106 |
| 400 | kvarn5,kvarn5 | 0.000167 | 0.001495 | 0.030120 |
| 2048 | kvarn4 | 0.000102 | 0.000192 | 0.000888 |
| 2048 | kvarn5,kvarn4 | 0.000020 | 0.000121 | 0.003891 |
| 4096 | kvarn4 | 0.000126 | 0.000518 | 0.006072 |
| 4096 | kvarn5,kvarn4 | 0.000036 | 0.000366 | 0.007289 |
| 8192 | kvarn4 | 0.000010 | 0.000048 | 0.000880 |
| 8192 | kvarn5,kvarn4 | 0.000011 | 0.000122 | 0.005001 |
