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

Note on BeeLlama comparison: measured on this 4090 (beellama.cpp
main @0ba48c55, sm_89 CUDA build, Qwen3.8-27B Q4_K_XL, llama-bench
`-p 8192,16384,32768 -n 256 -ngl 99`, 100MB-free VRAM rule enforced):

| ctx | f16 pp | kvarn4 pp | f16 tg256 | kvarn4 tg256 | f16 VRAM | kvarn4 VRAM |
|-----|--------|-----------|-----------|--------------|----------|-------------|
| 8192 | 3124 | 2946 | 46.1 | 44.0 | 18.1GB used | n/a |
| 16384 | 3025 | 2844 | 46.2 | 44.0 | 18.6GB used | 17.6GB used |
| 32768 | 2837 | 2641 | 46.1 | 44.0 | 19.7GB used | 17.9GB used |

Bee's online-dequant architecture (fused `flash_attn_ext_kvarn`,
windowed fp16 staging of tail_groups+1 groups, no persistent image)
pays ~5% decode tax (44.0 vs 46.1, flat across ctx) for real,
growing VRAM savings (1.1GB @16k -> 1.7GB @32k). Ours pays ~30%
(59 vs 88 @8k) for none. Absolute tok/s across harnesses is not
comparable (different weight quants/kernels); the ratios are the
signal. Porting means (1) windowed staging/exact plus (2) fused
dequant-attention replacing our serve path; our store side, dequant
kernels, and methodology transfer. Scoped as a separate branch when
the speed loop closes.

Protocol: prefill at 8k/16k/32k + 256 greedy decode tokens (tg) at
each length, 27B dense 1.40bpw, `EXL3_KVARN_TRITON=1` parity-off
(PARITY=1 alongside for the gate). Peaks are allocator peaks per
phase (weights + cache + temps; prefill peaks include the full-length
fp32 scoring logits, decode peaks are the honest cache comparison).

| ctx | fp16 pre | kvarn pre | fp16 dec (peak) | kvarn dec (peak) | KLD same-top |
|-----|----------|-----------|-----------------|------------------|--------------|
| 8192 | 3.3s, 2449 tok/s (12.5GB) | 4.8s, 1696 tok/s (13.1GB) | 88.1 tok/s (11.9GB) | 58.3 tok/s (12.0GB) | 100.00% |
| 16384 | 6.7s, 2458 tok/s (14.1GB) | 9.5s, 1722 tok/s (15.2GB) | 83.0 tok/s (14.1GB) | 58.7 tok/s (14.1GB) | 100.00% |
| 32768 | 14.3s, 2293 tok/s (17.4GB) | 19.9s, 1643 tok/s (19.5GB) | 76.2 tok/s (18.4GB) | 56.9 tok/s (18.4GB) | 100.00% |

PARITY=1 same code: 58.0/58.4/56.3 tok/s decode; KLD digits identical
at all lengths. End-to-end allocator peaks are equal within 0.1GB --
but that is shared weights/temps dominating, NOT cache parity.
Cache-only accounting at 8k (per-tensor bytes, 16 layers):

| store | fp16 | kvarn |
|-------|------|-------|
| pages (fp16 full ctx) | 554MB | — |
| image `_img_k/_v` (fp16 full ctx) | — | 554MB |
| staging `stage_k/_v` (fp16, all groups) | — | 554MB |
| exact `exact_k/_v` (tail dtype, all groups) | — | 554MB |
| records (quantized) | — | 151MB |
| overlay stash + masks | — | ~17MB |
| total | 0.55GB | 1.83GB |

The quantized records (the actual win: 151MB vs 554MB) are buried
under three full-context fp16 duplicates. The persistent image (the
speed play) costs exactly one fp16 cache by construction, so the
memory goal is currently INVERTED: more VRAM, not less. Reclaiming it
needs imageless serve (online dequant, Bee-style fused attention)
and/or windowed staging+exact instead of per-group statics -- a
storage-layer redesign, scoped separately.
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
- Page-math micro-cuts (`02581f6`, mask-filtered pages for single-row
  batches + constant page-groups buffer, ~10 launches/layer saved):
  20.2 tok/s.
- Fused open-group serve (`52ba530`, gather + full head WHT + scatter
  in 4 launches / zero syncs, replacing ~45 torch launches per layer;
  twin-tested bit-exact across seal boundaries at every head dim):
  28.8 tok/s (+43%, past Path A's ~25 estimate; fp16 86.5 same run).
  Kineto shape-attribution was the guide (torch `with_stack` yields
  empty stacks on this build, so ops were attributed by input shape).
  KLD digits identical throughout; PARITY=1 clean at 8k.
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
- Sync-free vectorized evict (`_evict_q` -> `_evict_tick` tick gate +
  `_evict_exact_all(n_rows)`: one vectorized mask over the resident set,
  no `int(max())`/per-group `int()` reads; prefill-scale calls scan
  immediately, decode-scale every 256th): 29.0 tok/s (+0.7%, runs
  29.0-29.6, fp16 85.4-86.9 same runs; kvarn prefill 4.9s). KLD digits
  identical (median 0.000001, mean 0.000020, max 0.000395, p99
  0.000249, p99.9 0.000380, same-top 100.00%); PARITY=1 clean at 8k;
  CPU suite 73 passed. Post-evict Kineto: index 15.5ms still top,
  copy_ down to 1145 calls / 5.4ms, nonzero 560 calls / 13.0ms.
- Global dirty sweep + in-place WHT (no new kernels): get_kv consumes
  `_dirty_mask.nonzero()` directly instead of the ~13-op
  resident-pages -> page-groups -> dirty chain (refresh is idempotent,
  recomputed from records/staging, so the pages restriction was pure
  rediscovery); `kvarn_triton_wht_rows(..., inplace=True)` runs the head
  kernel on the fresh serve/store fp32 temps minus alloc+copy (parity
  path stays out-of-place: its input feeds the torch reference):
  32.0 tok/s (+10% over 29.0, runs 31.9-32.0, fp16 86.2-86.7 same runs;
  kvarn prefill 5.0-5.1s vs 4.9s, run noise). KLD digits identical;
  PARITY=1 clean at 8k; CPU suite 73 passed; triton twins 9 passed
  1 skipped (per-step path agreement + image state asserted).
- Steady-path call-trimming (all numerics-preserving): `_touch_batch`
  bsz==1 fast path (whole-row validate + owner max-update, no arange +
  2D mask, ~8 launches saved; -1 padding filtered, never wraps onto
  the last owner slot); update_kv length==1 `pos` is a slice, not an
  alloc+add; fused store takes fp16 exact rows (kernel downcasts on
  load, same RNE bits -- twin-asserted) with a persistent per-layer
  status buffer (no per-call alloc); redundant `bt.long()` in the
  overlay branch dropped: 35.2 tok/s (+10% over 32.0, runs 34.7-35.2,
  fp16 85.3-86.3 same runs; kvarn prefill 4.9-5.1s, no regression).
  KLD digits identical; PARITY=1 clean at 8k; CPU 73 + twins green.
- Serve-from-image, stash-first (no clones): the Triton overlay lands
  in place on the persistent image; pre-overlay rows are stashed
  in-kernel to per-layer buffers and restored by a separate
  `kvarn_triton_unoverlay` launch in update_kv (own grid barrier: the
  tail slides every step, so same-kernel ordering would race). The
  dirty-writeback variant was measured first and reverted (17.4-17.6
  tok/s: every overlay dirtied sealed tail groups, forcing a batched
  dequant refresh per step, doubled by PARITY=1): 36.2 tok/s parity-off
  (+3% over 35.2, 35.7 PARITY=1 same code, fp16 85.5-86.7 same runs;
  kvarn prefill 4.8-5.1s, no regression). KLD digits identical
  (median 0.000001, mean 0.000020, max 0.000395, p99 0.000249,
  p99.9 0.000380, same-top 100.00%); PARITY=1 clean at 8k; CPU 77
  passed 6 skipped + triton twins 10 passed.
- Store write-through + dirty flag (no per-step refresh): the fused
  store kernel lands the WHT'd row in staging AND the image (same fp32
  row, single final RNE cast -- bit-identical to refresh-from-staging,
  twin-asserted across head dims), so pure appends dirty nothing; a
  Python-side `_dirty_any` mirror lets the sweep skip the nonzero sync
  when the mask is empty (steady decode: almost every step; seals and
  fresh-group resets still dirty host-side, first image build forces a
  full refresh): 49.3 tok/s parity-off (+36% over 36.2, runs 49.2-49.3,
  45.6 PARITY=1 same code, fp16 86.4-87.0 same runs; kvarn prefill
  4.8-5.2s, no regression). KLD digits identical; PARITY=1 clean at 8k;
  CPU 77 + twins 10 green. Kineto over 5 steps: Self CPU 129.0ms ->
  98.7ms, nonzero gone from the top, index 640 -> 400 calls, serve WHT
  kernels eliminated (the open refresh now runs only on seals).
- Tick-gated touch (isolated probe: touch 0.25ms + store 0.68ms per
  layer, serve 0.07ms): steady single-row appends skip `_touch_batch`
  (the fused store already maintains the appended page's owner; owners
  are only read by the evict scan, so the full path runs on the evict
  tick and owners are current whenever the scan runs; the range
  validation still fires within <=256 steps plus the gather
  bounds-check backstop every step; prefill/multi-row always run):
  64.5 tok/s parity-off (+31% over 49.3, runs 63.4-64.5, 63.0 PARITY=1
  same code, fp16 86.1-87.0 same runs; kvarn prefill 5.0s, no
  regression). KLD digits identical; PARITY=1 clean at 8k; CPU 77 +
  twins 10 green.
- Batched single-group seal (256-step runs exposed an 88ms/seal cliff:
  37.7 tok/s over 256 steps vs 64.5 over 120): `_seal_group` delegated
  to `_seal_groups_batched` (one Sinkhorn+quantize per K/V over all
  tiles; records bit-identical): 59.1-59.5 tok/s over 256 steps from 8k
  (58.0 PARITY=1, fp16 88.1-88.5 same runs). KLD identical; CPU 77 +
  twins 10 green.
- Image to 160 pages (~40k tokens): 32k+256 needs 129 pages and the
  legacy path rematerializes the whole context per step (8.2 tok/s at
  32k vs 64.4 fp16). 56.9 tok/s over 256 steps from 32k (56.3 PARITY=1,
  fp16 76.1-76.4 same runs); 16k holds 59.1 (58.4 PARITY=1, fp16 83.0).
  KLD same-top 100% at all three lengths; decode peaks equal to fp16
  within 0.1GB (see protocol table). Harness now prints prefill tok/s
  and per-phase peak VRAM.
  (Copy+overlay single-kernel fusion was considered and rejected: the
  copy grid and overlay grid would write the same temp rows from
  different programs with a required order and no cross-CTA barrier.)
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
