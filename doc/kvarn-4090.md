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
| 8192 | 3124 | 2946 | 46.1 | 44.0 | 18.1GB used | 180MB KV resident |
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

| ctx | fp16 pp | kvarn pp | fp16 tg256 (peak) | kvarn tg256 (peak) | KLD same-top |
|-----|---------|----------|-------------------|---------------------|--------------|
| 8192 | 3.3s, 2449 tok/s (12.5GB) | 4.8s, 1696 tok/s (13.1GB) | 88.1 tok/s (11.9GB) | 58.3 tok/s (12.0GB) | 100.00% |
| 16384 | 6.7s, 2458 tok/s (14.1GB) | 9.5s, 1722 tok/s (15.2GB) | 83.0 tok/s (14.1GB) | 58.7 tok/s (14.1GB) | 100.00% |
| 32768 | 14.3s, 2293 tok/s (17.4GB) | 19.9s, 1643 tok/s (19.5GB) | 76.2 tok/s (18.4GB) | 56.9 tok/s (18.4GB) | 100.00% |

PARITY=1 same code: 58.0/58.4/56.3 tok/s decode; KLD digits identical
at all lengths. End-to-end allocator peaks are equal within 0.1GB --
but that is shared weights/temps dominating, NOT cache parity.

Imageless baseline @76b7fb0 (2026-09-29, `EXL3_KVARN_TRITON=1
EXL3_KVARN_IMAGELESS=1`, kvarn4; 8k rows reuse 2026-09-28 same-HEAD
runs; CPU suite 90 passed 7 skipped; fused-kernel parity maxdiff 0.0
throughout; ARMATTN PASS @8k):

| ctx | fp16 pp | kvarn pp | fp16 tg256 (peak) | kvarn tg256 (peak) | KLD med/mean/max/p99 | same-top |
|-----|---------|----------|-------------------|---------------------|----------------------|----------|
| 8192 | 3.2-3.3s, ~2525 tok/s (12.5GB) | 8.3-8.4s, ~980 tok/s (12.6GB) | 87.5-88.1 (10.4GB) | 43.0 (10.6GB) | 1e-6 / 2.3e-5 / 6.2e-4 / 3.29e-4 | 100.00% |
| 16384 | 6.5s, ~2532 tok/s (13.2GB) | 16.4-16.5s, ~997 tok/s (13.2GB) | 83.1-83.2 (11.1GB) | 42.1/42.4 (11.3GB) | 1e-6 / 1.0e-5 / 1.85e-4 / 1.65e-4 | 100.00% |

Probe-arm per-layer (us, imageless): store 245.8, qwht 11.6, eref
44.7, serve 97.2, stats 68.6, mask 26.8, tail 412.4, merge 112.9,
fmerge 17.9, ftail 50.0, fgat 34.0, full arm 805.7.
Run-variance rule: first-run-of-day tg reads low (38.3 @8k, 34.1
@16k) vs warmed repeats (43.0 @8k; 42.1/42.4 @16k) with fp16 and KLD
digits bit-identical across all runs -- always warm up / re-run
before comparing tg.

Cache-only accounting (MB; 16 cached layers; ours via
`eval/_probe_vram.py` per-tensor bytes, Bee via `llama-bench
--kv-memory` `kv_resident_bytes` + component fields; `~` = summed from
measured components, direct flag-on run queued):

| store | ours 8k | Bee 8k | ours 16k | Bee 16k |
|-------|--------:|-------:|---------:|--------:|
| fp16 K+V | 554 | 537 | 1091 | 1074 |
| q8 payload+scales | 294 | 285 | 579 | 570 |
| kvarn4 image (fp16 full ctx) | 554 | 0 (native online) | 1091 | 0 (native online) |
| kvarn4 records / Bee payload | 151 | 147 | 298 | 294 |
| kvarn4 staging / Bee staging | 50 | 25 | 50 | 25 |
| kvarn4 exact+stash / Bee exact | 84 | 25 | 84 | 25 |
| kvarn4 served total | 806 | 180 | 1490 | 327 |
| kvarn4 imageless (no image) | 235 | — | 380 | — |
| kvarn4,kvarn2 served total | ~770calc | — | ~1422calc | — |
| kvarn4,kvarn2 imageless (no image) | ~199calc | — | ~314calc | — |

Bee 8k components: payload K 73.4 + V 73.4, staging 25.2, exact
tail/history/overlay ~8.4 each (categories overlap; resident 180.4 is
the headline). Ours 16k components: image 1091, records 298.2,
staging 50.4, exact 67.2, stash 16.8. Imageless measured
(`EXL3_KVARN_IMAGELESS=1` probe): 8k = 151.4+67.2+50.4, 16k =
298.2+67.2+50.4 — no image AND no overlay stash (lazy, never
allocated without the image path). Decode-time online workspace
(~2.5MB/layer @8k geometry) not included — serves from dispatch arm.
Gap @16k imageless: 380-327 = 53MB = staging 9 + exact/stash 42 +
payload 4, misc −2. K4V4 is the comparison vehicle (K4V2 parked as
reference; reasoning benches dead last per policy). Fork long-ctx tg
check (valujin/beellama-kvarn own build 3cb90bf): @61440 tg64 fork
45.94 vs origin 45.74, @90112 fork 45.95 vs origin 45.69 — identical
within noise, tg flat with length on both. Claim NOT reproduced on
4090/sm_89/Q4_K_XL/kvarn4: comparison/tg target stays origin.
Two-point slopes are IDENTICAL on both implementations: fp16 16.0,
q8 8.5, kvarn4 body 4.37 bits/element (kvarn4,kvarn2 body ~3.4bpE,
"effectively 3-bit" per the paper author, vLLM #46613). The whole gap
is the intercept: Bee ~34MB fixed vs ours ~84MB (staging 34 + exact
50; stash absent without image) plus our image on the default path.
Fixed-fraction shrinks with length (18% @8k -> 10% @16k -> ~5% @32k),
which is why 16k is now the dev reference length for VRAM (a 16k probe
costs ~22s wall).

Gospel presets (paper author, vLLM #46613): K4V4 is the safe baseline,
K4V2 the pick ("appears lossless", author's personal choice), K2V2 for
limits only. KLD alone is NOT proof — reasoning benchmarks decide, so
KLD same-top is necessary but not sufficient. Ours K4V2 KLD-8k:
same-top 100% but ~20x the divergence of K4V4 (median 2.7e-4 vs 1e-6,
max 7.7e-3 vs 6.2e-4) — smoke PASS, reasoning bench queued as the real
gate. K4V4 stays the default; K2V2 deprioritized (INT2 support poor).

History: the quantized records (151MB vs 554MB @8k) were buried under
fp16 duplicates (image 554 + staging 336 + exact 554 = 1.59GB). Tasks
3+4 windowed staging/exact; incremental seal cut staging 40 slots to 8
(`dceb314`) and residency data cut staging+exact to 6 slots
(`affa0cf`, exact still SWA-sized at 8 via the window formula). What
remains is the persistent image (the speed play) at exactly one fp16
cache by construction: reclaiming it needs imageless serve (online
dequant, Bee-style fused attention) -- decided below (Task 5: NO-GO).

### Task 5 decision: NO-GO on imageless online serve (stop Phase 2)

Spike: true-fused decode kernel (program per kv-head x sealed-group,
128-token online loop straight from records, no per-token FWHT -- the
WHT is symmetric orthogonal so dot(Q,Hk) = dot(HQ,k): Q rows are WHT'd
once per step with the proven kernel, attention accumulates in the WHT
domain, output WHT'd once at the end; stage-2 block-combine in torch).
Spike deleted after the decision per plan; design preserved in
`.superpowers/sdd/kvarn-mem-plan/task-5-report.md`.

- Row-math vs `kvarn_triton_dequant_groups` (kvh=4, hd=256): torch.equal
  on every tile, 4-bit and 5-bit presets. PASS.
- Attention vs torch fp32 reference: RMSE 3e-08 (online-vs-two-pass
  softmax agrees to rounding). PASS.
- 8k decode probe, one 27B layer (kvh=4, hd=256, 6 q-heads/kv, 63 sealed
  groups = 8064 rows): fp16 production paged attention 0.1150 ms/step
  (14.03 ns/row) vs spike 0.2531 ms/step incl. Q/out WHTs (31.38 ns/row).
  Per-row ratio spike/fp16 = 2.24; GO needed <= 1.11. NO-GO.
- Breakdown: the fused stage-1 core runs at 9.8 ns/row (BEATS fp16) --
  the algorithm is sound. The gap is dispatch-bound scaffolding (~60%:
  torch block-combine 98us + two tiny WHT launches 95us). A Task-6
  fusion would project to ~1.2-1.3x, still missing the line; the
  everything-goes-right projection sits exactly on it with zero margin.
  A gate that needs everything to go right is not a GO.
- Consequence: stop Phase 2 (Tasks 6-7 dead: no full kernel, no default
  flip). Keep Phase 1 gains: windowed staging+exact, 1.83 -> 1.11GB
  cache-only, KLD-identical. Imageless serve stays future work (a
  prefill/varlen variant has more parallelism and might GO -- out of
  scope; the HQ-trick reformulation is validated and reusable).
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
