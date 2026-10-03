# KVarN: 4090 (sm_89) machine handoff

Workflow knowledge lives in `wiki/` (WikiSkill-lite: `wiki/index.md`
catalog, `wiki/patterns/` durable rules, `wiki/skill-impact.md`
accept/reject trail). This doc is the chronological record; reusable
rules go to the wiki, not here. Check the wiki before proposing
process/skill changes.

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

| ctx (tok) | preset | median | mean | max | p99 | p99.9 | fp16 pre (s) | kvarn pre (s) |
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

| step | change (commit) | fp16 pre (s) | kvarn pre (s) | ratio (x) |
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

| ctx (tok) | f16 pp (tok/s) | kvarn4 pp (tok/s) | f16 tg256 (tok/s) | kvarn4 tg256 (tok/s) | f16 VRAM (GB used) | kvarn4 VRAM (GB resident) |
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

Standing bench requirements (2026-09-29, user-locked):
- tg at 64k AND 128k, not just 8/16/32k: imageless kvarn4 + fp16
  ref, 256 greedy decode tokens, same flags as the protocol table.
- Every numbers table shows the BeeLlama baseline alongside ours
  (`llama-bench -m Qwen3.8-27B-UD-Q4_K_XL.gguf -p <ctx> -n 256
  -ctk kvarn4 -ctv kvarn4`, `--kv-memory` resident bytes) AND VRAM
  (ours: cache-only per-tensor + decode-peak allocator; Bee:
  `kv_resident_bytes`).
- After every change: re-run the gates, show the updated table in
  the report, and update this file. Every table header carries its
  measure unit (tok, s, tok/s, GB, MB, %); KLD median/mean/max/p99
  columns are divergences (unitless) unless noted.
- Gate order before deep work: smoke first (needle item 0,
  "capital of France" -> Paris, aborts the probe on fail) + short
  reasoning (`bbeh_mini --limit 3 -fresh`) -- never go deep on a
  wrong path.
- VRAM guard (project-wide rule, see `AGENTS.md` on `fork-overview`):
  every GPU run is wrapped with `eval/smi_guard.py`, which polls
  `nvidia-smi memory.free` and KILLS the task the moment free VRAM
  drops under 100MB (only PIDs that appeared after launch are ever
  killed). `0 used` before starting, idle after finishing, 1GB
  headroom over the expected peak before starting.

| ctx (tok) | fp16 pp (s, tok/s, peak GB) | kvarn pp (s, tok/s, peak GB) | fp16 tg256 (tok/s, peak GB) | kvarn tg256 (tok/s, peak GB) | KLD same-top (%) |
|-----|---------|----------|-------------------|---------------------|--------------|
| 8192 | 3.3s, 2449 tok/s (12.5GB) | 4.8s, 1696 tok/s (13.1GB) | 88.1 tok/s (11.9GB) | 58.3 tok/s (12.0GB) | 100.00% |
| 16384 | 6.7s, 2458 tok/s (14.1GB) | 9.5s, 1722 tok/s (15.2GB) | 83.0 tok/s (14.1GB) | 58.7 tok/s (14.1GB) | 100.00% |
| 32768 | 14.3s, 2293 tok/s (17.4GB) | 19.9s, 1643 tok/s (19.5GB) | 76.2 tok/s (18.4GB) | 56.9 tok/s (18.4GB) | 100.00% |

PARITY=1 same code: 58.0/58.4/56.3 tok/s decode; KLD digits identical
at all lengths. End-to-end allocator peaks are equal within 0.1GB --
but that is shared weights/temps dominating, NOT cache parity.

Imageless baseline @0bf462a, rebased onto `origin/fork-overview`
(code-identical to the validated @76b7fb0: delta is README +
local-build docs only; 2026-09-29, `EXL3_KVARN_TRITON=1
EXL3_KVARN_IMAGELESS=1`, kvarn4; 8k rows reuse 2026-09-28 same-code
runs; CPU suite 90 passed 7 skipped; fused-kernel parity maxdiff 0.0
throughout; ARMATTN PASS @8k. Bold = winner.)

tg baseline (tok/s, 256 greedy decode; 128k: 256-step decode):

Code version for ours columns: post-graphs-v2 (2026-10-02;
seal-direct fast path + hierarchical serve + graphs, protocol v3:
clean perf + parity validation separate). Bee columns are external
(beellama.cpp). Untagged older numbers predate versioning.
PARITY TAX (2026-10-02): every tg/pp number below was measured
with PARITY=1, whose full-refresh asserts cost ~15% throughput.
CleanPerf (no parity) in the v3 rows; parity runs remain the
correctness validation (asserts must stay green there). Protocol v3:
perf runs PARITY=0 (+ GRAPH default), validation runs PARITY=1
(either GRAPH setting); never compare across parity settings.

| ctx (tok) | tg ours fp16 (tok/s) | tg ours kvarn4 (tok/s) | tg Bee kvarn4 (tok/s) | tg Bee f16 (tok/s) | ours/Bee (%) | KLD med/mean/max/p99 (unitless) | same-top (%) |
|-----|--------------|----------------|---------------|------------|----------|----------------------|----------|
| 8192 | 87.6 | 48.2 eager / 55.8 graph (was 40.8 parity-taxed; pre-serve-v2, re-measure) | **44.0** | 46.1 | 110% / 127% | 1e-6 / 1.8e-5 / 5.05e-4 / n/a | 100.00% |
| 16384 | 82.2 | 46.8 eager (pre-serve-v2, re-measure) / 58.3 graph (+3.2% cap64) | **44.0** | 46.2 | 106% / 132% | 1e-6 / 1.1e-5 / 2.96e-4 / n/a | 100.00% |
| 65536 | 62.2 | 46.8 eager / 52.2 graph (+9.0% serve-v2 + cap64) | **44.0** | 46.1 | 106% / 119% | 1e-6 / 2.8e-5 / 1.36e-3 / n/a | 100.00% |
| 131072 | 47.1 | 35.2 graph (pre-serve-v2, re-measure; Bee tg@128k unknown) | n/a | n/a | n/a | 1e-6 / 5e-6 / 6.9e-5 / 4.9e-5 | 100.00% |
Peak VRAM @64k, protocol v3 (microkld `peak` lines): prefill 19.0GB
fp16 / 19.1GB kvarn4; decode 14.7GB / 15.3GB. Prefill-vs-decode gap
is structural (chunk fp32 logits + full-ctx remat temps + allocator
retention), NOT a leak — see research 2026-10-03. Standing rules
(2026-10-03): (1) every box harness prints peak allocated per phase
(microkld already does) and every ledger table transcribes it —
no number without its peak; (2) q8 proxy rule: fp16 ref where it
fits, `-ref q8` (KLD digit-identical to fp16, med 1e-6 / mean 4e-6 /
max 4.6e-5 @128k) only where fp16 spills; one-time ≤32k
proxy calibration (kvarn-vs-fp16 AND kvarn-vs-q8 digit-equal +
identical needle HIT/MISS) before q8-only at 64k/128k.
Combine WHT-split (2026-10-02, `90074b7`, default ON, kill-switch
`EXL3_KVARN_COMBINE_SPLIT=0`): tg@64k graph 47.5 hot vs 47.8 pre-cut
(neutral within noise; run1 34.9 was one-time triton recompile of the
two new specializations). Fresh online-path Kineto re-ranks the
audit: combine is 0.28ms/step (not 1.2ms), so the +3-5% estimate is
retired to ~+1% ceiling; serve 4.5ms is the top kvarn item, MoE ~8ms
dominates device. Twin maxabs <1e-5, KLD digits identical @64k,
PARITY=1 @8k clean (29.5 tok/s, asserts green).
n-mirror (2026-10-02, `c50905e`, default ON, kill-switch
`EXL3_KVARN_N_MIRROR=0`): host mirror of int(cache_seqlens[0]),
kills ~16 DtoH syncs/step; tg@64k graph 47.9 hot vs 47.8 (neutral,
syncs were hidden). Fail-closed: q_len!=1/PARITY/128-cadence
resync, eager-rest + legacy paths always real-sync. KLD identical,
PARITY=1 @8k clean (42.7 tok/s, asserts green).
longskip (2026-10-03, `633660b`): _as_long skip-when-long guards +
_touch_batch early-out before conversions (~96 .long()/step in
Kineto); tg@64k graph 47.8 hot vs 47.9 (neutral -- host dispatch
count is not the binding constraint either). KLD identical,
PARITY=1 @8k clean (41.3 tok/s, asserts green).
Task-6 phase-0 spike (2026-10-03, `eval/_spike8_layer.py`, no code
change): post-graphs device busy @64k measured for the first time --
16.51ms/step over 1682 kernels (kvarn serve 4.46, exl3 GEMV/GEMM
8.20, tail 0.78, combine 0.40, GDN 0.53) vs 19.27ms in-process wall
(51.9 tok/s, argmax loop) -> host bubble 2.77ms/step = 14.4% of
wall, device ceiling 60.6 tok/s in-process. Closes open question 1
of `wiki/reports/2026-10-02-tg64-host-bubbles.md`: the remaining
host-side pool is 2.77ms, not the 4-7ms the ranking assumed. Whole-
layer graph capture is NOT viable: GDN and GatedMLP decode already
self-capture via `bc.run_bszN` / `BC_GatedMLP` (nested capture trips
`exllamav3_ext/graph.cu:186`, exit 900), BCAttn declines on all 16
kvarn layers, and the attn-layer capture is invalidated by the eager
serve's host read at `dispatch.py:491`; store alone is 0.228ms/layer
(3.50ms/step) and is host-sync bound. The one shape with a prize left
-- two disjoint graphs per attn layer (project_qkv+rope / gate+o_proj,
store+serve eager) -- was then BUILT and measured: 16/16 layers
capture, bit-exact (region worst maxabs 0.0, re-verified over 10
advancing steps x 16 layers), but worth only +0.041 ms/step =
19.272 -> 19.231 ms (51.9 -> 52.0 tok/s in-process, ~48.0 projected on
the 47.9 baseline) because the removed dispatches were already
overlapped with the GPU. Verdict STOP + re-rank; details in
`wiki/plans/task6-whole-layer-graphs.md` §8. Model is dense
(GatedMLP), so the "MoE ~8ms" label above is really the dense exl3
GEMV/GEMM path (measured 8.20ms).
Serve v2 / per-group metadata hoist (2026-10-03, task-7 attempt 1,
default OFF, kill-switch `EXL3_KVARN_SERVE_V2=0`): the serve tile loop
becomes a (group, tile) nest, so the block-table gather and the three
per-channel metadata vectors (K sc, K zp, V oth) load once per 128-row
group instead of once per 16-row tile (8x fewer) -- a TOK=16 tile can
never straddle a 128-row group, so the values are identical.
Phase-0 bill (isolated serve @64k, differential ablation of 18 kernel
variants, 7 interleaved windows): metadata loads are 54.0% of the
222.8us/layer kernel, K+V payload reads 37.8%, dequant ALU 24.8%,
exact-direct tail 13.4%, partials stores 3.2%, exp 5.9%, one extra MMA
-1.5% (i.e. the tensor cores are NOT a cost -- §3 dot narrowing and
exp fast paths have no prize). Serve kernel 222.8 -> 162.7 us/layer
(-27%). Occupancy proof: grid 512 CTAs unchanged, regs/thread
219 -> 254, spills 0, smem 16384B unchanged, CTA/SM 2 -> 2 (8 warps/SM,
17%) -- preserved, so this is a win at constant occupancy, not a
traffic-reduction illusion. (254 regs is one step from the 255 cliff
that would halve CTA/SM; watch it if the body grows.)
Gates: twin `test_serve_v2_group_hoist_bit_exact` 7 shapes
(CPG=4 production, non-pow2 gc, short-prefix mask rows, sink+tail,
CPG=1) bit-exact via `torch.equal` on out AND m/l/acc partials;
KLD same-top 100.00%, mean 2.8e-5 < 1e-4, and *identical to 6 digits
across arms* (the bit-exactness prediction); PARITY=1 @8k green both
arms; CPU suite 85 passed / 14 skipped.
Perf (64k, `-dec 256`): graph path, arms interleaved over 3 rounds --
legacy 47.9 / 47.9 / 48.0 tok/s, v2 51.0 / 50.8 / 51.0 tok/s = **+6.5%**.
Eager in-process alternating (one model load, gate flipped per window,
6 x 32 steps, `eval/_spike9_ab.py`): 21.254 -> 19.970 ms/step = -6.0%,
1505.6 -> 1604 tok/s. Note: this box has a +/-25% BETWEEN-PROCESS
spread for identical code (expandable_segments is unsupported here, so
each process lays the 64k cache out differently) -- one v2 graph run
measured 39.3 tok/s against 50.8 for the same config. Single-process
alternating A/B is the only reliable protocol here; both arms also
show one ~25ms outlier window at the same step count (a periodic
event, pre-existing, not this cut).
Serve v2 default ON (2026-10-03, follow-up to `3ccc7e8`, kill-switch
`EXL3_KVARN_SERVE_V2=0` restores the legacy kernel): box-green, so
the gate now defaults to 1. Re-validated with the env UNSET: PARITY=1
@8k asserts green (43.3 tok/s, KLD identical), 64k graph 50.9 and
50.8 tok/s. Full CUDA twin suite green on both arms (20 passed) with
one PRE-EXISTING failure, `test_promoted_serve_matches_eval_spike`:
the promoted serve kernel and the eval spike7 original it was copied
from differ by 4.5e-08 in one q-row, and the failure reproduces
identically with `EXL3_KVARN_SERVE_V2=0` (legacy) and `=1` (v2), so
it predates this task -- the "promoted == eval original" guard has
been broken by drift and should be re-synced (or the twin relaxed to a
tight allclose) separately.
Serve-groups cap 128 -> 64 (2026-10-03, task-7 attempt 2, default 64,
kill-switch `EXL3_KVARN_SERVE_GROUPS=128`): the hierarchical subgroup
cap trades CTA count against partials traffic, and at 2 CTA/SM that
trade is decided by wave quantization, not by partials bytes. At 64k
this goes from 128 groups x CPG=4 (512 CTAs = 4 exact waves, 4MB/layer
of partials) to 64 x CPG=8 (256 CTAs = 2 waves, 2MB/layer): HALF the
partials traffic AND faster. Isolated serve 163.7 -> 154.2 us/layer
(-5.8%); CPG=2 (cap 256) and CPG=1 (cap 512) both measured slower
(+8.2, +8.6 us), so the old cap sat past the optimum the other way.
Occupancy: CTAs 512 -> 256, regs/thread 254 unchanged, spills 0, smem
16384B unchanged, CTA/SM 2 -> 2 -- the grid halves AND the wall
improves, the opposite of the flat-32 failure mode.
Perf, 4 interleaved rounds per arm at 64k graph: cap128 50.7 / 50.8 /
50.8 / 50.9, cap64 52.1 / 52.2 / 52.3 (one 72.7 outlier process,
discarded) = **+2.8%**, and 52.2 with the env unset. Also +3.2% @16k
(56.5 -> 58.3) and +1.8% @32k (38.4 -> 39.1). Gates: KLD same-top
100.00% and mean 2.8e-5 at 64k/32k, 1.1e-5 at 16k -- identical to the
cap128 arms to 6 digits (the reduction tree keeps its kind, only its
depth changes: 4 tiles/CTA -> 8); PARITY=1 @8k green (42.4 tok/s);
twin suite pass/fail IDENTICAL at cap=64 and cap=128, so the 4
failures the protocol env causes (EXL3_KVARN_IMAGELESS=1 /
EXL3_KVARN_GRAPH=1 break four store/image twins that do not set those
themselves) are pre-existing and not this cut -- run the twin suite
WITHOUT the protocol env.
Four more negatives, so nobody re-proposes them (all vs the same base of
155.9 us/layer, one interleaved process). The six per-slot (16,)
metadata loads (K oth, V sc, V zp x SL=2) are the largest section
left -- `nopermeta` measures them at 43.4 us/layer = 0.70ms/step =
27.9% -- and none of the five ways to get them out of the inner loop
wins: rewriting the three (16, HD) expansions from SL selects each into
a (16,2,128) broadcast of a joined pair (same tensor, same values)
+4.4us; unrolling the tile loop so the compiler can merge the eight
adjacent 16-element chunks into one 256B load +88.7us with 124 spills;
issuing the six loads one tile ahead by hand +17.4us with 10 spills;
`num_stages=2` +29.1us; and keeping the loads while deleting only the
expansions (`permeta`) changes NOTHING (+1.8us, regs 254 -> 251). That
last one is the load-bearing measurement: the 27.9% is the loads' own
issue and latency, not the select ALU, and not the registers the
expansions hold -- which also refutes the "get under 170 regs for
3 CTA/SM" theory. The serve kernel is at a local optimum for this tile
shape; the next candidate is structural (fold K sc/zp into `qwT` per
group, split the QK dot per slice so `k_ot` never reaches the (16, HD)
domain, rewrite the V side as `dot(trans(qqv), (v_sc*e))` plus a rank-1
`v_zp` term). Those reassociate -> allclose + KLD gates, not bit-exact,
and are a much larger change; not attempted. Serve device time is now
~2.49ms/step for 16 layers, down from 4.474ms at the start of task 7.
Tail+merge fusion -- STOP (2026-10-03, `wiki/plans/task4-tailmerge-fusion.md`,
no code change; spike `eval/_spike10_tailmerge.py`, log `t4_spike10.log`).
The hand-fused kernel is BIT-EXACT against the two separate launches
(`torch.equal` True, maxabs 0.0, 0/6144 differing elements: same op
order, and the fp32 tail-stat DRAM round-trip it removes is
value-preserving, so register residency cannot change the result). It is
also FASTER in isolation on the device -- 52.4 -> 50.6 us/layer (-3.5%,
reproduced to +-0.15us over three interleaved runs) and 16.8-18.2
us/layer cheaper to ISSUE from python (two wrappers + three reshapes
become one launch). And it is SLOWER end to end: in-process
interleaved eager A/B at 64k, 10 windows of 16 steps, split 19.737
ms/step (50.67 tok/s, range 19.43-20.40) vs fused 20.638 ms/step
(48.45 tok/s, range 20.29-31.09) = **-4.5%**. All 10 fused windows beat
the split median and 7 of 10 beat EVERY split window; even the fused
arm's fastest window (20.29) beat 9 of the 10 split windows. Plan §0's
guard fires ("fused slower than separate -> STOP"). Why the two
disagree: the pair is only 52.4us x 16 = 0.84ms of a 19.7ms eager step
(4.2%), so even the full
device win is +0.15% -- the plan's "+1-2%" premise was ~10x optimistic
-- while fusing turns two independent 24-CTA kernels per layer into one
longer dependent chain, which plausibly costs the cross-layer overlap
(tail of layer N+1 has no dependency on merge of layer N, so with them
separate the long tail loop can hide under it; fused, it cannot). That
overlap loss is ~60x the 15us the removed launch actually saved.
Mechanism inferred from the timings, not separately profiled -- but the
verdict does not depend on it: the cut loses on every end-to-end
metric. Do NOT re-propose this fusion; the prize is not there in
either direction. Premise correction for the plan: the EAGER
baseline is 46.8 / 46.7 tok/s @64k and 47.7 @8k
(`EXL3_KVARN_GRAPH=0`, protocol v3, this run), not the 44.1 written
down -- the task-7 serve work moved eager too, so a +2% cut would have
been 46.8 -> 47.7, not 44.1 -> 45. Eager traffic is real (tabbyAPI
`max_batch_size: 2`, so bsz=2 decode declines the graph at
`dispatch.py:249` and runs eager every step), which is why the gate
question was worth asking -- the answer just does not make this cut
worthwhile. Also note the isolated 52us/layer is ~50x off roofline for
1.25MB of reads: the cost is the tail kernel's serial R=256-iteration
reduction chain (`tl.sum(where(roff==r, e, 0))` per r), not the tail
DRAM round-trip. That loop, not the fusion, is where a future tail-side
cut would have to aim.
Sampler + terminal sync -- STOP (2026-10-03,
`wiki/plans/task5-sample-in-graph.md`, no production code change; spike
`eval/_spike11_sampler_bill.py`, log `t5_bill.log`). Measured in the
SERVER path (real `Generator` + `Job`, greedy `ArgmaxSampler`, 65536-token
prompt, protocol v3, 6 windows x 40 steps interleaved): **17.344 ms/step
= 57.66 tok/s**. Host split per step: model forward **16.382 ms** (94%;
of which 11.72 ms host dispatch + 4.66 ms blocked in the fused store's
status readback), terminal `torch.cuda.synchronize` at
`generator.py:1128` **0.667 ms** (exactly 1 per step), everything else
0.245 ms (block-table/positions/input-id staging, sampler launch,
`receive_sample`, requeue). Sampler device tail 0.024 ms.
**The step is device-bound with zero slack**: adding 5 ms of pure-python
spin at the step boundary (where the device is provably idle, the
previous step having ended in a full sync) costs 5.01 ms of step time,
not one microsecond absorbed. Therefore no host-side cut can recover host
time. Confirmed directly by a **device-floor loop** (phase F: same
forward, same pinned staging and the same block_table params the
generator passes, but the sampled token stays on the device and there is
one `torch.cuda.synchronize` per *window* instead of per step, run on its
own cache and its own prefill in the same process): floor **17.160 ms =
58.28 tok/s** vs base 17.366 = 57.58, with the control `floor_sync`
(identical loop plus the per-step terminal sync and a `.cpu().item()`
readback) landing within 0.25% of base, which is what makes the floor
usable. **So the entire host cost of the server decode loop -- every
staging byte, every bookkeeping call, the terminal sync, all the
readbacks -- is 0.197-0.25 ms/step (1.1-1.5%), and that is the hard
ceiling for this whole task.** A second full run reproduced it (base
17.394 / floor 17.141 / floor_sync 17.436, control -0.24%), so the
ceiling is 0.252 ms/step = +1.45%, best case 58.34 tok/s. The floor
loop samples with a bare `argmax` rather than the real sampler chain, so
it also skips the 0.024 ms sampler device tail and the true ceiling is
nearer 0.23-0.27 ms. Note `torch.profiler` could NOT be used to measure
device-busy here: its kernel-sum disagreed with itself on identical code
across the two runs (21.5 ms/step then 14.8 ms/step, neither consistent
with the 17.3 ms wall) -- CUDA-graph replay plus CUPTI is not a reliable
kernel-sum. The floor loop's wall time is the measurement that stands,
and it needs no profiler: with the host out of the loop, wall time per
step IS the device cost. Deleting the terminal sync (incorrect: it feeds stale
tokens) measures 17.109 = 58.45 tok/s, i.e. it already reaches the floor
within 0.3% -- there is nothing beyond it for a correct implementation to
find. A perfect sample-in-graph (device-side input staging, static
Philox, no host stall) buys 58.3 tok/s instead of 57.6, and the plan's
own ranked attacks cannot do better: deleting the terminal sync outright
(illegal -- it feeds stale tokens) is worth **-0.257 ms/step = +1.48%**
(17.366 -> 17.109, 58.45 tok/s, consistent across all 6 windows, i.e.
already at the floor), and the "LAST, hardest" attack (argmax in the
graph + device-side input staging) is bounded by the same 0.197-0.25 ms. Two
ranked attacks were each attacked and measured, not argued: (a) "batch
the `.item()` reads" EXISTS upstream at
`generator.py:1102-1128` -- one pinned buffer, one synchronize per
step -- and the 4 remaining token readbacks (`job.py:620/621/809/826`)
cost 3.8 us/step total; (b) replacing the full sync with a stream-event
wait was IMPLEMENTED as a fourth arm (`evwait`, same loop, same
interleaving, 6 windows x 40 steps) and is worth **+0.0010 ms/step, i.e.
zero**: blocked time 0.6725 -> 0.6735 ms. Three supporting facts: the
wait primitive was never the cost (`torch.cuda.synchronize` idle =
3.9-4.1 us, kernel+sync round trip = 11.8 us, and the event form is the
SLOWER of the two at 9.0-9.7 us idle / 17.2 us with work queued); there
is nothing for a narrower wait to skip, because the decode path creates
NO side streams (the only `torch.cuda.Stream(...)` sites are
`model/moe_cpu_host.py:1180`, CPU MoE offload, and this model is dense
`use_moe=False`, and `modules/quant/exl3_lib/quantize.py:1725`, weight
quantization at load time); and the sync is not a cross-stream handoff --
the token is produced by kernels queued behind the forward on the same
stream, so waiting for the token IS waiting for the forward. +1.2-1.5% (the measured ceiling; three runs give 0.197 / 0.206 / 0.252
ms) is inside this box's own
window-to-window spread (-23% seen in the same run, in every arm and
every phase) and only reachable behind a static-Philox RNG problem: the sampler seed is a host int
(`job.py:578` -> `sampler/custom.py:1185-1190`). Do NOT re-propose
sample-in-graph / terminal-sync removal. Premise corrections to the
plan: the "~1.7 ms loop overhead above the model" pool is really 0.9 ms
of which **0.197-0.25 ms** is recoverable (the device-floor
measurement);
`generator.py:670` is the DRAFT path,
not the main greedy sample (that is `job.receive_logits`,
`job.py:571-583`); `DefaultSampler` is at `sampler/presets.py:3`, not
`job.py:166-168`. Two findings worth keeping, neither chased here: the
largest single host-blocking item in a server decode step is the fused
store's status readback at `kvarn_triton.py:502` -- **4.63 ms/step, 27%
of the step**, 16 blocking `tolist()` per step, reached via
`kvarn.py:2019 _store_rows` <- `kvarn.py:2986 update_kv_direct` <-
`dispatch.py:259 _try_kvarn_graph_decode` (previously unnumbered in
`wiki/reports/2026-10-02-tg64-host-bubbles.md`, which had flagged it
HARD); and the SERVER harness measures 57.66 tok/s at 64k where the
microkld ledger harness measures 47.8 for the same model/cache/protocol,
so the ledger understates the server by ~20% and server-path work must
be measured on the server harness.
Note: KLD divergence trend across approximation cuts (mean
1.7e-5 base -> 2.3e-5 Sinkhorn -> 3.2e-5 deferred seals @64k;
max 6e-4 -> 8.2e-4 -> 1.3e-3; same-top 100% throughout,
absolute values tiny). Budgeted: each cut must keep same-top
100% and mean < 1e-4 @64k; reasoning benches (not KLD) decide
ultimately per gospel policy.
Reasoning smoke (2026-09-29, bbeh_mini --limit 3 -fresh, current
code): kvarn4 0/3, fp16 baseline 0/3 on the same 3 (BBEH-mini is
frontier-hard; both ramble to the 16k cap). Honest reading: this
config has ZERO discriminative power (0-baseline gate decides
nothing) — do not cite it as a quality gate. What it does prove:
end-to-end stability (3× up-to-16k-token generations, no crash/
hang/OOM, slot windows hold). Quality gate stays KLD (same-top
100% + digits). Reasoning calibration MINED 2026-09-30 (fp16,
greedy `-temp 0 -topp 1`, `-mt 2048 -fresh`, mini items 0-60,
`fp16_mine.jsonl` on box): 0/60 solved -- all ramble-to-cap
(4-9KB answers, no judgeable output). Mini-first-N is barren for
calibration (floor effect on fp16 too, not a cache signal); do NOT
extend blindly. Next pond if ever needed: full-BBEH task-split
stratification (some tasks are easier) or a different bench, never
more first-N. Needle stays the long-context gate; bbeh_mini stays
stability-smoke only.
Scope directive (2026-09-30, user): K4V4 ONLY from now on. No new
K4V2 runs (KLD/twins/needles); K4V2 stays parked (was co-vehicle
on speed 39-40 tok/s tied, deficit KLD-only). Existing k4v2 twin
stays as regression cover; all future gates `-cq kvarn4`.
Needle gate (eval/kvarn_needle.py -- smoke + passcode at 5/50/95%
depths + multi-conjunction + recency-update, greedy, substring
check; K4V4 only per scope): fp16 6/6 (39s), kvarn4 6/6 (109s),
all depths HIT. Multi needed -mt 256 headroom (170 tok: preamble
+ 3 codes; the old 64-cap cut it mid-reasoning --gate bug, not a
model failure). Update item keys on the NEW code only. THIS stays
the discriminative long-context gate (baseline solves it): any
retention regression (evict/slot/seal) shows as MISS. Run per cut
when touching store/evict/seal/serve paths.
| 131072 | 47.4-47.5 | 28.1/28.7 | **44.0** | OOM Bee f16 only (>24GB: 17.9GB Q4_K_XL + 8.59GB KV = 26.5GB; ours-fp16 fits at ~19-23GB, tight) | 65% | 1e-6 / 3e-6 / 3.1e-5 / 2.7e-5 | 100.00% |

128k q8-ref (2026-09-29, current code, same box/flags): q8 prefill
85.7s (1529 tok/s, peak 18.3GB), q8 tg 56.3; kvarn prefill 148.4s
(883 tok/s, peak 19.3GB, was 171.6-219.4s pre-Sinkhorn), kvarn tg
22.8 (base 28.8; expandable_segments: 29.0, i.e. parity-or-better);
KLD kvarn-vs-q8 median 1e-6 / mean 4e-6 / max 4.6e-5, same-top
100%. Fragmentation verdict, third confirmation: default-alloc
tg gaps (-3.5%@8k, -24%@64k, -21%@128k) all collapse under
expandable (8k 42.9, 64k 35.9 vs 37.0, 128k 29.0 vs 28.8). The
code is perf-neutral-or-better everywhere; the pool layout is the
lever. Recommendation stands: expandable_segments for 64k+.

pp baseline (tok/s, full-context prefill):

Code version for ours columns: `f73271f` (2026-09-30; seal-direct
fast path, protocol v2 hot-cache, chunk 8192 unless noted). Bee
columns are external (beellama.cpp). Untagged older numbers predate
versioning.

| ctx (tok) | pp ours fp16 (s, tok/s) | pp ours kvarn4 (s, tok/s) | pp Bee kvarn4 (s, tok/s) | ours/Bee (%) |
|-----|--------------|----------------|---------------|----------|
| 8192 | 3.2s, ~2560 | 3.6s, ~2280 (was 3.75 parity-taxed) | **2.8s, 2946** | 78% |
| 16384 | 6.3s, ~2586 | 7.1s, ~2300 (was 7.55 parity-taxed) | **5.8s, 2844** | 82% |
| 65536 | 32.1s, ~2044 | 35.5s, ~1844 (was 40.0 parity-taxed) | **28.1s, 2336** | 79% |
| 131072 | 198.1s, ~662 (chunk 4096; swap pressure) | 157.9s, ~830 (chunk 4096, warm-restart) | **69.5s, 1887** | 44% |

Long-context degradation verdict (2026-09-30, code `f73271f`,
chunk 8192, protocol v2): NO kvarn cliff. pp kvarn/fp16 slips
88% -> 85% -> 80% over 8x ctx (shared O(n^2) prefill attention in
the 16 full-attn layers hits everyone: fp16 itself falls
2525 -> 2565 -> 2044, Bee 2946 -> 2844 -> 2336). The old 16k cliff
(997 tok/s) was stale measurement, killed by re-measure (2175).
tg slope is the real scale story: 40.8 -> 39.4 (-3%) -> 35.1
(-14%), vs Bee flat 44 -- serve partials traffic is O(n) in gc
(acc + m/l combine reads double per doubling) while Bee's fused
serve + parallel combine is flat. Hierarchical serve+combine
(`3647f54`, groups capped at 128 past gc 64: 8k/16k bit-identical
direct-equiv, 64k 4x fewer partials) recovered +3% (34 -> 35.1,
K4V2@64k 35.5, all gates green). Lesson inside the lesson: the
first attempt (flat 32 groups) LOST 4% -- serve is
parallelism-bound, not traffic-bound (128 CTAs underfilled the
144 SMs); 512 CTAs saturate, so the cap is 128, not 32.
New-context-length = new triton specializations (nbpad): first run
at each ctx reads 15-25% low (16k: 31.8 -> 39.4; 64k: 28.4 ->
33.7 -> 34.2); always run twice per ctx, take the hot number
(protocol v2 amendment). Same for ANY kernel source change
(comments included: triton hashes source text).
Chunk 8192 (single forward, 2026-09-29): kvarn pre 4.8s, 1718 tok/s
(+5% vs chunk-4096 5.05s; fp16 peak 14.5GB vs 12.5GB), KLD median
1e-6 / mean 1.9e-5 / max 5.15e-4 (slightly BETTER than chunk-4096:
fewer chunk boundaries), same-top 100%. Recipe: biggest chunk
that fits VRAM (fewer forwards amortize per-call fixed costs).
MMA floor (2026-09-30, `561588c`, REVERTED `2a0b382`): wrapper
padded QPAD to >=8 (QPK=4 gave QPAD=4 -> QK/EV dots SIMT). Twins
green, KLD identical, same-top 100% -- but hot-cache A/B shows
ZERO tg effect (K4V4 39.4->39.4, K4V2 39.3->39.5, noise). The
apparent +4% was triton-compile contamination (see protocol v2).
Reverted: neutral cuts don't ship, and it doubled serve-acc
buffers (idle lanes) against the VRAM axis. Lesson: dots are NOT
the bottleneck (SIMT->MMA invisible) -- serve is bound elsewhere
(dequant loads / tail / bubbles). Fusion is next, by profile.
Stable A/B protocol v2 (2026-09-30, mandatory): warmed box (first
run after idle reads ~20% low) + `expandable_segments:True` +
fp16-decode anchor 87.2-87.5 + HOT TRITON CACHE (every code or
preset change recompiles inside the timed region: 31-32 reads go
39+ hot; run twice per code version, take the hot number) +
back-to-back pairs + anti-bias ordering (control LAST). All
pre-v2 tg numbers (incl. the 43.6-43.8 K4V4 and 34.9 K4V2) carry
cold-cache/box noise -- the old K4V2 -20% gap is GONE under v2:
K4V4 39.4 vs K4V2 39.3 (tied). K4V2 un-parked as co-vehicle on
speed; its deficit is now KLD-only (270x median, same-top 100%
holds). Host-desync cut (2026-09-30, `bf91130`): serve sticky-flag read
goes periodic (every 128 serves/layer; PARITY=1 checks every call --
spec-authorized: "keeps its sync until green", green since Spec A),
tpos memoized across layers (same key per step, read-only
downstream), update_kv .to(long) dtype-guarded. Zero math change.
Twins + suite green, KLD identical, same-top 100%. tg 39.4 -> 40.2
(+2%). Status.tolist stays synchronous (drives code 0/2/1 control
flow -- cannot speculate).
Qh view (2026-09-30, `6dda3ec`): Qh as reshape-view of Qf (was a
second fp32 copy; identical values). tg 40.2 -> 40.0 (noise),
KLD identical. Keep: fewer allocs/copies, zero risk.
Kineto @8k (2026-09-30, 5 steps, K4V4): CUDA 12.6ms/step vs wall
~25ms -- GPU half-idle. kvarn kernels 1.9ms (serve 0.74 + tail
0.78 + combine 0.16 + wht 0.23); exl3 gemv/mgemm ~8.6ms; rest is
host CPU (~19ms: copy_/to/_to_copy + 128 syncs + ~600 dispatches
for 16 full-attn layers -- the model is HYBRID: 16 attn + 48
linear). Bee gap anatomy: Bee kvarn4 44 vs Bee f16 46.1 (4.5%
overhead, C++ engine); ours 40 vs our fp16 87.6 (54% overhead,
Python engine). Closing tg structurally needs graphs (blocked on
status.tolist code-branch + seqlens int); parked after trims.
Prefill store trims (2026-09-30, `277efe8`): full-group zero-gate
(128/128 rows overwrites skip the slot zero_ -- numel is shape-only;
partial groups keep it) + batched touched-slot eref refresh (one WHT
over all touched slots vs one per group; per-row independent so
bit-identical). Twins + suite green, KLD identical, same-top 100%,
needle 4/4 (44s). pp 5.2 -> 5.2 (NEUTRAL), tg 40.2 -> 40.7-40.9
(noise; decode path untouched). Kept as foundation (fewer launches
+ less traffic, zero risk), not as a gain. GOTCHA that cost an
hour: `w[_ess].copy_(x)` with tensor _ess is a silent no-op
(advanced indexing returns a copy; copy_ writes the discarded
temporary). Use indexed assignment `w[_ess] = x`. The eref twin
caught it; without that twin it would have shipped silent staleness.
pp restructure SHIPPED (2026-09-30, `476eb94` + `3ce676f`):
seal-direct-from-rows fast path with per-group split (eligible
groups commit seal-direct; partial/sink/reset/pressure groups go
legacy; shared tail unchanged). Twin-tested bit-identical (aligned
+ 31+1-split + reset cases) incl. rev-resolved copy/deferred tests.
Box gates: suite 35/35, KLD same-top 100%, needle 4/4 (38s).
pp 5.2s -> 4.0s TWICE (+30%, 71% of Bee 2.8s); tg untouched (40.8).
PTIMES: fast_E31 x32 calls (31 groups fast + 1 partial legacy --
the real 4032+64 box pattern), fast_E0 x16 (64-row tails, legacy);
loop-or-fast 1129 -> 471ms, seals 791 -> 469ms (fastseal 319 +
legacy 150), total 1946 -> 1015ms.
Median print wobble (1e-6 legacy vs 2e-6 fast, same-top 100%,
mean/max identical, parity asserts x4096 green): PROVEN external --
fast and legacy run the SAME batched seal kernel on the SAME values
(records bit-identical by construction), and parity asserts prove
per-step cache exactness on box; the last print digit at 1e-6 is
harness noise. Not pursued.
Two test-assumption fixes (both over-specification, not weakening):
copy_page slot indices now resolve via rev maps (slot ASSIGNMENT
order legitimately differs: fast consumes no staging slots);
deferred test is a 2x2 fast/defer matrix (seal-calls assert only on
the legacy pair -- its subject is deferral; records/kv match across
all four). Staging content of sealed groups is dead by construction
(no reader: copy rebuilds from records, spills restore from rows).
Remaining pp (~1.2s gap): seals Sinkhorn torch elementwise
(~470ms) is the next target (fused Triton seal kernel); loop
remainder + WHT + evict after.
Chunk-8192 recipe re-tested post-fast-path (2026-09-30): single 8k
forward pp 4.0 -> 3.7-3.8s (+5%, holds), KLD BETTER (fewer chunk
boundaries: median 1e-6/mean 1.8e-5/max 5.05e-4, same-top 100%),
peak 14.6GB (fits 24GB). Recipe stands: biggest chunk that fits.
torch.compile seal experiment (2026-09-30, REJECTED in 20 min via
standalone probe `eval/_probe_seal_compile.py`, no model load):
inductor G232 inexact (maxdiff 16.0!) + K4V2 Dynamo failure, and
the prize was only +17% on ~50ms of seal math (~+0.2% pp). The
early-break data-dependent branch graph-breaks; hand-Triton seal
remains possible but unaudited for ROI (device math is only
~60-100ms/chunk -- the gap is stalls, not math).
Honest pp anatomy (fp16-profile diff): kvarn-specific gap 0.45s =
seals-math ~0.06 + store-syncs ~0.1 + host dispatch saturation +
idle ~0.25. Piece-wise host cuts are now +1-2% each (diminishing).
Structural options: prefill CUDA graphs (sync-free region needed;
pages-loop ints + status branch block it today), fused tail QK
(small, clean, twin-testable, queued), or accept ~74-80% pp
(engine GEMM diff is out of scope under the KVarN-only mandate:
our fp16 itself is 3.3s vs Bee 2.8s).
K4V2 vehicle check @8k (2026-09-29, same flags): pre 5.0s (same),
tg 34.9 (vs 43.6 K4V4, -20%), KLD median 2.7e-4 / mean 5.2e-4 /
max 7.8e-3 (270x K4V4 median), same-top 100%. K4V4 stays the
vehicle (confirmed by data, not just policy).
K4V2 quad fast path (2026-09-29, serve V unpack, twin
maxabs 9.2e-5 vs torch): correct but NO tg gain (34.7 vs 34.9) --
serve is compute-bound (SIMT dots), not unpack-bound, so unpack
traffic is invisible. K4V2's -20% lives elsewhere (not in serve
unpack; mechanism open, not goal-blocking). Quad path stays
(correct, exercised by twin; helps if K4V2 revives). K4V2 parked:
slower AND 270x KLD despite author's pick; its memorandum value
is VRAM (smaller v payload), not speed.
(The 16k/64k pp rows that lived here are superseded by the versioned
pp baseline table above; 128k below is the latest available.)
| 131072 | 198.1s, ~662 (chunk 4096; swap pressure) | 157.9s, ~830 (chunk 4096, warm-restart; faster than fp16: 4-bit cache stays resident while fp16 swaps) | **69.5s, 1887** | 44% |

128k protocol (2026-10-02): full KLD twice is too slow AND the
guard kills valid runs (transient dips under min-free at 20GB+).
Warm-restart instead: short run (`-dec 32`, compiles everything)
then restart + ONE measured run. Chunk 4096 (8192-chunk OOMs:
activations + 20GB cache don't fit 24GB). Peak 23.1GB -- 0.9GB
headroom (K4V2's memorandum VRAM value lives here, parked).
KLD@128k: median 1e-6 / mean 5e-6 / max 6.9e-5, same-top 100%.
tg slope 64k->128k: 47.0 -> 35.2 (-25% for 2x ctx, O(n) serve).

Probe-arm per-layer (us, imageless): store 245.8, qwht 11.6, eref
44.7, serve 97.2, stats 68.6, mask 26.8, tail 412.4, merge 112.9,
fmerge 17.9, ftail 50.0, fgat 34.0, full arm 805.7.
Run-variance rule: first-run-of-day tg reads low (38.3 @8k, 34.1
@16k, 27.7 @64k, 22.4 @128k) vs warmed repeats (43.0 @8k; 42.1/42.4
@16k; 36.3 @64k; 28.7 @128k) with fp16 and KLD digits bit-identical
across all runs -- always warm up / re-run before comparing tg.
Incremental-eref A/B @8k (2026-09-28, commits 30a81d9 -> 349d24a,
fused in-kernel eref write-through): warmed current 41.6-42.1 (4
runs) vs pre-eref base 43.3-43.8 (2 runs, same warm box), KLD
digits bit-identical, parity asserts exact, VRAM identical. Path:
40.4 (host slot refresh: +1 sync + 1 launch) -> 41.8 (fused, sync
and launch removed). Residual -3.5% has no isolated mechanism:
store +0.6us/layer and serve +0.0us/layer by micro A/B (raw kernel
+ alternating serve, 200 iters), prefill equal, gap persists under
expandable_segments, fp16 drifted -0.8% in-window (latency-bound
regime: 64% SM vs 99% fp16). RESOLVED by the Sinkhorn cut
(0b1b9c7, see below): the residual was prefill pool pollution,
not decode code — fewer prefill launches/syncs leave a cleaner
pool, and tg recovered to base parity (8k 43.6-43.8, 64k 36.1
vs 36.4) under the default allocator.
64k A/B (2026-09-29, same box/flags/harness, current b24efff vs base
1f29695): default allocator base 36.4 vs current 27.6 (-24%);
expandable_segments base 37.0 vs current 35.9 (-3%). The 8k residual
likewise shrinks under expandable (current 42.9 vs ~43.5 base, ~-1.5%).
Mechanism: pool fragmentation, not kernel code (store +0.6us/layer
and serve +0.0us/layer by micro A/B; prefill identical 73.6/73.7s;
KLD identical). The persistent eref buffer + changed temp cadence
shift the pool layout; under default allocator at near-full VRAM the
decode temps fragment into retry/split storms (uniform ~1000x op
inflation, zero net growth, zero alloc retries). Rule: use
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for 64k+ runs (and
note it); code-level fragmentation hygiene (preallocated decode
temps) is follow-up work.
Env pitfall (2026-09-29, cost real debugging time): cmd.exe `set
VAR=1 && ...` bakes a TRAILING SPACE into the value (`'1 '`), so
every `== "1"` gate silently fails. All direct-chain probe/parity
runs that day ran torch fallbacks (the phantom "0.1 tok/s 64k cliff",
phantom OOMs, vacuous parity). Bat files (`set VAR=1` at line end)
are clean, as is `set VAR=1&&`. Verified via in-process ENV print.
Affected: only ad-hoc probe/parity runs; all bat-driven table
numbers and in-process-env test twins stand.
Test-hygiene gap (pre-existing, not a code regression): many
kvarn tests assume TRITON/IMAGELESS unset globally and fail when
the box pytest sets them (11 failures incl. overlay/fused-serve/
prefill twins that build the image path: imageless leaves _img_k
None; CPU tests hit triton branches with CPU tensors). Proven
pre-existing by identical failure sets pre/post Sinkhorn+split;
env-unset runs are 33/33 green incl. the new split twin. The
twins need per-test env management (like the fused-store twin
already does); queued as test-only cleanup, no prod impact
(KLD same-top 100% throughout).
Spec B integrated 2026-09-28 (commit b24efff, rebased from
/tmp/impl-wht with offset; only conflict was the helper insert site
next to _eref_wht): prefill store WHT inplace-on-fresh-temp (n <=
65536 rows, torch fallback beyond), _group_block + refresh-rot via
_triton_wht_head_maybe_triton; twins: prefill-4096 chunks + WHT
shapes + eref-buffer compares. On-box: 31 passed, PARITY=1 green,
KLD digits bit-identical, same-top 100%. Warmed perf perf-neutral
at 8k (prefill 8.3/8.7s, 984/938 tok/s; decode 41.7/42.0) -- the
row-WHT is not the prefill bottleneck (dequant/seals/image
dominate); value is alloc-hygiene + the inplace pattern for later
fusion. tg residual -3.5% carries over unchanged.

KV head-to-head (GB cache-only, 16 layers; ours measured per-tensor,
Bee `kv_resident_bytes`; Bee `llama-bench -m Qwen3.8-27B-UD-Q4_K_XL.gguf
-p <ctx> -n 256 -ctk kvarn4 -ctv kvarn4 --kv-memory`):

| ctx (tok) | KV ours fp16 / q8 / q4 / kvarn4 (GB) | KV Bee kvarn4 (GB) | gap (MB) |
|-----|----------------------------|---------------|-----|
| 8192 | 0.55 / 0.29 / 0.16 / 0.24 | **0.18** | +60MB |
| 16384 | 1.09 / 0.58 / 0.31 / 0.38 | **0.33** | +53MB |
| 65536 | 4.31 / 2.29 / 1.21 / 1.26 | **1.21** | +50MB |
| 131072 | 8.61 / 4.57 / 2.42 / 2.44 | **2.38** | +60MB |

Bee tg is flat 44.0 at all lengths; ours drops 43.0 -> 42.3 ->
36.3 -> 28.7, so the gap widens with length (98% -> 96% -> 83%
-> 65%) -- the online serve grid grows with ctx (position-masked
over all groups) while Bee's fused dequant-attention does not. Bee
pp is 94% of its fp16 (2844 vs 3025 @16k); ours is 39% (997 vs
2532) -- our prefill pays row-WHT + Sinkhorn seals + per-layer
syncs that Bee never does. VRAM is within 4% at every length; the
remaining gap is staging +9 and exact/stash +42MB @16k.
Why no allocator peaks in the metric columns: the old `(15.5GB)`
style numbers were decode-peak allocator totals (weights + cache +
temps + full-length fp32 scoring logits), not KV cache -- they
prove end-to-end fit and nothing about cache parity. Fit record
(ours kvarn decode peaks): 10.6 / 11.3 / 15.5 / 21.1GB
@8/16/64/128k. Cache truth is the KV table above.
Components: Bee 128k payload K+V 2.35GB, staging 25MB, exact ~25MB.
Ours 128k measured (`alloc-only`, seconds): records 2353.4MB, exact
50.4MB, staging 33.6MB. Plain q4 is 60-80MB SMALLER than KVarN-4 at
every length (0.16/0.31/1.21/2.42 vs 0.24/0.38/1.26/2.44GB) -- the
intercept price of the staging+exact window; KVarN's argument over
q4 is quality-per-bit (vLLM #46613 gospel), not size.
Swap verdict @128k (measured): q8-ref run gives pp q8 87.7s/1494
tok/s, kvarn 173.9s/754, tg q8 46.6 / kvarn 28.1, KLD kvarn-vs-q8
same-top 100% with digits identical to the fp16-ref KLD, min-free
2461MB -- comfortable, no swap. The fp16 648 tok/s was
swap-throttled (q8 2.3x faster at the same ctx); kvarn itself ran
27% faster just from sharing the machine with the lighter ref
(173.9s vs 219.4s). So `-ref q8` is confirmed policy at 128k.
Min-free record (guard, mandatory with every run): alloc-only
128k probe hit 23MB -> guard KILL exit 2 on its first day of duty
(all four totals already captured); q8-ref KLD 2461MB; Bee kvarn4
@128k 4397MB (18GB weights + 2.4GB KV, never near swap).
Equal weight footing @128k (tok/s; IQ2_XXS 7.27GB/2.13bpw
approximates our EXL3 dir 7.88GB, vs Q4_K_XL 17.9GB/5.14bpw;
27.32B params):

| path | pp (tok/s) | tg (tok/s) | KV (GB) | min-free (MB) |
|------|------------|------------|---------|---------------|
| Bee Q4_K_XL kvarn4 | 1887 | 44.0 | 2.38 | 4397 |
| Bee IQ2_XXS kvarn4 | 1882 | **80.9** | 2.38 | 13987 |
| Bee IQ2_XXS f16 | **2068** | **88.4** | 8.59 | 8069 |
| ours 1.40bpw kvarn4 | 754 | 28.1 | 2.44 | 2461 |
| ours 1.40bpw fp16 | 648 (swap) | 47.5 | 8.61 | n/a (old guard) |

Two things fall out. First, Bee f16 @128k exists after all --
8.59GB resident cross-validates our 8.61GB probe to 0.2%. Second,
decode is weight-bandwidth-bound and the weights set the ceiling:
Bee-IQ2 streams 7.27GB/token at 80.9 tok/s = 588GB/s (~60% of
4090 peak, credible); Bee-Q4 44.0 x 17.9GB = 788GB/s. Ours moves
~5GB/token at 28.1 tok/s = ~140GB/s -- overhead-bound by ~4-5x,
which corroborates the subagents independently (launches, syncs,
serial gc-reduce -- not math). The tg prize is ~3x before
bandwidth even binds.
Bee f16 @128k does not fit 24GB (18GB weights + 8.6GB KV ->
offload crawl, run killed at 39MB free); ours fits end-to-end
on small 1.4bpw weights. Gates this round: smoke Paris HIT +
needle 4/4 in 66s; bbeh-mini --limit 3 0/3 clean exit (jsonl utf-8
fix); smi showed 0 used before every run and after every run, min
free ~1GB (128k kvarn prefill peak 23.1GB).

Agent findings (2026-09-29, 3 parallel read-only subagents,
convergent): our body serve is O(n) per step AND serially reduced --
serve grid `(kvh, gc)` with gc growing in n plus a single-warp
`for b in range(NB=gc)` combine loop (97us@8k -> ~1.5ms/layer@128k).
Bee: fixed-64 splits, parallel reduce, 3 launches, zero syncs. Our
acc buffers also grow O(n) (~65MB/layer @128k); the tail's exact-WHT
refreshes fully every step (~44us waste); prefill pays the torch
store path (fused store_row is T==1-only) with ~300-500 Sinkhorn
micro-launches and ~120 syncs per layer per 4096-chunk.
Proposals, best (fast, easy, performant) first:

1. Incremental eref + kill per-layer syncs (fast, easy): WHT only
   the written exact slot, drop `flag`/`status` tolists to device-side
   checks. ~50-100us/layer/step + unstalls the pipeline.
2. Split-parallel body + parallel combine (medium, biggest tg gain):
   fixed-64 splits, Q_TILE>1, parallel reduce. Restores flat tg.
3. Prefill WHT via existing `kvarn_triton_wht_rows` (fast reuse):
   kills ~30-40 launches/layer/chunk, ~30-40% of the kvarn pp tax.
4. Tensorize `_store_rows` loop (medium): slot-gathered scatter,
   zero `.item/.tolist/.any` on steady path. ~15-20% pp wall.
5. Fuse tail into body kernel (harder): -4 launches/layer/step.
6. Lazy/fused Sinkhorn seals (hardest pp item): seal only
   evicted/served groups; removes the 2.5x-vs-fp16 floor.
7. Streaming acc buffers (hygiene): fixed-size, never
   materialize per-chunk acc; unlocks length, minor speed.
8. SWA exact-margin + slot tightening (small): tens of MB VRAM.

Plan to goal (2026-09-29, refreshed): KV-cache VRAM about equal
or better than BeeLlama under KVarN quant, same or better pp
and tg. Status: Spec A done (incremental eref + fused
write-through, parity-proven, tg at base parity); Spec B done
(prefill WHT inplace, exact but perf-neutral — WHT is NOT the
prefill bottleneck); harness fixed (phase reorder + ref free +
no-grad; 64k KLD green, same-top 100%); fragmentation diagnosed
(default-alloc gaps collapse under expandable); Sinkhorn cut done
(pp 8.3->6.7s); deferred-batched pressure seals done (pp 6.7s ->
5.05s@8k, 62s -> 48.2s@64k, 29 seal-calls/chunk -> 1 batched;
KLD same-top 100%, mean +9%@8k/+35%@64k budgeted).
Remaining gaps need structural work, not tweaks (post-cut
Kineto: GEMM 65% shared; kvarn-specific is a <5%-each long tail
of launch overhead): CUDA graphs for prefill chunks (kill all
launch overhead), tensorized serve (MMA + reg-cap + fuse
combine), VRAM window tightening (staging/exact).
Decode-step Kineto @64k (26ms wall, 17ms device): serve 4.0ms
(23%) + combine 1.2ms (7%, serial NB=gc loop) + tail 0.8ms;
~9ms wall is bubbles (host gaps/syncs/allocator), not kernels.
Hoist loop-invariants evaluated and SKIPPED (loads not dominant;
dots+MMA shape + bubbles are). Priority: (1) launch/bubble
reduction via per-layer fusion (serve+combine+tail+merge),
(2) MMA reshape of QK dots (N=QPAD<16 falls back to SIMT),
(3) prefill graphs.
Next order (evidence-driven, one cut per commit):
1. Prefill breakdown (measure first): attribute kvarn prefill
   (8.3s@8k / 73.6s@64k vs Bee 2.8s/28.1s) across dequant /
   Sinkhorn seals / store-WHT / remat / fixed costs with a Kineto
   probe; attack the biggest piece (seal inventory suspects
   Sinkhorn: 16 iters x K,V x ~31 groups/chunk).
2. Split-parallel body serve (tg @length): serve grid (kvh, gc)
   is O(n) + serial combine (ours drops 43->36 with length, Bee
   flat 44); shard body over fixed token blocks, parallel combine.
   RESULT 2026-09-29 (reverted 312eefb): split S=2048 (grid
   (kvh,ns), ns<=32) is SLOWER at both lengths (8k 31.3 vs 43.6;
   64k 30.6 vs 36.1), KLD-identical. Lesson: serve is
   throughput-bound on total work, not parallelization-starved.
   Bee's flatness = less work per step + better kernels (MMA,
   reg-capped, 3 launches). Next lever is kernel efficiency
   (tensorize serve), not work partitioning.
3. Fragmentation hygiene: preallocated decode temps (or
   expandable_segments as 64k+ standard).
4. VRAM: close imageless gap 53MB@16k (staging 9 + exact/stash
   42 + payload 4): window exact 8->6, staging pressure.
Success bars: flat-with-length tg; kvarn pp within 20% of our
fp16 pp (Bee is at 94% of its); imageless <= Bee resident at
every length. Gate every cut: CPU suite, real PARITY=1
(`set VAR=1&&`, never `set VAR=1 &&` — trailing space kills
every =="1" gate), KLD-8k + KLD-64k identical, warmed pp/tg
@8k/64k, VRAM, table, commit.
128k swap note: fp16 pp collapses 2008 -> 648 tok/s from 64k to
128k while prefill peak hits 22.1GB -- swap spillover, not compute.
Policy: at 128k the reference is q8 (`-ref q8`; KLD then reads
kvarn-vs-q8, labeled as such), fp16 gospel stays at <=64k where
it fits. q4 figures now probed alongside (`CacheLayer_quant`
takes 2-8 bits); `_probe_vram.py` gained `alloc-only` (seconds
even at 128k; live slots read 0 without populate).
Peak+free rule (mandatory): every run report carries smi
free-before, min-free-during (guard prints START/min_free/DONE),
free-after, plus harness allocator peaks. A falling min-free
across phases is the swap early-warning; <100MB kills the task.
Operating model (speed without chaos): the GPU is serial -- one
run at a time, smi-guarded, never parallelize runs. Analysis
parallelizes: each round, subagents dissect the next target
(serve-path audit, Bee-structural-compare, seal-path inventory)
while the GPU validates the previous cut. Then implement the top
cut only (one cut per commit keeps KLD attribution clean), gate
(CPU, parity, KLD-8k, warmed tg, VRAM), table, commit, push.
K4V4 stays the comparison vehicle; KLD same-top 100% + direct
arm-vs-torch validation on every dispatch change; reasoning
benches last. Implementation parallelizes across disjoint scopes
only (no shared files without explicit coordination): one agent per
isolated worktree, each returns a unified diff + CPU suite result,
no commits; integrator applies sequentially (A before B), GPU-gates
each, one cut per commit.

### Subagent findings, round 1 (2026-09-29, read-only, convergent)

Serve audit: per-step serve grid `(kvh, gc)` is O(n) FLOPs AND
O(n) record traffic (`_kvarn_online_serve_kernel`, 8x16-row
iters/chunk); per-q-head combine loops `for b in range(NB=gc)`
serially (single-warp); `_ov_serve_acc` + m/l grow O(n)
(~4MB/layer@8k -> ~65MB/layer@128k, realloc on gc change);
merge reloads O(n) m/l over `arange(GCPAD)`. Ruled O(1): tail
(R<=sink+tail_eff), store (grid `2*kvh` + 1 status sync),
evict (resident-only scan 1x/128 rows).
Bee-compare: Bee splits fixed 64 tok in parallel + parallel
combine (ours: growing grid + serial reduce); Bee m16n8k16
MMA + unpack2 pair-loads, reg-capped (ours: SIMT-ish tl.dot +
per-element unpack); Bee same-kernel masked tail fallback
(ours: 3 extra launches + full `Ew` refresh every step);
Bee 3 launches total, device-side mask-skip, zero syncs (ours:
~6/layer/step + Python aranges + 2 tolist syncs).
Seal inventory (@4096-chunk/layer): row-WHT fwd+inv ~14-20
launches x3 passes; Sinkhorn 16 iters x(K,V) ~300-500 launches
for ~31 groups; `_store_rows` loop ~4-5 DtoH per group (~120
syncs); serve dequant+unpack remat per chunk; overlay per-group
syncs + indexed assigns. Fused `store_row` is T==1-only; prefill
pays the full torch path.

### Tech watch: TIRx-Harness (2026-09-30, evaluated, NO-GO)

MLC's TIRx-Harness (blog 2026-09-29: thin PTX-level foundation +
kernel zoo + sync/race/numerical analyses + KCoral benchmark
server for agentic kernel dev, 2.94x/6.84x on KDA) does not fit
this project: our bottleneck was never lowering unpredictability
(the one occurrence, SIMT-vs-MMA, died in one A/B); our
measurement pain was cold-cache/warmup, not shared-GPU contention
(solved by protocol v2, no remote queue wanted on a dedicated
GPU); the zoo is TIRx-specific (nothing transfers to our bespoke
Triton kernels); our bugs are host-side logic caught by twins in
seconds (our equivalent of their GPU-less numerical sim).
Adopting it would mean a new PTX-level toolchain on Windows for
kernels deeply embedded in exllamav3's torch runtime -- high cost,
no payoff against the current bottleneck list (host orchestration
+ torch-structured seal math). Revisit only if serve ever needs a
from-scratch sub-Triton rewrite (nothing on the roadmap requires
it). Transferable meta-lesson (already our practice): shape the
environment so agent budget goes to the optimization, not to
resolving uncertainty around it (protocol v2, PTIMES, twins,
parity asserts, this ledger).

### Spec A: incremental eref + sync-kill (implement first)

Current: full exact refresh `dispatch.py:184`
(`Ew=wht_rows(exact_v.float())`, dead slots included); syncs at
`dispatch.py:167` (`int(cache_seqlens[0])`),
`kvarn_triton.py:1810` (`int(flag[0])`), `:482`
(`status.tolist()`), `kvarn.py:1343` (`bool((~ev).any())`,
skipped via `_tail_exact_certain` `:1340`, set `:1010`,
cleaned `:1359`); parity gates at `kvarn_triton.py:56`,
`kvarn.py:1609/1878/2021`, imageless gate `:330`.
Recipe: cache `Ew` per layer as `_ov_eref_w (E,128,kvh,hd)`
fp32 (shard like `exact_k/v`, `kvarn.py:978`; rev
`_exact_rev (G,) i64` `:1021`, valid `(G,) bool` `:983`);
each step WHT only touched slots (from `status[1]` `:482`) and
`copy_` into cache; serve reads cached `Ew` (`dispatch.py:187`).
Invalidate (clear rows or drop cache) on torch-store clear
`:1592`, evict release `:1747`, copy-page lapse `:2386`,
seal/reset `:1646`. Hoist seqlens to caller int; device-side
flag with async check next step; 1-step-delayed status read;
keep KLD-8k-green-gated delay (fail-closed `:191` stays
synchronous until green). Parity twin: under PARITY=1 run full
`wht_rows` vs incremental `Ew`, `assert torch.equal` (+flag/
status equality), mirroring `:1609`/`:1878`/`:2021`.
Validate: CPU twin (incremental vs full Ew over
evict/copy/reset matrix) -> probe-arm parity maxdiff 0.0 ->
KLD-8k flag-on identical -> tg/VRAM re-measure.
Gain ~1.2ms/step @28 layers (+5-8% tg @8k, more at length).
Risks: stale `Ew` (silent wrong body -- clear-on-write +
parity twin); deferred fail-closed by one step (keep sync
until KLD green, then delay).

### Spec B: prefill WHT via triton kernel (implement second)

Current: fwd torch `kvarn_wht_head` (`kvarn.py:584` via
`kvarn_hadamard_128` `:539` + `kvarn_wht_slices` `:558`);
prefill store fwd `kvarn.py:1615` on `stack(...)`; T==1-only
gate `:1563` (`rows_k.shape[0]==1 and not is_swa and triton`);
single-row torch path `_store_row_single` `:1478` (called
`:1625`); multi-row loop `:1637`. Kernel
`kvarn_triton_wht_rows(x, head_dim, inplace=False)` (`:370`;
inplace asserts fp32-contig `:390-392`; flatten `:393-399`;
grid `[(n,)] num_warps=1` `:402`). Fused store
`kvarn_triton_store_row` (`:407`; inplace on stacked temp
`:440-443`). Entry `update_kv` `:2304` / `update_kv_direct`
`:2333` fan into `_store_rows` (`:2329,2357`).
Inverses: `_group_block` `:1921` (sealed `:1931-1932`, open
`:1941-1942`); `_refresh_into` `:2090` (rot batch
`:2156-2157`, `wht_done` slice-only `:2110-2114`,
open-serve `:2126-2139`); tail audit `:1350-1353`.
Recipe: under triton, replace torch fwd with the kernel for
ALL T (drop the shape gate for WHT only; keep
`store_row` T==1 gate); pass stacked `(T,2,kvh,hd).float()`
contig, `n=T*2*kvh`, `inplace=True` on the fresh temp only
(never persistent fp16 stage/image); route group-block,
refresh-rot, tail-audit through the kernel out-of-place;
keep `wht_done` branch. Parity invariant: reference always
out-of-place torch on pre-transform `stacked` (`:1607,1610`);
keep asserts `:1609,1878,2021`. Twins: extend
`test_wht_rows_matches_torch_head` (`:163`) to T=4096
(kvh 2..8, hd 128/256/512) and serve-match test (`:231`,
`:240,256`) to length-4096 chunks.
Validate: CPU suites (env unset) -> PARITY=1 KLD-8k
bit-identical (any assert trip = reject) -> warmed pp A/B
(TRITON on/off) + launch count.
Gain: low-single-digit % of the pp tax (Sinkhorn dominates;
WHT swap saves allocs+launches only). Risks: grid scale
(proven to 1500x128; 16-65k rows may regress -> row-cap +
torch fallback); inplace aliasing (restrict to fresh temp,
parity input pre-mutation).

Cache-only accounting (MB; 16 cached layers; ours via
`eval/_probe_vram.py` per-tensor bytes, Bee via `llama-bench
--kv-memory` `kv_resident_bytes` + component fields; `~` = summed from
measured components, direct flag-on run queued):

| store | ours 8k (MB) | Bee 8k (MB) | ours 16k (MB) | Bee 16k (MB) |
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

| ctx (tok) | preset | median | mean | max | fp16 pre (s) | kvarn pre (s) |
|-----|--------|--------|------|-----|----------|-----------|
| 400 | kvarn4 | 0.000043 | 0.006330 | 0.380000 | — | — |
| 400 | kvarn5,kvarn4 | 0.000039 | 0.000406 | 0.006960 | — | — |
| 8192 | kvarn4 | 0.000000 | 0.000011 | 0.000329 | 5.0s | 144.6s* |
| 8192 | kvarn5,kvarn4 | 0.000000 | 0.000005 | 0.000066 | 4.9s | 149.8s* |

\*: pre-optimization numbers (per-group Python loops); see 1.40bpw
history above for the optimized path. Quality digits match across
checkpoints.

### Qwen3.8-Flash-Next 3.05bpw (Qwen4Exp, MoE+QSA, `-mcl 40`)

| ctx (tok) | preset | median | mean | max |
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
