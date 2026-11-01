# T14 — deployment config catalog (swap targets)

Everything here is measured or derived on this box: production `3.18bpw_h6_ng8`, DFlash2 BF16 window 4,
win-perf 1.6.5 with `EXL3_MOE_THREAD_PRIO=1`, `max_batch_size 2` unless stated. "Split floor" is the
lowest `cpu_moe_split_experts` the engine will load (lower = more experts on the GPU = faster, less
VRAM); the floor is set by the card, and it is what consumes the headroom.

## A. The deployed config, bucketed by prompt length (live traffic)

Parsed from 289 real requests in the server log of 2026-10-31/11-01 — the operator's own harness, so
it is the workload that matters. Concurrency is in play (several jobs in flight), which is why the
p90 is much better than the median.

| prompt bucket | n | decode T/s median | p90 | prefill T/s median | draft acceptance |
|---|---|---|---|---|---|
| <16k | 3 | 51.7 | 51.7 | 987 | 79.8 % |
| 16–64k | 73 | 52.4 | 62.4 | 574 | 55.9 % |
| 64–128k | 93 | 52.3 | 63.2 | 666 | 50.6 % |
| 128–200k | 73 | 46.1 | 57.2 | 468 | 44.3 % |
| >200k | 47 | 53.1 | 63.0 | 439 | 56.7 % |

Decode is nearly flat with context (46–53 T/s median) because the drafter's rounds dominate; prefill
falls with length (987 → 439 T/s), which is the predictable part. Acceptance falls with context,
which is where the drafter earns less.

## B. The catalog

`size_check.py --draft DFlash2` picks the split floor; resident VRAM is therefore pinned to the card
(~20.5–21.5 GiB) in every row — the difference is where the trunk lives, not how much is used.

| # | config | cache | split floor | resident VRAM | host arena | decode (measured or predicted) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **deployed** | 262144 Q2, 2 slots | **378** | 21.4 GiB (763 MB free at the floor) | 33.1 GiB | **58.54 tok/s corpus**; 46–53 T/s live | best all-round |
| 2 | single slot | 262144 Q2, 1 slot | 378 | 21.4 GiB (−0.1) | 33.0 GiB | +1–2 % single stream (predicted, less batching) | use when one client only |
| 3 | **accuracy at 262k** | 262144 **Q4**, 2 slots | **416** | 20.2 GiB | 36.4 GiB | ≈ −19 % (predicted: 38 experts above the Q2 floor at 0.5–0.6 %/expert) | if cache accuracy matters more than speed |
| 4 | fastest decode, half context | 131072 Q2, 2 slots | **368** | 20.9 GiB | 32.2 GiB | highest of the set (fewest CPU experts); 131 k ceiling | for short-prompt, latency-critical work |
| 5 | fp16 cache, 131k | 131072 **FP16**, 2 slots | **~400** | 20.4 GiB | 34.2 GiB | ≈ −10 % (predicted: 22 experts above the Q2 floor) | accuracy at half context, cheap |
| 6 | **fp16 cache, 262k** | 262144 **FP16**, 2 slots | **~440** | 20.4 GiB | 37.6 GiB | ≈ −26 % (predicted: 62 experts above the Q2 floor) | full context, full precision cache |

**Correction, 2026-11-01.** An earlier version of this file said 262 k + FP16 was impossible, from a
15.8 GiB cache estimate. The estimate was wrong: it scaled the measured 2-bit figure linearly to 16
bits, but a QSA layer stores **fp16 indexer planes** (raw key per token, pooled key per 4-token
block) that do not shrink with the cache width. The true figures are 640 B/token/layer at Q2, 896 at
Q4 and **2368 at FP16** (2048 B of K/V + 320 B of planes) over 12 full-attention layers, i.e. **6.94
GiB of KV at 262 144** — 7.16 GiB with the recurrent state. The cache arithmetic now comes from the
engine's own shapes (`bench/size_check.py`, `cache/quant.py` and `cache/qsa.py`), and the Q2 point
still reproduces the measured 2.093 GiB exactly.

So 262 k + FP16 **fits**, at split ≈440 instead of 378: 62 more experts on the CPU, about −26 %
decode. It is a real option, not an arithmetic impossibility.

## C. Swapping

Only `config.yml` fields change; restart is `start_tuned.ps1` (31 s load). Nothing else in the
deployment (env, drafter, port, key) needs to move.

```yaml
# 1. deployed (fastest all-round)
cache_size: 262144
cache_mode: "2,2"
cpu_moe_split_experts: 378
max_batch_size: 2

# 2. single stream
max_batch_size: 1

# 3. accuracy at 262k
cache_mode: "4,4"
cpu_moe_split_experts: 416

# 4. fastest decode, 131k context
cache_size: 131072
cpu_moe_split_experts: 368

# 5. fp16 accuracy, 131k (about −10 % decode, +2 GiB RAM)
cache_mode: "16,16"
cpu_moe_split_experts: 400

# 6. fp16 accuracy at the full 262k (about −26 % decode, +4.5 GiB RAM)
cache_mode: "16,16"
cpu_moe_split_experts: 440
```

The split must move with the cache mode: **Q4 at 262k needs 416, not 378** — with 378 the engine
refuses (`Insufficient VRAM in split for model and cache`), the same way it refused 262144 at 376.

## D. What is measured vs predicted

- **Measured**: row 1's corpus throughput, its per-bucket live numbers (section A), the split floors
  for 262k Q2 (378, engine-verified) and the Q2 131k floor (368 ran in T10), and every fit number via
  `size_check`.
- **Predicted**: rows 2, 3, 5, 6 — both the split floor (from `size_check.py`, which the engine reads
  ~1–2 experts low at this card: it accepts 376 where the engine refuses it) and the decode delta
  (from the split curve measured at 262k: 392 → 378 = +7.8 % for 14 experts, i.e. ~0.56 %/expert).
  Each needs one `perf.py` ladder (~5 min) to become a measurement, and the box serves live traffic.
