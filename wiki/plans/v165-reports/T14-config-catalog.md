# T14 — deployment config catalog

Swappable configurations for the production `3.18bpw_h6_ng8` on this box, with DFlash2 BF16 window 4,
win-perf 1.6.5 with `EXL3_MOE_THREAD_PRIO=1`, `max_batch_size 2` unless stated. Every number is
measured here or derived from the checkpoint's own shapes.

**Split floor** is the lowest `cpu_moe_split_experts` the engine will load. Lower means more experts
on the GPU, which is faster, and that is what consumes the headroom. `size_check.py --draft DFlash2` picks the floor. It is why resident VRAM
sits at 20.5–21.5 GiB in every row: the rows differ in where the trunk lives, not in how much of the
card they use.

## Pick one

| you want | row | cost against the deployed config |
|---|---|---|
| best all-round at 262k for coding traffic (**deployed**) | 1 | — |
| one client only, less batching | 2 | +1–2 % single stream (predicted) |
| cache accuracy at 262k | 3 | about −19 % decode (predicted) |
| the lowest latency, 131k is enough | 4 | highest of the set, 131k ceiling |
| cache accuracy, cheap | 5 | about −10 % (predicted), +2 GiB RAM |
| cache accuracy at the full 262k | 6 | about −26 % (predicted), +4.5 GiB RAM |

## The catalog

| # | config | cache | split floor | resident VRAM | host arena | decode (measured or predicted) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **deployed** | 262144 Q2, 2 slots | **378** | 21.4 GiB (763 MB free at the floor) | 33.1 GiB | **58.54 tok/s corpus**; 46–53 T/s live | best all-round |
| 2 | single slot | 262144 Q2, 1 slot | 378 | 21.4 GiB (−0.1) | 33.0 GiB | +1–2 % single stream (predicted, less batching) | use when one client only |
| 3 | **accuracy at 262k** | 262144 **Q4**, 2 slots | **416** | 20.2 GiB | 36.4 GiB | ≈ −19 % (predicted: 38 experts above the Q2 floor at 0.5–0.6 %/expert) | if cache accuracy matters more than speed |
| 4 | fastest decode, half context | 131072 Q2, 2 slots | **368** | 20.9 GiB | 32.2 GiB | highest of the set (fewest CPU experts); 131 k ceiling | for short-prompt, latency-critical work |
| 5 | fp16 cache, 131k | 131072 **FP16**, 2 slots | **~400** | 20.4 GiB | 34.2 GiB | ≈ −10 % (predicted: 22 experts above the Q2 floor) | accuracy at half context, cheap |
| 6 | **fp16 cache, 262k** | 262144 **FP16**, 2 slots | **~440** | 20.4 GiB | 37.6 GiB | ≈ −26 % (predicted: 62 experts above the Q2 floor) | full context, full precision cache |

## Live traffic, bucketed by prompt length

Parsed from 289 real requests in the server log of 2026-10-31/11-01: the operator's own harness, so
this is the workload that matters. Several jobs were in flight, which is why p90 sits well above the
median.

| prompt bucket | n | decode T/s median | p90 | prefill T/s median | draft acceptance |
|---|---|---|---|---|---|
| <16k | 3 | 51.7 | 51.7 | 987 | 79.8 % |
| 16–64k | 73 | 52.4 | 62.4 | 574 | 55.9 % |
| 64–128k | 93 | 52.3 | 63.2 | 666 | 50.6 % |
| 128–200k | 73 | 46.1 | 57.2 | 468 | 44.3 % |
| >200k | 47 | 53.1 | 63.0 | 439 | 56.7 % |

Decode is nearly flat in context (46–53 T/s median) because the drafter's rounds dominate. Prefill
falls with length (987 → 439 T/s), which is the predictable part. Acceptance falls with context,
which is where the drafter earns less.

## Swapping

Only `config.yml` fields change. Restart with `start_tuned.ps1` (31 s load); env, drafter, port and
key stay as they are.

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

The split must move with the cache mode: **Q4 at 262k needs 416, not 378**. At 378 the engine
refuses with `Insufficient VRAM in split for model and cache`, the same way it refused 262144 at 376.

## Measured versus predicted

- **Measured**: row 1's corpus throughput, its per-bucket live numbers above, the split floors for
  262k Q2 (378, engine-verified) and Q2 at 131k (368 ran in T10), and every fit number through
  `size_check`.
- **Predicted**: rows 2, 3, 5 and 6, on both axes. The split floor comes from `size_check.py`, which
  reads 1–2 experts optimistic at this card: it accepts 376 where the engine refuses. The decode
  delta comes from the split curve measured at 262k, where 392 → 378 = +7.8 % for 14 experts, i.e.
  about 0.56 %/expert.
- Each prediction needs one `perf.py` ladder, about 5 min, and the box serves live traffic.

## Cache arithmetic, and the retracted 15.8 GiB

A QSA layer stores K and V at the requested width **plus fp16 indexer planes** (one raw key per
token, one pooled key per 4-token block). The planes keep their width whatever the cache does, so the
cost per token per layer is 640 B at Q2, 896 B at Q4 and **2368 B at FP16** (2048 B of K/V plus
320 B of planes), over 12 full-attention layers.

At 262 144 tokens that is **6.94 GiB** of KV for FP16, or 7.16 GiB with the recurrent state; Q2
costs 2.09 GiB, which matches the measured 2.093 GiB exactly.

**Retraction (2026-11-01).** An earlier version of this file called 262k + FP16 impossible, from a
15.8 GiB estimate. That estimate scaled the measured 2-bit figure linearly to 16 bits, which
multiplied the fixed-width planes by eight. The arithmetic now follows the engine's own shapes
(`bench/size_check.py`, `cache/quant.py`, `cache/qsa.py`), and row 6 is a real option at split ≈440:
62 experts on the CPU and about −26 % decode.
