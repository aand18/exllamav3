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
| 5 | fp16 cache, 131k | 131072 **FP16**, 2 slots | **496** | 20.0 GiB | 43.4 GiB | ≈ −30…40 % (predicted: 128 experts above the Q2 floor) | accuracy purist only; +10 GB RAM |
| 6 | fp16 cache, 262k | 262144 FP16 | **does not fit** | needs 27.4 GiB at split 512 (all experts on CPU) and still −4.9 GiB | — | — | **impossible on a 24 GB card** |

Row 6 is the arithmetic the operator asked about directly: at 262 144 the FP16 KV cache alone is
~15.8 GiB; with the non-expert trunk it needs 27.4 GiB **even with every routed expert on the CPU**,
so no split makes it fit. 262 k + FP16 is not a tuning choice on this card.

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

# 5. fp16 accuracy, 131k (expect +10 GB RAM, ~1/3 slower)
cache_mode: "16,16"
cpu_moe_split_experts: 496
```

The split must move with the cache mode: **Q4 at 262k needs 416, not 378** — with 378 the engine
refuses (`Insufficient VRAM in split for model and cache`), the same way it refused 262144 at 376.

## D. What is measured vs predicted

- **Measured**: row 1's corpus throughput, its per-bucket live numbers (section A), the split floors
  for 262k Q2 (378, engine-verified) and the Q2 131k floor (368 ran in T10), and every fit number via
  `size_check`.
- **Predicted**: rows 2, 3, 5's decode deltas, from the split curve measured at 262k (392 → 378 =
  +7.8 % for 14 experts). Each needs one `perf.py` ladder (~5 min) to become a measurement, and the
  box is serving live traffic right now.
