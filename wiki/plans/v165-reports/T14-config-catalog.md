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
| cache accuracy at 262k, coding traffic (**deployed**) | 1 | — |
| the same at Q2, about 2 % more decode | 3 | +1.7 % decode, −0.75 GiB cache |
| one client only, less batching | 2 | +1–2 % single stream (predicted) |
| the lowest latency, 131k is enough | 4 | highest of the set, 131k ceiling |
| fp16 accuracy, cheap | 5 | about −10 % (predicted), +2 GiB RAM |
| fp16 accuracy at the full 262k | 6 | about −26 % (predicted), +4.5 GiB RAM |

## The catalog

| # | config | cache | split floor | resident VRAM | host arena | decode (measured or predicted) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **deployed** | 262144 **Q4**, 2 slots | **381** | 21.2 GiB (839 MB free at the floor) | 32.6 GiB | Q2 measured **58.54 tok/s corpus**; Q4 about −1.7 % by the expert count, corpus not yet re-run | cache accuracy at full context |
| 2 | single slot | 262144 Q4, 1 slot | ~381 | 21.1 GiB (−0.1) | 32.5 GiB | +1–2 % single stream (predicted, less batching) | use when one client only |
| 3 | **fastest at 262k** | 262144 **Q2**, 2 slots | **378** | 20.7 GiB (763 MB free at the floor) | 32.3 GiB | **58.54 tok/s corpus**; 46–53 T/s live | the previous default, +1.7 % decode |
| 4 | fastest decode, half context | 131072 Q2, 2 slots | **368** | 20.9 GiB | 32.2 GiB | highest of the set (fewest CPU experts); 131 k ceiling | for short-prompt, latency-critical work |
| 5 | fp16 cache, 131k | 131072 **FP16**, 2 slots | **~400** | 20.4 GiB | 34.2 GiB | ≈ −10 % (predicted: 22 experts above the Q2 floor) | accuracy at half context, cheap |
| 6 | **fp16 cache, 262k** | 262144 **FP16**, 2 slots | **~440** | 20.4 GiB | 37.6 GiB | ≈ −26 % (predicted: 62 experts above the Q2 floor) | full context, full precision cache |

## Live traffic, bucketed by prompt length

Parsed from **731 requests** across the two days the Q2 cache served with the deployed profile
(2026-10-31 14:19 UTC to 2026-11-02 05:30 UTC). This is the operator's own harness, so it is the
workload that matters. Several jobs were in flight, which is why p90 sits above the median.

| prompt bucket | n | decode T/s median | p90 | prefill T/s median | draft acceptance median | acceptance p10 |
|---|---|---|---|---|---|---|
| <16k | 34 | 54.8 | 65.5 | — | 64.7 % | 34.5 % |
| 16–64k | 125 | 42.0 | 63.5 | 1166 | 48.9 % | 35.5 % |
| 64–128k | 272 | 55.2 | 73.4 | 1437 | 43.1 % | 31.9 % |
| 128–200k | 232 | 56.9 | 72.9 | 1398 | 42.4 % | 30.7 % |
| >200k | 68 | 51.5 | 67.1 | 1645 | 39.5 % | 31.5 % |

Totals: 491,452 tokens generated, **301,207 of 760,980 draft tokens accepted (39.6 %)**, prompt
lengths 10 to 222,875, of which 300 are at 128k or more and 68 at 200k or more.

**Prefix cache**: 95–96 % of the large requests hit a cached prefix, and the median hit covers
99–100 % of their prompt. Prefill is quoted only over requests that actually processed 4096 new
tokens or more; the median rises with length (1166 → 1645 T/s) because the fixed first-token cost
amortises.

Two readings that look wrong and are not: the 16–64k bucket decodes slower (42.0) than the longer
buckets, and it is the only bucket with fewer than 96 % prefix hits, so those are first-touch
prompts with prefill interleaved. Acceptance falls with context for the drafter (64.7 % → 39.5 %),
which is where it earns least.

An earlier version of this table sampled the first 289 requests and read differently in the long
buckets (128–200k at 46.1 T/s against 56.9 here). The two-day sample above replaces it.

## Swapping

Only `config.yml` fields change. Restart with `start_tuned.ps1` (31 s load); env, drafter, port and
key stay as they are.

```yaml
# 1. deployed (cache accuracy at 262k)
cache_size: 262144
cache_mode: "4,4"
cpu_moe_split_experts: 381
max_batch_size: 2

# 2. single stream
max_batch_size: 1

# 3. fastest at 262k (the previous default)
cache_mode: "2,2"
cpu_moe_split_experts: 378

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

The split must move with the cache mode: **Q4 at 262k needs 381, not 378**. At 378 the engine
refuses with `Insufficient VRAM in split for model and cache`, the same way it refused 262144 at 376.

## Evidence

Every number above traces to a record:

- **Row 1 (Q4) and row 3 (Q2)**: `t14-q4-262k-corpus-20261102-085612` at split 389 and
  `t14-q2-262k-corpus-20261102-091413` at split 378, both `ok`, same session. Q4 at the server's
  dynamic split 381 has no harness row, because `spec_decode` runs static placement and refuses
  there (`t14-q4-262k-corpus-20261102-085450`, 25 s).
- **The live bucket table**: parsed from the server logs of 2026-10-31 and 2026-11-01 (731 requests),
  not from a harness run, so it carries no run id. The parse is reproducible from `logs/server-*.log`.
- **Q2 corpus 58.54 tok/s**: `t13-378-corpus-20261031-162616`.
- **The floors**: 378 for Q2 and 381 for Q4 at 262144, both found by loading, not by arithmetic. 262144
  at split 376 refuses (`t13-262k-dflash-20261030-080138`) and at 392 runs
  (`t13-262k-dflash-s392-20261030-081052`).
- **Rows 2, 5 and 6** remain predictions; `bench/options_table.py` lists every corpus arm beside its
  cache mode and split.

The gate that requires these citations to exist and to be `ok`, with the two refusals named as
failures rather than measurements:

```
bench/verify_task.py --post --report wiki/plans/v165-reports/T14-config-catalog.md \
    --cites-failure t13-262k-dflash-20261030-080138 \
    --cites-failure t14-q4-262k-corpus-20261102-085450
```

## Measured versus predicted

- **Measured**: the Q2 corpus throughput in row 3, its per-bucket live numbers above, and the split
  floors at 262144 — **Q2 378, Q4 381**, both engine-verified (376 and 378 refuse at Q4). Q2 at
  131k ran in T10 at 368.
- **A floor is only a floor when the engine says so.** `size_check.py` predicted 416 for Q4 at 262k
  because it reserves 2048 MB plus a 1024 MB headroom by default; the engine accepted 381 (839 MB
  free at its own reserve). Use the tool to rule a configuration out, then load it to find the
  floor.
- **Predicted**: rows 2, 3, 5 and 6, on both axes. The split floor comes from `size_check.py`, which
  reads 1–2 experts optimistic at this card: it accepts 376 where the engine refuses. The decode
  delta comes from the split curve measured at 262k, where 392 → 378 = +7.8 % for 14 experts, i.e.
  about 0.56 %/expert. That is how Q4's 3 extra experts become ~1.7 %.
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
