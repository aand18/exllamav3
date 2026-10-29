# T04 — Arm A: the reference arm and the acceptance threshold

Date: 2026-10-29. Engine: stock 1.6.5 (installed wheel, sha `5337b620…`). Model: 3.68bpw_h6_ng8.
Cache: `-cq 2,2`, 262144 slots (operator-fixed). GPU: RTX 4090 24564 MiB.

## The config (and why it is not the plan's original one)

```
-MaxLength 131072 -Split 400 -NoSpecDec -Root <v165 root>
regime env: EXL3_MOE_MEMOPS=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWIZZLE=1
            EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0
watcher floors: 100 MB free VRAM / 1200 MB free sys RAM (operator, 2026-10-29)
```

- **131072, not 262144.** 262144 *runs*, but not reproducibly: pass 1 finished with min free
  VRAM 503 MB, the very next identically-configured pass breached at **57 MB** in the 65536 band.
  VRAM swings ~±450 MB and host RAM ~±2.5 GB run to run, and no split satisfies both sides (VRAM
  needs `-mcs` ≳ 404, host RAM needs ≲ 410). 131072 keeps ~3 GB free on both → reproducible.
  262144 becomes the single final confirmation (T13 prize, not a measurement length).
- **`-Split 400`.** The regime default 360 — tuned for the deleted 2.05bpw model — does not load
  the 3.68 model at all; 372 likewise. 392 loads but dies at the 65536 prefill band with 23 MB
  VRAM free. See `T02-models.md` for the full ladder.
- **`EXL3_HOST_MEM_RESERVE_MB=0`.** The 43.02 GB CPU arena leaves ~2.0 GB host RAM at its deepest
  chunk, so the engine's guard (default 2048, or 1024) aborts the load by ~50–100 MB. The watcher
  holds the operator's 1200 MB floor instead: one enforced margin, charged once.
- **`-NoSpecDec`, for comparability rather than necessity (corrected 2026-10-30, critique C10).**
  The earlier claim that spec decode costs ~1.1 GB "for a second draft cache sized like the main
  one" was wrong twice: `perf.py -sd` allocates **no** second cache (it loops verification lengths
  1..4 on the same cache), and the MTP head's own draft cache in `spec_decode.py` is **~160 MiB**
  at 262144/Q2 because the head is **one** layer, not 12. Arm A stays the no-drafting reference so
  that every later tg comparison names its drafting state explicitly — nothing here is a fit limit.

## The two passes

| pass | run id | PP @4096 | PP @131072 | TG range (S=1) | min VRAM | min RAM |
|---|---|---|---|---|---|---|
| A1 | `armA-1-20261029-195610` | 1637.69 | 1632.40 | 46.6–57.0 | 329 MB | 2443 MB |
| A2 | `armA-2-20261029-195914` | 1602.18 | 1642.92 | 45.8–57.6 | 863 MB | 2366 MB |

Full per-band tables: `results/pp.csv`, `results/tg.csv` (archived under
`artifacts/results/`), ratios in `artifacts/analysis/T04-armA-spread-hugepage0.md`.

## The acceptance threshold — A2/A1 per band

| band | pp ratio | tg ratio (S=1) |
|---|---|---|
| 256 | 1.09 | 1.07 |
| 512 | 0.94 | 1.03 |
| 1024 | 1.00 | 0.98 |
| 2048 | 0.95 | 1.05 |
| 4096 | 0.98 | 1.06 |
| 8192 | 1.00 | 1.02 |
| 16384 | 1.00 | 1.04 |
| 32768 | 1.00 | 1.02 |
| 65536 | 1.01 | 1.11 |
| 131072 / 130816 | 1.01 | 0.97 |

**Read it per metric, not as one number:**

- **pp at the measurement bands (4096–131072) is tight: ±2 %.** A prefill win above 2 % is real.
- **pp at 256–2048 is noisy: ±5–10 %.** No claim from those bands alone.
- **tg is the noisy metric: ±11 % at the worst band (65536), typically ±5 %.** A tg win under
  ~11 % is inside the noise and must be recorded as *no difference*; decisive tg claims need
  either a bigger effect or the ABBA median of two ratios.

## The variance source this report closes

A first A1/A2 pair measured **tg ratios 0.93–1.19** with identical configs. Cause: the OS grants
a variable slice of the ~43 GB CPU expert arena as large pages — 0.87 GB in one pass, 2.08 GB in
the next (both printed by the engine) — and CPU expert gathers run faster through huge pages. The
production regime now sets `EXL3_MOE_ARENA_HUGEPAGE=0`, which cut the spread to the table above.
That is a determinism control, not a perf claim: any pass measured before this change is not
comparable with arms measured after it.

## Re-derived on the swapped-in 3.18bpw model (operator swap, 2026-10-29)

The operator replaced the production checkpoint mid-campaign (3.68 → 3.18bpw_h6_ng8, "no
headroom, 8 GB lighter"), so the reference arm and its threshold are re-measured on the model that
will be deployed. The 3.68 numbers above stay as the reason the swap was made.

### Split ladder at 131072 on 3.18 (T10 pulled forward — the swap flips the ladder direction)

| split | min VRAM | min RAM | PP @4096 | PP @131072 | TG @130816 | verdict |
|---|---|---|---|---|---|---|
| 400 | 4933 | 9605 | 1711 | — | — | loads with ~5 GB spare (4096 probe) |
| 384 | 1777 | 7059 | 1774 | 1783 | 43.54 | ok, but 15 % slower on tg |
| **376** | **2141** | **8382** | **1935** | **1828** | **49.79** | **winner** |
| 368 | 15 | 8778 | — | — | — | does not load (VRAM) |

376 is the lowest split that loads and also the fastest: the 8-expert step from 384 to 376 is
worth ~15 % tg (43.5 → 49.8 t/s at 130816) and +2.5 % pp. **Arm A on 3.18 = split 376.**

### The two passes and the new threshold

| pass | run id | PP @4096 | PP @131072 | TG @130816 | min VRAM | min RAM |
|---|---|---|---|---|---|---|
| A1 | `split376-131k-20261029-221518` | 1935.22 | 1827.59 | 49.79 | 2141 MB | 8382 MB |
| A2 | `armA2-318-20261029-221855` | 1786.27 | 1800.29 | 48.87 | 2491 MB | 7774 MB |

| band | pp ratio (A2/A1) | tg ratio |
|---|---|---|
| 4096 | 0.92 | 0.97 |
| 8192 | 0.96 | 1.00 |
| 16384 | 0.95 | 0.99 |
| 32768 | 0.98 | 0.93 |
| 65536 | 0.99 | 0.99 |
| 131072 / 130816 | 0.99 | 0.98 |

**Acceptance threshold on 3.18 (A2/A1): pp ±5 % at the measurement bands (0.95–0.99), tg ±9 %
worst band (0.91–1.01).** Tighter than the 3.68 arm (tg 0.97–1.11), and both resources now have
~2 GB and ~8 GB of slack — the box finally has room to measure.

Ratios: `artifacts/analysis/T04b-armA-318-spread.md`.
