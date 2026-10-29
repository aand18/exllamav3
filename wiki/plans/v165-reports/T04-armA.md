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
- **`-NoSpecDec`.** Spec decode costs ~1.1 GB VRAM (a second draft cache sized like the main one),
  which 131072 does not have to spare at this split. Drafting is measured in T6/T7; Arm A is the
  no-drafting reference, and every later tg comparison must use a reference in the same drafting
  state.

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
