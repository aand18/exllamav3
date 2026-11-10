# T58 — distinct CPU experts per layer + slot ABBA: slots are not the lever

> **Correction (T66).** The headline "distinct CPU experts median 8, p90 18, mean 9.3" was an
> **all-rows** statistic: 97 920 of the 124 991 layer-rounds are the undrafted `rows=1` baseline arm.
> By row count the same CSVs give rows=1 **8**, rows=2 13, rows=3 16, rows=4 19, **rows=5 median 21**
> (mean 20.5). Everything downstream that quoted ~9.3 at the drafted window — T59, T63, T64 — should
> read **~20.5** instead; at 33.9 µs/expert the corrected figure closes T59's 0.696 ms `cmb` exactly.

## Setup

`EXL3_ROUND_PROF=1` with a new R1 counter (`enqueue_distinct`/`write_distinct` in
`exllamav3/util/round_prof.py`, pumped from `BlockSparseMLP_CPU.cpu_split_submit`) plus the
slots arm via `EXL3_MOE_CPU_SLOTS`. Production regime, pin on, `EXL3_MOE_PARTIAL_COMPLETION`
unset (default-off), static placement, 131072 (the T57 basis).

## Runs

| arm | run id | draft tok/s |
|---|---|---:|
| slots 4 | `t58A-slots4-131k-20261108-214726` | **55.42** |
| slots 8 | `t58B-slots8-131k-20261108-220339` | **55.80** |
| slots 8 | `t58C-slots8-131k-20261108-221944` | **55.80** |
| slots 4 | `t58D-slots4-131k-20261108-223615` | **54.14** |

Means (basis: draft tok/s, 131072, spec_decode, 8 rounds): slots 4 **54.78**, slots 8 **55.80**.
Difference **+1.86 %**, inside the same-session spread — the T57 identical-config spread is
56.23/54.44 = **3.3 %**. Acceptance identical (2.00/2.76, 2.04/2.83 vs 1.98/2.74, 2.01/2.79):
this is not a yield effect.

First attempt (two files deployed — a no-op for the instrument) is discarded: no phase or
distinct CSVs existed, so only its tok/s ABBA is citable and only separately: slots 4
55.53 mean (54.86/56.21), slots 8 57.05 (57.80/56.29), **+2.7 %**, same verdict, different
harness. That attempt also crashed twice on the way (`moe_cpu_host.py` ABI, then a missing
span-of-files `moe_lookahead`) before the 6-file set landed.

## R2 — slots 4→8 (the arm)

| phase `sub` (ms/layer) | mean | median |
|---|---:|---:|
| slots 4 (A/D) | 0.204 / 0.187 | 0.049 / 0.053 |
| slots 8 (B/C) | 0.181 / 0.185 | 0.049 / 0.052 |

Means: **0.196 → 0.183 ms (−0.013)**. The pre-registered pass mark was mean < 0.10 ms. The
12 ms → `sub` tail is **not slot-reuse** — doubling slots moved nothing. The mean/median gap
(0.19 vs 0.05) is a tail that lives somewhere else (ring backpressure is the remaining
suspect, but no measurement here pins it); the slot window was the hypothesis and it is dead.
No follow-up on 8→16: there is no slope to follow.

## R1 — distinct CPU experts (the count)

Per sampled round, per layer (basis: decode window only, static placement):

| | n | median | p90 | mean | max |
|---|---|---:|---:|---:|---:|
| slots 4 (A/D) | 123743 / 124655 | 8 / 8 | 18 / 18 | 9.35 / 9.25 | 68 |
| slots 8 (B/C) | 125471 / 128686 | 8 / 8 | 18 / 18 | 9.32 / 9.40 | 68–89 |

Round totals are drawn from the same CSVs (median 396, p90 ~900 per round).

The decision rule said ≤3 kills bytes, ≥10 justifies the global-pool work. Measured: median
**8**, mean **9.3**, p90 **18**. Neither gate fires cleanly. Bytes still track distinct —
the 0.58 ms CPU compute (T52) is consistent with a mean of ~9 distinct experts/layer at
42–45 GB/s — but the fork sits at 8, not the ~13 the analysis assumed, and the right tail
(p90 18, max 68) is where per-round compute varies. A Strata-shaped global pool would still
have to earn its keep: even cutting 8→2 saves at most a share of the 0.58 ms, and the GPU
window (T54: 0.67 ms occupancy-bound) stays untouched.

## Verdict

1. **Slots: dead.** 4→8 is noise on tok/s (+1.86 %) and moves `sub` by 0.013 ms. The
   slot-reuse hypothesis is withdrawn.
2. **Distinct: counted, between gates.** Bytes remain the mechanism by which CPU compute is
   spent, but nothing here licenses a build. If any lever is revisited, it is the
   **right tail** (p90 18, max 68) — per-round bytes variance — not the median, and a
   Strata global pool amortizes it differently than median arithmetic suggests.
3. The remaining CPU-wait mechanism (`cmb` ~0.39–0.50 ms vs compute 0.58 ms) is overlap, and
   0.25 ms/layer of it is what the next experiment has to measure directly rather than by
   elimination.

Verification: `tests/moe_cpu/test_distinct_counter.py` 3 passed; `tests/moe_cpu` 175
passed/41 skipped (same-session before the campaign).
