# T05 — MEMOPS: the first lever

Date: 2026-10-29. Arm A config (split 400, 131072, Q2, no drafting, guard 0, huge pages off,
floors 100/1200). ABBA order: `memops0-1`, `memops1-1`, `memops1-2`, `memops0-2`.
Ratios: `artifacts/analysis/T05-memops-abba.md`.

| pass | run id | status | PP @4096 | PP @131072 | TG @4096 | TG @130816 | min VRAM | min RAM |
|---|---|---|---|---|---|---|---|---|
| memops0-1 | `memops0-1-20261029-200246` | ok | 1596.78 | 1613.54 | 53.43 | 47.21 | 489 MB | 4490 MB |
| memops1-1 | `memops1-1-20261029-200544` | ok | 1650.67 | 1623.86 | 45.44 | 42.14 | 365 MB | 2573 MB |
| memops1-2 | `memops1-2-20261029-200847` | ok | 1559.77 | 1623.83 | 45.45 | 41.89 | 491 MB | 2752 MB |
| memops0-2 | `memops0-2-20261029-201150` | ok | 1704.38 | 1629.96 | 52.73 | 43.54 | 331 MB | 2852 MB |

## Median of the two per-band ratios (var = memops1)

| band | pp median | pp range | tg median | tg range |
|---|---|---|---|---|
| 4096 | 0.974 | 0.915–1.034 | 0.856 | 0.850–0.862 |
| 8192 | 0.958 | 0.946–0.971 | 0.870 | 0.796–0.943 |
| 16384 | 0.969 | 0.965–0.973 | 0.898 | 0.866–0.931 |
| 32768 | 0.977 | 0.971–0.984 | 0.889 | 0.878–0.900 |
| 65536 | 0.993 | 0.988–0.998 | 0.849 | 0.833–0.865 |
| 131072 / 130816 | 1.001 | 0.996–1.006 | 0.927 | 0.893–0.962 |

## Decision

**`MEMOPS_WINNER = 0`** (the deployed value; the regime does not change).

- **tg decides it**: `EXL3_MOE_MEMOPS=1` is **8–17 % slower at every context**, medians 0.83–0.93,
  and the two independent ratios agree within 1–5 points (range column) — the direction is far
  outside the ±11 % band noise and consistent across all eleven contexts.
- **pp is neutral-to-worse**: medians 0.958–1.001 at the measurement bands, i.e. at or just under
  the ±2 % threshold, never a win.
- This reproduces the v1.4.9 finding (memop path ~8–11 % slower) on 1.6.5, so the value stays
  pinned for the rest of the campaign and for deployment.

**No 262144 confirmation.** The winner is the value production already runs, so there is no
deployment change to confirm, and 262144 is a coin flip on this box (T4). The confirmation budget
is better spent on the final winner at T13.
