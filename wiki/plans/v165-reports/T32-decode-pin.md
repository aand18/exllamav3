# T32 — the arena pin is a decode win; promote-on-miss is not

Campaign of 2026-11-04 on the production regime (`split 381`, `cs 262144`, `cq 4,4`, DFlash2 window 4,
`max_length` 131072), one session, server stopped, `-dr 3`. Engine: branch `feat/adaptive-tier`
installed file-level into the venv (two files, md5 `7ff6aded…`, `7b9b705a…`); the pristine pair
(`328a82d8…`, `75ce3c64…`) is backed up and is restored when the campaigns end (T33 runs after this one).

## Why this campaign exists

The earlier campaign (t29) concluded "no proven decode gain" from **one draw of 100 tokens per
cell** (`perf.py`'s `decode_reps` default is 1, and the arms carried no `-dr`). Its own control arms
differed **9.6 %** (36.17 vs 39.66 tok/s for identical code), so the +1.13 % it reported was never
resolvable. This campaign fixes the metric (`-dr 3`, three reps per cell), removes the two debug
flags that were set in t29's pin arms (`EXL3_MOE_TIER_STATS`, `EXL3_MOE_CPU_SWAP_DEBUG`), and splits
the feature into its two parts.

## Arms

| arm | configuration | isolates |
|---|---|---|
| **A** | no pin (production path, periodic sweep) | the control |
| **B** | `EXL3_MOE_PINNED_ARENA=1`, no spares — unbounded, the driver stops it at 30 of 34 chunks / 30 720 MiB | the pinned arena alone |
| **C** | B + `EXL3_MOE_CPU_PROMOTE=16` spares/layer — promote-on-miss, budget 6/step, decay 0.98 | promotions, including their per-step hook |

Order **A B C C B A**, six passes, every one `status=ok`:

| arm | run id | status | elapsed |
|---|---|---|---|
| A1 | `t32-A1-pin-off-20261104-114457` | ok | 374 s |
| B1 | `t32-B1-pin-only-20261104-115118` | ok | 310 s |
| C1 | `t32-C1-pin-promote-20261104-115636` | ok | 314 s |
| C2 | `t32-C2-pin-promote-20261104-120157` | ok | 317 s |
| B2 | `t32-B2-pin-only-20261104-120721` | ok | 302 s |
| A2 | `t32-A2-pin-off-20261104-121230` | ok | 371 s |

## Decode (`TG_S1`, tok/s, mean of the two arms per config)

| context | A | B | C | B−A | C−B | spread A/B/C |
|---|---|---|---|---|---|---|
| 0 | 36.18 | 49.42 | 48.64 | **+36.6 %** | −1.6 % | 0.0 / 1.7 / 5.9 % |
| 256 | 37.53 | 46.80 | 46.09 | +24.7 % | −1.5 % | 1.0 / 4.0 / 5.2 % |
| 512 | 40.65 | 47.47 | 47.50 | +16.8 % | +0.1 % | 0.6 / 8.5 / 2.0 % |
| 1024 | 40.53 | 47.26 | 47.81 | +16.6 % | +1.2 % | 3.4 / 6.3 / 3.6 % |
| 2048 | 39.69 | 47.48 | 46.22 | +19.6 % | −2.7 % | 2.0 / 10.0 / 3.0 % |
| 4096 | 38.62 | 48.70 | 47.52 | +26.1 % | −2.4 % | 0.9 / 1.8 / 3.5 % |
| 8192 | 36.84 | 43.07 | 43.03 | +16.9 % | −0.1 % | 0.3 / 7.3 / 2.3 % |
| 16384 | 41.20 | 44.19 | 42.55 | +7.3 % | −3.7 % | 5.7 / 7.6 / 0.8 % |
| 32768 | 42.63 | 44.36 | 45.86 | +4.1 % | +3.4 % | 8.1 / 3.5 / 5.3 % |
| 65536 | 39.03 | 43.03 | 44.93 | +10.3 % | +4.4 % | 0.9 / 6.4 / 2.5 % |
| **130816** | **39.88** | **43.30** | **44.95** | **+8.6 %** | **+3.8 %** | **3.1 / 10.9 / 1.5 %** |

The pin is **positive at all eleven contexts**, from +4.1 % at 32768 to +36.6 % at 0 (sign test
11/11, p < 0.0005) — and the gain is largest where expert traffic dominates the step, which is what
a pinned arena is supposed to buy. At 130816 B's own repeat spread (10.9 %) is larger than the +8.6 %
difference, but B's **worse** draw (40.95) still beats A's mean (39.88), and C — which contains the
same pinned arena — independently reproduces the plateau at 44.95 with a 1.5 % spread.

Promotions do **not** add decode: C−B is −3.7 % at 16384, −2.4 % at 4096, −1.6 % at 0, +4.4 % at
65536 and +3.8 % at 130816 — **mean +0.08 %** over the ladder, with the positive tail inside B's
6–11 % spread at those contexts.

## Prefill (`pp`, tok/s, mean of the two arms)

| band | A | B | C | B−A | C−B |
|---|---|---|---|---|---|
| 4096 | 1914.50 | 2366.53 | 2325.52 | **+23.6 %** | −1.7 % |
| 131072 | 1896.40 | 2311.96 | 2251.95 | **+21.9 %** | −2.6 % |

## Wall clock, same work

A 371–374 s, B 302–310 s, C 314–317 s. The pinned arena is **18 % faster end to end**, and the
promotion arms give 8 s of that back.

## Decisions

- **The pinned arena is a decode win and ships.** +21.9 % prefill at 131072, +4…+37 % decode
  depending on context, −18 % wall clock, for pinned host memory and no engine-side arithmetic.
- **Promote-on-miss is killed** (ruling R16, taken on a decision tree pre-committed before the first
  pass): mean +0.08 % decode, −2.2 % prefill, +8 s wall clock, and the only positive tail is inside
  the control's spread. The 16 spare slots/layer, the 2.31 ms/step hook, and the promotion DMAs buy
  nothing measurable on this workload.
- **t29 is superseded, not reconciled** (ruling R17). Its pin arms carried `EXL3_MOE_TIER_STATS=1`
  and `EXL3_MOE_CPU_SWAP_DEBUG=1`; t32's carried neither; with identical engine files, regime and
  split, the same arm read **38.4** then **45.0** tok/s at 130816. No t29 decode number is citable.

## Defects found in the promotion policy, left unfixed on purpose

Both only matter to a feature that is now dead, so they are recorded rather than repaired:

1. `routed = cur > ps.prev` under a 0.98 per-step decay is **false for any expert whose count is
   ≥ 50** even when it was routed (one hit adds 1 while the decay removes 0.02·count). Hot misses are
   invisible to the policy.
2. Prefill feeds `_split_hist` (`submit_prefill` → `_split_translate` → `moe_split_map`), and the
   promote state never invalidates across prefill the way `_TierStats` does.
   In this harness the counter is empirical, not theoretical: the t29 log shows 643 routed/step in
   the first 128 decode steps and steady promotions from step 128, so the policy was not frozen — the
   defect changes *which* experts it can see.

## What this campaign does not claim

- **No drafter.** `perf.py` loads no DFlash2, so these are batch-1 decode cells. Production verifies
  5 rows per step and accepts ~3.4 tokens per round, which changes the per-step hook's amortisation
  (it would help promotions) and the miss profile.
- **One box, one workload, n = 2 per config.** The 130816 cell alone is underpowered for a 2–3 %
  claim; the ladder and the sign test are what carry the pin's verdict.
- **Capped pin untested here.** Unbounded leaves 5.5 GB of host RAM free; the cap curve is T33.

## How to re-run

```
# branch engine, file-level (pristine pair backed up in _agent/adaptive-t27/backup)
cp exllamav3/modules/block_sparse_mlp_cpu.py exllamav3/model/moe_cpu_host.py \
   /mnt/c/Users/yoho/Downloads/tabbyAPI/venv/Lib/site-packages/exllamav3/{modules,model}/
# arms, one session, server stopped
run_pass.ps1 -Tag t32-A1-pin-off      -MaxLength 131072 -Extra "-dr 3"
run_pass.ps1 -Tag t32-B1-pin-only     -Env "EXL3_MOE_PINNED_ARENA=1" -MaxLength 131072 -Extra "-dr 3"
run_pass.ps1 -Tag t32-C1-pin-promote  -Env "EXL3_MOE_PINNED_ARENA=1,EXL3_MOE_CPU_PROMOTE=16" -MaxLength 131072 -Extra "-dr 3"
# … C2, B2, A2; then the numbers from results/tg.csv and results/pp.csv
```
