# T66 — `cmb` is 33 µs per distinct CPU expert, and placement cannot change the count

Two things settled here, one of them a correction to this campaign's own record.

## A. The T58 number was wrong, and everything read from it was inherited

T58 reported **"distinct CPU experts per layer: median 8, p90 18, mean 9.3"**. That was an
**all-rows** statistic: 97 920 of its 124 991 layer-rounds were the undrafted baseline arm at
`rows=1`. Recomputed from the same CSVs by row count:

| rows | T58 A | T58 B | T65 (n=122 928) |
|---|---|---|---|
| 1 | 8 | 8 | 8 |
| 2 | 13 | 12 | 12 |
| 3 | 16 | 16 | 16 |
| 4 | 19 | 19 | 19 |
| **5** | **21** | **21** | **21** |

**At the drafted window the count is median 21, mean ~20.5 — two and a half times what the report
said.** The corrected figure propagates to T59, T63 and Argon's review, all of which used ~9.3.

## B. That correction closes the arithmetic exactly

T59 measured `cmb` = 0.696 ms/layer at rows=5. At 20.5 distinct experts that is **33.9 µs per
expert**, and the 1 014 545 matched layer-rounds of T65 give a slope of **+37 µs per additional
distinct expert** (rows=5) with a clean monotone ladder (4→0.047, 8→0.184, 12→0.328, 16→0.505,
20→0.807 ms). Per-expert cost at ~2 MB of payload is ~60 GB/s of DDR5 — plausible bus doctrine, not
a coincidence. **`cmb` is CPU DRAM read time, linear in distinct experts, and nothing else in the
window competes with it** (`ker` 0.118 ms, `sub` 0.047, `mixer` ~0.12).

Argon's mechanism stands. Its per-expert byte figure (~2 MB) stands; my 5.57 MB objection was the
arena's over-provisioned capacity, not payload.

## C. The experiment: adaptive placement on the drafted path

`-Runner spec_decode` forces `EXL3_MOE_CPU_SWAP=0`, so T51's dismissal of expert coverage was
measured in `perf.py` at 1 pick/layer and never on the drafted path. The harness permits an override
(`-Env "EXL3_MOE_CPU_SWAP=1"`), so this is the test T51 should have been.

Basis: `spec_decode`, `-CacheSize 262144` (production allocation), 131 072 decode, split 381, exl3
drafter (`_t4check/state-exl3.json` — the combination T62 proved loads; BF16 + 262 k + 381 refuses),
pin on, slots 4, roundprof `SAMPLE=8`.

| arm | run id | draft tok/s | accepted |
|---|---|---:|---:|
| static | `t66A-static-262k-20261110-172259` | 56.05 | 1.84 |
| dynamic | `t66B-dynamic-262k-20261110-174018` | **58.39** | 1.97 |
| dynamic | `t66C-dynamic-262k-20261110-175726` | 56.24 | 1.87 |
| static | `t66D-static-262k-20261110-181449` | 56.08 | 1.91 |

| arm | rows=5 distinct (median) | `cmb` median | µs/expert |
|---|---:|---:|---:|
| static (A) | 20 | 0.668 | 33.4 |
| dynamic (B) | 21 | 0.672 | 32.0 |
| dynamic (C) | 21 | 0.659 | 31.4 |
| static (D) | 21 | 0.680 | 32.4 |

**The falsifier fired: adaptive placement does not reduce distinct misses.** 20.2 → 20.5 — if
anything a little worse, and `cmb` is flat within 0.02 ms. The per-expert slope is identical across
arms (31.4–33.4 µs), which is the same statement from the other side: the mechanism is untouched.

Agentic-weighted primary: static 68.37 → dynamic 70.31, **ratio 1.0284 against a 1.045 floor →
`WITHDRAW`**. The per-category split is the usual trap: Agentic-code **+4.93 %**, Agentic-curl
**+8.99 %**, Coding **−6.62 %**, acceptance +0.138. A movement with no mechanism behind it
(distinct experts unchanged, `cmb` unchanged) and one category losing badly is not a win.

## What is now closed, on the right evidence

1. **Adaptive expert placement does not cut CPU bytes on the drafted path** — measured at 50
   picks/layer, not inferred from `perf.py`. T51's conclusion survives; its evidence did not.
2. **The window is DRAM-bound at the CPU.** 20.5 experts × 33 µs = 0.69 ms of the 0.86 ms window;
   the GPU's own work is 0.12 ms. No GPU-side change (kernels, graphs, grouping) can win a race
   against a bus that is already 92 % busy.
3. **The levers that would cut bytes are not available**: more VRAM does not help (T62: +3 experts =
   +0.28 %), KV residency to host is not in this engine (only a hash-keyed prefix cache × `-ccs`),
   and 2 bpw is an operator decision.

## The one number a future session should start from

Distinct CPU experts at rows=5 is **~20**, not 8. Anything that cannot halve that — a smaller draft
window, a byte-level expert representation, or genuinely KV-off-VRAM attention — cannot reach
Strata's 25 ms round; at 33 µs/expert the budget is arithmetic.
