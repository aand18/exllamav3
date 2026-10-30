# T10 — placement (split ladder), re-founded on 3.18 and re-measured warm

Date: 2026-10-30. Engine win-perf `1.6.5` (`1422a0a1…`). Model `3.18bpw_h6_ng8`, `-cs 131072`,
`-cq 2,2`, `-mct 16`, measurement length 131072, `perf.py` with `-NoSpecDec` (TG S=1 and the prefill
bands only). Every arm: frozen expert profile restored from `master` (`FROZEN_STATE=master->active`),
`EXL3_MOE_CPU_SWAP` left at the engine default (dynamic) — the regime the deployed server runs and
the one the archive's rows were measured under. Harness `fba75185…`.

## Step 1 — the ladder

| arm (run id) | split | PP 4096 | PP 65536 | PP 131072 | TG 65536 | TG 130816 | min free VRAM | min free RAM |
|---|---|---|---|---|---|---|---|---|
| `t10-s380-20261030-074338` | 380 | 1922.46 | 1856.65 | 1837.24 | 37.67 | 36.76 | 4093 MB | 8359 MB |
| `t10-s376-20261030-074632` | 376 | 1825.60 | 1835.71 | 1823.60 | 38.41 | **40.38** | 3617 MB | 8751 MB |
| `t10-s372-20261030-074927` | 372 | 1889.25 | 1864.28 | 1849.21 | 36.52 | 39.93 | 3205 MB | 9341 MB |
| `t10-s368-20261030-075220` | 368 | 1915.45 | 1880.82 | 1846.89 | 39.01 | 38.43 | 3081 MB | 10388 MB |

All four `status=ok`, `breach=0`.

**368 survives a full 131072 pass.** The archived `split368` died at the 131072 band on an allocator
scratch request; `perf.py` now calls `torch.cuda.empty_cache()` between bands (improvement I3), and
with that the band completes — `ALLOC peak_allocated_mb=20303 peak_reserved_mb=20716`, 3081 MB of
VRAM still free at the minimum. The old "368 does not load / cannot run long context" claim is
**closed**: it loads, warms, and finishes the ladder.

**PP is flat across the ladder** (1824–1849 at 131072, 1.4 %); the knob is a decode knob.

**TG at the long band**: 376 (40.38) > 372 (39.93) > 368 (38.43) > 380 (36.76). Three of the four
sit inside a 2 % band of each other; 380 is 9 % down and is the only arm outside it.

**VRAM floor direction is the opposite of the naive reading**: a *higher* `-mcs` parks more experts
in system RAM, so 380 leaves 4093 MB free where 368 leaves 3081 MB. 376 therefore dominates 368 —
more free VRAM *and* faster — and 380 buys 0.5 GB of VRAM for 9 % of decode. That is the whole
reason the ladder moves *down* from 400 and stops at 376.

## Step 3 — confirm the winner (and the spread it has to clear)

| arm (run id) | TG 130816 | TG 65536 | PP 131072 | min free VRAM |
|---|---|---|---|---|
| `t10-s376-20261030-074632` | 40.38 | 38.41 | 1823.60 | 3617 MB |
| `t10-s376-r2-20261030-075531` | 36.30 | 38.79 | 1841.10 | 3617 MB |
| `t10-s376-r3-20261030-075827` | 40.37 | 38.21 | 1821.62 | 3617 MB |

**The spread is 10 % peak-to-peak at the long band** (36.30 / 40.38 / 40.37, median 40.37) while
PP and the mid bands move <2 %. That single low run is not noise to be averaged away — it is the
instrument's resolution at this band, and it changes the ladder's verdict from "376 is fastest" to
what the evidence supports:

- **376 wins on the median** (40.37) and is 1 % above 372 (39.93) and 5 % above 368 (38.43) — all
  inside the spread, so those three are **tied on decode**.
- 380 (36.76, one pass) sits at the *bottom* of 376's own range, so the ladder cannot prove 380 is
  slower either; what it can prove is that 380 spends 0.5 GB more VRAM headroom (`-mcs` parks more
  experts in RAM) for nothing measurable.
- The tie is broken by VRAM, not by tok/s: **376 leaves 3617 MB free at the minimum, 368 leaves
  3081 MB and the winning drafter costs ~1.4 GB** (1150 MB weights + 248 MB draft cache). 376 is the
  only split that gives the drafter room without either the lowest split or the slowest one.

**Finding: split 376 confirmed as the operating point, on the median plus headroom — not on a
resolved speed difference.** The old "the 8-expert step from 384 to 376 is worth ~15 % tg" claim
does not survive this: measured warm, 380 → 376 is 9 % at one pass each and inside the spread.

## Step 2 — allocator scratch at the low end

`ALLOC peak_allocated/reserved` per arm: 380 → 19278/19704, 376 → 19666/20592, 372 → 20048/20592,
368 → 20303/20716. The reserved pool grows as the split falls (more GPU-resident experts), and 368's
`peak_reserved` (20716 MB) is the largest while its free VRAM minimum (3081 MB) is the smallest —
the cached-pool theory the critique raised now has its evidence, and `empty_cache` is what lets the
131072 band fit inside it.

## Step 5 — pinned arena / promote-on-miss

Not run this session. It is a VRAM/RAM trade the deployment does not need yet (the winner has
3.6 GB free and the drafter costs ~1.4 GB), and the prior 1.5.4 measurement (+21 % prefill, decode
unresolved) has to be re-earned with an ABBA that this session did not have time for. Carried
forward, not dropped.
