# T62 — the exl3 drafter moves the split floor 381 → 378, and decode does not notice

Pre-registered in `wiki/plans/2026-11-10-split-floor-exl3.md`. Basis: `spec_decode`, **cache
allocated at 262144** (the production allocation), decode priced at 131072, exl3 drafter
(`_t4check/state-exl3.json`), static placement, pin on, `-cq 4,4`, `-mct 16`, `-ambs 2`,
`-ndt 4 -dds`. Arms differ only in `-Split`. All seven arms in one session, 2026-11-10.

## The floor moved

| split | result | run id |
|---:|---|---|
| 381 | **ok** | `t62A-381-262k-20261110-085520`, `t62D-381-262k-20261110-091303` |
| 378 | **ok** | `t62E-378-262k-20261110-093222` |
| 375 | refused — `Insufficient VRAM in split for model and cache`, 26 s | `t62F-375-262k-20261110-094934` |
| 373 | refused, 26 s | `t62G-373-262k-20261110-095006` |
| 372 | refused, 26 s | `t62B-372-262k-20261110-091158`, `t62C-372-262k-20261110-091230` |

With the BF16 drafter the Q4 floor at 262144 was **381** (376 and 378 both refused — STATE.md).
With the 472 MB exl3 drafter resident, **378 loads and 375 does not**: the floor falls by exactly
**three experts**, the value the exl3 drafter's freed VRAM predicts when one expert is
87.7 MiB of trunk.

`bench/size_check.py` predicted 372 would fit (730 MiB free against a 512 MiB headroom rule) and
the engine refused it. That reproduces the known optimism of the tool versus the autosplit's own
accounting — the refusals, not the geometry, are the authority on the floor.

## Decode did not move

| arm | draft tok/s (corpus mean, 8 cats) | accepted of 4 |
|---|---:|---:|
| 381 | 59.03 | 1.926 |
| 381 | 58.09 | 1.819 |
| **378** | **58.60** | 1.861 |

381 pair mean **58.56**. Paired per-category ratio, 378 against the mean of the two adjacent 381
arms: **mean 1.0028, median 1.0030** — **+0.28 %**. The pre-registered deployable threshold was
**+2.0 %**; the verdict is **NULL**, and acceptance is flat (1.861 inside the 381 pair's
1.819–1.926 range).

## Why the null is the expected result, not a surprise

Two mechanisms both predict it, and either alone is sufficient:

1. **The three experts are the coldest ones.** The split is permuted hot-to-cold, so the experts
   sitting at the GPU/CPU boundary carry almost no routing. Handing three of them back to the GPU
   buys coverage in the ~0.6 % range — inside the spread at any run count.
2. **The binding constraint is the handoff wait, not GPU compute.** T59 measured the rows=5 MoE
   window at 0.883 ms/layer of which `cmb` (wait) is 0.696 — 79 %. Three more GPU experts shorten
   `ker` (0.117 ms of the window) by a few percent of it, i.e. by milliseconds per round at most.

So this arm closes the split lever with hard evidence: **the served VRAM headroom is real, the
floor really moved, and spending the headroom on experts buys nothing measurable.**

## What this changes

- **Do not change the served split on performance grounds.** 378 is available and costs nothing
  measured, so it is a *resource* choice (≈263 MiB less trunk VRAM, three fewer CPU experts of
  host traffic) rather than a speed choice. Deploying it needs the operator, and the argument for
  it is margin, not tok/s.
- **The drafter finding stands and is now complete on the split side**: T61 measured the exl3
  drafter's acceptance cost (−0.184) and its freed VRAM; T62 shows the freed VRAM cannot be
  converted into decode speed at this workload.
- **The performance frontier is where T59 left it**: the wait, not the kernels, not placement, not
  the split, not the drafter.

Verification: `bench/verify_task.py --post` on this report (cited runs ok, breach=0, warm split
components); three refusals are disclosed above as refusals and are not cited as measurements.
