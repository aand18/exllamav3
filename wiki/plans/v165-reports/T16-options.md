# T16 — the options table

One place to choose from. Every row is a configuration this box has run, or a compromise that the
measurements rule out. Corpus arms are `spec_decode` at 262 144 tokens unless the row says otherwise,
DFlash2 window 4, Qwen3.8-Flash-Next `3.18bpw_h6_ng8`.

## What is on the table

| option | cache | split | corpus t/s | vs deployed | draft accepted | min free VRAM | peak working set | min avail RAM | evidence |
|---|---|---|---|---|---|---|---|---|---|
| **deployed** | Q4 262k | 381 (dynamic) | not harness-measured | — | — | 1699 MB at boot | not tracked | not tracked | server boot 2026-11-02 18:49, 22440 MiB used |
| Q4 262k, static floor | Q4 262k | 389 | **52.96** | reference | 2.34/4 | 1777 MB | 41.7 GB | 4652 MB | `t14-q4-262k-corpus-20261102-085612` |
| Q2 262k | Q2 262k | 378 | **54.78** | +3.4 % | 2.39/4 | 777 MB | 40.7 GB | 6880 MB | `t14-q2-262k-corpus-20261102-091413` |
| **Q2 262k at Q4's split** | Q2 262k | 389 | **52.70** | −0.5 % | 2.34/4 | 2485 MB | 41.7 GB | 6971 MB | `t16-q2-389-corpus-20261102-170933` |
| FP16 262k | FP16 262k | 448 | **46.56** | −12.1 % | 2.39/4 | 2057 MB | 46.8 GB | 2094 MB | `t14-fp16-262k-corpus-20261102-093623` |
| FP16 262k, MTP head | FP16 262k | 434 | **50.62** | −4.4 % | 1.65/2 | 657 MB | 46.2 GB | 2773 MB | `t15-fp16-mtp-corpus-20261102-123715` |
| **FP16 131k** | FP16 131k | 408 | **50.64** | −4.4 % | 2.35/4 | 2363 MB | 43.2 GB | 4415 MB | `t16-fp16-131k-corpus-20261102-164829` |
| `-dds` on Q4, run 1 | Q4 262k | 389 | **53.71** | +1.4 % vs ref1 | 1.95/2.74 | 1777 MB | 41.7 GB | 6638 MB | `t14-q4-dds-corpus-20261102-102211` |
| `-dds` on Q4, run 2 | Q4 262k | 389 | **55.44** | +3.6 % vs ref2 | 2.07/2.90 | 1805 MB | 41.6 GB | 5632 MB | `t16-dds-confirm-20261102-160442` |
| Q4 reference, run 2 | Q4 262k | 389 | **53.51** | +1.0 % | 2.35/4 | 1777 MB | 41.7 GB | 6465 MB | `t16-q4-ref2-20261102-172722` |
| chunk 8192 | Q4 262k | 389 | 2702 T/s prefill | — | — | — | — | — | `t15-chunk8192-pp-20261102-123135`; the load at 381 refuses (server log of 12:35 UTC) |

Two slots are deployed, and the evidence is thinner than the phrase "flat" suggests. Two pairs are
valid, both at 256 generated tokens with no repeat: 1 × 20 k against 2 × 20 k is 12.77 against 12.69
(−0.6 %), and 1 × 130 k against 2 × 130 k is 3.19 against 3.12 (**−2.2 %**, a loss). The 131 k pair
never co-resided — 262 209 tokens against a 262 144 cache — so it is a serialization control, not a
slot test, and it ran 3.4 % faster end to end than co-residence at 130 k. The +7.8 % row
(`t16-conc-2jobs-20261102-162500`) is excluded: a slicing bug gave its jobs 131 k and 90 k prompts.
Scenario not measured: staggered arrival, which is what two agents with tool calls actually do.

## What each compromise costs

- **Q2 instead of Q4**: at one split, Q2 is **0.5 % slower**, not 3 % faster. The 3.4 % from the
  unequal-split comparison is placement: eleven experts on the static harness (389 against 378) and
  **three on the served box (381 against 378)**. No probe in this harness can see a quality
  difference, so the operator's traffic remains the only evidence that Q2 hurt.
- **FP16 instead of Q4 at 262k**: about 12 % slower, **5.2 GB** more host working set (47.9 against
  42.7 GB), and still no measurable quality gain on three probes. Not worth it on this card.
- **FP16 at 131k instead of Q4 at 262k**: −5.9 % median, Coding 0.998. **Most of that gap is the
  trunk, not the cache**: its no-draft baseline is 38.49 against the reference's 39.67, so 3.0 of the
  4.4 points are split and context, and the drafting cost per round barely moves (2.55 against 2.50).
  One draw, three variables, no Q4-at-131k control. Promising, not a ranking.
- **FP16 with the MTP head**: recovers half of the FP16 tax by dropping the split 14 experts, but its
  two-token drafts lose 19 % on Coding and Agentic code. Not viable for this traffic.
- **`-dds`**: an ABBA at split 389 — reference 52.96, `-dds` 53.71, `-dds` 55.44, reference 53.51.
  Paired correctly against the reference nearest each arm: **+1.4 % and +3.6 %**. The treated arm's
  own pair-to-pair spread (3.2 %) is larger than the pooled effect, and a pooled interval reaches
  from about −5 % to +10 %. Per category the second pair is Coding 1.003 and Agentic code 1.000,
  against 1.026 and 0.996 in the first, and two identical references differ 3.2 % on Coding.
  **The sign repeats; the size does not resolve.** Deployment is a bet on the sign, taken because
  the operator's categories are never negative across the four measurements.
- **Two slots** (deployed): aggregate throughput is flat in every pair measured, so the second slot
  buys latency hiding for two agents and nothing in tokens per second.
- **Chunk 8192**: 44 % faster prefill, refused at the deployed split; the split step it would need
  costs more decode than the prefill returns on this traffic.

## What is not measured, and why it matters

- **Q4 against Q2 at one split**: settled. 52.96 against 52.70 at split 389, acceptance identical,
  so the cache width has no runtime cost and the earlier 3.4 % was placement.
- **Production acceptance**: still the one number the harness cannot produce, because the box does
  not archive prompts.
- **Production acceptance.** The corpus accepts 58.5 % of drafts; the live log shows 39.6 %, which is
  1.58 of 4 against a break-even near 1.5. Every drafting number here is priced on a friendlier
  workload than production.

## The prediction held: the cache costs VRAM, not speed

Q2 was re-run at split 389, the split Q4 was measured at. Prediction: 52.96 tok/s and acceptance
2.34/4, a miss if it landed more than 5 % away.

| arm | split | corpus t/s | baseline | accepted |
|---|---|---|---|---|
| Q4 262k | 389 | 52.96 | 39.67 | 2.34/4.00 |
| Q2 262k | 389 | **52.70** | 39.58 | **2.34/4.00** |

Median per-category ratio **0.993**, Coding 0.991, Agentic code 0.978, Agentic curl 1.026. The
prediction holds to half a percent, and acceptance is identical to the last digit.

**Bounded, not zero.** Q2 and Q4 are indistinguishable at one split: 52.70 against 52.96, a
difference smaller than the 1.0 % spread between two runs of Q4 itself (52.96 and 53.51, category
median 2.2 % apart, Coding 3.2 %). With one draw on the Q2 side, the pair bounds the cache-width
effect at about 1 %, it does not establish equivalence.

The delivered cost is the split step: three experts on the served box (378 against 381), eleven on
the static harness (378 against 389). A dynamic-placement ladder without the drafter puts eight
experts at 0.7 %, so three experts are worth somewhere between 0.3 % and 1.7 % depending on which
coefficient holds. **Write it as: Q4 costs VRAM and at most about 1 %.**

## The placement term, measured on the served setting

Every earlier comparison priced the split from static arms. The served box runs **dynamic** expert
placement, where the engine migrates hot experts to the card, so the split is supposed to matter
less. Two `perf.py` ladders, no drafter, 262 144 cache, both with placement on:

| split | PP @131 072 | TG @130 816 | min free VRAM |
|---|---|---|---|
| 381 (served) | 1923.93 | 39.98 | 2267 MB |
| 389 | 1873.22 | 39.67 | 2983 MB |

Ratios 1.03 on prefill and 1.01 on decode, and the per-band swings are wider than the difference
itself: `bench/ratios.py` over these two runs prints both ranges. **Eight experts cost nothing measurable under dynamic placement**, against the
0.56 %/expert the campaign had derived from static arms.

That closes the chain the operator's choice rests on: Q4 costs VRAM, the VRAM costs up to three
experts on the served placement, and three experts under dynamic placement cost about 0.3 %, inside
the noise. The deployed Q4 is, for practical purposes, free.

## Evidence status

`bench/options_table.py` regenerates the measured rows from `passes.csv` and `arms.csv`, and
`bench/verify_task.py --post` gates the three reports this table draws on (T13, T14, T15).
