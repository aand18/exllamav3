# T75 — three independent verifiers on Argon's nine claims: agreement, and two things they caught

Three models (GLM 5.3 Flash, Muse Spark 1.3, DeepSeek V4.1 Flash) were given the identical brief:
verify each claim against the code, judge whether the repo now fixes it, and add their own
conclusions. None was told what to conclude. HEAD at issue was `8742a681`; DeepSeek's and GLM's
sharpest findings were then implemented, so the table below has a "after" column for those.

## The verdicts

| # | claim | GLM | Muse | DeepSeek | fix landed? |
|---|---|---|---|---|---|
| 1 | `spec_decode` aggregated rates quadratically | CORRECT | CORRECT | CORRECT | **yes** — `ΣN/ΣT`, `time_generate` stored, `CAT` tokens/seconds emitted |
| 2 | docstring said ratio-of-sums, code averaged rates; spread unweighted | CORRECT | CORRECT | **PARTLY** | spread **yes**; the *primary* was still a weighted mean of rates until DeepSeek's finding was implemented (T75) — now ratio-of-sums when the columns exist |
| 3 | `ARM` line is an unweighted 8-category mean | CORRECT | CORRECT | CORRECT | **scoped, not reweighted** — documented headline-only; decision metric is `spec_primary.py` |
| 4 | pass floor from `(max−min)/mean` grows with N | CORRECT | CORRECT | CORRECT | **yes**, then **re-fixed** (see below) |
| 5 | fixed 1.00 / 0.98 category lines on 3–5-prompt buckets | CORRECT | CORRECT | CORRECT | **yes** — per-category noise lines |
| 6 | `|Δacc| ≤ 0.05` barred drafting improvements | CORRECT | CORRECT | CORRECT | **yes**, opt-in `--drafting`; and a material acceptance **loss** now withdraws rather than merely blocking a pass |
| 7 | `spec_decode` runs an ignored Baseline arm; contaminates the profiler CSVs | CORRECT | CORRECT | CORRECT | harness **yes** (`-nbl` default); **residual fixed**: the distinct counter now records `rows > 1` only |
| 8 | `perf.py` is not a proxy for `spec_decode` | CORRECT | CORRECT | CORRECT | **no code fix** — recorded as method |
| 9 | `spares` carves GPU slots that some layers can never fill | CORRECT (mechanism imprecise) | CORRECT (mechanism imprecise) | CORRECT | **no code fix** — recorded rule `PROMOTE=0` for dynamic arms |

All three independently reproduced the T58 contamination arithmetic: 97 920 of 124 991 layer-rounds
were `rows=1`, so an all-rows median necessarily equals the baseline arm's 8 while the drafted window
reads 21. That closes the worst number in the campaign with three external confirmations.

## Two findings that were right and cost me a fix

**1. `1.96 × SEM` at n=2 is the range statistic relabelled.** DeepSeek: with two values,
`sd = |v₁−v₂|/√2`, so `1.96·sd/√2 = 0.98·|v₁−v₂|` — the exact quantity it replaced. Verified
numerically: `1.96·SEM/mean ÷ range/mean = 0.980` at n=2, 0.571 at n=3. So the T73 "floors fell from
1.045 to 1.020" was the **weighting** change, not the statistic, and my T74 §5 claim that item 4 was
"DONE" was an overclaim. GLM reached the same conclusion from the other side ("the SEM rarely binds;
the floor is still the fixed 2 %, laundered").

Fixed properly: the half-width now uses the small-sample t quantile (`t(0.975, n−1)` = 12.71 at n=2,
4.30 at n=3). The floors become what the data can actually support:

| arm pair | n per arm | floor | verdict |
|---|---:|---:|---|
| T70 `-ndt 3` | 3 / 2 | **1.141** | WITHDRAW (primary 0.9657, acc −0.423) |
| T61 exl3 drafter | 2 / 2 | **1.114** | WITHDRAW (acc −0.205) |
| T69 +8 experts | 3 / 2 | **1.071** | INCONCLUSIVE |
| T66 dynamic | 2 / 2 | **1.256** | INCONCLUSIVE |

The honest reading: **at two to three runs this harness can only resolve 7–26 % of tok/s.** The
reported "resolution" field now says so, and it is Argon's §4A point arriving as arithmetic rather
than as advice.

**2. The primary was still an average of rates.** DeepSeek: the docstring promised ratios of sums, the
primary was `Σ(w·rate_c)/Σw`. Implemented: `primary_of()` now takes a weighted `Σ tokens / Σ seconds`
when the CSV carries the columns (it does not for pre-T74 passes), and `acc` aggregates
`Σ acc·rounds / Σ rounds` instead of a prompt-count mean.

## Where the three disagree, and it is the useful part

- **Claim 2**: Muse said fixed, DeepSeek said PARTLY and was right at the time. The fix existed for
  the spread but not for the primary.
- **Claim 9's mechanism**: Muse and GLM both note `spares` is the carve-out, but the layers that can
  never fill it are the ones outside the pin budget where `promote_ready` refuses — the recorded
  rule (`PROMOTE=0`) is unchanged, the explanation was wrong.
- **Claim 8 is the one nobody thinks is fixed**, and GLM sharpened it beyond the claim: the "hardware
  timer" is `cmb`/`distinct` *sampled during free-running generation*, i.e. derived counters on an
  uncontrolled workload. There is no fixed-trace runner in this repo. That is a missing capability,
  not a rule.

## Evidence

Re-scored under the corrected gate: `t70A-ndt4-262k-20261111-055730`,
`t70D-ndt4-262k-20261111-061433`, `t70E-ndt4-262k-20261111-070757`,
`t70B-ndt3-262k-20261111-063421`, `t70C-ndt3-262k-20261111-065103`;
`t61A-bf16-131k-20261110-065833`, `t61D-bf16-131k-20261110-074755`,
`t61B-exl3-131k-20261110-071514`, `t61C-exl3-131k-20261110-073132`;
`t68E-cs40-mcs381-20261110-211419`, `t68H-cs40-mcs381-20261110-215158`,
`t68J-cs40-mcs381-20261110-221041`, `t69A-cs40-mcs373-20261110-223004`,
`t69B-cs40-mcs373-20261110-224657`; `t66A-static-262k-20261110-172259`,
`t66D-static-262k-20261110-181449`, `t66B-dynamic-262k-20261110-174018`,
`t66C-dynamic-262k-20261110-175726`.

Tests: `bench/test_spec_primary.py` 13 passed; `tests/moe_cpu` + spec_primary 191 passed / 41 skipped
after the `rows > 1` counter filter.

## Consequence

The recent verdicts survive on their mechanism evidence, not on tok/s: T66/T69/T70 were decided on
`cmb` flat to ±3 µs. What does not survive is any claim that this stack resolves a 2 % tok/s
difference at n=2–3 — it does not, and now says so. A future systems arm should either (a) run ≥6
replicates, or (b) gate on `ms/round` from a fixed trace, which the repository does not yet have.
