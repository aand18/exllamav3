# T78 — Kimi's review of my response: two repairs were incomplete, and the record over-claimed

Second pass, on HEAD `3d43921d`. Its verdict: the response was substantive rather than tidy, my account
of its review was **not** systematically misrepresented, but "all findings addressed" was not true and
several passages adopted its narrower vocabulary while keeping a stronger conclusion underneath.

## What it caught in my "fixed" list

| # | its finding | what was actually wrong | fixed |
|---|---|---|---|
| a | spread and primary still different estimators | I had called `throughput()` **inside each category**, then taken a 5:4:3 weighted mean of the category rates — which is not a cross-category ratio of sums. The effect and its resolution were still two different quantities | `primary_from_rows()` is now the **single** function, used by both the effect and every per-run spread value |
| d | basis guard incomplete | `basis()` inferred completeness with `any()` while `primary_of()` required `all()`, so two arms could both read "ratio of sums" with only one complete — exactly the comparison the guard exists to refuse | `basis()` now applies the same all-rows rule, `throughput()` pairs tokens/seconds **per row** (independent lists could divide different row subsets), and the gate refuses when the two arms' bases differ — with a test that reaches the refusal |
| — | category guards still used `max()` | I had combined variances only for the primary | category intervals now combine as `sqrt(a² + t²)` too |
| — | ratio-scale asymmetry | `1 − floor` made a loss easier to prove than a gain | both the loss line and the category line use `1 / (1 + floor)` |
| — | no regression tests for the repairs, and the docstring still described the superseded rules | true on both counts | 5 focused tests added (18 total): estimator agreement, `basis()` completeness, mixed-basis refusal, negative-inside-floor does not withdraw, a resolved loss does. The module docstring now states the **current** rule and lists the superseded one |
| — | my block replacement deleted `def ratio()` | caught by the suite immediately | restored |

## What it caught in the record

Three passages over-claimed in my favour, all now corrected in `v165-STATE.md`:

1. **"all zero", "both factors are closed", "the count lever is closed"** — while T69 and T66 are
   explicitly `INCONCLUSIVE`. That breaches reporting rule 1, which I had just adopted. Now reads
   *"not detected at these doses … a poor-ROI decision, not an empirical closure of the family"*, with
   T62's +0.28 %, T69's `cmb` 0.676 → 0.679 and T66's handicap stated beside it.
2. **"removing the fourth draft loses more than it saves"** — T70's withdrawal is driven by the
   material acceptance loss (−0.423); its throughput point estimate is negative but **inside** the
   floor. Now: withdrawn because the acceptance loss is material, net throughput effect unresolved.
3. **"the payload hypothesis survives intervention"** — no byte-width intervention has run. T69 changed
   *count*, not bytes. Now: the payload hypothesis is **untested**, not validated.
4. **"low-to-mid 70s"** — a conditional prediction, now written with its conditions attached.

## Its answer to the operator question

> Authorise a **compile-only, static-placement falsification build**; **not** a performance A/B or
> deployment until the sizing prerequisite is met.

And the single sentence it would require in the record first:

> *This is a static-only, non-deployment falsification experiment with a named numerical operator
> quality limit; before any performance A/B, `suh`/`svh` must be included in the sizing model, both
> exact same-regime split-381 arms must clear the VRAM/RAM floors with stated margins,
> `EXL3_MOE_CPU_SWAP` and `EXL3_MOE_CPU_PROMOTE` must be zero, and the fixed-trace runner must be
> ready.*

## Most serious omission, in its ranking

**The unpriced `suh`/`svh` allocations** — ~950 MiB missing from the sizing model, already the cause of
two VRAM-floor breaches, and a threat to attribution because a forced split change moves count, payload
and latency together. The missing **fixed-trace runner** is the larger *scientific* omission, second
because it does not block the build the way the sizing defect does. The bandwidth microbenchmark and
the historical tok/s rebuild are both down-ranked: a fresh same-session control makes the latter
unnecessary for this experiment.

## Final re-score after the repairs

| arm | ratio | floor | verdict | rule that produced it |
|---|---:|---:|---|---|
| T70 `-ndt 3` | 0.9657 | 1.152 | WITHDRAW | material acceptance loss (−0.423) |
| T61 exl3 drafter | 1.0033 | 1.135 | WITHDRAW | material acceptance loss (−0.205) |
| T69 +8 experts | 1.0185 | 1.089 | INCONCLUSIVE | inside resolution |
| T66 dynamic | 1.0284 | 1.336 | INCONCLUSIVE | inside resolution; arm handicapped |

18 tests pass. No verdict moved, but two of the four floors changed with the corrected variance
combination and symmetric loss line, and every verdict now names its rule.

## Evidence

Re-scored: `t70A-ndt4-262k-20261111-055730`, `t70D-ndt4-262k-20261111-061433`,
`t70E-ndt4-262k-20261111-070757`, `t70B-ndt3-262k-20261111-063421`,
`t70C-ndt3-262k-20261111-065103`; `t61A-bf16-131k-20261110-065833`,
`t61D-bf16-131k-20261110-074755`, `t61B-exl3-131k-20261110-071514`,
`t61C-exl3-131k-20261110-073132`; `t68E-cs40-mcs381-20261110-211419`,
`t68H-cs40-mcs381-20261110-215158`, `t68J-cs40-mcs381-20261110-221041`,
`t69A-cs40-mcs373-20261110-223004`, `t69B-cs40-mcs373-20261110-224657`;
`t66A-static-262k-20261110-172259`, `t66D-static-262k-20261110-181449`,
`t66B-dynamic-262k-20261110-174018`, `t66C-dynamic-262k-20261110-175726`.

Tests: `bench/test_spec_primary.py` 18 passed (13 before this task, 5 new for the repairs).

## Still open, in its own priority order

1. Finish the primary/basis repairs — **done in this task** — and treat the re-score as validation now.
2. Fix workload validity and build the fixed-trace systems measurement.
3. Add a per-category acceptance guard before any free-running drafting decision.
4. Increase paired replication instead of trying to make small-n intervals "friendlier".
5. Then make the corpus and secondary vetoes noise-aware.
