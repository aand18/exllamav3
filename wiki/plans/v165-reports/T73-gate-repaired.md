# T73 — the gate was defective: fixed, tested, and the historical verdicts re-run

Argon's methodology critique (`T72 §5`) named five defects in `bench/spec_primary.py`. Four are real
and are now fixed with tests; the fifth (trajectory divergence) is a property of the measurement, not
of the script, and is answered by trusting the hardware timer instead of tok/s — which is what this
campaign's recent verdicts already did.

## The four defects, and what changed

| # | defect | fix |
|---|---|---|
| C1 | `arm_spread` used the **range** `(max−min)/mean`, which *grows* with N — so each added replicate **raised** the pass floor (1.039 → 1.045 → 1.050 across T66–T70) | **1.96 × SEM / mean** of the per-run primary means. It shrinks as 1/√N, which is the quantity a floor should track |
| C2 | `arm_spread` measured the primary with an **unweighted** category mean while the primary itself is weighted **5:4:3** — two different quantities under one name | per-run primary means are now weighted 5:4:3, the same weights as the gate |
| C3 | a fixed per-category line (**≥1.00** to pass, **<0.98** to withdraw) applied to 3–5-prompt categories with 3–7 % jitter — a true +2.5 % uniform gain failed ~half the time | each category vetoes only past **its own** measured noise: `1 − max(0.02, that category's 1.96·SEM)` |
| C4 | `|Δacc| ≤ 0.05` made **every** genuine drafting improvement unpassable — T61/T63/T70 could only ever be judged on their losses | `--drafting` makes the acceptance guard **one-sided** (`Δacc ≥ −0.05`): a drafter or window arm must not *lose* acceptance; a kernel arm keeps the two-sided band |

Tests: `bench/test_spec_primary.py` **13 passed** (9 existing + 4 new, one per defect — a third
replicate no longer raises the floor; a drafting arm may improve acceptance; a drafting arm may not
lose it; a −1 % category passes while −10 % withdraws).

## The historical verdicts, re-run under the corrected gate

| task | arm | old floor | new floor | verdict then | verdict now |
|---|---|---:|---:|---|---|
| T66 | dynamic placement | 1.045 | 1.020 | INCONCLUSIVE | **INCONCLUSIVE** |
| T69 | split 373 (+8 experts) | 1.039 | 1.020 | INCONCLUSIVE | **INCONCLUSIVE** |
| T70 | `-ndt 3` | 1.050 | 1.025 | WITHDRAW | **WITHDRAW** (0.9657; acc −0.423 ≫ 0.05) |
| T61 | exl3 drafter | 1.024 | 1.020 | WITHDRAW | **WITHDRAW** (0.9657…1.0033 vs 1.020; curl −5.74 %) |

**No verdict flips.** What changed is the *resolution*: the floors fell to ~1.020 (from up to 1.050),
so a real 2.5 % improvement would now pass where before it needed 5 %. That is the defect C1 was
costing.

Two of the four remain inconclusive rather than pass, and for a reason the corrected gate now makes
explicit rather than silent: T66's dynamic arm **changed acceptance by +0.138**, so under the kernel
rule it cannot be scored as a pure systems change at all — the right reading was always "this arm
moves acceptance too, so its tok/s ratio is not attributable to placement", and the script now says so
in `acc.rule` instead of folding it into a generic inconclusive.

## What this does not fix

**Trajectory divergence (Argon's A).** Any intervention that perturbs floating-point order flips a
borderline draft and sends two runs into different text; on 3–5-prompt categories that injects 3–7 %
jitter into tok/s even when the round time is constant. The answer is not a better statistic but a
different observable: **gate systems changes on per-row round latency from the `ROUND_PROF` brackets**
(`ms/round` at rows=5, where 0.5 % resolves cleanly) and use tok/s as the cross-check. That is what
T66/T69/T70 already did in prose — `cmb` flat to ±3 µs — and it is the rule to make explicit in the
next campaign plan rather than a change to this script.

Recorded as the standing method: **`cmb` and `distinct` decide a systems arm; `spec_primary.py`
decides a drafting arm; tok/s alone decides neither.**

## Runs re-scored

`t66A-static-262k-20261110-172259`, `t66B-dynamic-262k-20261110-174018`,
`t66C-dynamic-262k-20261110-175726`, `t66D-static-262k-20261110-181449` (T66);
`t68E-cs40-mcs381-20261110-211419`, `t68H-cs40-mcs381-20261110-215158`,
`t68J-cs40-mcs381-20261110-221041`, `t69A-cs40-mcs373-20261110-223004`,
`t69B-cs40-mcs373-20261110-224657` (T69); `t70A-ndt4-262k-20261111-055730`,
`t70D-ndt4-262k-20261111-061433`, `t70E-ndt4-262k-20261111-070757`,
`t70B-ndt3-262k-20261111-063421`, `t70C-ndt3-262k-20261111-065103` (T70);
`t61A-bf16-131k-20261110-065833`, `t61D-bf16-131k-20261110-074755`,
`t61B-exl3-131k-20261110-071514`, `t61C-exl3-131k-20261110-073132` (T61).

Every one was re-scored with the corrected gate in this session; the script and its 13 tests are the
deliverable, and the table above is that re-score.
