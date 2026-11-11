# T77 — Kimi 3.1's final review: three overclaims withdrawn, four gate defects fixed

Reviewed HEAD `21e884fd`. Unlike the two earlier reviews this one attacked the *breadth* of the
conclusions rather than the arithmetic, and on three points it is right.

## 1. "Decode exhausted" is wider than the evidence — narrowed

Kimi's replacement wording, adopted:

> The tested 3.18 bpw configuration is locally exhausted for the placement, small-split, pinning and
> fixed-window arms that were **actually run**. The implementation remains dominated by CPU-side
> expert handling, and further small knob changes have poor expected returns.

What it does **not** establish: that no drafting change can help, or that the only remaining options
are a new representation or another engine. Specifically, T70 closed *one* proposition — "with this
drafter, removing the fourth draft position loses more than it saves" — and never tested **better
acceptance at the same five-row cost**, which is a different arm.

## 2. The acceptance requirement is the honest way to state the gap — and it is now quantified

At fixed round cost, `tok/s = (1 + acc) / T_round`. With the measured 57 tok/s at 2.88 tokens/round,
`T_round ≈ 50.5 ms`; reaching 93 tok/s needs

> `acc ≥ 93 × 0.0505 − 1 ≈ 3.70` accepted of four drafts.

Across the 55–58 tok/s range the requirement is **3.62–3.87 of 4**. That is a real, countable target
that the campaign never wrote down, and it is far more useful than "we are at 57 and Strata is at 93".
It also means a Strata-shaped 25 ms round is not portable: at 2.88 tokens/round, 93 tok/s implies a
**31 ms** round.

## 3. The ladder is not linear, and the coefficient was being read as a constant

Recomputed from the T66 medians, the marginal cost per distinct expert **rises**:

| segment | µs per additional expert |
|---|---:|
| 4 → 8 | 34.2 |
| 8 → 12 | 36.0 |
| 12 → 16 | 44.2 |
| **16 → 20** | **75.5** |
| mean 4 → 20 | 47.5 |

So "33.9 µs/expert" (median `cmb` ÷ median count) is an *average over a curved ladder*, not the
marginal cost at the operating point, which is roughly double. Two consequences, in opposite
directions, and both must be stated together:

- **Against the count lever:** the ladder is *observational* — it compares layer-rounds that happened
  to have different expert counts. The only interventional test (T69, +8 experts) moved `cmb` by
  **3 µs**, i.e. it did **not** reproduce the ladder's slope. So the steep segment is not evidence
  that removing experts saves 75 µs each.
- **For the payload lever:** the payload term is the one with a mechanism that survives intervention —
  bytes actually transferred — which is why the 2 bpw projection rests on payload, not on the count.

Kimi is also right that the 54–60 GB/s implied by 1.83 MB / 33.9 µs does **not** identify a hardware
ceiling: saturation, memory-level parallelism, worker serialisation, dequant cost and cache effects
are not separated by these observations. A same-access-pattern read microbenchmark would settle it
cheaply, and has not been run.

## 4. The 2 bpw lever is **static-placement-only** — a constraint that was missing

This is the sharpest correction. Under dynamic placement:

- `promote_expert` / `demote_expert` copy representations using the layer's **single** `expert_bytes`
  and same-shaped destination tensors;
- a cold 2 bpw expert promoted into a 3.18 bpw GPU slot — or a hot 3.18 bpw expert demoted into the
  2 bpw CPU arena — violates the region assumption;
- the `spares` offset registers CPU experts starting at `wfirst` while `proj_dims` come from expert
  `first`, which is a mixed-K layout the worker was not built for.

So the accurate claim is: **legal without kernel changes under `EXL3_MOE_CPU_SWAP=0`**, with every
CPU-region tensor satisfying the worker's uniform layout. Dynamic support would need format-aware
swapping. T72 said "the region split is legal"; it did not say "and only statically", and that
omission would have been discovered during a build.

Also on the projection: a 37 % payload cut implies **≈26 % lower round latency**, which as a rate is
**≈35 % higher** round throughput — not "25 % faster end to end". End-to-end, at 2.88 tokens/round
and the observed fixed overhead, the honest range is **low-to-mid 70s tok/s**, not 93.

## 5. Four gate defects in the carried-over fix — all fixed

Kimi found what the three verifiers missed, in the code I had just "repaired":

| # | defect | fix |
|---|---|---|
| a | `arm_spread()` measured each run as a **mean of rates** while `primary_of()` used a **ratio of sums** — the effect and its resolution were different quantities | spread now uses `throughput()` per run, the same estimator as the primary |
| b | the floor took `max(control_interval, treatment_interval)`; that is not uncertainty on a **ratio** | variances combine: `sqrt(a² + t²)` |
| c | `withdraw` fired on `p_ratio < 1.0`, so half of all true nulls "proved" a regression while a real win needed the full floor | a loss withdraws only at `p_ratio ≤ 1 − floor`; a material acceptance loss still withdraws outright |
| d | arms could be compared on **different bases** (one with `tokens`/`seconds`, one without) | the gate now returns "cannot judge" when the two arms' bases differ |

Still weak, and recorded rather than fixed: fixed (non-noise-adjusted) corpus and secondary vetoes
outside the primary; no per-category acceptance guard; small-n t intervals are an admission of
ignorance, not power; and workload validity — live acceptance 1.58 of 4 against a corpus 2.07 — remains
the largest external weakness, which no gate fixes.

## Evidence

Re-scored runs: `t70A-ndt4-262k-20261111-055730`, `t70D-ndt4-262k-20261111-061433`,
`t70E-ndt4-262k-20261111-070757`, `t70B-ndt3-262k-20261111-063421`,
`t70C-ndt3-262k-20261111-065103`; `t61A-bf16-131k-20261110-065833`,
`t61D-bf16-131k-20261110-074755`, `t61B-exl3-131k-20261110-071514`,
`t61C-exl3-131k-20261110-073132`; `t68E-cs40-mcs381-20261110-211419`,
`t68H-cs40-mcs381-20261110-215158`, `t68J-cs40-mcs381-20261110-221041`,
`t69A-cs40-mcs373-20261110-223004`, `t69B-cs40-mcs373-20261110-224657`;
`t66A-static-262k-20261110-172259`, `t66D-static-262k-20261110-181449`,
`t66B-dynamic-262k-20261110-174018`, `t66C-dynamic-262k-20261110-175726`.

Ladder recomputed from the T66 distinct/roundprof pair; the non-linearity (34 → 36 → 44 → 75.5 µs per
additional expert) is a re-analysis of those CSVs, not a new run.

## 6. Verdicts re-scored under the corrected floor

| arm | ratio | floor | verdict | why |
|---|---:|---:|---|---|
| T70 `-ndt 3` | 0.9657 | 1.152 | **WITHDRAW** | acceptance −0.423 (material loss) |
| T61 exl3 drafter | 1.0033 | 1.135 | **WITHDRAW** | acceptance −0.205 (material loss) |
| T69 +8 experts | 1.0185 | 1.089 | INCONCLUSIVE | inside resolution |
| T66 dynamic | 1.0284 | 1.336 | INCONCLUSIVE | inside resolution; arm was also handicapped |

None flips, and now each verdict names the rule that produced it.

## 7. What the campaign should stop saying

Kimi's list, adopted verbatim as reporting rules:

1. No "closed", "zero" or "exhausted" from an underpowered null — record estimate, uncertainty, exact
   arm and regime. "Not detected at this dose" is not "the family cannot work".
2. No changing an estimator mid-campaign without rebasing every quoted comparison. Raw tokens and
   seconds should have been mandatory from the first run.
3. No profiler output without arm and row-count invariants — the ignored baseline should never have
   been able to dominate the distinct-expert statistic.
4. No `perf.py` or cache-width proxy for a production drafting decision without a transfer validation.
5. No mixing measurement and served configurations in one baseline; drafter, placement, cache, split
   and vision belong on every row, not in one campaign-wide sentence.
6. No aggregate-corpus verdict over a pre-registered agentic primary.
7. No calling model consensus empirical verification — three reviewers agreeing on code structure is
   review, not a loader test or a measurement.
8. No truncating a reviewer's findings to a word limit (this cost the campaign the finding that
   reopened the largest lever).
9. No accepting a plausible hardware story as causal identification — "≈60 GB/s sounds like DRAM"
   should trigger a microbenchmark, not a conclusion.
10. No build ladders before the sizing model accounts for every allocation (the aux tensors that were
    never priced turned two doses into floor breaches).

## 8. Its recommended experiment, and stopping rule

**A quality-gated static 2 bpw CPU-tail A/B**: control 3.18 bpw throughout; treatment identical hot
131 GPU experts at 3.18, cold 381 at 2 bpw, static placement, 262144 cache, split 381, served exl3
drafter. Two complementary measurements — a fixed-trace systems test (payload, µs/expert, `cmb`,
round latency, no trajectory confounding) and a paired free-running evaluation (corrected
`Σtokens/Σseconds`, agentic primary, pre-registered quality suite).

Pre-registered predictions: payload 1.83 → 1.15 MB; coefficient 33.9 → 21–23 µs; `cmb` 0.69 →
0.43–0.48 ms/layer; round 47.2 → ~35 ms; end-to-end +25–30 % *if* acceptance and fixed overhead hold.

**Stop if** quality fails the operator's limit, or the mechanism works fully and still lands in the
mid-70s — that would show the largest validated byte reduction does not close the gap.

**Do not stop if** payload falls 37 % and `cmb` barely moves: that falsifies the bytes law and exposes
a fixed/compute/sync term, which is new diagnostic information, not exhaustion.
