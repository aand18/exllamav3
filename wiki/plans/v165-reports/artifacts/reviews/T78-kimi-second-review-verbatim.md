# Kimi 3.1 — second review, verbatim

Recorded so the reasoning survives the summary. Submitted against HEAD `3d43921d`, in response to the
disposition list in `T77`. Session `ses_e3582826cffePj7JCV78jDnNLH`.

---

## Overall verdict

This is a substantive response, not merely a tidy one, but it is **not yet an accurate "all findings
addressed" record**.

The acceptance arithmetic and static-placement correction are good. The largest remaining problems are:

1. **Two of the four gate repairs are incomplete**, so "all fixed" and the re-scored floors are not yet dependable.
2. `v165-STATE.md` still says **"all zero," "both factors are closed," and "the count lever is closed"**, despite the corresponding arms being inconclusive.
3. The record overstates the payload hypothesis as having **"survived intervention."** No byte-width intervention has run; that is what the 2 bpw experiment would test.

I do **not** see systematic misrepresentation of my review in your favour. I do see several places where
the response adopted my narrower vocabulary but retained a stronger conclusion underneath it.

## 1. Finding-by-finding assessment

### 1. "Decode exhausted" was too broad — mostly addressed, with residual overclaim

The replacement wording in T77 §1 is appropriately scoped to the placement, small-split, pinning and
fixed-window arms actually run. Explicitly distinguishing `-ndt 3` from "better acceptance at the same
five-row cost" is the important correction.

However, **"removing the fourth draft loses more than it saves" is still stronger than the displayed
evidence**. T70 has: primary ratio `0.9657`; floor `1.152`; withdrawal driven by the material
acceptance loss of `−0.423`. That justifies **withdrawing the arm**. It does not establish a
statistically resolved net throughput regression: the negative throughput point estimate lies inside
the stated resolution. The defensible sentence is:

> The tested `-ndt 3` arm is withdrawn because its acceptance loss is material and its throughput
> point estimate is negative; its exact net throughput effect is unresolved at the measured floor.

More importantly, STATE.md:211–219 still says: "Both factors are closed by measurement"; the
placement/count arms are "all zero"; "the count lever is closed." T69 and T66 are explicitly
**INCONCLUSIVE** in T77 §6. Calling them "zero" or "closed" directly violates T77 reporting rule 1.
Decision closure—"do not spend more runs on these doses because expected returns are poor"—is
legitimate; empirical closure is not.

### 2. Accepted tokens per round was the untouched factor — addressed

The arithmetic is correct: `2.88 / 57 ≈ 50.5 ms/round`; at that fixed round cost, 93 tok/s requires
`acc ≈ 3.70` of four; across 55–58 tok/s, the requirement is approximately `3.62–3.87`; 93 tok/s at
2.88 tokens/round implies a 31 ms round, not 25 ms. This does not prove that better drafting can reach
the target; it makes the size of the required improvement explicit, which was the point.

One provenance requirement remains: before using these numbers to price the proposed experiment,
identify the exact control runs and regime behind **57 tok/s, 2.88 tokens/round and the 47.2 ms round
in T77 §8**. T77 §8 names the served exl3 drafter, while STATE repeatedly warns that many campaign
figures came from the BF16 drafter. I am not asserting that the numbers are incompatible; I am saying
the record does not demonstrate that they form one same-regime calculation.

### 3. The ladder is nonlinear — partially addressed

The marginal-cost recomputation and the observational/interventional distinction are valuable and
correctly recorded. In particular, stating that T69 did not reproduce the ladder prevents the 75.5 µs
segment from being treated as a causal saving per removed expert.

But three qualifications are needed:

- **33.9 µs/expert is a ratio of medians, not an average of the reported marginal slopes.** It need not share an estimand with the 47.5 mean slope or the 75.5 endpoint segment. It can be reported as an average intensity, but not used as a causal marginal coefficient.
- T69's failure to reproduce the ladder does **not** strengthen the conclusion that the count lever is "closed." It shows that the ladder does not identify the effect of changing count.
- Nor does it show that the payload mechanism "survives intervention." No intervention changed bytes per expert while holding count and trajectory fixed. T69 therefore leaves the simple bytes story less identified, not newly validated.

The accurate conclusion is:

> Count changes at the tested doses have not produced resolved throughput gains, and the observational
> ladder does not explain their mechanism; reducing bytes per expert at fixed count remains an
> untested hypothesis.

That still fully motivates the 2 bpw falsification experiment. It does not require pretending the
payload law is already validated.

### 4. Mixed-K is static-only — constraint addressed; projection needs consistent conditioning

The static-placement correction is properly represented. STATE.md correctly includes both
`EXL3_MOE_CPU_SWAP=0` and `EXL3_MOE_CPU_PROMOTE=0`. T77's actual experiment recipe should repeat both
assertions rather than relying on "static placement" alone.

The reciprocal arithmetic is also right: approximately 26% lower latency corresponds to approximately
35% higher round rate. Low-to-mid-70s is a more credible conditional projection than 93 tok/s.

But it must remain explicitly conditional:

> **If** `cmb` scales approximately linearly with payload, acceptance and routing remain comparable,
> and fixed overhead behaves as assumed, the model projects low-to-mid-70s end-to-end.

The payload reduction is known from the representation change; the round-latency and end-to-end
results are predictions, not established implications. T77 §8 correctly treats them as pre-registered
predictions; §4's "implies" and "honest range" are occasionally more definitive than that.

### 5. Four gate defects — only partially fixed

This is the clearest place where the response claims more than the code implements.

#### (a) Spread and primary still do not use the same estimator — not fixed

`primary_of()` computes, when complete columns exist:
`Σ(category weight × tokens) / Σ(category weight × seconds)` across the primary categories
(`bench/spec_primary.py:193–207`). But `arm_spread()` computes each category's throughput and then
takes a **weighted arithmetic mean of those category rates** (`spec_primary.py:232–241`). Those
quantities differ whenever category rates and durations differ. Calling `throughput()` within each
category fixes only the within-category estimator; it does not make the per-run primary the same
ratio-of-sums as `primary_of()`. Consequently, the effect and its resolution can still be different
quantities, and the re-scored floors in T77 §6 do not yet validate the claimed repair.

#### (b) Combining primary-arm variances — implemented

The primary now uses `sqrt(a² + t²)` at `spec_primary.py:250–256`. That addresses the specific defect
for independent-arm intervals. The category guards still use the maximum of the two arm spreads at
line 318, so this remains a primary-only repair.

#### (c) Negative point estimates and material acceptance loss — implemented as described

The code now distinguishes a resolved primary loss from a mere negative point estimate, and material
acceptance loss withdraws outright (`spec_primary.py:333–361`). That is the intended improvement. A
smaller residual issue is that `1 ± floor` is not exactly symmetric on a ratio scale. A log-ratio
criterion, or a reciprocal loss threshold such as `1 / (1 + floor)`, would make treatment/control
reversal symmetric. This is secondary to fixing (a) and (d). Also, T77's table reports `1 + floor`
while its prose discusses `1 − floor`; defining the fractional uncertainty separately would prevent
misreading.

#### (d) Different-basis refusal — incomplete

`primary_of()` requires tokens and seconds on **all** relevant rows before selecting the ratio-of-sums
estimator (`spec_primary.py:199–207`). But `basis()` reports ratio-of-sums when **any** row has tokens
and seconds (`spec_primary.py:139–144`). Therefore two arms can both be labelled "ratio of sums" while
one actually has complete columns and the other falls back inside `primary_of()`. The comparison at
lines 282–284 will then allow exactly the mixed-basis comparison it claims to refuse. There is a
related partial-column hazard in `throughput()` itself: tokens and seconds are collected
independently, so incomplete or differently incomplete columns can produce sums over different row
subsets. The basis check must follow the estimator actually selected, require complete matched
token/second pairs, and test missingness per arm and per run—not infer it from `any()`.

#### Verification gap

I found no focused T77 regression tests in `bench/test_spec_primary.py` for: weighted-mean versus
ratio-of-sums divergence; combined-arm uncertainty; the new withdrawal boundary; complete/partial/
mismatched token-second bases. Re-scoring old arms and observing no verdict flips is not a substitute
for those tests. The module docstring also still describes `1.96 × SEM`, `floor = max(0.02, spread)`
and `withdraw = p_ratio < 1.00` as current rules at lines 8–19, despite the later code differing.
Therefore T77 §5's **"all fixed" is currently an overclaim**.

### 6. Ten procedural rules — faithfully recorded, inconsistently applied

The list in T77 §7 accurately preserves the substance of the review. But recording is not yet
adoption: STATE's "zero/closed" language breaches rule 1; beginning another GPU build ladder with
`suh`/`svh` unpriced would breach rule 10; the unnamed regimes behind the projection inputs sit
uneasily with rule 5's requirement that drafter, placement, cache and split accompany each comparison.

### 7. Experiment and stopping rule — faithfully recorded

T77 §8 preserves the recommended experiment and, importantly, the correct falsification logic: quality
failure stops the arm; full mechanism success landing in the mid-70s closes the claim that this byte
reduction closes the gap; reduced payload with little `cmb` movement is diagnostic falsification of
the bytes law, not evidence of exhaustion. That last distinction is correct and should remain. Before
execution, "quality-gated" needs a named numerical quality limit, and "full mechanism success" should
be tied to the measured payload/`cmb`/round predictions rather than judged retrospectively.

## 2. Is the record faithful to the review?

Broadly, yes. I do not see my position materially misquoted. The passages that are over-claimed in
your favour are instead:

1. **"Four gate defects … all fixed"** — two are incomplete.
2. **"All zero," "both factors closed," "count lever closed"** — incompatible with the displayed inconclusive verdicts and with your own reporting rule.
3. **"Payload … survives intervention"** — no byte-width intervention has yet tested it.
4. **"Removing the fourth draft loses more than it saves"** — a valid withdrawal decision, but not a resolved net-throughput measurement.
5. **"Low-to-mid 70s"** — reasonable as a conditional prediction, not as an experimentally established range.

These are substantive wording issues because they determine whether the next experiment is presented as
a falsification test or as confirmation of an already-established mechanism.

## 3. Most serious omission

**For the decision to start the next build/run, the unpriced `suh`/`svh` allocations are the most
serious omission.** That is not merely an unclosed explanatory detail: approximately 950 MiB is missing
from the sizing model, and the omission has already contributed to VRAM-floor breaches. It threatens
both **operational safety and feasibility** on a live 24 GiB box; and **experimental attribution**,
because a forced split change alters GPU/CPU expert counts, payload and round latency alongside the
intended representation change.

The missing fixed-trace runner is the next most important omission and the largest **scientific** one:
without it, routing and trajectory changes can confound the very payload/`cmb` mechanism the
experiment is meant to test. But the sizing defect comes first operationally because it has already
produced failed doses.

The memory-bandwidth microbenchmark would improve diagnosis but is not a prerequisite for the A/B. A
new same-session control makes rebuilding historical absolute tok/s less urgent for this experiment.
`-dds` is orthogonal, and awaiting quality sign-off before building is appropriate rather than a defect.

## 4. Are the gate fixes prioritised correctly?

**Yes in category, but the first category must actually be completed before adding more gate
sophistication.** My order would be:

1. **Finish the primary estimator and basis repairs**, add focused synthetic tests, then regenerate the T61/T66/T69/T70 scores. At present, the quoted floors are not reliable validation.
2. **Fix workload validity and add the fixed-trace systems measurement.** As T77 itself says, no statistical gate repairs a corpus whose acceptance differs materially from live traffic.
3. **Add a per-category acceptance guard before relying on free-running 2 bpw/drafting decisions.** Aggregate acceptance can conceal a severe regression in Agentic-curl or another category.
4. **Increase same-regime paired replication rather than trying to "fix" small-n t intervals.** The wide t intervals are honestly reporting insufficient information. The remedy is more paired evidence or lower-variance measurements, not a friendlier distribution.
5. **Then make corpus and secondary vetoes noise-aware.** Their fixed thresholds are real weaknesses, but with primary floors currently around 9–34%, they are not the immediate source of most unresolved verdicts.

## 5. Would I authorise the 2 bpw build?

**Yes to a compile-only, static-placement falsification build; no yet to a performance A/B or
deployment.** If "build" includes loading and measuring on the GPU, my answer is **not until the sizing
prerequisite is satisfied**.

The experiment is still worth doing: it is the first proposed arm that directly changes bytes per
expert at fixed count, its realistic upside remains meaningful even if it does not reach 93 tok/s, and
its stopping rule now permits a clean negative mechanistic result.

The single sentence I would require in the record is:

> **This is a static-only, non-deployment falsification experiment with a named numerical operator
> quality limit; before any performance A/B, `suh`/`svh` must be included in the sizing model, both
> exact same-regime split-381 arms must clear the VRAM/RAM floors with stated margins,
> `EXL3_MOE_CPU_SWAP` and `EXL3_MOE_CPU_PROMOTE` must be zero, and the fixed-trace runner must be
> ready.**

That sentence distinguishes permission to investigate from permission to deploy, and prevents the
build from repeating the known sizing and trajectory-confounding failures.

## Key Learnings (its own)

1. Calling `throughput()` inside each category does not make a weighted mean of category rates equivalent to a cross-category ratio of token/second sums.
2. A basis guard based on `any()` cannot validate an estimator selected using `all()`; partial token/second columns can silently permit mixed-basis comparisons.
3. A gate's `WITHDRAW` decision is not necessarily a statistically resolved regression, and an inconclusive count intervention neither closes the count family nor validates the byte-width hypothesis.
