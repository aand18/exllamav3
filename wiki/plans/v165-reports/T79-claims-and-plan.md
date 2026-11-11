# T79 — Kimi's claims, their status, and the plan that follows

Revised after Kimi reviewed the first version of this register and found it self-serving. Its
criticisms are recorded in `T80`, the corrections applied here. Companion documents:
`artifacts/reviews/T78-kimi-second-review-verbatim.md` (the review in full) and `T78-kimi-second-pass.md`
(what was repaired).

## Counting rule (added — the previous totals were not reproducible)

Each row is one atomic claim. **addressed** = the artifact now contains the fix, and where the claim
named a test, the test exists and passes. **partial** = something changed but part of the claim
survives. **open** = no work done. **recorded** = the item is a decision or standing rule rather than
an action; it is never counted as addressed.

## Part A — claims and status

| # | claim | status |
|---|---|---|
| O1 | Response substantive; "all findings addressed" not accurate | **addressed** (T78 opens by conceding it) |
| O2 | No systematic misrepresentation of the review | **recorded** |
| O3 | Narrower vocabulary adopted over stronger conclusions | **partial** — 4 of 5 corrected; the closure-language instance survived (see 1c) |
| 1a | "Decode exhausted" broader than the evidence | **addressed** |
| 1b | "Removing the 4th draft loses more than it saves" exceeds T70 | **addressed** |
| 1c | STATE still asserted empirical closure | **addressed in T80** — `STATE.md:211` now reads "No family below is empirically closed. What is closed is the *decision* to spend more runs on these doses" |
| 1d | Decision closure legitimate, empirical closure not | **recorded** as a wording rule |
| 2a | 3.70-of-4 and the 31 ms round are correct | **recorded** |
| 2b | Provenance of 57 tok/s / 2.88 tok-per-round / 47.2 ms unnamed | **open** |
| 3a | Nonlinear ladder recomputation valuable | **recorded** |
| 3b | 33.9 µs is a ratio of medians, not a marginal coefficient | **addressed** |
| 3c | T69's non-reproduction does not license "count lever closed" | **addressed** |
| 3d | Payload does not "survive intervention" | **addressed** |
| 4a | Static-placement constraint represented | **recorded** |
| 4b | Recipe must assert `SWAP=0` **and** `PROMOTE=0` explicitly | **open** |
| 4c | 26 % latency ≈ 35 % rate | **recorded** |
| 4d | 70s projection must stay conditional | **addressed** |
| 5a | Spread and primary used different estimators | **addressed** — one `primary_from_rows()` |
| 5b | Variance combination primary-only; categories used `max()` | **addressed** |
| 5c | Ratio scale asymmetric | **addressed** — `1/(1+f)` both sides |
| 5d | T77's table showed `1+f` while its prose discussed `1−f` | **addressed in T80** — T77 now labels the column and defines both bounds locally; the earlier docstring edit alone did not close this |
| 5e | Basis guard used `any()` vs estimator's `all()` | **addressed** |
| 5f | No focused tests for the repairs | **addressed in T80** — 21 tests, adding combined-variance, category-guard reciprocal, and resolved-loss-boundary cases to the earlier five |
| 5g | Module docstring described superseded rules | **addressed** |
| 6a | Ten procedural rules faithfully recorded | **recorded** |
| 6b | Rules recorded but inconsistently applied — rule 1 breached in STATE, rule 10 at risk before a ladder, rule 5 with unnamed regimes | **partial** — rule 1 now satisfied (T80); 5 open under 2b; 10 open under item 1 |
| 7a | Experiment and stopping rule faithfully recorded | **recorded** |
| 7b | "Quality-gated" needs a named numerical limit; success tied to predictions | **open** |
| §3 | Ranked #1 omission: `suh`/`svh` unpriced (~950 MiB); fixed-trace runner is the larger scientific omission | **recorded** (both unimplemented) |
| §4.1 | Finish estimator/basis **and add the focused tests** before re-scoring | **addressed in T80** (tests now exist; re-score unchanged) |
| §4.2 | Fix workload validity; add fixed-trace measurement | **open** |
| §4.3 | Per-category acceptance guard | **open** (item 4) |
| §4.4 | More same-regime paired replication instead of friendlier intervals | **open** — now an explicit action (item 8) |
| §4.5 | Noise-aware corpus and secondary vetoes | **open** — now an explicit action (item 9) |
| §5 | Authorise compile-only static build, not a GPU A/B, until sizing is fixed | **recorded**, with the loophole closed in item 7 |

Totals under the stated rule: **addressed 16 · partial 2 · recorded 10 · open 9**. The earlier
"20 addressed, 8 open" was not reproducible and is withdrawn.

## Part B — the plan, re-ordered after validation

| # | action | reason | cost |
|---|---|---|---|
| **1** | Price `suh`/`svh` and biases in `bench/size_check.py`; **then run the repaired model on both exact same-regime split-381 arms and record VRAM/RAM margins** | Ranked #1 omission; ~950 MiB unpriced caused two floor breaches. Adding tensors alone does not establish clearance — the acceptance step is part of the item | ~1 h + a sizing run, no GPU |
| **2** | Insert the **full** required record sentence into the 2 bpw spec and obtain the **named numerical quality limit from the operator** | Prevents a pass criterion chosen after the result. The limit is an operator decision, not something the plan author invents | typing minutes; sign-off is on you |
| **3** | Name the provenance of the projection inputs with **run ids, drafter, placement, cache width and split**; if no same-regime control exists, plan one | The 3.70-of-4 target prices the experiment. Rule 5 requires those fields per comparison | ~1 h if metadata suffices; **may become a control run** |
| **4** | Add `EXL3_MOE_CPU_SWAP=0` / `EXL3_MOE_CPU_PROMOTE=0` assertions to the recipe | Minutes; removes a documented build-time trap | minutes |
| **5** | **Fixed-trace systems runner** (fix tokens, rows **and routes**; gate on `ms/round`) | Largest scientific omission: replaying tokens without routes leaves the payload-vs-routing confound intact. This is now committed, not optional | days |
| **6** | **Workload validity**: reconcile corpus acceptance (2.07) against live (1.58), and state which conclusions survive for live traffic | The largest external weakness; no gate repairs it, and the plan previously only acknowledged it | days |
| **7** | Per-category acceptance guard | Aggregate acceptance hides an Agentic-curl regression (T61 −5.74 %, T70 acc −0.35 there) | ~1 h + tests |
| **8** | Same-regime paired replication for the next drafting arm (n from a pilot variance estimate, not a guess) | The wide t intervals honestly report insufficient information; the remedy is evidence, not a friendlier statistic | per arm |
| **9** | Make the corpus and secondary vetoes noise-aware | Real weakness, but with floors at 9–34 % it is not the source of most unresolved verdicts | ~1 h |
| **10** | The 2 bpw build — **compile-only** | Authorised on that basis alone | days |

**Order: 4 → 2 → 1 → 3 → then 5 and 6 in parallel → 7 → 8 → 9 → 10 (compile-only).**

Corrected from the first draft: item 7 (per-category guard) no longer precedes the scientific work.
Kimi's objection was that putting another gate repair ahead of the fixed-trace runner and workload
validity, and then making the runner optional, buried the urgent work behind cheap visible items.

## Evidence

No run of its own: a claims register and a plan, which is why the post gate reports "no run ids". The
measurements referred to are `t61A-bf16-131k-20261110-065833`, `t61B-exl3-131k-20261110-071514`,
`t61C-exl3-131k-20261110-073132`, `t61D-bf16-131k-20261110-074755`,
`t66A-static-262k-20261110-172259`, `t66B-dynamic-262k-20261110-174018`,
`t66C-dynamic-262k-20261110-175726`, `t66D-static-262k-20261110-181449`,
`t68E-cs40-mcs381-20261110-211419`, `t68H-cs40-mcs381-20261110-215158`,
`t68J-cs40-mcs381-20261110-221041`, `t69A-cs40-mcs373-20261110-223004`,
`t69B-cs40-mcs373-20261110-224657`, `t70A-ndt4-262k-20261111-055730`,
`t70B-ndt3-262k-20261111-063421`, `t70C-ndt3-262k-20261111-065103`,
`t70D-ndt4-262k-20261111-061433`, `t70E-ndt4-262k-20261111-070757`.

Tests: `bench/test_spec_primary.py` 21 passed.

## Part C — the authorisation clause (added)

> **Item 10 is compile-only: no GPU load, timing, performance A/B or deployment. Any performance A/B
> additionally requires the fixed-trace runner, recorded sizing results showing that both exact
> same-regime split-381 arms clear the VRAM/RAM floors with stated margins,
> `EXL3_MOE_CPU_SWAP=0`, `EXL3_MOE_CPU_PROMOTE=0`, and satisfaction of the named operator quality
> gate. Choosing item 10 does not replace or defer item 5 as an A/B prerequisite.**

This closes the loophole in "choose between 6 and 7": a compile-only build cannot itself show that
2 bpw moves tok/s past noise, so it never substitutes for the runner.

## Part D — what the plan still does not do

- It does not reopen the handoff, placement, split or pinning families: those are "not detected at
  these doses", and larger doses are a poor-ROI decision recorded in STATE.md.
- It does not promise 93 tok/s. The conditional projection is low-to-mid 70s, and the falsification
  logic stands: payload −37 % with `cmb` barely moving falsifies the bytes law — new information, not
  exhaustion.
- It does not claim the methodology is settled. Workload validity (item 6) is the largest external
  weakness and is now scheduled rather than acknowledged.
