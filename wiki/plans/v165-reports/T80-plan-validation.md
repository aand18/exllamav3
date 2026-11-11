# T80 — Kimi validated the plan and found three more overclaims

Sent the first version of `T79` (claims table + seven-action plan) back to Kimi for validation. Its
verdict: the plan is **not yet an accurate disposition of the review**, and the back half is
self-serving — it prefers cheap, visible gate work over the harder validity and attribution problems.

## What it found, and what changed

| # | its finding | correction |
|---|---|---|
| 1 | **`1c` was not fixed**: `v165-STATE.md:211` still literally read "Both factors are closed by measurement". Adding "not detected at these doses" elsewhere did not remove the contradictory sentence | replaced with "**No family below is empirically closed.** What is closed is the *decision* to spend more runs on these doses" |
| 2 | **Rule 1 was therefore not satisfied** (I had recorded it as done) | now satisfied by the same edit; 6b remains *partial* because rules 5 and 10 are still open |
| 3 | **`5f` was partly fixed**: the five tests covered estimator agreement, basis and withdrawal, but not **combined-arm uncertainty**, the **category variance combination**, or **reciprocal-threshold symmetry** | three tests added; `bench/test_spec_primary.py` now **21 passed** |
| 4 | **`5d` was not fixed at report level**: the complaint was T77's table showing `1+f` while its prose discussed `1−f`; editing a module docstring does not close a report-level contradiction | T77's floor column is now labelled `1 + f` and defines the reciprocal loss bound locally |
| 5 | **The totals were not reproducible**: 31 rows, 29 classified, no counting rule | counting rule stated; rows made atomic; totals **16 addressed · 2 partial · 10 recorded · 9 open**, and the old "20 addressed, 8 open" withdrawn |
| 6 | **Ordering was wrong**: putting the per-category guard before the fixed-trace runner and workload validity, then making the runner optional, buried the urgent scientific work | re-ordered: recipe assertions → quality limit → sizing → provenance → **fixed-trace and workload validity in parallel** → category guard → replication → noise-aware vetoes → compile-only build |
| 7 | **"Choose between 6 and 7" was a loophole**: a compile-only build cannot show that 2 bpw moves tok/s past noise, so it can never substitute for the runner | Part C adds the explicit clause: item 10 compile-only, no GPU load or timing; any A/B additionally requires the runner, sizing results for both split-381 arms with margins, both placement flags zero, and the operator quality gate |
| 8 | **Missing actions**: workload validity, paired replication, noise-aware vetoes were acknowledged but not scheduled | added as items 6, 8 and 9 |
| 9 | **Cost estimates were optimistic in ways that matter**: the quality limit needs operator sign-off, not 30 minutes of typing; provenance may require a control run; the sizing item needs an acceptance step (both arms cleared, margins recorded), not just tensor arithmetic | item 2 separates the sentence from the sign-off; item 3 flags the possible control run; item 1 carries the acceptance step |
| 10 | The fixed-trace runner must fix **routes as well as tokens and rows**, or the payload-versus-routing confound survives | item 5 now says so |

## Confirmed by it

- The `33.9 µs` wording is now correct: a ratio of medians / average intensity, not a causal marginal
  coefficient, with payload left explicitly untested.
- `5a`, `5b`, `5c`, `5e`, `5g` survive as implementation fixes — with the caveat that this is a
  documentary check, not an independent code re-audit.
- Its compile-only authorisation is satisfied by the revised plan; a performance A/B is not.

## Its sharpest line

> A plan written by the person being reviewed has an obvious bias toward actions that are cheap and
> look responsive; if that is what you see, say so.

That is what it saw, and it was right: my first draft did six cheap gate and documentation items,
put a seventh cheap one ahead of the two-day scientific work, and offered the two as interchangeable.
The corrected plan commits to the fixed-trace runner and workload validity as required work, and makes
the build explicitly compile-only.

## Evidence

Artifacts changed in this task: `wiki/plans/v165-STATE.md` (closure sentence),
`wiki/plans/v165-reports/T77-kimi-review.md` (floor convention),
`bench/test_spec_primary.py` (3 tests), `wiki/plans/v165-reports/T79-claims-and-plan.md` (rewritten).

No GPU work; no measurement claims of its own beyond re-reading the existing runs:
`t61A-bf16-131k-20261110-065833`, `t61B-exl3-131k-20261110-071514`,
`t61C-exl3-131k-20261110-073132`, `t61D-bf16-131k-20261110-074755`,
`t66A-static-262k-20261110-172259`, `t66B-dynamic-262k-20261110-174018`,
`t66C-dynamic-262k-20261110-175726`, `t66D-static-262k-20261110-181449`,
`t68E-cs40-mcs381-20261110-211419`, `t68H-cs40-mcs381-20261110-215158`,
`t68J-cs40-mcs381-20261110-221041`, `t69A-cs40-mcs373-20261110-223004`,
`t69B-cs40-mcs373-20261110-224657`, `t70A-ndt4-262k-20261111-055730`,
`t70B-ndt3-262k-20261111-063421`, `t70C-ndt3-262k-20261111-065103`,
`t70D-ndt4-262k-20261111-061433`, `t70E-ndt4-262k-20261111-070757`.
