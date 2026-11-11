# T81 — Kimi's third pass: 5b was still not fixed, and the totals were wrong

Its critique is recorded verbatim in `artifacts/reviews/T81-kimi-plan-critique-verbatim.md`. This is
what it found and what changed.

## 1. `5b` was still not fixed in the code — correct

`bench/spec_primary.py:368` read

```python
sp = max(FLOOR_MIN, category_spread(c, control) or 0.0, category_spread(c, treat) or 0.0)
```

while `T79` marked 5b **addressed** and the primary path already combined variances. The category path
had been left on `max()` through two revisions — the same class of failure as the estimator repair two
passes earlier: the fix was applied where it was being looked at, not everywhere the rule applies.

Now:

```python
ca, ct = category_spread(c, control) or 0.0, category_spread(c, treat) or 0.0
combined = (ca ** 2 + ct ** 2) ** 0.5
sp = max(FLOOR_MIN, combined)
cat_lines[c] = loss_bound(sp)
```

and the per-category spreads are exposed in the output (`category_spread`) so a test can assert the
combination exactly rather than asserting a shape.

## 2. The tests did not pin what their names claimed — correct

- `test_category_guard_uses_combined_variance_and_reciprocal_form()` asserted only
  `0.0 < line <= 1.0`, which passes under `max()`, under `sqrt(a²+t²)`, and under several wrong
  reciprocal forms. Replaced with `test_category_guard_uses_combined_variance_exactly()`, which reads
  the reported spreads and asserts `combined == sqrt(a²+t²)` to 1e-12, that `combined > 1.2 × max(a,t)`
  when both arms are noisy (the discriminating case), and that the guard equals `1/(1+max(0.02,
  combined))` exactly.
- `test_a_resolved_loss_is_the_reciprocal_of_the_pass_bound()` used a −14 % loss, past either
  threshold, plus an arbitrary `+0.02` allowance — the old `1−f` rule passed it too. Replaced with
  `test_bounds_are_reciprocal_and_the_gap_between_them_classifies_correctly()`, which for
  `f ∈ {0.02, 0.05, 0.10, 0.25}` asserts the exact bounds and then classifies a ratio in the
  **discriminating band** `(1−f, 1/(1+f))` — above the old linear line, at or below the reciprocal
  bound — where the two rules give different verdicts. `classify_ratio(0.905, 0.10) == "loss"` while
  the old rule called it "inside".

The bounds are now `pass_bound(f) = 1 + f` and `loss_bound(f) = 1/(1+f)`, with `classify_ratio()`
holding the comparison in one place, so the convention cannot drift between the ratio and the
categories again.

Tests: **22 passed**.

## 3. The stale comment — correct

`spec_primary.py:387` described `p_ratio <= 1 − floor` immediately above code that rejects it. The
comment now states the reciprocal convention and why `1 − floor` was wrong.

## 4. The totals were still wrong — correct

35 rows; the literal statuses gave **16 addressed · 2 partial · 10 recorded · 7 open**, not 9 open.
`T79` is corrected, and:

- **§3 split** into two atomic rows (`suh`/`svh` sizing; fixed-trace runner) with an explicit note that
  neither is done — an unimplemented action is **open**, not "recorded";
- **O3** no longer says the closure-language instance survives (1c is addressed);
- §4.3 now points at **item 7** (the category guard) and §5 at **item 10** (the build).

New totals under the stated rule: **16 addressed · 2 partial · 11 recorded · 8 open = 37 rows**. Every
row counted once, and the arithmetic is stated so it can be checked rather than trusted.

## 5. Re-score after the category repair

| arm | verdict |
|---|---|
| T70 `-ndt 3` | WITHDRAW |
| T61 exl3 drafter | WITHDRAW |
| T66 dynamic | INCONCLUSIVE |
| T69 +8 experts | INCONCLUSIVE |

No verdict moved — the category guard is stricter now, so a flip was possible; none occurred, which is
the re-score the critique asked for.

## What it accepted, unchanged

- the plan ordering;
- that workload validity, replication and noise-aware vetoes are scheduled;
- that the compile-only clause satisfies its precondition and a performance A/B stays blocked;
- that the numerical quality limit is an **operator decision** requiring a named baseline, metric and
  maximum tolerable degradation — it declined to invent one, as it should.

## Evidence

Re-scored after the category repair: `t70A-ndt4-262k-20261111-055730`,
`t70D-ndt4-262k-20261111-061433`, `t70E-ndt4-262k-20261111-070757`,
`t70B-ndt3-262k-20261111-063421`, `t70C-ndt3-262k-20261111-065103`;
`t61A-bf16-131k-20261110-065833`, `t61D-bf16-131k-20261110-074755`,
`t61B-exl3-131k-20261110-071514`, `t61C-exl3-131k-20261110-073132`;
`t68E-cs40-mcs381-20261110-211419`, `t68H-cs40-mcs381-20261110-215158`,
`t68J-cs40-mcs381-20261110-221041`, `t69A-cs40-mcs373-20261110-223004`,
`t69B-cs40-mcs373-20261110-224657`; `t66A-static-262k-20261110-172259`,
`t66D-static-262k-20261110-181449`, `t66B-dynamic-262k-20261110-174018`,
`t66C-dynamic-262k-20261110-175726`.

Tests: `bench/test_spec_primary.py` 22 passed.

## The pattern worth naming

Three consecutive external passes found the *same* defect shape: a rule fixed at the site under
discussion and left unfixed at a sibling site — the estimator (primary fixed, spread not), the basis
guard (`all()` in the estimator, `any()` in the report), and now the variance combination (primary
combined, categories maxed). The lesson is not "be more careful"; it is that a rule with two
implementations needs one implementation, which is why `pass_bound`/`loss_bound`/`classify_ratio` now
exist as the single source.
