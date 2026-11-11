# Kimi 3.1 — critique of the plan (T79), verbatim

Third pass, on `T79`/`T80` at HEAD `d2e9f75d`. Recorded in full; the corrections it forced are in
`T81-kimi-plan-critique-fixes.md`. Session `ses_e34e2d58fffeTnctcs0Tq2vRNV`.

---

## Updated assessment: substantially improved, but **not yet validated**

The response correctly adopts almost all of the plan-level feedback. The closure wording and T77 floor
conventions are fixed; the ordering now prioritises fixed trace and workload validity; provenance
includes placement/split; the runner includes routes; and the authorisation clause closes the
compile-only/A/B loophole.

However, checking the new tests against the implementation exposed a substantive remaining defect.

### 1. **5b is still not fixed in the code**

`bench/spec_primary.py:368` still calculates category spread using:

> `max(FLOOR_MIN, category_spread(control), category_spread(treat))`

It does **not** combine the two arms' variances with `sqrt(a² + t²)`. This directly contradicts T79's
**"5b addressed"** and T80's claim that the category variance repair survived validation.

The new test does not catch it: `test_category_guard_uses_combined_variance_and_reciprocal_form()` at
lines 252–257 merely asserts:

> `0.0 < line <= 1.0`

That passes with `max()`, with `sqrt(a²+t²)`, and with many incorrect reciprocal formulas. Its name
claims considerably more than it tests.

### 2. **5f remains partial despite 21 passing tests**

I directly executed all 21 test functions: **21 passed**. But passing is not the same as pinning the
repairs.

- The new **primary combined-arm uncertainty** test is meaningful.
- The category test above does not verify combined variance or the exact reciprocal value.
- `test_a_resolved_loss_is_the_reciprocal_of_the_pass_bound()` uses a −14 % loss far beyond either
  candidate threshold, then checks an inequality with an arbitrary `+0.02` allowance. The old `1−f`
  implementation would also pass this test.

Use exact numerical assertions with inputs where `max(a,t)` differs materially from `sqrt(a²+t²)`, and
a loss between `1−f` and `1/(1+f)` so the old and new rules produce different verdicts.

Consequently, **§4.1 cannot remain "addressed"**: fix the category implementation, strengthen the
tests, then re-score and confirm the recorded verdicts.

Minor related cleanup: `spec_primary.py:387` still describes `p_ratio <= 1 − floor`, immediately
before lines 388–391 correctly reject that convention.

### 3. The revised totals are still arithmetically wrong

The table contains **35 rows**. Reading its literal statuses gives:

- 16 addressed
- 2 partial
- 10 recorded
- **7 open**

That totals 35. The published **9 open** makes the claimed totals sum to **37**.

There are also consistency errors:

- **§3 is not atomic** and is misclassified under the stated rule: unimplemented sizing and fixed-trace
  work are actions, not merely a "decision or standing rule".
- **O3's explanation is stale**: it says the closure-language instance survives, although 1c is now
  marked addressed.
- §4.3 points to **item 4**, but the category guard is now **item 7**.
- §5 says the loophole is closed in **item 7**, but means **item 10**.

Do not publish another corrected total until 5b/5f are resolved and §3 is split or its counting
treatment clarified.

## What I now accept

- The **plan ordering is sound**.
- Workload validity, replication, and noise-aware vetoes are properly scheduled.
- The **compile-only authorisation clause satisfies my precondition**; performance A/B remains
  correctly blocked.
- The numerical quality limit is correctly identified as an **operator decision**. The artifacts do not
  supply a defensible number for me to invent; it needs a named baseline, quality metric/evaluation,
  and maximum tolerable degradation.

So item 2 remains the only outstanding **operator decision**, but it is not the only outstanding
repair. The next engineering correction is the category variance implementation and its ineffective
tests.

---

## Addendum (same session, after the fixes were described)

One practical addition: **item 10's compile-only build does not need to wait behind items 5–9.** My
authorisation permits it after recipe assertions, the record sentence/quality limit, sizing, and
provenance. Moving it earlier is optional, but could provide useful compiler feedback while
fixed-trace and workload-validity work proceeds. The performance A/B must remain blocked.

Otherwise, no new substantive findings. Minimum close-out remains:

1. Fix `spec_primary.py:368`.
2. Make the tests genuinely fail against the old `max()` and `1−f` implementations.
3. Re-score the affected arms.
4. Correct the register's totals and stale references.

The numerical quality limit remains an operator input, not something the reviewer should invent.
