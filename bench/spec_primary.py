#!/usr/bin/env python3
"""The agentic-weighted primary metric for a spec_decode arm pair, with its verdict.

    primary = workload-weighted tok_s over {Agentic; code (5), Agentic; curl (4), Coding (3)}

Gate (revised twice: T73, then T77/T78 after two external reviews found the first repair wanting).
This is the CURRENT rule; the superseded forms are listed below so the older reports can still be read.

    primary  = 5:4:3 weighted tok/s over {Agentic; code, Agentic; curl, Coding},
               as sum(tokens)/sum(seconds) when every row of a run carries both columns,
               else the 5:4:3 weighted mean of the per-category rates. `basis()` says which, and
               the gate REFUSES to compare two arms measured on different bases.
    run_value = the same function (`primary_from_rows`) applied per run - the effect and its
               resolution must be the same quantity.
    spread   = t(0.975, n-1) * SEM / mean over the per-run values
    floor    = max(0.02, sqrt(spread_control^2 + spread_treat^2))   # variances add for a ratio
    pass     = primary ratio >= 1 + floor
               AND every primary category >= 1 / (1 + its own combined spread)
               AND acceptance: |delta| <= 0.05 (kernel arm) or delta >= -0.05 (`--drafting`)
               AND corpus ratio >= 1.00 with no non-trivial category < 0.97
    withdraw = primary ratio <= 1 / (1 + floor)      # a RESOLVED loss, not any negative point
               or a material acceptance loss on a drafting arm
               or a primary category below its own noise line
    resolution = the floor, printed with the verdict: at n=2-3 this is 7-26 %, i.e. the data
               cannot support a 2 % claim, and the output says so instead of looking decisive.

Defects the revisions remove, each of which produced a wrong verdict in practice:
  * a fixed per-category line at 1.00 / 0.98 over 3-5-prompt buckets (a true +2.5 % gain failed
    ~half the time);
  * a two-sided acceptance band, which made every genuine drafting improvement unpassable;
  * a range-based floor that grew with the number of replicates, so more evidence made the gate
    stricter;
  * 1.96*SEM, which at n=2 equals 0.98*range - the statistic it replaced, relabelled;
  * a spread computed as a weighted mean of category rates while the effect was a ratio of sums;
  * `basis()` inferred from `any()` while the estimator required `all()`, so mixed bases slipped
    through the check meant to refuse them;
  * `withdraw` on any negative point estimate, which let ordinary noise prove a regression.

Superseded: pass = ratio >= max(1.02, 1 + range/mean) AND every category >= 1.00 AND
|acc delta| <= 0.05 AND corpus >= 1.00 AND no category < 0.97; withdraw = ratio < 1.00 or a
category < 0.98.

Input is `results/spec_decode.csv` as the harness writes it (one row per category per arm), with two
or more runs per arm so the same-arm spread is measurable.

Basis, stated because it was once claimed and not true: `throughput()` takes a **ratio of sums**
(tokens/seconds) only when the CSV carries those columns; otherwise it falls back to the mean of
per-run rates, which is an average of ratios. `basis()` reports which one was used and the verdict
carries it. A docstring promising a property the code does not compute survived three reviews of this
file before an external one caught it - the basis belongs in the output, not in the prose.

    bench/spec_primary.py --csv results/spec_decode.csv --control t39-A --treat t39-B
    bench/spec_primary.py --csv ... --control ... --treat ... --drafting   # drafter/window arms
    bench/spec_primary.py --csv ... --control ... --treat ... --json out.json

Exit 0 = pass, 1 = withdraw, 2 = cannot judge (missing arm, fewer than two runs on an arm, a category
the rule names that the CSV does not carry).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

# Workload counts from eval/spec_decode.py's corpus. The rule weights by these, not by category count.
PRIMARY_WEIGHTS = {"Agentic; code": 5, "Agentic; curl": 4, "Coding": 3}
TRIVIAL = "Trivial repetition"
DRAFT_ARMS = ("DFlash", "MTP", "Draft")

# Gate constants. FLOOR_MIN is the historical 2 % default, used when an arm has a single run and no
# observable spread.
#
# The multi-run half-width uses the SMALL-SAMPLE t quantile, not the normal 1.96. At n=2 the two
# differ in a way that matters: sd/sqrt(2) makes 1.96*SEM exactly 0.98*(max-min), i.e. the range
# statistic this replaced, relabelled. With df = n-1 the interval is honest about how little two runs
# say (t = 12.71 at n=2, 4.30 at n=3), which is the point: a floor that shrinks with n must start
# somewhere credible.
FLOOR_MIN = 0.02
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df: int) -> float:
    """Two-sided 95 % t quantile for df degrees of freedom (table, interpolated at the top).

    Why a t quantile and not 1.96: with two values, sd = |v1 - v2| / sqrt(2), so 1.96 * sd / sqrt(2)
    = 0.98 * |v1 - v2| - i.e. the normal quantile reproduced the range statistic it was meant to
    replace (measured ratio 0.980 at n=2, 0.571 at n=3). A floor that is supposed to shrink with n
    has to start from an interval that admits how little two runs say: t = 12.71 at df=1, 4.30 at
    df=2. This mistake shipped once and was caught by an external review, not by the tests.
    """
    if df < 1:
        return float("inf")
    if df in T975:
        return T975[df]
    keys = sorted(T975)
    for k in keys:
        if df <= k:
            return T975[k]
    return 1.96


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _matches(tag: str, prefixes: str) -> bool:
    return any(tag.startswith(p) for p in str(prefixes).split(",") if p)


def pick(rows, tag_prefix: str, category: str, arm_kind: str = "draft") -> list[dict]:
    """The rows of one arm (every run whose tag starts with any of the comma-separated prefixes)."""
    out = []
    for r in rows:
        if not _matches(r.get("tag", ""), tag_prefix):
            continue
        if r.get("category") != category:
            continue
        arm = r.get("arm", "")
        if arm_kind == "draft" and not any(arm.startswith(k) for k in DRAFT_ARMS):
            continue
        if arm_kind == "baseline" and not arm.startswith("Baseline"):
            continue
        out.append(r)
    return out


def rate(rows, key: str) -> float | None:
    """Sum of `key` over the rows; None when there are no rows (never a silent 0.0)."""
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def throughput(rows) -> float | None:
    """tok/s for a set of rows: a ratio of sums where the rows allow it, else the mean of rates.

    A ratio of sums needs the SAME rows to carry both numbers - collecting token and second lists
    independently (as an earlier version did) lets two different row subsets be divided by each
    other. The pairing is therefore per row, and any row missing either field forces the whole
    estimate onto the mean-of-rates path, where the basis is reported.
    """
    pairs = [(float(r["tokens"]), float(r["seconds"])) for r in rows
             if r.get("tokens") not in (None, "") and r.get("seconds") not in (None, "")]
    if pairs and len(pairs) == len(rows) and sum(sec for _, sec in pairs) > 0:
        return sum(tok for tok, _ in pairs) / sum(sec for _, sec in pairs)
    return rate(rows, "tok_s")


def primary_from_rows(rows) -> float | None:
    """The 5:4:3 primary for ONE run: a ratio of sums across categories, or the weighted mean of
    their rates.

    Why this exists as its own function: the effect and its uncertainty must be the same quantity.
    An earlier version computed the primary as a cross-category ratio of sums but the per-run spread
    as a weighted mean of the categories' rates - calling throughput() inside each category fixes the
    estimator within a category, not across categories, and the two differ whenever category rates
    and durations differ. Kimi caught this after the first repair was called complete.
    """
    cats = {}
    complete = True
    for c in PRIMARY_WEIGHTS:
        rs = [r for r in rows if r.get("category") == c
              and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]
        if not rs:
            continue
        if any(r.get("tokens") in (None, "") or r.get("seconds") in (None, "") for r in rs):
            complete = False
        cats[c] = rs
    if not cats:
        return None
    if complete:
        num = sum(PRIMARY_WEIGHTS[c] * sum(float(r["tokens"]) for r in rs) for c, rs in cats.items())
        den = sum(PRIMARY_WEIGHTS[c] * sum(float(r["seconds"]) for r in rs) for c, rs in cats.items())
        if den > 0:
            return num / den
    wsum = sum(PRIMARY_WEIGHTS[c] for c in cats)
    rates = {c: rate(rs, "tok_s") for c, rs in cats.items()}
    if any(v is None for v in rates.values()):
        return None
    return sum(PRIMARY_WEIGHTS[c] * rates[c] for c in cats) / wsum


def basis(rows) -> str:
    """Which estimator the primary will use for these rows, by the same completeness rule.

    The guard must not infer completeness from `any()`: two arms can both be labelled
    "ratio of sums" while only one actually has complete columns, and the mixed comparison the check
    exists to refuse goes through. If any row lacks either field, the estimator is the mean of rates.
    """
    if not rows:
        return "no rows"
    complete = all(r.get("tokens") not in (None, "") and r.get("seconds") not in (None, "")
                   for r in rows)
    return "ratio of sums (tokens/seconds)" if complete else \
        "mean of per-run rates (tokens/seconds missing on at least one row)"


def pass_bound(f: float) -> float:
    """Smallest ratio that counts as a resolved gain at fractional uncertainty `f`."""
    return 1.0 + f


def loss_bound(f: float) -> float:
    """Largest ratio that counts as a resolved loss.

    The reciprocal of the pass bound, not 1 - f: a ratio scale is not symmetric, so a +20 % pass
    bound corresponds to 1/1.2 = 0.833 as a loss, not 0.80. Using 1 - f made a loss easier to prove
    than a gain, which is the same asymmetry in the other direction.
    """
    return 1.0 / (1.0 + f)


def classify_ratio(p_ratio: float | None, f: float) -> str:
    """'pass' when the ratio clears the gain bound, 'loss' when it clears the loss bound."""
    if p_ratio is None:
        return "unknown"
    if p_ratio >= pass_bound(f):
        return "pass"
    if p_ratio <= loss_bound(f):
        return "loss"
    return "inside"


def ratio(control: float, treat: float) -> float | None:
    return (treat / control) if control else None


def evaluate(rows, control: str, treat: str, drafting: bool = False,
             acc_tolerance: float = 0.05) -> tuple[int, dict]:
    problems: list[str] = []
    cats = {}
    for cat in PRIMARY_WEIGHTS:
        c = pick(rows, control, cat)
        t = pick(rows, treat, cat)
        if not c or not t:
            problems.append(f"category '{cat}' is missing from {'control' if not c else 'treat'}")
            continue
        if len({r["run_id"] for r in c}) < 2 or len({r["run_id"] for r in t}) < 2:
            problems.append(f"category '{cat}' has fewer than two runs on an arm, so no spread")
        rc, rt = throughput(c), throughput(t)
        cats[cat] = {"control": rc, "treat": rt, "ratio": ratio(rc, rt),
                     "acc_control": rate(c, "acc"), "acc_treat": rate(t, "acc")}
    # every non-trivial category, as the guardrail needs them
    others = {}
    for cat in sorted({r["category"] for r in rows if r.get("category")}):
        if cat in PRIMARY_WEIGHTS or cat == TRIVIAL:
            continue
        c, t = pick(rows, control, cat), pick(rows, treat, cat)
        rc, rt = throughput(c), throughput(t)
        if rc and rt:
            others[cat] = ratio(rc, rt)
        else:
            problems.append(f"secondary category '{cat}' has no data on one arm")
    # corpus: every draft row of each arm, primary and secondary alike
    def allrows(prefix):
        return [r for r in rows if _matches(r.get("tag", ""), prefix)
                and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]
    corpus_c, corpus_t = throughput(allrows(control)), throughput(allrows(treat))

    if problems:
        return 2, {"problems": problems, "categories": cats, "secondary": others}

    wsum = sum(PRIMARY_WEIGHTS.values())
    # The primary is a workload-weighted THROUGHPUT, which is sum(tokens)/sum(seconds) over the
    # weighted prompt counts - not the weighted mean of three per-category rates. Those differ
    # whenever a category's prompts run to different lengths, and the campaign reported the second
    # under the name of the first for weeks. Where the columns are missing (every pass before they
    # were emitted) the mean of rates is used and `basis` says so, so the next reader is not misled
    # by the number's provenance.
    def primary_of(prefix, arm_kind):
        num = den = 0.0
        have = True
        for c in cats:
            rs = [r for r in rows if _matches(r.get("tag", ""), prefix) and r.get("category") == c
                  and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]
            if not rs or any(r.get("tokens") in (None, "") for r in rs) or \
                    any(r.get("seconds") in (None, "") for r in rs):
                have = False
                break
            num += PRIMARY_WEIGHTS[c] * sum(float(r["tokens"]) for r in rs)
            den += PRIMARY_WEIGHTS[c] * sum(float(r["seconds"]) for r in rs)
        if have and den > 0:
            return num / den
        return sum(PRIMARY_WEIGHTS[c] * cats[c][arm_kind] for c in cats) / wsum

    p_c = primary_of(control, "control")
    p_t = primary_of(treat, "treat")
    p_ratio = ratio(p_c, p_t)
    # the same-arm spread: the widest per-run deviation from its arm's mean, over the primary set
    def primary_rows(prefix):
        return [r for r in rows if _matches(r.get("tag", ""), prefix)
                and r.get("category") in PRIMARY_WEIGHTS
                and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]

    def arm_spread(prefix):
        # Spread of the same-arm primary mean as an INTERVAL, not a range: (max - min) grows with the
        # number of replicates, so adding passes RAISED the pass floor (1.039 -> 1.045 -> 1.050 across
        # T66-T70) - more evidence made the gate stricter, which is backwards. The interval is
        # t(0.975, n-1) * SEM / mean: SEM shrinks as 1/sqrt(N) and the t quantile handles the tiny-n
        # case honestly. Note 1.96 was wrong here for the reason spelled out at t975() - at n=2 it is
        # exactly 0.98 * the range it replaced.
        pr = primary_rows(prefix)
        runs = sorted({r["run_id"] for r in pr})
        vals = []
        for rid in runs:
            run_rows = [r for r in pr if r["run_id"] == rid]
            # the SAME function the effect uses. Calling throughput() inside each category fixes the
            # estimator within a category, not across categories: a weighted mean of category rates
            # is not a cross-category ratio of sums, and the effect and its resolution ended up being
            # different quantities (caught by a second external review after the first repair was
            # reported as complete).
            v = primary_from_rows(run_rows)
            if v:
                vals.append(v)
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        if not m:
            return None
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        sem = (var / len(vals)) ** 0.5
        return t975(len(vals) - 1) * sem / m
    spread_a, spread_t = arm_spread(control), arm_spread(treat)
    # The uncertainty of the RATIO is not the larger of the two arms' intervals: for independent
    # arms the variances add, so the half-width is sqrt(a^2 + t^2) (T77). max() understates a case
    # where both arms are noisy and overstates one where only one is.
    combined = ((spread_a or 0.0) ** 2 + (spread_t or 0.0) ** 2) ** 0.5
    # a single-run arm has no observable spread; the 2 % default stands in for it and is reported
    floor = max(FLOOR_MIN, combined)
    def acc_of(prefix, arm_kind):
        # Accepted drafts per verification round, aggregated as sum(acc*rounds)/sum(rounds). The
        # earlier form averaged the per-category means weighted by prompt count, which counts a
        # 5-prompt category five times but ignores that its prompts may have run far more rounds -
        # so the acceptance a long prompt contributed was thrown away.
        num = den = 0.0
        for c in cats:
            rs = [r for r in rows if _matches(r.get("tag", ""), prefix) and r.get("category") == c
                  and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]
            if rs and all(r.get("acc") not in (None, "") for r in rs) and \
                    all(r.get("rounds") not in (None, "", "0") for r in rs):
                for r in rs:
                    num += PRIMARY_WEIGHTS[c] * float(r["acc"]) * float(r["rounds"])
                    den += PRIMARY_WEIGHTS[c] * float(r["rounds"])
        if den > 0:
            return num / den
        return sum(PRIMARY_WEIGHTS[c] * cats[c][arm_kind] for c in cats) / wsum

    acc_c = acc_of(control, "acc_control")
    acc_t = acc_of(treat, "acc_treat")
    corpus_ratio = ratio(corpus_c, corpus_t)
    worst_other = min(others.values()) if others else None

    # (d) an arm pair measured on different estimators is not a comparison: refuse it rather than
    # silently falling back for one side only (T77)
    basis_c, basis_t = basis(primary_rows(control)), basis(primary_rows(treat))
    if basis_c != basis_t:
        return 2, {"problems": [f"arms measured on different bases: control={basis_c}, treat={basis_t}"]}

    missing = [c for c in cats if cats[c]["ratio"] is None or cats[c]["acc_control"] is None
               or cats[c]["acc_treat"] is None]
    if missing:
        return 2, {"problems": [f"no data for {c}" for c in missing],
                   "categories": cats, "secondary": others}

    # Per-category guards, tied to each category's own noise instead of two fixed lines.
    #
    # Why: the three primary categories are built from 3-5 prompts each and jitter 3-7 % run to run.
    # Requiring every one of them >= 1.00 rejected a TRUE +2.5 % uniform gain in roughly half of draws
    # (Argon's arithmetic: with sigma = 2.5 % per category, P(one lands < 1.00) = 21 %, so
    # 1 - 0.79^3 = 51 % across the three). Three independent gates over noisy buckets is a
    # multiple-testing trap, and it fired: T66-T70 all landed "inconclusive".
    #
    # The trade accepted here: a category now vetoes only past its own resolution, so with 2-3 runs a
    # real 4 % regression in one category also passes. That is the price of not rejecting real wins;
    # it is why the primary and the corpus ratio still have to clear their own floors.
    def category_spread(cat, prefix):
        rs = [r for r in rows if _matches(r.get("tag", ""), prefix) and r.get("category") == cat
              and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]
        runs = sorted({r["run_id"] for r in rs})
        vals = [v for v in (throughput([r for r in rs if r["run_id"] == rid]) for rid in runs) if v]
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        if not m:
            return None
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return t975(len(vals) - 1) * (var / len(vals)) ** 0.5 / m

    cat_lines = {}
    cat_spreads = {}
    for c in cats:
        # same combination rule as the primary: a ratio's uncertainty is sqrt(a^2 + t^2). This line
        # said max() for two revisions after the primary was fixed - the repair was reported complete
        # while the category path still took the larger arm alone (caught by a third external pass).
        ca, ct = category_spread(c, control) or 0.0, category_spread(c, treat) or 0.0
        combined = (ca ** 2 + ct ** 2) ** 0.5
        sp = max(FLOOR_MIN, combined)
        cat_spreads[c] = {"control": ca, "treat": ct, "combined": combined}
        cat_lines[c] = loss_bound(sp)
    # How small an effect this arm pair could see at all, printed with the verdict. At two or three
    # runs the t interval is 7-26 % wide, so a 2 % claim is simply not supported by the data; saying
    # that in the output is more useful than a verdict that reads as decisive. This number is the
    # honest answer to "why does every arm come back inconclusive".
    resolution = floor
    # a loss only vetoes past that category's own resolution; the old fixed 0.98 could not tell a
    # real 2 % loss from a 5 %-jitter category
    # A material acceptance LOSS withdraws; it does not merely block a pass.
    #
    # Why: with the one-sided band, a drafting arm that lost 0.42 accepted tokens could still report
    # "inconclusive" because the verdict branch only looked at the throughput ratio. That is how a
    # genuine regression hides - it is not that the arm failed to win, it is that it lost, and the
    # word "inconclusive" tells the operator to try again instead of to stop.
    acc_loss = (acc_t - acc_c) < -acc_tolerance if drafting else False
    # A NEGATIVE point estimate is not by itself a regression: at a 10-25 % resolution, half of all
    # true nulls land below 1.0. Withdrawing on p_ratio < 1.0 made the gate asymmetric in the worst
    # way - a coin flip could "prove" a regression, while a real win needed the full floor (T77).
    # A loss now withdraws only when it clears the same interval, expressed on the reciprocal scale:
    # p_ratio <= loss_bound(floor) = 1 / (1 + floor). A ratio is not symmetric, so a +20 % pass bound
    # corresponds to 0.833 as a loss, not 0.80; `1 - floor` made a loss easier to prove than a gain.
    real_loss = classify_ratio(p_ratio, floor) == "loss"
    withdraw = real_loss or acc_loss or any(cats[c]["ratio"] < cat_lines[c] for c in cats)

    # Acceptance: two-sided for kernel/placement arms, one-sided for arms whose point IS acceptance.
    #
    # Why: |delta acc| <= 0.05 was written for kernel arms, where acceptance is a confound that
    # should not move. Applied to drafter and window arms it made the better mechanism unpassable by
    # construction - a drafter that accepts MORE drafts fails the band. T61/T63/T70 were therefore
    # only ever judged on their losses, which is not a test, it is a filter. A drafting arm must not
    # LOSE acceptance; gaining it is the point.
    if drafting:
        acc_ok = (acc_t - acc_c) >= -acc_tolerance
        acc_rule = f"acc_treat >= acc_control - {acc_tolerance} (drafting arm)"
    else:
        acc_ok = abs(acc_t - acc_c) <= acc_tolerance
        acc_rule = f"|acc_treat - acc_control| <= {acc_tolerance} (kernel arm)"
    passed = bool(
        p_ratio is not None and p_ratio >= 1 + floor
        and all(cats[c]["ratio"] >= cat_lines[c] for c in cats)
        and acc_ok
        and corpus_ratio is not None and corpus_ratio >= 1.0
        and (worst_other is None or worst_other >= 0.97)
    )
    verdict = "withdraw" if withdraw else ("pass" if passed else "inconclusive")
    out = {
        "control": control, "treat": treat,
        "primary": {"control": p_c, "treat": p_t, "ratio": p_ratio,
                    "floor": 1 + floor, "spread_control": spread_a, "spread_treat": spread_t,
                    "spread_method": f"1.96->t(95%, df=n-1) x SEM / mean"},
        "categories": cats, "secondary": others, "corpus_ratio": corpus_ratio,
        "basis": basis(allrows(treat)) if "allrows" in dir() else "mean of per-run rates",
        "category_guard": cat_lines,
        "category_spread": cat_spreads,
        "resolution": {"min_effect_pct": resolution * 100.0,
                       "note": "the smallest primary change this many runs can separate at 95 %"},
        "acc": {"control": acc_c, "treat": acc_t, "delta": (acc_t - acc_c) if acc_c and acc_t else None,
                "rule": acc_rule, "ok": acc_ok},
        "verdict": verdict,
    }
    return (1 if withdraw else (0 if passed else 3)), out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/spec_decode.csv")
    ap.add_argument("--control", required=True, help="tag prefix of the control arm, e.g. t39-A")
    ap.add_argument("--treat", required=True, help="tag prefix of the treatment arm")
    ap.add_argument("--json", help="write the full evaluation here")
    ap.add_argument("--drafting", action="store_true",
                    help="the arm changes draft acceptance on purpose (drafter or window), so the "
                         "acceptance guard is one-sided: it must not LOSE acceptance")
    ap.add_argument("--acc-tolerance", type=float, default=0.05,
                    help="tokens/round the arm may lose before it fails (default 0.05)")
    args = ap.parse_args(argv)

    rows = load(args.csv)
    # a campaign can be split across tags (a repeat runs under a new one), so a side is a list of
    # prefixes and the evaluation matches any of them
    code, out = evaluate(rows, args.control, args.treat, drafting=args.drafting,
                         acc_tolerance=args.acc_tolerance)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    if code == 2:
        print("cannot judge:")
        for p in out["problems"]:
            print(f"  - {p}")
        return 2
    p = out["primary"]
    print(f"primary  control {p['control']:.2f}  treat {p['treat']:.2f}  ratio {p['ratio']:.4f} "
          f"({100 * (p['ratio'] - 1):+.2f} %)  floor {p['floor']:.3f} "
          f"(spread {p['spread_control']:.2%} / {p['spread_treat']:.2%})")
    for c, v in out["categories"].items():
        print(f"  {c:<15} {v['control']:7.2f} -> {v['treat']:7.2f}  {100 * (v['ratio'] - 1):+6.2f} %"
              f"   acc {v['acc_control']:.2f} -> {v['acc_treat']:.2f}")
    print(f"corpus   ratio {out['corpus_ratio']:.4f}   acc delta {out['acc']['delta']:+.3f}")
    if out["secondary"]:
        print("secondary worst: " + ", ".join(f"{k} {100 * (v - 1):+.1f} %" for k, v in
                                              sorted(out["secondary"].items(), key=lambda kv: kv[1])[:3]))
    print(f"verdict  {out['verdict'].upper()}")
    return code


if __name__ == "__main__":
    sys.exit(main())
