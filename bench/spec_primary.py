#!/usr/bin/env python3
"""The agentic-weighted primary metric for a spec_decode arm pair, with its verdict.

    primary = workload-weighted tok_s over {Agentic; code (5), Agentic; curl (4), Coding (3)}

Gate (revised 2026-11-11, T72 §5 — the earlier form is described below and was defective):

    spread   = 1.96 x SEM / mean of the per-run primary means, each run weighted 5:4:3
               (the old rule used (max-min)/mean, which GREW with the number of replicates and so
               raised the pass floor every time a repeat was added: 1.039 -> 1.045 -> 1.050 across
               T66-T70. The mean's standard error shrinks as 1/sqrt(N), which is the quantity a
               pass floor should track.)
    floor    = max(0.02, spread)
    pass     = primary ratio >= 1 + floor
               AND every primary category >= 1 - max(0.02, that category's own spread)
               AND acceptance: |delta| <= 0.05 for a kernel arm, or delta >= -0.05 for a drafting
                   arm (`--drafting`) whose point IS acceptance
               AND full-corpus ratio >= 1.00 with no non-trivial category < 0.97
    withdraw = primary ratio < 1.00, or any primary category past its own noise line

Two defects the revision removes, both of which produced wrong verdicts in practice: a fixed
per-category line at 1.00 (with 3-5 prompts per category and 3-7 % jitter, a true +2.5 % uniform gain
failed about half the time) and a two-sided acceptance band that made every genuine drafting
improvement unpassable.

Superseded rule, kept for reading the earlier reports: pass = ratio >= max(1.02, 1 + range/mean) AND
every category >= 1.00 AND |acc delta| <= 0.05 AND corpus >= 1.00 AND no category < 0.97;
withdraw = ratio < 1.00 or any category < 0.98.

Input is `results/spec_decode.csv` as the harness writes it (one row per category per arm), with two
or more runs per arm so the same-arm spread is measurable. Every ratio is a ratio of sums, never an
average of per-run averages.

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
# observable spread; SEM_Z is the normal quantile for the 95 % half-width of the same-arm mean.
FLOOR_MIN = 0.02
SEM_Z = 1.96


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
    """tok/s for a set of rows: a ratio of sums where the CSV makes it possible.

    `sum(tokens) / sum(seconds)` is the quantity a throughput claim is about. Where those columns are
    absent (every pass before the harness emitted them) this falls back to the mean of the per-run
    rates, which is an average of ratios and weights a short prompt the same as a long one; the
    fallback is reported by `basis()` so a report can say which one it used.
    """
    tk = [float(r["tokens"]) for r in rows if r.get("tokens") not in (None, "")]
    sc = [float(r["seconds"]) for r in rows if r.get("seconds") not in (None, "")]
    if tk and sc and sum(sc) > 0:
        return sum(tk) / sum(sc)
    return rate(rows, "tok_s")


def basis(rows) -> str:
    """Which throughput estimator `throughput` used, for the report to quote."""
    if any(r.get("tokens") not in (None, "") for r in rows) and \
       any(r.get("seconds") not in (None, "") for r in rows):
        return "ratio of sums (tokens/seconds)"
    return "mean of per-run rates (no tokens/seconds columns in this CSV)"


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
    p_c = sum(PRIMARY_WEIGHTS[c] * cats[c]["control"] for c in cats) / wsum
    p_t = sum(PRIMARY_WEIGHTS[c] * cats[c]["treat"] for c in cats) / wsum
    p_ratio = ratio(p_c, p_t)
    # the same-arm spread: the widest per-run deviation from its arm's mean, over the primary set
    def primary_rows(prefix):
        return [r for r in rows if _matches(r.get("tag", ""), prefix)
                and r.get("category") in PRIMARY_WEIGHTS
                and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]

    def arm_spread(prefix):
        # Spread of the same-arm primary mean, as a standard error of the mean and not a range:
        # (max - min) grows with the number of replicates, so adding passes RAISED the pass floor
        # (1.039 -> 1.045 -> 1.050 across T66-T70) instead of shrinking the uncertainty. The mean's
        # standard error shrinks as 1/sqrt(N), which is what the gate needs.
        pr = primary_rows(prefix)
        runs = sorted({r["run_id"] for r in pr})
        vals = []
        for rid in runs:
            run_rows = [r for r in pr if r["run_id"] == rid]
            # the same 5:4:3 weighting the primary uses: an unweighted mean here measured a
            # different quantity than the one being gated
            got = {}
            for c in PRIMARY_WEIGHTS:
                v = throughput([r for r in run_rows if r["category"] == c])
                if v:
                    got[c] = v
            if got:
                ws = sum(PRIMARY_WEIGHTS[c] for c in got)
                vals.append(sum(PRIMARY_WEIGHTS[c] * got[c] for c in got) / ws)
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        if not m:
            return None
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        sem = (var / len(vals)) ** 0.5
        return SEM_Z * sem / m
    spread_a, spread_t = arm_spread(control), arm_spread(treat)
    # a single-run arm has no observable spread; the 2 % default stands in for it and is reported
    floor = max(FLOOR_MIN, spread_a or 0.0, spread_t or 0.0)
    acc_c = sum(PRIMARY_WEIGHTS[c] * cats[c]["acc_control"] for c in cats) / wsum
    acc_t = sum(PRIMARY_WEIGHTS[c] * cats[c]["acc_treat"] for c in cats) / wsum
    corpus_ratio = ratio(corpus_c, corpus_t)
    worst_other = min(others.values()) if others else None

    missing = [c for c in cats if cats[c]["ratio"] is None or cats[c]["acc_control"] is None
               or cats[c]["acc_treat"] is None]
    if missing:
        return 2, {"problems": [f"no data for {c}" for c in missing],
                   "categories": cats, "secondary": others}

    # Per-category guards, tied to each category's own noise instead of two fixed lines. A primary
    # category has 3-5 prompts, so its run-to-run jitter is 3-7 %; requiring every category >= 1.00
    # rejected a true +2.5 % uniform gain in roughly half of draws (T66-T70 floors of 1.039-1.050
    # came from that same noise). A category now vetoes only when it loses more than it can resolve.
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
        return SEM_Z * (var / len(vals)) ** 0.5 / m

    cat_lines = {}
    for c in cats:
        sp = max(FLOOR_MIN, category_spread(c, control) or 0.0, category_spread(c, treat) or 0.0)
        cat_lines[c] = 1.0 - sp
    # a loss only vetoes past that category's own resolution; the old fixed 0.98 could not tell a
    # real 2 % loss from a 5 %-jitter category
    withdraw = bool(p_ratio is not None and p_ratio < 1.0) or any(
        cats[c]["ratio"] < cat_lines[c] for c in cats)

    # Acceptance: two-sided for kernel/placement arms, one-sided for arms whose point IS acceptance
    # (a drafter or window change must not LOSE acceptance; requiring |delta| <= 0.05 barred every
    # genuine drafting improvement from passing, which is why T61/T63/T70 could only be judged on
    # their losses).
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
                    "spread_method": f"SEM x {SEM_Z} / mean"},
        "categories": cats, "secondary": others, "corpus_ratio": corpus_ratio,
        "basis": basis(allrows(treat)) if "allrows" in dir() else "mean of per-run rates",
        "category_guard": cat_lines,
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
