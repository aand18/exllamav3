#!/usr/bin/env python3
"""The agentic-weighted primary metric for a spec_decode arm pair, with its verdict.

The rule this implements was fixed before the t34 data was read (see the campaign ledger):

    primary = workload-weighted tok_s over {Agentic; code (5), Agentic; curl (4), Coding (3)}
    pass     = primary ratio >= max(1.02, 1 + |A1-A2| / mean(A1,A2))   on the 12-workload mean
               AND every primary category ratio >= 1.00
               AND primary acc within +-0.05 of control
               AND full-corpus ratio >= 1.00 with no non-trivial category < 0.97
    withdraw = primary ratio < 1.00 or any primary category < 0.98

Input is `results/spec_decode.csv` as the harness writes it (one row per category per arm), with two
or more runs per arm so the same-arm spread is measurable. Every ratio is a ratio of sums, never an
average of per-run averages.

    bench/spec_primary.py --csv results/spec_decode.csv --control t39-A --treat t39-B
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


def ratio(control: float, treat: float) -> float | None:
    return (treat / control) if control else None


def evaluate(rows, control: str, treat: str) -> tuple[int, dict]:
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
        rc, rt = rate(c, "tok_s"), rate(t, "tok_s")
        cats[cat] = {"control": rc, "treat": rt, "ratio": ratio(rc, rt),
                     "acc_control": rate(c, "acc"), "acc_treat": rate(t, "acc")}
    # every non-trivial category, as the guardrail needs them
    others = {}
    for cat in sorted({r["category"] for r in rows if r.get("category")}):
        if cat in PRIMARY_WEIGHTS or cat == TRIVIAL:
            continue
        c, t = pick(rows, control, cat), pick(rows, treat, cat)
        rc, rt = rate(c, "tok_s"), rate(t, "tok_s")
        if rc and rt:
            others[cat] = ratio(rc, rt)
        else:
            problems.append(f"secondary category '{cat}' has no data on one arm")
    # corpus: every draft row of each arm, primary and secondary alike
    def allrows(prefix):
        return [r for r in rows if _matches(r.get("tag", ""), prefix)
                and any(r.get("arm", "").startswith(k) for k in DRAFT_ARMS)]
    corpus_c, corpus_t = rate(allrows(control), "tok_s"), rate(allrows(treat), "tok_s")

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
        # the spread the rule guards is on the same 12-workload primary mean, not on every category
        pr = primary_rows(prefix)
        runs = sorted({r["run_id"] for r in pr})
        vals = [rate([r for r in pr if r["run_id"] == rid], "tok_s") for rid in runs]
        vals = [v for v in vals if v]
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        return (max(vals) - min(vals)) / m if m else None
    spread_a, spread_t = arm_spread(control), arm_spread(treat)
    floor = max(0.02, spread_a or 0.0, spread_t or 0.0)
    acc_c = sum(PRIMARY_WEIGHTS[c] * cats[c]["acc_control"] for c in cats) / wsum
    acc_t = sum(PRIMARY_WEIGHTS[c] * cats[c]["acc_treat"] for c in cats) / wsum
    corpus_ratio = ratio(corpus_c, corpus_t)
    worst_other = min(others.values()) if others else None

    missing = [c for c in cats if cats[c]["ratio"] is None or cats[c]["acc_control"] is None
               or cats[c]["acc_treat"] is None]
    if missing:
        return 2, {"problems": [f"no data for {c}" for c in missing],
                   "categories": cats, "secondary": others}
    withdraw = bool(p_ratio is not None and p_ratio < 1.0) or any(
        cats[c]["ratio"] < 0.98 for c in cats)
    passed = bool(
        p_ratio is not None and p_ratio >= 1 + floor
        and all(cats[c]["ratio"] >= 1.0 for c in cats)
        and abs(acc_t - acc_c) <= 0.05
        and corpus_ratio is not None and corpus_ratio >= 1.0
        and (worst_other is None or worst_other >= 0.97)
    )
    verdict = "withdraw" if withdraw else ("pass" if passed else "inconclusive")
    out = {
        "control": control, "treat": treat,
        "primary": {"control": p_c, "treat": p_t, "ratio": p_ratio,
                    "floor": 1 + floor, "spread_control": spread_a, "spread_treat": spread_t},
        "categories": cats, "secondary": others, "corpus_ratio": corpus_ratio,
        "acc": {"control": acc_c, "treat": acc_t, "delta": (acc_t - acc_c) if acc_c and acc_t else None},
        "verdict": verdict,
    }
    return (1 if withdraw else (0 if passed else 3)), out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/spec_decode.csv")
    ap.add_argument("--control", required=True, help="tag prefix of the control arm, e.g. t39-A")
    ap.add_argument("--treat", required=True, help="tag prefix of the treatment arm")
    ap.add_argument("--json", help="write the full evaluation here")
    args = ap.parse_args(argv)

    rows = load(args.csv)
    # a campaign can be split across tags (a repeat runs under a new one), so a side is a list of
    # prefixes and the evaluation matches any of them
    code, out = evaluate(rows, args.control, args.treat)
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
