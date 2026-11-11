#!/usr/bin/env python3
"""The primary-metric rule, pinned on synthetic arms (no harness, no model)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spec_primary  # noqa: E402
from spec_primary import PRIMARY_WEIGHTS, evaluate  # noqa: E402


def _rows(arms):
    """Two runs per arm, one row per primary category, built from {category: (c1, c2, t1, t2)}."""
    rows = []
    for cat, (c1, c2, t1, t2) in arms.items():
        rows.append({"tag": "A1", "run_id": "A1", "category": cat, "arm": "DFlash (greedy)",
                     "tok_s": str(c1), "acc": "2.0"})
        rows.append({"tag": "A2", "run_id": "A2", "category": cat, "arm": "DFlash (greedy)",
                     "tok_s": str(c2), "acc": "2.0"})
        rows.append({"tag": "B1", "run_id": "B1", "category": cat, "arm": "DFlash (greedy)",
                     "tok_s": str(t1), "acc": "2.0"})
        rows.append({"tag": "B2", "run_id": "B2", "category": cat, "arm": "DFlash (greedy)",
                     "tok_s": str(t2), "acc": "2.0"})
    return rows


def test_a_clean_gain_passes():
    rows = _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
    code, out = evaluate(rows, "A", "B")
    assert code == 0 and out["verdict"] == "pass"
    assert abs(out["primary"]["ratio"] - 1.10) < 1e-9


def test_one_primary_category_below_the_floor_withdraws():
    d = {c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS}
    d["Agentic; curl"] = (100, 100, 95, 95)          # -5 %, below the 0.98 withdrawal line
    code, out = evaluate(_rows(d), "A", "B")
    assert code == 1 and out["verdict"] == "withdraw"


def test_a_flat_arm_is_inconclusive_not_a_pass():
    rows = _rows({c: (100, 100, 100, 100) for c in PRIMARY_WEIGHTS})
    code, out = evaluate(rows, "A", "B")
    assert out["verdict"] == "inconclusive" and code == 3


def test_a_single_run_per_arm_cannot_be_judged():
    rows = [r for r in _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
            if r["run_id"] in ("A1", "B1")]
    code, out = evaluate(rows, "A", "B")
    assert code == 2 and any("fewer than two runs" in p for p in out["problems"])


def _rows_acc(arms, acc_c="2.0", acc_t="2.0"):
    """Same shape as _rows but with per-arm acceptance, for the drafting-arm guard."""
    rows = []
    for cat, (c1, c2, t1, t2) in arms.items():
        for tag, rid, v, a in (("A1","A1",c1,acc_c), ("A2","A2",c2,acc_c),
                               ("B1","B1",t1,acc_t), ("B2","B2",t2,acc_t)):
            rows.append({"tag": tag, "run_id": rid, "category": cat,
                         "arm": "DFlash (greedy)", "tok_s": str(v), "acc": str(a)})
    return rows


def _arm(cat, rid, tag, tok_s, acc="2.0", arm="DFlash (greedy)"):
    return {"tag": tag, "run_id": rid, "category": cat, "arm": arm, "tok_s": str(tok_s), "acc": str(acc)}


def test_a_baseline_row_is_never_part_of_the_arm():
    rows = _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
    # a baseline row that would drag the arm down if the filter let it in
    rows.append(_arm("Agentic; code", "A1", "A1", 10, arm="Baseline"))
    code, out = evaluate(rows, "A", "B")
    assert out["categories"]["Agentic; code"]["control"] > 99


def test_an_accuracy_regression_blocks_a_pass():
    rows = _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
    for r in rows:
        if r["tag"].startswith("B"):
            r["acc"] = "1.80"          # -0.20, well outside the +-0.05 band
    code, out = evaluate(rows, "A", "B")
    assert out["verdict"] == "inconclusive"


def test_a_corpus_regression_blocks_a_pass():
    """A gain on the 12 primary workloads cannot carry a corpus that falls below parity: the
    corpus ratio is over every draft row, and it must be reported and must block the pass."""
    rows = _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
    rows += [_arm("Creative", "A1", "A1", 100), _arm("Creative", "A2", "A2", 100),
             _arm("Creative", "B1", "B1", 60), _arm("Creative", "B2", "B2", 60)]
    code, out = evaluate(rows, "A", "B")
    assert out["corpus_ratio"] < 1.0
    assert out["verdict"] != "pass"


def test_a_secondary_category_below_the_guardrail_blocks_a_pass():
    rows = _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
    rows += [_arm("Creative", "A1", "A1", 100), _arm("Creative", "A2", "A2", 100),
             _arm("Creative", "B1", "B1", 96), _arm("Creative", "B2", "B2", 96)]   # -4 %
    code, out = evaluate(rows, "A", "B")
    assert out["secondary"]["Creative"] < 0.97 and out["verdict"] == "inconclusive"


def test_missing_primary_data_cannot_be_judged():
    rows = _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
    rows = [r for r in rows if not (r["category"] == "Coding" and r["tag"].startswith("B"))]
    code, out = evaluate(rows, "A", "B")
    assert code == 2


TESTS = [
    test_missing_primary_data_cannot_be_judged,
    test_a_secondary_category_below_the_guardrail_blocks_a_pass,
    test_a_corpus_regression_blocks_a_pass,
    test_an_accuracy_regression_blocks_a_pass,
    test_a_baseline_row_is_never_part_of_the_arm,
    test_a_clean_gain_passes,
    test_one_primary_category_below_the_floor_withdraws,
    test_a_flat_arm_is_inconclusive_not_a_pass,
    test_a_single_run_per_arm_cannot_be_judged,
]

if __name__ == "__main__":
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except BaseException as e:  # noqa: BLE001 - a test runner that names the failure
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    sys.exit(1 if failures else 0)


# --- the gate's own defects (T72 §5), each pinned by a test -------------------------------------

def test_adding_a_repeat_does_not_raise_the_floor():
    """The old range-based spread grew with N, so a third replicate RAISED the pass floor."""
    two = _rows({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS})
    _, out2 = evaluate(two, "A", "B")
    three = two + [_arm(c, "A3", "A3", 100.0) for c in PRIMARY_WEIGHTS] \
               + [_arm(c, "B3", "B3", 110.0) for c in PRIMARY_WEIGHTS]
    _, out3 = evaluate(three, "A", "B")
    assert out3["primary"]["floor"] <= out2["primary"]["floor"] + 1e-9
    assert out3["verdict"] == "pass"


def test_a_drafting_arm_may_improve_acceptance():
    """|delta acc| <= 0.05 barred every genuine drafting improvement from passing."""
    rows = _rows_acc({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS}, acc_c="1.80", acc_t="2.30")
    _, kernel = evaluate(rows, "A", "B", drafting=False)
    assert kernel["verdict"] != "pass", "a kernel arm still may not move acceptance"
    _, draft = evaluate(rows, "A", "B", drafting=True)
    assert draft["acc"]["ok"] is True and draft["verdict"] == "pass"


def test_a_drafting_arm_may_not_lose_acceptance():
    rows = _rows_acc({c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS}, acc_c="2.30", acc_t="1.80")
    _, out = evaluate(rows, "A", "B", drafting=True)
    assert out["acc"]["ok"] is False and out["verdict"] != "pass"


def test_a_category_vetoes_only_past_its_own_noise():
    """A 3-prompt category jitters 3-7 %; the fixed 0.98 line vetoed real wins."""
    d = {c: (100, 100, 110, 110) for c in PRIMARY_WEIGHTS}
    d["Agentic; curl"] = (100, 100, 99, 99)          # -1 %: inside any noise floor
    _, out = evaluate(_rows(d), "A", "B")
    assert out["verdict"] == "pass", out
    d["Agentic; curl"] = (100, 100, 90, 90)          # -10 %: past it
    _, out = evaluate(_rows(d), "A", "B")
    assert out["verdict"] == "withdraw"


# --- the defects a second external review found in the first repair (T78) -----------------------

def _acc_rows(tok, acc, tokens=None, seconds=None, tag="A", rid="A1"):
    """One run per category with optional tokens/seconds columns."""
    rows = []
    for i, c in enumerate(PRIMARY_WEIGHTS):
        r = {"tag": tag, "run_id": rid, "category": c,
             "arm": "DFlash (greedy)", "tok_s": str(tok[i] if isinstance(tok, list) else tok),
             "acc": str(acc)}
        if tokens is not None:
            r["tokens"] = str(tokens[i])
            r["seconds"] = str(seconds[i])
        rows.append(r)
    return rows


def test_spread_uses_the_same_estimator_as_the_effect():
    """A weighted mean of category rates is NOT a cross-category ratio of sums: with categories of
    different durations the two differ, and the spread must measure the gated quantity."""
    import spec_primary as sp
    # two categories at different rates and durations: ratio-of-sums differs from mean-of-rates
    runs = []
    for rid in ("A1", "A2"):
        runs.append({"tag": rid, "run_id": rid, "category": "Agentic; code", "arm": "DFlash (greedy)",
                     "tok_s": "100", "acc": "2.0", "tokens": "1000", "seconds": "10"})
        runs.append({"tag": rid, "run_id": rid, "category": "Coding", "arm": "DFlash (greedy)",
                     "tok_s": "10", "acc": "2.0", "tokens": "10", "seconds": "1"})
        runs.append({"tag": rid, "run_id": rid, "category": "Agentic; curl", "arm": "DFlash (greedy)",
                     "tok_s": "50", "acc": "2.0", "tokens": "50", "seconds": "1"})
    primary = sp.primary_from_rows(runs)
    # ratio of sums: tokens weighted 5/4/3 -> (5*1000 + 4*50 + 3*10) / (5*10 + 4*1 + 3*1)
    assert abs(primary - (5000 + 200 + 30) / (50 + 4 + 3)) < 1e-9


def test_basis_requires_complete_columns_not_any():
    import spec_primary as sp
    rows = _acc_rows([100, 100, 100], "2.0", tokens=[1000, 1000, 1000], seconds=[10, 10, 10])
    assert "ratio of sums" in sp.basis(rows)
    rows[1].pop("seconds")                      # one row incomplete
    assert "mean of per-run rates" in sp.basis(rows)


def test_mixed_bases_are_refused():
    a = (_acc_rows([100, 100, 100], "2.0", tokens=[1000, 1000, 1000], seconds=[10, 10, 10],
                   tag="A", rid="A1")
         + _acc_rows([100, 100, 100], "2.0", tokens=[1000, 1000, 1000], seconds=[10, 10, 10],
                     tag="A2", rid="A2"))
    b = (_acc_rows([100, 100, 100], "2.0", tag="B", rid="B1")          # no columns at all
         + _acc_rows([100, 100, 100], "2.0", tag="B2", rid="B2"))
    code, out = evaluate(a + b, "A", "B")
    assert code == 2 and any("different bases" in p for p in out["problems"]), out.get("problems")


def test_a_negative_point_estimate_inside_the_floor_does_not_withdraw():
    """Half of true nulls land below 1.0; that is not evidence of a regression."""
    d = {c: (100, 100, 99, 99) for c in PRIMARY_WEIGHTS}   # -1 %, floor will be >= 2 %
    code, out = evaluate(_rows(d), "A", "B")
    assert out["verdict"] == "inconclusive", out


def test_a_real_loss_withdraws():
    d = {c: (100, 100, 80, 80) for c in PRIMARY_WEIGHTS}   # -20 %, past the 2 % floor
    code, out = evaluate(_rows(d), "A", "B")
    assert out["verdict"] == "withdraw"


def test_combined_arm_uncertainty_not_the_maximum():
    """The ratio's interval is sqrt(a^2 + t^2); taking max(a, t) understates two noisy arms."""
    noisy = _rows({c: (100, 110, 100, 110) for c in PRIMARY_WEIGHTS})   # both arms identical spread
    _, out = noisy_ = evaluate(noisy, "A", "B")
    a = out["primary"]["spread_control"] or 0.0
    t = out["primary"]["spread_treat"] or 0.0
    f = out["primary"]["floor"] - 1.0
    assert abs(f - max(0.02, (a ** 2 + t ** 2) ** 0.5)) < 1e-9, (f, a, t)


def test_category_guard_uses_combined_variance_exactly():
    """max(a, t) and sqrt(a^2 + t^2) differ materially when BOTH arms are noisy; only the exact
    combination passes this."""
    # both arms carry a 10 % spread, so max == a while sqrt == 1.414 * a
    rows = (_rows({c: (100, 110, 100, 110) for c in PRIMARY_WEIGHTS}))
    _, out = evaluate(rows, "A", "B")
    for c, sp in out["category_spread"].items():
        combined = (sp["control"] ** 2 + sp["treat"] ** 2) ** 0.5
        assert abs(sp["combined"] - combined) < 1e-12, (c, sp)
        if sp["control"] > 0 and sp["treat"] > 0:
            assert sp["combined"] > max(sp["control"], sp["treat"]) * 1.2, (c, sp)
        expected = 1.0 / (1.0 + max(0.02, combined))
        assert abs(out["category_guard"][c] - expected) < 1e-12, (c, out["category_guard"][c], expected)


def test_bounds_are_reciprocal_and_the_gap_between_them_classifies_correctly():
    """A ratio between 1 - f and 1 / (1 + f) must classify as a loss under the reciprocal rule and
    as 'inside' under the old linear one: at f = 0.10 those are 0.900 and 0.9091, so 0.905 splits
    them. The old rule would have called 0.905 'inside' and not withdrawn."""
    import spec_primary as sp
    for f in (0.02, 0.05, 0.10, 0.25):
        assert abs(sp.pass_bound(f) - (1 + f)) < 1e-12
        assert abs(sp.loss_bound(f) - 1 / (1 + f)) < 1e-12
        assert sp.loss_bound(f) > 1 - f                      # the reciprocal is the stricter loss line
        mid = (sp.loss_bound(f) + (1 - f)) / 2
        # the discriminating band: ABOVE the old linear line (so the old rule saw no loss) but AT or
        # BELOW the reciprocal bound (so the current rule does). This is the case that made
        # `1 - floor` easier to satisfy than the pass bound.
        assert mid > 1 - f, f
        assert sp.classify_ratio(mid, f) == "loss", f
        assert sp.classify_ratio(sp.loss_bound(f) * 0.999, f) == "loss"
        assert sp.classify_ratio(sp.pass_bound(f) * 1.001, f) == "pass"
    assert sp.classify_ratio(0.905, 0.10) == "loss"           # the case the old rule let through
    assert sp.classify_ratio(0.95, 0.10) == "inside"


def test_a_large_loss_still_withdraws_end_to_end():
    d = {c: (100, 100, 86, 86) for c in PRIMARY_WEIGHTS}      # -14 %
    _, out = evaluate(_rows(d), "A", "B")
    assert out["verdict"] == "withdraw"
