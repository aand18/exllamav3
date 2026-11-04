#!/usr/bin/env python3
"""The primary-metric rule, pinned on synthetic arms (no harness, no model)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


TESTS = [
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
