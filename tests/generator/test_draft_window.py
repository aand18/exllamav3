"""
Cost-modelled draft window (EXL3_DRAFT_COST_MODEL).

The dynamic draft path shrank the verified window on a confidence cutoff alone. On live traffic the
DFlash2 window accepts 39.6 % of its drafts (1.58 of 4) against 58.5 % on the benchmark corpus
(wiki/plans/v165-STATE.md:203), and every rejected draft still routes the round's experts across all
48 split layers, so a window that is likely to fail pays the CPU expert tier for tokens it never
returns. The cost model scores each window by expected tokens per millisecond of round cost and takes
the widest one within 2 % of the best score.

`choose_draft_window` is pure and never reads the environment; `cost_model_window` is the caller's
gate and is the only place `EXL3_DRAFT_COST_MODEL` is read. Unset (or 0) must reproduce today's
behaviour: the configured window, whatever the confidence cutoff decided.
"""

import pytest

from exllamav3.generator.generator import (
    choose_draft_window,
    cost_model_window,
    draft_cost_model_enabled,
)

pytestmark = pytest.mark.nogpu


# The two inputs the defect was measured with: live traffic at the long-context end (39.6 % of
# drafts accepted over 760,980 drafts, T14) and the benchmark corpus (58.5 %, v165-STATE.md:203).
LIVE_LONG = dict(acceptance = 0.25, context_len = 200_000, distinct_experts = 22.0, base = 4)
CORPUS_SHORT = dict(acceptance = 0.85, context_len = 4_096, distinct_experts = 8.0, base = 4)


def test_low_acceptance_on_long_context_shrinks_the_window():
    """
    A round that is likely to fail must not verify four rows: its expected tokens saturate while
    the CPU expert work keeps growing with the window.
    """
    assert choose_draft_window(**LIVE_LONG) < 4


def test_high_acceptance_keeps_the_window():
    """On the corpus, where most drafts are accepted, the full configured window still pays."""
    assert choose_draft_window(**CORPUS_SHORT) == 4


def test_the_score_never_returns_zero_or_more_than_base():
    """
    The selection is a window, not a skip: whatever the inputs, the result stays inside 1..base
    (the drafting loop indexes the drafted block with it).
    """
    for base in (1, 2, 3, 4, 5, 8, 16):
        for acceptance in (0.0, 0.05, 0.25, 0.5, 0.585, 0.75, 1.0):
            for context_len in (0, 4_096, 40_960, 131_072, 200_000, 500_000):
                for distinct_experts in (0.0, 1.0, 8.0, 22.0, 64.0, 512.0):
                    w = choose_draft_window(acceptance, context_len, distinct_experts, base)
                    assert 1 <= w <= base, (acceptance, context_len, distinct_experts, base, w)


def test_the_model_is_a_pass_through_when_off(monkeypatch):
    """
    Unset or 0: the caller keeps the configured window even when `choose_draft_window` would shrink
    it. This is the default-off guarantee - an unset env var must reproduce today's bytes.
    """
    monkeypatch.delenv("EXL3_DRAFT_COST_MODEL", raising = False)
    assert not draft_cost_model_enabled()
    # The model itself would shrink this round to a single row...
    assert choose_draft_window(**LIVE_LONG) < 4
    # ...and the gate does not ask it to.
    assert cost_model_window(LIVE_LONG["base"], LIVE_LONG["acceptance"],
                             LIVE_LONG["context_len"], LIVE_LONG["distinct_experts"]) == 4

    monkeypatch.setenv("EXL3_DRAFT_COST_MODEL", "0")
    assert cost_model_window(LIVE_LONG["base"], LIVE_LONG["acceptance"],
                             LIVE_LONG["context_len"], LIVE_LONG["distinct_experts"]) == 4

    monkeypatch.setenv("EXL3_DRAFT_COST_MODEL", "1")
    assert cost_model_window(LIVE_LONG["base"], LIVE_LONG["acceptance"],
                             LIVE_LONG["context_len"], LIVE_LONG["distinct_experts"]) < 4


def test_a_monotone_reduction():
    """
    More distinct experts per round at the same acceptance and context means more CPU expert work
    for the same expected tokens, which can widen the window for nobody. The choice must never grow
    as that cost rises.
    """
    for acceptance, context_len in ((0.85, 4_096), (0.6, 100_000), (0.25, 200_000)):
        widths = [
            choose_draft_window(acceptance, context_len, float(d), 4)
            for d in (1, 2, 4, 6, 8, 10, 12, 16, 20, 22, 24, 32, 48, 64)
        ]
        assert widths == sorted(widths, reverse = True), (acceptance, context_len, widths)


def test_the_acceptance_prior_falls_with_context():
    """
    T14's bands are the only acceptance evidence this box has: the same (over-confident) acceptance
    estimate has to buy a narrower window on a long context than on a short one. This is the linear
    interpolation between the documented bands, and it is only as good as those bands.
    """
    assert choose_draft_window(1.0, 200_000, 22.0, 4) < choose_draft_window(1.0, 4_096, 22.0, 4)
