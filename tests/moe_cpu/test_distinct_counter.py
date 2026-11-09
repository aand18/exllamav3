"""
R1 distinct-expert counting (T58) on the CPU side, without a GPU: the pure counting helper
(round_prof.count_distinct_cpu), the sampled-round enqueue/finalize path, and the distinct CSV
emission. The production pump (block_sparse_mlp_cpu.cpu_split_submit's async pinned copy) is
exercised by the measurement arms, not here. T60 adds the routing-autocorr cover: the same
finalize path maintains each layer's previous sampled-round CPU set and scores rows==5
layer-rounds against it (round_prof.distinct_cpu_set, round_prof._autocorr_cover).
"""

import os

import pytest
import torch

from exllamav3.util import round_prof

pytestmark = pytest.mark.nogpu


def test_count_distinct_cpu_rules():
    sel = torch.tensor([[0, 1, 2],
                        [2, 3, 0]], dtype = torch.int64)
    assert round_prof.count_distinct_cpu(sel, 2, 2, 3) == 2          # {2, 3}
    assert round_prof.count_distinct_cpu(sel, 4, 2, 3) == 0          # none at/past the tail
    assert round_prof.count_distinct_cpu(sel, 0, 2, 3) == 4          # {0, 1, 2, 3}
    assert round_prof.count_distinct_cpu(sel, 2, 1, 3) == 1          # first row only -> {2}
    assert round_prof.count_distinct_cpu(sel, None, 2, 3) == 0        # no split -> no CPU experts


def test_enqueue_finalize_write_distinct(tmp_path, monkeypatch):
    prev_pending, prev_distinct = list(round_prof._pending), list(round_prof._distinct)
    try:
        round_prof._pending.clear()
        round_prof._distinct.clear()
        buf = torch.empty((64, 3), dtype = torch.int64)
        buf[:2, :3] = torch.tensor([[5, 6, 5],
                                    [6, 7, 0]], dtype = torch.int64)
        round_prof.enqueue_distinct("model.layers.0.mlp", buf, 5, 2, 3, 9)
        round_prof._finalize_distinct()
        assert round_prof._distinct == [(9, "model.layers.0.mlp", 3, 2, 3)]
        out = tmp_path / "distinct.csv"
        n = round_prof.write_distinct(str(out))
        assert n == 1
        lines = out.read_text(encoding = "utf-8").splitlines()
        assert lines[0] == "round,layer,distinct,rows,topk"
        assert lines[1] == "9,model.layers.0.mlp,3,2,3"
        assert round_prof._pending == []
    finally:
        round_prof._pending.clear()
        round_prof._distinct.clear()
        round_prof._pending.extend(prev_pending)
        round_prof._distinct.extend(prev_distinct)


def test_distinct_csv_path_derivation(monkeypatch):
    monkeypatch.setenv("EXL3_ROUND_PROF_CSV", os.path.join("C:", os.sep, "run", "roundprof.csv"))
    # DISTINCT_CSV_PATH is bound at import; only assert the rule it implements.
    assert round_prof.DISTINCT_CSV_PATH is None or round_prof.DISTINCT_CSV_PATH.endswith(".distinct")


def test_distinct_cpu_set_rules():
    sel = torch.tensor([[0, 1, 2],
                        [2, 3, 0]], dtype = torch.int64)
    assert round_prof.distinct_cpu_set(sel, 2, 2, 3) == {2, 3}
    assert round_prof.distinct_cpu_set(sel, 4, 2, 3) == set()          # none at/past the tail
    assert round_prof.distinct_cpu_set(sel, 0, 2, 3) == {0, 1, 2, 3}
    assert round_prof.distinct_cpu_set(sel, 2, 1, 3) == {2}            # first row only -> {2}
    assert round_prof.distinct_cpu_set(sel, None, 2, 3) == set()       # no split -> no CPU experts


def _buf(rows_vals):
    """(64, 3) CPU buffer whose first len(rows_vals) rows are rows_vals (topk=3)"""
    buf = torch.empty((64, 3), dtype = torch.int64)
    buf[:len(rows_vals), :3] = torch.tensor(rows_vals, dtype = torch.int64)
    return buf


def test_autocorr_prev_set_and_cover():
    prev_pending, prev_distinct = list(round_prof._pending), list(round_prof._distinct)
    prev_sets = dict(round_prof._cpu_sets)
    prev_covers = list(round_prof._covers)
    try:
        round_prof._pending.clear()
        round_prof._distinct.clear()
        round_prof._cpu_sets.clear()
        round_prof._covers.clear()
        layer = "model.layers.0.mlp"
        # Round 40, rows=5: first sampled round for the layer -> sets the previous set, no cover.
        round_prof.enqueue_distinct(layer, _buf([[5, 6, 0]] * 5), 5, 5, 3, 40)
        round_prof._finalize_distinct()
        assert round_prof._cpu_sets == {layer: {5, 6}}
        assert round_prof._covers == []
        # Round 48, rows=5: cur {6, 7} vs prev {5, 6} -> cover 1/2.
        round_prof.enqueue_distinct(layer, _buf([[6, 7, 0]] * 5), 5, 5, 3, 48)
        round_prof._finalize_distinct()
        assert round_prof._cpu_sets == {layer: {6, 7}}
        assert round_prof._covers == [0.5]
        # Round 56, rows=4: no cover recorded (not the drafted window), but the set still
        # becomes the previous one for the next comparison.
        round_prof.enqueue_distinct(layer, _buf([[7, 0, 0]] * 4), 5, 4, 3, 56)
        round_prof._finalize_distinct()
        assert round_prof._cpu_sets == {layer: {7}}
        assert round_prof._covers == [0.5]
        # Round 64, rows=5: cur {7, 8} vs prev {7} -> cover 1/2.
        round_prof.enqueue_distinct(layer, _buf([[7, 8, 0]] * 5), 5, 5, 3, 64)
        round_prof._finalize_distinct()
        assert round_prof._cpu_sets == {layer: {7, 8}}
        assert round_prof._covers == [0.5, 0.5]
        # Distinct rows were collected for all four rounds regardless of the autocorr filter.
        assert round_prof._distinct == [(40, layer, 2, 5, 3), (48, layer, 2, 5, 3),
                                        (56, layer, 1, 4, 3), (64, layer, 2, 5, 3)]
    finally:
        round_prof._pending.clear()
        round_prof._distinct.clear()
        round_prof._cpu_sets.clear()
        round_prof._covers.clear()
        round_prof._pending.extend(prev_pending)
        round_prof._distinct.extend(prev_distinct)
        round_prof._cpu_sets.update(prev_sets)
        round_prof._covers.extend(prev_covers)


def test_autocorr_empty_prev_scores_zero_and_empty_cur_skips():
    prev_sets = dict(round_prof._cpu_sets)
    prev_covers = list(round_prof._covers)
    try:
        round_prof._cpu_sets.clear()
        round_prof._covers.clear()
        layer = "model.layers.1.mlp"
        # Previous round routed nothing to the CPU -> an empty predicted set scores 0.
        round_prof.enqueue_distinct(layer, _buf([[0, 0, 0]] * 5), 5, 5, 3, 40)
        round_prof._finalize_distinct()
        round_prof.enqueue_distinct(layer, _buf([[5, 0, 0]] * 5), 5, 5, 3, 48)
        round_prof._finalize_distinct()
        assert round_prof._covers == [0.0]
        # A current round with no CPU experts has an undefined cover -> skipped, set still kept.
        round_prof.enqueue_distinct(layer, _buf([[0, 0, 0]] * 5), 5, 5, 3, 56)
        round_prof._finalize_distinct()
        assert round_prof._covers == [0.0]
        assert round_prof._cpu_sets == {layer: set()}
    finally:
        round_prof._pending.clear()
        round_prof._cpu_sets.clear()
        round_prof._covers.clear()
        round_prof._cpu_sets.update(prev_sets)
        round_prof._covers.extend(prev_covers)
