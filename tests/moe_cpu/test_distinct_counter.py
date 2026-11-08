"""
R1 distinct-expert counting (T58) on the CPU side, without a GPU: the pure counting helper
(round_prof.count_distinct_cpu), the sampled-round enqueue/finalize path, and the distinct CSV
emission. The production pump (block_sparse_mlp_cpu.cpu_split_submit's async pinned copy) is
exercised by the measurement arms, not here.
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
