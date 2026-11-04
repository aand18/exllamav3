#!/usr/bin/env python3
"""The lookahead probe's arithmetic, and the refusal that must never be silent.

`bench/router_lookahead_probe.py` needs the engine and a GPU to answer its question, but every
number it prints is derived from three pure helpers - the overlap, the cosine, and the split-tail
coverage - and from a reason string for each layer the probe could not instrument. None of that
needs torch, and none of it may fail quietly: a probe that instruments nothing and reports a clean
zero would read as "the router is unpredictable", which is the opposite of a measurement.

Run on a box with no pytest (the Linux side of this repo has neither pytest nor torch):

    python3 bench/test_router_lookahead_probe.py

or under pytest where it exists (the Windows box, in its venv):

    python3 -m pytest bench/test_router_lookahead_probe.py -q
"""

import pathlib
import sys
import traceback

# `pytest bench/...` puts bench/ on sys.path only in the default import mode; this keeps a direct
# `python3 bench/test_router_lookahead_probe.py` working too.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from router_lookahead_probe import (  # noqa: E402
    LayerStats,
    corpus_summary,
    cosine,
    probe_verdict,
    router_support_reason,
    tail_coverage,
    topk_overlap,
)


def test_overlap_counts_a_perfect_prediction():
    """Same top-k, listed in a different order: the overlap is all of it."""
    true_rows = [[3, 7, 11, 42], [0, 1, 2, 3]]
    pred_rows = [[42, 11, 3, 7], [3, 2, 1, 0]]
    assert topk_overlap(true_rows, pred_rows) == [1.0, 1.0]

    # A reordering is not the only way to be right: a prediction may be a superset (a wider
    # lookahead) and still cover every true pick. The denominator is the true pick count.
    assert topk_overlap([[5, 9]], [[5, 9, 13, 21]]) == [1.0]


def test_overlap_counts_a_disjoint_prediction():
    """Every pick wrong: zero, not a missing row and not a one-sided fraction."""
    true_rows = [[3, 7, 11, 42], [100, 101]]
    pred_rows = [[4, 8, 12, 43], [200, 201]]
    assert topk_overlap(true_rows, pred_rows) == [0.0, 0.0]

    # Half the picks right is half, and the rows may be of different widths
    assert topk_overlap([[3, 7, 11, 42]], [[3, 7, 90, 91]]) == [0.5]

    # A row-count mismatch is a bug in the caller, not a measurement of zero
    try:
        topk_overlap([[1, 2]], [[1, 2], [3, 4]])
    except ValueError:
        pass
    else:
        raise AssertionError("a row-count mismatch must raise, not report an overlap")


def test_cosine_of_identical_and_orthogonal_vectors():
    assert abs(cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-12
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-12
    assert abs(cosine([0.5, 0.25, -2.0], [-0.5, -0.25, 2.0]) + 1.0) < 1e-12
    # A zero vector has no direction: the probe reports 'no data', and the mean excludes it,
    # because a 0.0 here would read as a measured disagreement
    assert cosine([0.0, 0.0], [1.0, 2.0]) is None
    assert cosine([], []) is None


def test_a_layer_without_a_linear_router_is_reported_not_skipped_silently():
    """A router whose selection is not a function of the hidden state gets a named reason that
    reaches the output rows and the corpus summary - it is reported, not dropped."""

    class Stub:
        """Only the attributes the support check reads, so the test needs no torch."""

        def __init__(self, **kw):
            self.routing_gate = object()
            self.router_type = "std"
            self.tid2eid = None
            self.alt_residual_channel = False
            self.__dict__.update(kw)

    hash_router = Stub(router_type="sqrtsp_hash", tid2eid=object())
    unknown_router = Stub(router_type="quantum_magic")
    no_router = Stub(routing_gate=None)
    plain = Stub()

    assert router_support_reason(plain) is None, "a plain linear router must be instrumentable"

    r_hash = router_support_reason(hash_router)
    r_unknown = router_support_reason(unknown_router)
    r_none = router_support_reason(no_router)
    assert r_hash and r_unknown and r_none, "every refusal must carry a reason"
    assert len({r_hash, r_unknown, r_none}) == 3, "the reasons must name different causes"
    assert "hash" in r_hash and "hidden state" in r_hash
    assert "quantum_magic" in r_unknown
    assert "broadcast" in r_none

    rows = [
        LayerStats(layer_idx=0, name="model.layers.0.mlp").finished_row(),
        LayerStats(layer_idx=3, name="model.layers.3.mlp", status="skipped",
                   reason=r_hash).finished_row(),
    ]
    assert rows[0]["status"] == "measured"
    assert rows[1]["status"] == "skipped" and rows[1]["reason"] == r_hash

    summary = corpus_summary(rows)
    assert summary["layers_measured"] == 1 and summary["layers_skipped"] == 1
    assert summary["skipped_reasons"] == [r_hash], "the reason must survive into the summary"


def test_tail_coverage_counts_only_the_offloaded_picks():
    """With a CPU split, a lookahead prefetch can only help the picks that are not resident."""
    true_rows = [[10, 100, 101, 200]]
    pred_rows = [[10, 100, 300, 301]]
    # first CPU-resident (tail) router id = 100: picks 100, 101, 200 are misses, 100 predicted
    assert tail_coverage(true_rows, pred_rows, 100) == (3, 1)
    assert tail_coverage(true_rows, pred_rows, None) == (0, 0)


def test_the_engine_names_the_probe_hooks_still_exist():
    """The probe wraps four instance attributes and reads one params key. A rename in the engine
    would leave the probe silently measuring nothing, so the names are checked against the source
    (text, not import: the Linux side has no torch)."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    sources = {
        "mlp": (repo / "exllamav3" / "modules" / "block_sparse_mlp.py").read_text(),
        "block": (repo / "exllamav3" / "modules" / "transformer.py").read_text(),
        "cpu": (repo / "exllamav3" / "modules" / "block_sparse_mlp_cpu.py").read_text(),
    }
    for text, needle in [
        (sources["mlp"], "self.routing_fn = "),            # 3. wrapped: the true picks
        (sources["mlp"], "self.routing_gate = "),          #    the gate the prediction re-applies
        (sources["mlp"], "self.router_pre_norm = "),
        (sources["mlp"], "self.routing_cfg = RoutingCFG("),
        (sources["mlp"], "self.num_experts_per_tok = "),
        (sources["mlp"], "self.hidden_size = "),
        (sources["block"], 'params["residual"] = x'),      # 2. the post-attention residual
        (sources["block"], 'params["resid_pending"]'),     # 1. the deferred add the pre state needs
        (sources["block"], "self.mlp_norm = mlp_norm"),
        (sources["cpu"], "self.cpu_split_first = first"),  # the CPU tail the tail coverage counts
    ]:
        assert needle in text, f"the engine no longer contains {needle!r}: the probe's hook moved"


def test_the_probe_refuses_to_report_a_clean_zero():
    """Both ways of measuring nothing must exit 2 with a line saying which condition fired."""
    code, line = probe_verdict(0, [])
    assert code == 2 and "instrumented == 0" in line
    code, line = probe_verdict(3, [])
    assert code == 2 and "layers_with_rows == 0" in line
    # one layer with one compared row is a measurement, however thin
    rows = [{"layer": 0, "status": "measured",
             "cosine_min": 0.9,
             "counts": {"rows": 1, "cos_sum": 0.9, "cos_rows": 1, "overlap_sum": 0.5,
                        "overlap_rows": 1, "picks": 2, "covered": 1}}]
    code, line = probe_verdict(1, rows)
    assert code == 0 and line is None
    # a layer that was instrumented but measured nothing is still a refusal
    code, line = probe_verdict(1, [{"layer": 0, "status": "measured", "counts": {"rows": 0}}])
    assert code == 2 and "layers_with_rows == 0" in line


TESTS = [
    test_overlap_counts_a_perfect_prediction,
    test_overlap_counts_a_disjoint_prediction,
    test_cosine_of_identical_and_orthogonal_vectors,
    test_the_probe_refuses_to_report_a_clean_zero,
    test_a_layer_without_a_linear_router_is_reported_not_skipped_silently,
    test_tail_coverage_counts_only_the_offloaded_picks,
    test_the_engine_names_the_probe_hooks_still_exist,
]


if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            test()
        except BaseException:
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)
