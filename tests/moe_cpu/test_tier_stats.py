"""
The pinned-arena budget (EXL3_MOE_PINNED_ARENA_GIB) and the tier statistics (EXL3_MOE_TIER_STATS),
neither of which needs a GPU: the budget is arithmetic on chunk sizes, and the statistics are sums
over two snapshots of the per-layer hit counts (the CUDA side only decides how a snapshot travels).
What the tests pin down is the part a GPU-less box can still get wrong: the chunk that straddles the
budget, the zero budget meaning "no budget", the decay-corrected step arithmetic, and the padding
rule for layers of different width.
"""

import numpy as np
import pytest
import torch

from exllamav3.model.moe_cpu_host import _pin_budget_spent
from exllamav3.modules.block_sparse_mlp_cpu import _TierStats, _PROMOTE_DECAY

pytestmark = [pytest.mark.nogpu]


# -- pin budget ----------------------------------------------------------------------------------

def test_pin_budget_zero_is_no_budget():
    gib = 1 << 30
    assert not _pin_budget_spent(0, 0, 40 * gib)
    assert not _pin_budget_spent(0, 30 * gib, 2 * gib)


def test_pin_budget_admits_whole_chunks_only():
    gib = 1 << 30
    budget = 8 * gib
    pinned = 0
    admitted = 0
    for _ in range(33):
        if _pin_budget_spent(budget, pinned, gib):
            break
        pinned += gib
        admitted += 1
    assert admitted == 8 and pinned == budget


def test_pin_budget_refuses_a_chunk_that_straddles_it():
    gib = 1 << 30
    # 7 GiB in, a 2 GiB chunk would end at 9: refused, and the pinned bytes are left alone
    assert _pin_budget_spent(8 * gib, 7 * gib, 2 * gib)
    # A chunk that lands exactly on the budget is admitted
    assert not _pin_budget_spent(8 * gib, 6 * gib, 2 * gib)


# -- tier statistics -----------------------------------------------------------------------------

class _FakeSplit:
    """The attributes _TierStats reads off a dynamically placed split module"""

    def __init__(self, key, first, slots, spares = 0):
        self.key = key
        self.num_experts = len(slots)
        self.cpu_split_first = first
        self._split_spares = spares
        self._split_gen = 0
        self._split_map = torch.tensor(slots, dtype = torch.long)   # marker: dynamic placement on
        if spares:
            # Only the promote-on-miss path keeps a placement map; without it the map is the identity
            self._split_mp = list(slots)
        self._split_hist = torch.zeros(len(slots), dtype = torch.float)
        self.device = "cpu"
        self.cpu_host = None


def _consume(st, block):
    return st._consume(np.array(block, dtype = np.float32))


def test_tier_stats_counts_evaluations_and_offload_share():
    # 4 experts, slots [0, 1) on the GPU and [2, 3) on the worker: experts 0 and 1 are resident
    mod = _FakeSplit("layer0", first = 2, slots = [0, 1, 2, 3])
    st = _TierStats([mod])
    assert st.shape == (1, 4)
    assert st.decay == 1.0          # no promote-on-miss: the sweep's counts decay only at a sweep
    # The first block only arms the comparison: there is nothing before it to be a change of
    assert _consume(st, [[0, 0, 0, 0]]) is False
    assert (st.routed, st.missed, st.steps) == (0, 0, 0)

    assert _consume(st, [[1, 0, 1, 0]]) is True     # 2 evaluations, 1 of them on the worker
    assert (st.routed, st.missed, st.steps) == (2, 1, 1)

    assert _consume(st, [[2, 1, 1, 1]]) is True     # +3 evaluations (experts 0,1,3), +1 on the worker
    assert (st.routed, st.missed, st.steps) == (5, 2, 2)

    line = st.report()
    assert "60.0% coverage" in line and "5 expert evaluations" in line and "2 on the CPU" in line
    # The report resets the counters
    assert (st.routed, st.missed, st.steps) == (0, 0, 0)
    assert st.report() is None


def test_tier_stats_survives_the_promote_decay():
    # Promote-on-miss decays every count by 0.98 per step, so a routed expert's count can *fall*
    # between two snapshots; the step's evaluations are recovered from the sum, not per expert
    mod = _FakeSplit("layer0", first = 1, slots = [0, 1, 2], spares = 1)
    st = _TierStats([mod])
    assert st.decay == pytest.approx(_PROMOTE_DECAY)
    st.prev = np.array([[100.0, 200.0, 0.0]], dtype = np.float32)
    # Expert 0 (resident) routed once from a count of 100: 98 + 1 = 99. Expert 1 (worker) not routed
    # and decayed to 196. Expert 2 untouched
    assert _consume(st, [[99.0, 196.0, 0.0]]) is True
    assert (st.routed, st.missed) == (1, 0)
    # Now the worker expert is routed from a high count: 196 * 0.98 + 1 = 193.08, a *fall* of 2.92
    # against the undecayed value but +1 against the decayed one
    assert _consume(st, [[97.02, 193.08, 0.0]]) is True
    assert (st.routed, st.missed) == (2, 1)


def test_tier_stats_rebaselines_after_a_hard_decay():
    # A sweep multiplies every count by 0.5, which no step of routing can explain: the block must
    # re-baseline rather than report a negative step
    mod = _FakeSplit("layer0", first = 1, slots = [0, 1, 2])
    st = _TierStats([mod])
    st.prev = np.array([[100.0, 100.0, 0.0]], dtype = np.float32)
    assert _consume(st, [[50.0, 50.0, 0.0]]) is False
    assert (st.routed, st.missed, st.steps) == (0, 0, 0)
    assert np.array_equal(st.prev, np.array([[50.0, 50.0, 0.0]], dtype = np.float32))
    # And the next step counts normally against the new baseline
    assert _consume(st, [[51.0, 50.0, 0.0]]) is True
    assert (st.routed, st.missed) == (1, 0)


def test_tier_stats_follows_a_promotion():
    # The instrument exists to measure promotion, so it must not score a promoted expert at its old
    # home: the placement mirror is refreshed whenever a module's placement generation moves.
    mod = _FakeSplit("layer0", first = 2, slots = [0, 1, 2, 3], spares = 1)
    st = _TierStats([mod])
    assert _consume(st, [[0, 0, 0, 0]]) is False
    assert _consume(st, [[0, 0, 1, 0]]) is True           # expert 2 routed: a worker slot, a miss
    assert (st.routed, st.missed) == (1, 1)

    # Promote expert 2 into GPU slot 0 and demote expert 0 into the worker slot it vacated
    mod._split_mp = [mod.cpu_split_first + 0, 1, 0, 3]
    mod._split_gen += 1
    assert _consume(st, [[0, 0, 2, 0]]) is True           # expert 2 routed again, now resident
    assert (st.routed, st.missed) == (2, 1), "a promoted expert must count as a hit"
    # And the demoted one is now a miss
    assert _consume(st, [[1, 0, 2, 0]]) is True
    assert (st.routed, st.missed) == (3, 2), "a demoted expert must count as a miss"


def test_tier_stats_follows_a_sweep_without_a_placement_map():
    # The sweep path keeps only the device map (`_split_map`) and no host mirror, so the refresh
    # reads it back - once per placement change, never per step
    mod = _FakeSplit("layer0", first = 2, slots = [0, 1, 2, 3])
    st = _TierStats([mod])
    assert _consume(st, [[0, 0, 0, 0]]) is False
    assert _consume(st, [[0, 0, 1, 0]]) is True            # expert 2 on the worker
    assert (st.routed, st.missed) == (1, 1)
    # A sweep swaps experts 0 and 2 in the device map
    mod._split_map = torch.tensor([2, 1, 0, 3], dtype = torch.long)
    mod._split_gen += 1
    assert _consume(st, [[0, 0, 2, 0]]) is True            # expert 2 again, now in a GPU slot
    assert (st.routed, st.missed) == (2, 1), "the sweep's move must be visible to the mask"


def test_tier_stats_ignores_a_step_that_routed_nothing():
    mod = _FakeSplit("layer0", first = 2, slots = [0, 1, 2, 3])
    st = _TierStats([mod])
    assert _consume(st, [[0, 0, 0, 0]]) is False        # arms
    assert _consume(st, [[0, 0, 0, 0]]) is False        # an idle step is not a step
    assert (st.routed, st.missed, st.steps) == (0, 0, 0)


def test_tier_stats_invalidate_forces_a_rebaseline():
    # A prefill, a capture or a load moves the counts for a reason no decode step explains; the next
    # block must only re-baseline instead of reporting the whole prompt as one step's evaluations
    mod = _FakeSplit("layer0", first = 2, slots = [0, 1, 2, 3])
    st = _TierStats([mod])
    assert _consume(st, [[1, 1, 1, 1]]) is False
    st.invalidate()
    assert st.prev is None
    assert _consume(st, [[500, 400, 300, 200]]) is False   # the prompt's routing, not the conversation's
    assert (st.routed, st.missed, st.steps) == (0, 0, 0)
    assert _consume(st, [[501, 400, 300, 200]]) is True    # and the next real step counts again
    assert (st.routed, st.missed, st.steps) == (1, 0, 1)


def test_tier_stats_padding_never_counts():
    # Two layers of different width (the draft model's layers are shorter): the padding columns must
    # not be counted as evaluations of either kind, and the per-layer figures must not mix
    wide = _FakeSplit("layer0", first = 2, slots = [0, 1, 2, 3])
    narrow = _FakeSplit("layer1", first = 1, slots = [0, 1])
    st = _TierStats([wide, narrow])
    assert st.shape == (2, 4)
    assert _consume(st, [[0, 0, 0, 0], [0, 0, 0, 0]]) is False
    # layer 0 routes expert 2 (a worker slot: a miss); layer 1 routes expert 0 (resident: a hit)
    assert _consume(st, [[0, 0, 1, 0], [1, 0, 0, 0]]) is True
    assert (st.routed, st.missed) == (2, 1)
    assert int(st.per_layer[0].sum()) == 1 and int(st.per_layer[1].sum()) == 0


def test_tier_stats_detail_lists_the_worst_layers():
    mods = [_FakeSplit(f"layer{i}", first = 1, slots = [0, 1, 2]) for i in range(3)]
    st = _TierStats(mods)
    assert _consume(st, [[0.0] * 3] * 3) is False
    block = [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    assert _consume(st, block) is True
    line = st.report(detail = 2)
    assert "worst layers:" in line and "layer0:2" in line and "layer2:0" in line
