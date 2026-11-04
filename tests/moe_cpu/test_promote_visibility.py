"""
The promote-on-miss policy at the three places where it could not see what it exists to move.

1. Visibility. A routing adds 1 to an expert's count, and the policy's decay removes
   (1 - decay) * count of it per step. Comparing a snapshot against the raw previous count
   (``cur > prev``) is therefore false for every expert at count >= 1 / (1 - decay) -- 50 at the
   shipped 0.98 -- even when it *was* routed: the hotter a CPU expert is, the more certainly the
   policy was blind to it. ``_routed_mask`` undoes the decay before comparing, so a step's hits are
   recovered at any count magnitude.

2. The prefill boundary. A prefill-size batch routes the prompt through the same ``_split_hist`` the
   policy reads (``_split_translate`` -> ``moe_split_map``), so without a reset the first decode
   steps rank experts by prompt routing. ``_promote_reset`` zeroes the counts and drops the
   comparison window, and the step returns before it issues anything.

3. The cadence (``EXL3_MOE_CPU_PROMOTE_EVERY`` / ``_DECAY``). A pass every N steps that decays by
   ``decay ** N`` keeps the history's half-life in wall-clock steps, which is what makes Strata's
   adapt-every-N-rounds a delta here rather than a fork. The tier instrument has to read the same
   pair, or it scores a pass's routing against one step's decay.

No GPU: the histograms are plain torch CPU tensors, and the one test that has to reach past the
policy's CUDA half (the pinned snapshot the next pass compares against) stands a stub in for it.
Nothing here builds a model.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import exllamav3.modules.block_sparse_mlp_cpu as bsm_cpu
from exllamav3.modules.block_sparse_mlp_cpu import (
    _TierStats, _promote_cadence, _routed_mask, _split_promote_step)

pytestmark = [pytest.mark.nogpu]

MIN_ROWS = 32     # cpu_host.stream_min_rows: at or above this a batch is a prefill, not a decode step


# -- 1/2: visibility at high counts ---------------------------------------------------------------

def test_a_hot_routed_expert_is_visible():
    # An expert whose count is already 50, routed once: the next snapshot reads 50 * 0.98 + 1 = 50,
    # no greater than the count it already had, and at 60 it reads 59.8, strictly lower. That is the
    # defect: `cur > prev` called a routed expert unrouted from count 1 / (1 - decay) up, i.e. exactly
    # where the count is high enough for the promotion to pay
    prev = np.array([[50.0]], dtype = np.float32)
    cur = prev * 0.98 + 1.0
    assert _routed_mask(cur, prev, 0.98)[0, 0]
    hot = np.array([[60.0]], dtype = np.float32)
    assert not bool(hot * 0.98 + 1.0 > hot)              # what the policy asked, and why it missed
    assert _routed_mask(hot * 0.98 + 1.0, hot, 0.98)[0, 0]


def test_an_expert_that_was_not_routed_is_invisible():
    # The same count, one step later, with nothing routed: the snapshot is the pure decay
    prev = np.array([[50.0]], dtype = np.float32)
    cur = prev * 0.98
    assert not _routed_mask(cur, prev, 0.98)[0, 0]


# -- 3: the prefill boundary ----------------------------------------------------------------------

class _FakeSplit:
    """Only the attributes _split_promote_step reads off a module before it reaches its CUDA half"""

    def __init__(self, experts = 8, first = 6, min_rows = MIN_ROWS, device = "cpu"):
        self.key = "model.layers.0.mlp"
        self.num_experts = experts
        self.cpu_split_first = first
        self.device = device
        self._split_spares = experts - first
        self._split_pending = []          # nothing in flight
        self._split_free_gpu = []
        self._split_free_cpu = []
        self._split_mp = np.arange(experts, dtype = np.int64)
        self._split_map = torch.arange(experts, dtype = torch.long)
        self._split_gen = 0
        self._split_hist = torch.zeros(experts, dtype = torch.float)
        self._split_step_count = 0
        self._split_promo_n = self._split_demo_n = 0
        self.cpu_host = SimpleNamespace(stream_min_rows = min_rows)


def _state(modules, prev = None, snap_ev = None):
    """The promote bookkeeping _split_promote_step reads, without _PromoteState's pinned snapshot"""
    return SimpleNamespace(modules = modules, prev = prev, snap_ev = dict(snap_ev or {}),
                           stats = dict(routed = 0, missed = 0, steps = 0, ms = 0.0))


def test_prefill_resets_the_histograms():
    m = _FakeSplit()
    m._split_hist.fill_(500.0)                                  # the prompt's routing
    state = _state([m], prev = np.full((1, m.num_experts), 400.0, dtype = np.float32),
                   snap_ev = {0: None})                         # a snapshot in flight
    ip = SimpleNamespace(moe_cpu_promote_state = state)
    _split_promote_step(ip, [m], rows = 64)                     # prefill-size
    assert float(m._split_hist.abs().sum()) == 0.0
    assert state.prev is None
    assert not state.snap_ev


# -- 4: the cadence ------------------------------------------------------------------------------

def test_every_n_runs_once_per_n_steps(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_PROMOTE_EVERY", "4")
    monkeypatch.delenv("EXL3_MOE_CPU_PROMOTE_DECAY", raising = False)
    m = _FakeSplit()
    ip = SimpleNamespace(moe_cpu_promote_state = _state([m]))
    passes = []

    def snapshot(ps, decay):
        # Stands in for the pinned copy and its events. The histogram decay the pass applies stays
        # the production one, so what this checks is the cadence and the decay the pass is handed
        passes.append(decay)
        return None

    monkeypatch.setattr(bsm_cpu, "_promote_snapshot", snapshot)
    m._split_hist.fill_(100.0)
    for step in range(1, 5):
        m._split_step_count = step        # _split_swap_tick increments before it promotes
        m._split_hist.add_(1.0)           # every expert routed once between passes
        _split_promote_step(ip, [m], rows = 1)
    assert len(passes) == 1, "a pass per step: EXL3_MOE_CPU_PROMOTE_EVERY was not honoured"
    assert passes[0] == pytest.approx(0.98 ** 4), "the pass must decay by the four steps it spans"
    assert float(m._split_hist[0]) == pytest.approx(104.0 * 0.98 ** 4)


# -- the instrument reads the same cadence -------------------------------------------------------

def test_the_tier_instrument_reads_the_same_cadence(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_PROMOTE_EVERY", "4")
    monkeypatch.delenv("EXL3_MOE_CPU_PROMOTE_DECAY", raising = False)
    every, decay = _promote_cadence()
    assert every == 4
    assert decay == pytest.approx(0.98 ** 4)
    # A pass decays by decay ** every, so the coverage instrument must undo the same amount: at one
    # step's decay it would report the three steps it never decayed as routed
    assert _TierStats([_FakeSplit()]).decay == pytest.approx(0.98 ** 4)
