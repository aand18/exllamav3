"""
Task 4 of the decode-to-strata plan (wiki/plans/2026-11-04-decode-to-strata.md).

eval/spec_decode.py measures two arms in one process (baseline, then the same model with a drafter).
With dynamic expert placement on (EXL3_MOE_CPU_SWAP) the first arm's promotions are the second arm's
starting state, so the arm-to-arm ratio compares two placements rather than two drafters.
reset_placement() puts every split module back to the placement its load built.

A synthetic split module carries the state that decides a placement - the router->slot map, the free
slot lists, the transfers in flight, the hit histogram and the promote bookkeeping - with no
checkpoint, no CPU worker and no device. The weights behind the map are restored from the checkpoint
and need a worker, so the box exercises those; this file covers the placement state itself.
"""

import ast
import os
from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.block_sparse_mlp_cpu import (
    _split_map_initialise,
    reset_placement,
)

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(REPO, "eval", "spec_decode.py")


def split_module(num_experts, first, spares = 0):
    """What load_cpu_split() + cpu_post_load() leave on a split module, on the CPU: `first` GPU slots
    (the last `spares` of them free), the remaining experts on the worker. No checkpoint is read and
    no worker is started, so nothing here can have moved."""
    m = SimpleNamespace(
        device = torch.device("cpu"),
        num_experts = num_experts,
        cpu_split_first = first,
        _split_spares = spares,
        _split_perm = None,
        _split_saved = None,      # no worker state: the weights cannot have moved
    )
    _split_map_initialise(m)
    return m


def test_the_map_returns_to_the_load_time_placement():
    # 8 experts, 5 GPU slots, no spares: the map loads as the identity. One promotion and one
    # demotion later, router 1 computes on the worker (physical slot 5) and router 5 in GPU slot 1
    m = split_module(num_experts = 8, first = 5)
    m._split_map = torch.tensor([0, 5, 2, 3, 4, 1, 6, 7], dtype = torch.long)

    reset_placement(SimpleNamespace(modules = [m]))

    fresh = split_module(num_experts = 8, first = 5)
    assert m._split_map.tolist() == list(range(8))
    assert torch.equal(m._split_map, fresh._split_map)
    assert m._split_map.dtype == fresh._split_map.dtype


def test_pending_transfers_and_histograms_are_cleared():
    # 9 experts, 5 GPU slots, 2 spares: experts 3..8 live in worker slots 0..5, GPU slots 3 and 4
    # start free, and router 1 was promoted into spare slot 3 during the first arm
    m = split_module(num_experts = 9, first = 5, spares = 2)
    ip = SimpleNamespace(moe_cpu_promote_state = object(), moe_cpu_swap_pending = True)
    m.config = SimpleNamespace(infer_params = ip)
    m._split_hist = torch.full((9,), 12.0)
    m._split_pending = [dict(kind = "p", r = 1, gpu = 3, cpu = 2, ev = None)]
    m._split_busy = {1}
    m._split_free_gpu = []
    m._split_free_cpu = [2]
    m._split_mp = [0, 3, 2, 5, 6, 7, 8, 9, 10]
    m._split_map = torch.tensor(m._split_mp, dtype = torch.long)

    reset_placement(SimpleNamespace(modules = [m]))

    fresh = split_module(num_experts = 9, first = 5, spares = 2)
    assert float(m._split_hist.abs().sum()) == 0.0
    assert m._split_pending == []
    assert m._split_busy == set()
    assert m._split_free_gpu == fresh._split_free_gpu == [3, 4]
    assert m._split_free_cpu == fresh._split_free_cpu == []
    # the slot mirror goes back to the map the load built, not to zero: the promote path reads it as
    # the physical slot of each expert, so zeros would claim every expert sits in worker slot 0
    assert m._split_mp == fresh._split_mp
    assert torch.equal(m._split_map, fresh._split_map)
    # rebuilt from the restored map on the next decode step
    assert ip.moe_cpu_promote_state is None


def test_an_unlanded_transfer_is_waited_for_even_when_the_map_is_unchanged():
    # _split_map only records transfers that have LANDED. A promotion or demotion the last arm issued
    # can still be queued while the map equals the load-time map, and the rebuild drops
    # _split_pending: that copy would then land in a slot the load-time map has just made live again.
    # So the quiesce cannot be conditional on the map having moved.
    class Event:
        def __init__(self):
            self.waited = 0

        def synchronize(self):
            self.waited += 1

    m = split_module(num_experts = 9, first = 5, spares = 2)
    ev = Event()
    m._split_pending = [dict(kind = "d", r = 3, gpu = 0, cpu = 0, ev = ev)]
    fresh = split_module(num_experts = 9, first = 5, spares = 2)
    assert m._split_map.tolist() == fresh._split_map.tolist()      # nothing has landed yet

    reset_placement(SimpleNamespace(modules = [m]))

    assert ev.waited == 1
    assert m._split_pending == []


def test_a_module_without_a_split_map_is_untouched():
    static = SimpleNamespace(device = torch.device("cpu"), num_experts = 8,
                             cpu_split_first = None, _split_map = None)
    absent = SimpleNamespace(device = torch.device("cpu"), num_experts = 8)
    before = (dict(vars(static)), dict(vars(absent)))

    assert reset_placement(SimpleNamespace(modules = [static, absent])) == 0

    assert (dict(vars(static)), dict(vars(absent))) == before
    assert not hasattr(absent, "_split_hist")
    assert not hasattr(absent, "_split_pending")


def test_the_runner_resets_between_arms():
    # The reset existing is not the fix: main() has to call it at the arm boundaries, and deleting
    # those calls once would leave every other test here green
    with open(RUNNER, "r") as f:
        tree = ast.parse(f.read())
    mains = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(mains) == 1, "main() is gone"
    calls = [c for c in ast.walk(mains[0])
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == "reset_placement"
             and c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == "model"]
    assert len(calls) >= 2, "main() must reset the measured model's placement between arms"
