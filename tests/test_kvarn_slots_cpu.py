"""
KVarN slot-remap CPU tests: windowed staging/exact bookkeeping
(no CUDA/Triton/ext).

Loads the real ``exllamav3.cache`` sources with stubbed parent packages so
the heavy ``exllamav3/__init__`` (model, compiled ext) is never executed.
Run with the CPU venv, e.g.::

    ./venv/bin/python -m pytest tests/test_kvarn_slots_cpu.py -q
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
EXL = ROOT / "exllamav3"


def _stub(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXL / rel)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    parent, _, attr = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], attr, m)
    spec.loader.exec_module(m)
    return m


_pkg = _stub("exllamav3")
_pkg.__path__ = [str(EXL)]
_cache_pkg = _stub("exllamav3.cache")
_cache_pkg.__path__ = [str(EXL / "cache")]

_constants = _load("exllamav3.constants", "constants.py")
_cache_mod = _load("exllamav3.cache.cache", "cache/cache.py")
CacheLayer = _cache_mod.CacheLayer

_fp16m = _stub("exllamav3.cache.fp16")
_fp16m.CacheLayer_fp16 = type("CacheLayer_fp16", (CacheLayer,), {})
_quantm = _stub("exllamav3.cache.quant")
_quantm.CacheLayer_quant = type("CacheLayer_quant", (CacheLayer,), {})

_qsa = _load("exllamav3.cache.qsa", "cache/qsa.py")
kvarn = _load("exllamav3.cache.kvarn", "cache/kvarn.py")


def _layer():
    attn = SimpleNamespace(num_kv_heads=2, head_dim=128,
                           qsa_indexer=None)
    # 8192 tokens -> 32 pages -> 64 groups: groups 5/9/60 used below
    # must be valid indices (a 512-token layer has only 4 groups).
    lay = kvarn.CacheLayer_kvarn(None, attn, 0, 8192, k_bits=4, v_bits=4)
    lay.alloc(torch.device("cpu"))
    return lay


def test_slot_assign_and_reuse():
    lay = _layer()
    s0 = lay._stage_slot(5)
    assert lay._stage_slot(5) == s0
    assert int(lay._stage_rev[5]) == s0
    s1 = lay._stage_slot(9)
    assert s1 != s0
    lay._stage_release(5)
    assert int(lay._stage_rev[5]) == -1
    s2 = lay._stage_slot(5)
    assert int(lay._stage_rev[5]) == s2


def test_slot_overflow_is_loud():
    lay = _layer()
    for g in range(kvarn.KVAR_N_STAGE_SLOTS):
        lay._stage_slot(g)
    try:
        lay._stage_slot(63)
    except AssertionError:
        return
    raise SystemExit("expected AssertionError on slot overflow")


def test_stage_roundtrip_through_slots():
    lay = _layer()
    g, s = 7, lay._stage_slot(7)
    lay.stage_k[s, 3].fill_(1.5)
    lay.present[g, 3] = True
    assert bool((lay.stage_k[lay._stage_slot(g), 3] == 1.5).all())
    lay._stage_release(g)
    assert lay._stage_rev[g] == -1


def test_exact_roundtrip_through_slots():
    lay = _layer()
    assert lay.exact_k.shape[0] == kvarn.KVAR_N_EXACT_SLOTS
    g, s = 11, lay._exact_slot(11)
    lay.exact_k[s, 9].fill_(2.5)
    lay.exact_valid[g] = True
    assert bool((lay.exact_k[lay._exact_slot(g), 9] == 2.5).all())
    lay._exact_release(g)
    assert lay._exact_rev[g] == -1


def _occupy_exact_sealed(lay, groups):
    for g in groups:
        lay._exact_slot(g)
        lay.sealed[g] = True
        lay.group_base[g] = g * kvarn.KVAR_N_GROUP
        lay.exact_valid[g] = True


def test_exact_pressure_evicts_dead():
    lay = _layer()
    assert lay.has_sink
    # All 6 slots: sink group 0 + sealed groups 1..5, every page owned
    # by a long-advanced sequence (windows far above).
    _occupy_exact_sealed(lay, range(6))
    lay.page_owner_n.fill_(5000)
    s = lay._exact_slot(60)
    lay.exact_valid[60] = True  # _alloc_exact_block does this in prod
    # Dead groups below-window are evicted (forced scan on pressure);
    # sink (base 0) is never a victim.
    assert int(lay._exact_rev[0]) >= 0
    assert int(lay._exact_rev[1]) == -1
    assert not bool(lay.exact_valid[1])
    assert int(lay._exact_rev[60]) == s
    # valid ⟺ assigned invariant holds everywhere.
    assert bool(((lay.exact_valid) == (lay._exact_rev >= 0)).all())


def test_exact_pressure_keeps_live_window():
    lay = _layer()
    _occupy_exact_sealed(lay, range(6))
    # Owner 700: live window [444, 700): groups 3..5 (bases 384..640)
    # intersect it, groups 1..2 are below it.
    lay.page_owner_n.fill_(700)
    s = lay._exact_slot(60)
    assert int(lay._exact_rev[1]) == -1  # oldest dead goes first
    assert int(lay._exact_rev[60]) == s
    # Everything still live keeps its slot.
    for g in (0, 3, 4, 5):
        assert int(lay._exact_rev[g]) >= 0


def test_exact_pressure_refreshes_live_owners():
    lay = _layer()
    assert lay.has_sink
    # Slots full: sink 0 + sealed groups 52..56 (bases 6656..7168).
    for g in [0, 52, 53, 54, 55, 56]:
        lay._exact_slot(g)
        lay.group_base[g] = g * kvarn.KVAR_N_GROUP
        lay.exact_valid[g] = True
        if g:
            lay.sealed[g] = True
    # Same-sequence owners, 127-stale (touch lag): group 52's base+128
    # (6784) sits above the stale drop line (6744), so a bare alloc
    # still overflows...
    lay.page_owner_n.fill_(7000)
    try:
        lay._exact_slot(61)
    except AssertionError:
        pass
    else:
        raise SystemExit("expected AssertionError without live context")
    # ...but with live context (true n=7127 on pages 26..28) the refresh
    # + forced scan frees group 52 and the alloc succeeds.
    live_bt = torch.tensor([26, 27, 28])
    s = lay._exact_slot(61, live_bt, 7127)
    assert int(lay._exact_rev[52]) == -1
    assert int(lay._exact_rev[61]) == s
    assert int(lay.page_owner_n[28]) == 7127


def test_exact_pressure_empty_means_loud():
    lay = _layer()
    _occupy_exact_sealed(lay, range(6))
    # Owner 500: live window [244, 500) covers every live group (plus
    # sink protection for group 0) -- nothing droppable, still loud.
    lay.page_owner_n.fill_(500)
    try:
        lay._exact_slot(60)
    except AssertionError:
        return
    raise SystemExit("expected AssertionError on unreclaimable overflow")


def test_stage_reclaim_seals_dead_partial():
    lay = _layer()
    assert lay.has_sink
    # All 4 staging slots: open groups 1..4 with rows, far below the
    # owners' windows (dead partial tails of finished sequences).
    for g in range(1, 5):
        lay._stage_slot(g)
        lay.group_base[g] = g * kvarn.KVAR_N_GROUP
        lay.present[g, :10] = True
    lay.page_owner_n.fill_(5000)
    s = lay._stage_slot(60)
    # Oldest dead partial (group 1) sealed; its slot recycled.
    assert bool(lay.sealed[1])
    assert int(lay._stage_rev[60]) == s
    # Sealed partial keeps its rows readable (present-gated).
    assert bool(lay.present[1, :10].all())


def test_below_windows_strict():
    lay = _layer()
    lay.page_owner_n.fill_(-1)
    lay.page_owner_n[0] = 14699
    # Stale owner vetoes by its own window (strict): base 14336 overlaps
    # [14443, 14699), so it is NOT reclaimable without a refresh -- this
    # is what the live-context refresh in _exact_slot is for.
    assert not bool(lay._below_all_windows(torch.tensor([14336]))[0])
    # Clearly below: reclaimable.
    assert bool(lay._below_all_windows(torch.tensor([14000]))[0])
    # Untouched pages vote nothing (no owners at all -> no victims).
    lay.page_owner_n.fill_(-1)
    assert not bool(lay._below_all_windows(torch.tensor([14000]))[0])
