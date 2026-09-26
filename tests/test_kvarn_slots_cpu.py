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
