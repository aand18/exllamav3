"""
KVarN Triton bootstrap tests (no GPU needed).

Covers what is coverable without hardware: module import, the availability
gate (must be False on a CUDA-less box even when opted in), loud failure of
the wrapper when unrunnable, and the default-off guarantee (the stock torch
path is untouched). Numerics and launch behavior of the kernels are
explicitly NOT covered here -- EXL3_KVARN_TRITON_PARITY=1 on a CUDA box is
the acceptance test (see kvarn_triton.py).
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
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

_afn = _stub("exllamav3.modules")
_afn.__path__ = [str(EXL / "modules")]
_attn_pkg = _stub("exllamav3.modules.attention_fn")
_attn_pkg.__path__ = [str(EXL / "modules" / "attention_fn")]
kt = _load("exllamav3.modules.attention_fn.kvarn_triton",
           "modules/attention_fn/kvarn_triton.py")


def test_triton_module_imports():
    assert hasattr(kt, "kvarn_triton_available")
    assert hasattr(kt, "kvarn_triton_dequant_side")
    assert hasattr(kt, "kvarn_triton_dequant_group")


def test_availability_false_without_cuda():
    # This box has no GPU: even opted in, the path must report unavailable
    # (never silently half-enable).
    old = os.environ.get("EXL3_KVARN_TRITON")
    os.environ["EXL3_KVARN_TRITON"] = "1"
    try:
        if not torch.cuda.is_available():
            assert kt.kvarn_triton_available() is False
    finally:
        if old is None:
            del os.environ["EXL3_KVARN_TRITON"]
        else:
            os.environ["EXL3_KVARN_TRITON"] = old


def test_wrapper_fails_loud_when_unavailable():
    if kt.kvarn_triton_available():
        pytest.skip("CUDA + triton present; loud-failure test N/A")
    pay = torch.zeros((1, 8192), dtype=torch.uint8)
    sc = torch.zeros((1, 128), dtype=torch.float16)
    with pytest.raises(RuntimeError):
        kt.kvarn_triton_dequant_side(pay, sc, sc, sc, 4)


def test_default_path_is_torch():
    # Default env (unset): the gate is off, so sealed-group reads use the
    # tested torch loop. Any regression here breaks the whole CPU suite,
    # which runs with this default.
    assert kvarn._kvarn_use_triton() is False or \
        os.environ.get("EXL3_KVARN_TRITON") == "1"
    if "EXL3_KVARN_TRITON" in os.environ:
        del os.environ["EXL3_KVARN_TRITON"]
    assert kvarn._kvarn_use_triton() is False
