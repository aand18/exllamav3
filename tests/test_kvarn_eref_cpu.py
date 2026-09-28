"""
KVarN Spec A CPU twin: incremental eref vs full WHT over the
evict/copy/reset matrix (no CUDA, Triton, or compiled ext required).

Loads the real ``exllamav3.cache`` sources with stubbed parent packages
(same pattern as test_kvarn_cpu.py). The incremental cache
(``kvarn_eref_cached`` + store/evict/copy slot hooks) must stay
bit-identical to the full ``kvarn_wht_head(exact_v)`` refresh after every
mutation: sequential appends (single + multi-row store paths), evict
scans, copy_page (full + partial + unwritten-source reset), and an
explicit page-reuse reset via _store_rows.
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

PAGE_SIZE = _constants.PAGE_SIZE


def _attn(kvh=2, hd=128):
    return SimpleNamespace(num_kv_heads=kvh, head_dim=hd)


def _layer(kvh=2, hd=128, ntok=1024, **kw):
    layer = kvarn.CacheLayer_kvarn(None, _attn(kvh, hd), 0, ntok, **kw)
    layer.alloc(torch.device("cpu"))
    return layer


def _ids(ntok, bsz=1, pages=None):
    pages = pages or (ntok + PAGE_SIZE - 1) // PAGE_SIZE
    return torch.arange(bsz * pages, dtype=torch.int32).view(bsz, pages)


def _assert_eref_matches_full(layer):
    Ew = layer.kvarn_eref_cached()
    ref = kvarn.kvarn_wht_head(layer.exact_v.float(), layer.head_dim)
    assert Ew.shape == ref.shape
    assert Ew.dtype == torch.float32
    assert torch.equal(Ew, ref)


@torch.inference_mode()
def test_eref_incremental_over_evict_copy_reset():
    torch.manual_seed(0)
    kvh, hd, ntok = 2, 128, 1024
    layer = _layer(kvh, hd, ntok)
    bt = _ids(ntok)
    # Sequential appends across single-row (fast path) and multi-row
    # (general loop) store paths, with seals + evict scans along the way.
    pos = 0
    for chunk in (1, 1, 5, 40, 127, 128, 64, 200, 140):
        if pos + chunk > 700:
            break
        k = torch.randn(1, chunk, kvh, hd).half()
        v = torch.randn(1, chunk, kvh, hd).half()
        layer.update_kv_direct(torch.tensor([pos], dtype=torch.int32),
                               bt, k, v, chunk)
        pos += chunk
        _assert_eref_matches_full(layer)
    # Force an evict scan (row-budgeted tick may not have fired yet).
    layer._evict_exact_all(128)
    _assert_eref_matches_full(layer)
    # copy_page matrix on a second layer: full sealed page, partial page,
    # and an unwritten-source reset (destination release path).
    dst = _layer(kvh, hd, ntok)
    dst.copy_page(layer, 0, 1, 256)
    _assert_eref_matches_full(dst)
    _assert_eref_matches_full(layer)  # source untouched
    dst.copy_page(layer, 1, 0, 44)
    _assert_eref_matches_full(dst)
    fresh = _layer(kvh, hd, ntok)
    dst.copy_page(fresh, 0, 0, 0)  # degenerate empty copy: no-op path
    _assert_eref_matches_full(dst)
    # Explicit page-reuse reset: rewrite group 0 (page 0, slots 0..127)
    # at a new logical base; the loop's reset branch releases + reassigns.
    rk = torch.randn(4, kvh, hd).half()
    rv = torch.randn(4, kvh, hd).half()
    pages = torch.zeros(4, dtype=torch.long)
    offs = torch.arange(4, dtype=torch.long)
    reset_pos = torch.full((4,), 768, dtype=torch.long) + torch.arange(4)
    layer._store_rows(rk, rv, pages, offs, reset_pos, 772, bt[0])
    _assert_eref_matches_full(layer)
