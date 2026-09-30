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


@torch.inference_mode()
def test_store_fast_matches_legacy():
    # Prefill fast-path twin: seal-direct-from-rows for complete fresh
    # non-sink groups must match the legacy per-group loop bit-exact
    # (records, sealed, present, group_base, exact blocks, eref,
    # owners, pinned). Kill switch EXL3_KVARN_FASTSTORE=0 forces legacy
    # on the twin. Aligned chunks take the fast path; sink/partial/
    # reset chunks stay legacy (asserted via _last_store_fast).
    import os as _os
    torch.manual_seed(7)
    kvh, hd, ntok = 2, 128, 1024
    bt = _ids(ntok)
    G = kvarn.KVAR_N_GROUP
    assert ntok // G >= 6, "need room for aligned chunks past sink"

    def run(env_on):
        _os.environ["EXL3_KVARN_FASTSTORE"] = "1" if env_on else "0"
        layer = _layer(kvh, hd, ntok)
        took = []
        pos = 0
        for chunk in (G, G, G, 40):
            k = torch.randn(1, chunk, kvh, hd).half()
            v = torch.randn(1, chunk, kvh, hd).half()
            layer.update_kv_direct(torch.tensor([pos], dtype=torch.int32),
                                   bt, k, v, chunk)
            pos += chunk
            took.append(bool(getattr(layer, "_last_store_fast", False)))
        return layer, took

    torch.manual_seed(7)
    fast, took_fast = run(True)
    torch.manual_seed(7)
    slow, took_slow = run(False)
    # Chunk 1 touches the sink group (base 0) -> legacy; chunk 2..3 are
    # complete fresh non-sink groups -> fast; the 40-row tail is partial
    # (2 full + 1 partial group... all fresh but incomplete) -> legacy.
    assert took_fast == [False, True, True, False], took_fast
    assert took_slow == [False, False, False, False], took_slow
    for name in ("records", "sealed", "present", "group_base",
                 "exact_k", "exact_v", "page_owner_n", "page_pinned"):
        a, b = getattr(fast, name), getattr(slow, name)
        assert torch.equal(a, b), name
    _assert_eref_matches_full(fast)
    # Reset case stays legacy on both and still matches.
    torch.manual_seed(11)
    rk = torch.randn(4, kvh, hd).half()
    rv = torch.randn(4, kvh, hd).half()
    pages = torch.zeros(4, dtype=torch.long)
    offs = torch.arange(4, dtype=torch.long)
    reset_pos = torch.full((4,), 768, dtype=torch.long) + torch.arange(4)
    _os.environ["EXL3_KVARN_FASTSTORE"] = "1"
    fast._store_rows(rk, rv, pages, offs, reset_pos, 772, bt[0])
    _os.environ["EXL3_KVARN_FASTSTORE"] = "0"
    slow._store_rows(rk, rv, pages, offs, reset_pos, 772, bt[0])
    assert not fast._last_store_fast and not slow._last_store_fast
    for name in ("records", "sealed", "present", "group_base",
                 "exact_k", "exact_v", "page_owner_n", "page_pinned"):
        a, b = getattr(fast, name), getattr(slow, name)
        assert torch.equal(a, b), name
    _assert_eref_matches_full(fast)
    _os.environ.pop("EXL3_KVARN_FASTSTORE", None)
