"""
KVarN M4 CPU tests: full 36-combo width table, SWA-window tail, compact
arenas (no CUDA/Triton/ext).

Loads the real ``exllamav3.cache`` sources with stubbed parent packages so
the heavy ``exllamav3/__init__`` (model, compiled ext) is never executed.
Reference geometry mirrors beellama ``src/llama-kvarn.cpp`` (desc table
15-62, layout builder 576-613) and ``src/llama-kv-cache-kvarn.cpp``
(SWA ring 611-634, SWA tail groups). Run with the CPU venv, e.g.::

    kvarn-venv\\Scripts\\python.exe -m pytest tests/test_kvarn_m4_cpu.py -x -q
"""

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
QSAPlanes = _qsa.QSAPlanes
kvarn = _load("exllamav3.cache.kvarn", "cache/kvarn.py")

PAGE_SIZE = _constants.PAGE_SIZE
VALID = (2, 3, 4, 5, 6, 8)


def _attn(kvh=2, hd=128, qsa_indexer=None, sliding_window=-1):
    return SimpleNamespace(num_kv_heads=kvh, head_dim=hd,
                           qsa_indexer=qsa_indexer,
                           sliding_window=sliding_window)


def _layer(kvh=2, hd=128, ntok=512, swa_window=-1, **kw):
    layer = kvarn.CacheLayer_kvarn(
        None, _attn(kvh, hd, sliding_window=swa_window), 0, ntok, **kw)
    layer.alloc(torch.device("cpu"))
    return layer


def _ids(ntok, bsz=1, pages=None):
    pages = pages or (ntok + PAGE_SIZE - 1) // PAGE_SIZE
    return torch.arange(bsz * pages, dtype=torch.int32).view(bsz, pages)


def _rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()))


def _bee_formula_offsets(key_bits, value_bits, head=128, group=128):
    """Independent re-derivation of llama_kvarn_make_layout (576-613)."""
    k_pay = (head * group * key_bits + 7) // 8
    v_pay = (group * head * value_bits + 7) // 8
    off = 0
    k_po = off
    off += k_pay
    k_sc, off2 = off, off + head * 2
    k_zp, off2 = off2, off2 + head * 2
    k_sr, off2 = off2, off2 + group * 2
    v_po = off2
    off2 += v_pay
    v_sc, off3 = off2, off2 + head * 2
    v_sr, off3 = off3, off3 + group * 2
    v_zp, off3 = off3, off3 + group * 2
    tile = (off3 + 7) // 8 * 8
    return {
        "k_payload_bytes": k_pay, "v_payload_bytes": v_pay,
        "k_payload_off": k_po, "k_s_col_off": k_sc, "k_zp_off": k_zp,
        "k_s_row_off": k_sr, "v_payload_off": v_po,
        "v_s_col_off": v_sc, "v_s_row_off": v_sr, "v_zp_off": v_zp,
        "tile_bytes": tile,
        "k_record_bytes": k_pay + 2 * head * 2 + group * 2,
        "v_record_bytes": v_pay + head * 2 + 2 * group * 2,
    }


def _bee_tiles():
    k = torch.empty(128, 128)
    v = torch.empty(128, 128)
    for r in range(128):
        for c in range(128):
            k[r, c] = math.sin(r * 0.071) + math.cos(c * 0.113) + ((r * 17 + c * 13) % 29 - 14) * 0.015
            v[r, c] = math.cos(r * 0.057) - math.sin(c * 0.091) + ((r * 11 + c * 19) % 31 - 15) * 0.012
    return k, v


def _tile_rmse(tile_k, tile_v, k_bits, v_bits):
    L = kvarn.kvarn_make_layout(128, 128, k_bits, v_bits)
    rec = torch.zeros(L.tile_bytes, dtype=torch.uint8)
    kvarn.kvarn_quantize_k_tile(tile_k, 16, k_bits, L, rec)
    kvarn.kvarn_quantize_v_tile(tile_v, 16, v_bits, L, rec)
    kd = kvarn.kvarn_dequantize_k_tile(rec, k_bits, L)
    vd = kvarn.kvarn_dequantize_v_tile(rec, v_bits, L)
    assert torch.isfinite(kd).all() and torch.isfinite(vd).all()
    return _rmse(tile_k, kd), _rmse(tile_v, vd)


# --------------------------------------------------------------------------
# M4.1: full 36-combo width table
# --------------------------------------------------------------------------

def test_width_table_all_36_layout():
    """Every Bee desc-table pair matches the layout formula (spot-checks
    for 2/3/6/8-bit sides included by construction: the loop covers all
    K,V in {2,3,4,5,6,8})."""
    seen = set()
    for kb in VALID:
        for vb in VALID:
            L = kvarn.kvarn_make_layout(128, 128, kb, vb)
            exp = _bee_formula_offsets(kb, vb)
            assert L.k_payload_bytes == exp["k_payload_bytes"] == (128 * 128 * kb + 7) // 8
            assert L.v_payload_bytes == exp["v_payload_bytes"] == (128 * 128 * vb + 7) // 8
            assert (L.k_payload_off, L.k_s_col_off, L.k_zp_off, L.k_s_row_off) == \
                (exp["k_payload_off"], exp["k_s_col_off"], exp["k_zp_off"], exp["k_s_row_off"])
            assert (L.v_payload_off, L.v_s_col_off, L.v_s_row_off, L.v_zp_off) == \
                (exp["v_payload_off"], exp["v_s_col_off"], exp["v_s_row_off"], exp["v_zp_off"])
            assert L.tile_bytes == exp["tile_bytes"]
            assert L.tile_bytes % 8 == 0
            assert L.k_record_bytes + L.v_record_bytes == L.tile_bytes
            seen.add((kb, vb))
    assert seen == kvarn.KVAR_N_SUPPORTED_PRESETS
    # Extreme corners (2/8-bit sides).
    assert kvarn.kvarn_make_layout(128, 128, 2, 2).tile_bytes == 9728
    assert kvarn.kvarn_make_layout(128, 128, 8, 8).tile_bytes == 34304
    assert kvarn.kvarn_make_layout(128, 128, 2, 8).tile_bytes == 22016
    assert kvarn.kvarn_make_layout(128, 128, 8, 2).tile_bytes == 22016
    assert kvarn.kvarn_make_layout(128, 128, 3, 6).tile_bytes == 19968
    assert kvarn.kvarn_make_layout(128, 128, 6, 3).tile_bytes == 19968


def test_width_rmse_monotonic_and_caps():
    """RMSE strictly decreases with bits per side; absolute caps."""
    k, v = _bee_tiles()
    sym = {}
    for b in VALID:
        rk, rv = _tile_rmse(k, v, b, b)
        sym[b] = (rk, rv)
    for i in range(len(VALID) - 1):
        lo, hi = VALID[i], VALID[i + 1]
        assert sym[hi][0] < sym[lo][0], (hi, lo, sym)
        assert sym[hi][1] < sym[lo][1], (hi, lo, sym)
    assert sym[8][0] < 0.01 and sym[8][1] < 0.01, sym[8]
    assert sym[2][0] < 0.30 and sym[2][1] < 0.30, sym[2]
    # Per-side independence across the wider table.
    rk_a, _ = _tile_rmse(k, v, 6, 2)
    rk_b, _ = _tile_rmse(k, v, 6, 8)
    assert rk_a == rk_b
    _, rv_a = _tile_rmse(k, v, 2, 3)
    _, rv_b = _tile_rmse(k, v, 8, 3)
    assert rv_a == rv_b


def test_preset_names_cover_table():
    """kvarnN + kvarnK,kvarnV names cover exactly the 36 combos."""
    assert set(kvarn.KVAR_N_PRESETS.values()) == kvarn.KVAR_N_SUPPORTED_PRESETS
    for kb in VALID:
        assert kvarn.KVAR_N_PRESETS[f"kvarn{kb}"] == (kb, kb)
        for vb in VALID:
            assert kvarn.KVAR_N_PRESETS[f"kvarn{kb},kvarn{vb}"] == (kb, vb)


@torch.inference_mode()
def test_all_36_construct_and_seal():
    """Every table pair constructs, seals one group, and round-trips."""
    torch.manual_seed(41)
    for kb in VALID:
        for vb in VALID:
            layer = kvarn.CacheLayer_kvarn(None, _attn(1, 128), 0, 256,
                                           k_bits=kb, v_bits=vb)
            try:
                layer.alloc(torch.device("cpu"))
                bt = _ids(256)
                kk = torch.randn(128, 1, 128).half()
                vv = torch.randn(128, 1, 128).half()
                layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                                       kk.unsqueeze(0), vv.unsqueeze(0), 128)
                # Single full group at base 0 is the sink: served exact.
                assert not bool(layer.sealed[0])
                got_k, _ = layer.get_kv(torch.tensor([128], dtype=torch.int32), bt)
                assert torch.equal(got_k[bt[0]].reshape(-1, 1, 128)[:128], kk)
            finally:
                layer.free()


# --------------------------------------------------------------------------
# M4.2: SWA-window tail
# --------------------------------------------------------------------------

def test_swa_helpers_unit():
    vis = kvarn.kvarn_swa_visible_groups
    ring = kvarn.kvarn_swa_ring_groups
    # Bee kvarn_swa_visible_groups: ((min(kv,n_swa)+127)/128)+1.
    assert vis(512, 512) == 5
    assert vis(4096, 512) == 5
    assert vis(100, 512) == 2
    assert vis(0, 512) == 1
    assert vis(4096, 0) == 33
    # Bee record ring: max(1, visible + ceil(ubatch/128) - 1).
    assert ring(512, 512, 256) == 6
    assert ring(4096, 512, 256) == 6
    assert ring(100, 512, 256) == 3
    assert ring(512, 512, 1) == 5
    # Tail window cap: min(size, n_swa), unknown window falls back.
    assert kvarn.kvarn_swa_tail_window(4096, 512) == 512
    assert kvarn.kvarn_swa_tail_window(256, 512) == 256
    assert kvarn.kvarn_swa_tail_window(4096, 0) == 4096


def test_swa_window_cap_policy():
    layer = kvarn.CacheLayer_kvarn(None, _attn(2, 128, sliding_window=512),
                                   0, 4096)
    try:
        assert layer.is_swa and not layer.has_sink
        assert layer.swa_window == 512
        assert layer.tail_window == 512       # capped, not max_num_tokens
        assert layer.tail_effective == 128    # intrinsic floor
        assert layer.swa_ring_groups == 6
        assert not layer.tail_native_exact
    finally:
        layer.free()

    layer = kvarn.CacheLayer_kvarn(None, _attn(2, 128, sliding_window=512),
                                   0, 4096, tail_tokens=300)
    try:
        assert layer.tail_effective == 384
        assert not layer.tail_native_exact
    finally:
        layer.free()

    # Full SWA-window request => native exact (nothing sealed).
    layer = kvarn.CacheLayer_kvarn(None, _attn(2, 128, sliding_window=512),
                                   0, 4096, tail_tokens=1024)
    try:
        assert layer.tail_window == 512
        assert layer.tail_effective == 512 and layer.tail_native_exact
    finally:
        layer.free()

    # Explicit SWA without a discoverable window: M2 full-window fallback.
    layer = kvarn.CacheLayer_kvarn(None, _attn(2, 128), 0, 512, is_swa=True)
    try:
        assert layer.is_swa and layer.swa_window == 0
        assert layer.tail_window == 512
    finally:
        layer.free()


@torch.inference_mode()
def test_swa_compact_no_sink_ring():
    """SWA: no sink (group 0 seals), residency under the ring bound."""
    torch.manual_seed(42)
    layer = _layer(2, 128, 1024, swa_window=512)
    try:
        assert layer.swa_ring_groups == 6
        bt = _ids(1024)
        k = torch.randn(600, 2, 128).half()
        v = torch.randn(600, 2, 128).half()
        layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                               k.unsqueeze(0), v.unsqueeze(0), 600)
        # No sink on SWA: completed groups 0..3 all sealed.
        assert bool(layer.sealed[0]) and bool(layer.sealed[3])
        assert not bool(layer.sealed[4])
        # Compact window [600-256, 600): groups 2,3,4 (+staging for open 4).
        assert sorted(layer.exact_blocks.keys()) == [2, 3, 4]
        assert layer._live_stage_groups() == [4]
        assert len(layer.exact_blocks) <= layer.swa_ring_groups
        kk, _ = layer.get_kv(torch.tensor([600], dtype=torch.int32), bt)
        got_k = kk[bt[0]].reshape(-1, 2, 128)[:600]
        assert torch.equal(got_k[600 - 128:].half(), k[600 - 128:])  # tail exact
        assert not torch.equal(got_k[:128], k[:128])                 # no sink
        assert _rmse(k.float(), got_k.float()) < 0.15
    finally:
        layer.free()


@torch.inference_mode()
def test_swa_full_window_native_exact():
    torch.manual_seed(43)
    layer = _layer(1, 128, 1024, swa_window=512, tail_tokens=1024)
    try:
        assert layer.tail_native_exact
        bt = _ids(1024)
        k = torch.randn(400, 1, 128).half()
        v = torch.randn(400, 1, 128).half()
        layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                               k.unsqueeze(0), v.unsqueeze(0), 400)
        assert not bool(layer.sealed.any())
        kk, vv = layer.get_kv(torch.tensor([400], dtype=torch.int32), bt)
        # n <= tail window: every row is inside the exact overlay.
        assert torch.equal(kk[bt[0]].reshape(-1, 1, 128)[:400], k)
        assert torch.equal(vv[bt[0]].reshape(-1, 1, 128)[:400], v)
        # Past the window, pre-window rows fall back to staging (WHT
        # roundtrip, ~1 ulp) while the window tail stays bit-exact.
        k2 = torch.randn(200, 1, 128).half()
        v2 = torch.randn(200, 1, 128).half()
        layer.update_kv_direct(torch.tensor([400], dtype=torch.int32), bt,
                               k2.unsqueeze(0), v2.unsqueeze(0), 200)
        assert not bool(layer.sealed.any())
        kk, _ = layer.get_kv(torch.tensor([600], dtype=torch.int32), bt)
        got = kk[bt[0]].reshape(-1, 1, 128)[:600]
        assert torch.equal(got[88:].half(), torch.cat([k, k2])[88:])
        assert _rmse(got[:88].float(), k[:88].float()) < 0.005
    finally:
        layer.free()


# --------------------------------------------------------------------------
# M4.3: compact tail arenas (memory win + exactness at 4K/8K)
# --------------------------------------------------------------------------

def _write_full(layer, ntok, kvh, hd, seed, chunk=1024):
    torch.manual_seed(seed)
    pages = ntok // PAGE_SIZE
    bt = torch.arange(pages, dtype=torch.int32).view(1, pages)
    n = 0
    all_k, all_v = [], []
    while n < ntok:
        c = min(chunk, ntok - n)
        kk = torch.randn(c, kvh, hd).half()
        vv = torch.randn(c, kvh, hd).half()
        layer.update_kv_direct(torch.tensor([n], dtype=torch.int32), bt,
                               kk.unsqueeze(0), vv.unsqueeze(0), c)
        all_k.append(kk)
        all_v.append(vv)
        n += c
    return bt, torch.cat(all_k), torch.cat(all_v)


@torch.inference_mode()
def test_compact_memory_4k():
    """Total (records + compact tail + staging) << fp16 at 4K."""
    for bits in ((4, 4), (5, 4)):
        layer = _layer(2, 128, 4096, k_bits=bits[0], v_bits=bits[1])
        try:
            bt, K, V = _write_full(layer, 4096, 2, 128, seed=44)
            fp16_bytes = 2 * 4096 * 2 * 128 * 2
            total = layer.storage_size() + layer.overhead_size()
            assert total < 0.5 * fp16_bytes, (bits, total, fp16_bytes)
            # Resident set: sink group 0 + trailing N+R window [3840,4096).
            assert sorted(layer.exact_blocks.keys()) == [0, 30, 31], \
                (bits, sorted(layer.exact_blocks.keys()))
            assert layer._live_stage_groups() == [0], \
                (bits, layer._live_stage_groups())
            assert int(layer.sealed.sum()) == 31  # all but the sink
            rec = layer.get_kvarn_records()
            assert rec["swa_window"] == 0 and rec["tail_window"] == 4096
            # Sink + tail served bit-exact from the compact blocks.
            kk, _ = layer.get_kv(torch.tensor([4096], dtype=torch.int32), bt)
            got = kk[bt[0]].reshape(-1, 2, 128)
            assert torch.equal(got[:128].half(), K[:128])
            assert torch.equal(got[4096 - 128:].half(), K[4096 - 128:])
            body = got[128:4096 - 128].float()
            assert _rmse(body, K[128:4096 - 128].float()) < 0.10
        finally:
            layer.free()


@torch.inference_mode()
def test_compact_memory_8k():
    """Total << fp16 at 8K for (4,4) and (5,4)."""
    for bits in ((4, 4), (5, 4)):
        layer = _layer(2, 128, 8192, k_bits=bits[0], v_bits=bits[1])
        try:
            bt, K, V = _write_full(layer, 8192, 2, 128, seed=45)
            fp16_bytes = 2 * 8192 * 2 * 128 * 2
            total = layer.storage_size() + layer.overhead_size()
            assert total < 0.5 * fp16_bytes, (bits, total, fp16_bytes)
            assert sorted(layer.exact_blocks.keys()) == [0, 62, 63], \
                (bits, sorted(layer.exact_blocks.keys()))
            assert layer._live_stage_groups() == [0]
            assert int(layer.sealed.sum()) == 63
            kk, _ = layer.get_kv(torch.tensor([8192], dtype=torch.int32), bt)
            got = kk[bt[0]].reshape(-1, 2, 128)
            assert torch.equal(got[:128].half(), K[:128])
            assert torch.equal(got[8192 - 128:].half(), K[8192 - 128:])
            assert _rmse(got.float(), K.float()) < 0.12
        finally:
            layer.free()


@torch.inference_mode()
def test_page_reuse_rewrite_reseals():
    """Overwriting a sealed group unseals + reseals it: records track."""
    torch.manual_seed(46)
    layer = _layer(1, 128, 512)
    try:
        bt = _ids(512)
        k1 = torch.randn(256, 1, 128).half()
        v1 = torch.randn(256, 1, 128).half()
        layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                               k1.unsqueeze(0), v1.unsqueeze(0), 256)
        assert bool(layer.sealed[1])
        rec_before = layer.records[1].clone()
        # Overwrite the sealed body group with new content.
        k2 = torch.randn(128, 1, 128).half()
        v2 = torch.randn(128, 1, 128).half()
        layer.update_kv_direct(torch.tensor([128], dtype=torch.int32), bt,
                               k2.unsqueeze(0), v2.unsqueeze(0), 128)
        assert bool(layer.sealed[1])  # resealed fresh
        assert not torch.equal(layer.records[1], rec_before)
        kk, _ = layer.get_kv(torch.tensor([256], dtype=torch.int32), bt)
        got = kk[bt[0]].reshape(-1, 1, 128)[:256]
        assert torch.equal(got[:128], k1[:128])    # sink untouched
        assert torch.equal(got[128:], k2)          # new content (tail)
    finally:
        layer.free()


@torch.inference_mode()
def test_remapped_reuse_resets_base():
    """A physical group reused under a new logical base resets (no stale
    seal/present), and seals as body when it is not the sink."""
    torch.manual_seed(47)
    layer = _layer(1, 128, 512)
    try:
        bt = _ids(512)
        k1 = torch.randn(256, 1, 128).half()
        v1 = torch.randn(256, 1, 128).half()
        layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                               k1.unsqueeze(0), v1.unsqueeze(0), 256)
        assert int(layer.group_base[0]) == 0
        # Simulate page reuse under a remapped logical range: same physical
        # page now backs positions 256..383 (group bases 256/384).
        k2 = torch.randn(128, 1, 128).half()
        v2 = torch.randn(128, 1, 128).half()
        layer._store_rows(k2, v2,
                          torch.zeros(128, dtype=torch.long),
                          torch.arange(128, dtype=torch.long),
                          torch.arange(256, 384, dtype=torch.long), 384)
        assert int(layer.group_base[0]) == 256
        assert bool(layer.present[0].all())
        assert bool(layer.sealed[0])  # base != 0: body, seals
        assert 0 not in layer._live_stage_groups()  # staging freed on seal
    finally:
        layer.free()
