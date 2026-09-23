"""
KVarN M3 CPU tests: asymmetric widths incl. kvarn5/kvarn4 (no CUDA/Triton/ext).

Loads the real ``exllamav3.cache`` sources with stubbed parent packages so
the heavy ``exllamav3/__init__`` (model, compiled ext) is never executed.
Reference geometry mirrors beellama ``src/llama-kvarn.cpp`` (desc table
15-62, layout builder 576-613, pack/unpack 627-666). Run with the CPU
venv, e.g.::

    kvarn-venv\\Scripts\\python.exe -m pytest tests/test_kvarn_widths_cpu.py -x -q
"""

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

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


def _attn(kvh=2, hd=128, qsa_indexer=None, sliding_window=-1):
    return SimpleNamespace(num_kv_heads=kvh, head_dim=hd,
                           qsa_indexer=qsa_indexer,
                           sliding_window=sliding_window)


def _layer(kvh=2, hd=128, ntok=512, **kw):
    layer = kvarn.CacheLayer_kvarn(None, _attn(kvh, hd), 0, ntok, **kw)
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


# --------------------------------------------------------------------------
# Layout offsets vs Bee formula
# --------------------------------------------------------------------------

def test_layout_55_matches_bee_formula():
    L = kvarn.kvarn_make_layout(128, 128, 5, 5)
    exp = _bee_formula_offsets(5, 5)
    assert L.k_payload_bytes == exp["k_payload_bytes"] == 10240
    assert (L.k_payload_off, L.k_s_col_off, L.k_zp_off, L.k_s_row_off) == \
        (exp["k_payload_off"], exp["k_s_col_off"], exp["k_zp_off"], exp["k_s_row_off"]) == \
        (0, 10240, 10496, 10752)
    assert (L.v_payload_off, L.v_payload_bytes) == \
        (exp["v_payload_off"], exp["v_payload_bytes"]) == (11008, 10240)
    assert (L.v_s_col_off, L.v_s_row_off, L.v_zp_off) == \
        (exp["v_s_col_off"], exp["v_s_row_off"], exp["v_zp_off"]) == \
        (21248, 21504, 21760)
    assert L.tile_bytes == exp["tile_bytes"] == 22016
    assert L.k_record_bytes == exp["k_record_bytes"] == 11008
    assert L.v_record_bytes == exp["v_record_bytes"] == 11008
    assert L.k_record_bytes + L.v_record_bytes == L.tile_bytes


def test_layout_54_asymmetric_matches_bee_formula():
    L = kvarn.kvarn_make_layout(128, 128, 5, 4)
    exp = _bee_formula_offsets(5, 4)
    # K side identical to (5,5): same group boundaries, K payload width.
    L55 = kvarn.kvarn_make_layout(128, 128, 5, 5)
    assert L.k_payload_bytes == L55.k_payload_bytes == 10240
    assert (L.k_payload_off, L.k_s_col_off, L.k_zp_off, L.k_s_row_off) == \
        (L55.k_payload_off, L55.k_s_col_off, L55.k_zp_off, L55.k_s_row_off)
    # V side identical to (4,4) widths, shifted after the wider K side.
    assert (L.v_payload_off, L.v_payload_bytes) == \
        (exp["v_payload_off"], exp["v_payload_bytes"]) == (11008, 8192)
    assert (L.v_s_col_off, L.v_s_row_off, L.v_zp_off) == \
        (exp["v_s_col_off"], exp["v_s_row_off"], exp["v_zp_off"]) == \
        (19200, 19456, 19712)
    assert L.tile_bytes == exp["tile_bytes"] == 19968
    assert L.k_record_bytes == 11008 and L.v_record_bytes == 8960
    assert L.k_record_bytes + L.v_record_bytes == L.tile_bytes
    assert L.tile_bytes % 8 == 0


@torch.inference_mode()
def test_multislice_head_dim_256_asymmetric():
    """256-dim heads store 2 slice tiles each with the (5,4) geometry."""
    torch.manual_seed(31)
    layer = kvarn.CacheLayer_kvarn(None, _attn(1, 256), 0, 256,
                                   k_bits=5, v_bits=4)
    try:
        layer.alloc(torch.device("cpu"))
        assert layer.slices == 2
        assert layer.records.shape == (2, 2, 19968)
        bt = _ids(256)
        k = torch.randn(140, 1, 256).half()
        v = torch.randn(140, 1, 256).half()
        layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                               k.unsqueeze(0), v.unsqueeze(0), 140)
        assert not bool(layer.sealed[0]) and not bool(layer.sealed[1])
        kk, _ = layer.get_kv(torch.tensor([140], dtype=torch.int32), bt)
        got_k = kk[bt[0]].reshape(-1, 1, 256)[:140]
        assert _rmse(k.float(), got_k.float()) < 0.15
    finally:
        layer.free()


# --------------------------------------------------------------------------
# Pack/unpack incl. the 5-bit plane path (llama-kvarn.cpp:627-666)
# --------------------------------------------------------------------------

def test_pack_roundtrip_all_valid_bits():
    for bits in (2, 3, 4, 5, 6, 8):
        for n in (1, 7, 257):  # odd sizes exercise the pad-to-byte path
            mask = (1 << bits) - 1
            values = torch.tensor([(i * 7 + 3) & mask for i in range(n)],
                                  dtype=torch.uint8)
            packed = kvarn.kvarn_pack_bits(values, bits)
            assert packed.numel() == kvarn.kvarn_packed_bytes(n, bits), bits
            assert packed.numel() == (n * bits + 7) // 8, bits
            got = kvarn.kvarn_unpack_bits(packed, n, bits)
            assert torch.equal(got, values), (bits, n)


# --------------------------------------------------------------------------
# Tile RMSE: kvarn5 better than kvarn4, per-side independence
# --------------------------------------------------------------------------

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


def test_tile_rmse_ordering_and_cap():
    k, v = _bee_tiles()
    rk44, rv44 = _tile_rmse(k, v, 4, 4)
    rk55, rv55 = _tile_rmse(k, v, 5, 5)
    rk54, rv54 = _tile_rmse(k, v, 5, 4)
    # Absolute cap (Bee ballpark: kvarn4 < 0.12; 5-bit must clear 0.10).
    assert rk55 < 0.10 and rv55 < 0.10, (rk55, rv55)
    assert rk54 < 0.10 and rv54 < 0.10, (rk54, rv54)
    # Ordering: 5-bit K/V sides beat their 4-bit counterparts.
    assert rk55 < rk44, (rk55, rk44)
    assert rv55 < rv44, (rv55, rv44)
    # Per-side independence: K error depends only on K bits, V only on V.
    assert rk54 == rk55, (rk54, rk55)
    assert rv54 == rv44, (rv54, rv44)


def test_asymmetric_k_side_bytes_identical():
    """Same tile sealed under (5,5) vs (5,4): K side bytes are identical
    (same group boundaries, K payload width); only the V side differs."""
    k, v = _bee_tiles()
    L55 = kvarn.kvarn_make_layout(128, 128, 5, 5)
    L54 = kvarn.kvarn_make_layout(128, 128, 5, 4)
    rec55 = torch.zeros(L55.tile_bytes, dtype=torch.uint8)
    rec54 = torch.zeros(L54.tile_bytes, dtype=torch.uint8)
    kvarn.kvarn_quantize_k_tile(k, 16, 5, L55, rec55)
    kvarn.kvarn_quantize_v_tile(v, 16, 5, L55, rec55)
    kvarn.kvarn_quantize_k_tile(k, 16, 5, L54, rec54)
    kvarn.kvarn_quantize_v_tile(v, 16, 4, L54, rec54)
    assert torch.equal(rec55[:L54.k_record_bytes], rec54[:L54.k_record_bytes])
    assert L55.tile_bytes != L54.tile_bytes  # V payload width differs


# --------------------------------------------------------------------------
# Asymmetric end-to-end paged parity vs the fp16 overlay
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_asymmetric_paged_parity():
    torch.manual_seed(32)
    kvh, hd, ntok = 2, 128, 300
    layers = {
        bits: _layer(kvh, hd, 512, k_bits=bits[0], v_bits=bits[1])
        for bits in ((4, 4), (5, 5), (5, 4))
    }
    try:
        bt = _ids(512)
        k = torch.randn(ntok, kvh, hd).half()
        v = torch.randn(ntok, kvh, hd).half()
        got = {}
        for bits, layer in layers.items():
            layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                                   k.unsqueeze(0), v.unsqueeze(0), ntok)
            assert not bool(layer.sealed[0])   # permanent sink group
            assert bool(layer.sealed[1])        # K+V sealed together
            assert not bool(layer.sealed[2])
            rec = layer.get_kvarn_records()
            assert (rec["k_bits"], rec["v_bits"]) == bits
            kk, vv = layer.get_kv(torch.tensor([ntok], dtype=torch.int32), bt)
            got[bits] = (kk[bt[0]].reshape(-1, kvh, hd)[:ntok].float(),
                         vv[bt[0]].reshape(-1, kvh, hd)[:ntok].float())
        # Sink + tail rows are served exact from the overlay at any width.
        gk54 = got[(5, 4)][0]
        assert torch.equal(gk54[:128].half(), k[:128])
        assert torch.equal(gk54[ntok - 128:].half(), k[ntok - 128:])
        # Asymmetric body error sits between the symmetric presets.
        e44 = _rmse(k.float(), got[(4, 4)][0])
        e55 = _rmse(k.float(), got[(5, 5)][0])
        e54 = _rmse(k.float(), got[(5, 4)][0])
        assert e54 < 0.10, e54
        assert e55 <= e54 <= e44, (e55, e54, e44)
    finally:
        for layer in layers.values():
            layer.free()


@torch.inference_mode()
def test_asymmetric_sdpa_parity_explicit_mask():
    """Single-softmax SDPA over the merged image vs exact fp16.

    NOTE: uses an explicit bottom-right causal mask, not is_causal with
    q_len < kv_len (torch 2.14 CPU mis-aligns that case top-left, which
    would only attend exact sink rows and make the bound vacuous).
    """
    torch.manual_seed(33)
    kvh, qh, hd, past = 2, 4, 128, 300
    qq = torch.randn(1, 8, qh, hd).half()
    kq = torch.randn(1, 8, kvh, hd).half()
    vq = torch.randn(1, 8, kvh, hd).half()
    mask = torch.ones(8, past + 8, dtype=torch.bool) \
        .tril(past).unsqueeze(0).unsqueeze(0)
    errs = {}
    layers = {}
    try:
        for bits in ((4, 4), (5, 5), (5, 4)):
            layers[bits] = _layer(kvh, hd, 512,
                                  k_bits=bits[0], v_bits=bits[1])
        bt = _ids(512)
        torch.manual_seed(34)
        kp = torch.randn(past, kvh, hd).half()
        vp = torch.randn(past, kvh, hd).half()
        for bits, layer in layers.items():
            layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                                   kp.unsqueeze(0), vp.unsqueeze(0), past)
            kk, vv = layer.get_kv(torch.tensor([past], dtype=torch.int32), bt)
            kb = kk[bt[0]].reshape(-1, kvh, hd)[:past].float()
            vb = vv[bt[0]].reshape(-1, kvh, hd)[:past].float()
            qt = qq.transpose(1, 2).float()
            kf = torch.cat([kb.unsqueeze(0).half(), kq], dim=1) \
                .transpose(1, 2).float()
            vf = torch.cat([vb.unsqueeze(0).half(), vq], dim=1) \
                .transpose(1, 2).float()
            kr = torch.cat([kp.unsqueeze(0), kq], dim=1) \
                .transpose(1, 2).float()
            vr = torch.cat([vp.unsqueeze(0), vq], dim=1) \
                .transpose(1, 2).float()
            o_q = F.scaled_dot_product_attention(
                qt, kf, vf, attn_mask=mask, enable_gqa=True).transpose(1, 2)
            o_r = F.scaled_dot_product_attention(
                qt, kr, vr, attn_mask=mask, enable_gqa=True).transpose(1, 2)
            errs[bits] = _rmse(o_q, o_r)
        assert errs[(5, 4)] < 0.02, errs
        assert errs[(5, 5)] <= errs[(5, 4)] <= errs[(4, 4)], errs
    finally:
        for layer in layers.values():
            layer.free()


# --------------------------------------------------------------------------
# Storage accounting per width
# --------------------------------------------------------------------------

def test_storage_ordering_per_width():
    layers = {
        bits: _layer(4, 128, 4096, k_bits=bits[0], v_bits=bits[1])
        for bits in ((4, 4), (5, 5), (5, 4))
    }
    try:
        s44 = layers[(4, 4)].storage_size()
        s55 = layers[(5, 5)].storage_size()
        s54 = layers[(5, 4)].storage_size()
        assert s44 == 32 * 4 * 17920
        assert s55 == 32 * 4 * 22016
        assert s54 == 32 * 4 * 19968
        # Asymmetric records sit strictly between the symmetric presets.
        assert s44 < s54 < s55, (s44, s54, s55)
        fp16_bytes = 2 * 4096 * 4 * 128 * 2
        assert s55 < 0.5 * fp16_bytes
    finally:
        for layer in layers.values():
            layer.free()


# --------------------------------------------------------------------------
# CLI preset parsing (the -cq gate in model_init.py delegates here) +
# fail-closed construction
# --------------------------------------------------------------------------

def test_cli_preset_parse_accept():
    p = kvarn.kvarn_parse_preset
    assert p("kvarn4") == (4, 4)
    assert p("kvarn4,kvarn4") == (4, 4)
    assert p("kvarn5") == (5, 5)
    assert p("kvarn5,kvarn5") == (5, 5)
    assert p("kvarn5,kvarn4") == (5, 4)
    # Case-insensitive, '/' separator equivalent to ','.
    assert p("KVARN5") == (5, 5)
    assert p("Kvarn5/Kvarn4") == (5, 4)
    # Bare numeric pair restricted to the M3 set.
    assert p("5,4") == (5, 4)
    # M4: the full Bee 36-combo table parses (symmetric + asymmetric,
    # any separator, any case, bare numerics).
    bits = (2, 3, 4, 5, 6, 8)
    seen = set()
    for kb in bits:
        assert p(f"kvarn{kb}") == (kb, kb)
        assert p(f"{kb}") == (kb, kb)
        for vb in bits:
            assert p(f"kvarn{kb},kvarn{vb}") == (kb, vb)
            assert p(f"kvarn{kb}/kvarn{vb}") == (kb, vb)
            assert p(f"KVARN{kb},Kvarn{vb}") == (kb, vb)
            assert p(f"{kb},{vb}") == (kb, vb)
            seen.add((kb, vb))
    assert seen == kvarn.KVAR_N_SUPPORTED_PRESETS
    assert len(seen) == 36


def test_cli_preset_parse_reject():
    p = kvarn.kvarn_parse_preset
    for bad in ("kvarn7", "kvarn1", "kvarn0", "kvarn9", "kvarn4,kvarn7",
                "kvarn7,kvarn4", "kvarn", "kvarn5,kvarn4,kvarn4", "",
                "kvarn4,kvarn", "abc", "2,7", "7", "4,4,4", "kvarn4,4,4"):
        with pytest.raises(ValueError):
            p(bad)
    # The error names the supported width set.
    with pytest.raises(ValueError, match="36 combos"):
        p("kvarn7")
    # Non-kvarn numerics stay on the quant-cache path (no 'kvarn' prefix).
    assert not "4".startswith("kvarn") and not "4,4".startswith("kvarn")


def test_layer_construction_fail_closed():
    for bits in ((4, 7), (7, 4), (1, 4), (0, 4), (9, 9), (7, 7), (4, 1)):
        with pytest.raises(AssertionError):
            kvarn.CacheLayer_kvarn(None, _attn(2, 128), 0, 512,
                                   k_bits=bits[0], v_bits=bits[1])
    # M4: every Bee table pair constructs (spot-check alloc/free here;
    # the full 36-combo matrix is exercised in test_kvarn_m4_cpu.py).
    for bits in ((4, 4), (5, 5), (5, 4), (2, 2), (8, 8), (3, 6), (6, 3)):
        layer = kvarn.CacheLayer_kvarn(None, _attn(2, 128), 0, 512,
                                       k_bits=bits[0], v_bits=bits[1])
        layer.alloc(torch.device("cpu"))
        layer.free()


@torch.inference_mode()
def test_copy_page_width_mismatch_rejected():
    torch.manual_seed(35)
    src = _layer(1, 128, 256, k_bits=4, v_bits=4)
    dst = _layer(1, 128, 256, k_bits=5, v_bits=4)
    try:
        bt = _ids(256)
        k = torch.randn(128, 1, 128).half()
        v = torch.randn(128, 1, 128).half()
        src.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                             k.unsqueeze(0), v.unsqueeze(0), 128)
        with pytest.raises(AssertionError):
            dst.copy_page(src, 0, 0, 128)
    finally:
        src.free()
        dst.free()


# --------------------------------------------------------------------------
# QSA + tail unchanged across widths
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_qsa_asymmetric_planes_geometry_and_tail():
    torch.manual_seed(36)
    idx = SimpleNamespace(head_dim=32, compress_ratio=4)
    layer = kvarn.CacheLayer_kvarn_qsa(None, _attn(2, 128, idx), 0, 512,
                                       k_bits=5, v_bits=4,
                                       tail_tokens=200, tail_type="f16")
    try:
        assert layer.tail_effective == 256
        assert layer.has_sink
        layer.alloc(torch.device("cpu"))
        # QSA indexer planes stay fp16 whatever the K/V storage widths.
        assert layer.raw_k.shape == (2, 256, 32)
        assert layer.pooled.shape == (2, 64, 32)
        assert layer.raw_k.dtype == torch.half
        assert layer.records.shape[2] == 19968
        bt = _ids(512)
        k = torch.randn(300, 2, 128).half()
        v = torch.randn(300, 2, 128).half()
        layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                               k.unsqueeze(0), v.unsqueeze(0), 300)
        assert not bool(layer.sealed[0]) and bool(layer.sealed[1])
        kk, _ = layer.get_kv(torch.tensor([300], dtype=torch.int32), bt)
        got = kk[bt[0]].reshape(-1, 2, 128)[:300]
        assert torch.equal(got[:128], k[:128])            # sink exact
        assert torch.equal(got[300 - 256:], k[300 - 256:])  # tail exact
        d = layer.tp_export(None)
        assert d["cls"] is kvarn.CacheLayer_kvarn_qsa
        assert (d["args"]["k_bits"], d["args"]["v_bits"]) == (5, 4)
        rebuilt = kvarn.CacheLayer_kvarn_qsa(
            None, _attn(2, 128, idx), **d["args"])
        assert (rebuilt.k_bits, rebuilt.v_bits) == (5, 4)
        assert rebuilt.tail_effective == 256
    finally:
        layer.free()
