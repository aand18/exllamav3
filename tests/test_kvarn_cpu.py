"""
KVarN M1 CPU tests (no CUDA, Triton, or compiled ext required).

Loads the real ``exllamav3.cache`` / ``attention_fn`` sources with stubbed
parent packages so the heavy ``exllamav3/__init__`` (model, compiled ext)
is never executed. Reference vectors mirror beellama ``tests/test-kvarn.cpp``.
Run with the CPU venv, e.g.::

    kvarn-venv\\Scripts\\python.exe -m pytest tests/test_kvarn_cpu.py -x -q
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

# Fake fp16/quant bases so the real qsa.py loads without the compiled ext.
# (Only the real QSAPlanes is used below; the fakes are never instantiated.)
_fp16m = _stub("exllamav3.cache.fp16")
_fp16m.CacheLayer_fp16 = type("CacheLayer_fp16", (CacheLayer,), {})
_quantm = _stub("exllamav3.cache.quant")
_quantm.CacheLayer_quant = type("CacheLayer_quant", (CacheLayer,), {})

_qsa = _load("exllamav3.cache.qsa", "cache/qsa.py")
QSAPlanes = _qsa.QSAPlanes
kvarn = _load("exllamav3.cache.kvarn", "cache/kvarn.py")

PAGE_SIZE = _constants.PAGE_SIZE


def _attn(kvh=2, hd=128, qsa_indexer=None):
    return SimpleNamespace(num_kv_heads=kvh, head_dim=hd, qsa_indexer=qsa_indexer)


def _layer(kvh=2, hd=128, ntok=512, **kw):
    layer = kvarn.CacheLayer_kvarn(None, _attn(kvh, hd), 0, ntok, **kw)
    layer.alloc(torch.device("cpu"))
    return layer


def _ids(ntok, bsz=1, pages=None):
    pages = pages or (ntok + PAGE_SIZE - 1) // PAGE_SIZE
    bt = torch.arange(bsz * pages, dtype=torch.int32).view(bsz, pages)
    return bt


# --------------------------------------------------------------------------
# Layout / packing (llama-kvarn.cpp:576-666)
# --------------------------------------------------------------------------

def test_layout_matches_bee():
    L = kvarn.kvarn_make_layout(128, 128, 4, 4)
    assert L.k_payload_bytes == 8192
    assert (L.k_payload_off, L.k_s_col_off, L.k_zp_off, L.k_s_row_off) == (0, 8192, 8448, 8704)
    assert (L.v_payload_off, L.v_payload_bytes) == (8960, 8192)
    assert (L.v_s_col_off, L.v_s_row_off, L.v_zp_off) == (17152, 17408, 17664)
    assert L.tile_bytes == 17920
    assert L.tile_bytes % 8 == 0


def test_pack_roundtrip():
    for bits in (2, 4, 8):
        n = 257
        mask = (1 << bits) - 1
        values = torch.tensor([(i * 7 + 3) & mask for i in range(n)], dtype=torch.uint8)
        packed = kvarn.kvarn_pack_bits(values, bits)
        assert packed.numel() == kvarn.kvarn_packed_bytes(n, bits)
        got = kvarn.kvarn_unpack_bits(packed, n, bits)
        assert torch.equal(got, values), bits


# --------------------------------------------------------------------------
# Hadamard (llama-kvarn.cpp:668-686, test-kvarn.cpp Hadamard roundtrip)
# --------------------------------------------------------------------------

def test_hadamard_roundtrip():
    v = torch.tensor([math.sin(i * 0.19) + (i - 64) * 0.002 for i in range(128)],
                     dtype=torch.float32)
    w = kvarn.kvarn_hadamard_128(kvarn.kvarn_hadamard_128(v))
    assert torch.allclose(w, v, atol=1e-5, rtol=0)


def test_wht_head_involution():
    torch.manual_seed(0)
    for hd in (128, 256, 512):
        x = torch.randn(3, 5, hd)
        y = kvarn.kvarn_wht_head(kvarn.kvarn_wht_head(x, hd), hd)
        assert torch.allclose(y, x, atol=1e-4, rtol=0), hd


def test_head_slices_fail_closed():
    assert kvarn.kvarn_head_slices(128) == 1
    assert kvarn.kvarn_head_slices(256) == 2
    assert kvarn.kvarn_head_slices(512) == 4
    assert kvarn.kvarn_head_slices(64) == 0
    with pytest.raises(AssertionError):
        _layer(kvh=2, hd=64)


# --------------------------------------------------------------------------
# Tile quantization (test-kvarn.cpp:837-874, max_rmse[4] = 0.12)
# --------------------------------------------------------------------------

def _bee_tiles():
    k = torch.empty(128, 128)
    v = torch.empty(128, 128)
    for r in range(128):
        for c in range(128):
            k[r, c] = math.sin(r * 0.071) + math.cos(c * 0.113) + ((r * 17 + c * 13) % 29 - 14) * 0.015
            v[r, c] = math.cos(r * 0.057) - math.sin(c * 0.091) + ((r * 11 + c * 19) % 31 - 15) * 0.012
    return k, v


def _rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()))


def test_tile_rmse_kvarn4():
    k, v = _bee_tiles()
    L = kvarn.kvarn_make_layout(128, 128, 4, 4)
    rec = torch.zeros(L.tile_bytes, dtype=torch.uint8)
    kvarn.kvarn_quantize_k_tile(k, 16, 4, L, rec)
    kvarn.kvarn_quantize_v_tile(v, 16, 4, L, rec)
    kd = kvarn.kvarn_dequantize_k_tile(rec, 4, L)
    vd = kvarn.kvarn_dequantize_v_tile(rec, 4, L)
    assert torch.isfinite(kd).all() and torch.isfinite(vd).all()
    assert _rmse(k, kd) < 0.12, _rmse(k, kd)
    assert _rmse(v, vd) < 0.12, _rmse(v, vd)


def test_rotated_domain_equivalence():
    """Port of test-kvarn.cpp test_rotated_domain_equivalence (kvarn4)."""
    k, v = _bee_tiles()
    L = kvarn.kvarn_make_layout(128, 128, 4, 4)
    rec = torch.zeros(L.tile_bytes, dtype=torch.uint8)
    kvarn.kvarn_quantize_k_tile(k, 16, 4, L, rec)
    kvarn.kvarn_quantize_v_tile(v, 16, 4, L, rec)
    k_rot = kvarn.kvarn_dequantize_k_tile(rec, 4, L)  # [dim, token]
    v_rot = kvarn.kvarn_dequantize_v_tile(rec, 4, L)  # [token, dim]

    q = torch.tensor([math.sin(d * 0.037) + 0.25 * math.cos(d * 0.0131) for d in range(128)])
    rq = kvarn.kvarn_hadamard_128(q)
    k_max_abs = k_max_diff = 0.0
    for c in range(128):
        kcol = k_rot[:, c]
        korig = kvarn.kvarn_hadamard_128(kcol)
        ref = float((q * korig).sum())
        rot = float((rq * kcol).sum())
        k_max_abs = max(k_max_abs, abs(ref))
        k_max_diff = max(k_max_diff, abs(ref - rot))
    assert k_max_diff < 1e-3 * (1.0 + k_max_abs)

    w = torch.tensor([0.5 + 0.5 * math.sin(t * 0.083) + 0.01 * t for t in range(128)])
    w = w / w.sum()
    ref_o = torch.zeros(128)
    for t in range(128):
        ref_o += w[t] * kvarn.kvarn_hadamard_128(v_rot[t])
    o_rot = (w.unsqueeze(1) * v_rot).sum(dim=0)
    o_rot = kvarn.kvarn_hadamard_128(o_rot)
    v_max_abs = float(ref_o.abs().max())
    v_max_diff = float((ref_o - o_rot).abs().max())
    assert v_max_diff < 1e-3 * (1.0 + v_max_abs)


# --------------------------------------------------------------------------
# Cache layer: paging, sealing, copy_page, storage accounting
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_paging_roundtrip_and_seal():
    torch.manual_seed(1)
    kvh, hd, ntok = 2, 128, 300
    layer = _layer(kvh, hd, 512)
    bt = _ids(512)
    k = torch.randn(ntok, kvh, hd).half()
    v = torch.randn(ntok, kvh, hd).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt, k.unsqueeze(0), v.unsqueeze(0), ntok)
    assert not bool(layer.sealed[0])           # M2: group 0 is the permanent sink
    assert bool(layer.sealed[1])               # group 1 sealed (256 tokens)
    assert not bool(layer.sealed[2])             # group 2 partial -> fp16 staging
    assert not bool(layer.sealed[3])
    kk, vv = layer.get_kv(torch.tensor([ntok], dtype=torch.int32), bt)
    got_k = kk[bt[0]].reshape(-1, kvh, hd)[:ntok]
    got_v = vv[bt[0]].reshape(-1, kvh, hd)[:ntok]
    assert _rmse(k.float(), got_k.float()) < 0.15
    assert _rmse(v.float(), got_v.float()) < 0.15


@torch.inference_mode()
def test_multislice_head_dim_256():
    torch.manual_seed(9)
    kvh, hd, ntok = 1, 256, 140
    layer = _layer(kvh, hd, 256)
    bt = _ids(256)
    k = torch.randn(ntok, kvh, hd).half()
    v = torch.randn(ntok, kvh, hd).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt, k.unsqueeze(0), v.unsqueeze(0), ntok)
    assert not bool(layer.sealed[0]) and not bool(layer.sealed[1])  # M2: group 0 sink
    assert layer.records.shape == (2, 2, 17920)  # 2 slices per head
    kk, vv = layer.get_kv(torch.tensor([ntok], dtype=torch.int32), bt)
    got_k = kk[bt[0]].reshape(-1, kvh, hd)[:ntok]
    assert _rmse(k.float(), got_k.float()) < 0.15


@torch.inference_mode()
def test_update_kv_persists_from_dequant_temps():
    """get_kv -> merge rows (as the attn fallback does) -> update_kv roundtrip."""
    torch.manual_seed(2)
    kvh, hd = 1, 128
    layer = _layer(kvh, hd, 256)
    bt = _ids(256)
    seqlens = torch.zeros(1, dtype=torch.int32)
    k, v = layer.get_kv(seqlens, bt)
    nk = torch.randn(1, 40, kvh, hd).half()
    nv = torch.randn(1, 40, kvh, hd).half()
    k[0, :40] = nk[0]
    v[0, :40] = nv[0]
    layer.update_kv(seqlens, bt, k, v, 40)
    seqlens2 = torch.tensor([40], dtype=torch.int32)
    k2, v2 = layer.get_kv(seqlens2, bt)
    assert _rmse(nk[0].float(), k2[0, :40].reshape(40, kvh, hd).float()) < 0.15


@torch.inference_mode()
def test_copy_page():
    torch.manual_seed(3)
    layer = _layer(2, 128, 512)
    bt = _ids(512)
    k = torch.randn(300, 2, 128).half()
    v = torch.randn(300, 2, 128).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt, k.unsqueeze(0), v.unsqueeze(0), 300)

    dst = _layer(2, 128, 512)
    dst.copy_page(layer, 0, 1, 256)   # full page: sealed group travels, sink stays unsealed
    assert not bool(dst.sealed[2])    # M2: sink group never sealed
    assert bool(dst.sealed[3])
    assert torch.equal(dst.records[3], layer.records[1])
    # M2: the exact overlay is logical (per block_table), so compare the
    # shared page through a remapped block table, not raw page tensors.
    bt2 = bt.clone()
    bt2[0, 0] = 1
    kk, _ = dst.get_kv(torch.tensor([300], dtype=torch.int32), bt2)
    ref, _ = layer.get_kv(torch.tensor([300], dtype=torch.int32), bt)
    # Only the copied 256-token span is comparable (the copy overwrote dst
    # page 1, which used to back positions 256..299).
    assert torch.equal(kk[bt2[0]].reshape(-1, 2, 128)[:256],
                       ref[bt[0]].reshape(-1, 2, 128)[:256])

    dst2 = _layer(2, 128, 512)
    dst2.copy_page(layer, 1, 0, 44)   # partial page: staging travels unsealed
    assert not bool(dst2.sealed[0])
    assert torch.equal(dst2.stage_k[0, :44], layer.stage_k[1, :44])


def test_storage_size_beats_fp16_and_quant():
    layer = _layer(4, 128, 4096)
    fp16_bytes = 2 * 4096 * 4 * 128 * 2
    # quant4 reference geometry: token_dim//32*k_bits int32-equivalent bytes/token approx
    got = layer.storage_size()
    assert got < 0.5 * fp16_bytes, (got, fp16_bytes)
    # records dominate: 32 groups * 4 heads * 17920 B
    assert got == 32 * 4 * 17920
    assert layer.overhead_size() > 0  # fp16 staging + exact tail buffers
    assert len(layer.get_tensors()) == 5


def test_tp_export():
    layer = _layer(2, 128, 512)
    d = layer.tp_export(None)
    assert d["cls"] is kvarn.CacheLayer_kvarn
    assert d["args"] == {"cache_id": 0, "max_num_tokens": 512, "k_bits": 4, "v_bits": 4,
                         "tail_tokens": 0, "tail_type": "f16", "is_swa": False}
    q = kvarn.CacheLayer_kvarn_qsa(None, _attn(2, 128,
        SimpleNamespace(head_dim=32, compress_ratio=4)), 7, 512)
    dq = q.tp_export(None)
    assert dq["cls"] is kvarn.CacheLayer_kvarn_qsa
    assert dq["args"]["cache_id"] == 7


def test_qsa_planes_geometry():
    idx = SimpleNamespace(head_dim=32, compress_ratio=4)
    layer = kvarn.CacheLayer_kvarn_qsa(None, _attn(2, 128, idx), 0, 512)
    assert isinstance(layer, QSAPlanes)
    assert isinstance(layer, kvarn.CacheLayer_kvarn)
    layer.alloc(torch.device("cpu"))
    assert layer.raw_k.shape == (2, 256, 32)
    assert layer.pooled.shape == (2, 64, 32)
    assert layer.raw_k.dtype == torch.half
    ref = kvarn.CacheLayer_kvarn(None, _attn(2, 128), 0, 512)
    ref.alloc(torch.device("cpu"))
    assert layer.storage_size() > ref.storage_size()  # planes cost extra
    layer.free()
    assert layer.raw_k is None and layer.records is None


@torch.inference_mode()
def test_qsa_copy_page_carries_planes():
    idx = SimpleNamespace(head_dim=32, compress_ratio=4)
    a = _attn(1, 128, idx)
    s = kvarn.CacheLayer_kvarn_qsa(None, a, 0, 512)
    d = kvarn.CacheLayer_kvarn_qsa(None, a, 0, 512)
    s.alloc(torch.device("cpu"))
    d.alloc(torch.device("cpu"))
    s.raw_k[0, :10] = 1.0
    d.copy_page(s, 0, 1, 128)
    assert bool((d.raw_k[1, :10] == 1.0).all())
    assert bool((d.raw_k[1, 10:128] == 0.0).all())


# --------------------------------------------------------------------------
# Attention equivalence: kvarn dequant + SDPA vs fp16 (manual gather)
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_sdpa_close_to_fp16():
    torch.manual_seed(5)
    kvh, qh, hd = 2, 4, 128
    layer = _layer(kvh, hd, 256)
    bt = _ids(256)
    past = 200
    kp = torch.randn(1, past, kvh, hd).half()
    vp = torch.randn(1, past, kvh, hd).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt, kp, vp, past)
    kk, vv = layer.get_kv(torch.tensor([past], dtype=torch.int32), bt)
    k_buf = kk[bt[0, :1]].reshape(-1, kvh, hd)[:past].float()
    v_buf = vv[bt[0, :1]].reshape(-1, kvh, hd)[:past].float()

    q = torch.randn(1, 8, qh, hd).half()
    kq = torch.randn(1, 8, kvh, hd).half()
    vq = torch.randn(1, 8, kvh, hd).half()
    k_full = torch.cat([k_buf.unsqueeze(0).half(), kq], dim=1)
    v_full = torch.cat([v_buf.unsqueeze(0).half(), vq], dim=1)
    o_q = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k_full.transpose(1, 2).float(),
        v_full.transpose(1, 2).float(), is_causal=True, enable_gqa=True).transpose(1, 2)
    o_ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        torch.cat([kp, kq], dim=1).transpose(1, 2).float(),
        torch.cat([vp, vq], dim=1).transpose(1, 2).float(),
        is_causal=True, enable_gqa=True).transpose(1, 2)
    # 4-bit KVarN with sink + 128 tail floor: output error is quantization-dominated.
    assert _rmse(o_q, o_ref) < 0.12, _rmse(o_q, o_ref)


# --------------------------------------------------------------------------
# Dispatch end-to-end on CPU (real dispatch, stubbed ext/model/triton)
# --------------------------------------------------------------------------

def _load_dispatch():
    if "exllamav3.modules.attention_fn.dispatch" in sys.modules:
        import importlib as _il
        return _il.import_module("exllamav3.modules.attention_fn.dispatch")
    import numpy  # noqa: F401  (fp16/quant/mla use numpy)
    _stub("exllamav3.model").Config = object
    _extm = _stub("exllamav3.ext")
    _extm.exllamav3_ext = SimpleNamespace()
    _stub("exllamav3.util").__path__ = []
    _stub("exllamav3.util.memory").malloc_trim = lambda *a, **k: None
    # Real cache submodules under their canonical names (ext/model stubs above
    # satisfy fp16/quant imports), then exec the real cache/__init__ so that
    # `from ...cache import ...` in dispatch resolves package attributes.
    for name, rel in [
        ("exllamav3.cache.fp16", "cache/fp16.py"),
        ("exllamav3.cache.quant", "cache/quant.py"),
        ("exllamav3.cache.mla", "cache/mla.py"),
        ("exllamav3.cache.recurrent", "cache/recurrent.py"),
        ("exllamav3.cache.dsa", "cache/dsa.py"),
        ("exllamav3.cache.qsa", "cache/qsa.py"),
        ("exllamav3.cache.kvarn", "cache/kvarn.py"),
    ]:
        sys.modules.pop(name, None)
        _load(name, rel)
    spec = importlib.util.spec_from_file_location(
        "exllamav3.cache", EXL / "cache" / "__init__.py")
    spec.loader.exec_module(sys.modules["exllamav3.cache"])
    _stub("exllamav3.modules").__path__ = [str(EXL / "modules")]
    _afn = _stub("exllamav3.modules.attention_fn")
    _afn.__path__ = [str(EXL / "modules" / "attention_fn")]
    for name, rel in [
        ("exllamav3.modules.attention_fn.common", "modules/attention_fn/common.py"),
        ("exllamav3.modules.attention_fn.bighead_scalar", "modules/attention_fn/bighead_scalar.py"),
        ("exllamav3.modules.attention_fn.torch", "modules/attention_fn/torch.py"),
        ("exllamav3.modules.attention_fn.xformers", "modules/attention_fn/xformers.py"),
        ("exllamav3.modules.attention_fn.dispatch", "modules/attention_fn/dispatch.py"),
    ]:
        _load(name, rel)
    global kvarn
    kvarn = sys.modules["exllamav3.cache.kvarn"]
    return sys.modules["exllamav3.modules.attention_fn.dispatch"]


@torch.inference_mode()
def test_dispatch_kvarn_vs_fp16_cpu():
    dispatch = _load_dispatch()
    fp16_cls = sys.modules["exllamav3.cache.fp16"].CacheLayer_fp16
    torch.manual_seed(7)
    kvh, qh, hd = 2, 4, 128
    a = _attn(kvh, hd)
    kl = kvarn.CacheLayer_kvarn(None, a, 0, 512)
    fl = fp16_cls(None, a, 0, 512)
    kl.alloc(torch.device("cpu"))
    fl.alloc(torch.device("cpu"))
    bt = _ids(512)
    for step, n in ((0, 100), (100, 1), (101, 5)):
        q = torch.randn(1, n, qh, hd).half()
        k = torch.randn(1, n, kvh, hd).half()
        v = torch.randn(1, n, kvh, hd).half()
        seqlens = torch.tensor([step], dtype=torch.int32)
        o_k = dispatch.attn_dispatch(q.clone(), k.clone(), v.clone(), cache=kl,
                                     causal=True, block_table=bt, cache_seqlens=seqlens)
        o_f = dispatch.attn_dispatch(q.clone(), k.clone(), v.clone(), cache=fl,
                                     causal=True, block_table=bt, cache_seqlens=seqlens)
        assert o_k.shape == (1, n, qh, hd)
        assert _rmse(o_k.float(), o_f.float()) < 0.08, (step, n, _rmse(o_k.float(), o_f.float()))
