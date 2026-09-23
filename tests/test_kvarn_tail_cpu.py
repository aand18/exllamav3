"""
KVarN M2 CPU tests: Bee sink + configurable exact tail (no CUDA/Triton/ext).

Loads the real ``exllamav3.cache`` / ``attention_fn`` sources with stubbed
parent packages so the heavy ``exllamav3/__init__`` (model, compiled ext)
is never executed. Run with the CPU venv, e.g.::

    kvarn-venv\\Scripts\\python.exe -m pytest tests/test_kvarn_tail_cpu.py -x -q
"""

import importlib.util
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


# --------------------------------------------------------------------------
# Tail policy (llama-kv-cache-kvarn.h:29-49)
# --------------------------------------------------------------------------

def test_tail_policy_floor_rounding_cap():
    p = kvarn.kvarn_tail_policy_for
    assert p(0, 4096)["effective"] == 128       # omitted/0 => intrinsic floor
    assert p(1, 4096)["effective"] == 128
    assert p(128, 4096)["effective"] == 128
    assert p(129, 4096)["effective"] == 256     # ceil to 128-groups
    assert p(256, 4096)["effective"] == 256
    assert p(300, 4096)["effective"] == 384
    assert not p(300, 4096)["native_exact"]
    over = p(5000, 4096)                        # cap at window
    assert over["effective"] == 4096 and over["native_exact"]
    full = p(4096, 4096)
    assert full["effective"] == 4096 and full["native_exact"]
    assert full["exact_groups"] == 32
    small = p(0, 100)                           # window < floor: intrinsic
    assert small["effective"] == 100 and small["native_exact"]
    zero = p(0, 0)
    assert zero["effective"] == 0
    assert p(200, 512)["exact_groups"] == 2     # 200 -> 256 -> 2 groups


def test_tail_type_parse():
    assert kvarn.kvarn_parse_tail_type("f16") == torch.float16
    assert kvarn.kvarn_parse_tail_type("BF16") == torch.bfloat16
    assert kvarn.kvarn_parse_tail_type(torch.bfloat16) == torch.bfloat16
    with pytest.raises(AssertionError):
        kvarn.kvarn_parse_tail_type("q8")


def test_layer_tail_defaults_and_request():
    layer = _layer(2, 128, 512)
    assert layer.has_sink and not layer.is_swa
    assert layer.tail_requested_raw == 0
    assert layer.tail_effective == 128          # Bee floor, not M1 zero-tail
    assert layer.tail_dtype == torch.float16
    assert not layer.tail_native_exact
    layer.free()

    layer = _layer(2, 128, 512, tail_tokens=300, tail_type="bf16")
    assert layer.tail_effective == 384
    assert layer.tail_exact_groups == 3
    assert layer.tail_dtype == torch.bfloat16
    assert not layer.tail_native_exact
    layer.free()

    layer = _layer(2, 128, 512, tail_tokens=512)
    assert layer.tail_native_exact and layer.tail_effective == 512
    layer.free()


def test_swa_derived_from_attention_and_explicit():
    a = _attn(2, 128, sliding_window=512)
    layer = kvarn.CacheLayer_kvarn(None, a, 0, 512)
    assert layer.is_swa and not layer.has_sink
    layer.alloc(torch.device("cpu"))
    layer.free()

    layer = kvarn.CacheLayer_kvarn(None, _attn(2, 128), 0, 512, is_swa=True)
    assert layer.is_swa and not layer.has_sink
    layer.alloc(torch.device("cpu"))
    layer.free()


# --------------------------------------------------------------------------
# Sink permanence + eager seal inside the tail window
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_sink_permanence_dense():
    torch.manual_seed(11)
    kvh, hd, ntok = 2, 128, 300
    layer = _layer(kvh, hd, 512)
    bt = _ids(512)
    k = torch.randn(ntok, kvh, hd).half()
    v = torch.randn(ntok, kvh, hd).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                           k.unsqueeze(0), v.unsqueeze(0), ntok)
    assert not bool(layer.sealed[0])   # permanent sink group stays exact
    assert bool(layer.sealed[1])        # eager seal stays, even though group 1
                                       # overlaps the tail window
    assert not bool(layer.sealed[2])
    kk, vv = layer.get_kv(torch.tensor([ntok], dtype=torch.int32), bt)
    got_k = kk[bt[0]].reshape(-1, kvh, hd)[:ntok]
    got_v = vv[bt[0]].reshape(-1, kvh, hd)[:ntok]
    # Sink rows bit-exact (each key counted once, served exact, not dequant).
    assert torch.equal(got_k[:128], k[:128])
    assert torch.equal(got_v[:128], v[:128])
    # Tail rows (last 128) exact even though group 1 was sealed eagerly.
    assert torch.equal(got_k[ntok - 128:], k[ntok - 128:])
    assert torch.equal(got_v[ntok - 128:], v[ntok - 128:])


@torch.inference_mode()
def test_swa_no_sink_ring():
    torch.manual_seed(12)
    layer = _layer(2, 128, 512, is_swa=True)
    bt = _ids(512)
    k = torch.randn(300, 2, 128).half()
    v = torch.randn(300, 2, 128).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                           k.unsqueeze(0), v.unsqueeze(0), 300)
    assert bool(layer.sealed[0]) and bool(layer.sealed[1])
    assert not bool(layer.sealed[2])
    kk, _ = layer.get_kv(torch.tensor([300], dtype=torch.int32), bt)
    got_k = kk[bt[0]].reshape(-1, 2, 128)[:300]
    # No sink: group 0 is quantized body (close but not bit-exact).
    assert not torch.equal(got_k[:128], k[:128])
    assert _rmse(k.float(), got_k.float()) < 0.15


@torch.inference_mode()
def test_full_window_native_exact():
    torch.manual_seed(13)
    layer = _layer(1, 128, 512, tail_tokens=512)
    assert layer.tail_native_exact
    bt = _ids(512)
    k = torch.randn(300, 1, 128).half()
    v = torch.randn(300, 1, 128).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                           k.unsqueeze(0), v.unsqueeze(0), 300)
    assert not bool(layer.sealed.any())   # no compressed body for the span
    kk, vv = layer.get_kv(torch.tensor([300], dtype=torch.int32), bt)
    assert torch.equal(kk[bt[0]].reshape(-1, 1, 128)[:300], k)
    assert torch.equal(vv[bt[0]].reshape(-1, 1, 128)[:300], v)


@torch.inference_mode()
def test_no_body_bit_exact_when_covered():
    """tail 129 -> effective 256; with sink, all 300 tokens are exact."""
    torch.manual_seed(14)
    layer = _layer(1, 128, 512, tail_tokens=129)
    assert layer.tail_effective == 256
    bt = _ids(512)
    k = torch.randn(300, 1, 128).half()
    v = torch.randn(300, 1, 128).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                           k.unsqueeze(0), v.unsqueeze(0), 300)
    kk, vv = layer.get_kv(torch.tensor([300], dtype=torch.int32), bt)
    assert torch.equal(kk[bt[0]].reshape(-1, 1, 128)[:300], k)
    assert torch.equal(vv[bt[0]].reshape(-1, 1, 128)[:300], v)


# --------------------------------------------------------------------------
# Attention parity: single-softmax merge vs fp16 overlay reference
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_body_tail_single_count_parity():
    torch.manual_seed(15)
    kvh, qh, hd = 2, 4, 128
    layer = _layer(kvh, hd, 512)
    bt = _ids(512)
    past = 300
    kp = torch.randn(1, past, kvh, hd).half()
    vp = torch.randn(1, past, kvh, hd).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt, kp, vp, past)
    kk, vv = layer.get_kv(torch.tensor([past], dtype=torch.int32), bt)
    k_buf = kk[bt[0]].reshape(-1, kvh, hd)[:past].float()
    v_buf = vv[bt[0]].reshape(-1, kvh, hd)[:past].float()

    # Single-count check: gathered image has exactly one row per position,
    # sink + tail rows bit-exact, body rows dequant approximations.
    assert k_buf.shape[0] == past
    assert torch.equal(k_buf[:128].half(), kp[0, :128])
    assert torch.equal(k_buf[past - 128:].half(), kp[0, past - 128:])
    body = k_buf[128:past - 128]
    assert body.numel() > 0
    assert _rmse(body, kp[0, 128:past - 128].float()) < 0.15
    assert _rmse(body, kp[0, 128:past - 128].float()) > 0.0

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
    # Only 44 body tokens are quantized (sink 128 + tail 128 exact).
    assert _rmse(o_q, o_ref) < 0.05, _rmse(o_q, o_ref)


# --------------------------------------------------------------------------
# Storage accounting with tail
# --------------------------------------------------------------------------

def test_storage_accounting_with_tail():
    base = _layer(4, 128, 4096)
    tailed = _layer(4, 128, 4096, tail_tokens=1024, tail_type="bf16")
    try:
        fp16_bytes = 2 * 4096 * 4 * 128 * 2
        # Compressed records are tail-independent.
        assert tailed.storage_size() == base.storage_size() == 32 * 4 * 17920
        assert tailed.storage_size() < 0.5 * fp16_bytes
        # Overhead carries staging + exact tail buffers + bit flags.
        expect = (2 * tailed.stage_k.numel() * torch.half.itemsize +
                  2 * tailed.exact_k.numel() * torch.bfloat16.itemsize +
                  tailed.present.numel() + tailed.sealed.numel())
        assert tailed.overhead_size() == expect
        assert tailed.overhead_size() == base.overhead_size()  # bf16 == f16 bytes
        assert len(tailed.get_tensors()) == 5
    finally:
        base.free()
        tailed.free()


def test_tp_export_m2_roundtrip():
    layer = _layer(2, 128, 512, tail_tokens=300, tail_type="bf16")
    try:
        d = layer.tp_export(None)
        assert d["cls"] is kvarn.CacheLayer_kvarn
        assert d["args"] == {"cache_id": 0, "max_num_tokens": 512,
                             "k_bits": 4, "v_bits": 4,
                             "tail_tokens": 300, "tail_type": "bf16",
                             "is_swa": False}
        rebuilt = kvarn.CacheLayer_kvarn(None, _attn(2, 128), **d["args"])
        assert rebuilt.tail_effective == 384
        assert rebuilt.tail_dtype == torch.bfloat16
        assert rebuilt.has_sink
    finally:
        layer.free()


def test_get_kvarn_records_m2():
    layer = _layer(1, 128, 256)
    try:
        rec = layer.get_kvarn_records()
        assert rec["records"].shape == (2, 1, 17920)
        assert rec["k_bits"] == rec["v_bits"] == 4
        assert rec["has_sink"] and rec["tail_effective"] == 128
    finally:
        layer.free()


# --------------------------------------------------------------------------
# copy_page with sink + tail
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_copy_page_sink_tail():
    torch.manual_seed(16)
    layer = _layer(2, 128, 512)
    bt = _ids(512)
    k = torch.randn(300, 2, 128).half()
    v = torch.randn(300, 2, 128).half()
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                           k.unsqueeze(0), v.unsqueeze(0), 300)
    # Full-page copy: sealed group travels, sink group stays unsealed.
    dst = _layer(2, 128, 512)
    dst.copy_page(layer, 0, 1, 256)
    assert not bool(dst.sealed[2])   # sink group: never sealed
    assert bool(dst.sealed[3])
    assert torch.equal(dst.records[3], layer.records[1])
    assert torch.equal(dst.exact_k[1], layer.exact_k[0])
    # Logical comparison through a remapped block table: dst page 1 now
    # backs logical positions 0..255 (prompt-cache page sharing). The M2
    # exact overlay is logical, so raw page tensors are not comparable.
    bt2 = bt.clone()
    bt2[0, 0] = 1
    kk, _ = dst.get_kv(torch.tensor([300], dtype=torch.int32), bt2)
    ref, _ = layer.get_kv(torch.tensor([300], dtype=torch.int32), bt)
    # Only the copied 256-token span is comparable (the copy overwrote dst
    # page 1, which used to back positions 256..299).
    assert torch.equal(kk[bt2[0]].reshape(-1, 2, 128)[:256],
                       ref[bt[0]].reshape(-1, 2, 128)[:256])
    # Partial copy stays unsealed staging + exact.
    dst2 = _layer(2, 128, 512)
    dst2.copy_page(layer, 1, 0, 44)
    assert not bool(dst2.sealed[0])
    assert torch.equal(dst2.exact_k[0, :44], layer.exact_k[1, :44])


# --------------------------------------------------------------------------
# QSA + tail
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_qsa_tail_planes_untouched():
    torch.manual_seed(17)
    idx = SimpleNamespace(head_dim=32, compress_ratio=4)
    layer = kvarn.CacheLayer_kvarn_qsa(None, _attn(2, 128, idx), 0, 512,
                                      tail_tokens=200, tail_type="f16")
    try:
        assert layer.tail_effective == 256
        assert layer.has_sink
        layer.alloc(torch.device("cpu"))
        assert layer.raw_k.shape == (2, 256, 32)
        assert layer.pooled.shape == (2, 64, 32)
        assert layer.raw_k.dtype == torch.half
        bt = _ids(512)
        k = torch.randn(300, 2, 128).half()
        v = torch.randn(300, 2, 128).half()
        layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                               k.unsqueeze(0), v.unsqueeze(0), 300)
        assert not bool(layer.sealed[0]) and bool(layer.sealed[1])
        kk, _ = layer.get_kv(torch.tensor([300], dtype=torch.int32), bt)
        got = kk[bt[0]].reshape(-1, 2, 128)[:300]
        assert torch.equal(got[:128], k[:128])          # sink exact on dense KV
        assert torch.equal(got[300 - 256:], k[300 - 256:])  # tail exact
        d = layer.tp_export(None)
        assert d["cls"] is kvarn.CacheLayer_kvarn_qsa
        assert d["args"]["tail_tokens"] == 200
    finally:
        layer.free()


# --------------------------------------------------------------------------
# Dispatch end-to-end on CPU with M2 policy
# --------------------------------------------------------------------------

def _load_dispatch():
    if "exllamav3.modules.attention_fn.dispatch" in sys.modules:
        import importlib as _il
        return _il.import_module("exllamav3.modules.attention_fn.dispatch")
    import numpy  # noqa: F401
    _stub("exllamav3.model").Config = object
    _extm = _stub("exllamav3.ext")
    _extm.exllamav3_ext = SimpleNamespace()
    _stub("exllamav3.util").__path__ = []
    _stub("exllamav3.util.memory").malloc_trim = lambda *a, **k: None
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
def test_dispatch_m2_vs_fp16_cpu():
    dispatch = _load_dispatch()
    fp16_cls = sys.modules["exllamav3.cache.fp16"].CacheLayer_fp16
    torch.manual_seed(18)
    kvh, qh, hd = 2, 4, 128
    a = _attn(kvh, hd)
    kl = kvarn.CacheLayer_kvarn(None, a, 0, 512)   # default: sink + 128 floor
    fl = fp16_cls(None, a, 0, 512)
    kl.alloc(torch.device("cpu"))
    fl.alloc(torch.device("cpu"))
    bt = _ids(512)
    for step, n in ((0, 100), (100, 1), (101, 5), (106, 100)):
        q = torch.randn(1, n, qh, hd).half()
        k = torch.randn(1, n, kvh, hd).half()
        v = torch.randn(1, n, kvh, hd).half()
        seqlens = torch.tensor([step], dtype=torch.int32)
        o_k = dispatch.attn_dispatch(q.clone(), k.clone(), v.clone(), cache=kl,
                                     causal=True, block_table=bt, cache_seqlens=seqlens)
        o_f = dispatch.attn_dispatch(q.clone(), k.clone(), v.clone(), cache=fl,
                                     causal=True, block_table=bt, cache_seqlens=seqlens)
        assert o_k.shape == (1, n, qh, hd)
        # Sink + tail exact: tighter than the M1 no-floor bound.
        assert _rmse(o_k.float(), o_f.float()) < 0.06, (step, n, _rmse(o_k.float(), o_f.float()))
