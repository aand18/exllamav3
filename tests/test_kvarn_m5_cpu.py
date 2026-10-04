"""
KVarN M5 CPU tests: SWA K/V pair overrides, autosplit/BC integration
decisions, prompt-cache/state versioning (no CUDA/Triton/ext).

Loads the real ``exllamav3.cache`` sources with stubbed parent packages so
the heavy ``exllamav3/__init__`` (model, compiled ext) is never executed.
The BC-attn builder (``modules/attention_fn/bc_attn.py``) is loaded with a
stubbed ``exllamav3.ext`` plus the real torch-only ``util.tensor`` helper
to prove the per-layer KVarN decline. Run with the CPU venv, e.g.::

    kvarn-venv\\Scripts\\python.exe -m pytest tests/test_kvarn_m5_cpu.py -x -q
"""

import importlib.util
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
_cache_pkg.CacheLayer_fp16 = _fp16m.CacheLayer_fp16
_quantm = _stub("exllamav3.cache.quant")
_quantm.CacheLayer_quant = type("CacheLayer_quant", (CacheLayer,), {})
_cache_pkg.CacheLayer_quant = _quantm.CacheLayer_quant

_qsa = _load("exllamav3.cache.qsa", "cache/qsa.py")
QSAPlanes = _qsa.QSAPlanes
kvarn = _load("exllamav3.cache.kvarn", "cache/kvarn.py")

PAGE_SIZE = _constants.PAGE_SIZE
VALID = (2, 3, 4, 5, 6, 8)
QIDX = SimpleNamespace(head_dim=32, compress_ratio=4)


def _attn(kvh=2, hd=128, qsa_indexer=None, sliding_window=-1):
    return SimpleNamespace(num_kv_heads=kvh, head_dim=hd,
                           qsa_indexer=qsa_indexer,
                           sliding_window=sliding_window)


def _layer(kvh=2, hd=128, ntok=512, swa_window=-1, qsa=False, **kw):
    attn = _attn(kvh, hd, QIDX if qsa else None, sliding_window=swa_window)
    cls = kvarn.CacheLayer_kvarn_qsa if qsa else kvarn.CacheLayer_kvarn
    layer = cls(None, attn, 0, ntok, **kw)
    layer.alloc(torch.device("cpu"))
    return layer


def _ids(ntok, bsz=1, pages=None):
    pages = pages or (ntok + PAGE_SIZE - 1) // PAGE_SIZE
    return torch.arange(bsz * pages, dtype=torch.int32).view(bsz, pages)


def _rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()))


@torch.inference_mode()
def _write(layer, k, v, bt, n):
    layer.update_kv_direct(torch.zeros(1, dtype=torch.int32), bt,
                           k.unsqueeze(0), v.unsqueeze(0), n)


@torch.inference_mode()
def _read(layer, bt, n, kvh=2, hd=128):
    kk, vv = layer.get_kv(torch.tensor([n], dtype=torch.int32), bt)
    return (kk[bt[0]].reshape(-1, kvh, hd)[:n],
            vv[bt[0]].reshape(-1, kvh, hd)[:n])


# --------------------------------------------------------------------------
# M5.1: SWA override parsing (mirrors Bee --cache-type-k-swa/-v pairing)
# --------------------------------------------------------------------------

def test_swa_bits_parse_valid():
    p = kvarn.kvarn_parse_bits
    assert p("kvarn8") == 8
    assert p("8") == 8
    assert p("KVARn5") == 5
    assert p(" 6 ") == 6
    assert p("kvarn2") == 2


def test_swa_bits_parse_invalid():
    p = kvarn.kvarn_parse_bits
    for bad in ("kvarn7", "7", "kvarn", "", "q4", "kvarn4x", "1", "16", None):
        with pytest.raises(ValueError):
            p(bad)


def test_swa_pair_default_pairing_and_36_combos():
    f = kvarn.kvarn_parse_swa_pair
    assert f(None, None, (5, 4)) == (5, 4)          # default: main preset
    assert f("", "  ", (4, 4)) == (4, 4)           # empty == omitted
    assert f("kvarn8", "kvarn6", (4, 4)) == (8, 6)
    assert f("5", "4", (4, 4)) == (5, 4)           # bare N spelling
    assert f("Kvarn3", "2", (4, 4)) == (3, 2)
    for kb in VALID:                                # full 36-combo table
        for vb in VALID:
            assert f(f"kvarn{kb}", f"kvarn{vb}", (4, 4)) == (kb, vb)
    with pytest.raises(ValueError):                 # Bee pairing rule
        f("kvarn8", None, (4, 4))
    with pytest.raises(ValueError):
        f(None, "kvarn6", (4, 4))
    with pytest.raises(ValueError):                 # invalid K side
        f("kvarn7", "kvarn6", (4, 4))
    with pytest.raises(ValueError):                 # invalid V side
        f("kvarn8", "q4", (4, 4))


# --------------------------------------------------------------------------
# M5.1: per-group preset application + tail-cap interaction
# --------------------------------------------------------------------------

def test_swa_per_group_preset_application():
    dense = _layer(2, 128, 512, k_bits=5, v_bits=4,
                   swa_k_bits="kvarn8", swa_v_bits="kvarn6")
    swa = _layer(2, 128, 512, swa_window=512, k_bits=5, v_bits=4,
                 swa_k_bits="kvarn8", swa_v_bits="kvarn6")
    try:
        assert not dense.is_swa and swa.is_swa
        assert (dense.k_bits, dense.v_bits) == (5, 4)      # dense: main
        assert dense.swa_override == (8, 6)                # (kept for export)
        assert (swa.k_bits, swa.v_bits) == (8, 6)          # SWA: override
        assert (swa.main_k_bits, swa.main_v_bits) == (5, 4)
        assert swa.layout.tile_bytes != dense.layout.tile_bytes
        assert swa.records.shape != dense.records.shape   # payload widths
        plain = _layer(2, 128, 512, swa_window=512, k_bits=5, v_bits=4)
        try:
            assert plain.swa_override is None
            assert (plain.k_bits, plain.v_bits) == (5, 4)  # inherit main
            assert plain.records.shape == dense.records.shape
        finally:
            plain.free()
    finally:
        dense.free()
        swa.free()


def test_swa_constructor_validation():
    with pytest.raises(ValueError):   # one-sided override
        _layer(2, 128, 512, swa_k_bits="kvarn8")
    with pytest.raises(ValueError):   # bad V width
        _layer(2, 128, 512, swa_k_bits="kvarn8", swa_v_bits="kvarn7")
    with pytest.raises(ValueError):   # bad K width
        _layer(2, 128, 512, swa_window=512,
               swa_k_bits=7, swa_v_bits=6)


@torch.inference_mode()
def test_swa_tail_cap_and_sink_policy():
    # Dense window 2048 vs SWA window 512: same tail request caps
    # differently; SWA keeps the ring (no sink).
    dense = _layer(1, 128, 2048, tail_tokens=1000)
    swa = _layer(1, 128, 512, swa_window=512, tail_tokens=1000,
                 swa_k_bits="kvarn8", swa_v_bits="kvarn6")
    try:
        assert dense.tail_window == 2048 and swa.tail_window == 512
        assert dense.tail_effective == 1024 and not dense.tail_native_exact
        assert swa.tail_effective == 512 and swa.tail_native_exact
        assert dense.has_sink and not swa.has_sink
        assert swa.swa_ring_groups > 0 and dense.swa_ring_groups == 0
        torch.manual_seed(11)
        n = 300
        k = torch.randn(n, 1, 128).half()
        v = torch.randn(n, 1, 128).half()
        _write(dense, k, v, _ids(2048), n)
        _write(swa, k, v, _ids(512), n)
        assert not bool(dense.sealed[0])      # sink group never seals
        assert not bool(swa.sealed.any())     # native-exact: nothing seals
        ring = _layer(1, 128, 512, swa_window=512, tail_tokens=0,
                      swa_k_bits="kvarn8", swa_v_bits="kvarn6")
        try:
            _write(ring, k, v, _ids(512), n)
            assert bool(ring.sealed[0])       # ring has no sink: seals
            assert (ring.k_bits, ring.v_bits) == (8, 6)
        finally:
            ring.free()
        gd, _ = _read(dense, _ids(2048), n, kvh=1)
        gs, _ = _read(swa, _ids(512), n, kvh=1)
        assert torch.equal(gd[:128], k[:128])          # dense sink exact
        assert torch.equal(gd[n - 128:], k[n - 128:])  # dense tail exact
        assert torch.equal(gs[n - 128:], k[n - 128:])  # SWA tail exact
    finally:
        dense.free()
        swa.free()


# --------------------------------------------------------------------------
# M5.1: behavior matrix (SWA override x tail x QSA)
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_behavior_matrix_swa_tail_qsa():
    torch.manual_seed(12)
    n = 300
    k = torch.randn(n, 1, 128).half()
    v = torch.randn(n, 1, 128).half()
    for tail_tokens, te in ((0, 128), (256, 256)):
        for qsa in (False, True):
            dense = _layer(1, 128, 512, qsa=qsa, k_bits=5, v_bits=4,
                           tail_tokens=tail_tokens,
                           swa_k_bits="kvarn8", swa_v_bits="kvarn6")
            swa = _layer(1, 128, 512, swa_window=512, qsa=qsa,
                         k_bits=5, v_bits=4, tail_tokens=tail_tokens,
                         swa_k_bits="kvarn8", swa_v_bits="kvarn6")
            try:
                assert dense.tail_effective == swa.tail_effective == te
                assert (dense.k_bits, dense.v_bits) == (5, 4)
                assert (swa.k_bits, swa.v_bits) == (8, 6)
                if qsa:
                    assert dense.raw_k.dtype == torch.half
                    assert swa.raw_k.dtype == torch.half
                _write(dense, k, v, _ids(512), n)
                _write(swa, k, v, _ids(512), n)
                gd, _ = _read(dense, _ids(512), n, kvh=1)
                gs, _ = _read(swa, _ids(512), n, kvh=1)
                # sink: dense only, always exact
                assert torch.equal(gd[:128], k[:128])
                # tail: both groups, always exact (per-group preset aware)
                assert torch.equal(gd[n - te:], k[n - te:])
                assert torch.equal(gs[n - te:], k[n - te:])
                # body: dequantized, bounded error wherever it exists
                for got, lo, hi, name in (
                        (gd, 128, n - te, "dense"), (gs, 0, n - te, "swa")):
                    if hi > lo:
                        assert _rmse(k[lo:hi].float(), got[lo:hi].float()) < 0.2, \
                            (tail_tokens, qsa, name)
                # QSA-vs-plain image equality is covered by
                # test_qsa_layer_matches_plain_layer_image.
            finally:
                dense.free()
                swa.free()


@torch.inference_mode()
def test_qsa_layer_matches_plain_layer_image():
    """QSA mapping keeps the per-group preset: a QSA layer serves the
    bit-identical KV image of the planes-free layer on the same data."""
    torch.manual_seed(13)
    n, kvh, hd = 300, 1, 128
    k = torch.randn(n, kvh, hd).half()
    v = torch.randn(n, kvh, hd).half()
    for swa_window in (-1, 512):
        kw = dict(k_bits=5, v_bits=4, tail_tokens=0,
                  swa_k_bits="kvarn8", swa_v_bits="kvarn6")
        a = _layer(kvh, hd, 512, swa_window=swa_window, qsa=False, **kw)
        b = _layer(kvh, hd, 512, swa_window=swa_window, qsa=True, **kw)
        try:
            assert (a.k_bits, a.v_bits) == (b.k_bits, b.v_bits)
            assert b.raw_k.dtype == torch.half  # planes stay fp16
            _write(a, k, v, _ids(512), n)
            _write(b, k, v, _ids(512), n)
            assert torch.equal(a.records, b.records)
            assert torch.equal(a.sealed, b.sealed)
            ga, _ = _read(a, _ids(512), n, kvh, hd)
            gb, _ = _read(b, _ids(512), n, kvh, hd)
            assert torch.equal(ga, gb)
        finally:
            a.free()
            b.free()


# --------------------------------------------------------------------------
# M5.2: autosplit -- dummy measurement must not corrupt seals
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_autosplit_rewrite_matches_clean():
    """Simulate the autosplit load-time measuring forward (garbage prefix
    write, possibly sealing groups) followed by the real prefix rewrite:
    the layer must end up identical to a clean layer."""
    torch.manual_seed(14)
    n, kvh, hd = 300, 2, 128
    k = torch.randn(n, kvh, hd).half()
    v = torch.randn(n, kvh, hd).half()
    gk = torch.randn(n, kvh, hd).half()  # "dummy-measure" garbage
    gv = torch.randn(n, kvh, hd).half()
    for swa_window in (-1, 512):
        kw = dict(k_bits=5, v_bits=4, tail_tokens=0,
                  swa_k_bits="kvarn8", swa_v_bits="kvarn6")
        dirty = _layer(kvh, hd, 512, swa_window=swa_window, **kw)
        clean = _layer(kvh, hd, 512, swa_window=swa_window, **kw)
        try:
            bt = _ids(512)
            _write(dirty, gk, gv, bt, n)
            _write(dirty, k, v, bt, n)   # real prefix rewrite
            _write(clean, k, v, bt, n)
            assert torch.equal(dirty.sealed, clean.sealed)
            assert torch.equal(dirty.group_base, clean.group_base)
            assert torch.equal(dirty.records, clean.records)
            gd, _ = _read(dirty, bt, n, kvh, hd)
            gc, _ = _read(clean, bt, n, kvh, hd)
            assert torch.equal(gd, gc)
        finally:
            dirty.free()
            clean.free()


def test_autosplit_probe_declined_transient_conservative():
    ok, reason = kvarn.kvarn_autosplit_probe_supported()
    assert ok is False and isinstance(reason, str) and reason
    # The synthetic probe zeroes page tensors KVarN does not have.
    layer = _layer(1, 128, 512)
    try:
        assert not hasattr(layer, "k") and not hasattr(layer, "qk")
        t = kvarn.kvarn_autosplit_transient_bytes(2, 2, 128)
        assert t == 2 * 256 * 2 * 128 * 2 * 2
        torch.manual_seed(15)
        n = 300
        _write(layer, torch.randn(n, 1, 128).half(),
               torch.randn(n, 1, 128).half(), _ids(512), n)
        # Conservative: fp16-size transient covers the real footprint.
        assert layer.storage_size() + layer.overhead_size() <= t
    finally:
        layer.free()


# --------------------------------------------------------------------------
# M5.2: BC graph path declines KVarN per layer (real builder)
# --------------------------------------------------------------------------

def _load_bc():
    _stub("exllamav3.model").Config = object
    _extm = _stub("exllamav3.ext")
    _extm.exllamav3_ext = SimpleNamespace()
    _util = _stub("exllamav3.util")
    _util.__path__ = [str(EXL / "util")]
    for name, rel in [
        ("exllamav3.constants", "constants.py"),
        ("exllamav3.cache.cache", "cache/cache.py"),
        ("exllamav3.cache.qsa", "cache/qsa.py"),
        ("exllamav3.cache.kvarn", "cache/kvarn.py"),
        ("exllamav3.util.device_copy", "util/device_copy.py"),
        ("exllamav3.util.tensor", "util/tensor.py"),
    ]:
        sys.modules.pop(name, None)
        _load(name, rel)
    _stub("exllamav3.modules").__path__ = [str(EXL / "modules")]
    _afn = _stub("exllamav3.modules.attention_fn")
    _afn.__path__ = [str(EXL / "modules" / "attention_fn")]
    sys.modules.pop("exllamav3.modules.attention_fn.bc_attn", None)
    bc = _load("exllamav3.modules.attention_fn.bc_attn",
               "modules/attention_fn/bc_attn.py")
    global kvarn
    kvarn = sys.modules["exllamav3.cache.kvarn"]
    # rebind the stub page-tensor names the builder imports
    sys.modules["exllamav3.cache"].CacheLayer_fp16 = _fp16m.CacheLayer_fp16
    sys.modules["exllamav3.cache"].CacheLayer_quant = _quantm.CacheLayer_quant
    return bc


def _eligible_module():
    proj = SimpleNamespace(quant_type="exl3", inner=SimpleNamespace(bc=object()),
                           out_features=2048)
    return SimpleNamespace(
        layer_idx=0, device=torch.device("cpu"), qsa_indexer=None,
        rope=SimpleNamespace(), q_norm=None, headwise_gate=False,
        interleaved_gate=False, full_gate=False, g_proj=None, v_norm=None,
        sinks=None, sliding_window=-1, logit_softcapping=None,
        tp_span_heads_norm=False, multi_kv=None, use_k_as_v=False,
        head_dim=128, num_q_heads=4, num_kv_heads=2,
        q_proj=proj, k_proj=proj, v_proj=proj, o_proj=proj,
    )


def test_bc_attn_declines_kvarn_layers():
    bc = _load_bc()
    ok, reason = kvarn.kvarn_bc_attn_supported()
    assert ok is False and isinstance(reason, str) and reason
    m = _eligible_module()
    assert bc._module_eligible(m)  # module side eligible: decline is KVarN's
    layers = [
        _layer(2, 128, 512),
        _layer(2, 128, 512, swa_window=512,
               swa_k_bits="kvarn8", swa_v_bits="kvarn6"),
        _layer(2, 128, 512, qsa=True),
    ]
    try:
        for layer in layers:
            assert bc.build_bc_attn(m, layer) is None  # graceful, no raise
    finally:
        for layer in layers:
            layer.free()


# --------------------------------------------------------------------------
# M5.3: tp_export versioning + prompt-cache reuse with state
# --------------------------------------------------------------------------

@torch.inference_mode()
def test_tp_export_roundtrip_with_sink_tail_compact():
    torch.manual_seed(16)
    n, kvh, hd = 300, 2, 128
    k = torch.randn(n, kvh, hd).half()
    v = torch.randn(n, kvh, hd).half()
    layer = _layer(kvh, hd, 512, k_bits=5, v_bits=4,
                   tail_tokens=256, tail_type="f16",
                   swa_k_bits="kvarn8", swa_v_bits="kvarn6")
    try:
        bt = _ids(512)
        _write(layer, k, v, bt, n)
        assert bool(layer.sealed[1]) and not bool(layer.sealed[0])
        d = layer.tp_export(None)
        assert d["cls"] is kvarn.CacheLayer_kvarn
        a = d["args"]
        assert a["kvarn_version"] == kvarn.KVAR_N_STATE_VERSION
        assert (a["k_bits"], a["v_bits"]) == (5, 4)      # main pair
        assert (a["swa_k_bits"], a["swa_v_bits"]) == (8, 6)
        assert (a["tail_tokens"], a["tail_type"]) == (256, "f16")
        rebuilt = kvarn.CacheLayer_kvarn(None, _attn(kvh, hd), **a)
        try:
            assert (rebuilt.main_k_bits, rebuilt.main_v_bits) == (5, 4)
            assert (rebuilt.k_bits, rebuilt.v_bits) == (5, 4)
            assert rebuilt.swa_override == (8, 6)
            assert rebuilt.has_sink and rebuilt.tail_effective == 256
            assert rebuilt.tp_export(None)["args"] == a  # stable re-export
        finally:
            rebuilt.free()
        # SWA side roundtrip keeps the override preset.
        s = _layer(kvh, hd, 512, swa_window=512, k_bits=5, v_bits=4,
                   tail_tokens=256, swa_k_bits="kvarn8", swa_v_bits="kvarn6")
        try:
            ds = s.tp_export(None)
            rs = kvarn.CacheLayer_kvarn(
                None, _attn(kvh, hd, sliding_window=512), **ds["args"])
            try:
                assert (rs.k_bits, rs.v_bits) == (8, 6)
                assert not rs.has_sink
                assert rs.tp_export(None)["args"] == ds["args"]
            finally:
                rs.free()
        finally:
            s.free()
        # Prompt-cache reuse: copy_page preserves sink+tail exactness.
        dst = _layer(kvh, hd, 512, k_bits=5, v_bits=4,
                     tail_tokens=256, tail_type="f16",
                     swa_k_bits="kvarn8", swa_v_bits="kvarn6")
        try:
            dst.copy_page(layer, 0, 1, 256)
            bt2 = bt.clone()
            bt2[0, 0] = 1
            kk, _ = dst.get_kv(torch.tensor([n], dtype=torch.int32), bt2)
            ref, _ = layer.get_kv(torch.tensor([n], dtype=torch.int32), bt)
            assert torch.equal(kk[bt2[0]].reshape(-1, kvh, hd)[:256],
                               ref[bt[0]].reshape(-1, kvh, hd)[:256])
        finally:
            dst.free()
    finally:
        layer.free()


def test_tp_export_version_and_preset_fail_closed():
    layer = _layer(2, 128, 512)
    try:
        d = layer.tp_export(None)
        stale = dict(d["args"], kvarn_version=d["args"]["kvarn_version"] - 1)
        with pytest.raises(ValueError):   # stale state rejected
            kvarn.CacheLayer_kvarn(None, _attn(2, 128), **stale)
        other = _layer(2, 128, 512, k_bits=8, v_bits=6)
        try:
            with pytest.raises(AssertionError):   # cross-preset copy
                other.copy_page(layer, 0, 0, 128)
        finally:
            other.free()
        swad = _layer(2, 128, 512, swa_window=512)   # SWA-group mismatch
        try:
            with pytest.raises(AssertionError):
                swad.copy_page(layer, 0, 0, 128)
        finally:
            swad.free()
    finally:
        layer.free()
