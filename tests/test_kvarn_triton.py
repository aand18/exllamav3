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


def _cuda_triton():
    return torch.cuda.is_available() and \
        importlib.util.find_spec("triton") is not None


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_kernel_side_matches_torch_math():
    # The fused unpack+dequant kernel must equal the torch reference
    # (kvarn_unpack_bits + (q*sc+zp)*other) bit-exact in fp32.
    torch.manual_seed(0)
    for bits in (2, 4, 5, 8):
        NT, PAY = 3, 16384 * bits // 8
        pay = torch.randint(0, 256, (NT, PAY), dtype=torch.uint8,
                            device="cuda")
        sc = torch.randn(NT, 128, dtype=torch.float16, device="cuda")
        zp = torch.randn(NT, 128, dtype=torch.float16, device="cuda")
        oth = torch.randn(NT, 128, dtype=torch.float16, device="cuda")
        got = kt.kvarn_triton_dequant_side(pay, sc, zp, oth, bits)
        assert got.dtype == torch.float32
        q = kvarn.kvarn_unpack_bits(pay.reshape(-1), NT * 16384, bits) \
            .reshape(NT, 128, 128)
        ref = torch.stack([
            kvarn.kvarn_dequantize_tile(q[i].float(), sc[i].float(),
                                        zp[i].float(), oth[i].float())
            for i in range(NT)])
        assert torch.equal(got, ref), bits
        got_w = kt.kvarn_triton_dequant_side(pay, sc, zp, oth, bits,
                                             do_wht=True)
        ref_w = torch.stack([
            kvarn.kvarn_hadamard_128(
                kvarn.kvarn_dequantize_tile(q[i].float(), sc[i].float(),
                                            zp[i].float(), oth[i].float()))
            for i in range(NT)])
        assert torch.equal(got_w, ref_w), (bits, "wht")


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_groups_matches_group_loop():
    # Batched multi-group entry must equal looping the per-group entry
    # (assembly/permute logic), for symmetric and asymmetric widths.
    torch.manual_seed(1)
    for k_bits, v_bits in ((4, 4), (5, 4)):
        layout = kvarn.kvarn_make_layout(128, 128, k_bits, v_bits)
        kvh, sl, Gg = 2, 1, 3
        recs = torch.randint(0, 256, (Gg, kvh * sl, layout.tile_bytes),
                             dtype=torch.uint8, device="cuda")
        f16 = recs.view(torch.float16)
        f16[..., layout.k_s_col_off // 2:] = \
            torch.randn(Gg, kvh * sl,
                        layout.tile_bytes // 2 - layout.k_s_col_off // 2,
                        dtype=torch.float16, device="cuda")
        bk, bv = kt.kvarn_triton_dequant_groups(
            recs, layout, k_bits, v_bits, kvh, sl)
        rk, rv = [], []
        for g in range(Gg):
            tk, tv = kt.kvarn_triton_dequant_group(
                recs[g], layout, k_bits, v_bits, kvh, sl)
            rk.append(tk)
            rv.append(tv)
        ref_k = torch.stack(rk)
        ref_v = torch.stack(rv)
        assert torch.equal(bk, ref_k), (k_bits, v_bits)
        assert torch.equal(bv, ref_v), (k_bits, v_bits)


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_wht_rows_matches_torch_head():
    # Fused row-WHT kernel must equal kvarn_wht_head bit-exact for all
    # supported head dims (FWHT is an involution: same kernel both ways).
    torch.manual_seed(2)
    for hd in (128, 256, 512):
        x = torch.randn(5, 3, hd, dtype=torch.float32, device="cuda")
        got = kt.kvarn_triton_wht_rows(x, hd)
        ref = kvarn.kvarn_wht_head(x, hd)
        assert torch.equal(got, ref), hd
    # Scale regression: the per-128 FWHT stages exchange values across
    # lanes, and tl.debug_barrier does not sync warps (triton 3.8/sm_89:
    # 0/6 exact at 1200 rows with 4/8 warps, nondeterministic scattered
    # corruption). Kernels launch single-warp now; these grids must stay
    # exact (they failed every trial before the fix).
    for hd, nrows in ((128, 1500), (256, 800)):
        y = torch.randn(nrows, hd, dtype=torch.float32, device="cuda")
        assert torch.equal(kt.kvarn_triton_wht_rows(y, hd),
                           kvarn.kvarn_wht_head(y, hd)), (hd, nrows)


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_fused_store_matches_torch_path():
    # Twin layers on CUDA: identical single-row appends across seal
    # boundaries, fused path (env on) vs torch path (env off). All state
    # plus get_kv outputs must be identical.
    from types import SimpleNamespace

    def make():
        attn = SimpleNamespace(num_kv_heads=2, head_dim=128,
                               qsa_indexer=None)
        lay = kvarn.CacheLayer_kvarn(None, attn, 0, 512,
                                     k_bits=4, v_bits=4)
        lay.alloc(torch.device("cuda"))
        return lay

    torch.manual_seed(7)
    A, B = make(), make()
    bt = torch.arange(2, dtype=torch.int32, device="cuda").view(1, 2)
    old = os.environ.get("EXL3_KVARN_TRITON")
    try:
        with torch.inference_mode():
            for step in range(300):
                k = torch.randn(1, 1, 2, 128, dtype=torch.float16,
                                device="cuda")
                v = torch.randn(1, 1, 2, 128, dtype=torch.float16,
                                device="cuda")
                se = torch.tensor([step], dtype=torch.int32, device="cuda")
                os.environ["EXL3_KVARN_TRITON"] = "1"
                B.update_kv_direct(se, bt, k, v, 1)
                del os.environ["EXL3_KVARN_TRITON"]
                A.update_kv_direct(se, bt, k, v, 1)
        for name in ("records", "sealed", "present", "group_base",
                     "page_owner_n", "stage_k", "stage_v",
                     "exact_valid", "exact_k", "exact_v"):
            assert torch.equal(getattr(A, name), getattr(B, name)), name
        with torch.inference_mode():
            se = torch.tensor([300], dtype=torch.int32, device="cuda")
            ka, va = A.get_kv(se, bt)
            kb, vb = B.get_kv(se, bt)
        assert torch.equal(ka, kb) and torch.equal(va, vb)
    finally:
        if old is None:
            os.environ.pop("EXL3_KVARN_TRITON", None)
        else:
            os.environ["EXL3_KVARN_TRITON"] = old


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_overlay_matches_torch_loop():
    # Fused overlay kernel must equal _apply_exact_overlay row-for-row,
    # including absent-block skips (sink + tail window over 300 tokens).
    from types import SimpleNamespace

    attn = SimpleNamespace(num_kv_heads=2, head_dim=128, qsa_indexer=None)
    lay = kvarn.CacheLayer_kvarn(None, attn, 0, 512, k_bits=4, v_bits=4,
                                 is_swa=False)
    lay.alloc(torch.device("cuda"))
    torch.manual_seed(11)
    bt = torch.arange(2, dtype=torch.int32, device="cuda").view(1, 2)
    k = torch.randn(1, 300, 2, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 300, 2, 128, dtype=torch.float16, device="cuda")
    se0 = torch.tensor([0], dtype=torch.int32, device="cuda")
    se = torch.tensor([300], dtype=torch.int32, device="cuda")
    gps = 2  # PAGE_SIZE // KVAR_N_GROUP
    with torch.inference_mode():
        lay.update_kv_direct(se0, bt, k, v, 300)
        # Build the persistent image via a throwaway get_kv (torch path:
        # image stays pre-overlay, overlay lands on the discarded clone).
        lay.get_kv(se, bt)
        t1k = lay._img_k.clone()
        t1v = lay._img_v.clone()
        lay._apply_exact_overlay(t1k, t1v, se, bt)
        t2k = lay._img_k.clone()
        t2v = lay._img_v.clone()
        kt.kvarn_triton_overlay(t2k, t2v, lay, se, bt[0], gps, 128,
                                lay.tail_effective)
    assert torch.equal(t1k, t2k)
    assert torch.equal(t1v, t2v)


def test_default_path_is_torch():
    # Default env (unset): the gate is off, so sealed-group reads use the
    # tested torch loop. Any regression here breaks the whole CPU suite,
    # which runs with this default.
    assert kvarn._kvarn_use_triton() is False or \
        os.environ.get("EXL3_KVARN_TRITON") == "1"
    if "EXL3_KVARN_TRITON" in os.environ:
        del os.environ["EXL3_KVARN_TRITON"]
    assert kvarn._kvarn_use_triton() is False
