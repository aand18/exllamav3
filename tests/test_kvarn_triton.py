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
    # Spec B prefill shape: stacked (T,2,kvh,hd) fresh temp, n=T*2*kvh
    # rows, T=4096 (kvh endpoints 2..8, all head dims). Both out-of-place
    # and inplace-on-fresh-temp must equal the torch reference bit-exact.
    for kvh in (2, 8):
        for hd in (128, 256, 512):
            z = torch.randn(4096, 2, kvh, hd, dtype=torch.float32,
                            device="cuda")
            assert torch.equal(kt.kvarn_triton_wht_rows(z, hd),
                               kvarn.kvarn_wht_head(z, hd)), (kvh, hd)
            zi = z.clone()
            got_i = kt.kvarn_triton_wht_rows(zi, hd, inplace=True)
            assert torch.equal(got_i, kvarn.kvarn_wht_head(z, hd)), \
                (kvh, hd, "inplace")


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
            # Build the incremental eref cache up front so the fused
            # path exercises the in-kernel eref write-through (DO_EREF)
            # while the torch path uses the host slot refresh: the
            # final compare below then isolates write-through exactness.
            A._eref_ensure()
            B._eref_ensure()
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
        # In-kernel eref write-through (B) vs host slot refresh (A).
        assert torch.equal(A._ov_eref_w, B._ov_eref_w), "eref"
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
def test_fused_serve_matches_torch_path():
    # Twin layers, mixed single/multi-row writes with a get_kv after
    # every step: full Triton (fused store + serve + overlay + dequant)
    # vs pure torch. Served outputs must match every step and all state
    # (including the persistent image) at the end, across seal
    # boundaries, for every head dim (per-128 vs cross-slice WHT).
    from types import SimpleNamespace

    for hd in (128, 256, 512):
        def make():
            attn = SimpleNamespace(num_kv_heads=2, head_dim=hd,
                                   qsa_indexer=None)
            lay = kvarn.CacheLayer_kvarn(None, attn, 0, 1024,
                                         k_bits=4, v_bits=4, is_swa=False)
            lay.alloc(torch.device("cuda"))
            return lay

        torch.manual_seed(100 + hd)
        A, B = make(), make()
        bt = torch.arange(4, dtype=torch.int32, device="cuda").view(1, 4)
        old = os.environ.get("EXL3_KVARN_TRITON")
        try:
            pos, step = 0, 0
            while pos < 512:
                length = 1 if step % 3 else 100
                length = min(length, 512 - pos)
                k = torch.randn(1, length, 2, hd, dtype=torch.float16,
                                device="cuda")
                v = torch.randn(1, length, 2, hd, dtype=torch.float16,
                                device="cuda")
                se = torch.tensor([pos], dtype=torch.int32, device="cuda")
                os.environ["EXL3_KVARN_TRITON"] = "1"
                B.update_kv_direct(se, bt, k, v, length)
                kb, vb = B.get_kv(se + length, bt)
                del os.environ["EXL3_KVARN_TRITON"]
                A.update_kv_direct(se, bt, k, v, length)
                ka, va = A.get_kv(se + length, bt)
                assert torch.equal(ka, kb) and torch.equal(va, vb), (hd, step)
                pos += length
                step += 1
            # The Triton path serves the persistent image in place: the
            # last get_kv's stashed overlay + dirty bits are transient
            # serve state, not a records disagreement. Restore the stash
            # (get_kv entry would do it on the next call) then refresh
            # dirty groups (idempotent recompute from records) before
            # comparing image state; served outputs were already asserted
            # equal every step above.
            if bool(getattr(B, "_ov_pending", False)):
                kt.kvarn_triton_unoverlay(B._img_k, B._img_v, B)
            assert not bool(getattr(B, "_ov_pending", False))
            _d = B._dirty_mask.nonzero().flatten()
            if _d.numel():
                B._refresh_groups(_d)
                B._dirty_mask[_d] = False
            for name in ("records", "sealed", "present", "group_base",
                         "page_owner_n", "stage_k", "stage_v",
                         "exact_valid", "exact_k", "exact_v",
                         "_img_k", "_img_v"):
                assert torch.equal(getattr(A, name), getattr(B, name)), \
                    (hd, name)
        finally:
            if old is None:
                os.environ.pop("EXL3_KVARN_TRITON", None)
            else:
                os.environ["EXL3_KVARN_TRITON"] = old


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_online_serve_k4v2_matches_torch():
    # Quad fast path (V_BITS==2) vs torch reference with the generic
    # bit-loop unpack: synthetic K4V2 records, all-body (sink 0,
    # tail_eff 0, all sealed), direct online_serve (combine folds the
    # out-WHT, so kernel output is original-domain) vs torch
    # WHT-domain attention + inverse WHT.
    # Different reduction orders, so allclose (not equal): a bit-flip
    # in unpack shows as O(1) errors, association noise is ~1e-6.
    from types import SimpleNamespace
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from _spike2_online import _make_records

    torch.manual_seed(15)
    kvh, sl, hd, qpk = 2, 1, 128, 2
    qh = kvh * qpk
    layout = kvarn.kvarn_make_layout(128, 128, 4, 2)
    G, n = 8, 1024
    gps, scale = 2, hd ** -0.5
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    Qf = Q.float()
    Qw = kt.kvarn_triton_wht_rows(Q.float(), hd)
    records = _make_records(G, kvh, sl, layout, 4, 2)
    exact_k = torch.zeros(G, 128, kvh, hd, dtype=torch.float16,
                          device="cuda")
    exact_v_w = torch.zeros(G, 128, kvh, hd, dtype=torch.float32,
                            device="cuda")
    exrev = torch.arange(G, dtype=torch.int64, device="cuda")
    sealed = torch.ones(G, dtype=torch.bool, device="cuda")
    bt = torch.arange((n + 255) // 256, dtype=torch.int32, device="cuda")
    n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")
    lay = SimpleNamespace(
        records=records, layout=layout, k_bits=4, v_bits=2,
        num_kv_heads=kvh, head_dim=hd, slices=sl)
    out_b, flag_b = kt.kvarn_triton_online_serve(
        lay, Qw, Qf, exact_k, exact_v_w, exrev, sealed, bt, n_0d,
        qpk, scale, 0, 0, gps, gc=G)
    assert flag_b == 0
    # Torch reference: generic-unpack dequant + full attention in the
    # WHT domain (same math as the kernel: dot(Qw, Kw) by WHT
    # symmetry), then the inverse WHT the combine kernel folds in.
    # Different reduction orders, so allclose (not equal).
    Ks, Vs = [], []
    for g in range(G):
        for h in range(kvh):
            rec = records[g, h * sl]
            tk = kvarn.kvarn_dequantize_k_tile(rec, 4, layout)
            tv = kvarn.kvarn_dequantize_v_tile(rec, 2, layout)
            Ks.append(tk.T.contiguous())
            Vs.append(tv)
    K_all = torch.stack(Ks).reshape(G, kvh, 128, hd).permute(1, 0, 2, 3)
    V_all = torch.stack(Vs).reshape(G, kvh, 128, hd).permute(1, 0, 2, 3)
    refs = []
    for h in range(kvh):
        q = Qw[h * qpk:(h + 1) * qpk].float()
        st = torch.bmm(q.unsqueeze(0).expand(1, -1, -1),
                       K_all[h].reshape(-1, hd).T.unsqueeze(0)
                       ).squeeze(0) * scale
        pe = torch.softmax(st, dim=-1)
        refs.append(pe @ V_all[h].reshape(-1, hd))
    ref = kvarn.kvarn_wht_head(torch.cat(refs).reshape(qh, hd), hd)
    d = (out_b.float() - ref.float()).abs()
    print(f"k4v2 serve: maxabs={float(d.max()):.3e} "
          f"meanabs={float(d.mean()):.3e}")
    assert float(d.max()) < 5e-3, float(d.max())


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_prefill_4096_chunks_match_torch_path():
    # Spec B prefill twin: twin layers, 4096-token prefill in 4096-wide
    # chunks (single chunk) plus 1024-wide chunks, fused WHT (env on) vs
    # torch (env off). State plus get_kv outputs must be identical for
    # every head dim (covers the T=4096, n=T*2*kvh inplace path).
    from types import SimpleNamespace

    for hd in (128, 256, 512):
        def make():
            attn = SimpleNamespace(num_kv_heads=2, head_dim=hd,
                                   qsa_indexer=None)
            lay = kvarn.CacheLayer_kvarn(None, attn, 0, 8192,
                                         k_bits=4, v_bits=4, is_swa=False)
            lay.alloc(torch.device("cuda"))
            return lay

        torch.manual_seed(200 + hd)
        A, B = make(), make()
        bt = torch.arange(32, dtype=torch.int32, device="cuda").view(1, 32)
        old = os.environ.get("EXL3_KVARN_TRITON")
        try:
            with torch.inference_mode():
                # Build the eref cache up front so the multi-row hook
                # path maintains it on both (triton-WHT vs torch
                # prefill): the final compare isolates hook exactness.
                A._eref_ensure()
                B._eref_ensure()
                pos = 0
                for length in (4096, 1024, 1024):
                    if pos + length > 6144:
                        break
                    k = torch.randn(1, length, 2, hd, dtype=torch.float16,
                                    device="cuda")
                    v = torch.randn(1, length, 2, hd, dtype=torch.float16,
                                    device="cuda")
                    se = torch.tensor([pos], dtype=torch.int32,
                                      device="cuda")
                    os.environ["EXL3_KVARN_TRITON"] = "1"
                    B.update_kv_direct(se, bt, k, v, length)
                    kb, vb = B.get_kv(se + length, bt)
                    del os.environ["EXL3_KVARN_TRITON"]
                    A.update_kv_direct(se, bt, k, v, length)
                    ka, va = A.get_kv(se + length, bt)
                    assert torch.equal(ka, kb) and torch.equal(va, vb), \
                        (hd, length, pos)
                    pos += length
            if bool(getattr(B, "_ov_pending", False)):
                kt.kvarn_triton_unoverlay(B._img_k, B._img_v, B)
            _d = B._dirty_mask.nonzero().flatten()
            if _d.numel():
                B._refresh_groups(_d)
                B._dirty_mask[_d] = False
            for name in ("records", "sealed", "present", "group_base",
                         "page_owner_n", "stage_k", "stage_v",
                         "exact_valid", "exact_k", "exact_v",
                         "_img_k", "_img_v", "_ov_eref_w"):
                assert torch.equal(getattr(A, name), getattr(B, name)), \
                    (hd, name)
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
    # Setup runs on the torch path (the fused overlay stashes + sets the
    # pending flag; the documented unit-test purity below assumes torch).
    old = os.environ.pop("EXL3_KVARN_TRITON", None)
    try:
        with torch.inference_mode():
            lay.update_kv_direct(se0, bt, k, v, 300)
            # Build the persistent image via a throwaway get_kv (torch path:
            # image stays pre-overlay, overlay lands on the discarded clone).
            lay.get_kv(se, bt)
    finally:
        if old is not None:
            os.environ["EXL3_KVARN_TRITON"] = old
    with torch.inference_mode():
        t1k = lay._img_k.clone()
        t1v = lay._img_v.clone()
        lay._apply_exact_overlay(t1k, t1v, se, bt)
        t2k = lay._img_k.clone()
        t2v = lay._img_v.clone()
        # stash=None: unit-test purity on throwaway clones (no stash, no
        # pending flag); production get_kv passes the layer stash.
        kt.kvarn_triton_overlay(t2k, t2v, lay, se, bt[0], gps, 128,
                                lay.tail_effective)
        assert not bool(getattr(lay, "_ov_pending", False))
    assert torch.equal(t1k, t2k)
    assert torch.equal(t1v, t2v)
    assert not bool(lay._dirty_mask.any())


def test_default_path_is_torch():
    # Default env (unset): the gate is off, so sealed-group reads use the
    # tested torch loop. Any regression here breaks the whole CPU suite,
    # which runs with this default.
    assert kvarn._kvarn_use_triton() is False or \
        os.environ.get("EXL3_KVARN_TRITON") == "1"
    if "EXL3_KVARN_TRITON" in os.environ:
        del os.environ["EXL3_KVARN_TRITON"]
    assert kvarn._kvarn_use_triton() is False


@pytest.mark.skipif(not _cuda_triton(), reason="needs CUDA + triton")
def test_promoted_serve_matches_eval_spike():
    # Promoted _kvarn_online_serve_kernel is a verbatim copy of eval
    # spike7: identical inputs must give bit-identical outputs
    # (torch.equal, not RMSE). Guards promotion drift.
    sys.path.insert(0, str(ROOT / "eval"))
    import _spike7_coal as s7
    from _spike2_online import _make_records
    torch.manual_seed(5)
    kvh, sl, hd, qpk, Gg = 4, 2, 256, 6, 4
    bits = (4, 4)
    layout = kvarn.kvarn_make_layout(128, 128, bits[0], bits[1])
    records = _make_records(Gg, kvh, sl, layout, bits[0], bits[1])
    qh = kvh * qpk
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    qw = kt.kvarn_triton_wht_rows(Q.float(), hd)
    n = 400
    exact_k = torch.randn(Gg, 128, kvh, hd, dtype=torch.float16,
                          device="cuda")
    exact_v = torch.randn_like(exact_k)
    # exrev[0] = -1: sink group without an exact slot exercises the
    # -inf path for unassigned tail rows (staging-fallback rows the torch
    # tail path serves). Both kernels must agree bit-exactly.
    exrev = torch.tensor([-1, 1, 2, 3], dtype=torch.int64, device="cuda")
    sealed = torch.tensor([False, True, True, True], device="cuda")
    bt = torch.arange(2, dtype=torch.int32, device="cuda")
    n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")
    Qf = Q.float()
    exact_v_w = kt.kvarn_triton_wht_rows(exact_v.float(), hd)
    out_e, flag_e = s7.serve_online_s7(
        qw, Qf, records, layout, bits[0], bits[1], exact_k, exact_v_w,
        exrev, sealed, bt, n_0d, kvh, qpk, sl, hd, 2, 128, 128)
    lay = types.SimpleNamespace(
        records=records, layout=layout, k_bits=bits[0], v_bits=bits[1],
        num_kv_heads=kvh, head_dim=hd, slices=sl)
    out_p, flag_p = kt.kvarn_triton_online_serve(
        lay, qw, Qf, exact_k, exact_v_w, exrev, sealed, bt, n_0d, qpk,
        0.0625, 128, 128, 2)
    assert flag_e == flag_p == 0
    assert torch.equal(out_e, out_p)
