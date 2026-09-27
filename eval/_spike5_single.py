"""Match-Bee: single-kernel fused serve spike (UNTRACKED).

Design (from the launch-economics diagnosis): ONE kernel, grid
(QH, GMAX), does everything per (q-head, 128-row chunk):
- rows partitioned BY POSITION in-kernel: body [sink, n-tail) via
  record dequant; sink/tail window via exact-direct; open-body rows
  (argued impossible for dense) zeros + sticky flag tripwire.
- No ids array (no nonzero sync), no tail temps (no gather), no
  version caching. n via 0-d tensor (no recompile churn).
- Online softmax over the chunk -> partials (m, l, acc) per (qh, chunk).
Second kernel (grid QH): reduce + WHT-folded combine. Plus separate
Q-WHT launch. TOTAL 3 launches (fp16 path: ~2).

Usage: python eval/_spike5_single.py [attn|probe]
  attn: RMSE vs torch full attention (body-ref + exact rows).
  probe: 8k timing vs fp16 paged attention on real layer records.
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, 'eval')
from _spike2_online import _kcol as _kcol_v2, _vrow as _vrow_v2  # noqa: E402,E501


@triton.jit
def _serve_kernel(
    qw_ptr, qf_ptr, rec_ptr, rec_f16_ptr,
    exact_k_ptr, exact_v_ptr, exrev_ptr, sealed_ptr,
    bt_ptr, n_ptr, flag_ptr,
    m_ptr, l_ptr, out_ptr,  # (KVH, QPAD, GMAX), ..., (KVH, QPAD, GMAX, HD)
    K_PAY_OFF, K_SC2, K_ZP2, K_OT2, K_BITS: tl.constexpr,
    V_PAY_OFF, V_SC2, V_ZP2, V_OT2, V_BITS: tl.constexpr,
    C: tl.constexpr, B: tl.constexpr, SL: tl.constexpr, GPS: tl.constexpr,
    KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
    HD: tl.constexpr, GMAX: tl.constexpr, SCALE: tl.constexpr,
    SINK_N: tl.constexpr, TAIL_EFF: tl.constexpr,
):
    """Grid (KVH, GMAX): GQA-shared record reads (once per kv-head, not
    per q-head). QPAD q-heads inside. Per-row body/tail select, exact
    direct, sticky flag. Partials (KVH, QPAD, GMAX) feed the QPAD-aware
    combine (v2 pattern)."""
    pid_h = tl.program_id(0)
    pid_c = tl.program_id(1)
    lane = tl.arange(0, HD)
    qoff = tl.arange(0, QPAD)
    qmask = qoff < QPK
    qw = tl.load(qw_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                 + lane[None, :], mask=qmask[:, None], other=0.0)
    qf = tl.load(qf_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                 + lane[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
    n = tl.load(n_ptr)
    tail_start = n - TAIL_EFF
    m = tl.full([QPAD], -1e30, dtype=tl.float32)
    l = tl.zeros([QPAD], dtype=tl.float32)
    acc = tl.zeros([QPAD, HD], dtype=tl.float32)
    for t in tl.range(128):
        pos = pid_c * 128 + t
        in_range = pos < n
        in_body = (pos >= SINK_N) & (pos < tail_start) & in_range
        in_tail = (~in_body) & in_range
        pos_a = tl.where(in_range, pos, 0)
        page = tl.load(bt_ptr + pos_a // 256)
        offs = pos_a % 256
        g = page * GPS + offs // 128
        s = offs % 128
        gbase_u8 = rec_ptr + g * C * B
        gbase_f16 = rec_f16_ptr + (g * C * B) // 2
        kval = _kcol_v2(gbase_u8, gbase_f16, C, B,
                        K_PAY_OFF, K_SC2, K_ZP2, K_OT2, K_BITS,
                        pid_h, SL, s, lane)
        vval = _vrow_v2(gbase_u8, gbase_f16, C, B,
                        V_PAY_OFF, V_SC2, V_ZP2, V_OT2, V_BITS,
                        pid_h, SL, s, lane)
        es = tl.load(exrev_ptr + g)
        ok = (es >= 0) & in_tail
        e_off = (es * 128 + s) * KVH * HD + pid_h * HD + lane
        ek = tl.load(exact_k_ptr + e_off,
                     mask=ok, other=0.0).to(tl.float32)
        ev = tl.load(exact_v_ptr + e_off,
                     mask=ok, other=0.0).to(tl.float32)
        bad = in_body & (~tl.load(sealed_ptr + g))
        tl.store(flag_ptr, 1, mask=bad)
        krow = tl.where(in_body, kval, ek)
        vrow = tl.where(in_body, vval, ev)
        s_hq = tl.sum(qw * krow[None, :], axis=1) * SCALE
        s_o = tl.sum(qf * krow[None, :], axis=1) * SCALE
        sc = tl.where(in_range, tl.where(in_body, s_hq, s_o),
                      float("-inf"))
        m_new = tl.maximum(m, sc)
        alpha = tl.exp(m - m_new)
        e = tl.exp(sc - m_new)
        l = l * alpha + e
        acc = acc * alpha[:, None] + e[:, None] * vrow[None, :]
        m = m_new
    qoff2 = tl.arange(0, QPAD)
    qmask2 = qoff2 < QPK
    tl.store(m_ptr + (pid_h * QPAD * GMAX) + qoff2 * GMAX + pid_c, m,
             mask=qmask2)
    tl.store(l_ptr + (pid_h * QPAD * GMAX) + qoff2 * GMAX + pid_c, l,
             mask=qmask2)
    tl.store(out_ptr + ((pid_h * QPAD * GMAX) + qoff2[:, None] * GMAX
                        + pid_c) * HD + lane[None, :], acc,
             mask=qmask2[:, None])


@triton.jit
def _merge2_kernel(
    m_ptr, l_ptr, acc_ptr,   # (QH, GMAX), ..., (QH, GMAX, HD)
    out_ptr,                 # (QH, HD) fp32 ORIGINAL domain
    QH: tl.constexpr, GMAX: tl.constexpr, GPAD: tl.constexpr,
    HD: tl.constexpr, SL: tl.constexpr, SSCALE: tl.constexpr,
):
    """One program = one q-head: reduce chunk partials + final in-place
    WHT. num_warps=1 REQUIRED. den==0 guarded."""
    pid = tl.program_id(0)
    lane = tl.arange(0, HD)
    goff = tl.arange(0, GPAD)
    gmask = goff < GMAX
    m = tl.load(m_ptr + pid * GMAX + goff, mask=gmask, other=float("-inf"))
    l = tl.load(l_ptr + pid * GMAX + goff, mask=gmask, other=0.0)
    m_all = tl.max(m)
    e = tl.exp(m - m_all)
    den = tl.sum(l * e)
    num = tl.zeros([HD], dtype=tl.float32)
    for b in tl.range(GPAD):
        ab = tl.load(acc_ptr + (pid * GMAX + b) * HD + lane,
                     mask=(b < GMAX), other=0.0)
        num += ab * tl.sum(tl.where(goff == b, e, 0.0))
    row = tl.where(den > 0, num / den, 0.0)
    base = out_ptr + pid * HD
    tl.store(base + lane, row)
    for _sl in tl.static_range(4):
        if _sl < SL:
            sbase = base + _sl * 128
            scols = tl.arange(0, 128)
            tl.debug_barrier()
            for _s in tl.static_range(7):
                s = 1 << _s
                cur = tl.load(sbase + scols)
                prt = tl.load(sbase + (scols ^ s))
                tl.store(sbase + scols,
                         tl.where((scols & s) == 0, cur + prt, prt - cur))
                tl.debug_barrier()
            tl.store(sbase + scols,
                     tl.load(sbase + scols) * 0.08838834764831845)
    tl.debug_barrier()
    if SL > 1:
        cur = tl.load(base + lane)
        prt = tl.load(base + (lane ^ 128))
        tl.store(base + lane,
                 tl.where((lane & 128) == 0, cur + prt, prt - cur))
        tl.debug_barrier()
    if SL > 2:
        cur = tl.load(base + lane)
        prt = tl.load(base + (lane ^ 256))
        tl.store(base + lane,
                 tl.where((lane & 256) == 0, cur + prt, prt - cur))
        tl.debug_barrier()
    tl.store(base + lane, tl.load(base + lane) * SSCALE)


def serve_online(qw, Qf, records, layout, k_bits, v_bits, exact_k, exact_v_w,
                 exrev, sealed, bt, n_0d, kvh, qpk, sl, hd, gps,
                 sink_n, tail_eff, scale=0.0625):
    """Single-kernel serve host entry. exact_k: (G,128,kvh,hd) ORIG tail
    K blocks; exact_v_w: same blocks in WHT domain (pre-WHT'd by caller
    with the proven kernel); Qf: (QH,HD) fp32 orig-domain Q for tail
    scores. exrev maps group -> exact slot. Returns (QH, HD) fp32
    ORIGINAL + flag value. Buffers per call (spike; production persists).
    """
    dev = qw.device
    qh = kvh * qpk
    gmax = int(records.shape[0])
    qpad = 1 << (qpk - 1).bit_length()
    global _S5BUFS
    try:
        _S5BUFS
    except NameError:
        _S5BUFS = {}
    key = (qh, gmax, hd)
    if key not in _S5BUFS:
        _S5BUFS[key] = (
            torch.empty((kvh, qpad, gmax), dtype=torch.float32, device=dev),
            torch.empty((kvh, qpad, gmax), dtype=torch.float32, device=dev),
            torch.empty((kvh, qpad, gmax, hd), dtype=torch.float32,
                        device=dev),
            torch.empty((qh, hd), dtype=torch.float32, device=dev),
            torch.zeros((1,), dtype=torch.uint8, device=dev))
    m, l, acc, out, flag = _S5BUFS[key]
    flag.zero_()
    rec_f16 = records.view(torch.float16)
    _serve_kernel[(kvh, gmax,)](
        qw, Qf, records, rec_f16, exact_k, exact_v_w, exrev, sealed, bt,
        n_0d,
        flag, m, l, acc,
        layout.k_payload_off,
        layout.k_s_col_off // 2, layout.k_zp_off // 2,
        layout.k_s_row_off // 2, k_bits,
        layout.v_payload_off,
        layout.v_s_row_off // 2, layout.v_zp_off // 2,
        layout.v_s_col_off // 2, v_bits,
        records.shape[1], records.shape[2], sl, gps,
        kvh, qpk, qpad, hd, gmax, scale, sink_n, tail_eff,
        num_warps=4)
    sscale = 1.0 if sl == 1 else (0.7071067811865475 if sl == 2 else 0.5)
    from _spike2_online import _combine_kernel as _combine_v2
    nbpad = 1 << (gmax - 1).bit_length()
    _combine_v2[(qh,)](
        m, l, acc, out, kvh, qpk, qpad, gmax, nbpad, hd, sl, sscale,
        num_warps=1)
    return out, int(flag[0])


def cmd_attn():
    from exllamav3.cache.kvarn import kvarn_make_layout, kvarn_wht_head
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_dequant_groups, kvarn_triton_wht_rows)
    torch.manual_seed(5)
    kvh, sl, hd, qpk, Gg = 4, 2, 256, 6, 4
    bits = (4, 4)
    layout = kvarn_make_layout(128, 128, bits[0], bits[1])
    import _spike2_online as S2
    records = S2._make_records(Gg, kvh, sl, layout, bits[0], bits[1])
    qh = kvh * qpk
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    qw = kvarn_triton_wht_rows(Q.float(), hd)
    # Exact blocks: synthesize valid exact for sink+tail of n=400.
    n = 400
    exact_k = torch.randn(Gg, 128, kvh, hd, dtype=torch.float16,
                          device="cuda")
    exact_v = torch.randn_like(exact_k)
    exrev = torch.arange(Gg, dtype=torch.int64, device="cuda")
    sealed = torch.tensor([False, True, True, True], device="cuda")
    bt = torch.arange(2, dtype=torch.int32, device="cuda")
    n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows as _wht2)
    Qf = Q.float()
    exact_v_w = _wht2(exact_v.float(), hd)
    out, flag = serve_online(qw, Qf, records, layout, bits[0], bits[1],
                             exact_k, exact_v_w, exrev, sealed, bt, n_0d,
                             kvh, qpk, sl, hd, 2, 128, 128)
    print("flag (expect 0):", flag, flush=True)
    # Reference: torch attention over true rows: body (dequant, sealed
    # non-tail) + exact (sink/tail). Positions: sink [0,128), tail
    # [272,400); body rows [128,272) live in groups 1 (128-255) and 2
    # (256-271).
    bk, bv = kvarn_triton_dequant_groups(
        records, layout, bits[0], bits[1], kvh, sl, do_wht=False)
    Kb = kvarn_wht_head(bk, hd)
    Vb = kvarn_wht_head(bv, hd)
    Brow = list(range(128, 272))
    Trow = list(range(0, 128)) + list(range(272, 400))
    # body rows: (group, slot) map: rows 128-255 -> g1 s0-127; 256-271 ->
    # g2 s0-15.
    Kb_body = torch.cat([Kb[1, :, :, :], Kb[2, :16, :, :]]).reshape(-1, kvh,
                                                                   hd)
    Vb_body = torch.cat([Vb[1, :, :, :], Vb[2, :16, :, :]]).reshape(-1, kvh,
                                                                   hd)
    # exact rows for Trow: group = pos//128, slot = pos%128.
    g = torch.tensor(Trow, device="cuda") // 128
    s = torch.tensor(Trow, device="cuda") % 128
    Ke = exact_k[g, s].float()
    Ve = exact_v[g, s].float()
    outs = []
    for h in range(kvh):
        q = Q[h * qpk:(h + 1) * qpk].float()
        Kall = torch.cat([Kb_body[:, h, :], Ke[:, h, :]])
        Vall = torch.cat([Vb_body[:, h, :], Ve[:, h, :]])
        p = torch.softmax((q @ Kall.T) * 0.0625, dim=-1)
        outs.append(p @ Vall)
    ref = torch.cat(outs)
    d = (out - ref).abs()
    print(f"single maxdiff={float(d.max()):.3e} "
          f"RMSE={float((d**2).mean().sqrt()):.3e}", flush=True)
    assert flag == 0
    assert float((d**2).mean().sqrt()) < 1e-6
    print("SINGLE PASS", flush=True)



def cmd_probe():
    from exllamav3 import Config, Model, Tokenizer, Cache
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows)
    from exllamav3.cache import CacheLayer_kvarn
    from exllamav3.cache.kvarn import kvarn_parse_preset
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows as _wht2)
    from exllamav3.modules.attention_fn.triton_paged import (
        paged_attn_triton_decode)
    from kvarn_microkld import SAMPLER_TEXT, populate
    MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
             "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
    NTOK, CHUNK, ITERS = 8192, 4096, 200
    k_bits, v_bits = kvarn_parse_preset("kvarn4")
    config = Config.from_directory(MODEL)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=NTOK + 512,
                  layer_type=CacheLayer_kvarn, k_bits=k_bits, v_bits=v_bits)
    model.load("cuda:0", progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    reps = max(16, (NTOK // 24) + 2)
    ids = tokenizer.encode(SAMPLER_TEXT * reps)[:, :NTOK]
    n = int(ids.shape[1])
    states, _ = populate(model, cache, ids, CHUNK, n)
    del states
    torch.cuda.synchronize()
    lay0 = next(iter(cache.layers.values()))
    kvh, hd, sl = lay0.num_kv_heads, lay0.head_dim, lay0.slices
    qh, qpk = 24, 6
    assert kvh * qpk == qh and sl == 2
    gmax = int(lay0.records.shape[0])
    dev = "cuda"
    torch.manual_seed(9)
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    Qf = Q.float()
    qw = torch.empty((qh, hd), dtype=torch.float32, device="cuda")
    qscratch = torch.empty_like(qw)
    import _spike3_online as _s3
    _s3.qwht_fused(Q, qscratch, qw, sl, 0.7071067811865475)
    # Exact blocks WHT'd once here (production would refresh incrementally;
    # timed separately below as the steady-state delta).
    Ew = _wht2(lay0.exact_k.float(), hd)
    bt = torch.arange(33, dtype=torch.int32, device="cuda")
    n_0d = torch.tensor([n], dtype=torch.int32, device="cuda")

    def hot(fn, iters=ITERS):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters

    pages = NTOK // 256 + 1
    k_cache = torch.randn(pages, 256, kvh, hd, dtype=torch.float16,
                          device="cuda")
    v_cache = torch.randn_like(k_cache)
    btt = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    se = torch.tensor([NTOK], dtype=torch.int32, device="cuda")
    q1 = torch.randn(1, 1, qh, hd, dtype=torch.float16, device="cuda")
    k1 = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
    v1 = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
    paged_attn_triton_decode(q1, k1, v1, k_cache, v_cache, btt, se,
                             causal=True, softmax_scale=0.0625,
                             softcap=0.0, sinks=None)
    t_fp16 = hot(lambda: paged_attn_triton_decode(
        q1, k1, v1, k_cache, v_cache, btt, se, causal=True,
        softmax_scale=0.0625, softcap=0.0, sinks=None))

    def do_spike():
        _s3.qwht_fused(Q, qscratch, qw, sl, 0.7071067811865475)
        o, _f = serve_online(qw, Qf, lay0.records, lay0.layout, k_bits,
                             v_bits, lay0.exact_k, Ew, lay0._exact_rev,
                             lay0.sealed, bt, n_0d, kvh, qpk, sl, hd, 2,
                             128, 128)
        return o
    o = do_spike()
    print("spike5 full path ok", flush=True)
    t_spike = hot(do_spike)
    e2 = lay0.exact_k[:2].float()
    t_eref = hot(lambda: kvarn_triton_wht_rows(e2, hd), iters=500)
    print(f"exact-V refresh (2 blocks, pessimistic): {t_eref * 1e3:.1f} us",
          flush=True)
    t_spike += t_eref
    rows = NTOK
    per_fp16 = t_fp16 / rows * 1e6
    per_spike = t_spike / rows * 1e6
    print(f"fp16 paged attn: {t_fp16:.4f} ms/step over {rows} rows "
          f"({per_fp16:.2f} ns/row)", flush=True)
    print(f"spike5 single:   {t_spike:.4f} ms/step over {rows} rows "
          f"({per_spike:.2f} ns/row)", flush=True)
    print(f"per-row ratio spike/fp16: {per_spike / per_fp16:.3f} "
          f"(gate <= 1.111)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "attn"
    {"attn": cmd_attn, "probe": cmd_probe}[mode]()

