"""Match-Bee: fused tail-merge spike (UNTRACKED).

v1 dispatch analysis: ~17 launches + 3-4 syncs/step (torch tail SDPA +
merge dominate). This spike fuses the tail path:
- _tail_block_kernel, grid (KVH, TCHMAX=2): GQA-shared (qpk accumulators
  inside the kv-head program: tail rows read ONCE, not per q-head).
  Standard flash-split over fp16 tail rows (convert on load, exact),
  ORIG domain, fp32 math. Fully-masked chunks emit m=-1e30/l=0.
- _mega_merge_kernel, grid (QH,), num_warps=1: reduce body partials +
  reduce tail partials + WHT tail-num via out-row scratch + global
  merge + final in-place WHT. Replaces torch body-reduce + tail-SDPA +
  merge + 2 WHT launches.
Target step: Q-WHT(1) + stage-1(1) + tail-gather(~2) + tail-split(1) +
mega-merge(1) ≈ 6 launches (image path ≈ 5).

Usage: python eval/_spike4_online.py [tail|merge|probe]
  tail: tail-split RMSE vs torch tail attention.
  merge: full mega-merge RMSE vs torch full attention (body+tail).
  probe: timing breakdown on real 8k layer records + exact tail.
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, 'eval')

TCHMAX = 2  # tail capacity 256 rows (sink 128 + tail 128, dense)


@triton.jit
def _tail_block_kernel(
    q_ptr,                  # (QH, HD) fp16 query rows
    k_ptr, v_ptr,           # (MAXW, KVH, HD) fp32 tail rows (compacted)
    m_ptr, l_ptr, out_ptr,  # (KVH, QPK, 2), ..., (KVH, QPK, 2, HD)
    n_ptr,                  # (1,) int32 current length (R derived in-kernel)
    SINK_TAIL: tl.constexpr,  # sink + tail rows (R = min(n, SINK_TAIL))
    KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
    HD: tl.constexpr, SCALE: tl.constexpr,
):
    """One program = one (kv head, 128-row tail chunk). Flash-split over
    fp16 rows (exact convert on load). R derives in-kernel (no recompile
    churn across lengths); empty chunks emit m=-1e30/l=0."""
    pid_h = tl.program_id(0)
    pid_c = tl.program_id(1)
    n = tl.load(n_ptr)
    R = tl.minimum(n, SINK_TAIL)
    lane = tl.arange(0, HD)
    qoff = tl.arange(0, QPAD)
    qmask = qoff < QPK
    q = tl.load(q_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                + lane[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
    m = tl.full([QPAD], -1e30, dtype=tl.float32)
    l = tl.zeros([QPAD], dtype=tl.float32)
    acc = tl.zeros([QPAD, HD], dtype=tl.float32)
    for t in tl.range(128):
        tt = pid_c * 128 + t
        active = tt < R
        krow = tl.load(k_ptr + (tt * KVH + pid_h) * HD + lane,
                       mask=active, other=0.0)
        vrow = tl.load(v_ptr + (tt * KVH + pid_h) * HD + lane,
                       mask=active, other=0.0)
        s = tl.where(active, tl.sum(q * krow[None, :], axis=1) * SCALE,
                     float("-inf"))
        m_new = tl.maximum(m, s)
        alpha = tl.exp(m - m_new)
        e = tl.exp(s - m_new)
        l = l * alpha + e
        acc = acc * alpha[:, None] + e[:, None] * vrow[None, :]
        m = m_new
    qoff2 = tl.arange(0, QPAD)
    qmask2 = qoff2 < QPK
    tl.store(m_ptr + (pid_h * QPAD) * 2 + qoff2 * 2 + pid_c, m,
             mask=qmask2)
    tl.store(l_ptr + (pid_h * QPAD) * 2 + qoff2 * 2 + pid_c, l,
             mask=qmask2)
    tl.store(out_ptr + ((pid_h * QPAD) * 2 + qoff2[:, None] * 2
                        + pid_c) * HD + lane[None, :], acc,
             mask=qmask2[:, None])


@triton.jit
def _mega_merge_kernel(
    bm_ptr, bl_ptr, bacc_ptr,  # body partials (KVH, QPK, NB[, HD])
    tm_ptr, tl_ptr, tacc_ptr,  # tail partials (KVH, QPK, TCHMAX[, HD])
    out_ptr,                   # (QH, HD) fp32 ORIGINAL domain
    KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
    NB: tl.constexpr, NBPAD: tl.constexpr,
    HD: tl.constexpr, SL: tl.constexpr, SSCALE: tl.constexpr,
):
    """One program = one q-head: reduce body partials + reduce tail
    partials + global merge + final in-place WHT. num_warps=1 REQUIRED
    (FWHT exchange needs lockstep). den==0 guarded (short prefix)."""
    pid = tl.program_id(0)
    ph = pid // QPK
    pq = pid % QPK
    lane = tl.arange(0, HD)
    # Body reduce (WHT domain).
    nboff = tl.arange(0, NBPAD)
    nbmask = nboff < NB
    bm = tl.load(bm_ptr + (ph * QPAD + pq) * NB + nboff, mask=nbmask,
                 other=float("-inf"))
    bl = tl.load(bl_ptr + (ph * QPAD + pq) * NB + nboff, mask=nbmask,
                 other=0.0)
    m_b = tl.max(bm)
    eb = tl.exp(bm - m_b)
    den_b = tl.sum(bl * eb)
    num_b = tl.zeros([HD], dtype=tl.float32)
    for b in tl.range(NBPAD):
        ab = tl.load(bacc_ptr + ((ph * QPAD + pq) * NB + b) * HD + lane,
                     mask=(b < NB), other=0.0)
        num_b += ab * tl.sum(tl.where(nboff == b, eb, 0.0))
    # Tail reduce (ORIG domain).
    tcoff = tl.arange(0, 2)
    tm = tl.load(tm_ptr + (ph * QPAD + pq) * 2 + tcoff, mask=tcoff < 2,
                 other=float("-inf"))
    tl_ = tl.load(tl_ptr + (ph * QPAD + pq) * 2 + tcoff, mask=tcoff < 2,
                  other=0.0)
    m_t = tl.max(tm)
    et = tl.exp(tm - m_t)
    den_t = tl.sum(tl_ * et)
    num_t = tl.zeros([HD], dtype=tl.float32)
    for b in tl.range(2):
        at = tl.load(tacc_ptr + ((ph * QPAD + pq) * 2 + b) * HD + lane)
        num_t += at * tl.sum(tl.where(tcoff == b, et, 0.0))
    # Global merge in WHT domain: WHT the tail numerator via the out
    # row as scratch, then merge, then final in-place WHT.
    base = out_ptr + pid * HD
    tl.store(base + lane, num_t)
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
    num_tw = tl.load(base + lane) * SSCALE
    m_all = tl.maximum(m_b, m_t)
    wb = tl.exp(m_b - m_all)
    wt = tl.exp(m_t - m_all)
    den = den_b * wb + den_t * wt
    num = num_b * wb + num_tw * wt
    row = tl.where(den > 0, num / den, 0.0)
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


def _tail_partials(Q, Kt, Vt, kvh, qpk, hd, scale=0.0625):
    """Tail-split kernel + host. Q (QH,HD) fp16; Kt/Vt (R,KVH,HD) fp32.
    Returns (m, l, acc) QPAD-padded partials."""
    import torch as _t
    dev = Q.device
    R = int(Kt.shape[0])
    qpad = 1 << (qpk - 1).bit_length()
    m = _t.empty((kvh, qpad, 2), dtype=_t.float32, device=dev)
    l = _t.empty_like(m)
    acc = _t.empty((kvh, qpad, 2, hd), dtype=_t.float32, device=dev)
    n_0 = _t.tensor([R], dtype=_t.int32, device=dev)
    _tail_block_kernel[(kvh, 2,)](
        Q, Kt, Vt, m, l, acc, n_0, 10 ** 9, kvh, qpk, qpad, hd, scale,
        num_warps=4)
    return m, l, acc


def cmd_tail():
    import torch as _t
    _t.manual_seed(21)
    kvh, qpk, hd, R = 4, 6, 256, 200
    qh = kvh * qpk
    Q = _t.randn(qh, hd, dtype=_t.float16, device="cuda")
    Kt = _t.randn(R, kvh, hd, dtype=_t.float32, device="cuda")
    Vt = _t.randn(R, kvh, hd, dtype=_t.float32, device="cuda")
    m, l, acc = _tail_partials(Q, Kt, Vt, kvh, qpk, hd)
    m, l, acc = m[:, :qpk], l[:, :qpk], acc[:, :qpk]
    m_all = m.amax(dim=2, keepdim=True)
    e = _t.exp(m - m_all)
    den = (l * e).sum(dim=2)
    num = (acc * e.unsqueeze(-1)).sum(dim=2)
    got = (num / den.unsqueeze(-1)).reshape(qh, hd)
    outs = []
    for h in range(kvh):
        q = Q[h * qpk:(h + 1) * qpk].float()
        s = (q @ Kt[:, h, :].T) * 0.0625
        p = _t.softmax(s, dim=-1)
        outs.append(p @ Vt[:, h, :])
    ref = _t.cat(outs)
    d = (got - ref).abs()
    print(f"tail RMSE={float((d**2).mean().sqrt()):.3e} "
          f"max={float(d.max()):.3e}", flush=True)
    assert float((d**2).mean().sqrt()) < 1e-6
    print("TAIL PASS", flush=True)


def cmd_merge():
    import sys as _s
    _s.path.insert(0, 'eval')
    from _spike2_online import _make_records
    from exllamav3.cache.kvarn import kvarn_make_layout, kvarn_wht_head
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_dequant_groups, kvarn_triton_wht_rows)
    import torch as _t
    _t.manual_seed(7)
    kvh, sl, hd, qpk, Gg = 4, 2, 256, 6, 4
    bits = (4, 4)
    layout = kvarn_make_layout(128, 128, bits[0], bits[1])
    records = _make_records(Gg, kvh, sl, layout, bits[0], bits[1])
    qh = kvh * qpk
    Q = _t.randn(qh, hd, dtype=_t.float16, device="cuda")
    qw = kvarn_triton_wht_rows(Q.float(), hd)
    # Body = groups 0..1 full, tail rows = 128 + 64 synthetic exact rows.
    n = 256 + 64
    ids = _t.tensor([0, 1], dtype=_t.int64, device="cuda")
    nb = 2
    qpad = 8
    dev = "cuda"
    m = _t.empty((kvh, qpad, nb), dtype=_t.float32, device=dev)
    l = _t.empty_like(m)
    acc = _t.empty((kvh, qpad, nb, hd), dtype=_t.float32, device=dev)
    n_0 = _t.tensor([n], dtype=_t.int32, device=dev)
    import _spike3_online as _s3
    _s3._block_kernel_h2[(kvh, nb,)](
        qw, records, records.view(_t.float16), ids, m, l, acc,
        layout.k_payload_off,
        layout.k_s_col_off // 2, layout.k_zp_off // 2,
        layout.k_s_row_off // 2, bits[0],
        layout.v_payload_off,
        layout.v_s_row_off // 2, layout.v_zp_off // 2,
        layout.v_s_col_off // 2, bits[1],
        records.shape[1], records.shape[2],
        kvh, qpk, qpad, hd, nb, 0.0625,
        num_warps=4)
    # NOTE: h2 kernel here is unmasked (spike-era); this test declares
    # all 256 record rows body by construction, plus 192 synthetic tail
    # rows. The merge algebra is position-free (union softmax); position
    # masking is validated separately (production masked kernel PASS).
    # Tail rows: first 128 (sink-ish) + 64 → synthetic exact.
    Kt = _t.randn(192, kvh, hd, dtype=_t.float32, device=dev)
    Vt = _t.randn(192, kvh, hd, dtype=_t.float32, device=dev)
    tm, tl_, tacc = _tail_partials(Q, Kt, Vt, kvh, qpk, hd)
    out = _t.empty((qh, hd), dtype=_t.float32, device=dev)
    _mega_merge_kernel[(qh,)](
        m, l, acc, tm, tl_, tacc, out,
        kvh, qpk, qpad, nb, 1 << (nb - 1).bit_length(),
        hd, sl, 0.7071067811865475,
        num_warps=1)
    # Reference: single softmax over (body dequant rows + tail rows).
    bk, bv = kvarn_triton_dequant_groups(
        records[:2], layout, bits[0], bits[1], kvh, sl, do_wht=False)
    Kb = kvarn_wht_head(bk, hd).reshape(256, kvh, hd)
    Vb = kvarn_wht_head(bv, hd).reshape(256, kvh, hd)
    outs = []
    for h in range(kvh):
        q = Q[h * qpk:(h + 1) * qpk].float()
        Kall = _t.cat([Kb[:, h, :], Kt[:, h, :]])
        Vall = _t.cat([Vb[:, h, :], Vt[:, h, :]])
        s = (q @ Kall.T) * 0.0625
        p = _t.softmax(s, dim=-1)
        outs.append(p @ Vall)
    ref = _t.cat(outs)
    d = (out - ref).abs()
    print(f"merge maxdiff={float(d.max()):.3e} "
          f"RMSE={float((d**2).mean().sqrt()):.3e}", flush=True)
    assert float((d**2).mean().sqrt()) < 1e-6
    print("MERGE PASS", flush=True)




def cmd_probe():
    from exllamav3 import Config, Model, Tokenizer, Cache
    from exllamav3.cache import CacheLayer_kvarn
    from exllamav3.cache.kvarn import kvarn_parse_preset
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_wht_rows)
    from exllamav3.modules.attention_fn.triton_paged import (
        paged_attn_triton_decode)
    from kvarn_microkld import SAMPLER_TEXT, populate
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_qwht, _kvarn_online_block_kernel)
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
    gids = lay0.sealed.nonzero().flatten().to(torch.int64)
    nb = int(gids.numel())
    nbpad = 1 << (nb - 1).bit_length()
    qpad = 8
    dev = "cuda"
    torch.manual_seed(9)
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    L = lay0.layout
    rec_f16 = lay0.records.view(torch.float16)
    # Persistent buffers (production design persists all of these).
    qw = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    qscratch = torch.empty_like(qw)
    m = torch.empty((kvh, qpad, nb), dtype=torch.float32, device=dev)
    l = torch.empty_like(m)
    acc = torch.empty((kvh, qpad, nb, hd), dtype=torch.float32, device=dev)
    out = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    # Tail rows: sink + tail window from exact (all valid here).
    gps = 2
    sink_n, t0 = 128, n - 128
    tm = torch.empty((kvh, qpad, 2), dtype=torch.float32, device=dev)
    tll = torch.empty_like(tm)
    tacc = torch.empty((kvh, qpad, 2, hd), dtype=torch.float32, device=dev)
    n_0 = torch.tensor([n], dtype=torch.int32, device=dev)
    btp = torch.arange(33, dtype=torch.int32, device=dev)
    R = sink_n + (n - t0)

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
    bt = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    se = torch.tensor([NTOK], dtype=torch.int32, device="cuda")
    q1 = torch.randn(1, 1, qh, hd, dtype=torch.float16, device="cuda")
    k1 = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
    v1 = torch.randn(1, 1, kvh, hd, dtype=torch.float16, device="cuda")
    paged_attn_triton_decode(q1, k1, v1, k_cache, v_cache, bt, se,
                             causal=True, softmax_scale=0.0625,
                             softcap=0.0, sinks=None)
    t_fp16 = hot(lambda: paged_attn_triton_decode(
        q1, k1, v1, k_cache, v_cache, bt, se, causal=True,
        softmax_scale=0.0625, softcap=0.0, sinks=None))

    def do_spike():
        kvarn_triton_qwht(Q, qscratch, qw, sl, 0.7071067811865475)
        # Tail gather (per-step production cost).
        pos = torch.cat([torch.arange(sink_n, device=dev),
                         torch.arange(t0, n, device=dev)]).long()
        pages = btp[pos // 256]
        offs = pos % 256
        g = pages * gps + offs // 128
        s = offs % 128
        es = lay0._exact_rev[g.long()].clamp_min(0)
        # Direct indexed-row gather (NOT fancy-full-then-pick: that
        # materializes (R,128,kvh,hd) 8MB temps per side).
        Kt = lay0.exact_k[es, s].float()
        Vt = lay0.exact_v[es, s].float()
        _kvarn_online_block_kernel[(kvh, nb,)](
            qw, lay0.records, rec_f16, gids, m, l, acc,
            L.k_payload_off,
            L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2,
            k_bits,
            L.v_payload_off,
            L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2,
            v_bits,
            lay0.records.shape[1], lay0.records.shape[2], sl,
            kvh, qpk, qpad, hd, nb, 0.0625,
            n_0, sink_n, 128,
            num_warps=4)
        _tail_block_kernel[(kvh, 2,)](
            Q, Kt, Vt, tm, tll, tacc, n_0, sink_n + 128, kvh, qpk, qpad,
            hd, 0.0625,
            num_warps=4)
        _mega_merge_kernel[(qh,)](
            m, l, acc, tm, tll, tacc, out,
            kvh, qpk, qpad, nb, nbpad, hd, sl, 0.7071067811865475,
            num_warps=1)
        return out
    o = do_spike()
    print("spike4 full path ok", flush=True)
    t_spike = hot(do_spike)
    rows_fp16, rows_spike = NTOK, nb * 128 + R
    per_fp16 = t_fp16 / rows_fp16 * 1e6
    per_spike = t_spike / rows_spike * 1e6
    print(f"fp16 paged attn: {t_fp16:.4f} ms/step over {rows_fp16} rows "
          f"({per_fp16:.2f} ns/row)", flush=True)
    print(f"spike4 fused:    {t_spike:.4f} ms/step over {rows_spike} rows "
          f"({per_spike:.2f} ns/row)", flush=True)
    print(f"per-row ratio spike/fp16: {per_spike / per_fp16:.3f} "
          f"(v2prod 1.04, gate <= 1.111)", flush=True)



def cmd_break():
    """Piece-wise timing on the REAL populated layer (settles the 774us
    question): qwht / masked stage-1 / tail gather / tailsplit /
    megamerge, each hot-timed in the same process."""
    from exllamav3 import Config, Model, Tokenizer, Cache
    from exllamav3.cache import CacheLayer_kvarn
    from exllamav3.cache.kvarn import kvarn_parse_preset
    from exllamav3.modules.attention_fn.kvarn_triton import (
        kvarn_triton_qwht, _kvarn_online_block_kernel)
    from kvarn_microkld import SAMPLER_TEXT, populate
    MODEL = ("C:/Users/yoho/Downloads/tabbyAPI/models/"
             "Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3")
    NTOK, CHUNK, ITERS = 8192, 4096, 100
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
    gids = lay0.sealed.nonzero().flatten().to(torch.int64)
    nb = int(gids.numel())
    nbpad = 1 << (nb - 1).bit_length()
    qpad = 8
    dev = "cuda"
    torch.manual_seed(9)
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    L = lay0.layout
    rec_f16 = lay0.records.view(torch.float16)
    qw = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    qscratch = torch.empty_like(qw)
    m = torch.empty((kvh, qpad, nb), dtype=torch.float32, device=dev)
    l = torch.empty_like(m)
    acc = torch.empty((kvh, qpad, nb, hd), dtype=torch.float32, device=dev)
    out = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    tm = torch.empty((kvh, qpad, 2), dtype=torch.float32, device=dev)
    tll = torch.empty_like(tm)
    tacc = torch.empty((kvh, qpad, 2, hd), dtype=torch.float32, device=dev)
    n_0 = torch.tensor([n], dtype=torch.int32, device=dev)
    gps, sink_n, t0 = 2, 128, n - 128
    btp = torch.arange(33, dtype=torch.int32, device=dev)

    def hot(fn, iters=ITERS):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0e = torch.cuda.Event(enable_timing=True)
        t1e = torch.cuda.Event(enable_timing=True)
        t0e.record()
        for _ in range(iters):
            fn()
        t1e.record()
        torch.cuda.synchronize()
        return t0e.elapsed_time(t1e) / iters

    def do_qwht():
        kvarn_triton_qwht(Q, qscratch, qw, sl, 0.7071067811865475)
    qw_ = do_qwht()

    def do_stage1():
        _kvarn_online_block_kernel[(kvh, nb,)](
            qw, lay0.records, rec_f16, gids, m, l, acc,
            L.k_payload_off,
            L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2,
            k_bits,
            L.v_payload_off,
            L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2,
            v_bits,
            lay0.records.shape[1], lay0.records.shape[2], sl,
            kvh, qpk, qpad, hd, nb, 0.0625,
            n_0, sink_n, 128,
            num_warps=4)

    def do_gather():
        pos = torch.cat([torch.arange(sink_n, device=dev),
                         torch.arange(t0, n, device=dev)]).long()
        pages = btp[pos // 256]
        offs = pos % 256
        g = pages * gps + offs // 128
        s = offs % 128
        es = lay0._exact_rev[g.long()].clamp_min(0)
        # Direct indexed-row gather (NOT fancy-full-then-pick: that
        # materializes (R,128,kvh,hd) 8MB temps per side).
        Kt = lay0.exact_k[es, s].float()
        Vt = lay0.exact_v[es, s].float()
        return Kt, Vt
    Kt, Vt = do_gather()

    def do_tailsplit():
        _tail_block_kernel[(kvh, 2,)](
            Q, Kt, Vt, tm, tll, tacc, n_0, sink_n + 128, kvh, qpk, qpad,
            hd, 0.0625,
            num_warps=4)

    def do_merge():
        _mega_merge_kernel[(qh,)](
            m, l, acc, tm, tll, tacc, out,
            kvh, qpk, qpad, nb, nbpad, hd, sl, 0.7071067811865475,
            num_warps=1)

    print(f"qwht:      {hot(do_qwht) * 1e3:7.1f} us", flush=True)
    print(f"stage1:    {hot(do_stage1) * 1e3:7.1f} us", flush=True)
    print(f"gather:    {hot(do_gather) * 1e3:7.1f} us", flush=True)
    print(f"tailsplit: {hot(do_tailsplit) * 1e3:7.1f} us", flush=True)
    print(f"merge:     {hot(do_merge) * 1e3:7.1f} us", flush=True)
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tail"
    {"tail": cmd_tail, "merge": cmd_merge, "probe": cmd_probe, "break": cmd_break}[mode]()
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tail"
    {"tail": cmd_tail, "merge": cmd_merge, "probe": cmd_probe, "break": cmd_break}[mode]()

