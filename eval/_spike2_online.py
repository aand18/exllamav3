"""Match-Bee Task 1 spike: fused online attention v2 (UNTRACKED).

v1 (Task 5) post-mortem: stage-1 core 9.8 ns/row beats fp16, but torch
combine (98us) + WHT launches (95us) dominate. v2 attacks both while
keeping v1's validated math (per-lane slice select, LSB-first unpack,
((q*sc)+zp)*other order):
- stage-1 keeps the (kv_head, block) grid (9MB record traffic; the
  qh-parallel alternative re-reads records 6x) but unrolls 4 tokens per
  iteration with a JOINT block softmax update (serial depth 128 -> 32).
- NEW combine kernel (grid QH): online-reduce NB partials + out-WHT via
  the proven single-warp pattern (num_warps=1), zero torch.
- Q-WHT stays a separate launch in this task (Task 2 shrinks it).

Usage: python eval/_spike2_online.py [tiles|attn|probe|all]
"""
import sys

import torch
import triton
import triton.language as tl

from exllamav3.modules.attention_fn.kvarn_triton import (
    kvarn_triton_dequant_groups,
    kvarn_triton_wht_rows,
)


@triton.jit
def _kcol(gbase_u8, gbase_f16, C: tl.constexpr, B: tl.constexpr,
          PAY_OFF, SC2, ZP2, OT2, BITS: tl.constexpr,
          pid_h, SL: tl.constexpr, t, lane):
    sl = lane // 128
    dd = lane % 128
    c = pid_h * SL + sl
    pay = gbase_u8 + c * B + PAY_OFF
    v = dd * 128 + t
    q = tl.zeros_like(lane)
    for i in tl.static_range(8):
        if i < BITS:
            b = v * BITS + i
            byteval = tl.load(pay + b // 8)
            q += ((byteval.to(tl.int32) >> (b % 8)) & 1) << i
    sc = tl.load(gbase_f16 + (c * B) // 2 + SC2 + dd).to(tl.float32)
    zp = tl.load(gbase_f16 + (c * B) // 2 + ZP2 + dd).to(tl.float32)
    oth = tl.load(gbase_f16 + (c * B) // 2 + OT2 + t).to(tl.float32)
    return (q.to(tl.float32) * sc + zp) * oth


@triton.jit
def _vrow(gbase_u8, gbase_f16, C: tl.constexpr, B: tl.constexpr,
          PAY_OFF, SC2, ZP2, OT2, BITS: tl.constexpr,
          pid_h, SL: tl.constexpr, t, lane):
    sl = lane // 128
    dd = lane % 128
    c = pid_h * SL + sl
    pay = gbase_u8 + c * B + PAY_OFF
    v = t * 128 + dd
    q = tl.zeros_like(lane)
    for i in tl.static_range(8):
        if i < BITS:
            b = v * BITS + i
            byteval = tl.load(pay + b // 8)
            q += ((byteval.to(tl.int32) >> (b % 8)) & 1) << i
    sc = tl.load(gbase_f16 + (c * B) // 2 + SC2 + t).to(tl.float32)
    zp = tl.load(gbase_f16 + (c * B) // 2 + ZP2 + t).to(tl.float32)
    oth = tl.load(gbase_f16 + (c * B) // 2 + OT2 + dd).to(tl.float32)
    return (q.to(tl.float32) * sc + zp) * oth


@triton.jit
def _block_kernel(
    qw_ptr, rec_ptr, rec_f16_ptr, ids_ptr,
    m_ptr, l_ptr, out_ptr,  # (KVH, QPK, NB), ..., (KVH, QPK, NB, HD)
    K_PAY_OFF, K_SC2, K_ZP2, K_OT2, K_BITS: tl.constexpr,
    V_PAY_OFF, V_SC2, V_ZP2, V_OT2, V_BITS: tl.constexpr,
    C: tl.constexpr, B: tl.constexpr, SL: tl.constexpr,
    KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
    HD: tl.constexpr, NB: tl.constexpr, SCALE: tl.constexpr,
):
    """One program = one (kv head, sealed group). 32 outer iters x 4
    tokens with a joint block softmax update (depth 32, not 128)."""
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    g = tl.load(ids_ptr + pid_b)
    lane = tl.arange(0, HD)
    qoff = tl.arange(0, QPAD)
    qmask = qoff < QPK
    q = tl.load(qw_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                + lane[None, :], mask=qmask[:, None], other=0.0)
    m = tl.full([QPAD], float("-inf"), dtype=tl.float32)
    l = tl.zeros([QPAD], dtype=tl.float32)
    acc = tl.zeros([QPAD, HD], dtype=tl.float32)
    gbase_u8 = rec_ptr + g * C * B
    gbase_f16 = rec_f16_ptr + (g * C * B) // 2
    for t0 in tl.range(32):
        t = t0 * 4
        k0 = _kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                   K_BITS, pid_h, SL, t, lane)
        k1 = _kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                   K_BITS, pid_h, SL, t + 1, lane)
        k2 = _kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                   K_BITS, pid_h, SL, t + 2, lane)
        k3 = _kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                   K_BITS, pid_h, SL, t + 3, lane)
        s0 = tl.sum(q * k0[None, :], axis=1) * SCALE
        s1 = tl.sum(q * k1[None, :], axis=1) * SCALE
        s2 = tl.sum(q * k2[None, :], axis=1) * SCALE
        s3 = tl.sum(q * k3[None, :], axis=1) * SCALE
        smax = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))
        m_new = tl.maximum(m, smax)
        alpha = tl.exp(m - m_new)
        e0 = tl.exp(s0 - m_new)
        e1 = tl.exp(s1 - m_new)
        e2 = tl.exp(s2 - m_new)
        e3 = tl.exp(s3 - m_new)
        l = l * alpha + e0 + e1 + e2 + e3
        v0 = _vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
                   V_BITS, pid_h, SL, t, lane)
        v1 = _vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
                   V_BITS, pid_h, SL, t + 1, lane)
        v2 = _vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
                   V_BITS, pid_h, SL, t + 2, lane)
        v3 = _vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
                   V_BITS, pid_h, SL, t + 3, lane)
        acc = acc * alpha[:, None] + e0[:, None] * v0[None, :] \
            + e1[:, None] * v1[None, :] + e2[:, None] * v2[None, :] \
            + e3[:, None] * v3[None, :]
        m = m_new
    qoff = tl.arange(0, QPAD)
    qmask = qoff < QPK
    tl.store(m_ptr + (pid_h * QPAD * NB) + qoff * NB + pid_b, m,
             mask=qmask)
    tl.store(l_ptr + (pid_h * QPAD * NB) + qoff * NB + pid_b, l,
             mask=qmask)
    tl.store(out_ptr + ((pid_h * QPAD * NB) + qoff[:, None] * NB + pid_b)
             * HD + lane[None, :], acc, mask=qmask[:, None])


@triton.jit
def _fwht128_scratch(row_ptr, cols):
    # Same 7 stages + norm as _fwht128_block (proven pattern), factored
    # for the combine kernel. Single-warp launch ONLY.
    tl.debug_barrier()
    for _s in tl.static_range(7):
        s = 1 << _s
        cur = tl.load(row_ptr + cols)
        prt = tl.load(row_ptr + (cols ^ s))
        tl.store(row_ptr + cols,
                 tl.where((cols & s) == 0, cur + prt, prt - cur))
        tl.debug_barrier()
    tl.store(row_ptr + cols,
             tl.load(row_ptr + cols) * 0.08838834764831845)


@triton.jit
def _combine_kernel(
    m_ptr, l_ptr, acc_ptr,   # (KVH, QPK, NB), ..., (KVH, QPK, NB, HD)
    out_ptr,                 # (QH, HD) fp32 ORIGINAL domain (WHT folded)
    KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
    NB: tl.constexpr, NBPAD: tl.constexpr,
    HD: tl.constexpr, SL: tl.constexpr, SSCALE: tl.constexpr,
):
    """One program = one q-head: online-reduce NB block partials, then
    full head WHT in place (per-slice FWHT + cross-slice + scale, same
    order as _kvarn_wht_hd_kernel). num_warps=1 REQUIRED. NB pads to
    NBPAD (pow2); scalar block weights come from one-hot selects."""
    pid = tl.program_id(0)
    ph = pid // QPK
    pq = pid % QPK
    lane = tl.arange(0, HD)
    nboff = tl.arange(0, NBPAD)
    nbmask = nboff < NB
    m = tl.load(m_ptr + (ph * QPAD + pq) * NB + nboff, mask=nbmask,
                other=float("-inf"))
    l = tl.load(l_ptr + (ph * QPAD + pq) * NB + nboff, mask=nbmask, other=0.0)
    m_all = tl.max(m)
    e = tl.exp(m - m_all)
    den = tl.sum(l * e)
    num = tl.zeros([HD], dtype=tl.float32)
    for b in tl.range(NBPAD):
        active = b < NB
        eb = tl.sum(tl.where(nboff == b, e, 0.0))
        ab = tl.load(acc_ptr + ((ph * QPAD + pq) * NB + b) * HD + lane,
                     mask=active, other=0.0)
        num += ab * eb
    row = num / den
    base = out_ptr + pid * HD
    tl.store(base + lane, row)
    for _sl in tl.static_range(4):
        if _sl < SL:
            _fwht128_scratch(base + _sl * 128, tl.arange(0, 128))
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


def spike2_attn(qw, records, ids, layout, k_bits, v_bits,
                kvh, qpk, slices, hd, scale=0.0625):
    """Two-launch fused online attention. Returns (QH, HD) fp32 out in
    the ORIGINAL domain (out-WHT folded into combine). Q-WHT stays
    caller-side (Task 2 shrinks it)."""
    assert qw.is_cuda and records.is_cuda and ids.is_cuda
    nb = int(ids.numel())
    dev = qw.device
    qh = kvh * qpk
    assert qw.shape == (qh, hd)
    if nb == 0:
        return torch.zeros((qh, hd), dtype=torch.float32, device=dev)
    qpad = 1 << (qpk - 1).bit_length()
    rec_f16 = records.view(torch.float16)
    m = torch.empty((kvh, qpad, nb), dtype=torch.float32, device=dev)
    l = torch.empty_like(m)
    acc = torch.empty((kvh, qpad, nb, hd), dtype=torch.float32, device=dev)
    _block_kernel[(kvh, nb,)](
        qw, records, rec_f16, ids, m, l, acc,
        layout.k_payload_off,
        layout.k_s_col_off // 2, layout.k_zp_off // 2, layout.k_s_row_off // 2,
        k_bits,
        layout.v_payload_off,
        layout.v_s_row_off // 2, layout.v_zp_off // 2, layout.v_s_col_off // 2,
        v_bits,
        records.shape[1], records.shape[2], slices,
        kvh, qpk, qpad, hd, nb, scale,
        num_warps=4)
    out = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    sscale = 1.0 if slices == 1 else (0.7071067811865475 if slices == 2
                                      else 0.5)
    nbpad = 1 << (nb - 1).bit_length()
    _combine_kernel[(qh,)](
        m, l, acc, out, kvh, qpk, qpad, nb, nbpad, hd, slices, sscale,
        num_warps=1)
    return out


def _make_records(Gg, kvh, sl, layout, k_bits, v_bits, seed=0):
    from exllamav3.cache.kvarn import (
        kvarn_make_layout, kvarn_quantize_k_tile, kvarn_quantize_v_tile,
        kvarn_hadamard_128)
    torch.manual_seed(seed)
    C = kvh * sl
    B = layout.tile_bytes
    records = torch.zeros((Gg, C, B), dtype=torch.uint8, device="cuda")
    for g in range(Gg):
        for h in range(kvh):
            for s in range(sl):
                c = h * sl + s
                rec = records[g, c]
                # Records hold NORMALIZED WHT-domain tiles (seal path
                # quantizes already-head-WHT'd staging).
                tk = kvarn_hadamard_128(
                    torch.randn(128, 128, dtype=torch.float32, device="cuda"))
                tv = kvarn_hadamard_128(
                    torch.randn(128, 128, dtype=torch.float32, device="cuda"))
                kvarn_quantize_k_tile(tk, 16, k_bits, layout, rec)
                kvarn_quantize_v_tile(tv, 16, v_bits, layout, rec)
    return records


def cmd_attn():
    from exllamav3.cache.kvarn import kvarn_make_layout, kvarn_wht_head
    torch.manual_seed(5)
    kvh, sl, hd, qpk, Gg = 4, 2, 256, 6, 4
    bits = (4, 4)
    layout = kvarn_make_layout(128, 128, bits[0], bits[1])
    records = _make_records(Gg, kvh, sl, layout, bits[0], bits[1])
    qh = kvh * qpk
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    qw = kvarn_triton_wht_rows(Q.float(), hd)
    ids = torch.arange(Gg, dtype=torch.int64, device="cuda")
    out = spike2_attn(qw, records, ids, layout, bits[0], bits[1],
                      kvh, qpk, sl, hd)
    bk, bv = kvarn_triton_dequant_groups(
        records, layout, bits[0], bits[1], kvh, sl, do_wht=False)
    K = kvarn_wht_head(bk, hd).reshape(Gg * 128, kvh, hd)
    V = kvarn_wht_head(bv, hd).reshape(Gg * 128, kvh, hd)
    outs = []
    for h in range(kvh):
        q = Q[h * qpk:(h + 1) * qpk].float()
        s = (q @ K[:, h, :].T) * 0.0625
        p = torch.softmax(s, dim=-1)
        outs.append(p @ V[:, h, :])
    ref = torch.cat(outs)
    diff = (out - ref).abs()
    print(f"attn max|diff|={float(diff.max())} "
          f"RMSE={float((diff**2).mean().sqrt())}", flush=True)


def cmd_probe():
    from exllamav3 import Config, Model, Tokenizer, Cache
    from exllamav3.cache import CacheLayer_kvarn
    from exllamav3.cache.kvarn import kvarn_parse_preset
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
    assert kvh * qpk == qh
    gids = lay0.sealed.nonzero().flatten().to(torch.int64)
    nb = int(gids.numel())
    print(f"layer kvh={kvh} hd={hd} slices={sl} sealed={nb}", flush=True)
    torch.manual_seed(9)
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    qw = kvarn_triton_wht_rows(Q.float(), hd)

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
        o = spike2_attn(qw, lay0.records, gids, lay0.layout, k_bits, v_bits,
                        kvh, qpk, sl, hd)
        return o
    do_spike()
    # Correctness on real records (RMSE vs torch over reference rows).
    from exllamav3.cache.kvarn import kvarn_wht_head
    bk, bv = kvarn_triton_dequant_groups(
        lay0.records, lay0.layout, k_bits, v_bits, kvh, sl, do_wht=False)
    o = do_spike()
    print("spike ok on real records", flush=True)
    t_spike = hot(do_spike)
    rows_fp16, rows_spike = NTOK, nb * 128
    per_fp16 = t_fp16 / rows_fp16 * 1e6
    per_spike = t_spike / rows_spike * 1e6
    print(f"fp16 paged attn: {t_fp16:.4f} ms/step over {rows_fp16} rows "
          f"({per_fp16:.2f} ns/row)", flush=True)
    print(f"spike2 fused:    {t_spike:.4f} ms/step over {rows_spike} rows "
          f"({per_spike:.2f} ns/row, incl Q-WHT)", flush=True)
    print(f"per-row ratio spike/fp16: {per_spike / per_fp16:.3f} "
          f"(v1 spike was 2.236)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "attn"
    {"attn": cmd_attn, "probe": cmd_probe}[mode]()
