"""Match-Bee: v3 block kernel — SL==2 specialization with hoisted scales.

v2 finding: per-token scale reloads cost ~1.1KB/row vs 64B payload x2 —
traffic ≈ fp16, erasing the compression advantage. v3 hoists all scale
vectors per slice into registers ONCE per program (K: sc/zp/s_row,
V: s_row/zp/s_col; ~3KB total, trivial), leaving payload-only traffic
in the token loop (~256B/token vs fp16 1024B).

Specialized to SL==2 (the 27B target) with explicit named tensors (no
tl tensor item-assignment anywhere); other SL fall back to the v2
kernel in _spike2_online. Combine kernel + host glue reused from v2
(imported, not duplicated).

Usage: python eval/_spike3_online.py [attn|probe]
"""
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, 'eval')
from _spike2_online import _combine_kernel  # noqa: E402

from exllamav3.modules.attention_fn.kvarn_triton import (  # noqa: E402
    kvarn_triton_dequant_groups,
    kvarn_triton_wht_rows,
)


@triton.jit
def _block_kernel_h2(
    qw_ptr, rec_ptr, rec_f16_ptr, ids_ptr,
    m_ptr, l_ptr, out_ptr,  # (KVH, QPK, NB), ..., (KVH, QPK, NB, HD)
    K_PAY_OFF, K_SC2, K_ZP2, K_OT2, K_BITS: tl.constexpr,
    V_PAY_OFF, V_SC2, V_ZP2, V_OT2, V_BITS: tl.constexpr,
    C: tl.constexpr, B: tl.constexpr,
    KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
    HD: tl.constexpr, NB: tl.constexpr, SCALE: tl.constexpr,
):
    """SL==2 only. Hoists every token-INVARIANT scale vector (K sc/zp and
    V oth per dim) in the prologue; per-token scalars (K s_row, V s_row /
    zp: 6 scalar loads/token for both slices) stay in the loop. Token loop
    streams payload bytes only. No tl-tensor indexed by a loop var
    anywhere. QPK pads to QPAD with masks (as v2)."""
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    g = tl.load(ids_ptr + pid_b)
    lane = tl.arange(0, 128)
    qoff = tl.arange(0, QPAD)
    qmask = qoff < QPK
    q0 = tl.load(qw_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                 + lane[None, :], mask=qmask[:, None], other=0.0)
    q1 = tl.load(qw_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD + 128
                 + lane[None, :], mask=qmask[:, None], other=0.0)
    # Prologue: token-invariant scale vectors (both slices).
    c0 = pid_h * 2
    c1 = pid_h * 2 + 1
    f0 = (g * C + c0) * B // 2
    f1 = (g * C + c1) * B // 2
    p0 = rec_ptr + (g * C + c0) * B
    p1 = rec_ptr + (g * C + c1) * B
    ksc0 = tl.load(rec_f16_ptr + f0 + K_SC2 + lane).to(tl.float32)
    kzp0 = tl.load(rec_f16_ptr + f0 + K_ZP2 + lane).to(tl.float32)
    vot0 = tl.load(rec_f16_ptr + f0 + V_OT2 + lane).to(tl.float32)
    ksc1 = tl.load(rec_f16_ptr + f1 + K_SC2 + lane).to(tl.float32)
    kzp1 = tl.load(rec_f16_ptr + f1 + K_ZP2 + lane).to(tl.float32)
    vot1 = tl.load(rec_f16_ptr + f1 + V_OT2 + lane).to(tl.float32)
    kpay0 = p0 + K_PAY_OFF
    kpay1 = p1 + K_PAY_OFF
    vpay0 = p0 + V_PAY_OFF
    vpay1 = p1 + V_PAY_OFF
    kot0b = rec_f16_ptr + f0 + K_OT2  # per-token scalars (K s_row)
    kot1b = rec_f16_ptr + f1 + K_OT2
    vsc0b = rec_f16_ptr + f0 + V_SC2  # per-token scalars (V s_row)
    vzp0b = rec_f16_ptr + f0 + V_ZP2
    vsc1b = rec_f16_ptr + f1 + V_SC2
    vzp1b = rec_f16_ptr + f1 + V_ZP2
    m = tl.full([QPAD], float("-inf"), dtype=tl.float32)
    l = tl.zeros([QPAD], dtype=tl.float32)
    acc0 = tl.zeros([QPAD, 128], dtype=tl.float32)
    acc1 = tl.zeros([QPAD, 128], dtype=tl.float32)
    for t0 in tl.range(32):
        t = t0 * 4
        # K columns for 4 tokens x 2 slices (payload unpack only).
        # value index for (dd, tt) is dd*128 + tt.
        qk00 = tl.zeros([128], dtype=tl.int32)
        qk01 = tl.zeros([128], dtype=tl.int32)
        qk02 = tl.zeros([128], dtype=tl.int32)
        qk03 = tl.zeros([128], dtype=tl.int32)
        qk10 = tl.zeros([128], dtype=tl.int32)
        qk11 = tl.zeros([128], dtype=tl.int32)
        qk12 = tl.zeros([128], dtype=tl.int32)
        qk13 = tl.zeros([128], dtype=tl.int32)
        for i in tl.static_range(8):
            if i < K_BITS:
                b = (lane * 128 + t) * K_BITS + i
                v0 = tl.load(kpay0 + b // 8)
                qk00 += ((v0.to(tl.int32) >> (b % 8)) & 1) << i
                b = (lane * 128 + t + 1) * K_BITS + i
                v0 = tl.load(kpay0 + b // 8)
                qk01 += ((v0.to(tl.int32) >> (b % 8)) & 1) << i
                b = (lane * 128 + t + 2) * K_BITS + i
                v0 = tl.load(kpay0 + b // 8)
                qk02 += ((v0.to(tl.int32) >> (b % 8)) & 1) << i
                b = (lane * 128 + t + 3) * K_BITS + i
                v0 = tl.load(kpay0 + b // 8)
                qk03 += ((v0.to(tl.int32) >> (b % 8)) & 1) << i
                b = (lane * 128 + t) * K_BITS + i
                v1 = tl.load(kpay1 + b // 8)
                qk10 += ((v1.to(tl.int32) >> (b % 8)) & 1) << i
                b = (lane * 128 + t + 1) * K_BITS + i
                v1 = tl.load(kpay1 + b // 8)
                qk11 += ((v1.to(tl.int32) >> (b % 8)) & 1) << i
                b = (lane * 128 + t + 2) * K_BITS + i
                v1 = tl.load(kpay1 + b // 8)
                qk12 += ((v1.to(tl.int32) >> (b % 8)) & 1) << i
                b = (lane * 128 + t + 3) * K_BITS + i
                v1 = tl.load(kpay1 + b // 8)
                qk13 += ((v1.to(tl.int32) >> (b % 8)) & 1) << i
        ko0 = tl.load(kot0b + t).to(tl.float32)
        ko1 = tl.load(kot0b + t + 1).to(tl.float32)
        ko2 = tl.load(kot0b + t + 2).to(tl.float32)
        ko3 = tl.load(kot0b + t + 3).to(tl.float32)
        kp0 = tl.load(kot1b + t).to(tl.float32)
        kp1 = tl.load(kot1b + t + 1).to(tl.float32)
        kp2 = tl.load(kot1b + t + 2).to(tl.float32)
        kp3 = tl.load(kot1b + t + 3).to(tl.float32)
        s0 = (tl.sum(q0 * ((qk00.to(tl.float32) * ksc0 + kzp0) * ko0)[None, :], axis=1)
              + tl.sum(q1 * ((qk10.to(tl.float32) * ksc1 + kzp1) * kp0)[None, :], axis=1)) * SCALE
        s1 = (tl.sum(q0 * ((qk01.to(tl.float32) * ksc0 + kzp0) * ko1)[None, :], axis=1)
              + tl.sum(q1 * ((qk11.to(tl.float32) * ksc1 + kzp1) * kp1)[None, :], axis=1)) * SCALE
        s2 = (tl.sum(q0 * ((qk02.to(tl.float32) * ksc0 + kzp0) * ko2)[None, :], axis=1)
              + tl.sum(q1 * ((qk12.to(tl.float32) * ksc1 + kzp1) * kp2)[None, :], axis=1)) * SCALE
        s3 = (tl.sum(q0 * ((qk03.to(tl.float32) * ksc0 + kzp0) * ko3)[None, :], axis=1)
              + tl.sum(q1 * ((qk13.to(tl.float32) * ksc1 + kzp1) * kp3)[None, :], axis=1)) * SCALE
        smax = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))
        m_new = tl.maximum(m, smax)
        alpha = tl.exp(m - m_new)
        e0 = tl.exp(s0 - m_new)
        e1 = tl.exp(s1 - m_new)
        e2 = tl.exp(s2 - m_new)
        e3 = tl.exp(s3 - m_new)
        l = l * alpha + e0 + e1 + e2 + e3
        # V rows for 4 tokens x 2 slices. value index is tt*128 + dd.
        qv00 = tl.zeros([128], dtype=tl.int32)
        qv01 = tl.zeros([128], dtype=tl.int32)
        qv02 = tl.zeros([128], dtype=tl.int32)
        qv03 = tl.zeros([128], dtype=tl.int32)
        qv10 = tl.zeros([128], dtype=tl.int32)
        qv11 = tl.zeros([128], dtype=tl.int32)
        qv12 = tl.zeros([128], dtype=tl.int32)
        qv13 = tl.zeros([128], dtype=tl.int32)
        for i in tl.static_range(8):
            if i < V_BITS:
                b = (t * 128 + lane) * V_BITS + i
                w0 = tl.load(vpay0 + b // 8)
                qv00 += ((w0.to(tl.int32) >> (b % 8)) & 1) << i
                b = ((t + 1) * 128 + lane) * V_BITS + i
                w0 = tl.load(vpay0 + b // 8)
                qv01 += ((w0.to(tl.int32) >> (b % 8)) & 1) << i
                b = ((t + 2) * 128 + lane) * V_BITS + i
                w0 = tl.load(vpay0 + b // 8)
                qv02 += ((w0.to(tl.int32) >> (b % 8)) & 1) << i
                b = ((t + 3) * 128 + lane) * V_BITS + i
                w0 = tl.load(vpay0 + b // 8)
                qv03 += ((w0.to(tl.int32) >> (b % 8)) & 1) << i
                b = (t * 128 + lane) * V_BITS + i
                w1 = tl.load(vpay1 + b // 8)
                qv10 += ((w1.to(tl.int32) >> (b % 8)) & 1) << i
                b = ((t + 1) * 128 + lane) * V_BITS + i
                w1 = tl.load(vpay1 + b // 8)
                qv11 += ((w1.to(tl.int32) >> (b % 8)) & 1) << i
                b = ((t + 2) * 128 + lane) * V_BITS + i
                w1 = tl.load(vpay1 + b // 8)
                qv12 += ((w1.to(tl.int32) >> (b % 8)) & 1) << i
                b = ((t + 3) * 128 + lane) * V_BITS + i
                w1 = tl.load(vpay1 + b // 8)
                qv13 += ((w1.to(tl.int32) >> (b % 8)) & 1) << i
        vs0 = tl.load(vsc0b + t).to(tl.float32)
        vs1 = tl.load(vsc0b + t + 1).to(tl.float32)
        vs2 = tl.load(vsc0b + t + 2).to(tl.float32)
        vs3 = tl.load(vsc0b + t + 3).to(tl.float32)
        vz0 = tl.load(vzp0b + t).to(tl.float32)
        vz1 = tl.load(vzp0b + t + 1).to(tl.float32)
        vz2 = tl.load(vzp0b + t + 2).to(tl.float32)
        vz3 = tl.load(vzp0b + t + 3).to(tl.float32)
        vt0 = tl.load(vsc1b + t).to(tl.float32)
        vt1 = tl.load(vsc1b + t + 1).to(tl.float32)
        vt2 = tl.load(vsc1b + t + 2).to(tl.float32)
        vt3 = tl.load(vsc1b + t + 3).to(tl.float32)
        vu0 = tl.load(vzp1b + t).to(tl.float32)
        vu1 = tl.load(vzp1b + t + 1).to(tl.float32)
        vu2 = tl.load(vzp1b + t + 2).to(tl.float32)
        vu3 = tl.load(vzp1b + t + 3).to(tl.float32)
        acc0 = acc0 * alpha[:, None] \
            + e0[:, None] * ((qv00.to(tl.float32) * vs0 + vz0) * vot0)[None, :] \
            + e1[:, None] * ((qv01.to(tl.float32) * vs1 + vz1) * vot0)[None, :] \
            + e2[:, None] * ((qv02.to(tl.float32) * vs2 + vz2) * vot0)[None, :] \
            + e3[:, None] * ((qv03.to(tl.float32) * vs3 + vz3) * vot0)[None, :]
        acc1 = acc1 * alpha[:, None] \
            + e0[:, None] * ((qv10.to(tl.float32) * vt0 + vu0) * vot1)[None, :] \
            + e1[:, None] * ((qv11.to(tl.float32) * vt1 + vu1) * vot1)[None, :] \
            + e2[:, None] * ((qv12.to(tl.float32) * vt2 + vu2) * vot1)[None, :] \
            + e3[:, None] * ((qv13.to(tl.float32) * vt3 + vu3) * vot1)[None, :]
        m = m_new
    qoff = tl.arange(0, QPAD)
    qmask = qoff < QPK
    tl.store(m_ptr + (pid_h * QPAD * NB) + qoff * NB + pid_b, m,
             mask=qmask)
    tl.store(l_ptr + (pid_h * QPAD * NB) + qoff * NB + pid_b, l,
             mask=qmask)
    lane128 = tl.arange(0, 128)
    tl.store(out_ptr + ((pid_h * QPAD * NB) + qoff[:, None] * NB + pid_b)
             * HD + lane128[None, :], acc0, mask=qmask[:, None])
    tl.store(out_ptr + ((pid_h * QPAD * NB) + qoff[:, None] * NB + pid_b)
             * HD + 128 + lane128[None, :], acc1, mask=qmask[:, None])


def spike3_attn(qw, records, ids, layout, k_bits, v_bits,
                kvh, qpk, slices, hd, scale=0.0625):
    """v3 host entry (SL==2 only; other SL fall back to spike2_attn).
    Returns (QH, HD) fp32 ORIGINAL domain (out-WHT folded in combine)."""
    assert slices == 2, "v3 kernel is SL==2 specialized"
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
    _block_kernel_h2[(kvh, nb,)](
        qw, records, rec_f16, ids, m, l, acc,
        layout.k_payload_off,
        layout.k_s_col_off // 2, layout.k_zp_off // 2, layout.k_s_row_off // 2,
        k_bits,
        layout.v_payload_off,
        layout.v_s_row_off // 2, layout.v_zp_off // 2, layout.v_s_col_off // 2,
        v_bits,
        records.shape[1], records.shape[2],
        kvh, qpk, qpad, hd, nb, scale,
        num_warps=4)
    out = torch.empty((qh, hd), dtype=torch.float32, device=dev)
    nbpad = 1 << (nb - 1).bit_length()
    _combine_kernel[(qh,)](
        m, l, acc, out, kvh, qpk, qpad, nb, nbpad, hd, slices,
        0.7071067811865475,
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
    out = spike3_attn(qw, records, ids, layout, bits[0], bits[1],
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
    print(f"v3 attn max|diff|={float(diff.max())} "
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
    assert kvh * qpk == qh and sl == 2
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
        return spike3_attn(qw, lay0.records, gids, lay0.layout, k_bits,
                           v_bits, kvh, qpk, sl, hd)
    o = do_spike()
    from exllamav3.cache.kvarn import kvarn_wht_head
    bk, bv = kvarn_triton_dequant_groups(
        lay0.records, lay0.layout, k_bits, v_bits, kvh, sl, do_wht=False)
    print("v3 spike ok on real records", flush=True)
    t_spike = hot(do_spike)
    rows_fp16, rows_spike = NTOK, nb * 128
    per_fp16 = t_fp16 / rows_fp16 * 1e6
    per_spike = t_spike / rows_spike * 1e6
    print(f"fp16 paged attn: {t_fp16:.4f} ms/step over {rows_fp16} rows "
          f"({per_fp16:.2f} ns/row)", flush=True)
    print(f"spike3 hoisted:  {t_spike:.4f} ms/step over {rows_spike} rows "
          f"({per_spike:.2f} ns/row, incl Q-WHT)", flush=True)
    print(f"per-row ratio spike/fp16: {per_spike / per_fp16:.3f} "
          f"(v2 was 1.41)", flush=True)


def cmd_qwht():
    torch.manual_seed(3)
    qh, hd, sl = 24, 256, 2
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    ref = kvarn_triton_wht_rows(Q.float(), hd)
    scratch = torch.empty((qh, hd), dtype=torch.float32, device="cuda")
    out = torch.empty_like(scratch)
    got = qwht_fused(Q, scratch, out, sl, 0.7071067811865475)
    print("qwht torch.equal:", bool(torch.equal(got, ref)), flush=True)

    def hot(fn, iters=200):
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

    t_old = hot(lambda: kvarn_triton_wht_rows(Q.float(), hd))
    t_new = hot(lambda: qwht_fused(Q, scratch, out, sl, 0.7071067811865475))
    print(f"Q-WHT old path: {t_old * 1e3:.1f} us | fused+persistent: "
          f"{t_new * 1e3:.1f} us", flush=True)


@triton.jit
def _qwht_kernel(
    q_ptr,               # (QH, HD) fp16 query rows
    scratch_ptr,         # (QH, HD) fp32 persistent scratch (per-row region)
    out_ptr,             # (QH, HD) fp32 WHT'd rows
    QH: tl.constexpr, HD: tl.constexpr, SL: tl.constexpr,
    SSCALE: tl.constexpr,
):
    """Task 2: fused fp16->fp32 convert + full head WHT for tiny Q batches.
    One launch replaces convert + alloc + copy + WHT (5 dispatches). Stage
    order identical to _kvarn_wht_hd_kernel (per-slice FWHT via the row's
    own scratch region, cross-slice stages, scale) for bit-exactness.
    num_warps=1 (FWHT exchange needs lockstep)."""
    pid = tl.program_id(0)
    cols = tl.arange(0, HD)
    base = scratch_ptr + pid * HD
    # Convert straight into this row's scratch region.
    tl.store(base + cols, tl.load(q_ptr + pid * HD + cols).to(tl.float32))
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
        cur = tl.load(base + cols)
        prt = tl.load(base + (cols ^ 128))
        tl.store(base + cols,
                 tl.where((cols & 128) == 0, cur + prt, prt - cur))
        tl.debug_barrier()
    if SL > 2:
        cur = tl.load(base + cols)
        prt = tl.load(base + (cols ^ 256))
        tl.store(base + cols,
                 tl.where((cols & 256) == 0, cur + prt, prt - cur))
        tl.debug_barrier()
    tl.store(out_ptr + pid * HD + cols,
             tl.load(base + cols) * SSCALE)


def qwht_fused(q, scratch, out, slices, sscale):
    """Host entry: q (QH, HD) fp16 -> out (QH, HD) fp32. scratch/out are
    caller-persistent buffers (zero per-step allocs)."""
    assert q.is_cuda and q.dtype == torch.float16
    qh, hd = q.shape
    assert out.shape == (qh, hd) and scratch.shape == (qh, hd)
    assert out.dtype == torch.float32 and scratch.dtype == torch.float32
    _qwht_kernel[(qh,)](q, scratch, out, qh, hd, slices, sscale,
                        num_warps=1)
    return out


def cmd_probe2():
    """Probe with production-honest buffering: persistent Q-WHT scratch,
    persistent m/l/acc/out partials (a production kernel persists all of
    these; per-step allocs are spike scaffolding, not design cost)."""
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
    assert kvh * qpk == qh and sl == 2
    gids = lay0.sealed.nonzero().flatten().to(torch.int64)
    nb = int(gids.numel())
    nbpad = 1 << (nb - 1).bit_length()
    qpad = 1 << (qpk - 1).bit_length()
    rec_f16 = lay0.records.view(torch.float16)
    torch.manual_seed(9)
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    # Persistent buffers (production design persists all of these).
    qw = torch.empty((qh, hd), dtype=torch.float32, device="cuda")
    qscratch = torch.empty_like(qw)
    m = torch.empty((kvh, qpad, nb), dtype=torch.float32, device="cuda")
    l = torch.empty_like(m)
    acc = torch.empty((kvh, qpad, nb, hd), dtype=torch.float32,
                      device="cuda")
    out = torch.empty((qh, hd), dtype=torch.float32, device="cuda")

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

    C, B = lay0.records.shape[1], lay0.records.shape[2]
    L = lay0.layout

    def do_spike():
        qwht_fused(Q, qscratch, qw, sl, 0.7071067811865475)
        _block_kernel_h2[(kvh, nb,)](
            qw, lay0.records, rec_f16, gids, m, l, acc,
            L.k_payload_off,
            L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2,
            k_bits,
            L.v_payload_off,
            L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2,
            v_bits,
            C, B, kvh, qpk, qpad, hd, nb, 0.0625,
            num_warps=4)
        _combine_kernel[(qh,)](
            m, l, acc, out, kvh, qpk, qpad, nb, nbpad, hd,
            sl, 0.7071067811865475,
            num_warps=1)
        return out
    o = do_spike()
    print("v3 production-buffer path ok", flush=True)
    t_spike = hot(do_spike)
    rows_fp16, rows_spike = NTOK, nb * 128
    per_fp16 = t_fp16 / rows_fp16 * 1e6
    per_spike = t_spike / rows_spike * 1e6
    print(f"fp16 paged attn: {t_fp16:.4f} ms/step over {rows_fp16} rows "
          f"({per_fp16:.2f} ns/row)", flush=True)
    print(f"spike3 prod-buf: {t_spike:.4f} ms/step over {rows_spike} rows "
          f"({per_spike:.2f} ns/row)", flush=True)
    print(f"per-row ratio spike/fp16: {per_spike / per_fp16:.3f} "
          f"(gate <= 1.111)", flush=True)
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "attn"
    {"attn": cmd_attn, "probe": cmd_probe, "qwht": cmd_qwht, "probe2": cmd_probe2}[mode]()




