"""Match-Bee Task 3 spike6: MMA-dot single-kernel serve (UNTRACKED).

Bee inspiration (fattn-mma-kvarn-decode.cuh): ONE K-tile dequant reused
across Q rows via tensor-core mma(scores, k_a, q_b) + mma(out, v_a, p_b),
not per-Q CUDA-core scalar dots. Spike5's wall: 1024x tl.sum(q*k,axis=1)
dots/chunk (CUDA cores) + 8-wide QPAD serial softmax depth.

Spike6 re-tiles the same math MMA-first:
- TOK=16 rows/iter x 8 iters cover the 128-row chunk (same total dequant
  traffic as spike5's 4x32, fewer iters than 32x4 depth).
- K/V tiles built 2D natively (TOK,HD): per-row group/slot from the same
  bt/page map spike5 uses; bit-unpack loop over BITS with 2D pointer
  tensors (row-dependent gbase). No per-row helper calls.
- QK: dot(K_fp16 (TOK,HD), QwT_fp16 (HD,QPAD)) -> S_body (TOK,QPAD) +
  dot(K_fp16, QfT_fp16) -> S_tail; per-row body/tail select, r-valid mask
  to -inf. M=16,N=8 fits mma.m16n8k16 (spike5's (8,256)x(256,1) never can).
- EV: E (TOK,QPAD) exps -> trans -> (QPAD,TOK) @ V_fp16 (TOK,HD) ->
  (QPAD,HD) MMA accumulation into acc (no 16-way scalar axpy chain).
- V tile unified WHT-domain (body dequant vv + pre-WHT'd exact ev, same
  mixed-domain fix as spike5: tail V host-pre-WHT'd, tail scores Q-orig).
- Merge path unchanged (merge2/combine already validated).

Precision note: dot operands fp16 / accum fp32 (tensor-core path). Spike
RMSE gate relaxed to 5e-4 (fp16-MMA rounding); the REAL gate is KLD
same-top vs fp16-cache (fp16-image asymmetry alone is 1.4e-04).

Usage: python eval/_spike6_mma.py [attn|probe]
"""

import sys

import torch
import triton
import triton.language as tl

TOK: int = 16  # rows per iter; 8 iters cover 128

_S6BUFS = {}


@triton.jit
def _serve_mma_kernel(
    qw_ptr, qf_ptr, rec_ptr, rec_f16_ptr,
    exact_k_ptr, exact_v_ptr, exrev_ptr, sealed_ptr,
    bt_ptr, n_ptr, flag_ptr,
    m_ptr, l_ptr, out_ptr,
    K_PAY_OFF, K_SC2, K_ZP2, K_OT2, K_BITS: tl.constexpr,
    V_PAY_OFF, V_SC2, V_ZP2, V_OT2, V_BITS: tl.constexpr,
    C: tl.constexpr, B: tl.constexpr, SL: tl.constexpr, GPS: tl.constexpr,
    KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
    HD: tl.constexpr, GMAX: tl.constexpr, SCALE: tl.constexpr,
    SINK_N: tl.constexpr, TAIL_EFF: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_c = tl.program_id(1)
    lane = tl.arange(0, HD)  # (HD,)
    sl_c = lane // 128
    dd_c = lane % 128
    qoff = tl.arange(0, QPAD)  # (QPAD,)
    qmask = qoff < QPK
    qw = tl.load(qw_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                 + lane[None, :], mask=qmask[:, None], other=0.0)
    qf = tl.load(qf_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                 + lane[None, :], mask=qmask[:, None], other=0.0)
    qwT = tl.trans(qw.to(tl.float16))  # (HD, QPAD)
    qfT = tl.trans(qf.to(tl.float16))
    n = tl.load(n_ptr)
    tail_start = n - TAIL_EFF
    m = tl.full([QPAD], -1e30, dtype=tl.float32)
    l = tl.zeros([QPAD], dtype=tl.float32)
    acc = tl.zeros([QPAD, HD], dtype=tl.float32)

    tok = tl.arange(0, 16)  # TOK=16 rows/iter; 8 iters cover 128
    for t0 in tl.range(8):
        t = t0 * 16
        p = pid_c * 128 + t + tok  # (TOK,)
        r = p < n
        body = (p >= SINK_N) & (p < tail_start) & r
        pa = tl.where(r, p, 0)
        page = tl.load(bt_ptr + pa // 256)
        offs = pa % 256
        g = page * GPS + offs // 128  # (TOK,)
        s = offs % 128  # (TOK,)

        # --- K tile 2D (TOK, HD), row-dependent gbase ---
        kc = pid_h * SL + sl_c  # (HD,) col channel
        # per-row payload/f16 bases broadcast to 2D via pointer tensors
        kpay_row = rec_ptr + g * C * B + K_PAY_OFF  # (TOK,)
        kf16_row = rec_f16_ptr + (g * C * B) // 2  # (TOK,)
        # v index: dd[c]*128 + s[r]
        vv = dd_c[None, :] * 128 + s[:, None]  # (16, HD)
        qq = tl.zeros([16, HD], dtype=tl.int32)
        for i in tl.static_range(8):
            if i < K_BITS:
                b = vv * K_BITS + i
                ptr = (kpay_row[:, None] + kc[None, :] * B + b // 8)
                byteval = tl.load(ptr).to(tl.int32)
                qq += ((byteval >> (b % 8)) & 1) << i
        sc_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + K_SC2
                  + dd_c[None, :])
        zp_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + K_ZP2
                  + dd_c[None, :])
        ot_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + K_OT2
                  + s[:, None])
        kk = (qq.to(tl.float32) * tl.load(sc_ptr).to(tl.float32)
              + tl.load(zp_ptr).to(tl.float32)) \
            * tl.load(ot_ptr).to(tl.float32)  # (16, HD)

        # --- V tile 2D (16, HD) ---
        vpay_row = rec_ptr + g * C * B + V_PAY_OFF  # (TOK,)
        vv2 = s[:, None] * 128 + dd_c[None, :]
        qqv = tl.zeros([16, HD], dtype=tl.int32)
        for i in tl.static_range(8):
            if i < V_BITS:
                bv = vv2 * V_BITS + i
                ptrv = (vpay_row[:, None]
                        + kc[None, :] * B + bv // 8)
                byteval = tl.load(ptrv).to(tl.int32)
                qqv += ((byteval >> (bv % 8)) & 1) << i
        vsc_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + V_SC2
                   + s[:, None])
        vzp_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + V_ZP2
                   + s[:, None])
        vot_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + V_OT2
                   + dd_c[None, :])
        vv_tile = ((qqv.to(tl.float32) * tl.load(vsc_ptr).to(tl.float32)
                    + tl.load(vzp_ptr).to(tl.float32))
                   * tl.load(vot_ptr).to(tl.float32))

        # --- exact-direct 2D tiles (orig-K / WHT-V) ---
        es = tl.load(exrev_ptr + g)  # (TOK,)
        ok_tail = (es >= 0) & (~body) & r  # (TOK,)
        ek_ptr = (exact_k_ptr + (es[:, None] * 128 + s[:, None]) * KVH * HD
                  + pid_h * HD + lane[None, :])
        ev_ptr = (exact_v_ptr + (es[:, None] * 128 + s[:, None]) * KVH * HD
                  + pid_h * HD + lane[None, :])
        ek = tl.load(ek_ptr, mask=ok_tail[:, None], other=0.0)
        ev = tl.load(ev_ptr, mask=ok_tail[:, None], other=0.0)
        k_tile = tl.where(body[:, None], kk, ek)  # open rows: r-masked below
        v_tile = tl.where(body[:, None], vv_tile, ev)

        bad = body & (tl.load(sealed_ptr + g) == 0)
        tl.store(flag_ptr, 1, mask=tl.sum(bad.to(tl.int32)) > 0)

        # --- MMA QK: (TOK,HD) @ (HD,QPAD) -> (TOK,QPAD), both domains ---
        sb = tl.dot(k_tile.to(tl.float16), qwT) * SCALE
        st = tl.dot(k_tile.to(tl.float16), qfT) * SCALE
        sc = tl.where(body[:, None], sb.to(tl.float32),
                      st.to(tl.float32))
        sc = tl.where(r[:, None], sc, float("-inf"))

        smax = tl.max(sc, axis=0)  # (QPAD,)
        m_new = tl.maximum(m, smax)
        alpha = tl.exp(m - m_new)
        e = tl.exp(sc - m_new[None, :])  # (TOK, QPAD)
        l = l * alpha + tl.sum(e, axis=0)
        # --- MMA EV: (QPAD,TOK) @ (TOK,HD) ---
        eT = tl.trans(e.to(tl.float16))  # (QPAD, TOK)
        acc = acc * alpha[:, None] + tl.dot(eT, v_tile.to(tl.float16))
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


def serve_online_mma(qw, Qf, records, layout, k_bits, v_bits, exact_k,
                     exact_v_w, exrev, sealed, bt, n_0d, kvh, qpk, sl, hd,
                     gps, sink_n, tail_eff, scale=0.0625):
    dev = qw.device
    qh = kvh * qpk
    gmax = int(records.shape[0])
    qpad = 1 << (qpk - 1).bit_length()
    key = (qh, gmax, hd)
    if key not in _S6BUFS:
        _S6BUFS[key] = (
            torch.empty((kvh, qpad, gmax), dtype=torch.float32, device=dev),
            torch.empty((kvh, qpad, gmax), dtype=torch.float32, device=dev),
            torch.empty((kvh, qpad, gmax, hd), dtype=torch.float32,
                        device=dev),
            torch.empty((qh, hd), dtype=torch.float32, device=dev),
            torch.zeros((1,), dtype=torch.uint8, device=dev))
    m, l, acc, out, flag = _S6BUFS[key]
    flag.zero_()
    rec_f16 = records.view(torch.float16)
    _serve_mma_kernel[(kvh, gmax,)](
        qw, Qf, records, rec_f16, exact_k, exact_v_w, exrev, sealed, bt,
        n_0d, flag, m, l, acc,
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
    out, flag = serve_online_mma(qw, Qf, records, layout, bits[0], bits[1],
                                 exact_k, exact_v_w, exrev, sealed, bt, n_0d,
                                 kvh, qpk, sl, hd, 2, 128, 128)
    print("flag (expect 0):", flag, flush=True)
    bk, bv = kvarn_triton_dequant_groups(
        records, layout, bits[0], bits[1], kvh, sl, do_wht=False)
    Kb = kvarn_wht_head(bk, hd)
    Vb = kvarn_wht_head(bv, hd)
    Kb_body = torch.cat([Kb[1, :, :, :], Kb[2, :16, :, :]]).reshape(-1, kvh,
                                                                   hd)
    Vb_body = torch.cat([Vb[1, :, :, :], Vb[2, :16, :, :]]).reshape(-1, kvh,
                                                                   hd)
    Trow = list(range(0, 128)) + list(range(272, 400))
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
    print(f"mma maxdiff={float(d.max()):.3e} "
          f"RMSE={float((d**2).mean().sqrt()):.3e}", flush=True)
    assert flag == 0
    assert float((d**2).mean().sqrt()) < 5e-4
    print("SPIKE6-MMA PASS", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "attn"
    assert mode == "attn", "probe runs on the Win GPU host (queued)"
    cmd_attn()
