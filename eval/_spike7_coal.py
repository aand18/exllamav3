"""Match-Bee Task 3 spike7: coalesced-K single-kernel serve (UNTRACKED).

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

Spike7 delta (coalesced K): the K payload is dim-major (v = dd*128+s),
so a row-tile read strides 64B per dim (~64x L2-sector amplification;
microbench _dbg_coal.py: strided 41.4us vs coalesced 10.8us = 3.84x on
payload reads alone). V payload is already slot-major (coalesced).
Spike7 stores/serves K slot-major (v = s*128+dd): each row = 64
contiguous bytes. Host helper transpose_k_payload() permutes real
records for the probe (production store change is queued behind this
measurement). 4-bit only; other widths keep the legacy index.

Usage: python eval/_spike7_coal.py [attn|probe]
"""

import sys

import torch
import triton
import triton.language as tl

TOK: int = 16  # rows per iter; 8 iters cover 128

_S7BUFS = {}


@triton.jit
def _serve_s7_kernel(
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
        # Group-uniformity: K sc/zp and V oth are per-dim values shared by
        # all rows of a group. Interior tiles (the common case) touch one
        # group -> load them once as (256,) vectors + broadcast instead of
        # (16,256) 2D tiles (16x fewer metadata loads). Boundary tiles
        # take the slow path. Bit-identical (same addresses).
        g0 = tl.sum(tl.where(tok == 0, g, 0))
        uniform = tl.sum(tl.where(g == g0, 1, 0).to(tl.int32)) == 16
        kf16_g0 = rec_f16_ptr + (g0 * C * B) // 2
        if uniform:
            k_sc0 = tl.load(kf16_g0 + (kc * B) // 2 + K_SC2
                            + dd_c).to(tl.float32)  # (256,)
            k_zp0 = tl.load(kf16_g0 + (kc * B) // 2 + K_ZP2
                            + dd_c).to(tl.float32)
            v_ot0 = tl.load(kf16_g0 + (kc * B) // 2 + V_OT2
                            + dd_c).to(tl.float32)
        else:
            k_sc0 = tl.zeros([HD], dtype=tl.float32)
            k_zp0 = tl.zeros([HD], dtype=tl.float32)
            v_ot0 = tl.zeros([HD], dtype=tl.float32)
        # K value index: slot-major (transposed store) for 4-bit, legacy
        # dim-major otherwise. Slot-major row s = 64 contiguous bytes.
        vv = s[:, None] * 128 + dd_c[None, :]  # (16, HD) transposed
        vv_legacy = dd_c[None, :] * 128 + s[:, None]
        if K_BITS == 4:
            # Nibble fast path: value v occupies stream bits [4v,4v+4),
            # i.e. low/high nibble of byte v//2. One byte load per
            # element instead of 4 bit-loop loads. Matches the generic
            # LSB-first loop bit-for-bit (verified: even v -> low
            # nibble, odd v -> high nibble).
            nb = vv // 2
            nptr = (kpay_row[:, None] + kc[None, :] * B + nb)
            nbyte = tl.load(nptr).to(tl.int32)
            qq = ((nbyte >> ((vv % 2) * 4)) & 0xF)
        else:
            qq = tl.zeros([16, HD], dtype=tl.int32)
            for i in tl.static_range(8):
                if i < K_BITS:
                    b = vv_legacy * K_BITS + i
                    ptr = (kpay_row[:, None] + kc[None, :] * B + b // 8)
                    byteval = tl.load(ptr).to(tl.int32)
                    qq += ((byteval >> (b % 8)) & 1) << i
        sc_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + K_SC2
                  + dd_c[None, :])
        zp_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + K_ZP2
                  + dd_c[None, :])
        ot_ptr = (kf16_row[:, None] + (kc[None, :] * B) // 2 + K_OT2
                  + s[:, None])
        if uniform:
            # Per-slot K oth: 2 values/row (one per slice) -> where-expand
            # over sl_c. Bit-identical (same addresses, same op order).
            k_ot = tl.zeros([16, HD], dtype=tl.float32)
            for _sl in tl.static_range(4):
                if _sl < SL:
                    _v = tl.load(kf16_g0 + ((pid_h * SL + _sl) * B) // 2
                                 + K_OT2 + s).to(tl.float32)  # (16,)
                    k_ot = tl.where(sl_c[None, :] == _sl, _v[:, None], k_ot)
            kk = (qq.to(tl.float32) * k_sc0[None, :]
                  + k_zp0[None, :]) * k_ot  # (16, HD)
        else:
            kk = (qq.to(tl.float32) * tl.load(sc_ptr).to(tl.float32)
                  + tl.load(zp_ptr).to(tl.float32)) \
                * tl.load(ot_ptr).to(tl.float32)  # (16, HD)

        # --- exact-direct 2D tiles (orig-K / WHT-V) ---
        # (V payload + ev deferred below the QK dots to halve peak live.)
        es = tl.load(exrev_ptr + g)  # (TOK,)
        ok_tail = (es >= 0) & (~body) & r  # (TOK,)
        ek_ptr = (exact_k_ptr + (es[:, None] * 128 + s[:, None]) * KVH * HD
                  + pid_h * HD + lane[None, :])
        ek = tl.load(ek_ptr, mask=ok_tail[:, None], other=0.0)
        k_tile = tl.where(body[:, None], kk, ek)  # open rows: r-masked below

        bad = body & (tl.load(sealed_ptr + g) == 0)
        tl.store(flag_ptr, 1, mask=tl.sum(bad.to(tl.int32)) > 0)

        # --- MMA QK: (TOK,HD) @ (HD,QPAD) -> (TOK,QPAD), both domains ---
        # Tail-domain dot only when the tile actually has non-body rows
        # (all-body is the common case): saves 1 MMA/iter there.
        sb = tl.dot(k_tile.to(tl.float16), qwT) * SCALE
        nbody = tl.sum(body.to(tl.int32))
        if nbody == 16:
            sc = tl.where(r[:, None], sb.to(tl.float32), float("-inf"))
        else:
            st = tl.dot(k_tile.to(tl.float16), qfT) * SCALE
            sc = tl.where(body[:, None], sb.to(tl.float32),
                          st.to(tl.float32))
            sc = tl.where(r[:, None], sc, float("-inf"))

        smax = tl.max(sc, axis=0)  # (QPAD,)
        m_new = tl.maximum(m, smax)
        alpha = tl.exp(m - m_new)
        e = tl.exp(sc - m_new[None, :])  # (TOK, QPAD)
        l = l * alpha + tl.sum(e, axis=0)
        # --- V side deferred here so K-side tiles (qq/kk/ek/k_tile, ~50KB
        # live) are dead past the QK dots: peak live state roughly halves.
        # Bit-identical: same addresses, same per-element op order.
        vpay_row = rec_ptr + g * C * B + V_PAY_OFF  # (TOK,)
        vv2 = s[:, None] * 128 + dd_c[None, :]
        if V_BITS == 4:
            # Same nibble fast path as K (stream bits [4v,4v+4)).
            nbv = vv2 // 2
            nptrv = (vpay_row[:, None]
                     + kc[None, :] * B + nbv)
            nbytev = tl.load(nptrv).to(tl.int32)
            qqv = ((nbytev >> ((vv2 % 2) * 4)) & 0xF)
        else:
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
        if uniform:
            # Per-slot V sc/zp: (16,SL) loads + where-expand. Bit-identical.
            v_sc = tl.zeros([16, HD], dtype=tl.float32)
            v_zp = tl.zeros([16, HD], dtype=tl.float32)
            for _sl in tl.static_range(4):
                if _sl < SL:
                    _s = tl.load(kf16_g0 + ((pid_h * SL + _sl) * B) // 2
                                 + V_SC2 + s).to(tl.float32)  # (16,)
                    _z = tl.load(kf16_g0 + ((pid_h * SL + _sl) * B) // 2
                                 + V_ZP2 + s).to(tl.float32)
                    _m = sl_c[None, :] == _sl
                    v_sc = tl.where(_m, _s[:, None], v_sc)
                    v_zp = tl.where(_m, _z[:, None], v_zp)
            vv_tile = ((qqv.to(tl.float32) * v_sc + v_zp)
                       * v_ot0[None, :])
        else:
            vv_tile = ((qqv.to(tl.float32) * tl.load(vsc_ptr).to(tl.float32)
                        + tl.load(vzp_ptr).to(tl.float32))
                       * tl.load(vot_ptr).to(tl.float32))
        ev = tl.load(exact_v_ptr + (es[:, None] * 128 + s[:, None]) * KVH
                     * HD + pid_h * HD + lane[None, :],
                     mask=ok_tail[:, None], other=0.0)
        v_tile = tl.where(body[:, None], vv_tile, ev)
        # --- MMA EV via transposed form: (HD,TOK) @ (TOK,QPAD) = (HD,QPAD)
        # M=256,N=8,K=16 hits m16n8k16; the old (QPAD,TOK) form had M=8
        # (SIMT fallback suspect). Trans back and accumulate.
        d = tl.dot(tl.trans(v_tile.to(tl.float16)),
                   e.to(tl.float16))  # (HD, QPAD)
        acc = acc * alpha[:, None] + tl.trans(d)
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


def serve_online_s7(qw, Qf, records, layout, k_bits, v_bits, exact_k,
                     exact_v_w, exrev, sealed, bt, n_0d, kvh, qpk, sl, hd,
                     gps, sink_n, tail_eff, scale=0.0625):
    dev = qw.device
    qh = kvh * qpk
    gmax = int(records.shape[0])
    qpad = 1 << (qpk - 1).bit_length()
    key = (qh, gmax, hd)
    if key not in _S7BUFS:
        _S7BUFS[key] = (
            torch.empty((kvh, qpad, gmax), dtype=torch.float32, device=dev),
            torch.empty((kvh, qpad, gmax), dtype=torch.float32, device=dev),
            torch.empty((kvh, qpad, gmax, hd), dtype=torch.float32,
                        device=dev),
            torch.empty((qh, hd), dtype=torch.float32, device=dev),
            torch.zeros((1,), dtype=torch.uint8, device=dev))
    m, l, acc, out, flag = _S7BUFS[key]
    flag.zero_()
    rec_f16 = records.view(torch.float16)
    _serve_s7_kernel[(kvh, gmax,)](
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
        num_warps=8)
    sscale = 1.0 if sl == 1 else (0.7071067811865475 if sl == 2 else 0.5)
    from _spike2_online import _combine_kernel as _combine_v2
    nbpad = 1 << (gmax - 1).bit_length()
    _combine_v2[(qh,)](
        m, l, acc, out, kvh, qpk, qpad, gmax, nbpad, hd, sl, sscale,
        num_warps=1)
    return out, int(flag[0])


def transpose_k_payload(records, layout, k_bits):
    """Return a clone with each (group, channel) K payload permuted from
    dim-major (value v = dd*128+t) to slot-major (w = t*128+dd).
    Metadata regions are untouched. 4-bit only (probe uses kvarn4);
    the s7 kernel reads slot-major only when K_BITS == 4, so the
    transpose and the reader stay in lockstep by construction."""
    assert k_bits == 4, "spike7 transpose is 4-bit only"
    assert layout.head_dim == 128 and layout.group == 128
    R = records.clone()
    G, C, B = R.shape
    kp = layout.k_payload_bytes  # 8192 for 4-bit
    off = layout.k_payload_off
    u8 = R.view(torch.uint8) if R.dtype != torch.uint8 else R
    pay = u8[:, :, off:off + kp].reshape(G * C, kp)
    lo = (pay & 0xF).to(torch.int32)
    hi = ((pay >> 4) & 0xF).to(torch.int32)
    nib = torch.empty((G * C, 2 * kp), dtype=torch.int32, device=R.device)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    # nib[w], w = dd*128+t -> (dd, t) -> permute -> (t, dd)
    nib = nib.reshape(G * C, 128, 128).permute(0, 2, 1).reshape(G * C, -1)
    repacked = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    u8[:, :, off:off + kp] = repacked.reshape(G, C, kp)
    return R


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
    recordsT = transpose_k_payload(records, layout, bits[0])
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
    out, flag = serve_online_s7(qw, Qf, recordsT, layout, bits[0], bits[1],
                                 exact_k, exact_v_w, exrev, sealed, bt, n_0d,
                                 kvh, qpk, sl, hd, 2, 128, 128)
    print("flag (expect 0):", flag, flush=True)
    # Reference dequants the ORIGINAL (dim-major) records: same values,
    # so the reference is unchanged; only the kernel's read pattern moved.
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
    print("SPIKE7-COAL PASS", flush=True)


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
    torch.manual_seed(9)
    Q = torch.randn(qh, hd, dtype=torch.float16, device="cuda")
    Qf = Q.float()
    qw = torch.empty((qh, hd), dtype=torch.float32, device="cuda")
    qscratch = torch.empty_like(qw)
    import _spike3_online as _s3
    _s3.qwht_fused(Q, qscratch, qw, sl, 0.7071067811865475)
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
        o, _f = serve_online_s7(qw, Qf, recT, lay0.layout, k_bits,
                                v_bits, lay0.exact_k, Ew, lay0._exact_rev,
                                lay0.sealed, bt, n_0d, kvh, qpk, sl, hd, 2,
                                128, 128)
        return o
    # Host-side K transpose of the populated records (one-off; production
    # store change queued behind this measurement). Timed below.
    torch.cuda.synchronize()
    tT0 = torch.cuda.Event(enable_timing=True)
    tT1 = torch.cuda.Event(enable_timing=True)
    tT0.record()
    recT = transpose_k_payload(lay0.records, lay0.layout, k_bits)
    tT1.record()
    torch.cuda.synchronize()
    print(f"host K-transpose (one-off): {tT0.elapsed_time(tT1):.2f} ms",
          flush=True)
    o = do_spike()
    print("spike7 full path ok", flush=True)
    t_spike = hot(do_spike)
    # Piece split: qwht alone, serve+combine (qw precomputed once, timing
    # only), combine alone over the live buffers. serve~ = (serve+combine)
    # - combine. One GPU round-trip, no extra model load.
    def do_servecombine():
        o, _f = serve_online_s7(qw, Qf, recT, lay0.layout, k_bits,
                                v_bits, lay0.exact_k, Ew, lay0._exact_rev,
                                lay0.sealed, bt, n_0d, kvh, qpk, sl, hd, 2,
                                128, 128)
        return o
    t_qwht = hot(lambda: _s3.qwht_fused(Q, qscratch, qw, sl,
                                       0.7071067811865475), iters=500)
    t_servecombine = hot(do_servecombine)
    from _spike2_online import _combine_kernel as _combine_v2
    qpad = 1 << (qpk - 1).bit_length()
    nbpad = 1 << (gmax - 1).bit_length()
    sscale = 0.7071067811865475
    m, l, acc, out = _S7BUFS[(qh, gmax, hd)][:4]
    t_combine = hot(lambda: _combine_v2[(qh,)](
        m, l, acc, out, kvh, qpk, qpad, gmax, nbpad, hd, sl, sscale),
        iters=500)
    print(f"pieces us: qwht={t_qwht * 1e3:.1f} "
          f"serve+combine={t_servecombine * 1e3:.1f} "
          f"combine={t_combine * 1e3:.1f} "
          f"serve~={(t_servecombine - t_combine) * 1e3:.1f}", flush=True)
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
    print(f"spike7 coal:      {t_spike:.4f} ms/step over {rows} rows "
          f"({per_spike:.2f} ns/row)", flush=True)
    print(f"per-row ratio spike/fp16: {per_spike / per_fp16:.3f} "
          f"(gate <= 1.111)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "attn"
    {"attn": cmd_attn, "probe": cmd_probe}[mode]()
