"""
KVarN online-dequant Triton kernels (bootstrap, NOT YET RUN ON GPU).

Status: validated on RTX 4090 (sm_89, triton 3.8.0). The per-row kernel
is numerically bit-exact vs the torch reference (container GPU tests +
in-harness ``EXL3_KVARN_TRITON_PARITY=1`` asserts on a 27B 8k run with
zero mismatches). Perf: ~zero end-to-end win yet -- by the time this
fused, dequant was no longer the bottleneck (batched torch + incremental
image); the next fuse candidate is the inverse WHT. Keep PARITY=1 on the
first run of any kernel change.

Why this exists: ``CacheLayer_kvarn.get_kv`` dequantizes sealed 128-groups
with device-agnostic torch ops (correct on CUDA, but one Python loop per
group plus per-tile vector ops). This module fuses the hot part -- LSB-first
linear unpack + asymmetric RTN dequant -- into one Triton kernel launch per
sealed group. It deliberately does NOT fuse the inverse WHT or attention:
those reuse the tested torch path (``kvarn_wht_head``) and the existing
Triton paged-attention kernels until hardware measurements justify fusion.

Math contract (must match ``kvarn_unpack_bits`` + ``kvarn_dequantize_tile``):
- Records pack values LSB-first linear: value v occupies stream bits
  [v*BITS, (v+1)*BITS), byte b holds stream bits [8b, 8b+8).
- tile[r, c] = (q * sc[r] + zp[r]) * other[c], fp32 compute and store
  (fp32 store keeps the Triton path bit-exact with the torch reference
  through the downstream WHT).
- K tiles are [dim, token], V tiles [token, dim]; orientation is the
  caller's business (strides), the kernel always sees [row, col] tiles.

Parity mode: with ``EXL3_KVARN_TRITON_PARITY=1`` the ``CacheLayer_kvarn``
hook runs the torch reference alongside and asserts allclose before
returning the Triton result. That is the acceptance test for this file.
"""

from __future__ import annotations
import os
import torch

try:
    import triton
    import triton.language as tl
    _have_triton = True
except ImportError:  # CPU-only hosts without triton
    _have_triton = False


def kvarn_triton_available() -> bool:
    """
    True only when the Triton path may be attempted: triton importable, a
    CUDA device present, and explicitly opted in via EXL3_KVARN_TRITON=1.
    Default off, so stock behavior never touches this module.
    """
    return _have_triton and torch.cuda.is_available() and \
        os.environ.get("EXL3_KVARN_TRITON", "0") == "1"


def kvarn_triton_parity_check() -> bool:
    """True when EXL3_KVARN_TRITON_PARITY=1: run torch reference + assert."""
    return os.environ.get("EXL3_KVARN_TRITON_PARITY", "0") == "1"


if _have_triton:
    @triton.jit
    def _fwht128_block(row_ptr, cols):
        # 7 in-place FWHT stages + 1/sqrt(128) norm over 128 fp32 values.
        # Caller seeds row_ptr first. Single-warp launch ONLY (num_warps=1
        # at every call site): stages exchange values across lanes, and
        # tl.debug_barrier does NOT synchronize warps in compiled triton
        # 3.8 kernels on sm_89 (nondeterministic corruption at 1000+ rows,
        # 0/6 exact with 4/8 warps vs 6/6 with 1 warp). One warp stays in
        # lockstep over this branch-free code, so no barrier is needed.
        # Subtractions use (lower - upper), matching the torch reference
        # (odd-parity positions carry the minus).
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
    def _kvarn_wht_hd_kernel(x_ptr, HD: tl.constexpr, SLICES: tl.constexpr,
                             SSCALE: tl.constexpr):
        # Head-wide WHT for one (HD,) row: per-128 FWHT per slice, then
        # cross-slice stages. Matches kvarn_wht_head bit-exact (FWHT is an
        # involution, so forward and inverse share this kernel).
        pid = tl.program_id(0)
        cols = tl.arange(0, HD)
        base = x_ptr + pid * HD
        for _sl in tl.static_range(4):
            if _sl < SLICES:
                _fwht128_block(base + _sl * 128, tl.arange(0, 128))
        tl.debug_barrier()
        if SLICES > 1:
            cur = tl.load(base + cols)
            prt = tl.load(base + (cols ^ 128))
            tl.store(base + cols,
                     tl.where((cols & 128) == 0, cur + prt, prt - cur))
            tl.debug_barrier()
        if SLICES > 2:
            cur = tl.load(base + cols)
            prt = tl.load(base + (cols ^ 256))
            tl.store(base + cols,
                     tl.where((cols & 256) == 0, cur + prt, prt - cur))
            tl.debug_barrier()
        tl.store(base + cols, tl.load(base + cols) * SSCALE)

    @triton.jit
    def _kvarn_row_kernel(
        pay_ptr, sc_ptr, zp_ptr, oth_ptr, out_ptr,
        PAY: tl.constexpr,   # payload bytes per tile (drop-in bound, unused)
        BITS: tl.constexpr,  # 2, 3, 4, 5, 6 or 8
        DO_WHT: tl.constexpr = 0,  # 1: fold the 128-point FWHT (+ norm) in
    ):
        # One program = one row of one 128x128 tile.
        # pid = tile * 128 + row.
        pid = tl.program_id(0)
        tile = pid // 128
        row = pid % 128
        cols = tl.arange(0, 128)

        # Linear bit positions of this row's 128 values in the stream.
        v = (row * 128 + cols) * BITS
        q = tl.zeros([128], dtype=tl.int32)
        for i in tl.static_range(BITS):
            b = v + i
            byte = b // 8
            bit = b % 8
            byteval = tl.load(pay_ptr + tile * PAY + byte)
            q += ((byteval.to(tl.int32) >> bit) & 1) << i

        sc = tl.load(sc_ptr + tile * 128 + row).to(tl.float32)
        zp = tl.load(zp_ptr + tile * 128 + row).to(tl.float32)
        oth = tl.load(oth_ptr + tile * 128 + cols).to(tl.float32)
        tile_out = (q.to(tl.float32) * sc + zp) * oth
        row_ptr = out_ptr + (tile * 128 + row) * 128
        if DO_WHT == 0:
            tl.store(row_ptr + cols, tile_out)
        else:
            # In-place FWHT over the 128-vector using the out row as
            # scratch (XOR butterfly: partner of i at stride s is i^s).
            # Same stage order and norm as kvarn_hadamard_128.
            tl.store(row_ptr + cols, tile_out)
            _fwht128_block(row_ptr, cols)


def kvarn_triton_dequant_side(payload: torch.Tensor, sc: torch.Tensor,
                               zp: torch.Tensor, oth: torch.Tensor,
                               bits: int, do_wht: bool = False) -> torch.Tensor:
    """
    Dequantize NT tiles of one side (K or V).

    payload: (NT, PAY) uint8 CUDA. sc/zp: (NT, 128) fp16 CUDA (per row).
    oth: (NT, 128) fp16 CUDA (per col). Returns (NT, 128, 128) fp32 CUDA.
    With do_wht=True each tile additionally gets the 128-point FWHT (+
    norm), matching kvarn_hadamard_128 bit-exact.
    Loud failure (never silent) when the Triton path cannot run.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton dequant requires triton (import failed on this host).")
    if not payload.is_cuda:
        raise RuntimeError(
            "KVarN Triton dequant requires CUDA tensors, got "
            f"{payload.device}.")
    NT, PAY = payload.shape
    assert sc.shape == (NT, 128) and zp.shape == (NT, 128)
    assert oth.shape == (NT, 128)
    out = torch.empty((NT, 128, 128), dtype=torch.float32,
                       device=payload.device)
    # num_warps=1 on the DO_WHT path: it runs _fwht128_block, which is
    # only exact single-warp (see its comment). The pure-dequant path
    # keeps the default 4 warps (no cross-lane exchange there). One row
    # per program either way.
    _kvarn_row_kernel[(NT * 128,)](payload, sc, zp, oth, out,
                                    PAY, bits, 1 if do_wht else 0,
                                    num_warps=1 if do_wht else 4)
    return out


def kvarn_triton_dequant_group(records_g: torch.Tensor, layout,
                               k_bits: int, v_bits: int,
                               num_kv_heads: int, slices: int):
    """
    Dequantize one sealed group's combined records to rotated-domain fp32.

    records_g: (ncols, tile_bytes) uint8 CUDA, ncols = kv_heads * slices.
    Returns (bk, bv) float32 CUDA shaped (128, kvh, hd), matching the torch
    loop in CacheLayer_kvarn (K tiles [dim, token] transposed on assembly,
    V tiles [token, dim] as-is). Scale gather is plain torch slicing
    (device ops); only unpack+dequant is fused Triton.
    """
    ncols = num_kv_heads * slices
    assert records_g.shape[0] == ncols
    rec_f16 = records_g.view(torch.float16)

    def side(payload_off, payload_bytes, sc_off, zp_off, oth_off, bits):
        pay = records_g[:, payload_off: payload_off + payload_bytes]
        sc = rec_f16[:, sc_off // 2: sc_off // 2 + 128]
        zp = rec_f16[:, zp_off // 2: zp_off // 2 + 128]
        oth = rec_f16[:, oth_off // 2: oth_off // 2 + 128]
        return kvarn_triton_dequant_side(pay.contiguous(), sc.contiguous(),
                                         zp.contiguous(), oth.contiguous(),
                                         bits).float()

    k_tiles = side(layout.k_payload_off, layout.k_payload_bytes,
                   layout.k_s_col_off, layout.k_zp_off, layout.k_s_row_off,
                   k_bits)   # (ncols, 128, 128) [dim, token]
    v_tiles = side(layout.v_payload_off, layout.v_payload_bytes,
                   layout.v_s_row_off, layout.v_zp_off, layout.v_s_col_off,
                   v_bits)   # (ncols, 128, 128) [token, dim]

    hd = slices * 128
    bk = torch.empty((128, num_kv_heads, hd), dtype=torch.float32,
                     device=records_g.device)
    bv = torch.empty_like(bk)
    for h in range(num_kv_heads):
        for sl in range(slices):
            c = h * slices + sl
            d0, d1 = sl * 128, (sl + 1) * 128
            bk[:, h, d0:d1] = k_tiles[c].T
            bv[:, h, d0:d1] = v_tiles[c]
    return bk, bv


if _have_triton:
    @triton.jit
    def _kvarn_store_row_kernel(
        rk_ptr, rv_ptr, ek_ptr, ev_ptr,  # (kvh, HD) fp32 WHT'd / tail-dtype rows
        pages_ptr, offs_ptr, pos_ptr,    # (1,) int64: page, offset, position
        stage_k_ptr, stage_v_ptr,        # (G, 128, kvh, HD) fp16
        exact_k_ptr, exact_v_ptr,        # (G, 128, kvh, HD) tail dtype
        exact_valid_ptr,                 # (G,) bool
        base_ptr,                        # (G,) int64 group base
        sealed_ptr,                      # (G,) bool
        present_ptr,                     # (G, 128) bool
        owner_ptr,                       # (P,) int64 page owners
        pinned_ptr,                      # (P,) bool
        dirty_ptr,                       # (G,) bool
        status_ptr,                      # (2,) int64 out: [code, group]
        KVH: tl.constexpr, HD: tl.constexpr, GPS: tl.constexpr,
        TAIL_KEEP: tl.constexpr, SINK_N: tl.constexpr,
        HAS_SINK: tl.constexpr, TAIL_IS_BF16: tl.constexpr,
    ):
        # Fused single-row decode store (bsz 1, length 1, non-SWA).
        # Grid (2*KVH,) programs x HD lanes. All policy is predicated
        # (no Python branches on tensor values); steady appends need
        # zero CPU syncs to launch, one status read after.
        # Codes: 0 ok, 1 slow-path fallback, 2 seal group status[1].
        pid = tl.program_id(0)
        is_k = (pid // KVH) == 0
        h = pid % KVH
        cols = tl.arange(0, HD)

        pos = tl.load(pos_ptr)
        n_new = pos + 1
        page = tl.load(pages_ptr)
        offs = tl.load(offs_ptr)
        g = page * GPS + offs // 128
        s = offs % 128
        bnew = pos - s
        base = tl.load(base_ptr + g)
        is_sealed = tl.load(sealed_ptr + g)
        slow = ((base != bnew) & (base >= 0)) | is_sealed
        fresh = (base != bnew) & (~slow)
        go = ~slow

        row_k = tl.load(rk_ptr + h * HD + cols)
        row_v = tl.load(rv_ptr + h * HD + cols)
        s_off = (g * 128 + s) * KVH * HD + h * HD + cols
        tl.store(stage_k_ptr + s_off, row_k.to(tl.float16),
                 mask=is_k & go)
        tl.store(stage_v_ptr + s_off, row_v.to(tl.float16),
                 mask=(~is_k) & go)
        tl.store(present_ptr + g * 128 + s, True, mask=go)

        tl.store(sealed_ptr + g, False, mask=fresh)
        tl.store(exact_valid_ptr + g, False, mask=fresh)
        tl.store(base_ptr + g, bnew, mask=fresh)
        tl.store(pinned_ptr + page, False, mask=fresh)

        keep = (pos >= n_new - TAIL_KEEP) | (HAS_SINK & (pos < SINK_N))
        e_off = (g * 128 + s) * KVH * HD + h * HD + cols
        # Exact blocks hold ORIGINAL-domain rows (ek/ev), unlike staging.
        e_row_k = tl.load(ek_ptr + h * HD + cols)
        e_row_v = tl.load(ev_ptr + h * HD + cols)
        if TAIL_IS_BF16:
            tl.store(exact_k_ptr + e_off, e_row_k.to(tl.bfloat16),
                     mask=is_k & keep & go)
            tl.store(exact_v_ptr + e_off, e_row_v.to(tl.bfloat16),
                     mask=(~is_k) & keep & go)
        else:
            tl.store(exact_k_ptr + e_off, e_row_k.to(tl.float16),
                     mask=is_k & keep & go)
            tl.store(exact_v_ptr + e_off, e_row_v.to(tl.float16),
                     mask=(~is_k) & keep & go)
        tl.store(exact_valid_ptr + g, True, mask=keep & go)

        cur_owner = tl.load(owner_ptr + page)
        new_owner = tl.where(cur_owner < 0, n_new,
                             tl.minimum(cur_owner, n_new))
        new_owner = tl.where(n_new > new_owner, n_new, new_owner)
        tl.store(owner_ptr + page, new_owner, mask=go)
        tl.store(dirty_ptr + g, True, mask=go)

        full = tl.sum(tl.load(present_ptr + g * 128 + tl.arange(0, 128))
                      .to(tl.int32)) == 128
        sink_skip = HAS_SINK & (bnew == 0)
        code = tl.where(slow, 1, tl.where(full & (~sink_skip), 2, 0))
        tl.store(status_ptr, code)
        tl.store(status_ptr + 1, g)

    # NOTE: kvarn_triton_wht_rows defined below (unchanged position).


def kvarn_triton_wht_rows(x, head_dim: int, inplace: bool = False):
    """
    Head-wide forward WHT over (..., HD) fp32 CUDA, HD in {128, 256, 512}.
    Matches kvarn_wht_head bit-exact (FWHT is an involution). One launch.
    With inplace=True the kernel runs directly on x (must be a contiguous
    fp32 temp -- same values out as the copy path, minus the alloc+copy;
    used for the fresh serve/store buffers). Default stays out-of-place.
    Loud failure (never silent) when the Triton path cannot run.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton WHT requires triton (import failed on this host).")
    if not x.is_cuda:
        raise RuntimeError(
            "KVarN Triton WHT requires CUDA tensors, got "
            f"{x.device}.")
    slices = head_dim // 128
    assert head_dim in (128, 256, 512) and slices * 128 == head_dim
    sscale = 1.0 if slices == 1 else (0.7071067811865475 if slices == 2
                                      else 0.5)
    if inplace:
        assert x.dtype == torch.float32 and x.is_contiguous(), \
            "KVarN Triton in-place WHT needs a contiguous fp32 temp."
        out = x.reshape(-1, head_dim)
        n = out.shape[0]
    else:
        flat = x.float().reshape(-1, head_dim)
        out = torch.empty_like(flat)
        out.copy_(flat)
        n = flat.shape[0]
    # num_warps=1: per-128 FWHT stages exchange values across lanes and
    # tl.debug_barrier does not sync warps (see _fwht128_block comment).
    _kvarn_wht_hd_kernel[(n,)](out, head_dim, slices, sscale,
                               num_warps=1)
    return out.reshape(*x.shape[:-1], head_dim)


def kvarn_triton_store_row(layer, rows_k, rows_v, pages_1, offs_1, pos_1,
                           gps, sink_tokens, rollback_tokens):
    """Fused single-row decode store (bsz 1, length 1, non-SWA caller).

    Row WHT (1 launch) + fused write kernel (1 launch): stage/exact
    writes, present/base/owner/valid/dirty updates. Policy events
    (reuse with live content, sealed overwrite) bail with code 1;
    a completed group reports code 2 and its id for the torch sealer.
    Returns (code: int, group: int) -- one status read, the only CPU
    sync. Small ints (gps, sink/rollback sizes) come from the caller so
    this module never imports the cache package (no cycle, no stub
    fragility). Loud failure (never silent) when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton store requires triton (import failed on this host).")
    dev = rows_k.device
    if dev.type != "cuda":
        raise RuntimeError(
            "KVarN Triton store requires CUDA tensors, got "
            f"{rows_k.device}.")
    kvh, hd = layer.num_kv_heads, layer.head_dim
    stacked = torch.stack((rows_k.float(), rows_v.float()))
    # stacked is a fresh fp32 contiguous temp: transform in place
    # (same values, minus the alloc+copy).
    rkv = kvarn_triton_wht_rows(stacked, hd, inplace=True)
    rk = rkv[0].reshape(kvh, hd).contiguous()
    rv = rkv[1].reshape(kvh, hd).contiguous()
    tail_keep = int(layer.tail_effective) + rollback_tokens
    sink_n = sink_tokens if layer.has_sink else 0
    # Exact rows ride in as fp16 (no .to(tail_dtype) alloc+copy): the
    # kernel downcasts on load (same RNE result the torch fallback gets
    # from rows.to(tail_dtype); the fused-store twin test asserts it).
    status = layer._store_status
    _kvarn_store_row_kernel[(2 * kvh,)](
        rk, rv,
        rows_k.reshape(kvh, hd).contiguous(),
        rows_v.reshape(kvh, hd).contiguous(),
        pages_1.to(torch.int64), offs_1.to(torch.int64),
        pos_1.to(torch.int64),
        layer.stage_k, layer.stage_v,
        layer.exact_k, layer.exact_v, layer.exact_valid,
        layer.group_base, layer.sealed, layer.present,
        layer.page_owner_n, layer.page_pinned, layer._dirty_mask,
        status,
        kvh, hd, gps,
        tail_keep, sink_n, bool(layer.has_sink),
        layer.tail_dtype == torch.bfloat16,
    )
    code, g = status.tolist()
    return int(code), int(g)


if _have_triton:
    @triton.jit
    def _kvarn_overlay_kernel(
        img_k_ptr, img_v_ptr,             # (P, 256, kvh, HD) fp16 temps
        exact_k_ptr, exact_v_ptr,         # (G, 128, kvh, HD) tail dtype
        exact_valid_ptr,                  # (G,) bool
        seqlens_ptr, bt_ptr,              # (1,) int32 n, (P,) int32 pages
        KVH: tl.constexpr, HD: tl.constexpr, GPS: tl.constexpr,
        TAIL_EFF: tl.constexpr, SINK_N: tl.constexpr,
        HAS_SINK: tl.constexpr, TAIL_IS_BF16: tl.constexpr, MAXW: tl.constexpr,
    ):
        # Fused exact overlay: sink + tail rows gathered from the exact
        # blocks into the fp16 temps. Grid (MAXW,) programs x (kvh*HD)
        # lanes; each program serves one window row, predicated. Positions,
        # paging and validity all resolve in-kernel: zero CPU syncs.
        # Matches _apply_exact_overlay row-for-row (absent blocks keep
        # the refreshed body: predicated skip, same as `continue`).
        pid = tl.program_id(0)
        lane = tl.arange(0, KVH * HD)
        h = lane // HD
        d = lane % HD
        n = tl.load(seqlens_ptr)
        sink_count = tl.minimum(n, SINK_N) if HAS_SINK else 0
        tail_start = tl.maximum(n - TAIL_EFF, 0)
        in_sink = pid < sink_count
        tp = pid - sink_count
        tail_count = n - tail_start
        active = (pid < sink_count + tail_count) & (pid < MAXW)
        pos = tl.where(in_sink, pid, tail_start + tp)
        # Clamp inactive lanes into range so every load below is safe;
        # all stores stay predicated on active.
        pos_safe = tl.where(active, pos, 0)
        page = tl.load(bt_ptr + pos_safe // 256)
        offs = pos_safe % 256
        g = page * GPS + offs // 128
        s = offs % 128
        valid = tl.load(exact_valid_ptr + g) & active
        e_off = (g * 128 + s) * KVH * HD + h * HD + d
        i_off = (page * 256 + offs) * KVH * HD + h * HD + d
        if TAIL_IS_BF16:
            ek = tl.load(exact_k_ptr + e_off, mask=valid).to(tl.float16)
            ev = tl.load(exact_v_ptr + e_off, mask=valid).to(tl.float16)
        else:
            ek = tl.load(exact_k_ptr + e_off, mask=valid)
            ev = tl.load(exact_v_ptr + e_off, mask=valid)
        tl.store(img_k_ptr + i_off, ek, mask=valid)
        tl.store(img_v_ptr + i_off, ev, mask=valid)


def kvarn_triton_overlay(image_k, image_v, layer, seqlens_1, bt_1,
                         gps, sink_tokens, tail_effective):
    """Fused exact overlay into fp16 image temps (non-SWA caller).

    One launch replaces the per-group Python loop (tolist + int syncs +
    per-group assigns). Absent exact blocks are skipped, matching the
    torch fallback row-for-row. Loud failure when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton overlay requires triton (import failed on this host).")
    if not image_k.is_cuda:
        raise RuntimeError(
            "KVarN Triton overlay requires CUDA tensors, got "
            f"{image_k.device}.")
    kvh, hd = layer.num_kv_heads, layer.head_dim
    dev = image_k.device
    maxw = sink_tokens + int(tail_effective)
    _kvarn_overlay_kernel[(maxw,)](
        image_k, image_v,
        layer.exact_k, layer.exact_v, layer.exact_valid,
        seqlens_1.to(dtype=torch.int32, device=dev),
        bt_1.to(dtype=torch.int32, device=dev),
        kvh, hd, gps,
        int(tail_effective), sink_tokens, bool(layer.has_sink),
        layer.tail_dtype == torch.bfloat16, maxw,
    )


def kvarn_triton_wht_slices(x):
    """
    Per-128-slice FWHT over the last dim (any head_dim = slices * 128),
    WITHOUT the cross-slice stage. Triton counterpart of applying
    ``kvarn_hadamard_128`` per slice: each 128-slice becomes one kernel
    row (HD=128, single warp -- see _fwht128_block comment). One launch.
    Loud failure (never silent) when the Triton path cannot run.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton WHT requires triton (import failed on this host).")
    if not x.is_cuda:
        raise RuntimeError(
            "KVarN Triton WHT requires CUDA tensors, got "
            f"{x.device}.")
    assert x.shape[-1] % 128 == 0
    prefix = x.shape[:-1]
    flat = x.float().reshape(-1, 128).contiguous()
    out = torch.empty_like(flat)
    out.copy_(flat)
    _kvarn_wht_hd_kernel[(flat.shape[0],)](out, 128, 1, 1.0, num_warps=1)
    return out.reshape(*prefix, x.shape[-1])


if _have_triton:
    @triton.jit
    def _kvarn_serve_gather_kernel(
        stage_k_ptr, stage_v_ptr,  # (G, 128, kvh, HD) fp16
        ids_ptr,                   # (D,) int64 group ids
        buf_k_ptr, buf_v_ptr,      # (D, 128, kvh, HD) fp32 out
        KVH: tl.constexpr, HD: tl.constexpr,
    ):
        # Gather staging rows to fp32. Bit-identical to the torch
        # .float() gather, including never-written slots: those read
        # whatever staging holds there (zeros by the static-tensor
        # reset invariant), exactly like the torch path, which applies
        # no present-mask either. Sealed groups in ids gather harmlessly
        # (their rows are discarded at scatter). Mask style (not
        # branches) matches the store/overlay kernels: masked-off lanes
        # issue no traffic and their values never reach a store.
        pid_d = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h2 = tl.program_id(2)
        is_k = pid_h2 < KVH
        h = pid_h2 % KVH
        lane = tl.arange(0, HD)
        gid = tl.load(ids_ptr + pid_d)
        s_off = (gid * 128 + pid_s) * KVH * HD + h * HD + lane
        d_off = (pid_d * 128 + pid_s) * KVH * HD + h * HD + lane
        tl.store(buf_k_ptr + d_off,
                 tl.load(stage_k_ptr + s_off, mask=is_k).to(tl.float32),
                 mask=is_k)
        tl.store(buf_v_ptr + d_off,
                 tl.load(stage_v_ptr + s_off, mask=(~is_k)).to(tl.float32),
                 mask=(~is_k))

    @triton.jit
    def _kvarn_serve_scatter_kernel(
        buf_k_ptr, buf_v_ptr,      # (D, 128, kvh, HD) fp32, fully WHT'd
        tgt_k_ptr, tgt_v_ptr,      # (Gg_flat, 128, kvh, HD) fp16 image/temps
        ids_ptr,                   # (D,) int64 group ids
        sealed_ptr,                # (G,) bool
        KVH: tl.constexpr, HD: tl.constexpr,
    ):
        # Scatter fp32 rows to the fp16 target, skipping sealed groups
        # (torch serves those via the dequant path -- disjoint sets, so
        # the split is order-irrelevant). Matches the torch indexed
        # scatter bit-exact (fp32 compute, single final cast).
        pid_d = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h2 = tl.program_id(2)
        is_k = pid_h2 < KVH
        h = pid_h2 % KVH
        lane = tl.arange(0, HD)
        gid = tl.load(ids_ptr + pid_d)
        go = ~tl.load(sealed_ptr + gid)
        d_off = (pid_d * 128 + pid_s) * KVH * HD + h * HD + lane
        t_off = (gid * 128 + pid_s) * KVH * HD + h * HD + lane
        # Combined masks (side AND seal): a V-program must never write
        # the K image and vice versa, even for open groups.
        tl.store(tgt_k_ptr + t_off,
                 tl.load(buf_k_ptr + d_off).to(tl.float16),
                 mask=go & is_k)
        tl.store(tgt_v_ptr + t_off,
                 tl.load(buf_v_ptr + d_off).to(tl.float16),
                 mask=go & (~is_k))


def kvarn_triton_serve_open(tgt_k, tgt_v, layer, ids):
    """Fused open-group refresh: gather staging + full head WHT + scatter.

    tgt_k/v: (pages, 256, kvh, hd) fp16 image or legacy temps. ids: (D,)
    int64 open-group ids (sealed members, if any, are skipped at
    scatter). 4 launches (gather, K WHT, V WHT, scatter), zero CPU
    syncs. Bit-identical to the torch rot path (same fp32 elementwise
    math in the same order); the WHT reuses the proven single-warp
    head kernel. Loud failure (never silent) when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton serve requires triton (import failed on this host).")
    if not tgt_k.is_cuda:
        raise RuntimeError(
            "KVarN Triton serve requires CUDA tensors, got "
            f"{tgt_k.device}.")
    kvh, hd = layer.num_kv_heads, layer.head_dim
    assert hd in (128, 256, 512)
    D = ids.numel()  # shape only, no sync
    if D == 0:
        return
    dev = tgt_k.device
    buf_k = torch.empty((D, 128, kvh, hd), dtype=torch.float32, device=dev)
    buf_v = torch.empty_like(buf_k)
    grid = (D, 128, 2 * kvh)
    _kvarn_serve_gather_kernel[grid](
        layer.stage_k, layer.stage_v, ids, buf_k, buf_v, kvh, hd)
    # bufs are fresh fp32 contiguous temps: transform in place
    # (same values, minus two allocs+copies).
    buf_k = kvarn_triton_wht_rows(buf_k, hd, inplace=True)
    buf_v = kvarn_triton_wht_rows(buf_v, hd, inplace=True)
    flat_k = tgt_k.reshape(-1, 128, kvh, hd)
    flat_v = tgt_v.reshape(-1, 128, kvh, hd)
    _kvarn_serve_scatter_kernel[grid](
        buf_k, buf_v, flat_k, flat_v, ids, layer.sealed, kvh, hd)


def kvarn_triton_dequant_groups(records_G, layout,
                               k_bits: int, v_bits: int,
                               num_kv_heads: int, slices: int,
                               do_wht: bool = False):
    """
    Batched multi-group entry: records_G (Gg, ncols, tile_bytes) uint8
    CUDA, ncols = kv_heads * slices. One kernel launch per side (K, V).

    Returns (bk, bv) float32 CUDA shaped (Gg, 128, kvh, hd) rotated-domain,
    matching ``CacheLayer_kvarn._dequant_groups_batched`` torch order
    (K transposed to [token, dim]). Bit-exact with the torch path: same
    fp32 elementwise math, fp32 store. With do_wht=True each side gets the
    per-128 FWHT (+ norm), i.e. the output already passed the per-slice
    ``kvarn_hadamard_128`` and only the cross-slice stage (if any) remains.

    K-axis note: K tiles are stored [dim, token] but served [token, dim],
    so the in-kernel row-FWHT (over tile columns = tokens) would transform
    the WRONG axis for K. K is therefore dequantized raw, transposed on
    assembly, then passed through ``kvarn_triton_wht_slices`` (FWHT over
    the head dim, one extra launch). V tiles are [token, dim): the
    in-kernel row-FWHT is already over the head dim.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton dequant requires triton (import failed on this host).")
    if not records_G.is_cuda:
        raise RuntimeError(
            "KVarN Triton dequant requires CUDA tensors, got "
            f"{records_G.device}.")
    ncols = num_kv_heads * slices
    Gg = records_G.shape[0]
    assert records_G.shape[1] == ncols
    rec_f16 = records_G.view(torch.float16)

    def side(payload_off, payload_bytes, sc_off, zp_off, oth_off, bits,
             wht):
        NT = Gg * ncols
        pay = records_G[:, :, payload_off: payload_off + payload_bytes] \
            .reshape(NT, payload_bytes).contiguous()
        sc = rec_f16[:, :, sc_off // 2: sc_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        zp = rec_f16[:, :, zp_off // 2: zp_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        oth = rec_f16[:, :, oth_off // 2: oth_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        return kvarn_triton_dequant_side(pay, sc, zp, oth, bits, wht)

    hd = slices * 128
    kt = side(layout.k_payload_off, layout.k_payload_bytes,
              layout.k_s_col_off, layout.k_zp_off, layout.k_s_row_off,
              k_bits, False).reshape(Gg, num_kv_heads, slices, 128, 128)
    bk_raw = kt.permute(0, 4, 1, 2, 3).reshape(Gg, 128, num_kv_heads, hd)
    # K-transpose first, FWHT over the head dim second (see K-axis note
    # above). do_wht=False above: the in-kernel row-FWHT stays off for K.
    bk = kvarn_triton_wht_slices(bk_raw) if do_wht else bk_raw
    vt = side(layout.v_payload_off, layout.v_payload_bytes,
              layout.v_s_row_off, layout.v_zp_off, layout.v_s_col_off,
              v_bits, do_wht).reshape(Gg, num_kv_heads, slices, 128, 128)
    bv = vt.permute(0, 3, 1, 2, 4).reshape(Gg, 128, num_kv_heads, hd)
    return bk, bv
