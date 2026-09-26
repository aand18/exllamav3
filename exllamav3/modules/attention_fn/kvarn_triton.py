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
        stage_k_ptr, stage_v_ptr,        # (S, 128, kvh, HD) fp16 slots
        exact_k_ptr, exact_v_ptr,        # (E, 128, kvh, HD) tail-dtype slots
        exact_valid_ptr,                 # (G,) bool
        base_ptr,                        # (G,) int64 group base
        sealed_ptr,                      # (G,) bool
        present_ptr,                     # (G, 128) bool
        owner_ptr,                       # (P,) int64 page owners
        pinned_ptr,                      # (P,) bool
        stage_rev_ptr,                   # (G,) int64 group -> staging slot
        exact_rev_ptr,                   # (G,) int64 group -> exact slot
        img_k_ptr, img_v_ptr,            # (P, 256, kvh, HD) fp16 image (or dummy)
        status_ptr,                      # (2,) int64 out: [code, group]
        KVH: tl.constexpr, HD: tl.constexpr, GPS: tl.constexpr,
        TAIL_KEEP: tl.constexpr, SINK_N: tl.constexpr,
        HAS_SINK: tl.constexpr, TAIL_IS_BF16: tl.constexpr,
        DO_IMG: tl.constexpr,
    ):
        # Fused single-row decode store (bsz 1, length 1, non-SWA).
        # Grid (2*KVH,) programs x HD lanes. All policy is predicated
        # (no Python branches on tensor values); steady appends need
        # zero CPU syncs to launch, one status read after.
        # Staging is slot-windowed: the slot comes from the host rev map
        # (fail-closed: unassigned, fresh, or sealed groups bail with
        # code 1 into the torch fallback, which assigns/resets host-side).
        # Exact is slot-windowed the same way: a keep-window row with no
        # exact slot yet bails too (host assigns via _alloc_exact_block;
        # valid ⟺ assigned, so steady appends never bail). Exact_valid
        # stays group-indexed (flag tensor, written in place).
        # Codes: 0 ok (image current via write-through), 1 slow-path
        # fallback, 2 seal group status[1].
        # Write-through keeps the persistent image current for pure
        # appends (the WHT'd row lands in staging AND the image), so the
        # per-step dirty sweep finds an empty mask and skips the open
        # refresh. Bit-identical to refresh-from-staging (same fp32 row,
        # single final RNE cast, WHT is per-row over HD). Pruned entirely
        # when the layer has no image yet (DO_IMG off: dummy pointers
        # never touched).
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
        slot = tl.load(stage_rev_ptr + g)
        eslot = tl.load(exact_rev_ptr + g)
        # Fail closed: fresh groups (any base change), sealed groups,
        # and unassigned staging slots all bail into the torch fallback,
        # which assigns/resets host-side. A keep-window row with no exact
        # slot yet bails the same way (host assigns). The fast path only
        # ever touches assigned slots of a known-open group. keep is
        # computed here (pure position math, no dependencies) so the
        # bail precedes every store below.
        keep = (pos >= n_new - TAIL_KEEP) | (HAS_SINK & (pos < SINK_N))
        slow = (base != bnew) | is_sealed | (slot < 0) | \
            (keep & (eslot < 0))
        go = ~slow

        row_k = tl.load(rk_ptr + h * HD + cols)
        row_v = tl.load(rv_ptr + h * HD + cols)
        s_off = (slot * 128 + s) * KVH * HD + h * HD + cols
        tl.store(stage_k_ptr + s_off, row_k.to(tl.float16),
                 mask=is_k & go)
        tl.store(stage_v_ptr + s_off, row_v.to(tl.float16),
                 mask=(~is_k) & go)
        tl.store(present_ptr + g * 128 + s, True, mask=go)
        # Image write-through (pure appends keep the image current, so
        # the sweep stays empty).
        if DO_IMG:
            i_off = (page * 256 + offs) * KVH * HD + h * HD + cols
            tl.store(img_k_ptr + i_off, row_k.to(tl.float16),
                     mask=is_k & go)
            tl.store(img_v_ptr + i_off, row_v.to(tl.float16),
                     mask=(~is_k) & go)

        # Exact blocks hold ORIGINAL-domain rows (ek/ev), unlike staging.
        # The slot is assigned whenever keep rows land here (unassigned
        # slots bailed above); clamp keeps the pointer in range on the
        # masked-off path all the same.
        eslot_c = tl.where(eslot >= 0, eslot, 0)
        e_off = (eslot_c * 128 + s) * KVH * HD + h * HD + cols
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
        # No device-side dirty: pure appends are image-current via
        # write-through; completion/fresh are reported by code and the
        # host dirties (it owns the Python-side dirty flag, which the
        # device cannot set).

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
                           gps, sink_tokens, rollback_tokens,
                           img_k=None, img_v=None):
    """Fused single-row decode store (bsz 1, length 1, non-SWA caller).

    Row WHT (1 launch) + fused write kernel (1 launch): stage/exact
    writes, present/base/owner/valid updates, image write-through.
    Staging is slot-windowed (slot from the host rev map; unassigned,
    fresh, or sealed groups bail with code 1 into the torch fallback,
    which assigns/resets host-side). Exact is slot-windowed the same
    way (a keep-window row with no exact slot yet bails; valid ⟺
    assigned, so steady appends never bail). A completed group reports
    code 2 and its id for the torch sealer. Returns (code: int,
    group: int) -- one status read, the only CPU sync. Small ints (gps,
    sink/rollback sizes) come from the caller so this module never
    imports the cache package (no cycle, no stub fragility). Loud
    failure (never silent) when unrunnable.

    img_k/img_v: the persistent fp16 image for write-through, or None
    (no image yet / legacy temps: the write-through is pruned and the
    refresh path owns the rows as before).
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
    # One convert (not two): stack the fp16 rows, then widen once.
    # Identical values, one fewer dispatch + one fewer temp.
    stacked = torch.stack((rows_k, rows_v)).float()
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
    if img_k is None:
        img_k, img_v, do_img = rows_k, rows_v, False
    else:
        do_img = True
    # Callers pass long already (block_table.long() upstream): skip the
    # no-op converts (same tensor object .to() would return, minus three
    # dispatches per layer per step).
    if pages_1.dtype != torch.int64:
        pages_1 = pages_1.to(torch.int64)
    if offs_1.dtype != torch.int64:
        offs_1 = offs_1.to(torch.int64)
    if pos_1.dtype != torch.int64:
        pos_1 = pos_1.to(torch.int64)
    _kvarn_store_row_kernel[(2 * kvh,)](
        rk, rv,
        rows_k.reshape(kvh, hd).contiguous(),
        rows_v.reshape(kvh, hd).contiguous(),
        pages_1, offs_1, pos_1,
        layer.stage_k, layer.stage_v,
        layer.exact_k, layer.exact_v, layer.exact_valid,
        layer.group_base, layer.sealed, layer.present,
        layer.page_owner_n, layer.page_pinned,
        layer._stage_rev, layer._exact_rev,
        img_k, img_v,
        status,
        kvh, hd, gps,
        tail_keep, sink_n, bool(layer.has_sink),
        layer.tail_dtype == torch.bfloat16,
        do_img,
    )
    code, g = status.tolist()
    return int(code), int(g)


if _have_triton:
    @triton.jit
    def _kvarn_overlay_kernel(
        img_k_ptr, img_v_ptr,             # (P, 256, kvh, HD) fp16 temps
        exact_k_ptr, exact_v_ptr,         # (E, 128, kvh, HD) tail-dtype slots
        exact_valid_ptr,                  # (G,) bool
        exact_rev_ptr,                    # (G,) int64 group -> exact slot
        seqlens_ptr, bt_ptr,              # (1,) int32 n, (P,) int32 pages
        sk_ptr, sv_ptr,                   # (MAXW, kvh, HD) fp16 stash
        slin_ptr, svalid_ptr,             # (MAXW,) int32 lin / bool valid
        KVH: tl.constexpr, HD: tl.constexpr, GPS: tl.constexpr,
        TAIL_EFF: tl.constexpr, SINK_N: tl.constexpr,
        HAS_SINK: tl.constexpr, TAIL_IS_BF16: tl.constexpr, MAXW: tl.constexpr,
        DO_STASH: tl.constexpr,
    ):
        # Fused exact overlay: sink + tail rows gathered from the exact
        # blocks into the fp16 temps. Grid (MAXW,) programs x (kvh*HD)
        # lanes; each program serves one window row, predicated. Positions,
        # paging and validity all resolve in-kernel: zero CPU syncs.
        # Exact is slot-windowed (slot from the host rev map; valid ⟺
        # assigned, belt-and-braces the slot sign into validity).
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
        eslot = tl.load(exact_rev_ptr + g)
        valid = tl.load(exact_valid_ptr + g) & active & (eslot >= 0)
        eslot_c = tl.where(eslot >= 0, eslot, 0)
        e_off = (eslot_c * 128 + s) * KVH * HD + h * HD + d
        i_off = (page * 256 + offs) * KVH * HD + h * HD + d
        if TAIL_IS_BF16:
            ek = tl.load(exact_k_ptr + e_off, mask=valid).to(tl.float16)
            ev = tl.load(exact_v_ptr + e_off, mask=valid).to(tl.float16)
        else:
            ek = tl.load(exact_k_ptr + e_off, mask=valid)
            ev = tl.load(exact_v_ptr + e_off, mask=valid)
        # Stash-first (active rows only; every pid owns its slot, so no
        # cross-program race): stash the pre-overlay rows, then land the
        # overlay. The restore kernel in update_kv puts these back after
        # the forward, which retires the 2 full-image clones. valid[]
        # doubles as the activity record: inactive pids clear it every
        # call, so restore never replays a stale row. Pruned entirely on
        # throwaway temps (DO_STASH off: dummy pointers never touched).
        if DO_STASH:
            s_off = pid * KVH * HD + h * HD + d
            oldk = tl.load(img_k_ptr + i_off, mask=active)
            oldv = tl.load(img_v_ptr + i_off, mask=active)
            tl.store(sk_ptr + s_off, oldk, mask=active)
            tl.store(sv_ptr + s_off, oldv, mask=active)
            tl.store(slin_ptr + pid, page * 256 + offs, mask=active)
            tl.store(svalid_ptr + pid, active)
        tl.store(img_k_ptr + i_off, ek, mask=valid)
        tl.store(img_v_ptr + i_off, ev, mask=valid)


if _have_triton:
    @triton.jit
    def _kvarn_unoverlay_kernel(
        img_k_ptr, img_v_ptr,             # (P, 256, kvh, HD) fp16 image
        sk_ptr, sv_ptr,                   # (MAXW, kvh, HD) fp16 stash
        slin_ptr, svalid_ptr,             # (MAXW,) int32 lin / bool valid
        KVH: tl.constexpr, HD: tl.constexpr, MAXW: tl.constexpr,
    ):
        # Restore stashed rows after the forward consumed the overlay.
        # Separate launch from the overlay (grid barrier): the tail slides
        # every step, so restore rows and overlay rows of consecutive
        # calls overlap under different pid mappings -- same-kernel
        # ordering would be a cross-program race. Consumes the valid bits
        # (clears them) so a later call without an overlay is a no-op.
        pid = tl.program_id(0)
        lane = tl.arange(0, KVH * HD)
        h = lane // HD
        d = lane % HD
        v = tl.load(svalid_ptr + pid)
        lin = tl.load(slin_ptr + pid)
        s_off = pid * KVH * HD + h * HD + d
        i_off = lin * KVH * HD + h * HD + d
        tl.store(img_k_ptr + i_off, tl.load(sk_ptr + s_off), mask=v)
        tl.store(img_v_ptr + i_off, tl.load(sv_ptr + s_off), mask=v)
        tl.store(svalid_ptr + pid, False)


def _kvarn_overlay_stash(layer, maxw, dev):
    """Lazy persistent overlay-stash buffers (no per-call alloc)."""
    sk = getattr(layer, "_ov_stash_k", None)
    if sk is None or sk.shape[0] != maxw:
        kvh, hd = layer.num_kv_heads, layer.head_dim
        layer._ov_stash_k = torch.zeros((maxw, kvh, hd),
                                        dtype=torch.half, device=dev)
        layer._ov_stash_v = torch.zeros((maxw, kvh, hd),
                                        dtype=torch.half, device=dev)
        layer._ov_lin = torch.zeros((maxw,), dtype=torch.int32, device=dev)
        layer._ov_valid = torch.zeros((maxw,), dtype=torch.bool, device=dev)
        layer._ov_pending = False
    return (layer._ov_stash_k, layer._ov_stash_v,
            layer._ov_lin, layer._ov_valid)


def kvarn_triton_overlay(image_k, image_v, layer, seqlens_1, bt_1,
                         gps, sink_tokens, tail_effective, stash=None):
    """Fused exact overlay into fp16 image temps (non-SWA caller).

    One launch replaces the per-group Python loop (tolist + int syncs +
    per-group assigns). Exact blocks are slot-windowed (slot from the
    host rev map, resolved in-kernel; absent blocks are skipped,
    matching the torch fallback row-for-row). Loud failure when
    unrunnable.

    stash: None (overlay onto throwaway temps: legacy path, unit tests)
    or the tuple from _kvarn_overlay_stash (serve-from-image: rows are
    stashed in-kernel for kvarn_triton_unoverlay after the forward).
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
    if stash is None:
        sk, sv, slin, svalid, do_stash = image_k, image_v, image_k, image_k, False
    else:
        sk, sv, slin, svalid, do_stash = (*stash, True)
    _kvarn_overlay_kernel[(maxw,)](
        image_k, image_v,
        layer.exact_k, layer.exact_v, layer.exact_valid, layer._exact_rev,
        seqlens_1.to(dtype=torch.int32, device=dev),
        bt_1.to(dtype=torch.int32, device=dev),
        sk, sv, slin, svalid,
        kvh, hd, gps,
        int(tail_effective), sink_tokens, bool(layer.has_sink),
        layer.tail_dtype == torch.bfloat16, maxw,
        do_stash,
    )
    if do_stash:
        layer._ov_pending = True


def kvarn_triton_unoverlay(image_k, image_v, layer):
    """Restore stashed rows after the forward consumed the overlay.

    Gated by the layer's pending flag (plain bool, zero syncs): without
    a preceding stashed overlay this is a no-op. Loud failure when
    unrunnable.
    """
    if not bool(getattr(layer, "_ov_pending", False)):
        return
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton unoverlay requires triton (import failed on this host).")
    kvh, hd = layer.num_kv_heads, layer.head_dim
    maxw = int(layer._ov_valid.shape[0])
    _kvarn_unoverlay_kernel[(maxw,)](
        image_k, image_v,
        layer._ov_stash_k, layer._ov_stash_v,
        layer._ov_lin, layer._ov_valid,
        kvh, hd, maxw,
    )
    layer._ov_pending = False


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
        stage_k_ptr, stage_v_ptr,  # (S, 128, kvh, HD) fp16 slots
        slot_ptr,                  # (D,) int64 staging slot ids
        buf_k_ptr, buf_v_ptr,      # (D, 128, kvh, HD) fp32 out
        KVH: tl.constexpr, HD: tl.constexpr,
    ):
        # Gather staging rows to fp32. Bit-identical to the torch
        # .float() gather, including never-written groups: those have no
        # slot (-1) and serve zeros (the old static zero invariant),
        # matching the torch clamp+zero-fill. Mask style (not branches)
        # matches the store/overlay kernels: the clamped address keeps
        # every pointer in range, masked-off lanes issue no traffic,
        # other=0.0 fills the served zeros. Sealed groups in ids gather
        # harmlessly (their rows are discarded at scatter).
        pid_d = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h2 = tl.program_id(2)
        is_k = pid_h2 < KVH
        h = pid_h2 % KVH
        lane = tl.arange(0, HD)
        slot = tl.load(slot_ptr + pid_d)
        ok = slot >= 0
        slot_c = tl.where(ok, slot, 0)
        s_off = (slot_c * 128 + pid_s) * KVH * HD + h * HD + lane
        d_off = (pid_d * 128 + pid_s) * KVH * HD + h * HD + lane
        tl.store(buf_k_ptr + d_off,
                 tl.load(stage_k_ptr + s_off, mask=is_k & ok,
                         other=0.0).to(tl.float32),
                 mask=is_k)
        tl.store(buf_v_ptr + d_off,
                 tl.load(stage_v_ptr + s_off, mask=(~is_k) & ok,
                         other=0.0).to(tl.float32),
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


def kvarn_triton_serve_open(tgt_k, tgt_v, layer, ids, slots):
    """Fused open-group refresh: gather staging + full head WHT + scatter.

    tgt_k/v: (pages, 256, kvh, hd) fp16 image or legacy temps. ids: (D,)
    int64 open-group ids (sealed members, if any, are skipped at
    scatter); slots: (D,) int64 staging slot ids gathered from.
    4 launches (gather, K WHT, V WHT, scatter), zero CPU syncs.
    Bit-identical to the torch rot path (same fp32 elementwise
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
        layer.stage_k, layer.stage_v, slots, buf_k, buf_v, kvh, hd)
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


if _have_triton:
    @triton.jit
    def _kvarn_online_kcol(gbase_u8, gbase_f16, C: tl.constexpr, B: tl.constexpr,
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
    def _kvarn_online_vrow(gbase_u8, gbase_f16, C: tl.constexpr, B: tl.constexpr,
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
    def _kvarn_online_block_kernel(
        qw_ptr, rec_ptr, rec_f16_ptr, ids_ptr,
        m_ptr, l_ptr, out_ptr,  # (KVH, QPK, NB), ..., (KVH, QPK, NB, HD)
        K_PAY_OFF, K_SC2, K_ZP2, K_OT2, K_BITS: tl.constexpr,
        V_PAY_OFF, V_SC2, V_ZP2, V_OT2, V_BITS: tl.constexpr,
        C: tl.constexpr, B: tl.constexpr, SL: tl.constexpr,
        KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
        HD: tl.constexpr, NB: tl.constexpr, SCALE: tl.constexpr,
        seq_ptr,                     # (1,) int32: current sequence length n
        SINK_N: tl.constexpr, TAIL_EFF: tl.constexpr,
    ):
        """One program = one (kv head, sealed group). 32 outer iters x 4
        tokens with a joint block softmax update (depth 32, not 128).

        Position-masked (match-bee Task 3): only rows in
        [SINK_N, n - TAIL_EFF) participate (body); sink/tail-overlap rows
        are served exact by the tail block, so gathering them here would
        double-count. Masked scores are -inf (zero weight). Fully masked
        groups emit m=-1e30/l=0 (the combine guards den==0).
        m starts at finite -1e30, NOT -inf: with all-masked rows,
        alpha = exp(m - m_new) would be exp(NaN); the finite floor keeps
        every lane NaN-free and is numerically identical otherwise
        (exp underflows to 0 either way).
        """
        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        g = tl.load(ids_ptr + pid_b)
        n = tl.load(seq_ptr)
        tail_start = n - TAIL_EFF
        lane = tl.arange(0, HD)
        qoff = tl.arange(0, QPAD)
        qmask = qoff < QPK
        q = tl.load(qw_ptr + (pid_h * QPK) * HD + qoff[:, None] * HD
                    + lane[None, :], mask=qmask[:, None], other=0.0)
        m = tl.full([QPAD], -1e30, dtype=tl.float32)
        l = tl.zeros([QPAD], dtype=tl.float32)
        acc = tl.zeros([QPAD, HD], dtype=tl.float32)
        gbase_u8 = rec_ptr + g * C * B
        gbase_f16 = rec_f16_ptr + (g * C * B) // 2
        for t0 in tl.range(32):
            t = t0 * 4
            k0 = _kvarn_online_kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                       K_BITS, pid_h, SL, t, lane)
            k1 = _kvarn_online_kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                       K_BITS, pid_h, SL, t + 1, lane)
            k2 = _kvarn_online_kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                       K_BITS, pid_h, SL, t + 2, lane)
            k3 = _kvarn_online_kcol(gbase_u8, gbase_f16, C, B, K_PAY_OFF, K_SC2, K_ZP2, K_OT2,
                       K_BITS, pid_h, SL, t + 3, lane)
            a0 = (g * 128 + t >= SINK_N) & (g * 128 + t < tail_start) & (g * 128 + t < n)
            a1 = (g * 128 + t + 1 >= SINK_N) & (g * 128 + t + 1 < tail_start) & (g * 128 + t + 1 < n)
            a2 = (g * 128 + t + 2 >= SINK_N) & (g * 128 + t + 2 < tail_start) & (g * 128 + t + 2 < n)
            a3 = (g * 128 + t + 3 >= SINK_N) & (g * 128 + t + 3 < tail_start) & (g * 128 + t + 3 < n)
            s0 = tl.where(a0, tl.sum(q * k0[None, :], axis=1) * SCALE, float("-inf"))
            s1 = tl.where(a1, tl.sum(q * k1[None, :], axis=1) * SCALE, float("-inf"))
            s2 = tl.where(a2, tl.sum(q * k2[None, :], axis=1) * SCALE, float("-inf"))
            s3 = tl.where(a3, tl.sum(q * k3[None, :], axis=1) * SCALE, float("-inf"))
            smax = tl.maximum(tl.maximum(s0, s1), tl.maximum(s2, s3))
            m_new = tl.maximum(m, smax)
            alpha = tl.exp(m - m_new)
            e0 = tl.exp(s0 - m_new)
            e1 = tl.exp(s1 - m_new)
            e2 = tl.exp(s2 - m_new)
            e3 = tl.exp(s3 - m_new)
            l = l * alpha + e0 + e1 + e2 + e3
            v0 = _kvarn_online_vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
                       V_BITS, pid_h, SL, t, lane)
            v1 = _kvarn_online_vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
                       V_BITS, pid_h, SL, t + 1, lane)
            v2 = _kvarn_online_vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
                       V_BITS, pid_h, SL, t + 2, lane)
            v3 = _kvarn_online_vrow(gbase_u8, gbase_f16, C, B, V_PAY_OFF, V_SC2, V_ZP2, V_OT2,
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
    def _kvarn_online_combine_kernel(
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
        # NaN-proof: a fully masked call (short prefix, everything tail)
        # has den == 0; serve zeros (the tail merge owns the output).
        # tl.where selects elementwise: no trap on the discarded NaN.
        row = tl.where(den > 0, num / den, 0.0)
        base = out_ptr + pid * HD
        tl.store(base + lane, row)
        for _sl in tl.static_range(4):
            if _sl < SL:
                _fwht128_block(base + _sl * 128, tl.arange(0, 128))
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




if _have_triton:
    @triton.jit
    def _kvarn_online_qwht_kernel(
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




def _kvarn_online_buffers(layer, qh, qpad, hd, dev):
    """Lazy persistent online-serve workspace (no per-step allocs).

    Sized for max groups (num_groups); steps slice [:nb]. Mirrors the
    _ov_stash pattern: allocate once, reuse across steps. Holds (m, l)
    block stats, fp32 block outputs, WHT'd queries + Q-WHT scratch, and
    the final output rows. Small vs the image it replaces (~2.5MB/layer
    at 8k-class geometry vs 35MB image).
    """
    G = int(layer.num_groups)
    need = (qh, qpad, G, hd)
    have = getattr(layer, "_ov_online_shape", None)
    if have != need or getattr(layer, "_ov_online_m", None) is None:
        kvh = int(layer.num_kv_heads)
        layer._ov_online_m = torch.empty((kvh, qpad, G), dtype=torch.float32,
                                         device=dev)
        layer._ov_online_l = torch.empty((kvh, qpad, G), dtype=torch.float32,
                                         device=dev)
        layer._ov_online_acc = torch.empty((kvh, qpad, G, hd),
                                           dtype=torch.float32, device=dev)
        layer._ov_online_qw = torch.empty((qh, hd), dtype=torch.float32,
                                          device=dev)
        layer._ov_online_qs = torch.empty((qh, hd), dtype=torch.float32,
                                          device=dev)
        layer._ov_online_out = torch.empty((qh, hd), dtype=torch.float32,
                                           device=dev)
        layer._ov_online_shape = need
    return (layer._ov_online_m, layer._ov_online_l, layer._ov_online_acc,
            layer._ov_online_qw, layer._ov_online_qs, layer._ov_online_out)


def kvarn_triton_qwht(q, scratch, out, slices, sscale):
    """Fused fp16->fp32 convert + full head WHT for tiny Q batches.

    One launch replaces convert + alloc + copy + WHT (5 dispatches).
    Stage order identical to _kvarn_wht_hd_kernel: bit-exact vs
    kvarn_triton_wht_rows. Loud failure (never silent) when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton Q-WHT requires triton (import failed on this host).")
    if not q.is_cuda:
        raise RuntimeError(
            "KVarN Triton Q-WHT requires CUDA tensors, got "
            f"{q.device}.")
    qh, hd = q.shape
    assert out.shape == (qh, hd) and scratch.shape == (qh, hd)
    assert out.dtype == torch.float32 and scratch.dtype == torch.float32
    assert slices * 128 == hd
    _kvarn_online_qwht_kernel[(qh,)](q, scratch, out, qh, hd, slices,
                                     sscale, num_warps=1)
    return out


def kvarn_triton_online_decode(layer, qw, ids, qpk, scale, n_new,
                               sink_n, tail_eff):
    """Fused online attention over sealed groups (imageless serve).

    qw: (QH, HD) fp32 WHT'd queries (via kvarn_triton_qwht). ids: (NB,)
    int64 sealed group ids (need not exclude tail-overlapping groups).
    n_new: (1,) int32 current sequence length. Only rows in
    [sink_n, n_new - tail_eff) participate (body); sink/tail-overlap
    rows are served exact by the tail block, so gathering them here
    would double-count. Fully masked calls emit zeros (merge-owned).
    Returns (QH, HD) fp32 out in the ORIGINAL domain (out-WHT folded
    into the combine). 2 launches + 0 syncs (NB from shape). Tail/open
    rows are NOT covered here: the caller serves them from
    exact/staging and merges (global online combine). Loud failure
    (never silent) when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton online decode requires triton (import failed).")
    dev = qw.device
    if dev.type != "cuda":
        raise RuntimeError(
            "KVarN Triton online decode requires CUDA tensors, got "
            f"{qw.device}.")
    kvh, hd = int(layer.num_kv_heads), int(layer.head_dim)
    sl = int(layer.slices)
    assert hd == sl * 128
    qh = kvh * qpk
    assert qw.shape == (qh, hd)
    nb = int(ids.numel())
    if nb == 0:
        return torch.zeros((qh, hd), dtype=torch.float32, device=dev)
    qpad = 1 << (qpk - 1).bit_length()
    nbpad = 1 << (nb - 1).bit_length()
    m, l, acc, _, _, out = _kvarn_online_buffers(layer, qh, qpad, hd, dev)
    L = layer.layout
    rec = layer.records
    rec_f16 = rec.view(torch.float16)
    _kvarn_online_block_kernel[(kvh, nb,)](
        qw, rec, rec_f16, ids, m, l, acc,
        L.k_payload_off,
        L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2,
        int(layer.k_bits),
        L.v_payload_off,
        L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2,
        int(layer.v_bits),
        rec.shape[1], rec.shape[2], sl,
        kvh, qpk, qpad, hd, nb, scale,
        n_new, int(sink_n), int(tail_eff),
        num_warps=4)
    _kvarn_online_combine_kernel[(qh,)](
        m, l, acc, out, kvh, qpk, qpad, nb, nbpad, hd, sl,
        1.0 if sl == 1 else (0.7071067811865475 if sl == 2 else 0.5),
        num_warps=1)
    return out
