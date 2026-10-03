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
        K_TPOSE: tl.constexpr = 0,  # 1: K payload stored [token, dim] (v6):
            # sc/zp index columns (dims), other indexes rows (tokens).
            # V and pre-v6 K keep 0 (sc/zp on rows, other on columns).
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

        if K_TPOSE == 0:
            sc = tl.load(sc_ptr + tile * 128 + row).to(tl.float32)
            zp = tl.load(zp_ptr + tile * 128 + row).to(tl.float32)
            oth = tl.load(oth_ptr + tile * 128 + cols).to(tl.float32)
        else:
            sc = tl.load(sc_ptr + tile * 128 + cols).to(tl.float32)
            zp = tl.load(zp_ptr + tile * 128 + cols).to(tl.float32)
            oth = tl.load(oth_ptr + tile * 128 + row).to(tl.float32)
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
                               bits: int, do_wht: bool = False,
                               k_tpose: bool = False) -> torch.Tensor:
    """
    Dequantize NT tiles of one side (K or V).

    payload: (NT, PAY) uint8 CUDA. sc/zp: (NT, 128) fp16 CUDA (per row).
    oth: (NT, 128) fp16 CUDA (per col). Returns (NT, 128, 128) fp32 CUDA.
    With do_wht=True each tile additionally gets the 128-point FWHT (+
    norm), matching kvarn_hadamard_128 bit-exact.
    k_tpose=True: K payload stored [token, dim] (v6) -- sc/zp index tile
    columns (dims), other indexes rows (tokens). V and pre-v6 K keep the
    row/column convention. Loud failure (never silent) when the Triton
    path cannot run.
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
                                    1 if k_tpose else 0,
                                    num_warps=1 if do_wht else 4)
    return out


def kvarn_triton_dequant_group(records_g: torch.Tensor, layout,
                               k_bits: int, v_bits: int,
                               num_kv_heads: int, slices: int):
    """
    Dequantize one sealed group's combined records to rotated-domain fp32.

    records_g: (ncols, tile_bytes) uint8 CUDA, ncols = kv_heads * slices.
    Returns (bk, bv) float32 CUDA shaped (128, kvh, hd), matching the torch
    loop in CacheLayer_kvarn (K tiles stored [token, dim], V tiles
    [token, dim] as-is). Scale gather is plain torch slicing
    (device ops); only unpack+dequant is fused Triton.
    """
    ncols = num_kv_heads * slices
    assert records_g.shape[0] == ncols
    rec_f16 = records_g.view(torch.float16)

    def side(payload_off, payload_bytes, sc_off, zp_off, oth_off, bits,
             k_tpose=False):
        pay = records_g[:, payload_off: payload_off + payload_bytes]
        sc = rec_f16[:, sc_off // 2: sc_off // 2 + 128]
        zp = rec_f16[:, zp_off // 2: zp_off // 2 + 128]
        oth = rec_f16[:, oth_off // 2: oth_off // 2 + 128]
        return kvarn_triton_dequant_side(pay.contiguous(), sc.contiguous(),
                                         zp.contiguous(), oth.contiguous(),
                                         bits, False, k_tpose).float()

    k_tiles = side(layout.k_payload_off, layout.k_payload_bytes,
                   layout.k_s_col_off, layout.k_zp_off, layout.k_s_row_off,
                   k_bits, True)   # (ncols, 128, 128) [token, dim]
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
            bk[:, h, d0:d1] = k_tiles[c]
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
        eref_v_ptr,                      # (E, 128, kvh, HD) fp32 WHT'd-V cache
        status_ptr,                      # (2,) int64 out: [code, group]
        KVH: tl.constexpr, HD: tl.constexpr, GPS: tl.constexpr,
        TAIL_KEEP: tl.constexpr, SINK_N: tl.constexpr,
        HAS_SINK: tl.constexpr, TAIL_IS_BF16: tl.constexpr,
        DO_IMG: tl.constexpr, DO_EREF: tl.constexpr,
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
        # Eref write-through (incremental eref cache): the fp32 WHT'd V
        # row lands directly in the cache slot, so the host pays no rev
        # sync and no 128-row refresh launch. Bit-identical to a slot
        # refresh: same fp32 row (rv), same e_off, V rows only. Pruned
        # entirely when the cache is not yet built (DO_EREF off: dummy
        # pointer never touched).
        if DO_EREF:
            tl.store(eref_v_ptr + e_off, row_v,
                     mask=(~is_k) & keep & go)

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
    # Incremental eref write-through: the kernel stores the fp32 WHT'd V
    # row directly (no host rev sync, no refresh launch). Cache-absent
    # (None, pre-first-serve) prunes the write; the dummy is never
    # touched (same pattern as DO_IMG above).
    eref_w = layer._ov_eref_w
    if eref_w is None:
        eref_w, do_eref = rows_v, False
    else:
        do_eref = True
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
        eref_w,
        status,
        kvh, hd, gps,
        tail_keep, sink_n, bool(layer.has_sink),
        layer.tail_dtype == torch.bfloat16,
        do_img, do_eref,
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

    K-axis note: K tiles are stored [token, dim] (slot-major, same as V)
    and served [token, dim], so no transpose on assembly (pre-v6 records
    stored [dim, token] and needed one). do_wht stays False for K with
    the separate ``kvarn_triton_wht_slices`` launch (unchanged path);
    folding it in is queued (columns are now dims, so the row-FWHT would
    hit the right axis). V tiles are [token, dim): the in-kernel
    row-FWHT is already over the head dim.
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
             wht, k_tpose=False):
        NT = Gg * ncols
        pay = records_G[:, :, payload_off: payload_off + payload_bytes] \
            .reshape(NT, payload_bytes).contiguous()
        sc = rec_f16[:, :, sc_off // 2: sc_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        zp = rec_f16[:, :, zp_off // 2: zp_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        oth = rec_f16[:, :, oth_off // 2: oth_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        return kvarn_triton_dequant_side(pay, sc, zp, oth, bits, wht,
                                         k_tpose)

    hd = slices * 128
    kt = side(layout.k_payload_off, layout.k_payload_bytes,
              layout.k_s_col_off, layout.k_zp_off, layout.k_s_row_off,
              k_bits, False, True).reshape(Gg, num_kv_heads, slices, 128, 128)
    # Stored [token, dim]: serve rows map straight (was: stored
    # [dim, token], permute (0,4,1,2,3) transposed on assembly).
    bk_raw = kt.permute(0, 3, 1, 2, 4).reshape(Gg, 128, num_kv_heads, hd)
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
        # K payload is stored [token, dim] (slot-major rows; same as V),
        # so value (t, dd) sits at stream offset t*128+dd.
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
        DO_WHT: tl.constexpr = 1,
    ):
        """One program = one q-head: online-reduce NB block partials, then
        full head WHT in place (per-slice FWHT + cross-slice + scale, same
        order as _kvarn_wht_hd_kernel). num_warps=1 REQUIRED when the
        folded WHT runs (DO_WHT=1: _fwht128_block is single-warp-only).
        DO_WHT=0 skips the WHT (out stays normalized-domain); the
        caller then runs _kvarn_wht_hd_kernel separately, which frees
        the reduce to num_warps=4. NB pads to NBPAD (pow2); scalar
        block weights come from one-hot selects."""
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
        # Tight bound NB (not NBPAD): iters b>=NB contribute exactly +0.0
        # (e is 0 there via -inf padding, ab masked to 0.0; den is computed
        # above the loop), so skipping them is bit-identical while cutting
        # e.g. 128->64 iters at 8k/GMAX=64.
        for b in tl.range(NB):
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
        if DO_WHT:
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

    def _kvarn_launch_combine(m, l, acc, out, kvh, qpk, qpad, nb, nbpad,
                              hd, sl, sscale):
        """Combine reduce + out-WHT with the WHT split into its own
        launch (EXL3_KVARN_COMBINE_SPLIT=1, default): reduce-only
        combine at num_warps=4 (the serial b-loop is unchanged, only
        the independent HD/NB lanes spread over warps) followed by
        the proven _kvarn_wht_hd_kernel. =0 restores the legacy
        single-launch folded path (num_warps=1). Fail-closed: any
        throw in the split path falls back to the legacy launch,
        loudly (never half-runs: the fallback rewrites out fully)."""
        import os as _os
        qh = int(out.shape[0])
        if _os.environ.get("EXL3_KVARN_COMBINE_SPLIT", "1") == "1":
            try:
                _kvarn_online_combine_kernel[(qh,)](
                    m, l, acc, out, kvh, qpk, qpad, nb, nbpad, hd, sl,
                    sscale, 0, num_warps=4)
                _kvarn_wht_hd_kernel[(qh,)](out, hd, sl, sscale)
                return
            except Exception as _e:
                print("KVARN-COMBINE split failed, legacy fallback: "
                      f"{_e}", flush=True)
        _kvarn_online_combine_kernel[(qh,)](
            m, l, acc, out, kvh, qpk, qpad, nb, nbpad, hd, sl,
            sscale, 1, num_warps=1)


if _have_triton:
    @triton.jit
    def _kvarn_online_merge_kernel(
        m_ptr, l_ptr,             # (KVH, QPAD, GC) fp32 serve partials
        out_b_ptr,               # (QH, HD) fp32 body out (original domain)
        tail_m_ptr, tail_den_ptr,  # (QH,) fp32 tail stats
        tail_num_ptr,            # (QH, HD) fp32 tail block
        out_ptr,                 # (QH, HD) fp16 merged output
        KVH: tl.constexpr, QPK: tl.constexpr, QPAD: tl.constexpr,
        GC: tl.constexpr, GCPAD: tl.constexpr, HD: tl.constexpr,
    ):
        """One program = one q-head: body (m, den) stats + original-domain
        merge with the torch tail block. Replaces ~13 torch dispatches
        (stats amax/exp/sum + merge maximum/exp/mul/add/div/half) with one
        launch. Same per-element math as the torch merge (gc-contiguous
        accumulation; elementwise exp), so results agree to fp32 assoc
        noise (~1e-7 relative, far inside the 5e-4 arm gate). GC pads to
        GCPAD (pow2) with -inf/0.0 like the combine kernel: padded lanes
        contribute exactly +0.0, so the bound is bit-identical. The den==0
        NaN-proof mirrors the combine kernel (fully masked short prefix:
        the tail owns the output). num_warps=4."""
        pid = tl.program_id(0)
        ph = pid // QPK
        pq = pid % QPK
        coff = tl.arange(0, GCPAD)
        cmask = coff < GC
        m = tl.load(m_ptr + (ph * QPAD + pq) * GC + coff, mask=cmask,
                    other=float("-inf"))
        l = tl.load(l_ptr + (ph * QPAD + pq) * GC + coff, mask=cmask,
                    other=0.0)
        m_b = tl.max(m)
        den_b = tl.sum(l * tl.exp(m - m_b))
        lane = tl.arange(0, HD)
        tm = tl.load(tail_m_ptr + pid)
        td = tl.load(tail_den_ptr + pid)
        m_g = tl.maximum(m_b, tm)
        eb = tl.exp(m_b - m_g)
        et = tl.exp(tm - m_g)
        den = den_b * eb + td * et
        nb = tl.load(out_b_ptr + pid * HD + lane) * den_b
        tn = tl.load(tail_num_ptr + pid * HD + lane)
        row = tl.where(den > 0, (nb * eb + tn * et) / den, 0.0)
        tl.store(out_ptr + pid * HD + lane, row.to(tl.float16))


def kvarn_triton_online_merge(m, l, out_b, tail_m, tail_den, tail_num,
                              qpk, gc, _out=None):
    """Fused body-stats + original-domain merge for the imageless arm.

    m/l: (kvh, qpad, gc) fp32 serve partials; out_b: (qh, hd) fp32 body
    output (combine already folded the out-WHT: original domain, so it
    un-normalizes by den with NO extra WHT -- see the cacd7af fix);
    tail_m/tail_den (qh,) + tail_num (qh, hd) fp32 torch tail block.
    Returns (qh, hd) fp16. Loud failure when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton online merge requires triton (import failed).")
    dev = out_b.device
    if dev.type != "cuda":
        raise RuntimeError(
            "KVarN Triton online merge requires CUDA tensors, got "
            f"{out_b.device}.")
    qh, hd = out_b.shape
    # Optional persistent out (dispatch passes a layer buffer: same
    # values, minus 1 alloc/layer/step; shape static per layer).
    out = _out if _out is not None else torch.empty(
        (qh, hd), dtype=torch.float16, device=dev)
    gcpad = 1 << (gc - 1).bit_length()
    _kvarn_online_merge_kernel[(qh,)](
        m, l, out_b, tail_m, tail_den, tail_num, out,
        m.shape[0], qpk, m.shape[1], gc, gcpad, hd, num_warps=4)
    return out


if _have_triton:
    @triton.jit
    def _kvarn_online_tail_kernel(
        st_ptr,                  # (KVH, QPK, R) fp32 scaled tail scores
        vt_ptr,                  # (R, KVH, HD) fp32 tail values
        tg_ptr,                  # (R,) int64 tail group ids
        exrev_ptr,               # (G,) int64 group -> exact slot
        tail_m_ptr, tail_den_ptr,  # (KVH, QPK) fp32
        tail_num_ptr,            # (KVH, QPK, HD) fp32
        KVH: tl.constexpr, QPK: tl.constexpr,
        R: tl.constexpr, RPAD: tl.constexpr, HD: tl.constexpr,
    ):
        """One program = one q-head: tail rowwise max + masked exp +
        weighted value sum + weight sum. Replaces ~5 torch dispatches
        (amax, sub/exp/mul, bmm, sum) with one launch, and the (kvh,qpk,R)
        pe temporary with on-chip weights. The assignment mask reads the
        SAME exrev array the serve kernel reads (in-kernel this time, so
        the torch tg/ok computation vanishes too): kernel covers
        exrev>=0, torch covers exrev<0 -- the partition is airtight by
        construction. Same per-element math as the torch block (exp
        exact; masked lanes contribute exactly +0.0), so results agree
        to fp32 assoc noise (~1e-7 relative, far inside the 5e-4 arm
        gate). R pads to RPAD (pow2) with -inf scores. num_warps=4."""
        pid = tl.program_id(0)
        ph = pid // QPK
        pq = pid % QPK
        roff = tl.arange(0, RPAD)
        rmask = roff < R
        s = tl.load(st_ptr + (ph * QPK + pq) * R + roff, mask=rmask,
                    other=float("-inf"))
        m = tl.max(s)
        t = tl.load(tg_ptr + roff, mask=rmask, other=0)
        ev = tl.load(exrev_ptr + t, mask=rmask, other=-1)
        e = tl.exp(s - m) * tl.where(ev < 0, 1.0, 0.0)
        den = tl.sum(e)
        tl.store(tail_m_ptr + pid, m)
        tl.store(tail_den_ptr + pid, den)
        lane = tl.arange(0, HD)
        num = tl.zeros([HD], dtype=tl.float32)
        # Tight bound R (not RPAD): padded lanes carry exactly +0.0
        # (e is 0 there), so skipping them is bit-identical.
        for r in tl.range(R):
            er = tl.sum(tl.where(roff == r, e, 0.0))
            v = tl.load(vt_ptr + (r * KVH + ph) * HD + lane)
            num += v * er
        tl.store(tail_num_ptr + (pid * HD) + lane, num)


def kvarn_triton_online_tail_reduce(st, vt, tg, exrev, _bufs=None):
    """Fused tail softmax + value reduction for the imageless arm.

    st: (kvh, qpk, R) fp32 scaled scores (torch bmm, already includes the
    sm scale); vt: (R, kvh, hd) fp32 tail values; tg: (R,) int64 tail
    group ids; exrev: (G,) int64 group -> exact slot (same array the
    serve kernel reads: torch owns exrev<0 rows). Returns (tail_m,
    tail_den, tail_num): (kvh, qpk), (kvh, qpk), (kvh, qpk, hd) fp32.
    Loud failure when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton online tail requires triton (import failed).")
    dev = st.device
    if dev.type != "cuda":
        raise RuntimeError(
            "KVarN Triton online tail requires CUDA tensors, got "
            f"{st.device}.")
    kvh, qpk, R = st.shape
    hd = vt.shape[2]
    # Optional persistent outs (dispatch passes layer buffers: same
    # values, minus 3 allocs/layer/step; shapes are static per layer).
    if _bufs is None:
        tail_m = torch.empty((kvh, qpk), dtype=torch.float32, device=dev)
        tail_den = torch.empty((kvh, qpk), dtype=torch.float32, device=dev)
        tail_num = torch.empty((kvh, qpk, hd), dtype=torch.float32, device=dev)
    else:
        tail_m, tail_den, tail_num = _bufs
    rpad = 1 << (R - 1).bit_length()
    _kvarn_online_tail_kernel[(kvh * qpk,)](
        st, vt, tg, exrev, tail_m, tail_den, tail_num,
        kvh, qpk, R, rpad, hd, num_warps=4)
    return tail_m, tail_den, tail_num


if _have_triton:
    @triton.jit
    def _kvarn_online_tail_gather_kernel(
        tpos_ptr,                # (R,) int64 tail positions
        bt_ptr,                  # (P,) int32 block table row
        exact_k_ptr, exact_v_ptr,  # (E, 128, KVH, HD) tail-dtype exact rows
        exrev_ptr,               # (G,) int64 group -> exact slot
        valid_ptr,               # (G,) bool exact residency
        k_out_ptr, v_out_ptr,    # (MAXW, KVH, HD) fp32 persistent temps
        ev_out_ptr,              # (R,) bool per-row exact hit
        g_out_ptr,               # (R,) int64 per-row group
        s_out_ptr,               # (R,) int64 per-row slot-in-group
        GPS: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr,
    ):
        """One program = one (row, kv-head): tail positions to exact rows.
        Replaces ~10 torch dispatches (pos/pages/offs/g/s index math, two
        indexed gathers, two fp16->fp32 converts) with one launch, writing
        straight into the layer's persistent temps. Position/group math is
        identical to kvarn_online_tail (page * gps + offs // 128,
        offs % 128); unassigned slots read zeros via the validity mask
        (valid ⟺ assigned: the mask matches the old clamp_min(0) +
        exact_valid-gated read). ev/g/s ride out for the cert check, the
        staging fallback, and the caller's assignment mask. num_warps=4."""
        r = tl.program_id(0)
        h = tl.program_id(1)
        lane = tl.arange(0, HD)
        pos = tl.load(tpos_ptr + r).to(tl.int64)
        page = tl.load(bt_ptr + pos // 256).to(tl.int64)
        offs = pos % 256
        g = page * GPS + offs // 128
        s = offs % 128
        ev = tl.load(valid_ptr + g)
        es = tl.load(exrev_ptr + g)
        es_c = tl.where(es >= 0, es, 0)
        e_off = ((es_c * 128 + s) * KVH + h) * HD + lane
        k = tl.load(exact_k_ptr + e_off, mask=ev, other=0.0).to(tl.float32)
        v = tl.load(exact_v_ptr + e_off, mask=ev, other=0.0).to(tl.float32)
        o_off = (r * KVH + h) * HD + lane
        tl.store(k_out_ptr + o_off, k)
        tl.store(v_out_ptr + o_off, v)
        if h == 0:
            tl.store(ev_out_ptr + r, ev)
            tl.store(g_out_ptr + r, g)
            tl.store(s_out_ptr + r, s)


def kvarn_triton_online_tail_gather(layer, tpos, bt_row, K, V, gps, _bufs=None):
    """Fused tail exact-gather for the imageless arm.

    tpos: (R,) int64 positions (built once by the caller, shared with the
    assignment mask); bt_row: block-table row; K/V: (MAXW, kvh, hd) fp32
    persistent temps (rows [0, R) written). Returns (K[:R], V[:R], ev, g,
    s): gathered rows plus per-row exact-hit, group, and slot-in-group
    for the cert check, staging fallback, and mask. Loud failure when
    unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton online tail gather requires triton.")
    dev = tpos.device
    if dev.type != "cuda":
        raise RuntimeError(
            "KVarN Triton online tail gather requires CUDA tensors, got "
            f"{tpos.device}.")
    kvh = int(layer.num_kv_heads)
    hd = int(layer.head_dim)
    R = int(tpos.numel())
    # Optional persistent outs (caller passes R-sized buffers or
    # [:R] views of MAXW buffers: same values, minus 3 allocs).
    if _bufs is None:
        ev = torch.empty((R,), dtype=torch.bool, device=dev)
        g = torch.empty((R,), dtype=torch.int64, device=dev)
        s = torch.empty((R,), dtype=torch.int64, device=dev)
    else:
        ev, g, s = _bufs
    _kvarn_online_tail_gather_kernel[(R, kvh)](
        tpos, bt_row, layer.exact_k, layer.exact_v, layer._exact_rev,
        layer.exact_valid, K, V, ev, g, s, gps, kvh, hd, num_warps=4)
    return K[:R], V[:R], ev, g, s


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


def kvarn_triton_online_partials(layer, qw, ids, qpk, scale, n_new,
                                  sink_n, tail_eff):
    """Stage-1 only: block partials for merging (imageless serve).

    Same launch as kvarn_triton_online_decode minus the combine: returns
    (m, l, acc) in persistent per-layer buffers (QPAD layout; the caller
    slices to qpk). The caller reduces/merges (torch for v1, fused later)
    and WHTs once at the end. Loud failure when unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton online partials requires triton (import failed).")
    dev = qw.device
    if dev.type != "cuda":
        raise RuntimeError(
            "KVarN Triton online partials requires CUDA tensors, got "
            f"{qw.device}.")
    kvh, hd = int(layer.num_kv_heads), int(layer.head_dim)
    sl = int(layer.slices)
    assert hd == sl * 128
    qh = kvh * qpk
    assert qw.shape == (qh, hd)
    qpad = 1 << (qpk - 1).bit_length()
    m, l, acc, _, _, _ = _kvarn_online_buffers(layer, qh, qpad, hd, dev)
    L = layer.layout
    rec = layer.records
    rec_f16 = rec.view(torch.float16)
    _kvarn_online_block_kernel[(kvh, int(ids.numel()),)](
        qw, rec, rec_f16, ids, m, l, acc,
        L.k_payload_off,
        L.k_s_col_off // 2, L.k_zp_off // 2, L.k_s_row_off // 2,
        int(layer.k_bits),
        L.v_payload_off,
        L.v_s_row_off // 2, L.v_zp_off // 2, L.v_s_col_off // 2,
        int(layer.v_bits),
        rec.shape[1], rec.shape[2], sl,
        kvh, qpk, qpad, hd, int(ids.numel()), scale,
        n_new, int(sink_n), int(tail_eff),
        num_warps=4)
    return m, l, acc


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
    _kvarn_launch_combine(
        m, l, acc, out, kvh, qpk, qpad, nb, nbpad, hd, sl,
        1.0 if sl == 1 else (0.7071067811865475 if sl == 2 else 0.5))
    return out


# --------------------------------------------------------------------------
# Promoted single-kernel serve (spike7 -> production, inert until wired).
# Body-verbatim copy of eval/_spike7_coal._serve_s7_kernel (modulo indent
# + name; verified IDENTICAL by construction script): grid (KVH, chunks),
# per-row body(record)/tail(exact) select, sticky flag, online softmax
# partials feeding _kvarn_online_combine_kernel. Keep in lockstep with
# the eval original until dispatch owns this path; then delete the eval
# copy. Promotion delta vs v1 partials: no ids array, no tail temps.
# --------------------------------------------------------------------------
if _have_triton:
    @triton.jit
    def _kvarn_online_serve_kernel(
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
        CPG: tl.constexpr, GROUPS: tl.constexpr,
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
    
        tok = tl.arange(0, 16)  # TOK=16 rows/iter
        # Hierarchical super-chunk: each program covers CPG chunks
        # (CPG*128 rows) with one online state instead of one chunk per
        # program. CPG=1 is exactly the old path (8 iters cover 128).
        # Past-n tiles are no-ops via the r-mask (same mechanism as the
        # old last-chunk partial tiles); the reduction order changes
        # (tree of depth 2 vs flat: fp32 assoc noise only, allclose-gated).
        for t0 in tl.range(8 * CPG):
            t = t0 * 16
            p = (pid_c * CPG) * 128 + t + tok  # (TOK,)
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
            # K value index: slot-major v6 store for all widths (v = s*128+dd;
            # rows read coalesced). nibble path below for 4-bit, bit loop else.
            vv = s[:, None] * 128 + dd_c[None, :]  # (16, HD) transposed
            if K_BITS == 4:
                # Nibble fast path: value v occupies stream bits [4v,4v+4),
                # i.e. low/high nibble of byte v//2. One byte load per
                # element instead of 4 bit-loop loads. Matches the generic
                # LSB-first loop bit-for-bit (verified: even v -> low
                # nibble, odd v -> high nibble).
                nb = vv // 2
                nptr = (kpay_row[:, None] + kc[None, :] * B + nb)
                nbyte = tl.load(nptr).to(tl.int32)
                # uint8: values are 0-15 (bit-identical through the later
                # .to(fp32)), 4KB live instead of 16KB -> fewer regs/thread.
                qq = (((nbyte >> ((vv % 2) * 4)) & 0xF).to(tl.uint8))
            else:
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
                # Unassigned tail rows (staging-fallback rows the torch tail
                # path serves): -inf, else their zero exact values would still
                # draw softmax weight and downscale real rows.
                st = tl.where(ok_tail[:, None], st, float("-inf"))
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
                # uint8 (see K side): 4KB live instead of 16KB.
                qqv = (((nbytev >> ((vv2 % 2) * 4)) & 0xF).to(tl.uint8))
            elif V_BITS == 2:
                # Quad fast path (K4V2 vehicle): stream bits [2v,2v+2),
                # 4 values/byte, LSB-first exactly like the generic
                # bit loop below (value v lives at bits [2j,2j+2) of
                # byte j=vv2//4). One byte load per element instead of
                # 2 predicated bit-loop loads; bit-identical.
                qbv = vv2 // 4
                qptrv = (vpay_row[:, None]
                         + kc[None, :] * B + qbv)
                qbytev = tl.load(qptrv).to(tl.int32)
                qqv = (((qbytev >> ((vv2 % 4) * 2)) & 0x3).to(tl.uint8))
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
        # Partials strided by GROUPS (subgroup count), not GMAX: with
        # CPG=1 (GROUPS==GMAX) this is bit-identical to the old layout.
        tl.store(m_ptr + (pid_h * QPAD * GROUPS) + qoff2 * GROUPS + pid_c,
                 m, mask=qmask2)
        tl.store(l_ptr + (pid_h * QPAD * GROUPS) + qoff2 * GROUPS + pid_c,
                 l, mask=qmask2)
        tl.store(out_ptr + ((pid_h * QPAD * GROUPS) + qoff2[:, None] * GROUPS
                            + pid_c) * HD + lane[None, :], acc,
                 mask=qmask2[:, None])


    # ------------------------------------------------------------------
    # Serve v2 (task-7 attempt 1): per-GROUP metadata hoist. Env-gated
    # EXL3_KVARN_SERVE_V2=1, default OFF, fail-closed on the launcher.
    # Body-verbatim copy of _kvarn_online_serve_kernel with ONE change:
    # the single tile loop becomes a (group, tile) nest, so the
    # block-table gather and the three per-channel metadata vectors
    # (K sc, K zp, V oth) are loaded once per 128-row group instead of
    # once per 16-row tile -- 8x fewer, and provably the same values
    # (a TOK=16 tile can never straddle a 128-row group).
    # Bit-exact vs legacy: same addresses, same per-element op order, no
    # reassociation. Rows masked out by r = p < n read a different
    # in-bounds record address than the legacy pa=0 clamp gave them;
    # every use of those rows is masked or discarded by the body/tail
    # select, so the outputs are identical.
    # ------------------------------------------------------------------
    @triton.jit
    def _kvarn_online_serve_kernel_v2(
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
        CPG: tl.constexpr, GROUPS: tl.constexpr,
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
    
        tok = tl.arange(0, 16)  # TOK=16 rows/iter
        # Hierarchical super-chunk: each program covers CPG chunks
        # (CPG*128 rows) with one online state instead of one chunk per
        # program. CPG=1 is exactly the old path (8 iters cover 128).
        # Past-n tiles are no-ops via the r-mask (same mechanism as the
        # old last-chunk partial tiles); the reduction order changes
        # (tree of depth 2 vs flat: fp32 assoc noise only, allclose-gated).
        kc = pid_h * SL + sl_c  # (HD,) col channel
        for cg in tl.range(CPG):
            # One group == 128 token slots == 8 TOK=16 tiles.
            pa0 = (pid_c * CPG + cg) * 128
            # Groups past n (the grid covers 8*CPG tiles regardless of n,
            # exactly like the legacy loop) must not index past the block
            # table or the records. Clamp the page index to the highest
            # page the serve may touch and the group index to GMAX-1;
            # both clamps are no-ops for a live group, and every value a
            # dead group derives is discarded by the r / body masks.
            bt_max = tl.maximum((n - 1) // 256, 0)
            page0 = tl.load(bt_ptr + tl.minimum(pa0 // 256, bt_max))
            g0 = tl.minimum(page0 * GPS + (pa0 % 256) // 128, GMAX - 1)
            kf16_g0 = rec_f16_ptr + (g0 * C * B) // 2
            k_sc0 = tl.load(kf16_g0 + (kc * B) // 2 + K_SC2
                            + dd_c).to(tl.float32)  # (256,)
            k_zp0 = tl.load(kf16_g0 + (kc * B) // 2 + K_ZP2
                            + dd_c).to(tl.float32)
            v_ot0 = tl.load(kf16_g0 + (kc * B) // 2 + V_OT2
                            + dd_c).to(tl.float32)
            # Fail-closed guard, evaluated once per group: the page of the
            # group's last row must equal the page of its first. For a
            # group inside n this is an identity (groups are 128-aligned,
            # so pa0//256 == (pa0+127)//256), i.e. a 2-scalar-load
            # invariant check rather than a heuristic; a partially live
            # last group clamps and takes the per-row path below, which
            # serves those tiles exactly as the legacy kernel does.
            uniform = tl.load(bt_ptr + tl.minimum((pa0 + 127) // 256,
                                                  bt_max)) == page0
            for t in tl.range(8):
                p = pa0 + t * 16 + tok  # (TOK,)
                r = p < n
                body = (p >= SINK_N) & (p < tail_start) & r
                if uniform:
                    # Group index and in-group slot are arithmetic here
                    # (no block-table gather per tile).
                    g = tl.zeros([16], dtype=tl.int32) + g0
                    s = tl.zeros([16], dtype=tl.int32) + (t * 16 + tok)
                else:
                    pa = tl.where(r, p, 0)
                    pg = tl.load(bt_ptr + pa // 256)
                    offs = pa % 256
                    g = pg * GPS + offs // 128  # (TOK,)
                    s = offs % 128  # (TOK,)

                # --- K tile 2D (TOK, HD), row-dependent gbase ---
                kpay_row = rec_ptr + g * C * B + K_PAY_OFF  # (TOK,)
                kf16_row = rec_f16_ptr + (g * C * B) // 2  # (TOK,)
                # K value index: slot-major v6 store for all widths (v = s*128+dd;
                # rows read coalesced). nibble path below for 4-bit, bit loop else.
                vv = s[:, None] * 128 + dd_c[None, :]  # (16, HD) transposed
                if K_BITS == 4:
                    # Nibble fast path: value v occupies stream bits [4v,4v+4),
                    # i.e. low/high nibble of byte v//2. One byte load per
                    # element instead of 4 bit-loop loads. Matches the generic
                    # LSB-first loop bit-for-bit (verified: even v -> low
                    # nibble, odd v -> high nibble).
                    nb = vv // 2
                    nptr = (kpay_row[:, None] + kc[None, :] * B + nb)
                    nbyte = tl.load(nptr).to(tl.int32)
                    # uint8: values are 0-15 (bit-identical through the later
                    # .to(fp32)), 4KB live instead of 16KB -> fewer regs/thread.
                    qq = (((nbyte >> ((vv % 2) * 4)) & 0xF).to(tl.uint8))
                else:
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
                    # Unassigned tail rows (staging-fallback rows the torch tail
                    # path serves): -inf, else their zero exact values would still
                    # draw softmax weight and downscale real rows.
                    st = tl.where(ok_tail[:, None], st, float("-inf"))
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
                    # uint8 (see K side): 4KB live instead of 16KB.
                    qqv = (((nbytev >> ((vv2 % 2) * 4)) & 0xF).to(tl.uint8))
                elif V_BITS == 2:
                    # Quad fast path (K4V2 vehicle): stream bits [2v,2v+2),
                    # 4 values/byte, LSB-first exactly like the generic
                    # bit loop below (value v lives at bits [2j,2j+2) of
                    # byte j=vv2//4). One byte load per element instead of
                    # 2 predicated bit-loop loads; bit-identical.
                    qbv = vv2 // 4
                    qptrv = (vpay_row[:, None]
                             + kc[None, :] * B + qbv)
                    qbytev = tl.load(qptrv).to(tl.int32)
                    qqv = (((qbytev >> ((vv2 % 4) * 2)) & 0x3).to(tl.uint8))
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
            # Partials strided by GROUPS (subgroup count), not GMAX: with
            # CPG=1 (GROUPS==GMAX) this is bit-identical to the old layout.
            tl.store(m_ptr + (pid_h * QPAD * GROUPS) + qoff2 * GROUPS + pid_c,
                     m, mask=qmask2)
            tl.store(l_ptr + (pid_h * QPAD * GROUPS) + qoff2 * GROUPS + pid_c,
                     l, mask=qmask2)
            tl.store(out_ptr + ((pid_h * QPAD * GROUPS) + qoff2[:, None] * GROUPS
                                + pid_c) * HD + lane[None, :], acc,
                     mask=qmask2[:, None])


def _kvarn_serve_groups(gc):
    """Hierarchical subgroup count shared by serve and its callers.

    Direct (one chunk per program) while gc <= 64; capped at
    EXL3_KVARN_SERVE_GROUPS beyond (default 64: 64k -> 64 groups x
    CPG=8; 16k -> 64 x CPG=2). Single source so kernel strides and
    merge/combine counts cannot drift apart.

    Why the cap came down from 128 (task-7 attempt 2, measured 64k
    isolated serve, 7 interleaved windows): the cap trades CTA count
    against partials traffic, and at 2 CTA/SM the trade is dominated by
    wave quantization, not by the partials bytes. 128 groups x CPG=4
    = 512 CTAs = 4 exact waves at 16 warps of work per SM; 64 groups x
    CPG=8 = 256 CTAs = 2 waves, HALF the partials traffic (2MB/layer
    vs 4MB), and 5.8% faster per layer (163.7 -> 154.2 us/layer) with
    registers, spills and smem unchanged. CPG=2 (cap 256) and CPG=1
    (cap 512) both measured slower (+8.2, +8.6 us), so the old cap was
    past the optimum in the other direction.

    Kill-switch: EXL3_KVARN_SERVE_GROUPS=128 restores the old cap
    exactly. A malformed value falls back to the default with a loud
    print (never raises into the decode path).
    """
    cap = 64
    _v = os.environ.get("EXL3_KVARN_SERVE_GROUPS")
    if _v is not None:
        try:
            cap = max(1, int(_v))
        except ValueError:
            print(f"KVARN-SERVE-GROUPS invalid ({_v!r}), using {cap}",
                  flush=True)
    return gc if gc <= 64 else min(gc, cap)

def kvarn_triton_online_serve(layer, qw, Qf, exact_k, exact_v_w, exrev,
                              sealed, bt, n_0d, qpk, scale, sink_n,
                              tail_eff, gps, gc=None, rec_f16=None,
                              sync_flag=True):
    """Promoted spike7 single-kernel body serve (imageless path).

    INERT until dispatch wires it (no callers yet): same math as eval
    serve_online_s7 (grid-trimmed gc, MMA dots, coalesced slot-major K
    reads, sticky flag, online partials into _kvarn_online_combine).
    Buffers persist on the layer (_ov_serve_*, mirroring _ov_online_*),
    keyed by (qh, gc, hd); realloc on shape change. exact_v_w is exact
    blocks pre-WHT'd caller-side (production eref owns the incremental
    refresh; spikes pass a full refresh). rec_f16 hoisting + sync_flag
    mirror the eval entry (CUDA-graph safe). Loud failure when
    unrunnable.
    """
    if not _have_triton:
        raise RuntimeError(
            "KVarN Triton online serve requires triton (import failed).")
    dev = qw.device
    if dev.type != "cuda":
        raise RuntimeError(
            "KVarN Triton online serve requires CUDA tensors, got "
            f"{qw.device}.")
    kvh = int(layer.num_kv_heads)
    hd = int(layer.head_dim)
    sl = int(layer.slices)
    records, layout = layer.records, layer.layout
    k_bits, v_bits = int(layer.k_bits), int(layer.v_bits)
    qh = kvh * qpk
    gmax = int(records.shape[0])
    if gc is None:
        gc = gmax
    assert gc <= gmax
    qpad = 1 << (qpk - 1).bit_length()
    # Hierarchical subgroups: direct (one chunk per program) while
    # gc <= 64 (today's exact path, incl. all of 8k), else cap at 128
    # subgroups (16k: 128 groups = direct-equivalent; 64k: 128 groups
    # x CPG=4; 128k: x CPG=8). The first attempt (flat 32) proved serve
    # is parallelism-bound, not traffic-bound: 128 CTAs underfilled the
    # GPU and lost 4% despite 16x less traffic. 512 CTAs saturate the
    # 144 SMs, so 128 groups keep full occupancy AND cut partials
    # traffic 4x at 64k. Combine/merge read GROUPS partials with
    # matching strides (stride == count invariant holds, both UNCHANGED).
    groups = _kvarn_serve_groups(gc)
    cpg = (gc + groups - 1) // groups
    need = (qh, qpad, groups, hd)
    if getattr(layer, "_ov_serve_shape", None) != need or \
            getattr(layer, "_ov_serve_m", None) is None:
        layer._ov_serve_m = torch.empty((kvh, qpad, groups),
                                        dtype=torch.float32, device=dev)
        layer._ov_serve_l = torch.empty((kvh, qpad, groups),
                                        dtype=torch.float32, device=dev)
        layer._ov_serve_acc = torch.empty((kvh, qpad, groups, hd),
                                          dtype=torch.float32, device=dev)
        layer._ov_serve_out = torch.empty((qh, hd), dtype=torch.float32,
                                          device=dev)
        layer._ov_serve_flag = torch.zeros((1,), dtype=torch.uint8,
                                           device=dev)
        layer._ov_serve_shape = need
    m = layer._ov_serve_m
    l = layer._ov_serve_l
    acc = layer._ov_serve_acc
    out = layer._ov_serve_out
    flag = layer._ov_serve_flag
    flag.zero_()
    if rec_f16 is None:
        rec_f16 = records.view(torch.float16)
    import os as _os
    if _os.environ.get("EXL3_KVARN_DEBUG_HASH") == "1":
        # Temporary input-fingerprinting (arm-vs-direct divergence hunt):
        # sums are order-sensitive enough to catch any differing input.
        print(f"SERVE-HASH gc={gc} n={[int(n_0d[0])]} "
              f"qw={float(qw.double().sum()):.6e} "
              f"qf={float(Qf.double().sum()):.6e} "
              f"rec={float(records.double().sum()):.6e} "
              f"ek={float(exact_k.double().sum()):.6e} "
              f"evw={float(exact_v_w.double().sum()):.6e} "
              f"exrev={int(exrev.sum())} sealed={int(sealed.sum())} "
              f"bt={int(bt.sum())} scale={scale} sink={sink_n} "
              f"tail={tail_eff} gps={gps} qpk={qpk}", flush=True)
    _serve_args = (
        qw, Qf, records, rec_f16, exact_k, exact_v_w, exrev, sealed, bt,
        n_0d, flag, m, l, acc,
        layout.k_payload_off,
        layout.k_s_col_off // 2, layout.k_zp_off // 2,
        layout.k_s_row_off // 2, k_bits,
        layout.v_payload_off,
        layout.v_s_row_off // 2, layout.v_zp_off // 2,
        layout.v_s_col_off // 2, v_bits,
        records.shape[1], records.shape[2], sl, gps,
        kvh, qpk, qpad, hd, gc, scale, sink_n, tail_eff, cpg, groups)
    # Serve v2 (per-group metadata hoist, task-7 attempt 1): DEFAULT ON
    # (box-green 2026-10-03: tg@64k graph 47.9 -> 51.0 tok/s over 3
    # interleaved rounds, twin bit-exact, KLD identical). Kill-switch
    # EXL3_KVARN_SERVE_V2=0 restores the legacy kernel. Same arg tuple,
    # same grid, same partials layout -- only the kernel body differs,
    # and it is bit-exact. Fail-closed loud fallback on throw (triton
    # compiles before it launches, so a throw means nothing ran): copy
    # the _kvarn_launch_combine pattern.
    if _os.environ.get("EXL3_KVARN_SERVE_V2", "1") == "1":
        try:
            _kvarn_online_serve_kernel_v2[(kvh, groups,)](
                *_serve_args, num_warps=4, num_stages=1)
        except Exception as _e:
            print("KVARN-SERVE-V2 launch failed, legacy fallback: "
                  f"{_e}", flush=True)
            _kvarn_online_serve_kernel[(kvh, groups,)](
                *_serve_args, num_warps=4, num_stages=1)
    else:
        _kvarn_online_serve_kernel[(kvh, groups,)](
            *_serve_args, num_warps=4, num_stages=1)
    sscale = 1.0 if sl == 1 else (0.7071067811865475 if sl == 2 else 0.5)
    nbpad = 1 << (groups - 1).bit_length()
    _kvarn_launch_combine(
        m, l, acc, out, kvh, qpk, qpad, groups, nbpad, hd, sl, sscale)
    import os as _os2
    if _os2.environ.get("EXL3_KVARN_DEBUG_HASH") == "1":
        print(f"SERVE-OUT m={float(m.double().sum()):.6e} "
              f"l={float(l.double().sum()):.6e} "
              f"acc={float(acc.double().sum()):.6e} "
              f"out={float(out.double().sum()):.6e} "
              f"flag={int(flag[0])}", flush=True)
    if sync_flag:
        return out, int(flag[0])
    # Unchecked (production periodic-check path): int 0, no DtoH sync.
    # The kernel still zeroes/sets the sticky device flag; the caller
    # reads it every Kth call (a real trip fires every step, so
    # periodic catches it; PARITY=1 tests check every call).
    return out, 0
