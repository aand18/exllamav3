"""
KVarN online-dequant Triton kernels (bootstrap, NOT YET RUN ON GPU).

Status: written against the CPU-tested math in ``exllamav3/cache/kvarn.py``
and import-checked only. No GPU was available, so nothing here has executed:
no numerics, no perf numbers, no launch validation. The first 4090 run must
use ``EXL3_KVARN_TRITON_PARITY=1`` (see below) before trusting this path.

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
    def _kvarn_row_kernel(
        pay_ptr, sc_ptr, zp_ptr, oth_ptr, out_ptr,
        PAY: tl.constexpr,   # payload bytes per tile (drop-in bound, unused)
        BITS: tl.constexpr,  # 2, 3, 4, 5, 6 or 8
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
        tl.store(out_ptr + (tile * 128 + row) * 128 + cols, tile_out)


def kvarn_triton_dequant_side(payload: torch.Tensor, sc: torch.Tensor,
                              zp: torch.Tensor, oth: torch.Tensor,
                              bits: int) -> torch.Tensor:
    """
    Dequantize NT tiles of one side (K or V).

    payload: (NT, PAY) uint8 CUDA. sc/zp: (NT, 128) fp16 CUDA (per row).
    oth: (NT, 128) fp16 CUDA (per col). Returns (NT, 128, 128) fp32 CUDA.
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
    _kvarn_row_kernel[(NT * 128,)](payload, sc, zp, oth, out,
                                   PAY, bits)
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


def kvarn_triton_dequant_groups(records_G, layout,
                               k_bits: int, v_bits: int,
                               num_kv_heads: int, slices: int):
    """
    Batched multi-group dequant: records_G (Gg, ncols, tile_bytes) uint8
    CUDA, ncols = kv_heads * slices. One kernel launch per side (K, V).

    Returns (bk, bv) float32 CUDA shaped (Gg, 128, kvh, hd) rotated-domain,
    matching ``CacheLayer_kvarn._dequant_groups_batched`` torch order
    (K transposed to [token, dim]). Bit-exact with the torch path: same
    fp32 elementwise math, fp32 store.
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

    def side(payload_off, payload_bytes, sc_off, zp_off, oth_off, bits):
        NT = Gg * ncols
        pay = records_G[:, :, payload_off: payload_off + payload_bytes] \
            .reshape(NT, payload_bytes).contiguous()
        sc = rec_f16[:, :, sc_off // 2: sc_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        zp = rec_f16[:, :, zp_off // 2: zp_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        oth = rec_f16[:, :, oth_off // 2: oth_off // 2 + 128] \
            .reshape(NT, 128).contiguous()
        return kvarn_triton_dequant_side(pay, sc, zp, oth, bits)

    kt = side(layout.k_payload_off, layout.k_payload_bytes,
              layout.k_s_col_off, layout.k_zp_off, layout.k_s_row_off,
              k_bits).reshape(Gg, num_kv_heads, slices, 128, 128)
    bk = kt.permute(0, 4, 1, 2, 3).reshape(Gg, 128, num_kv_heads, slices * 128)
    vt = side(layout.v_payload_off, layout.v_payload_bytes,
              layout.v_s_row_off, layout.v_zp_off, layout.v_s_col_off,
              v_bits).reshape(Gg, num_kv_heads, slices, 128, 128)
    bv = vt.permute(0, 3, 1, 2, 4).reshape(Gg, 128, num_kv_heads, slices * 128)
    return bk, bv
