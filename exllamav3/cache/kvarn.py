"""
KVarN compressed KV cache, M3 (CPU-testable, upstream-mergeable).

Reference: BeeLlama KVarN (Huawei arXiv:2606.03458), ported from
beellama ``src/llama-kvarn.h`` / ``src/llama-kvarn.cpp``,
``src/llama-kv-cache-kvarn.h`` (tail policy) /
``src/llama-kv-cache-kvarn.cpp`` (sink + stage depth) and the CPU
reference kernels in ``ggml/src/ggml-cpu/ops.cpp``.

M3 scope (presets ``kvarn4`` (4,4 symmetric), ``kvarn5`` (5,5 symmetric)
and ``kvarn5,kvarn4`` (5-bit K / 4-bit V asymmetric, Bee balanced
default); K and V widths are independent per side, sealed together per
128-group with different payload widths):
- Permanent 128-token sink (physical group 0) kept exact on non-SWA
  layers only; SWA layers get no sink (ring). Mirrors Bee's
  ``sink_tokens = 128`` (``llama-kvarn.cpp`` validation) and the
  non-SWA sink slot / SWA no-sink stage accounting
  (``llama-kv-cache-kvarn.h:304-315``). Logical sink exactness (first
  128 committed tokens per sequence) is guaranteed by the get_kv
  overlay regardless of paging; the seal skip targets physical group 0,
  which coincides with the logical sink when sequences start at page 0
  (true for single-sequence CPU validation and the paged generator's
  per-sequence page assignment).
- Configurable exact tail: ``tail_tokens`` (raw ``--kv-tail-tokens``)
  and ``tail_type`` (``--kv-tail-type f16/bf16``, default f16 for KVarN)
  plumbed through ``model_init.py``. Policy is Bee's tail helper
  (``llama-kv-cache-kvarn.h:29-49``, applied in
  ``llama-context.cpp:526-541``): intrinsic floor 128 (request
  0/omitted => effective 128), positive requests ceil to 128-groups,
  cap at window (here ``max_num_tokens``); full-window request =>
  native exact (no compressed body for that span, nothing is sealed).
- Attention M2: ``get_kv`` serves one merged fp16 image per position --
  dequantized body with sink + tail rows overwritten exact -- so the
  downstream single-softmax SDPA counts each key once across
  sink/body/tail. This mirrors Bee's portable mask semantics
  (``fattn-kvarn-portable.cuh``: body rows covered by the exact tail are
  masked out, tail rows attended once). Eager seal stays: completed
  groups are sealed even while inside the tail window; the overlay
  serves the exact copy alongside the sealed record.
- QSA: sink/tail apply to dense KV; raw_k/pooled indexer planes stay
  fp16 exact (unchanged).
- State/copy: ``copy_page`` carries staging + exact buffers; sealed
  flags travel for full groups except physical group 0 on sink layers
  (never sealed). Copies are expected group-aligned (128) for sealed
  groups; partial copies travel unsealed. Prompt-cache state compat is
  still M2-provisional (no version bump).
- M2 CPU keeps full-page exact mirrors (staging + tail buffers) as
  overhead; production CUDA uses compact tail arenas (remaining work).

M3 still explicitly out of scope (later): SWA pair overrides (SWA tail
uses the full ``max_num_tokens`` window here, not the SWA window),
Triton/CUDA online kernels (``get_kvarn_records`` now exposes the
records for them; attention stays dequant-then-SDPA on CPU), the full
36-combo Bee width table (M3 ships (4,4), (5,5) and (5,4) only;
anything else is fail-closed).

M3 non-comparability note: widths change the record payload, so
records sealed under one preset are NOT comparable / interchangeable
with another preset's records or layouts. M3 is still CPU mirrors
only (no CUDA kernels); end-to-end parity is validated against the
fp16 overlay on CPU.

Numerical convention (matches BeeLlama):
- Each token's head is transformed by the head-wide normalized WHT
  (per-128 FWHT plus cross-slice FWHT for 256/512, ``test-kvarn.cpp``
  ``apply_reference_kvarn_wht_head``) before staging. The WHT is a
  symmetric involution, so the inverse applied after dequantization is
  the same op. Attention itself runs in the ORIGINAL domain, so no
  query rotation / output correction is needed anywhere else.
- K tiles are stored transposed: tile[row=dim, col=token]; V tiles are
  tile[row=token, col=dim] (``kvarn_cpu_quantize_stage``).
- QSA indexer planes (raw_k / pooled) are written from ORIGINAL-domain
   keys and stay fp16, so block scores are bit-identical to fp16 cache.
"""

from __future__ import annotations
from typing_extensions import override
import math
import torch
from ..constants import PAGE_SIZE
from .cache import CacheLayer
from .qsa import QSAPlanes
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention
    from ..model import Config


# --------------------------------------------------------------------------
# Preset / geometry constants
# --------------------------------------------------------------------------

KVAR_N_GROUP = 128
KVAR_N_INV_SQRT_128 = 0.08838834764831845
KVAR_N_SUPPORTED_HEAD_DIMS = (128, 256, 512)
KVAR_N_SINKHORN_ITERS = 16
# M3: Bee balanced default is kvarn5/kvarn4 (K 5 bit, V 4 bit); symmetric
# kvarn4 and kvarn5 also ship. Any other width combo is fail-closed in
# M3 (the full 36-combo Bee desc table, src/llama-kvarn.cpp:15-62, is M4).
KVAR_N_M3_PRESETS = frozenset({(4, 4), (5, 5), (5, 4)})
KVAR_N_PRESETS = {
    "kvarn4": (4, 4),
    "kvarn4,kvarn4": (4, 4),
    "kvarn5": (5, 5),
    "kvarn5,kvarn5": (5, 5),
    "kvarn5,kvarn4": (5, 4),
}

# M2: permanent exact sink on non-SWA layers (llama-kvarn.cpp: sink_tokens
# = 128, validated "exactly 128 unquantized sink tokens").
KVAR_N_SINK_TOKENS = 128
# M2: intrinsic exact-tail floor (llama-kv-cache-kvarn.h tail policy:
# max(128, ...) even for a zero request).
KVAR_N_TAIL_FLOOR_TOKENS = 128
KVAR_N_TAIL_TYPES = {"f16": torch.float16, "bf16": torch.bfloat16}


def kvarn_valid_bits(bits: int) -> bool:
    """Bee bit widths (llama-kvarn.cpp llama_kvarn_valid_bits)."""
    return int(bits) in (2, 3, 4, 5, 6, 8)


def kvarn_parse_preset(spec) -> tuple:
    """
    Parse a ``-cq`` KVarN preset string (case-insensitive; ``/`` and ``,``
    separators are equivalent) into a ``(k_bits, v_bits)`` pair.

    M3 ships ``kvarn4`` (4,4), ``kvarn5`` (5,5) and the Bee balanced
    default ``kvarn5,kvarn4`` (5,4); a bare ``k,v`` numeric pair is also
    accepted when it lands on the M3 set. Anything else raises
    ValueError (fail-closed; the full 36-combo Bee table is M4).
    """
    key = str(spec).strip().lower().replace("/", ",").replace(" ", "")
    if key in KVAR_N_PRESETS:
        return KVAR_N_PRESETS[key]
    bits: list = []
    for part in key.split(","):
        name = part[5:] if part.startswith("kvarn") else part
        if name.isdigit():
            bits.append(int(name))
    if len(bits) == 2 and tuple(bits) in KVAR_N_M3_PRESETS:
        return (bits[0], bits[1])
    raise ValueError(
        f"Unsupported KVarN preset {spec!r}: M3 supports kvarn4 [=(4,4)], "
        f"kvarn5 [=(5,5)] and kvarn5,kvarn4 [=(5,4)]; "
        f"parsed bits {bits or 'unparseable'}.")


def kvarn_tail_policy_for(raw_requested_tokens: int, window: int) -> dict:
    """
    Bee tail helper (src/llama-kv-cache-kvarn.h:29-49).

    - window == 0 -> no tail (native_exact True, vacuous).
    - intrinsic floor 128 (request 0/omitted => effective 128).
    - positive requests ceil to 128-groups, cap at window.
    - effective == window => native_exact (no compressed body).
    Returns dict(requested, effective, exact_groups, native_exact).
    """
    group = KVAR_N_GROUP
    raw = int(raw_requested_tokens)
    assert raw >= 0
    window = int(window)
    assert window >= 0
    if window == 0:
        return {"requested": 0, "effective": 0, "exact_groups": 0,
                "native_exact": True}
    intrinsic = min(group, window)
    rounded = 0 if raw == 0 else ((raw + group - 1) // group) * group
    effective = max(intrinsic, min(rounded, window))
    return {
        "requested": effective,
        "effective": effective,
        "exact_groups": (effective + group - 1) // group,
        "native_exact": effective == window,
    }


def kvarn_parse_tail_type(tail_type) -> torch.dtype:
    """'f16'/'bf16' (default f16 for KVarN) or a torch dtype."""
    if isinstance(tail_type, torch.dtype):
        assert tail_type in (torch.float16, torch.bfloat16), \
            f"KVarN tail type must be f16 or bf16, got {tail_type}"
        return tail_type
    t = str(tail_type).lower()
    assert t in KVAR_N_TAIL_TYPES, \
        f"KVarN tail type must be f16 or bf16, got {tail_type}"
    return KVAR_N_TAIL_TYPES[t]


def kvarn_packed_bytes(n_values: int, bits: int) -> int:
    """packed_bytes=(n*bits+7)/8 LSB-first (llama-kvarn.cpp:627-666)."""
    assert n_values >= 0 and bits in (2, 3, 4, 5, 6, 8)
    return (n_values * bits + 7) // 8


def _align_up(x: int, a: int = 8) -> int:
    return (x + a - 1) // a * a


class KvarnTileLayout:
    """
    Mirror of llama_kvarn_make_layout (llama-kvarn.cpp:576-613):
    k_payload(head*128*Kb) + k_s_col(head*u16) + k_zp(head*u16) +
    k_s_row(128*u16) + v_payload + v_s_col + v_s_row + v_zp,
    tile_bytes=align_up(...,8).

    K and V sides are independent: each side's payload width comes from
    its own bit count while the u16 metadata scales stay fixed shape, so
    asymmetric presets (e.g. kvarn5/kvarn4) share group boundaries with
    different payload widths (``k_record_bytes`` / ``v_record_bytes``).

    M1 always builds 128-token x 128-dim slice tiles, so head == group
    == 128 here; head_dim 256/512 heads are stored as 2/4 slice tiles.
    """

    def __init__(self, head_dim: int = 128, group: int = 128,
                 key_bits: int = 4, value_bits: int = 4):
        assert head_dim == group == KVAR_N_GROUP
        assert kvarn_valid_bits(key_bits) and kvarn_valid_bits(value_bits)
        off = 0
        self.k_payload_off = off
        self.k_payload_bytes = kvarn_packed_bytes(head_dim * group, key_bits)
        off += self.k_payload_bytes
        self.k_s_col_off = off
        off += head_dim * 2
        self.k_zp_off = off
        off += head_dim * 2
        self.k_s_row_off = off
        off += group * 2
        self.v_payload_off = off
        self.v_payload_bytes = kvarn_packed_bytes(group * head_dim, value_bits)
        off += self.v_payload_bytes
        self.v_s_col_off = off
        off += head_dim * 2
        self.v_s_row_off = off
        off += group * 2
        self.v_zp_off = off
        off += group * 2
        self.tile_bytes = _align_up(off, 8)
        self.head_dim = head_dim
        self.group = group
        self.key_bits = key_bits
        self.value_bits = value_bits
        assert self.tile_bytes == _align_up(
            self.k_record_bytes + self.v_record_bytes, 8)

    @property
    def k_record_bytes(self) -> int:
        """K-side record bytes: K payload (K bits) + s_col + zp + s_row."""
        return (self.k_payload_bytes + self.head_dim * 2 +
                self.head_dim * 2 + self.group * 2)

    @property
    def v_record_bytes(self) -> int:
        """V-side record bytes: V payload (V bits) + s_col + s_row + zp."""
        return (self.v_payload_bytes + self.head_dim * 2 +
                self.group * 2 + self.group * 2)


def kvarn_make_layout(head_dim: int = 128, group: int = 128,
                      key_bits: int = 4, value_bits: int = 4) -> KvarnTileLayout:
    return KvarnTileLayout(head_dim, group, key_bits, value_bits)


def kvarn_head_slices(head_dim: int) -> int:
    """llama_kvarn_head_slices: 0 unless 128/256/512 (fail-closed)."""
    if head_dim not in KVAR_N_SUPPORTED_HEAD_DIMS:
        return 0
    return head_dim // KVAR_N_GROUP


# --------------------------------------------------------------------------
# Bit packing (LSB-first, matches llama_kvarn_pack_bits/unpack_bits_value)
# --------------------------------------------------------------------------

def kvarn_pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    values = values.to(torch.uint8).reshape(-1)
    n = values.numel()
    b = torch.arange(bits, device=values.device, dtype=torch.uint8)
    stream = ((values.unsqueeze(1).to(torch.int32) >> b.to(torch.int32)) & 1).reshape(-1)
    pad = (-stream.numel()) % 8
    if pad:
        stream = torch.cat([stream, torch.zeros(pad, dtype=torch.int32, device=values.device)])
    stream = stream.reshape(-1, 8)
    weights = (1 << torch.arange(8, device=values.device, dtype=torch.int32))
    return (stream * weights).sum(dim=1).to(torch.uint8)


def kvarn_unpack_bits(payload: torch.Tensor, n_values: int, bits: int) -> torch.Tensor:
    payload = payload.to(torch.uint8).reshape(-1)
    pos8 = torch.arange(8, device=payload.device, dtype=torch.int32)
    stream = ((payload.unsqueeze(1).to(torch.int32) >> pos8) & 1).reshape(-1)[: n_values * bits]
    stream = stream.reshape(n_values, bits)
    wb = (1 << torch.arange(bits, device=payload.device, dtype=torch.int32))
    return (stream * wb).sum(dim=1).to(torch.uint8)


# --------------------------------------------------------------------------
# Head-wide normalized WHT (symmetric involution; forward == inverse)
# --------------------------------------------------------------------------

def kvarn_hadamard_128(x: torch.Tensor) -> torch.Tensor:
    """FWHT + *1/sqrt(128) over the last dim (llama-kvarn.cpp:668-686)."""
    x = x.clone().contiguous()
    s = 1
    while s < KVAR_N_GROUP:
        v = x.reshape(*x.shape[:-1], -1, 2, s)
        a = v[..., 0, :].clone()
        b = v[..., 1, :].clone()
        v[..., 0, :] = a + b
        v[..., 1, :] = a - b
        s *= 2
    x.mul_(KVAR_N_INV_SQRT_128)
    return x


def kvarn_wht_head(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Head-wide WHT: per-128 FWHT then cross-slice FWHT with 1/sqrt(slices)
    (matches test-kvarn.cpp apply_reference_kvarn_wht_head and
    kvarn_cpu_hadamard_head in ggml-cpu/ops.cpp).
    """
    slices = kvarn_head_slices(head_dim)
    assert slices > 0, f"KVarN: unsupported head_dim {head_dim}"
    prefix = x.shape[:-1]
    v = kvarn_hadamard_128(x.reshape(-1, slices, KVAR_N_GROUP))
    if slices == 1:
        return v.reshape(*prefix, head_dim)
    scale = 0.7071067811865475 if slices == 2 else 0.5
    # FWHT over the slice axis
    s = 1
    while s < slices:
        vv = v.reshape(v.shape[0], -1, 2, s, KVAR_N_GROUP)
        a = vv[:, :, 0].clone()
        b = vv[:, :, 1].clone()
        vv[:, :, 0] = a + b
        vv[:, :, 1] = a - b
        s *= 2
    return (v * scale).reshape(*prefix, head_dim)


# --------------------------------------------------------------------------
# Variance normalization: log-domain Sinkhorn, 16 iters
# (llama-kvarn.cpp:688-777, kvarn_cpu_* in ggml-cpu/ops.cpp)
# --------------------------------------------------------------------------

def _sample_std_cols(cur: torch.Tensor) -> torch.Tensor:
    return torch.std(cur, dim=-2, correction=1)


def _sample_std_rows(cur: torch.Tensor) -> torch.Tensor:
    return torch.std(cur, dim=-1, correction=1)


def kvarn_imbalance(cur: torch.Tensor) -> torch.Tensor:
    col = _sample_std_cols(cur)
    row = _sample_std_rows(cur)
    return col.amax(dim=-1) / col.amin(dim=-1).clamp_min(1e-8) + \
           row.amax(dim=-1) / row.amin(dim=-1).clamp_min(1e-8)


def kvarn_variance_normalize(tile: torch.Tensor, sinkhorn_iters: int = KVAR_N_SINKHORN_ITERS):
    """
    tile: (..., 128, 128) float32. Returns (balanced, s_col_best, s_row_best)
    with balanced = tile / (s_col[c] * s_row[r]). Batched over leading dims.
    """
    assert tile.shape[-2:] == (KVAR_N_GROUP, KVAR_N_GROUP)
    assert sinkhorn_iters > 0
    tile = tile.float()
    log_c = torch.zeros_like(tile[..., 0, :])
    log_r = torch.zeros_like(tile[..., :, 0])
    s_col_best = torch.ones_like(log_c)
    s_row_best = torch.ones_like(log_r)
    imb_best = kvarn_imbalance(tile)

    def rebuild():
        return tile / (log_c.exp().unsqueeze(-2) * log_r.exp().unsqueeze(-1))

    cur = tile
    for _ in range(sinkhorn_iters):
        std_c = _sample_std_cols(cur).clamp(1e-3, 1e3)
        log_c = (log_c + std_c.log()).clamp(-0.3, 10.0)
        cur = rebuild()
        std_r = _sample_std_rows(cur).clamp(1e-3, 1e3)
        log_r = (log_r + std_r.log()).clamp(-0.3, 10.0)
        cur = rebuild()
        imb = kvarn_imbalance(cur)
        better = imb <= imb_best
        if bool(better.all()):
            imb_best = imb
            s_col_best = log_c.exp()
            s_row_best = log_r.exp()
        elif bool(better.any()):
            imb_best = torch.where(better, imb, imb_best)
            ec, er = log_c.exp(), log_r.exp()
            s_col_best = torch.where(better.unsqueeze(-1), ec, s_col_best)
            s_row_best = torch.where(better.unsqueeze(-1), er, s_row_best)

    balanced = tile / (s_col_best.unsqueeze(-2) * s_row_best.unsqueeze(-1))
    return balanced, s_col_best, s_row_best


# --------------------------------------------------------------------------
# Per-tile RTN quantization.
# quantize_k/v_tile: per-row lo/hi, store k_s_col=s_row*scale,
# k_zp=s_row*lo, k_s_row=s_col (same for V) (llama-kvarn.cpp:790-895).
# Dequant: tile=(q*scale+zp)*other (897-935).
# --------------------------------------------------------------------------

def kvarn_quantize_tile(tile: torch.Tensor, bits: int,
                        sinkhorn_iters: int = KVAR_N_SINKHORN_ITERS):
    balanced, s_col, s_row = kvarn_variance_normalize(tile, sinkhorn_iters)
    qmax = (1 << bits) - 1
    lo = balanced.amin(dim=-1)
    hi = balanced.amax(dim=-1)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    q = torch.round((balanced - lo.unsqueeze(-1)) / scale.unsqueeze(-1)) \
        .clamp(0, qmax).to(torch.uint8)
    return q, s_row * scale, s_row * lo, s_col


def kvarn_dequantize_tile(q: torch.Tensor, sc: torch.Tensor,
                          zp: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return (q.float() * sc.unsqueeze(-1) + zp.unsqueeze(-1)) * other.unsqueeze(-2)


def _rec_f16(record: torch.Tensor) -> torch.Tensor:
    """fp16 view of a uint8 record row (last dim must be even-sized)."""
    return record.view(torch.float16)


def kvarn_quantize_k_tile(tile: torch.Tensor, sinkhorn_iters: int, bits: int,
                          layout: KvarnTileLayout, record: torch.Tensor):
    """tile: (128,128) [dim, token]. Writes K region of combined record."""
    q, sc, zp, other = kvarn_quantize_tile(tile, bits, sinkhorn_iters)
    assert q.numel() == KVAR_N_GROUP * KVAR_N_GROUP
    record[layout.k_payload_off: layout.k_payload_off + layout.k_payload_bytes] \
        .copy_(kvarn_pack_bits(q, bits))
    f16 = _rec_f16(record)
    o = layout.k_s_col_off // 2
    f16[o: o + 128].copy_(sc.half())
    o = layout.k_zp_off // 2
    f16[o: o + 128].copy_(zp.half())
    o = layout.k_s_row_off // 2
    f16[o: o + 128].copy_(other.half())


def kvarn_quantize_v_tile(tile: torch.Tensor, sinkhorn_iters: int, bits: int,
                          layout: KvarnTileLayout, record: torch.Tensor):
    """tile: (128,128) [token, dim]. Writes V region of combined record."""
    q, sc, zp, other = kvarn_quantize_tile(tile, bits, sinkhorn_iters)
    assert q.numel() == KVAR_N_GROUP * KVAR_N_GROUP
    record[layout.v_payload_off: layout.v_payload_off + layout.v_payload_bytes] \
        .copy_(kvarn_pack_bits(q, bits))
    f16 = _rec_f16(record)
    o = layout.v_s_row_off // 2
    f16[o: o + 128].copy_(sc.half())
    o = layout.v_zp_off // 2
    f16[o: o + 128].copy_(zp.half())
    o = layout.v_s_col_off // 2
    f16[o: o + 128].copy_(other.half())


def kvarn_dequantize_k_tile(record: torch.Tensor, bits: int,
                            layout: KvarnTileLayout) -> torch.Tensor:
    """Returns (128,128) float32 tile in [dim, token] orientation."""
    q = kvarn_unpack_bits(
        record[layout.k_payload_off: layout.k_payload_off + layout.k_payload_bytes],
        KVAR_N_GROUP * KVAR_N_GROUP, bits).float().reshape(128, 128)
    f16 = _rec_f16(record).float()
    sc = f16[layout.k_s_col_off // 2: layout.k_s_col_off // 2 + 128]
    zp = f16[layout.k_zp_off // 2: layout.k_zp_off // 2 + 128]
    other = f16[layout.k_s_row_off // 2: layout.k_s_row_off // 2 + 128]
    return kvarn_dequantize_tile(q, sc, zp, other)


def kvarn_dequantize_v_tile(record: torch.Tensor, bits: int,
                            layout: KvarnTileLayout) -> torch.Tensor:
    """Returns (128,128) float32 tile in [token, dim] orientation."""
    q = kvarn_unpack_bits(
        record[layout.v_payload_off: layout.v_payload_off + layout.v_payload_bytes],
        KVAR_N_GROUP * KVAR_N_GROUP, bits).float().reshape(128, 128)
    f16 = _rec_f16(record).float()
    sc = f16[layout.v_s_row_off // 2: layout.v_s_row_off // 2 + 128]
    zp = f16[layout.v_zp_off // 2: layout.v_zp_off // 2 + 128]
    other = f16[layout.v_s_col_off // 2: layout.v_s_col_off // 2 + 128]
    return kvarn_dequantize_tile(q, sc, zp, other)


# --------------------------------------------------------------------------
# M2 hooks for online-dequant Triton/CUDA kernels (later work)
# --------------------------------------------------------------------------

def kvarn_m2_triton_available() -> bool:
    """Online-dequant kernels (decode/prefill/varlen). Always False on CPU M2."""
    return False


# --------------------------------------------------------------------------
# Cache layer
# --------------------------------------------------------------------------

class CacheLayer_kvarn(CacheLayer):
    """
    KVarN compressed KV cache layer (M3).

    Storage per 128-token physical group x per (kv_head, 128-dim slice):
    one combined K+V tile record (see KvarnTileLayout). M3 presets:
    kvarn4 (4,4), kvarn5 (5,5) and kvarn5,kvarn4 (asymmetric, Bee
    balanced default); K and V tiles of the same 128-group are sealed
    together (same group boundaries) with different payload widths.

    Storage per 128-token physical group x per (kv_head, 128-dim slice):
    one combined K+V tile record (see KvarnTileLayout). Partial groups are
    held in fp16 staging buffers (rotated domain) and sealed on fill.
    ``get_kv`` dequantizes sealed groups plus inverse-WHTs the staging
    rows, returning standard paged fp16 temps in the ORIGINAL domain, with
    sink + tail rows overwritten exact from the tail buffers so attention
    sees one merged image (single softmax, each key counted once).

    M2 policy per layer:
    - ``has_sink`` (non-SWA only): physical group 0 is never sealed and
      the first 128 committed tokens of every sequence are served exact.
    - ``tail_effective`` exact tokens at the committed-prefix end, served
      from the tail buffers (``tail_dtype``).
    - ``tail_native_exact`` (full-window request): nothing is sealed.
    """

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int = 4,
        v_bits: int = 4,
        sinkhorn_iters: int = KVAR_N_SINKHORN_ITERS,
        tail_tokens: int = 0,
        tail_type="f16",
        is_swa: bool | None = None,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        assert PAGE_SIZE % KVAR_N_GROUP == 0
        assert (k_bits, v_bits) in KVAR_N_M3_PRESETS, \
            f"M3 supports kvarn4 (4,4), kvarn5 (5,5) and kvarn5,kvarn4 " \
            f"(5,4), got {(k_bits, v_bits)}"

        head_dim = attention.head_dim
        self.slices = kvarn_head_slices(head_dim)
        assert self.slices > 0, \
            f"KVarN M2 fail-closed: unsupported head_dim {head_dim} " \
            f"(need one of {KVAR_N_SUPPORTED_HEAD_DIMS})"
        self.head_dim = head_dim
        self.num_kv_heads = attention.num_kv_heads
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.sinkhorn_iters = sinkhorn_iters
        self.layout = kvarn_make_layout(128, 128, k_bits, v_bits)

        # M2 sink/tail policy. SWA layers (sliding_window > 0) get no sink
        # (ring); the tail window here is max_num_tokens (SWA-window
        # overrides are later work).
        if is_swa is None:
            sw = getattr(attention, "sliding_window", -1) or 0
            is_swa = sw > 0
        self.is_swa = bool(is_swa)
        self.has_sink = not self.is_swa
        self.tail_requested_raw = int(tail_tokens or 0)
        assert self.tail_requested_raw >= 0
        policy = kvarn_tail_policy_for(self.tail_requested_raw, max_num_tokens)
        self.tail_requested = policy["requested"]
        self.tail_effective = policy["effective"]
        self.tail_exact_groups = policy["exact_groups"]
        self.tail_native_exact = policy["native_exact"]
        self.tail_dtype = kvarn_parse_tail_type(tail_type)
        self.tail_type_name = "bf16" if self.tail_dtype == torch.bfloat16 else "f16"

        self.num_pages = max_num_tokens // PAGE_SIZE
        self.num_groups = max_num_tokens // KVAR_N_GROUP
        self.ncols = self.num_kv_heads * self.slices

        self.records = None       # uint8 (num_groups, ncols, tile_bytes)
        self.stage_k = None       # fp16 (num_pages, 256, kvh, hd), rotated
        self.stage_v = None
        self.exact_k = None       # tail_dtype (num_pages, 256, kvh, hd), ORIGINAL
        self.exact_v = None       # exact sink+tail source for the get_kv overlay
        self.present = None       # bool (num_groups, 128)
        self.sealed = None        # bool (num_groups,)
        self.device = None

    # -- alloc / free ------------------------------------------------------

    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.records = torch.zeros(
            (self.num_groups, self.ncols, self.layout.tile_bytes),
            dtype=torch.uint8, device=device)
        self.stage_k = torch.zeros(
            (self.num_pages, PAGE_SIZE, self.num_kv_heads, self.head_dim),
            dtype=torch.half, device=device)
        self.stage_v = torch.zeros_like(self.stage_k)
        self.exact_k = torch.zeros(
            (self.num_pages, PAGE_SIZE, self.num_kv_heads, self.head_dim),
            dtype=self.tail_dtype, device=device)
        self.exact_v = torch.zeros_like(self.exact_k)
        self.present = torch.zeros((self.num_groups, KVAR_N_GROUP),
                                   dtype=torch.bool, device=device)
        self.sealed = torch.zeros((self.num_groups,), dtype=torch.bool, device=device)

    @override
    def free(self):
        self.device = None
        self.records = None
        self.stage_k = None
        self.stage_v = None
        self.exact_k = None
        self.exact_v = None
        self.present = None
        self.sealed = None

    # -- M2 record access (for online-dequant Triton/CUDA kernels) ----------

    def get_kvarn_records(self):
        """M2: raw (records, layout, bits, seal map) for online-dequant kernels."""
        assert self.records is not None, "KVarN layer not allocated"
        return {
            "records": self.records,
            "layout": self.layout,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "sealed": self.sealed,
            "present": self.present,
            "has_sink": self.has_sink,
            "tail_effective": self.tail_effective,
        }

    # -- internal: store / seal / materialize --------------------------------

    @torch.inference_mode()
    def _store_rows(self, rows_k: torch.Tensor, rows_v: torch.Tensor,
                    pages: torch.Tensor, offs: torch.Tensor):
        """
        rows_k/rows_v: (T, kvh, hd) fp16 ORIGINAL domain.
        pages/offs: (T,) long physical locations. Rotates into staging,
        mirrors ORIGINAL-domain exact copies into the tail buffers, marks
        present, seals newly completed groups (eager seal stays: completed
        groups seal even inside the tail window; the overlay keeps serving
        them exact). Physical group 0 on sink layers is never sealed;
        native-exact layers (full-window tail) never seal at all.
        """
        if rows_k.numel() == 0:
            return
        dev = self.stage_k.device
        pages = pages.to(torch.long)
        offs = offs.to(torch.long)
        rk = kvarn_wht_head(rows_k.float(), self.head_dim).half().to(dev)
        rv = kvarn_wht_head(rows_v.float(), self.head_dim).half().to(dev)
        self.stage_k[pages, offs] = rk
        self.stage_v[pages, offs] = rv
        self.exact_k[pages, offs] = rows_k.to(self.tail_dtype)
        self.exact_v[pages, offs] = rows_v.to(self.tail_dtype)
        g = pages * (PAGE_SIZE // KVAR_N_GROUP) + offs // KVAR_N_GROUP
        s = offs % KVAR_N_GROUP
        self.present[g, s] = True
        if self.tail_native_exact:
            return
        ug = torch.unique(g)
        done = self.present[ug].all(dim=1) & ~self.sealed[ug]
        for gi in ug[done].tolist():
            if self.has_sink and int(gi) == 0:
                continue  # permanent sink group stays exact
            self._seal_group(int(gi))

    @torch.inference_mode()
    def _seal_group(self, g: int):
        page = g // (PAGE_SIZE // KVAR_N_GROUP)
        half = g % (PAGE_SIZE // KVAR_N_GROUP)
        base = half * KVAR_N_GROUP
        bk = self.stage_k[page, base: base + KVAR_N_GROUP].float()  # (128, kvh, hd) rotated
        bv = self.stage_v[page, base: base + KVAR_N_GROUP].float()
        for h in range(self.num_kv_heads):
            for sl in range(self.slices):
                c = h * self.slices + sl
                d0, d1 = sl * KVAR_N_GROUP, (sl + 1) * KVAR_N_GROUP
                k_tile = bk[:, h, d0:d1].T.contiguous()   # [dim, token]
                v_tile = bv[:, h, d0:d1].contiguous()     # [token, dim]
                rec = self.records[g, c]
                kvarn_quantize_k_tile(k_tile, self.sinkhorn_iters, self.k_bits,
                                      self.layout, rec)
                kvarn_quantize_v_tile(v_tile, self.sinkhorn_iters, self.v_bits,
                                      self.layout, rec)
        self.sealed[g] = True

    @torch.inference_mode()
    def _group_block(self, g: int, out_k: torch.Tensor, out_v: torch.Tensor):
        """Materialize group g (original domain) into page rows of out_k/out_v."""
        page = g // (PAGE_SIZE // KVAR_N_GROUP)
        half = g % (PAGE_SIZE // KVAR_N_GROUP)
        base = half * KVAR_N_GROUP
        ok = out_k[page, base: base + KVAR_N_GROUP]  # (128, kvh, hd) fp16
        ov = out_v[page, base: base + KVAR_N_GROUP]
        if self.has_sink and g == 0:
            # Permanent sink: always serve the exact staging rows.
            ok.copy_(kvarn_wht_head(
                self.stage_k[page, base: base + KVAR_N_GROUP].float(),
                self.head_dim).half())
            ov.copy_(kvarn_wht_head(
                self.stage_v[page, base: base + KVAR_N_GROUP].float(),
                self.head_dim).half())
        elif bool(self.sealed[g]):
            kk = torch.empty((KVAR_N_GROUP, self.num_kv_heads, self.head_dim),
                             dtype=torch.float32, device=ok.device)
            vv = torch.empty_like(kk)
            for h in range(self.num_kv_heads):
                for sl in range(self.slices):
                    c = h * self.slices + sl
                    d0, d1 = sl * KVAR_N_GROUP, (sl + 1) * KVAR_N_GROUP
                    rec = self.records[g, c]
                    kk[:, h, d0:d1] = kvarn_dequantize_k_tile(rec, self.k_bits, self.layout).T
                    vv[:, h, d0:d1] = kvarn_dequantize_v_tile(rec, self.v_bits, self.layout)
            ok.copy_(kvarn_wht_head(kk, self.head_dim).half())
            ov.copy_(kvarn_wht_head(vv, self.head_dim).half())
        else:
            ok.copy_(kvarn_wht_head(
                self.stage_k[page, base: base + KVAR_N_GROUP].float(),
                self.head_dim).half())
            ov.copy_(kvarn_wht_head(
                self.stage_v[page, base: base + KVAR_N_GROUP].float(),
                self.head_dim).half())

    @torch.inference_mode()
    def _apply_exact_overlay(self, k: torch.Tensor, v: torch.Tensor,
                             cache_seqlens: torch.Tensor,
                             block_table: torch.Tensor):
        """
        Overwrite sink + tail rows of the materialized temps with exact
        values from the tail buffers (ORIGINAL domain, cast to temp dtype).

        Per batch entry with committed length n: logical positions
        [0, min(128, n)) (sink, non-SWA only) and
        [max(0, n - tail_effective), n) (tail) are served exact. The union
        with the caller-merged in-flight rows is the single-softmax merge:
        each key appears exactly once across sink/body/tail (Bee portable
        mask semantics). In-flight rows merged by the attention kernel
        after get_kv returns stay exact by construction.
        """
        bsz = cache_seqlens.numel()
        bt = block_table.long()
        seqlens = cache_seqlens.long()
        for b in range(bsz):
            n = int(seqlens[b])
            if n <= 0:
                continue
            parts = []
            if self.has_sink:
                parts.append(torch.arange(min(KVAR_N_SINK_TOKENS, n)))
            if self.tail_effective > 0:
                parts.append(torch.arange(max(0, n - self.tail_effective), n))
            if not parts:
                continue
            pos = torch.unique(torch.cat(parts)).long()
            pages = bt[b, pos // PAGE_SIZE]
            offs = pos % PAGE_SIZE
            # NOTE: indexed assignment (not .copy_ on a gather, which would
            # hit a temporary) so the overlay lands in the temps.
            k[pages, offs] = self.exact_k[pages, offs].to(k.dtype)
            v[pages, offs] = self.exact_v[pages, offs].to(v.dtype)

    # -- CacheLayer interface --------------------------------------------------

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor,
               sliding_window: int = -1) -> tuple:
        # Dense and QSA layers pass -1 here (SWA/GDN layers are recurrent
        # and never kvarn-cached). A window, if ever passed, is applied by
        # the attention kernel itself. Sink/tail exactness is applied here
        # as an overlay, so the temps form one merged image for the single
        # downstream softmax.
        k = torch.empty((self.num_pages, PAGE_SIZE, self.num_kv_heads, self.head_dim),
                        dtype=torch.half, device=self.device)
        v = torch.empty_like(k)
        for g in range(self.num_groups):
            self._group_block(g, k, v)
        self._apply_exact_overlay(k, v, cache_seqlens, block_table)
        return k, v

    @override
    def update_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor, length: int):
        # k/v are the paged fp16 temps returned by get_kv (with the new rows
        # merged in by the attention fallback); persist rows
        # [seqlens, seqlens+length) per batch entry.
        if length == 0:
            return
        bsz = cache_seqlens.numel()
        bt = block_table.long()
        seqlens = cache_seqlens.long()
        for b in range(bsz):
            pos = seqlens[b] + torch.arange(length, device=bt.device)
            pages = bt[b, pos // PAGE_SIZE]
            offs = pos % PAGE_SIZE
            self._store_rows(k[pages, offs], v[pages, offs], pages, offs)

    @override
    def update_kv_direct(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor,
                         k: torch.Tensor, v: torch.Tensor, length: int):
        # k/v: (bsz, length, kvh, hd) new contiguous rows at
        # positions cache_seqlens..+length.
        if length == 0:
            return
        bsz = cache_seqlens.numel()
        bt = block_table.long()
        seqlens = cache_seqlens.long()
        for b in range(bsz):
            pos = seqlens[b] + torch.arange(length, device=bt.device)
            pages = bt[b, pos // PAGE_SIZE]
            offs = pos % PAGE_SIZE
            self._store_rows(k[b], v[b], pages, offs)

    @override
    def copy_page(self, source: CacheLayer_kvarn, from_page: int, to_page: int,
                  num_tokens: int):
        assert self.records.shape == source.records.shape
        assert (self.k_bits, self.v_bits) == (source.k_bits, source.v_bits), \
            "KVarN copy_page requires matching K/V widths (records are " \
            "not comparable across presets)"
        assert self.tail_dtype == source.tail_dtype and \
            self.has_sink == source.has_sink, \
            "KVarN copy_page requires matching tail dtype and sink policy"
        gps = PAGE_SIZE // KVAR_N_GROUP
        self.stage_k[to_page, :num_tokens].copy_(
            source.stage_k[from_page, :num_tokens], non_blocking=True)
        self.stage_v[to_page, :num_tokens].copy_(
            source.stage_v[from_page, :num_tokens], non_blocking=True)
        self.exact_k[to_page, :num_tokens].copy_(
            source.exact_k[from_page, :num_tokens], non_blocking=True)
        self.exact_v[to_page, :num_tokens].copy_(
            source.exact_v[from_page, :num_tokens], non_blocking=True)
        for hh in range(gps):
            gf, gt = from_page * gps + hh, to_page * gps + hh
            lo, hi = hh * KVAR_N_GROUP, min((hh + 1) * KVAR_N_GROUP, num_tokens)
            if hi <= lo:
                continue
            self.present[gt, : hi - lo].copy_(source.present[gf, : hi - lo])
            full = (hi - lo) == KVAR_N_GROUP and bool(source.present[gf].all())
            if full and source.sealed[gf]:
                # Sealed flags travel, except the permanent sink group on
                # sink layers (never sealed, stays exact).
                if self.has_sink and gt == 0:
                    self.sealed[gt] = False
                else:
                    self.records[gt].copy_(source.records[gf], non_blocking=True)
                    self.sealed[gt] = True
            else:
                self.sealed[gt] = False

    @override
    def get_tensors(self):
        return [self.records, self.stage_k, self.stage_v,
                self.exact_k, self.exact_v]

    @override
    def storage_size(self):
        # Compressed records only; fp16 staging + exact tail buffers are overhead.
        return int(self.records.numel()) * torch.uint8.itemsize

    @override
    def overhead_size(self):
        return 2 * int(self.stage_k.numel()) * torch.half.itemsize + \
            (int(self.exact_k.numel()) + int(self.exact_v.numel())) * \
            self.tail_dtype.itemsize + \
            int(self.present.numel()) + int(self.sealed.numel())

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_kvarn,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
                "tail_tokens": self.tail_requested_raw,
                "tail_type": self.tail_type_name,
                "is_swa": self.is_swa,
            }
        }


class CacheLayer_kvarn_qsa(QSAPlanes, CacheLayer_kvarn):
    """
    KVarN KV cache layer with the fp16 QSA indexer planes (raw_k/pooled).
    Mirrors cache/qsa.py CacheLayer_qsa_quant: planes stay fp16 whatever the
    K/V storage is, so sparse block scores never change.
    """

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int = 4,
        v_bits: int = 4,
        sinkhorn_iters: int = KVAR_N_SINKHORN_ITERS,
        tail_tokens: int = 0,
        tail_type="f16",
        is_swa: bool | None = None,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens,
                         k_bits, v_bits, sinkhorn_iters,
                         tail_tokens, tail_type, is_swa)
        self._init_planes(attention, max_num_tokens)

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_kvarn_qsa,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
                "tail_tokens": self.tail_requested_raw,
                "tail_type": self.tail_type_name,
                "is_swa": self.is_swa,
            }
        }
