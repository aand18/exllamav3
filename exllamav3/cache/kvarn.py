"""
KVarN compressed KV cache, M4 (CPU-testable, upstream-mergeable).

Reference: BeeLlama KVarN (Huawei arXiv:2606.03458), ported from
beellama ``src/llama-kvarn.h`` / ``src/llama-kvarn.cpp``,
``src/llama-kv-cache-kvarn.h`` (tail policy) /
``src/llama-kv-cache-kvarn.cpp`` (sink + stage depth + SWA ring) and the
CPU reference kernels in ``ggml/src/ggml-cpu/ops.cpp``.

M4 scope (full CPU parity except CUDA kernels):
- Full 36-combo Bee width table: every cartesian K,V pair in
  {2,3,4,5,6,8} (``src/llama-kvarn.cpp:15-62``). ``kvarn_parse_preset``
  accepts symmetric ``kvarnN`` plus asymmetric ``kvarnK,kvarnV``
  (``/``-separated spelling equivalent, case-insensitive) and bare
  ``K,V`` / ``N`` numerics. Layout, packing and tile quant are
  width-generic; anything outside the table (or head_dim not in
  128/256/512) is fail-closed.
- SWA-window tail: each layer learns its sliding window from the
  attention module (``sliding_window`` attr; the attention kernel keeps
  applying the window itself via the dispatch ``window_size``). The
  exact-tail policy runs against ``min(max_num_tokens, swa_window)``
  (Bee ``llama-context.cpp`` SWA tail group: window ``min(size,
  n_swa)``), SWA layers keep no sink, and the analytical record-ring
  bound ``visible = ceil(min(kv,n_swa)/128)+1`` /
  ``ring = max(1, visible + ceil(ubatch/128) - 1)``
  (``llama-kv-cache-kvarn.cpp:611-634``) is exposed for tests. Same
  preset for SWA layers in M4 (window-capped); per-side SWA pair
  overrides are M5.
- Compact tail arenas: exact sink+tail rows live in on-demand
  per-128-group blocks (the resident set is sink + the effective-tail
  window + one rollback group, i.e. Bee's compact N+R exact-history
  ring, ``llama-kv-cache-tail.cpp`` ``history_stride = N + R``).
  Staging (rotated-domain fp16) exists only for unsealed groups (the
  single open 128-group in steady single-sequence state) and is freed
  on seal. ``storage_size`` (records + resident exact) plus
  ``overhead_size`` (resident staging + bit flags) land well below fp16.
- Attention M2 carry-over: ``get_kv`` serves one merged fp16 image per
  position -- dequantized body with sink + tail rows overwritten exact
  -- so the downstream single-softmax SDPA counts each key once across
  sink/body/tail (Bee portable mask semantics). Eager seal stays:
  completed groups seal even inside the tail window; the overlay keeps
  serving the exact copy alongside the sealed record.
- QSA: sink/tail apply to dense KV; raw_k/pooled indexer planes stay
  fp16 exact (unchanged).
- State/copy: ``copy_page`` carries staging + exact blocks (remapped to
  the destination groups) plus the logical-base / owner / pin metadata;
  sealed flags travel for full groups except physical group 0 on sink
  layers (never sealed). Rewriting a sealed group (page reuse by a new
  sequence, speculative overwrite) unseals and reseals it, so records
  can never go stale. Prompt-cache state compat is still M2-provisional
  (no version bump).

M5 scope (CPU-testable; Triton/CUDA online kernels still OUT):
- SWA K/V pair overrides (``--kv-swa-k`` / ``--kv-swa-v``, Bee
  ``--cache-type-k-swa`` / ``--cache-type-v-swa``): SWA layers use a
  different KVarN preset than full-attention layers (both-or-neither,
  same 36-combo validation; default is the main preset). Each layer
  self-selects its effective preset from ``is_swa``; QSA mapping and
  the tail policy (SWA window cap) respect the per-group preset.
- Autosplit/BC integration (CPU-verifiable half): the BC graph path
  declines KVarN layers per layer (``kvarn_bc_attn_supported`` documents
  why; ``build_bc_attn`` returns None -> dispatch fallback, never
  silent). The QSA synthetic zero-page probe is declined for KVarN
  (``kvarn_autosplit_probe_supported``); the load-time measuring forward
  already covers the fp16-size ``get_kv`` transient
  (``kvarn_autosplit_transient_bytes``) and seal bookkeeping is
  rewrite-safe, so dummy writes cannot corrupt seals.
- Prompt-cache/state versioning: ``tp_export`` carries a
  ``kvarn_version`` tag (``KVAR_N_STATE_VERSION``) plus the SWA override
  pair; ``copy_page`` asserts version + effective widths and carries
  seal/sink/tail metadata, so prompt-cache reuse across the generator
  paths preserves exactness.
- Still OUT (documented fallbacks, force fp16): Triton/CUDA
  online-dequant kernels, the model KLD parity harness, TP loader
  composition beyond ``tp_export`` (``get_tensors`` is not page-major),
  and the CPU second-tier page cache (requires page-major CUDA
  tensors).

M4 non-comparability note: widths change the record payload, so
records sealed under one preset are NOT comparable / interchangeable
with another preset's records or layouts. M4 is still CPU blocks only
(no CUDA kernels); end-to-end parity is validated against the fp16
overlay on CPU.

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
import os
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
# M4: full Bee width table (llama-kvarn.cpp:15-62): every cartesian
# K,V pair in {2,3,4,5,6,8} (36 combos). Bee's balanced default is
# kvarn5/kvarn4 (K 5 bit, V 4 bit).
KVAR_N_VALID_BITS = (2, 3, 4, 5, 6, 8)
KVAR_N_SUPPORTED_PRESETS = frozenset(
    (kb, vb) for kb in KVAR_N_VALID_BITS for vb in KVAR_N_VALID_BITS)
assert len(KVAR_N_SUPPORTED_PRESETS) == 36
# M3 shipped {(4,4),(5,5),(5,4)} only; kept as an alias for compat.
KVAR_N_M3_PRESETS = frozenset({(4, 4), (5, 5), (5, 4)})


def _kvarn_preset_dict() -> dict:
    d: dict = {}
    for kb, vb in sorted(KVAR_N_SUPPORTED_PRESETS):
        d[f"kvarn{kb},kvarn{vb}"] = (kb, vb)
        if kb == vb:
            d[f"kvarn{kb}"] = (kb, vb)
    return d


KVAR_N_PRESETS = _kvarn_preset_dict()

# M2: permanent exact sink on non-SWA layers (llama-kvarn.cpp: sink_tokens
# = 128, validated "exactly 128 unquantized sink tokens").
KVAR_N_SINK_TOKENS = 128
# M2: intrinsic exact-tail floor (llama-kv-cache-kvarn.h tail policy:
# max(128, ...) even for a zero request).
KVAR_N_TAIL_FLOOR_TOKENS = 128
KVAR_N_TAIL_TYPES = {"f16": torch.float16, "bf16": torch.bfloat16}
# M4: SWA keeps only local tail groups exact, no sink
# (llama-kv-cache-kvarn.cpp:119-121 KVAR_N_SWA_TAIL_GROUPS=2).
KVAR_N_SWA_TAIL_GROUPS = 2
# M4: compact exact-history ring rollback reserve (tokens): the resident
# exact window per owning sequence is [n - N - R, n) plus the sink, where
# N is the effective tail (llama-kv-cache-tail.cpp history_stride=N+R).
KVAR_N_TAIL_ROLLBACK_TOKENS = 128


def kvarn_valid_bits(bits: int) -> bool:
    """Bee bit widths (llama-kvarn.cpp llama_kvarn_valid_bits)."""
    return int(bits) in KVAR_N_VALID_BITS


def kvarn_valid_pair(k_bits: int, v_bits: int) -> bool:
    """Full 36-combo Bee desc table (llama-kvarn.cpp:15-62)."""
    return (int(k_bits), int(v_bits)) in KVAR_N_SUPPORTED_PRESETS


def kvarn_parse_preset(spec) -> tuple:
    """
    Parse a ``-cq`` KVarN preset string (case-insensitive; ``/`` and ``,``
    separators are equivalent) into a ``(k_bits, v_bits)`` pair.

    M4 accepts the full Bee 36-combo table: symmetric ``kvarnN`` and
    asymmetric ``kvarnK,kvarnV`` for K,V in {2,3,4,5,6,8}, plus bare
    ``N`` / ``K,V`` numerics over the same set. Anything else raises
    ValueError (fail-closed).
    """
    key = str(spec).strip().lower().replace("/", ",").replace(" ", "")
    if key in KVAR_N_PRESETS:
        return KVAR_N_PRESETS[key]
    bits: list = []
    for part in key.split(","):
        name = part[5:] if part.startswith("kvarn") else part
        if name.isdigit() and int(name) in KVAR_N_VALID_BITS:
            bits.append(int(name))
        else:
            bits = []
            break
    if len(bits) == 1:
        return (bits[0], bits[0])
    if len(bits) == 2 and (bits[0], bits[1]) in KVAR_N_SUPPORTED_PRESETS:
        return (bits[0], bits[1])
    raise ValueError(
        f"Unsupported KVarN preset {spec!r}: M4 supports kvarnN and "
        f"kvarnK,kvarnV for K,V in {list(KVAR_N_VALID_BITS)} "
        f"(36 combos, e.g. kvarn4, kvarn5, kvarn5,kvarn4).")


# M5: prompt-cache / TP state version. Bump whenever the sealed-record
# layout, seal/sink/tail metadata, or tp_export args change incompatibly.
# tp_export carries this tag; the constructor rejects stale versions
# (fail-closed: rebuild the cache / re-export instead of silently
# misreading another preset's records).
KVAR_N_STATE_VERSION = 5


def kvarn_parse_bits(spec) -> int:
    """
    Parse one side of an SWA K/V override: ``kvarnN`` or bare ``N``
    (case-insensitive) for N in {2,3,4,5,6,8}. Anything else raises
    ValueError (fail-closed).
    """
    key = str(spec).strip().lower().replace(" ", "")
    name = key[5:] if key.startswith("kvarn") else key
    if name.isdigit() and int(name) in KVAR_N_VALID_BITS:
        return int(name)
    raise ValueError(
        f"Unsupported KVarN SWA bit width {spec!r}: expected kvarnN or "
        f"bare N with N in {list(KVAR_N_VALID_BITS)} (e.g. kvarn8, 6).")


def kvarn_parse_swa_pair(k_spec, v_spec, default_pair: tuple) -> tuple:
    """
    Resolve the ``--kv-swa-k`` / ``--kv-swa-v`` override pair (Bee
    ``--cache-type-k-swa`` / ``--cache-type-v-swa`` semantics).

    - Both omitted (None/empty) -> ``default_pair`` (the main preset).
    - Exactly one given -> ValueError (Bee requires the pair; a lone
      side would silently mix precisions).
    - Both given -> each parsed with ``kvarn_parse_bits`` and the pair
      validated against the full 36-combo table (``kvarn_valid_pair``).
    """
    empty = lambda s: s is None or str(s).strip() == ""
    if empty(k_spec) and empty(v_spec):
        assert kvarn_valid_pair(*default_pair), \
            f"KVarN SWA default pair must be a valid preset, got {default_pair}"
        return (int(default_pair[0]), int(default_pair[1]))
    if empty(k_spec) or empty(v_spec):
        raise ValueError(
            "KVarN SWA overrides require both --kv-swa-k and --kv-swa-v "
            f"(got k={k_spec!r}, v={v_spec!r}); omit both to reuse the "
            "main -cq preset for SWA layers.")
    try:
        kb = kvarn_parse_bits(k_spec)
    except ValueError as e:
        raise ValueError(f"Invalid --kv-swa-k: {e}") from None
    try:
        vb = kvarn_parse_bits(v_spec)
    except ValueError as e:
        raise ValueError(f"Invalid --kv-swa-v: {e}") from None
    if not kvarn_valid_pair(kb, vb):
        raise ValueError(
            f"Unsupported KVarN SWA pair (kvarn{kb},kvarn{vb}): K,V must "
            f"each be in {list(KVAR_N_VALID_BITS)} (36 combos).")
    return (kb, vb)


def kvarn_bc_attn_supported() -> tuple:
    """
    M5 BC-integration decision (CPU-testable): the graph-captured BC
    attention path does NOT support KVarN layers.

    BC bakes page-major K/V tensors into CUDA graphs; KVarN serves
    dequantized fp16 temps from sealed records plus an exact sink+tail
    overlay, and its online-dequant kernels are out of scope (no CUDA
    on this box). Callers (``build_bc_attn``) decline per layer
    (return None -> dispatch fallback). Returns (False, reason).
    """
    return (False,
            "KVarN has no page-major K/V tensors to bake into BC graphs "
            "(dequant-then-SDPA temps + exact overlay); online-dequant "
            "kernels are out of scope. Decline per layer, use dispatch.")


def kvarn_autosplit_probe_supported() -> tuple:
    """
    M5 autosplit decision (CPU-testable): the QSA synthetic zero-page
    sparse-regime probe (zero page 0, run a measuring forward) is NOT
    run on KVarN layers.

    It would seal garbage groups and touch page tensors KVarN does not
    have (``layer.k``/``layer.qk``); the load-time measuring forward
    through the real cached path already accounts the transient, so the
    probe is skipped and seal state stays intact. Returns
    (False, reason).
    """
    return (False,
            "QSA synthetic zero-page probe would seal garbage groups on "
            "KVarN layers; the load forward already measures the "
            "fp16-size transient. Skip probe, keep seals intact.")


def kvarn_autosplit_transient_bytes(num_pages: int, num_kv_heads: int,
                                    head_dim: int) -> int:
    """
    Conservative autosplit transient for a KVarN layer: the fp16-size
    ``get_kv`` materialization (K+V temps over every page). The real
    transient never exceeds this (records + compact exact residency are
    far smaller), so measuring the layer as fp16-size is a safe upper
    bound that cannot corrupt seal bookkeeping (no synthetic writes).
    """
    return int(num_pages) * PAGE_SIZE * int(num_kv_heads) * int(head_dim) \
        * 2 * torch.half.itemsize


def _kvarn_use_triton() -> bool:
    """
    Opt-in gate for the fused Triton dequant path (attention_fn/
    kvarn_triton.py). Default off: stock behavior is the tested torch loop.
    The kernel module re-checks availability and fails loud when set but
    unrunnable -- an env typo must never silently change numerics.
    """
    return os.environ.get("EXL3_KVARN_TRITON", "0") == "1"


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


def kvarn_swa_tail_window(max_num_tokens: int, swa_window: int) -> int:
    """
    Effective tail-policy window for an SWA layer: the SWA window caps
    the cache window (Bee llama-context.cpp SWA tail group runs the tail
    helper against ``min(size, n_swa)``). Non-positive ``swa_window``
    means "unknown" and falls back to the full cache window (M2
    behavior, documented).
    """
    max_num_tokens = int(max_num_tokens)
    swa_window = int(swa_window)
    assert max_num_tokens >= 0 and swa_window >= 0
    if swa_window <= 0:
        return max_num_tokens
    return min(max_num_tokens, swa_window)


def kvarn_swa_visible_groups(kv_size: int, n_swa: int) -> int:
    """
    Bee ``kvarn_swa_visible_groups`` (llama-kv-cache-kvarn.cpp:620-623):
    ``((min(kv, n_swa) + 127) / 128) + 1`` -- the metadata window may
    span one more 128-token tile than its nominal size.
    """
    kv_size = int(kv_size)
    n_swa = int(n_swa)
    assert kv_size >= 0 and n_swa >= 0
    window_cells = min(kv_size, n_swa) if n_swa > 0 else kv_size
    return (window_cells + KVAR_N_GROUP - 1) // KVAR_N_GROUP + 1


def kvarn_swa_ring_groups(kv_size: int, n_swa: int, ubatch_tokens: int) -> int:
    """
    Bee ``kvarn_record_groups_per_stream`` SWA branch
    (llama-kv-cache-kvarn.cpp:625-634): the record ring stores only tiles
    older than the exact tail, sized ``max(1, visible + in_flight - 1)``
    with ``in_flight = max(1, ceil(ubatch / 128))``. On CPU the ubatch
    analog is one page (PAGE_SIZE).
    """
    ubatch_tokens = int(ubatch_tokens)
    assert ubatch_tokens > 0
    visible = kvarn_swa_visible_groups(kv_size, n_swa)
    in_flight = max(1, (ubatch_tokens + KVAR_N_GROUP - 1) // KVAR_N_GROUP)
    return max(1, visible + in_flight - 1)


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
    assert n_values >= 0 and kvarn_valid_bits(bits)
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
    """FWHT + *1/sqrt(128) over the last dim (llama-kvarn.cpp:668-686).

    Ping-pong buffers instead of per-stage clones: identical math, ~3x
    fewer launches/allocs (Kineto: clone/copy dominated decode CPU).
    """
    src = x.clone().contiguous()
    dst = torch.empty_like(src)
    s = 1
    while s < KVAR_N_GROUP:
        v = src.reshape(*src.shape[:-1], -1, 2, s)
        w = dst.reshape(*dst.shape[:-1], -1, 2, s)
        w[..., 0, :] = v[..., 0, :] + v[..., 1, :]
        w[..., 1, :] = v[..., 0, :] - v[..., 1, :]
        src, dst = dst, src
        s *= 2
    return src.mul_(KVAR_N_INV_SQRT_128)


def kvarn_wht_slices(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """
    Cross-slice FWHT stage only (no-op for head_dim 128). The input must
    already have the per-128 FWHT applied -- e.g. Triton DO_WHT output.
    Same stage order as the tail of kvarn_wht_head.
    """
    slices = kvarn_head_slices(head_dim)
    assert slices > 0, f"KVarN: unsupported head_dim {head_dim}"
    if slices == 1:
        return x
    prefix = x.shape[:-1]
    src = x.reshape(-1, slices, KVAR_N_GROUP)
    dst = torch.empty_like(src)
    scale = 0.7071067811865475 if slices == 2 else 0.5
    # FWHT over the slice axis (ping-pong, no per-stage clones)
    s = 1
    while s < slices:
        vv = src.reshape(src.shape[0], -1, 2, s, KVAR_N_GROUP)
        ww = dst.reshape(dst.shape[0], -1, 2, s, KVAR_N_GROUP)
        ww[:, :, 0] = vv[:, :, 0] + vv[:, :, 1]
        ww[:, :, 1] = vv[:, :, 0] - vv[:, :, 1]
        src, dst = dst, src
        s *= 2
    return (src * scale).reshape(*prefix, head_dim)


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
    return kvarn_wht_slices(v.reshape(*prefix, head_dim), head_dim)


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
    # Bee uses std::round (half away from zero); torch.round is banker's
    # (half to even) and differs on exact .5 fractions. The argument is
    # always non-negative (lo is the row min), so floor(x + 0.5) matches Bee.
    q = torch.floor((balanced - lo.unsqueeze(-1)) / scale.unsqueeze(-1) + 0.5) \
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
    """Online-dequant kernels (decode/prefill/varlen). Always False on CPU M4."""
    return False


# --------------------------------------------------------------------------
# Cache layer
# --------------------------------------------------------------------------

class CacheLayer_kvarn(CacheLayer):
    """
    KVarN compressed KV cache layer (M4).

    Storage per 128-token physical group x per (kv_head, 128-dim slice):
    one combined K+V tile record (see KvarnTileLayout). M4 accepts the
    full Bee 36-combo table; K and V tiles of the same 128-group are
    sealed together (same group boundaries) with different payload
    widths.

    Exact rows (ORIGINAL domain, ``tail_dtype``) live in on-demand
    per-group exact rows (``exact_k``/``exact_v`` static tensors gated by
    ``exact_valid``) covering only the resident window: the
    logical sink (non-SWA) plus the trailing ``tail_effective`` tokens
    plus one rollback group (Bee compact N+R exact-history ring).
    Rotated-domain fp16 staging (``stage_k``/``stage_v`` static tensors,
    zeroed on seal/reset) is only meaningful for unsealed groups
    (the single open 128-group in steady state) and are freed on seal,
    so ``storage_size`` (records + resident exact) and
    ``overhead_size`` (resident staging + flags) stay well below fp16.

    ``get_kv`` dequantizes sealed groups plus inverse-WHTs the staging
    rows, returning standard paged fp16 temps in the ORIGINAL domain,
    with sink + tail rows overwritten exact from the exact blocks so
    attention sees one merged image (single softmax, each key counted
    once).

    M5 policy per layer:
    - ``has_sink`` (non-SWA only): the logical sink group (base 0) is
      never sealed and the first 128 committed tokens of every sequence
      are served exact.
    - ``tail_effective`` exact tokens at the committed-prefix end,
      served from the exact blocks (``tail_dtype``). SWA layers cap the
      policy window at ``swa_window`` and use the SWA override preset
      when one is configured (``swa_override``); dense layers always
      use the main preset. QSA indexer planes stay fp16 either way.
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
        swa_k_bits: int | str | None = None,
        swa_v_bits: int | str | None = None,
        kvarn_version: int = KVAR_N_STATE_VERSION,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        if int(kvarn_version) != KVAR_N_STATE_VERSION:
            raise ValueError(
                f"Stale KVarN state (kvarn_version={kvarn_version!r}, "
                f"current={KVAR_N_STATE_VERSION}): records sealed under "
                f"another layout version are NOT comparable. Rebuild the "
                f"cache / re-export instead of reusing this state.")
        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        assert PAGE_SIZE % KVAR_N_GROUP == 0
        assert kvarn_valid_pair(k_bits, v_bits), \
            f"KVarN M4 supports the full Bee 36-combo table K,V in " \
            f"{list(KVAR_N_VALID_BITS)}, got {(k_bits, v_bits)}"
        # M5 SWA pair override (both-or-neither, same 36-combo table;
        # None/None inherits the main preset). Parsed here so Cache and
        # TP-import call sites share one validation path.
        if (swa_k_bits is None) != (swa_v_bits is None):
            raise ValueError(
                "KVarN SWA overrides require both swa_k_bits and "
                f"swa_v_bits (got {swa_k_bits!r}, {swa_v_bits!r}); pass "
                f"neither to reuse the main preset {(k_bits, v_bits)}.")
        if swa_k_bits is None:
            self.swa_override = None
        else:
            try:
                sk = kvarn_parse_bits(swa_k_bits)
            except ValueError as e:
                raise ValueError(f"Invalid swa_k_bits: {e}") from None
            try:
                sv = kvarn_parse_bits(swa_v_bits)
            except ValueError as e:
                raise ValueError(f"Invalid swa_v_bits: {e}") from None
            assert kvarn_valid_pair(sk, sv), \
                f"KVarN SWA pair must be in the 36-combo table K,V in " \
                f"{list(KVAR_N_VALID_BITS)}, got {(sk, sv)}"
            self.swa_override = (sk, sv)

        head_dim = attention.head_dim
        self.slices = kvarn_head_slices(head_dim)
        assert self.slices > 0, \
            f"KVarN M4 fail-closed: unsupported head_dim {head_dim} " \
            f"(need one of {KVAR_N_SUPPORTED_HEAD_DIMS})"
        self.head_dim = head_dim
        self.num_kv_heads = attention.num_kv_heads
        self.main_k_bits = int(k_bits)
        self.main_v_bits = int(v_bits)
        self.sinkhorn_iters = sinkhorn_iters

        # M4 SWA: learn the sliding window from the attention module
        # (attn.py uses -1 for dense). The tail policy runs against the
        # capped window min(max_num_tokens, swa_window); SWA layers get
        # no sink (ring). An explicitly-SWA layer without a discoverable
        # window keeps the M2 full-window fallback (documented).
        if is_swa is None:
            sw = getattr(attention, "sliding_window", -1)
            try:
                sw = int(sw)
            except (TypeError, ValueError):
                sw = 0
            is_swa = sw > 0
        self.is_swa = bool(is_swa)
        # M5: per-group preset -- SWA layers use the override pair when
        # one was configured, full-attention layers always use the main
        # preset. The window cap and the QSA planes (fp16, untouched)
        # are orthogonal to the widths.
        if self.is_swa and self.swa_override is not None:
            self.k_bits, self.v_bits = self.swa_override
        else:
            self.k_bits, self.v_bits = self.main_k_bits, self.main_v_bits
        self.layout = kvarn_make_layout(128, 128, self.k_bits, self.v_bits)
        self.has_sink = not self.is_swa
        try:
            sw = int(getattr(attention, "sliding_window", -1) or 0)
        except (TypeError, ValueError):
            sw = 0
        self.swa_window = sw if (self.is_swa and sw > 0) else 0
        self.tail_window = kvarn_swa_tail_window(max_num_tokens, self.swa_window) \
            if self.is_swa else int(max_num_tokens)
        self.tail_requested_raw = int(tail_tokens or 0)
        assert self.tail_requested_raw >= 0
        policy = kvarn_tail_policy_for(self.tail_requested_raw, self.tail_window)
        self.tail_requested = policy["requested"]
        self.tail_effective = policy["effective"]
        self.tail_exact_groups = policy["exact_groups"]
        self.tail_native_exact = policy["native_exact"]
        self.tail_dtype = kvarn_parse_tail_type(tail_type)
        self.tail_type_name = "bf16" if self.tail_dtype == torch.bfloat16 else "f16"
        # Bee SWA record-ring bound (analytical; the CPU layer seals
        # eagerly and serves the window from exact blocks, so residency
        # stays under it). CPU ubatch analog is one page.
        self.swa_ring_groups = kvarn_swa_ring_groups(
            self.tail_window, self.swa_window, PAGE_SIZE) if self.is_swa else 0

        self.num_pages = max_num_tokens // PAGE_SIZE
        self.num_groups = max_num_tokens // KVAR_N_GROUP
        self.ncols = self.num_kv_heads * self.slices

        self.records = None       # uint8 (num_groups, ncols, tile_bytes)
        self.stage_k = None  # half (num_groups, 128, kvh, hd) rotated fp16
        self.stage_v = None  # staging for unsealed groups; all-zero == none
        self.exact_k = None  # tail_dtype (num_groups, 128, kvh, hd) ORIGINAL
        self.exact_v = None  # exact rows, resident window only; valid mask gates
        self.exact_valid = None  # bool (num_groups,): group has exact rows
        self.present = None       # bool (num_groups, 128)
        self.sealed = None        # bool (num_groups,)
        self.group_base = None    # int64 (num_groups,): logical pos of slot 0, -1 unwritten
        self.page_owner_n = None  # int64 (num_pages,): last known owning seqlen, -1 untouched
        self.page_pinned = None   # bool (num_pages,): prompt-cache shared pages skip eviction
        self.device = None

    # -- alloc / free ------------------------------------------------------

    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.records = torch.zeros(
            (self.num_groups, self.ncols, self.layout.tile_bytes),
            dtype=torch.uint8, device=device)
        self.stage_k = torch.zeros(
            (self.num_groups, KVAR_N_GROUP, self.num_kv_heads, self.head_dim),
            dtype=torch.half, device=device)
        self.stage_v = torch.zeros_like(self.stage_k)
        self.exact_k = torch.zeros(
            (self.num_groups, KVAR_N_GROUP, self.num_kv_heads, self.head_dim),
            dtype=self.tail_dtype, device=device)
        self.exact_v = torch.zeros_like(self.exact_k)
        self.exact_valid = torch.zeros((self.num_groups,), dtype=torch.bool,
                                       device=device)
        self.present = torch.zeros((self.num_groups, KVAR_N_GROUP),
                                   dtype=torch.bool, device=device)
        self.sealed = torch.zeros((self.num_groups,), dtype=torch.bool, device=device)
        self.group_base = torch.full((self.num_groups,), -1,
                                     dtype=torch.int64, device=device)
        self.page_owner_n = torch.full((self.num_pages,), -1,
                                       dtype=torch.int64, device=device)
        self.page_pinned = torch.zeros((self.num_pages,), dtype=torch.bool, device=device)
        self._img_k = None  # persistent pre-overlay fp16 image (idea 2),
        self._img_v = None  # allocated lazily in get_kv; None = not yet built
        self._dirty_mask = torch.zeros((self.num_groups,), dtype=torch.bool,
                                       device=device)
        # Python-side mirror of "mask is non-empty": every mask-True site
        # sets this, the sweep clears it. Lets steady steps skip the
        # nonzero sync entirely (the fused store keeps the image current
        # via write-through, so the mask is empty almost every step).
        self._dirty_any = False
        # Slot-remap tables for windowed staging/exact (memory plan):
        # live groups map 1:1 onto S_STAGE/S_EXACT slots; group-id
        # indexed tensors shrink to slot-indexed windows in Task 3/4.
        # Until then these tables exist but nothing reads them.
        self._stage_slots = torch.full((4,), -1, dtype=torch.int64,
                                       device=device)
        self._stage_rev = torch.full((self.num_groups,), -1,
                                     dtype=torch.int64, device=device)
        self._exact_slots = torch.full((4,), -1, dtype=torch.int64,
                                       device=device)
        self._exact_rev = torch.full((self.num_groups,), -1,
                                     dtype=torch.int64, device=device)
        # Page -> groups map (num_pages, gps): constant gather replacing
        # per-call pages[:,None]*gps + arange in get_kv (~3 launches).
        _gps = PAGE_SIZE // KVAR_N_GROUP
        self._page_groups = (
            torch.arange(self.num_pages, device=device)[:, None] * _gps
            + torch.arange(_gps, device=device)).to(torch.int64)
        # Incremental image only pays when the allocation is modest; huge
        # contexts keep the memory-slim full rematerialization path.
        # 160 pages ~= 40k tokens: covers the 32k protocol (129 pages
        # with decode headroom) at ~160MB/layer for kvh4/hd256. The
        # legacy path above this is prefill-grade only (it rematerializes
        # the whole context per step: 8.2 tok/s at 32k vs 64 fp16).
        self._img_ok = self.num_pages <= 160
        self._evict_tick = 0  # store calls since the last evict scan
        # Scratch for the fused single-row store's [code, group] report:
        # the kernel overwrites both words every launch, so one persistent
        # buffer replaces a per-call alloc (no fill needed, no staleness).
        self._store_status = torch.zeros(2, dtype=torch.int64,
                                         device=device)

    @override
    def free(self):
        self.device = None
        self.records = None
        self.stage_k = None
        self.stage_v = None
        self.exact_k = None
        self.exact_v = None
        self.exact_valid = None
        self.present = None
        self.sealed = None
        self.group_base = None
        self.page_owner_n = None
        self.page_pinned = None
        self._img_k = None
        self._img_v = None
        self._dirty_mask = None
        self._dirty_any = False
        self._stage_slots = None
        self._stage_rev = None
        self._exact_slots = None
        self._exact_rev = None
        self._page_groups = None
        self._evict_tick = 0
        self._store_status = None

    # -- Slot-remap for windowed staging/exact (memory plan) -----------------

    def _stage_slot(self, g: int) -> int:
        """Slot holding group g's staging rows; assigns a free slot."""
        s = int(self._stage_rev[g])
        if s >= 0:
            return s
        free = (self._stage_slots < 0).nonzero().flatten()
        assert free.numel(), "KVarN: staging slot overflow"
        s = int(free[0])
        self._stage_slots[s] = g
        self._stage_rev[g] = s
        return s

    def _stage_release(self, g: int) -> None:
        """Free group g's staging slot (idempotent)."""
        s = int(self._stage_rev[g])
        if s >= 0:
            self._stage_slots[s] = -1
            self._stage_rev[g] = -1

    def _exact_slot(self, g: int) -> int:
        """Slot holding group g's exact rows; assigns a free slot."""
        s = int(self._exact_rev[g])
        if s >= 0:
            return s
        free = (self._exact_slots < 0).nonzero().flatten()
        assert free.numel(), "KVarN: exact slot overflow"
        s = int(free[0])
        self._exact_slots[s] = g
        self._exact_rev[g] = s
        return s

    def _exact_release(self, g: int) -> None:
        """Free group g's exact slot (idempotent)."""
        s = int(self._exact_rev[g])
        if s >= 0:
            self._exact_slots[s] = -1
            self._exact_rev[g] = -1

    # -- M4 record access (for online-dequant Triton/CUDA kernels) ----------

    def get_kvarn_records(self):
        """M4: raw (records, layout, bits, seal map) for online-dequant kernels."""
        assert self.records is not None, "KVarN layer not allocated"
        return {
            "records": self.records,
            "layout": self.layout,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "main_k_bits": self.main_k_bits,
            "main_v_bits": self.main_v_bits,
            "swa_override": self.swa_override,
            "is_swa": self.is_swa,
            "kvarn_version": KVAR_N_STATE_VERSION,
            "sealed": self.sealed,
            "present": self.present,
            "has_sink": self.has_sink,
            "tail_effective": self.tail_effective,
            "tail_window": self.tail_window,
            "swa_window": self.swa_window,
            "swa_ring_groups": self.swa_ring_groups,
            "stage_groups": sorted(self._live_stage_groups()),
            "exact_groups": sorted(self._live_exact_groups()),
        }

    # -- internal: store / seal / materialize --------------------------------

    def _block_shape(self) -> tuple:
        return (KVAR_N_GROUP, self.num_kv_heads, self.head_dim)

    def _live_stage_groups(self):
        """Groups with live staging: unsealed with any present row
        (equivalent to the old stage_blocks dict keys)."""
        live = (~self.sealed) & self.present.any(dim=1)
        return live.nonzero().flatten().tolist()

    def _alloc_stage_block(self, g: int):
        self.stage_k[g].zero_()
        self.stage_v[g].zero_()
        return [self.stage_k[g], self.stage_v[g]]

    def _live_exact_groups(self):
        """Groups with resident exact rows (equivalent to the old
        exact_blocks dict keys)."""
        return self.exact_valid.nonzero().flatten().tolist()

    def _alloc_exact_block(self, g: int):
        # Fresh blocks read zero (matches the old dict behavior where a
        # missing entry meant zeros); callers then fill resident slots.
        self.exact_k[g].zero_()
        self.exact_v[g].zero_()
        self.exact_valid[g] = True
        return [self.exact_k[g], self.exact_v[g]]

    def _exact_keep(self, pos: torch.Tensor, n_new: int) -> torch.Tensor:
        """
        Which written logical positions need exact residency: the sink
        (non-SWA) plus the trailing compact window [n_new - N - R, n_new)
        (Bee compact history_stride = N + R). Everything else is served
        from the sealed body.
        """
        keep = pos >= (n_new - self.tail_effective - KVAR_N_TAIL_ROLLBACK_TOKENS)
        if self.has_sink:
            keep = keep | (pos < KVAR_N_SINK_TOKENS)
        return keep

    @torch.inference_mode()
    def _touch_batch(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor,
                     length: int):
        """
        Record the owning-sequence length for every page backing the
        current batch entries (each entry's full committed prefix at its
        post-write length, not just the rows being written: a sequence
        advances through different pages on successive calls, so
        per-page owners must be refreshed from the batch union). Pages of
        idle sequences keep their last owner (stale-but-correct: an idle
        sequence never advances). Prompt-cache sharing must go through
        ``copy_page``, which pins the shared pages against max-owner
        eviction.
        """
        bt = block_table.long()
        seqlens = cache_seqlens.long()
        bsz = seqlens.numel()  # shape only, no sync
        if bsz == 0:
            return
        # Steady single-row decode appends (bsz 1, length 1) refresh
        # per-page owners the fused store kernel already maintains for
        # the appended page -- and nothing else reads page_owner_n
        # between evict scans. So this path runs fully only on the
        # evict tick (same counter the scan gates on: owners are current
        # whenever the scan runs). Multi-row appends always run fully.
        # The out-of-range validation still fires within <=256 steps for
        # systematic table bugs (plus the gather bounds-check backstop
        # on every step); prefill/multi-row always validate.
        if bsz == 1 and length == 1 and (self._evict_tick & 255):
            return
        if bsz == 1:
            # Steady single-sequence path (decode appends and batch-1
            # prefill): the whole row validates and updates without the
            # arange + 2D-mask machinery (~8 launches saved). The
            # whole-row range check is a superset of the in-use check
            # (padding is -1 by contract; anything else out of range is
            # a harness bug), and -1 padding is filtered (never wraps
            # onto the last owner's slot).
            row = bt[0]
            if bool(((row >= self.num_pages) | (row < -1)).any()):
                raise AssertionError(
                    "KVarN: block table page out of range")
            idx = row[(row >= 0) & (row < self.num_pages)]
            n_b = seqlens[0] + int(length)
            cur = self.page_owner_n[idx]
            self.page_owner_n[idx] = torch.where(cur < n_b, n_b, cur)
            return
        # Vectorized over the batch: no .item()/.tolist() in steady
        # state (was 3 syncs per entry: seqlens int + min/max asserts).
        # n_vec stays a tensor; per-entry slicing uses a broadcast mask
        # so no Python int is ever needed. One validation sync for the
        # whole batch preserves the loud out-of-range failure.
        n_vec = seqlens + int(length)
        npages = (n_vec + PAGE_SIZE - 1) // PAGE_SIZE
        npages = torch.where(n_vec > 0, npages,
                             torch.zeros_like(npages))
        max_cols = bt.shape[1]
        cols = torch.arange(max_cols, device=bt.device)
        mask = cols.unsqueeze(0) < npages.unsqueeze(1)
        # Single validation sync for the whole batch (was 2 per entry
        # via min/max). Empty selection validates clean (no-op below).
        sel = bt[mask]
        if sel.numel() and bool(((sel < 0) | (sel >= self.num_pages)).any()):
            raise AssertionError(
                "KVarN: block table page out of range")
        # Max-update per entry (identical semantics to the per-entry
        # loop; _store_rows still applies the min-drop for page reuse
        # after this, so reuse final state is unchanged). Empty masks
        # are no-ops: no per-entry syncs.
        for b in range(bsz):
            idx = bt[b][mask[b]]
            n_b = n_vec[b]
            cur = self.page_owner_n[idx]
            self.page_owner_n[idx] = torch.where(cur < n_b, n_b, cur)

    def _store_row_single(self, pages, offs, pos, n_new,
                          rk, rv, ek, ev, g, s) -> bool:
        """Single-row decode fast path. Writes the row when it purely
        appends to an open (or fresh) group; returns False for any policy
        event (reuse with live content, sealed overwrite) so the caller
        falls through to the general loop. Mirrors the loop body exactly
        for the T == 1 case (same reset/write/exact/owner semantics)."""
        gps = PAGE_SIZE // KVAR_N_GROUP
        gi = int(g[0])
        si = int(s[0])
        pos0 = int(pos[0])
        bold = int(self.group_base[gi])
        bnew = pos0 - si
        if bold != bnew and bold >= 0:
            return False  # page reuse with live content: general path
        if bool(self.sealed[gi]):
            return False  # overwrite of sealed content: general path
        page = gi // gps
        if bold != bnew:
            # Fresh group: reset bookkeeping (nothing live to preserve).
            self.sealed[gi] = False
            self.present[gi] = False
            self.stage_k[gi].zero_()
            self.stage_v[gi].zero_()
            self.exact_valid[gi] = False
            self.group_base[gi] = bnew
            self.page_pinned[page] = False
        self.stage_k[gi, si] = rk[0]
        self.stage_v[gi, si] = rv[0]
        self.present[gi, si] = True
        if pos0 >= n_new - self.tail_effective - KVAR_N_TAIL_ROLLBACK_TOKENS or \
           (self.has_sink and pos0 < KVAR_N_SINK_TOKENS):
            if not bool(self.exact_valid[gi]):
                self._alloc_exact_block(gi)
            self.exact_k[gi, si] = ek[0]
            self.exact_v[gi, si] = ev[0]
        cur_owner = int(self.page_owner_n[page])
        self.page_owner_n[page] = n_new if cur_owner < 0 \
            else min(cur_owner, n_new)
        # Trailing owner-max, mirroring the general path's refresh loop.
        if n_new > int(self.page_owner_n[page]):
            self.page_owner_n[page] = n_new
        return True

    @torch.inference_mode()
    def _store_rows(self, rows_k: torch.Tensor, rows_v: torch.Tensor,
                    pages: torch.Tensor, offs: torch.Tensor,
                    pos: torch.Tensor, n_new: int):
        """
        rows_k/rows_v: (T, kvh, hd) fp16 ORIGINAL domain, at logical
        positions ``pos`` ((T,) long) with owning-sequence length
        ``n_new`` (plain int or 0-d tensor: callers pass the tensor form
        to avoid a CPU sync; only the torch fallback paths below
        materialize the Python int, the fused Triton steady path never
        touches it). Rotates into staging, mirrors exact copies for the
        resident window (sink + trailing N+R), marks present, seals newly
        completed groups (eager seal stays: completed groups seal even
        inside the tail window; the overlay keeps serving them exact).
        The logical sink group (base 0) on sink layers is never sealed;
        native-exact layers (full-window tail) never seal at all.

        Rewrites are safe: replacing a group's content (page reuse by a
        new sequence, detected via the logical base) resets it, and
        overwriting a sealed group unseals it first, so records can never
        go stale -- the group reseals with fresh staging below.
        """
        if rows_k.numel() == 0:
            return
        dev = self.device
        pages = pages.to(torch.long)
        offs = offs.to(torch.long)
        pos = pos.to(torch.long)
        # Fused single-row store (decode appends): 2 launches + 1 status
        # sync instead of ~50 launches + ~10 syncs. Policy events bail
        # (code 1) into the torch paths below; completed groups come back
        # for the torch sealer (code 2). The fused path never reads
        # n_new (position math resolves in-kernel), so the int()
        # materialization stays below it: steady decode pays zero syncs
        # here, fallback paths keep the exact old behavior.
        if rows_k.shape[0] == 1 and not self.is_swa and _kvarn_use_triton():
            from ..modules.attention_fn.kvarn_triton import (
                kvarn_triton_available, kvarn_triton_store_row)
            assert kvarn_triton_available(), \
                "EXL3_KVARN_TRITON=1 but the Triton path is unavailable " \
                "(needs triton + CUDA); unset it for the torch path."
            img = self._img_k if self._img_k is not None else None
            code, gg = kvarn_triton_store_row(
                self, rows_k, rows_v, pages, offs, pos,
                PAGE_SIZE // KVAR_N_GROUP, KVAR_N_SINK_TOKENS,
                KVAR_N_TAIL_ROLLBACK_TOKENS,
                img, self._img_v if img is not None else None)
            if code == 0:
                # Pure append: the image is current via write-through,
                # nothing dirtied (the sweep below stays empty).
                self._evict_exact_all(1)
                return
            if code == 2:
                self._evict_exact_all(1)
                self._seal_group(gg)
                self._dirty_mask[gg] = True
                self._dirty_any = True
                return
            if code == 3:
                # Fresh group reset: the row itself wrote through, but
                # the reset siblings need a refresh from staging.
                self._evict_exact_all(1)
                self._dirty_mask[gg] = True
                self._dirty_any = True
                return
            # code == 1: fall through to the torch paths below.
        # Torch fallbacks need the Python int (single sync, same as the
        # old caller-side int); the fused path above never paid it.
        n_new = int(n_new)
        # One batched WHT for K+V (was two calls): same per-element
        # math, ~half the launches. Bit-exact (batching preserves order).
        # With EXL3_KVARN_TRITON=1 the fused Triton row-WHT runs instead
        # (1 launch); PARITY=1 asserts it against this torch reference.
        if _kvarn_use_triton():
            from ..modules.attention_fn.kvarn_triton import (
                kvarn_triton_available, kvarn_triton_wht_rows,
                kvarn_triton_parity_check)
            assert kvarn_triton_available(), \
                "EXL3_KVARN_TRITON=1 but the Triton path is unavailable " \
                "(needs triton + CUDA); unset it for the torch path."
            stacked = torch.stack((rows_k.float(), rows_v.float()))
            rkv_t = kvarn_triton_wht_rows(stacked, self.head_dim)
            if kvarn_triton_parity_check():
                rkv_r = kvarn_wht_head(stacked, self.head_dim)
                assert torch.equal(rkv_t, rkv_r), \
                    "KVarN Triton store WHT disagrees with torch"
            rkv = rkv_t
        else:
            rkv = kvarn_wht_head(torch.stack((rows_k.float(), rows_v.float())),
                                 self.head_dim)
        rkv = rkv.half().to(dev)
        rk, rv = rkv[0], rkv[1]
        ek = rows_k.to(self.tail_dtype)
        ev = rows_v.to(self.tail_dtype)
        g = pages * (PAGE_SIZE // KVAR_N_GROUP) + offs // KVAR_N_GROUP
        s = offs % KVAR_N_GROUP
        keep = self._exact_keep(pos, n_new)
        base = pos - s
        if rows_k.shape[0] == 1 and \
                self._store_row_single(pages, offs, pos, n_new,
                                        rk, rv, ek, ev, g, s):
            self._evict_exact_all(1)
            gi = int(g[0])
            if bool(self.present[gi].all()) and not bool(self.sealed[gi]):
                if not (self.has_sink and int(self.group_base[gi]) == 0):
                    self._seal_group(gi)
            gv = g.to(device=self.device, dtype=torch.long)
            self._dirty_mask[gv[(gv >= 0) & (gv < self.num_groups)]] = True
            self._dirty_any = True
            return
        for gi in torch.unique(g).tolist():
            gi = int(gi)
            m = (g == gi)
            slots = s[m]
            bnew = int(base[m][0])
            bold = int(self.group_base[gi])
            if bold != bnew:
                # New content (first write or page reuse): reset the group.
                # A reused page stops aliasing its prompt-cache sibling.
                self.sealed[gi] = False
                self.present[gi] = False
                self.stage_k[gi].zero_()
                self.stage_v[gi].zero_()
                self.exact_valid[gi] = False
                self.group_base[gi] = bnew
                page = gi // (PAGE_SIZE // KVAR_N_GROUP)
                self.page_pinned[page] = False
                # The page's owner drops to the new writer: a reused page
                # otherwise keeps its dead sequence's length and the
                # end-of-call eviction (owner - N - R) would discard the
                # just-written tail as if it were ancient history. min()
                # is the safe direction (a low owner only keeps more
                # exact rows); untouched pages (-1) take the new length.
                cur_owner = int(self.page_owner_n[page])
                self.page_owner_n[page] = n_new if cur_owner < 0 \
                    else min(cur_owner, n_new)
            elif bool(self.sealed[gi]):
                # Overwrite of sealed content: unseal so the fresh rows
                # reseal below (records never go stale). The owner drops
                # to the new writer for the same reason as above
                # (page reuse at the same base with a shorter sequence).
                self.sealed[gi] = False
                page = gi // (PAGE_SIZE // KVAR_N_GROUP)
                cur_owner = int(self.page_owner_n[page])
                if cur_owner >= 0:
                    self.page_owner_n[page] = min(cur_owner, n_new)
            self.stage_k[gi, slots] = rk[m]
            self.stage_v[gi, slots] = rv[m]
            self.present[gi, slots] = True
            km = m & keep
            if bool(km.any()):
                if not bool(self.exact_valid[gi]):
                    self._alloc_exact_block(gi)
                self.exact_k[gi, s[km]] = ek[km]
                self.exact_v[gi, s[km]] = ev[km]
        # Every touched group changed content (stage write, reset or
        # unseal): refresh it on next materialization (idea 2).
        gm = g.to(device=self.device, dtype=torch.long)
        self._dirty_mask[gm[(gm >= 0) & (gm < self.num_groups)]] = True
        self._dirty_any = True
        for p in torch.unique(pages).tolist():
            p = int(p)
            if n_new > int(self.page_owner_n[p]):
                self.page_owner_n[p] = n_new
        self._evict_exact_all(int(rows_k.shape[0]))
        if self.tail_native_exact:
            return
        ug = torch.unique(g)
        done = self.present[ug].all(dim=1) & ~self.sealed[ug]
        seal_gs = ug[done]
        if seal_gs.numel():
            if self.has_sink:
                # Logical sink group (base 0) stays exact, never seals.
                seal_gs = seal_gs[self.group_base[seal_gs] != 0]
            if seal_gs.numel():
                self._seal_groups_batched(seal_gs)

    @torch.inference_mode()
    def _evict_exact_all(self, n_rows: int = 0):
        """
        Drop every resident exact block fully below its owner's compact
        window (``base + 128 <= owner_n - N - R``), keeping the sink and
        prompt-cache-pinned pages. Owners are refreshed by
        ``_touch_batch`` on every update call, so a periodic scan over the
        (small) resident set keeps memory bounded as the window slides.

        Fully sync-free: dropped blocks are strictly below the served
        sink+tail overlay window, so WHEN eviction runs is numerically
        invisible -- no high-water read needed. Prefill-scale calls
        (n_rows >= 128, a shape-only int) scan immediately, covering
        multi-quantum jumps; decode-scale calls scan every 256th call.
        The old int(max()) quantum gate cost a DtoH sync per layer per
        call (16/step, Kineto top-5); the per-group int() reads are gone
        too (one vectorized mask over the resident set).
        """
        self._evict_tick += 1
        if n_rows < KVAR_N_GROUP and self._evict_tick < 256:
            return
        self._evict_tick = 0
        floor = self.tail_effective + KVAR_N_TAIL_ROLLBACK_TOKENS
        gps = PAGE_SIZE // KVAR_N_GROUP
        res = self.exact_valid.nonzero().flatten()
        if res.numel():  # shape only, no sync
            p = res // gps
            owner = self.page_owner_n[p]
            b = self.group_base[res]
            drop = (owner >= 0) & (b >= 0) & \
                (b + KVAR_N_GROUP <= owner - floor) & \
                ~self.page_pinned[p]
            if self.has_sink:
                # Logical sink (base 0) stays exact.
                drop = drop & ~((b >= 0) & (b < KVAR_N_GROUP))
            self.exact_valid[res[drop]] = False

    @torch.inference_mode()
    def _seal_groups_batched(self, gs: torch.Tensor):
        """Seal several groups with one batched Sinkhorn+quantize+pack per
        K and V. Bit-exact vs looping _seal_group: variance_normalize,
        RTN quantize and pack are all per-tile independent (leading batch
        dims broadcast), and each record payload is byte-exact so the
        packed stream concatenates without padding."""
        L = self.layout
        kvh, sl = self.num_kv_heads, self.slices
        C = kvh * sl
        bk = self.stage_k[gs].float()
        bv = self.stage_v[gs].float()
        G = bk.shape[0]
        bkr = bk.reshape(G, KVAR_N_GROUP, kvh, sl, KVAR_N_GROUP)
        bvr = bv.reshape(G, KVAR_N_GROUP, kvh, sl, KVAR_N_GROUP)
        k_tiles = bkr.permute(0, 2, 3, 4, 1).reshape(G * C, 128, 128)
        v_tiles = bvr.permute(0, 2, 3, 1, 4).reshape(G * C, 128, 128)
        qk, sck, zpk, otk = kvarn_quantize_tile(
            k_tiles, self.k_bits, self.sinkhorn_iters)
        qv, scv, zpv, otv = kvarn_quantize_tile(
            v_tiles, self.v_bits, self.sinkhorn_iters)
        recs = self.records[gs]  # (G, C, B) copy; written back below
        recs[:, :, L.k_payload_off:L.k_payload_off + L.k_payload_bytes] = \
            kvarn_pack_bits(qk.reshape(-1), self.k_bits) \
            .reshape(G, C, L.k_payload_bytes)
        recs[:, :, L.v_payload_off:L.v_payload_off + L.v_payload_bytes] = \
            kvarn_pack_bits(qv.reshape(-1), self.v_bits) \
            .reshape(G, C, L.v_payload_bytes)
        f16 = _rec_f16(recs)
        f16[:, :, L.k_s_col_off // 2: L.k_s_col_off // 2 + 128] = \
            sck.half().reshape(G, C, 128)
        f16[:, :, L.k_zp_off // 2: L.k_zp_off // 2 + 128] = \
            zpk.half().reshape(G, C, 128)
        f16[:, :, L.k_s_row_off // 2: L.k_s_row_off // 2 + 128] = \
            otk.half().reshape(G, C, 128)
        f16[:, :, L.v_s_row_off // 2: L.v_s_row_off // 2 + 128] = \
            scv.half().reshape(G, C, 128)
        f16[:, :, L.v_zp_off // 2: L.v_zp_off // 2 + 128] = \
            zpv.half().reshape(G, C, 128)
        f16[:, :, L.v_s_col_off // 2: L.v_s_col_off // 2 + 128] = \
            otv.half().reshape(G, C, 128)
        self.records[gs] = recs
        self.sealed[gs] = True
        self._dirty_mask[gs.to(self.device)] = True
        self._dirty_any = True
        # M4: staging is freed on seal; only the single open group (plus
        # the sink) retains fp16 staging. Exact blocks stay for the
        # overlay over sealed tail groups.
        self.stage_k[gs].zero_()
        self.stage_v[gs].zero_()

    @torch.inference_mode()
    def _seal_group(self, g: int):
        # Single-group seal via the batched core: one Sinkhorn+quantize
        # per K/V over all tiles instead of per-tile loops (~16x fewer
        # launches and status syncs; the per-tile math is independent so
        # the packed records are bit-identical). Decode seals every 128
        # steps per layer, so the loop form showed up as an 88ms/step
        # cliff on 256-step runs.
        assert bool(self.present[g].any()), \
            f"KVarN: sealing group {g} without staging (base {int(self.group_base[g])})"
        self._seal_groups_batched(
            torch.tensor([g], device=self.device, dtype=torch.long))

    @torch.inference_mode()
    def _sealed_tiles(self, g: int):
        """
        Dequantize sealed group g to rotated-domain fp32 blocks
        (bk, bv) shaped (128, kvh, hd). Default is the tested torch loop;
        with EXL3_KVARN_TRITON=1 (and triton + CUDA present) the fused
        Triton kernel in attention_fn/kvarn_triton.py is used instead.
        With EXL3_KVARN_TRITON_PARITY=1 both run and must agree (the
        acceptance test for the untested-on-GPU kernel path).
        """
        if _kvarn_use_triton():
            from ..modules.attention_fn.kvarn_triton import (
                kvarn_triton_available, kvarn_triton_dequant_group,
                kvarn_triton_parity_check)
            assert kvarn_triton_available(), \
                "EXL3_KVARN_TRITON=1 but the Triton path is unavailable " \
                "(needs triton + CUDA); unset it for the torch path."
            bk_t, bv_t = kvarn_triton_dequant_group(
                self.records[g], self.layout, self.k_bits, self.v_bits,
                self.num_kv_heads, self.slices)
            if kvarn_triton_parity_check():
                bk_r, bv_r = self._sealed_tiles_torch(g)
                assert torch.equal(bk_t.half(), bk_r.half()) and \
                    torch.equal(bv_t.half(), bv_r.half()), \
                    "KVarN Triton dequant disagrees with the torch " \
                    f"reference on group {g}"
            return bk_t, bv_t
        return self._sealed_tiles_torch(g)

    @torch.inference_mode()
    def _sealed_tiles_torch(self, g: int):
        """Torch reference for _sealed_tiles (tested on CPU, runs anywhere)."""
        bk = torch.empty(self._block_shape(), dtype=torch.float32, device=self.device)
        bv = torch.empty_like(bk)
        for h in range(self.num_kv_heads):
            for sl in range(self.slices):
                c = h * self.slices + sl
                d0, d1 = sl * KVAR_N_GROUP, (sl + 1) * KVAR_N_GROUP
                rec = self.records[g, c]
                bk[:, h, d0:d1] = kvarn_dequantize_k_tile(rec, self.k_bits, self.layout).T
                bv[:, h, d0:d1] = kvarn_dequantize_v_tile(rec, self.v_bits, self.layout)
        return bk, bv

    @torch.inference_mode()
    def _staging_from_records(self, g: int, records=None):
        """Rotated-domain fp16 staging rows rebuilt from sealed records."""
        if records is not None and records is not self.records:
            # Cross-layer rebuild (copy_page): torch path only, the Triton
            # hook serves own records (parity self-check needs own layout).
            bk = torch.empty(self._block_shape(), dtype=torch.float32, device=self.device)
            bv = torch.empty_like(bk)
            for h in range(self.num_kv_heads):
                for sl in range(self.slices):
                    c = h * self.slices + sl
                    d0, d1 = sl * KVAR_N_GROUP, (sl + 1) * KVAR_N_GROUP
                    rec = records[g, c]
                    bk[:, h, d0:d1] = kvarn_dequantize_k_tile(rec, self.k_bits, self.layout).T
                    bv[:, h, d0:d1] = kvarn_dequantize_v_tile(rec, self.v_bits, self.layout)
            return [bk.half(), bv.half()]
        bk, bv = self._sealed_tiles(g)
        return [bk.half(), bv.half()]

    @torch.inference_mode()
    def _group_block(self, g: int, out_k: torch.Tensor, out_v: torch.Tensor):
        """Materialize group g (original domain) into page rows of out_k/out_v."""
        gps = PAGE_SIZE // KVAR_N_GROUP
        page = g // gps
        half = g % gps
        base = half * KVAR_N_GROUP
        ok = out_k[page, base: base + KVAR_N_GROUP]  # (128, kvh, hd) fp16
        ov = out_v[page, base: base + KVAR_N_GROUP]
        if bool(self.sealed[g]):
            kk, vv = self._sealed_tiles(g)
            ok.copy_(kvarn_wht_head(kk, self.head_dim).half())
            ov.copy_(kvarn_wht_head(vv, self.head_dim).half())
        else:
            # Static staging reads zeros for never-written groups, matching
            # the old dict-miss path.
            ok.copy_(kvarn_wht_head(self.stage_k[g].float(), self.head_dim).half())
            ov.copy_(kvarn_wht_head(self.stage_v[g].float(), self.head_dim).half())

    @torch.inference_mode()
    def _apply_exact_overlay(self, k: torch.Tensor, v: torch.Tensor,
                             cache_seqlens: torch.Tensor,
                             block_table: torch.Tensor):
        """
        Overwrite sink + tail rows of the materialized temps with exact
        values from the exact blocks (ORIGINAL domain, cast to temp dtype).

        Per batch entry with committed length n: logical positions
        [0, min(128, n)) (sink, non-SWA only) and
        [max(0, n - tail_effective), n) (tail) are served exact. The union
        with the caller-merged in-flight rows is the single-softmax merge:
        each key appears exactly once across sink/body/tail (Bee portable
        mask semantics). In-flight rows merged by the attention kernel
        after get_kv returns stay exact by construction. Positions whose
        exact block is absent (unreachable in normal flows: the resident
        window always covers the overlay window) fall back to the body.
        """
        bsz = cache_seqlens.numel()
        bt = block_table.long()
        seqlens = cache_seqlens.long()
        gps = PAGE_SIZE // KVAR_N_GROUP
        for b in range(bsz):
            n = int(seqlens[b])
            if n <= 0:
                continue
            parts = []
            if self.has_sink:
                parts.append(torch.arange(min(KVAR_N_SINK_TOKENS, n), device=bt.device))
            if self.tail_effective > 0:
                parts.append(torch.arange(max(0, n - self.tail_effective), n, device=bt.device))
            if not parts:
                continue
            pos = torch.unique(torch.cat(parts)).long()
            pages = bt[b, pos // PAGE_SIZE]
            offs = pos % PAGE_SIZE
            g = pages * gps + offs // KVAR_N_GROUP
            s = offs % KVAR_N_GROUP
            for gi in torch.unique(g).tolist():
                gi = int(gi)
                if not bool(self.exact_valid[gi]):
                    continue
                m = (g == gi)
                pm, om, sm = pages[m], offs[m], s[m]
                # NOTE: indexed assignment (not .copy_ on a gather, which would
                # hit a temporary) so the overlay lands in the temps.
                k[pm, om] = self.exact_k[gi, sm].to(k.dtype)
                v[pm, om] = self.exact_v[gi, sm].to(v.dtype)

    # -- CacheLayer interface --------------------------------------------------

    @override
    @torch.inference_mode()
    def _dequant_groups_batched(self, Gs: torch.Tensor):
        """Dequantize sealed groups Gs (1D long tensor, on device) to
        rotated-domain fp32 (bk, bv) shaped (len(Gs), 128, kvh, hd).

        Default is the tested torch batch; with EXL3_KVARN_TRITON=1 (and
        triton + CUDA present) the fused Triton kernel in
        attention_fn/kvarn_triton.py is used instead. With
        EXL3_KVARN_TRITON_PARITY=1 both run and must agree bit-exact
        (fp32 == fp32); that is the acceptance test for the kernel path.
        """
        if _kvarn_use_triton():
            from ..modules.attention_fn.kvarn_triton import (
                kvarn_triton_available, kvarn_triton_dequant_groups,
                kvarn_triton_parity_check)
            assert kvarn_triton_available(), \
                "EXL3_KVARN_TRITON=1 but the Triton path is unavailable " \
                "(needs triton + CUDA); unset it for the torch path."
            bk_t, bv_t = kvarn_triton_dequant_groups(
                self.records[Gs], self.layout, self.k_bits, self.v_bits,
                self.num_kv_heads, self.slices, do_wht=True)
            if kvarn_triton_parity_check():
                bk_r, bv_r = self._dequant_groups_batched_torch(Gs)
                # Per-128-slice reference (NOT the full head WHT): the
                # fused output has the per-slice FWHT done with the
                # cross-slice stage pending (applied by _refresh_into via
                # kvarn_wht_slices, same split as the torch path would
                # get from kvarn_wht_head = per-slice + cross-slice).
                sl = self.slices
                ref_k = kvarn_hadamard_128(
                    bk_r.reshape(-1, sl, KVAR_N_GROUP)).reshape_as(bk_r)
                ref_v = kvarn_hadamard_128(
                    bv_r.reshape(-1, sl, KVAR_N_GROUP)).reshape_as(bv_r)
                assert torch.equal(bk_t, ref_k) and \
                    torch.equal(bv_t, ref_v), \
                    "KVarN Triton fused dequant+WHT disagrees with the " \
                    f"torch reference on {Gs.numel()} groups"
            return bk_t, bv_t, True
        bk, bv = self._dequant_groups_batched_torch(Gs)
        return bk, bv, False

    @torch.inference_mode()
    def _dequant_groups_batched_torch(self, Gs: torch.Tensor):
        """Torch reference for _dequant_groups_batched (tested on CPU, runs
        anywhere): identical elementwise ops (unpack, (q*sc+zp)*other),
        one kernel launch each instead of ~30 per group."""
        C = self.num_kv_heads * self.slices
        assert self.records.shape[1] == C
        recs = self.records[Gs]  # (Gg, C, B) uint8
        Gg = recs.shape[0]
        kvh, sl = self.num_kv_heads, self.slices
        N = Gg * C * KVAR_N_GROUP * KVAR_N_GROUP
        f16 = _rec_f16(recs).float()  # (Gg, C, B//2)

        def deq_tiles(payload_off, payload_bytes, bits, sc_o, zp_o, ot_o, transpose):
            pay = recs[:, :, payload_off:payload_off + payload_bytes].reshape(-1)
            q = kvarn_unpack_bits(pay, N, bits).float() \
                .reshape(Gg, kvh, sl, KVAR_N_GROUP, KVAR_N_GROUP)
            sc = f16[:, :, sc_o // 2: sc_o // 2 + 128].reshape(Gg * C, 128)
            zp = f16[:, :, zp_o // 2: zp_o // 2 + 128].reshape(Gg * C, 128)
            ot = f16[:, :, ot_o // 2: ot_o // 2 + 128].reshape(Gg * C, 128)
            t = kvarn_dequantize_tile(q.reshape(Gg * C, 128, 128), sc, zp, ot) \
                .reshape(Gg, kvh, sl, 128, 128)
            if transpose:  # K records are [dim, token]; V are [token, dim]
                t = t.permute(0, 4, 1, 2, 3)
            else:
                t = t.permute(0, 3, 1, 2, 4)
            return t.reshape(Gg, KVAR_N_GROUP, kvh, self.head_dim)

        L = self.layout
        bk = deq_tiles(L.k_payload_off, L.k_payload_bytes, self.k_bits,
                       L.k_s_col_off, L.k_zp_off, L.k_s_row_off, True)
        bv = deq_tiles(L.v_payload_off, L.v_payload_bytes, self.v_bits,
                       L.v_s_row_off, L.v_zp_off, L.v_s_col_off, False)
        return bk, bv

    @torch.inference_mode()
    def _refresh_groups_legacy(self, Gs: torch.Tensor,
                               k: torch.Tensor, v: torch.Tensor):
        """Full rematerialization of Gs into caller temps (huge-context
        path: no persistent image). Same math as _refresh_groups."""
        self._refresh_into(Gs, k, v)

    @torch.inference_mode()
    def _refresh_into(self, Gs: torch.Tensor,
                      k_tgt: torch.Tensor, v_tgt: torch.Tensor):
        """(Re)materialize groups Gs (1D long, on device) into the paged
        fp16 temps k_tgt/v_tgt (shaped (pages, 256, kvh, hd)).

        Sealed groups come from the batched dequant (torch, or the fused
        Triton kernel when opted in); open groups from stacked staging
        (torch batched path, or the fused serve kernel when opted in:
        gather + full head WHT + scatter, disjoint from the sealed set).
        Rows needing the torch inverse-WHT share one batched call; rows
        the kernel already 128-WHT'd get only the cross-slice stage.
        Per-row math is identical in all combinations."""
        kvh, hd = self.num_kv_heads, self.head_dim
        dev = self.device
        flat_k = k_tgt.reshape(-1, KVAR_N_GROUP, kvh, hd)
        flat_v = v_tgt.reshape(-1, KVAR_N_GROUP, kvh, hd)
        sealed_m = self.sealed[Gs]
        Gs_s = Gs[sealed_m]
        rot_k, rot_v, rot_gs = [], [], []
        if Gs_s.numel():
            bk, bv, wht_done = self._dequant_groups_batched(Gs_s)
            if wht_done:
                # Fused kernel output: 128-FWHT done, cross-slice remains.
                flat_k[Gs_s] = kvarn_wht_slices(bk, hd).half()
                flat_v[Gs_s] = kvarn_wht_slices(bv, hd).half()
            else:
                rot_k.append(bk)
                rot_v.append(bv)
                rot_gs.append(Gs_s)
        open_m = ~sealed_m
        # Emptiness guard on COUNT (shape only, zero syncs) -- not
        # bool(any()) (a CPU sync per layer per call). Appending an
        # empty staging read would make `if rot_k:` truthy below and
        # crash the all-sealed refresh inside the WHT reshape.
        Gs_o = Gs[open_m]
        if Gs_o.numel():
            if _kvarn_use_triton():
                from ..modules.attention_fn.kvarn_triton import (
                    kvarn_triton_available, kvarn_triton_serve_open)
                assert kvarn_triton_available(), \
                    "EXL3_KVARN_TRITON=1 but the Triton path is unavailable " \
                    "(needs triton + CUDA); unset it for the torch path."
                # Fused open serve (4 launches, zero syncs) replaces the
                # rot-path gather + batched WHT + scatter below. Sealed
                # members of Gs_o, if any, are skipped in-kernel (the
                # sealed branch above owns them; sets are disjoint).
                kvarn_triton_serve_open(k_tgt, v_tgt, self, Gs_o)
            else:
                # Static staging reads zeros for never-written groups:
                # no tolist loop, no per-group syncs.
                rot_k.append(self.stage_k[Gs_o].float())
                rot_v.append(self.stage_v[Gs_o].float())
                rot_gs.append(Gs_o)
        if rot_k:
            mat_k = kvarn_wht_head(torch.cat(rot_k), hd).half()
            mat_v = kvarn_wht_head(torch.cat(rot_v), hd).half()
            rGs = torch.cat(rot_gs)
            flat_k[rGs] = mat_k
            flat_v[rGs] = mat_v

    @torch.inference_mode()
    def _refresh_groups(self, Gs: torch.Tensor):
        """(Re)materialize groups Gs (1D long, on device) into the
        persistent pre-overlay image. Same math as the legacy full
        rematerialization, restricted to Gs: batched sealed dequant plus
        stacked open staging, one inverse-WHT, indexed scatter."""
        self._refresh_into(Gs, self._img_k, self._img_v)

    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor,
               sliding_window: int = -1) -> tuple:
        # Dense and QSA layers pass -1 here (SWA/GDN layers are recurrent
        # and never kvarn-cached). A window, if ever passed, is applied by
        # the attention kernel itself; the layer's own tail/ring were
        # capped at the learned swa_window at construction. Sink/tail
        # exactness is applied here as an overlay, so the temps form one
        # merged image for the single downstream softmax.
        #
        # Idea 2: the image persists across forwards; only groups dirtied
        # since the last call are rematerialized (decode: ~1 group). The
        # overlay is per-call (depends on seqlens). The Triton path serves
        # the persistent image directly (in-place overlay + in-kernel
        # stash, restored after the forward, no clones); the torch
        # fallback still serves a clone (the overlay mutates its target).
        # Modest allocations only
        # (num_pages <= 160); huge contexts keep the memory-slim
        # full rematerialization below.
        kvh, hd = self.num_kv_heads, self.head_dim
        dev = self.device
        gps = PAGE_SIZE // KVAR_N_GROUP
        if self._img_ok:
            if self._img_k is None:
                self._img_k = torch.zeros(
                    (self.num_pages, PAGE_SIZE, kvh, hd),
                    dtype=torch.half, device=dev)
                self._img_v = torch.zeros_like(self._img_k)
                # First build: every appended row predates the image (and
                # fused stores before the image existed wrote staging
                # only), so the whole image is stale by construction.
                # Mark all groups dirty: the sweep below materializes
                # everything once, then write-through keeps it current.
                self._dirty_mask.fill_(True)
                self._dirty_any = True
            # A previous serve-from-image overlay may still be pending
            # (twin-test order is update_direct -> get_kv without an
            # update_kv in between; production always has update_kv, which
            # also restores). Restore first: idempotent, gated, zero syncs
            # when nothing is pending.
            if bool(getattr(self, "_ov_pending", False)):
                from ..modules.attention_fn.kvarn_triton import (
                    kvarn_triton_unoverlay)
                kvarn_triton_unoverlay(self._img_k, self._img_v, self)
            # Global dirty sweep, no resident-pages restriction: refresh
            # recomputes image rows from records/staging (the truth), so
            # it is idempotent -- a not-currently-resident dirty group
            # refreshes to the same values it would get when it turns
            # resident, and any later content change re-dirties before
            # serving. The Python-side flag skips the nonzero sync when
            # nothing dirtied since the last sweep (steady decode: the
            # fused store keeps the image current via write-through, so
            # this is almost every step).
            if self._dirty_any:
                dirty = self._dirty_mask.nonzero().flatten()
                if dirty.numel():
                    self._refresh_groups(dirty)
                    self._dirty_mask[dirty] = False
                self._dirty_any = False
            if _kvarn_use_triton() and not self.is_swa and \
                    cache_seqlens.numel() == 1:
                # Serve-from-image: overlay lands in place on the
                # persistent image (2 full-image clones + 2 casts saved
                # per layer per step). Overwritten rows are stashed
                # in-kernel and put back by kvarn_triton_unoverlay in
                # update_kv after the forward -- served rows are
                # bit-identical to overlay-on-clone (the overlay region
                # is disjoint from the append rows update_kv reads back,
                # and refresh-from-records backstops every dirty group).
                from ..modules.attention_fn.kvarn_triton import (
                    kvarn_triton_available, kvarn_triton_overlay,
                    _kvarn_overlay_stash)
                assert kvarn_triton_available(), \
                    "EXL3_KVARN_TRITON=1 but the Triton path is unavailable " \
                    "(needs triton + CUDA); unset it for the torch path."
                maxw = KVAR_N_SINK_TOKENS + int(self.tail_effective)
                stash = _kvarn_overlay_stash(self, maxw, dev)
                kvarn_triton_overlay(self._img_k, self._img_v, self,
                                     cache_seqlens,
                                     block_table[0].to(dtype=torch.int32,
                                                       device=dev),
                                     gps, KVAR_N_SINK_TOKENS,
                                     self.tail_effective,
                                     stash=stash)
                return self._img_k, self._img_v
            # Torch fallback (SWA / multi-row / TRITON=0): the overlay
            # mutates its target, so it still lands on throwaway clones
            # (bandwidth-cheap; the loop was the cost).
            bt = block_table.long()
            k = self._img_k.clone()
            v = self._img_v.clone()
            self._apply_exact_overlay(k, v, cache_seqlens, block_table)
            return k, v
        else:
            bt = block_table.long()
            if bt.shape[0] == 1:
                # Steady decode: one row, pages distinct by construction, so
                # a range mask replaces the sort (same set, same order, no
                # sync, ~5 launches saved). Multi-row batches keep unique
                # (rows may alias pages via prompt-cache sharing).
                row = bt[0]
                pages = row[(row >= 0) & (row < self.num_pages)].to(dev)
            else:
                pages = torch.unique(bt).to(dev)
                pages = pages[(pages >= 0) & (pages < self.num_pages)]
            k = torch.zeros((self.num_pages, PAGE_SIZE, kvh, hd),
                            dtype=torch.half, device=dev)
            v = torch.zeros_like(k)
            if pages.numel():
                Gs = self._page_groups[pages].reshape(-1)
                Gs = Gs[Gs < self.num_groups]
                if Gs.numel():
                    self._refresh_groups_legacy(Gs, k, v)
        # Huge-context legacy path: k/v are fresh temps owned by this
        # call, so both overlay variants mutate them directly (no clones
        # anywhere here, no dirty writeback: there is no image to keep).
        if _kvarn_use_triton() and not self.is_swa and cache_seqlens.numel() == 1:
            from ..modules.attention_fn.kvarn_triton import (
                kvarn_triton_available, kvarn_triton_overlay)
            assert kvarn_triton_available(), \
                "EXL3_KVARN_TRITON=1 but the Triton path is unavailable " \
                "(needs triton + CUDA); unset it for the torch path."
            kvarn_triton_overlay(k, v, self, cache_seqlens, bt[0],
                                 gps, KVAR_N_SINK_TOKENS, self.tail_effective)
        else:
            self._apply_exact_overlay(k, v, cache_seqlens, block_table)
        return k, v

    @override
    def update_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor, length: int):
        # k/v are the paged fp16 temps returned by get_kv (with the new rows
        # merged in by the attention fallback); persist rows
        # [seqlens, seqlens+length) per batch entry.
        # Restore any pending serve-from-image overlay first (even when
        # length == 0): the forward has consumed it, and the next get_kv
        # must see the pre-overlay image. Append rows are disjoint from
        # the overlay window, so restore-before-store is safe.
        if bool(getattr(self, "_ov_pending", False)):
            from ..modules.attention_fn.kvarn_triton import (
                kvarn_triton_unoverlay)
            kvarn_triton_unoverlay(self._img_k, self._img_v, self)
        if length == 0:
            return
        bsz = cache_seqlens.numel()
        bt = block_table.long()
        seqlens = cache_seqlens.long()
        self._touch_batch(seqlens, bt, length)
        for b in range(bsz):
            # Length-1 (every decode step) is a slice, not an alloc+add.
            pos = seqlens[b:b + 1] if length == 1 else \
                seqlens[b] + torch.arange(length, device=bt.device)
            pages = bt[b, pos // PAGE_SIZE]
            offs = pos % PAGE_SIZE
            self._store_rows(k[pages, offs], v[pages, offs], pages, offs,
                             pos, seqlens[b] + length)

    @override
    def update_kv_direct(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor,
                         k: torch.Tensor, v: torch.Tensor, length: int):
        # k/v: (bsz, length, kvh, hd) new contiguous rows at
        # positions cache_seqlens..+length.
        # Twin-test order can leave a served overlay pending across a
        # direct append (update_direct -> get_kv has no update_kv between);
        # get_kv entry already restores, this is the belt-and-braces copy
        # for direct-only flows. Gated no-op when nothing is pending.
        if bool(getattr(self, "_ov_pending", False)):
            from ..modules.attention_fn.kvarn_triton import (
                kvarn_triton_unoverlay)
            kvarn_triton_unoverlay(self._img_k, self._img_v, self)
        if length == 0:
            return
        bsz = cache_seqlens.numel()
        bt = block_table.long()
        seqlens = cache_seqlens.long()
        self._touch_batch(seqlens, bt, length)
        for b in range(bsz):
            # Length-1 (every decode step) is a slice, not an alloc+add.
            pos = seqlens[b:b + 1] if length == 1 else \
                seqlens[b] + torch.arange(length, device=bt.device)
            pages = bt[b, pos // PAGE_SIZE]
            offs = pos % PAGE_SIZE
            self._store_rows(k[b], v[b], pages, offs, pos, seqlens[b] + length)

    @override
    def copy_page(self, source: CacheLayer_kvarn, from_page: int, to_page: int,
                  num_tokens: int):
        assert self.records.shape == source.records.shape
        assert (self.k_bits, self.v_bits) == (source.k_bits, source.v_bits), \
            "KVarN copy_page requires matching K/V widths (records are " \
            "not comparable across presets)"
        assert (self.num_kv_heads, self.slices, self.head_dim) == \
            (source.num_kv_heads, source.slices, source.head_dim), \
            "KVarN copy_page requires matching geometry (records.shape " \
            "alone is ambiguous: e.g. kvh8/hd128 and kvh4/hd256 share " \
            "ncols and tile bytes but slice differently)"
        assert self.tail_effective == source.tail_effective and \
            self.tail_window == source.tail_window, \
            "KVarN copy_page requires matching tail policy (a different " \
            "effective tail would leave the destination without exact " \
            "rows the source was served)"
        assert self.swa_override == source.swa_override and \
            self.is_swa == source.is_swa, \
            "KVarN copy_page requires matching SWA group (is_swa) and " \
            "SWA override pair (per-group presets are not comparable)"
        assert self.tail_dtype == source.tail_dtype and \
            self.has_sink == source.has_sink, \
            "KVarN copy_page requires matching tail dtype and sink policy"
        gps = PAGE_SIZE // KVAR_N_GROUP
        for hh in range(gps):
            gf, gt = from_page * gps + hh, to_page * gps + hh
            lo, hi = hh * KVAR_N_GROUP, min((hh + 1) * KVAR_N_GROUP, num_tokens)
            if hi <= lo:
                continue
            nrows = hi - lo
            fb = int(source.group_base[gf])
            if fb < 0:
                # Source span never written: destination stays unwritten.
                self.group_base[gt] = -1
                self.present[gt] = False
                self.sealed[gt] = False
                self.stage_k[gt].zero_()
                self.stage_v[gt].zero_()
                self.exact_valid[gt] = False
                continue
            self.group_base[gt] = fb
            self.present[gt] = False
            self.present[gt, :nrows].copy_(source.present[gf, :nrows])
            full = nrows == KVAR_N_GROUP and bool(source.present[gf].all())
            sink_span = self.has_sink and fb == 0
            if full and bool(source.sealed[gf]) and not sink_span:
                # Sealed flags travel with the records (the logical sink
                # group on sink layers is never sealed, stays exact).
                self.records[gt].copy_(source.records[gf], non_blocking=True)
                self.sealed[gt] = True
                self.stage_k[gt].zero_()
                self.stage_v[gt].zero_()
                # Exact blocks travel too: the destination sequence's tail
                # may cover these positions. (Sealed branch is always
                # full-group, so whole-block copies match the old code.)
                if bool(source.exact_valid[gf]):
                    self._alloc_exact_block(gt)
                    self.exact_k[gt].copy_(source.exact_k[gf],
                                           non_blocking=True)
                    self.exact_v[gt].copy_(source.exact_v[gf],
                                           non_blocking=True)
                else:
                    self.exact_valid[gt] = False
            else:
                self.sealed[gt] = False
                # Staging travels (or is rebuilt from the sealed records
                # for partial copies out of sealed groups); exact blocks
                # travel when present.
                if not bool(source.sealed[gf]) and \
                        bool(source.present[gf].any()):
                    self.stage_k[gt, :nrows] \
                        .copy_(source.stage_k[gf, :nrows], non_blocking=True)
                    self.stage_v[gt, :nrows] \
                        .copy_(source.stage_v[gf, :nrows], non_blocking=True)
                    if nrows < KVAR_N_GROUP:
                        self.stage_k[gt, nrows:].zero_()
                        self.stage_v[gt, nrows:].zero_()
                elif bool(source.sealed[gf]):
                    rec = self._staging_from_records(gf, source.records)
                    self.stage_k[gt, :nrows].copy_(rec[0][:nrows],
                                                   non_blocking=True)
                    self.stage_v[gt, :nrows].copy_(rec[1][:nrows],
                                                   non_blocking=True)
                    if nrows < KVAR_N_GROUP:
                        self.stage_k[gt, nrows:].zero_()
                        self.stage_v[gt, nrows:].zero_()
                else:
                    self.stage_k[gt].zero_()
                    self.stage_v[gt].zero_()
                if bool(source.exact_valid[gf]):
                    self._alloc_exact_block(gt)
                    self.exact_k[gt, :nrows] \
                        .copy_(source.exact_k[gf, :nrows], non_blocking=True)
                    self.exact_v[gt, :nrows] \
                        .copy_(source.exact_v[gf, :nrows], non_blocking=True)
                    if nrows < KVAR_N_GROUP:
                        self.exact_k[gt, nrows:].zero_()
                        self.exact_v[gt, nrows:].zero_()
                else:
                    self.exact_valid[gt] = False
        # Destination content changed in every branch above (copied,
        # rebuilt or reset): refresh it on next materialization (idea 2).
        self._dirty_mask[to_page * gps: to_page * gps + gps] = True
        self._dirty_any = True
        # The shared content aliases one logical prefix now: the source
        # page is known-shared and the destination page is a new alias, so
        # both skip eviction until one of them is rewritten (which unpins
        # it via the base-change path).
        if num_tokens > 0:
            source.page_pinned[from_page] = True
            self.page_pinned[to_page] = True

    @override
    def get_tensors(self):
        # M5 decision (documented fallback, forces fp16 elsewhere): group
        # records and compact blocks are not page-major, so the CPU
        # second-tier page cache (page-major CUDA slices per layer
        # tensor) and TP loader composition beyond tp_export cannot
        # consume KVarN layers. Those paths must fall back to fp16 (or
        # stay out of scope); the GPU-tier prompt-cache path (copy_page)
        # is fully supported and version-checked.
        out = [self.records]
        for g in self._live_stage_groups():
            out += [self.stage_k[g], self.stage_v[g]]
        for g in self._live_exact_groups():
            out += [self.exact_k[g], self.exact_v[g]]
        return out

    def _resident_bytes(self, blocks: dict) -> int:
        total = 0
        for blk in blocks.values():
            for t in blk:
                total += int(t.numel()) * t.element_size()
        return total

    @override
    def storage_size(self):
        # Compressed records plus the compact exact (N+R) history: the
        # persistent footprint. Resident exact is bounded by the sink +
        # tail window, so this lands far below fp16.
        n_ex = int(self.exact_valid.sum())
        blk = KVAR_N_GROUP * self.num_kv_heads * self.head_dim * \
            self.tail_dtype.itemsize * 2
        return int(self.records.numel()) * torch.uint8.itemsize + n_ex * blk

    @override
    def overhead_size(self):
        # Transient workspace: single-open-group staging plus bit flags
        # and the per-group/per-page compact metadata. Staging is a
        # static dense tensor (mostly zeros); only live groups
        # (unsealed with present rows, i.e. the old dict keys) count.
        n_live = int((~self.sealed & self.present.any(dim=1)).sum())
        blk = KVAR_N_GROUP * self.num_kv_heads * self.head_dim * \
            torch.half.itemsize * 2
        return n_live * blk + \
            int(self.present.numel()) + int(self.sealed.numel()) + \
            int(self.group_base.numel()) * torch.int64.itemsize + \
            int(self.page_owner_n.numel()) * torch.int64.itemsize + \
            int(self.page_pinned.numel())

    @override
    def tp_export(self, plan):
        # M5: version tag + main/SWA pairs. The main pair plus the
        # override reconstruct the identical per-group preset on import
        # (is_swa re-derives from the attention module there); a stale
        # version fails closed in the constructor.
        swa_k, swa_v = self.swa_override if self.swa_override else (None, None)
        return {
            "cls": CacheLayer_kvarn,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.main_k_bits,
                "v_bits": self.main_v_bits,
                "swa_k_bits": swa_k,
                "swa_v_bits": swa_v,
                "tail_tokens": self.tail_requested_raw,
                "tail_type": self.tail_type_name,
                "is_swa": self.is_swa,
                "sinkhorn_iters": self.sinkhorn_iters,
                "kvarn_version": KVAR_N_STATE_VERSION,
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
        swa_k_bits: int | str | None = None,
        swa_v_bits: int | str | None = None,
        kvarn_version: int = KVAR_N_STATE_VERSION,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens,
                         k_bits, v_bits, sinkhorn_iters,
                         tail_tokens, tail_type, is_swa,
                         swa_k_bits, swa_v_bits, kvarn_version)
        self._init_planes(attention, max_num_tokens)

    @override
    def tp_export(self, plan):
        swa_k, swa_v = self.swa_override if self.swa_override else (None, None)
        return {
            "cls": CacheLayer_kvarn_qsa,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.main_k_bits,
                "v_bits": self.main_v_bits,
                "swa_k_bits": swa_k,
                "swa_v_bits": swa_v,
                "tail_tokens": self.tail_requested_raw,
                "tail_type": self.tail_type_name,
                "is_swa": self.is_swa,
                "sinkhorn_iters": self.sinkhorn_iters,
                "kvarn_version": KVAR_N_STATE_VERSION,
            }
        }
