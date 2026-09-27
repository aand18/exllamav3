import torch
from ...cache import CacheLayer, Cache, CacheLayer_quant, CacheLayer_kvarn
from .common import AttnArgs, AttnFn
from .bighead_scalar import fn_bighead_scalar_attn
from .torch import (
    fn_torch_sdpa_fallback_cache,
    fn_torch_sdpa_fallback_nocache,
    fn_torch_sdpa_paged_cpu_cache,
)
from .xformers import fn_xformers_cutlass_fallback_cache, fn_xformers_cutlass_fallback_nocache
try:
    from .triton_paged import (
        _qc_staging,
        fn_triton_paged_attn,
        fn_triton_paged_attn_longq,
        fn_triton_paged_attn_decode,
        fn_triton_paged_attn_prefill,
        fn_triton_varlen_attn,
        fn_triton_paged_attn_decode_qc,
        fn_triton_paged_attn_prefill_qc,
        fn_triton_attn_nocache,
    )
    _have_triton_paged = True
except ImportError:
    # CPU-only hosts without triton: fall through to the torch/xformers fallbacks
    _have_triton_paged = False

# Candidate attn functions in order of preference: the Triton decode/prefill/varlen kernels
# serve every shape they support (any head_dim <= 512, zero-padded to a power of two), then the
# older Triton kernels and the torch/xformers fallbacks for the rest
_fns_triton_fast: list[AttnFn] = [
    fn_triton_paged_attn_decode,
    fn_triton_paged_attn_prefill,
    fn_triton_varlen_attn,
] if _have_triton_paged else []

# Quant-direct calls carry the packed cache in q_cache and leave k_cache/v_cache as None, which makes them
# indistinguishable from cache-less attention to any backend that only checks has_kv_cache(). Such a backend
# would silently attend over just the new K/V rows and ignore the cached context, so quant-direct calls only
# ever dispatch over the qc-aware functions
_fns_qc: list[AttnFn] = [
    fn_triton_paged_attn_decode_qc,
    fn_triton_paged_attn_prefill_qc,
] if _have_triton_paged else []

# Quantized caches feed the attention kernels directly (online dequant or prefill staging by
# EXL3_QC_STAGING level, see triton_paged); level 2 restores the dequantize-then-attend path
# with full-size fp16 temporaries for A/B testing
_qc_attn = (_qc_staging < 2) if _have_triton_paged else False

_triton_fallbacks: list[AttnFn] = [
    fn_triton_attn_nocache,
    fn_triton_paged_attn,
    fn_triton_paged_attn_longq,
] if _have_triton_paged else []

attn_fns: list[AttnFn] = _fns_triton_fast + _triton_fallbacks + [
    fn_bighead_scalar_attn,
    fn_xformers_cutlass_fallback_cache,
    fn_xformers_cutlass_fallback_nocache,
    fn_torch_sdpa_paged_cpu_cache,
    fn_torch_sdpa_fallback_cache,
    fn_torch_sdpa_fallback_nocache
]

# Every Triton entry point is CUDA-only: _check_tensor raises ValueError on
# CPU tensors instead of declining with None, which breaks the dispatch
# contract ("candidate functions return None on incompatible arguments").
# triton-windows is a mandatory Windows dependency, so without this guard
# every CPU-tensor dispatch on a GPU-less box crashes instead of reaching
# the torch/xformers fallbacks. GPU behavior is untouched (q is CUDA).
_fns_triton_all = frozenset(_fns_triton_fast + _triton_fallbacks + _fns_qc)

def _tensor_desc(t: torch.Tensor | None) -> str:
    if t is None:
        return "None"
    return f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} contiguous={t.is_contiguous()}"


def _print_no_attn_match_report(args: AttnArgs):
    tried = ", ".join(fn.__name__ for fn in attn_fns)
    print(
        "No matching attention function found.\n"
        f"  shape: bsz={args.bsz} q_len={args.q_len} kv_len={args.kv_len} "
        f"q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} dim={args.dim}\n"
        f"  flags: cache={args.has_kv_cache()} varlen={args.is_varlen()} gqa={args.is_gqa()} "
        f"causal={args.causal} window_size={args.window_size} softcap={args.softcap} "
        f"non_causal_spans={args.non_causal_spans is not None} sinks={args.sinks is not None}\n"
        f"  scale: sm_scale={args.sm_scale} max_seqlen={args.max_seqlen}\n"
        f"  q: {_tensor_desc(args.q)}\n"
        f"  k: {_tensor_desc(args.k)}\n"
        f"  v: {_tensor_desc(args.v)}\n"
        f"  k_cache: {_tensor_desc(args.k_cache)}\n"
        f"  v_cache: {_tensor_desc(args.v_cache)}\n"
        f"  block_table: {_tensor_desc(args.block_table)}\n"
        f"  cache_seqlens: {_tensor_desc(args.cache_seqlens)}\n"
        f"  cu_seqlens: {_tensor_desc(args.cu_seqlens)}\n"
        f"  tried: {tried}"
    )


def _try_kvarn_online_decode(q, k, v, cache, cache_idx, cache_instance,
                             block_table, cache_seqlens, q_len, sm_scale,
                             causal, window_size, softcap, sinks,
                             non_causal_spans, cu_seqlens):
    """Imageless KVarN decode arm (match-bee Task 3): serves single-token
    decode attention online from records (fused kernels) + exact/staging
    tail block + global merge, with NO persistent image. Returns fp16
    (1, 1, qh, hd) or None (fail-closed to the get_kv path).

    Stores new rows first (mirrors the qc path, so the post-attention
    write-back is already done and this returns directly). v1 is
    torch-heavy in tail/merge (validated patterns); fused-merge follows
    once timed. Loud paths only: every gate declines with None, never
    half-runs.
    """
    import os
    if os.environ.get("EXL3_KVARN_IMAGELESS", "0") != "1":
        return None
    if os.environ.get("EXL3_KVARN_TRITON", "0") != "1":
        return None
    if q_len != 1 or q.shape[0] != 1:
        return None
    if cu_seqlens is not None or non_causal_spans:
        return None
    if sinks is not None or (softcap or 0.0) != 0.0:
        return None
    if window_size not in (None, -1):
        return None
    if q.device.type != "cuda" or q.dtype != torch.float16:
        return None
    if k.dtype != torch.float16 or v.dtype != torch.float16:
        return None
    layer = cache if isinstance(cache, CacheLayer) else \
        cache.layers[cache_idx, cache_instance or 0]
    if not isinstance(layer, CacheLayer_kvarn) or layer.is_swa:
        return None
    bsz, _, qh, dim = q.shape
    kvh, hd = int(layer.num_kv_heads), int(layer.head_dim)
    if dim != hd or qh % kvh != 0:
        return None
    if dim not in (128, 256, 512) or dim % 128 != 0:
        return None
    if k.shape[2] != kvh or v.shape[2] != kvh:
        return None
    if k.shape[3] != hd or v.shape[3] != hd:
        return None
    import importlib.util
    if not torch.cuda.is_available() or \
            importlib.util.find_spec("triton") is None:
        return None
    from .kvarn_triton import (
        kvarn_triton_available, kvarn_triton_qwht,
        kvarn_triton_online_partials, kvarn_triton_wht_rows,
        _kvarn_online_buffers)
    from ...cache.kvarn import KVAR_N_SINK_TOKENS
    if not kvarn_triton_available():
        return None
    qpk = qh // kvh
    sl = hd // 128
    scale = sm_scale if sm_scale is not None else dim ** (-0.5)
    dev = q.device
    # Store first (write-back already done on this path).
    layer.update_kv_direct(cache_seqlens, block_table, k, v, q_len)
    n = int(cache_seqlens[0]) + q_len
    n_0d = cache_seqlens[:1] + q_len
    sink_n = KVAR_N_SINK_TOKENS if layer.has_sink else 0
    tail_eff = int(layer.tail_effective)
    Q = q[0, 0]
    # QWHT (persistent buffers) + body partials (persistent buffers).
    qpad = 1 << (qpk - 1).bit_length()
    m_b, l_b, acc_b, qw, qs, _o = _kvarn_online_buffers(
        layer, qh, qpad, hd, dev)
    sscale = 1.0 if sl == 1 else (0.7071067811865475 if sl == 2 else 0.5)
    kvarn_triton_qwht(Q, qs, qw, sl, sscale)
    ids = layer.sealed.nonzero().flatten()
    m, l, acc = kvarn_triton_online_partials(
        layer, qw, ids, qpk, scale, n_0d, sink_n, tail_eff)
    m, l, acc = m[:, :qpk], l[:, :qpk], acc[:, :qpk]
    with torch.inference_mode():
        m_all = m.amax(dim=2, keepdim=True)
        e = torch.exp(m - m_all)
        body_den = (l * e).sum(dim=2).reshape(qh)
        body_num = (acc * e.unsqueeze(-1)).sum(dim=2).reshape(qh, hd)
        body_m = m_all.reshape(qh)
        # Tail block (exact-first + staging fallback, original domain).
        Kt, Vt = layer.kvarn_online_tail(n, block_table[0])
        t_ms, t_ns, t_ds = [], [], []
        for h in range(kvh):
            qh_ = Q[h * qpk:(h + 1) * qpk].float()
            st = (qh_ @ Kt[:, h, :].T) * scale
            tm = st.amax(dim=-1)
            pe = torch.exp(st - tm.unsqueeze(-1))
            t_ms.append(tm)
            t_ns.append(pe @ Vt[:, h, :])
            t_ds.append(pe.sum(dim=-1))
        tail_m = torch.cat(t_ms)
        tail_num = torch.cat(t_ns)
        tail_den = torch.cat(t_ds)
        # Merge in WHT domain, single final WHT.
        tail_num_w = kvarn_triton_wht_rows(tail_num, hd)
        m_g = torch.maximum(body_m, tail_m)
        eb = torch.exp(body_m - m_g)
        et = torch.exp(tail_m - m_g)
        den = body_den * eb + tail_den * et
        num = body_num * eb.unsqueeze(-1) + tail_num_w * et.unsqueeze(-1)
        out_w = num / den.unsqueeze(-1)
        out = kvarn_triton_wht_rows(out_w, hd).half()
    return out.reshape(bsz, q_len, qh, hd)


def attn_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: CacheLayer | Cache | None = None,
    cache_idx: int | None = None,
    cache_instance: int | None = None,
    causal: bool = True,
    sm_scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    window_size: int | None = None,
    softcap: float = 0.0,
    block_table: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    non_causal_spans: list | None = None,
    sinks: torch.Tensor | None = None,
    dispatch_cache: dict | None = None,
    max_kv_len: int | None = None,
):
    """
    Select and run the first compatible attention implementation for the supplied tensors.

    The dispatcher builds an AttnArgs description covering regular, varlen and paged-cache attention modes, obtains
    K/V cache tensors when a Cache or CacheLayer is provided, and tries registered attention backends in preference
    order. After a cached attention call, any updated K/V tensors are written back through the same cache interface.
    """
    bsz, q_len, num_q_heads, dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    # Get cache tensors. Quantized layers pass their packed tensors straight to the attention
    # kernels when possible: new K/V are quantized into the cache up front and never
    # materialized as full fp16 cache-sized temporaries
    q_cache = None
    if cache is not None:
        assert block_table is not None
        assert cache_seqlens is not None
        layer = cache if isinstance(cache, CacheLayer) else cache.layers[cache_idx, cache_instance or 0]
        # Imageless KVarN decode arm (match-bee): stores rows first and
        # serves online with no image; returns directly (write-back
        # already done). Declines (None) unless every gate holds.
        kvarn_o = _try_kvarn_online_decode(
            q, k, v, cache, cache_idx, cache_instance, block_table,
            cache_seqlens, q_len, sm_scale, causal, window_size, softcap,
            sinks, non_causal_spans, cu_seqlens)
        if kvarn_o is not None:
            return kvarn_o
        if (
            _qc_attn and
            isinstance(layer, CacheLayer_quant) and
            not isinstance(layer, CacheLayer_kvarn) and  # KVarN always takes the
                                                          # dequant path (q_cache=None):
                                                          # get_kv serves one merged
                                                          # image (sealed body +
                                                          # exact sink/tail overlay)
                                                          # for single-softmax SDPA;
                                                          # online kernels are later work
            layer.compand_a == 0.0 and
            q.dtype == torch.float16 and
            dim <= 512 and dim % 32 == 0 and   # packed groups of 32; non-pow2 dims run zero-padded
            cu_seqlens is None
        ):
            layer.update_kv_direct(cache_seqlens, block_table, k, v, q_len)
            q_cache = layer.get_qkv()
            k_cache, v_cache = None, None
        else:
            k_cache, v_cache = layer.get_kv(cache_seqlens, block_table, window_size if window_size is not None else -1)
    else:
        k_cache, v_cache = None, None

    # Defaults
    if sm_scale is None:
        sm_scale = dim ** (-0.5)

    # Dispatch
    args = AttnArgs(
        bsz, q_len, num_q_heads, dim,
        kv_len, num_kv_heads,
        q, k, v,
        k_cache, v_cache,
        causal,
        sm_scale,
        cu_seqlens, max_seqlen,
        window_size,
        softcap,
        block_table, cache_seqlens,
        non_causal_spans,
        q_cache,
        sinks,
        max_kv_len = max_kv_len,
    )
    # Quant-direct calls select among the qc-aware backends only; a separate hint slot keeps a function that
    # won a cache-less or fp16-cache call from being retried on quant-direct arguments (it cannot see q_cache
    # and would accept them as cache-less)
    candidates = _fns_qc if q_cache is not None else attn_fns
    hint_key = "fn_qc" if q_cache is not None else "fn"
    if q.device.type == "cpu":
        # See _fns_triton_all: Triton entries raise (not decline) on CPU
        # tensors, so they must be excluded before the scan, and a stale
        # hint at one of them must not be retried either.
        candidates = [fn for fn in candidates if fn not in _fns_triton_all]

    # Retry the backend that matched last time for this caller before scanning the full list.
    # Candidate functions return None on incompatible arguments, so a stale hint self-corrects
    fn = dispatch_cache.get(hint_key) if dispatch_cache is not None else None
    if fn is not None and fn not in candidates:
        fn = None
    o = fn(args) if fn is not None else None

    if o is None:
        args.sanity_check()
        for fn in candidates:
            o = fn(args)
            if o is not None:
                break
        else:
            _print_no_attn_match_report(args)
            raise ValueError("No matching attention function")
        if dispatch_cache is not None:
            dispatch_cache[hint_key] = fn

    # Update cache (quant-direct mode already wrote the new K/V before the attention call)
    if cache is not None and q_cache is None:
        if isinstance(cache, CacheLayer):
            cache.update_kv(cache_seqlens, block_table, k_cache, v_cache, q_len)
        elif isinstance(cache, Cache):
            cache.update_layer(cache_idx, cache_seqlens, block_table, k_cache, v_cache, q_len, cache_instance)

    return o
