import os
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


def _kvarn_check_flag(layer) -> bool:
    # Sticky-flag tripwire cadence: a real trip fires every serve, so
    # checking every 128 serves per layer still catches systematic
    # trips within any KLD gate (256+ steps); PARITY=1 tests check
    # every call. Skipping the int(flag) read saves a DtoH sync per
    # layer per step (64/step); zero math change (flag is a tripwire,
    # not data). Plain-Python counter, no syncs.
    if os.environ.get("EXL3_KVARN_TRITON_PARITY", "0") == "1":
        return True
    n = getattr(layer, "_serve_flag_tick", 0) + 1
    layer._serve_flag_tick = n
    return (n % 128) == 0


# Tail-position memo: tpos depends only on (dev, n, tail_eff, sink)
# (identical for all same-type layers in a step); building it once
# saves 64 x (2 aranges + cat + long) per step. Values are read-only
# downstream (gather indices + mask). Bounded (cleared past 8 keys).
_tpos_memo: dict = {}


def _kvarn_tpos(dev, n: int, tail_eff: int, sn_: int):
    key = (str(dev), n, tail_eff, sn_)
    tpos = _tpos_memo.get(key)
    if tpos is None:
        tpos = torch.cat([torch.arange(sn_, device=dev),
                          torch.arange(max(0, n - tail_eff), n,
                                       device=dev)]).long()
        if len(_tpos_memo) > 8:
            _tpos_memo.clear()
        _tpos_memo[key] = tpos
    return tpos


# Lazy-once process constants for the decode arm: find_spec +
# cuda-availability + submodule imports cost ~0.5-1ms/step when paid
# per layer per step (16x). Values cannot change at runtime (module
# objects, spec probes), so caching is exact. Env-derived flags
# (IMAGELESS/TRITON/PARITY) are still read live per call (tests flip
# them). First call pays what today pays every call.
_kvarn_arm = None
_g_calls = 0


def _kvarn_arm_load():
    global _kvarn_arm
    if _kvarn_arm is None:
        import importlib.util
        ok = torch.cuda.is_available()
        ok = ok and importlib.util.find_spec("triton") is not None
        if not ok:
            _kvarn_arm = False
        else:
            from .kvarn_triton import (
                kvarn_triton_available, kvarn_triton_qwht,
                kvarn_triton_online_partials, kvarn_triton_wht_rows,
                kvarn_triton_online_serve, kvarn_triton_online_merge,
                kvarn_triton_online_tail_reduce, _kvarn_online_buffers,
                _kvarn_serve_groups, kvarn_triton_online_tail_gather)
            from ...cache.kvarn import (KVAR_N_SINK_TOKENS, KVAR_N_GROUP,
                _ptime_count)
            from ...constants import PAGE_SIZE
            from types import SimpleNamespace
            _kvarn_arm = SimpleNamespace(
                t_avail=kvarn_triton_available, t_qwht=kvarn_triton_qwht,
                t_partials=kvarn_triton_online_partials,
                t_wht_rows=kvarn_triton_wht_rows,
                t_serve=kvarn_triton_online_serve,
                t_merge=kvarn_triton_online_merge,
                t_tailred=kvarn_triton_online_tail_reduce,
                t_bufs=_kvarn_online_buffers, t_groups=_kvarn_serve_groups,
                t_tailgather=kvarn_triton_online_tail_gather,
                c_sink=KVAR_N_SINK_TOKENS, c_group=KVAR_N_GROUP,
                c_ptime=_ptime_count,
                c_page=PAGE_SIZE)
    return _kvarn_arm


# Graphs v2: per-layer decode sublattice replay (Task 3).
# Store (ungraphed, status-gated) -> static input copies -> bucket
# (gc, R) lookup -> replay, else eager-rest + capture-after. Kill
# switch EXL3_KVARN_GRAPH=1 (default off). PARITY=1 runs the eref
# assert but still graphs (validates both). Fallback (status 1/2,
# cert fail, flag-due, bucket miss/bad, shape change) runs eager.
def _graph_bufs(layer, dev, qh, kvh, hd, qpk, maxw, npages, sledtype,
                btdtype):
    """Ensure the static input buffers for graph capture/replay.
    Idempotent (allocates once, reallocs only on shape/dtype
    change). All addresses stable afterwards, which is what makes
    replay valid. Returns (Qbuf, Qfbuf, nbuf, tbuf, btbuf)."""
    Qbuf = getattr(layer, "_ov_dec_g_q", None)
    if Qbuf is None or tuple(Qbuf.shape) != (qh, hd):
        Qbuf = torch.empty((qh, hd), dtype=torch.float16, device=dev)
        layer._ov_dec_g_q = Qbuf
    Qfbuf = getattr(layer, "_ov_dec_qf", None)
    if Qfbuf is None or tuple(Qfbuf.shape) != (qh, hd):
        Qfbuf = torch.empty((qh, hd), dtype=torch.float32, device=dev)
        layer._ov_dec_qf = Qfbuf
    nbuf = getattr(layer, "_ov_dec_g_n", None)
    if nbuf is None or tuple(nbuf.shape) != (1,) or nbuf.dtype != sledtype:
        nbuf = torch.empty((1,), dtype=sledtype, device=dev)
        layer._ov_dec_g_n = nbuf
    tbuf = getattr(layer, "_ov_dec_g_t", None)
    if tbuf is None or tuple(tbuf.shape) != (maxw,):
        tbuf = torch.empty((maxw,), dtype=torch.int64, device=dev)
        layer._ov_dec_g_t = tbuf
    btbuf = getattr(layer, "_ov_dec_g_b", None)
    if btbuf is None or tuple(btbuf.shape) != (npages,) or btbuf.dtype != btdtype:
        btbuf = torch.empty((npages,), dtype=btdtype, device=dev)
        layer._ov_dec_g_b = btbuf
    return Qbuf, Qfbuf, nbuf, tbuf, btbuf


def _graph_tail_bufs(layer, dev, kvh, hd, maxw):
    """Ensure tail temps + ev/g/s (K/V reuse the method temp attr
    names so eager-fallback and graph paths share them)."""
    K = getattr(layer, "_ov_online_tail_k", None)
    V = getattr(layer, "_ov_online_tail_v", None)
    if K is None or V is None or K.shape[0] != maxw:
        K = torch.zeros((maxw, kvh, hd), dtype=torch.float32, device=dev)
        V = torch.zeros((maxw, kvh, hd), dtype=torch.float32, device=dev)
        layer._ov_online_tail_k = K
        layer._ov_online_tail_v = V
    for _an, _ad in (("_ov_dec_ev", torch.bool),
                     ("_ov_dec_gg", torch.int64),
                     ("_ov_dec_ss", torch.int64)):
        _t = getattr(layer, _an, None)
        if _t is None or _t.shape[0] != maxw:
            _t = torch.empty((maxw,), dtype=_ad, device=dev)
            setattr(layer, _an, _t)
    return K, V


def _try_kvarn_graph_decode(layer, q, k, v, cache_seqlens,
                            block_table, q_len, qh, kvh, hd, qpk,
                            sl, scale, sscale, dev, arm):
    """Graph fast path: returns (stored, out|None). stored False =
    decline before store (run full eager); (True, None) = stored,
    run eager rest (skip store); (True, out) = replayed. Never
    half-runs (loud fallback to eager on any trip)."""
    # Default ON (green twice at 8k/16k/64k + review fixes; kill with
    # EXL3_KVARN_GRAPH=0 to restore pure eager).
    if os.environ.get("EXL3_KVARN_GRAPH", "1") != "1":
        return (False, None)
    if not arm:
        return (False, None)
    if q.shape[0] != 1 or int(cache_seqlens.numel()) != 1:
        return (False, None)
    layer._ov_dec_pending_capture = None
    from ...cache.kvarn import _ptime_count as _pgc, _kvarn_ptimes_on, _ptimes_report, _kvarn_past_cached, _kvarn_past_commit
    global _g_calls
    _g_calls += 1
    if _kvarn_ptimes_on() and _g_calls % 2048 == 0:
        _ptimes_report("graph-engagement")
    _past, _hit = _kvarn_past_cached(layer, cache_seqlens, q_len)
    n = _past + q_len
    code = layer.update_kv_direct(cache_seqlens, block_table, k, v,
                                 q_len)
    if code is None or code != 0:
        _pgc("g_fb_code")
        _kvarn_past_commit(layer, n, False)
        return (True, None)
    _kvarn_past_commit(layer, n, True)
    tail_eff = int(layer.tail_effective)
    sink_n = arm.c_sink if layer.has_sink else 0
    gps = arm.c_page // arm.c_group
    _gc = min((n + 127) // 128, int(layer.records.shape[0]))
    _sn = min(arm.c_sink, n) if layer.has_sink else 0
    _t0 = max(0, n - tail_eff)
    _R = _sn + (n - _t0)
    if _R <= 0:
        _pgc("g_fb_shape")
        return (True, None)
    maxw = int(layer.kvarn_online_maxw())
    if _R > maxw:
        _pgc("g_fb_shape")
        return (True, None)
    if not bool(getattr(layer, "_tail_exact_certain", False)):
        _pgc("g_fb_cert")
        return (True, None)
    # NOTE: no serve-tick probe here (review #1): PARITY=1 must
    # still replay (the assert above validates eref); the sticky
    # tripwire is covered by the post-replay due read below, at the
    # same every-128 cadence as eager.
    if os.environ.get("EXL3_KVARN_TRITON_PARITY", "0") == "1":
        Ew = layer.kvarn_eref_cached()
        Ew_ref = arm.t_wht_rows(layer.exact_v.float(), hd)
        assert torch.equal(Ew, Ew_ref), "eref diverged (graph run)"
    try:
        Qbuf, Qfbuf, nbuf, tbuf, btbuf = _graph_bufs(
            layer, dev, qh, kvh, hd, qpk, maxw,
            int(block_table.shape[1]), cache_seqlens.dtype,
            block_table.dtype)
        Qbuf.copy_(q[0, 0])
        Qfbuf.copy_(q[0, 0])
        nbuf.copy_(cache_seqlens[:1])
        nbuf += q_len
        torch.arange(_sn, device=dev, out=tbuf[:_sn])
        torch.arange(_t0, n, device=dev, out=tbuf[_sn:_R])
        btbuf.copy_(block_table[0])
        gb = getattr(layer, "_ov_dec_graphs", None)
        if gb is None:
            layer._ov_dec_graphs = gb = {}
        key = (_gc, _R)
        bad = getattr(layer, "_ov_dec_bad", None)
        if bad is not None and key in bad:
            return (True, None)
        if key not in gb:
            layer._ov_dec_pending_capture = {
                "gc": _gc, "R": _R, "scale": scale, "sscale": sscale,
                "qh": qh, "kvh": kvh, "hd": hd, "qpk": qpk, "sl": sl,
                "sink_n": sink_n, "tail_eff": tail_eff, "gps": gps}
            _pgc("g_miss")
            return (True, None)
        _ent = gb[key]
        # Buffer-identity verify (review #3/#6): serve realloc on gc
        # change or st-cache eviction orphans baked addresses. Mismatch
        # drops the bucket (recapture next miss); never replays stale.
        _st_now = getattr(layer, "_ov_dec_st_cache", {}).get(_R)
        if (_ent["m"] is not layer._ov_serve_m
                or _ent["acc"] is not layer._ov_serve_acc
                or _st_now is not _ent["st"]):
            del gb[key]
            _pgc("g_stale")
            return (True, None)
        _ent["graph"].replay()
        _pgc("g_replay")
        # Post-replay sticky-flag due read (review #2): same every-128
        # cadence as eager; trip discards outputs + clears graphs.
        _rn = int(getattr(layer, "_ov_dec_replays", 0)) + 1
        layer._ov_dec_replays = _rn
        if _rn % 128 == 0:
            if int(layer._ov_serve_flag[0]) != 0:
                gb.clear()
                print("KVARN-GRAPH sticky trip during replay: graphs cleared, eager fallback", flush=True)
                return (True, None)
        return (True, layer._ov_dec_out)
    except Exception as _e:
        print(f"KVARN-GRAPH unexpected, eager fallback: {_e}", flush=True)
        import traceback as _tb
        print("".join(_tb.format_exc(limit=8)), flush=True)
        return (True, None)



def _graph_capture(layer, cap):
    """Record the sublattice graph for one bucket (called after an
    eager-rest warmed kernels; inputs already in static bufs).
    Raises on failure (caller marks bucket bad, stays eager)."""
    arm = _kvarn_arm_load()
    dev = layer.device
    gc, R = cap["gc"], cap["R"]
    qh, kvh, hd, qpk, sl = (cap["qh"], cap["kvh"], cap["hd"],
                             cap["qpk"], cap["sl"])
    groups = arm.t_groups(gc)
    Qbuf = layer._ov_dec_g_q
    Qfbuf = layer._ov_dec_qf
    nbuf = layer._ov_dec_g_n
    tbuf = layer._ov_dec_g_t
    btbuf = layer._ov_dec_g_b
    _mb, _lb, _ab, qw, qs, _o = arm.t_bufs(
        layer, qh, 1 << (qpk - 1).bit_length(), hd, dev)
    Qh = Qfbuf.reshape(kvh, qpk, hd)
    maxw = int(layer.kvarn_online_maxw())
    K, V = _graph_tail_bufs(layer, dev, kvh, hd, maxw)
    Kt = K[:R].permute(1, 2, 0)
    Vt = V[:R]
    st = layer._ov_dec_st_cache[R]
    tm, td, tn = layer._ov_dec_tail
    tm_v = tm.reshape(qh)
    td_v = td.reshape(qh)
    tn_v = tn.reshape(qh, hd)
    evb = layer._ov_dec_ev[:R]
    ggb = layer._ov_dec_gg[:R]
    ssb = layer._ov_dec_ss[:R]
    m_out = layer._ov_dec_out
    Ew = layer.kvarn_eref_cached()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        arm.t_qwht(Qbuf, qs, qw, sl, cap["sscale"])
        _ob, _fb = arm.t_serve(
            layer, qw, Qfbuf, layer.exact_k, Ew, layer._exact_rev,
            layer.sealed, btbuf, nbuf, qpk, cap["scale"],
            cap["sink_n"], cap["tail_eff"], cap["gps"], gc=gc,
            sync_flag=False)
        arm.t_tailgather(
            layer, tbuf, btbuf, K, V, cap["gps"],
            (evb, ggb, ssb))
        torch.bmm(Qh, Kt, out=st)
        torch.mul(st, cap["scale"], out=st)
        arm.t_tailred(st, Vt, ggb, layer._exact_rev, (tm, td, tn))
        arm.t_merge(layer._ov_serve_m, layer._ov_serve_l, _ob,
                    tm_v, td_v, tn_v, qpk, groups, m_out)
    layer._ov_dec_graphs[(gc, R)] = {"graph": g, "Kt": Kt,
                                        "Vt": Vt, "gg": ggb,
                                        "Qh": Qh, "tm_v": tm_v,
                                        "td_v": td_v, "tn_v": tn_v,
                                        "m": layer._ov_serve_m,
                                        "acc": layer._ov_serve_acc,
                                        "st": st}


def _try_kvarn_online_decode(q, k, v, cache, cache_idx, cache_instance,
                             block_table, cache_seqlens, q_len, sm_scale,
                             causal, window_size, softcap, sinks,
                             non_causal_spans, cu_seqlens,
                             window_right: int = 0, sink_key0: bool = False):
    """Imageless KVarN decode arm (match-bee Task 3): serves single-token
    decode attention online from records (promoted single-kernel serve)
    + torch tail block + original-domain merge, with NO persistent image.
    Returns fp16 (1, 1, qh, hd) or None (fail-closed to the get_kv path).

    Stores new rows first (mirrors the qc path, so the post-attention
    write-back is already done and this returns directly). Body served
    by kvarn_triton_online_serve (in-kernel online partials + production
    combine); tail (unassigned rows only) and merge stay torch.
    Loud paths only: every gate declines with None, never half-runs.
    """
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
    # 1.5.4 added window_right / sink_key0 to attn_dispatch. This arm returns
    # directly and so bypasses the generic path that forwards them into
    # AttnArgs (see the attn_dispatch call site), which would silently drop
    # both. The kernels have no windowed path, so decline rather than honor
    # them part-way -- fail-closed, same as every other gate here.
    if window_right != 0 or sink_key0:
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
    _arm = _kvarn_arm_load()
    if not _arm:
        return None
    kvarn_triton_available = _arm.t_avail
    kvarn_triton_qwht = _arm.t_qwht
    kvarn_triton_online_partials = _arm.t_partials
    kvarn_triton_wht_rows = _arm.t_wht_rows
    kvarn_triton_online_serve = _arm.t_serve
    kvarn_triton_online_merge = _arm.t_merge
    kvarn_triton_online_tail_reduce = _arm.t_tailred
    _kvarn_online_buffers = _arm.t_bufs
    _kvarn_serve_groups = _arm.t_groups
    KVAR_N_SINK_TOKENS = _arm.c_sink
    KVAR_N_GROUP = _arm.c_group
    _ptime_count = _arm.c_ptime
    PAGE_SIZE = _arm.c_page
    if not kvarn_triton_available():
        return None
    qpk = qh // kvh
    sl = hd // 128
    scale = sm_scale if sm_scale is not None else dim ** (-0.5)
    sscale = 1.0 if sl == 1 else (0.7071067811865475 if sl == 2 else 0.5)
    dev = q.device
    # Graph attempt (includes store; kill-switched inside, default off).
    # Returns (stored, out): stored False -> full eager below; (True,
    # None) -> stored, eager rest skips store; (True, out) -> replayed.
    _g_stored, _g_out = _try_kvarn_graph_decode(
        layer, q, k, v, cache_seqlens, block_table, q_len,
        qh, kvh, hd, qpk, sl, scale, sscale, dev, _arm)
    if _g_out is not None:
        return _g_out.reshape(bsz, q_len, qh, hd)
    # Store first (write-back already done on this path). Eager-rest
    # runs only on graph decline/fallback/miss: always take the real
    # sync here (rare path) so a stale mirror self-corrects before
    # doing math; the mirror refreshes below for the next step.
    from ...cache.kvarn import _kvarn_past_commit as _pc2
    if not _g_stored:
        _code = layer.update_kv_direct(cache_seqlens, block_table, k, v,
                                       q_len)
    else:
        _code = 0
    n = int(cache_seqlens[0]) + q_len
    _pc2(layer, n, _code == 0)
    n_0d = cache_seqlens[:1] + q_len
    sink_n = KVAR_N_SINK_TOKENS if layer.has_sink else 0
    tail_eff = int(layer.tail_effective)
    Q = q[0, 0]
    # QWHT (persistent buffers). Body served by the promoted
    # single-kernel serve below (own persistent partials buffers).
    qpad = 1 << (qpk - 1).bit_length()
    _mb, _lb, _ab, qw, qs, _o = _kvarn_online_buffers(
        layer, qh, qpad, hd, dev)
    kvarn_triton_qwht(Q, qs, qw, sl, sscale)
    # Persistent fp32 Q (saves 1 alloc/layer/step; shape static).
    # copy_ converts in the same kernel .float() would run.
    _dq = getattr(layer, "_ov_dec_qf", None)
    if _dq is None or tuple(_dq.shape) != (qh, hd):
        _dq = torch.empty((qh, hd), dtype=torch.float32, device=dev)
        layer._ov_dec_qf = _dq
    _dq.copy_(Q)
    Qf = _dq
    # Body via promoted single-kernel serve (in-kernel online partials +
    # production combine, ORIGINAL-domain normalized body out).
    # Incremental eref (Spec A): serve reads the per-layer cached Ew
    # (slot-wise WHT maintained by the kvarn.py store/evict/copy hooks;
    # _eref_ensure full-refreshes once on first build). No per-step full
    # remat. The serve sticky flag is now periodic-check (every 128
    # serves/layer; PARITY=1 checks every call): KLD-green since Spec A,
    # so the every-step fail-closed sync is retired. Status code stays
    # synchronous (drives control flow: 0 append / 2 seal / 1 torch
    # fallback -- cannot speculate).
    Ew = layer.kvarn_eref_cached()
    if os.environ.get("EXL3_KVARN_TRITON_PARITY", "0") == "1":
        Ew_ref = kvarn_triton_wht_rows(layer.exact_v.float(), hd)
        assert torch.equal(Ew, Ew_ref), \
            "KVarN incremental eref disagrees with full refresh"
    gps = PAGE_SIZE // KVAR_N_GROUP
    gc_eff = min((n + 127) // 128, int(layer.records.shape[0]))
    out_b, flag_b = kvarn_triton_online_serve(
        layer, qw, Qf, layer.exact_k, Ew, layer._exact_rev, layer.sealed,
        block_table[0], n_0d, qpk, scale, sink_n, tail_eff, gps,
        gc=gc_eff, sync_flag=_kvarn_check_flag(layer))
    if flag_b:
        # Fail-closed: sticky flag means open-body rows reached the
        # kernel (argued impossible for dense); the get_kv path serves.
        return None
    with torch.inference_mode():
        # Tail positions first: the same tpos feeds the tail gather
        # (passed in so kvarn_online_tail skips rebuilding it) and the
        # assignment mask below -- one cat, consistent by construction.
        sn_ = min(KVAR_N_SINK_TOKENS, n) if layer.has_sink else 0
        # Memoized across layers (same key per step): saves 64 x
        # (2 aranges + cat + long) per step. Read-only downstream.
        tpos = _kvarn_tpos(dev, n, tail_eff, sn_)
        # Tail block (exact-first + staging fallback, original domain),
        # UNASSIGNED rows only: assigned tail rows are already inside
        # out_b (exact-direct); counting them again would corrupt the
        # merge. Mask by the SAME array the kernel reads (exrev) so the
        # partition is airtight even if the valid⟺assigned invariant
        # wobbles: kernel covers exrev>=0, torch covers exrev<0.
        Kt, Vt, tg = layer.kvarn_online_tail(n, block_table[0], pos=tpos)
        # R-bucket observability for graphs planning (shape-only, PTIMES-gated).
        _ptime_count(f"tailR_{int(Kt.shape[0])}")
        # Batched scores over heads (one bmm: identical per-element
        # contraction order), fused masked-softmax + value reduction
        # (one launch, was ~5 dispatches + the pe temporary). The mask
        # lives inside the kernel now (same exrev array the serve kernel
        # reads: torch owns exrev<0 rows, airtight by construction).
        Qh = Qf.reshape(kvh, qpk, hd)  # view of Qf (was a second
        # fp32 copy of Q: identical values, saves 16 _to_copy/step)
        # Persistent bmm out, R-keyed (R takes few values; cap 4).
        # bmm out= + mul out= : zero allocs, identical values.
        R = int(Kt.shape[0])  # shape only, no sync
        _std = getattr(layer, "_ov_dec_st_cache", None)
        if _std is None:
            layer._ov_dec_st_cache = _std = {}
        st_buf = _std.get(R)
        if st_buf is None:
            st_buf = torch.empty((kvh, qpk, R), dtype=torch.float32,
                                 device=dev)
            if len(_std) >= 4:
                _std.pop(next(iter(_std)))
            _std[R] = st_buf
        st = torch.bmm(Qh, Kt.permute(1, 2, 0), out=st_buf)
        torch.mul(st, scale, out=st)  # (kvh, qpk, R)
        # Persistent tail outs (shapes static per layer).
        _tb = getattr(layer, "_ov_dec_tail", None)
        if _tb is None or tuple(_tb[0].shape) != (kvh, qpk):
            _tb = (torch.empty((kvh, qpk), dtype=torch.float32,
                               device=dev),
                   torch.empty((kvh, qpk), dtype=torch.float32,
                               device=dev),
                   torch.empty((kvh, qpk, hd), dtype=torch.float32,
                               device=dev))
            layer._ov_dec_tail = _tb
        tail_m, tail_den, tail_num = kvarn_triton_online_tail_reduce(
            st, Vt, tg, layer._exact_rev, _tb)
        tail_m = tail_m.reshape(qh)
        tail_den = tail_den.reshape(qh)
        tail_num = tail_num.reshape(qh, hd)
        # Fused body-stats + original-domain merge (one launch, was
        # ~13 torch dispatches): out_b is already original-domain
        # normalized body attention (combine folds the out-WHT), so it
        # un-normalizes by den with NO extra WHT (cacd7af). Groups (not
        # gc_eff): serve partials are groups-strided under hierarchical
        # subgroups; read the stride from the buffer itself.
        # Persistent merge out (shape static per layer).
        _mo = getattr(layer, "_ov_dec_out", None)
        if _mo is None or tuple(_mo.shape) != (qh, hd):
            _mo = torch.empty((qh, hd), dtype=torch.float16, device=dev)
            layer._ov_dec_out = _mo
        out = kvarn_triton_online_merge(
            layer._ov_serve_m, layer._ov_serve_l, out_b,
            tail_m, tail_den, tail_num, qpk,
            int(layer._ov_serve_m.shape[2]), _mo)
    # Graph capture-after (miss path set pending above; kernels warm
    # from this eager rest). Failure pins the bucket eager (loud).
    _cap = getattr(layer, "_ov_dec_pending_capture", None)
    if _cap is not None:
        layer._ov_dec_pending_capture = None
        try:
            _graph_capture(layer, _cap)
            _ptime_count("g_captured")
        except Exception as _e:
            _ptime_count("g_capfail")
            _bad = getattr(layer, "_ov_dec_bad", None)
            if _bad is None:
                layer._ov_dec_bad = _bad = set()
            _bad.add((_cap["gc"], _cap["R"]))
            import traceback as _tb
            print(f"KVARN-GRAPH capture failed, eager pinned: {_e}",
                  flush=True)
            print("".join(_tb.format_exc(limit=12)), flush=True)
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
    window_right: int = 0,
    sink_key0: bool = False,
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
            sinks, non_causal_spans, cu_seqlens, window_right, sink_key0)
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
        window_right = window_right,
        sink_key0 = sink_key0,
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
        if isinstance(layer, CacheLayer_kvarn):
            # Legacy/get_kv path doesn't track the n-mirror (only the
            # online arm commits it): invalidate so the next online
            # step resyncs once instead of trusting a stale total.
            # Unreachable on the online path (it returns early above).
            layer._ov_dec_n_mirror = None
        if isinstance(cache, CacheLayer):
            cache.update_kv(cache_seqlens, block_table, k_cache, v_cache, q_len)
        elif isinstance(cache, Cache):
            cache.update_layer(cache_idx, cache_seqlens, block_table, k_cache, v_cache, q_len, cache_instance)

    return o
