"""
EXL3_MOE_LOOKAHEAD: a pre-attention router lookahead that prefetches the host-resident experts a
split MoE block is predicted to route.

Backlog item #1 (`wiki/plans/2026-11-04-decode-backlog.md`) proposes computing a block's router
picks from the residual that enters the block -- i.e. *before* the attention/GDN sublayer has run --
and pulling the predicted CPU-resident experts' bytes into cache while attention runs, so the
worker's miss is an L3 hit by the time the true routing asks for the bytes.

`bench/router_lookahead_probe.py` measured the half of the claim that is a prediction-quality
question: the pre-attention residual predicts **49.7 %** of the router's top-k (7 680 picks, 48
layers, one 16-token decode), with `redundancy = 2.01x` -- half of what such a prefetch touches is
not routed. This module is the other half, the mechanism.

Shape (three knobs, all default off):

    EXL3_MOE_LOOKAHEAD       0/1, default 0
    EXL3_MOE_LOOKAHEAD_MAX   experts prefetched per layer per step, default 8
    EXL3_MOE_LOOKAHEAD_STATS 0/1, default 0: print the counters at exit

What it does per split block forward, in order:

  1. collapse the block's input to the residual a lookahead would have (the stream mean for a
     hyper-connection block, exactly the probe's fallback -- the predictor whose 49.7 % was
     measured; `_mix_ref` is deliberately NOT used, it is three extra linears and it needs source
     weights an EXL3 checkpoint does not keep);
  2. re-run the layer's own router pipeline on it (`mlp_norm` -> `router_pre_norm` ->
     `routing_fn` with the layer's `RoutingCFG`), so a bias, a scale or a group rule carries over;
  3. take the predicted ids that are host-resident under the layer's current placement, at most
     `MAX` of them;
  4. touch one 64-byte line per cache line of each of their trellis blocks in the shared arena
     (a numpy strided reduce over the parent's mapping of the worker's arena).

Correctness: it reads memory. It never writes the arena, the placement map or the routing. The only
engine state it consumes is the model's own router and the arena's byte offsets; a wrong prediction
is wasted bandwidth, never a different output.

Cost, stated here because it is the reason the knob is off by default (see the task report for the
measurement):

  * one router call (norm + gate) per layer per step, and one device-to-host copy of the ids --
    the ids have to be on the CPU before any CPU-side read can be issued, and that copy is a
    synchronisation point in the middle of the forward pass;
  * `MAX` expert blocks touched per layer per step. On this box one 1.92 MB expert block costs
    ~130 us to pull line-by-line from one thread (~15 GB/s), the touch is bandwidth-bound, and at
    49.7 % coverage with 2.01x redundancy about half the touched bytes are never routed;
  * the parent can only reach the arena when `EXL3_MOE_PINNED_ARENA=1` (that is what maps the
    worker's chunks into this process). With it off -- the served regime -- every layer is skipped
    by name and the counters say so. Nothing here maps, pins or promotes anything itself.
"""

from __future__ import annotations

import os

# Router types whose selection is a function of the hidden state. Same tuple as the probe's
# `LINEAR_ROUTER_TYPES`: a router outside it has no hidden-state function to re-apply, so the layer
# is refused by name rather than predicted badly.
LINEAR_ROUTER_TYPES = ("std", "std_bias", "dots", "sqrtsp", "ds3")

DEFAULT_MAX_EXPERTS = 8
LINE_BYTES = 64               # x86-64 cache line; one load per line is the cheapest full touch
DEFAULT_STREAM_MIN_ROWS = 32  # the engine's decode/prefill boundary (MoeCpuHost.stream_min_rows)


# ---------------------------------------------------------------------------------------- config
# Pure: the flags are parsed from a mapping so the tests can drive every combination without
# touching os.environ or re-importing the module.

def _flag(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() not in ("0", "", "false", "no", "off")


def _positive(value, default):
    if value is None or str(value).strip() == "":
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n > 0 else default


class LookaheadCFG:
    """The three knobs. `enabled` gates everything: with it False no object is created and no
    method in this file is reachable from the forward pass."""

    __slots__ = ("enabled", "max_experts", "stats")

    def __init__(self, enabled=False, max_experts=DEFAULT_MAX_EXPERTS, stats=False):
        self.enabled = bool(enabled)
        self.max_experts = max(1, int(max_experts))
        self.stats = bool(stats)

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        return cls(
            enabled = _flag(env.get("EXL3_MOE_LOOKAHEAD"), False),
            max_experts = _positive(env.get("EXL3_MOE_LOOKAHEAD_MAX"), DEFAULT_MAX_EXPERTS),
            stats = _flag(env.get("EXL3_MOE_LOOKAHEAD_STATS"), False),
        )

    def __repr__(self):
        return (f"LookaheadCFG(enabled={self.enabled}, max_experts={self.max_experts}, "
                f"stats={self.stats})")


_CFG = None


def lookahead_cfg():
    """This process's flags (read once, at first use)."""
    global _CFG
    if _CFG is None:
        _CFG = LookaheadCFG.from_env()
    return _CFG


# ---------------------------------------------------------------------------------------- refusals

def support_reason(mlp) -> str | None:
    """Why this module cannot issue a pre-attention prefetch, or None.

    Attribute-only, so the refusal is testable without the engine and without torch. The reasons
    name the cause: a layer that is skipped because routing is not a hidden-state function must not
    look like a layer that measured zero coverage. The first four are the probe's reasons
    verbatim (`bench/router_lookahead_probe.py:router_support_reason`); the last two are this
    mechanism's own -- a prefetch needs host-resident bytes and a mapping to reach them.
    """
    if getattr(mlp, "routing_gate", None) is None:
        return ("no router on this module: the selection arrives by broadcast from the routing rank "
                "(a tensor-parallel shard without the gate), so there is no hidden-state function here")
    if getattr(mlp, "tid2eid", None) is not None or getattr(mlp, "router_type", "std") == "sqrtsp_hash":
        return ("hash-selected router: the picks are read from a tid2eid table indexed by input_ids, not "
                "computed from the hidden state, so a pre-attention lookahead cannot predict them")
    rtype = getattr(mlp, "router_type", None)
    if rtype not in LINEAR_ROUTER_TYPES:
        return (f"router type {rtype!r} is not one of {LINEAR_ROUTER_TYPES}: its selection is not a "
                f"plain gate over the hidden state, so the lookahead has no router to re-apply")
    if getattr(mlp, "alt_residual_channel", False):
        return ("alt_residual_channel: the router reads params['residual'] instead of the block input, "
                "so the pre-attention residual is not the vector a lookahead would give the router")
    if getattr(mlp, "cpu_offload", False):
        return ("whole-layer offload: every expert of this layer is on the worker, and this mechanism "
                "is defined on a split's tail (that is the placement the 49.7 % overlap was measured "
                "in); a whole-layer prefetch would need the worker's own expert order")
    if getattr(mlp, "cpu_split_first", None) is None:
        return ("not a split layer: every expert of this layer is GPU-resident, so a prefetch has no "
                "host-resident bytes to reach")
    if getattr(mlp, "cpu_host", None) is None:
        return "no CPU worker on this layer: there is no host arena to prefetch from"
    return None


# ---------------------------------------------------------------------------------------- counters

def _bump(d, key, n = 1):
    d[key] = d.get(key, 0) + n


class LookaheadStats:
    """Counters for the whole model, summed over layers.

    Three different bases, kept apart on purpose:
      * `layers_*` counts LAYERS (attached, and skipped with a reason);
      * `predictions` counts ROUTER CALLS issued by the lookahead (one per layer per step);
      * `experts`/`nbytes` count expert BLOCKS touched, i.e. the transfers -- the unit of the cost;
      * `checked`/`hits` count EXPERT BLOCKS that the true router then selected, over the blocks the
        prediction named. That share (`hits / checked`) is the number that says whether the
        mechanism works; it is only measured with EXL3_MOE_LOOKAHEAD_STATS, because it costs the
        readback of the true selection.
    """

    def __init__(self):
        self.layers_attached = 0
        self.layer_skips = {}       # reason -> layers
        self.step_skips = {}        # reason -> steps (a layer-step that issued nothing)
        self.predictions = 0
        self.experts = 0
        self.nbytes = 0
        self.checked = 0            # expert blocks whose fate was compared with the true routing
        self.hits = 0

    # -- accumulation -------------------------------------------------------------------------
    @property
    def layers_skipped(self):
        return sum(self.layer_skips.values())

    def skip_layer(self, reason):
        _bump(self.layer_skips, reason)

    def skip_step(self, reason):
        _bump(self.step_skips, reason)

    def predicted(self):
        self.predictions += 1

    def touched(self, blocks, nbytes):
        self.experts += blocks
        self.nbytes += nbytes

    def compared(self, prefetched, hits):
        self.checked += prefetched
        self.hits += hits

    # -- reporting -----------------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "layers_attached": self.layers_attached,
            "layers_skipped": self.layers_skipped,
            "layer_skip_reasons": dict(sorted(self.layer_skips.items())),
            "predictions": self.predictions,
            "experts_prefetched": self.experts,
            "bytes_touched": self.nbytes,
            "expert_precision": (self.hits / self.checked) if self.checked else None,
            "experts_checked": self.checked,
            "experts_hit": self.hits,
            "step_skips": dict(sorted(self.step_skips.items())),
        }

    def report(self) -> str:
        s = self.summary()
        lines = [f" -- MoE lookahead (EXL3_MOE_LOOKAHEAD): {s['layers_attached']} layer(s) attached, "
                 f"{s['layers_skipped']} skipped"]
        for reason, n in s["layer_skip_reasons"].items():
            lines.append(f"    skipped {n} layer(s): {reason}")
        lines.append(f"    predictions {s['predictions']} issued, {s['experts_prefetched']} expert "
                     f"block(s) prefetched, {s['bytes_touched'] / 1e6:.1f} MB touched")
        for reason, n in s["step_skips"].items():
            lines.append(f"    {n} layer-step(s) issued nothing: {reason}")
        if s["expert_precision"] is None:
            lines.append("    expert precision: not measured (EXL3_MOE_LOOKAHEAD_STATS=0)")
        else:
            lines.append(f"    expert precision: {100.0 * s['expert_precision']:.1f}% of the "
                         f"prefetched experts were routed ({s['experts_hit']}/{s['experts_checked']})")
        return "\n".join(lines)


_STATS = LookaheadStats()
_REPORTED = False


def stats() -> LookaheadStats:
    """The process-wide counters (one lookahead spans every split layer of a model)."""
    return _STATS


def reset_stats():
    """Forget everything counted so far. Test and sweep hook only."""
    global _STATS, _REPORTED
    _STATS = LookaheadStats()
    _REPORTED = False
    return _STATS


def report_stats(out = None):
    """Print the counters once (no-op after the first call, so atexit plus an explicit close cannot
    double-print). Safe at interpreter exit: every number is already an int."""
    global _REPORTED
    if _REPORTED or _STATS is None:
        return
    _REPORTED = True
    print(_STATS.report(), file = out)


def _register_exit_report():
    import atexit
    atexit.register(report_stats)


# ---------------------------------------------------------------------------------------- the arena

class ArenaToucher:
    """Touches the bytes of a host-resident expert, in the parent's mapping of the worker's arena.

    Only `MoeCpuHost.arena_views` (the parent's mapping, per chunk) and `MoeCpuHost.layer_blocks`
    (per layer, per worker slot, the (chunk, byte offset) of the expert's contiguous
    [gate | up | down] trellis block) are read. Nothing here maps, pins, promotes, demotes or
    writes: those live in `model/moe_cpu_host.py` and are untouched.

    Why the parent's mapping can be touched at all: with EXL3_MOE_PINNED_ARENA the worker allocates
    the arena as a shared memfd (Windows: a named section) and the parent mmaps the same pages, so a
    line pulled here is the line the worker reads.

    Why strided: `block[::32]` on the int16 view loads one element per 64-byte line, which is the
    minimum work that makes the line resident, while a contiguous reduce of the same block moves 32x
    the bytes through L1/L2 for the same DRAM traffic. Measured on this box (7950X3D, one thread,
    cold buffer, 8 expert blocks): strided 51 GB/s of lines against sequential 19 GB/s.
    """

    def __init__(self, numpy_module = None, line_bytes = LINE_BYTES):
        self._np = numpy_module
        self._line_bytes = line_bytes
        # [(host, {chunk: numpy view or None})], matched by identity: the arena is attached once per
        # host, and holding the host in the entry keeps its id() from being reused under us
        self._maps = []
        self.reason = None          # a named refusal, when there is one

    # -- setup ---------------------------------------------------------------------------------
    def _numpy(self):
        if self._np is None:
            try:
                import numpy
            except Exception as exc:                            # noqa: BLE001 - recorded, not swallowed
                self.reason = (f"numpy is not importable ({type(exc).__name__}: {exc}); the touch is a "
                               f"numpy strided reduce, so there is no way to issue one")
                return None
            self._np = numpy
        return self._np

    def _chunk_views(self, host):
        for entry in self._maps:
            if entry[0] is host:
                return entry[1]
        entry = [host, {}]
        self._maps.append(entry)
        return entry[1]

    def _chunk_view(self, host, chunk):
        """numpy view over arena chunk `chunk` of `host`, or None when this process has no mapping
        of it (the pinned arena is off, or the driver ran out of page-locked memory before it)."""
        views = self._chunk_views(host)
        if chunk in views:
            return views[chunk]
        np = self._numpy()
        if np is None:
            return None
        mapped = host.arena_views or []
        if chunk >= len(mapped) or mapped[chunk] is None:
            if self.reason is None:
                self.reason = ("the worker's arena is not mapped in this process "
                               "(EXL3_MOE_PINNED_ARENA=0, or the driver's page-locked budget ran out "
                               "before this chunk). That is what a parent-side touch needs: it is the "
                               "reason this regime cannot prefetch, not a zero it measured")
            views[chunk] = None
            return None
        t = mapped[chunk]
        try:
            # A torch tensor the host built with frombuffer over its mmap: numpy shares the memory,
            # no copy. Fall back to a buffer-protocol wrap if that bridge is unavailable.
            arr = t.numpy() if hasattr(t, "numpy") else np.frombuffer(t, dtype = np.int16)
        except Exception as exc:                                # noqa: BLE001 - recorded, not swallowed
            self.reason = (f"the arena mapping is not readable as a numpy array "
                           f"({type(exc).__name__}: {exc})")
            views[chunk] = None
            return None
        views[chunk] = arr
        return arr

    def block(self, host, layer_idx, cpu_local):
        """(view, start, stop) of worker slot `cpu_local` of `layer_idx`, or None."""
        blocks = getattr(host, "layer_blocks", None) or []
        if layer_idx is None or layer_idx >= len(blocks) or not blocks[layer_idx]:
            return None
        slots = blocks[layer_idx]
        if cpu_local < 0 or cpu_local >= len(slots):
            return None
        loc = slots[cpu_local]
        if loc is None:
            return None
        chunk, off = loc[0], loc[1]
        spec = host.specs[layer_idx] if layer_idx < len(host.specs) else None
        nbytes = (spec or {}).get("expert_bytes")
        if not nbytes:
            return None
        arr = self._chunk_view(host, chunk)
        if arr is None:
            return None
        item = arr.itemsize
        start = off // item
        stop = (off + nbytes) // item
        if stop > arr.size:
            return None
        return arr, start, stop

    # -- the touch -----------------------------------------------------------------------------
    def touch(self, host, layer_idx, cpu_locals) -> tuple[int, int, int]:
        """Issue one line-load per cache line of each block. Returns (blocks touched, bytes, missed)
        where `missed` counts blocks that could not be reached."""
        arr_blocks = []
        missed = 0
        for cpu_local in cpu_locals:
            b = self.block(host, layer_idx, cpu_local)
            if b is None:
                missed += 1
            else:
                arr_blocks.append(b)
        if not arr_blocks:
            return 0, 0, missed
        np = self._np
        stride = max(1, self._line_bytes // arr_blocks[0][0].itemsize)
        nbytes = 0
        for arr, start, stop in arr_blocks:
            # np.max, not np.add.reduce: an int16 accumulator wraps, and the value is unused either
            # way -- what matters is that every 64-byte line is loaded into the cache
            np.max(arr[start:stop:stride])
            nbytes += (stop - start) * arr.itemsize
        return len(arr_blocks), nbytes, missed


# ---------------------------------------------------------------------------------------- the layer

def collapse_pre(x):
    """The residual a lookahead would have, from the vector the block is handed.

    Hyper-connection blocks carry a (b, s, hc_mult, hidden) fp32 stream stack and the sublayer sees
    a *mixed* collapse of it. The mix needs the site's source weights (`_mix_ref` calls
    `_require_source_weights`), which a quantized EXL3 checkpoint does not keep, so the collapse
    here is the plain mean over the streams -- the probe's documented fallback and the predictor its
    49.7 % was measured with. A weaker predictor than the gated mix, never a different answer: the
    mean only decides what is prefetched.
    """
    if x.dim() == 4 and x.shape[-2] > 1:
        return x.mean(dim = -2)
    return x


class LayerLookahead:
    """One split MoE layer's lookahead: predict at the block's input, prefetch before attention.

    `mlp` is the `BlockSparseMLP` (for its router pipeline and its host), `router` is the piece the
    prediction re-runs -- defaulted to the module's own `routing_fn`, `routing_cfg`, `mlp_norm` and
    `router_pre_norm`, so the tests can hand a stub in without a torch install.
    """

    def __init__(self, mlp, cfg, stats_obj, toucher = None, layer_idx = None,
                 stream_min_rows = DEFAULT_STREAM_MIN_ROWS, torch_module = None, router = None):
        self.mlp = mlp
        self.cfg = cfg
        self.stats = stats_obj
        self.toucher = toucher if toucher is not None else ArenaToucher()
        self.layer_idx = layer_idx if layer_idx is not None else getattr(mlp, "cpu_layer_idx", None)
        self.stream_min_rows = stream_min_rows
        self._torch = torch_module
        self._router = router
        self.failed = None            # a named per-layer failure: one message, then silent
        self._pred_router = set()     # router ids named by this step's prediction
        self._pred_slots = []         # worker slots prefetched for this step

    # -- dependencies (lazy: no torch import at module level, the tests have no torch) ---------
    def _t(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _r(self):
        """(routing_fn, routing_cfg, mlp_norm, router_pre_norm, hidden) -- injected or the module's"""
        if self._router is not None:
            return self._router
        m = self.mlp
        return (m.routing_fn, m.routing_cfg, getattr(m, "mlp_norm", None),
                getattr(m, "router_pre_norm", None), m.hidden_size)

    # -- placement -----------------------------------------------------------------------------
    def cpu_slot(self, router_id):
        """The worker slot that holds `router_id`'s weights now, or None when the expert is
        GPU-resident (nothing to prefetch) or its placement cannot be read without a device copy.

        Two placement regimes, both read from what the module already keeps on the host:
          * static (`_split_map is None`): worker slot = id - cpu_split_first, the load-time split.
          * dynamic without spares: nothing can move (`_split_promote_step` requires spares), so the
            same arithmetic holds.
          * dynamic with spares: `_split_mp` is the host mirror the promote step maintains, and it is
            authoritative -- a promoted expert's router id maps to a GPU slot, and prefetching its
            (now demoted) worker slot would fetch the wrong expert's bytes.
        """
        m = self.mlp
        first = m.cpu_split_first
        if first is None:
            return None
        mp = getattr(m, "_split_mp", None)
        if mp is not None:
            slot = mp[router_id]
            return None if slot < first else slot - first
        if getattr(m, "_split_spares", 0) and getattr(m, "_split_map", None) is not None:
            # Movable, and no host mirror to ask: refuse rather than prefetch another expert's block
            return None
        return router_id - first if router_id >= first else None

    # -- the two hooks -------------------------------------------------------------------------
    def issue(self, x, params = None):
        """Called at the START of the split block's forward: predict, then prefetch.

        Returns the number of expert blocks touched (0 when nothing was issued). Never raises into
        the forward pass: a failure is recorded once, by name, and the layer goes quiet.
        """
        if not self.cfg.enabled or self.failed is not None:
            return 0
        # This step's prediction starts empty: a step that issues nothing must not leave the previous
        # step's set for note_true() to score
        self._pred_router = set()
        self._pred_slots = []
        params = params or {}
        if params.get("prefill"):
            self.stats.skip_step("prefill pass (the prompt's routing is not decode's)")
            return 0
        if params.get("autosplit_measure"):
            self.stats.skip_step("autosplit measuring forward")
            return 0
        if params.get("tp_warmup"):
            self.stats.skip_step("tensor-parallel warmup (the selection is random)")
            return 0
        try:
            torch = self._t()
        except Exception as exc:                                # noqa: BLE001
            return self._fail(f"torch is not importable ({type(exc).__name__}: {exc})")
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                # A device-to-host copy cannot go in a captured graph, and issuing the prediction
                # outside a replayed graph would put a synchronisation in every decode step that
                # the graph exists to avoid. Turn the layer off, by name, rather than pay either.
                self._fail("CUDA graph capture covers this block: a lookahead needs a device-to-host "
                           "copy, which is not capturable, and running it outside the graph would "
                           "sync every step -- set EXL3_GRAPHS=0 to use the mechanism")
                return 0
        except Exception:                                       # noqa: BLE001 - a probe, not a demand
            pass

        pre = collapse_pre(x)
        rows = pre.numel() // self.mlp.hidden_size
        if rows >= self.stream_min_rows:
            self.stats.skip_step("prefill-size batch")
            return 0
        ids = self.predict(pre, torch)
        if ids is None:
            return 0
        self.stats.predicted()

        # The predicted ids that are host-resident under the current placement, in the router's own
        # order, deduplicated by worker slot and bounded by MAX
        picked = []          # (router id, worker slot)
        seen = set()
        for row in ids:
            for r in row:
                slot = self.cpu_slot(r)
                if slot is None or slot in seen:
                    continue
                seen.add(slot)
                picked.append((r, slot))
        picked = picked[:self.cfg.max_experts]
        self._pred_router = {r for r, _slot in picked}
        self._pred_slots = [slot for _r, slot in picked]
        if not picked:
            self.stats.skip_step("no host-resident pick predicted")
            return 0
        host = self.mlp.cpu_host
        touched, nbytes, missed = self.toucher.touch(host, self.layer_idx, self._pred_slots)
        if touched == 0:
            # The parent has no mapping of these bytes: the pinned arena is off, or the driver's
            # page-locked budget ran out before their chunk was registered
            self.stats.skip_step(self.toucher.reason or
                                 "the worker's arena is not mapped in this process "
                                 "(EXL3_MOE_PINNED_ARENA=0, or no page-locked memory for the chunk)")
            return 0
        if missed:
            self.stats.skip_step("predicted expert block not reachable in the parent's mapping")
        self.stats.touched(touched, nbytes)
        return touched

    def predict(self, pre, torch = None):
        """The predicted router ids, one list per row, or None on a recorded failure."""
        if self.failed is not None:
            return None
        torch = torch or self._t()
        routing_fn, routing_cfg, mlp_norm, pre_norm, hidden = self._r()
        try:
            z = pre.reshape(-1, hidden)
            if not z.is_contiguous():
                z = z.contiguous()
            # The layer's own router input pipeline, with the pre-attention residual substituted for
            # the post-attention one: same norms, same gate, same RoutingCFG. Nothing reimplemented,
            # so a bias or a per-expert scale carries over exactly.
            if mlp_norm is not None:
                z = mlp_norm.forward(z, {}, out_dtype = torch.half)
            if pre_norm is not None:
                z = pre_norm.forward(z, {}, out_dtype = torch.half)
            if z.dtype != torch.half:
                z = z.half()
            sel, _w = routing_fn(z.shape[0], routing_cfg, z, {})
            # The ids have to cross to the host: the prefetch is a CPU-side read. This is the one
            # synchronisation the mechanism costs (see the module docstring).
            return [[int(i) for i in row] for row in sel.detach().to("cpu").tolist()]
        except Exception as exc:                                # noqa: BLE001 - recorded, not swallowed
            return self._fail(f"the lookahead could not re-run the layer's router on the "
                              f"pre-attention residual: {type(exc).__name__}: {exc}")

    def note_true(self, selected_experts):
        """Called at the routing site with the TRUE selection: records how many of the prefetched
        experts were routed. Does nothing unless EXL3_MOE_LOOKAHEAD_STATS is on, because it costs a
        second device-to-host copy per layer per step."""
        if not self.cfg.stats or self.failed is not None or not self._pred_router:
            return
        try:
            true = selected_experts.detach().to("cpu").tolist()
        except Exception as exc:                                # noqa: BLE001
            self._fail(f"the lookahead could not read the true selection: {type(exc).__name__}: {exc}")
            return
        seen = {int(i) for row in true for i in row}
        self.stats.compared(len(self._pred_router), len(self._pred_router & seen))
        # One prediction is scored once: a layer entered twice in a step must not count the same
        # prefetch twice
        self._pred_router = set()

    # -- failure -------------------------------------------------------------------------------
    def _fail(self, reason):
        """Record a per-layer failure once and go quiet. Returns None so callers can `return` it."""
        if self.failed is None:
            self.failed = reason
            self.stats.skip_layer(reason)
        return None


# ---------------------------------------------------------------------------------------- install

def install(mlp, cfg = None, stats_obj = None, layer_idx = None,
            stream_min_rows = None, **kwargs):
    """Attach a lookahead to a loaded split MoE module: the object, or None when the mechanism is off
    or the layer is refused (in which case the refusal is counted by name, never dropped).

    Called from `BlockSparseMLP.load()`; `**kwargs` carries the test's stubs through.
    """
    cfg = cfg if cfg is not None else lookahead_cfg()
    stats_obj = stats_obj if stats_obj is not None else stats()
    if not cfg.enabled:
        return None
    reason = support_reason(mlp)
    if reason is not None:
        stats_obj.skip_layer(reason)
        return None
    if stream_min_rows is None:
        # The engine's own decode/prefill boundary, so the gate agrees with the split's fast path
        stream_min_rows = getattr(getattr(mlp, "cpu_host", None), "stream_min_rows",
                                  DEFAULT_STREAM_MIN_ROWS)
    la = LayerLookahead(mlp, cfg, stats_obj, layer_idx = layer_idx,
                        stream_min_rows = stream_min_rows, **kwargs)
    if la.toucher.reason is not None:
        stats_obj.skip_layer(la.toucher.reason)
        return None
    stats_obj.layers_attached += 1
    if cfg.stats:
        _register_exit_report()
    return la
