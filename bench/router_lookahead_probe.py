#!/usr/bin/env python3
"""Does the residual before attention predict the router's picks? Backlog item #1's measurement.

`wiki/plans/2026-11-04-decode-backlog.md` row 1 proposes prefetching the CPU-resident experts from a
router lookahead computed *before* the attention/GDN sublayer of each block. Its load-bearing claim
has two halves and neither has been measured on this model:

  * the pre- and post-attention residuals are close (>0.92 cosine, the claim);
  * the router's top-k applied to the pre-attention residual overlaps the router's true top-k often
    enough that fetching the predicted experts is cheaper than the misses it saves.

This tool measures both, per block-sparse MoE layer, over one short decode-only generation:

    python bench/router_lookahead_probe.py -m D:\\llms\\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 \
        -cs 8192 -cq 4,4 -chunk_size 4096 -ambs 2 -mcs 381 -mct 16 -tokens 16 --json probe.json

What is recorded, per layer and per decode step (no engine change: four instance-level wrappers, see
`RouterLookaheadProbe._attach`):

  * `pre`  - the residual stream entering the block, i.e. the vector a pre-attention probe would
             have. Captured at the block's `forward`, before anything mutates it in place (the
             block's deferred-residual `params["resid_pending"]` is added first, as the block does).
  * `post` - the residual entering the router: `params["residual"]`, which `TransformerBlock.forward`
             publishes at the MLP site, captured at the MoE module's `forward`.
  * the true picks and weights - the module's own `routing_fn` outputs, cloned at the call (the
             bsz-1 path returns shared buffers, so a clone is mandatory).

The prediction re-applies the layer's own router to `pre`: the same `mlp_norm`, the same
`router_pre_norm`, the same `routing_fn` with the same `RoutingCFG` (so a bias, a scale or a group
rule carries over exactly). Nothing is reimplemented, so the probe cannot drift from the engine.

Refusals are named, never silent. A layer whose selection is not a function of the hidden state (a
`tid2eid` hash router), a module whose router lives on another rank (a TP shard without the gate), a
parallel attn+MLP block (no attention-updated residual exists) or a dense MLP all produce a row with
`status = skipped` and a `reason` string; the corpus line counts them. If no layer can be
instrumented at all the tool exits 2, because a probe that measures nothing and exits 0 reads as a
clean zero - the opposite of a measurement.

No tokenizer and no sampling: the prompt is built from bare in-vocabulary ids and the generation is
greedy, so two runs of the same arm see the same tokens.

What this tool does NOT answer: lead time. It shows whether the picks are predictable from the
pre-attention residual; whether the prefetch can be issued early enough to hide the DMA is a
scheduling measurement, not this one.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Router types whose selection is a function of the hidden state and of tensors already on the
# module (bias, per-expert scale, group rule). All of them are re-run through the engine's own
# routing_fn, so this list gates *reachability*, not an independent reimplementation.
# `sqrtsp_hash` is excluded deliberately: its picks come from a tid2eid table indexed by input_ids.
LINEAR_ROUTER_TYPES = ("std", "std_bias", "dots", "sqrtsp", "ds3")

# The prompt is bare ids: small, in-vocabulary, spread over a few magnitudes so the n-gram/PLE tables
# and the embedding both see real rows. Qwen3.8-Flash-Next's vocab is 248320.
PROMPT_IDS = [1000, 42, 7, 4096, 555, 12, 9001, 300, 66, 1234, 77, 8, 20000, 913, 3, 777]


# --------------------------------------------------------------------------------------- arithmetic
# These helpers are the whole of the measurement's arithmetic, and they are pure so that
# bench/test_router_lookahead_probe.py can hold them to account without torch or a GPU.
#
# Two conventions they all follow, because both were wrong once:
#
#   * "no data" is None, never 0.0. A zero row that contributes 0.0 to a mean reads as a measured
#     disagreement; a dead row has no opinion, so it is excluded and counted (`undefined_rows`).
#   * ids are counted as SETS. A step that routes two tokens into one expert costs one transfer of
#     that expert's bytes, and a top-k cannot repeat an expert inside one token, so the unit of the
#     byte claim is the distinct expert, not the (token, expert) assignment.

def cosine(a, b) -> float | None:
    """Cosine of two equal-length sequences of floats, or None when there is no direction to compare.

    None covers the empty row and the zero vector: a zero row is a dead (or zeroed) activation, and
    reporting 0.0 would put a fabricated "totally different" reading into the layer mean.
    """
    if len(a) != len(b):
        raise ValueError(f"cosine of rows of different width: {len(a)} against {len(b)}")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        x = float(x)
        y = float(y)
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return None
    return dot / math.sqrt(na * nb)


def row_overlap(true_ids, pred_ids) -> float | None:
    """Share of one row's true top-k that the prediction covers, or None when the row picked nothing.

    Sets: the denominator is the number of DISTINCT true picks, so a wider prediction is not rewarded,
    an order change is free, a superset covers fully, and a repeat in either list is counted once.
    """
    true_set = {int(i) for i in true_ids}
    pred_set = {int(i) for i in pred_ids}
    if not true_set:
        return None
    return len(true_set & pred_set) / len(true_set)


def topk_overlap(true_rows, pred_rows) -> list[float | None]:
    """Per-row overlap fractions (None for a row with no picks). A row-count mismatch is a caller bug
    and raises."""
    if len(true_rows) != len(pred_rows):
        raise ValueError(
            f"row-count mismatch: {len(true_rows)} true rows, {len(pred_rows)} predicted rows; "
            f"the pre-attention residual and the router did not see the same tokens")
    return [row_overlap(t, p) for t, p in zip(true_rows, pred_rows)]


def tail_coverage(true_rows, pred_rows, first) -> tuple[int, int]:
    """(distinct picks at or beyond `first`, of those covered by the prediction).

    `first` is the module's `cpu_split_first`: with a per-layer expert split, router ids from `first`
    on are the CPU-resident tail, i.e. the ones a prefetch would have to fetch. `first = None` (no
    split) counts nothing, because on a fully GPU-resident layer a prefetch has no miss to cover.

    Set semantics, as in `row_overlap`, and here the reason is the byte claim itself: within one
    forward, every token that routes the same expert shares ONE transfer of that expert's bytes, so
    the count is over distinct ids, not over (token, expert) assignments. Decode hands the probe one
    token per step, where the two counts are identical; a wider row would otherwise inflate the
    number a prefetch is credited with.
    """
    if first is None:
        return (0, 0)
    true_set = {int(i) for row in true_rows for i in row if int(i) >= int(first)}
    pred_set = {int(i) for row in pred_rows for i in row}
    return (len(true_set), len(true_set & pred_set))


def router_support_reason(mlp) -> str | None:
    """Why this layer's router cannot be probed from the pre-attention residual, or None.

    Attribute-only, so the refusal is testable without the engine. The reasons name the cause: a
    layer that is skipped because routing is not a hidden-state function must not look like a layer
    that measured zero overlap.
    """
    if getattr(mlp, "routing_gate", None) is None:
        return ("no router on this module: the selection arrives by broadcast from the routing rank "
                "(a tensor-parallel shard without the gate), so there is no hidden-state function here")
    if getattr(mlp, "tid2eid", None) is not None or getattr(mlp, "router_type", "std") == "sqrtsp_hash":
        return ("hash-selected router: the picks are read from a tid2eid table indexed by input_ids, not "
                "computed from the hidden state, so a pre-attention probe cannot predict them")
    rtype = getattr(mlp, "router_type", None)
    if rtype not in LINEAR_ROUTER_TYPES:
        return (f"router type {rtype!r} is not one of {LINEAR_ROUTER_TYPES}: its selection is not a "
                f"plain gate over the hidden state, so the probe has no router to re-apply")
    if getattr(mlp, "alt_residual_channel", False):
        return ("alt_residual_channel: the router reads params['residual'] instead of the block input, "
                "so the pre-attention residual is not the vector a lookahead would give the router")
    return None


def _wmean(pairs) -> float | None:
    """Weighted mean over (value, weight) pairs, skipping undefined values and zero weights."""
    num = den = 0.0
    for value, weight in pairs:
        if value is None or not weight:
            continue
        num += float(value) * float(weight)
        den += float(weight)
    return (num / den) if den else None


@dataclass
class LayerStats:
    """One MoE layer's accumulated rows. Pure data: the probe fills it, `finished_row` renders it.

    Two counting bases, kept apart on purpose. `rows`/`overlap_sum`/`cos_sum` are per ROW (a row is
    one token), and a row with no data contributes nothing (`*_rows` counts only the defined rows, so
    the undefined ones show up in `undefined_rows`). `picks`/`covered` are per STEP over distinct
    experts, which is the unit of a transfer: that is the byte claim, not a prediction-quality one.
    """

    layer_idx: int
    name: str
    status: str = "measured"          # "measured" | "skipped"
    reason: str | None = None
    router_type: str | None = None
    num_experts: int = 0
    top_k: int = 0
    cpu_split_first: int | None = None
    steps: list = field(default_factory=list)
    rows: int = 0
    cos_sum: float = 0.0
    cos_rows: int = 0
    cos_min: float | None = None
    zcos_sum: float = 0.0
    zcos_rows: int = 0
    overlap_sum: float = 0.0
    overlap_rows: int = 0
    top1_hits: int = 0
    exact_rows: int = 0
    picks: int = 0                    # distinct experts the steps routed
    covered: int = 0                  # of those, distinct experts the prediction named
    tail_picks: int = 0               # distinct CPU-tail experts the steps routed
    tail_hits: int = 0
    failures: list = field(default_factory=list)   # step-level reasons, deduped by the caller

    def finished_row(self) -> dict:
        row = {
            "layer": self.layer_idx,
            "name": self.name,
            "status": self.status,
            "router_type": self.router_type,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "cpu_split_first": self.cpu_split_first,
            "cpu_split_slot": getattr(self, "cpu_split_slot", None),
            "cpu_split_spares": getattr(self, "cpu_split_spares", 0),
            "steps": len(self.steps),
            "rows": self.rows,
            "picks": self.picks,
            "covered": self.covered,
            "counts": {
                "rows": self.rows,
                "cos_sum": self.cos_sum,
                "cos_rows": self.cos_rows,
                "zcos_sum": self.zcos_sum,
                "zcos_rows": self.zcos_rows,
                "overlap_sum": self.overlap_sum,
                "overlap_rows": self.overlap_rows,
                "top1_hits": self.top1_hits,
                "exact_rows": self.exact_rows,
                "picks": self.picks,
                "covered": self.covered,
                "tail_picks": self.tail_picks,
                "tail_hits": self.tail_hits,
            },
        }
        if self.status != "measured":
            row["reason"] = self.reason
            return row
        row.update({
            "cosine_mean": (self.cos_sum / self.cos_rows) if self.cos_rows else None,
            "cosine_min": self.cos_min if self.cos_rows else None,
            "router_input_cosine_mean": (self.zcos_sum / self.zcos_rows) if self.zcos_rows else None,
            "overlap_mean": (self.overlap_sum / self.overlap_rows) if self.overlap_rows else None,
            "miss_rate": (1.0 - self.overlap_sum / self.overlap_rows) if self.overlap_rows else None,
            "byte_share": (self.covered / self.picks) if self.picks else None,
            "top1_agreement": (self.top1_hits / self.rows) if self.rows else None,
            "exact_row_share": (self.exact_rows / self.overlap_rows) if self.overlap_rows else None,
            "undefined_rows": {
                "cosine": self.rows - self.cos_rows,
                "overlap": self.rows - self.overlap_rows,
            },
            "tail_picks": self.tail_picks,
            "tail_hits": self.tail_hits,
            "tail_overlap": (self.tail_hits / self.tail_picks) if self.tail_picks else None,
            "reason": self.reason,
            "step_failures": self.failures,
            "step_rows": self.steps,
        })
        return row


def corpus_summary(rows) -> dict:
    """The corpus line, from the row counts only - every number is a ratio of sums, never an average
    of per-layer averages, because the layers do not carry equal weight.

    `overlap_mean` is the per-row (per-token) prediction quality. `byte_share_covered` is the transfer
    claim: distinct (step, expert) pairs the prediction named, over the distinct pairs the steps
    routed. On this checkpoint every routed expert of every layer is the same size, so that share is
    also the share of routed bytes a lookahead prefetch would cover. `cpu_tail_overlap` is the same
    ratio restricted to the CPU-resident picks a per-layer split actually pays for.
    """
    measured = [r for r in rows if r.get("status") == "measured"]
    skipped = [r for r in rows if r.get("status") != "measured"]
    with_rows = [r for r in measured if (r.get("counts") or {}).get("rows")]

    def c(key):
        return sum((r.get("counts") or {}).get(key, 0) for r in with_rows)

    rows_n, cos_rows, overlap_rows = c("rows"), c("cos_rows"), c("overlap_rows")
    picks, covered = c("picks"), c("covered")
    tail_picks, tail_hits = c("tail_picks"), c("tail_hits")
    overlap = (c("overlap_sum") / overlap_rows) if overlap_rows else None
    byte_share = (covered / picks) if picks else None
    return {
        "layers_discovered": len(rows),
        "layers_measured": len(measured),
        "layers_with_rows": len(with_rows),
        "layers_skipped": len(skipped),
        "skipped_layers": [r.get("layer") for r in skipped],
        "skipped_reasons": sorted({r["reason"] for r in skipped if r.get("reason")}),
        "rows": rows_n,
        "picks": picks,
        "covered": covered,
        # Rows that had no data (a zero vector, an empty pick list): excluded from the means above,
        # counted here so a layer of dead rows cannot look like a measured disagreement.
        "undefined_rows": {
            "cosine": rows_n - cos_rows,
            "overlap": rows_n - overlap_rows,
        },
        "cosine_mean": (c("cos_sum") / cos_rows) if cos_rows else None,
        # A layer can be measured and still carry no cosine_min (every one of its rows was
        # undefined), and min() over an empty sequence raises - which would crash the summary on
        # exactly the input the verdict below exists to refuse.
        "cosine_min": (min([r["cosine_min"] for r in with_rows
                            if r.get("cosine_min") is not None] or [None])
                       if with_rows else None),
        "router_input_cosine_mean": (c("zcos_sum") / c("zcos_rows")) if c("zcos_rows") else None,
        "overlap_mean": overlap,
        "miss_rate": (1.0 - overlap) if overlap is not None else None,
        "byte_share_covered": byte_share,
        # Bytes fetched per byte of useful fetch: k predictions to cover byte_share*k of them
        "prefetch_redundancy": (1.0 / byte_share) if byte_share else None,
        "top1_agreement": (c("top1_hits") / rows_n) if rows_n else None,
        "exact_row_share": (c("exact_rows") / overlap_rows) if overlap_rows else None,
        "cpu_tail_pick_share": (tail_picks / picks) if picks else None,
        "cpu_tail_picks": tail_picks,
        "cpu_tail_overlap": (tail_hits / tail_picks) if tail_picks else None,
    }


def probe_verdict(instrumented: int, rows) -> tuple[int, str | None]:
    """The exit code the probe must return, and the line that says which condition fired.

    Both refusals are here rather than inline in `run`, so the offline test can hold them without the
    engine and so the two messages cannot drift. A probe that measured nothing must not exit 0.
    """
    if instrumented == 0:
        return 2, ("condition `instrumented == 0`: no block-sparse MoE layer could be instrumented, so "
                   "nothing was measured. This is not a zero overlap - it is a broken probe.")
    summary = corpus_summary(rows)
    if summary["layers_with_rows"] == 0:
        return 2, ("condition `layers_with_rows == 0`: the generation ran, every instrumented layer "
                   "was entered, and not one step produced a comparison. Not a zero overlap - a "
                   "broken probe.")
    return 0, None


def build_prompt_ids(length: int) -> list[int]:
    """A fixed prompt of bare ids, repeated to `length`. No tokenizer: two runs see the same ids."""
    length = max(int(length), 1)
    out = []
    while len(out) < length:
        out.extend(PROMPT_IDS)
    return out[:length]


# ------------------------------------------------------------------------------ engine instrumentation

class RouterLookaheadProbe:
    """Instance-level wrappers over the loaded model. The engine is not edited.

    Three wraps per block-sparse MoE layer:

      1. `block.forward`   - records the residual entering the block (adding the previous block's
                             deferred `resid_pending` first, exactly as the block will), because the
                             tensor is mutated in place later in the same call.
      2. `mlp.forward`     - records `params["residual"]` (the post-attention residual the router is
                             fed from) and, *after* the forward returns, runs the prediction. After,
                             because the bsz-1 router path hands back shared `RoutingCFG` buffers:
                             re-running the router before the expert compute consumed them would
                             overwrite the live selection and change the generated tokens.
      3. `mlp.routing_fn`  - records the true picks/weights/router input, cloned (shared buffers).
    """

    def __init__(self, model, dump_ids: bool = False):
        self.model = model
        self.dump_ids = dump_ids
        self.stats: list[LayerStats] = []
        self.step = 0
        self.active = False
        self._attach(model)

    # -- attachment ---------------------------------------------------------------------------
    def _attach(self, model):
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
        from exllamav3.modules.transformer import ParallelDecoderBlock, TransformerBlock

        counter = 0
        for block in model.modules:
            if not hasattr(block, "mlp"):
                continue                      # final norm / head: not a layer
            mlp = block.mlp
            if not isinstance(mlp, BlockSparseMLP):
                self.stats.append(LayerStats(
                    layer_idx=(getattr(block, "layer_idx", None)
                               if getattr(block, "layer_idx", None) is not None else len(self.stats)),
                    name=getattr(block, "key", "?"), status="skipped",
                    reason=f"dense MLP ({type(mlp).__name__}): this layer routes nothing, so there is "
                           f"no router to predict"))
                continue

            layer_idx = getattr(block, "layer_idx", None)
            if layer_idx is None:
                layer_idx = counter
            counter += 1
            # The CPU tail by ROUTER id: `cpu_post_load` permutes the gate columns into the placement
            # order, so ids from `cpu_split_first` on are the worker's. With promote-on-miss the
            # `_split_spares` slots just below `cpu_split_first` are GPU slots that hold CPU-resident
            # experts at load, so the tail starts at `cpu_split_first - spares`.
            split_slot = getattr(mlp, "cpu_split_first", None)
            spares = int(getattr(mlp, "_split_spares", 0) or 0)
            cpu_first = None if split_slot is None else max(0, int(split_slot) - spares)
            st = LayerStats(
                layer_idx=layer_idx,
                name=mlp.key,
                router_type=getattr(mlp, "router_type", None),
                num_experts=mlp.num_experts,
                top_k=mlp.num_experts_per_tok,
                cpu_split_first=cpu_first,
            )
            st.cpu_split_slot = split_slot
            st.cpu_split_spares = spares
            st.hidden_size = mlp.hidden_size            # not rendered; used by the prediction
            st.mlp_norm = getattr(block, "mlp_norm", None)
            st.router_pre_norm = mlp.router_pre_norm
            st.orig_routing = mlp.routing_fn
            st.routing_cfg = mlp.routing_cfg
            st.probing = False                          # set while the probe itself calls the router
            st.pre = st.post = st.sel = st.weights = st.router_in = None
            st.pre_step = -1

            if isinstance(block, ParallelDecoderBlock):
                st.status, st.reason = "skipped", (
                    "parallel attn+MLP block: the MLP reads the block input, so no attention-updated "
                    "residual exists to compare against (pre == post by construction)")
            else:
                if not isinstance(block, TransformerBlock):
                    st.status, st.reason = "skipped", (
                        f"{type(block).__name__} is not a TransformerBlock: the pre-attention residual "
                        f"is not reachable from this block's forward without an engine change")
                else:
                    st.reason = router_support_reason(mlp)
                    if st.reason is not None:
                        st.status = "skipped"

            self.stats.append(st)
            if st.status != "measured":
                continue
            self._wrap_block(block, st)
            self._wrap_mlp(mlp, st)
            self._wrap_routing(mlp, st)

    def _wrap_block(self, block, st):
        probe = self
        orig = block.forward

        def forward(x, params, out_dtype = None):
            if probe.active:
                pending = params.get("resid_pending")
                pre = x if pending is None else x + pending
                st.pre = pre.detach().clone()
                st.pre_step = probe.step
            return orig(x, params, out_dtype)

        block.forward = forward

    def _wrap_mlp(self, mlp, st):
        probe = self
        orig = mlp.forward

        def forward(x, params, out_dtype = None):
            if not probe.active:
                return orig(x, params, out_dtype)
            residual = params.get("residual")
            st.post = None if residual is None else residual.detach().clone()
            st.sel = st.weights = st.router_in = None
            out = orig(x, params, out_dtype)
            probe._evaluate(st)
            return out

        st._probe = probe
        mlp.forward = forward

    def _wrap_routing(self, mlp, st):
        probe = self
        orig = mlp.routing_fn

        def routing(bsz, cfg, z, params):
            sel, weights = orig(bsz, cfg, z, params)
            if probe.active and not st.probing:
                st.sel = sel.detach().clone()
                st.weights = weights.detach().clone()
                st.router_in = z.detach().clone()
            return sel, weights

        mlp.routing_fn = routing

    # -- measurement --------------------------------------------------------------------------
    def _fail(self, st: LayerStats, reason: str):
        if reason not in st.failures:
            st.failures.append(reason)
        if st.rows == 0:
            st.status, st.reason = "skipped", reason

    def _evaluate(self, st: LayerStats):
        import torch

        if st.sel is None:
            self._fail(st, "the router did not run in this forward: the selection was broadcast or "
                            "the module took a path that skips routing")
            return
        if getattr(st, "pre", None) is None or st.pre_step != self.step:
            self._fail(st, "no pre-attention residual was captured for this step: the block forward "
                           "the probe wraps was not entered (or is not the module that owns this MLP)")
            return
        if getattr(st, "post", None) is None:
            self._fail(st, "the block did not publish params['residual'] at the MLP site, so the "
                           "post-attention residual is not reachable")
            return

        H = st.hidden_size
        pre = st.pre.reshape(-1, H)
        post = st.post.reshape(-1, H)
        if pre.shape[0] != st.sel.shape[0]:
            self._fail(st, f"row mismatch: {pre.shape[0]} pre-attention rows against "
                           f"{st.sel.shape[0]} routed rows")
            return

        # The layer's own router input pipeline, with the pre-attention residual substituted for the
        # post-attention one. Same norm stack, same gate, same RoutingCFG - nothing reimplemented.
        z_pre = pre.contiguous()
        try:
            if st.mlp_norm is not None:
                z_pre = st.mlp_norm.forward(z_pre, {}, out_dtype = torch.half)
            if st.router_pre_norm is not None:
                z_pre = st.router_pre_norm.forward(z_pre, {}, out_dtype = torch.half)
            st.probing = True
            try:
                pred_sel, _pred_w = st.orig_routing(z_pre.shape[0], st.routing_cfg, z_pre, {})
            finally:
                st.probing = False
            pred_sel = pred_sel.detach().float().cpu()
        except Exception as exc:                                  # noqa: BLE001 - recorded, not swallowed
            self._fail(st, f"the probe could not re-run the layer's router on the pre-attention "
                           f"residual: {type(exc).__name__}: {exc}")
            return

        true_rows = st.sel.detach().cpu().tolist()
        pred_rows = pred_sel.long().tolist()
        if not true_rows:
            self._fail(st, "the router returned zero rows for this forward: there is nothing to "
                           "compare against")
            return
        try:
            overlaps = topk_overlap(true_rows, pred_rows)
        except ValueError as exc:
            self._fail(st, str(exc))
            return

        cos = [cosine(a, b) for a, b in zip(pre.float().cpu().tolist(), post.float().cpu().tolist())]
        zcos = ([] if st.router_in is None else
                [cosine(a, b) for a, b in zip(z_pre.float().cpu().tolist(),
                                              st.router_in.float().cpu().tolist())])
        tail_picks, tail_hits = tail_coverage(true_rows, pred_rows, st.cpu_split_first)
        top1 = sum(1 for t, p in zip(true_rows, pred_rows) if t and p and int(t[0]) == int(p[0]))
        # A row with no data (a zero vector, an empty pick list) has no opinion: it is left out of the
        # means and shows up as an undefined row, rather than dragging a mean towards zero.
        cos_valid = [c for c in cos if c is not None]
        zcos_valid = [c for c in zcos if c is not None] if len(zcos) == len(cos) else []
        ov_valid = [o for o in overlaps if o is not None]
        exact = sum(1 for o in ov_valid if o >= 1.0 - 1e-9)
        # Distinct-expert counts for the transfer claim (one transfer per step and expert)
        true_union = {int(i) for row in true_rows for i in row}
        pred_union = {int(i) for row in pred_rows for i in row}

        st.rows += len(true_rows)
        st.cos_sum += sum(cos_valid)
        st.cos_rows += len(cos_valid)
        if cos_valid:
            step_min = min(cos_valid)
            st.cos_min = step_min if st.cos_min is None else min(st.cos_min, step_min)
        st.zcos_sum += sum(zcos_valid)
        st.zcos_rows += len(zcos_valid)
        st.overlap_sum += sum(ov_valid)
        st.overlap_rows += len(ov_valid)
        st.top1_hits += top1
        st.exact_rows += exact
        st.picks += len(true_union)
        st.covered += len(true_union & pred_union)
        st.tail_picks += tail_picks
        st.tail_hits += tail_hits

        entry = {
            "step": self.step,
            "rows": len(true_rows),
            "cosine": (sum(cos_valid) / len(cos_valid)) if cos_valid else None,
            "router_input_cosine": (sum(zcos_valid) / len(zcos_valid)) if zcos_valid else None,
            "overlap": (sum(ov_valid) / len(ov_valid)) if ov_valid else None,
            "byte_share": (len(true_union & pred_union) / len(true_union)) if true_union else None,
            "top1_agreement": top1 / len(true_rows),
            "undefined_rows": {
                "cosine": len(cos) - len(cos_valid),
                "overlap": len(overlaps) - len(ov_valid),
            },
            "tail_picks": tail_picks,
            "tail_covered": tail_hits,
        }
        if self.dump_ids:
            entry["true_ids"] = [[int(i) for i in r] for r in true_rows]
            entry["pred_ids"] = [[int(i) for i in r] for r in pred_rows]
            entry["true_weights"] = [[float(w) for w in r]
                                     for r in st.weights.detach().float().cpu().tolist()]
        st.steps.append(entry)

    # -- reporting ----------------------------------------------------------------------------
    def rows(self) -> list[dict]:
        return [st.finished_row() for st in self.stats]

    def instrumented(self) -> int:
        return sum(1 for st in self.stats if st.status == "measured")

    def attach_report(self) -> str:
        lines = [
            f" -- instrumented {self.instrumented()} layer(s) of {len(self.stats)} discovered: "
            f"pre = residual at block.forward, post = params['residual'] at mlp.forward, "
            f"picks = routing_fn outputs (cloned)",
        ]
        reasons = {}
        for st in self.stats:
            if st.status != "measured":
                reasons.setdefault(st.reason or "unknown", []).append(st.layer_idx)
        for reason, layers in reasons.items():
            lines.append(f" -- skipped {len(layers)} layer(s) [{layers[0]}..{layers[-1]}]: {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------- the run

def _arm(probe: RouterLookaheadProbe, step: int):
    probe.step = step
    probe.active = True


def _disarm(probe: RouterLookaheadProbe):
    probe.active = False


def run(args) -> int:
    import torch
    from exllamav3 import model_init

    model, config, cache, tokenizer = model_init.init(args, load_tokenizer = False)
    try:
        bpw_layer, bpw_head, _ = model.get_storage_info()
    except Exception:                                             # noqa: BLE001
        bpw_layer = bpw_head = None

    probe = RouterLookaheadProbe(model, dump_ids = args.dump_ids)
    print(probe.attach_report())
    if probe.instrumented() == 0:
        print(" !! no block-sparse MoE layer could be instrumented, so nothing was measured. This is "
              "not a zero overlap: it is a broken probe.", file = sys.stderr)
        for st in probe.stats:
            print(f"    layer {st.layer_idx}: {st.reason}", file = sys.stderr)
        return 2

    prompt = build_prompt_ids(args.prompt_len)
    decode = int(args.tokens)
    is_recurrent = model.caps.get("recurrent_states", False)
    prefill_ids = prompt[:-1]
    past = len(prefill_ids)
    # the cache allocates in 256-token pages and asserts the shape is a whole number of them
    batch_shape = (1, max(256, ((past + decode + 256 + 255) // 256) * 256))

    generated = []
    with torch.inference_mode():
        # the recurrent state is created from inference tensors, and its constructor zeroes them in
        # place: outside InferenceMode that raises (first real run, 2026-11-05)
        recurrent = [cache.get_test_state(0)] if is_recurrent else None
        params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": batch_shape,
            "recurrent_states": recurrent,
        }
        model.prefill(torch.tensor([prefill_ids], dtype = torch.long), params)

        token = prompt[-1]
        for step in range(decode):
            params = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": past + step,
                "batch_shape": batch_shape,
                "recurrent_states": recurrent,
            }
            ids = torch.tensor([[token]], dtype = torch.long)
            _arm(probe, step)
            try:
                logits = model.forward(ids, params)
            finally:
                _disarm(probe)
            token = int(torch.argmax(logits.reshape(-1, logits.shape[-1])[-1]).item())
            generated.append(token)
        if is_recurrent:
            recurrent[0].free()

    rows = probe.rows()
    summary = corpus_summary(rows)
    print_table(rows, summary)

    report = {
        "tool": "router_lookahead_probe",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec = "seconds"),
        "question": "does the pre-attention residual predict the router's picks?",
        "model_dir": args.model_dir,
        "harness": {
            "decode_steps": decode,
            "prompt_tokens": len(prompt),
            "sampler": "greedy (argmax)",
            "prompt_ids": prompt,
            "generated_ids": generated,
            "cache_size": getattr(args, "cache_size", None),
            "cache_quant": getattr(args, "cache_quant", None),
            "moe_cpu_split": getattr(args, "moe_cpu_split", 0),
            "moe_cpu_threads": getattr(args, "moe_cpu_threads", None),
            "chunk_size": getattr(args, "chunk_size", None),
            "autosplit_max_batch_size": getattr(args, "autosplit_max_batch_size", None),
            "bpw_layer": bpw_layer,
            "bpw_head": bpw_head,
        },
        "router_types": sorted({str(st.router_type) for st in probe.stats if st.status == "measured"}),
        "layers": rows,
        "corpus": summary,
        "notes": [
            "overlap_mean is |true top-k AND predicted top-k| / k per row, averaged over rows",
            "corpus.overlap_mean is pick-weighted, i.e. the share of routed-expert BYTES a lookahead "
            "prefetch would cover: every routed pick is one expert of the same size on every layer",
            "cpu_tail_overlap restricts the picks to router ids >= cpu_split_first, i.e. the picks a "
            "per-layer expert split must fetch from host RAM; that is the number a prefetch pays for",
            "router ids are the module's own (post-permutation placement) ids; cpu_split_first is the "
            "load-time boundary (cpu_split_slot - spares), and dynamic placement can move experts across "
            "it during the run",
            "pre == the residual entering the block (after the previous block's deferred residual add), "
            "post == params['residual'] at the MLP site",
            "the prediction re-runs the layer's own routing_fn on norm(pre) - no reimplementation",
        ],
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent = 2)
        print(f" -- wrote {args.json}")

    if summary["layers_with_rows"] == 0:
        print(" !! every instrumented layer produced zero rows: the generation ran but nothing was "
              "recorded. Not a zero overlap - a broken probe.", file = sys.stderr)
        return 2
    if summary["layers_with_rows"] < summary["layers_discovered"]:
        print(f" !! {summary['layers_skipped']} layer(s) skipped, {summary['layers_with_rows']} "
              f"measured: the corpus line covers only the layers named in the rows above",
              file = sys.stderr)
    return 0


def _fmt(value, spec = ".3f", width = 0) -> str:
    s = "-" if value is None else format(value, spec)
    return s.rjust(width) if width else s


def print_table(rows, summary):
    header = (f"{'layer':>5}  {'name':<28} {'router':<9} {'rows':>4} {'cos(pre,post)':>13} "
              f"{'cos(router)':>11} {'overlap':>7} {'miss':>6} {'top1':>5} {'tail':>5} {'tail_cov':>8}")
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] != "measured":
            print(f"{r['layer']:>5}  {r['name']:<28} {str(r.get('router_type')):<9} "
                  f"{'-':>4}  skipped: {r.get('reason')}")
            continue
        print(f"{r['layer']:>5}  {r['name']:<28} {str(r.get('router_type')):<9} {r['rows']:>4} "
              f"{_fmt(r['cosine_mean'], '.4f', 13)} {_fmt(r['router_input_cosine_mean'], '.4f', 11)} "
              f"{_fmt(r['overlap_mean'], '.4f', 7)} {_fmt(r['miss_rate'], '.4f', 6)} "
              f"{_fmt(r['top1_agreement'], '.3f', 5)} {r['tail_picks']:>5} "
              f"{_fmt(r['tail_overlap'], '.3f', 8)}")
        if r.get("step_failures"):
            print(f"        !! {len(r['step_failures'])} step-level failure(s): {r['step_failures'][0]}")
    print()
    print(
        "CORPUS "
        f"layers={summary['layers_with_rows']}/{summary['layers_discovered']} "
        f"skipped={summary['layers_skipped']} picks={summary['picks']} "
        f"cosine={_fmt(summary['cosine_mean'], '.4f')} "
        f"cosine_min={_fmt(summary['cosine_min'], '.4f')} "
        f"router_input_cosine={_fmt(summary['router_input_cosine_mean'], '.4f')} "
        f"overlap={_fmt(summary['overlap_mean'], '.4f')} "
        f"miss={_fmt(summary['miss_rate'], '.4f')} "
        f"redundancy={_fmt(summary['prefetch_redundancy'], '.2f')}x "
        f"top1={_fmt(summary['top1_agreement'], '.3f')}"
    )
    print(
        "CORPUS "
        f"byte_share_covered={_fmt(summary['overlap_mean'], '.4f')} "
        f"cpu_tail_pick_share={_fmt(summary['cpu_tail_pick_share'], '.4f')} "
        f"cpu_tail_overlap={_fmt(summary['cpu_tail_overlap'], '.4f')} "
        f"cpu_tail_picks={summary['cpu_tail_picks']}"
    )
    if summary["skipped_reasons"]:
        for reason in summary["skipped_reasons"]:
            print(f"SKIPPED {reason}")
    print()
    print(" byte_share_covered = the share of routed-expert bytes a lookahead prefetch would reach; "
          "cpu_tail_overlap = the share of the SPLIT's host-resident picks it would reach "
          "(the number a per-layer expert split pays for).")


# ------------------------------------------------------------------------------------------- CLI

def _loader_fallbacks(parser):
    """The loader flags this tool takes straight from `exllamav3.model_init.add_args`, declared as
    stubs only when the engine cannot be imported (the Linux side of this repo has no torch)."""
    for flag, kw in (
        ("-m", dict(dest = "model_dir", type = str, default = None)),
        ("-gs", dict(dest = "gpu_split", type = str, default = None)),
        ("-cs", dict(dest = "cache_size", type = int, default = 8192)),
        ("-cq", dict(dest = "cache_quant", type = str, default = None)),
        ("-chunk_size", dict(dest = "chunk_size", type = int, default = 4096)),
        ("-ambs", dict(dest = "autosplit_max_batch_size", type = int, default = 1)),
        ("-mcs", dict(dest = "moe_cpu_split", type = int, default = 0)),
        ("-mct", dict(dest = "moe_cpu_threads", type = int, default = None)),
        ("-nw", dict(dest = "no_warmup", action = "store_true")),
    ):
        parser.add_argument(flag, help = "engine loader flag (this box cannot import the engine; "
                                         "the real definition comes from model_init.add_args)", **kw)


def main(argv = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Measure whether a pre-attention router lookahead predicts the router's picks "
                      "(backlog item 1 of 2026-11-04-decode-backlog.md).",
        allow_abbrev = False,
        epilog = "Exits 2 when no layer can be instrumented: a probe that measured nothing must not "
                 "read as a clean zero.",
    )
    have_engine = True
    try:
        from exllamav3 import model_init
    except Exception as exc:                                       # noqa: BLE001
        have_engine = False
        engine_error = f"{type(exc).__name__}: {exc}"
        _loader_fallbacks(parser)
    else:
        model_init.add_args(parser, default_cache_size = 8192, default_autosplit_max_batch_size = 1)

    parser.add_argument("-tokens", "--tokens", type = int, default = 16,
                        help = "decode steps to record (default 16)")
    parser.add_argument("-plen", "--prompt_len", type = int, default = 64,
                        help = "prefill length of the synthetic id prompt (default 64)")
    parser.add_argument("--json", type = str, default = None, metavar = "FILE",
                        help = "write the rows and the corpus summary to FILE as JSON")
    parser.add_argument("--dump-ids", action = "store_true",
                        help = "include the per-step true/predicted ids and the true weights in the "
                               "JSON (large, but it is the raw evidence)")
    args = parser.parse_args(argv)

    if not have_engine:
        print(f" !! this probe loads the model through exllamav3.model_init, and the engine is not "
              f"importable in this interpreter ({engine_error}). Run it on the Windows box, in the "
              f"engine's venv.", file = sys.stderr)
        return 2
    if not args.model_dir:
        parser.error("-m/--model_dir is required")
    if args.prompt_len < 2:
        parser.error("-plen/--prompt_len must be at least 2: the last prompt id is the first decode row")
    if args.tokens < 1:
        parser.error("-tokens/--tokens must be at least 1")

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
