"""
Specter-style MoE offloading: static budget / break-even calculator.

Answers the sizing questions that decide whether a "Specter"-shaped design
(quantized draft as a multi-step router look-ahead, expert weights streamed
host->device ahead of verification) can pay off for a given MoE model on a
given box. Pure arithmetic, no torch, no GPU, no model download: feed it the
model geometry and the box's link/CPU characteristics and it prints the
budgets that decide the design.

What it is for
--------------
The Specter paper (arXiv:2511.14102) hides PCIe weight-transfer latency behind
a multi-step speculative look-ahead. That only pays off if the weights
actually have to cross the bus. This script makes the premise explicit for
any model by contrasting two offload shapes:

  * "compute on CPU"  (exllamav3's current CPU MoE offload): the expert stays
    in host RAM, the activation goes down and the result comes back. PCIe
    traffic per token is ~2 x hidden_size x 2 bytes per MoE layer, and the
    cost is host compute.

  * "stream to GPU"   (Specter, Mixtral-Offloading, KTransformers-style): the
    expert weights cross the bus and the GPU computes them. PCIe traffic per
    token is bytes_per_expert x (offloaded experts per layer) x layers.

For Qwen3.8-Flash-Next the second shape moves ~2500x more bytes per token
than the first, which is the core of the feasibility verdict in
doc/specter-feasibility.md.

Usage
-----
    python eval/specter_budget.py
    python eval/specter_budget.py --preset qwen3.8-flash-next --vram-gb 24 \
        --link-gbps 25 --cpu-threads 12
    python eval/specter_budget.py --preset dsv2-lite --json

The CPU numbers are a model, not a measurement. Override them with
--cpu-us-per-expert once the 4090 box has been measured (EXL3_MOE_CPU_PROF),
or with --dequant-gbs-per-thread / --gemv-gflops-per-thread.
"""

import argparse
import json
import sys

# --------------------------------------------------------------------------
# Model geometry
# --------------------------------------------------------------------------
# dims: hidden, moe_intermediate, experts, top_k, shared_expert_intermediate,
#       layers, n_dense_layers (layers without MoE), attention/other params
PRESETS = {
    "qwen3.8-flash-next": {
        "label": "Qwen3.8-Flash-Next (Qwen4Exp) - the target of this study",
        "hidden": 2560,
        "moe_interm": 640,
        "experts": 512,
        "top_k": 10,
        "shared_interm": 640,
        "layers": 48,
        "other_params": 2.6e9,      # attention + GDN + gates + norms, 12 QSA + 36 GDN layers
        "embed_params": 2 * 248320 * 2560,  # token embedding + untied lm_head
        "mtp_params": 2.5e9,        # the 1-layer MTP head (its own 512-expert MoE)
        "ngram_rows": 320_000_000,  # PLE n-gram table, 160 dims/row
        "ngram_dim": 160,
    },
    "dsv2-lite": {
        "label": "DeepSeek-V2-Lite (Specter paper)",
        "hidden": 2048,
        "moe_interm": 1408,
        "experts": 64,
        "top_k": 6,
        "shared_interm": 4096,
        "layers": 26,
        "other_params": 1.5e9,
        "embed_params": 2 * 102400 * 2048,
        "mtp_params": 0.0,
        "ngram_rows": 0,
        "ngram_dim": 0,
    },
    "qwen1.5-moe": {
        "label": "Qwen1.5-MoE (Specter paper)",
        "hidden": 2048,
        "moe_interm": 1408,
        "experts": 60,
        "top_k": 4,
        "shared_interm": 4096,
        "layers": 24,
        "other_params": 1.0e9,
        "embed_params": 2 * 151936 * 2048,
        "mtp_params": 0.0,
        "ngram_rows": 0,
        "ngram_dim": 0,
    },
    "phi3.5-moe": {
        "label": "Phi-3.5-MoE (Specter paper)",
        "hidden": 4096,
        "moe_interm": 6400,
        "experts": 16,
        "top_k": 2,
        "shared_interm": 0,
        "layers": 32,
        "other_params": 2.0e9,
        "embed_params": 2 * 32064 * 4096,
        "mtp_params": 0.0,
        "ngram_rows": 0,
        "ngram_dim": 0,
    },
}

# Measured host->device link bandwidths on the box the fork was tuned on
# (see the stream_t calibration in exllamav3/model/moe_cpu_host.py).
LINK_GBPS = {
    "gen5 x16": 57.0,
    "gen5 x8": 29.0,
    "gen4 x16": 15.8,
    "gen4 x4": 6.7,
}

# The paper's measured expert-activation fidelity between an INT4 draft and the
# FP16 target, and the measured distinct-expert working set at K=16.
PAPER_TOPK_MATCH = 0.908        # Qwen1.5-MoE, INT4 GPTQ draft vs FP16 target
PAPER_DSV2_DISTINCT_K16 = 31.7  # distinct experts per layer, DeepSeek-V2-Lite
PAPER_DSV2_DISTINCT_K1 = 6.0


def routed_expert_params(geo):
    """Parameters in one routed expert (gate + up + down)."""
    return 3 * geo["hidden"] * geo["moe_interm"]


def model_budget(geo, bpw):
    """Parameter counts and byte footprints."""
    p_expert = routed_expert_params(geo)
    per_layer = p_expert * geo["experts"]
    routed = per_layer * geo["layers"]
    shared = (3 * geo["hidden"] * geo["shared_interm"] * geo["layers"]
              if geo["shared_interm"] else 0.0)
    other = geo["other_params"]
    embed = geo["embed_params"]
    total = routed + shared + other + embed
    byte = bpw / 8.0
    return {
        "expert_params": p_expert,
        "routed_expert_params": routed,
        "shared_expert_params": shared,
        "total_params": total,
        "bytes_per_expert": p_expert * byte,
        "routed_expert_bytes": routed * byte,
        "model_bytes": total * byte,
        "mtp_bytes": geo["mtp_params"] * byte,
        "ngram_table_bytes_fp16": geo["ngram_rows"] * geo["ngram_dim"] * 2,
    }


def distinct_experts(experts, top_k, depth, skew):
    """Distinct experts per layer after `depth` speculated tokens.

    Real MoE routing is heavily skewed: the paper measured 31.7 distinct
    experts per layer at K=16 on DeepSeek-V2-Lite (64 experts, top-6) where a
    uniform router would give ~64. This recurrence reproduces that with a
    single concentration factor `skew`: a fresh token contributes
    top_k * skew * (1 - S/E) new experts on top of the S already seen.
    """
    s = 0.0
    for i in range(depth):
        if i == 0:
            s = top_k
        else:
            s = min(experts, s + top_k * skew * (1.0 - s / experts))
    return s


def fit_skew(experts, top_k, depth, target):
    """Bisect `skew` so the recurrence hits the paper's measured working set."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if distinct_experts(experts, top_k, depth, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cpu_us_per_expert(geo, bpw, args, rows = 1):
    """Modelled host cost of evaluating one offloaded expert for `rows` tokens.

    Two terms, and their ratio is the whole design question:

      * weight traffic: the expert's bytes have to be read (and dequantized)
        once, no matter how many rows are attached to it;
      * compute: the GEMV cost grows with the number of rows.

    So a host-computed expert is nearly free per additional row, while a
    *streamed* expert costs the same transfer every time. That asymmetry is
    why speculation (K rows per expert) shifts the balance towards streaming.
    """
    if args.cpu_us_per_expert is not None:
        return args.cpu_us_per_expert
    b = model_budget(geo, bpw)["bytes_per_expert"]
    bw = args.dequant_gbs_per_thread * args.cpu_threads * 1e9  # bytes/s
    t_weight_us = b / bw * 1e6
    flops = 2.0 * routed_expert_params(geo) * rows
    t_compute_us = flops / (args.gemv_gflops_per_thread * args.cpu_threads * 1e9) * 1e6
    return t_weight_us + t_compute_us


def fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024.0
    return f"{n:.2f} TiB"


def fmt_ms(x):
    return f"{x:.2f} ms"


def report(geo, args):
    b = model_budget(geo, args.bpw)
    layers = geo["layers"]
    top_k = geo["top_k"]
    experts = geo["experts"]
    bpe = b["bytes_per_expert"]

    out = {}
    lines = []

    def p(s = ""):
        lines.append(s)

    p(f"=== {geo['label']} ===")
    p(f"geometry: hidden {geo['hidden']}, moe_interm {geo['moe_interm']}, "
      f"{experts} experts/layer, top-{top_k}, {layers} layers, {args.bpw} bpw")
    p()

    # ---- 1. footprint -----------------------------------------------------
    p("[1] Weight footprint")
    p(f"  routed experts        {b['routed_expert_params']/1e9:9.1f} B params  "
      f"{fmt_bytes(b['routed_expert_bytes']):>12}")
    p(f"  one routed expert     {b['expert_params']/1e6:9.1f} M params  "
      f"{fmt_bytes(bpe):>12}")
    p(f"  non-expert + embed    {(b['shared_expert_params']+geo['other_params']+geo['embed_params'])/1e9:9.1f} B params  "
      f"{fmt_bytes((b['shared_expert_params']+geo['other_params']+geo['embed_params'])*args.bpw/8):>12}")
    p(f"  whole model (excl. n-gram table) {b['total_params']/1e9:5.1f} B params  "
      f"{fmt_bytes(b['model_bytes']):>12}")
    if geo["mtp_params"]:
        p(f"  MTP head (1 layer)    {geo['mtp_params']/1e9:9.1f} B params  "
          f"{fmt_bytes(b['mtp_bytes']):>12}")
    if geo["ngram_rows"]:
        p(f"  PLE n-gram table      {geo['ngram_rows']*geo['ngram_dim']/1e9:9.1f} B params  "
          f"{fmt_bytes(b['ngram_table_bytes_fp16']):>12} (fp16, streamed from disk)")
    p()

    # ---- 2. GPU resident capacity ----------------------------------------
    p("[2] What fits in VRAM as a resident expert set")
    per_layer_all = experts * bpe
    for vram in args.vram_gb:
        budget = vram * (1 << 30) * args.expert_vram_share
        per_layer = budget / layers
        n_fit = int(per_layer // bpe)
        p(f"  VRAM {vram:5.1f} GB, {args.expert_vram_share*100:3.0f}% for experts -> "
          f"{n_fit:4d}/{experts} experts/layer resident "
          f"({n_fit*100.0/experts:5.1f}%), cold tail {experts-n_fit} experts/layer")
    p(f"  (all {experts} experts of one layer would need {fmt_bytes(per_layer_all)})")
    p()

    # ---- 3. PCIe traffic per token: compute-on-CPU vs stream-to-GPU ------
    p("[3] Host<->device traffic per decode token")
    act_bytes = 2 * geo["hidden"] * 2  # fp16 activation down + result up, per layer
    cpu_shape = act_bytes * layers
    p(f"  compute-on-CPU shape  {fmt_bytes(cpu_shape):>12}/token  "
      f"({layers} layers x down+up of one activation)")
    for off_frac, name in ((1.0, "all experts offloaded"), (0.3, "30% of selections offloaded")):
        sel = layers * top_k * off_frac
        vol = sel * bpe
        p(f"  stream-to-GPU shape   {fmt_bytes(vol):>12}/token  "
          f"({name}: {sel:.0f} experts x {fmt_bytes(bpe)})")
        if off_frac == 1.0:
            out["pcie_ratio"] = vol / cpu_shape
    p(f"  ratio (stream / compute-on-CPU): {out.get('pcie_ratio', float('nan')):.0f}x")
    p()

    # ---- 4. cost per token: host compute vs bus --------------------------
    p("[4] Cost of the offloaded part of one step")
    p("  A step that verifies R rows needs the UNION of the experts those rows select,")
    p("  so the weight cost is paid once per distinct expert, and the row cost per token.")
    p("  Host compute: weight read once per distinct expert + GEMV per row.")
    p("  Bus: one transfer per distinct expert, on the critical path unless prefetched.")
    skew = fit_skew(64, 6, 16, PAPER_DSV2_DISTINCT_K16)
    b = model_budget(geo, args.bpw)
    bw_mem = args.dequant_gbs_per_thread * args.cpu_threads * 1e9
    t_weight_us = bpe / bw_mem * 1e6
    t_row_us = 2.0 * b["expert_params"] / (args.gemv_gflops_per_thread * args.cpu_threads * 1e9) * 1e6
    links = list(LINK_GBPS.items()) + ([("custom", args.link_gbps)] if args.link_gbps else [])
    p(f"  host: {args.cpu_threads} threads, weight {t_weight_us:.0f} us/expert, "
      f"row {t_row_us:.0f} us/expert-row")
    p(f"  {'rows':>5} {'exp/layer':>10} {'host ms/step':>13} " +
      " ".join(f"{n:>12}" for n, _ in links) + "   per token")
    row_costs = {}
    for rows in args.rows:
        d = distinct_experts(experts, top_k, rows, skew)
        host_ms = layers * (d * t_weight_us + rows * top_k * t_row_us) / 1e3
        cells = []
        for name, bw in links:
            bus_ms = layers * d * bpe / (bw * 1e9) * 1e3
            cells.append(f"{bus_ms:8.1f}{'*' if bus_ms < host_ms else ' '}")
        row_costs[rows] = {"host_ms": host_ms, "experts_per_layer": d}
        p(f"  {rows:5d} {d:10.1f} {host_ms:13.1f} " + " ".join(f"{c:>12}" for c in cells) +
          f"   host {host_ms/rows:6.1f} ms/tok")
    p("  (* = bus faster than host compute for that link)")
    be = bpe / (t_weight_us * 1e3)  # bytes/us -> GB/s
    p(f"  break-even link bandwidth for one expert (weight read only): {be:.1f} GB/s")
    out["t_cpu_ms"] = row_costs.get(1, {}).get("host_ms")
    out["t_expert_ms"] = t_weight_us / 1e3
    out["breakeven_gbps"] = be
    out["host_ms_by_rows"] = row_costs
    p()

    # ---- 5. speculative working set --------------------------------------
    p("[5] Verification working set vs speculation depth K")
    p(f"  model skew factor {skew:.3f}, fitted so the recurrence reproduces the paper's")
    p(f"  DeepSeek-V2-Lite measurement ({PAPER_DSV2_DISTINCT_K16} distinct experts/layer at K=16)")
    vram_bytes = args.vram_gb[0] * (1 << 30)
    p(f"  {'K':>3} {'distinct/layer':>15} {'per step':>14} {'vs VRAM':>10}  note")
    for k in args.depths:
        d = distinct_experts(experts, top_k, k, skew)
        ws = d * layers * bpe
        fits = "fits" if ws < vram_bytes * args.expert_vram_share else "EXCEEDS"
        note = ("all of it must be resident at once during verification"
                if ws < vram_bytes * args.expert_vram_share
                else "must be streamed per layer, cannot live in a cache")
        p(f"  {k:3d} {d:15.1f} {fmt_bytes(ws):>14} {fits:>10}  {note}")
    out["skew"] = skew
    p()

    # ---- 6. draft model options ------------------------------------------
    p("[6] Draft / predictor options (VRAM cost vs the 'look-ahead' they provide)")
    opts = []
    # full INT4 copy of the target (Specter's choice)
    full = b["routed_expert_bytes"] + (b["shared_expert_params"] + geo["other_params"]
                                       + geo["embed_params"]) * args.bpw / 8
    opts.append(("full INT4 copy of the target (Specter)", full,
                 f"routing fidelity ~{PAPER_TOPK_MATCH*100:.0f}% (paper, other models)"))
    # MTP head
    if geo["mtp_params"]:
        opts.append(("MTP head from the same checkpoint", b["mtp_bytes"],
                     "target-consistent tokens, but 1 layer: no per-layer routing"))
    # hot-expert subset draft
    for m in (16, 32, 64):
        opts.append((f"target draft restricted to the top-{m} experts/layer",
                     m * layers * bpe + (geo["other_params"] + geo["embed_params"]) * args.bpw / 8,
                     "routing fidelity unknown (selected experts missing -> hidden-state drift)"))
    # router-only predictor
    router_bytes = layers * geo["hidden"] * experts * args.bpw / 8
    opts.append(("router-only predictor (no experts at all)", router_bytes,
                 "needs a hidden-state model to feed the routers"))
    p(f"  {'option':<52} {'VRAM':>12}  note")
    for name, size, note in opts:
        verdict = "OK" if size < args.vram_gb[0] * (1 << 30) * args.expert_vram_share else "no VRAM"
        p(f"  {name:<52} {fmt_bytes(size):>12}  [{verdict}] {note}")
    p(f"  (budget: {args.vram_gb[0]} GB x {args.expert_vram_share*100:.0f}% for experts)")
    p()

    # ---- 7. verdict -------------------------------------------------------
    p("[7] Verdict from the arithmetic above")
    ratio = out.get("pcie_ratio", float("nan"))
    if ratio > 100:
        p(f"  - streaming expert weights costs {ratio:.0f}x more PCIe traffic per token than")
        p("    computing the same experts on the host, so the bus, not the GPU, is the ceiling.")
        p("    A look-ahead that hides the transfer can only pay off if the bus is faster than")
        p("    the host at the per-expert level (see the break-even bandwidth above).")
    if full > args.vram_gb[0] * (1 << 30) * args.expert_vram_share:
        p(f"  - Specter's own draft (a full INT4 copy of the target) needs "
          f"{fmt_bytes(full)},")
        p("    which does not fit next to the resident expert set. The paper's models are")
        p("    3-8x smaller in total parameters and have 6-16x fewer experts per layer.")
    p()

    out["text"] = "\n".join(lines)
    out["numbers"] = {
        "bytes_per_expert": bpe,
        "routed_expert_bytes": b["routed_expert_bytes"],
        "model_bytes": b["model_bytes"],
        "pcie_ratio_stream_vs_cpu": out.get("pcie_ratio"),
        "t_cpu_ms": out.get("t_cpu_ms"),
        "t_expert_ms": out.get("t_expert_ms"),
        "breakeven_gbps": out.get("breakeven_gbps"),
        "skew": out.get("skew"),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="qwen3.8-flash-next", choices=sorted(PRESETS))
    ap.add_argument("--bpw", type=float, default=4.25,
                    help="bits per weight of the expert tensors (EXL3 mul1/trellis territory)")
    ap.add_argument("--vram-gb", type=float, nargs="+", default=[24.0],
                    help="GPU memory budgets to evaluate (a list)")
    ap.add_argument("--expert-vram-share", type=float, default=0.55,
                    help="fraction of VRAM spendable on resident experts")
    ap.add_argument("--offloaded-frac", type=float, default=1.0,
                    help="fraction of selected experts assumed to live on the host")
    ap.add_argument("--cpu-threads", type=int, default=12,
                    help="host threads the CPU MoE worker uses")
    ap.add_argument("--dequant-gbs-per-thread", type=float, default=1.5,
                    help="modelled expert dequant throughput per thread (GB/s)")
    ap.add_argument("--gemv-gflops-per-thread", type=float, default=20.0,
                    help="modelled expert GEMV throughput per thread (GFLOP/s)")
    ap.add_argument("--cpu-us-per-expert", type=float, default=None,
                    help="measured host cost per expert; overrides the model above")
    ap.add_argument("--link-gbps", type=float, default=None,
                    help="extra host->device bandwidth to evaluate")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="rows (tokens) per expert per step: 1 = plain decode, K = verification")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32],
                    help="speculation depths to evaluate")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    geo = dict(PRESETS[args.preset])
    res = report(geo, args)
    if args.json:
        print(json.dumps(res["numbers"], indent=2))
    else:
        print(res["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
