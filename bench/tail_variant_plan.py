#!/usr/bin/env python3
"""What a re-quantized cold tail would cost, and which tensors a quantizer must rewrite.

Backlog item 3 (`wiki/plans/2026-11-04-decode-backlog.md`) wants the CPU worker's cold-tail experts
at a lower bpw than the GPU head. This tool answers, from the checkpoint's own headers and without
loading anything:

  * which experts are the CPU tail at a given `-mcs` (after the saved hot-first permutation, when a
    profile is available),
  * what each of their tensors currently costs,
  * what they would cost at `--tail-bpw`, with the **scales left alone** -- EXL3 stores `suh`/`svh`
    per expert and those do not shrink with the trellis rate, so this is tighter than the flat
    `size_check.py --tail-bpw` linear model and is the number a quantizer run has to beat.

    bench/tail_variant_plan.py --model /mnt/d/llms/... --mcs 381 --tail-bpw 2.0 --out plan.json
    bench/tail_variant_plan.py --model ... --mcs 381 --tail-bpw 2.0 --stats profile.json

Exit 0 when the plan was produced, 2 when the checkpoint has no routed-expert tensors for a given
`--mcs` (so an empty plan cannot read as "nothing to save").
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

SCALE_SUFFIXES = (".suh", ".svh", ".bias", ".mul1")   # codebooks and scales: do not shrink


def read_headers(model_dir: str) -> dict[str, dict]:
    """Every tensor of every shard: name -> {shape, dtype, nbytes}."""
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            out[name] = {"shape": meta.get("shape"), "dtype": meta.get("dtype"), "nbytes": end - start}
    return out


def parse_expert(name: str):
    """('model.language_model.layers.N.mlp.experts.R.<proj>.<suffix>') -> (layer, expert, proj, suffix)."""
    if ".mlp.experts." not in name:
        return None
    head, rest = name.split(".mlp.experts.", 1)
    try:
        layer = int(head.split(".layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None
    parts = rest.split(".")
    if len(parts) < 3:
        return None
    try:
        expert = int(parts[0])
    except ValueError:
        return None
    return layer, expert, parts[1], ".".join(parts[2:])


def is_scale(tensor_name: str) -> bool:
    return any(tensor_name.endswith(s) for s in SCALE_SUFFIXES)


def _profile_counts(layer: int, num_experts: int, profile: dict):
    """The counts for one layer, in any of the shapes this box has written.

    `save_expert_stats` writes `{"layers": {"model.language_model.layers.N.mlp": [counts...]}}`;
    older scratch files used the bare index or "layerN". A shape this does not recognise must not
    silently fall back to index order without saying so - the caller reports it.
    """
    layers = profile.get("layers") if isinstance(profile.get("layers"), dict) else profile
    if isinstance(layers, dict):
        for key, counts in layers.items():
            if not isinstance(counts, (list, tuple)) or len(counts) != num_experts:
                continue
            k = str(key)
            if k == str(layer) or k == f"layer{layer}" or f"layers.{layer}." in k:
                return list(counts)
    return None


def hot_first_order(layer: int, num_experts: int, profile: dict | None) -> list[int]:
    """The order the load path puts experts in: hottest first, by the saved counts when present."""
    if profile:
        counts = _profile_counts(layer, num_experts, profile)
        if counts:
            return sorted(range(num_experts), key=lambda e: -counts[e])
    return list(range(num_experts))


def build_plan(headers: dict[str, dict], mcs: int, tail_bpw: float, source_bits: float,
               profile: dict | None = None, spares: int = 0) -> dict:
    by_layer: dict[int, dict[int, dict]] = {}
    for name, meta in headers.items():
        parsed = parse_expert(name)
        if parsed is None:
            continue
        layer, expert, proj, suffix = parsed
        by_layer.setdefault(layer, {}).setdefault(expert, {})[f"{proj}.{suffix}"] = meta

    layers = {}
    total_now = total_then = 0
    for layer in sorted(by_layer):
        experts = by_layer[layer]
        num = max(experts) + 1
        order = hot_first_order(layer, num, profile)
        first = num - mcs                      # the GPU keeps the first `first` of the order
        # with the pinned arena the worker also holds the `spares` experts the GPU slice gave up
        tail = order[first - spares:] if spares else order[first:]
        now = then = 0
        keys = []
        for e in tail:
            for tname, meta in sorted(experts.get(e, {}).items()):
                nb = meta["nbytes"]
                now += nb
                if is_scale(tname):
                    then += nb                # scales do not shrink with the trellis rate
                else:
                    then += int(round(nb * (tail_bpw / source_bits)))
            keys.append(f"model.language_model.layers.{layer}.mlp.experts.{e}")
        layers[layer] = {"num_experts": num, "gpu_experts": first, "tail_experts": len(tail),
                         "bytes_now": now, "bytes_at_tail_bpw": then,
                         "saving": now - then, "expert_prefixes": keys}
        total_now += now
        total_then += then
    return {"mcs": mcs, "tail_bpw": tail_bpw, "source_bits": source_bits, "spares": spares,
            "layers": layers,
            "tail_bytes_now": total_now, "tail_bytes_at_tail_bpw": total_then,
            "saving": total_now - total_then,
            "saving_fraction": (1 - total_then / total_now) if total_now else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mcs", type=int, required=True, help="CPU-tail experts per layer (the -mcs value)")
    ap.add_argument("--tail-bpw", type=float, required=True, help="target bits per weight for the tail")
    ap.add_argument("--stats", help="JSON of per-layer routing counts (the saved expert profile)")
    ap.add_argument("--spares", type=int, default=0,
                    help="EXL3_MOE_CPU_PROMOTE spare slots/layer: the worker holds these too")
    ap.add_argument("--out", help="write the plan here")
    args = ap.parse_args(argv)

    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    source_bits = float((cfg.get("quantization_config") or {}).get("bits") or 0)
    if source_bits <= 0:
        print("no quantization_config.bits in the checkpoint: cannot price a re-quantization",
              file=sys.stderr)
        return 2
    if args.tail_bpw >= source_bits:
        print(f"--tail-bpw {args.tail_bpw} is not below the checkpoint's {source_bits} bpw",
              file=sys.stderr)
        return 2
    profile = json.load(open(args.stats, encoding="utf-8")) if args.stats else None
    plan = build_plan(read_headers(args.model), args.mcs, args.tail_bpw, source_bits, profile,
                      max(0, args.spares))
    if not plan["layers"]:
        print("no routed-expert tensors found: nothing to plan", file=sys.stderr)
        return 2
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=1)
    mib = plan["tail_bytes_now"] / 2**20
    after = plan["tail_bytes_at_tail_bpw"] / 2**20
    print(f"tail at -mcs {args.mcs}: {plan['layers'][sorted(plan['layers'])[0]]['tail_experts']} "
          f"experts/layer over {len(plan['layers'])} layers")
    print(f"bytes now {mib:9.1f} MiB -> {after:9.1f} MiB at {args.tail_bpw:g} bpw "
          f"(saving {100 * plan['saving_fraction']:.1f} %; scales unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
