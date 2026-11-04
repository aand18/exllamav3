#!/usr/bin/env python3
"""Pre-flight geometry check: what a configuration costs, without loading anything.

Every number here comes from the checkpoint's own safetensors headers and `config.json`, so it
answers in ~1 s what used to cost a 60 GB load and a crash:

  * VRAM for the trunk at a given `-mcs` (GPU experts), plus the draft component if asked;
  * the pinned host-RAM arena for the CPU experts;
  * the cache at a given `-cs` / `-cq`, split into KV and recurrent parts;
  * the minimum `-cs` a benchmark script can run with (prompt + generation, page-rounded).

Protocol rule (see `wiki/plans/v165-critique-implementation.md`, I1): no GPU pass launches until
this script says the configuration fits inside the card and the host-RAM budget.

Usage:
  bench/size_check.py --model /mnt/d/llms/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 --mcs 376 --cs 131072
  bench/size_check.py --model ... --mcs 376 --cs 262144 --draft mtp --json
  bench/size_check.py --model ... --min-cs 34560 1024        # min cache for prompt+generation

Cache model: the engine's own tensor shapes, not a scaled measurement. A QSA layer
(`cache/qsa.py`) stores K and V at the requested width plus fp16 indexer side planes (raw key
per token, pooled key per compress-ratio block), and those planes do NOT scale with the cache
width. At `-cq 2,2` that is 320 B of packed K/V + 320 B of planes = 640 B/token/layer, which is
the measured 2.093 GiB at `-cs 262144`, slots 2 (12 full-attention layers) plus 0.218 GiB of GDN
recurrent state. Earlier versions scaled that 640 B linearly to 16 bits and so overestimated the
fp16 cache 2.3x (15.8 GiB instead of 6.9 GiB at 262k).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import sys

KV_BYTES_PER_TOKEN_PER_LAYER_2BIT = 640.0   # measured, see module docstring (fallback only)
RECURRENT_GIB_2_SLOTS = 0.218               # measured (GDN states, 48 layers, 2 slots)
PAGE = 256


def read_shards(directory: str) -> dict[str, int]:
    """tensor name -> exact bytes, straight from each shard's header."""
    sizes: dict[str, int] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            sizes[name] = end - start
    return sizes


def layer_of(name: str) -> int | None:
    if ".layers." not in name:
        return None
    try:
        return int(name.split(".layers.")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def split_costs(sizes: dict[str, int], num_experts: int, mcs: int) -> dict:
    """Split one component's bytes into GPU and CPU-arena parts for a given CPU-expert count."""
    trunk = arena = 0
    expert_gpu: dict[int, int] = {}
    expert_all: dict[int, int] = {}
    for name, nb in sizes.items():
        if ".mlp.experts." in name:
            li = layer_of(name)
            if li is None:
                continue
            expert_all[li] = expert_all.get(li, 0) + nb
        else:
            trunk += nb
    if expert_all:
        gpu_share = (num_experts - mcs) / num_experts
        for li, total in expert_all.items():
            expert_gpu[li] = int(total * gpu_share)
            arena += int(total * (mcs / num_experts))
        trunk += sum(expert_gpu.values())
    else:
        # a dense component (no routed experts) sits entirely where it is loaded
        pass
    return {"gpu_bytes": trunk, "arena_bytes": arena, "has_experts": bool(expert_all)}


def cache_bytes(cfg: dict, cs: int, bits_k: float, bits_v: float, slots: int) -> dict:
    """Bytes the engine allocates: cache/quant.py + cache/qsa.py shapes (see module docstring)."""
    layer_types = cfg.get("layer_types") or []
    kv_layers = sum(1 for t in layer_types if t == "full_attention") or 1
    token_dim = int(cfg.get("num_key_value_heads") or 0) * int(cfg.get("head_dim") or 0)
    idx_dim = int(cfg.get("indexer_head_dim") or 0)
    idx_ratio = int(cfg.get("indexer_compress_ratio") or 1) or 1
    if not token_dim:
        # config without the fields this model has: fall back to the one calibrated point
        bits = (bits_k + bits_v) / 2.0
        kv = int(KV_BYTES_PER_TOKEN_PER_LAYER_2BIT * (bits / 2.0) * cs * kv_layers)
    else:
        # fp16 planes: one raw indexer key per token plus one pooled key per compress-ratio block
        planes = (idx_dim + idx_dim // idx_ratio) * 2 if idx_dim else 0
        groups = token_dim // 32
        if bits_k >= 16 and bits_v >= 16:
            per_token_layer = token_dim * 2 * 2 + planes           # k, v fp16
        else:
            # packed words: token_dim//32 groups of k_bits (and v_bits) int32, plus fp16 scales
            per_token_layer = groups * (bits_k + bits_v) * 4 + 2 * groups * 2 + planes
        kv = int(per_token_layer * cs * kv_layers)
    rec = int(RECURRENT_GIB_2_SLOTS * (1024 ** 3) * slots / 2)
    return {"kv_bytes": kv, "recurrent_bytes": rec, "kv_layers": kv_layers,
            "bytes": kv + rec}


def gib(n: float) -> float:
    return n / (1024 ** 3)


def mib(n: float) -> float:
    return n / (1024 ** 2)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--model", required=True, help="model directory (config.json + shards)")
    ap.add_argument("--mcs", type=int, default=None, help="CPU experts per layer (the -mcs value)")
    ap.add_argument("--cs", type=int, default=None, help="cache size in tokens (the -cs value)")
    ap.add_argument("--cq", default="2,2", help="cache quantisation, e.g. 2,2 (default)")
    ap.add_argument("--slots", type=int, default=2, help="cache slots / -ambs (default 2)")
    ap.add_argument("--draft", default="none",
                    help="'mtp' to include the model's own MTP head, or a draft model directory")
    ap.add_argument("--budget-vram-mb", type=int, default=24564, help="card size in MiB")
    ap.add_argument("--budget-ram-mb", type=int, default=63600, help="host RAM in MiB")
    ap.add_argument("--headroom-mb", type=int, default=1024,
                    help="required free VRAM after everything is resident (default 1024)")
    ap.add_argument("--reserve-mb", type=int, default=2048,
                    help="the autosplit's own reserve, observed at ~2 GiB; counted against the card (0 to ignore)")
    ap.add_argument("--tail-bpw", type=float, default=0.0,
                    help="re-quantise the CPU worker's cold-tail experts at this bpw (backlog item 3). "
                         "The arena scales linearly from the checkpoint's own bits; the per-expert "
                         "scales do not shrink, so the saving is an upper bound")
    ap.add_argument("--ccs", type=float, default=0.0,
                    help="host KV second tier in GB (the -ccs value). Its VRAM cost is NOT modelled, so a "
                         "configuration that uses it is reported UNCERTIFIED instead of FITS (T37)")
    ap.add_argument("--min-cs", nargs=2, type=int, metavar=("PROMPT_TOKENS", "NEW_TOKENS"),
                    help="print the smallest -cs that can run prompt + generation")
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    args = ap.parse_args(argv)

    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tcfg = cfg.get("text_config", cfg)
    num_experts = tcfg.get("num_experts") or 0
    sizes = read_shards(args.model)

    if args.min_cs:
        prompt, new = args.min_cs
        need = prompt + new
        print(json.dumps({"min_cs_tokens": int(math.ceil(need / PAGE) * PAGE),
                          "prompt_tokens": prompt, "new_tokens": new, "pages": math.ceil(need / PAGE)})
              if args.json else
              f"minimum -cs for {prompt} prompt + {new} new tokens: "
              f"{int(math.ceil(need / PAGE) * PAGE)} tokens "
              f"({math.ceil(need / PAGE)} pages of {PAGE})")
        return 0

    if args.mcs is None or args.cs is None:
        ap.error("--mcs and --cs are required unless --min-cs is used")

    bits_k, bits_v = (float(x) for x in args.cq.split(","))

    host_streamed = {n: b for n, b in sizes.items() if "ngram_embedding" in n}
    mtp = {n: b for n, b in sizes.items() if n.startswith("mtp.")}
    trunk_sizes = {n: b for n, b in sizes.items()
                   if n not in host_streamed and n not in mtp}

    out: dict = {"model": args.model, "mcs": args.mcs, "cs": args.cs, "cq": args.cq}

    trunk = split_costs(trunk_sizes, num_experts, args.mcs)
    cache = cache_bytes(tcfg, args.cs, bits_k, bits_v, args.slots)
    vram = trunk["gpu_bytes"] + cache["bytes"]
    ram = trunk["arena_bytes"]

    out["trunk"] = {"gpu_mib": round(mib(trunk["gpu_bytes"]), 1),
                    "arena_mib": round(mib(trunk["arena_bytes"]), 1)}
    if args.tail_bpw:
        src_bits = float((cfg.get("quantization_config") or {}).get("bits") or 0)
        if src_bits <= 0:
            ap.error("--tail-bpw needs quantization_config.bits in the checkpoint's config.json")
        scale = min(1.0, args.tail_bpw / src_bits)
        ram = int(round(trunk["arena_bytes"] * scale))
        out["trunk"]["arena_source_mib"] = out["trunk"]["arena_mib"]
        out["trunk"]["arena_mib"] = round(mib(ram), 1)
        out["tail"] = {"bpw": args.tail_bpw, "source_bits": src_bits, "scale": round(scale, 4),
                       "note": "linear in bits; the per-expert scales do not shrink, so this is an upper bound"}
    out["cache"] = {"kv_mib": round(mib(cache["kv_bytes"]), 1),
                    "recurrent_mib": round(mib(cache["recurrent_bytes"]), 1),
                    "kv_layers": cache["kv_layers"],
                    "total_mib": round(mib(cache["bytes"]), 1)}

    if args.draft != "none":
        ddir = args.model if args.draft == "mtp" else args.draft
        if ddir != args.model:
            dsizes = read_shards(ddir)
        else:
            dsizes = mtp
        if not dsizes:
            raise SystemExit("no draft tensors found (expected mtp.* keys or a draft directory)")
        # the component's layer count comes from its own tensors, not from config.json: the MTP
        # head lives in the trunk's checkpoint as `mtp.layers.0.*` and is ONE full-attention layer
        # (the earlier estimate that treated it as a 48-layer trunk overpriced its cache 48x)
        dlayers = {layer_of(n) for n in dsizes if layer_of(n) is not None}
        dcount = (max(dlayers) + 1) if dlayers else 1
        draft = split_costs(dsizes, num_experts, args.mcs)
        dcfg_path = os.path.join(ddir, "config.json")
        dcfg = {}
        if os.path.exists(dcfg_path):
            with open(dcfg_path, encoding="utf-8") as f:
                dcfg = json.load(f)
            dcfg = dcfg.get("text_config", dcfg)
        if "dflash_config" in dcfg or "dspark" in str(dcfg.get("model_type") or ""):
            # DFlash/DSpark stores ONE shared KV row of head_dim per token per layer
            # (cache/dsa.py: CacheLayer_dspark), not num_kv_heads rows
            dcache = {"bytes": int(dcfg.get("head_dim") or 0) * 2 * dcount * args.cs}
        else:
            # MTP head: one plain full-attention layer with the trunk's K/V geometry
            fields = {"num_key_value_heads": tcfg.get("num_key_value_heads"),
                      "head_dim": tcfg.get("head_dim"),
                      "indexer_head_dim": None, "indexer_compress_ratio": None}
            if ddir == args.model:
                pass
            dcache = cache_bytes({**fields, "layer_types": ["full_attention"] * dcount},
                                 args.cs, bits_k, bits_v, args.slots)
        vram += draft["gpu_bytes"] + dcache["bytes"]
        ram += draft["arena_bytes"]
        out["draft"] = {"kind": args.draft,
                        "gpu_mib": round(mib(draft["gpu_bytes"]), 1),
                        "arena_mib": round(mib(draft["arena_bytes"]), 1),
                        "cache_mib": round(mib(dcache["bytes"]), 1),
                        "layers": dcount}

    out["vram_mib"] = round(mib(vram), 1)
    out["ram_mib"] = round(mib(ram), 1)
    out["vram_free_after_mib"] = round(
        mib(args.budget_vram_mb * 1024 ** 2 - vram - args.reserve_mb * 1024 ** 2), 0)
    out["ram_free_after_mib"] = round(mib(args.budget_ram_mb * 1024 ** 2 - ram), 0)
    out["fits"] = bool(out["vram_free_after_mib"] >= args.headroom_mb and out["ram_free_after_mib"] > 0)
    out["ngram_table_mib"] = round(mib(sum(host_streamed.values())), 1)
    # A host KV second tier (-ccs) costs VRAM somewhere - in T37 an arm this script certified as
    # FITS (2 041 MiB free) was killed by the watcher at 23 MiB free. Nothing here models that term,
    # so a configuration that uses it is not certified rather than certified against arithmetic that
    # does not include its cost.
    out["ccs_gb"] = args.ccs or 0.0
    out["certified"] = out["fits"] and not args.ccs
    if args.ccs:
        out["uncertified_reason"] = (
            f"-ccs {args.ccs:g} GB: the host KV tier's VRAM cost is not modelled "
            f"(T37 measured ~2 GB unaccounted at this regime)")

    if args.json:
        print(json.dumps(out, indent=1))
        return 0 if out["certified"] else 1

    print(f"model      {args.model}")
    print(f"config     -mcs {args.mcs}  -cs {args.cs}  -cq {args.cq}  slots {args.slots}"
          + (f"  draft {args.draft}" if args.draft != "none" else ""))
    print(f"  trunk    {out['trunk']['gpu_mib']:9.0f} MiB VRAM   {out['trunk']['arena_mib']:9.0f} MiB arena")
    print(f"  cache    {out['cache']['total_mib']:9.0f} MiB VRAM   "
          f"(KV {out['cache']['kv_mib']:.0f} over {out['cache']['kv_layers']} layers"
          f" + recurrent {out['cache']['recurrent_mib']:.0f})")
    if "draft" in out:
        d = out["draft"]
        print(f"  draft    {d['gpu_mib']:9.0f} MiB VRAM   {d['arena_mib']:9.0f} MiB arena   "
              f"{d['cache_mib']:.0f} MiB cache  ({d['layers']} layer(s))")
    print(f"  ngram    {out['ngram_table_mib']:9.0f} MiB streamed from disk (not resident)")
    print(f"  TOTAL    {out['vram_mib']:9.0f} MiB VRAM of {args.budget_vram_mb}"
          f"  -> {out['vram_free_after_mib']:.0f} MiB free")
    print(f"  reserve  {args.reserve_mb:9.0f} MiB held by the autosplit (counted against the card)")
    print(f"           {out['ram_mib']:9.0f} MiB host arena of {args.budget_ram_mb}"
          f"  -> {out['ram_free_after_mib']:.0f} MiB free")
    if "tail" in out:
        t = out["tail"]
        print(f"  tail     {t['bpw']:9.2f} bpw for the CPU tail (from {t['source_bits']:.2f}): "
              f"arena {out['trunk']['arena_source_mib']:.0f} -> {out['trunk']['arena_mib']:.0f} MiB "
              f"(linear, upper bound)")
    if args.ccs:
        print(f"  host KV  {args.ccs:9.1f} GB (-ccs)   cost NOT modelled, see T37")
        print(f"verdict    UNCERTIFIED — {out['uncertified_reason']}")
        return 1
    print(f"verdict    {'FITS' if out['fits'] else 'DOES NOT FIT'} "
          f"(needs >= {args.headroom_mb} MiB VRAM headroom)")
    return 0 if out["fits"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
