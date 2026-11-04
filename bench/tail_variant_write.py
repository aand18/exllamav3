#!/usr/bin/env python3
"""Write a variant shard for the cold tail, and print the `-or` argument that mounts it.

Backlog item 3 re-quantizes the CPU worker's cold-tail experts at a lower bpw. The engine already
knows how to prefer an override: `-or <glob>=<dir>` builds a `VariantSafetensorsCollection` that
routes every tensor matching `<glob>` to `<dir>` before the main checkpoint. What was missing is the
shard itself and the glob that names exactly the planned experts.

This tool is the plumbing, not the quantizer. In its only mode (`--copy`) it rewrites the planned
tensors **byte for byte** into `<out>/tail.safetensors`, which proves the selection, the file and the
mount end to end without a GPU; a later quantizer swaps the payload for a re-encoded one and leaves
everything else about this file alone.

    bench/tail_variant_write.py --plan plan.json --model /mnt/d/llms/... --out /mnt/d/llms/tailvariant
    # then:  run_pass.ps1 -Extra "-or 'model.language_model.layers.*.mlp.experts.*=D:\\llms\\tailvariant'"

Pure Python: the shard is copied as raw bytes, so no torch and no checkpoint load. Exit 2 when a
planned tensor is missing from every shard (a plan that names tensors the checkpoint does not have
would produce a variant that silently overrides nothing).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

ALIGN = 8


def read_headers(model_dir: str) -> dict[str, tuple[str, dict]]:
    """name -> (shard path, metadata), over every shard of the checkpoint."""
    out: dict[str, tuple[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            out[name] = (path, meta)
    return out


def iter_tensors(model_dir: str):
    """(name, shard path, header length, metadata) - the header length is where the payload starts."""
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            yield name, path, n, meta


def write_variant(model_dir: str, wanted: list[str], out_dir: str, filename: str = "tail.safetensors") -> dict:
    """Copy `wanted` tensors into one shard. Returns the written header."""
    index = {name: (path, n, meta) for name, path, n, meta in iter_tensors(model_dir)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise KeyError(f"{len(missing)} planned tensor(s) are not in the checkpoint, e.g. {missing[0]}")
    os.makedirs(out_dir, exist_ok=True)
    header: dict[str, dict] = {}
    payload = bytearray()
    for name in wanted:
        path, n, meta = index[name]
        start, end = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(8 + n + start)
            data = f.read(end - start)
        assert len(data) == end - start, f"short read for {name}"
        # align every tensor to 8 bytes, as safetensors requires
        pad = (-len(payload)) % ALIGN
        payload.extend(b"\0" * pad)
        off = len(payload)
        payload.extend(data)
        header[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                        "data_offsets": [off, off + len(data)]}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-len(blob)) % ALIGN)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(bytes(payload))
    return header


def glob_for(plan: dict) -> str:
    """One glob that covers every planned expert and nothing else in the checkpoint."""
    layers = sorted(plan["layers"])
    if len(layers) == 1:
        return f"model.language_model.layers.{layers[0]}.mlp.experts.*"
    return "model.language_model.layers.*.mlp.experts.*"


def planned_tensors(plan: dict) -> list[str]:
    """Every tensor of every planned expert: the plan names expert prefixes, the shard needs names."""
    out = []
    for layer, info in sorted(plan["layers"].items()):
        for prefix in info["expert_prefixes"]:
            out.append(prefix)
    return out


def expand(model_dir: str, prefixes: list[str]) -> list[str]:
    names = [n for n, _, _, _ in iter_tensors(model_dir)]
    keep = []
    for p in prefixes:
        hits = [n for n in names if n.startswith(p + ".")]
        if not hits:
            raise KeyError(f"no tensors under '{p}'")
        keep.extend(sorted(hits))
    return keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="the JSON from bench/tail_variant_plan.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="variant directory (mount it with -or)")
    ap.add_argument("--filename", default="tail.safetensors")
    args = ap.parse_args(argv)

    plan = json.load(open(args.plan, encoding="utf-8"))
    prefixes = planned_tensors(plan)
    if not prefixes:
        print("the plan names no experts: nothing to write", file=sys.stderr)
        return 2
    try:
        wanted = expand(args.model, prefixes)
        header = write_variant(args.model, wanted, args.out, args.filename)
    except KeyError as e:
        print(f"{e}", file=sys.stderr)
        return 2
    total = sum(m["data_offsets"][1] - m["data_offsets"][0] for m in header.values())
    print(f"wrote {len(header)} tensor(s), {total / 2**20:.1f} MiB -> {os.path.join(args.out, args.filename)}")
    print(f"mount with: -or \"{glob_for(plan)}={os.path.abspath(args.out)}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
