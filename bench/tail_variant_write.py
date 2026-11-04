#!/usr/bin/env python3
r"""Write a variant shard for the cold tail, and print the `-or` argument that mounts it.

Backlog item 3 re-quantizes the CPU worker's cold-tail experts at a lower bpw. The engine already
knows how to prefer an override: `-or <glob>=<dir>` builds a `VariantSafetensorsCollection` that
routes every tensor matching `<glob>` to `<dir>` before the main checkpoint. What was missing is the
shard itself and the glob that names exactly the planned experts.

This tool is the plumbing, not the quantizer. In its only mode (`--copy`) it rewrites the planned
tensors **byte for byte** into `<out>/tail.safetensors`, which proves the selection, the file and the
mount end to end without a GPU; a later quantizer swaps the payload for a re-encoded one and leaves
everything else about this file alone.

That quantizer is `bench/tail_requantize.py`, and it reaches this file's layout through two entry
points, so there is one writer and not two:

* `write_variant(..., payloads={name: payload})` - substitute a re-encoded payload for a tensor;
* `write_requantized_shard(model_dir, plan, out_dir, requantizer)` - ask a callable for each planned
  expert's payloads and stream them through `ShardWriter` (peak memory: one tensor).

A payload is raw `bytes` (then the checkpoint's `dtype` and `shape` are kept: only a re-encode whose
width does not change can be expressed that way) or a tensor-like object with `dtype`/`shape` (its own
header entry is used, which is what a re-encoded `trellis` needs - the third dimension is `16 * K`).

    bench/tail_variant_write.py --plan plan.json --model /mnt/d/llms/... --out /mnt/d/llms/tailvariant
    # then mount the YAML it writes:  -or D:\llms\tailvariant\override.yml

Pure Python: the shard is copied as raw bytes, so no torch and no checkpoint load. Exit 2 when a
planned tensor is missing from every shard (a plan that names tensors the checkpoint does not have
would produce a variant that silently overrides nothing).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import struct
import sys

ALIGN = 8

# torch dtype spellings kept here so this module never imports torch: the writer only ever sees the
# names, and a caller hands over a tensor-like object (detach/cpu/contiguous/numpy)
TORCH_DTYPES = {"torch.float16": "F16", "torch.bfloat16": "BF16", "torch.float32": "F32",
                "torch.float64": "F64", "torch.int8": "I8", "torch.uint8": "U8",
                "torch.int16": "I16", "torch.int32": "I32", "torch.int64": "I64"}


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


def read_tensor_bytes(path: str, header_len: int, meta: dict, name: str) -> bytes:
    """The payload of one tensor, read straight out of its shard."""
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + start)
        data = f.read(end - start)
    assert len(data) == end - start, f"short read for {name}"
    return data


class ShardWriter:
    """One safetensors file, laid out tensor by tensor.

    The payload is spooled to `<path>.payload.tmp` while the header accumulates, because a safetensors
    header precedes the payload and its length is not known until the last tensor is added. `close()`
    writes `header + payload` and removes the spool, so peak memory is one tensor: a requantized tail is
    tens of GiB and must not be assembled in RAM (the disk pays one extra copy instead).
    """

    def __init__(self, path: str):
        self.path = path
        self._spool_path = path + ".payload.tmp"
        self._spool = open(self._spool_path, "wb")
        self._header: dict[str, dict] = {}
        self._length = 0

    def add(self, name: str, data: bytes, dtype: str, shape) -> None:
        pad = (-self._length) % ALIGN          # align every tensor to 8 bytes, as safetensors requires
        self._spool.write(b"\0" * pad)
        off = self._length + pad
        self._spool.write(data)
        self._header[name] = {"dtype": dtype, "shape": list(shape),
                              "data_offsets": [off, off + len(data)]}
        self._length = off + len(data)

    def close(self) -> dict:
        self._spool.close()
        blob = json.dumps(self._header, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((-len(blob)) % ALIGN)
        try:
            with open(self.path, "wb") as out, open(self._spool_path, "rb") as spool:
                out.write(struct.pack("<Q", len(blob)))
                out.write(blob)
                shutil.copyfileobj(spool, out, 1 << 20)
        finally:
            os.remove(self._spool_path)
        return self._header


def payload_parts(payload):
    """(dtype, shape, bytes) of a substitute payload.

    Raw bytes carry no header entry and return `(None, None, data)`: the caller is saying the tensor's
    dtype and shape did not change. A tensor-like carries both, which a re-encoded `trellis` needs.
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return None, None, bytes(payload)
    dtype = getattr(payload, "dtype", None)
    shape = getattr(payload, "shape", None)
    if dtype is None or shape is None:
        raise TypeError(f"payload of type {type(payload).__name__} is neither bytes nor a tensor-like "
                        "object (needs dtype, shape and detach/cpu/contiguous/numpy)")
    name = TORCH_DTYPES.get(str(dtype))
    if name is None:
        raise ValueError(f"payload dtype {dtype} has no safetensors spelling in TORCH_DTYPES")
    data = payload.detach().cpu().contiguous().numpy().tobytes()
    return name, list(shape), data


def _tensor_payload(meta: dict, override) -> tuple[bytes, str, list]:
    """The (bytes, dtype, shape) to write for one tensor: the checkpoint's own dtype/shape by default."""
    dtype, shape, data = payload_parts(override)
    return data, dtype or meta["dtype"], shape if shape is not None else meta["shape"]


def write_variant(model_dir: str, wanted: list[str], out_dir: str, filename: str = "tail.safetensors",
                  payloads: dict | None = None) -> dict:
    """Copy `wanted` tensors into one shard, substituting `payloads[name]` where given.

    Returns the written header. Every tensor not named in `payloads` is copied byte for byte.
    """
    index = {name: (path, n, meta) for name, path, n, meta in iter_tensors(model_dir)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise KeyError(f"{len(missing)} planned tensor(s) are not in the checkpoint, e.g. {missing[0]}")
    os.makedirs(out_dir, exist_ok=True)
    payloads = payloads or {}
    writer = ShardWriter(os.path.join(out_dir, filename))
    for name in wanted:
        path, n, meta = index[name]
        raw = read_tensor_bytes(path, n, meta, name)
        if name in payloads:
            data, dtype, shape = _tensor_payload(meta, payloads[name])
        else:
            data, dtype, shape = raw, meta["dtype"], meta["shape"]
        writer.add(name, data, dtype, shape)
    return writer.close()


def write_requantized_shard(model_dir: str, plan: dict, out_dir: str, requantizer,
                            filename: str = "tail.safetensors",
                            prefixes: list[str] | None = None) -> dict:
    """Write the planned tensors, substituting what `requantizer` returns for one expert at a time.

    `requantizer(prefix, source)` is called once per planned expert, with that expert's tensors as
    `{suffix: bytes}` (read from the checkpoint, so the callable needs no torch to receive them), and
    returns `{suffix: payload}` for the tensors it re-encoded. Anything it does not name is copied.
    Peak memory is one expert, which is what lets the re-encode stream 20+ GiB to disk.
    """
    index = {name: (path, n, meta) for name, path, n, meta in iter_tensors(model_dir)}
    prefixes = planned_tensors(plan) if prefixes is None else prefixes
    by_prefix = names_by_prefix(model_dir, prefixes)
    # A prefix with no tensors would write nothing for those experts while the override spec still
    # names them, and the collection has no fallback: refuse before the first byte.
    blank = [p for p in prefixes if not by_prefix[p]]
    if blank:
        raise KeyError(f"{len(blank)} planned expert(s) are not in the checkpoint, e.g. {blank[0]}")
    os.makedirs(out_dir, exist_ok=True)
    writer = ShardWriter(os.path.join(out_dir, filename))
    for prefix in prefixes:
        names = by_prefix[prefix]
        source = {name[len(prefix) + 1:]: read_tensor_bytes(index[name][0], index[name][1],
                                                           index[name][2], name)
                  for name in names}
        overrides = requantizer(prefix, source) or {}
        unknown = [s for s in overrides if f"{prefix}.{s}" not in index]
        if unknown:
            raise KeyError(f"the requantizer returned '{unknown[0]}', which the plan does not name "
                           f"under {prefix}")
        for name in names:
            suffix = name[len(prefix) + 1:]
            if suffix in overrides:
                data, dtype, shape = _tensor_payload(index[name][2], overrides[suffix])
            else:
                data, dtype, shape = source[suffix], index[name][2]["dtype"], index[name][2]["shape"]
            writer.add(name, data, dtype, shape)
    return writer.close()


def prefix_of(name: str) -> str | None:
    """The expert a tensor belongs to: `...layers.N.mlp.experts.E.proj.suffix` -> `...experts.E`.

    Derived from the name's own structure rather than by scanning a list of prefixes: a full tail is
    ~18k experts and ~300k tensors in the checkpoint, so a `startswith` per prefix would be billions of
    comparisons. An expert tensor always has a projection and a suffix (that is how the plan tool reads
    them too), so a name without both belongs to no expert.
    """
    i = name.find(".mlp.experts.")
    if i < 0:
        return None
    head_len = i + len(".mlp.experts.")
    rest = name[head_len:].split(".")
    if len(rest) < 3:
        return None
    return name[:head_len] + rest[0]


def names_by_prefix(model_dir: str, prefixes: list[str]) -> dict[str, list[str]]:
    """prefix -> sorted tensor names, in one pass over the checkpoint's headers."""
    wanted = set(prefixes)
    out: dict[str, list[str]] = {p: [] for p in prefixes}
    for name, _, _, _ in iter_tensors(model_dir):
        head = prefix_of(name)
        if head in wanted:
            out[head].append(name)
    return {p: sorted(v) for p, v in out.items()}


def expert_globs(plan: dict) -> list[str]:
    """One glob per planned expert, and **only** the planned ones.

    A single `...experts.*` glob is wrong, not merely broad: `VariantSafetensorsCollection.get_tensor`
    does not fall back to the main checkpoint when the override collection lacks the key, so a glob
    that matches a GPU-resident expert makes that expert's load raise. The tail is a contiguous set
    of router ids and a glob cannot express a range, so the list is emitted per expert.
    """
    return [f"{prefix}.*" for _, info in sorted(plan["layers"].items())
            for prefix in info["expert_prefixes"]]


def override_yaml(plan: dict, variant_dir: str, source_id: str = "tail") -> str:
    """The `-or/--override` spec: a YAML file with one source and one key per planned expert."""
    lines = ["sources:", f"  - id: {source_id}", f"    model_dir: {variant_dir}", "overrides:"]
    for key in expert_globs(plan):
        lines.append(f'  - key: "{key}"')
        lines.append(f"    source: {source_id}")
    return "\n".join(lines) + "\n"


def planned_tensors(plan: dict) -> list[str]:
    """Every tensor of every planned expert: the plan names expert prefixes, the shard needs names."""
    out = []
    for layer, info in sorted(plan["layers"].items()):
        for prefix in info["expert_prefixes"]:
            out.append(prefix)
    return out


def expand(model_dir: str, prefixes: list[str]) -> list[str]:
    by_prefix = names_by_prefix(model_dir, prefixes)
    keep = []
    for p in prefixes:
        if not by_prefix[p]:
            raise KeyError(f"no tensors under '{p}'")
        keep.extend(by_prefix[p])
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
    spec = os.path.join(args.out, "override.yml")
    with open(spec, "w", encoding="utf-8") as f:
        f.write(override_yaml(plan, args.out))
    print(f"wrote {len(header)} tensor(s), {total / 2**20:.1f} MiB -> {os.path.join(args.out, args.filename)}")
    print(f"wrote {len(expert_globs(plan))} override key(s) -> {spec}")
    print(f"mount with: -or {spec}   (a Windows process needs the Windows path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
