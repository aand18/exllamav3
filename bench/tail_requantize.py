#!/usr/bin/env python3
r"""Re-encode the cold tail at a lower bpw: the quantizer behind the variant shard.

Backlog item 3 (`wiki/plans/2026-11-04-decode-backlog.md`) re-quantizes the CPU worker's tail experts
at a lower bpw. Three quarters of that already ship: `bench/tail_variant_plan.py` selects the tail and
prices every tensor, `bench/tail_variant_write.py` writes the shard and the `-or` spec that mounts it,
and the engine's `VariantSafetensorsCollection` routes a matched expert to the variant. What was
missing is the middle: a driver that reads a stored EXL3 tensor, dequantizes it with the engine's own
kernels and writes a **re-encoded** payload instead of a copy.

Two modes, and the cheap one is the default:

* `--dry-run` (default) reads only headers and the engine source: it validates the plan against the
  checkpoint, prints per projection what would be re-encoded and what it would cost, and prints the
  exact engine call it would make - function, module, argument names - **parsed out of the engine
  source**, so a renamed or missing kernel is reported instead of guessed at. No torch, no CUDA, no
  model, no writes.
* `--apply` re-encodes. Per planned expert, per projection: build a `LinearEXL3` from the stored
  `trellis`/`suh`/`svh` (+ the `mcg`/`mul1` codebook marker), call its `get_weight_tensor()` to get the
  fp16 weight back in the original basis, hand that to `quantize_exl3` with the uncalibrated Hessian
  placeholder, and write the returned `trellis`/`suh`/`svh` (and the marker) into the variant shard.
  Every other tensor is copied. `apply` needs the card: `fallback_quant` synchronizes a CUDA stream
  (`exllamav3/modules/quant/exl3_lib/quantize.py:825-827`), and so does `get_weight_tensor`.

    bench/tail_requantize.py --plan plan.json --model /mnt/d/llms/... --out /mnt/d/llms/tailvariant --bpw 2.0
    bench/tail_requantize.py --plan plan.json --model ... --out ... --bpw 2.0 --limit 3 --apply   # smoke
    # then mount the YAML it writes:  -or D:\llms\tailvariant\override.yml

Exit 0 when the dry run validated, 2 for every refusal: a plan tensor the checkpoint does not have,
a target rate that is not below the tensor's own source rate, a rate the encoder cannot express, a
codebook a half-integer rate needs and the tensors do not carry, an engine source without the function
this tool would call, or `--apply` with torch/the engine not importable.

**The basis of every number this tool prints.** The source rate of a projection is `trellis.shape[-1] /
16` - the engine's own derivation (`exllamav3/modules/quant/exl3.py:67`), not the checkpoint's
`quantization_config.bits`, which is a mean over the whole model (this box's 3.18 bpw checkpoint stores
3.0 bpw and 3.5 bpw tails). The target size is `tiles_k x tiles_n x 16*K` int16 words, computed from the
tile counts, not scaled from the source byte count: the plan's figure is a linear bound, this one is
exact. `suh`/`svh`/`mul1` do not shrink - the loader needs them at full width - so they are counted apart
and copied unchanged in size (they are re-derived in value by the encoder).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tail_variant_plan import read_headers  # noqa: E402
from tail_variant_write import (expert_globs, override_yaml, planned_tensors,  # noqa: E402
                               prefix_of, write_requantized_shard)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The four engine sources this tool names, and the functions it reads out of them.
ENGINE_SOURCES = {
    "quantize": "exllamav3/modules/quant/exl3_lib/quantize.py",
    "exl3": "exllamav3/modules/quant/exl3.py",
    "linear": "exllamav3/modules/linear.py",
    "convert": "exllamav3/conversion/convert_model.py",
}

# (source key, class or None, function) -> the entries the dry run must find before it prints a call.
REQUIRED_DEFS = [
    ("exl3", "LinearEXL3", "__init__"),
    ("exl3", "LinearEXL3", "get_weight_tensor"),
    ("linear", "Linear", "init_H_data"),
    ("quantize", None, "finalize_capture_H"),
    ("quantize", None, "quantize_exl3"),
    ("quantize", None, "fallback_quant"),
    ("quantize", None, "codebook_markers"),
    ("quantize", None, "trellis_words"),
    ("quantize", None, "frac_k"),
    ("convert", None, "make_quant_args"),
]

PROJECTION_ORDER = ("gate_proj", "up_proj", "down_proj")
TORCH_DTYPES = {"F16": "float16", "BF16": "bfloat16", "F32": "float32",
                "I16": "int16", "I32": "int32", "I8": "int8"}


class Refusal(RuntimeError):
    """A condition the tool will not guess its way past; the message names the thing that is wrong."""


# --------------------------------------------------------------------------------------------------
# what the engine says - read from the source, never from memory
# --------------------------------------------------------------------------------------------------

def _find_def(tree: ast.Module, name: str, cls: str | None = None):
    body = tree.body
    if cls is not None:
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                body = node.body
                break
        else:
            return None
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _arg_names(node) -> list[str]:
    args = [a.arg for a in node.args.posonlyargs + node.args.args]
    if node.args.vararg is not None:
        args.append("*" + node.args.vararg.arg)
    args += [a.arg for a in node.args.kwonlyargs]
    if node.args.kwarg is not None:
        args.append("**" + node.args.kwarg.arg)
    return args


def _function_spans(tree: ast.Module) -> list[tuple[str, int, int]]:
    """(qualified name, first line, last line) for every function and method in a module."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.name, node.lineno, getattr(node, "end_lineno", node.lineno)))
    return spans


def _enclosing(spans: list[tuple[str, int, int]], line: int) -> str:
    best = ""
    best_span = None
    for name, start, end in spans:
        if start <= line <= end and (best_span is None or end - start < best_span):
            best, best_span = name, end - start
    return best


def _ext_reconstruct_calls(tree: ast.Module, lines: list[str]) -> list[dict]:
    """The `ext.reconstruct` launches inside the module: the payload behind a dequantized weight.

    `LinearEXL3.get_weight_tensor` -> `get_inner_weight_tensor` -> `_reconstruct_inner` launches one of
    these; the fused `reconstruct_had_*` kernels belong to the prefill GEMM path and are left out. Each
    call carries the function it sits in, so a line from an unrelated path is visible as one.
    """
    spans = _function_spans(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id == "ext" and \
                node.func.attr in ("reconstruct", "reconstruct_slice"):
            out.append({"function": f"ext.{node.func.attr}", "line": node.lineno,
                        "enclosing": _enclosing(spans, node.lineno),
                        "source": lines[node.lineno - 1].strip()})
    return sorted(out, key=lambda c: c["line"])


def _dict_assignment(tree: ast.Module, target: str) -> tuple[int, list[str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(t, ast.Name) and t.id == target for t in node.targets):
                return node.lineno, [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    return 0, []


def engine_calls(engine_root: str = REPO) -> dict:
    """Every function this tool calls, as the engine source defines it: module, file, line, arguments.

    A missing definition is a refusal, not a fallback: the whole point of the dry run is that the call
    it prints is the call the source carries, so a renamed kernel must stop it here.
    """
    found: dict[str, dict] = {}
    trees: dict[str, ast.Module] = {}
    lines: dict[str, list[str]] = {}
    missing: list[str] = []
    for key, rel in ENGINE_SOURCES.items():
        path = os.path.join(engine_root, rel)
        if not os.path.isfile(path):
            raise Refusal(f"the engine source is not there: {path} (pass --engine-root)")
        text = open(path, encoding="utf-8").read()
        trees[key] = ast.parse(text, filename=path)
        lines[key] = text.splitlines()
    for key, cls, name in REQUIRED_DEFS:
        node = _find_def(trees[key], name, cls)
        label = f"{cls}.{name}" if cls else name
        if node is None:
            missing.append(f"{label} ({ENGINE_SOURCES[key]})")
            continue
        found[f"{cls}.{name}" if cls else name] = {
            "label": label,
            "module": ENGINE_SOURCES[key][:-3].replace("/", "."),
            "file": ENGINE_SOURCES[key],
            "line": node.lineno,
            "args": _arg_names(node),
        }
    if missing:
        raise Refusal("the engine source has no " + ", ".join(missing) +
                      ": the call this tool would print is gone, so it will not print a guess")
    found["ext_calls"] = _ext_reconstruct_calls(trees["exl3"], lines["exl3"])
    found["out_tensors_line"], found["out_tensors"] = _dict_assignment(trees["quantize"], "out_tensors")
    found["root"] = engine_root
    return found


def _call_str(entry: dict) -> str:
    return f"{entry['label']}({', '.join(entry['args'])})"


def print_engine_calls(calls: dict) -> None:
    print("engine calls, read from the source:")
    init = calls["LinearEXL3.__init__"]
    print(f"  build     {init['module']}.{_call_str(init)}")
    print(f"            {init['file']}:{init['line']}  (su/sv or suh/svh, trellis, mcg, mul1, key)")
    get = calls["LinearEXL3.get_weight_tensor"]
    print(f"  dequant   {_call_str(get)}")
    print(f"            {get['file']}:{get['line']}  on a LinearEXL3 built from the stored tensors")
    print("            inner payload, the module's own reconstruct launches:")
    for c in calls["ext_calls"]:
        print(f"              {c['function']}  in {c['enclosing']}()")
        print(f"              {ENGINE_SOURCES['exl3']}:{c['line']}  {c['source']}")
    hid = calls["Linear.init_H_data"]
    print(f"  hessian   {_call_str(hid)} -> a meta (in,in) H, no calibration data")
    print(f"            {hid['file']}:{hid['line']}; {ENGINE_SOURCES['quantize']}:"
          f"{calls['finalize_capture_H']['line']} ({_call_str(calls['finalize_capture_H'])}) turns "
          f"that into q_fallback")
    enc = calls["quantize_exl3"]
    print(f"  encode    {enc['module']}.{_call_str(enc)}")
    print(f"            {enc['file']}:{enc['line']}; q_fallback takes {_call_str(calls['fallback_quant'])} "
          f"({ENGINE_SOURCES['quantize']}:{calls['fallback_quant']['line']})")
    args = calls["make_quant_args"]
    print(f"  args      {_call_str(args)}")
    print(f"            {args['file']}:{args['line']}  (the quant_args keys the encoder reads)")
    print(f"  outputs   {', '.join(calls['out_tensors'])}  ({ENGINE_SOURCES['quantize']}:"
          f"{calls['out_tensors_line']}), plus the codebook marker from "
          f"{_call_str(calls['codebook_markers'])}")
    print(f"            the marker is copied when a projection carries mcg/mul1; without one the "
          f"encoder writes the default 3INST trellis")


# --------------------------------------------------------------------------------------------------
# the plan, checked against the checkpoint
# --------------------------------------------------------------------------------------------------

MAX_PROBLEMS = 12  # a 381-expert tail can fail 1 143 times over one missing tensor: say so once each


def source_rate(shape) -> float | None:
    """K = trellis.shape[-1] / 16, the engine's own derivation (`exl3.py:67`) or None.

    A tile is a whole number of 32-bit words at any rate the encoder can express, so the width is a
    multiple of 8: 48 -> 3.0 bpw, 56 -> 3.5 bpw, 32 -> 2.0 bpw.
    """
    if not shape or len(shape) != 3 or shape[-1] % 8:
        return None
    return shape[-1] / 16


def trellis_words(bpw: float) -> int:
    """int16 words per 256-weight tile at rate `bpw` (`quantize.py:115-121`).

    A rate the encoder cannot express is refused here rather than 17 000 experts into a run: `frac_k`
    accepts an integer K or a half-integer K = KA + 0.5 and asserts the half-integer form otherwise
    (`quantize.py:103-114`).
    """
    k = float(bpw)
    if k <= 0:
        raise Refusal(f"{bpw:g} bpw is not a rate: a tile carries at least one bit per weight")
    if not k.is_integer() and abs(int(k) + 0.5 - k) > 1e-9:
        raise Refusal(f"{bpw:g} bpw is neither an integer nor a half-integer: the encoder's frac_k "
                      f"asserts K = KA + 0.5 (quantize.py:103-114), so it cannot express this rate")
    words = 16 * k
    if not words.is_integer():
        raise Refusal(f"{bpw:g} bpw is not a whole number of trellis words per tile "
                      f"(16 * {bpw:g} = {words})")
    return int(words)


def half_rate(bpw: float) -> bool:
    """A half-integer rate: alternating K/K+1-bit steps, mul1 codebook only (`quantize.py:103-114`)."""
    return not float(bpw).is_integer()


def plan_prefixes(plan: dict) -> list[str]:
    return planned_tensors(plan)


def limit_plan(plan: dict, limit: int | None) -> dict:
    """The plan truncated to the first `limit` experts, so a smoke run writes a mountable shard."""
    if not limit:
        return plan
    left = limit
    layers = {}
    for layer in sorted(plan["layers"]):
        info = dict(plan["layers"][layer])
        keep = info.get("expert_prefixes", [])[:max(0, left)]
        left -= len(keep)
        info["expert_prefixes"] = keep
        info["tail_experts"] = len(keep)
        layers[layer] = info
    out = dict(plan)
    out["layers"] = layers
    return out


def expert_index(headers: dict, prefixes: list[str]) -> dict[str, dict[str, dict[str, dict]]]:
    """prefix -> proj -> suffix -> header meta, in one pass over the checkpoint's headers.

    The projections come back in `PROJECTION_ORDER` first, so a report reads gate/up/down every time.
    A planned prefix with nothing under it maps to `{}`, which is how `analyse` names it.
    """
    wanted = set(prefixes)
    out: dict[str, dict[str, dict[str, dict]]] = {p: {} for p in prefixes}
    for name, meta in headers.items():
        prefix = prefix_of(name)
        if prefix not in wanted:
            continue
        rest = name[len(prefix) + 1:].split(".")
        proj, suffix = rest[0], ".".join(rest[1:])
        out[prefix].setdefault(proj, {})[suffix] = meta
    for prefix, projs in out.items():
        out[prefix] = {p: projs[p] for p in
                       sorted(projs, key=lambda q: (PROJECTION_ORDER.index(q)
                                                    if q in PROJECTION_ORDER else 99, q))}
    return out


def codebook_of(tensors: dict[str, dict]) -> str:
    has_mcg, has_mul1 = "mcg" in tensors, "mul1" in tensors
    if has_mcg and has_mul1:
        raise Refusal("carries both a .mcg and a .mul1 marker")
    return "mcg" if has_mcg else "mul1" if has_mul1 else "3INST"


def analyse(headers: dict, plan: dict, bpw: float) -> tuple[list[str], dict]:
    """Every reason to refuse, and the per-projection cost table for the projections that pass."""
    problems: list[str] = []
    suppressed = [0]

    def note(msg: str) -> None:
        """Keep a report readable: a tail of 1 143 projections fails the same way 1 143 times."""
        if len(problems) < MAX_PROBLEMS:
            problems.append(msg)
        else:
            suppressed[0] += 1

    prefixes = plan_prefixes(plan)
    if not prefixes:
        return ["the plan names no experts: nothing to re-encode"], {}

    source_bits = plan.get("source_bits")
    if source_bits is None and not any(source_rate(m.get("shape")) for m in headers.values()):
        note("the plan carries no source_bits and the checkpoint's trellis shapes give none: the "
             "target rate cannot be shown to be below the source")
    if source_bits is not None and bpw >= float(source_bits):
        note(f"target {bpw:g} bpw is not below the plan's source {float(source_bits):g} bpw")

    rows: dict[str, dict] = {}
    codebooks: set[str] = set()
    index = expert_index(headers, prefixes)
    for prefix in prefixes:
        projs = index[prefix]
        if not projs:
            note(f"the plan names {prefix}, which is not in the checkpoint")
            continue
        for proj, tensors in projs.items():
            for suffix in ("trellis", "suh", "svh"):
                if suffix not in tensors:
                    note(f"{prefix}.{proj}.{suffix} is missing from the checkpoint (the plan names "
                         f"{prefix}; the loader needs trellis, suh and svh)")
            if "trellis" not in tensors:
                continue
            rate = source_rate(tensors["trellis"].get("shape"))
            if rate is None:
                note(f"{prefix}.{proj}.trellis has shape {tensors['trellis'].get('shape')}: its "
                     f"source rate (shape[-1] / 16) cannot be read")
                continue
            if bpw >= rate:
                note(f"target {bpw:g} bpw is not below the source {rate:g} bpw of {prefix}.{proj}"
                     f".trellis (shape {tensors['trellis']['shape']})")
                continue
            tk, tn, _ = tensors["trellis"]["shape"]
            if tn % 8:
                note(f"{prefix}.{proj} has {tn * 16} out-features: the encoder asserts "
                     f"size_n % 128 == 0 (quantize.py:835-842), so --apply would die deep inside it")
                continue
            try:
                words = trellis_words(bpw)
            except Refusal as e:
                note(str(e))
                words = None
            try:
                codebook = codebook_of(tensors)
            except Refusal as e:
                note(f"{prefix}.{proj} {e}")
                continue
            codebooks.add(codebook)
            if half_rate(bpw) and codebook != "mul1":
                note(f"target {bpw:g} bpw is a half-integer rate, which needs the mul1 codebook "
                     f"(quantize.py:103-114, 130); {prefix}.{proj} carries {codebook} markers")
                continue
            row = rows.setdefault(proj, {"projections": 0, "rates": {}, "tiles": 0,
                                         "trellis_now": 0, "trellis_then": 0, "scales": 0,
                                         "codebooks": set()})
            now = sum(m["nbytes"] for m in tensors.values())
            trellis_now = tensors["trellis"]["nbytes"]
            trellis_then = tk * tn * words * 2 if words is not None else trellis_now
            row["projections"] += 1
            row["rates"][rate] = row["rates"].get(rate, 0) + 1
            row["tiles"] += tk * tn
            row["trellis_now"] += trellis_now
            row["trellis_then"] += trellis_then
            row["scales"] += now - trellis_now
            row["codebooks"].add(codebook)
    if len(codebooks) > 1 and not problems:
        note(f"the planned experts mix codebooks ({', '.join(sorted(codebooks))}): one encoder "
             f"setting cannot reproduce all of them")
    if suppressed[0]:
        problems.append(f"... and {suppressed[0]} more like the above")
    return problems, rows


def print_report(rows: dict, plan: dict, bpw: float) -> None:
    prefixes = plan_prefixes(plan)
    layers = len(plan["layers"])
    print()
    print(f"plan: {len(prefixes)} experts over {layers} layers; target {bpw:g} bpw "
          f"(trellis words/tile {trellis_words(bpw)}, {2 * trellis_words(bpw)} B per tile)")
    print(f"{'projection':<11}{'projs':>7}{'stored rates (count)':>26}{'trellis now':>15}"
          f"{'at target':>14}{'scales':>13}{'saving':>9}")
    total_now = total_then = total_scales = 0
    for proj, row in sorted(rows.items(), key=lambda kv: (PROJECTION_ORDER.index(kv[0])
                                                          if kv[0] in PROJECTION_ORDER else 99, kv[0])):
        rates = ", ".join(f"{r:g} ({row['rates'][r]:,})" for r in sorted(row["rates"]))
        now, then, scales = row["trellis_now"], row["trellis_then"], row["scales"]
        total_now += now
        total_then += then
        total_scales += scales
        cut = 100 * (1 - (then + scales) / (now + scales)) if now + scales else 0.0
        print(f"{proj:<11}{row['projections']:>7,}{rates:>26}{now / 2**20:>12,.1f} MiB"
              f"{then / 2**20:>11,.1f} MiB{scales / 2**20:>10,.1f} MiB{cut:>8.1f} %")
    cut = 100 * (1 - (total_then + total_scales) / (total_now + total_scales)) \
        if total_now + total_scales else 0.0
    print(f"{'total':<11}{sum(r['projections'] for r in rows.values()):>7,}{'':>26}"
          f"{total_now / 2**20:>12,.1f} MiB{total_then / 2**20:>11,.1f} MiB"
          f"{total_scales / 2**20:>10,.1f} MiB{cut:>8.1f} %")
    print(f"every planned tensor is a trellis, a suh, a svh or a codebook marker: the trellis is "
          f"re-encoded, the other three are re-derived by the encoder at the same size")
    plan_now, plan_then = plan.get("tail_bytes_now"), plan.get("tail_bytes_at_tail_bpw")
    if plan_now and plan_then:
        print(f"the plan's own bound for the same tail: {plan_now / 2**20:,.1f} -> "
              f"{plan_then / 2**20:,.1f} MiB ({100 * (1 - plan_then / plan_now):.1f} %); it scales the "
              f"trellis linearly, by the checkpoint's mean rate over the tensor's own rate")
    if plan.get("source_bits") is not None:
        print(f"note: the plan's source_bits {float(plan['source_bits']):g} is the checkpoint's mean "
              f"rate; the rate that gates the re-encode is each trellis's own width / 16, and the two "
              f"disagree by every bit the head and MTP layers add")


# --------------------------------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------------------------------

def engine_available() -> tuple[bool, str]:
    """(ok, why not) for the path that needs torch, the extension and a CUDA device."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        return False, (f"--apply needs torch and the engine, and torch is not importable here ({e}). "
                       f"The dry run needs neither: drop --apply")
    try:
        from exllamav3.ext import exllamav3_ext as ext  # noqa: F401
    except Exception as e:  # noqa: BLE001 - any import failure means the kernels are not there
        return False, (f"--apply needs the built extension exllamav3.ext.exllamav3_ext, which failed "
                       f"to import ({type(e).__name__}: {e})")
    if not torch.cuda.is_available():
        return False, ("--apply needs a CUDA device: the encoder synchronizes CUDA streams "
                       "(quantize.py:825-827, 903-904) and the dequant kernels are CUDA-only")
    return True, ""


def uncalibrated_h_data(torch, in_features: int, device, key: str) -> dict:
    """The placeholder `Linear.init_H_data(False)` builds (`exllamav3/modules/linear.py:613-627`).

    A tensor with no `--hessians` file is quantized through exactly this dict
    (`convert_model.py:474-479`): `finalize_capture_H` sees a meta H and switches to q_fallback
    (`quantize.py:1021-1032`), which is the uncalibrated path. Reproducing the keys here keeps the
    re-encode on the engine's own branch without importing a model.
    """
    return {
        "H": torch.zeros(in_features, in_features, dtype=torch.float32, device="meta"),
        "packed": False,
        "n": in_features,
        "first_key": key,
        "count": 0,
        "finalized": False,
        "num_total": 0,
        "inf_nan": torch.zeros(2, dtype=torch.long),
        "device": device,
    }


def quant_args_for(bpw: float, device, seed: int, codebook: str, debug_dir: str) -> dict:
    """The quant_args keys the encoder reads, as `make_quant_args` builds them (`convert_model.py:510-531`).

    `apply_out_scales: None` is the converter's `--out_scales auto` default (`convert_model.py:316`).
    """
    # `quantize_tiles_scratch(index, K, ...)` (`quantize.py:82`) takes an **integer** K, so the
    # scratch sizing path cannot express a half-integer rate; refuse it by name instead of letting a
    # float reach the extension as a TypeError
    kf = float(bpw)
    if abs(kf - round(kf)) > 1e-9:
        raise Refusal(f"--bpw {bpw:g} is not an integer: the encoder's scratch path takes an integer K "
                      f"(quantize.py:82); half-integer rates would need the convert path")
    args = {"seed": seed, "K": int(round(kf)), "devices": [str(device)], "apply_out_scales": None,
            "debug_dir": debug_dir}
    if codebook == "mcg":
        args["mcg"] = True
    elif codebook == "mul1":
        args["mul1"] = True
    return args


def make_engine_requantizer(index: dict, bpw: float, device: str, seed: int, debug_dir: str):
    """A `requantizer(prefix, source)` for `write_requantized_shard`, built on the engine's C++ path.

    `index` is `expert_index`'s output; `source` is `{suffix: bytes}`. Shapes come from the checkpoint's
    headers, so nothing is inferred from a payload length. Returns `{proj.suffix: tensor}` for the
    tensors the encoder produced - trellis, suh, svh and the codebook marker - which the shard carries
    in place of the stored ones.
    """
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    def tensor(name: str, meta: dict, data: bytes):
        dtype = TORCH_DTYPES.get(meta["dtype"])
        if dtype is None:
            raise Refusal(f"{name} stores dtype {meta['dtype']}, which this tool cannot rebuild")
        flat = torch.frombuffer(bytearray(data), dtype=getattr(torch, dtype))
        return flat.reshape(meta["shape"]).to(device)

    def requantize(prefix: str, source: dict[str, bytes]) -> dict:
        out = {}
        for proj, suffixes in index[prefix].items():
            names = {s: f"{prefix}.{proj}.{s}" for s in suffixes}
            # `source` is keyed by the tensor's suffix *relative to the expert prefix*
            # ("gate_proj.trellis"), not by the bare suffix
            tensors = {s: tensor(names[s], suffixes[s], source[f"{proj}.{s}"])
                       for s in ("trellis", "suh", "svh", "mcg", "mul1") if s in suffixes}
            trellis = tensors["trellis"]
            in_features, out_features = trellis.shape[0] * 16, trellis.shape[1] * 16
            layer = LinearEXL3(None, in_features, out_features,
                               suh=tensors["suh"], svh=tensors["svh"], trellis=trellis,
                               mcg=tensors.get("mcg"), mul1=tensors.get("mul1"),
                               key=names["trellis"])
            weight = layer.get_weight_tensor().float()
            del layer
            codebook = codebook_of(suffixes)
            quant_args = quant_args_for(bpw, device, seed, codebook, debug_dir)
            H_data = uncalibrated_h_data(torch, in_features, device, names["trellis"])
            _, _, encoded = quantize_exl3(weight, H_data, quant_args, False)
            del weight, H_data
            for suffix, payload in encoded.items():
                out[f"{proj}.{suffix}"] = payload
        return out

    return requantize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="the JSON from bench/tail_variant_plan.py")
    ap.add_argument("--model", required=True, help="the checkpoint the plan was built from")
    ap.add_argument("--out", required=True, help="variant directory (mount it with -or)")
    ap.add_argument("--bpw", type=float, required=True,
                    help="target bits per weight for the tail: an integer or half-integer rate")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="default: validate and print, no torch, no writes")
    mode.add_argument("--apply", action="store_true", help="re-encode and write the variant shard")
    ap.add_argument("--filename", default="tail.safetensors")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="re-encode only the first N experts (a smoke run that still mounts)")
    ap.add_argument("--engine-root", default=REPO, help="the checkout holding exllamav3/")
    args = ap.parse_args(argv)

    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)
    plan = limit_plan(plan, args.limit)
    if plan.get("tail_bpw") is not None and float(plan["tail_bpw"]) != args.bpw:
        print(f"note: the plan prices {float(plan['tail_bpw']):g} bpw, --bpw is {args.bpw:g}: its byte "
              f"totals do not apply to this run", file=sys.stderr)

    try:
        headers = read_headers(args.model)
        calls = engine_calls(args.engine_root)
        problems, rows = analyse(headers, plan, args.bpw)
        if problems:
            for p in problems:
                print(f"refusing: {p}", file=sys.stderr)
            return 2
        print_engine_calls(calls)
        print_report(rows, plan, args.bpw)
        if args.limit:
            print(f"limited to the first {args.limit} expert(s) of the plan")
        if not args.apply:
            print()
            print("--apply would, per expert and projection: build the LinearEXL3 above from the stored "
                  "trellis/suh/svh, dequantize it to fp16, encode it at "
                  f"{args.bpw:g} bpw with the uncalibrated H_data placeholder, and write the returned "
                  f"trellis/suh/svh into {os.path.join(args.out, args.filename)}; nothing was written")
            return 0
        ok, why = engine_available()
        if not ok:
            print(f"refusing: {why}", file=sys.stderr)
            return 2
        debug_dir = os.path.join(args.out, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        index = expert_index(headers, plan_prefixes(plan))
        requantizer = make_engine_requantizer(index, args.bpw, args.device, args.seed, debug_dir)
        header = write_requantized_shard(args.model, plan, args.out, requantizer, args.filename)
        spec = os.path.join(args.out, "override.yml")
        with open(spec, "w", encoding="utf-8") as f:
            f.write(override_yaml(plan, args.out))
        total = sum(m["data_offsets"][1] - m["data_offsets"][0] for m in header.values())
        print(f"wrote {len(header)} tensor(s), {total / 2**20:,.1f} MiB -> "
              f"{os.path.join(args.out, args.filename)}")
        print(f"wrote {len(expert_globs(plan))} override key(s) -> {spec}")
        print(f"mount with: -or {spec}   (a Windows process needs the Windows path)")
        return 0
    except Refusal as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
