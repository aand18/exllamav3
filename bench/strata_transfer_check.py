#!/usr/bin/env python3
"""Strata's half of the transfer review: the placement constants, read from its own source.

Strata is the sibling engine this fork took its adaptive expert tier from. The review has to compare
the two engines' defaults, and prose is the wrong witness for that: a constant quoted in a report
drifts the moment either source changes. So this module reads the SOURCE - two named files, by regex -
and returns what it says, together with the file:line each value came from. It never imports Strata,
never builds it, never runs it: stdlib only, no GPU, no torch.

What each value is, and why the review needs it:

* ``adapt_every`` / ``adapt_swaps`` / ``adapt_decay`` - the adaptive tier's cadence. Strata swaps up
  to ``adapt_swaps`` experts per adaptation round OVER THE WHOLE TIER, every ``adapt_every`` rounds,
  decayed by ``adapt_decay``; the fork's nearest knobs are per-step budgets, which is the unit trap
  the matrix's M2-churn row exists to catch.
* ``slot_pool`` - one shared admission counter (``global``) or a per-layer slot range
  (``per-layer``). The default in ``src/program/generate.cpp`` decides; the header's member default
  is the fallback when the flag is gone, which is also the corroboration the selftest checks.
* ``swap_source`` - WHERE an adapted expert's bytes come from. Strata's adaptive swap is a RAM <-> VRAM
  exchange: the expert the GPU gives up is copied back into an exchange buffer while the incoming one
  is copied out of the resident RAM copy, so ``no swap reads the file``. That is the semantics the
  fork's name-matching "swap" does NOT have (there, a swap is two checkpoint reads), and the matrix's
  M2 row turns on it. ``checkpoint`` is recorded when a swap path names a checkpoint (or ``get_tensor``)
  read instead, ``unknown`` when neither shape is in the text.
* ``kv_resident_tokens`` / ``prefill_chunk`` - the two scale levers (KV cells per QSA layer kept in
  VRAM, prompt tokens per chunk) that set how much VRAM is left for expert slots. 0 means "not
  streamed" / "the token path", which is not the same as absent: absent is ``None``.

Conventions: floats are read as written (``0.7f`` -> ``0.7``), not as their float32 value, because a
review compares defaults as authored. A value the source does not carry is ``None`` and gets no
``source_lines`` entry; ``source_lines`` otherwise maps the value's key to ``"<relative path>:<line>"``
so a report can quote the line the number came from.

Usage::

    python3 bench/strata_transfer_check.py            # print the extraction as JSON
    python3 bench/strata_transfer_check.py --selftest # run this module's own assertions (no pytest)
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

#: Strata's tree. Read-only: it belongs to the sibling investigation. Override with ``STRATA_ROOT``
#: (or ``--root``) to point at another checkout, e.g. the commit the review pinned.
STRATA = pathlib.Path(os.environ.get("STRATA_ROOT", "/home/dev/UnbiasedStrata"))

#: This fork's root, the module's own repository. Unused here; the fork-side extractor lands next to
#: this one and needs the same answer.
REPO = pathlib.Path(__file__).resolve().parent.parent

#: The only two files the Strata side is read from. Relative to the tree's root.
GENERATE = "src/program/generate.cpp"
EXPERT_CACHE = "include/strata/core/expert_cache.hpp"

# Struct members, anchored on the declared type so an assignment or a usage string cannot match:
# `o.adapt_swaps = ...`, `--adapt-swaps` and `peer_adapt_swaps` are all different text.
ADAPT_EVERY = re.compile(r"\bint\s+adapt_every\s*=\s*(-?\d+)\s*;")
ADAPT_SWAPS = re.compile(r"\bint\s+adapt_swaps\s*=\s*(-?\d+)\s*;")
ADAPT_DECAY = re.compile(r"\bfloat\s+adapt_decay\s*=\s*(-?[0-9.]+)f?\s*;")
KV_RESIDENT = re.compile(r"\bint64_t\s+kv_resident\s*=\s*(-?\d+)\s*;")
PREFILL_CHUNK = re.compile(r"\bint64_t\s+prefill_chunk\s*=\s*(-?\d+)\s*;")

#: The option's own default: `--expert-cache-per-layer` is off unless this says true.
PER_LAYER_FLAG = re.compile(r"\bbool\s+expert_cache_per_layer\s*=\s*(true|false)\s*;")
#: The slot pool's member default, two files down: per-layer admission is off by default.
PER_LAYER_MEMBER = re.compile(r"\bbool\s+per_layer_\s*=\s*(true|false)\s*;")

#: The sentence that states the adaptive swap reads no file: the resident RAM copy is the source.
EXCHANGE_COMMENT = re.compile(r"exchange them with the GPU cache \(no file reads\)")

#: A swap path that names a checkpoint read: all three notions on ONE line, in any order.
#: Strata's conversation checkpoints mention "read" freely, so the swap word is what scopes this.
SWAP_NAMES_CHECKPOINT = re.compile(
    r"(?=[^\n]*\b(?:swap|swaps|swapping|adapt_swaps?|exchang\w*)\b)"
    r"(?=[^\n]*\b(?:checkpoint|get_tensor)\b)"
    r"(?=[^\n]*\b(?:read|reads|re-?reads|load|loads|copy|copies)\b)",
    re.IGNORECASE,
)

#: (key, file, pattern, cast) for the plain defaults, in the order the interface lists them.
_SCALARS = (
    ("adapt_every", GENERATE, ADAPT_EVERY, int),
    ("adapt_swaps", GENERATE, ADAPT_SWAPS, int),
    ("adapt_decay", GENERATE, ADAPT_DECAY, float),
    ("kv_resident_tokens", GENERATE, KV_RESIDENT, int),
    ("prefill_chunk", GENERATE, PREFILL_CHUNK, int),
)


def _lines(path: pathlib.Path) -> list[str]:
    """The file's lines, or an empty list when it is not there (a fixture tree may name only one)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []


def _slot_pool(text: dict[str, list[str]]) -> tuple[str, str | None]:
    """`(value, file:line)` of the admission policy's default, or `("global", None)` if neither file says."""
    for rel, pattern in ((GENERATE, PER_LAYER_FLAG), (EXPERT_CACHE, PER_LAYER_MEMBER)):
        for number, line in enumerate(text[rel], 1):
            found = pattern.search(line)
            if found:
                return ("per-layer" if found.group(1) == "true" else "global"), f"{rel}:{number}"
    return "global", None


def _swap_source(text: dict[str, list[str]]) -> tuple[str, str | None]:
    """Where a swapped expert's bytes come from, and the line that says so."""
    for number, line in enumerate(text[GENERATE], 1):
        if EXCHANGE_COMMENT.search(line):
            return "ram-copy", f"{GENERATE}:{number}"
    for number, line in enumerate(text[GENERATE], 1):
        if SWAP_NAMES_CHECKPOINT.search(line):
            return "checkpoint", f"{GENERATE}:{number}"
    return "unknown", None


def extract_strata(root: pathlib.Path = STRATA) -> dict:
    """Read Strata's placement defaults out of `root`'s source text.

    Returns ``{"adapt_every", "adapt_swaps", "adapt_decay", "slot_pool", "swap_source",
    "kv_resident_tokens", "prefill_chunk", "source_lines"}`` - see the module docstring for what each
    one means, and for the ``source_lines`` contract (``key -> "<relative path>:<line>"``, absent for a
    value the source does not carry).
    """
    root = pathlib.Path(root)
    text = {rel: _lines(root / rel) for rel in (GENERATE, EXPERT_CACHE)}
    out: dict[str, object] = {}
    source_lines: dict[str, str] = {}

    for key, rel, pattern, cast in _SCALARS:
        out[key] = None
        for number, line in enumerate(text[rel], 1):
            found = pattern.search(line)
            if found:
                out[key] = cast(found.group(1))
                source_lines[key] = f"{rel}:{number}"
                break

    out["slot_pool"], slot_pool_line = _slot_pool(text)
    if slot_pool_line is not None:
        source_lines["slot_pool"] = slot_pool_line
    out["swap_source"], swap_source_line = _swap_source(text)
    if swap_source_line is not None:
        source_lines["swap_source"] = swap_source_line

    # The interface's order: the cadence, the pool, the swap's source, then the two scale levers.
    ordered = {key: out[key] for key, _, _, _ in _SCALARS[:3]}
    ordered["slot_pool"] = out["slot_pool"]
    ordered["swap_source"] = out["swap_source"]
    ordered["kv_resident_tokens"] = out["kv_resident_tokens"]
    ordered["prefill_chunk"] = out["prefill_chunk"]
    ordered["source_lines"] = source_lines
    return ordered


def _fake_root(tree: dict[str, str]) -> pathlib.Path:
    """A root whose named files contain the given text (a fresh temporary tree, removed at exit).

    The one body the test module and the selftest both use: a value asserted in one and not the other
    would be a contract with two readings.
    """
    root = pathlib.Path(tempfile.mkdtemp(prefix="strata-tree-"))
    atexit.register(shutil.rmtree, root, True)
    for name, body in tree.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def _selftest(root: pathlib.Path = STRATA) -> int:
    """The assertions pytest would run, for a box that has no pytest. 0 = all passed."""
    failures: list[str] = []
    total = 0

    def check(name: str, got: object, want: object) -> None:
        nonlocal total
        total += 1
        if got != want:
            failures.append(f"FAIL {name}: got {got!r}, want {want!r}")

    # 1. The plan's fixture: the shape of the real file, not the file.
    fixture = _fake_root({
        GENERATE: (
            "    int adapt_every = 4;\n"
            "    float adapt_decay = 0.7f;\n"
            "    int adapt_swaps = 96;\n"
            '    "exchange them with the GPU cache (no file reads)"\n'),
    })
    got = extract_strata(fixture)
    check("fixture.adapt_every", got["adapt_every"], 4)
    check("fixture.adapt_swaps", got["adapt_swaps"], 96)
    check("fixture.adapt_decay", got["adapt_decay"], 0.7)
    check("fixture.swap_source", got["swap_source"], "ram-copy")
    check("fixture.slot_pool", got["slot_pool"], "global")
    check("fixture.kv_resident_tokens", got["kv_resident_tokens"], None)
    check("fixture.prefill_chunk", got["prefill_chunk"], None)
    check("fixture.source_lines.adapt_every", got["source_lines"].get("adapt_every"), f"{GENERATE}:1")
    check("fixture.source_lines.adapt_swaps", got["source_lines"].get("adapt_swaps"), f"{GENERATE}:3")
    check("fixture.source_lines.slot_pool", "slot_pool" in got["source_lines"], False)

    # 2. A per-layer flag default makes the pool per-layer.
    per_layer = _fake_root({GENERATE: "    bool expert_cache_per_layer = true;\n"})
    check("per-layer.slot_pool", extract_strata(per_layer)["slot_pool"], "per-layer")

    # 2b. The header decides when generate.cpp no longer declares the flag.  Without this arm the
    # fallback could be renamed out of the regex and nothing would say so: the real tree answers from
    # the flag first, so the fallback firing would only be noticed by a reviewer reading the code.
    header_only = _fake_root({EXPERT_CACHE: "    bool per_layer_ = true;\n"})
    got = extract_strata(header_only)
    check("header-fallback.slot_pool", got["slot_pool"], "per-layer")
    check("header-fallback.source_lines.slot_pool", got["source_lines"].get("slot_pool"), f"{EXPERT_CACHE}:1")

    # 3. A swap path that names a checkpoint read is "checkpoint"; nothing named is "unknown".
    checkpoint = _fake_root({
        GENERATE: "    // a swap re-reads the expert from the checkpoint before the copy\n"})
    check("checkpoint.swap_source", extract_strata(checkpoint)["swap_source"], "checkpoint")
    check("empty.swap_source", extract_strata(_fake_root({}))["swap_source"], "unknown")
    check("empty.slot_pool", extract_strata(_fake_root({}))["slot_pool"], "global")
    check("empty.adapt_every", extract_strata(_fake_root({}))["adapt_every"], None)

    # 4. The real tree, when it is on this box: the values the review records, and honest file:lines.
    if (root / GENERATE).is_file():
        real = extract_strata(root)
        check("real.adapt_every", real["adapt_every"], 4)
        check("real.adapt_swaps", real["adapt_swaps"], 96)
        check("real.adapt_decay", real["adapt_decay"], 0.7)
        check("real.swap_source", real["swap_source"], "ram-copy")
        check("real.slot_pool", real["slot_pool"], "global")
        check("real.kv_resident_tokens", real["kv_resident_tokens"], 0)
        check("real.prefill_chunk", real["prefill_chunk"], 0)
        # The header's member default is the corroboration for the flag's: on the real tree the two
        # must agree, or one of them moved and the review is quoting a shape the pool no longer has.
        header_pool, header_line = _slot_pool({GENERATE: [], EXPERT_CACHE: _lines(root / EXPERT_CACHE)})
        check("real.header_slot_pool", header_pool, real["slot_pool"])
        check("real.header_slot_pool_line", header_line is not None, True)
        text = {rel: _lines(root / rel) for rel in (GENERATE, EXPERT_CACHE)}
        for key, where in real["source_lines"].items():
            rel, number = where.rsplit(":", 1)
            total += 1
            if rel not in text or not 0 < int(number) <= len(text[rel]):
                failures.append(f"FAIL real.source_lines[{key}]: {where} is not a line of {rel}")
                continue
            # The line must carry the value it is claimed for: a mis-claimed line is the drift
            # this extraction exists to make loud.
            line = text[rel][int(number) - 1]
            value = real.get(key)
            if key in {"swap_source", "slot_pool"}:
                token = {"swap_source": "exchange them with the GPU cache",
                         "slot_pool": "expert_cache_per_layer" if rel == GENERATE else "per_layer_"}[key]
            elif isinstance(value, float):
                token = repr(value).rstrip("0").rstrip(".") or "0"
            else:
                token = str(value)
            if token not in line:
                failures.append(f"FAIL real.source_lines[{key}]: {where} does not carry {token!r}: {line.strip()!r}")
    else:
        print(f"skip real tree: no {GENERATE} under {root} (set STRATA_ROOT to check another checkout)")

    for line in failures:
        print(line)
    print(f"{total - len(failures)}/{total} checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true",
                        help="run this module's own assertions and exit non-zero on a failure")
    parser.add_argument("--root", type=pathlib.Path, default=STRATA,
                        help=f"Strata's tree to read (default: {STRATA}, or $STRATA_ROOT)")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest(args.root)
    if not (args.root / GENERATE).is_file():
        print(f"no Strata tree at {args.root}: pass --root or set STRATA_ROOT", file=sys.stderr)
        return 2
    print(json.dumps(extract_strata(args.root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
