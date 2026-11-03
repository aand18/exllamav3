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

This fork's side is the same exercise with the other engine's names, and it is the side the transfer
review actually moves numbers FROM:

* ``promote_budget`` / ``promote_ratio`` / ``promote_decay`` / ``spares`` - promote-on-miss: ``spares``
  are the extra GPU slots a split layer registers for the worker's experts, ``promote_budget`` caps how
  many of them a step may fill, ``promote_ratio`` is the miss-to-least-used-resident count a candidate
  must beat, ``promote_decay`` the per-step halving of that history. Strata's equivalent is
  ``adapt_swaps`` per round; the unit differs, which is the M2-churn row's point.
* ``sweep_interval`` / ``sweep_max`` - the legacy (non-promote) placement sweep: every
  ``sweep_interval`` decode steps, up to ``sweep_max`` swaps across the tier.
* ``sweep_source`` - where a swap's bytes come from, read out of ``_split_swap_experts``' body:
  ``checkpoint`` when it re-reads through ``stc.get_tensor``, ``ram-copy`` when it copies from the
  worker's ``arena_views`` and nothing else, ``unknown`` when the function is gone or carries neither.
  The fork's answer is ``checkpoint`` and Strata's is ``ram-copy`` - the difference the review's M2
  row turns on.
* ``stream_min_rows`` - the row count below which prefill is streamed expert-by-expert instead of
  batched, i.e. the shape of the prefill the VRAM budget has to leave room for.
* ``cache_mode`` / ``cache_size`` / ``split`` - the served regime, read from
  ``bench/windows/state.json``: the KV cache's quantisation (the fork spells it ``cache_quant``; Q2 is
  ``"2,2"``, not ``Q2``), its size in tokens, and the ``-mcs`` / ``--moe_cpu_split`` tail-expert count
  ``run_pass.ps1`` passes (``bench/windows/run_pass.ps1:234``). ``512 - split`` is the number of
  experts per layer that run on the CPU, which a report should derive rather than copy.

Conventions: floats are read as written (``0.7f`` -> ``0.7``), not as their float32 value, because a
review compares defaults as authored. A value the source does not carry is ``None`` and gets no
``source_lines`` entry; ``source_lines`` otherwise maps the value's key to ``"<relative path>:<line>"``
so a report can quote the line the number came from.

Usage::

    python3 bench/strata_transfer_check.py            # print the extraction as JSON
    python3 bench/strata_transfer_check.py --fork     # print this fork's side instead
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

#: This fork's root, the module's own repository: `extract_exllama`'s default tree.
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


# --- this fork's side: the CPU expert tier's placement constants ------------------------------

#: The three files the fork side is read from, relative to `REPO`.
CPU_MLP = "exllamav3/modules/block_sparse_mlp_cpu.py"
MOE_HOST = "exllamav3/model/moe_cpu_host.py"
STATE = "bench/windows/state.json"

#: The function whose body decides where a swapped expert's bytes come from.
SWAP_FN = "_split_swap_experts"

# The knobs' defaults are written at the env read, `os.environ.get("NAME", DEFAULT)` - the read, not
# the mention. Anchoring on the call shape is what keeps a comment out: `block_sparse_mlp_cpu.py:1184`
# names `EXL3_MOE_CPU_SWAP_INTERVAL` in prose and carries no number, and a name-only pattern would read
# that line as the knob's value.
def _env_default(name: str) -> re.Pattern[str]:
    return re.compile(rf'os\.environ\.get\(\s*"{re.escape(name)}"\s*,\s*(-?\d+(?:\.\d+)?)\s*\)')


#: (key, file, env name, cast) for the knobs whose default is written at the env read.
_CPU_ENV = (
    ("promote_budget", CPU_MLP, "EXL3_MOE_CPU_PROMOTE_BUDGET", int),
    ("promote_ratio", CPU_MLP, "EXL3_MOE_CPU_PROMOTE_RATIO", float),
    ("promote_decay", CPU_MLP, "EXL3_MOE_CPU_PROMOTE_DECAY", float),
    ("spares", CPU_MLP, "EXL3_MOE_CPU_PROMOTE", int),
    ("sweep_interval", CPU_MLP, "EXL3_MOE_CPU_SWAP_INTERVAL", int),
    ("sweep_max", CPU_MLP, "EXL3_MOE_CPU_SWAP_MAX", int),
    ("stream_min_rows", MOE_HOST, "EXL3_MOE_STREAM_MIN_ROWS", int),
)

#: The regime file's own spellings, which are not the interface's: the fork calls the cache mode
#: `cache_quant` (its `-cq`), and `state.json`'s `split` is the `-mcs` value. (key, field, cast).
_STATE_FIELDS = (
    ("cache_mode", "cache_quant", str),
    ("cache_size", "cache_size", int),
    ("split", "split", int),
)

#: The fork interface's key order, so the JSON reads like the brief's dict, not like insertion order.
_FORK_KEYS = (
    "promote_budget", "promote_ratio", "promote_decay", "spares", "sweep_interval", "sweep_max",
    "sweep_source", "stream_min_rows", "cache_mode", "cache_size", "split",
)

#: The regime's live block: `cache_quant`/`split` appear again in prose under `history`.
ACTIVE_BLOCK = re.compile(r'^\s*"active"\s*:\s*\{\s*$')
#: A JSON field's value and whether it was quoted, from the line it is written on.
STATE_FIELD = re.compile(r'^\s*"{name}"\s*:\s*("[^"]*"|-?\d+)\s*,?\s*$')

#: A re-read through the checkpoint handle: the fork's swap source.
GET_TENSOR_CALL = re.compile(r"\bstc\.get_tensor\(")
#: A copy out of the worker's resident arena: Strata's shape, not this fork's swap.
ARENA_COPY = re.compile(r"\barena_views\b")


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


def _fn_body(lines: list[str], name: str) -> list[tuple[int, str]]:
    """`(line number, text)` for `name`'s def line and body, up to the next def at that indent."""
    for number, line in enumerate(lines):
        found = re.match(rf"^(\s*)def\s+{re.escape(name)}\s*\(", line)
        if not found:
            continue
        indent = len(found.group(1))
        body = [(number + 1, line)]
        for offset, later in enumerate(lines[number + 1:], number + 2):
            if later.strip() and len(later) - len(later.lstrip()) <= indent:
                break
            body.append((offset, later))
        return body
    return []


def _sweep_source(lines: list[str]) -> tuple[str, str | None]:
    """Where a swapped expert's bytes come from, and the line that says so.

    ``checkpoint`` iff the swap re-reads through ``stc.get_tensor``; ``ram-copy`` iff it copies from
    the worker's ``arena_views`` and nothing else; ``unknown`` when the function is gone or its body
    carries neither shape. The scan is scoped to the FUNCTION: ``arena_views`` also appears in the
    promote verify that compares an arena against the checkpoint
    (``block_sparse_mlp_cpu.py:600,620``, in ``_split_promote_verify``), and a file-wide search would
    read a RAM copy off a line the swap never runs.
    """
    body = _fn_body(lines, SWAP_FN)
    for number, line in body:
        if GET_TENSOR_CALL.search(line):
            return "checkpoint", f"{CPU_MLP}:{number}"
    for number, line in body:
        if ARENA_COPY.search(line):
            return "ram-copy", f"{CPU_MLP}:{number}"
    return "unknown", None


def _state_field(lines: list[str], name: str, cast: type) -> tuple[object, str | None]:
    """`(value, file:line)` of the regime's `name`, from the line it is written on in `active`.

    The value is read from the text rather than through `json.loads` so that the line is part of the
    answer: the review quotes the number and the file:line together, and a parser gives no position.
    """
    start = 0
    for number, line in enumerate(lines):
        if ACTIVE_BLOCK.match(line):
            start = number
            break
    pattern = re.compile(STATE_FIELD.pattern.format(name=re.escape(name)))
    for number, line in enumerate(lines[start:], start + 1):
        found = pattern.search(line)
        if found:
            return cast(found.group(1).strip('"')), f"{STATE}:{number}"
    return None, None


def extract_exllama(root: pathlib.Path = REPO) -> dict:
    """Read this fork's placement defaults out of `root`'s source text.

    Returns ``{"promote_budget", "promote_ratio", "promote_decay", "spares", "sweep_interval",
    "sweep_max", "sweep_source", "stream_min_rows", "cache_mode", "cache_size", "split",
    "source_lines"}`` - see the module docstring for what each one means, and for the
    ``source_lines`` contract (``key -> "<relative path>:<line>"``, absent for a value the source
    does not carry).
    """
    root = pathlib.Path(root)
    text = {rel: _lines(root / rel) for rel in (CPU_MLP, MOE_HOST, STATE)}
    out: dict[str, object] = {}
    source_lines: dict[str, str] = {}

    for key, rel, name, cast in _CPU_ENV:
        out[key] = None
        pattern = _env_default(name)
        for number, line in enumerate(text[rel], 1):
            found = pattern.search(line)
            if found:
                out[key] = cast(found.group(1))
                source_lines[key] = f"{rel}:{number}"
                break

    out["sweep_source"], sweep_line = _sweep_source(text[CPU_MLP])
    if sweep_line is not None:
        source_lines["sweep_source"] = sweep_line
    for key, field, cast in _STATE_FIELDS:
        out[key], state_line = _state_field(text[STATE], field, cast)
        if state_line is not None:
            source_lines[key] = state_line

    ordered = {key: out[key] for key in _FORK_KEYS}
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


def _selftest(root: pathlib.Path = STRATA, repo: pathlib.Path = REPO) -> int:
    """The assertions pytest would run, for a box that has no pytest. 0 = all passed.

    `root` is the Strata tree, `repo` this fork's; either side skips its real-tree arm when that tree
    is not there, so a fixture-only box still runs the branches.
    """
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

    # 5. This fork's side, same rule: the shape of the three files, not the files.  The comment on
    # line 7 names the interval knob and carries no number (as block_sparse_mlp_cpu.py:1184 does), and
    # line 8 is where the value comes from - the fixture separates the mention from the read.
    fork = _fake_root({
        CPU_MLP: (
            '_PROMOTE_BUDGET = max(0, int(os.environ.get("EXL3_MOE_CPU_PROMOTE_BUDGET", 6)))\n'
            '_PROMOTE_RATIO = float(os.environ.get("EXL3_MOE_CPU_PROMOTE_RATIO", 1.5))\n'
            '_PROMOTE_DECAY = float(os.environ.get("EXL3_MOE_CPU_PROMOTE_DECAY", 0.98))\n'
            '            spares = max(0, int(os.environ.get("EXL3_MOE_CPU_PROMOTE", 4)))\n'
            '    budget = int(os.environ.get("EXL3_MOE_CPU_SWAP_MAX", 64))\n'
            '\n'
            '        # every EXL3_MOE_CPU_SWAP_INTERVAL steps, quiesce (one full synchronize,\n'
            '        interval = int(os.environ.get("EXL3_MOE_CPU_SWAP_INTERVAL", 128))\n'
            '    def _split_swap_experts(self, r_cold, r_hot, mp):\n'
            '        stc = self.config.stc\n'
            '        dst.trellis.copy_(stc.get_tensor(src_key + ".trellis", dst.device))\n'
            '\n'
            '    def _split_saved_lists_ref(self):\n'
            '        return self._split_saved\n'),
        MOE_HOST: '        self.stream_min_rows = int(os.environ.get("EXL3_MOE_STREAM_MIN_ROWS", 32))\n',
        STATE: (
            '{\n  "active": {\n    "cache_size": 131072,\n    "cache_quant": "2,2",\n'
            '    "split": 376\n  },\n  "history": [\n'
            '    { "change": "split re-founded 400 -> 376" }\n  ]\n}\n'),
    })
    got = extract_exllama(fork)
    check("fork.promote_budget", got["promote_budget"], 6)
    check("fork.promote_ratio", got["promote_ratio"], 1.5)
    check("fork.promote_decay", got["promote_decay"], 0.98)
    check("fork.spares", got["spares"], 4)
    check("fork.sweep_max", got["sweep_max"], 64)
    check("fork.sweep_interval", got["sweep_interval"], 128)
    check("fork.sweep_source", got["sweep_source"], "checkpoint")
    check("fork.stream_min_rows", got["stream_min_rows"], 32)
    check("fork.cache_mode", got["cache_mode"], "2,2")
    check("fork.cache_size", got["cache_size"], 131072)
    check("fork.split", got["split"], 376)
    check("fork.keys", tuple(got), _FORK_KEYS + ("source_lines",))
    check("fork.source_lines.promote_budget", got["source_lines"].get("promote_budget"), f"{CPU_MLP}:1")
    check("fork.source_lines.spares", got["source_lines"].get("spares"), f"{CPU_MLP}:4")
    check("fork.source_lines.sweep_interval", got["source_lines"].get("sweep_interval"), f"{CPU_MLP}:8")
    check("fork.source_lines.sweep_source", got["source_lines"].get("sweep_source"), f"{CPU_MLP}:11")
    check("fork.source_lines.cache_mode", got["source_lines"].get("cache_mode"), f"{STATE}:4")
    check("fork.source_lines.split", got["source_lines"].get("split"), f"{STATE}:5")

    # 6. The swap-source branches the real tree does not use.  arena_views in the SWAP body is the
    # sibling engine's shape; this file mentions arena_views elsewhere (_split_promote_verify), so
    # this arm also says which function the scan is scoped to.
    ram = _fake_root({CPU_MLP: (
        "    def _split_swap_experts(self, r_cold, r_hot, mp):\n"
        "        arena = self.cpu_host.arena_views[slot]\n"
        "        dst.trellis.copy_(arena)\n")})
    check("ram-copy.sweep_source", extract_exllama(ram)["sweep_source"], "ram-copy")
    check("ram-copy.source_lines.sweep_source",
          extract_exllama(ram)["source_lines"].get("sweep_source"), f"{CPU_MLP}:2")
    elsewhere = _fake_root({CPU_MLP: (
        "    def _split_sweep_layer(self, budget):\n"
        "        arena = host.arena_views[ci]\n"
        "    def _split_swap_experts(self, r_cold, r_hot, mp):\n"
        "        dst.trellis.copy_(stc.get_tensor(src_key, dst.device))\n")})
    check("scoped.sweep_source", extract_exllama(elsewhere)["sweep_source"], "checkpoint")
    check("empty.sweep_source", extract_exllama(_fake_root({}))["sweep_source"], "unknown")
    check("empty.promote_budget", extract_exllama(_fake_root({}))["promote_budget"], None)

    # 7. This fork's real tree.  Every value whose regex could miss is asserted twice: the value, and
    # the line recorded for it.  A regex that fails closed to the number the tree already carries would
    # pass the value check on its own while pointing at a line that never states it (mutation m1 in the
    # task report reads 128 off the tier-stats knob this way).
    if (repo / CPU_MLP).is_file():
        real = extract_exllama(repo)
        check("real_fork.promote_budget", real["promote_budget"], 6)
        check("real_fork.promote_ratio", real["promote_ratio"], 1.5)
        check("real_fork.promote_decay", real["promote_decay"], 0.98)
        check("real_fork.spares", real["spares"], 4)
        check("real_fork.sweep_interval", real["sweep_interval"], 128)
        check("real_fork.sweep_max", real["sweep_max"], 64)
        check("real_fork.sweep_source", real["sweep_source"], "checkpoint")
        check("real_fork.stream_min_rows", real["stream_min_rows"], 32)
        check("real_fork.cache_mode", real["cache_mode"], "2,2")
        check("real_fork.cache_size", real["cache_size"], 131072)
        # The regime's split, not the review plan's: the plan says 381, state.json says 376 and its
        # own history records the 400 -> 376 re-founding.  512 - split also moves with it (136, not
        # 131), so a report must derive that number from this one rather than copy the plan.
        check("real_fork.split", real["split"], 376)
        fork_text = {rel: _lines(repo / rel) for rel in (CPU_MLP, MOE_HOST, STATE)}
        # Written out here rather than imported from _CPU_ENV: the point is an INDEPENDENT statement
        # of which knob each line must name, so a renamed table entry is visible instead of agreeing
        # with itself.  `sweep_interval` needs both halves - `EXL3_MOE_CPU_SWAP_INTERVAL` also appears
        # in the docstring at :1184, and only the token separates a comment from the read.
        anchors = {
            "promote_budget": "EXL3_MOE_CPU_PROMOTE_BUDGET",
            "promote_ratio": "EXL3_MOE_CPU_PROMOTE_RATIO",
            "promote_decay": "EXL3_MOE_CPU_PROMOTE_DECAY",
            "spares": '"EXL3_MOE_CPU_PROMOTE"',
            "sweep_interval": "EXL3_MOE_CPU_SWAP_INTERVAL",
            "sweep_max": "EXL3_MOE_CPU_SWAP_MAX",
            "sweep_source": "stc.get_tensor(",
            "stream_min_rows": "EXL3_MOE_STREAM_MIN_ROWS",
            "cache_mode": '"cache_quant"',
            "cache_size": '"cache_size"',
            "split": '"split"',
        }
        for key in _FORK_KEYS:
            where = real["source_lines"].get(key)
            total += 1
            if where is None:
                failures.append(f"FAIL real_fork.source_lines[{key}]: no line recorded")
                continue
            rel, number = where.rsplit(":", 1)
            total += 1
            if rel not in fork_text or not 0 < int(number) <= len(fork_text[rel]):
                failures.append(f"FAIL real_fork.source_lines[{key}]: {where} is not a line of {rel}")
                continue
            line = fork_text[rel][int(number) - 1]
            token = "stc.get_tensor(" if key == "sweep_source" else str(real[key])
            total += 1
            if anchors[key] not in line or token not in line:
                failures.append(
                    f"FAIL real_fork.source_lines[{key}]: {where} does not carry "
                    f"{token!r} for {anchors[key]!r}: {line.strip()!r}")
    else:
        print(f"skip fork tree: no {CPU_MLP} under {repo}")

    for line in failures:
        print(line)
    print(f"{total - len(failures)}/{total} checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true",
                        help="run this module's own assertions and exit non-zero on a failure")
    parser.add_argument("--fork", action="store_true",
                        help="print this fork's extraction instead of Strata's")
    parser.add_argument("--root", type=pathlib.Path, default=None,
                        help=f"the tree to read (default: {STRATA} or $STRATA_ROOT, "
                             f"{REPO} for --fork)")
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO,
                        help=f"this fork's tree, for --selftest (default: {REPO})")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest(args.root or STRATA, args.repo_root)
    if args.fork:
        tree = args.root or REPO
        if not (tree / CPU_MLP).is_file():
            print(f"no fork tree at {tree}: pass --root", file=sys.stderr)
            return 2
        print(json.dumps(extract_exllama(tree), indent=2))
        return 0
    root = args.root or STRATA
    if not (root / GENERATE).is_file():
        print(f"no Strata tree at {root}: pass --root or set STRATA_ROOT", file=sys.stderr)
        return 2
    print(json.dumps(extract_strata(root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
