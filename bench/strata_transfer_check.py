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

The review's own record is ``MATRIX``: one row per mechanism (M1 budget, M2 churn, M3 miss split, M4
speculation yield, M5 PLE reads, M7 deployment), each carrying the values the two extractors read, the
arithmetic that turns them into the row's claim, a verdict from ``faithful``/``diverged``/``absent``/
``incompatible``, the justification for anything but ``faithful``, and the run id that measured it or
the literal ``unmeasured``. ``check`` compares every recorded value against the extractors (and
demands the source line), every derived number against its own recomputation, and the citation
discipline (a run id or ``unmeasured``; a justification where one is owed). The report that quotes
this matrix is ``wiki/plans/v165-reports/T31-strata-transfer.md``.

Usage::

    python3 bench/strata_transfer_check.py            # print the extraction as JSON
    python3 bench/strata_transfer_check.py --fork     # print this fork's side instead
    python3 bench/strata_transfer_check.py --check    # the matrix against both trees, one line a row
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


# --- the matrix: one row per Strata mechanism, and this review's verdict on it ------------------

#: The verdict vocabulary. `faithful` alone may omit a justification: it says the mechanism
#: transferred with its constants intact, and the compared values are the record of that. Every other
#: verdict says the fork differs, lacks the mechanism, or cannot take it - and a reader who finds the
#: difference with no reason recorded will "fix" it back, so the reason is mandatory.
VERDICTS = ("faithful", "diverged", "absent", "incompatible")

#: A harness run id: ``<tag>-<YYYYMMDD>-<HHMMSS>``, the ``run_id`` column of ``results/passes.csv``.
RUN_ID = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-\d{8}-\d{6}$")

#: The checkpoint's own geometry, from its ``config.json`` (``text_config``): 512 experts per layer,
#: 48 MoE layers, top-10 routing, 47.03 GB of expert bytes = about 1.91 MB per expert.
EXPERTS_PER_LAYER = 512
MOE_LAYERS = 48

#: The matrix. One row per mechanism: the values the two extractors read (`strata`, `fork`), the
#: arithmetic that turns them into the row's claim (`derived`), the verdict, why, and the run id that
#: measured it or the literal ``unmeasured``. `check` compares every recorded value against the
#: extractors and every derived number against its own recomputation, so a constant that drifts in
#: either engine - or a division that no longer adds up - fails loudly instead of reading plausibly.
#:
#: Two regime values are recorded twice on purpose (M1, M7). This worktree's
#: ``bench/windows/state.json`` says ``split 376, cache_size 131072, cache_quant "2,2"``; the arms in
#: ``results/passes.csv`` ran ``-mcs 381 -cs 262144 -cq 4,4`` (the deployed ``_agent/v165`` copy of
#: the same file). Both are stated, and M1's "131 of 512 experts per layer" is derived from the
#: MEASURED 381 and labelled as such - 376 would give 136.
MATRIX: list[dict] = [
    {
        "id": "M1-budget",
        "strata": {"kv_resident_tokens": 0, "prefill_chunk": 0},
        "fork": {"cache_size": 131072, "cache_mode": "2,2", "split": 376},
        "derived": {
            "split_measured": 381,              # results/passes.csv, t27-*/t29-* args: -mcs 381
            "experts_per_layer_repo": 136,      # 512 - 376, this worktree's state.json
            "experts_per_layer_measured": 131,  # 512 - 381, the regime the arms ran
            "strata_slots": 8301,               # log: "expert cache 8301 slots, 15.74 GiB"
            "strata_experts_per_layer": 173,    # 8301 / 48
            "experts_short": 42,                # 173 - 131
            # The lever's own arithmetic, in bytes rather than in a quoted range. A QSA layer holds
            # 2 KV heads x 256 head_dim = 512 values/token: at "4,4" that is 16 groups x (4+4) bits
            # x 4 B + 2 x 16 fp16 scales x 2 B = 576 B, plus the fp16 indexer planes
            # (128 + 128/4) x 2 B = 320 B that do NOT scale with the width (cache/qsa.py:16-19) =
            # 896 B/token/layer over 12 full-attention layers = 10 752 B/token (bench/size_check.py
            # gives the same 2688 MiB at 262144 cells). At 1.91 MB per expert (47.03 GB / 512 / 48).
            "kv_bytes_per_token_layer": 896,
            "kv_layers": 12,
            "kv_cells_measured": 262144,
            "kv_cells_65536": 65536,
            "kv_cells_32768": 32768,
            "freed_gb_65536": 2.11,             # (262144 - 65536) x 896 x 12 / 1e9
            "freed_gb_32768": 2.47,             # (262144 - 32768) x 896 x 12 / 1e9
            "expert_mb": 1.91,
            "slots_65536": 1106,                # freed bytes / 1.91 MB
            "slots_32768": 1291,
            "gain_65536": 23,                   # slots / 48
            "gain_32768": 27,
            "experts_after_65536": 154,         # 131 + 23
            "experts_after_32768": 158,         # 131 + 27
            "short_65536": 19,                  # 173 - 154
            "short_32768": 15,                  # 173 - 158
            # The card, from bench/size_check.py's own output in the t29 artefact section 1.
            "trunk_mib": 17551,
            "trunk_gib": 17.1,
        },
        "verdict": "absent",
        "justification": (
            "the fork's 131 experts/layer is a static prefix of one uniform per-layer budget at the "
            "measured split 381 (512 - 381; this worktree's state.json says 376, i.e. 136); Strata's "
            "173/layer is 8301 global slots over 48 layers (15.74 GiB) that follow routing. Three "
            "levers would move the fork: cache_size 262144 -> 32768/65536, memory.sysmem_kv_cache "
            "~1800 MB, then cpu_moe_split_experts below 381. The KV bytes, from the checkpoint's "
            "config and the engine's cache shapes: 2 KV heads x 256 head_dim = 512 values/token/layer, "
            "at \"4,4\" 16 groups x 8 bits x 4 B + 32 B of fp16 scales = 576 B, plus 320 B of fp16 "
            "indexer planes that do not scale with the width (cache/qsa.py:16-19) = 896 B/token/layer "
            "over 12 attention layers, so 2.82 GB at 262144 cells. Dropping to 65536 or 32768 cells "
            "frees 2.11 or 2.47 GB, which at 1.91 MB per expert (47.03 GB / 512 / 48) is 1106 or 1291 "
            "slots = +23..27 experts/layer, so 131 -> 154..158: still 15-19 short of 173. Research note "
            "1's M1 verdict (\"~1.9-2.4 GB VRAM ~ 500-650 more expert slots ~ +10-13 experts per "
            "layer\") does not divide by the 1.91 MB per expert that note states in its own section "
            "M9: 1.9-2.4 GB / 1.91 MB is 1000-1250 slots, and the conversion above is the corrected "
            "one. Reaching 80% by residency "
            "instead needs 410 of 512 experts/layer = 410 x 48 x 1.91 MB = 37.6 GB ~ 38 GB of VRAM, "
            "against a 24 GiB card whose dense trunk already holds 17 551 MiB = 17.1 GiB "
            "(bench/size_check.py, t29 artefact section 1). And residency alone does not buy Strata's "
            "coverage: a static profile at 8000 slots covers 28.4% of reads (note 2 section 2.2), the "
            "rest comes from the tier following routing, i.e. from M2. Absent, not diverged."
        ),
        "measured": "unmeasured",
    },
    {
        "id": "M2-churn",
        "strata": {
            "adapt_every": 4, "adapt_swaps": 96, "adapt_decay": 0.7,
            "slot_pool": "global", "swap_source": "ram-copy",
        },
        "fork": {
            "promote_budget": 6, "promote_ratio": 1.5, "promote_decay": 0.98, "spares": 4,
            "sweep_interval": 128, "sweep_max": 64, "sweep_source": "checkpoint",
        },
        "derived": {
            "measured_swaps_per_adaptation": 60.5,  # strata-vs-exllama-research.md section 2.2
            "swaps_per_round_cap": 24,              # 96 / 4
            "swaps_per_round_measured": 15.1,       # 60.5 / 4
            "promotions_t27": 710,                  # perf-t27-B-pin8-f4-20261103-122212.log, final
            "ladder_steps": 3528,                   # 27 x 128 + 72: the perf ladder's windows
            "promotions_per_step_t27": 0.201,       # 710 / 3528
            "sweep_swaps_per_step": 0.5,            # 64 / 128
            "churn_ratio": 75,                      # 15.1 / 0.201
            "promotions_t29": 8669,                 # t29-C1, "/ 8669, at step 3584"
            "steps_t29": 3584,                      # 28 x 128
            "promotions_per_step_t29": 2.42,        # 8669 / 3584
            "budget_raise_ratio": 12,               # 2.42 / 0.201 (8669 / 710 = 12.2)
        },
        "verdict": "diverged",
        "justification": (
            "units and rate, not names: Strata spends up to adapt_swaps 96 on ONE adaptation every "
            "adapt_every 4 decode rounds over ONE global pool (slot_pool global), fed from the "
            "resident RAM copy (swap_source ram-copy, 0 file reads), so 96/4 = 24 swaps/round is the "
            "cap and a measured adaptation did 60.5 -> 15.1/round. The fork at the t27 operating "
            "point promoted 710 times over the 3528-step perf ladder = 0.20/step (spares 4, "
            "PROMOTE_BUDGET 6/step, ratio 1.5, decay 0.98), and its legacy sweep is 64 swaps per 128 "
            "steps = 0.5/step with two checkpoint reads per swap (sweep_source checkpoint). "
            "15.1/0.201 = 75x, the review's 75x. Per committed token it is 31x (Strata 15.1/2.45 = "
            "6.2; a perf step is one token because perf.py registers no draft), and t29-C1 shows "
            "what the budget does once the pool stops capping it: 8669 promotions / 3584 steps = "
            "2.42/step, 12x the t27 rate (2.42/0.201 = 12.0, 8669/710 = 12.2; the arm also raised "
            "the spare count 4 -> 16), for +40.6 points of coverage (31.7 -> 72.3 % pooled)."
        ),
        "measured": "t29-C1-pin-unset-20261103-221302",
    },
    {
        "id": "M3-miss-split",
        "strata": {},
        "fork": {"stream_min_rows": 32},
        "derived": {
            "window_rows": 4,        # the deployed draft cap, i.e. the widest a verify window gets
            "rows_short": 28,        # 32 - 4
            "gate_factor": 8.0,      # 32 / 4
        },
        "verdict": "absent",
        "justification": (
            "the miss goes somewhere else. Strata's pcie_frac default is the model's own (-1.0 at "
            "generate.cpp:470, read as 0.55 DMA for the native packs and scaled by a live PCIe "
            "probe), so ~55% of a verify round's distinct missed experts are DMA'd and computed on "
            "the GPU. The fork's streamed path exists (moe_cpu_host.py:1968-1998) but is gated "
            "twice: rows < stream_min_rows 32 returns before it (:1981), and an expert needs "
            "stream_t 8 assigned tokens (:139, read by hand - not an extracted key) - so a 4-row "
            "window reaches neither and 100% of decode misses go to the worker process. The delta "
            "that would test the fork's half is env-only, EXL3_MOE_STREAM_MIN_ROWS=2 plus "
            "EXL3_MOE_STREAM_T=2, with no regime flag; nothing on this box has run it."
        ),
        "measured": "unmeasured",
    },
    {
        "id": "M4-spec-yield",
        "strata": {},
        "fork": {},
        "derived": {
            "window_cap": 4,          # draft_num_tokens, the deployed DFlash2 regime
            "tokens_low": 2.2,        # fork tokens/window, note 1 section 2
            "tokens_high": 2.4,
            "strata_low": 2.5,        # Strata tokens/window, note 1 section 2
            "strata_high": 3.5,
            "yield_ratio_low": 1.04,  # 2.5 / 2.4
            "yield_ratio_high": 1.59,  # 3.5 / 2.2
            "window_cap_minus_high": 1.6,  # 4 - 2.4: how much of the window goes unused
        },
        "verdict": "diverged",
        "justification": (
            "window arithmetic: tokens/window = 1 + the sum of per-position acceptance, so the "
            "deployed DFlash2 cap num_draft_tokens 4 truncates the sum at 4 and the fork measures "
            "2.2-2.4 of 4 rows, dynamic_draft only ever shortening (generator.py:1128-1146). Strata "
            "drafts <= 3 with the model's own MTP head plus a prompt-lookup drafter picked per round "
            "by a cost model (docs/upstream/DETAILS.md:1032-1035) and measures 2.5-3.5 "
            "tokens/window: 3.5/2.2 = 1.59x at best, 2.5/2.4 = 1.04x at worst. The second research "
            "note's own log pass (strata-vs-exllama-research.md section 1) reads 2.42-2.47 on both "
            "sides, so the divergence stands on note 1's window arithmetic and is not reproduced by "
            "note 2's requests. No arm of either campaign carries a draft (perf.py registers none; "
            "draft arguments belong to -Runner spec_decode), so this row is unmeasured."
        ),
        "measured": "unmeasured",
    },
    {
        "id": "M5-ple",
        "strata": {},
        "fork": {},
        "derived": {
            "rows_per_token": 16,       # the model's ngram_size x heads_per_ngram
            "row_bytes": 90,            # IQ4_NL row, ple_reader.hpp:3
            "bytes_per_token": 1440,    # 16 x 90
            "cache_rows": 1048576,      # 1 << 20 rows, Strata's row cache
            "cache_mib": 90,            # 1048576 x 90 B / 1 MiB
            "cache_read_share": 0.82,   # "up to ~82% of reads"
            "recurrence_low": 0.20,     # "within one long prompt 20-34% of rows recur"
            "recurrence_high": 0.34,
            "unhidden_ms_low": 2.1,     # generate.cpp:393-395, the un-hidden gather
            "unhidden_ms_high": 2.6,
        },
        "verdict": "absent",
        "justification": (
            "absent, and cheap to bound: Strata hides the 16 unbuffered row reads of a decode token "
            "(90 B/row, 26.8 GiB table never in RAM) behind the embedding and layer 0 with an "
            "issue/collect split, and serves repeats from a bounded 1M-row (90 MiB) cache that "
            "measures up to 82% of reads with 20-34% recurrence inside one prompt "
            "(ple_reader.hpp:13-16). The fork has no such row cache: every gather goes to "
            "ext.ngram_gather_cpu over a FILE_FLAG_NO_BUFFERING handle (row_table.py:282-312, opened "
            "at loader/safetensors.py:247-248), so a repeated row is read from disk again. Its two "
            "cheaper substitutes both miss decode: prefetch() declines decode-sized inputs "
            "(PREFETCH_MIN_TOKENS 256 positions, row_table.py:27, checked at "
            "ngram_embedding.py:334), and the read-ahead hint advise() (row_table.py:208-215) is "
            "posix_fadvise, i.e. a no-op on the Windows box this fork serves. 16 x 90 B = "
            "1440 B per token is the byte scale; Strata's own un-hidden gather is 2.1-2.6 ms/token "
            "(generate.cpp:393-395), which is the ceiling on a port, not a promise."
        ),
        "measured": "unmeasured",
    },
    {
        "id": "M7-deploy",
        "strata": {},
        "fork": {"cache_size": 131072, "cache_mode": "2,2", "split": 376},
        "derived": {
            "kv_vram_tokens_measured": 262144,   # the deployed regime: -cs 262144, -cq 4,4
            "kv_resident_tokens_strata": 32768,  # start-logged.ps1:54, --kv-resident 32768
            "kv_ratio": 8,                       # 262144 / 32768
            "batch_fork": 2,                     # tabbyAPI/config.yml:217, max_batch_size
            "batch_strata": 0,                   # generate.cpp:456, one request at a time
            "batch_delta": 2,                    # 2 - 0
            # Per-token bytes for both sides at the same context, so the VRAM delta is arithmetic and
            # not a range. The two engines agree on the K/V itself (2 heads x 256 dims at 4 bits + fp16
            # scales = 576 B/token/layer: the fork's packed words, Strata's kv_q4 cells); they differ
            # in the indexer planes (fork: 320 B/token/layer, fp16 at any width) and in where the
            # bytes LIVE. 12 full-attention layers in both.
            "fork_kv_bytes_token_layer": 896,     # 576 + 320 planes
            "strata_kv_bytes_token_layer": 576,   # Q4_0, K+V both heads, no planes (kv_q4.hpp:3-6)
            "kv_layers": 12,
            "fork_kv_gb": 2.82,                   # 262144 x 896 x 12 / 1e9
            "strata_kv_whole_gb": 1.81,           # 262144 x 576 x 12 / 1e9
            "strata_kv_whole_gib": 1.69,          # that table, in GiB: the log's "1.69 GiB" (log:378)
            "strata_kv_resident_gb": 0.23,        # 32768 x 576 x 12 / 1e9, in VRAM
            "kv_vram_delta_gb": 2.59,             # 2.82 - 0.23
        },
        "verdict": "diverged",
        "justification": (
            "deployment, not engine: Strata serves one request at a time (batch 0 default, "
            "src/program/generate.cpp:456, and /home/dev/UnbiasedStrata/start-logged.ps1 passes no "
            "--slots) with 32768 of 262144 KV cells resident "
            "(/home/dev/UnbiasedStrata/start-logged.ps1:54); the fork's deployed regime serves "
            "max_batch_size 2 (tabbyAPI/config.yml:217, outside this tree) with the whole cache in "
            "VRAM (sysmem_kv_cache 0, config.yml:441) - 262144 cells at the measured regime and "
            "131072 in this worktree's state.json. Per token per layer the two engines agree on the "
            "K/V: 576 B (2 heads x 256 dims at 4 bits + fp16 scales) = 6912 B/token over 12 layers, "
            "1.81 GB at 262144 cells, which is exactly the log's \"K/V in 1.69 GiB of pinned RAM\" "
            "(/mnt/c/Users/yoho/Downloads/UnbiasedStrata/logs/server-20261029-172352.err.log:378). "
            "The fork adds 320 B/token/layer of fp16 indexer planes (896 B/token/layer) and keeps "
            "all of it in VRAM: 2.82 GB. Strata keeps 32768 cells resident (0.23 GB) and the other "
            "229376 cells, 1.59 GB, in pinned RAM. So the VRAM delta at one context is 2.82 - 0.23 = "
            "2.59 GB - the same quantity the M1 lever frees (2.11-2.47 GB by target cell count), not "
            "a second one. Concurrency is anti-scaled: the log's one overlapping pair ran 49.8 T/s "
            "on the 19.5k-token stream and 8.1 T/s on the 551-token stream that shared the window, "
            "against 57.2-61.3 T/s for the three single-stream requests (note 1 section 2). No arm of "
            "either campaign varied batch or KV residency, so the divergence is real and unmeasured."
        ),
        "measured": "unmeasured",
    },
]

#: Sentinel: "the extractor has no such key", distinct from a key whose value is None.
_MISSING = object()


def _derived(row: dict) -> list[tuple[str, object]]:
    """The row's arithmetic, recomputed from the row's own inputs.

    Independent of `row["derived"]` by construction: the numbers on the right are the ones the
    sources wrote down, the operation is the one the justification claims, and the recorded result is
    compared against it. A row whose prose and arithmetic parted company fails here.
    """
    rid, d = row["id"], row.get("derived") or {}
    if rid == "M1-budget":
        measured = EXPERTS_PER_LAYER - d["split_measured"]
        per_layer = round(d["strata_slots"] / MOE_LAYERS)
        # The lever's arithmetic, recomputed in bytes: the KV per token, then the freed bytes at each
        # target cache size converted to expert slots at the expert's own size.
        kv_per_token = d["kv_bytes_per_token_layer"] * d["kv_layers"]
        expert_bytes = round(d["expert_mb"] * 1e6)
        out = [
            ("experts_per_layer_repo", EXPERTS_PER_LAYER - row["fork"]["split"]),
            ("experts_per_layer_measured", measured),
            ("strata_experts_per_layer", per_layer),
            ("experts_short", per_layer - measured),
            ("trunk_gib", round(d["trunk_mib"] / 1024, 1)),
        ]
        for label in ("65536", "32768"):
            freed = (d["kv_cells_measured"] - d[f"kv_cells_{label}"]) * kv_per_token
            slots = freed // expert_bytes
            gain = round(slots / MOE_LAYERS)
            after = measured + gain
            out += [
                (f"freed_gb_{label}", round(freed / 1e9, 2)),
                (f"slots_{label}", slots),
                (f"gain_{label}", gain),
                (f"experts_after_{label}", after),
                (f"short_{label}", per_layer - after),
            ]
        return out
    if rid == "M2-churn":
        s, f = row["strata"], row["fork"]
        measured = round(d["measured_swaps_per_adaptation"] / s["adapt_every"], 1)
        rate = round(d["promotions_t29"] / d["steps_t29"], 2)
        return [
            ("swaps_per_round_cap", s["adapt_swaps"] // s["adapt_every"]),
            ("swaps_per_round_measured", measured),
            ("sweep_swaps_per_step", round(f["sweep_max"] / f["sweep_interval"], 3)),
            ("promotions_per_step_t27", round(d["promotions_t27"] / d["ladder_steps"], 3)),
            ("churn_ratio", round(measured / d["promotions_per_step_t27"])),
            ("promotions_per_step_t29", rate),
            # The raise the review quotes: the t29 arm's rate over the t27 arm's, on the same ladder.
            ("budget_raise_ratio", round(rate / d["promotions_per_step_t27"])),
        ]
    if rid == "M3-miss-split":
        gate = row["fork"]["stream_min_rows"]
        return [
            ("rows_short", gate - d["window_rows"]),
            ("gate_factor", gate / d["window_rows"]),
        ]
    if rid == "M4-spec-yield":
        return [
            ("yield_ratio_low", round(d["strata_low"] / d["tokens_high"], 2)),
            ("yield_ratio_high", round(d["strata_high"] / d["tokens_low"], 2)),
            ("window_cap_minus_high", round(d["window_cap"] - d["tokens_high"], 2)),
        ]
    if rid == "M5-ple":
        return [
            ("bytes_per_token", d["rows_per_token"] * d["row_bytes"]),
            ("cache_mib", d["cache_rows"] * d["row_bytes"] // (1 << 20)),
        ]
    if rid == "M7-deploy":
        cells = d["kv_vram_tokens_measured"]
        fork_gb = cells * d["fork_kv_bytes_token_layer"] * d["kv_layers"] / 1e9
        whole_gb = cells * d["strata_kv_bytes_token_layer"] * d["kv_layers"] / 1e9
        resident_gb = d["kv_resident_tokens_strata"] * d["strata_kv_bytes_token_layer"] \
            * d["kv_layers"] / 1e9
        return [
            ("kv_ratio", cells // d["kv_resident_tokens_strata"]),
            ("batch_delta", d["batch_fork"] - d["batch_strata"]),
            ("fork_kv_gb", round(fork_gb, 2)),
            ("strata_kv_whole_gb", round(whole_gb, 2)),
            ("strata_kv_whole_gib", round(whole_gb * 1e9 / (1 << 30), 2)),
            ("strata_kv_resident_gb", round(resident_gb, 2)),
            ("kv_vram_delta_gb", round(fork_gb - resident_gb, 2)),
        ]
    return []


def _brief(items: list[str], limit: int) -> str:
    """The first `limit` items, then a count of the rest: one line, still names what was compared."""
    if not items:
        return "no extracted constant on either side (see justification)"
    head = ", ".join(items[:limit])
    return head if len(items) <= limit else f"{head} +{len(items) - limit} more"


def check(strata: dict, fork: dict, matrix: list[dict] | None = None) -> list[str]:
    """One `PASS`/`FAIL` line per matrix row: what was compared, and what the sources say.

    Three things are compared, in this order: every value a row records against the extractor's
    (`strata`, `fork`), and that the value has a source line (a value with no line is a claim with no
    witness); every number in `derived` against its own recomputation in `_derived`; and the review's
    citation discipline - a verdict from `VERDICTS`, a justification for every verdict but
    `faithful`, and `measured` a run id or the literal `unmeasured`.

    `matrix` defaults to `MATRIX`. A `FAIL` line names the row and the key, so the failure points at
    the constant rather than at the matrix; a row with several disagreements yields several lines.
    """
    matrix = MATRIX if matrix is None else matrix
    sources = {"strata": strata, "fork": fork}
    out: list[str] = []

    for row in matrix:
        rid = row.get("id", "?")
        fails: list[str] = []
        compared: list[str] = []
        count = 0

        for side in ("strata", "fork"):
            recorded = row.get(side) or {}
            if not isinstance(recorded, dict):
                fails.append(f"FAIL {rid}: {side} must be a dict of recorded values")
                continue
            extracted = sources[side]
            for key, want in recorded.items():
                count += 1
                where = (extracted.get("source_lines") or {}).get(key)
                got = extracted.get(key, _MISSING)
                if got is _MISSING:
                    fails.append(f"FAIL {rid}: {side}.{key} recorded {want!r}, "
                                 f"but the extractor carries no such value")
                elif got != want:
                    at = f" ({where})" if where else ""
                    fails.append(f"FAIL {rid}: {side}.{key} recorded {want!r}, "
                                 f"source says {got!r}{at}")
                elif where is None:
                    fails.append(f"FAIL {rid}: {side}.{key} is {want!r} "
                                 f"with no source line recorded")
                else:
                    compared.append(f"{side}.{key}={want!r} ({where})")

        recorded_derived = row.get("derived") or {}
        derived_items: list[str] = []
        try:
            computed = _derived(row)
        except KeyError as missing:
            fails.append(f"FAIL {rid}: derived check cannot run, the row has no {missing}")
            computed = []
        for name, got in computed:
            count += 1
            if name not in recorded_derived:
                fails.append(f"FAIL {rid}: derived.{name} = {got!r} is computed but not recorded")
            elif recorded_derived[name] != got:
                fails.append(f"FAIL {rid}: derived.{name} recorded {recorded_derived[name]!r}, "
                             f"arithmetic gives {got!r}")
            else:
                derived_items.append(f"{name}={got!r}")

        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            fails.append(f"FAIL {rid}: verdict {verdict!r} is not one of {', '.join(VERDICTS)}")
        elif verdict != "faithful" and not str(row.get("justification") or "").strip():
            fails.append(f"FAIL {rid}: a {verdict!r} verdict needs a justification")
        if verdict == "faithful" and not all(row.get(side) for side in ("strata", "fork")):
            fails.append(f"FAIL {rid}: a 'faithful' verdict needs a recorded value on both sides")
        measured = row.get("measured")
        if measured != "unmeasured" and not (isinstance(measured, str) and RUN_ID.match(measured)):
            fails.append(f"FAIL {rid}: measured {measured!r} is neither a run id "
                         f"(tag-YYYYMMDD-HHMMSS) nor 'unmeasured'")

        if fails:
            out.extend(fails)
            continue
        out.append(f"PASS {rid}: {count} comparison(s) - {_brief(compared, 3)}; "
                   f"derived {', '.join(derived_items) or 'none'}; "
                   f"{verdict}, measured {measured}")
    return out


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

    def expect(name: str, got: object, want: object) -> None:
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
    expect("fixture.adapt_every", got["adapt_every"], 4)
    expect("fixture.adapt_swaps", got["adapt_swaps"], 96)
    expect("fixture.adapt_decay", got["adapt_decay"], 0.7)
    expect("fixture.swap_source", got["swap_source"], "ram-copy")
    expect("fixture.slot_pool", got["slot_pool"], "global")
    expect("fixture.kv_resident_tokens", got["kv_resident_tokens"], None)
    expect("fixture.prefill_chunk", got["prefill_chunk"], None)
    expect("fixture.source_lines.adapt_every", got["source_lines"].get("adapt_every"), f"{GENERATE}:1")
    expect("fixture.source_lines.adapt_swaps", got["source_lines"].get("adapt_swaps"), f"{GENERATE}:3")
    expect("fixture.source_lines.slot_pool", "slot_pool" in got["source_lines"], False)

    # 2. A per-layer flag default makes the pool per-layer.
    per_layer = _fake_root({GENERATE: "    bool expert_cache_per_layer = true;\n"})
    expect("per-layer.slot_pool", extract_strata(per_layer)["slot_pool"], "per-layer")

    # 2b. The header decides when generate.cpp no longer declares the flag.  Without this arm the
    # fallback could be renamed out of the regex and nothing would say so: the real tree answers from
    # the flag first, so the fallback firing would only be noticed by a reviewer reading the code.
    header_only = _fake_root({EXPERT_CACHE: "    bool per_layer_ = true;\n"})
    got = extract_strata(header_only)
    expect("header-fallback.slot_pool", got["slot_pool"], "per-layer")
    expect("header-fallback.source_lines.slot_pool", got["source_lines"].get("slot_pool"), f"{EXPERT_CACHE}:1")

    # 3. A swap path that names a checkpoint read is "checkpoint"; nothing named is "unknown".
    checkpoint = _fake_root({
        GENERATE: "    // a swap re-reads the expert from the checkpoint before the copy\n"})
    expect("checkpoint.swap_source", extract_strata(checkpoint)["swap_source"], "checkpoint")
    expect("empty.swap_source", extract_strata(_fake_root({}))["swap_source"], "unknown")
    expect("empty.slot_pool", extract_strata(_fake_root({}))["slot_pool"], "global")
    expect("empty.adapt_every", extract_strata(_fake_root({}))["adapt_every"], None)

    # 4. The real tree, when it is on this box: the values the review records, and honest file:lines.
    if (root / GENERATE).is_file():
        real = extract_strata(root)
        expect("real.adapt_every", real["adapt_every"], 4)
        expect("real.adapt_swaps", real["adapt_swaps"], 96)
        expect("real.adapt_decay", real["adapt_decay"], 0.7)
        expect("real.swap_source", real["swap_source"], "ram-copy")
        expect("real.slot_pool", real["slot_pool"], "global")
        expect("real.kv_resident_tokens", real["kv_resident_tokens"], 0)
        expect("real.prefill_chunk", real["prefill_chunk"], 0)
        # The header's member default is the corroboration for the flag's: on the real tree the two
        # must agree, or one of them moved and the review is quoting a shape the pool no longer has.
        header_pool, header_line = _slot_pool({GENERATE: [], EXPERT_CACHE: _lines(root / EXPERT_CACHE)})
        expect("real.header_slot_pool", header_pool, real["slot_pool"])
        expect("real.header_slot_pool_line", header_line is not None, True)
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
    expect("fork.promote_budget", got["promote_budget"], 6)
    expect("fork.promote_ratio", got["promote_ratio"], 1.5)
    expect("fork.promote_decay", got["promote_decay"], 0.98)
    expect("fork.spares", got["spares"], 4)
    expect("fork.sweep_max", got["sweep_max"], 64)
    expect("fork.sweep_interval", got["sweep_interval"], 128)
    expect("fork.sweep_source", got["sweep_source"], "checkpoint")
    expect("fork.stream_min_rows", got["stream_min_rows"], 32)
    expect("fork.cache_mode", got["cache_mode"], "2,2")
    expect("fork.cache_size", got["cache_size"], 131072)
    expect("fork.split", got["split"], 376)
    expect("fork.keys", tuple(got), _FORK_KEYS + ("source_lines",))
    expect("fork.source_lines.promote_budget", got["source_lines"].get("promote_budget"), f"{CPU_MLP}:1")
    expect("fork.source_lines.spares", got["source_lines"].get("spares"), f"{CPU_MLP}:4")
    expect("fork.source_lines.sweep_interval", got["source_lines"].get("sweep_interval"), f"{CPU_MLP}:8")
    expect("fork.source_lines.sweep_source", got["source_lines"].get("sweep_source"), f"{CPU_MLP}:11")
    expect("fork.source_lines.cache_mode", got["source_lines"].get("cache_mode"), f"{STATE}:4")
    expect("fork.source_lines.split", got["source_lines"].get("split"), f"{STATE}:5")

    # 6. The swap-source branches the real tree does not use.  arena_views in the SWAP body is the
    # sibling engine's shape; this file mentions arena_views elsewhere (_split_promote_verify), so
    # this arm also says which function the scan is scoped to.
    ram = _fake_root({CPU_MLP: (
        "    def _split_swap_experts(self, r_cold, r_hot, mp):\n"
        "        arena = self.cpu_host.arena_views[slot]\n"
        "        dst.trellis.copy_(arena)\n")})
    expect("ram-copy.sweep_source", extract_exllama(ram)["sweep_source"], "ram-copy")
    expect("ram-copy.source_lines.sweep_source",
          extract_exllama(ram)["source_lines"].get("sweep_source"), f"{CPU_MLP}:2")
    elsewhere = _fake_root({CPU_MLP: (
        "    def _split_sweep_layer(self, budget):\n"
        "        arena = host.arena_views[ci]\n"
        "    def _split_swap_experts(self, r_cold, r_hot, mp):\n"
        "        dst.trellis.copy_(stc.get_tensor(src_key, dst.device))\n")})
    expect("scoped.sweep_source", extract_exllama(elsewhere)["sweep_source"], "checkpoint")
    expect("empty.sweep_source", extract_exllama(_fake_root({}))["sweep_source"], "unknown")
    expect("empty.promote_budget", extract_exllama(_fake_root({}))["promote_budget"], None)

    # 7. This fork's real tree.  Every value whose regex could miss is asserted twice: the value, and
    # the line recorded for it.  A regex that fails closed to the number the tree already carries would
    # pass the value check on its own while pointing at a line that never states it (mutation m1 in the
    # task report reads 128 off the tier-stats knob this way).
    if (repo / CPU_MLP).is_file():
        real = extract_exllama(repo)
        expect("real_fork.promote_budget", real["promote_budget"], 6)
        expect("real_fork.promote_ratio", real["promote_ratio"], 1.5)
        expect("real_fork.promote_decay", real["promote_decay"], 0.98)
        expect("real_fork.spares", real["spares"], 4)
        expect("real_fork.sweep_interval", real["sweep_interval"], 128)
        expect("real_fork.sweep_max", real["sweep_max"], 64)
        expect("real_fork.sweep_source", real["sweep_source"], "checkpoint")
        expect("real_fork.stream_min_rows", real["stream_min_rows"], 32)
        expect("real_fork.cache_mode", real["cache_mode"], "2,2")
        expect("real_fork.cache_size", real["cache_size"], 131072)
        # The regime's split, not the review plan's: the plan says 381, state.json says 376 and its
        # own history records the 400 -> 376 re-founding.  512 - split also moves with it (136, not
        # 131), so a report must derive that number from this one rather than copy the plan.
        expect("real_fork.split", real["split"], 376)
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

    # 8a. The matrix's own schema, independent of either tree: unique ids, the verdict vocabulary, a
    # justification wherever one is owed, and `measured` a run id or the literal "unmeasured".
    ids = [row["id"] for row in MATRIX]
    expect("matrix.ids_unique", len(set(ids)), len(ids))
    expect("matrix.verdicts", [r["id"] for r in MATRIX if r["verdict"] not in VERDICTS], [])
    expect("matrix.justified",
           [r["id"] for r in MATRIX
            if r["verdict"] != "faithful" and not str(r.get("justification") or "").strip()], [])
    expect("matrix.measured",
           [r["id"] for r in MATRIX
            if r["measured"] != "unmeasured" and not RUN_ID.match(str(r["measured"]))], [])
    expect("matrix.no_faithful_without_values",
           [r["id"] for r in MATRIX
            if r["verdict"] == "faithful" and not (r.get("strata") and r.get("fork"))], [])

    # 8b. The matrix against the real trees: every row passes, and each way of being wrong fails and
    # names itself.  Run only when both trees are here; a fixture-only box still ran 8a.
    if (root / GENERATE).is_file() and (repo / CPU_MLP).is_file():
        lines = check(extract_strata(root), extract_exllama(repo))
        expect("matrix.rows", len(lines), len(MATRIX))
        expect("matrix.all_pass", [line for line in lines if line.startswith("FAIL")], [])

        def first_fail(rows: list[dict]) -> str:
            found = [line for line in check(extract_strata(root), extract_exllama(repo), matrix=rows)
                     if line.startswith("FAIL")]
            return found[0] if found else ""

        # The plan's first test, and the two mutations around it: a wrong value on a key the row does
        # not carry (so the checker must compare what a row records rather than what it expects), a
        # wrong value on one it does, and a `derived` number the row's own arithmetic contradicts.
        expect("matrix.wrong_strata_value",
               "adapt_swaps" in first_fail(
                   [dict(MATRIX[0], strata=dict(MATRIX[0]["strata"], adapt_swaps=64))]), True)
        churn = next(r for r in MATRIX if r["id"] == "M2-churn")
        expect("matrix.wrong_fork_value",
               "sweep_source" in first_fail(
                   [dict(churn, fork=dict(churn["fork"], sweep_source="ram-copy"))]), True)
        expect("matrix.wrong_derived",
               "churn_ratio" in first_fail(
                   [dict(churn, derived=dict(churn["derived"], churn_ratio=10))]), True)
        # The review's citation discipline, as a failure: an unowned justification and an
        # approximate `measured` are both defects, and both must be caught rather than printed.
        expect("matrix.missing_justification",
               "justification" in first_fail([dict(churn, justification="  ")]), True)
        expect("matrix.approximate_measured",
               "measured" in first_fail([dict(churn, measured="about 0.2 per step")]), True)
        expect("matrix.invented_verdict",
               "verdict" in first_fail([dict(churn, verdict="mostly")]), True)
    else:
        print(f"skip matrix-vs-tree: need {GENERATE} under {root} and {CPU_MLP} under {repo}")

    for line in failures:
        print(line)
    print(f"{total - len(failures)}/{total} checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true",
                        help="run this module's own assertions and exit non-zero on a failure")
    parser.add_argument("--check", action="store_true",
                        help="compare the matrix's recorded values and arithmetic with both trees")
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
    if args.check:
        root = args.root or STRATA
        if not (root / GENERATE).is_file():
            print(f"no Strata tree at {root}: pass --root or set STRATA_ROOT", file=sys.stderr)
            return 2
        if not (args.repo_root / CPU_MLP).is_file():
            print(f"no fork tree at {args.repo_root}: pass --repo-root", file=sys.stderr)
            return 2
        lines = check(extract_strata(root), extract_exllama(args.repo_root))
        for line in lines:
            print(line)
        # Rows, not lines: one row can disagree on several constants and still be one failed row.
        rows: dict[str, bool] = {}
        for line in lines:
            rid = line.split(":", 1)[0].split(" ", 1)[-1]
            rows[rid] = rows.get(rid, True) and not line.startswith("FAIL")
        passed = sum(1 for ok in rows.values() if ok)
        print(f"{passed}/{len(rows)} rows PASS")
        return 0 if passed == len(rows) else 1
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
