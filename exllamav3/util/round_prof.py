"""
Round-phase instrument (EXL3_ROUND_PROF): how much of a decode round is the mixer (attention / GDN)
and how much is the MoE block. Two event brackets per layer plus one wall bracket per round, chained
around the Python-visible calls in TransformerBlock.forward (modules/transformer.py) and
Generator.iterate (generator/generator.py), which is also where the capture boundaries are: every
CUDA graph in this tree begins and ends inside a single C++ call, so a Python boundary sits outside
it by construction.

    EXL3_ROUND_PROF=1              enable (unset or 0: every caller's `if _RPROF` is False)
    EXL3_ROUND_PROF_SAMPLE=8       profile one round in N
    EXL3_ROUND_PROF_WARMUP=64      skip the first rounds: graph capture and its eager precursor
    EXL3_ROUND_PROF_EVERY=64       one report line per this many sampled rounds
    EXL3_ROUND_PROF_CSV=path       write round,layer,phase,ms at exit, one file per pass
    EXL3_DISTINCT_CSV=path         write round,layer,distinct,rows,topk at exit (R1: distinct CPU
                                   experts per layer per sampled round; see enqueue_distinct).
                                   Defaults to "<EXL3_ROUND_PROF_CSV>.distinct" when that is set.
                                   Each sampled round's rows==5 layer-round is also scored
                                   against the same layer's previous sampled round (T60
                                   autocorr cover) in the periodic report line.

The flag is read once, at import. A caller imports PROF into its own module-level `_RPROF`, so with
the knob off the branch is a constant lookup and nothing else: no event object is built, and no
attribute is added to a module or a params dict.

Events are preallocated per (layer, phase) and reused, because constructing torch.cuda.Event is the
expensive part of a bracket (cf. _sprof_wrap in modules/block_sparse_mlp_cpu.py). Times are stream
time, except the round wall, which is perf_counter. The two brackets of a layer deliberately do not
share an event: the residual / pre-MoE norm between them belongs to neither phase, and folding it
into the MoE would answer the question with the wrong number.
"""

import atexit
import os
import time

import torch


def _env_int(name, default):
    # A knob that fails to parse falls back to the default rather than aborting a live pass
    try:
        v = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return v if v >= 0 else default


PROF = os.environ.get("EXL3_ROUND_PROF", "0").strip().lower() not in ("", "0", "off", "false")
SAMPLE = max(1, _env_int("EXL3_ROUND_PROF_SAMPLE", 8))
WARMUP = _env_int("EXL3_ROUND_PROF_WARMUP", 64)
EVERY = max(1, _env_int("EXL3_ROUND_PROF_EVERY", 64))
# Accept the spelling in the design note as well: a silently ignored path would cost a whole pass
CSV_PATH = os.environ.get("EXL3_ROUND_PROF_CSV") or os.environ.get("EXLL3_ROUND_PROF_CSV") or None
# R1 distinct-expert counts share the phase CSV's path by default (suffix .distinct), so one env
# var enables both files in a measurement arm.
DISTINCT_CSV_PATH = os.environ.get("EXL3_DISTINCT_CSV") or \
    (CSV_PATH + ".distinct" if CSV_PATH else None)

_pool = {}                          # (layer, phase) -> [enter event, exit event], built once per key
_null = []                          # two back-to-back events: the floor a bracket costs
_on = False                         # True while a sampled round is open
_t0 = 0.0                           # perf_counter at the top of the sampled round
_rounds = 0                         # rounds seen since import
_sampled = 0                        # sampled rounds read back so far
_brackets = []                      # this round's (round, layer, phase, ev0, ev1)
_acc = {"mixer": {}, "moe": {}}     # phase -> layer -> [sum ms, count]
_walls = []                         # round wall ms, one per sampled round
_floors = []                        # null-pair ms, one per sampled round
_rows = []                          # CSV rows: (round, layer, phase, ms)
_dropped = 0                         # brackets whose exit event had not completed at read time
_distinct = []                       # rows: (round, layer, distinct cpu experts, rows, topk)
_pending = []                        # this sampled round's async sel copies, counted once at round end
# T65 1b: raw selections, so a pool policy can be simulated against real routing instead of the
# stationary profile. EXL3_ROUTING_DUMP=<path.npz>; capped by EXL3_ROUTING_DUMP_ROUNDS (default 400)
_DUMP_PATH = os.environ.get("EXL3_ROUTING_DUMP") or None
_DUMP_ROUNDS = _env_int("EXL3_ROUTING_DUMP_ROUNDS", 400)
_dump = []                           # (round, layer, rows, topk, list-of-picks)
# T60 routing autocorr: each layer's distinct CPU expert set at the previous finalized sampled
# round, and the intersection covers of rows==5 layer-rounds against it (one entry per
# current-rows==5 layer-round with a previous set and a non-empty current set).
_cpu_sets = {}                       # layer key -> set of cpu experts at the previous sampled round
_covers = []                         # covers for rows==5 layer-rounds: |cur & prev| / |cur|


def _new_pair():
    return [torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)]


def _record(ev):
    # An event recorded inside a capture invalidates it and surfaces as a wrong forward pass, not as
    # an error, so the loud version is here at every record. The C++ captures (attention.cpp:885,
    # gated_delta_net.cpp:355, mlp.cpp:87) run on a private capture stream, which
    # is_current_stream_capturing() cannot see; this catches a boundary that a future Python-side
    # capture would put inside one.
    assert not torch.cuda.is_current_stream_capturing(), \
        "EXL3_ROUND_PROF: event recorded inside a CUDA graph capture"
    ev.record()


def round_begin():
    global _on, _rounds, _t0, _brackets
    _rounds += 1
    _on = False                     # a round that raised leaves nothing armed for the next one
    _brackets = []
    if _rounds <= WARMUP or (_rounds % SAMPLE) != 0:
        return
    if not _null:
        _null.extend(_new_pair())
    _record(_null[0])
    _record(_null[1])               # nothing between: one bracket, both events, no work
    _t0 = time.perf_counter()
    _on = True


def round_end():
    global _on, _sampled
    if not _on:
        return
    _on = False
    _walls.append((time.perf_counter() - _t0) * 1000.0)
    _read()
    _finalize_distinct()
    _sampled += 1
    if (_sampled % EVERY) == 0:
        _report()


def phase_begin(layer, phase):
    if not _on:
        return
    pair = _pool.get((layer, phase))
    if pair is None:
        pair = _pool[(layer, phase)] = _new_pair()
    _record(pair[0])


def phase_end(layer, phase):
    if not _on:
        return
    pair = _pool[(layer, phase)]    # phase_begin built it inside this same bracket
    _record(pair[1])
    _brackets.append((_rounds, layer, phase, pair[0], pair[1]))


def _read():
    global _dropped
    # .query() before .elapsed_time(): an unfinished bracket has no elapsed time to report, and the
    # round ends in a synchronise anyway, so a miss here is a bracket that tiled a stream the round
    # does not wait on. Drop it rather than introduce a wait of our own.
    seen = {}
    for _, layer, phase, _, _ in _brackets:
        seen[(layer, phase)] = seen.get((layer, phase), 0) + 1
    for rnd, layer, phase, ev0, ev1 in _brackets:
        if seen[(layer, phase)] > 1:
            # two blocks under one layer index share a pooled pair, so both timings are the later
            # block's: a dropped row is honest, a doubled one is a wrong number
            continue
        if not ev1.query():
            _dropped += 1
            continue
        ms = ev0.elapsed_time(ev1)
        s = _acc.setdefault(phase, {}).setdefault(layer, [0.0, 0])
        s[0] += ms
        s[1] += 1
        _rows.append((rnd, layer, phase, ms))
    if _null and _null[1].query():
        _floors.append(_null[0].elapsed_time(_null[1]))


def _quantiles(vals):
    if not vals:
        return 0.0, 0.0
    vals = sorted(vals)
    return vals[len(vals) // 2], vals[int(len(vals) * 0.9)]


def _report():
    # One mean per layer index, then med/p90 across the layer stack: the load-bearing shape is a
    # rise of a phase with layer index, and a round-level percentile would hide it
    def layer_means(phase):
        return [s / n for s, n in _acc[phase].values() if n]

    parts = []
    for phase in sorted(_acc):
        med, p90 = _quantiles(layer_means(phase))
        parts.append(f"{phase} {med:.3f}/{p90:.3f}")
    w_med, w_p90 = _quantiles(_walls)
    f_med, _ = _quantiles(_floors)
    print(" -- round prof: " + " | ".join(parts) +
          f" ms | round wall {w_med:.3f}/{w_p90:.3f} ms, floor {f_med:.3f} ms, "
          f"dropped {_dropped}, n={_sampled}", flush = True)
    # T60 routing autocorr over the rows==5 layer-rounds so far: the share of this round's
    # distinct CPU experts that the previous sampled round's same-layer set already held.
    if _covers:
        med, p90 = _quantiles(_covers)
        frac = sum(1 for c in _covers if c >= 0.8) / len(_covers)
        print(f" -- round prof autocorr cover: median {med:.3f} p90 {p90:.3f} "
              f"frac>=0.8 {frac:.3f} (n={len(_covers)} rows=5 layer-rounds)",
              flush = True)


def write_csv(path = None, rows = None):
    """Emit the raw brackets so a table is generated from them rather than typed from a log line"""
    path = path or CSV_PATH
    rows = _rows if rows is None else rows
    if not path or not rows:
        return 0
    with open(path, "w", encoding = "utf-8", newline = "") as f:
        f.write("round,layer,phase,ms\n")
        for rnd, layer, phase, ms in rows:
            f.write(f"{rnd},{layer},{phase},{ms:.4f}\n")
    print(f" -- round prof: {len(rows)} rows -> {path}", flush = True)
    return len(rows)


# ---------------------------------------------------------------------------
# R1: distinct CPU experts per layer per sampled round (no kernel change).
#
# block_sparse_mlp_cpu.cpu_split_submit enqueues an async D2H copy of the raw router
# selections into a module-owned pinned buffer while a sampled round is open, and hands
# round_prof the buffer + this layer's cpu_split_first. The counts are taken at round end
# with one host sync total -- outside every phase bracket, so sub/ker/cmb stay clean.
# The copy is enqueued before submit_issue_fused runs, hence on raw router ids;
# the threshold rule is exactly _split_translate's: sel >= first_cpu is CPU-resident.
# T60: each finalized layer-round's CPU set is also scored against the same layer's previous
# sampled-round set (rows==5 intersection cover, see _autocorr_cover), still under the same
# single round-end sync.
# ---------------------------------------------------------------------------
def is_sampling():
    """True while a sampled round is open (matches the event brackets' gate)"""
    return _on


def current_round():
    """The round index the sampled-round machinery is counting"""
    return _rounds


def enqueue_distinct(layer, buf_cpu, first_cpu, rows, topk, rnd):
    """Queue one layer's async selection copy for counting at this round's end"""
    _pending.append((rnd, layer, buf_cpu, first_cpu, rows, topk))


def count_distinct_cpu(buf_cpu, first_cpu, rows, topk):
    """Distinct sel >= first_cpu among the first `rows` rows of a CPU copy (pure, nogpu-testable)"""
    if first_cpu is None:
        return 0
    vals = buf_cpu[:rows, :topk].reshape(-1)
    return len({int(v) for v in vals.tolist() if v >= first_cpu})


def distinct_cpu_set(buf_cpu, first_cpu, rows, topk):
    """The set backing count_distinct_cpu (pure, nogpu-testable): the distinct sel >= first_cpu
    among the first `rows` rows of a CPU copy"""
    if first_cpu is None:
        return set()
    vals = buf_cpu[:rows, :topk].reshape(-1)
    return {int(v) for v in vals.tolist() if v >= first_cpu}


def _autocorr_cover(layer, cur, rows):
    """Score one finalized layer-round against the same layer's previous sampled round and
    remember this round's set for the next: T60's intersection cover, recorded only when the
    current round is rows==5 (the drafted window) with a previous set and a non-empty current
    set. The set is updated every finalized round, so a cover always compares two consecutive
    sampled rounds for the layer."""
    prev = _cpu_sets.get(layer)
    if prev is not None and rows == 5 and cur:
        _covers.append(len(cur & prev) / len(cur))
    _cpu_sets[layer] = cur


def _finalize_distinct():
    global _pending, _distinct
    if not _pending:
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    for rnd, layer, buf_cpu, first_cpu, rows, topk in _pending:
        _distinct.append((rnd, layer, count_distinct_cpu(buf_cpu, first_cpu, rows, topk),
                          rows, topk))
        _autocorr_cover(layer, distinct_cpu_set(buf_cpu, first_cpu, rows, topk), rows)
        if _DUMP_PATH and len(_dump) < _DUMP_ROUNDS * 64:
            _dump.append((rnd, layer, rows, topk,
                          buf_cpu[:rows, :topk].reshape(-1).to(torch.int16).tolist()))
    _pending = []


def write_dump(path = None, rows = None):
    """T65 1b: the raw per-round selections as an npz, so a pool policy can be simulated on them.

    Rows differ per layer-round (rows=1..5 -> 10..50 picks), so the array is padded to a fixed
    width with -1 and `meta` carries each record's real length.
    """
    path = path or _DUMP_PATH
    rows = _dump if rows is None else rows
    if not path or not rows:
        return 0
    import numpy as np
    width = max(len(r[4]) for r in rows)
    picks = np.full((len(rows), width), -1, dtype = np.int16)
    for i, r in enumerate(rows):
        picks[i, :len(r[4])] = r[4]
    meta = np.array([[r[0],
                      int(r[1].split(".")[-2]) if ".layers." in r[1] else -1,
                      r[2], r[3], len(r[4])] for r in rows], dtype = np.int32)
    with open(path, "wb") as f:
        np.savez_compressed(f, picks = picks, meta = meta)
    print(f" -- routing dump: {len(rows)} layer-rounds, width {width} -> {path}", flush = True)
    return len(rows)


def write_distinct(path = None, rows = None):
    """Emit the R1 counts: round,layer,distinct,rows,topk"""
    path = path or DISTINCT_CSV_PATH
    rows = _distinct if rows is None else rows
    if not path or not rows:
        return 0
    with open(path, "w", encoding = "utf-8", newline = "") as f:
        f.write("round,layer,distinct,rows,topk\n")
        for rnd, layer, n, rows_, topk in rows:
            f.write(f"{rnd},{layer},{n},{rows_},{topk}\n")
    print(f" -- round prof distinct: {len(rows)} rows -> {path}", flush = True)
    return len(rows)


if PROF and CSV_PATH:
    atexit.register(write_csv)
if PROF and DISTINCT_CSV_PATH:
    atexit.register(write_distinct)
if PROF and _DUMP_PATH:
    atexit.register(write_dump)
