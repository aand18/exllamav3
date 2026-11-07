"""
EXL3_ROUND_PROF (util/round_prof.py) is a default-off instrument, and its three ways of going wrong are
each cheap to pin: a hot path that still costs something with the knob off, an event pool rebuilt per
bracket (that one mistake costs more than the thing being measured), and a CSV whose rows do not line up
with the brackets that produced them.

The flag is read once, at import, so every test reloads the module against its own environment; the
fixture puts the environment and the module back afterwards.
"""

import ast
import importlib
import os
from pathlib import Path

import pytest
import torch

import exllamav3.util.round_prof as round_prof

_KNOBS = (
    "EXL3_ROUND_PROF",
    "EXL3_ROUND_PROF_SAMPLE",
    "EXL3_ROUND_PROF_WARMUP",
    "EXL3_ROUND_PROF_EVERY",
    "EXL3_ROUND_PROF_CSV",
)

REPO_DIR = Path(__file__).resolve().parents[1]

_NO_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA events")


@pytest.fixture(autouse = True)
def knobs():
    saved = {k: os.environ.get(k) for k in _KNOBS}
    try:
        yield
    finally:
        for k in _KNOBS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(round_prof)


def _load(**env):
    """Import the module against exactly this knob set (it reads the environment once, at import)"""
    for k in _KNOBS:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = str(v)
    return importlib.reload(round_prof)


@pytest.mark.nogpu
def test_disabled_is_a_no_op():
    rp = _load()
    assert rp.PROF is False

    rp.round_begin()
    rp.phase_begin(0, "mixer")
    rp.phase_end(0, "mixer")
    rp.phase_begin(0, "moe")
    rp.phase_end(0, "moe")
    rp.round_end()
    rp.round_end()

    assert rp._pool == {}, "an event pair was allocated with the knob off"
    assert rp._null == [] and rp._rows == [] and rp._walls == [] and rp._floors == []
    assert rp._sampled == 0


@pytest.mark.nogpu
@pytest.mark.parametrize("rel, calls", [
    ("exllamav3/modules/transformer.py", ("phase_begin", "phase_end")),
    ("exllamav3/generator/generator.py", ("round_begin", "round_end")),
])
def test_call_sites_are_one_guarded_statement(rel, calls):
    """Read once at import, so the off path is `if _RPROF:` and nothing else at each site"""
    tree = ast.parse((REPO_DIR / rel).read_text(encoding = "utf-8"))

    flags = [
        n for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_RPROF" for t in n.targets)
    ]
    assert len(flags) == 1, f"{rel}: no single module-level _RPROF"
    v = flags[0].value
    assert isinstance(v, ast.Attribute) and v.attr == "PROF"                       \
       and isinstance(v.value, ast.Name) and v.value.id == "round_prof",           \
        f"{rel}: _RPROF is not captured from round_prof.PROF"

    guarded = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.If) or not isinstance(n.test, ast.Name) or n.test.id != "_RPROF":
            continue
        assert len(n.body) == 1, f"{rel}: the guarded body is not a single statement"
        stmt = n.body[0]
        assert isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call), f"{rel}: not a call"
        fn = stmt.value.func
        assert isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) \
           and fn.value.id == "round_prof", f"{rel}: guards something other than round_prof"
        guarded[fn.attr] = guarded.get(fn.attr, 0) + 1

    for c in calls:
        assert guarded.get(c, 0), f"{rel}: no `if _RPROF:` guard around round_prof.{c}"


@_NO_GPU
def test_event_pool_is_allocated_once():
    rp = _load(EXL3_ROUND_PROF = 1, EXL3_ROUND_PROF_SAMPLE = 1, EXL3_ROUND_PROF_WARMUP = 0,
               EXL3_ROUND_PROF_EVERY = 1000000)

    def one_round():
        rp.round_begin()
        rp.phase_begin(3, "mixer")
        rp.phase_end(3, "mixer")
        rp.phase_begin(3, "moe")
        rp.phase_end(3, "moe")
        torch.cuda.synchronize()            # the round ends in a synchronise; the brackets are readable
        rp.round_end()

    one_round()
    assert set(rp._pool) == {(3, "mixer"), (3, "moe")}
    first = {k: (id(v[0]), id(v[1])) for k, v in rp._pool.items()}
    first_null = (id(rp._null[0]), id(rp._null[1]))

    one_round()
    second = {k: (id(v[0]), id(v[1])) for k, v in rp._pool.items()}
    assert second == first, "the pool was rebuilt: one torch.cuda.Event per bracket"
    assert (id(rp._null[0]), id(rp._null[1])) == first_null

    assert rp._sampled == 2
    assert sorted((r, phase) for r, layer, phase, ms in rp._rows) == \
           [(1, "mixer"), (1, "moe"), (2, "mixer"), (2, "moe")]
    assert len(rp._floors) == 2


@_NO_GPU
def test_sampling_is_one_round_in_n():
    rp = _load(EXL3_ROUND_PROF = 1, EXL3_ROUND_PROF_SAMPLE = 4, EXL3_ROUND_PROF_WARMUP = 2,
               EXL3_ROUND_PROF_EVERY = 1000000)
    armed = []
    for _ in range(10):
        rp.round_begin()
        armed.append(rp._on)
        rp.round_end()
    # two warmup rounds, then every 4th round, never every round
    assert armed == [False, False, False, True, False, False, False, True, False, False]


@pytest.mark.nogpu
def test_report_line_format(capsys):
    rp = _load()
    rp._acc["mixer"] = {i: [0.5 * (i + 1), 1] for i in range(4)}     # per-layer means 0.5 .. 2.0
    rp._acc["moe"] = {i: [1.0, 1] for i in range(4)}
    rp._walls = [50.0, 52.0, 54.0]
    rp._floors = [0.004, 0.006]
    rp._sampled = 3
    rp._report()
    out = capsys.readouterr().out
    assert out.startswith(" -- round prof: mixer med/p90 1.500/2.000 ms")
    assert "moe med/p90 1.000/1.000 ms" in out
    assert "round wall med/p90 52.000/54.000 ms" in out
    assert "floor 0.006 ms" in out
    assert out.rstrip().endswith("n=3")


@pytest.mark.nogpu
def test_csv_has_one_row_per_layer_and_phase(tmp_path):
    rp = _load()
    path = tmp_path / "round_prof.csv"
    rows = [(7, 0, "mixer", 0.25), (7, 0, "moe", 1.0), (7, 1, "mixer", 0.5), (7, 1, "moe", 2.25)]
    assert rp.write_csv(path, rows = rows) == len(rows)
    lines = path.read_text(encoding = "utf-8").splitlines()
    assert lines[0] == "round,layer,phase,ms"
    assert len(lines) == 1 + len(rows)
    assert lines[1] == "7,0,mixer,0.2500"
    assert lines[4] == "7,1,moe,2.2500"

    # no path or nothing recorded: no file, and a path that was never opened is not an error
    assert rp.write_csv(path, rows = []) == 0
    assert rp.write_csv(None, rows = rows) == 0
