"""The extractors' contract: the placement defaults each engine's source text carries.

Two sides, one module: `extract_strata` reads the sibling engine's C++, `extract_exllama` reads this
fork's Python plus the regime file. The rest of the assertions (the real trees, the divergence
branches, the source_lines bounds) live in `strata_transfer_check._selftest`, because the checker has
to run on a box with no pytest: `python3 bench/strata_transfer_check.py --selftest`.

The five matrix-contract tests read the real sibling tree, so they skip - naming the missing file and
`STRATA_ROOT` - on a box that has only this fork; `--selftest` still exits 2 there unless
`--allow-missing-tree` says the fixtures-only run is the intended one.
"""

import pathlib
import sys

import pytest

# `pytest bench/...` from the repo root puts bench/ on sys.path only in the default import mode;
# this keeps a direct `python3 bench/test_strata_transfer_check.py` working too.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from strata_transfer_check import (  # noqa: E402
    GENERATE, MATRIX, REPO, STRATA, _fake_root, check, extract_exllama, extract_strata,
)


def test_extract_strata_reads_the_defaults_from_source_text():
    tree = {  # the shape of the real file, not the file
        "src/program/generate.cpp": (
            "    int adapt_every = 4;\n"
            "    float adapt_decay = 0.7f;\n"
            "    int adapt_swaps = 96;\n"
            '    "exchange them with the GPU cache (no file reads)"\n'),
    }
    got = extract_strata(_fake_root(tree))
    assert got["adapt_every"] == 4 and got["adapt_swaps"] == 96
    assert got["adapt_decay"] == 0.7
    assert got["swap_source"] == "ram-copy"
    assert got["slot_pool"] == "global"


def test_extract_exllama_matches_the_branch():
    got = extract_exllama(REPO)
    assert (got["promote_budget"], got["promote_ratio"]) == (6, 1.5)
    assert got["sweep_interval"] == 128 and got["sweep_max"] == 64
    assert got["sweep_source"] == "checkpoint"      # _split_swap_experts re-reads via stc.get_tensor
    assert got["stream_min_rows"] == 32


def test_extract_exllama_reads_the_fork_defaults_from_source_text():
    tree = {  # the shape of the real files, not the files
        "exllamav3/modules/block_sparse_mlp_cpu.py": (
            '_PROMOTE_BUDGET = max(0, int(os.environ.get("EXL3_MOE_CPU_PROMOTE_BUDGET", 6)))\n'
            '    budget = int(os.environ.get("EXL3_MOE_CPU_SWAP_MAX", 64))\n'
            '        # every EXL3_MOE_CPU_SWAP_INTERVAL steps, quiesce\n'
            '        interval = int(os.environ.get("EXL3_MOE_CPU_SWAP_INTERVAL", 128))\n'
            '    def _split_swap_experts(self, r_cold, r_hot, mp):\n'
            '        dst.trellis.copy_(stc.get_tensor(src_key + ".trellis", dst.device))\n'),
        "exllamav3/model/moe_cpu_host.py":
            '        self.stream_min_rows = int(os.environ.get("EXL3_MOE_STREAM_MIN_ROWS", 32))\n',
        "bench/windows/state.json":
            '{\n  "active": {\n    "cache_quant": "2,2",\n    "split": 376\n  }\n}\n',
    }
    got = extract_exllama(_fake_root(tree))
    assert got["promote_budget"] == 6 and got["sweep_max"] == 64
    # The comment above the interval line names the knob and carries no number: the value has to
    # come from the env read on line 4, not from the line that merely mentions it.
    assert got["sweep_interval"] == 128
    assert got["source_lines"]["sweep_interval"].endswith(":4")
    assert got["sweep_source"] == "checkpoint"
    assert got["stream_min_rows"] == 32 and got["cache_mode"] == "2,2" and got["split"] == 376
    # A knob this fixture does not carry is absent, and gets no line: None, not a guessed default.
    assert got["promote_decay"] is None and "promote_decay" not in got["source_lines"]


# --- the matrix's contract: a wrong recorded constant fails, the real tree passes ---------------

# The five tests below read the REAL sibling tree, not a fixture: they call `extract_strata(STRATA)`,
# so the file the extractor opens has to be on this box. Where it is absent - the Windows copy of this
# fork, which carries no Strata checkout - they skip and name what is missing, rather than failing on a
# fixture no box can supply. The file is the same one the extractor and `_selftest` key on, so the guard
# cannot drift from the thing it guards.
STRATA_SOURCE = STRATA / GENERATE
needs_strata_tree = pytest.mark.skipif(
    not STRATA_SOURCE.is_file(),
    reason = (
        f"no Strata tree: {STRATA_SOURCE} is absent, and these read the real source rather than a "
        f"fixture. Point the checker at another checkout with STRATA_ROOT=/path/to/UnbiasedStrata; "
        f"for the fixture-only arms there is "
        f"`python3 bench/strata_transfer_check.py --selftest --allow-missing-tree`."
    ),
)


@needs_strata_tree
def test_a_wrong_recorded_constant_fails():
    """The plan's first test: one wrong recorded constant must FAIL and name itself."""
    bad = [dict(MATRIX[0], strata=dict(MATRIX[0]["strata"], adapt_swaps=64))]
    fails = [l for l in check(extract_strata(STRATA), extract_exllama(REPO), matrix=bad)
             if l.startswith("FAIL")]
    assert fails and "adapt_swaps" in fails[0]


@needs_strata_tree
def test_the_real_tree_passes_every_row():
    """The plan's second test: the tree as it stands yields one line per row, none of them FAIL."""
    lines = check(extract_strata(STRATA), extract_exllama(REPO))
    assert [l for l in lines if l.startswith("FAIL")] == []
    assert len(lines) == len(MATRIX) == 6


@needs_strata_tree
def test_a_wrong_fork_constant_fails_and_names_it():
    row = next(r for r in MATRIX if r["id"] == "M2-churn")
    bad = [dict(row, fork=dict(row["fork"], sweep_source="ram-copy"))]
    fails = [l for l in check(extract_strata(STRATA), extract_exllama(REPO), matrix=bad)
             if l.startswith("FAIL")]
    assert fails and "sweep_source" in fails[0]


@needs_strata_tree
def test_a_divergence_without_a_justification_fails():
    row = next(r for r in MATRIX if r["verdict"] != "faithful")
    bad = [dict(row, justification="   ")]
    fails = [l for l in check(extract_strata(STRATA), extract_exllama(REPO), matrix=bad)
             if l.startswith("FAIL")]
    assert fails and "justification" in fails[0]


@needs_strata_tree
def test_an_approximate_measured_fails():
    """`measured` is a run id or the literal `unmeasured`; "about 0.2 per step" is neither."""
    bad = [dict(MATRIX[0], measured="about 0.2 per step")]
    fails = [l for l in check(extract_strata(STRATA), extract_exllama(REPO), matrix=bad)
             if l.startswith("FAIL")]
    assert fails and "measured" in fails[0]
