"""The extractors' contract: the placement defaults each engine's source text carries.

Two sides, one module: `extract_strata` reads the sibling engine's C++, `extract_exllama` reads this
fork's Python plus the regime file. The rest of the assertions (the real trees, the divergence
branches, the source_lines bounds) live in `strata_transfer_check._selftest`, because the checker has
to run on a box with no pytest: `python3 bench/strata_transfer_check.py --selftest`.
"""

import pathlib
import sys

# `pytest bench/...` from the repo root puts bench/ on sys.path only in the default import mode;
# this keeps a direct `python3 bench/test_strata_transfer_check.py` working too.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from strata_transfer_check import REPO, _fake_root, extract_exllama, extract_strata  # noqa: E402


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
