"""The Strata-side extractor's contract: the placement defaults it must read out of source text.

The rest of the module's own assertions (the real tree, the divergence branches, the source_lines
bounds) live in `strata_transfer_check._selftest`, because the checker has to run on a box with no
pytest: `python3 bench/strata_transfer_check.py --selftest`.
"""

import pathlib
import sys

# `pytest bench/...` from the repo root puts bench/ on sys.path only in the default import mode;
# this keeps a direct `python3 bench/test_strata_transfer_check.py` working too.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from strata_transfer_check import _fake_root, extract_strata  # noqa: E402


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
