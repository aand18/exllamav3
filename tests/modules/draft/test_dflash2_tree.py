"""
The drafter's runner-up: `extract_candidates` in the DFlash2 draft module.

DFlash2 is non-causal, so one forward pass carries a distribution for every draft position and a
verification tree needs no second pass - only the candidates the chain path throws away. The extractor
is that read: per position, the k most likely `(token id, probability)` pairs in the drafter's own token
space, prepared exactly as `DFlash2Selector.walk_block` prepares its top-k (vocabulary slice, output
multiplier, softcap) so the extractor and the selector cannot drift apart on the same logits.

The function is pure - no model, no tensor required, no RNG, no globals. This file keeps it that way:
torch is absent on the Linux box this file is developed on and `exllamav3` will not import without it,
so the fallback loader pulls the function out of the source by AST and gives it `math` and nothing else.
A reference to torch, to the surrounding module or to any other global fails loudly in every test here.
Where torch is present the same tests run against the imported function, plus a chain-identity test that
drives the real selector walk.

pytest is absent on that Linux box, so the file also runs directly:
`python3 tests/modules/draft/test_dflash2_tree.py`.
"""

import ast
import math
import os
import sys
import traceback

try:
    import torch
except ImportError:
    # Pure extractor: the tests below need no tensor, only the source-level contract
    torch = None

try:
    import pytest
    pytestmark = pytest.mark.nogpu
except ImportError:
    # The direct runner at the bottom of the file is the gate when pytest is not installed
    pytest = None


DRAFT_MODULE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "exllamav3", "modules", "arch_specific", "dflash2.py",
)


def _load_extract_candidates():
    """
    The extractor itself, from the package when it imports and from the source otherwise.

    `exllamav3/modules/arch_specific/dflash2.py` imports torch, the compiled extension and the model
    package at module scope, all of which are absent where these tests are developed. The extractor is
    pure, so the fallback compiles just that one function out of the file - with the module's
    `from __future__ import annotations` ahead of it, so an annotation naming torch stays unevaluated -
    and runs it with `math` as its only global.
    """
    try:
        from exllamav3.modules.arch_specific.dflash2 import extract_candidates
        return extract_candidates
    except (ImportError, RuntimeError):
        pass
    with open(DRAFT_MODULE, encoding = "utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    node = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == "extract_candidates"),
        None,
    )
    if node is None:
        raise ImportError(f"no extract_candidates in {DRAFT_MODULE}")
    body = ast.parse("from __future__ import annotations").body + [node]
    module = ast.fix_missing_locations(ast.Module(body = body, type_ignores = []))
    namespace = {"math": math}
    exec(compile(module, DRAFT_MODULE, "exec"), namespace)
    return namespace["extract_candidates"]


extract_candidates = _load_extract_candidates()


def _walk_block_preparation(row, vocab_size, scale = 1.0, softcap = 0.0):
    """
    The three steps `DFlash2Selector.walk_block` applies before it takes its top-k, transcribed: the
    vocabulary slice (a padded head is wider than the vocabulary), the output multiplier, then the
    softcap. The selector's own torch fallback does this in one expression; the signature test at the
    bottom drives the real selector where torch exists, so this transcription is checked against it.
    """
    scores = [float(value) * scale for value in row[:vocab_size]]
    if softcap > 0.0:
        scores = [math.tanh(value / softcap) * softcap for value in scores]
    return scores


def test_a_known_argmax_and_runner_up_come_back_ordered_with_softmax_probabilities():
    """
    The first job: a row with a known argmax and runner-up returns both, in that order, with the
    probabilities a softmax over that row gives - not the logits, and not a share of the returned pair.
    """
    row = [0.0, 2.0, 1.0, -1.0]
    weights = [math.exp(value) for value in row]
    total = sum(weights)

    candidates = extract_candidates([row], k = 2)

    assert [token for token, _ in candidates[0]] == [1, 2]
    assert abs(candidates[0][0][1] - weights[1] / total) < 1e-12
    assert abs(candidates[0][1][1] - weights[2] / total) < 1e-12
    # Normalised over the position's vocabulary, so the pair carries the drafter's own probability
    # and a tree can compare positions
    assert sum(probability for _, probability in candidates[0]) < 1.0
    assert candidates[0][0][1] > candidates[0][1][1]


def test_k_1_is_the_top_1_the_chain_path_takes():
    """
    Rank 1 is what the chain verifies today, so at k = 1 the extractor must return the id the existing
    selection returns and nothing else. The row is a padded head's slice prepared the way `walk_block`
    prepares it: the vocabulary is 5 of 8 columns and the tallest values sit in the padding, which a
    slice taken after the argmax would pick instead.
    """
    row = [0.5, -1.0, 2.25, 0.0, 1.75, 9.0, 8.5, 7.0]
    vocab_size, scale, softcap = 5, 1.7, 30.0
    prepared = _walk_block_preparation(row, vocab_size, scale, softcap)
    # The existing selection's rank-1 id for this row: the argmax, ties broken by the lower id, over
    # the vocabulary slice that the selector's top-k sees
    expected = max(range(vocab_size), key = lambda token: (prepared[token], -token))

    candidates = extract_candidates([row], k = 1, vocab_size = vocab_size, scale = scale, softcap = softcap)

    assert len(candidates) == 1
    assert candidates[0][0][0] == expected
    # k truncates the list, it does not renormalise it: rank 1 keeps its own probability, which is what
    # a tree multiplies along a path
    weights = [math.exp(value - max(prepared)) for value in prepared]
    assert abs(candidates[0][0][1] - weights[expected] / sum(weights)) < 1e-12


def test_a_tied_pair_resolves_by_token_id_every_time():
    """
    fp16 logits tie exactly often enough to matter, and `torch.topk`'s choice at a tie is unspecified.
    The extractor's must not be: the same row gives the same order in the same process and in the next
    one, so a tree cannot reorder between calls. Lower token id first.
    """
    row = [0.0, 1.5, 1.5, 1.5, -2.0]

    first = extract_candidates([row], k = 3)
    for _ in range(5):
        assert extract_candidates([row], k = 3) == first

    assert [token for token, _ in first[0]] == [1, 2, 3]
    probabilities = [probability for _, probability in first[0]]
    assert probabilities == sorted(probabilities, reverse = True)
    assert abs(probabilities[0] - probabilities[1]) < 1e-15


def test_a_row_with_one_finite_logit_returns_it():
    """
    A masked row - everything at -inf but the token the mask lets through - is the boundary case of a
    distribution: that token carries all the probability. It must come back rather than raise, and it
    must lead.
    """
    row = [float("-inf"), float("-inf"), 3.0, float("-inf")]

    candidates = extract_candidates([row], k = 2)

    assert candidates[0][0] == (2, 1.0)
    # The runner-up is a -inf id at zero probability: still a candidate, never a claim
    assert candidates[0][1][0] in (0, 1, 3)
    assert candidates[0][1][1] == 0.0


def test_probabilities_are_normalised_over_the_vocabulary_not_the_padded_width():
    """
    The head pads its output width, so a row is wider than the vocabulary and the padded columns hold
    whatever the kernel left there. Probabilities are normalised over `vocab_size` only, and no id
    from the padding can be returned.
    """
    row = [1.0, 2.0, 3.0, 4.0, 5.0, 40.0, 40.0, 40.0]
    weights = [math.exp(value) for value in row[:5]]
    total = sum(weights)

    candidates = extract_candidates([row], k = 5, vocab_size = 5)

    assert [token for token, _ in candidates[0]] == [4, 3, 2, 1, 0]
    assert all(token < 5 for token, _ in candidates[0])
    assert abs(sum(probability for _, probability in candidates[0]) - 1.0) < 1e-12
    assert abs(candidates[0][0][1] - weights[4] / total) < 1e-12


def test_the_multiplication_and_the_softcap_are_the_ones_the_selector_applies():
    """
    `walk_block` scales the row by the config's output multiplier and then softcaps it, both monotone:
    they leave the ranking alone but they are not free - the probability a tree weights a branch by is
    the softmax of the capped value. A row whose uncapped order differs from the capped one would
    expose an implementation that skipped the softcap, so this row is built so the cap cannot reorder.
    """
    row = [0.0, 1.0, 2.0]
    scale, softcap = 4.0, 30.0
    prepared = _walk_block_preparation(row, 3, scale, softcap)
    weights = [math.exp(value - max(prepared)) for value in prepared]
    total = sum(weights)

    candidates = extract_candidates([row], k = 2, scale = scale, softcap = softcap)

    assert [token for token, _ in candidates[0]] == [2, 1]
    assert abs(candidates[0][0][1] - weights[2] / total) < 1e-12
    unscaled = math.exp(2.0) / sum(math.exp(value) for value in row)
    assert abs(candidates[0][0][1] - unscaled) > 1e-3


def test_k_is_capped_at_the_vocabulary():
    """Asking for more candidates than the vocabulary holds returns the vocabulary, not an error."""
    candidates = extract_candidates([[0.0, 1.0, 2.0]], k = 7)
    assert [token for token, _ in candidates[0]] == [2, 1, 0]


def test_a_row_with_no_distribution_raises():
    """
    Every logit at -inf is a position with no distribution at all, not a position with an unlikely
    winner: the same is true of a NaN or a +inf. Those raise rather than returning a fabricated
    candidate, because a tree built on one would verify a branch the drafter never proposed.
    """
    for row in ([float("-inf")] * 4, [0.0, float("nan"), 1.0], [0.0, float("inf")]):
        try:
            extract_candidates([row], k = 2)
        except ValueError:
            continue
        raise AssertionError(f"accepted a row with no distribution: {row}")


def test_k_below_one_raises():
    """The extractor's contract starts at one candidate: there is no meaningful rank 0."""
    for k in (0, -1):
        try:
            extract_candidates([[0.0, 1.0]], k = k)
        except ValueError:
            continue
        raise AssertionError(f"accepted k = {k}")


def test_a_block_with_a_batch_axis_is_rejected():
    """
    The caller slices one sequence off the block (`logits[b, 1:]; the anchor row is row 0`). A 3-D
    argument is a batch - sequences that must not be mixed and a row that predicts nothing - so it is
    refused rather than read as a longer vocabulary.
    """
    block = [[[0.0, 1.0], [2.0, 3.0]]]
    for argument in (block, [[[0.0], [1.0]]]):
        try:
            extract_candidates(argument, k = 1)
        except ValueError:
            continue
        raise AssertionError(f"accepted a batch: {argument}")


def test_every_position_of_the_block_is_read():
    """One pass, one distribution per draft position: the result has one entry per row, in order."""
    rows = [[0.0, 1.0], [5.0, 0.0], [0.0, 0.25]]
    candidates = extract_candidates(rows, k = 1)
    assert len(candidates) == 3
    assert [position[0][0] for position in candidates] == [1, 0, 1]


def test_an_empty_block_returns_nothing():
    """No positions, no candidates - and no error, so a zero-length window short-circuits."""
    assert extract_candidates([], k = 2) == []


def test_the_selector_walk_returns_the_extractors_k_1_ids():
    """
    Chain identity against the existing path where that path can be run: `walk_block`'s top-k feeds
    `_walk_torch`, which picks its token by the selector's score - the row's score plus a codebook term
    conditioned on the previous token. With a zero predecessor codebook that term is identically zero,
    the score IS the prepared row, and the walk's ids must be the extractor's rank-1 ids for the same
    logits, prepared the same way, on the same padded width.

    This is the case in which the rerank cannot reorder. It is deliberately not a claim that the walk
    always equals per-position argmax: it does not, and the module's chain is the walk.

    Needs torch, absent on the development box; the direct runner reports it as skipped there.
    """
    if torch is None:
        if pytest is not None:
            pytest.skip("no torch on this interpreter")
        return "skipped: no torch on this interpreter"
    from exllamav3.modules.arch_specific.dflash2 import DFlash2Selector

    class Projection:
        def forward(self, hidden, params):
            return torch.ones((*hidden.shape[:-1], 1), device = hidden.device)

    vocab_size, width, rows, top_k = 6, 9, 4, 3
    selector = object.__new__(DFlash2Selector)
    selector.top_k = top_k
    selector.hidden_proj = Projection()
    selector.pred_codebook = torch.zeros(vocab_size, 1)         # the rerank term vanishes
    selector.succ_codebook = torch.ones(vocab_size, 1)
    # Fixed, widely spaced scores with a different argmax per row: the comparison below holds
    # whichever of float32 (the selector) and double (the extractor) rounds a value, which randn
    # values would not promise
    logits = torch.tensor([
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        [2.0, 1.0, 6.0, 5.0, 4.0, 3.0],
        [3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
    ]).view(1, rows, vocab_size)
    logits = torch.cat([logits, torch.full((1, rows, width - vocab_size), 1e4)], dim = -1)
    anchor = torch.tensor([2])

    path = selector.walk_block(
        torch.zeros(1, rows, 4), logits, anchor,
        vocab_size = vocab_size, scale = 1.7, softcap = 30.0,
    )[0]

    expected = [position[0][0] for position in extract_candidates(
        logits[0], k = 1, vocab_size = vocab_size, scale = 1.7, softcap = 30.0)]

    assert path.shape == (rows + 1,)
    assert path[0].item() == 2
    assert path[1:].tolist() == expected


def _main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            result = test()
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            if isinstance(result, str) and result.startswith("skipped"):
                print(f"SKIP {test.__name__} ({result})")
                continue
            print(f"PASS {test.__name__}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
