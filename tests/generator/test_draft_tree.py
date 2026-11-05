"""
The resident-set filter for the DFlash2 verification tree.

A verification tree buys accepted tokens per round with extra mask rows, and each extra row routes its
path's experts through the CPU tier. The tree therefore only pays when the branches it verifies are
served by experts the round already holds resident: a branch that would import an expert for a token
the verifier is likely to reject spends the round's expert bytes on nothing. `build_tree` makes that
decision and nothing else, so it is testable here without a GPU, a model or a drafter.

The chain is what the engine verifies today and is present in every result, first, even when the filter
rejects it - a tree whose branches all fail must degrade to today's block, not to an empty one.

A branch substitutes one position and keeps the chain for the rest, because DFlash2's draft positions are
independent: the row that predicts position `d + 1` does not depend on which token was taken at `d`, so a
branch that diverged at `d` may continue with the following positions' candidates. The tests below pin
that shape, the cap on how many positions one path may substitute, and that a substituted path is still
filtered on every token it carries - including the ones it kept from the chain.

pytest is absent on the Linux box this file is developed on, so it also runs directly:
`python3 tests/generator/test_draft_tree.py`.
"""

import importlib.util
import os
import sys
import traceback

try:
    from exllamav3.generator.draft_tree import build_tree
except (ImportError, RuntimeError):
    # No torch on this box: `exllamav3/__init__.py` refuses to import without it. The builder is pure,
    # so load it straight from its file and leave the package out of it.
    _PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "exllamav3", "generator", "draft_tree.py")
    _SPEC = importlib.util.spec_from_file_location("draft_tree_under_test", _PATH)
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    build_tree = _MODULE.build_tree

try:
    import pytest
    pytestmark = pytest.mark.nogpu
except ImportError:
    # The direct runner at the bottom of the file is the gate when pytest is not installed.
    pass


# One fixture, four words of tokens. 203 routes to expert 5, which is not resident; 301 the router has
# never seen. Everything else is served by experts 0..3.
RESIDENT = {0, 1, 2, 3}
ROUTES = {
    101: {0}, 102: {1}, 103: {2},
    201: {0, 3}, 202: {1}, 204: {2},
    203: {5},
}
CHAIN = [101, 102, 103]
CANDIDATES = [
    [(101, 0.80), (201, 0.15), (203, 0.05)],
    [(102, 0.70), (202, 0.20), (301, 0.10)],
    [(103, 0.55), (204, 0.45)],
]


def test_the_chain_is_always_present_and_first():
    """
    The chain is the block the engine verifies today, so it leads the result and survives every
    rejection. Every kept branch is a full root-to-leaf path, not a suffix, and a branch that diverges
    early still carries the chain's later tokens: the draft positions are independent, so position 1's
    alternative does not invalidate position 2's candidate.
    """
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert tree[0] == CHAIN
    assert CHAIN in tree
    assert tree[1:] == [
        [101, 102, 204],
        [101, 202, 103],
        [201, 102, 103],
    ]
    assert all(tree.count(path) == 1 for path in tree)


def test_a_branch_outside_the_resident_set_is_dropped():
    """
    [203, 102, 103] carries 203, which routes to expert 5 - an expert the round does not hold: verifying
    it would pay the CPU expert tier for a branch the resident set does not serve. Its sibling at the
    same position, [201, 102, 103], routes to {0, 3} and stays.
    """
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert [203, 102, 103] not in tree
    assert [201, 102, 103] in tree
    # ... and holding expert 5 makes it legal again, so the drop is the filter's and not the shape's.
    tree_all = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT | {5}, max_nodes = 20)
    assert [203, 102, 103] in tree_all


def test_a_branch_with_a_token_of_unknown_routes_is_dropped():
    """
    The router's view is incomplete by contract: a token with no `routes` entry has unknown experts, and
    a branch cannot be shown to be resident when its experts are not known. [101, 301, 103] is that
    branch; [101, 202, 103] shares the shape and keeps its place.
    """
    assert 301 not in ROUTES
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert [101, 301, 103] not in tree
    assert [101, 202, 103] in tree


def test_max_nodes_is_respected():
    """
    The returned paths' lengths sum to at most `max_nodes`, charged for the shared prefix as well. The
    branches are taken in probability order and any that no longer fits is skipped rather than closing
    the tree. The chain is never dropped to make the budget work: when it alone exceeds `max_nodes` it is
    returned alone.
    """
    for max_nodes in (3, 4, 5, 7, 20):
        tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = max_nodes)
        assert sum(len(path) for path in tree) <= max(max_nodes, len(CHAIN)), max_nodes
        assert tree[0] == CHAIN

    # 7 nodes, chain of 3: the 0.252 branch (3 nodes) fits and nothing else does, since every
    # substitution path costs the chain's full length.
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 7)
    assert tree == [[101, 102, 103], [101, 102, 204]]

    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 3) == [CHAIN]
    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 1) == [CHAIN]


def test_a_fully_rejected_tree_returns_the_chain_alone():
    """
    Nothing known to be resident, nothing resident at all, and a resident set that excludes the chain's
    own experts: the tree is the chain in each case. A rejected tree is a chain round, never an empty
    one (a round that verifies no rows returns no tokens).
    """
    assert build_tree(CHAIN, CANDIDATES, {}, set(), max_nodes = 20) == [CHAIN]
    assert build_tree(CHAIN, CANDIDATES, ROUTES, set(), max_nodes = 20) == [CHAIN]
    assert build_tree(CHAIN, CANDIDATES, ROUTES, {1, 2}, max_nodes = 20) == [CHAIN]
    assert build_tree([], [], {}, RESIDENT) == []


def test_paths_are_ordered_by_probability_after_the_chain():
    """
    The chain leads; the branches follow by cumulative probability, not by depth and not by the drafter's
    per-position rank. Every path is now the chain's length, so the products are comparable: the
    last-position substitution (0.8 * 0.7 * 0.45 = 0.252) outranks the middle one (0.8 * 0.2 * 0.55 =
    0.088), which outranks the one that diverges first (0.15 * 0.7 * 0.55 = 0.058).
    """
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert tree == [
        [101, 102, 103],
        [101, 102, 204],   # 0.252
        [101, 202, 103],   # 0.088
        [201, 102, 103],   # 0.058
    ]
    order = [[101, 102, 204], [101, 202, 103], [201, 102, 103]]
    assert tree[1:] == order
    # Same inputs, same result: the ordering carries no dependence on set or dict iteration.
    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20) == tree


# A second fixture, for the filter's reach over a path's UNCHANGED tokens: 402 routes to expert 7, which
# is not resident, while the alternatives 411 and 412 are resident on their own. A substitution at
# position 0 keeps 402, so the path must fall with it.
UNSERVED_CHAIN = [401, 402, 403]
UNSERVED_ROUTES = {
    401: {0}, 402: {7}, 403: {2},
    411: {1}, 412: {3},
}
UNSERVED_CANDIDATES = [
    [(401, 0.80), (411, 0.20)],
    [(402, 0.90)],
    [(403, 0.70), (412, 0.30)],
]


def test_a_substitution_keeps_the_rest_of_the_chain():
    """
    The point of the correction: a branch substitutes one position and continues with the chain's
    tokens. The alternative at index 0 gives chain[:0] + [201] + chain[1:] = [201, 102, 103] - not the
    truncated [201] the first builder returned, which would stop exactly where the verifier could still
    have accepted two more tokens. Same at the last index: chain[:2] + [204] + chain[3:] is
    [101, 102, 204].
    """
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)

    first = CHAIN[:0] + [201] + CHAIN[1:]
    assert first == [201, 102, 103]
    assert first in tree

    last = CHAIN[:2] + [204] + CHAIN[3:]
    assert last == [101, 102, 204]
    assert last in tree

    # Nothing is truncated at its divergence point, and nothing is a suffix.
    assert [201] not in tree
    assert [101, 202] not in tree
    assert all(len(path) == len(CHAIN) for path in tree)


def _substitutions(path, chain):
    """How many positions `path` replaces relative to `chain`."""
    return sum(1 for kept, original in zip(path, chain) if kept != original)


def test_substitutions_are_capped():
    """
    `max_substitutions` is what keeps the enumeration small: at 1 no path may replace two positions, at
    2 the two-position path [101, 202, 204] appears. The cap counts positions replaced, not alternatives
    offered, so one position can never be substituted twice.
    """
    one = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 40, max_substitutions = 1)
    assert max(_substitutions(path, CHAIN) for path in one) == 1
    assert [101, 202, 204] not in one
    assert all(_substitutions(path, CHAIN) <= 1 for path in one)

    two = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 40, max_substitutions = 2)
    assert [101, 202, 204] in two
    assert max(_substitutions(path, CHAIN) for path in two) == 2
    assert all(_substitutions(path, CHAIN) <= 2 for path in two)

    # 0 is the degenerate case: the chain alone, which is also what the default budget leaves when it
    # cannot hold a second full-length path.
    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 40,
                      max_substitutions = 0) == [CHAIN]


def test_a_substituted_path_is_filtered_on_every_token():
    """
    A substitution is filtered on the whole path, not just on the token it replaces. Here both
    alternatives are resident on their own, but every path through them carries 402, which routes to
    expert 7 - so the tree is the chain alone. Holding expert 7 puts both paths back, which is what
    proves the drop was the neighbour's doing and not the substitution's.
    """
    tree = build_tree(UNSERVED_CHAIN, UNSERVED_CANDIDATES, UNSERVED_ROUTES, RESIDENT, max_nodes = 20)
    assert tree == [UNSERVED_CHAIN]
    assert [411, 402, 403] not in tree
    assert [401, 402, 412] not in tree

    tree_all = build_tree(UNSERVED_CHAIN, UNSERVED_CANDIDATES, UNSERVED_ROUTES, RESIDENT | {7},
                          max_nodes = 20)
    assert [411, 402, 403] in tree_all
    assert [401, 402, 412] in tree_all


def test_max_nodes_still_bounds_the_total():
    """
    The budget is still the sum of the returned paths' lengths, chain included. Substitution paths are
    all the chain's length, so the bound reads as a branch count - and at 21 nodes a chain of 3 plus six
    full-length branches (three one-substitution and three two-substitution) fills it exactly.
    """
    for max_nodes in (1, 3, 4, 5, 6, 7, 9, 20):
        for max_substitutions in (0, 1, 2):
            tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT,
                              max_nodes = max_nodes, max_substitutions = max_substitutions)
            total = sum(len(path) for path in tree)
            assert total <= max(max_nodes, len(CHAIN)), (max_nodes, max_substitutions, total)
            assert tree[0] == CHAIN

    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 7) == [CHAIN, [101, 102, 204]]
    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 9) == [
        CHAIN, [101, 102, 204], [101, 202, 103],
    ]

    full = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 21, max_substitutions = 2)
    assert sum(len(path) for path in full) == 21
    assert len(full) == 7
    assert [101, 202, 204] in full
    # 21 nodes holds all six branches (18) under the chain's 3; at 20 the budget is 17 and only five
    # fit, so the lowest-ranked one goes.
    assert [101, 202, 204] in build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20,
                                         max_substitutions = 2)
    assert [201, 202, 103] not in build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20,
                                             max_substitutions = 2)


def _main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
