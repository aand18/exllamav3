"""
The resident-set filter for the DFlash2 verification tree.

A verification tree buys accepted tokens per round with extra mask rows, and each extra row routes its
path's experts through the CPU tier. The tree therefore only pays when the branches it verifies are
served by experts the round already holds resident: a branch that would import an expert for a token
the verifier is likely to reject spends the round's expert bytes on nothing. `build_tree` makes that
decision and nothing else, so it is testable here without a GPU, a model or a drafter.

The chain is what the engine verifies today and is present in every result, first, even when the filter
rejects it - a tree whose branches all fail must degrade to today's block, not to an empty one.

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
    rejection. Every kept branch is a full root-to-leaf path, not a suffix.
    """
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert tree[0] == CHAIN
    assert CHAIN in tree
    assert tree[1:] == [
        [101, 102, 204],
        [101, 202],
        [201],
    ]
    assert all(tree.count(path) == 1 for path in tree)


def test_a_branch_outside_the_resident_set_is_dropped():
    """
    [203] routes to expert 5, which the round does not hold: verifying it would pay the CPU expert tier
    for a branch the resident set does not serve. Its sibling [201] routes to {0, 3} and stays.
    """
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert [203] not in tree
    assert [201] in tree
    # ... and holding expert 5 makes it legal again, so the drop is the filter's and not the shape's.
    tree_all = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT | {5}, max_nodes = 20)
    assert [203] in tree_all


def test_a_branch_with_a_token_of_unknown_routes_is_dropped():
    """
    The router's view is incomplete by contract: a token with no `routes` entry has unknown experts, and
    a branch cannot be shown to be resident when its experts are not known. [101, 301] is that branch;
    [101, 202] shares the shape and keeps its place.
    """
    assert 301 not in ROUTES
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert [101, 301] not in tree
    assert [101, 202] in tree


def test_max_nodes_is_respected():
    """
    The returned paths' lengths sum to at most `max_nodes`, charged for the shared prefix as well. The
    branches are taken in probability order and any that no longer fits is skipped rather than closing
    the tree, so a cheaper branch may outlive an expensive one. The chain is never dropped to make the
    budget work: when it alone exceeds `max_nodes` it is returned alone.
    """
    for max_nodes in (3, 4, 5, 7, 20):
        tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = max_nodes)
        assert sum(len(path) for path in tree) <= max(max_nodes, len(CHAIN)), max_nodes
        assert tree[0] == CHAIN

    # 7 nodes, chain of 3: the 0.252 branch (3 nodes) and then the 0.15 branch (1 node) fit, the 0.16
    # branch (2 nodes) does not once those are in.
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 7)
    assert tree == [[101, 102, 103], [101, 102, 204], [201]]

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
    per-position rank. Here the deepest branch (0.8 * 0.7 * 0.45 = 0.252) outranks a shallower one
    (0.8 * 0.2 = 0.16), which outranks the one that diverges first (0.15).
    """
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20)
    assert tree == [
        [101, 102, 103],
        [101, 102, 204],   # 0.252
        [101, 202],        # 0.16
        [201],             # 0.15
    ]
    order = [[101, 102, 204], [101, 202], [201]]
    assert tree[1:] == order
    # Same inputs, same result: the ordering carries no dependence on set or dict iteration.
    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 20) == tree


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
