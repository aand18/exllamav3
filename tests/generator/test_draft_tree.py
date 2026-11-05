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

`max_nodes` is the mask's row count, not a sum of path lengths: every path shares the trunk, so the chain
is charged `len(chain)` rows once and each substitution taken adds the row where its path leaves that
trunk. The window this ships for - a 4-token chain at the default 7 - therefore holds three branches, and
one of the tests below pins exactly that.

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


def _substitutions(path, chain):
    """How many positions `path` replaces relative to `chain`."""
    return sum(1 for kept, original in zip(path, chain) if kept != original)


def _rows(tree, chain):
    """Mask rows a tree occupies: the shared trunk once, plus one row per substituted position."""
    return len(chain) + sum(_substitutions(path, chain) for path in tree)


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
    `max_nodes` bounds the rows of the verification mask, not the sum of the returned paths' lengths: every
    path shares the trunk, so the chain is charged once and each substitution costs the one row where its
    path leaves the trunk. Branches are taken in probability order and any that no longer fits is skipped
    rather than closing the tree. The chain is never dropped to make the budget work: when it alone
    reaches `max_nodes` it is returned alone, and a budget below its length still returns it.
    """
    for max_nodes in (3, 4, 5, 7, 20):
        tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = max_nodes)
        assert _rows(tree, CHAIN) <= max(max_nodes, len(CHAIN)), max_nodes
        assert tree[0] == CHAIN

    # 7 rows: the chain's 3 plus one row each for three substitutions. Six rows are used and no fourth
    # branch exists - the enumeration ends before the budget does.
    tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 7)
    assert tree == [[101, 102, 103], [101, 102, 204], [101, 202, 103], [201, 102, 103]]
    assert _rows(tree, CHAIN) == 6

    # One row of headroom past the trunk buys the best-ranked branch and nothing else.
    assert build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 4) == [CHAIN, [101, 102, 204]]

    # No headroom: the chain alone.
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

    # 0 is the degenerate case: the chain alone, whatever the budget would have bought.
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


def test_the_budget_counts_mask_rows_not_path_lengths():
    """
    The budget is the mask's row count under every substitution cap: the trunk once, one row per
    substituted position. A two-substitution path costs its two rows, so raising the cap moves which
    branches fit without moving the bound - and a branch that does not fit is skipped, not fatal, so a
    cheap branch ranked below an expensive one still gets in.
    """
    for max_nodes in (1, 3, 4, 5, 6, 7, 9, 20):
        for max_substitutions in (0, 1, 2):
            tree = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT,
                              max_nodes = max_nodes, max_substitutions = max_substitutions)
            assert _rows(tree, CHAIN) <= max(max_nodes, len(CHAIN)), (max_nodes, max_substitutions)
            assert tree[0] == CHAIN

    # At one substitution per path every branch is one row: 7 rows hold the trunk and three branches.
    one = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 7, max_substitutions = 1)
    assert one == [CHAIN, [101, 102, 204], [101, 202, 103], [201, 102, 103]]

    # At two, [101, 202, 204] ranks third and costs two rows: 3 + 1 + 1 + 2 = 7 fills the budget exactly.
    two = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 7, max_substitutions = 2)
    assert two == [CHAIN, [101, 102, 204], [101, 202, 103], [101, 202, 204]]
    assert _rows(two, CHAIN) == 7

    # One more row and the single-row branch ranked below it fits as well - the skip rule at work.
    wider = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 8, max_substitutions = 2)
    assert wider[4] == [201, 102, 103]
    assert _rows(wider, CHAIN) == 8

    # The whole enumeration is seven paths: three one-row, three two-row, and the trunk. It needs 12
    # rows, so a budget past that stops being the binding constraint.
    full = build_tree(CHAIN, CANDIDATES, ROUTES, RESIDENT, max_nodes = 21, max_substitutions = 2)
    assert len(full) == 7
    assert _rows(full, CHAIN) == 12


# The deployed window: tabbyAPI drafts four tokens and the mask's default is seven rows. Under the
# per-path budget this returned the chain alone, which is the tree the plan exists to build.
WINDOW_CHAIN = [101, 102, 103, 104]
WINDOW_ROUTES = {**ROUTES, 104: {3}, 205: {0}}
WINDOW_CANDIDATES = [
    [(101, 0.80), (201, 0.12)],
    [(102, 0.70), (202, 0.20)],
    [(103, 0.60), (204, 0.30)],
    [(104, 0.55), (205, 0.40)],
]


def test_the_deployed_window_builds_three_branches():
    """
    The case that motivated the change: a 4-token chain at the default `max_nodes = 7`. The trunk takes
    four rows and each substitution one more, so three substitutions fit - here the ones at positions 3,
    2 and 1 in probability order. The fourth is [201, 102, 103, 104] (0.80 * 0.12 * 0.60 * 0.55 =
    0.0317) and it is the budget that drops it, not the filter: every token on it is resident.

    Under the old per-path accounting the budget was 7 - 4 = 3 < len(chain) = 4, so no branch was
    admissible at all and the tree was the chain alone: four rows in use, three idle.
    """
    tree = build_tree(WINDOW_CHAIN, WINDOW_CANDIDATES, WINDOW_ROUTES, RESIDENT, max_nodes = 7)
    assert tree == [
        [101, 102, 103, 104],
        [101, 102, 103, 205],   # 0.8 * 0.7 * 0.6 * 0.40 = 0.1344
        [101, 102, 204, 104],   # 0.8 * 0.7 * 0.3 * 0.55 = 0.0924
        [101, 202, 103, 104],   # 0.8 * 0.2 * 0.6 * 0.55 = 0.0528
    ]
    assert len(tree) - 1 == 3
    assert _rows(tree, WINDOW_CHAIN) == 7

    # The dropped branch is resident on every token, so the budget is the only reason it is absent.
    assert all(RESIDENT.issuperset(WINDOW_ROUTES[token]) for token in [201, 102, 103, 104])
    assert build_tree(WINDOW_CHAIN, WINDOW_CANDIDATES, WINDOW_ROUTES, RESIDENT, max_nodes = 8)[-1] == \
        [201, 102, 103, 104]


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
