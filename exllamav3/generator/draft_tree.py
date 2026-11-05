"""
Resident-set filter for the DFlash2 verification tree (EXL3_DRAFT_TREE).

DFlash2 emits a distribution for every draft position from one pass; today the engine keeps the top-1
chain. A verification tree spends a few extra mask rows on the continuations the drafter ranked a close
second. Every extra branch still routes experts through the CPU tier along its whole path, so a branch
is worth verifying only when the resident expert set already serves every expert it would touch -
otherwise the round pays to bring in experts for a token it is likely to reject.

`build_tree` is that decision and nothing else: pure, no torch, no model, no globals. The caller passes
the router's view of each token's experts (`routes`); the builder never assumes that view is complete,
because a token the router has not seen is a token whose experts are unknown, and a branch through
unknown experts cannot be shown to be resident. A token whose `routes` entry exists but is empty is
known to route nowhere, so it passes the filter vacuously.

A path is one root-to-leaf continuation, given as its token ids from the root. The chain is the trunk of
every branch: a branch diverging at position `d` is `chain[:d] + [alternative]` and ends there, because
`candidates[d + 1]` was produced under the chain's token at `d` and says nothing about the alternative's
successor. Leaves therefore sit at different depths, which is what the verifier accepts against (the
longest root-to-leaf prefix that matches).

Ranking is by cumulative probability: the product of the per-position probabilities `candidates` reports
for the tokens on the path. A position the caller does not report - a `candidates` list shorter than the
chain, or a chain token missing from its own ranked list - contributes nothing rather than a guessed
value. Branches are taken in that order while they fit, skipping any that no longer does, so a cheap
third-ranked branch may outlive an expensive second-ranked one. The chain stays first whatever its own
probability and whatever the filter says about it: it is what the engine does today, and a tree whose
branches are all rejected is a round that still verifies the chain.

Node accounting: `max_nodes` bounds the sum of the returned paths' lengths, so a branch is charged for
the chain prefix it shares. That is the conservative reading of a tree mask that would share those
prefix nodes, and it keeps the contract checkable without modelling the mask. The budget covers the
chain as well; when the chain alone reaches it, the result is the chain alone, and when the chain alone
exceeds it the chain is still returned - a tree is never empty and never shorter than today's block.
"""


def _candidate_prob(candidates: list[list[tuple[int, float]]], depth: int, token: int):
    """The probability `candidates` reports for `token` at `depth`, or None when it reports none."""
    if depth >= len(candidates):
        return None
    for cand_token, prob in candidates[depth]:
        if cand_token == token:
            return float(prob)
    return None


def _score(path: list[int], candidates: list[list[tuple[int, float]]]) -> float:
    """Cumulative probability of a path; unreported positions contribute nothing."""
    score = 1.0
    for depth, token in enumerate(path):
        prob = _candidate_prob(candidates, depth, token)
        if prob is not None:
            score *= prob
    return score


def _served(path: list[int], routes: dict[int, set[int]], resident: set[int]) -> bool:
    """True when every token on the path has known experts and all of them are resident."""
    for token in path:
        experts = routes.get(token)
        if experts is None:
            return False
        if not resident.issuperset(experts):
            return False
    return True


def build_tree(chain: list[int],
               candidates: list[list[tuple[int, float]]],
               routes: dict[int, set[int]],
               resident: set[int],
               max_nodes: int = 7) -> list[list[int]]:
    """
    Build the verification tree around `chain`, keeping only the branches the resident expert set pays
    for.

    :param chain:
    The drafter's top-1 continuation: the block the engine verifies today.

    :param candidates:
    Per position, the drafter's ranked (token_id, probability) list. The chain's own token is normally
    the first entry; alternatives are every other entry.

    :param routes:
    Token id -> the experts it would route to. A missing entry means unknown, and a branch through an
    unknown token is dropped.

    :param resident:
    The experts the round can serve without a swap.

    :param max_nodes:
    Upper bound on the total number of nodes in the returned paths, chain included.

    :return:
    The chain first, then the kept branches by cumulative probability descending.
    """

    chain = list(chain)
    if not chain:
        return []

    budget = max_nodes - len(chain)

    branches = []
    seen = set()
    for depth in range(len(chain)):
        if depth >= len(candidates):
            break
        for token, _ in candidates[depth]:
            if token == chain[depth]:
                continue
            path = chain[:depth] + [token]
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            if not _served(path, routes, resident):
                continue
            branches.append((_score(path, candidates), path))

    # Deterministic: probability descending, then the path itself so equal scores never reorder between
    # calls (dict and set iteration order is not a promise the caller made).
    branches.sort(key = lambda branch: (-branch[0], branch[1]))

    tree = []
    remaining = budget
    for _, path in branches:
        if len(path) > remaining:
            continue
        tree.append(path)
        remaining -= len(path)

    return [chain] + tree
