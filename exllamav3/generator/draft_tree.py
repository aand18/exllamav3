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
every branch, and a branch substitutes at one position and keeps the chain for the rest:
`chain[:d] + [alternative] + chain[d + 1:]`. That is legal because DFlash2's draft positions are
independent - each row predicts its own position from the target taps and the anchor - so
`candidates[d + 1]` does not depend on which token was taken at `d`, and a diverged branch may continue
with the following positions' candidates. Ending a branch at its divergence point, as this builder first
did, under-enumerated the tree: every prefix a truncated branch can earn is also earned by the
substituted full path that contains it, and the full path can earn more, so the truncated form is
dropped rather than kept alongside.

`max_substitutions` caps how many positions one path may substitute, which keeps the enumeration small
(the default of 1 is one branch per alternative per position). The chain's own token at a substituted
position is never both kept and replaced: a path differs from the chain in at least one position, or it
is the chain. Two paths that come out equal - a `candidates` entry that repeats a token - collapse into
one.

Ranking is by cumulative probability: the product of the per-position probabilities `candidates` reports
for the tokens on the path. A position the caller does not report - a `candidates` list shorter than the
chain, or a chain token missing from its own ranked list - contributes nothing rather than a guessed
value. Branches are taken in that order, and the budget's skip rule still applies even though every
substitution path now costs the same: a branch that does not fit is passed over rather than closing the
tree. The chain stays first whatever its own probability and whatever the filter says about it: it is
what the engine does today, and a tree whose branches are all rejected is a round that still verifies
the chain.

Node accounting: `max_nodes` bounds the sum of the returned paths' lengths, so a branch is charged for
the chain prefix it shares. That is the conservative reading of a tree mask that would share those
prefix nodes, and it keeps the contract checkable without modelling the mask. Substitution paths make
that reading expensive: every branch is now exactly as long as the chain, so a budget of `max_nodes`
holds at most `(max_nodes - len(chain)) // len(chain)` branches, and a chain longer than half the budget
holds none. The budget covers the chain as well; when the chain alone reaches it, the result is the
chain alone, and when the chain alone exceeds it the chain is still returned - a tree is never empty and
never shorter than today's block.
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


def _served_token(token: int, routes: dict[int, set[int]], resident: set[int]) -> bool:
    """True when `token` has known experts and all of them are resident."""
    experts = routes.get(token)
    if experts is None:
        return False
    return resident.issuperset(experts)


def _substitutions(chain: list[int],
                   candidates: list[list[tuple[int, float]]],
                   routes: dict[int, set[int]],
                   resident: set[int],
                   max_substitutions: int) -> list[list[int]]:
    """
    Every path that differs from `chain` in one to `max_substitutions` positions and keeps the chain's
    tokens everywhere else, unordered and possibly empty.

    The walk is depth-first over positions and prunes as it goes: a token the filter rejects cannot be
    rescued by anything appended after it, because the rule is that every token on the path is served,
    so a rejected prefix ends that whole subtree of the enumeration. The prune is what keeps the count
    near the number of alternatives actually on offer rather than the product of the per-position
    widths, and it is also what makes a substituted path fail on a neighbour it keeps from the chain.
    """
    chain = list(chain)
    out: list[list[int]] = []

    def walk(depth: int, built: list[int], remaining: int) -> None:
        if depth == len(chain):
            out.append(list(built))
            return

        token = chain[depth]
        if _served_token(token, routes, resident):
            built.append(token)
            walk(depth + 1, built, remaining)
            built.pop()

        if remaining > 0 and depth < len(candidates):
            for alternative, _ in candidates[depth]:
                if alternative == token or not _served_token(alternative, routes, resident):
                    continue
                built.append(alternative)
                walk(depth + 1, built, remaining - 1)
                built.pop()

    walk(0, [], max_substitutions)

    chain_key = tuple(chain)
    return [path for path in out if tuple(path) != chain_key]


def build_tree(chain: list[int],
               candidates: list[list[tuple[int, float]]],
               routes: dict[int, set[int]],
               resident: set[int],
               max_nodes: int = 7,
               max_substitutions: int = 1) -> list[list[int]]:
    """
    Build the verification tree around `chain`, keeping only the branches the resident expert set pays
    for.

    :param chain:
    The drafter's top-1 continuation: the block the engine verifies today.

    :param candidates:
    Per position, the drafter's ranked (token_id, probability) list. The chain's own token is normally
    the first entry; alternatives are every other entry. A substituted branch keeps the chain's tokens
    at the other positions, because the draft positions are independent of one another.

    :param routes:
    Token id -> the experts it would route to. A missing entry means unknown, and a branch through an
    unknown token is dropped.

    :param resident:
    The experts the round can serve without a swap.

    :param max_nodes:
    Upper bound on the total number of nodes in the returned paths, chain included.

    :param max_substitutions:
    How many positions one path may replace with an alternative. 0 returns the chain alone; the default
    of 1 is one branch per alternative per position.

    :return:
    The chain first, then the kept branches by cumulative probability descending.
    """

    chain = list(chain)
    if not chain:
        return []

    # A substitution path is as long as the chain, so the chain's length is what every branch costs: a
    # budget that cannot hold a second full path holds no branch at all.
    budget = max_nodes - len(chain)
    if max_substitutions < 1 or budget < len(chain):
        return [chain]

    branches = []
    seen = {tuple(chain)}
    for path in _substitutions(chain, candidates, routes, resident, max_substitutions):
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
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
