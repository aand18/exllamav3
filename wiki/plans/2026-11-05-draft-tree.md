# Tree verification for DFlash2 — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: `subagent-driven-development`. Steps use `- [ ]`.

**Goal:** turn DFlash2's single non-causal pass into a 6–7 candidate verification tree and accept
longer prefixes per round, without paying extra drafter forwards or extra CPU-expert bytes.

**Architecture:** the drafter already emits distributions for every draft position from one pass; the
engine extracts a top-1 chain. This plan adds a candidate extractor, a router-overlap filter that keeps
only branches the resident expert set already serves, and a generator path that verifies a tree under a
tree mask instead of a chain.

**Tech Stack:** Python 3.12, torch 2.11/CUDA, exllamav3 generator + draft modules, pytest on the Windows
copy, `run_pass.ps1 -Runner spec_decode` for the arm.

**Spec:** backlog item 6 (`wiki/plans/2026-11-04-decode-backlog.md`) — "tree verification for DFlash2",
gated by nothing; it is a product-neutral optimisation of the drafting path.

## Global Constraints

- Everything default-off: `EXL3_DRAFT_TREE` unset must reproduce today's chain byte for byte.
- No change to the served config or the deployment until an arm passes the agentic-weighted bar.
- The drafter's own weights and the number of drafter forwards per round must not change; the tree comes
  from the **same** pass.
- Tests run on the Windows copy; `tests/moe_cpu` stays 124/63 and `tests/generator` must not regress.

## Review Focus

1. **A round whose tree is entirely rejected** — must fall back to the chain's accepted prefix, not to zero tokens.
2. **A tree branch that routes to an expert the filter excluded** — must be dropped before verification, never verified and then discarded.
3. **Positional accounting** — the bonus token after the accepted prefix must come from the same distribution the chain would have used.
4. **Tree size at the attention mask** — 7 candidates must not exceed the mask/cache page assumptions (256-token pages).
5. **Drafting off** — no tree code may run at all.

---

### Task 1: The drafter exports top-2 per position

**Files:** modify the DFlash2 draft module (candidate extraction); test `tests/modules/draft/test_dflash2_tree.py`.

**Interfaces:** `extract_candidates(logits, k: int = 2) -> list[list[tuple[int, float]]]` — per draft
position, the k most likely ids with their probabilities, in the drafter's own token space.

- [ ] **Step 1: failing test** — a synthetic logits tensor with a known argmax and runner-up returns both, ordered by probability.
- [ ] **Step 2: implement** the extraction next to the existing top-1 path; the chain path must call it with `k = 1` and produce identical ids.
- [ ] **Step 3: pass** — the new test plus the existing DFlash2 suite.
- [ ] **Step 4: commit.**

### Task 2: The router-overlap filter

**Files:** new `exllamav3/generator/draft_tree.py`; test `tests/generator/test_draft_tree.py`.

**Interfaces:** `build_tree(chain, candidates, routes: dict[int, set[int]], resident: set[int], max_nodes: int = 7, max_substitutions: int = 1) -> list[list[int]]`
— every root-to-leaf path is a candidate continuation; a branch substitutes up to `max_substitutions`
positions and keeps the chain's tokens elsewhere (`chain[:d] + [alt] + chain[d + 1:]`), because the draft
positions are independent; a branch is kept only when every expert it would route to is in `resident`, and
the total node count stays within `max_nodes`. The `routes` map arrived in implementation: a token the
caller does not report is unknown, and a branch through it is dropped.

- [ ] **Step 1: failing tests** — a branch outside `resident` is dropped; the chain is always present; `max_nodes` is respected; a fully rejected tree returns the chain.
- [ ] **Step 2: implement** the builder as a pure function (no model, no tensors) so it is testable without a GPU.
- [ ] **Step 3: pass**, commit.

### Task 3: Verify the tree in the generator

**Files:** modify the generator's speculative loop; test `tests/generator/test_tree_verify.py`.

**Interfaces:** the loop consumes `build_tree`'s output, runs one forward per tree depth with a tree
attention mask, and accepts the longest root-to-leaf prefix that matches; the bonus token is drawn from
the next distribution exactly as the chain path draws it.

- [ ] **Step 1: failing test** — a stub generator whose tree has one accepted branch of length 3 and a chain of length 2 accepts 3.
- [ ] **Step 2: failing test** — a tree where nothing matches accepts the chain and emits the same bonus token as today.
- [ ] **Step 3: implement**, guarded by `EXL3_DRAFT_TREE` (default 0).
- [ ] **Step 4: pass**, commit.

### Task 4: The arm

**Files:** report `wiki/plans/v165-reports/T44-draft-tree.md`.

- [ ] **Step 1:** pre-register (agentic-weighted bar, same rule as t41) and run `A B B A` with `EXL3_DRAFT_TREE=1` on B.
- [ ] **Step 2:** verdict with `bench/spec_primary.py`; the report states tokens/round and accepted drafts alongside the tok/s ratio.

---

## What this plan does not claim

The proposal's "+25–35 % accepted tokens" is an estimate. The router-overlap filter is what keeps the
extra branches cheap, and it is also what caps the gain: a branch that routes outside the resident set
is dropped, so the win depends on how often the second candidate shares the first's experts — the same
kind of number that turned backlog #1 from "60-75 % hidden" into a measured **49.7 %**.

## Corrections from Tasks 1–2 (2026-11-05)

Implementation contradicted the plan in three places. Task 3 must be read with these in force:

1. **DFlash2's chain is not a per-position top-1.** `walk_block` takes the top-16 of the prepared row and
   `_walk_torch` reranks greedily with predecessor-conditioned codebooks; `sample_from_state` consumes
   that walk. Calling `extract_candidates(k = 1)` from the chain path would replace a trained rerank with
   an argmax and change the drafted ids, so Task 1 **did not** wire it. The tested identity is narrower:
   `k = 1` returns the top-1 of the same prepared row the walk's gate sees, and where the rerank term
   vanishes the real walk reproduces it. **Task 3 must treat `chain` and `candidates` as two independent
   signals** — `build_tree` already tolerates that — not as one nested in the other.
2. **The extractor is a host path and is slow** (1.43 s per round measured in pure Python for 18 × 151 936).
   Task 3 must feed the tree from a device-side `softmax(-1).topk(k)`, and that device path must be tested
   against the host function on the Windows copy so the two cannot drift.
3. **The draft positions are independent** (non-causal rows each predict their own position from the
   target taps and the anchor), so `candidates[d+1]` does not depend on the token taken at `d`. Task 2's
   builder ends every branch at its divergence point and therefore **under-enumerates**: a diverged branch
   may legally continue with the following positions' candidates. Task 3 either extends the builder or
   documents the smaller tree it verifies. **Resolved in Task 2b:** the builder substitutes now
   (`chain[:d] + [alt] + chain[d+1:]`, at most `max_substitutions = 1` positions per path) and the
   truncated form is dropped rather than kept beside it — every prefix a truncated branch can earn is also
   earned by the full path that contains it. Two consequences for Task 3: every branch is now the chain's
   own length, so under the module's per-path node accounting (`max_nodes` = sum of path lengths) a
   `max_nodes` of 2n is the floor for any branch at all — a 4-token chain at the default 7 gets none; and
   the tree's leaves all sit at the same depth, so "longest matching root-to-leaf prefix" is decided by
   content, not by leaf depth.

Two smaller notes: zero-probability candidates are returned rather than dropped (a `routes` entry would
make a zero-score branch legal, so filter `p == 0` before verifying), and nothing in Task 1 touched the
checkpoint, so the runner-up's real probability is still unmeasured — a Windows run counting positions
where `_walk` differs from the argmax would settle it without a GPU pass.
