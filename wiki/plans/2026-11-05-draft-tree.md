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

**Interfaces:** `build_tree(chain, candidates, resident: set[int], max_nodes: int = 7) -> list[list[int]]`
— every root-to-leaf path is a candidate continuation; a branch is kept only when every expert it would
route to is in `resident`, and the total node count stays within `max_nodes`.

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
