# Slot assignment order is not a contract (rev maps are)

## Problem
Stage/exact slots are scratch: WHICH slot a group gets is an
allocation accident. Tests (and code) that hardcode slot indices
(`stage_k[2]`, `exact_k[0]`) break whenever anything changes slot
consumption (e.g. a fast path that skips staging) even though all
values are correct.

## Rule
- ALWAYS resolve through the rev maps: `_stage_rev[g]`,
  `_exact_rev[g]` (guard `>= 0`). Never index slots by group order.
- Snapshot helpers for tests must gather per-group rows via rev
  (see `test_deferred_pressure_seals_match_immediate`, which does
  this right and survived the fast-path change untouched).
- Staging content of SEALED groups is dead by construction (no
  reader: copy_page rebuilds sealed-group staging from records,
  spills restore from call rows). Do not assert on it across paths;
  do assert records/sealed/present/base/valid/owners.

## Evidence
- 2026-09-30: seal-direct fast path consumed no staging slots ->
  `test_copy_page` hardcoded `stage_k[2]` read an unused zeroed
  slot (content lived in slot 1, rev-correct). Fixed by rev
  resolution, not by code change (`3ce676f`).
- Same cut: deferred-test seal-call counts assumed pressure that
  the fast path avoids -> 2x2 fast/defer matrix (`3ce676f`).

## Scope
Tests and any new code touching stage/exact slots.
