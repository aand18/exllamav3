# Fancy-index in-place ops are silent no-ops

## Problem
`tensor[index_tensor].copy_(x)` / `.zero_()` / `.add_()` silently do
nothing: advanced (tensor) indexing returns a **copy**, so the
in-place method writes into a discarded temporary. No error, no
warning. Basic (int/slice) indexing returns a view and works.

## Rule
- Writes through fancy indexing MUST use indexed **assignment**:
  `t[idx] = x`, `t[idx] += ...` is NOT safe either (read-modify-write
  on a copy) -- use `t[idx] = t[idx] + ...` or `index_put_` /
  `index_fill_` / `scatter_*`.
- Reads/gathers via fancy indexing are fine (correct copies).
- Audit any `.copy_(` / `.zero_()` / `.*_(` whose base is `t[...]`
  with a tensor index. `t[mask]`, `t[idx_tensor]`, `t[[0]]` are all
  fancy; `t[0]`, `t[0:4]`, `t[es]` (Python int) are basic (safe).

## Evidence
- 2026-09-30: batched eref refresh `w[_ess].copy_(...)` silently
  dropped every refresh (stale cache, zero error); eref twin caught
  it in CI. Fix: `w[_ess] = ...` (`277efe8`, gotcha note in
  `cdfd476`). Debug cost ~1hr; the twin paid for itself.
- Same class verified safe: `present[gs] = True`, `group_base[gs]`,
  `records[gs] = recs` (all `__setitem__`, write through).

## Scope
Any torch code in this repo. Add to review checklist for every diff
touching indexed writes.
