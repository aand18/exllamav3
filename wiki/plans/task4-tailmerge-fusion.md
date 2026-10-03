# Task 4 plan: tail_reduce + merge fusion (handoff) — EAGER PATH ONLY

**Premise warning (read first): this cut likely does NOT move the
graph tg number.** The replay graph already covers tailgather →
bmm → tailred → merge (`dispatch.py:385-392` captured as one
replay), so fusing two kernels that replay together saves ~zero in
steady graph mode. The prize below is EAGER path only (44.1 tok/s).
If eager mode does not matter to the maintainer, SKIP this task
after §2 and record the skip — that is a valid, useful outcome.

Branch: `wip/kvarn-cache` (tip, do NOT rebase). Scope: K4V4
decode, eager path, 4090 box. One attempt, twin, env-gated,
fail-closed (project law).

## 0. Facts (measured / in-code)

- Eager per-step tail sequence (`dispatch.py:~540-610`): persistent
  `st_buf` bmm (`torch.bmm(Qh, Kt)`, stays torch — hard constraint)
  + `torch.mul` scale → `kvarn_triton_online_tail_reduce` kernel
  (0.8ms/step total) → 3 reshapes → `kvarn_triton_online_merge`
  kernel → out. Merge wrapper `kvarn_triton.py:1154`, tail kernel
  `:1193-~1230` (num_warps=4, grid `(qh,)` both).
- Prize shape: kill 1 launch/layer + tail-stat DRAM round-trip
  (`tail_m/den` (qh,) + `tail_num` (qh,hd) fp32 ≈ 65KB/layer) +
  the torch gap. Expectation: +1-2% EAGER (≈44.1 → ~44.5-45),
  ~+0% graph. If Phase-0 measurement contradicts this (fused
  slower than separate under replay), STOP.
- Merge math dependency: merge reads serve `m/l` + `out_b` +
  tail stats; tail stats derive from `st` (bmm out). Fusion =
  ONE kernel taking `(st, Vt, tg, exrev, m, l, out_b)` → final
  `(qh,hd)` fp16 out, with tail m/den/num purely on-chip.

## 1. Read this first (no code until done)

1. `kvarn_triton.py:1079-1153` (merge kernel + wrapper: exact
   input/output contract you must preserve).
2. `kvarn_triton.py:1193-~1310` (tail kernel + `tail_reduce` +
   `tail_gather` wrappers: the producer side).
3. `dispatch.py:~540-610` (eager call sequence + persistent
   buffers `_ov_dec_tail`, `_ov_dec_st_cache`, `_ov_dec_out` —
   the fused kernel MUST reuse these buffer names/shapes so
   eager-fallback and graph paths keep sharing temps).
4. `dispatch.py:377-399` (`_graph_capture`: the fused kernel
   must be capturable — no host reads, no retrievable shapes,
   static addresses only; capture will include it automatically
   if these hold).

## 2. Phase 0 — premise check + STOP gate (no kernel edits)

1. Eager `@64k`: `EXL3_KVARN_GRAPH=0` microkld pair (protocol §6
   of task-6 plan). Confirm eager baseline ~44.
2. If the maintainer says eager is irrelevant (graph is default
   ON and staying ON): STOP HERE, write the skip note into
   `doc/kvarn-4090.md` (2 lines: why fused-eager can't move graph
   tg), do not write a kernel. Done.
3. Otherwise spike (`eval/_spike10_tailmerge.py`, never commit):
   hand-fuse tailred+merge for ONE layer on synthetic inputs,
   verify `torch.equal` vs separate launches (same order = exact
   expected), measure per-call delta. Bar: ≥0.5% eager-step win
   projected, else STOP.

## 3. Phase 1 — production (only after §2 bar passes)

- New kernel `_kvarn_online_tailmerge_kernel` beside merge
  (do NOT edit the existing kernels: fallback needs them).
  Grid `(qh,)`, num_warps=4 (match both parents; retune only
  with bill proof).
- Wrapper `kvarn_triton_online_tailmerge(...)` reusing
  `_ov_dec_tail`/`_ov_dec_out` buffer names.
- Gate `EXL3_KVARN_TAILMERGE=1` (default OFF → ON after
  box-green), fail-closed to separate launches on throw, loud
  print. Kill-switch restores legacy.
- Twin `tests/test_kvarn_triton.py::test_tailmerge_matches_split`
  (CUDA-gated): synthetic `st/Vt/tg` incl. R non-pow2 + fully
  masked rows, `torch.equal` (same op order → exact).
- Same §6 box protocol as task-6 plan (EAGER pair control:
  run eager ×2 with gate ON vs OFF in ONE interleaved process
  if the box spread bites — see task-7 §8.4 note on ±25%
  between-process spread; quote only interleaved deltas).

## 4. Non-goals + guards (binding, cf. task-6 plan §5/§5b)

bmm stays torch. No record-format, grid, or dispatch-flow
changes. Hands-off everything outside scope; `git status` +
`git diff --stat` before every commit, explicit per-file `git
add`, never `git add -A`. Spikes untracked, one atomic commit
for the cut + one for the ledger lines.

## 5. Done means

Eager hot ≥45 (≈+2%) with twin + KLD + PARITY green, ledger
entry, pushed — or documented SKIP / revert with the measured
number. Graph number is not expected to move; do not chase it
from this task.
