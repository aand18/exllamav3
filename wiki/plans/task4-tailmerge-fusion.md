# Task 4 plan: tail_reduce + merge fusion (handoff) — EAGER PATH ONLY

> **VERDICT 2026-10-03: STOP. Do not implement §3.** The fused kernel is
> bit-exact and 3.5% faster per call, but **4.5% slower end-to-end**
> eager (50.67 → 48.45 tok/s @64k). No kernel was written. Full numbers
> and the re-aimed candidate in **§6**; ledger in `doc/kvarn-4090.md`;
> commit `ab94d74`.

**Premise warning (read first): this cut likely does NOT move the
graph tg number.** The replay graph already covers tailgather →
bmm → tailred → merge (`dispatch.py:385-392` captured as one
replay), so fusing two kernels that replay together saves ~zero in
steady graph mode. The prize below is EAGER path only (44.1 tok/s —
stale, see §6.1: it is 46.8).
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
  the torch gap. Expectation: +1-2% EAGER (≈46.8 → ~47.5,
  re-measured 2026-10-03 after task-7; was 44.1), ~+0% graph.
  If Phase-0 measurement contradicts this (fused slower than
  separate under replay), STOP.
- Eager traffic is REAL (not just warmup): box tabbyAPI
  `config.yml` sets active `max_batch_size: 2`, so any 2
  overlapping generation jobs batch into bsz=2 decode, and
  `_try_kvarn_graph_decode` declines (`q.shape[0] != 1`,
  `dispatch.py:249`) → the whole contended step runs eager on
  all kvarn layers at the eager rate. Prefill-mixed batches
  decline the same way. Single-stream bench never triggers it;
  under contention it IS the served tg.
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

*(The "≥45" bar below is stale — see §6.1. It was written when the
eager baseline was 44.1; it is now 46.8, so the bar should have read
≥47.7. Moot either way: §6 closed this as a STOP.)*

## 6. RESULTS (2026-10-03, box `4ff897b`, commits `ab94d74` ledger /
##    `ab61c98` this section) — STOP, fused kernel is 4.5% SLOWER
##    eager. No kernel written.

No code change. §3 Phase 1 was never started: the §2.3 bar failed and
§0's guard ("fused slower than separate → STOP") fired. Ledger entry
in `doc/kvarn-4090.md`.

### 6.1 Premise corrections (the plan's §0 numbers were stale)

- **Eager @64k is 46.8 / 46.7 tok/s**, not 44.1 (47.7 @8k; fp16 62.3).
  Measured this session under protocol v3 with `EXL3_KVARN_GRAPH=0`.
  Task-7's serve work moved *eager* too, so §5's "≥45" bar was already
  met by the baseline and a real +2% would have been 46.8 → 47.7.
- **§0's "+1-2% EAGER" was ~10× optimistic.** The tailred+merge pair is
  52.4us × 16 layers = **0.84ms of a 19.7ms eager step (4.2%)**, so the
  *entire* pair could vanish and buy only +4.2%. Fusing captures a
  fraction of that, so +1-2% was never on the table.
- The §2.2 gate question was worth asking and was answered *yes* — see
  §0's `max_batch_size: 2` note, added by the maintainer mid-task.
  Eager traffic is real; it just does not make this cut worthwhile.

### 6.2 The fused kernel: bit-exact, faster in isolation, slower live

Spike `eval/_spike10_tailmerge.py` (never committed), log `t4_spike10.log`
on the mirror. Real in-situ 64k inputs snapshotted at the tail/merge
wrapper boundaries of a live layer: `st=(4,6,256) vt=(256,4,256)
tg=(256,) exrev≈(518,) m=l=(4,8,64) out_b=(24,256)`, qpk=6, gc=64,
R=256 (qh=24, kvh=4, hd=256, 16 kvarn layers; R is pow2 here).

| metric | split (shipped) | fused | Δ |
|---|---|---|---|
| bit-exactness | — | `torch.equal` True, maxabs **0.0**, 0/6144 differ | win |
| device, per call | 52.4us | 50.6us | **−3.5%** |
| python issue, per call | 30.3us | 13.5us | **−16.8us** |
| **eager step (e2e)** | **19.737ms / 50.67 tok/s** | **20.638ms / 48.45** | **−4.5%** |

Exactness is structural, not luck: identical op order, and the fp32
tail-stat DRAM round-trip the fusion removes is value-preserving, so
register residency *cannot* change the result. Device and host-issue
numbers reproduced to ±0.15us across three interleaved runs; the
end-to-end A/B ran twice (8 and 10 windows of 16 steps, in-process,
order alternated per window — task-7 §8.4: between-process deltas on
this box are ±25% and worthless). In the 10-window run **all 10 fused
windows beat the split median and 7 of 10 beat EVERY split window**;
even the fused arm's fastest window (20.29ms) beat 9 of the 10 split
windows. Fused 20.29–20.86 excluding two outliers (30.92, 31.09 — the
periodic box event task-7 §8.4 recorded); split 19.43–20.39.

### 6.3 Why both microbenchmarks won and the step still lost

Fusing turns two independent 24-CTA kernels per layer into one longer
dependent chain, and that costs cross-layer overlap: tail of layer N+1
has **no** dependency on merge of layer N, so while they are separate
the long tail loop hides underneath it; fused, it cannot. That loss
(~0.9ms/step) is ~60× the ~15us the removed launch actually saved.
Mechanism inferred from the timings, not separately profiled — but the
verdict does not rest on it: the cut loses on every end-to-end metric,
and it fails §2.3's bar on the isolated device delta alone (+0.15%).

**Method note for the next person:** device-event and python-issue
timings of a fused-vs-split pair can BOTH say "win" while the step
loses. Take the interleaved end-to-end A/B as the decider, always.

### 6.4 Where the tail time actually is (re-aims a future cut)

52.4us/layer to read 1.25MB is ~50× off roofline. The cost is the tail
kernel's **serial `for r in tl.range(R)` reduction chain** —
`er = tl.sum(tl.where(roff == r, e, 0.0))` recomputes a full RPAD-wide
reduction once per row, R=256 times
(`kvarn_triton.py:1182`, `_kvarn_online_tail_kernel`). Not the
tail-stat DRAM round-trip this task targeted. A future tail-side cut
should attack that loop (blocked reduction, or a `tl.dot` over
`e × vt`); note a dot formulation reassociates, so it gates on allclose
+ KLD, **not** `torch.equal`.

### 6.5 Do not re-propose

Tail+merge fusion, in any direction, on either path. Negative in
isolation-prize terms (+0.15% ceiling) and negative end-to-end
(−4.5%). The graph path was never in scope — §0's premise there holds:
tailgather → bmm → tailred → merge already replay as one captured unit
(`dispatch.py:385-392`), so graph tg is untouched by this question.
