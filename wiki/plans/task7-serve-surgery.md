# Task 7 plan: serve traffic/compute surgery (handoff) — HIGH RISK

Goal: cut the 4.5ms/step kvarn-serve device cost @64k (23% of the
19.27ms wall, the biggest device item left). tg@64k graph is 47.9;
serve alone bounds the prize at ~+25%, realistic target +3-8% for
a clean win (hot ≥ 49.5).

**Risk banner (read twice): perf can go BACKWARDS.** The flat-32
attempt cut partials traffic 16x and LOST 4%: serve is
parallelism-bound, not traffic-bound
(`kvarn_triton.py:~1845-1857`, the comment at the launcher — the
first thing you read in §1). Every attempt must prove occupancy
preserved (CTA count + wall) or REVERT. No stacking attempts; one
at a time, each independently A/B'd.

Branch: `wip/kvarn-cache` (tip at handoff, do NOT rebase). Scope:
K4V4, decode serve only, 4090 box. One attempt per commit, twin
per kernel change, env-gated, fail-closed (project law).

## 0. Facts you must not re-derive (measured 2026-10-02/03)

- Serve kernel `_kvarn_online_serve_kernel`
  (`kvarn_triton.py:1556-~1790`): grid `(kvh=4, groups=128)` =
  512 CTAs @64k, `num_warps=4, num_stages=1`. 4.46ms/step total
  (task-6 spike bill; audit said 4.0-4.5, same).
- Hierarchical: `_kvarn_serve_groups` caps at 128 (`:1799-1806`);
  CPG=4 chunks/program @64k (`8*CPG` iters × TOK=16 rows).
- Per row per iter (read the loop `:1602-~1740` before touching):
  K nibble-unpack (K4 fast path) + sc/zp/oth f16 (uniform-group
  vector path common case) + K dequant MULs + exact-direct loads +
  seal check + 1-2 MMA QK dots (fp16, tail dot skipped when
  all-body) + online softmax (`exp`) + V unpack/dequant + EV dot
  + fp32 partials accumulate (`m/l/acc`, 8MB traffic/step).
- Twin precedent: memory id 74 (Sept spike: scale-hoisting was
  bit-exact RMSE 3e-08 but ZERO gain — stage-1 is ALU/latency
  bound: unpack chains + exp/div depth, NOT metadata reloads).
  Do not re-propose pure metadata-hoisting without new evidence.
- Store (0.228ms/layer) stays eager and OUT of serve capture.
  Combine already split (`_kvarn_launch_combine`); merge/tail
  fusion is task §4 (independent — do not touch here).

## 1. Read this first (in order, no code until done)

1. `kvarn_triton.py:~1840-1870` (launcher comment: flat-32
   lesson, occupancy math — your constraint envelope).
2. `kvarn_triton.py:1556-1790` (the whole serve kernel; note
   every traffic source per row: `kpay_row/nptr`, `sc/zp/ot_ptr`,
   `ek_ptr`, `vpay_row`, `m/l/acc` partials `:~1785-1795`).
3. `kvarn_triton.py:1799-1806` (`_kvarn_serve_groups`: grid math).
4. Memory id 74 context (ask maintainer for `obs-e81dd7d354fd805d`
   content if unavailable: what v3 hoisting tried and why it
   failed — your negative boundary).
5. `doc/perf-strats.md` §7 (this task) + task-6 plan §8 (why the
   host pool is only 2.77ms — device is where the prize is).

## 2. Phase 0 — roofline attribution + STOP gate (no kernel edits)

1. Kineto one decode step @64k (reuse
   `eval/kvarn_prof_kineto.py -ntok 65536 --stacks`): break the
   4.46ms serve time into sections (unpack/dequant vs MMA dots vs
   exp/softmax vs partials stores) using kernel self-time + issue
   counters. Write the bill down (ms per section).
2. For EACH candidate in §3, name the section it attacks and the
   section ms it can plausibly remove. If no candidate attacks a
   section holding ≥1ms with a concrete mechanism: STOP, report,
   do not write a kernel (same STOP discipline as task-6 §2.6).
3. Occupancy baseline FIRST: record grid (512 CTAs), registers/
   shared per CTA (from the compile log / `triton` annotations),
   achieved occupancy (Kineto), and wall. Every attempt re-reports
   all four; CTA-count drop without wall win = instant revert.

## 3. Ranked attacks (one at a time, A/B each, revert on red)

1. **Q-side conversion + dot narrowing.** `qwT/qfT` + per-iter
   `k_tile.to(fp16)`: test fp16 vs bf16 vs fp32-accumulate
   variants for the two MMA dots. Twin allclose (NOT equal:
   accumulation order may change — gate maxabs <1e-4 + KLD
   same-top 100% on box). Small, contained, no grid change.
2. **exp/softmax fast path.** `e = exp(sc - m_new)` per row per
   iter + combine's `num/den` div: `tl.exp` Zijlstra/fast-math
   flags, reciprocal-approx div. Twin allclose tight. Attacks
   the ALU-depth finding of id-74 directly.
3. **Partials precision.** `m/l` MUST stay fp32 (softmax state);
   `acc` fp32→fp16/bf16 halves partials traffic (8MB→4MB) with
   NO grid change (occupancy untouched — this is the safe shape
   flat-32 was not). Twin: allclose + KLD same-top mandatory
   (precision, not just order). If KLD moves: revert, no debate.
4. **CPG/occupancy re-tune (DANGER).** CPG 4→8 (fewer, fatter
   CTAs) or 4→2 (more, thinner): register/shared pressure moves
   opposite to parallelism. Only with the §2 baseline in hand;
   any wall regression = revert same day. Do NOT change `groups`
   (combine/merge stride invariant) — CPG only.
5. **Tail-dot elimination.** The second dot runs only on mixed
   tiles; measure its frequency @64k steady state (expect rare:
   tail window is small). If >2% of iters: route tail rows to a
   separate narrow kernel instead of the dual-dot branch (kills
   branch divergence in the common path). Needs the frequency
   number first — no number, no code.

Forbidden without maintainer sign-off: groups-count changes,
record-format changes, K4V2/width changes, prefill-path edits,
anything touching `dispatch.py` control flow.

## 4. Production rules (per attempt)

- Env gate `EXL3_KVARN_SERVE_V2=1` (default OFF until box-green;
  kill-switch restores legacy kernel, fail-closed loud fallback
  on throw — copy the `_kvarn_launch_combine` pattern).
- Twin in `tests/test_kvarn_triton.py` (CUDA-gated): exact-shape
  synthetic (non-pow2 groups + short-prefix mask rows, the two
  guards that break naive rewrites). Bit-exact where order is
  unchanged (`torch.equal`); allclose <1e-4 + box KLD where the
  attempt admits it changes order/precision.
- Occupancy proof in the commit message: CTAs before/after,
  wall before/after (hot runs), Kineto section bill delta.
- CPU suite green (skips without CUDA), atomic commit, ledger
  lines per §6 of the task-6 plan (same shape).

## 5. Non-goals + hands-off (same as task-6 plan §5/§5b, binding)

- Non-goals: prefill, sampler, MoE kernels, record format,
  `dispatch.py` control flow, `master`, rebase, force-push,
  `git config`, committing spikes/probes.
- Hands-off (read-only): `eval/_probe_geom.py`, `wiki/reports/*`,
  `doc/*.md` except ledger lines, workflow files, anything
  outside your attempt's scope. Spikes in new `eval/_spike9_*.py`
  only. `git status` + `git diff --stat` before every commit,
  explicit `git add <path>`, never `git add -A`; unknown paths =
  STOP + report. Recovery: `git stash -u`, `git checkout --
  <path>`; never `reset --hard`. Committed baseline is on
  `origin/wip/kvarn-cache` (pushed at plan time).

## 6. Box protocol (identical to task-6 plan §6 — follow it exactly)

Protocol v3 + env (`TRITON=1 IMAGELESS=1 PARITY=0/1`,
`expandable_segments`, box python/model paths); WITHOUT
`IMAGELESS=1` you measure the 5.3 legacy path. Sync via `cp` +
`unix2dos`; order 8k anchor, graph ×2 (hot counts), eager ×2
control-last, `-ntok 65536 -chunk 8192 -dec 256`; smi_guard
watchdog on every run. Gates: twin + KLD same-top 100% (mean
<1e-4) + PARITY=1 @8k green + hot ≥ baseline or revert.

## 7. Done means (per attempt)

- Hot tg@64k graph ≥ 49.5 (≈+3%) with occupancy proof, OR
- revert + ledger line recording the negative (negatives are
  data: flat-32 lives in the launcher comment for a reason).
- Stacking a second attempt on an uncommitted first is forbidden;
  each attempt lands (or reverts) independently.
