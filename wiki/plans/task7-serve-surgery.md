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

## 8. RESULTS (2026-10-03, branch `wip/kvarn-cache`, commits `3ccc7e8`
##    `fc24102` `4ab5caf` `ad7612a`) — two cuts landed, +9.5% tg@64k

`EXL3_KVARN_SERVE_V2=1` (default) + serve-groups cap 64 (default).
tg@64k graph **47.9 → 52.2 tok/s (+9.0%)**. Occupancy never dropped:
CTAs and CTA/SM are reported per attempt below.

### 8.1 Phase 0 PASSED the STOP gate — the bill inverted the ranking

`eval/_spike9_serve_bill.py`: 18 text-patched variants of the serve
kernel, one ablation per section, 7 interleaved windows in ONE process
(interleaving matters — see §8.4), CUDA-event timed, `n_regs` /
`n_spills` / smem per variant so a work-removal delta can be told
apart from an occupancy delta. Isolated serve = 222.8us/layer
(Kineto in-situ 279.7us/call, 4.474ms/step over 16 layers).

| section | us/layer | ms/step | % of serve | regs |
|---|---|---|---|---|
| metadata loads (K sc/zp, V oth, block table) | 120.4 | 1.93 | **54.0** | 219→239 |
| K+V payload reads | 84.3 | 1.35 | 37.8 | 219→200 |
| dequant ALU (per-element scale math) | 55.4 | 0.89 | 24.8 | 219→207 |
| exact-direct tail (ek/ev gather + selects) | 29.9 | 0.48 | 13.4 | 219→205 |
| partials stores | 7.2 | 0.12 | 3.2 | 219→209 |
| `exp` | 13.2 | 0.21 | 5.9 | 219→213 |
| one extra MMA dot | −3.3 | −0.05 | **−1.5** | 219→219 |

Occupancy baseline: grid 512 CTAs = 4 waves, **219 regs/thread, 0
spills, 16KB smem → 2 CTA/SM = 8 warps/SM = 17% of 48**.

Consequences for §3, all measured:
- **§3.1 (dot narrowing) and §3.5 (tail-dot) are dead**: one extra dot
  is *negative* (the tensor cores are free; `nbody == 16` skips the
  second dot already). Do not re-propose.
- **§3.2 (exp fast path)** ceiling is 0.21ms/step even if `exp` became
  free — under the +3% bar. Rejected.
- **§3.3 (partials fp16)** ceiling is 0.12ms/step (the stores are 3.2%,
  and half of that is 0.06ms). Rejected.
- **§3.4 (CPG) was the only ranked item with a real prize**, and it is
  bigger than §3.1-3.3 combined.
- The **unranked** item the plan did not list — per-tile metadata
  *hoisting* — is 54% of the kernel. §1.4 (memory id 74, "do not
  re-propose pure metadata-hoisting without new evidence") is a
  prefill-path negative; the new evidence is this bill, so it was
  worth exactly one attempt, taken bit-exactly.

### 8.2 Attempt 1 — per-group metadata hoist (BIT-EXACT, +6.5%)

A TOK=16 tile can never straddle a 128-row group, so `g`, `s` and the
three per-channel metadata vectors are loop-invariant across the 8
tiles of a group. The tile loop became a (group, tile) nest: the
block-table gather and the 3 metadata vectors load once per group
(CPG times) instead of once per tile (8*CPG times). Groups past `n`
clamp their page/group index — a no-op for live rows, and the only
rows that read through the clamp are already masked by `r`.

- **222.8 → 162.7us/layer (−27%)**; regs 219 → **254**, spills 0, smem
  unchanged, CTA/SM **2 → 2** (254*128 = 32512 ≤ 32768, so it just
  fits; 255 would drop it to 1 CTA/SM — watch this if the body grows).
- Twin `test_serve_v2_group_hoist_bit_exact`: 7 shapes (CPG=4
  production, non-pow2 gc, short-prefix mask rows, sink+tail, CPG=1),
  `torch.equal` on out AND on the written m/l/acc rows. This caught a
  real bug of mine: groups past `n` indexed past the block table until
  the clamps went in.
- Perf: graph 47.9/47.9/48.0 → 51.0/50.8/51.0 tok/s; eager in-process
  alternating 21.254 → 19.970 ms/step (−6.0%). KLD identical to 6
  digits across arms. Flipped default ON in `4ab5caf`.

### 8.3 Attempt 2 — serve-groups cap 128 → 64 (cpg 4 → 8, +2.8%)

At 2 CTA/SM the cap trade is decided by wave quantization, not
partials bytes: 512 CTAs = 4 exact waves → 256 CTAs = 2 waves, and
partials traffic *halves* (4MB → 2MB per layer) as a side effect.
cap 256 and cap 512 both measure slower, so the old cap was past the
optimum the other way.

- **163.7 → 154.2us/layer (−5.8%)**; CTAs 512 → 256, regs **254
  unchanged**, spills 0, smem unchanged, CTA/SM 2 → 2.
- 64k graph 50.8 → 52.2 tok/s (+2.8%, 4 interleaved rounds/arm), 16k
  56.5 → 58.3 (+3.2%), 32k 38.4 → 39.1 (+1.8%). KLD same-top 100.00%,
  mean 2.8e-5 @64k, identical to the cap128 arms.

### 8.4 Methodology finding — this box cannot be A/B'd across processes

Identical code, identical command, different process: 39.3 and 50.8
tok/s for the same v2 graph config; 35.1 and 56.3 for the same 8k
config; one cap64 run reported 72.7. `expandable_segments` is
unsupported on this platform (torch warns at every run), so each
process lays the 64k kvarn cache out differently in DRAM. **Every
number in this task that is quoted as a delta comes from a
single-process, gate-flipped, interleaved-window measurement** (the
bill harness and `eval/_spike9_ab.py`); the cross-process runs only
bound the spread. Both arms also show one ~25ms outlier window at the
same step count (w3 of 6) — a periodic event, pre-existing, and the
only run that ever crashed did so at exactly that window.

### 8.5 Next lever, and one negative to not re-propose

Post-v2 bill (base 163.5us/layer): payload 35.8% (record format, off
limits), per-slot metadata 31.0%, tail 13.5%, dequant ALU 12.0%,
partials 5.2%, exp 4.4%, MMA 2.3%.

**Negative**: replacing the three per-slot metadata expansions (K oth,
V sc, V zp) — SL `tl.where` selects each — with a `(16,2,128)`
broadcast of a `tl.join`ed pair is NOT faster (165.3 vs 163.7us/layer,
regs 254 → 243). Same tensor, same values. The 31% that removing those
loads exposes is mostly the *register* drop they enable (254 → 168, so
2 → 3 CTA/SM), not the select ALU.

**So the live prize is occupancy, not traffic**: 3 CTA/SM needs
≤170 regs/thread and the kernel is at 254. The concrete route is
folding K sc/zp into `qwT` once per group (a rank-1 correction term
for the zp half) and splitting the QK dot per slice so `k_ot` never
materialises in the (16, HD) domain — both reassociate, so they need
allclose + KLD gates, not `torch.equal`, and they are a bigger change
than anything attempted here. Ranked second: `nodeq` (12.0%, the
per-element scale math) rides along with the same fold.

### 8.6 Collateral finding

`test_promoted_serve_matches_eval_spike` fails on the box (4.5e-08 in
one q-row) with the gate forced to legacy AND to v2 — pre-existing
drift between the promoted kernel and the eval spike7 original it was
copied from, not any cut in this task. Four more twins fail when the
run inherits `EXL3_KVARN_IMAGELESS=1` / `EXL3_KVARN_GRAPH=1` (they
test the store/image paths and do not set those themselves); identical
at every serve-groups cap, so also pre-existing. Run the twin suite
WITHOUT the protocol env. Both want a separate cleanup.

Spike artifacts (never committed): `eval/_spike9_serve_bill.py`,
`eval/_spike9_ab.py`, `eval/_spike9_dbg.py`, `eval/_spike9_dbg2.py`,
`eval/_spike9*.bat`, mirror logs `spike9_bill*.log`, `t7_*.log`.
