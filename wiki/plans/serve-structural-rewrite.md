# Structural serve rewrite plan (handoff) — NEEDS MAINTAINER GO FIRST

**Do not start this plan unless the maintainer explicitly says GO.**
Default verdict is NO-GO (recorded 2026-10-03 on task-7
verification): five measured negatives, reassociating math, +1-2%
realistic prize. This plan exists so a GO decision executes
cleanly, not as authorization.

Goal: eliminate the six per-slot `(16,)` metadata loads (K oth, V
sc, V zp × SL=2) = 43.4us/layer = 0.70ms/step = 27.9% of the
remaining 2.49ms serve cost — NOT by loading faster (five
mechanisms failed: `bcast` +4.4, `static8` +88.7/124 spills,
`prefetch` +17.4/10 spills, `num_stages=2` +29.1, `permeta` +1.8;
details task-7 plan §8.5), but by restructuring the math so the
loads do not exist.

Branch: `wip/kvarn-cache` (tip, do NOT rebase). Scope: serve
kernel dequant math ONLY, K4V4 decode, 4090 box. This is a new
kernel beside the old one, env-gated, never an in-place rewrite
(the old kernel is the fallback AND the twin reference).

## 0. The three parts (in dependency order, land separately)

**Part A — fold K sc/zp into `qwT` per group.** Per group (not per
row, not per tile): precompute `qwT * sc` and the zp correction
as a rank-1 term, so the inner loop never loads K sc/zp. Math:
`q·((qq*s+z)*o) = (q*s)·(qq*o) + (q*z)*o` — the second term is
rank-1 over rows (rank it, do not expand it). Reassociates:
allclose + KLD gates, never `torch.equal`.

**Part B — split the QK dot per slice.** Compute the dot per
128-slice and combine after, so `k_ot` (the cross-slice `oth`
correction) never materializes in the `(16, HD)` domain. Kills
the K-oth loads + the `_sl` where-expand loop. Reassociates
(reduction order across slices): allclose + KLD.

**Part C — V side as `dot(trans(qqv), v_sc*e)` + rank-1 `v_zp`
term.** Mirror of A/B for V: the value-weighted sum factors into
a transposed dot against scaled values plus a separate zeropoint
term. Reassociates: allclose + KLD.

Each part lands as its own commit (A, then B on A, then C) so any
part can revert independently. If A fails its gates, B and C do
not start.

## 1. Read this first (no code until done)

1. Task-7 plan §8.5 (the five negatives + the `permeta`
   load-bearing result: the prize is loads issue/latency, regs
   are NOT the lever — your rewrite must delete loads, not
   rearranges that keep them).
2. Serve kernel K/V dequant sections (`kvarn_triton.py:~1620-1740`:
   nibble paths, `uniform` branch, `k_ot`/`v_sc`/`v_zp`
   where-expands — the exact code being replaced).
3. Combine-split precedent (`_kvarn_launch_combine` + twin
   `test_combine_split_matches_folded`): env-gate + fallback +
   twin shape to copy (but your twin is allclose, not equal).
4. Memory id 74 (Sept v3-hoisting negative: ALU-depth context).

## 2. Method rules (violations = revert, no debate)

1. **Spike each part FIRST** (`eval/_spike12_rewriteA.py` etc.,
   never commit): synthetic inputs vs legacy kernel, report
   (maxabs, meanabs, regs, spills, us/layer) vs base 155.9.
   Bar per part: maxabs <1e-4 AND regs ≤254 AND faster. Any part
   missing any leg: STOP that part, record, do not stack on it.
2. **Precision budget ledger.** Open a table in the plan file §8:
   part | maxabs | KLD mean/max drift vs base | regs. KLD gate
   per part: same-top 100% AND mean drift <2e-5 AND max drift
   <2e-4 @64k (tighter than the project 1e-4/absolute gate
   because drift COMPOUNDS across parts). Any part exceeding
   its budget: revert that part, keep the others.
3. **Register ceiling is informational only** (permeta proved
   regs ≠ lever here): record regs/spills per variant like the
   task-7 bill, but gate on WALL time (interleaved, one
   process — the ±25% between-process spread rule from task-7
   §8.4 applies to every number quoted).
4. **Never break the old kernel.** It stays default until the
   new kernel passes §4 gates; gate `EXL3_KVARN_SERVE_WHT2=1`
   (default OFF), fail-closed loud fallback, kill-switch =
   unset the var.

## 3. Twin + validation (per part, all must hold)

- CUDA twin vs legacy kernel on production shapes (non-pow2
  groups, short-prefix masks, sink+tail, CPG=8): `allclose`
  rtol=1e-4 atol=1e-5 + printed maxabs (NOT `torch.equal` —
  reassociation forbids it; anyone writing `assert torch.equal`
  here has misunderstood the task).
- Box: microkld KLD @64k (budget §2.2) + needle spot-check
  (`kvarn_needle.py`, same-top) + PARITY=1 @8k asserts green.
- tg@64k graph pair, interleaved gate-flipped if spread bites;
  quote hot runs only.
- CPU suite green (new twins skip without CUDA).

## 4. Non-goals + guards (binding, cf. task-6 plan §5/§5b)

No record-format, grid, groups-count, dispatch-flow, prefill,
sampler, or MoE changes. Hands-off everything outside the part;
`git status` + `git diff --stat` before every commit, explicit
per-file `git add`. Spikes untracked, atomic per-part commits +
ledger lines.

## 5. Done means (per part)

Hot tg@64k graph improves by the spiked amount (±30%) with
precision budget unspent, twin + KLD + PARITY + needle green,
ledger entry, pushed — or revert + negative recorded. Full-plan
done = all three parts landed ≈ +1-2% wall with total KLD drift
inside §2.2. If Part A fails: whole plan returns to NO-GO, do
not attempt B/C standalone (they depend on A's load deletion
for their register/issue budget).
