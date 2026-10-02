# Skill / process impact tracker (WikiSkill-lite)

Append one entry per accepted OR rejected workflow/process change:
proposal, target, validation, outcome. The wiki (`patterns/`,
`logs.md` below) is never rolled back; skill/process state is.
This file is the audit trail the maintainer pass consults so
rejected interventions are not proposed again.

Convention: update on every accept/reject, same commit if the
change ships, standalone docs commit if it doesn't.

## Accepted

- 2026-10-02 / decode graphs v2 (per-layer sublattice replay):
  tg 48.2 -> 57.8 @8k, 46.8 -> 54.0 @16k, 43.4 -> 47.0 @64k
  (clean). Gates: graph twin (maxabs 0.0), suite, KLD x5
  (parity asserts green while replaying), needle 6/6, fresh-eyes
  subagent review (caught parity-decline, flag-never-read,
  realloc-UAF, unwrapped post-store throw -- all fixed).
  Default flipped ON (kill switch kept).
- 2026-10-02 / PARITY TAX discovery: all historical tg/pp ran
  PARITY=1 (~15% hidden tax). Protocol v3: clean perf + separate
  parity validation; never compare across parity settings.

- 2026-09-30 / protocol-v2 (warmed + anchor + hot-cache + A/B):
  killed two false conclusions (MMA +4% mirage, K4V2 -20% gap).
  Validated by: re-measurement under v2. Status: mandatory.
- 2026-09-30 / host-desync cut (flag periodic, tpos memo,
  long-guards): tg 39.4 -> 40.2. Gates: twins + suite + KLD x2.
  Commit `bf91130`.
- 2026-09-30 / seal-direct fast path + split (31+1): pp 5.2 -> 4.0.
  Gates: twin (aligned+split+reset) + suite + KLD x2 + needle.
  Commits `476eb94`, `3ce676f`.
- 2026-09-30 / hierarchical serve+combine cap-128: tg@64k
  34.0 -> 35.1. Gates: grouped twin + suite + KLD x3 + needle.
  Commit `3647f54`.
- 2026-09-30 / needle 6/6 gate (multi+update, `-mt` 256):
  fp16 6/6, kvarn4 6/6. Commits `6905d6e`, `9cbbe6d`.
- 2026-09-30 / PTIMES + decline counters (undistorted attribution):
  found the real 4032+64 box pattern. Commit `6030603`.
- 2026-09-30 / graphs spike v1 (serve+combine capture ok, 2.2x,
  exact). Commit `8b4a6fb`.

## Rejected (with reason -- do not re-propose without new evidence)

- 2026-09-30 / MMA floor (`561588c`, reverted `2a0b382`):
  hot-cache A/B 0.0% + doubled serve-acc VRAM. Reason: dots aren't
  the bottleneck; serve is parallelism-bound.
- 2026-09-30 / K4V2 as vehicle: pre-protocol -20% was noise; tied
  39-40 tok/s under v2; KLD 270x worse. Parked (VRAM value only).
  User scope 2026-10-01: K4V4 ONLY, no new K4V2 runs.
- 2026-09-30 / fused tail-QK: would replace MMA bmm with SIMT
  dots (net loss). Rejected at design review, no code.
- 2026-09-30 / torch.compile seal core: inductor inexact
  (maxdiff 16.0) + Dynamo failure on K4V2 + tiny prize (+17% on
  ~50ms). Probe `54b1fee`.
- 2026-09-30 / hierarchical flat-32: -4% hot (128 CTAs starve
  144 SMs). Superseded by cap-128. Lesson kept, code gone.
- 2026-09-30 / TIRx-Harness adoption: wrong fight (lowering
  predictability not our bottleneck; dedicated GPU; bespoke
  kernels). Doc `649269d`.
- 2026-10-01 / bbeh_mini-first-N mining: fp16 0/60 (all
  ramble-to-cap) -- pond barren, not a cache signal. Doc `d236a1f`.

## Maintainer pass log

- 2026-10-01: initial consolidation. Mined 6 pattern pages from
  `doc/kvarn-4090.md` + session history; seeded this tracker from
  git log. Next pass: after graphs-v2 or next perf cut lands.
