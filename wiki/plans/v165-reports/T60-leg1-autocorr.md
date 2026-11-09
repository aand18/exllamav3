# T60 Leg 1 — routing autocorrelation: cover 25 % against an 80 % pass mark, predicted prefetch is dead

Leg 1 of `wiki/plans/2026-11-09-prefetch-spec.md` (predictions pre-registered there). The T58
distinct counter (`enqueue_distinct`/`write_distinct` in `exllamav3/util/round_prof.py`, pumped
from `BlockSparseMLP_CPU.cpu_split_submit`) now also keeps each layer's previous sampled-round
distinct CPU expert set and, at the same single round-end sync, scores cover: of the current
round's distinct set, the share already present in the previous round's same-layer set. One
sampled-round sync total, unchanged (accumulate + single `torch.cuda.synchronize()` at
`round_end`, outside every phase bracket). No kernel change, no `.pyd` build, env is
`EXL3_ROUND_PROF` only. The `block_sparse_mlp_cpu.py` edit is a comment addition only: the T58
hook already enqueues everything the cover needs (layer key, buffer, `cpu_split_first`, rows,
topk, round index), so no behavior there changed.

## Run

`bench/windows/run_pass.ps1`, production regime (state.json: 3.18bpw_h6_ng8, split 381, threads
16, pin on), `-Runner spec_decode`, 131072 basis (spec_decode cache 40960 per state.json),
`EXL3_MOE_CPU_SLOTS=4`, `EXL3_ROUND_PROF=1`, `EXL3_ROUND_PROF_SAMPLE=8`,
`EXL3_ROUND_PROF_WARMUP=32`, `EXL3_ROUND_PROF_EVERY=32`, static placement (drafting run, automatic).
The instrument set deployed to the venv was the T58 six (round_prof, transformer,
block_sparse_mlp, block_sparse_mlp_cpu, generator, moe_lookahead); the venv's original
moe_cpu_host.py stayed, because the worktree's newer host passes 13 args to
`cexl3_moe_cpu_worker_run` (per-expert partial completion) against the installed ext's 12-arg
binding. The pass that proved that — `t60A-autocorr-131k-20261109-180959`, died at worker
startup with `TypeError: exl3_moe_cpu_worker_run(): incompatible function arguments` (13 invoked
vs 12 supported) — is excluded as evidence and disclosed here. Evidence pass:
`t60B-autocorr-131k-20261109-181208`, `status=ok`, elapsed 1011 s, FLOORS 2687 MB VRAM /
8212 MB RAM, breach 0. Draft arm 55.91 tok/s, 2.03/2.86 accepted (same-session slots-4
comparison marks: T58 54.78; tok/s is not a decision metric in this leg, and the instrument
adds one sync per sampled round).

## The schedule (honest line)

Covers were recorded at sampled-round ends: the sets are those of the current sampled round
versus the previous sampled round, i.e. sampled rounds 8 rounds apart (SAMPLE=8), the first
sampled round at index 40 (WARMUP=32), stepping by 8 through 20968. The distinct CSV holds
**2603 sampled rounds** (indices 40 48 56 … 20968), 124991 layer-round rows; the periodic
report lines print at every 32 sampled rounds (last printed phase line: n=2240, the tail
363 rounds are in the CSV only). Of the layer-rounds, rows=5 is the minority — 15888
rows=5 entries (rows histogram: rows=1 97824, rows=2 4656, rows=3 4032, rows=4 2544,
rows=5 15888, rows=12 47) — 14809 of them produced a cover (first-round-per-layer and
empty-current-set rounds skip). Worth stating, because it bounds what this cost: the
production `bsz`/rows=5 leg covers only the drafted verify window, not every decode round.

## Numbers (basis: fraction of the current rows=5 round's distinct CPU experts)

Final report line (cumulative over the pass, every-32-sampled-round print):

```
 -- round prof autocorr cover: median 0.250 p90 0.500 frac>=0.8 0.004 (n=14809 rows=5 layer-rounds)
```

Earlier prints of the same line (n=8832 at the first one, median 0.273 and p90 0.533;
final, 0.250 and 0.500) hold the same shape throughout, and frac>=0.8 stays 0.004 at every
print — the verdict does not depend on which print is quoted.

The distinct counts in the same CSV remain T58-like (median distinct 8 per layer-round, rows
budget as in T58's table), so the covers are computed over sets of the size this campaign has
seen all along.

## Verdict vs the pre-registered pass mark

Pre-registered: cover ≥ 80 % → Leg 2 viable; < 80 % → predicted prefetch dead, shared-wait
family closes (record as negative). Measured median cover **0.250**, p90 **0.500**,
frac≥0.8 **0.004** — that is, only about 1 layer-round in 250 clears 0.8 on the rows=5
window, and half the layer-rounds have at most a quarter of their CPU experts already routed
last sampled round. **Leg 1 misses the pass mark: the predicted-prefetch leg is dead, Leg 2
does not run.** Routing at rows=5 with topk=10 and ~9 distinct CPU experts per layer churns
far faster than the spec's ~3-row-advance stability assumption.

**DEAD.** This closes the shared-wait family on this evidence; no Leg 2 arm.

## Caveats (what could not be verified)

- **Lag.** SAMPLE=8 puts the comparison 8 rounds (~24 rows of drafted progress) apart, so
  this is a lower bound on the adjacent-round cover Leg 2 would have used. The miss is by
  ~3×, not by a margin a lag discount plausibly closes, and the spec's pass mark was written
  against this same schedule, so the verdict stands under its own terms.
- **First-round poison.** The first sampled round of an arm boundary has no predecessor;
  run_pass restores the frozen profile, and sampling starts after round 32, so no cross-arm
  set can enter.
- No nsys trace was taken (the brackets show the sleep, not its cause), and nothing here
  measures warp-level stall composition.

## Verification

- `tests/moe_cpu/test_distinct_counter.py`: 6 passed (3 existing + 3 new: `distinct_cpu_set`
  rules, per-layer previous-set update with rows=5-only covers, empty-previous-set scoring 0
  and empty-current-set skipping). Run in a throwaway CPU-torch venv (the WSL worktrees
  carry no torch; the Windows venv's `exllamav3` resolves to site-packages, not the worktree).
- `bench/verify_task.py --pre`: all gates passed. `--post --cites-failure
  t60A-autocorr-131k-20261109-180959`: all gates passed.
- Server: stopped for the pass, venv restored byte-identical to the production backup
  (5 files `cmp`-equal, `util/round_prof.py` and `modules/moe_lookahead.py` removed as
  absent-then), restarted via `start_tuned.ps1`, `/v1/models` lists the production model and
  `/v1/chat/completions` returned 16 tokens at 51.79 tok/s with live drafting (9/28 accepted).

