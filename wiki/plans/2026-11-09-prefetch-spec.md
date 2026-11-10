# Spec: predictive-routing prefetch (lookahead re-run, T60)

Idea: at the start of a drafted round, pre-issue each layer's CPU expert weight pages
using the *previous* round's routing for that layer (the drafted window advances ~3 rows
per round, so routing per layer is expected to be strongly autocorrelated). The CPU then
has the GPU's foreground of all intervening layers to produce before the slot is due.
This is the per-round shared-wait family tried on the cheapest falsifiable member:
lookahead #1 was refuted on cost (+92 % DRAM for 49.3 % coverage, before the pinned arena
shipped) — the 24 GiB pinned arena (production since T29/T33) is the piece that assumption
predates, so the arm is re-measurement, not a new mechanism.

## Leg 1 — routing autocorrelation (no kernel change, no build)

Extend the T58 distinct counter to also record, per layer, the previous sampled round's
expert set and the current round's, and print the intersection cover (how many of this
round's distinct CPU experts were in the previous round's set).

Pre-registered:

- Expected: cover ≥ 80 % on the rows=5 window (routing near-stable across a 3-row advance).
- Pass mark: if cover ≥ 80 %, run Leg 2; if < 80 %, predicted prefetch is dead and the
  shared-wait family closes (record as negative).
- Miss threshold: treat < 80 % as a miss — a sub-80 % route wastes more page-in bandwidth
  than the foreground buys.

## Leg 2 — prefetch arm (only if leg 1 passes)

Earlier me in the night's LoRA budget said prior research showed lookahead+pin cost: measure
it again now pinned-arena is live, as a `bench/windows/lookahead` task, env-dispatched
(`EXL3_MOE_LOOKAHEAD` already exists), same-session ABBA at 131072 against slots-4
reference 54.78/55.80.

Expected from the laptop plan: prefetch cover × 0.25 ms/layer? No: predictions are per leg,
mechanical model: prefetching only hides *weight-load* time that wasn't already hidden by
`cpu_split_submit`'s own pipeline; it cannot hide GEMV time. Write both predictions before
the build and record both.

Pass marks for Leg 2 (pre-registered):

- decode draft ≥ 58 tok/s vs the same-session reference arm (T58 pass mark retained as the
  bar for any handoff experiment), cmb median down ≥ 0.10 ms/layer, and waste (pages
  prefetched but unused per round) ≤ 20 %.
- Fail = any of those missed. Do not extend T60 into grouped kernels or graphs on a miss.

## Rules carried from T58/T59

- `--post` gate before quoting any number. One report per task.
- Run ids read from the CSVs, never typed. Basis stated in every sentence (skip if
  identical to T58's 131072 spec_decode static/pin basis).
- `bench/verify_task.py --pre` before the first pass; `-DryRun` checks a spelling while the
  box is busy. Pass only deltas against state.json.
