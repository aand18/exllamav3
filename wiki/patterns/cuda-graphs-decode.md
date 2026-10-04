# CUDA-graphs decode: replay sublattice, fallback everything else

## Problem
Decode was host-bound (~19ms Python vs ~13ms kernels/step): ~8
launches + ~40 dispatches + 2-3 syncs per layer × 16 layers. Whole-step
graphs are infeasible (data-dependent branches); per-op trims asymptote.

## Rule
- Graph the STEADY-STATE sublattice per layer (qwht -> serve ->
  combine -> tail_gather -> bmm -> tail_reduce -> merge) with STATIC
  buffers; leave store/seal/`n`/branches ungraphed. Replay on bucket
  hit `(gc, R)`; eager fallback on any trip (status 1/2, cert fail,
  flag-due, miss/bad bucket, shape change). Nothing fundamental
  blocks it; everything data-dependent stays eager.
- Static addresses are the whole game: persistent bufs for every
  input/temp/output (Q/n/tpos/bt copies per step are fine -- values
  flow, addresses don't), views precomputed per bucket, reallocs
  keyed and verified (`is`-check baked tensors on replay, drop +
  recapture on mismatch).
- PARITY validation and graphs compose: run the eref assert, THEN
  proceed (never decline on parity -- that silently validated eager
  only). Kill switch defaults to graphs ON only after green twice;
  fallback IS eager (same code path, zero behavior delta when off).
- Sticky tripwires need a replay-cadence read (post-replay due
  check), NOT the serve-tick probe (it gets consumed without
  firing). Clear all graphs on trip (structural surprise).
- Twin bar for replay: maxabs 0.0 (deterministic re-execution; any
  nonzero diff is a capture bug, never "numerics").

## Evidence
- Spikes v1/v2/wrap: capture ok, 2.2-3.1x on pair/sublattice,
  bit-exact (`8b4a6fb`, `eea9687`).
- Prod: tg 48.2 -> 57.8 @8k, 46.8 -> 54.0 @16k, 43.4 -> 47.0 @64k
  (clean, no parity). PARITY=1 never replays pre-fix (review catch:
  flag probe consumed the due tick) -- all "graph+parity" numbers
  before the fix were eager twice.
- Subagent review (`ses_f05ead45...`) caught: parity-decline,
  flag-never-read, realloc-UAF, post-store unwrapped throw,
  pending leak, st eviction, multi-batch skip. Every one was real;
  fresh-context review of hot-path diffs is now mandatory practice.

## Scope
Decode graphs. Buckets `(gc, R)` recapture every ~128 steps;
captures amortize to ~0. Review the fallback matrix on ANY change
to store/cert/flag/bucket logic.
