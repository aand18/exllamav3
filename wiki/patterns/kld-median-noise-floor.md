# Gate on same-top + digits, not on print-level medians

## Problem
KLD medians at 1e-6 print as `0.000001` vs `0.000002`: a ~15%
relative wobble flips the last printed digit while mean/max stay
identical to 3 sig figs and same-top stays 100%. Chasing the digit
wastes runs; the parity asserts (4096/layer/run, incremental Ew vs
full refresh) already prove per-step cache exactness, and twins
prove bit-identity of records/state.

## Rule
- Ship/no-ship criteria: same-top 100% AND mean/max within
  historical band AND parity asserts green AND twins green.
- Median print-digit flips at <=2e-6 with everything else green =
  harness noise. Record and move on; revisit only if same-top,
  mean, or max move.
- When a real median shift is suspected, the decisive test is the
  parity assert + twin, not more KLD repeats.

## Evidence
- 2026-09-30: fast-path KLD median 1e-6 -> 2e-6 with identical
  mean/max/same-top; proven external (fast and legacy run the same
  batched seal kernel on the same values; parity green). Doc
  `f96114b`.

## Scope
KLD gate interpretation, all presets.
