# Triton recompile contaminates timings (hot-cache rule)

## Problem
Triton hashes kernel SOURCE (comments included) + constexpr
specializations. ANY kernel source change, and ANY new constexpr
combination (new context length via nbpad/gc, new bits, new CPG),
recompiles inside the timed region on first launch: reads 15-25%
low (31.3 vs 38.3 same code; 28.4 -> 33.7 -> 34.2 across 64k runs;
16k first-run 31.8 vs hot 39.4).

## Rule (protocol v2, mandatory)
- Warmed box (clocks up; first run after idle reads ~20% low) +
  `expandable_segments:True` + fp16-decode anchor must read
  87.2-87.5 @8k (else box state suspect) + HOT triton cache: run
  TWICE per code version per context length, take the hot number +
  back-to-back A/B pairs + anti-bias ordering (control LAST, so any
  residual warm-bias favors it).
- A single timed run after ANY kernel edit (even comments) is VOID
  for comparison. All pre-v2 numbers carry cold-cache/box noise
  (this killed two false conclusions: the +4% MMA mirage and the
  K4V2 -20% gap).

## Evidence
- MMA cut `561588c` read +4% (37.8 -> 39.5); hot-cache A/B proved
  0.0% (39.4 -> 39.4); reverted `2a0b382`. Doc `1b63d93`,
  corrected `97269fe`.
- K4V2 gap 34.9 vs 43.6 evaporated under v2 (39.3 vs 39.4 tied).
- 64k ladder: 28.4 (cold) -> 33.7 -> 34.2 (hot).

## Scope
Every on-box perf claim. Gate template: twin (correctness) ->
suite -> KLD x2 (hot number) -> needle if paths touched.
