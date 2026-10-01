# Benchmarks must be solvable by the baseline first

## Problem
A gate the fp16 baseline fails is not a quality gate, it's a
capability test: 0/3 on both tells you nothing about the cache
(floor effect, zero bits of information). Same for flaky harnesses
(temperature 0.8, no seed) and caps that cut answers mid-reasoning.

## Rule
- Mine fp16-first: run the baseline before any kvarn run. If fp16
  doesn't solve it, the item is disqualified (don't extend blindly;
  change ponds: stratify by task, never more first-N).
- Force determinism: greedy (`-temp 0 -topp 1`), log flags with the
  lock file, re-run single-item flips 2x before calling regression.
- Size generation caps from observed needs: single-needle EOS at
  ~30-90 tok but multi-needle needs ~170 (preamble + codes); a cap
  that truncates is a gate bug, not a model failure
  (`eval/kvarn_needle.py -mt` 64 -> 256).
- Power: to notice a 1-in-20 break rate needs 20-30 items minimum
  (64%/79% detection), 40-60 for a strong gate.
- Cadence: KLD same-top + digits stays the per-cut gate (seconds,
  deterministic). Reasoning/retention sets run on store/evict/seal/
  serve cuts, else nightly/weekly -- never per-cut.

## Evidence
- bbeh_mini 0/3 both (frontier-hard, decides nothing); fp16 mining
  0/60 on mini-first-60 (all ramble-to-cap) -> pond change recorded,
  no kvarn GPU wasted (`d236a1f`).
- Needle 6/6 (fp16 39s, kvarn4 109s): the discriminative gate that
  works (baseline solves it). `-mt` fix in `6905d6e`/`9cbbe6d`.

## Scope
All quality evaluation. Mining protocol: fp16-only first pass,
lock solvable + clean-EOS items with flags, gate on match to
locked target (never on distribution thresholds under sampling).
