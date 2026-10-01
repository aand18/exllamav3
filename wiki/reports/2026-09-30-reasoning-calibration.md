# Subagent report: Reasoning calibration design (2026-09-30)

## Provenance
RECOVERED from the opencode session store (`ses_f0dacbacaffepKXuwE7ZTjak3q`, dispatched
2026-09-30 12:38 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
**1. Mining procedure (fp16-first, then lock)**

`bbeh_mini.py` runs one fresh `Generator` per item (`-fresh` frees/reallocs KVarN slots), `max_rq_tokens 512`, `stop_on_loop (300,3)`, judged by `evaluate_correctness` (answer-after-`The answer is:` + fuzzy match). `-limit` slices `bbeh[:limit]` *before* the seeded (`123`) shuffle — so ids are unstable across limits. Note: `mini/data.json` has 460 items with only `input/target`, no task labels; no subset prioritization is possible inside mini. For stratification use full BBEH task splits; inside mini scan sequentially.

Run mining on fp16 only: `--limit 60 -mt 2048 -fresh` with greedy override (`--temperature 0 --top_p 1`, cf. §4) and `-o fp16_mine.jsonl`. Use `-mt 2048` per the doc note, not the 16384 default: observed behavior is ramble-to-cap (16k), and at ~40 tok/s kvarn / ~80 tok/s fp16 a 16k ramble costs ~200–400s/item. At 2048 caps, cost is ~25–30s/item fp16 (prefill few s + gen) → 60 items ≈ 30–35 min + load, one serial-4090 evening. Keep items where fp16 `correct=true` with `eos_reason` clean (no `max_new_tokens`/`loop_detected`) and `gen_tokens < 2048`. Aim to lock 20–30 (minimum) from the first 60; extend to 120 if yield <30%.

Lock-in format (jsonl, mirroring `kvarn_needle.py -o` + torture's differential style): `{orig_index, input_sha256, input, target, fp16_answer, fp16_eos, gen_tokens}` plus the exact harness flags. Gate on `evaluate_correctness(kvarn_answer, locked_target)` — exact/fuzzy match, **not** KLD thresholds or fp16-output match (sampling noise makes distribution thresholds meaningless here).

**2. Discriminative power**

0/3-all-fail is a floor effect: with P(fp16 solve)=0, P(kvarn solve)=0 regardless of cache quality — zero bits of information (the "0-baseline gate decides nothing"). Want the band where fp16 solves ~50–100% (30–80% ideal: headroom in both directions; needle's 4/4 is the working example of a solving baseline). Power: if kvarn breaks each item w.p. p=1/20, P(≥1 detection)=1−(1−p)^n. n=20 → 64%, n=30 → 79%, n=45 → 90%, n=60 → 95%. So 20–30 items is the smallest useful gate; 40–60 for a strong one.

**3. Cost model + cadence**

Needle baseline: 4 short-gen items = 25s fp16 / 36s q4 / 44s kvarn4 (~6–11s/item). Reasoning at `-mt 2048`: ~30s/item fp16, ~50s/item kvarn4 → 25-item set ≈ 13 min fp16, ~20 min kvarn4. Recommend: **not per-cut** (KLD same-top 100% + digits stays the per-cut gate — seconds, deterministic). Run the reasoning set on store/evict/seal/serve cuts only, otherwise nightly/weekly; per-cut keep only `-mt 2048 --limit 3` as a stability smoke (no-crash/no-hang, not quality).

**4. Failure modes + flake control**

`wrong answer` (clean EOS, fp16 solved) → quality drift/cache bug, bisect by KLD-8k; `ramble-to-cap/loop_detected` → attention degradation (lost instruction-following), suspect eviction/seal; `crash/hang` or smoke-item-0 fail → model path/harness, not a cache verdict (needle's abort rule) — rerun fp16 before triaging. Flake: **harness is not deterministic** — `bbeh_mini.py:292` defaults `temperature 0.8, top_p 0.8`, no sampler seed (needle forces `temperature 0.0`). Fix: always pass greedy flags, log them in the lock file, and re-run any single-item flip 2× before calling it a regression.
