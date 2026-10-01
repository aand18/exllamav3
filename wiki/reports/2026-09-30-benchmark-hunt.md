# Subagent report: Benchmark hunt for KV-cache quality (2026-09-30)

## Provenance
RECOVERED from the opencode session store (`ses_f0d7d3f9fffeq9Y0o1WDmhQE04`, dispatched
2026-09-30 13:30 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
Survey: `longctx.py` runs 9 qualitative probes (2 summaries, FR-translate-back, zoomer-slang ID, 3 Variable-Man QA with inline reference answers, char-list, pony-fanfic intrusion; +2 Pride&Prejudice XL) — prints completions, no scorer. `qbench_prompts.py` is a KLD trace sampler, not a benchmark. `prompts/*.json` are TabbyAPI request captures (agentic_code = huge tool specs; coding/translate = trivial single prompts) — no ground truth, unjudgeable as-is. `mini/data.json` not on disk (fetched remote by `bbeh_mini.py`); that script is the harness template to copy: per-question fresh Generator + KVarN slot reset, think-strip, fuzzy-match judge, JSONL + CI.

1. GSM8K-style short-answer math (GSM8K/SVAMP subset, n=100–200). Measures multi-step carry over short ctx. Discrimination: attention/quant noise corrupts carried operands → exact final number flips while fp16 27B scores ~80%+. Judge: regex last number + exact match, greedy, `-mt 256`. Harness: copy `bbeh_mini.py`, swap judge. Work: 2–3h. Cost: ~200+256 tok/item ≈ 7s kvarn → ~15–20 min; cadence: per-cut.
2. Scored `longctx.py` on-disk fiction (Variable-Man q1–q3 + char-list + pony/intrusion + Illustrious summary). Measures retrieval/multi-hop over 10–20k ctx — the exact eviction failure mode, contamination-free (renamed entities). Discrimination: fp16 answers reference strings; dropped early tokens → hallucinated names/plot. Judge: case-insensitive keyword-substring sets (e.g. q1 requires `variabl*`+`unpredictable|statistical`; char-list F1 on names from `variable_man_char.txt`; intrusion requires pony-passage quote). Harness: copy `longctx.py:make_job` + `kvarn_needle.py` scorer. Work: 3–4h. Cost: 9 items × ~12k ctx (~9s) + 200 tok gen ≈ 3–4 min; cadence: per-cut gate.
3. Multi-needle/conflicting needle variants (extend `kvarn_needle.py`: 2–3 passcodes, update `"passcode changed to X"`, conjunction query). Measures retention + interference directly. Discrimination: fp16 trivial; seal/evict bugs drop one depth. Judge: exact substring per key, greedy `-mt 64`. Work: 1–2h. Cost: 6–8 × 12k ctx ≈ 1–2 min; cadence: per-cut/nightly.

NOT: RULER-lite/LongBench (100MB–GB downloads, frontier-hard subtasks ramble-to-cap like bbeh-mini, needs LLM judge); `prompts/*.json` as scored sets (open-ended creative/coding, no ground truth, hand-labeling cost, low cache-sensitivity); IFBench (eval unimplemented, needs external repo script; instruction-following isn't cache-sensitive); MMLU single-token (ctx too short to touch eviction; KLD already covers it).
