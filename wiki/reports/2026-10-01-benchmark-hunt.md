# Subagent report: benchmark hunt for KV-cache quality (2026-10-01)

## Provenance (read first)

TRANS-REPORT: this file was transcribed from the subagent's chat
output AFTER the fact (the reports convention did not exist at
dispatch). The pasted output was TRUNCATED mid-sentence
(recommendation 1's detail is partial). Status per section:
- Survey paragraph (longctx.py / qbench_prompts.py / prompts/*.json
  / mini/data.json): UNVERIFIED except file existence (eval/*.py and
  eval/prompts/ listings confirmed; contents not re-read).
- Recommendations 2+3 and rejections: LOST with the truncation
  (only the coordinator's compressed summary survives, in chat).
- Independently verified by coordinator afterwards: needle harness
  structure, bbeh_mini CLI flags, model_init sampling args.

Do not treat truncated/unverified parts as established fact.

## Original report (verbatim, truncated)

Survey: longctx.py runs 9 qualitative probes (2 summaries,
FR-translate-back, zoomer-slang ID, 3 Variable-Man QA with inline
reference answers, char-list, pony-fanfic intrusion; +2
Pride&Prejudice XL) — prints completions, no scorer.
qbench_prompts.py is a KLD trace sampler, not a benchmark.
prompts/*.json are TabbyAPI request captures (agentic_code = huge
tool specs; coding/translate = trivial single prompts) — no ground
truth, unjudgeable as-is. mini/data.json not on disk (fetched remote
by bbeh_mini.py); that script is the harness template to copy:
per-question fresh Generator + KVarN slot reset, think-strip,
fuzzy-match judge, JSONL + CI.

GSM8K-style short-answer math (GSM8K/SVAMP subset, n=100–200).
Measures multi-step carry over short ctx. Discriminati...
[TRUNCATED HERE -- remainder lost]

## Coordinator's compressed outcome (what was actually used)

Recommendation adopted: extend kvarn_needle.py (proven gate) with
multi-conjunction + recency-update items, NOT a new GSM8K harness.
Result: needle 6/6 on fp16 (39s) and kvarn4 (109s); `-mt` 64 -> 256
headroom fix. Commits `6905d6e`, `9cbbe6d`. GSM8K/desiderata
(recommendations 2+3) remain unbuilt; rebuild from this file's
survey + fresh reads, not from memory of the lost text.
