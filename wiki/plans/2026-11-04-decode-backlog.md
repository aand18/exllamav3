# Decode backlog — the six architectural levers (queued 2026-11-04)

Queue entry for the Argon proposal `msg_1a7761647001FfsWQYm66exgdj` (openchamber
`ses_e58a52f10ffevEFMplvuOoLxc4`). It is **not** a plan: it is a triaged backlog, because several of
its items rest on claims this box can falsify or has already contradicted, and one item changes the
served distribution. Each row names the claim, the cheapest disproof, and what it would cost.

**Precondition for everything below:** the plan of 2026-11-04 (`2026-11-04-decode-to-strata.md`,
Tasks 1–4 merged, Task 5 pending) ends in a measurement session whose first step is the box
verification of the Task 4 weight restore. Nothing in this backlog starts before that session, because
every item is judged on the same agentic-weighted decode metric and the same box state.

## Triage

| # | lever | the load-bearing claim | cheapest falsification | verdict |
|---|---|---|---|---|
| 1 | pre-attention router lookahead prefetch | pre- and post-attention residuals have >0.92 cosine, and the predicted top-10 **router picks** match the true ones often enough to prefetch | capture pre/post residual per layer and the top-10 overlap for 20 decode steps (~10 min, one pass) | **gated on a measurement**; prefetch is correctness-neutral, so a poor overlap costs only wasted bandwidth |
| 2 | 3D V-Cache as a "Tier-1.5" | cold-expert streams flush the 96 MB V-Cache, so a hot pool could live there | count L3 hit rate with the current load pattern (perf counters or a microbenchmark of the gather) | **hardware verified, mechanism half-impossible**: `Win32_Processor` says AMD Ryzen 9 7950X3D, 16C/32T, L3 **131 072 KB** (96 + 32 MB across the two CCDs), so the premise about the silicon holds. But L3 is not software-addressable — "pin 45 experts in V-Cache" cannot be implemented; the achievable form is non-temporal loads for cold rows plus a hot pool bounded to ~86 MB |
| 3 | cold-tail 2.0 bpw via `-or` | a 3.18 → 2.0 bpw re-quantization of the bottom ~300 experts/layer costs <0.03-weight perplexity but 37 % fewer bytes | quantize one layer's tail and measure both perplexity and gather time | **highest expected value, highest cost**; needs a quantizer run, an `-or` variant tree, and its own plan |
| 4 | draft-row CPU-miss pruning | dropping tail picks on draft rows is a "standard capacity-aware policy" with <0.01 perplexity cost | it changes the served distribution **and** depends on placement, so two runs with different placement serve different models | **hold** — a product/correctness decision, not an optimization; needs an explicit decision that the served policy may depend on placement |
| 5 | 512 MiB hot-trigram table + 4-bit PLE | top ~3M of 20M rows serve >88 % of lookups; today every decode row is an unbuffered NVMe read | extends the row cache already merged (`e43ba35d`): measure its hit rate with a real corpus first | **next after the measurement session** — cheapest real win, no math change |
| 6 | tree verification for DFlash2 | a 6–7 token tree from one non-causal drafter pass, filtered by router overlap, yields +25–35 % accepted tokens for near-zero bytes | needs tree-attention masks and multi-candidate verification in the generator; nothing exists to measure yet | **own plan**; the engine has no tree path to extend |

## Order, and why

1. **Finish plan 1's measurement session** (weight-restore verification, Task 5's KV→experts arm, then
   the arms for Tasks 1–4). Every lever below is judged against the number that session produces.
2. **#5** — it extends code that already exists and shipped (`EXL3_NGRAM_ROW_CACHE_ROWS`), needs no math
   change, and its claim is measurable from the cache's own counters.
3. **#3** — the largest mechanical lever on the CPU-miss cost (37 % fewer bytes per miss) and it also
   shrinks the arena enough to make #2's non-temporal streaming and the plan's streaming gates viable.
4. **#1** — only after its overlap measurement; a prefetch that predicts 40 % of misses still helps, one
   that predicts 5 % is pure bandwidth waste.
5. **#2's achievable half** — non-temporal loads for cold rows and a cache-sized hot pool; skip the
   "pin data in L3" framing, which is not implementable.
6. **#6** — a separate plan with its own drafter work.
7. **#4** — parked unless the served-distribution question is answered explicitly.

## What this backlog does not accept

- The projected throughput table (57 → 95–105 → 145–170 tok/s). Those are estimates stacked on
  estimates: plan 1's own levers are unmeasured, #1's premise is unmeasured, and #3's perplexity claim
  is unmeasured. No row above may be quoted as a projection; each is quoted only after its own arm.
- "Zero change to model outputs" for #1 — true only because a prefetch hint cannot change the router's
  answer; it is not a property of the prediction, and it must not be used to justify #4.
- "0 % L3 hit rate" for #2 — asserted, not measured on this box.
