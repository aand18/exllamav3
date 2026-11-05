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
| 2 | 3D V-Cache as a "Tier-1.5" | cold-expert streams flush the 96 MB V-Cache, so a hot pool could live there | count L3 hit rate with the current load pattern (perf counters or a microbenchmark of the gather) | **CLOSED BY MEASUREMENT (t44-prof-phases)**: the engine's own phase profile shows the expert GEMVs at **47.4 and 45.8 GB/s against a 48 GB/s DDR5 peak** — the cold reads already saturate memory bandwidth, so there is no cache-miss headroom for streaming hints to recover. Was: **hardware verified, mechanism half-impossible**: `Win32_Processor` says AMD Ryzen 9 7950X3D, 16C/32T, L3 **131 072 KB** (96 + 32 MB across the two CCDs), so the premise about the silicon holds. But L3 is not software-addressable — "pin 45 experts in V-Cache" cannot be implemented; the achievable form is non-temporal loads for cold rows plus a hot pool bounded to ~86 MB |
| 3 | cold-tail 2.0 bpw via `-or` | a 3.18 → 2.0 bpw re-quantization of the bottom ~300 experts/layer costs <0.03-weight perplexity but 37 % fewer bytes | quantize one layer's tail and measure both perplexity and gather time | **highest expected value, highest cost**; needs a quantizer run, an `-or` variant tree, and its own plan. The payoff is now computed two ways that agree: `size_check.py --tail-bpw 2.0` puts the arena at **33 375 → 20 990 MiB** (flat linear bound), and `bench/tail_variant_plan.py --mcs 381 --tail-bpw 2.0`, which prices every tensor and leaves the EXL3 scales unscaled, says the tail is **33 375 → 21 115 MiB (−36.7 %)** over 48 layers × 381 experts by the checkpoint's mean 3.18 bpw — and **21 766 MiB (−34.8 %)** tile-exact, because 83 % of the tail is stored at 3.0 bpw, where dropping to 2.0 saves a third rather than 37 %; the mean-rate number is the optimistic bound — the 32.6 → 21 GiB claim holds, and the plan file names every tensor a quantizer must rewrite |
| 4 | draft-row CPU-miss pruning | dropping tail picks on draft rows is a "standard capacity-aware policy" with <0.01 perplexity cost | it changes the served distribution **and** depends on placement, so two runs with different placement serve different models | **hold** — a product/correctness decision, not an optimization; needs an explicit decision that the served policy may depend on placement |
| 5 | 512 MiB hot-trigram table + 4-bit PLE | top ~3M of 20M rows serve >88 % of lookups; today every decode row is an unbuffered NVMe read | extends the row cache already merged (`e43ba35d`): measure its hit rate with a real corpus first | **mechanism merged, premise corrected**: the served table is **320 001 536 rows** across 16 heads (20000003…20000171 each), not 20M (`bench/ngram_pin_manifest.py` on the header). A 4096-row pin is 648 KiB, 1M rows 162 MB, so the 512 MiB table is ~3.1M rows — the *row budget* is affordable. First real count, on a 182 650-token corpus of this repo's own docs (2.92M lookups, 1.10M distinct rows): **4 096 rows (648 KiB) cover 23.6 %**, **65 536 (10.4 MiB) 51.5 %**, **262 144 (41.5 MiB) 70.0 %**, **1 048 576 (162 MB) 98.3 %**. A pin inside 1M rows is therefore enough for this corpus — the 512 MiB table is more than the evidence needs, and the value on *production* traffic is still uncounted |
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

## Progress (2026-11-04)

- **#5, mechanism:** merged — `EXL3_NGRAM_ROW_CACHE_PIN` (+ `_PIN_MAX_BYTES`, named guards, never-evicted slots) on top of the row cache; `bench/ngram_hot_rows.py` counts a tokenized corpus; `bench/ngram_pin_manifest.py` derives the hash manifest from the checkpoint header. The pin file itself is one command away, after the campaign.
- **#5, premise:** corrected (320M rows, not 20M) — see the row.
- **#3, payoff:** computed — see the row.
- **#1, instrument:** merged — `bench/router_lookahead_probe.py` measures, per layer, the cosine between the pre- and post-attention residual and the top-k overlap between the router's true picks and the picks a pre-attention probe would make; it has not been run on the model yet, and the item stays gated on that number.
- **#2:** its achievable half (a hot pool inside a bounded row budget) is the #5 mechanism; the non-temporal-load half is a `exllamav3_ext` change and remains unstarted.
- **#4, #6:** unchanged.

## What this backlog does not accept

- The projected throughput table (57 → 95–105 → 145–170 tok/s). Those are estimates stacked on
  estimates: plan 1's own levers are unmeasured, #1's premise is unmeasured, and #3's perplexity claim
  is unmeasured. No row above may be quoted as a projection; each is quoted only after its own arm.
- "Zero change to model outputs" for #1 — true only because a prefetch hint cannot change the router's
  answer; it is not a property of the prediction, and it must not be used to justify #4.
- "0 % L3 hit rate" for #2 — asserted, not measured on this box.
