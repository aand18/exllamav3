# T52 — The round is CPU-late by ~23 %, and the worker is not saturated

One pass, 2026-11-07: `t52-A-profile-262k-20261107-172128`, production regime, `-MaxLength 262144`,
`-dr 1`, pin on, `EXL3_MOE_CPU_PROMOTE=0`, `-Prof` + `EXL3_MOE_HANDOFF_PROF=1`. `status=ok`, 310 s,
floors `min_free_vram 2267 MB`, `breach=0`. Prefill 2 153.83 at 262 144, single-stream decode 39.96 at
261 888 — the profiling pass cost nothing visible in the headline numbers.

No code was written for this pass. Two instruments were already in the tree and had never been run
together; the T7 design called one of them inert and that was wrong (corrected in
`2026-11-08-round-phase-instrument.md` §1).

## What the three instruments say

**Worker side** (`EXL3_MOE_HANDOFF_PROF`, 3 032 prints, 194 048 jobs; steady state, last three prints):

| segment | ms/job | meaning |
|---|---|---|
| gap | **0.08** | no job waiting — the pool is briefly idle |
| spin | **0.21** | job arrived, activation not yet copied |
| compute | **0.69** | the expert GEMVs |

Per job the pool is busy 92 % of the time and computing 71 %. **It is not accumulating a backlog**:
`gap` is small and flat, and the busiest layers vary from print to print (L44, L41, L2, L1, L14) rather
than drifting upward with depth. That is the "each layer misses its deadline by a fixed amount" signature,
not a queue-building one.

**Compute** (`EXL3_MOE_CPU_PROF`, 378 prints, 192 000+ jobs): `gemv_gu 393.5 µs at 42.9 GB/s`,
`gemv_d 186.8 µs at 45.1 GB/s` → **580 µs of expert work per job**.

**Exposed wait** (`EXL3_SPLIT_PROF`, 174 prints, 2 048 brackets each): issue median **0.012 ms** (enqueue is
free), collect-wait median **0.207–0.316 ms**, p90 0.35–0.43 ms, worst layers 0.44–0.82 ms.

## The per-layer arithmetic

At the quoted 52 ms round (5-row window, 3.05 tokens/round, 58.5 tok/s) the layer budget is
**52 / 48 = 1.08 ms**. Against that:

```
CPU compute            0.58 ms
exposed collect wait   0.25 ms   <- the stream is blocked here
------------------------------------------
CPU-related            0.83 ms of 1.08 ms   (77 %)
GPU-only window        0.82 ms              (1.08 - 0.25)
```

The CPU finishes **~0.25 ms after the GPU needs it, every layer**: 48 × 0.25 = **12 ms of a 52 ms round
(23 %)** is the stream blocked on the worker's done flag. If the CPU were fully hidden the round would be
~40 ms, i.e. **~30 % more tokens/s** — larger than every lever this campaign has measured except drafting.

This is the first direct measurement of the quantity T50's model estimated at 48.7 ms from a bytes formula.
The formula was wrong by a factor of ~4: the CPU is late, but by 12 ms, not 49.

## Why T51's coverage gain bought nothing, and this is not a contradiction

`EXL3_MOE_TIER_STATS` counts **`(row, expert)` evaluations**. The CPU's cost is **bytes**, which scale with
**distinct experts per layer**, not with evaluations. A 5-row window evaluates each routed expert once per
row it serves; moving 25 % of the *evaluations* to VRAM does not remove 25 % of the *distinct* expert reads
if the misses that remain are the tail, each still read in full for every row routed to it. So coverage can
rise 22 → 57 % while the 0.58 ms of CPU work falls much less — which is exactly the null T51 measured.

The number to instrument next, if the CPU side is pursued, is **distinct CPU experts per layer per round**,
not coverage. It is computable from the same job's `selected_experts` with no new kernel, and it is the
denominator that makes the 0.58 ms interpretable.

## What this leaves as levers

1. **Fewer bytes per CPU-expert miss** — the per-layer 2 bpw tail. It halves the 0.58 ms to ~0.29 ms, which
   puts the CPU back inside its window and is worth up to the full 12 ms. It was postponed on **quality**,
   not on performance, and this measurement is the reason to revisit that decision: it is now the only
   lever at a mandatory 262 k context that attacks the *measured* critical path.
2. **More GPU residency** — closed at 262 k (allocator refuses; `size_check` optimistically says otherwise).
3. **Hiding the tail** rather than shrinking it — e.g. splitting a layer's CPU job so the GPU consumes the
   hot rows first. Not designed; would need the per-layer job structure to expose partial completion.
4. **The GPU side** — 0.82 ms/layer of work the CPU is racing. Kernels, graph structure and row count live
   here; the round-phase instrument (T7) is still the way to split it, but it is no longer the *first*
   question.

## Caveats

- The 52 ms round is the **drafted corpus** figure (3.05 tokens/round); this pass measured the ladder, not
  the corpus, so the per-layer budget is an arithmetic bridge, not a measured round in this pass.
- `-dr 1`: one draw. The three instruments are counts and medians over thousands of brackets, so they are
  far more robust than a tok/s cell, but the wait median should be confirmed against a second pass before
  the 12 ms is quoted as a rate.
- The 25.95 ms `max` in the wait bracket is a stall (first-touch / page fault), excluded from the medians
  and not explained here.
- `EXL3_MOE_CPU_PROF` and `HANDOFF_PROF` are C++-side prints; `EXL3_SPLIT_PROF` is stream-time brackets.
  The three bases are not interchangeable and the table above keeps them separate.
