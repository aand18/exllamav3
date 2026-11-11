# T82 — KV residency assessed as a quality/capacity feature; the 2 bpw region is denied

Two decisions, recorded together because they are the same decision seen from both ends: what this
campaign is allowed to build next, and what it is not.

## 1. DENIED — the 2 bpw cold-CPU region (operator, 2026-11-11)

> **The mixed-representation model is a bad idea as a product configuration. Mark it denied.**

Accepted, and cancelled rather than deferred:

- 3.18 bpw hot experts on the GPU plus 2 bpw cold experts on the CPU means **two expert
  representations in one model** — extra sizing, extra loading code, two quality stories.
- It is legal **only under static placement**: `promote_expert`/`demote_expert` copy with a single
  `expert_bytes` and identically shaped destinations, so a cold 2 bpw expert cannot be promoted into
  a 3.18 bpw slot. Any future move to dynamic placement would make the checkpoint unshippable.
- The compile-only falsification build authorised in T77 is **cancelled with it**.

**What that costs the record, stated plainly:** the mechanism question it would have answered — *does
host-read bytes drive `cmb`?* — is now unpaid. The payload law stays **untested**, not validated. And
with it goes the last decode-speed lever that had real arithmetic behind it.

**Decode speed at this model, cache width and configuration is therefore closed by decision, not by
measurement.** Everything that could be measured has been: the CPU handoff line (T57–T60), expert
count at four doses (T62/T66/T67/T69), the draft window (T70), the drafter (T61/T63), pinning (T67).
The remaining question — whether bytes per expert is causal — is unfunded by choice.

Kept for anyone who reopens it: `dims_of()`/`proj_dims` do give the CPU worker its own K
(`block_sparse_mlp_cpu.py:1425-1429`, `:1464`), so a region split needs no kernel change; it requires
`EXL3_MOE_CPU_SWAP=0` and `EXL3_MOE_CPU_PROMOTE=0`; the conditional projection was ≈26 % lower round
latency ≈ 35 % higher round rate (low-to-mid 70s end to end), never measured.

## 2. KV residency — justified by quality and capacity, not by decode speed

Kimi's reframing, adopted:

- **FP16 KV** improves long-context quality (to be measured, not assumed).
- **Concurrency**: cold KV in host RAM allows more or longer resident sessions.
- **Decode speed**: unlikely to improve; the design should be *allowed* to trade latency for quality
  and capacity.

That is a different objective from the one this campaign has been optimising, and it changes the
arithmetic in the feature's favour — but not as far as its first estimate suggested.

### Correction: FP16 is ~2.6× the KV bytes, not 4×

`STATE.md:181` records the campaign's own figure: **6.94 GiB of KV at 262144 in FP16**, against
`size_check`'s Q4 figure of **2688 MiB** — **2.64×**, not 4×. The indexer planes (`raw_k`, `pooled`)
are already fp16 and ride along at every cache width, so only `qk/qv/sk/sv` scale.

| | first estimate | corrected |
|---|---|---|
| KV bytes at 262k | 4× → 10.7 GB | **2.64× → 7.1 GiB** |
| per-round gather (12 layers, 5 rows, ≤2048 tokens/row) | 284 MB → 11 ms | **~187 MB → ~7.2 ms** |
| share of a 50 ms round | 22 % | **~14 %** |

The conclusion survives — FP16 at 262k costs roughly an eighth to a seventh of a round if the gather
overlaps — but it is ~1.5× cheaper than first stated, and that matters when it is being compared
against the alternative.

### The comparator that was missing

**FP16 at 262k already fits today at split ~440.** So the choice is not "residency or nothing":

| route | cost | mechanism |
|---|---|---|
| split 440, FP16 KV in VRAM | +59 CPU experts/layer → distinct 20.5 → ~23.7 → **+5.2 ms/round ≈ +11 %** | known configuration, no code |
| KV residency, FP16 host-side + hot window | **~7.2 ms/round ≈ 14 %** if staging overlaps; worse if PCIe-latency-bound | one attention path + indexer, new kernels |

Host residency therefore has to beat a configuration that already exists and costs ~11 %. Its only
route to doing so is a **hot window large enough that most gathers hit VRAM** — a tunable, and
measurable.

### Four pre-measurements, in this order

1. **Comparator first:** what does split-440 FP16 actually cost? One `perf.py` ladder, no new code.
   This turns the target into a number and is currently missing from every comparison.
2. **Quality:** FP16 vs the current Q4 cache on representative long-context workloads, as a number.
3. **Sizing:** per-session VRAM *and* DRAM at target context and concurrency, including indexer,
   recurrent state, scratch and workspace — not just the KV tensors. The `suh`/`svh` lesson applies:
   an unpriced allocation is a floor breach waiting.
4. **Systems cost:** the real FP16 gather shape with concurrent streams, measuring achievable overlap.
   The arithmetic says 7.2 ms/round at 26 GB/s; the risk is that 576-byte random reads never approach
   that rate.

### What is scoped to the attention path

The model is hybrid: **12 of 48 layers carry KV**, all through one implementation — QSA block-sparse
with a 2048-token indexer budget. The recurrent 223 MiB is fixed-size and stateless with respect to
context, so it stays in VRAM and needs no residency. The work is therefore: keep `pooled` in VRAM for
the indexer (~200 MB at 262k), put `qk/qv/sk/sv` in pinned host RAM with a hot window, stage the
indexer's ≤2048 selected tokens per row into a per-layer scratch, and handle the recurrent-checkpoint
interaction for restored pages. One attention path, not every backend.

## Evidence

Figures used: `size_check` Q4 at 262144 (cache 2911 MiB = KV 2688 + recurrent 223);
`STATE.md:181` (FP16 6.94 GiB at 262k, split ~440); expert-at-the-floor value 33.9 µs (T66) and the
non-linear ladder (T79); the +8-expert null (T69, `t69A-cs40-mcs373-20261110-223004`,
`t69B-cs40-mcs373-20261110-224657`); the drafter and window results
(`t61A-bf16-131k-20261110-065833` … `t70E-ndt4-262k-20261111-070757`).

No new measurement in this task: it records a decision and re-prices an existing proposal.
