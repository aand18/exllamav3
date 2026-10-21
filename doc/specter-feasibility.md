# Specter (arXiv:2511.14102) in exllamav3 for Qwen3.8-Flash-Next — feasibility

**Status:** investigation only. No runtime code changed. Every number below comes
from `eval/specter_budget.py` (added in this branch) or from a code reference in
`fork/fork-overview`, which is byte-identical to upstream
`turboderp-org/exllamav3` master apart from the fork's own docs — i.e. the CPU
MoE offload path and the Qwen3.8-Flash-Next support discussed here are already
upstream, not fork-local work.

**The paper:** *Specter: Fast MoE Inference via Quantization-Enabled Multi-Step
Speculative Offloading* (Wang et al., ASPLOS'27, arXiv:2511.14102v2). Artifact:
`github.com/wwwwwf/specter_asplos` (HF Transformers + GPTQ + Marlin, Linux, A100
40 GB). Claims 1.50x average (up to 1.88x) TPOT over SpecMoEOff on
DeepSeek-V2-Lite, Qwen1.5-MoE and Phi-3.5-MoE at batch size 1.

---

## 0. Verdict

**Do not port the artifact. Do port the idea, in a different shape.**

| Question | Answer |
| --- | --- |
| Can the paper's *system* be reproduced in exllamav3 for Qwen3.8-Flash-Next? | **No.** Three independent blockers (§3): draft-model VRAM (61.8 GiB needed vs 24 GB available), a premise mismatch (exllamav3 does not have the PCIe bottleneck the paper hides), and a hard generator restriction (the AR-draft path rejects recurrent architectures, which is every Gated-DeltaNet model). |
| Can the paper's *mechanism* (multi-step look-ahead to hide expert-transfer latency) be used? | **Yes, but it is not the paper's mechanism that is needed here.** For Qwen3.8-Flash-Next the dominant cost is the *host* expert compute, not the bus. The useful move is to shift expert compute from the host to the GPU, which makes the bus the bottleneck — and *then* the paper's look-ahead becomes the way to hide it. |
| Is the work worth doing? | **Conditionally yes.** The first step is a measurement, not a port: one number (host cost per offloaded expert on the target box, vs the bus cost of the same expert) decides between two designs that differ by a factor of ~2 in achievable TPOT (§5). The measurement costs an afternoon on the 4090 box. |
| What would it buy? | **~1.5–2.5x end-to-end TPOT on a gen5 x16 box, ~1.2–1.5x on gen5 x8, ~2.4x on gen4 x16** (the consumer-card regime — but only **1.13x** there if the transfer stays exposed, so on a 4090-class link the look-ahead is the whole difference between a win and nothing), **and nothing at all (a regression) on gen4 x4**, where the host is faster than the bus (§4.5). The ceiling for this design is the GPU-bound floor at ~5–7x. The cheapest slice of it is available today: `--mtp` speculation amortizes the host's weight reads over 4 rows (1.55x on the offloaded part) with no porting at all. |
| Effort if it goes ahead | Phase 1 (streaming at decode) is a few hundred lines on existing machinery. Phase 2 (look-ahead) is the real work: a predictor, an async fetch path, and a residency policy. Phase 3 (the paper's full design) is **not** recommended for this model. |

A deliberate non-goal: this document does not propose reimplementing Specter's
ELB/planner/residency-manager trio verbatim. For a 512-expert-per-layer model
the paper's *cache-centric* design does not fit (§4.3), and its *draft-centric*
design does not fit either (§4.4).

---

## 1. What the paper actually does

Three modules, in the authors' framing:

1. **SIC — Speculative Inference Controller.** Builds a *hybrid-precision* pair
   from one checkpoint: the target runs FP16, and a **4-bit copy of the target
   itself** is the draft. Non-expert parameters (routers, attention, shared
   experts) stay FP16 because router logits are quantization-sensitive. The
   depth `K` is picked offline by sweeping TPOT over `K ∈ {2..32}` (their picks:
   16 / 8 / 16 for DSv2-Lite / Qwen1.5-MoE / Phi-3.5-MoE).
2. **PIO — Predictive I/O Orchestrator.** While the draft generates `K` tokens, the
   draft's routers fill an **Expert Look-ahead Buffer** (per-layer activation
   counters over the drafted tokens' top-k). At chunk boundaries the buffer is
   ranked per layer, the top-`M` not-yet-resident experts are packed into an H2D
   batch, and the batch is DMA'd on a separate CUDA stream while the next draft
   chunk computes. Chunk sizes are chosen so draft time ≈ transfer time.
3. **SEE — Streamlined Execution Engine.** Fused multi-expert INT4 draft kernels
   (one launch instead of many narrow GEMMs), **cache-aware expert-wise
   verification** (group the `K` drafted tokens by expert so the FFN runs as
   dense GEMMs, run cache-resident experts first, stream the misses
   concurrently), and a parallelism optimizer that keeps compute and copy on
   separate streams with pinned buffers.

The load-bearing idea, and the reason the paper is interesting: **a 4-bit
quantization of the target predicts the target's own expert routing** (90.8%
top-k match for Qwen1.5-MoE; 92.9–99.4% across models/methods in their Table 1).
That turns a data-dependent, sequential routing problem into a schedulable I/O
problem, and a speculative draft is the only cheap source of future routing.

Measured results worth carrying over: cache hit rate 99.85% / 98.62% / 96.25% at
16 / 24 / 32 GB for DSv2-Lite (vs LRU, which is far behind); 55.5% / 60.5% /
22.4% of total H2D load hidden; component ablation 69.2 → 83.5 (w/o PIO) → 94.4
(w/o PIO+SIC) → 130.4 ms/token (w/o SEE); and the robustness table — degrading
prefetch accuracy from 1.000 to 0.410 costs only 1.34x TPOT, because
verification keeps the output distribution lossless.

---

## 2. What exllamav3 has today (the parts that matter)

Everything below is upstream master, and it is a much closer fit to "MoE with
host-resident experts" than the paper's HF-Transformers baseline.

| Ingredient | Where | Notes |
| --- | --- | --- |
| Qwen3.8-Flash-Next support | `exllamav3/architecture/qwen4_exp.py` (`Qwen4ExpModel`, `arch_string = "Qwen4ExpForConditionalGeneration"`) | GDN + QSA hybrid attention, gated residual, PLE n-gram layer, 512 experts, top-10 + 1 shared, MoE intermediate 640, 48 layers (36 GDN + 12 QSA) |
| MTP head (1 layer, in-checkpoint) | `exllamav3/architecture/qwen4_exp_mtp.py` (`Qwen4ExpMTPModel`, `default_draft_size = 4`) | Uses `--mtp` with `draft_model_dir == model_dir`; shares the target's embedding and `lm_head` |
| Host-resident routed experts | `exllamav3/model/moe_cpu_host.py` (1739 lines) + `exllamav3/modules/block_sparse_mlp_cpu.py` | A child process owns the cold experts in a huge-page arena; the parent talks to it over pinned shared memory with stream-ordered flags (`exllamav3/exllamav3_ext/cpu/moe_handoff.h`) |
| Expert weight streaming (prefill only) | `moe_cpu_host.py:1272` `submit_prefill` → `:1384` `_submit_prefill_streamed` | Hot experts (≥ `stream_t` assignments) have their weights DMA'd through a pinned ring into a small VRAM ring on a dedicated copy stream, then computed on the GPU; the cold tail runs on the host, concurrently |
| Pinned-arena direct DMA | `moe_cpu_host.py:1455-1467`, `_attach_chunk:650` | The parent maps the worker's memfd chunks and `cuda_host_register`s them, so an expert can be DMA'd host→device with no staging copy (Windows-supported) |
| Hot/cold expert placement | `block_sparse_mlp_cpu.py:574-710` | Dynamic swap sweep every 128 steps: the hottest CPU-resident expert is promoted into the coldest GPU slot, in place, with hysteresis and a mass floor; the router→physical-slot map `_split_map` is applied by a kernel (`routing.cu:813`) |
| Router→slot translation | `block_sparse_mlp_cpu.py:250` `_split_translate`, `routing.cu:870` `moe_split_issue` | Selected experts are translated to worker-local ids, with `-1` sentinels for GPU-resident picks |
| Speculative decode loop | `exllamav3/generator/generator.py:603` (AR draft), `:703` (MTP), `:800` (DFlash), `:899` (n-gram) | Linear-chain verification only, no tree; dynamic draft length + online confidence calibration (`generator/draft_confidence.py`) |
| Prefetch pattern to copy | `exllamav3/modules/ngram_embedding.py:541` `prefetch` | Worker thread + two pinned staging sets + CUDA events + hit/miss stats; the PLE n-gram table is streamed from disk this way |
| Model-level prefetch hook | `exllamav3/model/model.py:76-79, 366-367, 391-392` | `caps["prefetch_ids"]` modules are prefetched before the first layers |
| Profiling | `EXL3_SPLIT_PROF`, `EXL3_MOE_SUBMIT_PROF`, `EXL3_MOE_CPU_PROF`, `EXL3_MOE_STREAM_DEBUG` | Per-layer collect-wait brackets, submit brackets, host-side worker stats, and the measured link bandwidth |

Measured on the fork's own box, and the only hard hardware datapoint that
exists in-tree (`moe_cpu_host.py:1215`):

> *Calibrated on Qwen3.8-Flash-Next (512 experts, 410 on the CPU, 12 threads)
> over gen5 x16 (57 GB/s), gen5 x8 (29 GB/s) and gen4 x4 (6.7 GB/s) links: 8 was
> best on both gen5 links … 16 on gen4 x4.*

So the fork has already been run with this model on a 24 GB-class card: ~102 of
512 experts per layer resident on the GPU, 410 on the host, 12 host threads.

**The target box is a gen4 x16 link.** An RTX 4090 on PCIe 4.0 x16: 26.8 GB/s
probed host→device. The paper calibrates over gen5 x16, gen5 x8 and gen4 x4, so
gen4 x16 is an extrapolation in every table below — and it is the regime most
consumer cards are in. The box has 16 cores / 32 logical threads; the model
assumes 12 host threads, so the thread count is itself one of the variables to
measure.

**The gap, stated plainly:** the router's top-k selection is a local variable
inside `BlockSparseMLP.forward` (`exllamav3/modules/block_sparse_mlp.py:997-1008`)
and is never exported. There is no GPU expert *cache* keyed by expert id, and no
decode-time path that fetches a single expert over PCIe. Those are the two
primitives every Specter-style design needs.

---

## 3. Premise check: the paper's premises against this model and this engine

| # | Paper's premise | Reality in exllamav3 / Qwen3.8-Flash-Next | Verdict |
| --- | --- | --- | --- |
| P1 | The bottleneck is PCIe weight transfer, because experts live in host RAM | exllamav3's CPU offload path ships **activations**, not weights: ~480 KiB/token vs ~1.17 GiB/token for the paper's shape — **2550x less traffic**. The host does the expert math | **Mismatch.** The bottleneck here is host compute; the paper's optimization attacks a cost this engine does not have |
| P2 | A 4-bit copy of the target is a cheap, high-fidelity routing predictor | The routed experts are 120.8 B of the model's 124.9 B parameters → an INT4 copy is **61.8 GiB**. It does not fit next to a resident expert set on a 24 GB card, and the AR-draft path additionally asserts `not draft_model.caps.get("recurrent_states")` (`generator.py:162`) — Qwen4Exp sets `recurrent_states: True` (`qwen4_exp.py:310`) | **Blocked.** Substitute predictors exist (§4.4) but none is the paper's |
| P3 | The expert working set fits in a GPU cache, so prefetching pays | 512 experts/layer, top-10: the union over `K` drafted tokens is ~67 distinct experts/layer at `K=16` → **7.8 GiB per step** (14.1 GiB at `K=32`), against ~13 GiB of expert budget on a 24 GB card that must also hold the resident set, KV cache, and the 95 GiB n-gram table's staging | **Does not fit.** The cache-centric design (ELB → top-M → cache) is not viable at this scale; a *streaming* design is |
| P4 | Verification is where the transfer can be hidden | exllamav3 verifies a linear chain of `K` tokens, one position at a time, with no tree and no expert-wise reordering (`generator.py:1156-1259`) | **Gap.** The reordering exists in the paper's SEE; here the MoE already runs a batched/expert-sorted path (`block_sparse_mlp.py:1106-1131`, `moe_batch_recon.py`), so part of it is already there |
| P5 | The draft runs on-device and is fast | The MTP head does run on-device and is target-consistent (`default_draft_size = 4`), but it is 1 layer: it predicts *tokens*, not per-layer routing | **Partial.** Tokens are useful; per-layer routing is the paper's actual currency |
| P6 | Expert weights are in host RAM in a form that can be DMA'd | True: the arena is huge-paged, and `EXL3_MOE_PINNED_ARENA=1` maps the worker's memfd into the parent and registers it with CUDA, so a direct host→device DMA of one expert is possible | **Available.** This is the strongest enabling fact in the whole fork |

**Update, 2026-10-21 — the second half of P2 no longer holds.** The
recurrent-state assertion only rejects an *autoregressive* draft.
`incoai/Qwen3.8-Flash-Next-DFlash2` (0.6 B, Apache-2.0, published 2026-10-19) is
a **block-diffusion** draft model for exactly this target: 5 dense
sliding-attention layers, `is_causal: false`, `use_cache: false`, block size 8,
a top-16 selector that traces one coherent path through the candidates. There is
no recurrent state for `generator.py:162` to reject, it drafts a whole block in
one pass — which is precisely the multi-step inter-token lookahead the paper
needs and an AR drafter cannot give — and its acceptance length beats the
model's own MTP on every task measured (GSM8K 3.26 vs 3.08, HumanEval 2.94 vs
2.52, MT-Bench 2.83 vs 2.50). The VRAM half of P2 stands (61.8 GiB is still
61.8 GiB); what changes is that a usable draft now exists, so P5 ("the draft
predicts tokens, not per-layer routing") can be attacked with a real drafter
rather than a hypothetical one. Its integration is not free: the capture patch
targets SGLang's `qwen4_exp.py`, and neither this fork nor exllamav3 has a
DFlash path.

**Conclusion of the premise check:** the paper's *artifact* is aimed at a
bottleneck this engine does not have, on a model whose scale breaks its cache
and draft assumptions. The paper's *insight* — that a cheap predictor of future
routing converts a sequential dependency into a schedulable one — is exactly
what the fork is missing, and it is missing it in the place where it would pay:
the fork's *hot/cold placement is purely retrospective* (`_split_hist`, a
decayed hit count) and its swap is a **quiesced, checkpoint-re-reading**
operation every 128 steps.

---

## 4. Budget arithmetic for Qwen3.8-Flash-Next

From `eval/specter_budget.py --preset qwen3.8-flash-next --vram-gb 24`
(4.25 bpw, 12 host threads — the target box has 16 cores / 32 logical threads,
so the thread count is a variable to measure too; the host-cost constants are
*assumptions*, see §6).

### 4.1 Footprint

| Quantity | Value |
| --- | --- |
| Routed experts | 120.8 B params → **59.8 GiB** |
| One routed expert | 4.9 M params → **2.49 MiB** |
| Non-expert + embeddings | 4.1 B params → 2.0 GiB |
| Whole model (excl. n-gram table) | 124.9 B params → 61.8 GiB |
| MTP head (1 layer) | 2.5 B params → 1.24 GiB |
| PLE n-gram table | 51.2 B params → 95.4 GiB fp16, streamed from disk |

### 4.2 What a 24 GB card can hold, and what each step costs

| VRAM | resident experts/layer | cold tail |
| --- | --- | --- |
| 24 GB (55% for experts) | 113 / 512 (22%) | 399 |
| 12 GB | 56 / 512 (11%) | 456 |

Per decode step, 48 layers × top-10 = **480 expert evaluations**:

| Shape | Host↔device traffic per token |
| --- | --- |
| compute-on-CPU (what the fork does) | **480 KiB** (48 × activation down + result up) |
| stream-to-GPU (the paper's shape) | **1.17 GiB** (480 × 2.49 MiB) |
| ratio | **2550x** |

Modelled cost of the offloaded part of one step, as a function of the number of
rows attached to each expert (`R = 1` is plain decode, `R = K` is verification):

| rows | distinct experts/layer | host compute | gen5 x16 | gen5 x8 | gen4 x16 | gen4 x4 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10.0 | 89.3 ms | 22.0 ms | 43.2 ms | 79.3 ms | 187.1 ms |
| 4 | 21.9 | 231.2 ms | 48.2 ms | 94.7 ms | 173.8 ms | 409.9 ms |
| 16 | 66.8 | 779.7 ms | 146.9 ms | 288.7 ms | 529.9 ms | 1249.7 ms |

Read this as: the host cost per step is dominated by *reading expert weights*
(145 µs/expert modelled) and grows only mildly with `R`; the bus cost is
proportional to the *union* of experts and therefore also grows sub-linearly in
`R`. The break-even link bandwidth for one expert is **18.0 GB/s** — below the
57 GB/s of a gen5 x16 slot, above the 6.7 GB/s of a gen4 x4 slot.

Two conclusions, both of which point the same way:

* On a gen5 box, **streaming an expert to the GPU is cheaper than computing it
  on the host**, even fully exposed on the critical path. The fork's host-compute
  path is not the obviously-correct choice at this model's scale.
* The advantage of the bus *grows* with speculation, because the union of
  experts over `R` rows is much smaller than `R × top-k`. This is the paper's
  argument, and it survives the transplant.

### 4.3 Why the paper's cache-centric design does not fit

Distinct experts per layer as a function of depth (the recurrence in
`eval/specter_budget.py` is fitted so that `K=16` reproduces the paper's
measured 31.7 for DeepSeek-V2-Lite):

| K | Qwen3.8-Flash-Next | working set | DeepSeek-V2-Lite (paper) | working set |
| --- | --- | --- | --- | --- |
| 4 | 21.9 | 2.6 GiB | 12.4 | 1.4 GiB |
| 8 | 37.4 | 4.4 GiB | 19.9 | 2.2 GiB |
| 16 | **66.8** | **7.8 GiB** | 31.7 | 3.5 GiB |
| 32 | 120.3 | 14.1 GiB | — | — |

The paper's DSv2-Lite working set at `K=16` is 3.5 GiB against a 16–24 GB
budget: a cache can hold it. Qwen3.8's is 7.8 GiB against a ~13 GiB expert
budget that must also cover the resident set and the KV cache: a cache cannot
hold it, and at `K=32` it is hopeless. A cache-centric Specter therefore has to
be re-thought as a *streaming* design, where the working set only has to fit
**one layer at a time** (10 experts ≈ 24.9 MiB), and the transfers are hidden by
look-ahead rather than absorbed by capacity.

### 4.4 Draft / predictor options that fit in VRAM

| Option | VRAM | What it gives |
| --- | --- | --- |
| Full INT4 copy of the target (the paper) | 61.8 GiB | **No.** Per-layer routing, ~91% fidelity (measured on other models) |
| MTP head from the same checkpoint | 1.24 GiB | Target-consistent *tokens* (4 deep), no per-layer routing |
| Target draft restricted to the top-16 / 32 / 64 experts per layer | 3.8 / 5.7 / 9.4 GiB | Per-layer routing, but missing experts ⇒ hidden-state drift ⇒ fidelity unknown |
| Router-only predictor (routers only, no experts) | 32 MiB | Needs a hidden-state model to feed the routers |
| *Routing-transition predictor* (`P(routing_L(t+1) | routing_L(t))`, learned from a trace) | << 1 MiB | Per-layer, per-step look-ahead with no draft model at all |

The last row is the one this document argues for. It is cheap, it is
measurable offline, and it fits the fork's structure: the router's top-k for
layer `L` is known at the *start* of layer `L`'s MoE, which is **a full layer of
compute ahead** of layer `L+1`'s need — and in the current design that layer is
the slow host-computed one, so there is ample wall-clock to fill it. This is
the paper's "intra-token" look-ahead that the paper dismisses as too short a
window; in this engine the window is one whole layer, not microseconds.

### 4.5 Expected improvement

Three regimes for the offloaded part of a decode step, per link speed
(`eval/specter_budget.py`, section `[4b]`):

* **A** — today: the host computes the cold experts.
* **B** — stream the weights to the GPU, transfer exposed on the critical path.
* **C** — stream the weights, and the look-ahead hides 60% of the transfer (the
  paper's measured hidden ratio for Qwen1.5-MoE).

| offloaded part, ms/step | gen5 x16 | gen5 x8 | gen4 x16 | gen4 x4 |
| --- | --- | --- | --- | --- |
| A — today (host computes) | 89.3 | 89.3 | 89.3 | 89.3 |
| B — stream, exposed | 22.0 | 43.2 | 79.3 | 187.1 |
| C — stream, 60% hidden | 8.8 | 17.3 | 31.7 | 74.8 |
| floor — all GPU, memory-bound | 3.4 | 3.4 | 3.4 | 3.4 |

| speedup of the offloaded part | gen5 x16 | gen5 x8 | gen4 x16 | gen4 x4 |
| --- | --- | --- | --- | --- |
| B vs A | **4.06x** | 2.07x | 1.13x | **0.48x** |
| C vs A | **10.15x** | 5.17x | 2.81x | 1.19x |

End-to-end, adding the GPU's own per-token work (attention, GDN, QSA indexer,
shared expert, embeddings, n-gram gather) modelled at 10 ms:

| end-to-end | gen5 x16 | gen5 x8 | gen4 x16 | gen4 x4 |
| --- | --- | --- | --- | --- |
| A — today | 99.3 ms (10.1 tok/s) | same | same | same |
| B — stream, exposed | 32.0 ms (31.3 tok/s) | 53.2 ms (18.8) | 89.3 ms (11.2) | 197.1 ms (5.1) |
| C — stream, 60% hidden | 18.8 ms (53.2 tok/s) | 27.3 ms (36.6) | 41.7 ms (24.0) | 84.8 ms (11.8) |
| floor | 13.4 ms (74.4 tok/s) | same | same | same |

Three caveats that matter more than the table:

1. **The ratios are more trustworthy than the absolutes.** The absolute numbers
   inherit the assumed host cost (1.5 GB/s/thread dequant, 20 GFLOP/s/thread
   GEMV, 12 threads) and an assumed 10 ms of GPU work. M1 and M3 replace both.
2. **The model is optimistic, and the paper says by how much.** For
   DeepSeek-V2-Lite on a PCIe 4.0 x16 A100 the model gives B 24.7 ms → C
   9.9 ms, i.e. 2.5x on the offloaded part, while the paper's *realized*
   end-to-end gain over its own streaming baseline was 1.5x average (1.88x
   max) — their implementation also pays for the draft, the ELB bookkeeping and
   the verification overhead. Apply a haircut of roughly 1.3–1.8x to the
   modelled ratios.
3. **On a gen4 x4 link the whole direction is wrong.** B is 0.48x of A: the
   bus is slower than the host, so streaming the weights would be a
   *regression*. The link speed therefore has to be known before any of this
   work starts (`EXL3_MOE_STREAM_DEBUG` already reports it).

**A second, independent gain comes from speculation itself**, and it is the
cheapest thing on the list. Verifying K rows amortizes the host's weight reads
over those rows: at K=4 the offloaded cost falls from 89.3 ms/token to
57.8 ms/token (1.55x) with no code change at all — just `--mtp`, because the
MTP head is already in the checkpoint. At the same time the *bus* cost per
token rises (the union of experts grows), so speculation helps the host path
and hurts the streaming path, which is precisely the tension the paper resolves
with prefetching. The MTP head's acceptance rate is unmeasured (M5), and the
paper's own profiling shows TPOT degrading past the optimal depth, so K=4 is a
starting point, not an answer.

**Realistic expectation**, after the haircut and with the GPU-side work
included: **~1.5–2.5x end-to-end TPOT on a gen5 x16 box** (roughly 15–25
tok/s from a ~10 tok/s modelled baseline), **~1.2–1.5x on gen5 x8**, and **~2.4x
on gen4 x16** (99.3 → 41.7 ms modelled, 10.1 → 24.0 tok/s) — but the gen4 x16
number is conditional: with the transfer exposed that link yields only
**1.13x**, so on a consumer-card link the look-ahead is not an optimization, it
is the entire difference between a win and nothing. **No gain at all — a
regression — on gen4 x4**. The hard ceiling for this design is
the GPU-bound floor at ~5–7x. The cheapest available improvement today is
`--mtp` (1.5x on the offloaded part, no porting), and the largest is the
look-ahead, which is where the paper's contribution would land.

---

## 5. The decision the measurement drives

Let `t_host` = host cost of one offloaded expert, `t_bus` = transfer cost of
the same expert, `w` = window available to hide the transfer.

| Measured | Design | Expected effect |
| --- | --- | --- |
| `t_bus < t_host` (modelled: 44 µs vs 186 µs at gen5) | **Stream to the GPU**, with look-ahead to hide the transfer | Move expert compute off the host: the 89 ms/token of modelled host compute per decode step is replaced by ~22 ms of bus time, less whatever the look-ahead hides |
| `t_host < t_bus` (gen4 x4, or a very fast host) | Keep the host-compute path; look-ahead is pointless | Look-ahead only helps if you also move compute to the GPU; otherwise the paper's mechanism has no target |
| Either, but `w` (prefetch window) < `t_bus` | Stream without look-ahead, accept exposed transfer | Still a win if `t_bus < t_host`; the paper's machinery buys the difference between "exposed" and "hidden" |

The interesting third column: the fork's layer structure provides the window for
free. Layer `L`'s host compute is the slowest thing in the step; using it as the
window in which layer `L+1`'s experts are DMA'd converts the host's slowness
into prefetch bandwidth. A "mixed schedule" (one host-computed layer, the rest
GPU-computed) falls out of this, and it is *not* the paper's design.

---

## 6. Measurement plan (4090 box; the GPU-less dev box cannot run this)

Everything in §4 is arithmetic over assumed host constants. The assumptions are
the weakest link: `1.5 GB/s/thread` dequant and `20 GFLOP/s/thread` GEMV are
guesses. One afternoon of measurement replaces the whole model with facts.

**M1 — Host cost per offloaded expert.** Run the existing CPU MoE worker on
Qwen3.8-Flash-Next and read the per-expert time from `EXL3_MOE_CPU_PROF` (or
`EXL3_SPLIT_PROF` brackets). Record: weight-read time, GEMV time per row, and
the number of threads actually used. This is the `t_host` of §5.

> A calibration point from outside this repo, measured on the same box
> (2026-10-21): the *other* engine for this model — the Strata fork — runs its
> own expert pool on the same Ryzen 9 7950X3D and the same expert bytes, and
> its per-window decode timing puts the host at **44 µs per offloaded expert**
> (3.53 ms of CPU-pool time per window / 48 layers / 1.68 experts). The 186 µs
> this study assumes is therefore not a property of the box but of the engine:
> a hand-tuned AVX2/AVX-512 pool is ~4x faster than a general CPU-MoE path.
> The number that matters here is exllamav3's own, and it can only come from
> M1 — but the spread means the model is not trustworthy to better than a
> factor of four until then.

**M2 — Bus cost per expert.** `EXL3_MOE_STREAM_DEBUG` already reports the
probed pinned→device bandwidth (57 / 29 / 15.8 / 6.7 GB/s on the links the
fork was calibrated on). Add a per-expert timing of the prefill streaming path
to get `t_bus` end-to-end, including the arena→pinned→VRAM hops, since the
pinned-arena path and the staged path differ.

**M3 — Baseline TPOT.** `eval/perf.py` with `--moe_cpu_split` at a few split
points, batch 1, long decode, on the same prompt stream (it already uses
wikitext2 so routing is realistic). This is the number every later change is
measured against.

**M4 — Routing statistics (the predictor's input data).** Export the router's
top-k per layer per token for a few hundred tokens, and measure:
  * per-layer distinct-expert count and its growth with `K` (replaces the fitted
    skew factor in §4.3 with the real curve for this model);
  * temporal overlap `|routing_L(t) ∩ routing_L(t+1)| / top-k` (how good a
    one-step transition predictor can be);
  * cross-layer overlap `|routing_L(t) ∩ routing_{L+1}(t)| / top-k` (how good a
    cross-layer predictor can be — this is the one that buys the prefetch
    window);
  * the GPU slice's hit rate as a function of slice size, i.e. the headroom
    the placement policy is leaving on the table.

M4 needs the router export (§7, item 1) and is the cheapest item on the list.

**M5 — Speculation acceptance.** `eval/spec_decode.py` with the MTP head
(`--mtp`) to get the acceptance rate of the 4-token draft, i.e. how many
tokens a look-ahead can actually cover.

**Prerequisites.** A Qwen3.8-Flash-Next EXL3 checkpoint does not exist upstream
yet: it means converting the 125 B FP16 release (`convert.py`, plus
`util/convert_ngram.py` for the separate n-gram table) — a ~250 GB download and
a long conversion, with the n-gram table (95 GiB fp16) needing its own
quantized file. Budget disk accordingly, and note the table is streamed from
disk at runtime (`EXL3_NGRAM_STREAM=1`), which adds its own I/O to the step.

> **Update, 2026-10-21:** the conversion step is no longer a prerequisite for
> the measurement. `turboderp/Qwen3.8-Flash-Next-exl3` publishes 2.05 bpw and
> 3.05 bpw branches for this model, and both are already on the target box
> (`D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw`, `-3.05bpw`, the latter also
> deployed through TabbyAPI). M1–M3 can run against those today; only a
> *fork-local* checkpoint with the fork's own quantization would need the
> conversion.
All CUDA runs go through the `eval/smi_guard.py` watchdog that this repo's
rules require (it does not exist in-tree yet — it has to be written).

---

## 7. Port plan

Ordered by value per unit of risk. Each phase is independently useful and
independently abandonable after its measurement.

### Phase 0 — Instruments (no behaviour change)

1. **Export the router's top-k.** In `BlockSparseMLP.forward`
   (`modules/block_sparse_mlp.py:997-1008`) write the selection into a
   per-layer buffer that the model exposes (e.g. `params["selected_experts"]`
   or a ring on the module). This is *the* missing primitive: it is needed by
   the predictor, by the profiler, and by any prefetch. Note the bsz-1 path
   already keeps a static `RoutingCFG.selected_experts_bsz1`
   (`modules/block_sparse_mlp_routing.py:62`) that captured graphs read at baked
   addresses — an export must not disturb that.
2. **Per-expert and per-layer counters** for the CPU path and the swap sweep
   (bytes moved, experts promoted/demoted, hit rate of the GPU slice), in the
   style of `NGramEmbedding.prefetch_stats`.
3. Write `eval/smi_guard.py` (VRAM watchdog) — required by the repo's rules for
   any CUDA run and missing from the tree.

Exit criterion: M1–M4 measured on the 4090 box, and `eval/specter_budget.py`
re-run with `--cpu-us-per-expert` set from M1 so the model is calibrated.

### Phase 1 — Stream experts to the GPU at decode

Today the weight-streaming path exists only for prefill-scale batches
(`EXL3_MOE_STREAM_MIN_ROWS=32`; decode rows are 1). Extend it to the decode
path: when the router fires at layer `L`, DMA the selected experts' weights
into a small per-layer VRAM ring on the copy stream (the ring already exists:
`EXL3_MOE_CPU_WSLOTS` × `EXL3_MOE_CPU_WSLOT_MB`) and compute them on the GPU
via the existing fused (`ext.exl3_moe`) or reconstruct tiers. The transfer is
exposed at first; the point is to establish the "bus is cheaper than the host"
claim on the target box with real numbers.

Risk: the fused kernel requires mul1 codebooks
(`block_sparse_mlp.py:540-548` `support_fused`), so the CPU-offload path and
the streaming path have overlapping but not identical quant requirements
(`modules/block_sparse_mlp_cpu.py:326-330` requires mul1 specifically, and
`moe_cpu_host.py:761-762` requires uniform expert dims and top-k across
layers — true for this model).

### Phase 2 — Look-ahead (the paper's contribution, in this engine's shape)

5. **Cross-layer / temporal routing predictor**, trained offline from the M4
   traces, tiny at runtime (a per-layer table or a small MLP over the previous
   selection's embedding). It answers: "which experts will layer `L+1` need?"
   while layer `L` is still running.
6. **Async fetch path**: a decode-time, non-quiescing, single-expert host→device
   DMA into a GPU-resident slot, with the router→slot map updated to match. The
   pieces are close to hand: pinned-arena DMA (`moe_cpu_host.py:1455-1467`), a
   copy stream with events, the `_split_map` translation kernel
   (`routing.cu:813`), and the `-1` sentinel that already tells the host worker
   "this expert is on the GPU". What is missing is the ability to *evict from
   the arena* (the worker must be told the expert now lives on the GPU) and to
   do it without the full-device sync the current swap sweep performs
   (`block_sparse_mlp_cpu.py:574-596`).
7. **Mixed schedule**: let the host compute the layer whose experts could not be
   predicted in time, and the GPU compute the rest. The host's slow layer
   becomes the prefetch window for the next one. This is the design that
   actually fits this model, and it has no analogue in the paper.
8. **Speculation on top** (optional, later): the MTP head gives 4 target-
   consistent tokens; combined with the predictor, the look-ahead covers a
   verification step rather than a single token. The verification reordering
   (group by expert, resident first) is partly already present in the batched
   MoE path.

### Phase 3 — Not recommended for this model

* The paper's full design: an INT4 copy of the target as the draft (61.8 GiB),
  the ELB → top-M → GPU-cache residency manager (7.8 GiB working set at `K=16`),
  and its own fused INT4 MoE draft kernels. None of it fits, and exllamav3's
  EXL3 mul1/trellis quant is not GPTQ/Marlin, so the artifact's kernels do not
  transfer either. **Do not attempt a literal port.**

---

## 8. Risks and open questions

| Risk / question | Why it matters | How to settle it |
| --- | --- | --- |
| The host-cost model may be wrong by 3x either way | §5's verdict flips with it | M1 |
| The link may be gen4 x4 (6.7 GB/s) on the target box | Below the 18 GB/s break-even → the paper's mechanism has no target at all | `EXL3_MOE_STREAM_DEBUG` |
| The n-gram table (95 GiB fp16) is itself an I/O bottleneck at decode | It is gathered per forward from disk (`ngram_embedding.py:570-616`); the paper does not address this at all | M3 with and without the PLE layer's gathers |
| The AR-draft path rejects recurrent architectures | Blocks "a copy of the target as draft" even if VRAM allowed it | `generator.py:162`; would need recurrent-state sharing between draft and target. A *non-AR* draft is not rejected by that assertion — DFlash 2's dense block-diffusion draft has no recurrent state, so this risk now applies to the paper's copy of the target only |
| The swap sweep perturbs logits | Same expert computed on host vs GPU differs numerically, which is why sweeps are deferred to a quiesced point (`block_sparse_mlp_cpu.py:587-592`) | A streaming design must keep the numerical difference explicit, or verification accuracy suffers |
| 320 M-row n-gram table needs its own quantized file | Conversion prerequisite | `util/convert_ngram.py` |
| Upstream merge | The fork's base *is* upstream master; a successful phase would be an upstream PR, not a fork-local feature | Decide the target repo before writing code |

---

## 9. What was not done

* No exllamav3 runtime code was modified. `eval/specter_budget.py` is the only
  addition, and it is pure arithmetic (no torch, no GPU).
* No measurement was taken: this dev box has no GPU (`nvidia-smi` is not
  present; `doc/local-build.md` records the same constraint), so every host-side
  number in §4 is a model, not a measurement.
* The paper's artifact was not read line by line; the analysis is based on the
  paper (v2 HTML) and the artifact's README, which describe the same system.
* No attempt was made to evaluate the paper's own baselines (SpecMoEOff,
  Mixtral-Offloading-SM) on this model; they are the comparison points a
  Phase-1 result would have to beat, and they are not in-tree.
