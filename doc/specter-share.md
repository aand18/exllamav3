# Specter in exllamav3 for Qwen3.8-Flash-Next — shareable summary

A condensed, self-contained version of [`doc/specter-feasibility.md`](./specter-feasibility.md)
for sharing outside the repo (Reddit, Hugging Face papers discussion, a blog
post). The study is the canonical version; this file is the short form.

---

## Can the Specter paper be implemented in exllamav3 for Qwen3.8-Flash-Next?

**Short answer: the paper's artifact should not be ported; its central idea should
be, in a different shape. And the first step is a measurement, not a port.**

[Specter](https://arxiv.org/abs/2511.14102) (ASPLOS'27) hides PCIe latency when
offloading MoE experts by using a **4-bit copy of the target model itself** as the
speculative draft: its routers predict the target's own expert routing with ~91%
top-k fidelity, which turns a sequential, data-dependent dependency into a
schedulable I/O problem. It reports 1.50x average (1.88x max) TPOT over
SpecMoEOff on DeepSeek-V2-Lite, Qwen1.5-MoE and Phi-3.5-MoE, on an A100 40 GB at
batch size 1.

Three things make that design a poor fit for Qwen3.8-Flash-Next in exllamav3.

**1. There is no PCIe stall to hide.** exllamav3's CPU MoE path keeps the cold
experts in host RAM and *computes them there*; only activations cross the bus.
That is 480 KiB/token against the 1.17 GiB/token the paper's design implies —
**2550x less traffic**. The bottleneck here is host compute, not the bus.

**2. The draft does not fit, and cannot be made to fit.** The routed experts are
120.8 B of the model's 124.9 B parameters, so an INT4 copy of the target is
**61.8 GiB** for a 24 GB card. Separately, exllamav3's speculative-decoding path
asserts that the draft model has no recurrent state
(`exllamav3/generator/generator.py:162`) — and Qwen3.8's Gated DeltaNet layers
are recurrent, so an in-family *autoregressive* draft model is rejected
outright. (That second half no longer binds: `incoai/Qwen3.8-Flash-Next-DFlash2`
is a **block-diffusion** draft for this exact target — 0.6 B, 5 dense
sliding-attention layers, `is_causal: false`, `use_cache: false` — so it has no
recurrent state for the assertion to reject, and it drafts a whole block in one
pass, which is the multi-step lookahead the paper needs. The VRAM half of this
blocker stands.)

**3. The working set does not fit a GPU cache.** The paper's design is
cache-centric: draft-time routing predictions fill a per-layer expert cache.
With 512 experts per layer and top-10, the union of experts over K=16 drafted
tokens is ~67 experts/layer = **7.8 GiB per step**, against a ~13 GiB expert
budget on a 24 GB card that must also hold the resident set, the KV cache and
the n-gram table's staging. At K=32 it is hopeless. Only a *streaming* design
survives, where the working set has to fit one layer at a time (10 experts ≈
24.9 MiB).

## What does transfer

The paper's real insight — *a cheap predictor of future routing converts a
sequential dependency into a schedulable one* — is exactly what exllamav3 is
missing. Its hot/cold expert placement is purely retrospective: a decayed hit
histogram and a quiesced, checkpoint-re-reading swap sweep every 128 steps
(`exllamav3/modules/block_sparse_mlp_cpu.py:574-710`).

And exllamav3 supplies the window the paper says does not exist. The paper
dismisses intra-token look-ahead as "a microsecond-level computation window". In
this engine the router fires a **whole layer** before those weights are needed —
and that layer is the slow host-computed one, so its wall clock becomes prefetch
bandwidth. The paper's own "expert explosion" argument, which says deeper
speculation is unaffordable, does not bite here: the host pays the weight read
once per expert regardless of how many rows are attached.

The predictor should therefore *not* be a quantized copy of the model. It should
be a routing-transition model (`P(routing_L(t+1) | routing_L(t))`), learnable
from a trace, costing well under a megabyte of VRAM. The enabling machinery
already exists: pinned-arena DMA straight out of the worker's memfd
(`exllamav3/model/moe_cpu_host.py:1455-1467`), a dedicated copy stream with
events, the fused `exl3_moe` kernel, and the `-1` sentinel that already tells
the host worker "this expert is on the GPU".

## Expected improvement

Per decode token, the offloaded part (48 layers x top-10 = 480 expert
evaluations), for a 24 GB card with 12 host threads:

| regime | gen5 x16 | gen5 x8 | gen4 x16 | gen4 x4 |
| --- | --- | --- | --- | --- |
| today — host computes the cold experts | 89.3 ms | 89.3 ms | 89.3 ms | 89.3 ms |
| stream the weights, transfer exposed | 22.0 ms | 43.2 ms | 79.3 ms | 187.1 ms |
| stream the weights, look-ahead hides 60% | 8.8 ms | 17.3 ms | 31.7 ms | 74.8 ms |
| floor — all compute on the GPU | 3.4 ms | 3.4 ms | 3.4 ms | 3.4 ms |

**~1.5–2.5x end-to-end TPOT on a gen5 x16 box, ~1.2–1.5x on gen5 x8, ~2.4x on
gen4 x16 — and nothing at all, a regression, on gen4 x4**, where the bus is
slower than the host. The model is optimistic by roughly 1.3–1.8x: run against
DeepSeek-V2-Lite it predicts a 2.5x offloaded-part gain, while the paper's own
implementation realized 1.5x end-to-end.

**The gen4 x16 column is the one most cards live in** (an RTX 4090 is PCIe 4.0
x16; 26.8 GB/s host-to-device, probed on the target box), and its number is
conditional in a way the others are not: with the transfer *exposed* that link
buys only 1.13x, so on a consumer card the look-ahead is not an optimization —
it is the whole difference between a win and nothing.

One gain needs no porting at all. Verifying K rows amortizes the host's weight
reads over those rows, so the **MTP head already in the checkpoint** takes the
offloaded part from 89.3 to 57.8 ms/token at K=4 (1.55x) just by running
`--mtp`.

## What is needed next

1. Measure the host cost of one offloaded expert against the bus cost of the
   same expert on the target box. This single number decides between two designs
   that differ by ~2x in achievable TPOT, and it decides whether the work is
   worth doing at all (it is not, on a gen4 x4 link). A calibration point from
   outside this repo: another engine for the same model on the same box measures
   **44 µs per offloaded expert** with a hand-tuned AVX2/AVX-512 pool, against
   the 186 µs this study assumes — the constant is engine-specific, so measure
   this engine's.
2. Export the router's top-k per layer per token. It is currently a local
   variable inside `BlockSparseMLP.forward` and never escapes the module; this
   is the missing primitive for any predictor, profiler or prefetch.
3. Measure routing autocorrelation (temporal and cross-layer) to see how good a
   predictor can be before building one.
4. Only then: a decode-time expert-streaming path, then the look-ahead on top.

A draft model is no longer on the critical path: `incoai/Qwen3.8-Flash-Next-DFlash2`
exists, is non-recurrent, and beats the model's own MTP on acceptance length
(GSM8K 3.26 vs 3.08, HumanEval 2.94 vs 2.52). Porting it into the fork is its
own piece of work — the capture patch today targets SGLang's `qwen4_exp.py` —
but it is no longer a reason to say no.

## Links

- Full study: [`doc/specter-feasibility.md`](./specter-feasibility.md)
- Budget calculator: `python eval/specter_budget.py --preset qwen3.8-flash-next --vram-gb 24`
  (pure arithmetic, no GPU; presets for the paper's three models too)
- Draft PR: <https://github.com/aand18/exllamav3/pull/3>
- Paper: <https://arxiv.org/abs/2511.14102> · artifact: <https://github.com/wwwwwf/specter_asplos>

The model and the CPU MoE path this studies are upstream exllamav3
(`turboderp-org/exllamav3`), not fork-local work.
