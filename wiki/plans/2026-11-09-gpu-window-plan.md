# Plan: the per-token GPU window (T59)

Derived position (unchanged from T58, same bases):

| per token | ours | Strata | basis |
|---|---|---|---|
| total | 17.05 ms | 10.16 ms | ours T52+arithmetic; Strata paper/config |
| CPU tail | 3.9 ms | ~3.2 ms | ours measured (cmb); Strata modelled |
| **GPU window** | **13.15 ms** | **~6.0 ms** | derived |

The CPU wait is roughly at parity with Strata's. **The 2.2× gap is in the GPU window.**
Nothing in this plan touches the CPU handoff again; coverage, placement, and per-expert
completion are all closed.

## Experiment 1 — roofline the window (cost: one short trace, no build)

Question: is the window latency-bound (kernel time flat in rows) or launch-bound (gaps
between kernels)?

- `nsys` (or torch profiler if nsys is absent) trace of 64 drafted rounds at the production
  regime in bench/windows. Extract: kernel list + duration, and the inter-kernel gap share
  of the 0.53 ms moe-phase median (T56/T58).
- Rows-scaling apples test: same kernels at rows 1 vs 5 — expert-kernel time linear in rows
  means throughput-bound (weights are the cost, batchable); flat means latency-bound
  (launches + small-GEMV spin-up are the cost).

Pre-register: flat rows-scaling launches Experiment 3; throughput-bound with large gapped
launches launches Experiment 2; throughput-bound and batched launches Experiment 4.

## Experiment 2 — one graph for the MoE block (kernel-lite)

The block does per-layer: sub enqueue → expert kernels → cmb wait/fold. Strata runs one
48-layer graph with mapped-flag doorbells. Steps: CUDA-graph capture of the fused issue +
collect pair (issue kernel, doorbell wait, fold kernel) — no new GEMMs, existing kernels.
Pass mark: moe-phase median down ≥ 0.10 ms/layer at 131k, tok/s with same-session reference.

## Experiment 3 — grouped expert kernel, rows as the batch dimension

The window's per-expert GEMVs are small and latency-bound; a grouped kernel reads each
expert's weights once for all 5 rows (weights reused across the drafted window — the
MMVQ idea). Effort: new kernel + wheel build. Pass mark: expert-kernel ms/row down ≥ 30%
at rows 5, no regression at rows 1.

## Experiment 4 — weight reuse, only if the roofline says bandwidth

If weights dominate, the Strata trick applies: read once per window, apply to all rows.
This is the same surgery as 3 from the demand side; roofline decides.

## Explicitly out

- Partial completion at any granularity (T57's regression is the mechanism proof).
- Placement, coverage, 2 bpw (closed T51/operator).
- Slots / ring tuning (closed T58).
- A reducer that only narrows round variance; the right tail helps variance, not the median.

## Order and stopping rule

1 → (2 | 3 | 4) → report per experiment with the post gate. Stop after the experiment that
moves the window ≥ 1.5 ms/token; anything smaller is inside the confound band and the
remaining arc is the CPU-tail overlap, which is a different plan.

Basises carried through: per-token ms for the window, per-layer ms for kernels, tok/s
131k spec_decode for the cross-check, never mixed.
