# T56 — The MoE is a handoff problem: 85 % waiting, 8 % kernel

One pass, 2026-11-07: `t57-A-moesplit-131k-20261107-232229`, `-Runner spec_decode`, production regime,
pin on, `EXL3_ROUND_PROF=1` with three brackets inside the split MoE. `status=ok`, 993 s, floors
`min_free_vram 2687 MB`, `breach=0`. Arms: baseline **38.33 tok/s**, drafted **56.07 tok/s** at
`acc 2.00/2.82` = **3.02 tokens/round**.

## The decomposition

Drafted rounds only (n = 1 323 of 2 654), two robust statistics because the distribution is skewed:

| phase | mean | median | p90 | what it is |
|---|---|---|---|---|
| mixer | 0.157 | 0.099 | 0.108 | attention / GDN, outside the MoE |
| **MoE block** | **1.006** | **0.528** | 1.028 | the whole `mlp.forward` |
| ├ `sub` | 0.364 | 0.014 | 0.020 | enqueue the CPU job |
| ├ `ker` | **0.077** | **0.064** | 0.123 | the fused GPU expert kernels |
| ├ `cmb` | **0.500** | **0.391** | 0.853 | wait for the CPU partial + fold it in |
| └ rest | 0.064 | 0.059 | — | router, latents, shared expert, norms |

Both columns close: **medians 0.014 + 0.064 + 0.391 + 0.059 = 0.528**, **means 0.364 + 0.077 + 0.500 + 0.064 =
1.005**. The means are the ones that sum to the round (48 × (0.157 + 1.006) = 55.8 ms, against the measured
56.07 tok/s at 3.02 tokens/round ≈ 53.9 ms).

## What it says

**The GPU expert kernel is not the cost.** 0.064–0.077 ms per layer — 8–12 % of the MoE, 6–8 % of the layer.
Two fused cooperative launches over every routed slot are already efficient; there is nothing there to win.

**The handoff is the cost.** Take the medians, which describe the typical layer: `cmb` alone is **0.391 of
0.528 ms (74 %)**. Take the means, which describe the aggregate: `cmb` 0.500 plus `sub` 0.364 is **85 %** of
the block. Either way, the MoE is waiting on the CPU pool, not computing on the GPU.

**`sub` has a heavy tail and that is a finding, not noise.** Its median is 0.014 ms but its mean is 0.364 ms:
a minority of layers block for a long time at enqueue. That is backpressure — the job ring is full because
the worker has not drained it. It is the same saturation the worker profiler reported from the other side
(`t52`: gap 0.08, spin 0.21, compute 0.69 ms/job, busy 92 %), and it is the mechanism that turns "coverage"
into a number that does not move: the CPU pool is the resource, and it is already running at 42–45 GB/s of
a ~48 GB/s bus (`t44`).

## Consequence for the levers

- **Kernel work on the MoE: closed.** 0.077 ms/layer, 6 % of the layer. Even making it free buys ~6 %.
- **The 48-layer graph: closed** (`t53`, +1.5 %).
- **Attention: closed** (0.157 ms, flat with context).
- **The only thing of size left is the CPU expert handoff** — 0.85 of the MoE's 1.01 ms — and the CPU is at
  ~90 % of its memory bus while doing it. That leaves exactly two shapes: **fewer CPU bytes** (the 2 bpw tail,
  dropped by operator decision; or more GPU residency, refused by the allocator at 262 k) and **better
  overlap** (start the job earlier, or return partial results so the GPU consumes the hot rows first).
- The overlap route is the only one not already closed by measurement or by policy, and it is a handoff
  redesign rather than a kernel or a placement change.

## Caveats

- The `sub` mean is dominated by a tail; a per-layer histogram (not in this CSV) would show whether it is a
  few layers or a recurring subset. The conclusion does not depend on it — `cmb` alone is 74 % by median.
- `-Runner spec_decode` forces `EXL3_MOE_CPU_SWAP=0`: static placement, the served shape, not T51's
  adaptive arm.
- The `moe` mean (1.006) is larger than T54's (0.932) because this pass caught more long-context rounds in
  its drafted population; the *ratio* inside the block is the stable quantity across both passes.
