# Don't fuse an MMA-optimal op into a SIMT loop

## Problem
Fusing two launches into one kernel can make the result SLOWER
when the fused math loses access to Tensor Cores: a torch/cuBLAS
MMA contraction replaced by per-element SIMT FMAs regresses even
though a launch and a temporary disappear.

## Rule
- Before fusing X into a custom kernel, classify X's compute: MMA
  (GEMM/bmm, min dims >= 16/8/16) stays out, or the fused form must
  keep an MMA-shaped dot. Fuse only launch-/traffic-bound pieces
  (gathers, elementwise chains, reductions) around it.
- Estimate first: launch saved (~10us) vs compute moved. A 20us
  bmm is not worth fusing into 300us of SIMT dots.

## Evidence
- 2026-09-30: fused tail-QK proposal (bmm `Qh @ Kt` into
  `_kvarn_online_tail_kernel`) REJECTED at design review:
  per-program scores would need RPAD(<=512) sequential HD-dots
  (~131k SIMT FMAs, ~0.3ms) vs the ~20us MMA bmm; QPK=4-8 < 16
  blocks the MMA reshape. No code written (`9cbbe6d` docs).

## Scope
Kernel fusion proposals. Write the arithmetic before the code.
