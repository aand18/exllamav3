# Serve is parallelism-bound, not traffic-bound

## Problem
Intuition says less memory traffic = faster decode serve. Two cuts
disproved it: quad unpack (halved V traffic, +0.0%), MMA floor
(Tensor Cores, +0.0% hot-cache). Then hierarchical subgroups with
flat 32 groups LOST 4% despite 16x less partials traffic.

## Rule
- Serve-kernel changes must preserve CTA parallelism first, traffic
  second. Grid `(kvh, G)` needs enough CTAs to fill 144 SMs
  (rule of thumb: >= 512 CTAs, i.e. groups cap 128, never 32).
- Traffic cuts that cost CTAs are net negative until Nsight/Kineto
  proves the traffic was the binding constraint. Our serve is
  dequant-load-latency-bound: it needs many CTAs to hide, not fewer
  bytes per CTA.
- Attention is O(n) per step, period: the slope game is constants
  (parallelism + fusion), and our engine's speed makes our constant
  overhead visible where Bee's slower engine hides it.

## Evidence
- Quad V unpack `fed57f0`: twin-exact (9.2e-5), tg 34.7 vs 34.9.
- MMA floor `561588c`: hot-cache A/B 39.4 -> 39.4; reverted.
- Hierarchical flat-32: 34.0 -> 32.5 hot (-4%); cap-128
  (`3647f54`): 34.0 -> 35.1 (+3%), twin maxabs 1.9e-5.
- Kineto @8k: serve 0.74ms of 25ms step; GPU half-idle.

## Scope
Serve/combine/merge kernel work. Check CTA math before any grid or
tile-shape change.
