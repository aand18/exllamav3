# T54 — Inside the drafted round: MoE 83 %, mixer 13 %, and nothing else of size

One pass, 2026-11-07: `t54-A-roundprof-131k-20261107-223143`, `-Runner spec_decode`, production regime,
pin on, `EXL3_ROUND_PROF=1` (sample 1-in-8, warmup 32 rounds, every 32). `status=ok`, 997 s,
floors `min_free_vram 2921 MB`, `breach=0`. Arms: baseline **37.86 tok/s** (1 token/round), drafted
**56.47 tok/s** at `acc 2.04/2.86` → **3.04 tokens/round**.

Instrument: `EXL3_ROUND_PROF` (`exllamav3/util/round_prof.py`), two preallocated event brackets per layer —
around `self.attn.forward` and around `self.mlp.forward` in `modules/transformer.py` — plus a `perf_counter`
round wall in `Generator.iterate`. Floor (null pair) 0.018 ms, 4 dropped brackets in 2 368 samples.

## The split, by round type

The CSV (`_t4check/roundprof.csv`, 253 816 rows, 2 652 rounds) is split at the median of per-layer
mixer+MoE time, which separates the two populations the arms already name:

| | **drafted round** (5 rows) | **baseline round** (1 row) |
|---|---|---|
| round wall | 53.8 ms *(56.47 tok/s × 3.04)* | 26.4 ms *(37.86 tok/s)* |
| per layer | **1.121 ms** | **0.550 ms** |
| mixer | **0.143 ms** (12.8 %) | 0.100 ms |
| MoE block | **0.932 ms** (83.2 %) | 0.265 ms |
| outside both brackets | **0.046 ms** (4.1 %) | 0.185 ms |

**The MoE owns the drafted window.** 0.932 of 1.121 ms per layer, against 0.143 for attention/GDN. The
mixer is not the growth term, which is consistent with single-stream decode being flat across context
(37.30 at 4 k → 39.96 at 261 888, `t52`): nothing in this model's attention path scales with context here.

**A correction to the mid-pass reading.** The live report line mixed both populations, so the "43 % of the
layer sits outside both brackets" I quoted from it was an artifact. Split properly, the unbracketed remainder
in a drafted round is **4 %** — the two brackets cover the layer almost completely. The remainder is only
large in the *baseline* population, where the round is short enough that the head and sampling are a big
fraction; it is not a hidden phase in the drafted path.

## What is inside the MoE's 0.93 ms

The bracket wraps `self.mlp.forward`, which in the split path spans `cpu_split_submit` → `cpu_split_combine`,
so it contains both the GPU expert kernel and the exposed wait for the CPU partial. `t52` measured that wait
at **0.207–0.316 ms** per layer (at 262 144). Taking the middle:

```
MoE block          0.932 ms
  CPU wait        ~0.260        (T52, SPLIT_PROF)
  GPU MoE         ~0.672
mixer              0.143
```

So of the GPU-only work per layer (0.815 ms), the MoE is **82 %** and the mixer **18 %** — the same verdict
with or without the wait.

**And the GPU side is nowhere near its bandwidth.** A 5-row window routes 5 × 10 = 50 picks; the distinct
experts per layer are ~30–40, each a 1.95 MB blob, so the MoE reads roughly **60–78 MB per layer**. At
0.672 ms that is **~90–116 GB/s effective**, against a 4090's ~1 TB/s. The kernel is **latency- or
occupancy-bound, not bandwidth-bound**: at this batch each expert serves about one row, so the grouped
kernel is a stack of tiny GEMVs. That is the same physics Strata's grouped kernel has, and it is why
coverage — which changes *which* experts, not how many rows each serves — cannot help.

## What this rules in and out

- **Out:** the 48-layer graph. `t53` measured the existing graphs at **+1.5 %** median, so launch overhead is
  small and a whole-model graph cannot recover more than a couple of percent.
- **Out:** attention as the target. 12.8 % of the layer, flat with context.
- **Out:** coverage/placement as a decode lever (T51), and lower bitrate as a policy (operator decision).
- **In:** the MoE's GPU kernel at ~10× below HBM peak. Levers there are occupancy (more rows per expert is
  not available at batch 1), kernel shape for tiny row counts, and hiding the 0.26 ms CPU wait inside the
  block. The first of those is a kernel question, not a placement question.

## Caveats

- `-Runner spec_decode` forces `EXL3_MOE_CPU_SWAP=0`, so this profile is the **static** placement — the
  served shape, not T51's adaptive arm.
- The distinct-expert estimate (30–40) is arithmetic on top-10 × 5 rows, not measured; the bandwidth figure
  scales with it linearly. Direction (two orders below peak) survives a 2× error.
- One pass, `-dr` not applicable to `spec_decode`; the phase medians are means over ~1 322 rounds per
  population, so they are far better conditioned than a tok/s cell, but the 4 % remainder is small enough
  that it should be read as "nothing else of size", not as "exactly 4 %".
