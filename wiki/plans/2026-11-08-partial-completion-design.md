# T57 — The one lever left: partial completion in the CPU handoff

Status: design not started. Everything else is closed by measurement. Written 2026-11-07.

## What the numbers say now

After T53–T56 the drafted round is fully attributed:

| | ms/layer | % of layer |
|---|---|---|
| mixer (attention/GDN) | 0.157 | 13 % |
| MoE `sub` (enqueue) | 0.014–0.36 (heavy tail) | backpressure |
| MoE `ker` (GPU kernel) | **0.077** | 7 % |
| MoE `cmb` (wait for CPU) | **0.39–0.50** | **35–45 %** |
| rest | 0.064 | 6 % |

Two facts make this interesting rather than merely additive:

1. **The job is *early* by design.** `cpu_split_submit` (at `block_sparse_mlp.py:1160`) runs before the
   branch chain, so the 0.077 ms GPU kernel and the 0.58 ms of CPU compute start at the same moment. The
   submission is not the bottleneck.
2. **The CPU finishes its job at 0.58 ms into a 1.08 ms layer.** It *should* be hidden. The 0.39 ms
   `cmb` wait is therefore **jitter in the CPU's finishing time**, not throughput — the pool is busy 92 %
   and computing 71 % (`t52`), with `gap 0.08 / spin 0.21 / compute 0.69 ms` per job.

So the exposure is the **last expert's finish time**: the layer waits for the whole job, and the job's
completion is the maximum of ~37 per-expert compute times plus staging. Any mechanism that lets the GPU
consume an expert's rows as soon as *that* expert lands is worth, at most, the spread of those 37 finish
times — plausibly a good part of the 0.39 ms.

## Why this is exactly the Strata shape

Strata runs **one job per distinct expert**, with every token routed to it attached
(`expert_source.cpp:2205-2233`), and splits each expert's rows across all threads so one expert streams at
the whole pool's bandwidth (`pool.hpp:166-171`). Its own header quantifies the result: of 19.035 ms of CPU
expert work per token, overlap hides **1.055 ms — 5.5 %** (`expert_cache.hpp:3-7`). That is a smaller prize
than ours would be, because our wait is larger relative to the round (23 % versus ~20 %).

The honest caveat: Strata's overlap number is small, and a 5.5 % figure is below the point where I'd bet a
campaign on an estimate. The reason to pursue it here is that our exposure is measured at 0.39 ms/layer with
a *mean* `sub` of 0.36 ms of enqueue backpressure, i.e. the system is visibly queue-limited in bursts, and
the same redesign (per-expert jobs, so the ring can run ahead of the GPU) attacks both.

## The redesign, in outline (to be specified in detail)

1. **One job per distinct CPU expert**, instead of one job per layer. Each job carries its own rows, so the
   pool can pick up a 48-entry queue rather than a 1-entry one and the stager can run ahead.
2. **A completion flag per expert**, published by the worker as each expert's output lands; `cpu_split_combine`
   waits per expert rather than on the whole job.
3. The GPU side then consumes the routed sum in the order the experts land — for a 5-row window, the first
   experts to land typically cover most rows, so the combine pays only the straggler.

Rejected alternatives, for the record: starting the job earlier (it already starts at the top of the block);
a bigger ring (256 slots against 48 jobs/round — the ring is not the limit, the stager's per-slot free-flag
spin is); more threads (the pool is already at the bus's 48 GB/s ceiling).

## What it costs and what it is worth

- Kernel + protocol work in `moe_cpu_host.py`, `cpu/moe_handoff.cu` and the `ext` bindings; the worker's
  job shape changes from "layer" to "expert". Expect a wheel rebuild on the Windows box and a full test
  pass; **no** config or checkpoint change, and it is default-off.
- Best case: recover most of the 0.39 ms median wait ≈ **20–35 % of decode** — the largest single gain left
  on the table, and the first since drafting.
- Why it is not free: it makes the routed sum observable part-way through a layer, which is what makes it a
  protocol change rather than a tuning one.

## Sequencing

1. Design (this task, cheap seat) — per-expert job shape, flag layout, the en-queue order that lets the
   stager run ahead, and where partial consumption changes numerics (reduction order per row).
2. Implement behind `EXL3_MOE_PARTIAL_COMPLETION`, default off; existing `EXL3_MOE_CPU_SPLIT_PROF` and
   `EXL3_ROUND_PROF` brackets must both read the new path so the gain is measured on the same instrument.
3. Arm at 131 072, same session, `-dr 3`, pre-registering `cmb` median < 0.20 ms and decode ≥ 60 tok/s.
