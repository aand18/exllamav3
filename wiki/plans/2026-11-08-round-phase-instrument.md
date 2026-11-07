# A round-phase instrument for the drafted decode path

Design only, 2026-11-08. Nothing was run: no GPU task, no pass, no restart, and no file outside this
one was written. The question is T51's open one — *of the ~52 ms round, how much is the layer waiting
on the CPU expert partial versus how much is GPU compute* — since coverage is ruled out
(`T51-adaptive-placement-at-262k.md`, "Verdict") and the fork has no per-stage instrument
(`/tmp/opencode/fork-strata-gap-ledger.md:258-264`, G11).

**Bases.** A *round* is one `Generator.iterate()` at batch 1: draft, verify 5 rows (1 + 4), sample,
accept. `52 ms` and `3.05 tokens/round` are the quoted corpus figures (3.05 / 0.052 = 58.7 tok/s,
against the served 58.54). Times are **stream time** (event timestamp to timestamp; this *includes*
GPU idle while the host is late) unless marked **wall**. Arithmetic is marked *(arith)*.

## 1. What already exists, and what each knob measures today

| knob | read site | measures today | cannot answer |
|---|---|---|---|
| `EXL3_SPLIT_PROF` | `block_sparse_mlp_cpu.py:14`; brackets `:1105-1106` (fused issue), `:1165-1169` (fused collect) | stream time of the issue enqueue and the collect wait+readback; percentiles per ~2048 brackets (`:25-40`) | nothing outside those two brackets, and no round denominator — a percentile has no share |
| `EXL3_MOE_CPU_PROF` | `moe_cpu_host.py:195` → `moe_mul1.cpp:3933-3938` | worker **µs per job** by phase: `gemv_gu 318.1 + gemv_d 164.7 = 482.8 µs/job` (`t44-prof-phases-20261105-141334`), every 512 jobs | **jobs per round**, so worker busy-ms/round is not derivable |
| `EXL3_MOE_SUBMIT_PROF` | `moe_cpu_host.py:1429-1447` | one bracket around `_issue_compute`+`_collect_compute` back to back — the whole-layer single-phase `submit()` (`:1417-1451`) | the split path, which is what the served box runs (`EXL3_MOE_CPU_SPLIT`, `config.py:46-48`); inert here |
| `EXL3_MOE_HANDOFF_PROF` | **`exllamav3_ext/cpu/moe_handoff.cu:437`** (a C++ read site — a Python-only search misses it) | every 64 jobs, **worker-side gap / spin / compute ms per job** (avg and max), empty-job count, cpu-assign per row (`moe_handoff.cu:542-550`) | has no round denominator, but it *does* say whether the worker is starved (large gap) or saturated (gap ≈ 0) — the first thing to run |
| `EXL3_MOE_TIER_STATS` | `block_sparse_mlp_cpu.py:543-574` | coverage, swaps, hook cost — a count instrument | time |

Both quantities that matter are half-instrumented: the exposed wait exists but has no round to be a
fraction of, and the worker cost exists as µs/job with no job count attached to a round. Those two
gaps plus the GPU side are the design.

## 2. The phase list

R = round, M = per module, L = inside the split MoE. Adjacent rows **share** an event, so the brackets
tile the timeline and every figure is the difference of two adjacent timestamps, not a subtraction of
independent measurements.

| id | phase | start → end | basis |
|---|---|---|---|
| R0 | round wall | `generator.py:855` `iterate` → wrapper end | wall |
| R1 | **drafter** (DFlash2 block) | `generator.py:980-986` → `:1592` | stream |
| R2 | **drafter→verifier host drain** | `generator.py:1592` → `:1632` | stream + wall |
| R3 | verify prologue: block table + staging H2D | `generator.py:1874` → `:1969` | stream |
| R4 | **verify forward** | `generator.py:1972` → return of `model.forward` | stream |
| R5 | sample / accept / reject, ending in the drain | `generator.py:2000` → `:2137` | wall |
| M1 | `prepare_for_device` | `model_ls.py:371` → `:377` | stream |
| M2 | module `forward`, **one bracket per entry in `fwd_modules`** | `model_ls.py:377` → next iteration | stream |
| B1 | attn hyper-connection mix + input norm | `transformer.py:177-187` → `:188` | stream |
| B2 | **mixer** (QSA full attention or GDN) | `transformer.py:188` → return | stream |
| B3 | attn residual / hc apply | `transformer.py:191-200` → `:202` | stream |
| B4 | mlp hyper-connection mix + norm | `transformer.py:203-215` → `:216` | stream |
| B5 | **MoE block** | `transformer.py:216` → return | stream |
| B6 | post norm / deferred residual / layer scalar | `transformer.py:217-230` → `:232` | stream |
| L1 | router pre-norm, routed pre-norm, latent_in | `block_sparse_mlp.py:1090` → `:1119` | stream |
| L2 | **router** (`routing_fn`, top-k, weights) | `block_sparse_mlp.py:1096` → `:1146` | stream |
| L3 | **CPU expert issue** | `block_sparse_mlp.py:1160` → `cpu.py:1080-1114` | stream |
| L4 | **GPU-resident experts** (`bc.run_bszN`) | `block_sparse_mlp.py:1449` → return | stream |
| L5 | **shared experts** (placed to cover the CPU job) | `block_sparse_mlp.py:1457-1459` → return | stream |
| L6 | **combine = wait for the CPU partial** | `block_sparse_mlp.py:1463` → `cpu.py:1157-1176` | stream |
| L7 | latent_out, post norms, output reduction | `block_sparse_mlp.py:1467` → `:1517` | stream |

(`cpu.py` = `exllamav3/modules/block_sparse_mlp_cpu.py`.)

Two phases beyond the brief, because both are large and neither is in the layer loop:

- **R2 is a full pipeline drain.** The drafter's ids must be on the host before the verifier can be
  built: `generator.py:1631` copies `new_ids` from CUDA into a pinned host tensor with a blocking
  `copy_`, and with `dynamic_draft: true` (`state.json`) `:1610-1611` already calls `.cpu()`. The GPU
  cannot be busy while the host waits there, so **no host optimisation can overlap the drafter with
  the verifier** — a structural ceiling, not a measurement.
- **The PLE n-gram lookup** (`ple.py:382` → `ngram_embedding.py:352` → `row_table.py:687-735`) is its
  own entry in `fwd_modules` (`qwen4_exp.py:250-267`) and stays inline at decode size
  (`ngram_embedding.py:308`: "decode-sized inputs stay inline: their 16-row gathers are already
  parallel preads"; `row_table.py:716` `_stage`). The table streams from disk unless `-ngr`
  (`eval/spec_decode.py:191-195`), which this regime does not set. It lands in M2 automatically; named
  here so a large M2 on the PLE module is not read as a block.

The KV/cache path is not a separate phase and cannot be made one: cache append, the QSA indexer and
the flash-decoding kernels are inside the mixer's captured graph (`attention.h:1-15`). Only B2 bounds
it.

## 3. How each phase is bounded

**One event chain per round, boundaries shared** — not one pair per phase: a chain of `N+1` events for
`N` phases tiles exactly and yields one overhead constant. Per sampled round: M2 gives the 48 blocks
plus the PLE entries, embedding/expand-streams, mixer and head (read the actual list from
`fwd_modules`, do not assume it); L3–L6 add 4 boundaries per split layer. Counts *(arith)*:
M2 ≈ 55, L3–L6 = 192, R = 8 → **255 events per sampled round** (B1–B6 optional if the intra-block
split is wanted: +192).

**48×N events is affordable**, if the pool is preallocated and only `record()` is called. Do **not**
copy `_sprof_wrap`, which builds two `torch.cuda.Event` objects per bracket (`cpu.py:18-19`) — that
allocation is the expensive part. At ~1.5 µs host per `record()` and ~0.5 µs stream per marker
*(arith)*: 255 × 1.5 µs = 0.38 ms + 255 × 0.5 µs = 0.13 ms per sampled round; sampling **1 round in
8** gives 0.048 + 0.016 ms per average round = **0.12 % of 52 ms**; every round would be ~1 %.

Do **not** sample one layer in four. The load-bearing prediction is a *shape across the layer stack*
(a serial worker falling behind shows as a monotone rise of L6 with layer index), and a 1-in-4 layer
sample aliases with the 48-layer repetition. Sample rounds, not layers: 19.2 rounds/s *(arith)*, so
1-in-8 over a 400 s pass is n ≈ 960 rounds; a 60 s probe gives n ≈ 115, per-layer n ≈ 115.

**Subtracting overhead.** Record one null pair (two events back to back, nothing between) at the top
of each sampled round; its median is the per-boundary floor — and it is where the spinning wait
kernel's backoff lands (`moe_handoff.cu:27-29`, up to 2048 ns; with `EXL3_MOE_MEMOPS=0` from
`state.json`'s env the wait *is* the spin kernel, `moe_handoff.cu:42-58`). Report raw and
floor-subtracted columns and never quote a phase whose corrected value is under twice the floor.

**Where numbers are read.** Top of the *next* sampled round, `.query()` before `.elapsed_time()` (the
existing pattern, `cpu.py:27`). The round already ends in a synchronise (`generator.py:2137`), so the
instrument adds **no new synchronisation**. `perf_counter` gives R0, so every stream figure is
reported against wall time and the difference (host time the GPU never saw) is its own named line.

## 4. The graph problem

**Where capture happens: one entry point.** `graph.cu:63`
`cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal)` inside `Graph::capture_begin`
(`:51-72`), which first syncs the device (`:55`) and creates a private capture stream (`:62`). Replay
is `cudaGraphLaunch(graph_exec, stream)` on the caller's stream (`:252`, from `:201`).

| module | site | when |
|---|---|---|
| `BC_Attention` | `attention.cpp:885` eager, `:892-897` capture | per `(bsz, q_len, regime)` slot (`:877`) |
| `BC_GatedDeltaNet` | `gated_delta_net.cpp:355-380`, `:603-626` | per slot |
| `BC_GatedMLP::run_bszN` — **the shared expert at 5 rows** | `mlp.cpp:87-112` | per `num_tokens` |
| `BC_GatedMLP` bsz 1 | `mlp.cpp:191-200` | not this regime |
| per-expert reconstruct | `exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:528-544` (`GraphCapture` at `:539`) | only above `MAX_BSZN = 8` |
| DSV4 / MLA / shared-KV | `dsv4_attn.cpp:599`, `mla_attention.cpp`, `shared_kv_attention.cpp` | not this model |

**Inside a capture:** B2 in full (q/k/v projections, q/k norm + RoPE, cache append, flash-decode,
o_proj — enumerated in `attention.h:1-15`) and L5. **Outside:** the fused MoE decode kernel
(`exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:70` `run_bszN` has no `GraphCapture`; `:82` says "routing tensors are read directly, so no per-bsz statics or graphs"), the collect (`moe_handoff.cu:295-315`),
the router, the PLE lookup.

**So events are usable, and the reason is structural: the capture boundary coincides with a
Python-visible call.** Every boundary in §2 is a point where control is back in Python, and every
capture begins and ends inside one C++ call. Recording immediately before and after
`module.forward` (`model_ls.py:377`) is outside the capture by construction — the same reasoning the
tree already uses when it needs an event or a D2H copy next to a capture
(`block_sparse_mlp_cpu.py:548-554`, `moe_lookahead.py:511-517`). The primary answer is **events
outside the capture boundary**; no separate non-captured path is built.

Two rules must be in code. (1) **Assert `not torch.cuda.is_current_stream_capturing()` at every record
site** — an event recorded inside a thread-local capture is illegal and invalidates the capture, which
surfaces as a wrong forward pass rather than an error; the assert makes a misplaced boundary loud.
This is precisely how a naive design here fails silently. (2) **Skip the first 64 rounds**: capture
and its eager precursor cost a first and second call per slot (`attention.cpp:885,892`) and
`capture_begin` syncs (`graph.cu:55`). `Model.warmup` already pays most of them
(`model.py:214-254` runs a decode-sized and a multi-token pass).

**Fallback if the assert fires:** `EXL3_GRAPHS=0`, which sends every graphed site down its eager
branch (`graph.cu:8-20`; `attention.cpp:885` `|| s.graph->disabled`). That is a *different regime* —
it puts back the per-site launch overhead the graphs remove — so it is a separately labelled arm and
its numbers are never mixed. It is only needed to subdivide *inside* the mixer, which is worth doing
only if B2 turns out to dominate.

## 5. The one number, and why it is direct

The collect is the first thing enqueued *after* the GPU's own expert work for the layer
(`block_sparse_mlp.py:1449`, then `:1459`, then `:1463`), and its first instruction is
`flag_wait_on(stream, done_flag, seq, abort_flag)` (`moe_handoff.cu:310`). So `elapsed(L6)` **is** the
time the stream spent blocked on the worker's `done` flag plus the readback-add kernel:

- `T_wait(L) = elapsed(L6)` — CPU lateness for layer L, in ms, directly;
- `T_gpu(L) = elapsed(L4) + elapsed(L5)` — GPU expert compute for layer L;
- the number: `Σ_L T_wait / round_wall` **and** `Σ_L T_wait / Σ_L T_gpu`, per round and per layer.

The host-side cross-check is free and needs no new code: read `host.v_jobs_head[0]` at the round
boundary (`moe_cpu_host.py:1571` uses it as worker progress; the child publishes it at
`moe_handoff.cu:475`). Its difference across a round is **jobs completed per round** — the missing
multiplier for `EXL3_MOE_CPU_PROF`, giving worker busy-ms/round = jobs/round × 482.8 µs *(arith)*.
Two independent routes to one quantity: if they disagree, one basis is wrong, and that is the finding.

## 6. Pre-registered prediction for the first profile pass

Written before any run, all as shares of the ~52 ms wall. Per layer, 52 / 48 = **1.083 ms** *(arith)*;
the t44 anchor is 482.8 µs/job *(arith sum of the printed phases)*, and if the pass shows more than one
job per layer, P1/P2 scale with it.

- **P1 — the host is hidden.** `Σ_L T_wait ≤ 15 %` (≤ 7.8 ms); median per-layer `T_wait ≤ 100 µs`.
  *Miss at* `Σ_L T_wait ≥ 40 %` (≥ 20.8 ms) or median `≥ 440 µs`.
- **P2 — the layer loop is the round, but not its host.** R4 is 40–60 % (20.8–31.2 ms). *Miss at*
  R4 ≥ 75 % (≥ 39 ms): then the layer loop alone is the round and one phase inside R4 must reach 40 %.
- **P3 — drafting is not the round.** R1 + R2 ≤ 25 % (≤ 13 ms). *Miss at* ≥ 40 % (≥ 20.8 ms): then the
  round is a drafting instrument and the 5-row verify window is the wrong frame.
- **P4 — attribution closes.** M2/L brackets ≥ 90 % of R4, named phases ≥ 95 % of R0, residual printed
  as its own line. The prior note leaves 5.3 ms/token unattributed = **16.2 ms/round** *(arith)*, so a
  residual of that size is the expected outcome if the instrument misses the real cost; the residual is
  the measurement, not a rounding entry.

**What would mean "host bytes dominate" is wrong.** P1 and P2 and P3 all holding: wait ≈ 0, verify
forward 20–31 ms, round 52 ms — 10–20 ms outside both the host and the layer loop, i.e. bounded by the
drains and the drafter. That explains T51 for free: with the CPU tail already hidden, moving 25 % of the
picks off the host (22 → 57 % coverage) could not have bought anything, because what it shortens was
never on the critical path. Conversely, **P1 missing with `T_wait` rising monotonically over the 48
layers** is the worker-queue signature and the first real evidence for the host-bytes model; P1 missing
with a *flat* large `T_wait` is a fixed per-layer handoff cost (~`T_wait` × 48) — a different fault
with a different fix.

## 7. Cost, risk, and putting the box back

**Expected slowdown** ≤ 0.2 % at 1-in-8 (§3), ≤ 1.5 % at every round. Both are *predictions*: the
instrument's own cost is unmeasured until the same-session A/B, which is exactly the defect the
coverage instrument carries (`fork-strata-gap-ledger.md:442`, row 4.8 — no row of 232 is `TIER_STATS=0`).
Run it: `EXL3_ROUND_PROF=0/1`, ≥ 3 repeats, ABBA. Until then quote phase *shares* only, never tok/s
from a profiled pass.

**At 262 144 without changing placement: yes.** The instrument is env-only. `run_pass.ps1` already
ships `-Prof`, which sets four knobs by env (`:383-386`) and touches neither `state.json` nor any
argv; `EXL3_ROUND_PROF` goes in through `-Env` the same way. Split 381, threads 16, Q4 `"4,4"` at
262 144 and DFlash2 window 4 are untouched, and `-Prof` is documented as "diagnosis, not tok/s"
(`bench/windows/README.md:73`).

One basis to state in every table: `-Runner spec_decode` forces `EXL3_MOE_CPU_SWAP=0`
(`run_pass.ps1:351-354`), so the profiled placement is the **static** tail split — the served path
(`fork-strata-gap-ledger.md:63-65`), but *not* the arm T51's adaptive numbers came from. A second arm
with `-Env "EXL3_MOE_CPU_SWAP=1"` measures adaptive placement on the same instrument if the sweep is
ever wanted again.

**Recipe.** Gate the box first (`verify_task.py --pre`; ask before the first GPU task — this box serves
live traffic). Cheap gate: one 60 s probe at 32 768 to confirm the columns exist and total. Then
`run_pass.ps1 -Tag roundprof-262k -Runner spec_decode -MaxLength 262144 -dr 2 -Prof -Env
"EXL3_ROUND_PROF=1,EXL3_ROUND_PROF_SAMPLE=8"` (20–25 min per `bench/windows/README.md:60`). Emit
`phase,sample,layer,elapsed_ms,floor_ms` rows to a CSV beside the pass so every table in the report is
generated from the CSV rather than typed from a log line.

**Restore: nothing on the box.** The pass is its own `eval/spec_decode.py` process; the served listener,
its port and its API key are untouched, and `run_pass.ps1` already restores the frozen expert/token
profile before every run (`:360-381`). The only thing left behind is the CSV.

**Risks, in order.** (1) The instrument's own cost, until the A/B exists. (2) `EXL3_GRAPHS=0` being
quietly used for the whole pass — then the numbers describe an unreal regime; forbid it in the primary
arm. (3) The floor: with `EXL3_MOE_MEMOPS=0` the wait is a spinning kernel whose backoff reaches
2048 ns, so a sub-µs `T_wait` is indistinguishable from zero — a real finding ("hidden by ≥ 48 × 2 µs")
but not a rate. (4) Event-pool growth if events are allocated per bracket instead of per
`(layer, phase)` and reused; that one mistake costs more than the thing being measured.
