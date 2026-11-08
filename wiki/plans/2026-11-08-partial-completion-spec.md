# T57 spec — per-expert partial completion (decode only, default-off `EXL3_MOE_PARTIAL_COMPLETION`)

## 0. Decision in one paragraph

One `MoeJob` per distinct CPU-routed expert per layer (kind `MOE_JOB_KIND_COMPUTE_EXPERT=3`), sharing one input slot per layer, writing a compact per-expert output segment into that slot, publishing a per-job flag in a new `expert_done[MOE_JOB_RING]` bank. `cpu_split_combine` enqueues `wait+fold` pairs in expert-ascending order on the stream. Prefill (`submit_prefill`, `moe_cpu_host.py:1980`) and the whole-layer path are untouched. Production behaviour is byte-identical when the flag is off.

## 1. New job granularity

- `MoeJob` keeps its layout and size (`(7+256+1)*4 = 1056 B` — arithmetic from `moe_cpu_host.py:48-51`, struct at `moe_handoff.h:77-88`). Repurposed fields: `experts[0]` = worker-local expert id; `rows` = layer row count (sel scan bound); `topk`, `layer`, `slot`, `seq`, `seq_hi` survive; `prev_seq` unused (0). No new struct fields; `MOE_JOB_BYTES` unchanged (`moe_cpu_host.py:50`).
- Output: each job writes ONLY its expert's contribution into a compact segment of the slot's out region (not a dense full-sum buffer). Slot layout becomes `[out: cap_rows*topk*Ho f32][rowidx: cap_rows*topk i32][segtab: E_cpu*(off,n) i32]`, sized in `ensure_started` (`moe_cpu_host.py:1267-1271`) and mirrored in the worker offsets (`moe_handoff.cu:429-433`). Segment offsets are a deterministic prefix sum over per-expert counts from a sel scan, so the GPU combine needs no host-computed offsets.
- A single shared out buffer is rejected: the worker adds experts sequentially into it today (`moe_mul1.cpp:3857`, `3298-3311`), so an early GPU read would race the next expert's add (torn sum). Disjoint compact segments make early reads safe.
- New worker entry `exl3_moe_cpu_forward_expert(handle, x, sel, wts, out_slice, rows, topk, expert, threads)` beside `exl3_moe_cpu_forward_raw` (`moe_mul1.h:93`): builds only that expert's chunks (`moe_mul1.cpp:3868-3883` loop restricted to one expert, scan order = sel scan order as at `3861-3867`), writes `weight*out` per row (no full-buffer `memset` at `3857`, no grouping vectors at `3860`, no arena regrow at `3891-3903`). Reusing `forward_raw` per expert is rejected — its fixed costs repeat D×.
- Fused path (`submit_issue_fused`, `moe_cpu_host.py:1475`; C++ at `moe_handoff.cu:245-293`): extended, not replaced. The staging kernel still stages `x/sel/w` once per layer. A new announce step in the same call writes the D descriptors *from the GPU* (thread 0 scans sel, writes distinct experts ascending, `__threadfence_system()`, then release-stores `jobs_tail`) — the host never polls or syncs, so run-ahead is preserved. Decided because descriptor count D is GPU-only knowledge on the fused path (no `tolist()` sync anywhere in `moe_cpu_host.py:1478-1488`).
- Gating: only decode fused submissions (`block_sparse_mlp_cpu.py:1094-1108`). Copy-path `submit_issue` (`moe_cpu_host.py:1453`) and `cpu_offload_issue` (`block_sparse_mlp_cpu.py:1134-1150`) keep whole-job behaviour; `cpu_split_combine` (`block_sparse_mlp_cpu.py:1157-1176`) branches on handle shape.

## 2. Per-expert completion flags and ordering

- New bank `expert_done[MOE_JOB_RING]`, 64 B stride, appended after `consumed[]` in the flag region; `MOE_FLAGS_SIZE` grows (`moe_handoff.h:95`, mirror `moe_cpu_host.py:58`). Flag indexed by ring position (as `data_ready/done` are by slot, `moe_handoff.h:22-25`); all D jobs of a layer share one `seq` (allocated once per layer), worker publishes `store_release_flag(expert_done[head % MOE_JOB_RING], job)` (`moe_handoff.cu:333-345`, `:529` pattern).
- Publish order per job: slice bytes → rowidx/segment table entry → release-store of `expert_done`. GPU waits (`moe_flag_wait`/memop `GEQ`, `moe_handoff.cu:181-186`) are acquire, so the flag is observed only after the slice is fully written. Slice regions are disjoint (deterministic prefix offsets), so no expert can tear another's sum.
- `cpu_split_combine` enqueues, per job in submission order: `flag_wait(expert_done[i], seq)` then `moe_split_collect_add_expert(final, out_dev, rowidx_ptr, segtab_ptr, expert, ho)`, then ONE `consumed` write after the last fold. `consumed` stays per-slot because slot reuse must wait for *all* folds (multi-device hazard, `moe_handoff.h:24-30`, `moe_cpu_host.py:1586-1591`).
- Worker dispatch is FIFO (`moe_handoff.cu:473-475`); completion order = enqueue order, and folds are stream-enqueued in that same order, so landing order can never reorder the sum (see §4).
- Crash hole closed by the existing watchdog: it must also satisfy `expert_done` entries (`moe_cpu_host.py:1391-1399`) and set `abort` (`moe_cpu_host.py:1400`); the GPU memop wait has no timeout (`moe_handoff.cu:65-71`), so this is load-bearing.

## 3. Submission order, in-flight budget, backpressure

- Parent enqueues layers in forward order; within a layer, descriptors expert-ascending. Ascending matches today's chunk/accumulation order (see §4) and costs no host sync.
- The weight stager thread (`moe_handoff.cu:389-427`, spins on `pinned_free` at `:411-417`) is untouched — separate `MOE_STAGE_RING` (`moe_handoff.h:100`), prefill-only. Per-expert jobs never enter it.
- Binding constraint is the job ring, not slots: `num_slots=4` (`moe_cpu_host.py:99`) only bounds input/output reuse, which is stream-ordered via `consumed` (`moe_handoff.cu:289-291`) and already deadlocksafe by the deferred-collect discipline (`moe_cpu_host.py:1537-1543`). The ring stalls when `tail-head >= MOE_JOB_RING-4` (`moe_handoff.cu:275`; copy-path spin at `moe_cpu_host.py:1571-1576`).
- Arithmetic: distinct CPU picks per decode layer `D <= rows*topk` (rows=5 ⇒ `D <= 40` at topk=8; T56's "~37 per-expert times" per layer-function). Full pass = 48 × D ≈ 1776 jobs > 252 ring capacity ⇒ ring must grow to 2048 for full-pass run-ahead. Without the grow, the host stalls mid-pass and the `sub` tail (mean 0.36 ms, `T56:34-39`) gets worse — this is the number that decides whether the design helps.

## 4. Reduction order / numerics

- Today the CPU sum is deterministic: chunks are grouped expert-ascending (`moe_mul1.cpp:3868-3883`) and `decode_out_block` adds `dst += weight*v` in chunk order per 128-col block from a single finisher (`moe_mul1.cpp:3298-3311`, last-finisher via `out_done` at `:3572`).
- Per-expert: each slice holds exactly `weight*v` for its expert's rows (same product rounding); GPU folds slices in the fixed ascending order (stream-enqueued, never landing order). The *sum-level* order is therefore preserved.
- Bit-for-bit is NOT preserved inside the GEMMs: split-K partitioning depends on total chunk count (`moe_mul1.cpp:3945-3973`: `num_grp_*` scale with `nc`, so `parts[]` and `finish_partials` grouping differ for nc=1 vs nc=37; small-job cap at `:3931` also changes). That is fp32 reassociation noise, same class as the existing GPU/CPU partial mix (`block_sparse_mlp_cpu.py:1172-1175`) and dynamic-placement steps (`block_sparse_mlp_cpu.py:1489-1492`).
- Existing test that bounds it: `tests/moe_cpu/test_cpu_decode.py:58` (`rtol = 5e-4, atol = 2e-5`) — the arm must still pass it. Non-deterministic *order* is not accepted; reassociation *within* an expert's GEMM is.

## 5. Fused path and prefill

- `submit_prefill` (`moe_cpu_host.py:1980-2018`) and its streamed path: untouched. Partial completion is decode-only; the prefill gate stays in `cpu_split_submit` (`block_sparse_mlp_cpu.py:1094`).
- `submit_issue_fused` extended behind `EXL3_MOE_PARTIAL_COMPLETION` (read once via `MoeCpuTuning`, `moe_cpu_host.py:84-199` pattern); off ⇒ current one-descriptor call (`moe_handoff.cu:277-285`) byte-identical. `EXL3_MOE_CPU_SPLIT_PROF` / `EXL3_ROUND_PROF` brackets must wrap the new path (design outline, `2026-11-08-partial-completion-design.md:71-72`).

## 6. Where flags/buffers live, added bytes, ring growth

- All in the existing pinned `SharedMemory` control/slot region (`moe_cpu_host.py:1283-1302`); no VRAM added (rowidx/segment table live host-side in the slot).
- `expert_done`: 2048 × 64 B = 128 KiB (arithmetic). Ring control: 2048 × 1056 B = 2.08 MiB vs current 264 KiB (arithmetic from `moe_cpu_host.py:50`). Slot out region grows topk×: `cap_rows*topk*Ho*4` vs `cap_rows*Ho*4` (e.g. cap 64, topk 8, Ho 4096 ⇒ 8 MiB vs 1 MiB per slot, +28 MiB over 4 slots — arithmetic).
- `MOE_JOB_RING` 256→2048 unconditionally (layout change, both mirrors `moe_handoff.h:54-104` / `moe_cpu_host.py:45-63` must change together; `MOE_CTRL_SIZE` recomputes; `check_shm_capacity` at `moe_cpu_host.py:1282` re-checks). Behaviour stays default-off.

## 7. Test plan

- Existing: `tests/kernels/moe/test_moe_split.py:210-233` (collect_add), `:289-291` (empty-skip) — add a `moe_split_collect_add_expert` gather case (non-contiguous rowidx, multi-expert fold equals dense sum).
- Existing: `tests/modules/moe/test_shared_expert_schedule.py:21-64` (`nogpu`) — add a partial-handle case asserting combine enqueues D wait/fold pairs then one `consumed`.
- Existing: `tests/moe_cpu/test_cpu_decode.py:58`, `tests/moe_cpu/test_wide_rows.py:200` (rtol 1e-4) — must pass unchanged with the flag on.
- New nogpu unit: C++/Python layout mirror assert (`MOE_JOB_BYTES`, flag bank offsets, ring × D budget) — arithmetic only, no GPU.
- Invariant: flag off ⇒ bit-identical outputs vs production (same code path). Flag on ⇒ reference tests within today's tolerances, and a same-session ABBA perf arm (below) before any number is quoted.

## 8. Risks and pre-registered prediction

- Biggest risk (correctness/performance): per-job fixed costs with D tiny jobs. `forward_raw` today pays grouping allocs, arena grow, a pool mutex, counter reset and two phase barriers per call (`moe_mul1.cpp:3860`, `:3891-3922`, `:3954`, `:3994-3999`); `forward_expert` must strip all of them, and the worker still dispatches jobs sequentially (`moe_handoff.cu:473-529`) — there is NO extra parallelism, only finer makespan pipelining plus fold overlap. Fold bytes are `rows*Ho*4` (e.g. 80 KiB at rows 5, Ho 4096 ⇒ ~2 µs at the measured 42–45 GB/s bus, `T56:38-39` — arithmetic), so fold overlap alone cannot buy 0.2 ms; the bet is on reduced cross-expert phase coupling (`moe_mul1.cpp:3944-3958` vs per-expert blocks).
- Second risk: descriptor volume re-creates the `sub` tail if ring growth is skipped (§3).
- Pre-register (same-session ABBA, `-dr 3`, 131 072, per `2026-11-08-partial-completion-design.md:73`): `cmb` median < 0.20 ms and decode ≥ 60 tok/s. If `cmb` improves but tok/s does not (fold-overlap arithmetic says it shouldn't move), or CPU total busy grows (per-job overhead), or the `sub` tail worsens — not worth doing; do not merge.
