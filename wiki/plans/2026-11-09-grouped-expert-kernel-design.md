# Grouped expert kernel — design (Experiment 3, T59)

Basis: T58 CSVs (rows=5 verify window: 1 pending + 4 drafts at
131072 spec_decode), T52 CPU compute, same-box Strata ~93.6 tok/s
vs fork ~56 tok/s. GPU window is 13.15 ms/tok vs Strata ~6.0 ms.
`ker` phase rows=1 median 0.061 ms, rows=5 median 0.117 ms
(1.9× over a 5× range — latency/occupancy-bound, not bandwidth).
moe phase ~0.53 ms median, cmb ~0.35–0.85 ms median, mixer ~0.10 ms.

## 1. Design — one pass over the verify window, weights read once per expert

The current decode path is already fused: `BC_BlockSparseMLP::run_bszN`
at `exllamav3/exllamav3_ext/libtorch/blocksparse_mlp.cpp:70` does
two launches per layer via `exl3_moe_coop_run`
(`exllamav3/exllamav3_ext/quant/exl3_moe_coop.cuh:83`), covering
every (token, expert) slot of the batch. The gap is not launch
count — it is **weight reuse**: each of the `bsz*topk` slots
re-streams its expert's gate/up/down weights independently, so
an expert hit by 3 of the 5 rows pays three weight streams.

The grouped kernel changes one thing: **the batch dim becomes the
row dim of a grouped GEMV**. One launch per layer (gate/up
stage + down stage with the rotation in between, same two-stage
structure), gridded over **distinct routed experts × N-tiles**,
not over slots:

- **Launch shape:** grid = `n_distinct × ceil(I/256)` (gate/up)
  then `n_distinct × ceil(Ho/256)` (down). `n_distinct ≤ 5·topk`;
  in practice ≤ 20 at rows=5, so the grid is tiny and the whole
  thing is one wave. Two launches per layer replace the coop
  pair; no per-row or per-slot launches remain.
- **Thread/block layout:** 256 threads (8 warps) per CTA, one
  256-wide N-tile per CTA. Each warp holds the decoded weight
  fragment for its sub-tile in registers and streams it against
  **all assigned rows** (up to 5) before moving on — MMVQ-style:
  one weight load, `rows` FMAs. Rows are packed as
  `[rows][H]` in registers/smem; the activation is rotated once
  (reuse the coop rotation), then gate/up/down run as grouped
  GEMVs with weights in smem broadcast across the row batch.
- **Routing:** unchanged. `routing.cu` top-k kernels
  (`routing_ds3_nogroup_topk_kernel` at
  `exllamav3/exllamav3_ext/routing.cu:304`,
  `routing_std_topk_kernel` at `:478`, `routing_ds3_group_topk_kernel`
  at `:664`) still produce `(bsz, topk)` selections; a cheap
  on-device compact step turns `sel` into
  `(distinct experts, row-run table)` — no host sync, same
  pattern as the existing run table in `MoeCoopParams.runs`
  (`exl3_moe_coop.cuh:65`). The grouped launcher reads that
  table; experts outside the local slice contribute exact
  zeros (as today).
- **CPU handoff:** unchanged by construction. Split layers
  keep `cpu_split_submit` → `exl3_moe_split_issue_fused`
  (`cpu/moe_handoff.cu:336`) publishing `data_ready`
  (`moe_handoff.h:22`, `flag_write_on` at `moe_handoff.cu:210`),
  and `exl3_moe_split_collect_fused` (`:421`) waiting
  `done` (`flag_wait_on` at `:226`) then publishing `consumed`
  after the readback. The grouped GPU kernel is a drop-in
  replacement for the body between issue and collect; the
  ring/flag discipline, announce kernel (`:259`) and worker
  loop are untouched.
- **Entry points:** wraps `BC_BlockSparseMLP::run_bszN`
  (`blocksparse_mlp.cpp:70`) and `exl3_moe_coop_run`
  (`exl3_moe_coop.cuh:83`) behind an env-flag dispatch in
  `BlockSparseMLP.forward` (`modules/block_sparse_mlp.py:1068`,
  fused eligibility at `:1092`, `cpu_split_submit` at `:1167`);
  reuses the coop kernel instance tables
  (`exl3_moe_kernel_instances` at `quant/exl3_moe.cu:35`) as
  the grouped down-path inner loop. CPU side `moe_mul1.cpp`
  (mul1 affine weight stream, `cpu/moe_mul1.cpp:36`) is out of
  scope — this is a GPU-window kernel only.

## 2. The pass mark

Pre-registered (T59 Experiment 3), all three gates required:

- `ker` ms/row at **rows=5 must fall ≥30% vs the 0.117 ms
  median → ≤0.082 ms**, measured per-layer from the T58 CSVs
  basis (same-session reference only).
- **No regression at rows=1**: must stay ≤ 0.061 ms median
  (the grouped path must not pessimise the single-row case;
  rows=1 falls back to the current coop launch).
- **Same-session ABBA tok/s land at ≥58 tok/s draft** at
  131k spec_decode (from the 55.4–55.8 baseline range),
  otherwise the build is not worth shipping even if `ker`
  improves — the win must survive the full window.

**Falsification — grouped-partial vs this design.** A cheap
"tracer" that groups rows into 2–4 arbitrary chunks (still
re-reading weights per chunk) tests occupancy/launch overhead
only; the full design groups **by expert identity** so one
weight stream serves all rows that hit that expert. If 2–4
grouped-partial already clears the 30% bar, the hypothesis
"weight reuse is the lever" is falsified — the win came from
fewer launches, and this kernel should not be built. The
tracer must run **before** the full kernel (and before any
wheel build) precisely to retire this distinction.

## 3. Effort and risk

- **Kernel-hours:** 24–40 h for the grouped GEMV (gate/up +
  down + run-table compact + flag-safe integration with the
  split handoff), plus 8 h for correctness (deterministic
  FUSED_DET-equivalent accumulation, expert-range masking,
  shared-expert merge). Debug on-device: add two passes.
  Estimated 3–5 working days end-to-end including ABBA.
- **Build time:** one wheel build is **~35+ min** on the
  Windows side (production box, live traffic — never
  stop/restart the server for a build). Budget two builds
  (one for the tracer if it needs ext changes, one for the
  real kernel); each failed build costs half a day of wall
  clock.
- **Percentage-done risk: ~40%.** The fused coop path already
  batches `bsz*topk` slots; the remaining kernel time is
  likely split between weight streams and tiny-GEMV spin-up.
  Probability the grouped kernel clears all three gates on
  first implementation: ~30–40%.
- **What could silently eat the win:**
  (a) **Handoff serialization** — `data_ready`/`consumed`
  flags serialize issue→collect per layer; a faster GPU
  kernel behind the same flag handshake shows as flat
  moe-phase medians.
  (b) **CPU pool dispatch** — `sub` phase (0.05–0.19 ms)
  and `cmb` (0.35–0.85 ms) bound the window, not `ker`;
  a 30% `ker` cut is ~0.035 ms/layer against a 0.58 ms
  CPU layer cost.
  (c) **Wave quantization** — at ≤20 distinct experts the
  grid underfills the GPU; the second wave of the down
  stage can add a full tail.
  (d) **Shared-expert reshape** — `sh_coop` runs its own
  pre-launch (`blocksparse_mlp.cpp:95`); reshaping it into
  the grouped launch re-serializes the merge order.

## 4. Alternatives if the tracer fails

Same code surface, cheaper levers, in order:

1. **Grouped-partial 2–4 groups** (hours, no new inner loop;
   env-flag chunking of the coop slot list). Pass mark: same
   30% `ker` cut at rows=5. This is the falsification arm
   above — if it passes, ship it and stop.
2. **One-graph wrapper** (T59 Experiment 2, kernel-lite):
   CUDA-graph capture of the existing issue+collect pair,
   no new GEMMs. Pass mark: moe-phase median down ≥ 0.10
   ms/layer at 131k, tok/s with same-session reference.
3. **Win a different phase** — attack `cmb` (median up to
   0.85 ms) via the worker pool or `moe_mul1.cpp` (MAX_M=4
   at `:71`, wide-row second pass) rather than the GPU
   window. Pass mark: ≥ 0.10 ms/layer off the moe-phase
   median, measured on the same T58 CSV basis.

## 5. Your next step

Do the tracer first and nothing else: one profiling pass
(nsight/torch profiler, 64 drafted rounds at production
regime) decomposing moe-phase median into kernel-time vs
inter-kernel gap share, plus a 2–4-group partial chunk
experiment against the coop path — both need no wheel build.
The roofline picks between graph, grouped, or weight-reuse
hands, and the partial arm falsifies the grouped hypothesis
without spending 35 minutes of build. Retire fast means stop
fast: if inter-kernel gaps are <10% of the moe median, or
grouped-partial clears the 30% bar, abandon the full grouped
kernel and spend the build budget on the graph wrapper or
the `cmb` phase instead.
