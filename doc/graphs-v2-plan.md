# Decode graphs v2: dispatch integration plan

> For agentic workers: implement task-by-task (one cut + gates per
> task). Steps use checkbox syntax.

**Goal:** replay the per-layer decode sublattice as one CUDA graph
per layer (16 replays/step), killing ~8 launches + Python dispatch
per layer (~1-2ms/step, +5-8% tg).

**Architecture:** per-layer sublattice graph
(qwht -> serve -> combine -> tail_gather -> bmm -> tail_reduce ->
merge) with static buffers; store/seal/`n` prologue stays
ungraphed; fail-closed eager fallback on any trip. Spike v1+v2
(`eval/_spike_graph_serve.py`, `eval/_spike_graph_full.py`)
proved capture mechanics (2.2-2.55x on the pair/sublattice,
bit-exact replay).

**Tech Stack:** torch.cuda.CUDAGraph, existing Triton kernels,
persistent layer buffers.

**Spec:** feasibility study `wiki/reports/2026-09-30-decode-graphs.md`;
spike evidence in this doc's graphs entries.

## Global Constraints

- Bit-exact replay required (twin bar: maxabs 0.0 on static shapes,
  same as spikes; KLD same-top 100% + digits on box).
- Kill switch `EXL3_KVARN_GRAPH=0` restores eager (default OFF until
  all gates green twice, then default ON).
- Fallback paths (status 1/2, flag trip, cert fail, shape change)
  must stay exactly today's behavior (loud None -> get_kv path).
- No `int()/bool()/tolist()` on device values inside capture;
  no per-replay allocs (pool or persistent only).

---

### Task 1: Buffer params for tail_gather (+ twin)

**Files:**
- Modify: `exllamav3/modules/attention_fn/kvarn_triton.py`
  (`kvarn_triton_online_tail_gather`: optional `ev/g/s` bufs)
- Test: extend `eval/_spike_graph_full.py` (already covers;
  assert buf reuse: same addresses across calls)

**Interfaces:**
- Consumes: existing wrapper signature + R-sized bufs.
- Produces: gather with zero allocs (needed for capture).

- [x] Step 1: Add optional trailing `_bufs=None` (tuple ev/g/s),
  defaulting to today's three `torch.empty` calls.
- [x] Step 2: CPU py_compile + box suite (callers without bufs
  behave identically).
- [x] Step 3: Spike v2 re-run (assert no new allocs: same ev/g/s
  addresses across replays).
- [x] Step 4: Commit.

### Task 2: Static R bucketing

DONE by analysis (2026-10-02, no box run needed): R = sn_ +
tail-window rows is structurally constant per (layer, run) once
n > tail_eff (window slides, size fixed); RPAD likewise. gc changes
every 128 steps (2-3 distinct (gc, RPAD) buckets per 256-step run),
so bucket key `(gc, RPAD)` with recapture on change is sufficient.
Added `tailR_*` PTIMES counters (env-gated observability) instead
of a measurement run.

**Files:**
- Modify: `exllamav3/modules/attention_fn/dispatch.py`
  (tpos/R handling), `exllamav3/cache/kvarn.py`
  (`kvarn_online_tail` MAXW views)

**Interfaces:**
- Consumes: per-step R (grows to cap, then stable).
- Produces: fixed RPAD bucket per capture epoch
  (pad R to next pow2 >= current, recapture on bucket change;
  masked lanes are bit-identical no-ops per kernel headers).

- [x] Step 1: bucket stability PROVEN analytically (R slides at
  fixed size; RPAD constant) -- no measurement run needed.
- [x] Step 2: pin RPAD per capture epoch (bucket key `(gc, RPAD)`).
- [x] Step 3: `tailR_*` observability committed (Task 1 commit).

### Task 3: Per-layer graph capture/replay + fallback

**Files:**
- Modify: `exllamav3/modules/attention_fn/dispatch.py`
  (graph objects keyed `(gc, R-bucket, qpk, hd)` on layer;
  capture on miss, replay on hit, eager fallback)
- Test: `tests/test_kvarn_triton.py` (graph-vs-eager twin, static
  shapes, maxabs 0.0 bar like spikes)

**Interfaces:**
- Consumes: Tasks 1-2 (zero-alloc sublattice, stable buckets).
- Produces: `EXL3_KVARN_GRAPH=1` replay path with identical outputs.

- [x] Step 1: Implement capture (warmup via eager-rest, then
  `torch.cuda.CUDAGraph` around the sublattice calls with
  persistent bufs + precomputed views).
- [x] Step 2: Implement fallback matrix: status != 0, flag trip,
  cert fail, bucket miss, any exception during capture ->
  eager today-path (loud, never half-run).
- [x] Step 3: Twin test (graph replay vs eager, static shapes).
- [x] Step 4: Box gates with `EXL3_KVARN_GRAPH=1`: suite + KLD-8k
  x2 (expect +3-8% tg, identical KLD) + needle.
- [x] Step 5: Commit; flip default ON only after green twice.

### Task 4: Docs + wiki pattern

- [ ] Step 1: Record outcome + measured tg delta in
  `doc/kvarn-4090.md` + `wiki/skill-impact.md` (accept or reject
  with reason).
- [ ] Step 2: If shipped, add `wiki/patterns/cuda-graphs-decode.md`
  (capture rules learned: static shapes/views, no syncs/allocs
  in capture, fallback matrix).
- [ ] Step 3: Commit.
