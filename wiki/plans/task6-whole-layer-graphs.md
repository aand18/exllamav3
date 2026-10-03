# Task 6 plan: whole-layer / whole-step graphs for tg@64k (handoff)

Goal: raise kvarn tg@64k from 47.9 tok/s toward the 61 tok/s device
ceiling by collapsing host dispatch (~3000 aten calls/step, ~5-7ms
of the 21ms wall) into CUDA-graph replay. Attn-sublattice graphs
already ship (`dispatch.py`); this extends capture outward.

Branch: `wip/kvarn-cache` (pushed to origin; base your work on the
tip, do NOT rebase). Scope: K4V4, decode only, 4090 box. All GPU
validation on the box mirror (see §7). One cut per commit, twin per
kernel change, env-gated with fail-closed fallback (project law).

## 0. Facts you must not re-derive (measured 2026-10-02/03)

- Wall/step @64k: graph 21ms (47.9 tok/s), eager 23ms, fp16 16ms.
- Device/step ~16.4ms: MoE exl3 GEMV/GEMM ~8ms (inherent, shared
  with fp16 — out of scope), serve 4.5ms, tail 0.8ms, combine 0.28ms.
- Host cpu_op wall 23.4ms = full wall: ~3000 aten calls/step, no
  single fat host function, syncs are cheap polls. Piecemeal host
  cuts (combine-split, n-mirror, longskip) all measured ±0%.
- Model: ~48 layers (16 full-attn kvarn + recurrent GDN + MoE MLP).
- Decode step shape is STATIC: bsz 1, q_len 1. Only contents mutate
  (hidden states, block_table pages, seqlens +1, sampler token).

## 1. Read this first (in order, no code until done)

1. `exllamav3/model/model_ls.py:317-330` (`forward_ls` — the
   decode loop you will eventually capture).
2. `exllamav3/modules/transformer.py:141-209`
   (`TransformerBlock.forward` — Phase-1 capture unit).
3. `exllamav3/modules/attention_fn/dispatch.py:236-341`
   (`_try_kvarn_graph_decode` — the SHIPPPED sublattice pattern to
   copy: static bufs `:187-233`, bucket key `(gc,R)` `:303`,
   buffer-identity verify `:315-324`, `replay()` `:325`, every-128
   sticky read `:329-335`, loud fallback everywhere).
4. `exllamav3/modules/attention_fn/dispatch.py:345-399`
   (`_graph_capture` — capture-after-eager-warmup pattern).
5. `doc/kvarn-4090.md` tg table (~line 216) + cut notes below it;
   `doc/perf-strats.md` §4-7 (this task is §6; do §4 first only if
   told to — it is independent and stacks).

## 2. Phase 0 — spike: single-layer capture, measure the prize (STOP gate)

Do NOT design Phase 1 until Phase 0 measures >5% per-step saving.
Work in `eval/_spike8_layer.py` (new file, NEVER commit spikes).

1. Load model + kvarn4 cache on box, populate 64k ctx (copy the
   preamble from `eval/kvarn_prof_kineto.py:15-45`: Config, Model,
   Cache with `CacheLayer_kvarn`, `populate`, 3 warmup steps).
2. Pick layer 0's `TransformerBlock` (a full-attn layer). Run one
   eager decode step through it alone, saving input `x` and output.
3. Capture: `g = torch.cuda.CUDAGraph()` +
   `with torch.cuda.graph(g): out = block.forward(x_static, params)`
   where `x_static` is a persistent buffer you `copy_` the live `x`
   into before replay (same input-copy pattern as
   `dispatch.py:293-299`). `params` must be the LIVE decode params
   dict (cache, seqlens, block_table) — capture bakes addresses, so
   reuse the same cache object for replay.
4. Replay 5× with fresh `x` per step (advance the real cache each
   step via the normal path on a SECOND identical layer? No —
   simpler: replay same step 5× for timing, then separately verify
   correctness step-by-step against eager with advancing cache).
5. Correctness bar: `torch.equal(replay_out, eager_out)` per step
   for 10 advancing steps. If not equal, dump maxabs per submodule
   (hook attn_out, mlp_out) to find the mutating op; most likely
   suspects: in-place residual adds on baked addresses, RNG (must
   be none in greedy path), `get_for_device` cache mutation.
6. Prize bar: `(eager_step_ms - replay_ms) / eager_step_ms` for one
   layer, × 48 layers projected. Compare against the 5-7ms host
   gap. If projected gain <5% wall: STOP, report, do not proceed
   (the capture overhead + input copies eat it).

Report format (append to this file §6): per-layer eager ms,
replay ms, equal Y/N, projected wall gain. Ask the maintainer
before Phase 1.

## 3. Phase 1 — per-layer graphs, attn layers (production)

Only after Phase-0 sign-off. Production rules: env gate
`EXL3_KVARN_LAYER_GRAPH=1` (default OFF until box-green, then flip
to ON like graphs-v2 did), fail-closed to eager on ANY trip,
loud `print(..., flush=True)` on fallback (grep-able on box).

1. Mirror the dispatch sublattice pattern at layer scope in a new
   module `exllamav3/modules/layer_graph.py` (do NOT bloat
   `dispatch.py`): static input buf per layer, capture-after-warmup
   (first N steps eager, capture on step N+1), bucket key =
   `(layer_idx, gc, R)` reusing the attn bucket when the layer has
   attn (recurrent layers: key `(layer_idx, n)` — n from the
   n-mirror, NOT a new sync).
2. Keep OUT of the capture (stay eager, in this order per step):
   kvarn store `update_kv_direct` (has the status-sync control
   flow) → layer replay → sticky/flag periodic reads. I.e. the
   capture covers attn-serve + MLP + norms, never the store.
3. Fallback invalidation (mirror EVERY rule from dispatch `:337-341`
   + these): seal/evict/pressure event → drop that layer's graphs;
   bucket miss → eager + capture-after; `PARITY=1` → still replay
   (asserts validate both, same as sublattice review #1); every-128
   sticky read → trip clears graphs.
4. Twin: `tests/test_kvarn_layer_graph.py::test_layer_replay_exact`
   (CUDA-gated like the graph twin): 10 advancing steps, assert
   `maxabs == 0.0` (same kernels, same addresses — nonzero means
   capture bug, never "numerics").
5. CPU suite must stay green (new test skips without CUDA):
   `venv/bin/python -m pytest tests/test_kvarn_cpu.py
   tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py
   tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py
   tests/test_kvarn_triton.py -q`

## 4. Phase 2 — recurrent + MLP-only layers, then Phase 3 (optional)

- Phase 2: same wrapper for GDN/recurrent layers (key on n;
   their state is recurrent — capture must include the state
   carry or explicitly exclude state-update ops to eager; spike
   FIRST per layer type, same §2 protocol, one layer type at a
   time).
- Phase 3 (only if Phase 1+2 leave >2ms host): whole-step capture
   across `forward_ls` (embedding + all layers + final norm;
   sampler stays OUT — see task §5 sample-in-graph). Needs cross-
   layer static buffers + a global invalidation bus. Design doc
   first, no code before maintainer sign-off.

## 5. Explicit non-goals (do not attempt)

- Prefill graphs (separate task, `doc/perf-strats.md` §3).
- Serve-kernel surgery (§7: HIGH RISK, needs its own box loop).
- Changing sampler behavior; MoE kernel work (upstream scope).
- Touching `master`, rebasing, force-pushing, `git config`
  changes, committing `eval/_spike*` or `_probe*` files.

## 5b. Hands-off + self-verification (read twice, follow always)

- HANDS-OFF, read-only unless your phase explicitly requires it:
  `eval/_probe_geom.py`, every `wiki/reports/*` file, every
  `doc/*.md` except the ledger lines §6 tells you to append,
  `BRANCHES.md`, `AGENTS.md`, workflow files. Spikes live ONLY in
  new `eval/_spike8_*.py` files; tests ONLY in the one twin file
  your phase names. Never rename, move, or reformat files outside
  your phase scope ("cleanup" is not your task).
- Before EVERY commit: run `git status --short` + `git diff
  --stat`. Every listed path must be one you intentionally edited
  for this phase. Anything else (modified file you don't
  recognize, unexpected deletion) = STOP, report, do not commit.
  Stage with explicit `git add <path>` per file, never `git add -A`
  / `git add .`.
- Recovery (if the tree looks wrong): committed work is safe on
  `origin/wip/kvarn-cache` (pushed through `6f51f35` at plan
  time). `git stash -u` parks YOUR mess including untracked;
  `git checkout -- <path>` reverts a tracked file to HEAD. Never
  `reset --hard`, never touch another process's files.

## 6. Box validation protocol (every perf commit, no exceptions)

Protocol v3 + env (copy EXACTLY into every `.bat`):
`EXL3_KVARN_TRITON=1 EXL3_KVARN_IMAGELESS=1`
`EXL3_KVARN_TRITON_PARITY=0` (perf) / `=1` (validation),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
`PYTHONPATH=C:\Users\yoho\Downloads\exllamav3-kvarn`,
python = `C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe`,
model = `C:\Users\yoho\Downloads\tabbyAPI\models\Qwen3.8-27B-exl3-SC_1.40bpw_H3_V3`.
WITHOUT `IMAGELESS=1` you measure the 5.3 tok/s legacy path
(trap documented 2026-10-02 — never debug perf without it).

1. Sync: `cp` changed files worktree → box mirror, `unix2dos`
   every copied file AND every new `.bat`.
2. Order per bat: 8k anchor+warmup, graph ×2 (hot counts),
   eager ×2 control-LAST:
   `eval/kvarn_microkld.py -m %MODEL% -cq kvarn4 -ntok 65536
   -chunk 8192 -dec 256`.
3. Gates (all must hold): twin maxabs 0.0 (or <1e-5 with written
   justification), KLD same-top 100% + mean <1e-4 @64k, PARITY=1
   @8k asserts green (`-ntok 8192 -dec 64`), tg hot-run ≥ baseline
   47.9 (first run may be lower = triton recompile, hot counts).
4. No `curl|wget` to stdout on this host; GPU runs need the VRAM
   watchdog (`eval/smi_guard.py --image python.exe --before <pids>`,
   kill under 100MB free). Never touch other processes' PIDs.
5. Record: append 3-5 ledger lines under the tg table in
   `doc/kvarn-4090.md` (numbers + twin + KLD + PARITY, same shape
   as the 2026-10-02/03 entries), one atomic commit per cut.

## 7. Done means (Phase 1)

- `EXL3_KVARN_LAYER_GRAPH=1` default-ON, kill-switch restores
  legacy, zero fallback trips in steady 256-step @64k.
- tg@64k graph hot ≥ 50 tok/s (≈+5%: 21.0ms → ≤20ms wall).
- Ledger entry + twin + CPU suite + KLD/PARITY green, pushed.
- If hot < 50: report measured number + Kineto split, do NOT
  stack more phases — hand back for re-rank.

## 8. PHASE 0 RESULT (2026-10-03, box `4ff897b`, spike `eval/_spike8_layer.py`,
##    logs `spike8_bill.log` / `spike8_cap.log` on the mirror) — STOP

Verdict: **whole-layer capture is not achievable as designed, and the
whole host-side pool is smaller than §0 assumed. Do not start Phase 1
without a maintainer re-rank.** Numbers, then why.

### 8.1 Plan facts that measurement corrected

- Model is **dense** (no `num_experts` in config.json ->
  `GatedMLP` 17408, not `BlockSparseMLP`). §0/§6 "MoE exl3
  GEMV/GEMM ~8ms" is the dense exl3 GEMV/GEMM path; it measures
  8.20ms/step, so the number stands, the label does not.
- **64 layers, not ~48**: 16 full-attn kvarn + 48 linear/GDN. §2.6's
  "x48" projection is really x16 attn + x48 GDN.
- **GDN and MLP decode are ALREADY graph-captured upstream**:
  `gated_delta_net.py:1053` runs the whole GDN layer through
  `bc.run_bszN` (internal CUDA graph, `exllamav3_ext/graph.cu`),
  `mlp.py:741` does the same for `GatedMLP` via `BC_GatedMLP`. Per
  layer: 2 `cudaGraphLaunch` + 3 launches + 1 copy, 36 aten ops.
  `BCAttn` (`bc_attn.py:10`) would graph the whole attn block too, but
  it **DECLINES on all 16 kvarn layers** (`EXL3_BC_ATTN_TRACE=1`:
  `BC-attn-kvarn: DECLINED layer 3..63`) because the online-dequant
  kvarn kernels are out of its scope. Step total: 128 graph launches
  (48 GDN + 48 MLP + 16 kvarn sublattice + 16 MLP), 467 kernel
  launches, 4192 aten ops.
- **The kvarn store's own budget is bigger than the whole bubble**:
  0.228ms/layer x16 = 3.50ms/step, and it is host-sync bound
  (`_store_rows_fast` 2-4 DtoH syncs), so it cannot be captured
  either way.
- **Host bubble measured for the first time post-graphs** (closes
  open question 1 of `wiki/reports/2026-10-02-tg64-host-bubbles.md`):
  device busy **16.51ms/step** (1682 kernels: serve 4.46, exl3
  GEMV/GEMM 8.20, tail 0.78, combine 0.40, GDN 0.53), wall
  **19.27ms** in-process (51.9 tok/s, argmax loop) -> bubble
  **2.77ms/step = 14.4% of wall**. Device ceiling 60.6 tok/s
  in-process. The §0 "5-7ms host gap" is CPU *work*, most of it
  hidden under the GPU tail or spent blocked on D2H; **no host-side
  cut, graphs included, can return more than 2.77ms/step.**

### 8.2 Measured (per layer, steady 64k decode, PARITY=0)

| layer type | eager host issue | wall | ops/call | x count | bill |
|---|---|---|---|---|---|
| attn (kvarn, L3) | 0.654ms | 0.701ms | 167 | 16 | 10.47ms/step |
| — attn half | 0.566ms | | | | 8.96ms/step |
| — mlp half (BC-GatedMLP) | 0.088ms | | | | 1.51ms/step |
| — store alone | 0.228ms | 0.231ms | | 16 | 3.50ms/step |
| gdn (L0) | 0.091ms | 0.160ms | 36 | 48 | 4.39ms/step |

Per-layer "issue" is *not* pure dispatch: the attn layer's top op is
one blocking `cudaMemcpyAsync` (kvarn status/seqlens DtoH, 1.57ms
under the profiler with a deep queue), and the GDN layer's top op is
one `aten::copy_` (60us, BC input staging). Both expose device time.
So the 14.86ms/step "layer bill" is mostly wait, and the honest
recoverable pool is the 2.77ms bubble.

### 8.3 Capture attempts (phase `cap`, separate process)

- **attn L3 whole layer (store stubbed, `EXL3_KVARN_GRAPH=0` so the
  outer graph would subsume the sublattice): FAILED** —
  `cudaErrorStreamCaptureInvalidated` (context then unusable). Cause
  is a host read inside the captured region: the eager serve path
  syncs at `dispatch.py:491` (`n = int(cache_seqlens[0]) + q_len`); the
  graph path is sync-free only because of the n-mirror
  (`kvarn.py:473`). With `EXL3_KVARN_GRAPH=1` instead, the capture
  would hit a nested replay, which is equally illegal.
- **gdn L0 whole layer: HARD FAIL** — `GPU assert: operation not
  permitted when stream is capturing ... exllamav3_ext/graph.cu:186`,
  exit 900. BC's own capture is the nested one. There is nothing left
  to graph in a GDN layer: it is already one graph launch.

So "equal Y/N" is N/A for both layer types: no whole-layer graph was
produced, hence no projected wall gain from the plan's design.

### 8.4 The only viable shape: two disjoint graphs per attn layer

Store + serve + MLP must stay eager/BC-owned, so the capturable
regions per attn layer are two disjoint graphs, not one:

- **graph A**: `project_qkv` + `rope` (attn_norm stays in the block),
  ending in static q/k/v/g buffers (no store, no host read; ~40 of the
  layer's 167 ops). The eager store then consumes A's static k/v.
- eager: kvarn store + serve (shipped sublattice graph, unchanged).
- **graph B**: gate mul + `project_o` (~15 ops), on a static copy of the
  serve output. The MLP can never be inside a graph (nested BC graph).

### 8.5 That split, MEASURED (spike `eval/_spike8_split.py`, log
###      `spike8b.log` on the mirror) — works, and is worth +0.21%

Built and measured end to end on all 16 attn layers (real store + real
serve, everything else replayed):

- **Captured**: 16/16 layers, both graphs, 64.0 MB of graph pools.
  exl3 GEMV + `ext.rope` capture cleanly — the capturability risk in
  §8.4 is retired.
- **Bit-exact**: region check worst maxabs **0.0** across all 16 layers
  (graph A vs eager `project_qkv`+rope for q/k/v/g, graph B vs eager
  gate+`project_o`); re-checked over 10 advancing steps x 16 layers at
  the live position, still **0.0**.
- **Perf A/B** (best of 3 windows x 40 steps, same cache, no fallbacks):
  baseline 19.272 ms/step (51.9 tok/s) -> split **19.231 ms/step
  (52.0 tok/s), delta +0.041 ms/step = +0.21%**. Projected onto the
  47.9 tok/s server baseline: **48.0 tok/s**.

The estimate in the pre-measurement draft of this section said ~0.9ms
(+1.8%); the measured value is 22x smaller, because the ops the split
removes were already overlapped with the GPU: the attn layer's host
time is dominated by the blocking D2H wait (kvarn status/seqlens), not
by the ~55 dispatches. This is the same reason the three earlier
host-side cuts measured neutral.

**Verdict: +0.21% vs a 5% bar. Phase 1 is not justified in any shape**
-- neither the plan's whole-layer graph (§8.3, impossible) nor the
two-graph split (§8.5, possible and bit-exact but ~nothing). Per §2.6
and §7: STOP, do not stack Phase 2/3, re-rank first. Phase 2 is moot
on its own terms (GDN layers already self-capture via `bc.run_bszN`)
and Phase 3's whole-step capture inherits every blocker above plus the
sampler.

One reusable design fact from the spike, if the split is ever revived:
the rope position must be fed through a **persistent device buffer**,
not `cache_seqlens` (`prepare_flash_attn` builds a fresh one per step,
so a capture bakes a frozen position) and not a per-layer `position`
copy. `cache_seqlens` is also wrong as a source because it is one
tensor shared by all 16 layers and each layer's store advances it
(measured: layers 7+ came out with position+1, maxabs 13.4). What works
is one `(n_attn_layers,)` int32 staging buffer, one row per layer,
filled once per step from the host `params["position"]` (one launch).

Re-rank candidates the measurement now supports (all inside the 2.77ms
bubble): sample-in-graph + the terminal sync (`perf-strats.md` §5,
~0.3-0.8ms by the host-bubbles report), the dispatch wrapper's 5
static-buffer copies + bucket math per attn layer (`perf-strats.md`
item (c)), and serve traffic surgery (device 4.46ms -- the only prize
larger than the bubble, but HIGH RISK and its own box loop).

Spike artifacts (never commit): `eval/_spike8_layer.py`,
`eval/_spike8.bat`, `eval/_spike8_split.py`, `eval/_spike8b.bat`,
mirror logs `spike8_bill.log` / `spike8_cap.log` / `spike8b.log`.

