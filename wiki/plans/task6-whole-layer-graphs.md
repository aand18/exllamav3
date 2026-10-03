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
