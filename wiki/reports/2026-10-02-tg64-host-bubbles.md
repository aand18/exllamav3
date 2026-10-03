# tg@64k host bubbles: what remains outside the graphed sublattice

Branch `wip/kvarn-cache` @ `5df2190`, read-only (no GPU runs; all numbers are
code-derived Fermi estimates ±50% unless cited from prior measured work).
Method: count launches/syncs/Python-dispatches per decode step on the steady
replay-hit path (batch=1, greedy, no draft), × unit costs
(launch+dispatch ~5–15µs, small-DtoH sync ~10–30µs, Python op ~1–5µs;
anchored to Kineto: `index 29ms / 448 calls ≈ 65µs` heavy-dispatch,
doc/kvarn-4090.md:919).

## Baseline

- Scoreboard (doc/kvarn-4090.md:218-220): tg kvarn4 @64k = **43.4 eager /
  47.0 graph** vs Bee kvarn4 **44.0** (KLD same-top 100%).
- Wall/step now: 1/47.0 ≈ **21.3ms**. Kineto @64k pre-graphs (doc:609-611):
  **26ms wall / 17ms device → ~9ms bubbles**. Graphs v2 removes per-layer
  ~10 dispatches + tail/merge launches; device ≈ unchanged (~17ms) →
  **remaining bubbles ≈ 4–4.5ms/step**. The ranking below sums to ~4–6ms
  (ranges overlap budget; syncs partly hide under the GPU tail, so the mean
  overcounts — treat as prioritization, not accounting).
- Model: Qwen3.8-27B dense (Qwen3_5, hd 256, **16 full-attn + 48 linear**,
  doc:81-82,344). All per-layer figures below ×16 attn layers unless noted.

## What the graph covers (boundary)

Per-layer decode sublattice only: qwht → serve → tail-gather → bmm →
tail-reduce → merge, replayed from static bufs
(exllamav3/modules/attention_fn/dispatch.py:379-399 `_graph_capture`,
replay at :325). Everything below runs eager every step.

## Ranked remaining host cost (steady replay-hit, @64k)

1. **Non-graphed eager dispatch, ~2.0–3.0ms** — 48 linear/recurrent layers
   plus, on each of the 16 attn layers, qkv/o projections, rope, norms, MLP,
   residuals; plus the model loop and `attn_dispatch` wrapper
   (`dispatch_cache` hint get, shape unpack, layer lookup) and per-layer
   `get_for_device` calls. Evidence: exllamav3/model/model.py:374 `forward`;
   exllamav3/modules/attn.py:1042-1149 `decode_flash_attn`
   (project 1093, norms/rope 1095-1115, o_proj 927);
   exllamav3/modules/attention_fn/dispatch.py:642-661/719-722;
   exllamav3/util/tensor.py:115 `get_for_device`. Prior: "~600 dispatches
   for 16 full-attn layers", doc:343. Length-independent → dominates more at
   short ctx, still the largest slice at 64k.
2. **Store + n/branches per layer, ~0.7–1.2ms** — fused store is 2 launches +
   1 sync (kvarn_triton.py:484-503), but the host still pays per layer:
   `torch.stack+float` temp alloc (:450), `status.tolist()` 1 DtoH sync
   (:502, drives code 0/2/1 — cannot speculate, doc:335-336), and the graph
   wrapper's `int(cache_seqlens[0])` sync (dispatch.py:262) plus
   gc/R/bucket math (:263-269). 16 layers × (1 alloc + ~4 launches +
   2 syncs). Evidence: dispatch.py:257-262; kvarn.py:1967-1993.
3. **Graph replay mechanics per layer, ~0.4–0.8ms** — 5 small `copy_` into
   static bufs, `nbuf += q_len`, 2 out-aranges, bucket dict lookup,
   buffer-identity verify (m/acc/st, :318-324), `graph.replay()` (:325).
   ~7 launches + ~15 Python dispatches per layer. Evidence: dispatch.py:289-336.
   Miss path (bucket miss every ~128 steps as gc rolls, stale on serve
   realloc) pays the full eager rest instead — rare, not in the mean.
4. **Sampler tail sync, ~0.3–0.8ms** — already batched to ONE
   `torch.cuda.synchronize` + pinned staging per step (generator.py:1108-1128),
   but `next_token.cpu()` + `.item()` ×2–3 per job (job.py:620-621,809,826)
   is the step's terminal GPU-tail sync; nothing after it can overlap.
   `sampler.forward` itself is device-side, sync-free (custom.py:1171+).
5. **Block-table / input-id staging, ~0.1–0.2ms** — `block_index.zero_()`
   over 272 cols (64k→257 pages, ×16-pad), per-seq `copy_`, scalar seqlens/
   positions sets, `torch.cat` into pinned staging. ~5–10 launches total
   (once per step, not per layer). Evidence: generator.py:965-1006
   (`_staging` at 929-940).
6. **Job bookkeeping, ~0.05–0.15ms** — `receive_sample`: tokenizer list
   lookup, string append, stop-token/set checks, result-dict build,
   `time.time()` ×2; `sequence_ids.append`, `kv_position += 1`. Page-hash
   only every 256 steps. Evidence: job.py:620-850; generator.py:993-995
   (`cuda_sync_active` only on first token — not steady).
7. **Amortized periodic, mean ~0.1–0.4ms (jitter, not mean)** — code-2 seals
   every 128 steps/layer → 0.125 seals/step (single-group seal via batched
   core, kvarn.py:2415-2432); evict scan every 128 rows, sync-free
   (kvarn.py:2186-2244); post-replay flag due-read every 128 replays/layer
   (dispatch.py:329-335); `_touch_batch` validation only on tick steps
   (kvarn.py:1713-1716). Seal cost is mostly device (Sinkhorn+quantize);
   host slice is small but spiky.

## Could graphs absorb next (ranked by prize)

- (a) `int(cache_seqlens[0])` (dispatch.py:262,481): n is +1/step in steady
  decode — host mirror counter, resync on tick/edge. Easy, kills 16 syncs.
- (b) Sample-in-graph (argmax inside, DtoH only the token): kills the
  terminal sync's serialization; standard next step, sampler already 1-sync.
- (c) Graph input copies (dispatch.py:293-299): baked addresses force the
  copy_; alternative is capturing upstream (q-proj output address) — invasive.
- (d) `status.tolist` (kvarn_triton.py:502): HARD — drives 0/2/1 control flow;
  needs kernel-side branch or speculative-replay+verify. Do not attempt cheap.
- (e) Block-table/positions staging: tiny prize (~0.1ms); content mutates per
  step (pages grow) so it needs the same static-buffer treatment. Defer.
- Out of scope (queued sibling): prefill-graphs subregions —
  see wiki/reports/2026-09-30-prefill-graphs.md. Boundary: prefill chunk
  graphs attack pp (4–5s@8k), this report is decode-step only; shared infra
  (static bufs, bucket keys) should stay compatible but neither design is
  attempted here.

## Dead ends (already retired — do not re-attack)

- `_kvarn_check_flag`: plain-Python counter, zero syncs (dispatch.py:103-114).
- `_tpos_memo`: dict hit in steady state (dispatch.py:121-134).
- `_touch_batch`: tick-gated skip 127/128 steps (kvarn.py:1713-1716).
- Evict: tick-gated + vectorized, sync-free (kvarn.py:2186-2244).
- `Qh` reshape-view, `.to(long)` dtype guards, fused-store fp16 exact rows —
  all done (doc:334-352). QSA path (`qsa_seqlens_cpu`, attn.py:1076-1077)
  is cold for this dense model.

## Open questions

1. Post-graphs device ms @64k unmeasured (no Kineto since v2 shipped) —
   the ~4ms bubble budget assumes ~17ms device; re-probe to fix the budget.
2. Engagement ratios at 64k unmeasured here: run `EXL3_KVARN_PTIMES=1`,
   read `g_replay / g_miss / g_stale / g_fb_{code,shape,cert}` via
   `_ptimes_report` (fires every 2048 graph calls ≈ 128 steps, dispatch.py:254-256).
   Note PTIMES-on inflates wall (event syncs) — diagnostic only.
3. Dispatch-vs-sync split needs nsys: if item 1 dominates, the lever is a
   parent/outer graph or fewer modules/step, not more micro-trims.
4. Bench geometry assumed 16 attn layers from doc:81-82,344 — confirm layer
   count in the scoreboard header before quoting per-layer figures.
5. Seal jitter at 64k: 0.125 seals/step mean, but per-seal host+device cost
   unmeasured post-batched-seal; check p99 step time, not just mean tok/s.
