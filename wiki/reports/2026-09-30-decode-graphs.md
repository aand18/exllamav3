# Subagent report: Decode-graphs feasibility study (2026-09-30)

## Provenance
RECOVERED from the opencode session store (`ses_f0dacbadbffewMG7PHkGzqnI0A`, dispatched
2026-09-30 12:38 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
Feasibility: YES for steady decode, NO for whole-step single graph. GPU ~1.68ms/step (serve 0.74 + tail 0.78 + combine 0.16 @8k) vs ~19ms host (~640 ops) → host-bound ~10x; graphing removes dispatch, not kernels.

1. Blockers per KVarN layer per step (16 layers/step)

- `kvarn_triton.py:502` `status.tolist()` in `kvarn_triton_store_row` → 1 DtoH sync/layer (16/step). Branch `kvarn.py:1979-1995` code 0/1/2. Code 1 → torch fallback with `int()/bool()` syncs (`kvarn.py:1770-1773,1799,1801,1808,1812,2050`; `km.any()` L2135, `tolist()` L2085, L2165). Code 2 → `_seal_group` (Sinkhorn+quantize, host-driven).
- `dispatch.py:202` `n=int(cache_seqlens[0])` → 1 sync/layer (16/step). `n` feeds `gc_eff=(n+127)//128` (grid changes every 128 steps → realloc `kvarn_triton.py:1802-1814`, `_kvarn_online_buffers:1367`), and tpos key.
- `dispatch.py:235` `flag.zero_()` graph-safe; `kvarn_triton.py:1861` `int(flag[0])` only when `_kvarn_check_flag` (`dispatch.py:102-114`) fires (1/128/layer, ~0.125 syncs/step) + `os.environ.get` per layer per step (`dispatch.py:153,155,226`; `kvarn_triton.py:1824,1853`). `flag_b` branch L236 is CPU-on-device.
- `dispatch.py:247` `_kvarn_tpos`: key includes `n` → miss/second-alloc every step until tail saturates (1 cat+2 aranges/step, shared across layers). Steady `R=sink+tail_eff` constant.
- `kvarn.py:1624` `bool((~ev).any())` + `1626 int(m.sum())` only when `_tail_exact_certain` false. Steady: sync-free (fused code 0/2 preserves cert; torch fallback clears it).
- Per-step allocs/layer: `Qf=Q.float()` + `st=bmm` (`dispatch.py:214,262`), `stacked` (`kvarn_triton.py:450`), 3×`empty` in `tail_reduce:1217-1219`, 1×`empty` in `merge:1141` → ~7 allocs/layer, ~112/step. `R/RPAD`, `gc/gcpad`, `groups/cpg/nbpad` are Triton constexprs → shape change = recapture.
- Between layers (`attn.py:1042-1156`, `model_ls.py:324-329`): `get_for_device` dict/cache, `project_qkv` mgemm (m==1 already preallocated L513-531), rope/norm/gate/o_proj launches, python loop + `prepare_for_device` per module. No device syncs steady-state, but ~500+ of the ~640 host ops.

2. Verdicts (by wall impact)

- Per-step allocs + python dispatch (~19ms host): **removable-trivially** — hoist to persistent buffers (pattern exists: `_ov_serve_*`, `_ov_online_tail_*`, `prealloc_qkv_*`).
- `status.tolist` + `n=int` (32 syncs/step, serializes): **needs-restructure** — run store kernel ungraphed, read status after; pass `n` as device scalar, derive `gc/R` from static step counter. Sketch: prologue kernel → `code,g` in pinned host mem → skip graph replay if `!=0`.
- `os.environ.get`, flag-tick counter, `get_for_device`: **removable-trivially** — hoist to constants/handles at capture.
- tpos rebuild: **removable-trivially** — static circular buffer, slice `[:R]`, pad `R→MAXW` with mask.
- flag/cert fail-closed (`flag_b`, `code 1/2`, staging fallback): **needs-restructure** — assume clean in graph; ungraphed fallback step on trip. Nothing **fundamental** — no data-dependent values enter math except via excluded fallback.

3. Strategy: graphed sublattice + ungraphed fallback (NOT whole-step, NOT per-layer-full)

Capture per-layer `qwht→serve→combine→tail_gather→bmm→tail_reduce→merge` as one graph per `(gc,R,qpk,hd)` bucket; store/seal/`n` prologue stays ungraphed; MoE/linears/norms/rope untouched. Recapture on `gc` change (every 128 steps, ~64 buckets @8k) and on any trip. Risk: **medium**. Riskiest: status-code speculation — a mid-step code-1/2 or flag trip cannot abort a launched graph; mitigation is fail-closed replay of the whole step ungraphed (correctness safe, one slow step).

4. NON-goals: sampler, MoE, linear/QKV/o-proj kernels, SWA layers unchanged.
