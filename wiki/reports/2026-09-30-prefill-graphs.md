# Subagent report: Prefill-graphs feasibility study (2026-09-30)

## Provenance
RECOVERED from the opencode session store (`ses_f0dacbad2ffeXUhztsZth19oLO`, dispatched
2026-09-30 12:38 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
**KVarN prefill-chunk CUDA-graph feasibility (analysis only, no edits)**

Prefill path per layer per chunk: `dispatch.py:347` get_kv (serve image) → SDPA → `dispatch.py:405` update_kv → `_store_rows` (`kvarn.py:1923`). bsz=1, T=4096/8192 (4032+64 splits). 16 full-attn layers/chunk.

**1. Capture-blocker inventory (per layer per chunk)**

* L1 legacy store loop `kvarn.py:2084-2145`: `torch.unique(g).tolist():2085` + per-group (G=1..32, data-dependent) `int(base):2089`, `int(group_base):2090`, `_stage_slot:2093` (`int(rev):1260`, `int(free[0]):1280`, `zero_:1281-82`), `bool(sealed):2121`, `int(owner):2118`, `bool(km.any()):2135`, `bool(exact_valid):2136`, `int(exact_rev):2138`, `_alloc_exact_block→_exact_slot:1395` (`int:1407`, `int(free[0]):1442`, `zero_`, diag `.tolist():1433`), indexed `stage/exact[slot,slots]=` scatters, `_eref_slots→torch.tensor:2152` HtoD alloc. ~12k DtoH+~6k HtoD/chunk total.
* L2 seal core `_seal_full_groups:2247` (`unique:2262`, mask `present.all/sealed/base/rev:2265-77`) → `_seal_groups_batched:2282` → `_seal_staged_blocks:2302` (+`_seal_spilled_batched:2364`, `zeros:2394`, asserts `bool(...all/any):2408-10`); Sinkhorn `kvarn_variance_normalize:762-810` ~16 iters × (std/log/exp/div/imbalance `amax/amin`) ≈100 launches/layer/chunk/side; early-break sync `bool((imb_best<imb_floor).any()):804` (≤4/seal/side); `kvarn_quantize_tile:820` (`amin/amax/floor/clamp`); `kvarn_pack_bits:656` (`arange(bits):659`, `arange(8):665` allocs, pad branch `pad:661-663` + `cat`); `recs=records[gs]:2325` copy + `records[gs]=recs:2348` writeback.
* L3 WHT `kvarn.py:2020-36`: `stack:2020`, torch `kvarn_wht_head:727` (`clone+empty_like:688-89`, 7 stages, `mul_`) or Triton `kvarn_triton_wht_rows:380` (1 launch, inplace); `half().to(dev):2036`, `ek/ev.to:2039-40`.
* L4 fast-path/frame `_store_rows_fast:1818-21` (`unique/bincount/scatter_reduce:1845-50`, `gs[bad].tolist():1880` 1 sync, `free[:NE].tolist():1905`, `index_fill_`, `_eref_wht:1916`); `n_new=int():2003`; single-row leftovers `int(g[0]):2049`, `bool(present.all/sealed):2050`; `torch.unique(pages).tolist():2165`; `_touch_batch:1717-32` (`bool(any):1726` sync); `_apply_exact_overlay:2521` (`int(seqlens):2543`, `arange/cat/unique`, `unique(g).tolist():2558`, `int rev:2564`); `status.tolist() kvarn_triton.py:502` (decode-only). `_evict_exact_all:2186` sync-free. PTIMES `_ptime.start/stop:430-70` + parity asserts add syncs (env-gated).

**2. Verdicts (impact order)**

1. Seal device ~469ms/8k: **needs-restructure** — graph saves only launch (~100×~7µs≈1ms/layer); real win is fusing Sinkhorn+quant+pack to 1–2 Triton kernels. Sketch: fix iters=16 (drop `:804` break), precompute pad (16384·bits%8==0 always → `:661-63` dead, **removable-trivially**), static `arange` consts, pad G to Gmax batch.
2. Store loop ~471ms: **needs-restructure** — ungraphable as-is (per-iter syncs+`zero_`/scatters); vectorize to batched `index_put` scatters + preassigned slots; `_store_rows_fast` proves eligibility is vectorizable. Remainder R-tail keeps 1–2 group eager path.
3. WHT+casts: **removable-trivially** — force Triton inplace WHT (static grid by T), hoist `stack/half/to` to static buffers.
4. Slot-pressure paths (`_seal_full_groups(None):1268`, `_reclaim:1374`, spill `unique().numel():1365`): **fundamental** — data-dependent control;Sketch: over-provision slots ≥Gmax+2 so pressure never fires in-capture; assert-free capture build, fallback to eager on trip.
5. Per-call allocs + PTIMES/parity: **removable-trivially** (static pool, env-off).

**3. Strategy: graphed subregions + eager remainder (risk: med)**

Whole-chunk graph rejected: spans get_kv↔SDPA per layer (NON-goal backend), G∈1..32 × R-tails × 2–4 chunk sizes = dozens of graphs + pressure re-capture storms (**high** risk). Instead: (a) graph WHT+vectorized store-scatter per chunk-size (2–4 shapes cached); (b) graph fixed-Gmax padded seal core per preset; remainder (pressure/spill/overlay-fallback) stays eager. Must change: kill `:804`/`:661` branches, static buffers for L1/L2 temps, slot over-provision, capture-build without asserts/PTIMES. Riskiest: bit-exact fused Sinkhorn seal kernel (op order, `floor(x+.5)` half-away-zero, fp32→fp16 RNE). No-graph host-trim grind remains viable fallback for ~80% of attainable win.

**NON-goal:** model forward/GEMMs/attention backend (shared fp16 path) untouched.
