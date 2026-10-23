# Specter → exllamav3: implementation plan

Branch `wip/spectre-moe`. Target model: Qwen3.8-Flash-Next (`architecture/qwen4_exp.py`), 48 MoE
layers, 512 experts, top-10, PLE n-gram table, MTP head. Box: RTX 4090 24 GB (WDDM), 7950X3D,
64 GB RAM, PCIe 4.0 x16 (26.8 GB/s probed host→device).

Paper: *Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading*
(arXiv:2511.14102, ASPLOS'27). Feasibility study: PR aand18/exllamav3#3
(`doc/specter-feasibility.md` on `feat/specter-feasibility`). Twin implementation and its
measurements: `aand18/UnbiasedStrata` branch `spectre-moe`, issues #1–#12.

## 1. Base

Rebased from `v1.4.9+9` (2026-09-25) onto upstream `v1.6.4` (2026-10-21, `da68d57d`).
Old tip preserved as branch `backup/spectre-moe-v149` and tag `spectre-moe-v149`.

Dropped as obsolete — upstream did it better or the code it patched no longer exists:

| old commit | what | why dropped |
| --- | --- | --- |
| `cfcee80d` | Windows 2 MB large pages in the expert arena, SeLockMemoryPrivilege | upstream `_win32_large_page_alloc` + `EXL3_MOE_ARENA_HUGE` ladder subsumes it |
| `f9211e02` | 64-bit truncation fix in that same mapping code + diagnostics | the code is gone |
| `715009e3` | `EXL3_MOE_ZERO_COPY` whole-layer handshake, `EXL3_MOE_ARENA_LARGEPAGES` | upstream rewrote submit and has its own fused issue/collect path |
| `337c5a9b` | WDDM memop handshake (flush batch, submit push, logging) | 233 lines against a rewritten `moe_handoff.cu`; upstream still has no WDDM fix, so **re-land only after a v1.6.4 A/B shows the stall** (`EXL3_MOE_MEMOPS=1` vs `0`) |
| `f4d32ed8` | docs for the above | follows the code |

Kept, default off, Windows only (`1a6531ec`): `EXL3_MOE_TIMER_RES`, `EXL3_MOE_THREAD_PRIO`,
`EXL3_MOE_BLOCKING_SYNC`, `EXL3_MOE_CPU_PIN_OFFSET`, with their `doc/env_vars.md` entries.

What the rebase buys, from upstream commits with measured numbers on this model class:
`fa822cf6` (+15% decode, +26% with MTP on Qwen3.8-Flash-Next, 408/512 experts on CPU),
`c48bb5f8` (~2× long-context prefill), `ec9a4b9b`/`d939b745` (MoE M-tiling, +20% routing),
`8114c493` (AVX2 all-reduce, +18% cold prefill), plus the pinned expert arena
(`EXL3_MOE_PINNED_ARENA`, the study's P6 enabling primitive) and the persist/promote machinery.

## 2. What transfers from the paper

The artifact does not. exllamav3 ships activations, not weights (480 KiB/token against
1.17 GiB/token for the paper's shape): the host computes the cold experts, so there is no
gigabyte-scale PCIe transfer to hide. What transfers is the **inter-token look-ahead**: predicting
which experts the next token(s) will route to, and acting on it a layer or more early.

In v1.6.4 that lands on machinery that already exists, as a **policy**, not a new cache:

- placement is reactive today — promote-on-miss (`_split_promote_step`), periodic hot/cold sweep,
  persisted per-layer frequency `.json`;
- the transport is `MoeCpuHost.promote_expert` / `demote_expert` (arena → GPU slot DMA on the
  promo stream) plus the streamed-prefill weight ring (`EXL3_MOE_CPU_WSLOTS` × `_WSLOT_MB`,
  `EXL3_MOE_PINNED_ARENA` for a direct arena→device DMA);
- so a look-ahead fetch is "call the existing promotion API for a prediction", and the paper's
  Expert Look-ahead Buffer becomes a small host-side table per layer.

## 3. Arms

All four stay open; each has a gate that kills it cheaply.

### A. Base measurement (the gate for everything else)

`M1` host cost per offloaded expert · `M2` bus cost per expert · `M3` baseline TPOT · `M5`
acceptance. The study's arithmetic: break-even link bandwidth **18.0 GB/s**; host cost modelled
at 145 µs/expert but measured at **44 µs** by the Strata twin on the same CPU. At 26.8 GB/s a
2.05 bpw expert blob (~1.06 MB) is ~40 µs on the bus — i.e. **one order of magnitude of
uncertainty** on which arm can win. Nothing gets built before M1/M2 are measured.

Instruments already in the tree: `EXL3_MOE_CPU_PROF` (worker per-phase ms), `EXL3_SPLIT_PROF`
(issue/wait percentiles), `EXL3_MOE_SUBMIT_PROF`, `EXL3_MOE_HANDOFF_PROF`, `EXL3_MOE_STREAM_DEBUG`
(probed bandwidth + per-layer streamed/tail split), `EXL3_MOE_CPU_SWAP_DEBUG` (routed/step,
miss %, hook cost).

Protocol for every arm below: single process, ≥5 repeats, same prompt stream, report window mean
and spread; this box cannot A/B across processes (the Strata twin measured 39.3 vs 50.8 tok/s for
identical code) and run-to-run variance is ±5–10%. Never compare across different placement or
cache settings without saying so.

### B. Routing capture and statistics (M4)

The number that sizes the whole direction: per layer, how many distinct experts a decode step
touches, how much consecutive steps overlap (temporal), and how much L overlaps L+1 (cross-layer —
this is the free prefetch window, since layer L's router fires a whole layer before L+1 needs its
experts). Then score the cheap predictors offline against an oracle:

1. recurrence: this layer's experts last step;
2. recurrence on misses only;
3. per-layer frequency top-M (the persisted `.json` already holds this);
4. first-order Markov `P(routing_L(t+1) | routing_L(t))`;
5. oracle.

Strata measured the same ladder on its own engine: profile top-64/layer 5.3%, previous window
22.6%, union of two windows 28.4% (~+5% end-to-end), oracle 100% (+19.4%). Expect the same
shape here; the absolute numbers are what the capture is for.

Capture seam: `params["capture_sens"]` (`modules/block_sparse_mlp.py:1070`) is the only existing
per-layer export of the selection, and it is a scatter of weights into a `(bsz, E)` tensor.
A capture hook must read `selected_experts_bsz1` **after** `:1047` and **before**
`cpu_split_submit` (`:1102`), clone, and never allocate or sync inside a bsz-1 capture window.

### C. Predictive promotion (the cheap arm)

Drive `promote_expert`/`demote_expert` from a prediction instead of a miss, reusing
`_split_free_gpu` as the only safe DMA target and the existing pinned snapshot + event pattern for
any host-side statistics. Gate: B says the cheap predictor recovers ≥ half of the oracle's headroom
and the promotion cost stays under the miss it avoids (Strata: each marginal swap costs 240 µs to
save 276 ± 19 µs — break-even).

### D. Decode-time weight streaming (the paper's arm)

Today weight streaming bails out at decode (`EXL3_MOE_STREAM_MIN_ROWS=32`). Extending it to decode
rows means the GPU computes what the host computes now, with the weights fetched per expert. Gate:
M2 < M1, i.e. the bus must be faster than the host for the same expert — true on this box only if
the 2.05 bpw blob is the resident shape (at 3.05 bpw the blob is ~1.5 MB ≈ 56 µs, close to the
host cost again, so measure both).

### E. Revisit after A/B on the new base

`EXL3_MOE_MEMOPS=1` vs `0` (the WDDM stall the dropped handshake work addressed), `--mtp` draft
length 5 + dynamic (already the production setting; re-tune on v1.6.4), and the PLE n-gram
streaming path, which upstream rewrote (RowTable) — the one place the upgrade is a substitution
rather than an addition, so prefill *and* decode need an A/B against v1.4.9.

## 4. Hazards (read before writing the hook)

1. bsz-1 CUDA graphs capture on the *first* forward for each row count 1..32
   (`ext/libtorch/blocksparse_mlp.cpp:528`, `capture_begin` does a `cudaDeviceSynchronize` under
   `cudaStreamCaptureModeThreadLocal`): no allocation, no `.item()`, no second stream inside it.
2. `selected_experts_bsz1` is overwritten with physical slot ids **in place**; capture the router
   ids before that and translate through `_split_map` / `_split_perm` for checkpoint ids.
3. Promotion must copy into the existing slot tensor (baked pointers stay valid); never replace the
   tensor object.
4. One promo stream and one shared scratch buffer per device — look-ahead fetches for several
   layers serialize there.
5. Statistics readout must use the pinned snapshot + event pattern, never `.tolist()`/`.cpu()`
   mid-decode.

## 5. Order of work

1. A (measure) — no code.
2. B (capture + offline analysis) — code that runs on the capture, validated by its own numbers;
   reuses the persisted-stats format.
3. C (predictive promotion) — the cheap arm, gated on B.
4. D (decode-time streaming) — gated on M2 < M1.
5. E — anything the new base changed that the measurements flag.
