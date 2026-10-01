# Subagent report: Adaptive bits sensitivity probe (2026-09-30)

## Provenance
RECOVERED from the opencode session store (`ses_f0e6cf4f5ffe01fbsKzBZdVIkg`, dispatched
2026-09-30 09:09 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
**Verdict: not expressible.** Bits are global (+1 SWA-group override), never per-layer.

Code points (`exllamav3/cache/kvarn.py:line`):
- `:180` `kvarn_parse_preset` → single `(k_bits,v_bits)`; `:221`/`236` single-side parsers only used for SWA pair.
- `cache.py:149-158` — `Cache` broadcasts the same `**kwargs` to every layer (`layer_type(..., **layer_kwargs)` per `attn.layer_idx`).
- `:895-974` `CacheLayer_kvarn.__init__`: `self.k_bits/v_bits = swa_override or main_*`; `:974` `self.layout = kvarn_make_layout(128,128,k,v)` — one layout per layer, no layer-index input.
- Flow into records: `:2086-2097` seal (`k_tiles/k_bits`, `v_tiles/v_bits` → `kvarn_pack_bits`); `:2212-2251`, `:2416-2418` serve/dequant read `self.k_bits/v_bits` + layout offsets. `:2707` `copy_page` asserts equal widths — cross-preset records incomparable. Per-layer would need a `layer_bits: dict[idx→pair]` plumbed through `Cache`, per-layer `records` shapes, and version bump (`:216` v6).

**Harness (leave-one-degraded):** for layer L in 0..63: all layers K4V4 except L at K4V3 (then V2). Score micro-KLD vs fp16 (`-ntok 8192 -ref fp16`, same-top + median/mean/max). Cost: warmed 8k run ≈ 3s ref + 5s kvarn prefill + overhead ≈ 1–2 min; 64 runs ≈ 2–4 h in one resident-model loop (amortize `model.load`). Cheap proxy first: `ntok 1024, chunk 1024, -dec 0` (seals 8 groups, scores 64 tokens; ~seconds/run) to rank layers, then confirm top candidates at 8k/32k. Cheaper still (no forward): offline dequant-error norm per layer from one calibration prefill — `||X−deq(X)||/||X||` at 4b vs 3b/2b per K/V — then correlate against 4–6 full-KLD spot checks; pick proxy only if rank-correlation holds.

**Tolerant-layer hypothesis:** no per-layer stats exist (grep: only probe-arm timings `:265` in `doc/kvarn-4090.md`; no scales histograms/eref deltas). Torture (kvarn4>q4 on repetitive-code, overlap 0.974 vs 0.941) suggests WHT+Sinkhorn handles low-entropy K — middle layers most tolerant; early layers (outlier channels, sink-adjacent) and final 2–4 layers (logit-shaping) sensitive. V first: K4V2 kept same-top 100% at 270× KLD — V error scales outputs linearly, K error reroutes attention.

**VRAM math** (`:491-494`, `:517-546`): tile = 128 tok × 128-dim slice; payload = 16384·b/8 (8KB@4b, 6KB@3b, 4KB@2b); meta fixed 768B+768B. Tile: K4V4 17920B, K4V3 15872B (−11%), K4V2 13824B (−23%). Per token/layer saving per side-bit: `n_kv·256/8` = 512B @16 kv-heads → 16MB/layer @32k, ≈1GB over 64 layers per side-bit; measured total K4V4 records 2353MB @128k, so full-model V4→V2 saves ≈0.5GB @128k.
