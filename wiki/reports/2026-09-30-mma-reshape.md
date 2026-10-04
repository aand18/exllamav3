# Subagent report: MMA reshape investigation (2026-09-30)

## Provenance
RECOVERED from the opencode session store (`ses_f0e6cf509ffedCci1H4JQXdg1Y`, dispatched
2026-09-30 09:09 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
Serve-kernel MMA analysis (`exllamav3/modules/attention_fn/kvarn_triton.py`):

**Dot sites (current `_kvarn_online_serve_kernel`):**
- `1646`: `sb = tl.dot(k_tile.to(fp16), qwT)` — QK body, `(TOK=16, HD) @ (HD, QPAD) → (16, QPAD)`
- `1651`: `st = tl.dot(k_tile.to(fp16), qfT)` — QK tail, same shapes (skipped when tile all-body)
- `1730-31`: `d = tl.dot(trans(v_tile.to(fp16)), e.to(fp16))` — EV, `(HD,16) @ (16,QPAD) → (HD,QPAD)`
- Operands fp16, accum fp32 (default). `HD = SL*128` (128/256/512); `QPAD = next_pow2(QPK)`, `QPK = qh/kvh` (typ. 4→`QPAD=4`, 8→`QPAD=8`).

**Why SIMT:** `tl.dot` needs M≥16, N≥8, K≥16 for `m16n8k16`. QK has M=16, K=HD OK, but **N=QPAD=4 (<8) → guaranteed SIMT**; N=8 is the bare minimum with masked lanes. File already documents this pattern at `1727-29` (old `(QPAD,TOK)` EV with M=8 was SIMT-suspect; fixed by transposing to M=HD). QK cannot be fixed by transposing — N is inherently QPAD.

**Triton/sm_90:** no blocker. MMA `m16n8k16` fp16 path exists in Triton 3.8; `allow_tf32` is irrelevant (fp32-input flag only). sm_90 WGMMA wants M≥64 but legacy MMA still lowers fine.

**Proposal — pad N to 16:** define `QPN=16` (constexpr); load `qw/qf` as `(16,HD)` with `qmask` zero-fill (already masked loads at `1538-41`); `qwT/qfT: (HD,16)`; dots become `(16,HD)@(HD,16)→(16,16)`, EV `(HD,16)@(16,16)→(HD,16)`; `m/l: [16]`, `acc: [16,HD]`; slice `[0:QPK]` on stores (`1735-43`). Critical: mask padded score columns to `-inf` (`sc[:, QPK:] = -inf`) and zero padded `e` columns, else `k·0=0` corrupts softmax. QW tiles from `kvarn_triton_qwht:1386` are row-major contiguous fp32 → already MMA-friendly; K/V tiles are in-register dequant → cast at dot, fine.

**Speedup/risk:** dots move SIMT (~single-digit TFLOPS) → Tensor Cores (~300+ TFLOPS fp16 on 4090) at ~2-4× dot FLOPs. Serve is 4.0ms/step but dequant-load bound, so expect ~1.2-2× serve (~2-3ms), ~10-15% tg (36.6→~40-42 tok/s) — not full Bee gap alone. Numerics: fp32 accum kept; fp16-MMA rounding ~1e-4 RMSE (per spike6 note) — KLD same-top gate should hold.

**Test plan:** 1) twin test padded-vs-torch ref, RMSE<5e-4 + sliced-lane exactness; 2) PTX check `tt.dot→mma` not SIMT; 3) KLD same-top 100% @8k; 4) Kineto A/B tg @64k.
