# Subagent report: Fused decode investigation (2026-09-30)

## Provenance
RECOVERED from the opencode session store (`ses_f0e6cf4fcffeCCVVxRpIJH2cCQ`, dispatched
2026-09-30 09:09 UTC): verbatim final-report text below (the subagent's prompts
capped these at ~400 words / 450 words, so this IS the complete
deliverable, not a compression). Pre-convention dispatch (no report
file was written at the time). Intermediate reasoning and tool
traces were not preserved. See `wiki/reports/README.md`.

---
Launch sequence per layer per decode step (`_try_kvarn_online_decode`, imageless):

1. `update_kv_direct` → fused store: WHT 1 + `_store_row` 1 launch, grid `(2*kvh)`. 1 sync (`status.tolist`).
2. `qwht` grid `(qh)` — fp16→fp32+WHT, persistent bufs, no sync.
3. `serve` grid `(kvh, gc)`, gc≈512@64k / ~1024@128k + `combine` grid `(qh)`. Serve writes `m/l` (kvh×QPAD×gc fp32, ~130KB) + `acc` tile (kvh×QPAD×gc×HD fp32) to DRAM; combine reads all back, writes `out_b` (qh×HD fp32, ~16KB). 1 sync (`int(flag[0])`).
4. Torch: `tpos` cat+2×arange, `kvarn_online_tail` → fused `tail_gather` grid `(R,kvh)`, R=sink(128)+tail_eff(~128–512); `bmm` Qh×Kt → `st` (kvh×qpk×R); `tail_reduce` grid `(qh)` writes `tail_m/den` (qh fp32) + `tail_num` (qh×HD fp32). 0 sync steady (cert flag skips `(~ev).any()`).
5. `merge` grid `(qh)`: reads serve m/l + out_b + tail triple, writes fp16 out.

Traffic/layer/step (ex: kvh8,qpk4,HD128): acc ~8MB@64k/~16MB@128k R+W (dominant), rest <0.5MB.

Top-2 fusion:
1. **serve+combine** (biggest): acc never hits DRAM, saves ~16–32MB R+W/layer/step (~0.5–1GB/s aggregate). Blocker: grid mismatch `(kvh,gc)` vs `(qh)`; combine needs cross-group reduce + single-warp WHT (`num_warps=1`) vs serve `num_warps=4`/stages=1; sticky-flag sync in between.
2. **tail_reduce+merge** (easiest): same `(qh)` grid; kills `tail_num` + `out_b` + stats round-trips (~50KB/layer) and 1 launch + torch gap. Blocker: `st` bmm stays torch (needs Kt/Vt); exact-fp16/bf16 path + exrev mask must move in-kernel; PARITY twin asserts on `tail_m/den/num` intermediates.

Bubbles (~9ms): per-layer flag/status syncs × N_layers serialized, torch gaps (cat/bmm/reshape/float) between tiny-grid kernels (qh≤64, tail R×kvh), launch overhead dominates 0.1–0.3ms kernels.

Test: `EXL3_KVARN_TRITON=1 EXL3_KVARN_IMAGELESS=1 EXL3_KVARN_TRITON_PARITY=1` twin ON vs OFF bit-exact @8k; Kineto before/after @64k/128k; WSL builds, Win 4090 runs only (triton-windows).
