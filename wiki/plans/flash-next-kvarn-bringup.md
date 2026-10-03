# Flash-Next kvarn bring-up plan (handoff) — parity + VRAM map, NO perf claims

Goal: prove kvarn is CORRECT on Qwen3.8-Flash-Next (2.05bpw_h4_ng4,
`D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw`, 33.93GB) and map what fits
on the 24GB 4090 at which ctx/offload settings. Decode will be
CPU-offload-bound: never quote tok/s comparatively.

Branch: `wip/kvarn-cache` (tip, do NOT rebase). Scope: bring-up
validation only (no kernel changes, no server integration). STOP
gates are binding (see per phase).

## 0. Facts (verified 2026-10-03, do not re-derive)

- Model: 48 layers (12 full-attn + 36 linear_attention, 1:3 from
  layer 0), head_dim 256 ✓ (triton gate passes), 24 q-heads /
  2 kv-heads (qpk 12, qpad 16), hidden 2560, **512 experts ×
  10 active**, max_pos 262144. Weights 33.93GB > 24GB VRAM, so
  experts MUST live partly on CPU (tabbyAPI uses
  `cpu_moe_offload_layers: 38`; microkld has `-mcl /
  --moe_cpu_offload` — read its `--help`).
- Serving reference (`tabbyAPI/config.yml`, model section):
  cache 262144, cache_mode 5,4, chunk 4096, rope auto/YaRN.
  For kvarn tests use `-cq kvarn4` (K4V4 scope); q8 (`-ref q8 /
  -cq 8,8) where fp16-cache spills, per the ledger q8 rule.
- Kvarn covers FULL-attention layers only (expect 12); the 36
  linear layers must decline (same precedent as GDN on 27B).
  A misrouted linear/indexer layer = silently wrong attention,
  not a crash — Phase 1 exists to catch exactly this.

## 1. Phase 0 — download + load (STOP on failure)

1. Verify download complete: 5 shards
   (`model-00001..5-of-00005.safetensors`) + `model.safetensors.
   index.json` + `config.json`; total ~33.93GB. If `hf` stalled,
   resume same command (it's idempotent); do NOT re-download
   finished shards. Check D: free >10GB after.
2. Confirm `config.json` on disk matches §0 (layer_types,
   head_dim, kv heads, experts). Any mismatch: STOP, report.
3. Load test (fp16 cache, tiny): model loads on cuda:0 with
   offload per `-mcl` docs; one 512-ctx forward; prints shapes.
   Failure (OOM, arch error, missing kernels): STOP with the
   exact traceback + `nvidia-smi` peak. Do NOT tune flags to
   force it — report first.

## 2. Phase 1 — per-layer cache audit (the load-bearing check)

1. Instrument (print, no behavior change — or a 20-line spike,
   never commit): for each of the 48 layers, which cache class
   serves it (kvarn vs fallback) and WHY (layer_type string,
   is_swa, gate result).
2. Bar: EXACTLY the 12 full-attention layers take kvarn, all 36
   linear decline, zero indexer/unknown types take kvarn.
   Anything else: STOP, do not proceed to KLD (numbers from a
   misrouted layer are fiction).
3. Record head_dim/qpk/kvh per full layer + triton-gate pass.

## 3. Phase 2 — KLD parity, small ctx first

1. `eval/kvarn_microkld.py -m <dir> -cq kvarn4 -ntok 2048
   -chunk 2048` (+ fp16 ref if it fits VRAM with offload, else
   `-ref q8` per the q8 rule + one-time ≤32k proxy calibration:
   kvarn-vs-fp16 AND kvarn-vs-q8 digit-equal where both fit).
2. Bar: same-top 100%, mean <1e-4. Then `-ntok 8192`,
   `PARITY=1` run (expect asserts to trip on a new arch — that
   is the test working; fix-forward only trivially, else STOP).
3. No tg/pp quoting beyond "completed in Ns" (offload-bound).

## 4. Phase 3 — VRAM map (the deliverable)

Grid (skip cells that OOM — record OOM, do not fight it):
ctx {8192, 32768, 65536, 131072} × cache {kvarn4, q8, fp16}
with offload per `-mcl` as needed. Per cell: peak allocated,
min-free, OOM Y/N, KLD same-top (parity cells only, not every
map cell). Present as one table: ctx × cache → peak / fits /
KLD. Watchdog (`smi_guard.py`, kill under 100MB free) on EVERY
run; 0-used before/after; warmed box; hot counts for any number
quoted (peaks are per-run maxima, 1 run suffices + 1 confirm).

## 5. Non-goals + guards (binding, cf. task-6 plan §5/§5b)

No kernel edits, no perf optimization, no server/tabbyAPI
integration, no sampler work, no 262k ctx (config max; stay
≤131072 this task). Hands-off everything outside scope;
`git status` + `git diff --stat` before every commit, explicit
per-file `git add`. Spikes/bats/logs untracked. Ledger:
one `doc/kvarn-4090.md` section (per-layer audit table + KLD
+ VRAM map), atomic commits, push when green.

## 6. Done means

- §1 audit: 12 full kvarn / 36 decline, recorded.
- KLD same-top 100% at 2k + 8k (+ PARITY green or triaged).
- VRAM map table filled (peaks + OOM boundaries).
- Ledger section + push. Perf numbers, if any were observed,
  are labeled offload-bound and NOT compared to anything.
