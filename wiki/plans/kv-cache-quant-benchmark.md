# KV-cache quantisation benchmark — quality + VRAM frontier at 256k (agentic coding)

Goal: decide the serving `cache_mode` for Qwen3.8-Flash-Next-exl3 3.05bpw on the
4090 box for the operator's real workload — agentic coding, max 256k ctx,
opencode-driven (xhigh thinking). Current production ships `cache_mode: 2,2`
@ `cache_size: 262144`; its **quality has never been measured** — the
flash-knobs plan shipped it on perf evidence only ("cache_mode quality was
never assessed; KLD owns quality", `flash-knobs-benchmark.md` §3.1). This plan
owns that gap: does KV quantisation cost recall/tool correctness at long
context, and what is the VRAM frontier per bit level?

**KVaRN is out of scope by operator decision (pinned memory 2026-10-07: "too
slow, minimal VRAM gain for this model").** Bare symmetric pairs only; never a
`kvarn*` preset; `-kvt/-kvt_type/-kvsk/-kvsv` are inert here.

## A. Where you are (read first, all load-bearing)

- **Machine:** RTX 4090 24 GB + 7950X3D + 64 GB RAM, Win11 + WSL2. The 64 GB
  RAM is the binding constraint at load (flash-knobs §0.9 #1) — never assume
  RAM headroom once the expert host arena is loaded.
- **Production baseline** (`C:\Users\yoho\Downloads\tabbyAPI\config.yml`,
  verify each in the startup log): `cache_size: 262144`, `cache_mode: 2,2`,
  `cpu_moe_split_experts: 380`, `max_batch_size: 2`, `chunk_size: 4096`,
  `draft_mode: mtp`, `draft_num_tokens: 3`, `dynamic_draft: true`,
  `draft_cache_mode: Q4`, `warmup: true`, `vision_offload: true`.
  Server via `start_tuned.ps1` (THREADS=16, PIN/SWIZZLE/ZERO_COPY; MEMOPS=0).
- **What `cache_mode` actually controls on this arch.** `cache_mode: k,v` =
  per-side bit width 2–8 for the paged KV cache
  (`exllamav3/cache/quant.py:30`, `CacheLayer_qsa_quant` in `cache/qsa.py`).
  On this model only the **12 QSA layers** hold KV pages; the 36 GDN layers are
  recurrent-state and unaffected. Measured cost model (obs #56): fp16 =
  24 KB/token; at `5,4` ≈ 10.75 KB/token = quant KV 576 B + **fp16 indexer
  planes 320 B/layer — the planes do NOT quantise**, so every arm shares a
  ~3.8 KB/token floor at 12 layers. Consequences:
  - fp16 @ 262144 ≈ 6.3 GB KV — will not boot next to the weights (≈ 1 GB
    margin). fp16 is a **quality reference at reduced `cache_size` only**,
    never a 256k candidate.
  - The quantisable part is small; the frontier question is "how many bits does
    recall need", not "how much VRAM does the ladder save".
- **QSA attention caps at 2048 tokens/query** (indexer, `qsa_indexer.py`,
  obs #140): a token attends to at most 512 selected blocks. Long-context
  quant damage therefore shows up as **wrong block selection or corrupted
  values inside the retrieved 2048-token window**, not as general attention
  decay. Design the quality probes around retrieval, not prose summarisation.
- **Determinism:** no RNG in the sampler (greedy = argmax, obs in
  flash-knobs §0.8); the residual nondeterminism is the DSA staged-prefill
  path. **Set `EXL3_DSA_QC_STAGE=0` for every quality run** (pins the online
  dequant path; the default staged path switches numerics past its entry cap
  and is a live risk at 250k on this box). Perf passes may use the default.
- **Draft side is NOT in this plan.** `draft_cache_mode` is measured and
  settled: keep Q4 (obs #161; cross-category 2,2/3,3 slower, obs #164). Do not
  conflate `draft_cache_mode` rows with `cache_mode` rows — that mislabel
  cost a retraction once (flash-knobs report footnote).
- **Existing cache_mode rows are perf-only and stale-baseline suspect.** The
  old Phase A rows (`a04-cq22/05-cq44/06-cq88`, 32k grid) predate the
  interleaved-reference rule (§0.8.1) and their Δ-VRAM column is internally
  inconsistent (4,4 shows a larger VRAM delta than 2,2 — impossible under the
  bits-monotone allocation). Treat them as directional only; re-measure with
  an interleaved baseline before quoting.

## B. Arms

| id | arm | role |
|---|---|---|
| `ref-fp16` | `cache_mode:` unset/FP16, `cache_size` = max bootable (est. ≤131072) | quality reference ceiling; NOT a 256k candidate |
| `cur-22` | `2,2` @ 262144 | current production; the arm under test |
| `q33` | `3,3` @ 262144 | ladder |
| `q44` | `4,4` @ 262144 | ladder |
| `q55` | `5,5` @ 262144 | ladder (historical production was `5,4`) |
| `q66` | `6,6` @ 262144 | ladder; expected first arm that fails to boot at 262144 |
| `q88` | `8,8` @ 262144, else max bootable | VRAM-frontier bound |

Symmetric pairs only (flash-knobs §0.8). Asymmetric pairs are not tested.
For any arm that will not boot at 262144, descend `cache_size`
(224k → 192k → 160k → 128k) until it boots and record `max_bootable_tokens` —
that table **is** the context-vs-bits frontier decision the operator needs.

## C. Phase V — VRAM frontier (offline first, live to confirm)

Offline `eval/perf.py` via the bench harness (`-cq <pair> -cs <size>`, server
stopped; perf.py reads nothing from config.yml — pass production knobs
explicitly: `-mcs 380 -ambs 2 -chunk_size 4096`). For each arm × cache_size
cell: boot/VRAM peak/min-free + RAM min-free. Live-confirm every arm that wins
a cell — the live server sits **+1.7–2.5 GB** above the offline harness
(flash-knobs §0.7), so offline "fits" proves nothing. Guards: kill <200 MB
free VRAM, abort <2 GB RAM before load.

## D. Phase Q — quality at length (live server, `EXL3_DSA_QC_STAGE=0`)

Primary metric is **task correctness on the real workload**, not KLD. Three
probes, all greedy (temperature 0), same message sets across arms:

1. **Tool-call divergence @ ~16k** — replay `eval/prompts/agentic_code_10.json`
   progressively (turn *i* sends `messages[0..i]`, the §0.8 sustained-replay
   protocol). Score = tool-call name+args mismatch and final-answer divergence
   vs `ref-fp16` at the same turn index. `cur-22` is the arm under trial; the
   other quant arms only run turns where 2,2 diverges (screen) unless 2,2 is
   clean, then spot-check at Tier 2 depth.
2. **Needle recall @ arm max safe context** — `eval/kvarn_needle.py`
   (`_spike17needle.bat` is its record wrapper). Gate: ≥ 4/4 at ~200k, the bar
   Flash-Next set at 131k. Quant arms that drop a needle while `2,2` holds it
   are worse at that length regardless of speed.
3. **Late-retrieval agentic probe @ ~250k** — one synthesised conversation
   from `eval/_kb_mklongctx.py` with a tool-result buried early and a
   question requiring it emitted at the end. Use the two-term calibration
   `actual ≈ 0.2383×text + 294.4×n_msg + 9110`, reject the batch if any variant
   exceeds `cache_size − 256`, and confirm prompt tokens from the server log
   before quoting (flash-knobs §0.7 — files labelled by target failed once).

Protocol rules carried over verbatim from flash-knobs §0.8/§0.8.1: interleave
the reference arm with every candidate; verify the reference is stable across
its own interleaved runs before quoting any ratio; no cross-window baselines;
≥4 distinct variants per boot at long context; a draft-acceptance-like ±3 pp
band is noise, and quality deltas inside the baseline's own spread are "no
evidence", not parity.

Optional tier (only if a quality delta appears and needs attribution): KLD of
arm vs fp16-cache logits at 16k via the qbench/microkld machinery, gated by
`wiki/patterns/kld-median-noise-floor.md` (same-top + mean/max band, not
median digits).

## E. Decision rule

Recommend the **lowest-bit arm** that simultaneously: boots at 262144 with
>200 MB sustained free; needle ≥4/4 at ≥200k; probe-1 divergence ≤ ref within
noise; probe-3 correct. If only `2,2` satisfies the frontier but fails a
quality probe, the honest output is the frontier table + the quality cost of
capping context (e.g. "8,8 at 160k beats 2,2 at 256k on recall") — the
context-vs-bits trade is the operator's call, so present it as a table, not a
verdict. Output is a proposed `config.yml` diff (never committed) + one-line
rationale per arm.

## F. Guards (inherited, binding)

- Everything in flash-knobs "Guards": `config.yml` → `.bak-<date>` before any
  edit, md5-invariant check only when idle, startup-log verification per value,
  restarts announced (server serves live sessions), restart between values,
  `start_tuned.ps1` always.
- Interleaved reference; within-boot ratios; no ratio before the last arm
  lands; log-derived prompt lengths only.
- Docs commit in this worktree only (`wip/kvarn-cache`), `config.yml` never
  committed.

## G. Run order

1. Baseline `cur-22` re-measure (interleaved anchor, quality probes).
2. `ref-fp16` at max bootable size — reference curves.
3. Ladder `3,3 → 4,4 → 5,5 → 6,6 → 8,8`, VRAM-frontier cells first (cheap,
   kills non-booters), then quality probes for survivors.
4. Frontier table + `config.yml` diff proposal + report under
   `wiki/reports/` (+ CSV via `_kb_mkcsv.py` convention).

Status: not started.