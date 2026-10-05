# Flash-Next knob benchmark plan (handoff) — pp/tg/VRAM/RAM/boot time

Goal: measured impact of every tunable on 3.05bpw Flash-Next serving
(48 MoE layers, MTP head), so `config.yml` stops being tuned by lore.
Metrics per knob setting: boot time (process start → first token),
pp tok/s, tg tok/s, VRAM peak + min-free, sys-RAM free before/after.

## A. Where you are (read this first, it is all load-bearing)

- **Machine:** RTX 4090 24GB + Ryzen 7950X3D (16C) + 64GB RAM,
  Windows 11 + WSL2. GPU headless (display on iGPU), optimally
  cooled (no thermal variance — do not blame thermals).
- **Repos:** tabbyAPI (server) at
  `C:\Users\yoho\Downloads\tabbyAPI` (own git repo — NEVER commit
  there; `config.yml` edits stay working-tree + `.bak`, see §0).
  Docs commits go ONLY to
  `/home/dev/exllamav3/.worktrees/kvarn-cache`, branch
  `wip/kvarn-cache` (verify with `git branch --show-current`
  before every commit).
- **How to run box commands:** from WSL, `cmd.exe /c "..."` for
  reads; write `.bat` files for sequences (unix2dos every bat or
  it silently misparses). Never use bare `curl|wget` to stdout.
- **Model:** `D:\llms\Qwen3.8-Flash-Next-exl3-3.05bpw` (52.5GB).
  Server starts via `start_tuned.ps1` — ALWAYS use it, never
  plain `start.bat` (the tuned env is load-bearing: THREADS=8,
  PIN/SWIZZLE/ZERO_COPY=1, MEMOPS=0, STREAM_T=6/BATCH=48).
- **Production baseline** (from `config.yml`, verify each in the
  startup log before trusting): `cpu_moe_offload_layers: 38`,
  `cache_size: 262144`, `cache_mode: 5,4`, `chunk_size: 4096`,
  `max_batch_size: 2`, `draft_mode: mtp`, `draft_num_tokens: 5`,
  `dynamic_draft: true`, `cuda_malloc_async: True` (differs from
  upstream default False — do not "fix", it is measured state).
- **Known numbers (do not re-measure):** MTP draft payoff +
  `draft_num_tokens` sweep (in `PERF_FINDINGS.md`); MEMOPS 0-vs-1
  gap ~29% (`MEMOPS=0` stands); box spread ±25% between processes
  (fragmentation, NOT thermals/clocks) → §Guards quoting rules.

## 0. Lock the baseline + instrument (no knob changes)

1. Read `tabbyAPI/PERF_FINDINGS.md` fully + `tabbyAPI/config.yml`
   model section. Derive the EXACT workload command from
   `PERF_FINDINGS.md` logs section + `bench.ps1` (prompt set,
   `-tokens 256`, greedy + one temp-1.0 arm). Write the chosen
   command into §4's table header — if you cannot find it, STOP
   and ask rather than inventing a workload.
2. `cp config.yml config.yml.bak-<date>` before the first edit.
   Every knob change = edit + server restart + grep the startup
   log for the intended value (settings fail silently on typo).
   If the server is live-serving users, confirm restart windows
   with the maintainer first (restarts drop connections).
3. Warmed server (2 throwaway generations), 3 reps, median.
   Boot: `Measure-Command { <start-cmd> }` to the ready line.
   VRAM per run: `nvidia-smi --query-gpu=memory.used,
   memory.free --format=csv` before/after + peak from the
   server log. RAM per run: PowerShell
   `[math]::Round((Get-CimInstance Win32_OperatingSystem).
   FreePhysicalMemory/1MB)` before/after.
4. Run the baseline battery once → the reference row. All deltas
   vs this row, same day. If the box state changed (reboot,
   driver, server version), re-run baseline, never reuse old.

## 1. Knob sweep — one at a time, in this order

For each: change ONLY it, restart, 3 reps, all six metrics,
restore before the next. STOP a knob early if its first rep is
>10% worse AND trips a guard (record + move on).

1. `cpu_moe_offload_layers`: 38 → 36 / 40 / 42. (VRAM↔RAM↔speed
   frontier; fewer offloaded = faster until the VRAM guard
   bites. Interacts with everything — first for a reason.)
2. `cache_mode`: 5,4 → 6,5 → 8,8. (Quality NOT measured here —
   label all rows perf-only; KLD owns quality separately.)
3. `chunk_size`: 4096 → 2048 / 8192. (8192 needs guard headroom
   — abort past it, mark UNSAFE not slow.)
4. `max_batch_size`: 2 → 1. (Single-stream ceiling vs contended
   reality. Do not "recommend" 1 on speed alone — it halves
   serving capacity; report both numbers.)
5. Draft: `draft_num_tokens` 5 → 3 / off; `dynamic_draft`
   on/off. (Narrow: MTP sweep exists; only re-probe gaps.)
6. Env threads/streams: `EXL3_MOE_CPU_THREADS` 8 → 4 / 12 / 16;
   `STREAM_T` 6 → 3 / 12 with `STREAM_BATCH_EXPERTS` 48 → 24.
   (16 threads contends with serving; interior optimum likely.
   Env read at server start — restart required, re-verify with
   a settings dump, not assumption.)
7. Ablations (one each, back to baseline between):
   `EXL3_MOE_CPU_PIN` 1→0, `SWIZZLE` 1→0, `ZERO_COPY` 1→0.
   (Keeps `start_tuned.ps1` honest — drop zero-effect lines.)
8. ADDED (in `config.yml`, commented — uncomment to test):
   `cpu_moe_split_experts` + `cpu_moe_threads` COMBINED
   (mutually exclusive with `cpu_moe_offload_layers` — set ONE,
   never both; finer + overlaps own-GPU-compute);
   `warmup` true/false (boot time vs cold-start stability);
   `recurrent_checkpoint_interval` (36 GDN layers);
   `cuda_malloc_async` True→False (upstream default; fork in
   allocator behavior — affects every fragmentation finding).

SKIP with reasons (do not re-litigate): `MEMOPS` (29% gap
measured); `PYTORCH_CUDA_ALLOC_CONF` (Windows no-op); TP
(single GPU); rope_* (model-driven); `ngram_ram` (tens of GB
RAM vs the 2GB floor rule).

## 2. Interactions (only the top-2 §1 winners)

Combine once. Combo underperforms the sum by >30% → record
the interaction (usually VRAM-headroom contention), keep the
single best. No three-way combos in this task.

## 3. Report + recommendation

One table: knob → boot/pp/tg/VRAM/RAM deltas vs baseline with
3-rep spreads + guard numbers attached to every claim. Then a
concrete `config.yml` DIFF PROPOSAL (not an edit: settings +
values + expected gain + risk per line) AND a
`start_tuned.ps1` verdict per env line (keep/drop). If a knob
was skipped per §1 STOP rule, it still gets a row (with why).

## Guards (binding, every run)

- VRAM kill under 200MB free; RAM abort unless ≥2GB free
  before, kill at 1GB during (swap poisons the machine —
  reboot territory, not a slow run).
- Restarts between values; startup-log verification every time.
- No cross-day/clock deltas (re-run baseline on state change);
  warmed, 3 reps, median; single-process discipline for A/B.
- Hands-off everything outside the knob list (no kernel,
  sampler, prefill, MoE-math, record-format, or dispatch work;
  no `master`/fork-overview/rebases; no force-push anywhere).
- Docs commits only in the kvarn-cache worktree on
  `wip/kvarn-cache`: `git status` + `git diff --stat` first,
  explicit per-file `git add`, atomic commits, push. Production
  `config.yml` is NEVER committed (proposed only). Spikes and
  all box logs stay untracked.
