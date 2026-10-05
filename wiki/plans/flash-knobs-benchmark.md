# Flash-Next knob benchmark plan (handoff) — pp/tg/VRAM/RAM/boot time

Goal: measured impact of every tunable on 3.05bpw Flash-Next serving
(48 MoE layers, MTP head), so `config.yml` stops being tuned by lore.
Metrics per knob setting: boot time (process start → first token),
pp tok/s, tg tok/s, VRAM peak + min-free, sys-RAM free before/after.
All on the 4090 box, Windows native, tabbyAPI server harness.

## 0. Lock the baseline + instrument (no knob changes)

1. Read `tabbyAPI/PERF_FINDINGS.md` fully (prior battery, MTP sweep,
   MEMOPS finding) + `tabbyAPI/config.yml` model section (current
   production values — the baseline). Do NOT re-run the MTP draft
   sweep or the MEMOPS 0/1 comparison (both measured on 1.5.4
   already: draft payoff recorded, MEMOPS gap ~29%).
2. Back up `config.yml` → `config.yml.bak-<date>` before the first
   edit. Every knob change = edit + server restart + verify the
   server reports the intended value in its startup log (many
   settings fail silently on typo — grep the log).
3. Fix the workload (same as PERF_FINDINGS or state the change):
   prompt set, `-tokens 256`, greedy + one temp-1.0 arm, warmed
   server (2 throwaway generations), 3 reps, median quoted.
   Boot time: `Measure-Command` around server start → ready line.
   VRAM: `nvidia-smi` peak + min-free per run; RAM: PowerShell
   free-physical before/after (the 2GB floor rule stands —
   abort any setting that crosses it, mark UNSAFE not slow).
4. Run the baseline battery once, record the reference row. All
   deltas are vs this row, same day, same clocks.

## 1. Knob sweep — one at a time, in this order

For each: change ONLY it, restart, 3 reps, record all six
metrics, restore before the next. STOP a knob early if its first
rep is >10% worse AND trips a guard (record + move on).

1. `cpu_moe_offload_layers`: 38 → 36 / 40 / 42. (VRAM↔RAM↔speed
   frontier; expect monotonic: fewer offloaded = faster until
   the VRAM guard bites. Interacts with everything — first for
   a reason.)
2. `cache_mode`: 5,4 → 6,5 → 8,8. (Quality NOT measured here —
   label all rows perf-only; KLD owns quality separately.)
3. `chunk_size`: 4096 → 2048 / 8192. (pp/VRAM + boot-time effect;
   8192 needs guard headroom — abort past it.)
4. `max_batch_size`: 2 → 1. (Isolates batching: single-stream
   ceiling vs contended reality. Do not "recommend" 1 on speed
   alone — it halves serving capacity; report both numbers.)
5. Draft: `draft_num_tokens` 5 → 3 / off; `dynamic_draft` on/off.
   (Narrow: MTP sweep exists; only re-probe what it left open.)
6. Env threads/streams: `EXL3_MOE_CPU_THREADS` 8 → 4 / 12 / 16;
   `STREAM_T` 6 → 3 / 12 with `STREAM_BATCH_EXPERTS` 48 → 24.
   (Box is 7950X3D 16C — 16 threads contends with serving;
   expect an interior optimum.)
7. Ablations (one each, back to baseline between):
   `EXL3_MOE_CPU_PIN` 1→0, `SWIZZLE` 1→0, `ZERO_COPY` 1→0.
   (Each isolates one `start_tuned.ps1` line; keeps the script
   honest — drop any line that measures zero.)

SKIP (already known): `MEMOPS` 0/1 (29% gap, MEMOPS=0 stands),
`PYTORCH_CUDA_ALLOC_CONF` (unsupported on Windows, no-op),
tensor_parallel (single GPU), rope_* (model-driven, not perf),
`ngram_ram` (tens of GB RAM for PLE table — violates the 2GB
floor rule; revisit only with RAM headroom).

ADDED 2026-10-05 (present in sample, now commented in
`config.yml` — uncomment to test, never both MoE modes):
- `cpu_moe_split_experts` + `cpu_moe_threads` COMBINED knob
  (finer than whole-layer offload + thread shape together;
  mutually exclusive with `cpu_moe_offload_layers`).
- `warmup` true/false (boot time vs cold-start stability).
- `recurrent_checkpoint_interval` (GDN-layer VRAM/compute).
- `cuda_malloc_async` True (current) vs False (upstream
  default since 2026-09-07 — allocator-behavior fork, affects
  all fragmentation findings; verify state before quoting
  any VRAM number).

## 2. Interactions (only the top-2 §1 winners)

Combine the two biggest independent wins, measure once. If the
combo underperforms the sum by >30%, record the interaction
(usually VRAM-headroom contention) and keep the single best.
No three-way combos in this task.

## 3. Report + recommendation

One table: knob → boot/pp/tg/VRAM/RAM deltas vs baseline with
3-rep spreads. Then a concrete `config.yml` diff proposal
(settings + values + expected gain + risk note per line) AND
a `start_tuned.ps1` verdict per env line (keep/drop). No
perf claim without its guard numbers attached.

## Guards (binding, every run)

- VRAM kill under 200MB free; system-RAM abort unless ≥2GB
  free before, kill at 1GB during (swap poisons the machine).
- Server restarts between knob values; verify each value in
  the startup log (silent-typo rule).
- Never compare across days/clocks for deltas (re-run baseline
  if the box state changed); warmed server, 3 reps median.
- Hands-off everything outside the knob list; `git status`
  before commits (docs-only here: the table + the config diff
  proposal — do NOT commit an edited production `config.yml`,
  propose it). Spikes/logs untracked. Atomic commits, push.
