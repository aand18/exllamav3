# KV-cache quantisation benchmark — degradation-vs-length frontier at 256k (agentic)

## 0. Deliverable (what "done" means)

A frontier table of the form **"use quant X up to length L; at length L' it
degrades by D"**, per arm, with practical task-level examples, e.g.:

> `2,2`: NIAH 40/40 @ 64k, 33/40 @ 131k, 26/40 @ 250k; agentic_code_10 @ 250k:
> did not finish within 32k output, +11 turns vs ref, +4 invalid tool calls,
> +3 min wall vs ref-fp16 @ same context (131k ceiling).

plus a recommended serving config per use case (128k agentic vs 256k max).
This is the first plain-`Cache` quality measurement on this box — **KVaRN is
out of scope by operator decision (pinned 2026-10-07)** and its ledger does not
transfer (different serve path, obs #101).

## 1. Why this arch changes what KV quant can break

- Only the **12 QSA layers** hold KV (36 GDN layers are recurrent state).
- The QSA indexer caps every query at **512 blocks × 4 = 2048 attended tokens**
  (`qsa_indexer.py`, obs #140) → damage is *retrieval at depth*: wrong block
  selection (indexer path) or corrupted values inside the attended window.
- **Indexer planes stay fp16** (~320 B/token/layer, obs #56) and are
  quant-invariant → the quantised arm's own forced-choice answers are a valid
  scoring reference; retrieval selection is not what's perturbed by quant.
- fp16 KV ≈ 24 KB/token (≈ 6.3 GB @ 262144); the fp16 indexer floor
  (~3.8 KB/token) is paid by every arm — quant savings are smaller than they
  look, and `FP16 @ 262144` is not expected to boot; measure, don't lore.
- **Causality trick**: KV through depth *L* is identical across depths, so one
  shared cold-prefill prefix serves all depth probes inside a boot. Depth is a
  variable, not a boot.

## 2. Adaptive protocol (2–3 exploratory runs before any ladder)

**Rule: never ladder the full {3,3…8,8} set up front.** Sequence:

1. **S0 endpoints** (2 boots, interleaved): `ref-fp16` @ max bootable size and
   `cur-22` @ 262144. Outputs: fp16 quality curve shape, and whether `2,2`
   already breaks at any depth.
2. **Gate 1 (decision point):**
   - `2,2` clean at all depths → ship `2,2`; ladder runs only as confidence
     nudge at the 2 depths nearest the operator's 200k+ regime (3 boots).
   - `2,2` breaks at L₁ → bisect the *space between* 2,2 and the endpoint that
     holds (test `4,4` at L₁ and just beyond; `3,3` only if 4,4 is borderline;
     skip 5,5/6,6 unless the frontier at high bits is the open question).
   - A depth where every quant fails but fp16 holds → the frontier is the
     memory ceiling, not quantisation; record and stop that arm.
3. **S2 targeted fill**: arms/lengths chosen from S0 data only. Budget target:
   whole battery ≤ ~12 boots, not 40.

## 3. Arms and VRAM maximisation (never leave VRAM on the table)

Benchmark-session config overrides (proposed diff, never committed; production
`config.yml` untouched outside runs):

- **`vision: false`** for benchmark boots — free the vision tower's resident
  bytes (measure the actual amount in pre-flight; `vision_offload` only freed
  ~150–190 MB on the old regime, full-disable value is unmeasured → measure it
  as part of S0).
- **MTP stays ON**: `draft_mode: mtp`, `draft_num_tokens: 3`,
  `draft_cache_mode: Q4` (settled, obs #161). Draft KV + head cost VRAM — that
  is included in the budget on purpose, because production ships it; arms must
  fit *with* drafting live.
- `cache_size` per arm = **max that boots while respecting §5 guards at
  runtime**, i.e. steady-state used + observed inference-time growth + transient
  margin ≤ 24 GB − 200 MB. `max_bootable_tokens` at both memory regimes is a
  first-class result. Note the pool preallocates at load, but **VRAM still
  climbs during inference** (attention working set, dequant/staging
  transients) — the frontier must be sized against the *run* peak, not the
  load peak. Load-OK/run-OOM arms are UNSAFE, not fast (mcl34 lesson).
- `cpu_moe_split_experts: 380` + `EXL3_MOE_CPU_THREADS=16` (`start_tuned.ps1`
  current state) — this regime frees ~724 MB vs the old mcl38 regime; old
  bootability verdicts are void, re-measure.

Arms: `fp16` (reference, max bootable size), `2,2` (production, under trial),
then bisected per §2; `fp8` control arm only if tabbyAPI's exllamav3 backend
exposes FP8 KV — if FP8 (different quantiser path) degrades too, suspicion
moves to the dequant/serve path, not bit width (obs #101 lesson).

## 4. Probes

**P1 depth ladder (primary, new harness).** One shared cold-prefill prefix
(tools + agentic preamble ~12k incl. template overhead, `_kb_mklongctx.py`
`--ladder` mode), then code-flavoured verbatim spans (timeouts, flag names,
error strings — the operator's actual KV content) planted at depths
{8k, 16k, 32k, 65k, 131k, 250k}, queried by **forced choice** (greedy,
auto-scored). NIAH-style batch sizing per user practice: **40 needles per
depth checkpoint (40/40 gate), 200-trial set at the decision length only**
(≥7/8 gate applies to the small set; at p≈0.85 a 4-trial test false-passes
~40% — 4 was the old record, not the target). Per (arm, depth) row: pass
count, pass-rate-vs-depth curve, latency-in-band check, and whether the span's
block was inside the selected 512 (selection failure) or inside the window
(value corruption). Scoring validity is the probe author's job: spans
verbatim-unique, outside the trailing 2048-token window at query time, no
recycler duplicates.

**P2 production task metric (practical examples for the table).**
`agentic_code_10.json` sustained replay (turn i = `messages[0..i]`), greedy,
at 2 lengths: 11–16k (regression screen) and the arm's frontier length.
Reported per arm as the user's example format: finished y/n, output tokens vs
ref, turns used vs ref, invalid tool calls (name/arg schema violations), wall
time vs ref (within-boot ratio only). Reference = `ref-fp16` same turns,
interleaved.

**P3 optional objective tier (KLD/PPL) — run only if cheap.** `eval/ppl.py`
wikitext2 stream is cached (perf.py shares the corpus); KLD of arm vs fp16-KV
logits at 16k and 65k, one offline boot per arm, gated by
`wiki/patterns/kld-median-noise-floor.md` (same-top + mean/max band; median
digit flips are noise). If the harness needs new plumbing, it gets dropped —
P1/P2 carry the decision.

**Determinism pre-flight first**: `EXL3_DSA_QC_STAGE` 0-vs-default cost on the
*plain* path is unmeasured (the 3.9× figure is kvarn-path). Measure on
arm-free boots; quality runs use whichever side keeps fingerprint noise below
the between-arm effect, always the same stage across arms; never mix.

All inherited protocol law applies verbatim: interleaved reference (§0.8.1),
≥4 variants per boot at long context, within-boot ratios, no cross-window
baselines, log-verified prompt lengths via the two-term law
(`0.2383×text + 294.4×n_msg + 9110`, reject batch if any variant exceeds
`cache_size − 256`), 250k fast/slow state matched within boot only.

## 5. Guards — monitor kills the *script*, not just the server

The run harness (not the operator) enforces via `eval/smi_guard.py` +
`_kb_monitor.ps1` (2 s poll, already LIVE):

- **VRAM free < 200 MB → kill harness + server immediately.**
- **Sys RAM free < 2 GB → kill.** (RAM binds at load on this box — flash-knobs
  §0.9 #1; kill means *no arm result*, not a slow run. Swap = reboot territory.)
- Poll continues **throughout generation**, not just at load: the kill trigger
  is the *run* peak (§3), since KV fill + attention transients push VRAM above
  the loaded steady state.
- A guard kill invalidates the arm's row; re-arm only after reducing
  `cache_size` one step and re-logging the fit. Two guard kills at the same
  cell = arm capped at the next lower size, no third attempt.
- `config.yml` `.bak-<date>` + md5 invariant checked only when idle;
  `start_tuned.ps1` always; restarts announced; arms write
  `config.yml.kb-<arm>` pristine-restore snapshots (naming convention: the
  file is the *restore source* for that arm, not the applied config).

## 6. Harness contract — the script must be sound (non-negotiable)

The `_kb_*` harness already proved (and paid for) most of these rules; the new
`_kq_*` scripts inherit them and close the gaps that remain (journaling, hang
detection, verify-before-measure). Reusing `_kb_lib.ps1` patterns is expected,
not optional.

1. **Never clobber a log.** Every run writes
   `logs\kq\<arm>-<probe>-<UTCstamp>-a<attempt>.log`; attempt counters
   increment, files are append-or-create only, results are append-only
   TSV/JSONL rows tagged `arm|probe|depth|variant|attempt`. The LogTag
   collision that already bit once (`_kb_lib.ps1` comment) generalises: a name
   reused across tiers/configs is a defect, not a shortcut.
2. **Per-cell journal + resume.** `_kq_run.jsonl` keyed by
   `arm|probe|depth|variant|configmd5`; a cell is `done` only when its row is
   fully written. Re-running the battery skips `done`, re-runs `invalidated`
   (guard kill / BAD-CONFIG / crash). A crash at minute 55 must cost ≤1 cell
   on resume, never the session.
3. **Verify-before-measure, fail fast.** Boot hard timeout 120 s (ready line
   expected ~65 s); the boot log must echo the arm's values (`cache_mode`,
   `cache_size`, split_experts, draft block) before the first request. Any
   mismatch → kill server, journal `BAD_CONFIG`, next arm. A typo'd YAML key
   must cost seconds, not an hour of silently-wrong data — this is THE way
   runs get lost.
4. **Watchdogs, and they kill the harness too.** 2 s poll: VRAM free <200 MB /
   sys-RAM free <2 GB → kill harness AND server (trip journaled, cell
   `invalidated`). Per-request inactivity timeout (no progress line 180 s) →
   `HANG` row, continue to next cell. Server-boot hangs and request hangs are
   different paths — both bounded. `smi_guard.py` stays the WSL-side backstop
   with its PID-diff rule (only PIDs present after `--before` are killable).
5. **Post-run self-check, loud.** Expected-cells vs journalled-cells count;
   every `done` row has non-null required fields; log-verified prompt tokens
   present for every long-context row; `config.yml` md5 == pristine when idle;
   no orphan `python.exe`. A phase finishing short of its schedule FAILS — it
   does not "complete with fewer rows".
6. **`_kq_selftest.ps1` before the first real boot** (pattern:
   `_kb_selftest.ps1`): full pipeline against a throwaway tiny-cache_size
   config with toy prompts; deliberately trips a guard kill, a BAD-CONFIG
   arm, and a mid-cell kill, then resumes and proves the journal skips `done`
   and re-runs `invalidated`. The battery is gated on selftest passing.
7. Windows mechanics per AGENTS.md: `.bat` via unix2dos, PowerShell
   `Tee-Object` for server stdout into the run's boot log, bounded-output
   commands only, `tasklist` PID-diff for kill safety.

## 7. Run order

1. Pre-flight (3 boots): vision-off VRAM delta; plain-path `DSA_QC_STAGE`
   cost; block-id logging availability probe (decides P1 attribution depth).
2. S0 endpoints (`fp16`, `2,2`) + Gate 1 decision.
3. Targeted bisection arms per §2.
4. P3 objective tier if budget survives.
5. Report → `wiki/reports/` + CSV (`_kb_mkcsv.py` convention): frontier table,
   per-arm quality-vs-depth curves, per-task rows in the §0 example format,
   proposed `config.yml` diff, and an explicit "frontier is memory-bound"
   verdict where quantisation is not the binding constraint.