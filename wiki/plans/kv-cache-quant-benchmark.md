# KV-cache quantisation benchmark — quality-vs-length frontier at 256k (agentic)

## 0. Goal and what "done" looks like

Decide the serving `cache_mode` for Qwen3.8-Flash-Next-exl3 3.05bpw on the
4090 box for the operator's real workload: code-heavy, tool-heavy, opencode
sessions at 128k–256k context. The deliverable is a frontier table written in
this shape:

> "use `2,2` up to 128k for agentic work; `2,2` degrades at 200k —
> NIAH 40/40 @ 64k, 22/40 @ 200k; agentic_code_10 @ 250k: did not finish,
> +14 turns, 5 rejected tool calls, +3 min wall vs uncompressed reference."

Per arm: max context it can honestly serve, accuracy-vs-length curve, and
real-task rows. Production today is `cache_mode: 2,2` @ `cache_size: 262144`
— shipped on perf evidence only; its quality has never been measured. This
plan owns that gap.

**Standing decisions (do not re-litigate):**
- KVaRN is out (operator decision, pinned 2026-10-07: too slow, minimal VRAM
  gain on this model). Plain-cache paths only; its quality ledger does not
  transfer.
- Draft-side KV (`draft_cache_mode`) is a separate, settled knob: keep Q4
  (FP16/Q8 neutral, 3,3/2,2 slower — obs #161/#164). Do not mix those rows
  with main-cache rows; that mislabel cost a retraction once.

## 1. Why this model's quantisation damage is a retrieval-at-depth problem

- Only the **12 QSA layers** hold KV pages; the 36 GDN layers are recurrent
  state and unaffected.
- The QSA attention selector caps every query at **2048 attended tokens**
  (`qsa_indexer.py`, obs #140). So quantisation cannot "blur the whole
  context"; it breaks one of two things, and they are different bugs:
  1. the model looks in the wrong region of the cache (selection failure), or
  2. it finds the right region and reads corrupted values out of it.
  The benchmark classifies misses into these two categories separately.
- The index structures themselves stay uncompressed (~320 B/token/layer,
  obs #56), so retrieval instructions survive quantisation — meaning the
  quantised arm's own forced-choice answers are a valid scoring reference.
- Cost reality: uncompressed KV ≈ 24 KB/token → ~6.3 GB at 262k; the
  un-quantisable index floor is ~3.8 KB/token. Compression savings are
  smaller than the bit count suggests, and uncompressed-at-256k probably
  does not fit this box — measure, don't assume.
- **Causality trick (makes the whole battery cheap):** content written later
  cannot change KV of earlier positions, so one shared cold-prefill
  conversation serves all probe depths inside one boot. Depth is a variable,
  not a boot. This also makes results common-mode against the machine-state
  drift that burned previous sessions.

## 2. Arms and VRAM maximisation

Production baseline (verify each in the boot log before trusting):
`cache_size: 262144`, `cpu_moe_split_experts: 380`, `max_batch_size: 2`,
`chunk_size: 4096`, `draft_mode: mtp`, `draft_num_tokens: 3`,
`dynamic_draft: true`, `draft_cache_mode: Q4`, warmup on. Server starts via
`start_tuned.ps1` (`EXL3_MOE_CPU_THREADS=16` etc).

- Arms: `fp16` (uncompressed reference at max bootable size), `2,2` (the arm
  under trial), then `3,3` / `4,4` / `5,5` / `6,6` / `8,8` as needed —
  symmetric pairs only, chosen adaptively (§3), never the full grid up front.
- **Fill the GPU, don't court it:** benchmark boots run `vision: false`
  (actual bytes freed is unmeasured — measure it in pre-flight; the old
  offload figure ~150–190 MB is not the same knob), while **MTP stays ON** at
  draft 3/Q4 — arms must fit with the draft head resident because that is
  what production runs.
- The limit that matters is peak memory **during generation**, not at load:
  KV fill plus attention/transient working sets push the run peak above the
  loaded steady state. Load-OK/dies-mid-run counts as unsafe, not fast
  (mcl34 lesson). Old bootability verdicts are void: current thr16+split380
  regime frees ~724 MB vs the regime the old ladder booted in.
- `fp8` control arm only if tabbyAPI's exllamav3 backend exposes it — it
  separates "too few bits" from "the read-back path is broken" (obs #101
  lesson: suspect the harness path before the mechanism).

## 3. Adaptive execution — never brute-force the grid

**Stage 0 — pre-flight (3 boots):**
1. `vision: false` VRAM delta (measure, don't inherit).
2. `EXL3_DSA_QC_STAGE` default-vs-0 cost on the plain cache path (the known
   3.9× figure is from a different code path; not transferable).
3. Whether the server logs which cache blocks the attention search selects —
   decides whether miss-classification (§1) is measurable or inferred.

**Stage 1 — endpoints (2 boots):** `fp16` reference + `2,2`, fully
interleaved. This is the "run 2–3 things, then decide" step.

**Gate 1 decision:**
- `2,2` clean at all depths → ship it; ladder reduces to a confidence nudge
  near 200k.
- `2,2` breaks at length L → test only the next-cheaper setting (4,4 first)
  at L and just beyond; 3,3 only if 4,4 is borderline; 5,5/6,6/8,8 only if
  the high-bit frontier is itself the open question.
- Every quantised setting fails where fp16 holds → that is the memory
  ceiling, not quantisation; record and stop that arm.

Budget target: whole battery ≈ 12 boots.

## 4. Probes

**P1 — depth ladder (primary, new harness).** One shared cold-prefill
conversation (tools + agentic preamble ~12k tokens including template
overhead) with 40 planted code-flavoured snippets per checkpoint (timeouts,
flag names, error strings, config values — what the operator's KV actually
contains) at depths 8k/16k/32k/65k/131k/250k, queried by forced choice
(greedy, auto-scored). Scoring validity is the probe author's job: spans
verbatim-unique, outside the trailing 2048-token window at query time, no
recycler duplicates. Each miss classified wrong-region vs corrupted-value
(§1), using the block-id log if stage 0 says it exists. Latency kept in the
known wall-clock band or the row is flagged.

**P2 — gate check.** Same mechanism, 8 trials per length at 131k and 200k,
pass bar ≥7/8; 200-trial set only at the decision length. Eight, not four:
at a true ~15% failure rate, a 4-trial test fakes a perfect pass ~40% of the
time. Report pass counts, not scores.

**P3 — real-task replay.** `agentic_code_10.json` sustained replay (turn i =
`messages[0..i]`), greedy, at 11–16k (regression screen) and at the arm's max
length. Reported per arm: finished y/n, output tokens vs reference, turns vs
reference, rejected tool calls (bad paths, stale symbols), wall-time delta.
This is the operator's actual workload and the harshest fair test of exact
recall — the property quantisation threatens — so it is included because it
is real and unforgiving, not because the model does well there. (The
"code is best case" note in obs #162 is about draft-acceptance *speed*, not
quality; #164 further confounds content vs tool count — do not import either
as a quality assumption.)
The one caveat: it is a single conversation — never generalise from it alone.

**P4 — FP8 control** (optional): if backend-supported, run it to attribute
damage between bit width and dequant code path.

**P5 — objective numeric tier (optional, drop if it needs new plumbing):**
wikitext2 perplexity (corpus already cached) + KLD vs uncompressed-cache
logits at 16k/65k, offline, one boot per arm, judged by the same-top rule
(`wiki/patterns/kld-median-noise-floor.md`): median-digit flips are noise;
same-top and mean/max band carry the verdict.

## 4b. Standard external benchmarks (comparability layer)

In-house probes share one blind spot: the same harness designs and scores
them. External suites catch a systematically wrong harness. All required
repos are already on disk (no installs — production machine rule):

- **NoLiMa** (`/home/dev/NoLiMa`, data local): industry-standard
  forgetting-in-context suite; distractor haystacks defeat keyword matching —
  the exact mechanism at risk. Multi-needle at 32k/64k/128k per arm;
  externally comparable scores.
- **τ-bench** (`/home/dev/tau2-bench`): standard multi-turn tool-agent
  benchmark with database-state verification at episode end — a hallucinated
  tool effect cannot pass grading. Speaks OpenAI-compatible, so tabbyAPI
  serves it directly. Subsets: airline-20 + retail-20 at ≤32k.
- **lm-evaluation-harness** MC screen at 16k/32k: independent scoring
  implementation that cross-validates the P1 scorer itself.
- BFCL: tiebreak only, if P3 and τ-bench disagree.

Rules: reference arm first; subsets frozen before the first boot; arms that
drop outside the reference's published-score band count as damaged, in units
the industry can read.

## 5. Guards — the monitor kills the script, not just the server

- Hard kill: VRAM free < 200 MB **or** system RAM free < 2 GB → kill harness
  AND server. Poll at 2 s **throughout generation**, not just at load — the
  run peak, not the load state, is what must stay under the line (§2).
- RAM is the binding constraint at load on this box (64 GB with the expert
  host arena); treat every load as a RAM event.
- A guard-killed cell is `invalidated`, never recorded as a result; re-arm
  only at the next-lower `cache_size`. Two kills at the same cell = that cell
  is capped, no third attempt.
- Never swap, never "one more attempt" past the guard — a swap here is a
  reboot, not a slow run.

## 6. Harness contract — the script must be sound

1. **Never clobber a log:** `logs\kq\<arm>-<probe>-<UTCstamp>-a<attempt>.log`,
   append-or-create only; results as append-only rows keyed
   `arm|probe|depth|variant|configmd5`. A log name reused across configs is a
   defect (this collision already cost one session).
2. **Journal + resume:** per-cell journal; `done` only when the row is fully
   written; re-runs skip `done`, re-run `invalidated`. A crash at minute 55
   costs ≤1 cell, never the session.
3. **Verify before measure:** boot hard timeout 120 s; the boot log must echo
   the arm's actual values before the first request; mismatch = `BAD_CONFIG`,
   seconds lost, not an hour of wrong data.
4. **Watchdogs bound everything:** boot timeout, per-request inactivity
   timeout (180 s → `HANG` row, next cell), guard thresholds kill the harness
   itself. `eval/smi_guard.py` stays the backstop with its PID-diff rule
   (only PIDs that appeared after `--before` are killable).
5. **Loud post-run self-check:** expected vs journalled cell counts; no
   `done` row with missing fields; log-verified prompt tokens on every long
   row; `config.yml` md5 vs pristine when idle; no orphan processes. A phase
   that finishes short of schedule FAILS.
6. **`_kq_selftest.ps1` gates the battery** (pattern: `_kb_selftest.ps1`):
   full pipeline on a throwaway tiny config, deliberately tripping a guard
   kill, a BAD-CONFIG arm, and a mid-cell kill; resume must prove journal
   semantics. Nothing real runs until selftest is green.
7. Windows mechanics per AGENTS.md: unix2dos `.bat`, `Tee-Object` for server
   stdout into the run's boot log, bounded-output commands only.

## 7. Measurement law (inherited, binding)

- **Interleave the reference.** Machine speed drifts ~10% across a session
  (measured: identical configs 8% apart; consecutive boots drift 0.07%). Run
  reference–candidate–reference–candidate and compare only within the pair;
  if the two reference runs disagree, the window is bad — repeat, don't
  average through it. This rule cost three wrong reports before it was law.
- Within-boot ratios for quality; ≥4 distinct variants per boot at long
  context; ≥10% machine-state effect vs <10% = unresolvable, say so.
- Prompt lengths are read from the server log, never from the file's
  nominal size: `actual ≈ 0.2383×text + 294.4×n_msgs + 9110`; reject the
  whole batch if any variant exceeds `cache_size − 256`.
- 250k has an unidentified fast/slow machine state — match state within a
  boot, never across boots.
- Config hygiene: `.bak-<date>` before edits, md5-invariant checked only when
  idle, per-arm pristine restore snapshots (`config.yml.kb-<arm>` = restore
  source, not applied config), restarts announced, `start_tuned.ps1` always.

## 8. Run order and output

1. Pre-flight (§3 stage 0, 3 boots).
2. Endpoints `fp16` + `2,2` → Gate 1.
3. Targeted bisection arms per Gate 1.
4. Optional tiers: P5 numeric tier; external layer (§4b) for arms that reach
   a verdict.
5. Report → `wiki/reports/` + CSVs (`_kb_mkcsv.py` convention).

**Report house style:** bullets, one fact per bullet; plain technical English
— the reader is a software engineer, not an LLM researcher; explain each
failure by what the model did wrong in the transcript; no insider shorthand,
no boilerplate the reader already assumes.

**Decision rule:** recommend the lowest-bit arm that passes the gate at the
lengths in use. If no arm passes at 256k, the honest output is still the
frontier table: "accuracy-critical work caps at ~128k (3,3/4,4 territory);
256k context is available at 2,2 with a measured deep-recall failure rate of
X — a workload decision, not a benchmark failure." Where every quantised arm
fails but fp16 holds, the frontier is memory-bound, not quantisation-bound —
say exactly that. Proposed `config.yml` diff accompanies the report; the file
itself is never committed.

## 9. Status

Not started. Executor: begin at §3 stage 0; everything above is either
measured (obs/reports cited) or explicitly marked unmeasured.