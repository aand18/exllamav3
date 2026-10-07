# KV-cache quantisation benchmark — quality-vs-length frontier at 256k (agentic)

## 0. Goal and what "done" looks like

Decide the serving `cache_mode` for Qwen3.8-Flash-Next-exl3 3.05bpw on the
4090 box for the operator's real workload: code-heavy, tool-heavy, opencode
sessions at 128k–256k context. The deliverable is a frontier table written in
this shape:

> "use `2,2` up to 128k for agentic work; `2,2` degrades at 200k —
> P1 ladder 40/40 @ 64k, 22/40 @ 200k; long agentic replay @ 200k: did not
> finish, +14 turns, 5 rejected tool calls, +3 min wall vs the band
> reference defined in §4."

Per arm: max context it can honestly serve (measured by the §3 stage-0.5 descent, not
by a single boot), accuracy-vs-length curve, VRAM-at-peak and speed at each
tested length, and real-task rows. Production today is `cache_mode: 2,2` @
`cache_size: 262144` — shipped on perf evidence only; its quality has never
been measured. This plan owns that gap.

Reference rule for every row (binding): comparisons run only within prompt
lengths that fit **both** arms' pools. Below the uncompressed ceiling the
reference is fp16 at the same depth; above it, the highest-bit arm that boots
there, marked as a proxy. Rows with no higher-bit coverage are reported
against the arm's own 8k baseline and labelled weaker evidence — never
against a reference that could not run there.

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
  (`qsa_indexer.py`, obs #140). So quantisation cannot blur the whole
  context; it breaks one of two things, and they are different bugs:
  1. the model looks in the wrong region of the cache (selection failure), or
  2. it finds the right region and reads corrupted values out of it.
  Misses are classified into these two categories where measurable (§4 P1);
  where the block-id log does not exist the miss-type column is N/A rather
  than inferred — do not let an unmeasurable distinction drive probe design.
- The index structures themselves stay uncompressed (~320 B/token/layer,
  obs #56), so retrieval instructions survive quantisation — which supports
  *attributing* a miss to value corruption rather than broken retrieval. It
  does not score anything: scoring is against the planted ground truth (§4).
- Cost reality, both units (GiB/GB mixing is how margins get silently eaten):
  uncompressed KV = 12 × 2 × 2 × 256 × 2 B = **24,576 B/token** → 6.44 GB
  decimal (6.00 GiB) at 262144; plus the un-quantisable index floor
  (12 × 320 B = 3,840 B/token → 1.01 GB) the loaded total is **7.45 GB
  decimal (6.94 GiB)**. Compression savings are smaller than the bit count
  suggests, and uncompressed-at-256k probably does not fit this box —
  measure, don't assume.
- **Causality trick (makes the whole battery cheap):** content written later
  cannot change KV of earlier positions, so one shared cold-prefill prefix
  serves every depth probe inside one boot — as **separate requests against
  the cached prefix, never an accumulating conversation**: probe questions
  and answers must not become context for deeper probes. Depth is a variable,
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
  `8,8` doubles as the high-bit anchor: it bounds the integer-quant path but
  rides the same dequant code, so it cannot separate bit-width from code-path
  effects — claim only what it covers. (There is no FP8 cache mode on this
  backend: `model.py:create_cache` accepts k,v integers 2–8 plus legacy
  Q4/Q6/Q8/FP16, and `CacheLayer_quant` asserts integer bits. An earlier
  draft's FP8 control arm is dropped, resolved here, not deferred.)
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
- All probes run single-stream; the table is labelled single-stream.
  A dual-stream run at full length is a capacity question (two 262k streams
  exceed the 262144 pool by construction), not a quantisation question —
  parked as a named capacity follow-up, not part of this battery.

## 3. Adaptive execution — never brute-force the grid

**Step 0 — harness + selftest (hard precondition, §6.6).** No _kq_* scripts
exist yet; writing them and passing selftest comes before pre-flight, not
after. Nothing below runs until selftest is green.

**Stage 0 — pre-flight (3 boot-pairs):**
1. `vision: false` VRAM delta (measure, don't inherit).
2. `EXL3_DSA_QC_STAGE` default-vs-0 cost on the plain cache path (the known
   3.9× figure is from a different code path; not transferable).
3. Whether the server logs which cache blocks the attention search selects —
   decides whether miss-classification (§1) is measured or N/A.
4. External-layer feasibility: NoLiMa cut to tiebreak-only (no further
   task unless P1 and LongBench-v2 disagree); τ-bench runner smoke against
   tabbyAPI with the agent-only split declared (§4b); LongBench-v2 item
   length filter validated (≤32k screen + 32–128k band addressable).
   Any layer that fails feasibility is cut here, loudly, before it consumes
   arm boots.

**Stage 0.5 — bootability descent per arm (the "max context" column).**
`cache_size` 262144 → 196608 → 163840 → 131072 → 98304, logging load peak,
run peak, and RAM at each step. An arm's ceiling is the largest size that
boots **and** survives a full generation run under §5 guards. Quality cells
run at or below each arm's own ceiling — never above it.

**Stage 1 — endpoints (2 boot-pairs: ref+cand, ref+cand).** `fp16` reference
+ `2,2`, interleaved per §7. This is the "run 2–3 things, then decide" step.
(The pair is the budget unit everywhere: one candidate boot bracketed by
reference boots. "2 boots" in older drafts meant one pair without the
stability check — that configuration is forbidden by §7.)

**Positive-control gate (from Stage 1 data, before Gate 1):** fp16 must
score ≥39/40 on the P1 ladder at 64k, or the probe harness is invalid and
nothing downstream is interpretable. Fix the harness, do not proceed.

**Gate 1 decision:**
- `2,2` clean at all depths → confirm, no change (it is already shipped; the
  report announces confirmation, not a decision). Minimum coverage still
  applies: run 4,4 and 3,3 at the two depths nearest 200k+ so the frontier
  table has three curves, not one verdict.
- `2,2` breaks at length L → test the next-higher-fidelity setting first
  (4,4 costs *more* memory than 2,2 — "cheaper" wording is retired): 4,4 at
  L and just beyond. **4,4 passes → 3,3 must be tested at the same lengths**
  (the decision rule wants the lowest-bit passer; a clean 4,4 never excuses
  skipping 3,3). 4,4 fails → go up, never down.
- Every quantised setting fails where fp16 holds → that is the memory
  ceiling, not quantisation; record and stop that arm.

**Budget: boot-pairs with a ceiling, plus a time ledger.** Target ≈ 6 pairs
(12 boots): 3 pre-flight + 2 endpoint + 1–2 bisection. P3's per-arm replay
boots, P5's one-boot-per-arm, and the external layer's arm boots are
line-itemed separately — they are outside the 12. Each phase lists
request-count × expected wall time (cold prefill ≈ 2 min at 250k in the fast
state + decode tails from standing battery rates); the first cut when the
ceiling binds is P5, then the external layer's longest lengths, in that
order. Greedy forced-choice accuracy rows run once per arm (no per-cell
interleaving — drift affects timing, not token choice); interleaved
bracketing applies to wall-time rows only, and only where the brackets agree
*and* share machine state, else wall is reported as unresolvable and turns /
rejected calls carry the row.

## 4. Probes

All quality probes pin `reasoning_effort: xhigh` (production reality —
opencode sends it; the template's silent medium default would measure a
different configuration) and greedy decoding.

**P1 — depth ladder (primary, new harness).** One shared cold-prefill prefix
(tools + agentic preamble ~12k tokens including template overhead) with
planted code-flavoured spans (timeouts, flag names, error strings, config
values — what the operator's KV actually contains) at depths
8k/16k/32k/65k/131k/**240k**, queried by forced choice, one independent
request per depth against the cached prefix. **"Depth" means total prompt
tokens as logged**, plant positions sized so each request fits: at 240k
total the batch keeps ~20k headroom under `cache_size − 256` per the §7
reject rule (the deepest cell of older drafts collided with it). 40 trials
per checkpoint; the 200-trial set runs at the decision length (deepest
length under test, normally 240k) with pass bar **≥170/200** (15% fail
budget at ±5% precision; a 165–175 borderline triggers a second 200-set
rather than a verdict). Before building the harness, run the §7 size formula
against the actual message packaging (spans batched into shared tool-result
messages, queries batched per request) — per-message scaffolding at ~294
tokens/message dominates if every span is its own message. Scoring validity
is the probe author's job: spans verbatim-unique, outside the trailing
2048-token window at query time, no recycler duplicates. Each miss classified
wrong-region vs corrupted-value (§1) where the block-id log exists, else
N/A. Latency is checked against that boot's own reference-run band — no
band at 240k without a same-boot, same-state reference, in which case the
latency column is dropped for that cell, not filled with cross-boot numbers.

**P2 — gate check.** Same mechanism, 8 trials per length at 131k, 200k, and
240k, pass bar ≥7/8 per length. Eight, not four: at a true 15% failure rate
a 4-trial test fakes a perfect pass 0.85⁴ ≈ 52% of the time (the "~40%" in
older drafts matches a 20% rate: 0.8⁴ ≈ 41%). Note what the bar is: a ≥7/8
gate still clears a true-85% arm ~66% of the time — it is a screen, and the
200-trial set in P1 is the arbiter. Report pass counts, not scores.

**P3 — real-task replay (full file, ~18k, plus a stretched fixture).**
`agentic_code_10.json` sustained replay (turn i = `messages[0..i]`), all 23
messages — the file replays to ≈17.8k prompt tokens as server-logged, and
the "11–16k" of older drafts understated its full length. That covers the
regression screen honestly labelled ~18k. The long row needs a stretched
fixture: extend the code_10 history with the `_kb_mklongctx.py` recycler
(tagged `[pass N]`, tool schemas attached once), validated with the §7
two-term law before first boot, at 128k and the arm's ceiling. If the fixture
fails validation, the long-task row is dropped from §0 and long-context
evidence rests on P1/P2 alone — stated, not silently missing. Reported per
arm: finished y/n, output tokens vs reference, turns vs reference, rejected
tool calls (bad paths, stale symbols), wall-time delta only where §7
bracketing holds. This is the operator's actual workload and the harshest
fair test of exact recall — included because it is real and unforgiving, not
because of any speed lore. (The "code is best case" note in obs #162 is
about draft-acceptance *speed*, not quality; #164 further confounds content
vs tool count — do not import either as a quality assumption.)
The one caveat: even stretched, it is one conversation family — never
generalise from it alone. Any arm showing a quality delta re-runs P1/P2 with
`draft_mode` off (one extra boot per suspect arm) to separate recall damage
from drafting damage; memory-frontier rows stay MTP-on.

**P4 — dropped.** Was the FP8 control; no such mode exists on this backend
(§2). `8,8` is the high-bit anchor.

**P5 — objective numeric tier (optional, first cut on budget pressure):**
wikitext2 perplexity (corpus already cached) + KLD vs uncompressed-cache
logits at 16k/65k, offline, one boot per arm, judged by the same-top rule
(`wiki/patterns/kld-median-noise-floor.md`): median-digit flips are noise;
same-top and mean/max band carry the verdict. If it needs new plumbing, it
gets dropped — P1/P2 carry the decision.

## 4b. Standard external benchmarks (regression + scorer-validation layer)

In-house probes share one blind spot: the same harness designs and scores
them. External suites catch a systematically wrong harness. They do **not**
reach the 200k+ decision lengths, so they are regression evidence and
independent scoring — never frontier evidence. Feasibility is gated in
stage 0.4; anything failing there is cut before spending arm boots:

- **NoLiMa** (`/home/dev/NoLiMa`): forgetting-in-context suite whose
  distractor haystacks defeat keyword matching — the exact mechanism at
  risk. Demoted to tiebreak-only: it needs dataset generation, 64k/128k
  run configs that don't exist (max shipped is 32K), and a venv build, for
  information the P1 ladder plus LongBench-v2 already cover. Run it only
  if P1 and LongBench-v2 disagree.
- **τ-bench** (`/home/dev/tau2-bench`): multi-turn tool-agent benchmark with
  database-state verification — a hallucinated tool effect cannot pass
  grading. Backend defaults point agent, user simulator, NL assertions, and
  env interface at the same cloud model, so the split must be declared:
  **agent = tabbyAPI arm under test; user simulator + judges = one pinned
  external model**, costed as an API dependency. Subsets: airline-20 +
  retail-20 at ≤32k. Agent-only; pointing any judge at the arm is
  self-grading and invalidates the layer.
- **LongBench-v2 via lm-evaluation-harness** (primary external layer):
  503 multiple-choice questions, contexts 8k–2M words, including code
  repository understanding and long-dialogue history — the closest
  off-the-shelf match to code-heavy agentic sessions at length. The
  harness's OpenAI-completions backend takes a `base_url`, so it points at
  tabbyAPI with no new installs; MC scoring doubles as the independent
  implementation that cross-validates the P1 scorer. Cost control is
  mandatory: filter items by context length (≤32k screen + 32–128k band),
  drop the tail items that exceed the cache by construction. Each item
  carries its own haystack (no shared-prefix trick) — budget prefill per
  item.
- BFCL: not on disk — dropped unless fetched; P3 already covers
  tool-call validity.

Rules: reference arm first; subsets frozen before the first boot; arms that
drop outside the reference's band count as damaged, in units the industry
can read — at the lengths these suites cover only.

## 5. Guards — the monitor kills the *script*, not just the server

- Hard kill: VRAM free < 200 MB **or** system RAM free < 2 GB → kill harness
  AND server. (Note: the inherited `_kb_lib.ps1` default kills RAM at 1 GB —
  the 2 GB figure here is a deliberate override for this battery; new
  scripts must set it explicitly, not inherit.) Poll at 2 s **throughout
  generation**, not just at load — the run peak, not the load state, is
  what must stay under the line (§2).
- RAM is the binding constraint at load on this box (64 GB with the expert
  host arena); treat every load as a RAM event.
- A guard-killed cell is `invalidated`, never recorded as a result; re-arm
  only at the next-lower `cache_size` from the §3 stage-0.5 ladder. Two kills at the same cell =
  that cell is capped, no third attempt.
- Never swap, never "one more attempt" past the guard — a swap here is a
  reboot, not a slow run.

## 6. Harness contract — the script must be sound

1. **Never clobber a log:** `logs\kq\<arm>-<probe>-<UTCstamp>-a<attempt>.log`
   (own LogRoot — the inherited lib default `logs\kb` must not be reused),
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
  Scope: wall-time rows only. Greedy accuracy against planted ground truth
  does not drift with machine state; accuracy rows run once per arm under
  the §3 positive-control gate instead of per-cell bracketing.
- Within-boot ratios for quality; ≥4 distinct variants per boot at long
  context (variants are requests — several fit one arm-boot); ≥10%
  machine-state effect vs <10% = unresolvable, say so.
- Prompt lengths are read from the server log, never from the file's
  nominal size: `actual ≈ 0.2383×text_tokens + 294.4×n_msgs + 9110`
  (inputs in tokens and message count respectively — the formula has been
  misapplied with character counts before); reject the whole batch if any
  variant exceeds `cache_size − 256`.
- 250k has an unidentified fast/slow machine state — match state within a
  boot, never across boots.
- Config hygiene: `.bak-<date>` before edits, md5-invariant checked only when
  idle, per-arm pristine restore snapshots (`config.yml.kb-<arm>` = restore
  source, not applied config), restarts announced, `start_tuned.ps1` always.

## 8. Run order and output

0. Harness + selftest green (§6.6). Nothing else starts.
1. Pre-flight incl. external-layer feasibility (§3 stage 0 + 0.4).
2. Descent ladder per arm (§3 stage 0.5) → the max-context column.
3. Endpoints `fp16` + `2,2` → positive-control gate → Gate 1.
4. Targeted bisection arms per Gate 1 (downward rule binding).
5. Optional tiers, in cut order: P5, then external layer (§4b) for arms that
   reach a verdict.
6. Report → `wiki/reports/` + CSVs (`_kb_mkcsv.py` convention): frontier
   table with VRAM-at-peak and speed per arm per length (the recommendation
   moves the operator off 2,2, which costs decode speed as well as memory —
   the table must show both prices), per-arm quality-vs-depth curves,
   per-task rows in the §0 shape, and the reference band each row was scored
   against.

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

Not started. Executor: begin at step 0; everything above is either
measured (obs/reports cited) or explicitly marked unmeasured. Fix list
from four external reviews adjudicated 2026-10-07 — no known open item
above is unaddressed; challenge any line that fails on contact rather than
routing around it.
