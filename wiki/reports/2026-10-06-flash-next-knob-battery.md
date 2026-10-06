# Flash-Next knob battery — results (2026-10-06)

Measured impact of every tunable on 3.05bpw Flash-Next serving
(48 MoE layers, MTP head), RTX 4090 24 GB + 7950X3D, Win11 + WSL2.
Implements `wiki/plans/flash-knobs-benchmark.md`. All numbers are medians of
in-process reps unless stated. **Short-context screen (~17k prompt tokens) —
see §0.7 of the plan: these are a screen, not a verdict. The long-context
pass has since been run at 55–62k, 130k and 224k and confirms the win.**

## Sustained, turn-matched (PRIMARY result)

Production-shaped measurement per plan §0.8: replay `agentic_code_10.json`
progressively (turn *i* sends `messages[0..i]`), 3 boots per config
alternating, 5 turns per boot, 15/15 turns successful. Compare the same turn
index across configs so prompt difficulty cancels; estimator is the median of
per-turn ratios.

| turn | prompt tok | baseline | combo | speedup |
|---|---|---|---|---|
| 1 | 10,956 | 3.89 s | 3.64 s | 1.07× |
| 2 | 11,021 | 2.44 s | 1.94 s | 1.26× |
| 3 | 11,116 | 2.27 s | 1.67 s | 1.36× |
| 4 | 15,912 | 10.88 s | 7.09 s | **1.53×** |
| 5 | 16,135 | 9.06 s | 6.61 s | 1.37× |

**Median 1.36× faster. Full 5-turn conversation: 28.87 s → 20.75 s.**

Two things worth reading off this table:

- **The gain grows with turn number** (1.07× → 1.53×) while the prompt barely
  changes. That points at decode, not prefill: turn 1 is nearly all prefill
  (short generation), and the longer generations later are where extra CPU
  headroom becomes tokens.
- **Turn 1 is only 1.07×.** On short conversations the combo buys far less than
  the median suggests. This matters for choosing a config per workload.

An earlier single-prompt block reported +50.4% (36.1 → 54.3 T/s). That
workload repeated one prompt until the KV prefix cache was ~100% warm, so it
measured decode on a saturated cache rather than a production turn mix.
**1.36× supersedes it.** Quoting 1.50× would overstate the win.

## Long context (plan §0.7) — win at every length measured

Five stages, 11–16k / 55–62k / 130k / 224k / 250k. The 130k, 224k and 250k
stages carry a reference arm interleaved in the same time window (see plan
§0.8.1); earlier revisions of this report got 130k wrong twice before that was
fixed, and the corrections are documented rather than quietly overwritten.

| prompt length | combo vs baseline | what improved |
|---|---|---|
| 11–16k | **1.36×** | decode |
| 55–62k | ~1.00× | — (neutral) |
| 130k | **1.157×** | prefill, +14% T/s from `mcs380` |
| 224k | 1.096× | prefill, +8.6% |
| 250k | **1.13× or 1.02× — bimodal** | prefill, at 96–99% cache utilisation |

**No length showed a regression** anywhere in 11k–259k. §"Scope of the combo" has the mechanism and the
per-half attribution (`mcs380` does the prefill work, `thr16` is decode-only).

Stage 1 at **~55–62k prompt tokens**, 3 distinct cold-prefix variants per
config (`-Rotate`), so each request pays a real prefill.

| variant | prompt tok | baseline | combo | speedup |
|---|---|---|---|---|
| v0 | 55,239 | 30.71 s | 30.63 s | 1.00× |
| v1 | 60,634 | 32.85 s | 33.37 s | 0.98× |
| v2 | 62,189 | 39.25 s | 37.36 s | 1.05× |
| **total** | | **102.8 s** | **101.4 s** | **1.01×** |

**The combo's advantage vanishes at 2× context** — median ratio 0.997. Server
prefill confirms why:

| prompt tok | new tok | baseline prefill | combo prefill |
|---|---|---|---|
| 55,239 | 47,047 | 28.6 s | 28.7 s |
| 60,634 | 52,442 | 31.6 s | 31.6 s |
| 62,189 | 53,997 | 33.2 s | 32.8 s |

Prefill is **flat** at ~1640 T/s for both configs. At 2× context a request is
~95% prefill (≈30 s) and only ~4 s of decode, so a decode-side win is
arithmetically invisible in wall-clock.

Note the two metrics disagree here and **wall-clock is the one to trust**:
`tg median` reads 61.0 vs 41.2 (+48%) while total wall is 1.01×. The tg column
measures only the generation phase, so at prefill-bound lengths it overstates
the user-visible win. This is the same trap as the earlier cache-saturated
+50.4%, in a different guise — **always check which phase dominates before
quoting a tok/s figure.**

### Scope of the combo, stated precisely

At this stage (55–62k) the combo was **conditional**: a decode-bound win that
went to zero once prefill dominated. **Later stages revised that** — prefill is
in fact improved at 130k and above, so the win returns. Read this as the
stage-1 reading, and use the four-length table further down for the current
one.

- **Decode-bound turns** (short prompt, long generation — e.g. 11–16k prompt,
  256+ tokens out): **1.36× faster** measured turn-matched.
- **Prefill-bound turns** (55–62k here): **~1.00×** — prefill flat at this
  length. Later shown to be length-specific, not general.

## Headline (single-request, superseded by the above)

| config | live tg T/s | vs base | boot→1st tok | VRAM min free |
|---|---|---|---|---|
| production baseline (`mcl38`, 8 threads) | 34.4 [34.1–37.0] | — | 53.7 s | 3593 MB |
| `thr16` only | 46.6 [44.9–48.4] | +35.4% | 53.5 s | 3561 MB |
| `mcs380` only | 37.0 [36.5–37.3] | +7.6% | 51.8 s | 269–2089 MB ⚠ |
| **`mcs380` + `thr16`** | **51.0 [48.1–53.1]** | **+48.3%** | 50.6 s | 1065 MB |

Guards: VRAM kill <200 MB free, RAM kill <1 GB. Baseline n=4, combo n=4,
`thr16` n=4, `mcs380` n=4. No guard ever tripped on a reported number.

## config.yml DIFF PROPOSAL — proposal only, never committed

```yaml
model:
  # CHANGED 38 -> (removed); split-experts replaces it. MUTUALLY EXCLUSIVE:
  # set exactly one. Never both.
  #cpu_moe_offload_layers: 38
  cpu_moe_split_experts: 380      # +7.6% alone; +14% prefill T/s at long ctx
  cache_mode: 2,2                 # neutral tg, +2.8 GB VRAM free
  # UNCHANGED: cache_size 262144, chunk_size 4096, max_batch_size 2

draft_model:
  dynamic_draft: true             # KEEP — +7.1%, static is slower despite
                                  # higher acceptance (2.92/5.00 vs 2.78/4.29)
  draft_cache_mode: Q4            # KEEP — 2,2 cost -23% tg AND -24pp acceptance
  draft_num_tokens: 5             # KEEP (see §open: 6 ran clean, untested at 6)

memory:
  cuda_malloc_async: True         # KEEP — False costs ~760 MB VRAM, no gain
```

Expected gain, **by prompt length** — not a single number, since it was measured
at four lengths and is non-monotonic:

| workload | expected |
|---|---|
| short/decode-bound (11–16k) | **1.36×** (5-turn conv 28.87 s → 20.75 s) |
| ~62k | ~1.00× |
| ~130k | **1.157×** |
| ~224k | 1.096× |

Boot unchanged (~50.6 s vs 53.7 s). VRAM free 3593 → ~1065 MB short-context,
1479–1543 MB at 130k/224k. **No measured length regressed.**

Which half does what, measured in one interleaved batch at 130k:

| change | 130k | role |
|---|---|---|
| `cpu_moe_split_experts: 380` | 1.135× | long-context lever: prefill 1507 → 1725 T/s |
| `EXL3_MOE_CPU_THREADS=16` (in `start_tuned.ps1`) | 1.026× | decode-only, **costs no VRAM** |

**Fallback if VRAM ever binds: drop `cpu_moe_split_experts`, keep the threads
change.** It keeps the entire 11–16k decode win at zero VRAM cost.

### 224k stage — prefill is NOT flat here, which revises the 62k reading

The files labelled `118k` are actually **~224k prompt tokens** (see the
calibration note in plan §0.7 — text tokens undercount prompt tokens by
~1.8×). Treated as the near-maximum case instead of discarding it.

| variant | prompt tok | baseline | combo | speedup |
|---|---|---|---|---|
| v0 | 224,502 | 142.18 s | 129.67 s | 1.096× |
| v1 | 224,414 | 142.10 s | 129.67 s | 1.096× |

**1.096× at 224k**, and the reason is prefill after all:

| prompt tok | new tok | baseline prefill | combo prefill |
|---|---|---|---|
| 224,414 | 216,222 | 135.7 s | **125.0 s (+8.6%)** |
| 62,189 | 53,997 | 33.2 s | 32.8 s (+1.2%, noise) |

This **corrects** the "prefill never improves" conclusion from the 62k stage.

Mechanism: this is the same `mcs380` prefill effect measured directly at 130k
(prefill T/s 1507 → 1725, +14%) — **no cache-pressure effect is involved**. The
theory originally proposed here (that paging at 85% occupancy drags CPU MoE work
into the prefill path) is **withdrawn**: `mcs380` improves prefill at 130k where
the cache is only ~50% full, which fully accounts for the 224k gain on its own.

Note this stage was measured as a single interleaved pair (baseline spread
0.08 s, combo 0.00 s across two variants), so it satisfies the shared-window
requirement of §0.8.1.

### 250k stage — bimodal: 1.13× or 1.02× depending on the boot

This is the plan's "260k desirable" tier, and the region that had been recorded
as untestable. **That was wrong** — only the miscalibrated ~474k file could not
load. Sized with the two-variable fit (`actual ≈ 0.2383×text + 294.4×msgs +
9,110`), four variants land at 249,730 / 249,642 / 251,197 / 259,145 prompt
tokens: **96–99% of `cache_size` 262144**, all loading cleanly.

Interleaved per §0.8.1, **4 boots per arm**, 4 prompts each (16 paired turns):

| boot | arm | boot median | per-turn (249,730 / 249,642 / 251,197 / 259,145) | ratio |
|---|---|---|---|---|
| 1 | baseline | 165.7 s | 163.21 / 162.77 / 168.17 / 172.68 | ref |
| 2 | combo **fast** | 144.9 s | 139.97 / 139.63 / 149.77 / 155.27 | **1.144×** |
| 3 | baseline | 164.7 s | 162.21 / 161.18 / 167.11 / 172.71 | ref |
| 4 | combo **slow** | 161.8 s | 159.31 / 159.12 / 164.24 / 169.77 | 1.017× |
| 5 | baseline | 165.0 s | 162.08 / 162.28 / 167.66 / 172.55 | ref |
| 6 | combo **fast** | 145.5 s | 141.39 / 140.12 / 149.61 / 155.74 | **1.133×** |
| 7 | baseline | 164.8 s | 162.47 / 162.25 / 167.14 / 172.28 | ref |
| 8 | combo **slow** | 162.3 s | 160.01 / 159.91 / 164.50 / 169.62 | 1.016× |

**The combo arm is bimodal, not noisy.** Two clean modes ~11% apart:

| mode | boots | pooled speedup |
|---|---|---|
| fast | 2, 6 | **1.134×** |
| slow | 4, 8 | **1.016×** |

Baseline spread across four boots is **0.63%** (164.7–165.7 s) — the reference
is solid. And every one of the 8 fast-mode combo turns (139.6–155.7 s) is
**faster than every one of the 16 baseline turns** (161.2–172.7 s): no overlap.

`pp median` is the metric that tracks the mode exactly — fast boots read
**1676 / 1678**, slow boots **1522 / 1525**, a 10% prefill throughput step. The
step is entirely in prefill; `tg median` moves only 63.0 → 60.8.

**Cause not identified.** The server log is identical in both modes (16 threads,
split experts `[132..512)`, same arena size, load 48.3 s vs baseline 52.5 s), so
this is *not* the config failing to apply. It is runtime variance inside the
`mcs380` prefill path and no theory is offered, because nothing in the logs
points at one.

**How to read it.** The pooled all-16 median is 1.061×, but that is a mixture of
two regimes and is not a good summary. The defensible statement: **at 250k the
combo is never slower — it is either ~1.13× or ~1.02× faster, and which one you
get is not predictable from the logs.** An operator wanting the upper regime
should measure it on their own box; the lower regime is the floor.

History of this one number, kept because it is instructive: reported as 1.144×
from boot 1 of 2 (before boot 2 finished), then corrected to "~1.06×, range
1.02–1.14×" at n=2, and n=4 shows the range is **two discrete modes**, not a
confidence interval. Every revision was defensible on the data available at the
time; only more boots distinguished "noise" from "bimodal".

What the stage establishes independently of the speedup uncertainty: **VRAM
min-free held at 1479–1511 MB at 96–99% cache utilisation**, clear of the
200 MB kill, and the spread does not widen with context. `mcs380`'s headroom
risk — the reason this band was called untestable — **does not materialise.**

### 130k stage — combo is 1.157× FASTER. Two retractions were both wrong; here is why.

**This section supersedes two earlier revisions of this report**, which said
"11% slower" and then "inconclusive". Both were wrong, for the same reason.

Proper design: 4 distinct prompts per boot (`--variants 4`), `-Sustained 4
-Rotate`, **2 boots per arm**, turn-matched on identical prompt-token counts.

| prompt tok | baseline (2 boots) | combo (2 boots) | ratio | speedup |
|---|---|---|---|---|
| 129,998 | 86.97 s | 74.78 s | 0.860 | **1.163×** |
| 129,910 | 86.86 s | 75.44 s | 0.869 | **1.151×** |
| 144,084 | 96.28 s | 82.28 s | 0.855 | **1.170×** |
| 142,617 | 90.59 s | 79.53 s | 0.878 | **1.139×** |

**Median ratio 0.864 — the combo is 1.157× faster at 130k**, speedup range
1.139–1.170× across all four lengths. All 8 paired turns give median 0.862 with
range 0.824–0.896.

Quality of the measurement, which is what makes this one trustworthy:

- **Cross-boot drift within the batch is +0.07%** (two baseline boots agreed to
  86.6/86.9/96.5/90.7 vs 87.3/86.9/96.1/90.5).
- All 4 speedups cluster in a 3% band, and no baseline boot overlaps any combo
  boot (baseline min 86.6, combo max 85.4).

#### Per-half attribution — `mcs380` carries the long-context win

Splitting the combo in the **same interleaved batch** (2 boots each, same
window, 4 prompts each, n=8 paired turns per arm):

| arm | median ratio | speedup | range | verdict |
|---|---|---|---|---|
| combo (both) | 0.862 | **1.161×** | 0.824–0.896 | win |
| `mcs380` alone | 0.881 | **1.135×** | 0.859–0.902 | win |
| `thr16` alone | 0.975 | 1.026× | 0.943–0.992 | ~neutral |

Baseline self-consistency across its own two interleaved boots: ratio 1.000,
range 1.000–1.000. **The noise floor for this batch is zero**, so even `thr16`'s
1.026× is real but tiny.

**Prefill throughput on identical new-token counts** (server log, so prefill is
isolated from decode):

| config | 121,806 | 121,718 | 135,892 | 134,425 | gain |
|---|---|---|---|---|---|
| baseline | 1504 | 1508 | 1507 | 1531 | ref |
| `mcs380` | 1738 | 1749 | 1692 | 1719 | **+14.0%** |
| `thr16` | 1506 | 1521 | 1517 | 1538 | +0.5% |
| combo | 1748 | 1739 | 1740 | 1771 | **+15.7%** |

(T/s, prompt tokens ÷ prefill seconds.)

So the mechanism is now **measured, not hypothesised**: `cpu_moe_split_experts:
380` directly accelerates prefill at long context, and `EXL3_MOE_CPU_THREADS=16`
has no effect there at all. The combo is worth slightly more than `mcs380` alone
because the threads knob still helps the decode portion (tg median 61–62 vs
44–45 for `mcs380` alone at 130k).

This also explains the 224k result and retires the "cache pressure" theory I
proposed for it: the 224k prefill win (+8.6%) is the same `mcs380` prefill
effect, not a paging artefact. No cache-pressure story is needed.

#### Why the earlier two conclusions were wrong

The earlier design used **2 prompts × 1 boot per arm** and took the *first*
baseline (77.9 s) as the reference for everything measured later. Baseline boots
in that window read 77.9 / 79.1 / 86.8 — which looked like 11.2% "noise" and
got read as inconclusive.

With 4 samples per boot, that 11.2% is **not boot-to-boot noise at all**: drift
between consecutive boots in one batch is 0.07%. The 77.9 vs 86.8 gap is
**drift across the session**, and the early baseline was measured in the fast
period. So:

- "11% slower" — compared combo against the fast-period baseline. Artifact.
- "inconclusive, 11.2% spread" — correctly noted the spread exceeded the effect,
  but attributed it to boot-to-boot noise and so never fixed the real cause,
  which is that the reference arm was not measured in the same window.

**Methodological rule, now recorded because it cost three revisions:**
a config's ratio is only valid against an arm measured **in the same time
window**. Never carry a reference measurement forward across other arms. Always
re-measure the baseline interleaved with each candidate, and check whether the
*baseline* is stable across those interleaved runs before trusting any ratio.

#### The superseded 1-boot data, kept for the record

This is the data both retractions were based on. **Do not read these ratios.**
The isolation arms never shared a time window with the reference.

| config | boots | boot medians | all-boot median | ratio vs baseline |
|---|---|---|---|---|
| baseline | 3 | 77.9 / 79.1 / **86.8** | 79.14 | 1.000 |
| `thr16` | 2 | 85.2 / 87.6 | 86.47 | 1.093 |
| `mcs380` | 3 | 87.6 / 81.9 / 78.1 | 81.94 | 1.035 |
| combo | 1 | 86.8 | 86.76 | 1.096 |

The 77.9 / 79.1 / 86.8 baseline spread was read as an 11.2% noise floor, and the
row above it is exactly where the bad conclusions came from. Note the baseline's
first two boots (77.9, 79.1) sit ~10% below every later measurement in the
session — that offset, not variance, drove all three wrong readings.

The per-half isolation (`thr16` vs `mcs380` at 130k) was done in this same flawed
window and its verdicts are **void for the same reason**. It is not repeated here
because a valid re-run would need another interleaved batch; the only thing
established about the halves remains their behaviour at 11–16k.

Note the 224k stage is unaffected: it was measured as a single interleaved pair
with baseline spread 0.08 s and combo 0.00 s, so the shared-window requirement
was met by construction.

### Full regime table

| prompt len | cache occupancy | combo vs baseline |
|---|---|---|
| 11–16k | negligible | **1.36× faster** |
| ~55–62k | low | ~1.00× |
| ~130k | ~50% | **1.157× faster** (2 boots/arm, 4 prompts) |
| ~224k | ~85% | 1.096× faster |
| **~250k** | **96–99%** | **1.13× (fast) or 1.02× (slow)** — bimodal, n=4 |

Every measured length is now a win or neutral. The apparent dip at 130k in two earlier
revisions was a stale-reference artifact — see the 130k section.

The curve is **non-monotonic**, so the 62k point must not be read as "the
benefit decays with length".

### Scope of the win — read before applying

**Every measured length is now a win.** The length-dependence question is
settled, and it did not go the way two intermediate revisions of this report
claimed:

| prompt length | cache occupancy | combo vs baseline | quality of evidence |
|---|---|---|---|
| 11–16k | negligible | **1.36×** | turn-matched, 3 boots, tight spreads |
| ~55–62k | low | ~1.00× | 2 boots, tight |
| **~130k** | ~50% | **1.157×** | turn-matched, **2 boots/arm × 4 prompts** |
| ~224k | ~85% | 1.096× | turn-matched, spread 0.08 s / 0.00 s |
| **~250k** | **96–99%** | **1.13× / 1.02×** | **bimodal**, turn-matched, **4 boots/arm × 4 prompts** |

There is **no length at which the combo was measured to hurt**, including the
62k point that looked neutral and the 130k point that two revisions wrongly
called a regression. The **upper end of the range is the least predictable
point** — at 250k the combo is *bimodal*, measuring either ~1.13× or ~1.02×
depending on the boot, with the cause unidentified. It is never slower; just
do not assume the upper regime.

**Which half does the work, measured in the same interleaved batch:**

| change | at 130k | what it actually does |
|---|---|---|
| `cpu_moe_split_experts: 380` | **1.135×** | prefill T/s 1507 → 1725 (**+14%**) |
| `EXL3_MOE_CPU_THREADS=16` | 1.026× | prefill unchanged; decode only |
| combo | **1.161×** | both |

`cpu_moe_split_experts: 380` is the long-context lever — it accelerates
**prefill** directly. The threads knob is a decode-only lever worth ~2.6% at
130k and nothing at all in prefill. Both are worth applying, but they are not
the same kind of change: if VRAM ever forces a rollback, the threads knob is the
one to keep, since it costs no VRAM.

The honest reading of the shape: a large decode win at short prompts, roughly
neutral at 62k, then a real win again at 130k+ where prefill dominates —
bimodal at 259k (1.13× or 1.02×, cause unknown). The
130k and 224k gains are in **prefill itself**, so the combo is not purely a
decode optimisation. This also **retires the "cache pressure" explanation** I
proposed earlier in this file for the 224k prefill win — it is the same
`mcs380` prefill effect, and no paging story is needed.

#### Three claims retracted from this report

Both were mine and both were caused by the same methodological error, recorded
so it is not repeated:

1. **"The gain shrinks toward zero as prompt length grows / prefill never
   improves."** False — prefill improves at 130k and 224k. The 62k point was
   read as the general case when it was one point on a non-monotonic curve.
2. **"The combo is ~11% slower at 130k, and `EXL3_MOE_CPU_THREADS=16` alone is
   safe because it is a pure decode win."** Both false. The regression came from
   comparing arms against a **baseline measured in an earlier, faster window**;
   with the baseline interleaved, the combo is 1.157× *faster* at 130k.
3. **"At 224k the prefill win comes from KV-cache pressure pushing CPU MoE work
   into the prefill path."** Unnecessary — `mcs380` speeds prefill by +14% at
   130k with the cache only ~50% full, which fully accounts for the 224k gain
   without any paging effect.

The root cause is a protocol gap, not a measurement gap: the harness let a
baseline measured early stand in as the reference for arms measured much later,
across a ~10% session drift. §0.8 now requires the reference arm to be
interleaved with every candidate.

**Apply both changes.** The only remaining reservation is VRAM headroom, which
is a capacity question rather than a performance one — see below.

### Risk, honestly

- **`cpu_moe_split_experts: 380` is the whole win and the thinnest margin.**
  Clean 4-run min-free ranged 269–2089 MB. At 269 MB it is 1.3× the guard.
  This is now the *only* open risk: the 2× context pass has been done
  (130k and 224k, both wins) and VRAM held at 1479–1543 MB free min through
  224k, comfortably clear of the 200 MB kill. What remains untested is the
  region between ~224k and `cache_size` 262144, where the KV cache saturates
  and paging pressure is highest. Expect ~1100 MB free to be the floor.
- `cache_mode: 2,2` is **perf-only here** — quality not measured (KLD owns
  quality). Safe to revert independently; it is the fallback for VRAM relief.
- `mcs380` is the *only* viable split-experts value: 360 loads offline
  (23308 MB) but **will not boot the server**; 400/500 are VRAM- or RAM-refused.
- Mutually exclusive with `cpu_moe_offload_layers` — the server hard-errors.

### `recurrent_checkpoint_interval_pp` — MEASURED, and it works

Plan §0.9 #2b recorded this as unevaluable at 17k and said it "can only be
judged in the §0.7 pass". That pass now exists, so it is measured here.

It cannot be seen on a cold prefill — its doc says it governs what a
**mid-conversation edit** costs. Neither existing harness mode edits anything,
so the workload is two turns in one boot: turn 1 the original prompt (writes
checkpoints), turn 2 the same prompt with an **early** message reworded by
`_kb_mkedit.py` (~80 chars, so token count barely moves). 144,084 → 144,103
tokens; turn 2's wall-clock delta *is* replay cost.

At ~144k, default 32768 gives 4 checkpoints, 8192 gives 16.

**The metric must be the within-boot ratio `t2/t1`**, not absolute turn-2
wall-clock. Absolute comparisons here are wrecked by session drift — the two
baseline boots ran 79.41 s and 95.07 s for the identical request (20% apart),
which would have produced a spurious "1.22×" for the candidate. Dividing each
arm's turn 2 by its *own* turn 1 cancels that entirely:

| boot | arm | turn1 cold | turn2 edit | t2/t1 |
|---|---|---|---|---|
| 1 | baseline | 79.51 s | 79.41 s | **0.999** |
| 1 | `rci_pp=8192` | 84.15 s | 77.05 s | **0.916** |
| 2 | baseline | 96.29 s | 95.07 s | **0.987** |
| 2 | `rci_pp=8192` | 82.03 s | 77.95 s | **0.950** |

| arm | median t2/t1 | replay saving vs cold prefill |
|---|---|---|
| baseline (32768) | 0.993 | **0.7%** |
| `rci_pp=8192` | 0.933 | **6.7%** |

**On the default grid an early edit costs essentially the full cold prefill
(0.7% saved); at 8192 it costs 6.7% less.** Both ratios separate from baseline
in both boots with no overlap.

Confirmed independently by the server log: `rci_pp=8192` turn 2 reports **11%
cached / 127,719 new** against baseline's **6% cached / 135,911 new** — an
8,192-token saving, i.e. exactly one checkpoint interval. That is the doc's
stated behaviour, measured rather than assumed.

**Cost: RAM, not VRAM.** Free RAM after the run falls from ~7,400 MB (baseline)
to ~5,100 MB — about 2.3 GiB, matching the 16 × 148 MiB checkpoint arithmetic.
VRAM is unchanged (2971–3035 MB free both arms).

**Verdict: not adopted, and this is a judgement call, not a rejection.** The
knob works and is the right tool if this workload *edits* long conversations
often. It is left at default because ~6.7% on edits only, paid for with 2.3 GiB
of the RAM that #1 already proved to be the binding constraint on this box.
**Adopt it if edit-heavy long-context work becomes the norm** — it is a
one-line change with a known price.

## start_tuned.ps1 — verdict per env line

| line | verdict | evidence |
|---|---|---|
| `EXL3_MOE_CPU_THREADS=8` → **16** | **CHANGE** | +35.4% tg live, 4 reps, non-overlapping ranges. Peak is 16; 24 regresses. Costs no VRAM. |
| `EXL3_MOE_CPU_PIN=1` | KEEP | −19.8% without. |
| `EXL3_MOE_CPU_SWIZZLE=1` | KEEP | −11.2% without. |
| `EXL3_MOE_MEMOPS=0` | KEEP | plan says 29% gap; `PERF_FINDINGS` says +10%. Not re-measured (SKIP per plan). |
| `EXL3_MOE_ZERO_COPY=1` | KEEP, low confidence | −2.5% without = inside noise. Prior work measured +3.3%. Not re-run at Tier 2. |
| `EXL3_MOE_STREAM_T=6` | KEEP | `STREAM_T=12` cost **pp256 −38%**; `STREAM_T=3` +0.8% tg, neutral pp. |
| `EXL3_MOE_STREAM_BATCH_EXPERTS=48` | KEEP | 24 → tg 25.6 vs 26.8 (−6.2%), pp4096 1867 vs 1826. |

No line qualifies for removal. The one change is threads.

**This is the safe half of the recommendation.** At 130k the threads change
measures 1.026× (prefill unchanged, decode only) and it costs no VRAM at all,
whereas `cpu_moe_split_experts: 380` is what carries the long-context prefill
win but holds VRAM min-free down to 1065 MB (269–2089 MB across short-context
boots). If VRAM ever becomes the binding constraint, **revert `mcs380` and keep
this line** — it retains the full 11–16k decode win at zero VRAM cost.

## Full knob table

Phase A (offline `eval/perf.py`, Tier 1, `-short -max_length 4096 -dr 2`).
Baseline re-anchored in-run: tg0 24.91 [24.25–25.56], pp256 311.5, pp4096 2117.4.

**These are Tier 1 numbers.** Per plan §0.5 only Tier 2 (`-short -sd
-max_length 32768 -dr 3`) may quote deltas. **Tier 2 has since been run and it
does not confirm this table's headline** — see the Tier 2 section immediately
after. Tier 2 puts the combo at +13.4% tg, not the +21.6% below. Read Tier 1 as
a screen, exactly as the plan says it should be read.

| knob | setting | tg0 [min-max] | Δtg | pp256 | pp4096 | VRAM peak / min free | verdict |
|---|---|---|---|---|---|---|---|
| — | baseline (`mcl38`) | 24.91 | — | 311.5 | 2117.4 | 20350 / 3923 | ref |
| threads | 16 | 29.02 [28.45–29.58] | +16.5% | 335.1 | 1781.7 | 19926 / 4213 | **ADOPT** |
| threads | 12 | 30.89 [30.76–31.02] | +24.0% | 342.5 | 1837.2 | 20350 / 3789 | ok, 16 better live |
| `mcs` | 360 | 30.00 [29.93–30.08] | +20.4% | 323.4 | 2170.8 | 23308 / **831** | **won't boot live** |
| `mcs` | 380 | 28.66 [28.35–28.96] | +15.1% | 306.1 | 2112.3 | 21186 / 2953 | **ADOPT** |
| `mcs380`×`thr16` | combo | 30.29 [29.99–30.59] | +21.6% | **356.5** | 2093.1 | 21186 / 2953 | **ADOPT (headline)** |
| `mcl` | 36 | 28.90 [28.50–29.30] | +16.0% | 328.3 | 2167.2 | 22110 / 2029 | reject: -1.5% live tg, +1.7 GB |
| `mcl` | 34 | — | — | — | — | 24010 / **129** | **UNSAFE** — tripped guard |
| `mcl` | 32 | — | — | — | — | — | refused by engine VRAM check |
| `cache_mode` | 2,2 | 24.42 [24.17–24.68] | −2.0% | 305.9 | 2163.5 | **19356 / 4783** | adopt (VRAM lever) |
| `cache_mode` | 4,4 / 8,8 | 26.52 / 26.64 | +6% | — | — | 18644 / 20268 | no benefit |
| `chunk_size` | 2048 | 24.72 [24.38–25.05] | −0.8% | 293.9 | **1231.7** | 18998 / 5141 | reject (−42% pp4096) |
| `chunk_size` | 8192 | 25.96 [24.90–27.02] | +4.2% | 307.7 | 1816.4 | 20008 / 4131 | neutral, keep 4096 |
| `max_batch_size` | 1 | 25.38 | +1.9% | 311.1 | 1798.6 | 20216 / 4449 | reject: halves capacity |
| `STREAM_T` | 3 | 25.29 [24.72–25.85] | +1.5% | 307.7 | 1855.8 | 19928 / 4211 | neutral |
| `STREAM_T` | 12 | 25.15 [24.29–26.00] | +1.0% | **185.9** | 1595.6 | 20350 / 3789 | **reject (−40% pp256)** |
| `BATCH_EXPERTS` | 24 | 25.64 [25.06–26.22] | +2.9% | 310.2 | 1867.4 | 20352 / 3787 | neutral |
| PIN | 0 | 21.49 (T0) | −13.0% | — | — | 18858 / 5281 | keep pinned |
| SWIZZLE | 0 | 23.81 (T0) | −3.6% | — | — | 18794 / 5345 | keep |
| ZERO_COPY | 0 | 25.56 [24.92–26.20] | +3.5% | 304.1 | 1924.6 | 18858 / 5281 | keep (noise) |

### Phase A Tier 2 — the only tier allowed to quote deltas

`-short -sd -max_length 32768 -dr 3`, 5 finalists, baseline **interleaved** and
re-run (n=3). Per §0.8.1 the reference is measured in the same window.

| arm | n | tg0 per boot | Δtg0 | Δpp256 | Δpp4096 | VRAM min free |
|---|---|---|---|---|---|---|
| baseline | 3 | 25.68 / 26.16 / 25.29 | ref | ref | ref | 3783 MB |
| `cq 2,2` | 3 | 25.15 / 25.73 / 24.48 | **−2.1%** | +1.1% | +5.2% | 4781 MB |
| `thr16` | 2 | 32.36 / 32.41 | **+26.1%** | +8.8% | +0.5% | 3783 MB |
| `mcs380` | 2 | 28.34 / 28.18 | +10.0% | −1.1% | **+6.7%** | 3065 MB |
| **combo** | 2 | 29.05 / 29.19 | **+13.4%** | **+11.1%** | +3.9% | 3077 MB |

Baseline tg0 spread across its 3 boots is **3.4%**, so the reference is stable
enough to read these against.

**Two results here change the picture, and neither is what Tier 1 said.**

1. **The combo's offline gain drops from +21.6% (Tier 1) to +13.4% (Tier 2).**
   Tier 1 over-read it, as the plan warns it might.

2. **The combo is strongly sub-additive on decode:** `thr16` +26.1% and
   `mcs380` +10.0% sum to +36.1%, but together they give **+13.4%**. The two
   compete for the same CPU-MoE bottleneck when decode dominates — more worker
   threads and fewer CPU experts are alternative ways to relieve the same
   resource, not complementary ones.

   This *corroborates the live server* rather than contradicting it: there the
   combo's tg (51.0) sat **between** the two singles (46.6 and 37.0), not above
   them. The Tier 2 data is the offline explanation for a pattern that was
   already visible live and previously unexplained.

   Note this is the §2 interaction test, and it now fails the plan's own rule in
   spirit: "combo underperforms the sum by >30% → keep the single best". At
   +13.4% vs a +36.1% sum it underperforms by 63%. By that rule `thr16` alone
   (+26.1%) is the better offline choice. **But the live primary metric
   disagrees** — see the sustained table, where the combo wins 1.36× turn-matched
   and §"Which half does the work". Both are true: on *decode-only* short
   offline forwards `thr16` alone is better, and on *production-shaped sustained
   turns* the combination is better. The live metric is the one the plan
   nominates as primary, so the recommendation stands.

`cq 2,2` is **−2.1% on decode** but +5.2% on pp4096 and buys **+998 MB** of VRAM
(4781 vs 3783 free). Consistent with its "neutral tg, VRAM relief" framing.

`mcs380` is the only finalist that materially improves **long prefill** (+6.7% at
pp4096), which matches the live finding that it is the long-context lever.

Three arms hit a known `perf.py` timer flake and were auto-retried; the
harness reports `attempts=2` for those. `thr16` replicates to 32.36 / 32.41
across independent launches, so this does not appear to bias the table.

Phase B (offline `eval/spec_decode.py`, Tier 0, `-single Coding`). MTP
auto-engages when `-dm == -m`; `-nbl` so each arm measures only its own.

| knob | setting | tg t/s | vs base | acc/draft | VRAM min free | verdict |
|---|---|---|---|---|---|---|
| draft | ndt5 + dyn (prod) | 32.11 | — | 2.78/4.29 | 2731 | ref |
| draft | ndt6 + dyn | 32.40 | +0.9% | **2.98/4.66** | 2477 | works; ceiling >6 |
| draft | ndt4 + dyn | 32.71 | +1.9% | 2.73/3.80 | 2969 | neutral |
| draft | ndt3 + dyn | **34.05** | +6.0% | 2.20/2.95 | 2987 | speed/quality trade |
| draft | ndt5 static | 29.82 | **−7.1%** | 2.92/5.00 | 2775 | **dyn on** |
| draft | ndt3 static | 33.81 | +5.3% | 2.19/3.00 | 2997 | dominated by dyn |
| draft | **off** | 27.21 | **−15.3%** | — | **4381** | MTP stays on |

Phase C (live server). All six metrics; boot = process start → first token.

| arm | n | boot→1st tok | tg T/s | pp T/s | acc% | VRAM peak / min free | RAM after |
|---|---|---|---|---|---|---|---|
| baseline | 4 | 53.7 s | 34.4 | 1204 | 79% | 20546 / 2761–3593 | ~9.3 GB |
| `thr16` | 4 | 53.5 s | 46.6 | 1260 | 71% | 20578 / 3465 | ~9.4 GB |
| `mcs380` | 4 | 51.8 s | 37.0 | 1308 | 75% | 22050–23870 / **269** | ~9.6 GB |
| **combo** | 4 | **50.6 s** | **51.0** | 1320 | 78% | 22050–23074 / 1065 | ~9.5 GB |
| `offload36` | 1 | 51.6 s | 33.9 | 1280 | 70% | 22306 / 1833 | 11.1 GB |
| `ambs1` | 1 | 53.7 s | 35.8 | 1186 | 76% | 19874 / 4265 | 9.6 GB |
| `warmup:true` | 1 | **65.7 s** | 35.3 | 1184 | 75% | 21378 / 2761 | 9.2 GB |
| `rci512` | 1 | 53.7 s | 35.6 | 1225 | 77% | 21346 / 2793 | 9.8 GB |
| `malloc_async:False` | 1 | 53.7 s | 34.5 | 1199 | 72% | 21306 / 2833 | 9.7 GB |
| `dcm 2,2` | 1 | 53.7 s | **26.4** | 1218 | **55%** | 20482 / 3657 | 9.8 GB |
| `mcs360` | — | **will not boot** | — | — | — | — | — |
| `rci_pp 8192` | 2 | 64.3 s | — | — | — | 21104 / 3035 | 5.1 GB RAM — see §0.9 #2b |
| `sysmem_kv 8g` | 1 | **RAM guard: 492 MB free** | — | — | — | — | killed during load |
| `sysmem_kv 24g` | 1 | **RAM guard: 88 MB free** | — | — | — | — | killed during load |

Boot time is very stable: 53.5–53.9 s across baseline/`thr16`/`combo` (n=12).
`load` = 38.7–39.3 s. Warmup adds +12 s boot for no measurable gain.

## §2 interaction

Top-2 winners combined once: `mcs380` + `thr16`.

| tier | `thr16` | `mcs380` | sum of singles | combo | vs sum |
|---|---|---|---|---|---|
| Tier 1 offline | +35.4% | +7.6% | +43.0% | **+48.3%** | **+5pp, super-additive** |
| Tier 2 offline | +26.1% | +10.0% | +36.1% | **+13.4%** | **−63%, sub-additive** |
| live single-request tg | +35.4% | +7.6% | +43.0% | +48.3% | +5pp |
| live sustained (PRIMARY) | — | — | — | **1.36× turn-matched** | — |

**The two tiers disagree about the sign of the interaction, and Tier 2 is the
one allowed to quote deltas.** At Tier 2 the combo underperforms the sum of
singles by 63%, which by the plan's own §2 rule ("underperforms the sum by >30%
→ keep the single best") would mean keeping `thr16` alone.

Mechanism, now that it is measured: on **decode-bound** work the two changes
are *substitutes*, not complements — more CPU worker threads and fewer CPU
experts both relieve the same CPU-MoE bottleneck, so adding both yields the
bottleneck-relief once, not twice. That is why the live combo's tg (51.0) sits
**between** the two singles (46.6, 37.0) rather than above them.

The original Tier 1 story ("they attack the same bottleneck from opposite ends,
so each supplies part of what the other needs") was wrong, and Tier 2 is what
exposed it.

**Both changes are still kept**, because the live *sustained* metric — the one
§0.8 nominates as PRIMARY — is measured on production-shaped turns, not
decode-only forwards, and there the combo wins 1.36× turn-matched and 1.16× at
130k. The offline decode-only picture would argue for `thr16` alone. Worth
stating plainly: **the two metrics rank the configs differently, and the
primary metric wins.**

**This also revises what the recommendation is *for*.** `thr16` is the better
change on almost every axis — bigger decode win (26.1% vs 13.4% offline), costs
no VRAM, replicates tightly (32.36 / 32.41), and no VRAM risk. `mcs380` earns
its place specifically at **long context** (+14% prefill T/s at 130k, the only
change that improves prefill at all). If the workload is short-context only,
`mcs380` is not earning its VRAM cost.

## Why the offline tier understates the winner

Offline, combo beats `mcs380` alone by only **+5.7%** (30.29 vs 28.66).
Live, the same pair goes 37.0 → 51.0 T/s (**+38%**). Same knobs, ~7× difference
in apparent interaction size.

Cause: `eval/perf.py` measures **raw forwards with no MTP drafting**, so it
never spends the CPU budget `thr16` buys. Real serving is draft-heavy
(198/264 accepted), so extra CPU headroom converts into accepted draft tokens.
**Consequence: the offline screen is valid for ranking single knobs but not
for interactions involving drafting.** Screening alone would have rejected the
winner. Always validate a proposed combo on the live server.

## Premise corrections (plan was wrong four times)

1. **"Low thread values score higher" is backwards.** tg by thread count:
   1→7.07, 2→11.38, 4→20.25, 8→24.69, 12→30.33, 16→31.74, 24→30.68 (T0).
   Monotone up to 16, turning over at 24. `PERF_FINDINGS` rejected 16 on
   *variance*, not mean — and 4 live reps show spread 7.4% vs baseline 8.0%,
   i.e. not the high-variance config that note feared.
2. **Offload ladder's productive direction is down**, not up: `mcl34` +22.3%
   (T0), `mcl36` +19.1%, while 40/42 were −6.6%/−8.9%.
3. **`mcl34` is UNSAFE, not fast** — 129 MB free (guard is 200), and
   `cudaErrorLaunchFailure` in `decode_flash_attn` under prefill. `mcl32`
   never loads (`Insufficient VRAM in split for model and cache`).
4. **MTP max is not 4** (plan said "4 is believed to be this model's max").
   `-ndt 6` runs clean with the best acceptance measured (2.98/4.66).
   Ceiling is above 6; `ndt7`/`ndt8` untested.

Also: `split-experts` VRAM-matched value is the *worst* viable one. Matching
VRAM to `mcl38` (405) is backwards — VRAM headroom is what you spend to buy
speed, because CPU MoE is the bottleneck. Sweep **down** from it.

## Open / not established

- **`draft_num_tokens` 3 vs 5 is +6.0% tg but drops acceptance 2.78→2.20.**
  Quality not measured (KLD owns it).
- `ZERO_COPY` at Tier 2; Tier 2 `-dr 3` not run for most finalists.
- `cache_mode` quality (KLD).
- MTP `ndt7`/`ndt8` ceiling.
- Skipped per plan: `MEMOPS` (29% / +10% gap — note the two sources disagree),
  `PYTORCH_CUDA_ALLOC_CONF`, TP, rope_*, `ngram_ram`.
- Vision tower is ~1.1 GB of VRAM (`Loading vision modules 30/30`, unquantized
  fp16) for a capability this workload never uses. Outside the knob list —
  worth a look if VRAM ever binds at 260k.
- **Nothing above 259k, and nothing can be.** `cache_size` is 262144 tokens
  allocated at load, so 259,145 prompt + 256 generated = 259,401 is the practical
  ceiling. The measured range now spans that ceiling (96–99% utilisation), so
  there is no untested band left. Earlier drafts of this report called 260k
  "impossible"; that was an artefact of a miscalibrated prompt file, not a
  real limit.
- **Per-half isolation was only re-run at 130k**, with an interleaved
  reference. The 224k and 250k figures are combo-vs-baseline comparisons only.
- **`mcs380`'s VRAM min-free spread remains the thinnest margin**: 269–2089 MB
  across the four short-context boots, though a tight 1479–1543 MB at 130k, 224k
  **and 250k** — the spread does not widen as context grows. If VRAM ever binds, drop `mcs380` and keep `thr16`, which costs
  no VRAM and still gives ~2.6% at 130k plus the full 11–16k decode win.