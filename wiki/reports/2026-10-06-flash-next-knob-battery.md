# Flash-Next knob battery — results (2026-10-06)

Measured impact of every tunable on 3.05bpw Flash-Next serving
(48 MoE layers, MTP head), RTX 4090 24 GB + 7950X3D, Win11 + WSL2.
Implements `wiki/plans/flash-knobs-benchmark.md`. All numbers are medians of
in-process reps unless stated. **Short-context screen (~17k prompt tokens) —
see §0.7 of the plan: these are a screen, not a verdict, and the long-context
pass is still outstanding.**

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

## Long context (plan §0.7) — the gain is decode-bound, not prefill-bound

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

- **Decode-bound turns** (short prompt, long generation — e.g. 11–16k prompt,
  256+ tokens out): **1.36× faster** measured turn-matched.
- **Prefill-bound turns** (long prompt — 55–62k here): **~1.00×**, because
  prefill is unchanged.

Both regimes are real for agentic coding with long context, so the combo is a
**conditional** win rather than a uniform one. It does not slow anything down;
it simply stops helping once prefill dominates.

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
  cpu_moe_split_experts: 380      # +7.6% alone, +48.3% with thr16
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

Expected total: **~1.36× sustained throughput** (5-turn conversation
28.87 s → 20.75 s), boot unchanged, VRAM free 3593 → ~1065 MB.

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
The likely mechanism is **cache pressure**: at 62k the KV cache is lightly
occupied, while at 224k against `cache_size 262144` it is ~85% full, so paging
and eviction enter the prefill path — and that is CPU MoE work, which is
precisely what `thr16` accelerates. `cpu_moe_split_experts:380` likewise
reduces CPU-side expert work, so its benefit grows as the cache fills.

Reproducibility is high enough to trust: baseline spread 0.08 s, combo 0.00 s
across two variants.

### 130k stage — INCONCLUSIVE. Retracted twice; do not quote a 130k verdict.

Correctly sized (`--target 65664`; server confirmed **129,998 / 129,910** prompt
tokens, so the calibration formula held to ~1%). Boot medians, seconds:

| config | boots | boot medians | all-boot median | ratio vs baseline |
|---|---|---|---|---|
| baseline | 3 | 77.9 / 79.1 / **86.8** | 79.14 | 1.000 |
| `thr16` | 2 | 85.2 / 87.6 | 86.47 | 1.093 |
| `mcs380` | 3 | 87.6 / 81.9 / 78.1 | 81.94 | 1.035 |
| combo | 1 | 86.8 | 86.76 | 1.096 |

**The baseline's own boot-to-boot spread is 77.9 → 86.8 s = 11.2%**, which is
larger than every between-config difference above. Ranges overlap the baseline
range completely:

| config | range | vs baseline max 86.8 |
|---|---|---|
| `thr16` | 85.2–87.6 | overlaps |
| `mcs380` | 78.1–87.6 | overlaps |
| combo | 86.8 | marginal |

**Verdict: inconclusive at 130k.** The apparent regression is within run-to-run
noise. Resolving it needs many more boots (this stage ran 1–3 per arm, against
6 tightly-clustered requests at 224k where the spread was 0.08 s).

**Two retractions on this stage, both from over-reading too few samples:**

1. An earlier revision said "prefill is flat at 62k, therefore prefill never
   improves" — falsified by the 224k stage.
2. The next revision said "the combo is ~11% slower at 130k, both halves
   regress independently" — that came from comparing **2-boot arms against a
   1-boot baseline**, and it does not survive equal replication. `mcs380`
   especially is noise: 87.6 / 81.9 / 78.1 across three boots.

Note the 224k stage by contrast had baseline spread 0.08 s and combo 0.00 s, so
the noise problem is specific to this length, not a property of the harness.

### 130k isolation — BOTH halves regress independently

Splitting the combo at 130k, two boots each, against the same baseline:

| config | v0 | v1 | median ratio | pp median | verdict |
|---|---|---|---|---|---|
| baseline | 78.03 s | 77.77 s | 1.000 | **1703** | ref |
| `thr16` alone | 85.54 s | 84.77 s | **1.093** | 1525 | SLOWER |
| `mcs380` alone | 87.81 s | 87.29 s | **1.124** | 1501 | SLOWER |
| combo (both) | 87.19 s | 86.33 s | 1.114 | 1488 | SLOWER |

**Both halves regress on their own** — this is not an interaction artefact.
Prefill degrades monotonically (1703 → 1525 → 1501 → 1488 T/s) and wall-clock
tracks it, so the cost is squarely in prefill.

Mechanism, consistent with both halves: **at 130k, prefill dominates and CPU MoE
work is on the critical path, so both changes ADD CPU cost rather than removing
it.**

- `thr16` oversubscribes the 16 physical cores during long sequential prefill
  chunks — the same effect `PERF_FINDINGS.md` recorded ("16 workers + stager +
  main on 16 phys cores"), which is why it rejected 16 on grounds that only
  surfaced here under a decode-only protocol.
- `mcs380` interleaves CPU expert work with each layer's own GPU compute, adding
  synchronisation the short-decode case never pays.

### Full regime table

| prompt len | cache occupancy | combo vs baseline |
|---|---|---|
| 11–16k | negligible | **1.36× faster** |
| ~55–62k | low | ~1.00× |
| ~130k | ~50% | **inconclusive** — inside an 11.2% baseline spread |
| ~224k | ~85% | 1.096× faster |

No clean trend: the 130k point is unresolved, so the curve cannot be called
monotonic in either direction.

### Revised regime table

| prompt length | cache pressure | measured |
|---|---|---|
| 11–16k | negligible | **1.36×** (decode-bound) |
| ~55–62k | low | ~1.00× (prefill-bound, no cache pressure yet) |
| ~224k | ~85% of `cache_size` | **1.096×** (prefill itself improves) |

The curve is **non-monotonic**, so the 62k point must not be read as "the
benefit decays with length". A true ~128k stage (correctly sized) is still
missing and would show whether 1.096× holds or keeps improving toward the
262k ceiling.

### Scope of the win — read before applying

The combo's benefit is **decode-bound, not prefill-bound**:

| regime | measured effect |
|---|---|
| decode-bound turns (11–16k prompt, 256+ out) | **1.36× faster** (turn-matched, 3 boots) |
| prefill-bound turns (55–62k prompt) | **~1.00×** — prefill flat at ~1640 T/s |

At 2× context a request is ~95% prefill, so the decode win is invisible in
wall-clock. **The gain shrinks toward zero as prompt length grows.** For a
128k+ workload where most turns are prefill-bound, expect far less than 1.36×.

`cpu_moe_split_experts: 380` also costs VRAM headroom (3593 → ~1100 MB free)
and buys **nothing** in prefill, which is the dominant cost at the target
context length. That is the central open question, and it is what the 128k and
258k stages exist to settle.

**UPDATE — the 130k question is UNRESOLVED, not answered.** An earlier revision
of this report claimed a ~11% regression at 130k for both changes. That came
from comparing 2-boot arms against a **1-boot baseline**; with equal
replication (baseline 3 boots) the baseline's own spread is 11.2%, larger than
every between-config effect, and all ranges overlap. **The 130k stage supports
no verdict in either direction.** It is reported as inconclusive rather than
negative.

What survives: both changes are large, reproducible **decode** wins at 11–16k
(1.36×, tight spreads) and the combo is a reproducible **prefill** win at 224k
(1.096×, spread 0.08 s vs 0.00 s). Whether either is a loss at 128k is
**unknown** and needs a properly replicated run before the config decision.

**RETRACTED** — an earlier revision claimed `EXL3_MOE_CPU_THREADS=16` was
"separable and safe to take alone — a pure decode win at every length tested".
That was inferred from short-context evidence only and is **not** supported. The
general lesson, recorded twice in this file now: *do not generalise a
per-length result across lengths, and do not generalise a small-n result across
configs.*

### Risk, honestly

- **`cpu_moe_split_experts: 380` is the whole win and the thinnest margin.**
  Clean 4-run min-free ranged 269–2089 MB. At 269 MB it is 1.3× the guard.
  At 128k+ context this is the first thing that will break. **Do not ship
  without the 2× context pass.**
- `cache_mode: 2,2` is **perf-only here** — quality not measured (KLD owns
  quality). Safe to revert independently; it is the fallback for VRAM relief.
- `mcs380` is the *only* viable split-experts value: 360 loads offline
  (23308 MB) but **will not boot the server**; 400/500 are VRAM- or RAM-refused.
- Mutually exclusive with `cpu_moe_offload_layers` — the server hard-errors.

## start_tuned.ps1 — verdict per env line

| line | verdict | evidence |
|---|---|---|
| `EXL3_MOE_CPU_THREADS=8` → **16** | **CHANGE** | +35.4% tg live, 4 reps, non-overlapping ranges. Peak is 16; 24 regresses. |
| `EXL3_MOE_CPU_PIN=1` | KEEP | −19.8% without. |
| `EXL3_MOE_CPU_SWIZZLE=1` | KEEP | −11.2% without. |
| `EXL3_MOE_MEMOPS=0` | KEEP | plan says 29% gap; `PERF_FINDINGS` says +10%. Not re-measured (SKIP per plan). |
| `EXL3_MOE_ZERO_COPY=1` | KEEP, low confidence | −2.5% without = inside noise. Prior work measured +3.3%. Not re-run at Tier 2. |
| `EXL3_MOE_STREAM_T=6` | KEEP | `STREAM_T=12` cost **pp256 −38%**; `STREAM_T=3` +0.8% tg, neutral pp. |
| `EXL3_MOE_STREAM_BATCH_EXPERTS=48` | KEEP | 24 → tg 25.6 vs 26.8 (−6.2%), pp4096 1867 vs 1826. |

No line qualifies for removal. The one change is threads.

## Full knob table

Phase A (offline `eval/perf.py`, Tier 1, `-short -max_length 4096 -dr 2`).
Baseline re-anchored in-run: tg0 24.91 [24.25–25.56], pp256 311.5, pp4096 2117.4.

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

Boot time is very stable: 53.5–53.9 s across baseline/`thr16`/`combo` (n=12).
`load` = 38.7–39.3 s. Warmup adds +12 s boot for no measurable gain.

## §2 interaction

Top-2 winners combined once: `mcs380` + `thr16`.
Individual gains +7.6% and +35.4% sum to +43.0%; measured +48.3%.
**Combo does NOT underperform the sum by >30%** — it exceeds it by ~5pp, so
the interaction is mildly *sub-additive-but-positive* and both changes are kept.
Mechanism: they attack the same bottleneck from opposite ends (more CPU
workers vs less CPU work), so each supplies part of what the other needs.

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

- **Long context (§0.7): NOT DONE.** Everything above is ~17k prompt tokens.
  `mcs380`'s 269–2089 MB spread is the specific risk at 128k/260k.
- `draft_num_tokens` 3 vs 5 is +6.0% tg but drops acceptance 2.78→2.20.
  Quality not measured (KLD owns it).
- `ZERO_COPY` at Tier 2; Tier 2 `-dr 3` not run for most finalists.
- `cache_mode` quality (KLD).
- MTP `ndt7`/`ndt8` ceiling.
- Skipped per plan: `MEMOPS` (29% / +10% gap — note the two sources disagree),
  `PYTORCH_CUDA_ALLOC_CONF`, TP, rope_*, `ngram_ram`.
- Vision tower is ~1.1 GB of VRAM (`Loading vision modules 30/30`, unquantized
  fp16) for a capability this workload never uses. Outside the knob list —
  worth a look if VRAM ever binds at 260k.