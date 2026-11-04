# T16 — the adaptive expert tier: a pin budget, a promote rate, and the coverage number

> **Update 2026-11-04 — the decode question is answered, and it splits the two halves of this feature.**
> `T32-decode-pin.md` and `T33-capped-pin.md` measured both halves with three decode reps per cell
> (the campaigns below ran one): the **pinned arena** is worth **+17–24 % decode** at the production
> context and **+16 % prefill**, while **promote-on-miss is worth nothing** (+0.08 % mean decode,
> −2.2 % prefill) and is not deployed. The "no proven decode gain" readings in §7 and the transfer
> review's M2/M4 rows describe the unpinned, instrumented campaigns and are superseded by those two
> reports. The coverage numbers in §7 remain valid as coverage; they are not a speed claim.

Written 2026-11-02 on `feat/adaptive-tier` (base `165-win-perf`), and **run on 2026-11-03**: the four
arms in §7 are measured, the suite ran on the Windows venv (§9), a review doubled back over both (§10),
and §6 records how the patch was installed. What this file was first written as — an unbuilt plan — is
what §2 and §3 still describe; §6 onwards is what happened.

Companion note: `/tmp/opencode/strata-vs-exllama-research.md` (the cross-engine measurement this
follows from). Read §2.1 and §3.1 of it before the first arm.

## 1. The gap this closes

Measured (that note, §2.1): exllamav3 decodes at **58.5 tok/s** on this box at 262 k context with
~**41 %** of its routed experts served from VRAM; UnbiasedStrata does **87-105 tok/s** at 185 k with
**79-95 %**. Both compute a host expert in ~40-44 µs, so the whole difference is how many experts
land on the worker — ~0.5 GB vs ~0.23 GB of host reads per token.

The engine already has the cheap mechanism (`_split_promote_step`: per-step, globally ranked, async
DMA from the arena) and does not use it: `spares > 0` needs `-mcpa`
(`EXL3_MOE_PINNED_ARENA`), and production runs the sweep instead — **64 swaps total across all 48
layers, every 128 steps, deferred to queue-drain, each one re-reading two experts from the
checkpoint**. 64 of 6,528 slots per 128 steps is 1 %, i.e. the load-time profile, i.e. Strata's own
28.4 % static-coverage number.

Three things were missing to test the alternative: a way to page-lock *part* of the arena on a box
that refuses the whole thing (WDDM refuses large `cudaHostRegister` here — the twin measured it
failing at 24-30 GiB), a way to set the churn rate without editing code, and — the gate — a
reported number for the tier's coverage, which the engine never printed.

## 2. What the branch adds

| what | where | why the default is what it is |
|---|---|---|
| `EXL3_MOE_PINNED_ARENA_GIB=N` | `moe_cpu_host.py` (`MoeCpuTuning`, `_attach_chunk`, `_pin_budget_spent`, `arena_pin_report`) | Registers at most N GiB of arena chunks, then falls back chunk-by-chunk exactly like the driver-refusal path (`promote_ready` already refuses experts in unpinned chunks, so promotions simply do not target them). The arena fills in chunk order and the layers register in order, so the prefix is the **first layers, not the hot end of the tier** — §7's measured 8 of 33 chunks reached ~11 of 48 layers, and ~37 could not promote at all (that "hot end" claim was wrong; §10 C3 is the correction). Bounding the budget also leaves page-locked memory for the handoff ring, which the staged path runs on, and `arena_pin_report()` now prints how many layers the pin made promotable, so a capped pool cannot be mistaken for a settled policy. `0`/unset = today's behaviour. |
| `EXL3_MOE_CPU_PROMOTE_BUDGET` (default `6`) | `block_sparse_mlp_cpu.py` `_PROMOTE_BUDGET` | Promotions per step over all layers. One moves ~1.95 MB over PCIe (~73 µs pinned), one avoided host expert is ~41 µs, so a promotion pays after ~2 reuses. 6/step is the shipped value; the runbook sweeps up. |
| `EXL3_MOE_CPU_PROMOTE_RATIO` (1.5), `EXL3_MOE_CPU_PROMOTE_DECAY` (0.98) | same | How fast the resident set follows a change of subject. Defaults unchanged. |
| `EXL3_MOE_TIER_STATS=1` | `block_sparse_mlp_cpu.py` (`_TierStats`, `_tier_stats_step`, `report_tier_stats`), called from the split step tick (`_split_step_count % EXL3_MOE_TIER_STATS_EVERY`) | Prints, per generation, the share of the routed expert evaluations the tier served (a step's evaluations are the change in each layer's hit-histogram sum, corrected for the placement policy's decay, so it stays exact at any count magnitude), plus the arena's pin state and the promotion/demotion counts. The line goes out from the module's own step tick, **not** from a hook in `generator.py`: the first draft hung it on the generator's queue-drained hook, which a `perf.py` ladder never reaches, so nothing was ever printed (§6). Counters come from the hit histograms `moe_split_map` already maintains, snapshotted into pinned memory and read one step later — nothing synchronizes mid-decode, but the extra D2H is why it is a diagnostic and not a default. `=2` adds the six worst layers. |

Docs: `doc/env_vars.md` (the four entries above, with the payback arithmetic).
Tests: `tests/moe_cpu/test_tier_stats.py` — the budget arithmetic (whole chunks, the chunk that
straddles it, zero = no budget) and the coverage math (evaluations, the promote path's per-step
decay, the re-baseline after a sweep's halving, layers of different width, worst layers). Run on the
Windows venv — the counts are in §9 and §10.

**Not changed:** the split itself (still a uniform `first` per layer), the sweep's interval and
budget, `spares`' default of 4, and every production config. A global (per-layer-varying) budget is
the bigger change the twin's own C3 arm says is worth up to 15 % (their equal per-layer budgets cost
them 15.5 %); it needs the VRAM allocation to vary per layer, and `size_check.py` to follow, so it is
not in this branch.

`spares` is the real churn pool, and its cost is RAM: each spare is one more expert in the worker's
arena, `48 × 1.95 MB ≈ 94 MiB` per spare per layer set. 4 (today) → 0.4 GiB, 16 → 1.5 GiB, 32 → 3.0 GiB
on top of the 32.6 GiB arena.

## 3. The protocol

One session, one process per arm, the same prompt stream, interleaved with a reference arm: this box
cannot be compared across processes or days (identical code has produced 39.3 vs 50.8 tok/s; the
twin's own noise floor was 0.8 % within a session). Regime: the deployed one from `state.json`
(262144, `"4,4"`, split 381, DFlash2 w4, 16 threads, `EXL3_MOE_THREAD_PRIO=1`), 131072 for the
sweep if the VRAM headroom is needed.

Arms, in this order (each one answers the question the next depends on):

| # | arm | env | question |
|---|---|---|---|
| A | reference | — | today's deployment: tok/s, and the tier's coverage (`EXL3_MOE_TIER_STATS=1`) |
| B | pin, today's rate | `EXL3_MOE_PINNED_ARENA=1`, `EXL3_MOE_PINNED_ARENA_GIB=8`, `EXL3_MOE_CPU_PROMOTE=4` | does the cheap path work at all here, and does coverage move? Read `promotions since load` — **0 means the pinned prefix did not cover the hot experts and the arm is void** |
| C | pin + wider churn pool | as B, `EXL3_MOE_CPU_PROMOTE=16` then `32` | coverage vs RAM spent (1.5 / 3.0 GiB more arena) |
| D | pin + higher rate | best of C, `EXL3_MOE_CPU_PROMOTE_BUDGET=16` then `32` | where the transfer becomes the cost |
| E | pin budget sensitivity | best of D, `EXL3_MOE_PINNED_ARENA_GIB=4 / 16` | how much prefix the churn actually needs (the twin's partial-pin analogue was +15.9 % there) |

Gates, decided before the run:

- **A→B passes** if coverage rises by ≥ 10 points *and* tok/s by ≥ 3 %. Coverage moving with tok/s
  flat means the transfers ate the gain (record the promotion count); neither moving means the
  machinery is not engaging (check the pin line and `promote_ready`).
- **C/D pass** if tok/s peaks rather than plateauing — i.e. a setting between the arms beats both.
  Stop raising the rate at the first arm that loses, and keep the previous one.
- **Any arm** with a floor breach (200 MB VRAM / 2048 MB available RAM) or a traceback is `fail`,
  never `ok` (`bench/verify_task.py --post`).
- Fail the whole direction if **A's coverage is already > 70 %** — then the tier is not the gap and
  the next target is the unattributed ~5 ms/token of GPU time (§2.4 of the companion note), which
  needs a per-stage instrument this fork does not have.

Read per arm: `expert tier: N% coverage`, `misses/step`, `promotions/demotions since load`, and the
existing `split prof [wait]` median (`EXL3_SPLIT_PROF=1`) — the exposed stall is the quantity a
smaller pool is supposed to shrink; if coverage rises and the wait does not fall, the pool was not
the critical path.

Harness: `run_pass.ps1` with `-Runner spec_decode` and the env above; draft arguments are
`spec_decode`-only; `state.json` stays the regime and the arm passes a delta. Note
`EXL3_MOE_CPU_SWAP=0` is automatic for `-Runner spec_decode` — the *sweep* is off in that runner, so
arms B-E measure promote-on-miss against a frozen profile, which is the comparison we want.

## 4. What is not verified

1. **The code has now been executed, and the first run's defects were not the ones this list
   predicted.** No import error, no numpy dtype mismatch, no wrong attribute name: what the box found
   was a report hung on a hook a `perf.py` ladder never reaches, a prefill landing in the next decode
   delta (both §6), a `torch.cuda.Event(device = d)` that raised `TypeError` on torch 2.11, and
   `pinned_layers()` walking the worker's ack one nesting level short (§10 C5, C3). Each is a
   prediction this list got wrong, recorded here so the list is not read as the run's outcome.
2. **The `spares` pool may be the binding constraint, not the budget.** With 4 spares per layer only
   4 of 131 slots per layer can hold a freshly promoted expert, so arm B may be capped regardless of
   `EXL3_MOE_CPU_PROMOTE_BUDGET`. That is why C raises `EXL3_MOE_CPU_PROMOTE` first.
3. **The pinned prefix may not cover the hot experts** if the loaded profile is stale: the arena is
   permuted by the *saved* profile, and the profile is what the load reads. Arm B's promotion count
   is the check.
4. **The coverage metric counts routed evaluations per step**, not distinct experts: an expert
   routed twice in a step counts twice. It is therefore not the same number as the `hit rate` the
   promote path's debug line prints (that one counts distinct routed pairs one step apart). They
   should agree within a few points; if they do not, trust neither until the discrepancy is explained.
5. **Pin budget + handoff ring**: `_register_handoff` still takes arena chunks back if the handoff
   segment does not fit. A budget leaves room, but on a box this short of RAM the final pin state is
   whatever the load reaches — the `arena ... pinned in N of M chunk(s)` tail of the stats line is
   the record, and a low prefix with zero promotions means go lower on the budget, not higher.

## 5. Risks

- A wider pinned prefix costs page-locked RAM that the KV tiers and the OS also want; the twin
  measured the box at 0.0 GB available under a 33 GiB lock, so the pin is bounded here by design.
- More spares cost arena RAM (~94 MiB each), which the same floor measures.
- Promotions move an expert between the GPU and the worker, and the two devices round the expert's
  arithmetic differently: a generation in progress *can* change its output when the tier changes
  under it (the twin measured this as the reason a tok/s A/B cannot resolve tier changes below ~2 %
  of the window). For byte-identical repeats, `EXL3_MOE_CPU_PROMOTE=0`.

## 6. Run log (2026-11-03)

**Deployment restored, then left alone.** The first deploy from this branch overwrote
`_agent/v165` with the branch's older harness; the box's copy is fed from the `165-win-perf`
worktree, and its `run_pass.ps1` hashes to `9B98E754…`, the value recorded on the last run of
2026-11-02. Re-deploying from that worktree restored it (`run_pass.ps1` `9B98E754…`, `perf.py`
`b270a0e9…`, the production regime in `state.json`: split 381, 262144, `4,4`, DFlash2 window 4 with
dynamic drafting). Nothing in this branch is deployed there; the arms use its env deltas only.

**How the patch was installed (no wheel).** The change is pure Python, so it went into the venv's
`site-packages` as two files — `exllamav3/modules/block_sparse_mlp_cpu.py` and
`exllamav3/model/moe_cpu_host.py` — with the replaced originals backed up under
`_agent/adaptive-t27/backup/` (`ec103eb9…` and `f7342c9d…`) for an exact rollback. The wheel build
(`run_build.ps1`, ~33 min) is still the artifact for anything deployed.

**Two defects the first pass found, both fixed** (`73625217`):

1. the report was hung off the generator's queue-drained hook, which a `perf.py` ladder never
   reaches (that runner drives the model directly), so nothing was ever printed there;
2. a **prefill** routes the prompt through the same hit histograms and nothing marked the
   transition, so the prompt's whole routing landed in the next decode step's delta — 120x the true
   per-step count in the log. Prefill-size batches now *invalidate* the comparison window, and a
   block whose arithmetic cannot be a decode step (routed <= 0, missed < 0, missed > routed) only
   re-baselines. The line now goes out from the step tick every 128 decode steps
   (`EXL3_MOE_TIER_STATS_EVERY`), which is one mechanism in every runner and a fixed window.

**A measurement that is not a tok/s number.** A drafting pass with dynamic placement re-enables the
sweep, and a sweep can fire inline inside a long generation (the generator's queue-drain defers it,
but `_split_swap_tick` sweeps anyway once it is 4x overdue). The run degenerated into repetition —
endless tool-call XML, "All work and no play makes Jack a dull boy" — which is exactly why
`spec_decode.py` refuses dynamic placement for a multi-arm pass. Its tok/s is therefore void. Its
**coverage** readings are not, because a 128-step window whose evaluations are 480 per step is a
decode window whatever the text says:

| baseline, production regime, cs 40960, DFlash2 w4, loaded profile | value |
|---|---|
| decode-clean windows (evaluations / step = 480, i.e. one routed row) | 12 of 27 |
| coverage | min 15.9 %, **median 39.4 %**, mean 37.0 %, max 61.1 % |

So the ~41 % that the research note derived from the worker profile's bytes is what the engine
measures, and it moves 16-61 points with the text — the tier is already reacting somewhat to the
conversation, and the ceiling is not 100 % coverage but the *stable* part of it.

The all-arms campaign therefore runs `perf.py` ladders: no drafting, so a placement change cannot
corrupt a band's timing, and `tg` at 130816 is the repo's canonical decode number.

## 7. Arms, measured (2026-11-03)

Regime: the deployed one from `state.json` (3.18bpw_h6_ng8, split 381, 262144 Q4, chunk 4096,
`-ambs 2`, 16 threads), `-Runner perf -MaxLength 131072`, found-unit floors (200 MB VRAM / 2048 MB
available RAM) held in every arm (`breach=0`, `FLOORS` line present), each pass preceded by the
harness's frozen-profile restore so every arm starts from the same expert ranking. Coverage is the
pooled figure over the pass's 128-step windows; promotions are the engine's own counters.

| arm | run id | env delta | PP 4096 | PP 131072 | TG 130816 | coverage | promotions | hook ms/step | min avail RAM |
|---|---|---|---|---|---|---|---|---|---|
| **A** control | `t27-A-perf-20261103-121739` | stats only | **1952.7** | **1920.3** | **38.30** | **28.5 %** | – | – | 6009 |
| **B** pin 8 GiB + promote | `t27-B-pin8-f4-20261103-122212` | `-mcpa`, `PINNED_ARENA_GIB=8`, `PROMOTE=4` | 2048.7 | 2004.8 | 36.30 | 27.7 % | 695 | – | 5240 |
| **C** pin 8 GiB, wider pool | `t27-C-pin8-f16-20261103-122756` | as B, `PROMOTE=16`, `SWAP_DEBUG=1` | 1992.9 | 1941.1 | 34.63 | 25.4 % | 663 | **1.05** | 5348 |
| **D** pin 8 GiB, no promote | `t27-D-pin8-nopromo-20261103-123221` | as B, `PROMOTE=0` | 2043.6 | 2011.8 | 37.31 | 27.8 % | – | – | 7911 |

All four are `status=ok`, `breach=0`, in `results/passes.csv` of the deployed harness; the void
drafting attempt before them (`t27-A-base-20261103-121148`, `status=fail`) and its `-swap_ok` re-run
are the subject of section 6.

**What the counters say (deterministic, not timing).**

1. **The cheap path engages and is safe.** The pin budget landed exactly where it was asked to
   ("pin budget 8 GiB spent at chunk 8 (8192 MiB registered)"), the load came up in
   promote-on-miss mode ("4/16 spare slots, placement from saved profile"), and every pass held its
   floors with 8 GiB page-locked beside the 2.9 GiB pinned KV cache.
2. **Promotion does not converge to a better tier at this workload.** The count settles at **~680
   promotions for the whole pass**, and it does *not* scale with the pool: 695 at 4 spares, 663 at
   16. The policy, not the pool, is the cap — the ratio test (a miss at 1.5x the least-used
   unresident resident, on 0.98-decayed counts) stops qualifying candidates after the first few
   hundred steps. The engine's own line agrees: 31.2-39.1 % hit over its last 128 steps, the same
   band as the control's 28.5 % pooled.

   > **Superseded (2026-11-04, T31 §2/§7.1).** This item is the *pinned* campaign's reading and is
   > kept as history. In the second round of t29 both "the policy is the cap" and "the count settles"
   > were contradicted: with the pin budget unset and 16 spares,
   > `t29-C1-pin-unset-20261103-221302` promoted **8 669** times by step 3 584 = **2.42/step**, a
   > **12× realized promotion rate** over this campaign's 0.201/step (the 6/step `PROMOTE_BUDGET` was
   > never the binding constraint in either arm), and pooled coverage went 31.7 → 72.3 %. What survives
   > from this item is the hook's cost and the shape — the promote path shuffles each layer's own
   > spares, Strata re-ranks one global pool — not the plateau.

3. **Every spare is a hole in the tier.** At load the GPU slice is `first` slots with the spares'
   experts living in worker slots, so 4 spares cost 3 % of the tier and 16 cost 12 % *before any
   promotion happens*. Coverage fell monotonically with the spare count (28.5 → 27.7 → 25.4 %):
   the mechanism buys adaptability by giving away capacity, and on this workload the adaptability
   converts nothing.

   > **Superseded (2026-11-04, T31 §7.1).** Same pinned campaign, same caveat: the coverage column here
   > came from the instrument whose placement mirror is built once and never refreshed (section 10), so
   > the 28.5 → 27.7 → 25.4 % ordering is not usable as evidence about the spare count. The accounting
   > point — a spare is a slot the resident tier does not have — stands; "the adaptability converts
   > nothing" does not: on the fixed instrument the second-round t29 arms read 72.3 % pooled with
   > promotion against 31.7 % without, in one interleaved session.
4. **The hook is not free**: 1.03 to 1.06 ms/step measured by the engine (the per-step pinned snapshot
   plus the numpy ranking over 48 x 512 counts). At ~28 ms/step that is ~4 % of decode, which is
   the size of B's and C's decode deltas.

**What the timings say, with the noise stated.** The repo's own acceptance thresholds are +-2 % pp
and +-11 % tg (worst band), from T4, and these arms are separate processes.

- **Prefill gains, repeatably.** +4.7 / +4.8 % at 4096 / 131072 for the pin alone (D), +4.9 / +4.4 %
  with promotion on (B): three of four comparisons clear the pp threshold, in the direction the
  flag exists for (the pinned arena gives the streamed-prefill path a direct DMA source instead of
  the worker's staging ring). It is the one positive the campaign produced. It is also the same
  effect the Strata twin measured on its own expert copy (+15.9 % from a partial pin), with the
  difference that their pin covered the *hot* part of the complement and ours a 1 GiB-chunk prefix
  of the arena.
- **Decode does not gain anywhere.** The pin alone is -2.6 % at 130816 (inside the tg threshold,
  i.e. not proven); promotion costs a further ~2.6 % on top of the hook the engine reports (1.03 to 1.06 ms/step), and
  the wider pool costs more again (-9.6 % in C). No arm beats the control on decode.
- The pin also leaves *more* available RAM at the floor (7911 vs 6009 MB) — it replaces the staged
  path's separate pinned ring buffers with the arena mapping.
- **This is the second campaign to see it.** The first was on 1.5.4 (2.05 bpw, split 360, 32k,
  `EXL3_MOE_STREAM_T=6`/`STREAM_BATCH_EXPERTS=48`): streamed prefill 2417-2441 tok/s unpinned against
  2906-2979 pinned, **+21 %**, across 4096/8192/16384/32768, with decode unresolved because S=1 swung
  25.4-32.0 it/s within a single run (Engram #260). Two engine versions, two quants and two splits
  agree on the sign; the magnitudes differ (21 % then, 4.7 % now), which is what the different share
  of *streamed* experts per prefill chunk predicts — the pin only accelerates experts the stager
  would have copied, and the newer split leaves more of them resident.

**Verdict against the gate — corrected after review (see section 10).** Section 3 asked for
"coverage up >= 10 points *and* tok/s up >= 3 %". What the campaign actually established, and what it
did not:

- **The coverage column of B and C is not usable.** The instrument's placement mirror is built once
  at construction and never refreshed, so a promoted expert keeps being scored at its old home. The
  same logs carry a second, live-mirror instrument from the promote path, and it reads **31.1 % mean
  hit for C against the control's 28.5 %** — the opposite sign to the "coverage fell with the spares"
  reading of the frozen column (28.5 / 27.7 / 25.4 / 28.5). Fixed on this branch ("Tier stats: the placement mirror follows the map"): the mirror now
  follows a per-module placement generation (asserted by two new tests), so the next pass can answer
  this. Until then the gate is **open**, not failed.
- **The promotion counters had not plateaued.** Final values are **710 (B) and 679 (C)**, still
  climbing at the end of the pass; "settles at ~680" quoted the 25th of 28 windows. The slope is
  pool-independent (+95 over the last ten windows at 4 spares, +88 at 16), which is real evidence
  against the *pool* being the cap — but at ~0.07 promotions/step and with the budget knobs never
  swept (`PROMOTE_BUDGET`, `_RATIO`, `_DECAY` appear in no arm) it says nothing about the *policy*.
- **The pin budget also capped the experiment.** The arena fills in layer order, so 8 of 33 chunks
  pin the *first ~11 layers* — a prefix is not "the hot end" (that claim was wrong and is corrected
  in the code, `doc/env_vars.md` and here). ~37 of 48 layers could not promote at all. The tier line
  now prints the promotable layer range.
- **Prefill, recomputed per band from `pp.csv`:** D is >= +2 % in **10 of 10** bands (mean +6.9 %),
  B in 9 of 10 (+3.6 %), and C — same pin, same budget — in only **3 of 10** (+1.9 %, four bands
  negative). So all four arms clear the threshold at 131072, and the C non-reproduction is an
  unexplained caveat, not a clean win.
- **No interleaved reference.** Four single passes, one each, in sequence: that is what section 3
  forbids and `AGENTS.md` forbids. The prefill number needs an A/D/A/D interleave before it is quoted
  as a deployment candidate; the four decode deltas (-2.6, -5.2, -9.6 %) are all inside the +-11 %
  threshold and none of them is proven either.

**So the honest result of this campaign is narrower than its first draft: the cheap DMA path engages
on this box (710 promotions, a live-mirror hit rate above the control), the pin budget bounds an
experiment in a way that has to be measured per layer, and the promotion rate, the policy constants
and the instrument itself were not yet in a state to answer the gate.** Sections 10 and 11 list what
that costs and what settles it.

**How this closes the direction, and what it does not.** The gate in section 3 was "coverage rises
by >= 10 points and tok/s by >= 3 %, else fail". Promote-on-miss as tuned fails both, and the
reason is now measured rather than inferred: a just-missed expert is promoted one step after the
route that needed it, and in a ladder whose prompt changes every 256 tokens that expert is exactly
the one that will not be routed again. The next candidate in this family is not a bigger pool or a
faster budget: it is a *ranking* that is right about the next thousand tokens rather than the last
one — which is what the twin's `--adapt-forecast` tried, and lost 20.2 % there.

> **Superseded as a verdict (2026-11-04, T31 §2 and §7.1 — kept as the campaign's own closing).** This
> conclusion was drawn on the pinned arms and a frozen coverage column. The second-round t29 arms,
> interleaved in one session on the fixed instrument, put promote-on-miss at **72.3 % pooled coverage
> against 31.7 % for the pin-off control (+40.59 points)**, with decode still a null (+1.13 % TG inside
> a 4.60 % repeat spread) — so "fails both" is wrong on the coverage half and holds only for tok/s. The
> promote path's *source* is also not the divergence it was read as here: both measured arms are
> promote-on-miss DMA out of the pinned arena (`moe_cpu_host.py:1884-1918`), and only the legacy sweep
> re-reads the checkpoint. What this paragraph still contributes is the *ranking* argument — a policy
> that reacts one step after the route — which the t29 rate (2.42 promotions/step against Strata's
> 15.1 adaptations/round) leaves open. T31 is the current reading; this report is the campaign record.

## 8. End state of the box

- **Server**: stopped for the campaign, then restarted with the operator's own `start_tuned.ps1` and
  verified serving (`GET /v1/models` -> HTTP 200, `Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8`, key
  unchanged), 21.9 GiB VRAM in use. The served config was never changed.
- **Installed engine**: restored to the deployed files from `_agent/adaptive-t27/backup/`
  (`block_sparse_mlp_cpu.py` `ec103eb9…`, `moe_cpu_host.py` `f7342c9d…`, `generator.py`
  `6edbdb19…`), and the patched symbols are verified absent from the installed package. The
  measurement ran a file-level patch of a pure-Python change, not a wheel; a wheel build
  (`run_build.ps1`, ~33 min) is what a deployment would need.
- **Harness deployment**: `_agent/v165` untouched by this branch (see section 6).
- The four arms' evidence lives in the deployed harness's `results/` and `logs/` (`perf-t27-*.log`),
  and `bench/verify_task.py --post` reads this report's run ids against `results/passes.csv`.

The gate, re-run on 2026-11-04 from the deployment's worktree
(`…/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf`), no GPU and no process:

```
$ python3 bench/verify_task.py --post --report <this file> --cites-failure t27-A-base-20261103-121148
post-run gates
  PASS  t27-A-base-20261103-121148: status=fail, cited as failure evidence (not a measurement)
  PASS  t27-A-perf-20261103-121739: 48 CPU-split component(s), every one from the saved profile
  PASS  t27-B-pin8-f4-20261103-122212: 48 CPU-split component(s), every one from the saved profile
  PASS  t27-C-pin8-f16-20261103-122756: 48 CPU-split component(s), every one from the saved profile
  PASS  t27-D-pin8-nopromo-20261103-123221: 48 CPU-split component(s), every one from the saved profile
verdict: all gates passed
```

Its `INFO` lines (one per prose `%` figure, each "prose, not a ratio band - not checked") are omitted;
no `FAIL` line printed and the exit code is 0. The `--cites-failure` argument is this report's own
rule in practice: `t27-A-base` is the void drafting attempt of section 6, cited as evidence of a
failure, so without that flag the gate reads it as a measurement and fails.

## 9. Tests (2026-11-03, re-run 2026-11-04 after the handle fix)

The suite was run against the **source tree with the patch** (`tests/conftest.py` puts the repo first
on `sys.path`, so the tests exercise the branch, not the installed copy), on the Windows venv
(pytest 9.1.1, torch 2.11.0+cu130), per directory because the venv has no pytest-xdist and the whole
tree takes ~35 min serially. 7,052 tests collected.

| directory | result |
|---|---|
| `kernels/moe_cpu_host` | 26 passed, 14 skipped |
| `moe_cpu` | **109 passed, 63 skipped** (includes the new `test_tier_stats.py`: 10 tests) |
| `cache` | 109 passed, 63 skipped |
| `generator` | 55 passed |
| `graph` | 222 passed, 47 skipped |
| `kernels` | 5,001 passed, 531 skipped, **2 failed** |
| `modules` | 315 passed, 5 skipped, **131 failed** |
| `conversion` | 193 passed, **3 failed** |
| `tp` | 28 passed, **2 failed** |
| `tokenizer`, `loader`, `model_setup`, `e2e` | 16+4+15 passed, 26 skipped, no failures |
| `parity` | 69 skipped, **7 errors** |

**Every red cluster is pre-existing and environmental — proven, not asserted.** The same selections
were run on a pristine copy of the base commit (`2df1b005`) with the two patched engine files
restored, and the counts are identical:

| cluster | patched | base |
|---|---|---|
| `modules` (131) | 131 failed, 315 passed | **131 failed, 315 passed** |
| `kernels`: ptx + cross-process, `tp`: overflow | 4 failed, 31 passed | **4 failed, 31 passed** |
| `conversion`: optimizer targets | 3 failed, 193 passed | **3 failed, 193 passed** |

- the 131 are **one mechanism, not 131 defects**: `RuntimeError: Error opening file ... Too many open
  files (errno=24)` from `loader/safetensors.py`, i.e. the suite accumulates open synthetic
  checkpoints per test until the process runs out of handles. `tests/modules/mla/test_mla_dsa.py`
  **alone passes 12/12** on the patched tree; the same file is the tail of the cascade in the full
  directory run. A Windows-only limit: the suite was written on Linux.
- `test_ptx_portable` (no `nvcc`/arch match), `test_run_blocks_until_marker_across_processes`
  (cross-process sync), `test_sm_overflow` (shared-memory overflow segments) and the `parity` setup
  errors (HF `transformers` device-map API) are all platform/dependency gaps, identical on base.

### 9.1 After the handle fix (2026-11-04, `4edc3955`)

The suite was re-run as **one invocation** at `1fb1696b`, against the branch tree on the Windows venv,
with the production server stopped (the CUDA tests need the card):

```
11 failed, 6097 passed, 951 skipped, 7 errors in 638.35s (0:10:38)
```

| | before (`2df1b005` + the two patched files) | after the fix |
|---|---|---|
| `modules` | **131 failed**, 315 passed | **4 failed** |
| whole tree | 138 failed, 7 errors | **11 failed, 7 errors** |

The cascade is gone, and it was one mechanism: `tests/testlib/checkpoint.py` now returns a
`SelfClosingCollection` whose `__del__` closes the loader's streams (`4edc3955`). The count is 8
streams per collection off Linux (`STLOADER_THREADS`, `exllamav3_ext/stloader.h`), so the 512-entry
CRT fd table fills at **~64 collections**, not ~500 — the failure arrives at case 63. The regression
test (`tests/loader/test_collection_lifetime.py`) counts **kernel handles**, because CPython's own
I/O layer is not bounded by MSVCRT `maxstdio` and an `os.open` probe cannot see the leak.

The 11 remaining failures are pre-existing, measured as one selection on a pristine `2df1b005` copy:
**11 failed, 23 passed**, the same eleven ids. Three `conversion/test_optimizer_targets`, one
`kernels/portability/test_ptx_portable`, one `kernels/tp/test_tp_context`, four
`modules/draft/test_dflash2`, two `tp/test_sm_overflow`. The seven `parity` errors are setup errors —
`ValueError: Using a device_map … requires accelerate`. Against base the branch is **three reds
better**: the two `test_shm_capacity` cases and `test_swap_sweep_inference_mode`.

**Two suite defects were fixed**, because they were red *only* for staleness (commit `acfbd7ac`,
both verified to fail on base first): `test_swap_sweep_inference_mode`'s stub lacked
`Generator.sparse_head`, which the hook reads first, so the test died before asserting the inference
mode it exists for; and `test_shm_capacity` sizes against `/dev/shm` without the `platform("linux")`
marker the rest of the suite uses for OS-specific cases.

`bench/test_gates.py` (the harness's own offline gates, which is where the `--cites-failure` rule
this report needs comes from) passes: `OK`.

## 10. Review of this branch (T28) and what it changed

An independent critique of the branch, its four arms and this report was run after the campaign
(`/tmp/opencode/t28-critique.md`, 27 findings: 3 blocker, 12 major, 8 minor, 4 nit). Its blockers
were verified against the source and the logs before being acted on — the recomputations below are
mine, from the same `logs/perf-t27-*.log` and `results/*.csv` the critique used.

| finding | verified how | response |
|---|---|---|
| **C1/M1 blocker** — `_TierStats` freezes its placement mirror at construction, so it cannot see a promotion | by inspection (the mask was built once in `__init__`) and by recomputing both instruments from the logs: C's tier column 24.6 % mean, C's own promote line 31.1 % mean, A/D 27.8 % each | **fixed**: a per-module `_split_gen` is bumped by every landed promotion/demotion and by a sweep, and `_refresh()` re-reads the map and the worker mask when it moves (one D2H per move, never per step). Two tests: a promotion flips a miss to a hit, a demotion flips a hit to a miss, and the sweep path (device map only) is followed too. |
| **C3 major** — the pin budget pins a prefix of the *layers*, not "the hot end of the tier" | by inspection of `_HugeArena` (1 GiB chunks filled in registration order, layers register in order) and the log line "8 of 33 chunk(s)" | **fixed as a claim**: the code comment, `doc/env_vars.md` and section 7 above now state the layer-order consequence; `arena_pin_report()` gained `pinned_layers()` and prints the promotable count on every tier line — `0 of N`, `N of M (0..K)` or `all N`, so neither a pin that covered no layer nor one that covered every layer can be mistaken for the partial case. The *design* fix (order the arena by heat across layers) is not attempted. |
| **C2 major** — the tier state survives `cpu_unload`, holding the modules (and their GPU slices) alive | `grep moe_cpu_tier_stats` found no reset | **fixed**: `cpu_unload` drops it beside `moe_cpu_promote_state` |
| **C5 minor** — a `torch.cuda.Event()` created on device 0 and recorded on another | by inspection, then by the crash it caused in the first t29 pass | **fixed, but not as first written**: `Event(device = d)` *was the crash* — torch 2.11's `Event.__new__` takes `(enable_timing, blocking, interprocess, external)` and no `device` kwarg, so the kwarg raised `TypeError` and took every `EXL3_MOE_TIER_STATS` pass down after the prefill ladder. The fix constructs the event inside `torch.cuda.device(d)` and records it on that device's current stream, so each device gets its own event |
| **C6/C7 minor** — two accounting styles, `>> 30` truncation printing "0 GiB" for a sub-GiB budget, and 0 meaning "unlimited" without saying so | by inspection | **fixed**: one formatter, real byte totals, `0 = no budget` documented |
| **C8 nit** — the docs still described the generator hook the code no longer uses | `grep report_tier_stats` | **fixed**: the docs describe the step tick and `EXL3_MOE_TIER_STATS_EVERY` |
| **M2, M3, M4, M6, R1-R5, P1-P4** — the missing budget/policy arms, the still-climbing counters, no interleaved reference, the instrument's price, the untested M1/M3/M4 mechanisms, and the protocol slips | recomputed per band and per window | **recorded, not fixed**: section 7 now states each one, and section 11 is the arm list that settles them. None of it is code; all of it is another GPU session. |

Tests after the fixes: `tests/moe_cpu` **111 passed, 63 skipped** (the two new placement-following
tests included), on the box's venv against the patched source tree. On 2026-11-04 the same directory on
the same venv, at this branch's HEAD, reports **119 passed, 63 skipped** — the later pin-report cases
(`cd39e08b`) and this round's two (`test_pin_report_states_zero_and_all_layers_explicitly` and
`test_pin_report_counts_only_layers_that_acked_blocks`) are the difference.

## 11. What the next session must run, in order

1. **Re-run arm C unchanged** (`-mcpa`, `PINNED_ARENA_GIB=8`, `PROMOTE=16`) with the fixed mask: it
   decides whether the gate was met (the live-mirror number says "close, not 10 points") or whether
   the campaign's negative stands.
2. **The policy arm the plan promised**: `PROMOTE=16 PROMOTE_BUDGET=32 PROMOTE_DECAY=0.90`, with
   `EXL3_MOE_PINNED_ARENA_GIB` **unset** so the driver's own limit applies (more than 8 chunks) and
   `EXL3_SPLIT_PROF=1` for the exposed `split prof [wait]` median that section 3 made the decision
   variable. This separates "the policy plateaued" from "the prefix capped it".
3. **`EXL3_MOE_STREAM_MIN_ROWS=2` on top of arm D** — the research's M3 (decode-size PCIe split of
   misses), reachable as an env delta with no code.
4. **Move the budget, not the pin**: D at `-CacheSize 32768 -MaxLength 32768` (plus the `-Split` and
   `size_check.py` that follow) to buy the research's M1 (**+23–27 experts/layer**, not the +10–13 the
   research note divided by the wrong per-expert size — T31 §1 has the corrected arithmetic: 2.11 or
   2.47 GB freed at 1.91 MB per expert is 1106 or 1291 slots, i.e. 131 → 154..158 of 512), reading
   coverage with the fixed instrument and comparing pp at 4096/32768 rather than tg at 130816.
5. **A/D/A/D interleaved, one with `EXL3_MOE_TIER_STATS=0`** — the same-session reference `AGENTS.md`
   requires, which also prices the instrument (the decode deltas in section 7 are all inside the
   noise floor without it).
