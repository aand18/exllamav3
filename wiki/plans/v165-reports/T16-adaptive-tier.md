# T16 — the adaptive expert tier: a pin budget, a promote rate, and the coverage number

Written 2026-11-02 on `feat/adaptive-tier` (base `165-win-perf`). **Nothing here has been run:**
the code is unbuilt and the unit tests are unrun (this box has no CUDA and no torch; the fork's
tests run on the Windows side). What exists is the instrumentation the direction was missing, the
two knobs its price depends on, and the protocol below.

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
| `EXL3_MOE_PINNED_ARENA_GIB=N` | `moe_cpu_host.py` (`MoeCpuTuning`, `_attach_chunk`, `_pin_budget_spent`, `arena_pin_report`) | Registers at most N GiB of arena chunks, then falls back chunk-by-chunk exactly like the driver-refusal path (`promote_ready` already refuses experts in unpinned chunks, so promotions simply do not target them). The arena is laid out hottest-first by the loaded profile, so the prefix is the hot end of the CPU tier — the part promotions are drawn from. Bounding it also leaves page-locked memory for the handoff ring, which the staged path runs on. `0`/unset = today's behaviour. |
| `EXL3_MOE_CPU_PROMOTE_BUDGET` (default `6`) | `block_sparse_mlp_cpu.py` `_PROMOTE_BUDGET` | Promotions per step over all layers. One moves ~1.95 MB over PCIe (~73 µs pinned), one avoided host expert is ~41 µs, so a promotion pays after ~2 reuses. 6/step is the shipped value; the runbook sweeps up. |
| `EXL3_MOE_CPU_PROMOTE_RATIO` (1.5), `EXL3_MOE_CPU_PROMOTE_DECAY` (0.98) | same | How fast the resident set follows a change of subject. Defaults unchanged. |
| `EXL3_MOE_TIER_STATS=1` | `block_sparse_mlp_cpu.py` (`_TierStats`, `_tier_stats_step`, `report_tier_stats`), hook in `generator.py` | Prints, per generation, the share of the routed expert evaluations the tier served (a step's evaluations are the change in each layer's hit-histogram sum, corrected for the placement policy's decay, so it stays exact at any count magnitude), plus the arena's pin state and the promotion/demotion counts. Counters come from the hit histograms `moe_split_map` already maintains, snapshotted into pinned memory and read one step later — nothing synchronizes mid-decode, but the extra D2H is why it is a diagnostic and not a default. `=2` adds the six worst layers. |

Docs: `doc/env_vars.md` (the four entries above, with the payback arithmetic).
Tests: `tests/moe_cpu/test_tier_stats.py` — the budget arithmetic (whole chunks, the chunk that
straddles it, zero = no budget) and the coverage math (evaluations, the promote path's per-step
decay, the re-baseline after a sweep's halving, layers of different width, worst layers). Unrun.

**Not changed:** the split itself (still a uniform `first` per layer), the sweep's interval and
budget, `spares`' default of 4, and every production config. A global (per-layer-varying) budget is
the bigger change the twin's own C3 arm says is worth up to 15 % (their equal per-layer budgets cost
them 15.5 %); it needs the VRAM allocation to vary per layer, and `size_check.py` to follow, so it is
not in this branch.

`spares` is the real churn pool, and its cost is RAM: each spare is one more expert in the worker's
arena, `48 × 1.95 MB ≈ 94 MiB` per spare per layer set. 4 (today) → 0.4 GiB, 16 → 1.5 GiB, 32 → 3.0 GiB
on top of the 32.6 GiB arena.

## 3. The protocolOne session, one process per arm, the same prompt stream, interleaved with a reference arm: this box
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

1. **The code has not been executed at all.** No CUDA on this box; no torch to run the tests. Expect
   the usual first-run issues: an import error, a numpy dtype mismatch, a wrong attribute name.
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

| arm | env delta | PP 4096 | PP 131072 | TG 130816 | coverage | promotions | hook ms/step | min avail RAM |
|---|---|---|---|---|---|---|---|---|
| **A** control | stats only | **1952.7** | **1920.3** | **38.30** | **28.5 %** | – | – | 6009 |
| **B** pin 8 GiB + promote | `-mcpa`, `PINNED_ARENA_GIB=8`, `PROMOTE=4` | 2048.7 | 2004.8 | 36.30 | 27.7 % | 695 | – | 5240 |
| **C** pin 8 GiB, wider pool | as B, `PROMOTE=16`, `SWAP_DEBUG=1` | 1992.9 | 1941.1 | 34.63 | 25.4 % | 663 | **1.03-1.06** | 5348 |
| **D** pin 8 GiB, no promote | as B, `PROMOTE=0` | 2043.6 | 2011.8 | 37.31 | 27.8 % | – | – | 7911 |

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
3. **Every spare is a hole in the tier.** At load the GPU slice is `first` slots with the spares'
   experts living in worker slots, so 4 spares cost 3 % of the tier and 16 cost 12 % *before any
   promotion happens*. Coverage fell monotonically with the spare count (28.5 → 27.7 → 25.4 %):
   the mechanism buys adaptability by giving away capacity, and on this workload the adaptability
   converts nothing.
4. **The hook is not free**: 1.03-1.06 ms/step measured by the engine (the per-step pinned snapshot
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
  i.e. not proven); promotion costs a further ~2.6 % on top of the measured 1.03 ms/step hook, and
  the wider pool costs more again (-9.6 % in C). No arm beats the control on decode.
- The pin also leaves *more* available RAM at the floor (7911 vs 6009 MB) — it replaces the staged
  path's separate pinned ring buffers with the arena mapping.

**Verdict against the gate.** Section 3 asked for "coverage up >= 10 points *and* tok/s up >= 3 %".
Promote-on-miss fails on both, in every pool size, and for a reason the counters name: it promotes
one step late and cannot see past the current prompt. **The promote path stays off by default**
(it already was), and the branch's remaining value is the two instruments plus the pin budget.

**What is left that the data still supports.** One thing: `EXL3_MOE_PINNED_ARENA_GIB` with
`EXL3_MOE_CPU_PROMOTE=0` is prefill-positive and decode-neutral at this noise level, i.e. a candidate
for the deployed `state.json` env (`EXL3_MOE_PINNED_ARENA=1`,
`EXL3_MOE_PINNED_ARENA_GIB=8`) — but a prefill claim at +4.7 % needs a *within-session interleaved*
A/B to clear the box's own pp threshold honestly, and this campaign compared separate processes.
Per the twin's account of the same measurement, that is the next run, not a conclusion.

**How this closes the direction, and what it does not.** The gate in section 3 was "coverage rises
by >= 10 points and tok/s by >= 3 %, else fail". Promote-on-miss as tuned fails both, and the
reason is now measured rather than inferred: a just-missed expert is promoted one step after the
route that needed it, and in a ladder whose prompt changes every 256 tokens that expert is exactly
the one that will not be routed again. The next candidate in this family is not a bigger pool or a
faster budget: it is a *ranking* that is right about the next thousand tokens rather than the last
one — which is what the twin's `--adapt-forecast` tried, and lost 20.2 % there.
