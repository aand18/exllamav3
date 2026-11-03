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
