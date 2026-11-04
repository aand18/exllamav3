# T31 — Strata → exllamav3 transfer: the matrix, and its verdicts machine-checked

> **Update 2026-11-04 — two rows now have a measurement instead of a gap.** The campaigns this report
> cites (`t27`, `t29`) measured decode once per cell; `T32-decode-pin.md` remeasured with three reps
> and separated the two halves of the mechanism. The result: the **byte source** row stands (M2's
> transfer source is faithful, RAM → VRAM from the arena), but the **churn** half of M2 is measured
> dead — promotions add +0.08 % mean decode over eleven contexts and cost 2.2 % of prefill — so the
> feature is not deployed. M1's budget lever (free KV to buy expert slots) remains unmeasured, and the
> pinned arena does **not** achieve the same end from the other side: T32/T33's first reading of
> +17–24 % decode came from the dynamic-placement regime those arms ran, and in the served
> configuration (DFlash2 forces `EXL3_MOE_CPU_SWAP=0`) the arena is never read during decode — the
> drafter confirmation moved the agentic-weighted metric by −0.93 %. The pin's only real gain is
> prefill, and the deployment was reverted. M3/M4/M5/M7 are unchanged.

Written 2026-11-03/04 on `feat/adaptive-tier` (base `165-win-perf`). **No GPU and no measurement pass
was started for this report:** every performance number below was already measured, and what is new
here is the comparison — the one thing executed was `bench/size_check.py`, which reads the
checkpoint's tensor headers to price a configuration (§8). Six mechanisms (M1, M2, M3, M4, M5, M7) get
one row each: the value **both** engines' sources state, with the `file:line` it came from, a verdict,
the reason for it, and the run id that measured it or the literal `unmeasured` — with the run ids and
their numbers cashed row by row in §7, and what a port would cost in §8.

The matrix is not prose. It lives in `bench/strata_transfer_check.py` as `MATRIX`, and
`python3 bench/strata_transfer_check.py --check` compares every recorded constant against a
regex read of the two sources, every `derived` number against its own recomputation, and the citation
discipline (`measured` a run id or `unmeasured`; a justification wherever one is owed). If either
engine's default drifts, or a division stops adding up, the check fails and names the row and the key.
The rows' `justification` strings in the code are the sections below, quoted.

Sources: `bench/strata_transfer_check.py:MATRIX` (authoritative), Strata at `/home/dev/UnbiasedStrata`
(read-only), this branch, `results/passes.csv` of the deployed harness, `wiki/plans/v165-reports/
T16-adaptive-tier.md` §7, and `/tmp/opencode/t29-pin-unset.md`. "Note 1" below is
`/home/dev/research/strata-vs-exllamav3-2026-11-02.md` (the seven-mechanism note); "note 2" is
`/tmp/opencode/strata-vs-exllama-research.md` (the coverage-attribution note).

**How evidence is cited here.** A harness run is cited by its run id — tag + UTC stamp — because
that is the key `results/passes.csv` holds and the key the post gate resolves; its log is
`logs/perf-<run id>.log`, and written out in full that name is itself read as a run id and reported
as a missing row. Strata's own log is cited as `server log:<lines>`: it is the 2026-10-29 17:23:52
one, `server-<stamp>.err.log` under `/mnt/c/Users/yoho/Downloads/UnbiasedStrata/logs/`, stamp
`20261029-172352`, and the same gate reads that name the same way. Both elisions are of the *name*
only — the stamp, the run ids and the line numbers are exact.

## The matrix

`faithful` is the only verdict that may omit a justification; all six rows below carry one, so each
has a section with its arithmetic. Strata's side is quoted from the sibling tree; the fork's from
this branch unless the cell says otherwise.

| mechanism | Strata (value, file:line) | fork (value, file:line) | verdict | justification (short; full text in the row's section) | measured |
|---|---|---|---|---|---|
| **M1-budget** — VRAM on the expert tier instead of KV/embedding | `kv_resident 0` (default; deployed run passes 32768) `src/program/generate.cpp:311`, `prefill_chunk 0` `:431`; 8301 slots = **173 experts/layer** of 15.74 GiB, Strata's `server log:390,395` | `cache_size 262144` `bench/windows/state.json:10`, `cache_quant "4,4"` `:11`, `split 381` `:7` ⇒ **131 of 512 experts/layer** (the split is the CPU tail, so 512 − 381 stay on the card) | `absent` | three levers (cache_size, `sysmem_kv_cache`, split) are worth +23..27 experts/layer, i.e. 131 → 154..158 of 173; 80 % by residency needs ≈38 GB of expert weights on a 24 GiB card (§1) | `unmeasured` |
| **M2-churn** — the adaptive tier's cadence and source | `adapt_every 4` `generate.cpp:474`, `adapt_swaps 96` `:497`, `adapt_decay 0.7` `:475`, `slot_pool global` `:382` + `/home/dev/UnbiasedStrata/include/strata/core/expert_cache.hpp:221`, `swap_source ram-copy` `:4590` | `promote_budget 6` `block_sparse_mlp_cpu.py:161`, `ratio 1.5` `:162`, `decay 0.98` `:163`, `spares 4` `:1088`, `sweep_interval 128` `:1208`, `sweep_max 64` `:141`, `sweep_source checkpoint` `:1293` | `diverged` | 15.1 swaps/round (measured) against 0.20/step at the t27 operating point: **75×**, 31× per committed token — and the bytes come from RAM on both sides (the fork's promote path DMAs the pinned arena, `moe_cpu_host.py:1884-1918`; only the legacy sweep re-reads the checkpoint), so what diverges is the cadence and the pool, not the source (§2) | `t29-C1-pin-unset-20261103-221302` |
| **M3-miss-split** — the miss is split CPU/DMA | `pcie_frac -1.0` = the model's own, nominal **0.55** for native packs before the startup probe scales it `generate.cpp:470`, `:2051-2063` (read by hand: not an extracted key) | `stream_min_rows 32` `model/moe_cpu_host.py:141`; second gate `stream_t 8` `:139` (by hand) | `absent` | a 5-row window (the pending token + the 4 drafts appended to it) is 6.4× below the gate and 1.6× below the per-expert gate; the arm that tests it is `EXL3_MOE_STREAM_MIN_ROWS=2` + `EXL3_MOE_STREAM_T=2` (§3) | `unmeasured` |
| **M4-spec-yield** — what one verify pass buys | MTP ≤3 drafts + prompt lookup, cost-model chosen, `docs/upstream/DETAILS.md:1032-1035`; **2.5–3.5 tokens/window** `server log:425,444,453` | DFlash2 `num_draft_tokens 4`, `dynamic_draft` shortens only, `generator.py:1128-1146`; **2.2–2.4 tokens/window** from note 1 §2's live-log derivation, against **2.17–2.40 accepted drafts** = 3.17–3.40 tokens/round in `results/arms.csv` | `diverged` | a ceiling of 4 offered drafts in a 5-row window against a policy that picks a window per round; the fork is behind on note 1's reading and at parity or ahead on the harness's own column, so the size and the sign are unresolved (§4) | `unmeasured` |
| **M5-ple** — the n-gram table read on the decode path | row cache 1M rows (≈90 MiB), up to ~82 % of reads **on a frozen corpus**, 20–34 % recurrence `include/strata/ngram/ple_reader.hpp:13-16`; issue/collect split `:5-11`; un-hidden gather 2.1–2.6 ms/token `generate.cpp:393-395` | no row cache: `ext.ngram_gather_cpu` per gather `modules/row_table.py:282-312`; `PREFETCH_MIN_TOKENS 256` `:27` declined at `ngram_embedding.py:334`; `advise()` is `posix_fadvise`, a no-op on Windows `row_table.py:208-215` | `absent` | three defences absent; 16 × 90 B = 1440 B/token is the byte scale, Strata's own un-hidden 2.1–2.6 ms/token the ceiling (§5) | `unmeasured` |
| **M7-deploy** — deployment policy, not an engine property | one request at a time (`batch 0` `generate.cpp:456`; no `--slots`), `--kv-resident 32768` of 262144 `/home/dev/UnbiasedStrata/start-logged.ps1:54` | `max_batch_size 2` (`tabbyAPI/config.yml:217`, outside this tree), `sysmem_kv_cache 0` `:441`, whole cache in VRAM: `cache_size 262144` `bench/windows/state.json:10` | `diverged` | 2.82 GB of KV + indexer planes in VRAM against Strata's 0.23 GB resident (8× the cells, 2.59 GB of bytes); the one overlapping pair ran 49.8 T/s on the long stream and 8.1 on the short one against 50.5–61.3 T/s single (§6) | `unmeasured` |

M1 and M7 read the same three regime constants, so they are recorded twice on purpose: a drift in
`bench/windows/state.json` fails both. That file describes **production** — `split 381`,
`cache_size 262144`, `cache_quant "4,4"`, verified on the box 2026-11-02 per its own `_comment` — and
the t27/t29 arms in `results/passes.csv` ran exactly that regime (`-mcs 381 -cs 262144 -cq 4,4`), which
is why the matrix cites them: one value serves both, and M1's "131 of 512" is 512 − 381 (§9).

The `measured` column is what `MATRIX` records and therefore what `--check` prints, unchanged — the
five `unmeasured` cells are the checker's verdict, not an omission, and §7 cashes each one: the run
ids, the number they produced, and whether nothing measured the row or something measured a different
half of it.

`--check` on this tree prints, one line per row:

```
PASS M1-budget: 19 comparison(s) - strata.kv_resident_tokens=0 (src/program/generate.cpp:311), strata.prefill_chunk=0 (src/program/generate.cpp:431), fork.cache_size=262144 (bench/windows/state.json:10) +2 more; derived experts_per_layer_repo=131, strata_experts_per_layer=173, experts_short=42, trunk_gib=17.1, freed_gb_65536=2.11, slots_65536=1106, gain_65536=23, experts_after_65536=154, short_65536=19, freed_gb_32768=2.47, slots_32768=1291, gain_32768=27, experts_after_32768=158, short_32768=15; absent, measured unmeasured
PASS M2-churn: 19 comparison(s) - strata.adapt_every=4 (src/program/generate.cpp:474), strata.adapt_swaps=96 (src/program/generate.cpp:497), strata.adapt_decay=0.7 (src/program/generate.cpp:475) +9 more; derived swaps_per_round_cap=24, swaps_per_round_measured=15.1, sweep_swaps_per_step=0.5, promotions_per_step_t27=0.201, churn_ratio=75, promotions_per_step_t29=2.42, promotion_raise_ratio=12; diverged, measured t29-C1-pin-unset-20261103-221302
PASS M3-miss-split: 3 comparison(s) - fork.stream_min_rows=32 (exllamav3/model/moe_cpu_host.py:141); derived rows_short=27, gate_factor=6.4; absent, measured unmeasured
PASS M4-spec-yield: 3 comparison(s) - no extracted constant on either side (see justification); derived tokens_per_round_low=3.17, tokens_per_round_high=3.4, unused_draft_rows=1.6; diverged, measured unmeasured
PASS M5-ple: 2 comparison(s) - no extracted constant on either side (see justification); derived bytes_per_token=1440, cache_mib=90; absent, measured unmeasured
PASS M7-deploy: 10 comparison(s) - fork.cache_size=262144 (bench/windows/state.json:10), fork.cache_mode='4,4' (bench/windows/state.json:11), fork.split=381 (bench/windows/state.json:7); derived kv_ratio=8, batch_delta=2, fork_kv_gb=2.82, strata_kv_whole_gb=1.81, strata_kv_whole_gib=1.69, strata_kv_resident_gb=0.23, kv_vram_delta_gb=2.59; diverged, measured unmeasured
6/6 rows PASS
```

## 1. M1-budget — where the VRAM goes, and the ≈38 GB wall

**`MATRIX["M1-budget"]["justification"]`** (reflowed for the page; typography and spacing normalised —
`—`, `×`, `≈`, `≤`, `§`; the code is authoritative): the fork's 131 experts/layer is a static prefix of
one uniform per-layer budget at the served split 381 (the split is the CPU tail count,
`model_init.py:68`, and the CPU tail is 381, so 131 experts per layer stay on the GPU); Strata's
173/layer is 8301 global slots over 48 layers (15.74 GiB)
that follow routing. Three levers would move the fork: `cache_size` 262144 → 32768/65536,
`memory.sysmem_kv_cache` ~1800 MB, then `cpu_moe_split_experts` below 381. The KV bytes, from the
checkpoint's config and the engine's cache shapes: 2 KV heads × 256 head_dim = 512 values/token/layer,
at "4,4" 16 groups × 8 bits × 4 B + 32 B of fp16 scales = 576 B, plus 320 B of fp16 indexer planes that
do not scale with the width (`cache/qsa.py:16-19`) = 896 B/token/layer over 12 attention layers, so
2.82 GB at 262144 cells. Dropping to 65536 or 32768 cells frees 2.11 or 2.47 GB, which at 1.91 MB per
expert (47.03 GB / 512 / 48) is 1106 or 1291 slots = +23..27 experts/layer, so 131 → 154..158: still
15-19 short of 173. Research note 1's M1 verdict ("~1.9-2.4 GB VRAM ~ 500-650 more expert slots ~
+10-13 experts per layer") does not divide by the 1.91 MB per expert that note states in its own
section M9: 1.9-2.4 GB / 1.91 MB is 1000-1250 slots, and the conversion above is the corrected one.
Reaching 80 % by residency instead needs 410 of 512 experts/layer = 410 × 48 × 1.91 MB = 37.6 GB ≈
38 GB of VRAM, against a 24 GiB card whose dense trunk already holds 17 551 MiB = 17.1 GiB (`bench/size_check.py`,
t29 artefact §1). And residency alone does not buy Strata's coverage: a static profile at 8000 slots
covers 28.4 % of reads (note 2 §2.2), the rest comes from the tier following routing, i.e. from M2.
**Absent**, not diverged.

The arithmetic, spelled out:

| step | number | source |
|---|---|---|
| experts per layer **on the GPU** at the served regime (the CPU tail keeps 381) | 512 − 381 = **131** | `bench/windows/state.json:7`, and every t27/t29 `args` (`-mcs 381`) in `results/passes.csv` |
| expert bytes per expert | ≈1.91 MB (47.03 GB / 512 / 48) | note 1 §M1, from `quantization_config.json` |
| the fork's experts on the card | 131 × 48 × 1.91 MB = **12.0 GB** | derived |
| Strata's expert tier | 8301 slots / 48 layers = **173/layer**, 15.74 GiB | Strata's `logs/server-…err.log:390,395` |
| KV bytes per token per layer, fork at `-cq 4,4` | 16 groups × (4+4) bits × 4 B = 512 B + 2 × 16 fp16 scales × 2 B = 64 B + 320 B fp16 indexer planes = **896 B** | `cache/quant.py:36-47`, `cache/qsa.py:16-31`, config (`2` KV heads × `256` dims, `indexer_head_dim 128`, ratio `4`) |
| … per token, 12 full-attention layers | 896 × 12 = **10 752 B** | `full_attention_interval: 4` of 48 layers |
| the fork's KV at 262144 cells | 262144 × 10 752 B = **2.82 GB** (2688 MiB) | `bench/size_check.py`, t29 artefact §1 |
| lever 1+2: cache 262144 → 65536 / 32768 cells | 2.82 − 0.70 = **2.11 GB** / 2.82 − 0.35 = **2.47 GB** freed | derived |
| … at 1.91 MB per expert | 2.11 GB / 1.91 MB = **1106 slots**; 2.47 GB / 1.91 MB = **1291 slots** | derived |
| … in experts per layer | 1106/48 = **+23**; 1291/48 = **+27** → 131 → **154..158** | derived |
| still short of Strata | 173 − 154 = **19**; 173 − 158 = **15** experts/layer | derived |
| reaching 80 % by residency | 410 × 48 × 1.91 MB = **37.6 GB ≈ 38 GB** | derived |
| VRAM the card has | 24 564 MiB, of which the trunk alone is 17 551 MiB = **17.1 GiB** | `bench/size_check.py` output, t29 artefact §1 |

`-mcs` is the **CPU** tail (`moe_cpu_split`: "run the TAIL N routed experts of every eligible
block-sparse MoE layer on the CPU", `model_init.py:68` of the venv's engine), so every fork-side
"experts per layer" figure above is the card's: the worker keeps `-mcs` experts and the GPU holds
512 − `-mcs`.

Note 1's own M1 verdict says the same levers are worth "≈500-650 more expert slots ≈ +10-13 experts per
layer" (`/home/dev/research/strata-vs-exllamav3-2026-11-02.md:156-157`, repeated at `:492`), and that
pair is the one this task's review rounds on: at the 1.91 MB per expert that note states itself
(`:458`), 1.9-2.4 GB is ~1000-1250 slots, not 500-650, so the note divided 1.9-2.4 GB by roughly 3.7 MB
per slot. The corrected gain is larger — +23..27, not +10..13 — and the conclusion it feeds is
*narrower* in effect, not wider: 154..158 is still short of 173, and the residency count was never the
mechanism (28.4 % of reads from an 8000-slot static profile, note 2 §2.2).

So the levers are real but bounded: they close 23-27 of the 42-expert gap — about 55-64 %, not the
"third" an earlier draft of this section claimed — and the rest is not a setting. Strata reaches
79–95 % coverage *from* a 34 %-of-experts tier because the tier follows the conversation, which is
M2's mechanism, and the fork's static prefix has no such follower. Moving the split the other way is
not a loss either: `-mcs` counts the **CPU** tail (`model_init.py:68`), so `381 → 376` — the earlier
3.18 boundary the file's own `notes` still record — keeps 512 − 376 = **136** experts per layer on the
card against 131, 3.8 % *more* residency, and hands the worker 5 × 48 × 1.91 MB ≈ 0.44 GB of arena back
in exchange for the same bytes of VRAM. That is why a split change is not a lever of its own in §8: it
is where the cache lever's freed bytes go.

## 2. M2-churn — the unit trap: 15 swaps/round against 0.20/step

**`MATRIX["M2-churn"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): units and rate, not names: Strata spends up to
`adapt_swaps` 96 on ONE adaptation every `adapt_every` 4 decode rounds over ONE global pool
(`slot_pool global`), fed from the resident RAM copy (`swap_source ram-copy`, 0 file reads), so
96/4 = 24 swaps/round is the cap and a measured adaptation did 60.5 → **15.1/round**. The fork at the
t27 operating point promoted **710** times over the 3528-step perf ladder = **0.20/step** (`spares 4`,
`PROMOTE_BUDGET 6/step`, ratio 1.5, decay 0.98), and its legacy sweep is 64 swaps per 128 steps =
0.5/step with two checkpoint reads per swap (`sweep_source checkpoint`). 15.1/0.201 = **75×**, the
review's 75×. Per committed token it is **31×** (Strata 15.1/2.45 = 6.2; a perf step is one token
because `perf.py` registers no draft). The source of the bytes is faithful, and it is the cadence and
the pool that diverge: both arms run promote-on-miss, and `promote_expert`
(`moe_cpu_host.py:1884-1918`) DMAs the expert out of the pinned arena, RAM to VRAM with no checkpoint
read — only the legacy sweep re-reads the checkpoint per swap (`stc.get_tensor`,
`block_sparse_mlp_cpu.py:1293`, `sweep_source checkpoint`). `t29-C1` shows the rate that same policy
reaches once the pool stops capping it: 8669 promotions / 3584 steps = **2.42/step**, **12×** the t27
realized rate (2.42/0.201 = 12.0, 8669/710 = 12.2; the spare count went 4 → 16 and `PROMOTE_BUDGET`
stayed 6/step, which neither arm reached), for +40.6 points of coverage (31.7 → 72.3 % pooled).

The two campaigns' numbers, as read — and the fork's divergence lives in the cadence and the pool,
not in where the bytes come from:

| quantity | Strata | fork, t27-B (pin 8 GiB, 4 spares) | fork, t29-C1 (pin unset, 16 spares) |
|---|---|---|---|
| cadence | 96 per adaptation ÷ 4 rounds = **24/round** | 6/step budget, **0.201/step realized** | 6/step budget, **2.42/step realized** — the budget did not move, the rate did |
| pool | **one global pool**, 8000–8301 slots over the whole cache | **per-layer spare slots**, 4 of them, filled by promote-on-miss | same path, **16 spares** and no pin budget, so the pool stops capping the policy |
| source of the bytes | RAM ↔ VRAM exchange, "no file reads" (`generate.cpp:4590`); log: "0 blob reads from the file" | **faithful**: `promote_expert` DMAs the expert from the page-locked arena (`moe_cpu_host.py:1884-1918`), RAM → VRAM, no checkpoint read | same path, same source (`moe_cpu_host.py:1884-1918`) |
| the only checkpoint reader | — | the legacy **sweep** (`_split_swap_experts`), 64 swaps/128 steps = 0.5/step, `stc.get_tensor` per swap (`block_sparse_mlp_cpu.py:1293`); it is what the t29 **A** arms (pin off) ran, and no t27-B or t29-C arm uses it | — |
| what it bought | 79–95 % coverage, 94.1 % in the sweep regime | none: coverage 28.5 % control against 27.7 % (and the column was frozen — T16 §7) | +40.6 points (72.3 vs 31.7), hook 2.318 ms/step |

So the M2 finding is *not* "the fork re-reads the checkpoint for every promotion" — it does not, in
either measured arm. It is that the fork's promotion rate (0.201/step with a 4-spare pool, 2.42/step
with 16) is 75× below Strata's 15.1 adaptations per round — 31× per committed token, the shared unit —
and that the pools have different shapes: Strata re-ranks one global 8000-slot pool, the fork shuffles
each layer's own spares.

Sources: `t27-B-pin8-f4-20261103-122212` (its last tier line: **710** promotions, 28 windows =
27 × 128 + 72 = 3528 steps; the pass's own `env` carries `EXL3_MOE_PINNED_ARENA=1`,
`EXL3_MOE_CPU_PROMOTE=4`), `/tmp/opencode/t29-pin-unset.md` §4-§5 (8669 promotions "at step 3584",
`t29-C1-pin-unset-20261103-221302`, whose `env` carries `EXL3_MOE_CPU_PROMOTE=16` and the same
`PINNED_ARENA=1`), `strata-vs-exllama-research.md` §2.2 (the 60.5-swap round).

The unit that made the campaign's first churn estimate 75× too small is this: Strata's 96 is a
per-round budget over a single global pool; the fork's 64 is a total across 48 layers every 128 steps,
i.e. 0.5/step, and the prom path's 6/step is a *different* unit again (a step is one token, a round is
~2.45 tokens). Normalising to the token — the only unit both engines share — narrows the gap from 75×
to 31×, and the honest number to quote is the pair, not either one alone.

## 3. M3-miss-split — the env delta, and the second gate

**`MATRIX["M3-miss-split"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): the miss goes somewhere else. Strata's
`pcie_frac` default is the model's own (`-1.0` at `generate.cpp:470`), whose base for the native packs
is 0.55 — the **nominal pre-probe** share, not the deployed one: the startup probe scales it by the
measured H2D bandwidth below 20 GB/s and prints the result (`generate.cpp:2051-2063`), and no log on
this box is cited for that line — so a share of a verify round's distinct missed experts, nominally
0.55, is DMA'd and computed on the GPU. The fork's streamed path exists (`moe_cpu_host.py:1968-1998`)
but is gated twice: `rows < stream_min_rows 32` returns before it (`:1981`), and an expert needs
`stream_t 8` assigned tokens (`:139`, read by hand — not an extracted key) — so a 5-row window (1
pending token + the 4 drafts appended to it, `job.py:391-394`) reaches neither and **100 % of decode
misses go to the worker process**. The delta that would test the fork's half is env-only,
`EXL3_MOE_STREAM_MIN_ROWS=2` plus `EXL3_MOE_STREAM_T=2`, with no regime flag; nothing on this box has
run it.

The arithmetic:

- a verify window is at most `num_draft_tokens` + 1 = **5 rows** wide on the fork's deployed regime:
  `generator/job.py:391-394` concatenates the four drafts **to the pending token**, so the window is
  the drafts *plus* the row that was already there. The gate is 32: **6.4× short**, 27 rows short
  (`derived.rows_short`).
- the second gate is per expert: an expert qualifies for staging at `stream_t = 8` assigned tokens
  (`moe_cpu_host.py:139`, applied at `:2009`), and an explicit `EXL3_MOE_STREAM_T` bypasses the
  bandwidth calibration that would otherwise raise it (`:1726-1727`); 5 rows can route an expert at
  most 5 times, so **no decode expert can ever qualify** even after the row gate is dropped.
- therefore the arm that tests the fork's half is **two variables by design**, not an untidy command:
  `EXL3_MOE_STREAM_MIN_ROWS=2` opens the path and `EXL3_MOE_STREAM_T=2` is what makes the second gate
  reachable at a 5-row window. Dropping either one is a null arm — with `stream_t` still 8 no expert
  qualifies, and without the row gate the call never reaches the per-expert test — so the pair is the
  smallest delta that can move the mechanism at all. It remains an experiment, not a port: the fork's
  own calibration says 4 rows lost to 8 at prefill sizes (`:1078-1118`), which is a prefill
  measurement and does not transfer to a 5-row decode window either way.

The exact command, if the operator consents (it stops the served model; restore per T16 §8):

```
-Tag t31-M3-rows2 -Runner perf -MaxLength 131072 -Env "EXL3_MOE_TIER_STATS=1,EXL3_MOE_STREAM_MIN_ROWS=2,EXL3_MOE_STREAM_T=2"
```

## 4. M4-spec-yield — the window numbers, and the two notes that disagree

**`MATRIX["M4-spec-yield"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): window arithmetic, in two units that are not
yet reconciled: tokens/window is 1 + the sum of per-position acceptance, so the deployed DFlash2 cap
`num_draft_tokens 4` truncates it (`dynamic_draft` only ever shortening, `generator.py:1128-1146`) and
the window itself is **5 rows wide** — the pending token plus the 4 drafts appended to it,
`job.py:391-394`.
Strata drafts ≤3 with the model's own MTP head plus a prompt-lookup drafter picked per round by a cost
model (`docs/upstream/DETAILS.md:1032-1035`) and its log reads **2.5–3.5 tokens/window**. The fork
reads two ways and they do not agree, so no ratio is recorded: note 1 §2 derives **2.2–2.4
tokens/window** from the served deployment's own log (2.17 / 2.41 / 2.33 / 2.56), while
`results/arms.csv`'s `acc` column for the 4-draft arms reads **2.17–2.40 accepted drafts** per round
(`t7-dflash2-w4` 2.40 of 4.00, `t16-q4-ref2` 2.35, `t19-qdrafter` 2.17), i.e. 1 + `acc` = **3.17–3.40
tokens per round**. Against note 1's reading the fork is behind Strata's 2.5–3.5; against the
harness's own column it is at parity or ahead, and nothing in either tree says which reading shares
Strata's unit. No arm of either campaign carries a draft (`perf.py` registers none; draft arguments
belong to `-Runner spec_decode`), so the row is `unmeasured`: `diverged` on the mechanism (a hard
ceiling of 4 against a policy that chooses a window per round), its performance advantage
**unresolved**.

The numbers, and what they do not settle. Note 1's live-log derivation and `arms.csv`'s `acc` column
are kept in separate rows because they measure different things — a served deployment's log against
the harness's arm rows — and converting one into the other is exactly the step that was wrong here
before:

| | Strata | fork, deployed DFlash2 |
|---|---|---|
| window | MTP ≤3 + prompt lookup up to 5, chosen per round by measured tokens/ms | block-diffusion drafter, `num_draft_tokens 4`, `dynamic_draft: true` (shortens only), so 5 rows: 4 drafts + the pending token (`job.py:391-394`) |
| tokens/window, **note 1's live-log derivation** | 2.5–3.5 (log: 2.9 / 2.46 / 2.56 / 3.48 / 2.65 over five requests) | 2.2–2.4 (note 1 §2: 2.17 / 2.41 / 2.33 / 2.56 — computed as 1 + accepted/windows from the deployment's log) |
| **`results/arms.csv`'s `acc`**, accepted **drafts** per round | — (no harness arm; Strata is not run by this harness) | 2.17–2.40 accepted of 4.00 offered (`t7-dflash2-w4` 2.40, `t16-q4-ref2` 2.35, `t19-qdrafter` 2.17) |
| tokens per round from that column | — | **3.17–3.40** = 1 + `acc`; every round also emits the token that was pending |
| the window is | kept full by a policy that prices both drafters | truncated by its own ceiling: 1.6 of the 4 offered drafts go unused on average |

Both notes agree on the physics and disagree on the size: note 1 derives 2.5–3.5 against 2.2–2.4 from
five and four requests; note 2, reading a different session, gets 2.42–2.47 on both sides and calls it
a wash. That disagreement is recorded here rather than resolved, and one thing an earlier draft of this
section got wrong is corrected in §7.3: **the harness has measured this row** — `results/arms.csv`
holds DFlash2 acceptance at 2.40 of 4.00 drafted (`t7-dflash2-w4-20261030-071928`) and 2.94 of 6.00
(`t7-dflash2-w6-20261030-072942`) — so the fork's acceptance is a harness column, not only the served
deployment's log. What *no* arm of either campaign carries is a draft, because `-Runner perf` registers
no draft flags and draft arguments belong to `-Runner spec_decode`. And the column is **accepted
drafts**, not tokens: reading 2.17–2.40 as tokens/window is the error this section carried until the
review, because the round that accepts 2.40 drafts emits 3.40 tokens (the closing section, item 9).
A verdict of `diverged` therefore rests on the mechanism (a hard ceiling of 4 against a policy that
chooses a window per round) **only**; the ratio's size *and its sign* are unmeasured, and on the
harness's own column the fork may already be at parity or ahead.

## 5. M5-ple — the read the fork does not cache

**`MATRIX["M5-ple"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): absent, and cheap to bound: Strata hides the 16
unbuffered row reads of a decode token (90 B/row, 26.8 GiB table never in RAM) behind the embedding and
layer 0 with an issue/collect split, and keeps a bounded 1M-row (90 MiB) row cache that its own source
says **would serve up to 82 % of reads on a frozen corpus**, with 20–34 % recurrence inside one prompt
(`ple_reader.hpp:13-16`) — a measurement on that corpus, not on the served traffic. The
fork has no such row cache: every gather goes to `ext.ngram_gather_cpu` over a
`FILE_FLAG_NO_BUFFERING` handle (`row_table.py:282-312`, opened at `loader/safetensors.py:247-248`), so
a repeated row is read from disk again. Its two cheaper substitutes both miss decode: `prefetch()`
declines decode-sized inputs (`PREFETCH_MIN_TOKENS` 256 positions, `row_table.py:27`, checked at
`ngram_embedding.py:334`), and the read-ahead hint `advise()` (`row_table.py:208-215`) is
`posix_fadvise`, i.e. a no-op on the Windows box this fork serves. 16 × 90 B = 1440 B per token is the
byte scale; Strata's own un-hidden gather is 2.1–2.6 ms/token (`generate.cpp:393-395`), which is the
ceiling on a port, not a promise.

The 90 B/row and 26.8 GiB in that quoted justification are **Strata's** table — IQ4_NL, 90 B/row, the
size its own reader states (`ple_reader.hpp:3`) — and not this fork's: the fork's table is ≈8.1 bpw
(~162 B/row, 51.84 GB on disk, 49 439 MiB summed from the checkpoint's own shapes by
`bench/size_check.py`). §7.3 records the correction; the conclusion does not move, because either
figure is ~0.3 % of the expert weights a token already reads.

Arithmetic: 2^20 rows × 90 B = 90 MiB per table would serve up to 82 % of reads **on Strata's frozen
corpus** — the qualifier is its own (`ple_reader.hpp:15-16`, "Measured on the frozen corpus
(bench/results/2026-09-23-ngram-io)"), and it is a measurement of that corpus, not of the served
traffic; the fork's per-token traffic
is 16 of its own ~162 B rows, ~2.6 KB, against the 0.92 GB a token already reads in expert weights —
the byte traffic is negligible and **the stall is the point**: Strata measured its own un-hidden gather at
2.1–2.6 ms/token against a 9–11 ms token. The row is `absent` rather than `diverged` because the fork
has the streaming path and none of the three defences (row cache, issue/collect split, keep-alive), so
the port is ~50 lines in `row_table.py` rather than a translation.

One correction to the research notes, found while writing this row: note 1 cites `_gather_rows` at
`ngram_embedding.py:445-466`; in this branch the gather machinery has moved to `modules/row_table.py`
(`_gather_rows` at `:282`), and `ngram_embedding.py` is 353 lines. The behaviour described is the same;
the line is not. The `advise()` page-cache hint is newer than the note and is the one partial answer in
the tree — it is why the row above says "posix_fadvise, a no-op on Windows" rather than "no read-ahead
at all".

## 6. M7-deploy — batch and KV residency

**`MATRIX["M7-deploy"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): deployment, not engine: Strata serves one
request at a time (`batch 0` default, `src/program/generate.cpp:456`, and
`/home/dev/UnbiasedStrata/start-logged.ps1` passes no `--slots`) with 32768 of 262144 KV cells resident
(`/home/dev/UnbiasedStrata/start-logged.ps1:54`); the fork's deployed regime serves `max_batch_size 2`
(`tabbyAPI/config.yml:217`, outside this tree) with the whole cache in VRAM (`sysmem_kv_cache 0`,
`config.yml:441`) — 262144 cells at "4,4", the regime `bench/windows/state.json` records. Per
token per layer the two engines agree on the K/V: 576 B (2 heads × 256 dims at 4 bits + fp16 scales)
= 6912 B/token over 12 layers, 1.81 GB at 262144 cells, which is exactly the log's "K/V in 1.69 GiB of
pinned RAM" (`server log:378`). The
fork adds 320 B/token/layer of fp16 indexer planes (896 B/token/layer) and keeps all of it in VRAM:
2.82 GB. Strata keeps 32768 cells
resident (0.23 GB) and the other 229376 cells, 1.59 GB, in pinned RAM. So the VRAM delta at one context
is 2.82 − 0.23 = **2.59 GB** — the same class of quantity the M1 lever frees (fork VRAM KV against
host KV), though not the same number: that lever frees 2.11–2.47 GB by target cell
count. Concurrency is anti-scaled: the log's one overlapping pair ran 49.8 T/s on
the 19.5k-token stream and 8.1 T/s on the 551-token stream that shared the window, against 50.5–61.3
T/s for the three single-stream requests (note 1 §2). No arm of either campaign varied batch or KV
residency, so the divergence is real and `unmeasured`.

Arithmetic, in bytes rather than in a range:

| quantity | fork (deployed, `-cq 4,4`, `-cs 262144`) | Strata (`--kv q4_0`, `--kv-resident 32768`) |
|---|---|---|
| K/V bytes per token per layer | 512 B codes + 64 B fp16 scales = **576 B** | **576 B** (`kv_q4.hpp:3-6`, K+V of both heads) |
| indexer planes per token per layer | **320 B**, fp16 at any width (`cache/qsa.py:16-19`) | not in this figure |
| bytes per token, 12 full-attention layers | 896 × 12 = **10 752 B** | 576 × 12 = **6912 B** |
| the whole table at 262144 cells | 262144 × 10 752 = **2.82 GB** | 262144 × 6912 = **1.81 GB = 1.69 GiB** (the log's line, `:378`) |
| where it lives | **all in VRAM** (`sysmem_kv_cache 0`) | 32768 cells in VRAM = **0.23 GB**; 229376 cells = **1.59 GB** in pinned RAM |
| VRAM at this context | **2.82 GB** | **0.23 GB** |
| delta, fork − Strata | **2.82 − 0.23 = 2.59 GB** | — |

So the two engines do not differ in bytes per token — both are 576 B of 4-bit K/V per token per layer,
which is why Strata's 1.69 GiB line and the fork's `-cq 4,4` agree — they differ in *residency*. The
fork's 2.82 GB (planes included) against Strata's 0.23 GB is a **2.59 GB** VRAM delta, the same class
of quantity the M1 lever frees when the fork shrinks its own cache to 32768 or 65536 cells (2.47 or
2.11 GB) — VRAM KV against host KV, not the same number; the 1.9–2.4 GB an earlier draft quoted here
was neither side's figure. Concurrency, one
reading: the log's single overlapping pair ran 49.8 T/s on the 19 497-token stream (#6) and 8.1 T/s on
the 551-token stream (#5) that shared its window, against 50.5 / 61.3 / 57.2 T/s — a 50.5–61.3 T/s
band — for the three single-stream requests (cold, fully cached, warm) — so two sessions cost more than
they yield on this box, a deployment decision rather than an engine property, and the deployed
`max_batch_size: 2` was never A/B'd by either campaign.

## 7. The evidence, row by row

The five `unmeasured` cells in the matrix are not the same kind of blank, and this section says which
kind each one is. Every figure below was recomputed on 2026-11-04 from the deployed harness's own
files — `results/passes.csv`, `results/pp.csv`, `results/tg.csv`, `results/arms.csv` and
`logs/perf-<run id>.log` — not copied from a report or a note. Nothing was measured for it: the only
command executed was `bench/size_check.py` (§8), which reads the checkpoint's tensor headers, and no
GPU work and no pass was started.
The t29 arms used here are the **second round only**: the first round ended `status=fail` in all four
arms before a single tier line printed (t29 §2), so it is evidence of an instrument defect, not of a
mechanism, and its run ids appear nowhere in this report.

| row | `measured` (matrix) | what the runs say | what it settles | what is still open |
|---|---|---|---|---|
| **M1-budget** | `unmeasured` | the *pin* half: `t29-C1-pin-unset-20261103-221302` **72.347 %** pooled coverage against `t29-A1-pin-off-20261103-220854` **31.760 %** (+40.59 points; the repeats agree: C2 72.331, A2 31.735), with `arena 30720 MiB pinned in 30 of 34 chunk(s), driver limit, 42 of 48 layer(s) promotable (0..41)` | expert bytes in page-locked host RAM, with promote-on-miss filling the spare GPU slots, is worth +40.59 points of coverage at one context — the residency half of the row, measured | the row's own three levers moved in **no** arm: all 17 t27/t29 passes carry `-mcs 381`, and the 16 `perf` arms carry `-cs 262144` (the one `spec_decode` arm sizes its cache from `draft.spec_decode_cache_size`), so 131 of 512 experts/layer is the only residency point ever run (§8 prices the arm) |
| **M2-churn** | `t29-C1-pin-unset-20261103-221302` | **8 669** promotions by step 3 584 = **2.42/step** (C2: 8 646, 0.27 % below); promote-on-miss hook **2.318 ms/step** mean over 28 windows (1.39 to 4.23) against 0.201 promotions/step and ~1.05 ms/step at the 8 GiB arm. Both arms are promote-on-miss DMA out of the pinned arena (`moe_cpu_host.py:1884-1918`), so neither reads the checkpoint for a promotion | the rate and its price are both measured in one session: **12× the t27 realized promotion rate** (the 6/step `PROMOTE_BUDGET` did not move — neither arm reached it), 2.3 ms of a ~25.9 ms step, and the source of the bytes faithful | whether the traffic pays: TG at 130816 is **+1.13 %** against a 4.6 % repeat spread — unresolved, not absent |
| **M3-miss-split** | `unmeasured` | nothing: no row of `passes.csv` (186 rows, every campaign) sets `EXL3_MOE_STREAM_MIN_ROWS` or `EXL3_MOE_STREAM_T` in its `env` | the arithmetic (a 5-row verify window against a 32-row gate, and a per-expert gate at 8) stands on the sources alone; nothing contradicts it and nothing supports it | the arm: one `-Env` string, `t31-M3-rows2` (§8). **`unmeasured` — needs an arm** |
| **M4-spec-yield** | `unmeasured` | the campaigns carry no draft — every t27/t29 `perf` arm's `args` ends `-sd`, and `perf.py` registers no draft flag, while the campaign's only draft arm is the failed `t27-A-base` — **but the harness's `results/arms.csv` does hold acceptance**: `t7-dflash2-w4-20261030-071928` **2.40 accepted drafts of 4.00 drafted** over 8 rounds, `t7-dflash2-w6-20261030-072942` **2.94 of 6.00**, and two later 4-draft repeats whose `-dm` is confirmed DFlash2 in `passes.csv` — `t16-q4-ref2-20261102-172722` **2.35 of 4.00** (`-dm …-DFlash2`) and `t19-qdrafter-381-20261102-194803` **2.17 of 4.00** (`-dm …-DFlash2-exl3`, the exl3-quantized drafter, which is why its tag is `qdrafter`) | the fork's acceptance at 4 offered drafts is 2.17 to 2.40 **accepted drafts per round**, i.e. **3.17 to 3.40 tokens per round** (1 + `acc`) — a harness column, and a different measurement from note 1's log-derived 2.2–2.4 tokens/window | the comparison: `acc` is accepted *drafts* (`spec_decode.py:142-151`), so the two fork-side readings (note 1's log derivation and this column) measure different things and disagree by about a token. Which — if either — shares Strata's unit is open, and a window-4 arm at the current regime is what settles it (§8) |
| **M5-ple** | `unmeasured` | one of the three defences is measured and it is the one that fails: the RAM-resident table (`-ngr` / `EXL3_NGRAM_STREAM=0`) breached the RAM floor 27 s in with 637 MB free (tag `t12-ngram-ram`, T12 §2); the PLE gather knob that does exist, `EXL3_PLE_SUB_CHUNK`, was A/B'd in the same campaign — `t12-ple1024-20261030-105757` (default) against `t12-ple512-20261030-104213` and `t12-ple0-20261030-103851`, prefill only, inside a 2 % band | two of Strata's three defences are either absent from this engine or not a lever at decode: the RAM copy does not fit, and the sub-chunk is a prefill knob | the row cache itself: no arm, because no such object exists in this fork (`modules/row_table.py:282-312` goes straight to `ext.ngram_gather_cpu`). The price is code, not a flag (§8) |
| **M7-deploy** | `unmeasured` | no arm varied batch or KV residency: every t27/t29 `args` carries `-ambs 2`, a regime flag `-Extra` may not restate, and `sysmem_kv_cache 0` is a tabbyAPI setting no pass reads. The only concurrency measurement on the box is Strata's own (`server log:378` ff.): 49.8 T/s on the 19 497-token stream against 8.1 on the 551-token one sharing the window, vs 50.5 / 61.3 / 57.2 single, a 50.5–61.3 T/s band | that the divergence is real (2.82 GB against 0.23 GB of KV in VRAM) and that its *performance* claim is not this harness's to make | batch and residency on the fork: a served-config edit or a second instance, not an arm — so this row stays `unmeasured` by construction (§6, §8) |

### 7.1 The four arms, recomputed

`results/passes.csv` + `results/pp.csv` + `results/tg.csv` + the pass's own log, second round, one
session, interleaved A1, C1, C2, A2:

| | A1 pin off | C1 pin unset | C2 pin unset | A2 pin off |
|---|---|---|---|---|
| run id | `t29-A1-pin-off-20261103-220854` | `t29-C1-pin-unset-20261103-221302` | `t29-C2-pin-unset-20261103-221642` | `t29-A2-pin-off-20261103-222022` |
| `status` / `breach` | ok / 0 | ok / 0 | ok / 0 | ok / 0 |
| coverage windows | 28 | 28 | 28 | 28 |
| **coverage, pooled** | **31.760 %** | **72.347 %** | **72.331 %** | **31.735 %** |
| mean of the 28 windows | 30.989 % | 70.200 % | 70.186 % | 30.975 % |
| promotable layers | pin off | 42 of 48 (0..41) | 42 of 48 (0..41) | pin off |
| pin state | pinned arena off | 30 of 34 chunks, 30 720 MiB, driver limit | same | pinned arena off |
| promotions / demotions | — | 8 669 / 8 669 | 8 646 / 8 646 | — |
| promote-on-miss hook | — | 2.318 ms/step (1.39 to 4.23) | 2.304 ms/step (1.37 to 4.24) | — |
| PP 4096, tok/s | 1 853.55 | 2 084.06 | 2 306.32 | 1 940.60 |
| PP 131072, tok/s | 1 907.20 | 2 245.09 | 2 246.17 | 1 899.26 |
| TG 130816, tok/s | 36.17 | 38.65 | 38.04 | 39.66 |

The pooled figure is `1 − Σ CPU evaluations / Σ evaluations` over the pass's 28 `expert tier:` lines
(A1: 2 206 714 of 3 233 760); the mean is the arithmetic mean of the 28 printed percentages.

The two deltas the row set quotes, recomputed from the pairs rather than read: **PP 131072 +17.99 %**
(C mean 2 245.63 against A mean 1 903.23; the repeats straddle their own mean by 0.02 % on the C side
and 0.21 % on A), **PP 4096 +15.71 %** (2 195.19 against 1 897.07, the noisier band: C's repeats there
straddle by 5.06 %, A's by 2.29 %), and **TG 130816 +1.13 %** (38.34 against 37.91). The decode number
is smaller than the spread between A's own two repeats of identical code — 36.17 and 39.66, i.e.
4.60 % either side of their mean — so it is a null, not a gain: +40.6 points of coverage bought ~18 %
of prefill and no provable decode.

### 7.2 The instrument's price: `unmeasured`

The tier instrument produced every coverage figure above, and its own cost is **`instrument cost:
unmeasured` (all arms carried it)**. The evidence for that is one query over `results/passes.csv`:
`EXL3_MOE_TIER_STATS=1` appears in the `env` of all 5 `t27` rows and all 12 `t29` rows, and no row of
the file — 186 rows, `armA` through `t29` — carries `EXL3_MOE_TIER_STATS=0`. There is no arm to
difference against, so the price is not small-but-known, it is unknown; and the 2.318 ms/step above is
the promote-on-miss hook's own counter, which is a different cost and must not be read as the
instrument's. It is not estimated here: an estimate would be a number no arm supports, and the next-arm
list (§10) carries the A/B that would measure it.

### 7.3 Three corrections this table forced

**§4's "never measured by this harness" is wrong, and its own range comes from the harness.**
`results/arms.csv` holds DFlash2 acceptance rows from the T7 campaign: 2.40 accepted drafts of 4.00
drafted (`t7-dflash2-w4-20261030-071928`), 2.94 of 6.00 (`t7-dflash2-w6-20261030-072942`), and the
4-draft point repeats at 2.35 (`t16-q4-ref2-20261102-172722`) and 2.17
(`t19-qdrafter-381-20261102-194803`, a draft repeat under the `qdrafter` tag) in later campaigns. Both
`-dm` values are checked against `passes.csv` rather than assumed:
`…\Qwen3.8-Flash-Next-DFlash2` for t16 and `…-DFlash2-exl3` for t19, the exl3-quantized drafter. So the
fork's spec-yield *has* been measured by this harness; what no t27/t29 arm carried is a draft. The
verdict does not move — the row still diverges on the mechanism, a hard ceiling of 4 against a policy
that chooses a window per round — but the sentence that overstated the gap is corrected, and the
harness's column now has a citation **and its unit**: these are accepted *drafts*, so the tokens the
round emits are 1 + `acc` = 3.17–3.40, not 2.2–2.4. That is why §4 keeps this column apart from note
1's log-derived 2.2–2.4 tokens/window instead of equating them, and why the row's advantage is
recorded as unresolved rather than as a gap.

**§5's 90 B and 26.8 GiB are Strata's table, not this fork's.** Note 1 puts Strata's table at
`IQ4_NL`, 90 B/row, 26.8–28.8 GB (`/home/dev/research/strata-vs-exllamav3-2026-11-02.md:65`), and the
fork's at ≈8.1 bpw — 51.84 GB over 16 heads × 20 M rows × 160 dims, i.e. ~162 B/row, and
`bench/size_check.py` sums the checkpoint's tensors to 49 439 MiB. So a decode token's 16 rows are
~2.6 KB on the fork's side, not 1 440 B: the byte scale in §5 is Strata's row width compared against
the fork's row *count*. The conclusion survives intact — 2.6 KB against the 0.92 GB of expert weights
a token already reads is still ~0.3 % — but the figure belongs to the side it was measured on.

**§4's fork-side range mixed two units, and the verify window is one row wider than this report first
said.** `acc` in `results/arms.csv` is accepted *drafts* per round, so reading 2.17–2.40 as
tokens/window understated the harness's yield by about a token (1 + `acc` = 3.17–3.40); note 1's
live-log derivation is a different measurement, and §4 now keeps the two apart instead of equating
them. And `generator/job.py:391-394` appends the four drafts **to the pending token**, so a verify
window is 5 rows, not 4: M3's gate is 6.4× the window (27 rows short, `gate_factor 6.4`) and the
per-expert gate at `stream_t 8` is 1.6× it. Both corrections came from the review, both changed a
`derived` number in the matrix, and neither moves a verdict: M3 is still `absent` and M4 still
`diverged` on the mechanism — M4's *performance* advantage is what is now recorded as unresolved.

## 8. What a port would cost

For the four `absent` rows the port is one of three things, and the table names which: an environment
delta, N lines in a named file, or impossible with the reason. `M2-churn` and `M7-deploy` are
`diverged`, not `absent` — the fork has those mechanisms and has them differently (M2's is a rate to
raise, not a port; M7's is deployment policy, two lines of served config, and it is not an arm at all).
**No arm in this table has been run, and none may be without the operator's consent:** the box serves
live traffic, every one of these passes stops the served model, and restoring it afterwards is T16 §8.

| row | the port is | price | the arm, exactly | measured? |
|---|---|---|---|---|
| **M1-budget** | a regime delta: `-CacheSize 32768 -MaxLength 32768 -Split 354`. The split is not a separate decision — it is what the freed 2.47 GB buys (131 → 158 of 512 experts/layer), so a pass that shrinks the cache and keeps `-mcs 381` spends the VRAM on nothing. `sysmem_kv_cache` is **1 line** in the served config (`tabbyAPI/config.yml:441`, default `0`) and **no pass can measure it**: `perf.py` loads the model directly, so tabbyAPI is not in the loop | 3 harness flags + 1 served-config line | `-Tag t31-M1-cs32k -Runner perf -CacheSize 32768 -MaxLength 32768 -Split 354` | `unmeasured` |
| **M3-miss-split** | an env delta, and it is **two variables by design**: `EXL3_MOE_STREAM_MIN_ROWS=2` **and** `EXL3_MOE_STREAM_T=2`. §3 is the arithmetic: the row gate alone is inert for a 5-row window because the per-expert gate (`stream_t`, `moe_cpu_host.py:2009`) is 8, and an explicit `EXL3_MOE_STREAM_T` is what bypasses the bandwidth calibration that would otherwise hold it up (`:1726-1727`), so either variable alone spends a pass to change nothing | one `-Env` string; 0 flags, 0 lines of code; needs the operator's consent like every arm here — the pass stops the served model | `-Tag t31-M3-rows2 -Runner perf -MaxLength 131072 -Env "EXL3_MOE_TIER_STATS=1,EXL3_MOE_STREAM_MIN_ROWS=2,EXL3_MOE_STREAM_T=2"` | `unmeasured` — needs an arm |
| **M4-spec-yield** | a runner switch, not a flag: `-Runner spec_decode` is the only runner the harness emits draft args for (`-dm`/`-ndt`/`-dds` come from `state.active.draft`; `-MaxLength` is stripped for it), and the deployed state's `knob_arms_disable_swap: true` makes the harness set `EXL3_MOE_CPU_SWAP=0` by itself — the guard that killed the campaign's only draft arm (`t27-A-base`, `status=fail` 42 s in, no ARM line) | 1 word, plus `-Extra "-dstats <file>"` if per-position records are wanted | `-Tag t31-M4-w4 -Runner spec_decode` | the acceptance column is in `arms.csv` already (2.17 to 2.40 accepted **drafts** of 4, i.e. 3.17–3.40 tokens per round); what is missing is the same measurement at the current regime, so the arm re-runs it |
| **M5-ple** | code: a bounded row cache in front of `_gather_rows` (`exllamav3/modules/row_table.py:282-312`), keyed on the resolved uids and capped in rows — **≈50 lines in that file**, no new dependency and no kernel change. Neither knob that exists is the row cache: `EXL3_PLE_SUB_CHUNK` is a prefill gather knob (T12, ~2 % bands) and `-ngr`/`EXL3_NGRAM_STREAM=0` was measured unavailable (RAM floor breached in 27 s, T12 §2) | ≈50 lines of engine code | none until the code exists; then the arm is T12's repeat-heavy corpus at `-MaxLength 131072` | `unmeasured` |

Three notes on the table:

- Every arm above needs a **same-session reference**: the deployed regime's own env, one pass, in the
  same session — a single arm compared against a cross-day absolute is not a control (AGENTS.md,
  "Measurement rules"). For M3 the reference is one pass with the two `EXL3_MOE_STREAM_*` variables
  absent; for M1, one at `-CacheSize 262144 -MaxLength 32768` so both arms share their bands.
- `M1`'s arm is already known to **fit**: `bench/size_check.py --mcs 354 --cs 32768 --cq 4,4
  --draft … --headroom-mb 512` reports `TOTAL 21634 MiB VRAM of 24564 -> 882 MiB free`, host arena
  31 010 MiB, `verdict FITS`. The lever is self-financing — the cache drops 2911 → 559 MiB and the
  trunk grows 17 551 → 19 916 MiB, i.e. the same bytes move from KV to experts — so the arm measures
  coverage and throughput, not whether the box can hold it.
- The M3 arm carries two variables where the plan's spelling named `EXL3_MOE_STREAM_MIN_ROWS=2`
  alone. That is the mechanism, not an untidy command: with `stream_t 8` still in force no decode
  expert can qualify (the per-expert gate, `moe_cpu_host.py:2009`), and the row gate alone never
  reaches that test, so a one-variable arm would be a null. §3 is the arithmetic; run the line as
  printed, and only with the operator's consent — like every arm here, it stops the served model.

## 9. The regime file, and who ran it

`results/passes.csv`'s `t27-*` and `t29-*` rows and the t29 dry-run's `ARGS` line say what the measured
arms actually ran:

```
-m … -cs 262144 -chunk_size 4096 -ambs 2 -cq 4,4 -max_length 131072 -mcs 381 -mct 16 -sd
```

`bench/windows/state.json` describes that same regime — **production**, verified on the box 2026-11-02
per the file's own `_comment`: `split 381` (`:7`, "the dynamic floor; the static harness floor at this
cache is 389"), `cache_size 262144` (`:10`), `cache_quant "4,4"` (`:11`), DFlash2 with
`num_draft_tokens 4` and `dynamic_draft true` (`:24-32`). So the t27/t29 arms ran **exactly** the regime
the file states — `-mcs 381 -cs 262144 -cq 4,4` — which is why the matrix cites them beside it, and the
`-max_length` difference is the one deliberate departure: the arms ran 131072, one value the deployment
sets at 262144.

| value | `bench/windows/state.json` | the measured arms | in the matrix as |
|---|---|---|---|
| split | 381 (`:7`) | 381 (`-mcs 381`) | `fork.split 381`; 131 = 512 − 381 is derived from it |
| cache_size | 262144 (`:10`) | 262144 (`-cs 262144`) | `fork.cache_size 262144` |
| cache_quant | `"4,4"` (`:11`) | `4,4` (`-cq 4,4`) | `fork.cache_mode "4,4"` |
| draft | DFlash2, `num_draft_tokens 4`, `dynamic_draft true` (`:24-32`) | no t27/t29 `perf` arm carried a draft; the campaign's one `spec_decode` arm failed (`t27-A-base`) | M4's prose |

`extract_exllama` reads this file, so the machine-checked value is 381/262144/"4,4" — pinned to a line
in this tree, and no longer a value the review had to state twice. No report should quote 131 without
saying it is 512 − 381, the CPU tail (`model_init.py:68`).

## 10. How to re-run, in order

**First, with no GPU and no process** — from this worktree, on the dev box (stdlib only, no torch):

```bash
cd /home/dev/.local/share/opencode/worktree/3a6c9f/adaptive-expert-tier

# the gate: every row against both trees, one PASS/FAIL line per row
python3 bench/strata_transfer_check.py --check          # -> 6/6 rows PASS, exit 0

# the module's own assertions (the gate on a box with no pytest)
python3 bench/strata_transfer_check.py --selftest       # -> 123/123 checks passed, exit 0

# ... and a missing real tree is a failure, not a green fixtures-only run (exit 2, not 1)
python3 bench/strata_transfer_check.py --selftest --root /nonexistent   # -> NO SUCH TREE, exit 2
python3 bench/strata_transfer_check.py --selftest --root /nonexistent --allow-missing-tree  # -> exit 0

# the pytest form, including the two plan tests
uv run --no-project --with pytest python -m pytest bench/test_strata_transfer_check.py -q   # -> 8 passed

# what each side's extractor reads, with the file:line per value
python3 bench/strata_transfer_check.py                  # Strata's side as JSON
python3 bench/strata_transfer_check.py --fork           # this fork's side as JSON
```

To point the check at another Strata checkout (`--root`) or another fork tree (`--repo-root`), and to
see the failure paths — a drifted `adapt_swaps` and a drifted `split` (the second tree is a scratch copy
of this one with `split 376` and the promote budget raised to 16):

```bash
python3 bench/strata_transfer_check.py --check --root /tmp/opencode/drift3
# FAIL M2-churn: strata.adapt_swaps recorded 96, source says 64 (src/program/generate.cpp:497)
# 5/6 rows PASS, exit 1

python3 bench/strata_transfer_check.py --check --repo-root /tmp/opencode/forkdrift
# FAIL M1-budget: fork.split recorded 381, source says 376 (bench/windows/state.json:7)
# FAIL M2-churn: fork.promote_budget recorded 6, source says 16 (exllamav3/modules/block_sparse_mlp_cpu.py:161)
# FAIL M7-deploy: fork.split recorded 381, source says 376 (bench/windows/state.json:7)
# 3/6 rows PASS, exit 1
```

**Then the arms — none of them run for this report, and none without the operator's consent** (each
stops the served model; restore per T16 §8 and re-verify `/v1/models`). In the order that makes each
result usable by the next:

```bash
# 1. the instrument's price — the one number §7.2 cannot separate, because every arm carried =1.
#    Same session, same length: =1 then =0, and the difference IS the instrument's cost.
#    Until this runs, the report says: instrument cost: unmeasured (all arms carried it).
-Tag t31-inst-1 -Runner perf -MaxLength 131072 -Env "EXL3_MOE_TIER_STATS=1"
-Tag t31-inst-0 -Runner perf -MaxLength 131072 -Env "EXL3_MOE_TIER_STATS=0"

# 2. M3, the fork's half of the miss split: env deltas only, no regime flag, both gates (§3, §8).
#    Two variables on purpose — `stream_t` is gated per expert (moe_cpu_host.py:2009) and an explicit
#    value bypasses the bandwidth calibration (:1726), so either one alone is a null arm.
#    Needs the operator's consent: the pass stops the served model.
-Tag t31-M3-rows2 -Runner perf -MaxLength 131072 \
  -Env "EXL3_MOE_TIER_STATS=1,EXL3_MOE_STREAM_MIN_ROWS=2,EXL3_MOE_STREAM_T=2"
#    its control in the same session: the same line with no EXL3_MOE_STREAM_* variables

# 3. M1, the cache-size lever and the split that follows it (§8; size_check already says FITS)
-Tag t31-M1-cs32k -Runner perf -CacheSize 32768 -MaxLength 32768 -Split 354
#    its control in the same session: -CacheSize 262144 -MaxLength 32768, so both arms share bands

# 4. M4, a runner switch only; the acceptance column is already in arms.csv (accepted DRAFTS, so
#    1 + acc tokens/round), and what the arm adds is the same measurement at the current regime (§7.3)
-Tag t31-M4-w4 -Runner spec_decode                  # -Extra "-dstats <file>" for per-position records
```

**And what is already settled**, so a reader does not spend a pass on it: M2's mechanism, its rate
(2.42 promotions/step, 8 669 by step 3 584), its price (2.318 ms/step hook) and its source (the pinned
arena, not the checkpoint) — second-round t29 only;
M1's pinned-arena half (+40.59 points of coverage, 42 of 48 layers promotable); M3's two gates as
arithmetic, and the `-Env` string that would test the fork's half; M4's acceptance column (2.17 to 2.40
accepted **drafts** of 4 offered, i.e. 3.17–3.40 tokens per round) and the fact that the harness, not a
research note, produced it; M5's `-ngr` verdict
(measured unavailable under this box's RAM floor) and the ~50 lines in `row_table.py` that the row
cache would cost; M7 as deployment policy rather than an engine property. What remains unmeasured is
exactly this: **the tier instrument's own cost**, **M3's fork half**, **M1's cache-size lever and
therefore the +23 to +27 experts/layer**, **M4's unit at the current regime** (note 1's log-derived
2.2–2.4 tokens/window and the harness's 3.17–3.40 tokens/round are different measurements and only an
arm on both sides settles which compares), **M5's row cache** (no
code to measure yet), and **every M7 batch/residency number on the fork**.

## The gates

Both gates exist so this report's claims are checkable rather than asserted, and both were run on
2026-11-04 — the checker in this worktree, the post gate from the deployment's worktree
(`/home/dev/.local/share/opencode/worktree/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf`).
No GPU, no process.

```
$ python3 bench/strata_transfer_check.py --check            # this worktree
PASS M1-budget: 19 comparison(s) - … ; absent, measured unmeasured
PASS M2-churn: 19 comparison(s) - … ; diverged, measured t29-C1-pin-unset-20261103-221302
PASS M3-miss-split: 3 comparison(s) - … ; absent, measured unmeasured
PASS M4-spec-yield: 3 comparison(s) - no extracted constant on either side (see justification); … ; diverged, measured unmeasured
PASS M5-ple: 2 comparison(s) - no extracted constant on either side (see justification); … ; absent, measured unmeasured
PASS M7-deploy: 10 comparison(s) - … ; diverged, measured unmeasured
6/6 rows PASS
```

(That block is abbreviated at the `…`; the same command's output in full, unelided, is quoted under
"The matrix" above, and it is the same run.)

```
$ python3 bench/verify_task.py --post --report .../T31-strata-transfer.md
  PASS  t12-ple0-20261030-103851: 48 CPU-split component(s), every one from the saved profile
  PASS  t12-ple1024-20261030-105757: 48 CPU-split component(s), every one from the saved profile
  PASS  t12-ple512-20261030-104213: 48 CPU-split component(s), every one from the saved profile
  PASS  t16-q4-ref2-20261102-172722: 48 CPU-split component(s), every one from the saved profile
  PASS  t16-q4-ref2-20261102-172722: arm rows present in arms.csv or spec_decode.csv
  PASS  t19-qdrafter-381-20261102-194803: 48 CPU-split component(s), every one from the saved profile
  PASS  t19-qdrafter-381-20261102-194803: arm rows present in arms.csv or spec_decode.csv
  PASS  t27-B-pin8-f4-20261103-122212: 48 CPU-split component(s), every one from the saved profile
  PASS  t29-A1-pin-off-20261103-220854: 48 CPU-split component(s), every one from the saved profile
  PASS  t29-A2-pin-off-20261103-222022: 48 CPU-split component(s), every one from the saved profile
  PASS  t29-C1-pin-unset-20261103-221302: 48 CPU-split component(s), every one from the saved profile
  PASS  t29-C2-pin-unset-20261103-221642: 48 CPU-split component(s), every one from the saved profile
  PASS  t7-dflash2-w4-20261030-071928: 48 CPU-split component(s), every one from the saved profile
  PASS  t7-dflash2-w4-20261030-071928: arm rows present in arms.csv or spec_decode.csv
  PASS  t7-dflash2-w6-20261030-072942: 48 CPU-split component(s), every one from the saved profile
  PASS  t7-dflash2-w6-20261030-072942: arm rows present in arms.csv or spec_decode.csv
verdict: all gates passed
```

Its `INFO` lines (one per prose `%` figure, each reported as "prose, not a ratio band - not checked")
are omitted above; no `FAIL` line was printed, and the exit code is 0.

The gate's own rules explain two things in the text above: it resolves a run id against
`results/passes.csv` and refuses a cited id that is absent or not `status=ok`, which is why a
`status=fail` arm is named by tag (`t27-A-base`, `t12-ngram-ram`) and never by run id; and it rejects a
symmetric percentage band, which is why every spread here is written as one number ("straddle by
4.60 %") rather than as a plus-minus.

## What this could not verify

1. **M4 and M5 have no source constant in either extractor.** Tasks 1-2 froze
   `extract_strata`/`extract_exllama`, and neither covers the draft block or the PLE read path, so
   those two rows are checked on their arithmetic and their citation discipline only — their PASS lines
   say "no extracted constant on either side" rather than passing on nothing. Adding
   `state.json`'s `active.draft` (`:24-32`) and `row_table.py:27` to the extractors would close it;
   it is not done here because the interfaces are frozen and both sides' selftests assert the key sets.
2. **Which cadence is live at the served config.** The promote path needs `-mcpa`
   (`EXL3_MOE_CPU_PINNED_ARENA`); the t29 A arms (pin off) exercised the sweep, the C arms the
   promote path. `state.json` carries neither flag, only `EXL3_MOE_CPU_PIN=1`, so the row quotes both
   cadences and says which arm measured which.
3. **The plan says "8/8 rows PASS"**; the brief pins six mechanisms and the plan's own self-review
   lists M1-M5 + M7, so the CLI prints `6/6`. M6 (the CPU pool's handshake) and M9 (quantization) are
   deliberately out of this matrix — both notes put them out of the gap.
4. **`pcie_frac 0.55` is Strata's nominal pre-probe share, `stream_t 8`, `num_draft_tokens 4`,
   `max_batch_size 2`, `sysmem_kv_cache 0` are quoted by hand**, not extracted: the first two because
   the extractor has no such key, the last three because they live in `tabbyAPI/config.yml`, outside
   this tree. Each is cited inline. The 0.55 is the base `generate.cpp:2051-2063` starts from for
   native packs; the startup probe scales it to the deployed value and prints that value, and no log
   on this box records the line — so the report quotes 0.55 as a nominal share, never as the share the
   served model ran.
5. **The 60.5-swaps adaptation** is note 2's measured round, not re-derived here (the log prints a
   cumulative exchange count, and the per-adaptation figure needs the round count from a log the
   notes do not carry). The matrix records it as an input, and the derivation it feeds (÷4) is checked.
6. **Cross-day comparison** (Strata's 2026-10-29 log, the fork's 2026-11-03 arms) is used for
   mechanism and counts only, never as a timing control; every timing figure above is same-session or
   explicitly labelled as the served deployment's own log.
7. **The tier instrument's price is unknown, not small.** `EXL3_MOE_TIER_STATS=1` was in every arm of
   both campaigns and no row of `results/passes.csv` carries `EXL3_MOE_TIER_STATS=0`, so every
   coverage number in this report was bought at an unmeasured cost. §7.2 records it as
   `unmeasured` and §10 arm 1 is the same-session A/B that would price it; it is not estimated here.
8. **The four t29 arms did not run on pristine HEAD.** They ran on the installed engine at the branch
   HEAD plus the two one-line repairs t29 §3 documents (this torch's `torch.cuda.Event` form, and
   `pinned_layers()`'s nesting level); at pristine HEAD every pinned pass dies inside the instrument
   before its first tier line. The worktree itself was not modified — so those numbers measure the
   mechanism, not a shipped tree, and the two lines are the first thing a re-run needs.
9. **`acc` in `arms.csv` is accepted *drafts* per round**, not tokens per window
   (`spec_decode.py:142-151`: `acc_len = dacc / rounds`, and every round emits one token that is not
   an accepted draft). So the harness's own column converts to **1 + `acc` = 3.17–3.40 tokens per
   round**, and reading 2.17–2.40 as tokens/window — which this report did until the review — understated
   the harness's yield by about a token. That column is *not* note 1's live-log derivation (2.2–2.4
   tokens/window, computed as 1 + accepted/windows from the served deployment's own log): the two are
   different measurements of different runs, they disagree, and neither is reconciled here. §4 keeps
   them in separate rows for that reason. The mechanism claim survives — a ceiling of 4 offered drafts
   against a policy that chooses a window per round — but with the fork at 3.17–3.40 tokens per round
   on the harness's column, **neither the size nor the sign of the divergence is open-and-shut**: the
   fork may be at parity with or ahead of Strata's 2.5–3.5 on that reading, and the arm that would
   settle it is §8's `t31-M4-w4`.
