# T31 — Strata → exllamav3 transfer: the matrix, and its verdicts machine-checked

Written 2026-11-03/04 on `feat/adaptive-tier` (base `165-win-perf`). **No GPU, no torch, no process
was started for this report:** every number below was already measured, and what is new here is the
comparison. Six mechanisms (M1, M2, M3, M4, M5, M7) get one row each: the value **both** engines'
sources state, with the `file:line` it came from, a verdict, the reason for it, and the run id that
measured it or the literal `unmeasured`.

The matrix is not prose. It lives in `bench/strata_transfer_check.py` as `MATRIX`, and
`python3 bench/strata_transfer_check.py --check` compares every recorded constant against a
regex read of the two sources, every `derived` number against its own recomputation, and the citation
discipline (`measured` a run id or `unmeasured`; a justification wherever one is owed). If either
engine's default drifts, or a division stops adding up, the check fails and names the row and the key.
The rows' `justification` strings in the code are the sections below, quoted.

Sources: `bench/strata_transfer_check.py:MATRIX` (authoritative), Strata at `/home/dev/UnbiasedStrata`
(read-only), this branch, `results/passes.csv` of the deployed harness, `wiki/plans/v165-reports/
T16-adaptive-tier.md` §7, and `/tmp/opencode/t29-pin-unset.md`.

## The matrix

`faithful` is the only verdict that may omit a justification; all six rows below carry one, so each
has a section with its arithmetic. Strata's side is quoted from the sibling tree; the fork's from
this branch unless the cell says otherwise.

| mechanism | Strata (value, file:line) | fork (value, file:line) | verdict | justification (short; full text in the row's section) | measured |
|---|---|---|---|---|---|
| **M1-budget** — VRAM on the expert tier instead of KV/embedding | `kv_resident 0` (default; deployed run passes 32768) `src/program/generate.cpp:311`, `prefill_chunk 0` `:431`; 8301 slots = **173 experts/layer** of 15.74 GiB, `logs/server-20261029-172352.err.log:390,395` | `cache_size 131072` `bench/windows/state.json:10`, `cache_quant "2,2"` `:11`, `split 376` `:7` — **measured regime: `-cs 262144 -cq 4,4 -mcs 381`** (`results/passes.csv`, t27/t29 `args`) ⇒ **131 of 512 experts/layer** | `absent` | three levers (cache_size, `sysmem_kv_cache`, split) buy +10..13 experts/layer; 80 % by residency needs ≈38 GB of expert weights on a 24 GiB card (§1) | `unmeasured` |
| **M2-churn** — the adaptive tier's cadence and source | `adapt_every 4` `generate.cpp:474`, `adapt_swaps 96` `:497`, `adapt_decay 0.7` `:475`, `slot_pool global` `:382` + `expert_cache.hpp:221`, `swap_source ram-copy` `:4590` | `promote_budget 6` `block_sparse_mlp_cpu.py:161`, `ratio 1.5` `:162`, `decay 0.98` `:163`, `spares 4` `:1088`, `sweep_interval 128` `:1208`, `sweep_max 64` `:141`, `sweep_source checkpoint` `:1293` | `diverged` | 15.1 swaps/round (measured) against 0.20/step at the t27 operating point: **75×**, and a checkpoint read per swap against a RAM copy (§2) | `t29-C1-pin-unset-20261103-221302` |
| **M3-miss-split** — the miss is split CPU/DMA | `pcie_frac -1.0` = the model's own, **0.55 DMA** for native packs `generate.cpp:470` (read by hand: not an extracted key) | `stream_min_rows 32` `model/moe_cpu_host.py:141`; second gate `stream_t 8` `:139` (by hand) | `absent` | a 4-row window is 8× below the gate and 2× below the per-expert gate; the arm that tests it is `EXL3_MOE_STREAM_MIN_ROWS=2` + `EXL3_MOE_STREAM_T=2` (§3) | `unmeasured` |
| **M4-spec-yield** — what one verify pass buys | MTP ≤3 drafts + prompt lookup, cost-model chosen, `docs/upstream/DETAILS.md:1032-1035`; **2.5–3.5 tokens/window** `logs/server-20261029-172352.err.log:425,444,453` | DFlash2 `num_draft_tokens 4`, `dynamic_draft` shortens only, `generator.py:1128-1146`; **2.2–2.4 of 4 rows** (note 1 §2) | `diverged` | tokens/window = 1 + Σ acceptance, truncated at 4: 1.59× at best, 1.04× at worst; no arm carries a draft (§4) | `unmeasured` |
| **M5-ple** — the n-gram table read on the decode path | row cache 1M rows (≈90 MiB), up to ~82 % of reads, 20–34 % recurrence `include/strata/ngram/ple_reader.hpp:13-16`; issue/collect split `:5-11`; un-hidden gather 2.1–2.6 ms/token `generate.cpp:393-395` | no row cache: `ext.ngram_gather_cpu` per gather `modules/row_table.py:282-312`; `PREFETCH_MIN_TOKENS 256` `:27` declined at `ngram_embedding.py:334`; `advise()` is `posix_fadvise`, a no-op on Windows `row_table.py:208-215` | `absent` | three defences absent; 16 × 90 B = 1440 B/token is the byte scale, Strata's own un-hidden 2.1–2.6 ms/token the ceiling (§5) | `unmeasured` |
| **M7-deploy** — deployment policy, not an engine property | one request at a time (`batch 0` `generate.cpp:456`; no `--slots`), `--kv-resident 32768` of 262144 `start-logged.ps1:54` | `max_batch_size 2` (`tabbyAPI/config.yml:217`, outside this tree), `sysmem_kv_cache 0` `:441`, whole cache in VRAM: `cache_size 131072` `bench/windows/state.json:10`, measured `262144` | `diverged` | 8× KV residency and 2-vs-1 concurrency; the second stream costs 8.1 T/s against 57.2 single (§6) | `unmeasured` |

Two regime values are recorded twice on purpose (M1, M7): this worktree's `state.json` and the
deployment's disagree (376/131072/"2,2" against 381/262144/"4,4"), so both are stated and M1's
"131 of 512" is derived from the **measured** 381 and labelled as such. §7 works the disagreement out.

`--check` on this tree prints, one line per row:

```
PASS M1-budget: 9 comparison(s) - strata.kv_resident_tokens=0 (src/program/generate.cpp:311), strata.prefill_chunk=0 (src/program/generate.cpp:431), fork.cache_size=131072 (bench/windows/state.json:10) +2 more; derived experts_per_layer_repo=136, experts_per_layer_measured=131, strata_experts_per_layer=173, experts_short=42; absent, measured unmeasured
PASS M2-churn: 18 comparison(s) - strata.adapt_every=4 (src/program/generate.cpp:474), strata.adapt_swaps=96 (src/program/generate.cpp:497), strata.adapt_decay=0.7 (src/program/generate.cpp:475) +9 more; derived swaps_per_round_cap=24, swaps_per_round_measured=15.1, sweep_swaps_per_step=0.5, promotions_per_step_t27=0.201, churn_ratio=75, promotions_per_step_t29=2.42; diverged, measured t29-C1-pin-unset-20261103-221302
PASS M3-miss-split: 3 comparison(s) - fork.stream_min_rows=32 (exllamav3/model/moe_cpu_host.py:141); derived rows_short=28, gate_factor=8.0; absent, measured unmeasured
PASS M4-spec-yield: 3 comparison(s) - no extracted constant on either side (see justification); derived yield_ratio_low=1.04, yield_ratio_high=1.59, window_cap_minus_high=1.6; diverged, measured unmeasured
PASS M5-ple: 2 comparison(s) - no extracted constant on either side (see justification); derived bytes_per_token=1440, cache_mib=90; absent, measured unmeasured
PASS M7-deploy: 5 comparison(s) - fork.cache_size=131072 (bench/windows/state.json:10), fork.cache_mode='2,2' (bench/windows/state.json:11), fork.split=376 (bench/windows/state.json:7); derived kv_ratio=8, batch_delta=2; diverged, measured unmeasured
6/6 rows PASS
```

## 1. M1-budget — where the VRAM goes, and the ≈38 GB wall

**`MATRIX["M1-budget"]["justification"]`** (reflowed for the page; typography and spacing normalised — `—`, `×`, `≈`, `≤`, `§`; the code is
  authoritative): the fork's 131 experts/layer is a static
prefix of one uniform per-layer budget at the measured split 381 (512 − 381; this worktree's
`state.json` says 376, i.e. 136); Strata's 173/layer is 8301 global slots over 48 layers (15.74 GiB)
that follow routing. Three levers would move the fork: `cache_size` 262144 → 32768/65536,
`memory.sysmem_kv_cache` ~1800 MB, then `cpu_moe_split_experts` below 381. Their arithmetic bounds the
gain: 1.9–2.4 GB of freed KV = 500–650 slots = +10..13 experts/layer, so 131 → 141..144, still ~30
short of 173. Reaching 80 % by residency instead needs 410 of 512 experts/layer = 410 × 48 × 1.91 MB
= 37.6 GB ≈ 38 GB of VRAM, against a 24 GiB card whose dense trunk already holds 17.5 GiB
(`bench/size_check.py`, t29 artefact §1). No `cache_size` or split delta closes it, so the mechanism
is **absent**, not diverged.

The arithmetic, spelled out:

| step | number | source |
|---|---|---|
| experts per layer on the CPU at the measured regime | 512 − 381 = **131** | `results/passes.csv`, t27/t29 `args` (`-mcs 381`) |
| … at this worktree's regime | 512 − 376 = **136** | `bench/windows/state.json:7` |
| expert bytes per expert | ≈1.91 MB (47.03 GB / 512 / 48) | note 1 §M1, from `quantization_config.json` |
| the fork's expert tier | 131 × 48 × 1.91 MB = **12.0 GB** | derived |
| Strata's expert tier | 8301 slots / 48 layers = **173/layer**, 15.74 GiB | `logs/server-…err.log:390,395` |
| lever 1+2: KV 262144 → 32768/65536 + `sysmem_kv_cache` ~1800 | frees **1.9–2.4 GB** = 500–650 slots | note 1 §M1 |
| … in experts per layer | 500–650 / 48 = **+10..13**, i.e. 131 → 141..144 | derived |
| still short of Strata | 173 − 131 = **42** experts/layer | derived |
| reaching 80 % by residency | 410 × 48 × 1.91 MB = **37.6 GB ≈ 38 GB** | derived |
| VRAM the card has | 24 564 MiB, of which the trunk alone is 17 551 MiB | `bench/size_check.py` output, t29 artefact §1 |

So the levers are real but bounded: they close about a third of the slot gap, and the rest is not a
setting — Strata reaches 79–95 % coverage *from* a 34 %-of-experts tier because the tier follows the
conversation, which is M2's mechanism, and the fork's static prefix has no such follower. Passing
`split 381 → 376` in the other direction makes it worse, not better (136 experts, 3 % of the tier
given away), which is why the row states both regimes rather than quietly picking one.

## 2. M2-churn — the unit trap: 15 swaps/round against 0.20/step

**`MATRIX["M2-churn"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): units and rate, not names: Strata spends up to
`adapt_swaps` 96 on ONE adaptation every `adapt_every` 4 decode rounds over ONE global pool
(`slot_pool global`), fed from the resident RAM copy (`swap_source ram-copy`, 0 file reads), so
96/4 = 24 swaps/round is the cap and a measured adaptation did 60.5 → **15.1/round**. The fork at the
t27 operating point promoted **710** times over the 3528-step perf ladder = **0.20/step** (`spares 4`,
`PROMOTE_BUDGET 6/step`, ratio 1.5, decay 0.98), and its legacy sweep is 64 swaps per 128 steps =
0.5/step with two checkpoint reads per swap (`sweep_source checkpoint`). 15.1/0.201 = **75×**, the
review's 75×. Per committed token it is **31×** (Strata 15.1/2.45 = 6.2; a perf step is one token
because `perf.py` registers no draft), and `t29-C1` shows the budget raised 13×: 8669 promotions /
3584 steps = **2.42/step** for +40.6 points of coverage (31.7 → 72.3 % pooled).

The two campaign's numbers, as read:

| quantity | Strata | fork, t27-B (pin 8 GiB, 4 spares) | fork, t29-C1 (pin unset, 16 spares) |
|---|---|---|---|
| cap | 96 per adaptation ÷ 4 rounds = **24/round** | 6/step, and 64 per 128 steps = 0.5/step (sweep) | 6/step |
| measured churn | 60.5 per adaptation → **15.1/round** | 710 promotions / 3528 steps = **0.201/step** | 8669 / 3584 = **2.42/step** |
| source of the bytes | RAM ↔ VRAM exchange, "no file reads" (`generate.cpp:4590`); log: "0 blob reads from the file" | checkpoint re-read per swap (`stc.get_tensor`, `block_sparse_mlp_cpu.py:1293`) | same |
| what it bought | 79–95 % coverage, 94.1 % in the sweep regime | none: coverage 28.5 % control against 27.7 % (and the column was frozen — T16 §7) | +40.6 points (72.3 vs 31.7), hook 2.318 ms/step |

Sources: `perf-t27-B-pin8-f4-20261103-122212.log` (final counter 710; 28 tier windows = 27 × 128 + 72
= 3528 steps), `/tmp/opencode/t29-pin-unset.md` §4-§5 (8669 promotions "at step 3584",
`t29-C1-pin-unset-20261103-221302`), `strata-vs-exllama-research.md` §2.2 (the 60.5-swap round).

The unit that made the campaign's first churn estimate 75× too small is this: Strata's 96 is a
per-round budget over a single global pool; the fork's 64 is a total across 48 layers every 128 steps,
i.e. 0.5/step, and the prom path's 6/step is a *different* unit again (a step is one token, a round is
~2.45 tokens). Normalising to the token — the only unit both engines share — narrows the gap from 75×
to 31×, and the honest number to quote is the pair, not either one alone.

## 3. M3-miss-split — the env delta, and the second gate

**`MATRIX["M3-miss-split"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): the miss goes somewhere else. Strata's
`pcie_frac` default is the model's own (`-1.0` at `generate.cpp:470`, read as 0.55 DMA for the native
packs and scaled by a live PCIe probe), so ~55 % of a verify round's distinct missed experts are DMA'd
and computed on the GPU. The fork's streamed path exists (`moe_cpu_host.py:1968-1998`) but is gated
twice: `rows < stream_min_rows 32` returns before it (`:1981`), and an expert needs `stream_t 8`
assigned tokens (`:139`, read by hand — not an extracted key) — so a 4-row window reaches neither and
**100 % of decode misses go to the worker process**. The delta that would test the fork's half is env-only,
`EXL3_MOE_STREAM_MIN_ROWS=2` plus `EXL3_MOE_STREAM_T=2`, with no regime flag; nothing on this box has
run it.

The arithmetic:

- a verify window is at most `num_draft_tokens` + 1 = **4 rows** wide on the fork's deployed regime; the
  gate is 32: **8× short**, 28 rows short (`derived.rows_short`).
- the second gate is per expert: an expert qualifies for staging at `stream_t = 8` assigned tokens
  (`moe_cpu_host.py:139`); 4 rows can route an expert at most 4 times, so **no decode expert can ever
  qualify** even after the row gate is dropped.
- therefore the arm that tests the fork's half needs *both* env deltas — the one this report names,
  `EXL3_MOE_STREAM_MIN_ROWS=2`, and `EXL3_MOE_STREAM_T=2` — and it is still an experiment, not a port:
  the fork's own calibration says 4 rows lost to 8 at prefill sizes (`:1078-1118`), which is a
  prefill measurement and does not transfer to a 4-row window either way.

The exact command, if the operator consents (it stops the served model; restore per T16 §8):

```
-Tag t31-M3-rows2 -Runner perf -MaxLength 131072 -Env "EXL3_MOE_TIER_STATS=1,EXL3_MOE_STREAM_MIN_ROWS=2,EXL3_MOE_STREAM_T=2"
```

## 4. M4-spec-yield — the window numbers, and the two notes that disagree

**`MATRIX["M4-spec-yield"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): window arithmetic: tokens/window = 1 + the
sum of per-position acceptance, so the deployed DFlash2 cap `num_draft_tokens 4` truncates the sum at 4
and the fork measures **2.2–2.4 of 4 rows**, `dynamic_draft` only ever shortening
(`generator.py:1128-1146`). Strata drafts ≤3 with the model's own MTP head plus a prompt-lookup drafter
picked per round by a cost model (`docs/upstream/DETAILS.md:1032-1035`) and measures **2.5–3.5
tokens/window**: 3.5/2.2 = **1.59×** at best, 2.5/2.4 = **1.04×** at worst. The second research note's
own log pass (`strata-vs-exllama-research.md` §1) reads 2.42–2.47 on both sides, so the divergence
stands on note 1's window arithmetic and is not reproduced by note 2's requests. No arm of either
campaign carries a draft (`perf.py` registers none; draft arguments belong to `-Runner spec_decode`),
so this row is `unmeasured`.

The numbers, and what they do not settle:

| | Strata | fork, deployed DFlash2 |
|---|---|---|
| window | MTP ≤3 + prompt lookup up to 5, chosen per round by measured tokens/ms | block-diffusion drafter, `num_draft_tokens 4`, `dynamic_draft: true` (shortens only) |
| tokens/window | 2.5–3.5 (log: 2.9 / 2.46 / 2.56 / 3.48 / 2.65 over five requests) | 2.2–2.4 of 4 rows (note 1 §2: 2.17 / 2.41 / 2.33 / 2.56) |
| the window is | kept full by a policy that prices both drafters | truncated by its own ceiling: 1.6 rows of the 4 go unused on average |

Both notes agree on the physics and disagree on the size: note 1 derives 2.5–3.5 against 2.2–2.4 from
five and four requests; note 2, reading a different session, gets 2.42–2.47 on both sides and calls it
a wash. That disagreement is recorded here rather than resolved, because neither campaign ran an arm
with a draft: `-Runner perf` registers no draft flags, so **the fork's spec-yield has never been
measured by this harness** — the 2.2–2.4 is the served deployment's own log. A verdict of `diverged`
therefore rests on the mechanism (a hard ceiling of 4 against a policy that chooses a window per
round), and the timing claim it would support is unmeasured.

## 5. M5-ple — the read the fork does not cache

**`MATRIX["M5-ple"]["justification"]`** (reflowed; typography and spacing normalised — `×`, `≈`, `≤`, `§`): absent, and cheap to bound: Strata hides the 16
unbuffered row reads of a decode token (90 B/row, 26.8 GiB table never in RAM) behind the embedding and
layer 0 with an issue/collect split, and serves repeats from a bounded 1M-row (90 MiB) cache that
measures up to 82 % of reads with 20–34 % recurrence inside one prompt (`ple_reader.hpp:13-16`). The
fork has no such row cache: every gather goes to `ext.ngram_gather_cpu` over a
`FILE_FLAG_NO_BUFFERING` handle (`row_table.py:282-312`, opened at `loader/safetensors.py:247-248`), so
a repeated row is read from disk again. Its two cheaper substitutes both miss decode: `prefetch()`
declines decode-sized inputs (`PREFETCH_MIN_TOKENS` 256 positions, `row_table.py:27`, checked at
`ngram_embedding.py:334`), and the read-ahead hint `advise()` (`row_table.py:208-215`) is
`posix_fadvise`, i.e. a no-op on the Windows box this fork serves. 16 × 90 B = 1440 B per token is the
byte scale; Strata's own un-hidden gather is 2.1–2.6 ms/token (`generate.cpp:393-395`), which is the
ceiling on a port, not a promise.

Arithmetic: 2^20 rows × 90 B = 90 MiB per table buys up to 82 % of reads; the fork's per-token traffic
is 16 rows × 90 B = 1440 B, i.e. ~1.4 KB of the 0.92 GB a token already reads in expert weights — the
byte traffic is negligible and **the stall is the point**: Strata measured its own un-hidden gather at
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
request at a time (`batch 0` default, `generate.cpp:456`, and `start-logged.ps1` passes no `--slots`)
with 32768 of 262144 KV cells resident (`start-logged.ps1:54`); the fork's deployed regime serves
`max_batch_size 2` (`tabbyAPI/config.yml:217`, outside this tree) with the whole cache in VRAM
(`sysmem_kv_cache 0`, `config.yml:441`) — 262144 cells at the measured regime and 131072 in this
worktree's `state.json`. That 8× KV-residency delta is exactly the M1 lever, and concurrency is
anti-scaled on this box (49.8 T/s for the second stream against 57.2 T/s single, note 1 §2). No arm of
either campaign varied batch or KV residency, so the divergence is real and `unmeasured`.

Arithmetic: 262144 / 32768 = **8×**; at 896 B/token/layer × 12 attention layers the fork's VRAM cache is
≈2.8 GB where Strata's resident window is 1.69 GiB in pinned RAM — the difference is ~1.9–2.4 GB of
VRAM, which is exactly the M1 lever's input. Concurrency: 57.2 tok/s single-stream against 49.8 for the
stream that overlapped another (and 8.1 T/s for the second stream itself, note 1 §2), so two sessions
cost more than they yield on this box — a deployment decision, not an engine property, and the deployed
`max_batch_size: 2` was never A/B'd by either campaign.

## 7. The two regime files (R5, widened)

`results/passes.csv`'s `t27-*` and `t29-*` rows and the t29 dry-run's `ARGS` line say what the measured
arms actually ran:

```
-m … -cs 262144 -chunk_size 4096 -ambs 2 -cq 4,4 -max_length 131072 -mcs 381 -mct 16 -sd
```

The deployed harness's `state.json` (`_agent/v165/state.json`, and `165-win-perf`'s copy of it) agrees:
`split 381, cache_size 262144, cache_quant "4,4"`. **This branch's** `bench/windows/state.json` says
`split 376, cache_size 131072, cache_quant "2,2", max_length 131072`, and its own note explains the
131072 as "the measurement length; 262144 is a single final confirmation". So there are two regimes in
the tree, and the review quotes both wherever the number matters:

| value | this worktree's `state.json` | the measured arms | in the matrix as |
|---|---|---|---|
| split | 376 (`:7`) | 381 | `fork.split 376` + `derived.split_measured 381`; 131 = 512 − 381, labelled measured |
| cache_size | 131072 (`:10`) | 262144 | `fork.cache_size 131072` + `derived.kv_vram_tokens_measured 262144` |
| cache_quant | `"2,2"` (`:11`) | `4,4` | `fork.cache_mode "2,2"`; the measured `4,4` is in M7's prose |
| draft | `mode mtp`, `num_draft_tokens 2`, `dynamic_draft false` (`:24-32`) | DFlash2, 4, dynamic true | M4's prose: no arm carried either |

`extract_exllama` reads this branch's file, so the machine-checked value is 376/131072/"2,2" — that is
the only one a drift check can pin to a line in this tree. The measured column's numbers are the
deployment's, quoted with their provenance, and no report should quote 131 without saying it is 512 −
381.

## How to re-run

From this worktree, on the dev box (stdlib only, no GPU, no torch, no process started):

```bash
cd /home/dev/.local/share/opencode/worktree/3a6c9f/adaptive-expert-tier

# the gate: every row against both trees, one PASS/FAIL line per row
python3 bench/strata_transfer_check.py --check          # -> 6/6 rows PASS, exit 0

# the module's own assertions (the gate on a box with no pytest)
python3 bench/strata_transfer_check.py --selftest       # -> 113/113 checks passed, exit 0

# the pytest form, including the two plan tests
uv run --no-project --with pytest python -m pytest bench/test_strata_transfer_check.py -q   # -> 8 passed

# what each side's extractor reads, with the file:line per value
python3 bench/strata_transfer_check.py                  # Strata's side as JSON
python3 bench/strata_transfer_check.py --fork           # this fork's side as JSON
```

To point the check at another Strata checkout (`--root`) or another fork tree (`--repo-root`), and to
see the failure paths — a drifted `adapt_swaps` and a drifted `split`:

```bash
python3 bench/strata_transfer_check.py --check --root /tmp/opencode/drift3
# FAIL M2-churn: strata.adapt_swaps recorded 96, source says 64 (src/program/generate.cpp:497)
# 5/6 rows PASS, exit 1

python3 bench/strata_transfer_check.py --check --repo-root /tmp/opencode/forkdrift
# FAIL M1-budget: fork.split recorded 376, source says 381 (bench/windows/state.json:7)
# FAIL M2-churn: fork.promote_budget recorded 6, source says 16 (…:161)
# FAIL M7-deploy: fork.split recorded 376, source says 381 (bench/windows/state.json:7)
# 3/6 rows PASS, exit 1
```

The one arm this row set asks for, if the operator consents (it stops the served model — restore per
T16 §8 and re-verify `/v1/models`):

```bash
# M3, the fork's half of the miss split: env deltas only, no regime flag
-Tag t31-M3-rows2 -Runner perf -MaxLength 131072 \
  -Env "EXL3_MOE_TIER_STATS=1,EXL3_MOE_STREAM_MIN_ROWS=2,EXL3_MOE_STREAM_T=2"
```

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
4. **`pcie_frac 0.55`, `stream_t 8`, `num_draft_tokens 4`, `max_batch_size 2`, `sysmem_kv_cache 0`
   are quoted by hand**, not extracted: the first two because the extractor has no such key, the last
   three because they live in `tabbyAPI/config.yml`, outside this tree. Each is cited inline.
5. **The 60.5-swaps adaptation** is note 2's measured round, not re-derived here (the log prints a
   cumulative exchange count, and the per-adaptation figure needs the round count from a log the
   notes do not carry). The matrix records it as an input, and the derivation it feeds (÷4) is checked.
6. **Cross-day comparison** (Strata's 2026-10-29 log, the fork's 2026-11-03 arms) is used for
   mechanism and counts only, never as a timing control; every timing figure above is same-session or
   explicitly labelled as the served deployment's own log.
