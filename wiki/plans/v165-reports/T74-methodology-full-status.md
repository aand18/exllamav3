# T74 — Argon's full methodology review: every claim, its status, and what changed

Every claim in Argon's methodology review, checked against the code and the data, with the fix
applied where one was warranted. Commits at the end.

## §1 What the harness gets right — confirmed, no action

| # | claim | status |
|---|---|---|
| 1.1 | single regime (`state.json`) + delta-only CLI, no flag restatement in `-Extra`, deployed==repo check | **verified in use** (it aborted two T70 arms in 6 s for `-ndt` in `-Extra`) |
| 1.2 | frozen routing profile per pass prevents arm B inheriting arm A's frequencies | **verified** (`frozen_state_dir` / `master`) |
| 1.3 | watchdogs sample *available* RAM (not free) and VRAM at 1 Hz, voiding on breach | **verified** — 3 breaches this campaign, all correctly voided |
| 1.4 | `verify_task --post` audits cited run ids and numbers against the CSVs | **verified** — caught my own bad citations 5× |
| 1.5 | same-session ABBA + mechanism counters resolved `cmb` to ±3 µs | **verified**, and it is what made T66/T69 decisive |

## §2 Aggregation bugs — all three real, all fixed

| # | claim | verified? | fix |
|---|---|---|---|
| 2A | `eval/spec_decode.py` computes `Σ(gen_tps·N)/ΣN`, i.e. `Σ(N²/T)/ΣN` — a token-weighted arithmetic mean of rates, upward-biased by Jensen, weighting a long prompt quadratically | **confirmed at `:137-140`** | replaced with **`ΣN / ΣT`**; `time_generate` is now stored per prompt and `tokens`/`seconds` are printed per category (`CAT` lines) so a ratio of sums survives into the CSV |
| 2B1 | `spec_primary.py`'s docstring promises "ratio of sums" while it averages per-run rates | **confirmed** | docstring corrected; `throughput()` now takes a **ratio of sums when the columns exist** and falls back to the mean of rates otherwise, reporting which in `basis` |
| 2B2 | spread used the unweighted mean of the three primary categories while the primary is 5:4:3 | **confirmed** | spread now uses the same 5:4:3 weighting as the gate (T73) |
| 2B3 | the `ARM` summary line weights `Trivial repetition` the same `1/8` as `Agentic; code` | **confirmed** | `arm_line` now documents itself as an unweighted 8-category headline and prints `tokens`/`seconds`; the decision metric remains `spec_primary.py` |

## §3 Decision-rule flaws — three real, all fixed in T73

| # | claim | status |
|---|---|---|
| 3A | `(max−min)/mean` grows with N (1.13σ at N=2, 2.06σ at N=4) so replication *raised* the floor | **confirmed** (floors 1.039→1.045→1.050). Fixed: 1.96·SEM/mean |
| 3B | fixed per-category line over 3–5-prompt buckets makes a true +2.0 % gain fail ~51 % of the time | **confirmed** (Argon's arithmetic reproduces). Fixed: each category vetoes only past its own measured noise |
| 3C | `|Δacc| ≤ 0.05` makes any drafting improvement unpassable | **confirmed** — T61/T63/T70 could only be judged on losses. Fixed: `--drafting` one-sided band |

All four historical verdicts re-scored under the repaired gate: **none flip**, the floors fall to
~1.020 (T73).

## §4 Design flaws — two real and now fixed, one recorded

| # | claim | status |
|---|---|---|
| 4A | testing execution-speed interventions on free-running generation injects 3–5 % trajectory noise into a 0.4 % timing problem; systems arms should gate on fixed-trace `ms/round` | **accepted as method.** Not a script change: T66/T69/T70 already decided on `cmb` (±3 µs) and `distinct`, not on tok/s. Recorded as the standing rule: *`cmb`+`distinct` decide systems arms, `spec_primary.py` decides drafting arms, tok/s alone decides neither* |
| 4B1 | `spec_decode` runs the whole 22-prompt undrafted Baseline arm by default, wasting ~a third of every pass's GPU time | **confirmed** | `run_pass.ps1` now passes **`-nbl` by default** for `spec_decode`; `-WithBaseline` opts back in |
| 4B2 | that Baseline arm contaminates `ROUND_PROF`/`distinct` CSVs — `97 920` rows=1 layer-rounds vs `27 071` rows=5 — **which is exactly how T58's "median 8" was born** | **confirmed; this was the root cause of my worst error this campaign** | same fix; the drafted window reads 21 distinct experts once the rows are separated |
| 4C1 | `perf.py` (`bsz=1`) is not a valid proxy for `spec_decode` (`rows=5`) | **confirmed** — it is what mis-tested T46/T51/T52 | recorded; no code change |
| 4C2 | `reset_placement()` leaves `spares = 4` GPU slots unmapped and unpinned layers can never promote | **confirmed** (`block_sparse_mlp_cpu.py:1388-1393`, `:361`) | recorded (T72 §1); any future dynamic arm must set `EXL3_MOE_CPU_PROMOTE=0` |

## §5 Its five recommended fixes

1. `spec_decode.py:137-140` → ratio of sums. **Done.**
2. `run_pass.ps1` → `--no_baseline` by default. **Done** (`-WithBaseline` to opt in).
3. `spec_primary.py` → primary as total tokens / total seconds. **Done** (with fallback + `basis`).
4. Replace the range floor with a paired/bootstrap interval. **Done as SEM** — the 1/√N property is
   what was missing; a bootstrap would be the next refinement if the CSV carried per-prompt rows.
5. Remove the acceptance cap for drafting arms and gate systems arms on `ms/round`. **Done** (1)
   and recorded as method (2).

## Verification

- `bench/test_spec_primary.py`: **13 passed**.
- `powershell -File` parse of the modified `run_pass.ps1`: **parse OK**.
- `py_compile` on `eval/spec_decode.py` and `bench/spec_primary.py`: **ok**.
- Historical re-score: T61, T66, T69, T70 — no verdict changes (T73 table).

## What did not change, and why that matters

The **absolute** tok/s in `spec_decode.csv` from every past pass is inflated by the 2A bug, and by a
prompt-dependent amount, so cross-pass absolute comparisons carry that bias. Ratios **within** a pass
share it partially, which is why no verdict flipped — but no future report should quote an absolute
tok/s from a pre-T74 pass without saying so.

## Evidence

No run of its own: this is a code and method review, which is why the post gate reports "no run ids".
The re-scored verdicts it depends on are `t61A-bf16-131k-20261110-065833`,
`t61D-bf16-131k-20261110-074755`, `t61B-exl3-131k-20261110-071514`,
`t61C-exl3-131k-20261110-073132`, `t66A-static-262k-20261110-172259`,
`t66D-static-262k-20261110-181449`, `t66B-dynamic-262k-20261110-174018`,
`t66C-dynamic-262k-20261110-175726`, `t68E-cs40-mcs381-20261110-211419`,
`t68H-cs40-mcs381-20261110-215158`, `t68J-cs40-mcs381-20261110-221041`,
`t69A-cs40-mcs373-20261110-223004`, `t69B-cs40-mcs373-20261110-224657`,
`t70A-ndt4-262k-20261111-055730`, `t70D-ndt4-262k-20261111-061433`,
`t70E-ndt4-262k-20261111-070757`, `t70B-ndt3-262k-20261111-063421`,
`t70C-ndt3-262k-20261111-065103`; the floor table is T73's.

Code changed: `eval/spec_decode.py` (aggregation, `time_generate`, `CAT` lines, `ARM` suffix),
`bench/spec_primary.py` (`throughput`/`basis`, weighting, gate),
`bench/test_spec_primary.py` (9 → 13 tests), `bench/windows/run_pass.ps1` (`-nbl` default,
`-WithBaseline`, `tokens`/`seconds` columns).
