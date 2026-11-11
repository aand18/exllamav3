# T76 — the reasoning is now in the code, at the line it protects

Every methodological change made in T73–T75 now carries, at the site, **why it exists and what it
costs when absent** — because the tests pin the behaviour but not the reason, and a future reader who
"simplifies" the line back would pass the suite and repeat the mistake.

## What was annotated, and the failure each note records

| file | site | reasoning recorded |
|---|---|---|
| `bench/spec_primary.py` | module docstring | the docstring once promised "every ratio is a ratio of sums" while the code averaged rates; the note says the basis belongs in the **output** (`basis()`), not the prose — prose can promise what the code does not compute |
| | `t975()` | `1.96·sd/√2 = 0.98·|v₁−v₂|` at n=2 — the normal quantile **reproduced the range statistic it replaced** (measured 0.980 at n=2, 0.571 at n=3). Shipped once; caught by external review, not by the tests |
| | `arm_spread()` | `(max−min)` grows with N, so more evidence made the gate **stricter** (floors 1.039→1.045→1.050), which is backwards |
| | category guard | the multiple-testing arithmetic: 3 buckets × 21 % false-failure = **51 %** rejection of a true +2.5 % gain; and the accepted trade (a real 4 % single-category regression now passes) |
| | `resolution` | why every arm reads "inconclusive": at 2–3 runs the interval is 7–26 % wide, so a 2 % claim is unsupported |
| | acceptance rule | `|Δacc| ≤ 0.05` was written for kernel arms; on drafter/window arms it made the better mechanism **unpassable by construction**, so T61/T63/T70 could only be judged on losses |
| | acc-loss withdraws | with the one-sided band, an arm losing 0.42 accepted tokens still read "inconclusive" — which tells the operator to retry instead of to stop |
| | `primary_of()` | a weighted mean of rates was reported **under the name** of a ratio of sums; they differ whenever prompt lengths differ, and this campaign did it for weeks |
| | `acc_of()` | prompt-count averaging discarded how many rounds a long prompt contributed |
| `eval/spec_decode.py` | tps aggregation | `Σ(N²/T)/ΣN`: quadratic prompt weighting, Jensen-inflated, so **every absolute tok/s before T74 is inflated** by a category-dependent amount; ratios mostly survive, absolutes do not |
| | `CAT` emission | a rate is an average of ratios and cannot be un-weighted, so the numerator and denominator are emitted for downstream |
| `bench/windows/run_pass.ps1` | `-nbl` default | three measured reasons: a third of pass GPU time, ignored by the analysis, and its rows=1 rounds outnumbered the drafted ones 4:1 — **the origin of "median 8"** |
| | `tokens`/`seconds` columns | unrecoverable from the table; blank on pre-T74 passes, which `basis` reports |
| `exllamav3/modules/block_sparse_mlp_cpu.py` | `bsz > 1` counter filter | the pooling that produced "median 8 vs drafted 21", with the counts; plus the note that the harness `-nbl` is the **second, independent** guard |
| `exllamav3/util/round_prof.py` | `write_dump()` | ragged rows raised inside an `atexit` hook, the exception was swallowed, the file stayed **0 bytes**, and a 17-minute pass produced nothing — hence the pad and `meta` length |
| | `_finalize_distinct()` | the copy once sat inside the `sub` bracket and inflated that phase **3.5×**, then propagated into a published table: instrumentation that perturbs what it measures is worse than none |
| `bench/pool_sim.py` | module docstring + `GPU_SLOTS` | `-mcs 381` names the **CPU** side, so the GPU holds 131 — the first version passed 381 and reported 0.49 against 21, a 40× error that looked like a result; and the 8-versus-21 misread. Calibration now runs first and voids everything below it when it fails |

## One correctness fix fell out of writing the notes down

Annotating `pool_sim.py` exposed that the simulator was still *calling* `uniform_resident(381)` — the
docstring described the bug and the code kept it. Corrected to `GPU_SLOTS = 512 − 381 = 131` and the
target moved from "median 8" to "median 21, mean 20.5". It now reports **10.8 simulated vs 20.5
measured**: wrong by 1.9× rather than 40×, which is the real, documented finding that the stationary
routing profile is more concentrated than per-conversation routing. Calibration still fails, and the
tool says so and voids its own numbers — as it must.

## Verification

- `bench/test_spec_primary.py` + `tests/moe_cpu`: **191 passed, 41 skipped**.
- `run_pass.ps1` parses under the PowerShell parser (and the `-nbl` line is real code again — a
  comment-merge during editing briefly swallowed it, caught by reading the emitted lines).
- `py_compile` clean on `eval/spec_decode.py`, `bench/spec_primary.py`, `bench/pool_sim.py`.
