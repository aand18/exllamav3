# Task 4 report — `verify_task.py`: per-line warm check, a real percentage gate, `--draft`, deployed-state check

Status: DONE (two brief ambiguities resolved in code and flagged below)
Commit: `ffec2e50` bench: verify gates read every line, number and deployed file
Files: `bench/verify_task.py`, `bench/test_gates.py` (2 files, +534/−20)

## What was implemented

### 1. Warm gate is per placement line (`warm_gate(log, label, allow_cold)`)

- Reads the log line by line, keeps every line containing `CPU split experts` (49 in
  `perf-mtp-ref-318-20261029-222244.log`), and fails unless **every** one contains
  `placement from saved profile`.
- Offending lines are printed verbatim, indented under the FAIL, with the count
  (`1 of 49 CPU-split component(s) have no saved placement`). No truncation: a truncated list would
  again certify more than it shows.
- A log with no `CPU split experts` line at all → FAIL ("no expert placement, so its warmth is
  unproven"). A missing log → FAIL.
- `--allow-cold` keeps its old meaning: the same detail, as a WARN, verdict untouched.
- Checked against the real artefacts (read-only):
  `perf-mtp-ref-318-20261029-222244.log` → FAIL, line 2 `mtp.layers.0.mlp [136..512) of 512`.
  T06's own step-1 run `perf-rb-ab-166-20261029-202655.log` has the same shape (49 lines, 48 warm,
  the cold one is `mtp.layers.0.mlp [152..512) of 512`).

### 2. Percentage gate reads the numbers (`percentage_gate(text)`)

- `±N %` (including the range form `±5–10 %` seen in T04) → **FAIL**, message names the C5 one-sided
  rule and points at the ratio table's min/max.
- A decimal interval (`\d+\.\d{2,}` on both sides of `-`/`–`/`—`, spaces and parens allowed) with
  no endpoint above `RATIO_SCALE = 1.2` → treated as a ratio-band quote:
  - no ratio table in the report → **FAIL** "quote without a table - regenerate with
    `bench/ratios.py`";
  - table present → **PASS** only if `|lo − min| ≤ 0.005` and `|hi − max| ≤ 0.005`; otherwise
    **FAIL** quoting both table numbers ("the table's ratios run 0.92 to 0.99").
- Ratio table: a block of markdown rows whose **last** cell is a ratio (`[01]\.\d{2,3}`, which is
  what `bench/ratios.py` prints: `| band | ref | var | ratio |`). Every column of that block that is
  a ratio in every row is a candidate (see Interpretation A).
- Every other `\d+ %` → `  INFO  ... is prose, not a ratio band - not checked`; never PASS, never a
  verdict change. Spans already judged by the `±`/interval rules are skipped so one number is not
  reported twice.
- The old unconditional PASS (`if re.search(r"±|\b\d+…%")`) is gone.

### 3. `--draft` pass-through (`size_check_argv(state)`)

`pre()` now calls `sc.main(size_check_argv(state))`; the function returns
`--model <wsl path> --mcs <split> --cs <cache_size> --cq <cache_quant>` plus `--draft mtp` when
`state["active"]["draft"]["mode"] == "mtp"`. Live effect on the repo state:

```
config     -mcs 376  -cs 131072  -cq 2,2  slots 2  draft mtp
  draft          337 MiB VRAM         668 MiB arena   303 MiB cache  (1 layer(s))
  TOTAL        19812 MiB VRAM of 24564  -> 2704 MiB free
```

(vs 19172 MiB without the head — the 640 MiB VRAM and 668 MiB arena the old check never counted.)

### 4. Split-brain gate (`deployed_state_gate(repo_doc, deployed, label)`)

- New flag `--deployed`, default `os.path.dirname(RESULTS_DEFAULT)` = `_agent/v165`; `''` disables.
- `<deployed>/state.json` missing → `  INFO  ... no deployed copy to disagree with`, verdict
  untouched. Present and equal (compared as **parsed JSON**) → PASS. Present and different → FAIL
  naming the differing dotted paths (`active.split`) and ending `run bench/windows/deploy.sh`.
- No byte or hash comparison, so a CRLF re-checkout is not a difference.

### 5. Arm-row gate (`arm_runs`, `arm_gate`, `row_runner`)

- The runner of a `passes.csv` row is read from the `runner` column, or — because the box's
  passes.csv was created before that column existed and its header is still 20 columns — from
  DictReader's overflow list, which is where the appended values land.
- A cited run with `runner=spec_decode` and no arm row is a **FAIL** ("the run died before producing
  an arm, so it is not a measurement"). Arm evidence is unioned over `arms.csv` and
  `spec_decode.csv` (Interpretation B). Neither file present at all + a cited spec_decode run →
  FAIL, not a silent pass.

### 6. Tests (18 added, 27 total, stdlib `unittest`, tempfile fixtures)

`WarmPlacementGate` (3): every-line-warm PASS; the real 48-warm/1-cold shape FAIL and the offending
line named; a log with no placement line FAIL.
`PercentageGate` (5): `0.92–0.99` beside a 0.92–0.99 table PASS **and the report is byte-identical
afterwards**; `(0.95-0.99)` beside that table FAIL quoting 0.92/0.99; `±5 %` FAIL; interval with no
table FAIL naming `ratios.py`; `acceptance 41 %` → INFO, rc 0, `verdict: all gates passed`.
`ArmRowGate` (4): no arm row FAIL (file exists, covers another run); arm row PASS; the archive's
overflow-cell runner FAIL; `spec_decode.csv` alone counts as evidence.
`DeployedStateGate` (4): differing JSON FAIL naming `active.split` + `deploy.sh`; absent → PASS/INFO;
CRLF + reordered keys → PASS; `--pre` itself prints the deployed FAIL (wiring).
`SizeCheckArgv` (2): MTP state gets `--draft mtp`, the WSL path is still translated, `--mcs` once;
`draft.mode != mtp` and a missing `draft` block emit no `--draft`.

## Commands and output (real, this box)

### `python3 bench/test_gates.py -v` → rc 0

```
test_run_with_arm_rows_passes (__main__.ArmRowGate.test_run_with_arm_rows_passes) ... ok
test_run_without_arm_rows_fails (__main__.ArmRowGate.test_run_without_arm_rows_fails) ... ok
test_runner_is_read_from_the_overflow_cell (...) ... ok
test_spec_decode_csv_alone_is_arm_evidence (...) ... ok
test_absent_deployed_state_passes (...) ... ok
test_deployed_state_differing_fails (...) ... ok
test_key_order_and_line_endings_are_not_a_difference (...) ... ok
test_pre_consults_the_deployed_copy (...) ... ok
test_interval_without_a_table_fails (...) ... ok
test_plain_percentage_only_informs (...) ... ok
test_quote_disagreeing_with_the_table_fails (...) ... ok
test_quote_matching_the_table_passes (...) ... ok
test_symmetric_band_fails (...) ... ok
test_both_callers_use_the_one_reader (...) ... ok
test_every_prompt_parses (...) ... ok
test_stringified_arguments_are_decoded (...) ... ok
test_mtp_regime_asks_for_the_draft_head (...) ... ok
test_no_draft_regime_leaves_it_off (...) ... ok
test_arm_count_follows_the_arm_blocks (...) ... ok
test_dynamic_placement_reads_the_registry_and_the_modules (...) ... ok
test_main_wires_the_gate_in (...) ... ok
test_multi_arm_with_dynamic_placement_is_refused (...) ... ok
test_multi_arm_with_static_placement_is_allowed (...) ... ok
test_one_arm_is_allowed (...) ... ok
test_every_line_warm_passes (...) ... ok
test_log_with_no_placement_line_fails (...) ... ok
test_one_cold_component_fails_and_names_it (...) ... ok

Ran 27 tests in 0.041s

OK
```

### `python3 bench/verify_task.py --pre` → rc 0 (`verdict: all gates passed`)

```
pre-run gates
  PASS  state.json: model D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8, split 376, cs 131072, cq 2,2
  PASS  deployed state.json matches /home/dev/.local/share/opencode/worktree/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf/bench/windows/state.json (parsed, not bytes)
  WARN  installed engine blocks -or with a draft model (fine unless this task needs it)
  PASS  frozen expert profile present (4 file(s))
model      /mnt/d/llms/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8
config     -mcs 376  -cs 131072  -cq 2,2  slots 2  draft mtp
  trunk        17989 MiB VRAM       32937 MiB arena
  cache         1183 MiB VRAM   (KV 960 over 12 layers + recurrent 223)
  draft          337 MiB VRAM         668 MiB arena   303 MiB cache  (1 layer(s))
  ngram        49439 MiB streamed from disk (not resident)
  TOTAL        19812 MiB VRAM of 24564  -> 2704 MiB free
  reserve       2048 MiB held by the autosplit (counted against the card)
               33605 MiB host arena of 63600  -> 29995 MiB free
verdict    FITS (needs >= 1024 MiB VRAM headroom)
verdict: all gates passed
```

Edge cases of the new flag, smoke-tested:
`--deployed ''` → `INFO deployed state.json check disabled (--deployed '')`;
`--deployed /tmp/opencode/t4` (no state.json there) → `INFO deployed state.json absent at
/tmp/opencode/t4/state.json: no deployed copy to disagree with`.

### `python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md` → rc 1

The report was not edited.

```
post-run gates
  FAIL  rb-ab-166-20261029-202655: 1 of 49 CPU-split component(s) have no saved placement, so their placement was computed fresh and is not comparable:
          -- CPU split experts (worker, dynamic): mtp.layers.0.mlp [152..512) of 512
  PASS  rb-ab-166-20261029-202655: arm rows present in arms.csv or spec_decode.csv
  FAIL  `±5 %` is a symmetric band; this project reports one-sided ratio intervals (C5) - point at the ratio table's min/max instead
  FAIL  `0.87–1.10` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  FAIL  `0.47–1.04` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  PASS  `0.96–1.09` matches this report's own ratio table
  INFO  `1.14–2.01` is not a ratio band (an endpoint is above 1.2) - not checked against the ratio table
  INFO  `1.01–1.68` is not a ratio band (an endpoint is above 1.2) - not checked against the ratio table
  INFO  `1.12–2.41` is not a ratio band (an endpoint is above 1.2) - not checked against the ratio table
  INFO  `13 %` is prose, not a ratio band - not checked
  INFO  `1 %` is prose, not a ratio band - not checked
  INFO  `35 %` is prose, not a ratio band - not checked
  INFO  `35 %` is prose, not a ratio band - not checked
  INFO  `33 %` is prose, not a ratio band - not checked
  INFO  `84 %` is prose, not a ratio band - not checked
  INFO  `84 %` is prose, not a ratio band - not checked
  INFO  `40 %` is prose, not a ratio band - not checked
  INFO  `42 %` is prose, not a ratio band - not checked
verdict: CONTRADICTIONS FOUND
```

Reading: the per-line gate fires on the **run T06 cites in its own step 1** (`-rb_ab -mtp` screening
run), whose MTP head is cold exactly like the reference log's; the `±5 %` claim it makes about
cross-process baseline spread now fails instead of being certified; two category-ratio ranges quoted
from the archive do not equal the report's own ratio table. The one PASS is a quote that does equal
it (`0.96–1.09`).

### Cold reference itself (one-off, read-only, not a test fixture)

```
$ python3 -c "... verify_task.warm_gate('/mnt/c/.../logs/perf-mtp-ref-318-20261029-222244.log', ...)"
  FAIL  mtp-ref-318-20261029-222244: 1 of 49 CPU-split component(s) have no saved placement, ...
          -- CPU split experts (worker, dynamic): mtp.layers.0.mlp [136..512) of 512
```

### Bonus evidence — `--post` on `T04-armA.md` (rc 1)

```
  FAIL  `±2 %` / `±5–10 %` / `±11 %` / `±5 %` / `±5 %` / `±9 %` is a symmetric band; ...
  FAIL  `0.95–0.99` does not match this report's ratio table: the table's ratios run 0.92 to 0.99
  FAIL  `0.91–1.01` does not match this report's ratio table: the table's ratios run 0.92 to 0.99
  PASS  `0.97–1.11` matches this report's own ratio table
```

`0.95–0.99` is the exact string in this script's own docstring (quoted band vs a table that starts
at 0.92) — the gate now finds it unaided.

## Interpretations (the two places the brief was not mechanical)

**A. Which column is "the ratio column".** The brief defines the table as "markdown rows whose last
cell is a ratio like `0.92`" and then says "take its ratio column's min/max". A report can print two
ratio columns in one table — T04's band table is `| band | pp ratio | tg ratio |`, and quoting only
the pp column against the tg column's min/max would fail a correct quote for the wrong reason. I
kept the brief's table definition verbatim (last cell must be a ratio) and made every column of that
block that is a ratio in every row a candidate; a quote passes if it matches **any** candidate. On
T04 this yields the right pairs (pp 0.95–0.99 vs pp 0.92–0.99, tg 0.91–1.01 vs tg 0.93–1.00).
If the controller wants the literal single-column rule instead, it is one loop bound in
`ratio_columns`.

**B. Arm evidence when both tables exist.** The brief's fallback is file-level ("or, when that file
is absent, `spec_decode.csv`"). On this box `arms.csv` exists but covers only
`val-spec-20261029-233352`: the older drafting runs (`rb-ab-166-20261029-202655`,
`mtp-ref-318-…`, `mtp-dnt3-368-…`) printed no `ARM` line and live only in `spec_decode.csv`. A
literal file-level rule would therefore fail five correctly measured archived runs while still
catching the real defect (a run that dies before its first arm is in **neither** table). The gate
unions both files per run; `ArmRowGate.test_spec_decode_csv_alone_is_arm_evidence` pins that
reading.

**Note (not an ambiguity):** a decimal interval is treated as a ratio-band quote only when no
endpoint exceeds 1.2, because T06's acceptance counts (`1.14–2.01 / 4`) and throughput ranges
(`35.2–38.8 t/s`) are not pp/tg ratios and calling them "quotes without a table" would be wrong.
Intervals above that bound are printed as INFO rather than silently ignored.

## Self-review

Completeness against the six requirements:

1. per-line warm gate — done, verbatim lines, empty/missing log FAIL, no-cap reporting.
2. percentage gate — done: ±, table equality with both numbers in the failure, no-table FAIL, INFO
   for prose, no PASS for an unchecked number.
3. `--draft mtp` via `size_check_argv(state)` — done, tested.
4. `--deployed` with parsed-JSON comparison, INFO when absent, `deploy.sh` remedy — done, tested,
   default is `_agent/v165`, `''` disables.
5. the ten listed test cases plus 5 more (arm gate positives/negatives, overflow-cell runner,
   `--pre` wiring, CRLF/key-order) — done, fixtures only, `--results`/`--logs`/`--deployed`/
   `--state` overrides, no network, no pytest.
6. arm-row gate — done with a fixture whose `passes.csv` and `arms.csv` disagree.

Quality:

- Regexes are anchored: `RATIO_INTERVAL` has `(?<![\w.])` / `(?![\w.])`, so `x0.92–0.99`, run ids
  (`mtp-ref-318-20261029-222244`) and versions (`1.6.5-1.6.6`) do not match; `SYMMETRIC_PCT`
  requires the `±`; a percent glued to a word (`MOS 0.92%-0.99%`) is a `%` mention → INFO, not a
  band. Verified against those strings directly.
- JSON comparison is parsed and order-independent; the CRLF case is a test, and the message names
  the differing dotted paths so the operator knows what to deploy.
- No new flags besides `--deployed`; `--allow-cold` keeps its meaning; nothing else in either file
  was restructured. Two unclosed-file reads inside the functions I touched became `with` blocks
  (one pre-existing, needed to keep the test run warning-free).
- `--post` never writes: `PercentageGate.test_quote_matching_the_table_passes` asserts the report is
  byte-identical after the run.

Testing (mutation matrix — each mutation applied to `verify_task.py`, suite re-run, then restored):

| mutation | caught by |
|---|---|
| warm: warm if **any** line is placed (old whole-log substring) | `test_one_cold_component_fails_and_names_it` |
| warm: gate not called from `post()` | 3 warm tests |
| percent: the old no-op PASS restored in `post()` | 5 percentage tests |
| percent: interval rule removed | `test_interval_without_a_table_fails`, `test_quote_disagreeing_with_the_table_fails`, `test_quote_matching_the_table_passes` |
| percent: `±` accepted | `test_symmetric_band_fails` |
| percent: interval accepted without a table | `test_interval_without_a_table_fails` |
| draft: `--draft` left off the argv | `test_mtp_regime_asks_for_the_draft_head` |
| deployed: byte compare instead of parsed JSON | 3 deployed tests (incl. the CRLF case) |
| deployed: gate not called from `pre()` | `test_pre_consults_the_deployed_copy` |
| arm: gate not called from `post()` | 4 arm tests |
| arm: `arms.csv` only, no `spec_decode.csv` fallback | `test_spec_decode_csv_alone_is_arm_evidence` |

## Concerns

1. **Interpretations A and B above** are the only places I chose a reading over the letter of the
   brief. Both are documented in the code docstrings and pinned by tests; if the controller prefers
   the literal single-column rule or the file-level arm fallback, each is a small, test-visible
   change.
2. **T06 now fails on its own step-1 run's cold MTP head.** That is the intended behaviour (the head
   is the component the drafting arms are compared against), but it means T06's step-1 readback
   conclusions rest on a run whose head placement was fresh; the report already flags the reference
   run as cold and does not yet flag this one. I did not touch the report.
3. `warm_gate` prints every offending line; an entirely cold 49-line log therefore prints 49 lines.
   Deliberate (truncation would hide part of the evidence) but noisy in that case.
4. `test_pre_consults_the_deployed_copy` runs the real `--pre`, so it reads the box's venv
   `model_init.py` path and probes `D:\llms` — read-only, no process start, and the assertion covers
   only the deployed-gate line, so the test does not depend on what those paths contain.
5. The `%`-in-prose INFO lines are repeated per occurrence (T06 prints `35 %` and `84 %` twice
   each). Intentional: one line per claim, no dedupe.

## Commands used for verification (all offline)

```
python3 bench/test_gates.py -v                     # 27 tests, OK
python3 bench/verify_task.py --pre                 # verdict: all gates passed
python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md   # CONTRADICTIONS FOUND
python3 -m py_compile bench/verify_task.py bench/test_gates.py
```

---

# Fix round 1 — two gate defects found in review (commit `1c5b9670`)

Both findings reproduced against `ffec2e50` with the strings given, fixed, and covered by tests.
Tests are now **34** (`34 tests, OK`).

## Finding 1 — `ratio_columns` split a ratio table at a non-ratio cell (both directions)

Reproduced on `ffec2e50` with the shape `bench/ratios.py`'s `fmt()` produces (`-` for a missing
ratio):

```
| band | ratio |
|---|---|
| 256 | 0.85 |
| 512 | 0.92 |
| 1024 | - |
| 2048 | 0.96 |
| 4096 | 0.99 |
```

```
--- ratio_columns = [(0.85, 0.92), (0.96, 0.99)]
quote 0.96–0.99  →  PASS  `0.96–0.99` matches this report's own ratio table     # wrong: column runs 0.85-0.99
quote 0.92–0.99  →  FAIL  `0.92–0.99` does not match ... the table's ratios run 0.96 to 0.99   # wrong: 0.85-0.99
```

(The reported single-row string `| 256 | 0.85 | 512 | 0.92 | 1024 | - | 2048 | 0.96 | 4096 | 0.99 |`
gives `[(0.85, 0.85), (0.92, 0.92), (0.96, 0.96), (0.99, 0.99)]` on `ffec2e50` — one singleton per
cell, no column at all. Both shapes come from the same defect: the block is built from rows, and a
row that is not a ratio row either ends it or fails the all-rows test. The fix addresses both.)

**Fix.** `markdown_cells()` no longer discards the `|---|---|` separator (so a separator cannot end a
table either), and `ratio_columns()` now builds tables from contiguous markdown rows, then takes
`rows = [r for r in table if RATIO_CELL.fullmatch(r[-1])]` and the min/max per column **over those
rows**. Only a non-table line ends a table.

```
--- ratio_columns = [(0.85, 0.99)]
quote 0.85–0.99  →  PASS  `0.85–0.99` matches this report's own ratio table
quote 0.96–0.99  →  FAIL  `0.96–0.99` does not match ... the table's ratios run 0.85 to 0.99
quote 0.92–0.99  →  FAIL  `0.92–0.99` does not match ... the table's ratios run 0.85 to 0.99
```

Tests: `test_quote_of_the_whole_range_passes_across_a_missing_ratio` (true range passes),
`test_sub_block_range_does_not_pass` (0.96–0.99 fails, both real endpoints 0.85 and 0.99 named).

## Finding 2 — a sentence-final interval was silently unchecked

Reproduced: `"the pp ratio was 0.92–0.99."` matched nothing (`RATIO_INTERVAL.finditer` → `[]`),
so there was no FAIL and no INFO, and requirement 2's "interval with no table → FAIL" never fired
for the most natural prose form. Cause: the trailing guard `(?![\w.])` rejects a following full
stop.

**Fix.** Guard is `(?!\w)(?!\.\d)`: a following word character is still refused, and so is a
following `.digit`, so a partial number cannot be certified as a 2-decimal band.

```
sentence-final, no table   →  FAIL  `0.92–0.99`: quote without a table - regenerate with `bench/ratios.py` ...
sentence-final, with table →  PASS  `0.92–0.99` matches this report's own ratio table
"0.92–0.995."              →  matches ["0.92–0.995"], never "0.92–0.99"
```

Tests: `test_sentence_final_interval_without_a_table_fails`,
`test_sentence_final_interval_matching_the_table_passes`,
`test_a_longer_decimal_is_not_truncated_to_a_band`.

## The two "while you are there" items

- **`arm_gate`'s "neither CSV present" branch** is now covered:
  `ArmRowGate.test_no_arm_table_at_all_fails` (no `arms.csv`, no `spec_decode.csv`, cited
  `runner=spec_decode` run → FAIL `no arms.csv and no spec_decode.csv`).
- **`RATIO_CELL`** is now `[01]\.\d{1,3}` (1–3 decimals), so a ratio column printed with one decimal
  (`0.9`, `1.0`) is not silently dropped out of the candidate set, and the docstring says which
  widths are recognised and what is excluded (a band number, a t/s rate like `45.76`, a MiB figure
  like `2141`). New test: `test_one_decimal_ratio_column_is_recognised`.

## Verification after the fix

```
$ python3 bench/test_gates.py -v
Ran 34 tests in 0.052s
OK

$ python3 bench/verify_task.py --pre
...
verdict: all gates passed          (rc 0)

$ python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md
post-run gates
  FAIL  rb-ab-166-20261029-202655: 1 of 49 CPU-split component(s) have no saved placement, ...
          -- CPU split experts (worker, dynamic): mtp.layers.0.mlp [152..512) of 512
  PASS  rb-ab-166-20261029-202655: arm rows present in arms.csv or spec_decode.csv
  FAIL  `±5 %` is a symmetric band; ...
  FAIL  `0.87–1.10` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  FAIL  `0.47–1.04` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  PASS  `0.96–1.09` matches this report's own ratio table
  ... (INFO lines for prose %, `1.14–2.01`, `1.01–1.68`, `1.12–2.41` unchanged)
verdict: CONTRADICTIONS FOUND     (rc 1)
```

Identical lines to `ffec2e50` — the fix changes the wrong-range and sentence-final cases, not this
report. `T04-armA.md` output is also unchanged (same 6 `±` failures, `0.95–0.99` vs table
0.92–0.99, `0.91–1.01` vs 0.93–1.00). The report was not edited.

## Mutation check of the new tests (on a `/tmp` copy of the fixed file)

| mutation | result |
|---|---|
| fix 1 reverted (non-ratio row ends the table) | rc 1 — `test_quote_of_the_whole_range_passes_across_a_missing_ratio`, `test_sub_block_range_does_not_pass` |
| fix 2 reverted (`(?![\w.])`) | rc 1 — `test_sentence_final_interval_without_a_table_fails`, `test_sentence_final_interval_matching_the_table_passes`, `test_a_longer_decimal_is_not_truncated_to_a_band` |
| `RATIO_CELL` reverted to 2–3 decimals | rc 1 — `test_one_decimal_ratio_column_is_recognised` |
| interval quantifier fixed at 2 decimals (truncates `0.995` to `0.99`) | rc 1 — `test_a_longer_decimal_is_not_truncated_to_a_band` |
| `arm_gate` "neither CSV" branch turned into a pass | rc 1 — `test_no_arm_table_at_all_fails` |

The original 11-mutation matrix was re-run on the fixed file and still fails, now with the larger
test counts (e.g. the old no-op `%` PASS is caught by 11 tests, "no interval rule" by 9).

## Concerns added by this round

1. **`T05-memops.md`'s table is not detected as a ratio table.** Its shape is
   `| band | pp median | pp range | tg median | tg range |`, i.e. the last cell is a *range*
   (`0.850–0.862`), not a bare ratio, so the brief's table definition ("rows whose last cell is a
   ratio") does not recognise it and its 13 quoted per-band ranges read as "quote without a table".
   That is the literal rule, not the defect fixed here; accepting a range cell as a ratio cell is a
   design change I did not make. If the controller wants T05 to be checkable rather than merely
   INFO'd-free, it is a small change to the qualifying-row test.
2. The table's `rows` set is the ratio-row subset, so a table that prints `-` in the *last* column
   for some bands contributes only the ratios it printed — correct for a min/max, but it cannot tell
   "0.85 over 5 bands" from "0.85 over 4 of 5".
