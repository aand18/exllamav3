# Final fix report — seven findings from the whole-branch review

Branch `165-win-perf`, base `dfa36468`, head `09e3b084`. Plan of record:
`wiki/plans/v165-critique-fixes.md` (no GPU pass, stdlib-only tests, nothing installed or killed on the
host). Every command below was run from the worktree
`/home/dev/.local/share/opencode/worktree/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf`.

## Commits

| SHA | Subject | Findings |
|---|---|---|
| `7e49fbb2` | `bench: percentage gate reads a one-decimal band and names a table it cannot match` | F1, F7 (+ two new test concerns in `bench/test_gates.py`) |
| `3fa1174e` | `bench: -DryRun skips the one-owner gate and creates no output directory` | F2 (`run_pass.ps1` + README's `-DryRun` row and prose) |
| `16998ea9` | `bench: a drafting pass writes no max_length cell, because it got no max_length` | F4 (+ a pin test) |
| `e628c5af` | `bench: README names the remedy the -Extra guard prints, and -dds in the drafting block` | F5, F6 |
| `a38f20ac` | `plan: tracker says what D4 tested, cites the comment's lines, names the docs commits` | F3, `:955` citation, C-section commits line |
| `a30abcc8` | `bench: a band that carries a unit is a measurement, not a ratio` | residual finding against F1 (wave 2, below) |
| `09e3b084` | `bench: the percentage gate audits prose, not the table it checks against` | residual finding against F1/F7 (wave 3, below) |

F1 and F7 share one commit: both are the same function's handling of a quoted band, and the two edits
land in the same branch of `percentage_gate()` — splitting them would have produced a commit that is
not independently reviewable.

---

## F1 — the percentage gate was blind to a one-decimal band

`bench/verify_task.py`

- `RATIO_INTERVAL` now accepts **one or more** decimals per endpoint
  (`(\d+\.\d+)\s*[–—-]\s*(\d+\.\d+)`, keeping the `(?<![\w.])` / `(?!\w)(?!\.\d)` guards), so
  `0.9–1.0`, `0.87–1.1` and `0.92–0.995` are all read as one interval and `0.92–0.995` is still not
  truncated to `0.92–0.99`.
- Comparison is now at the quoted precision, not at a fixed 0.005: `ratio_endpoint_matches()` rounds
  the table's endpoint to the number of decimals the quote carries with `Decimal.quantize(...,
  ROUND_HALF_UP)` and requires equality. Stated in the function's docstring, with the reason a flat
  0.005 tolerance is wrong: at one decimal `0.87` is exactly 0.05 from `0.92`, so a tolerance would let
  the very quote this finding reproduces through. Half-up (not banker's) because a human writing
  `0.9` means `0.92 → 0.9`, and `0.85 → 0.8` would be a false FAIL.
- `RATIO_TOL` is gone (no other user).

Behaviour: `0.9–1.0` PASSes against the 0.92–0.99 table; `0.87–1.1` and `0.9–1.1` FAIL naming `0.92`
and `0.99`; the existing 2-decimal cases are unchanged (`0.92–0.99` PASS, `(0.95-0.99)` FAIL on a
0.92–0.99 table, `0.85–0.99` PASS across a `-` row, `0.96–0.99` FAIL against a 0.85–0.99 table).

## F7 — two different failures had one message

> **Case 2 is gone in wave 3 (`09e3b084`).** Table rows are no longer scanned at all, so a range cell
> can never be read as a quote and the third wording had nothing left to say — it was removed with the
> exclusion. The two messages the finding asked for (no table / no matching range) are what the gate
> prints now. The history below is kept as the wave-1 record.

- `table_ratio_ranges()` collects the ratio ranges the report's **tables** print (`0.915–1.034`), which
  is the shape `ratio_columns()` cannot turn into a min/max band.
- The no-band branch distinguished three cases, all FAIL (the verdict is unchanged — a quote
  nothing certifies is not evidence):
  1. no ratio ranges in any table → `` `0.92–0.99`: quote without a table - no ratio table in the
     report; regenerate with `bench/ratios.py` … `` (the old sentence, plus the words "no ratio
     table" so the two cases are told apart in the text);
  2. the quote **is** one of those range cells → `` `0.915–1.034` is a range the report's own table
     prints, not a band quoted in prose - this gate compares a quote with a ratio column's min/max``.
     This third wording is a deliberate addition: with only the two required messages, a table cell
     read as a quote produced "matches no range in the report's table(s) (ranges: 0.915–1.034, …)" —
     a message that lists the quote as absent, i.e. a message contradicting itself. It is still a
     FAIL.
  3. otherwise → `` `0.958–1.001` matches no range in the report's table(s) (ranges: 0.915–1.034,
     0.850–0.862, 0.946–0.971, 0.796–0.943 (+8 more)): this gate compares a quote with a ratio
     column's min/max, and those cells are ranges`` — the T05 case the finding names.

## F2 — `-DryRun` aborted in the case its own documentation cites

`bench/windows/run_pass.ps1`

- The one-owner gate (`Get-CimInstance … perf\.py|spec_decode\.py|tabbyAPI` → exit 3) is now inside
  `if (-not $DryRun)`. It is unchanged for a real pass: a dry run starts nothing, so there is no owner
  to protect, and a spelling check is exactly what an operator does while the card is busy.
- `New-Item -ItemType Directory -Force $LOGS, $RES` moved out of the state-load block (was line 133)
  to the top of the `--- Run ---` section, i.e. after the dry-run exit. Creating two directories is a
  write; the "writes nothing" claim is now literal. `Set-Location $Root` stays where it was (trap 2:
  it must precede every Python call).
- The dry run still runs every other gate — state file, `.active.model`, model present, the venv
  python/perf/watcher presence, engine/package/extension/torch identity, `$pkg` must be
  site-packages — and still prints `DRYRUN` / `ARGS` / `ENV`.
- Documentation: `bench/windows/README.md`'s `-DryRun` table row and the prose sentence at :94 now say
  the one-owner check is skipped and that no output directory is created.

## F3 — the tracker overstated what was tested

`wiki/plans/v165-critique-implementation.md`, D4 status cell, now says exactly this: the pure gate and
its wiring (`swap_refusal`, `count_arms`, `dynamic_placement`, the `main()` block) **and**
`verify_task.arm_gate` are covered by `bench/test_gates.py`; the `run_pass.ps1` verdict branch **ran
nowhere**, verified by `Parser::ParseFile` plus a code read; no multi-arm run has loaded a model since.
The D4 evidence cell's citation is corrected: the "visibly derails a generation in progress" comment is
`block_sparse_mlp_cpu.py:957-960` (checked in this worktree; `:955` is
`if self._swap_tick_count < interval:`, the `EXL3_MOE_CPU_SWAP_INTERVAL` guard, and the interval is
read at `:954`). Section C's intro now names the two commits that wrote the documentation:
`7c8754e2` (`bench/windows/README.md`) and `dfa36468` (the tracker itself).

## F4 — `max_length` carried a value the drafting run never received

`bench/windows/run_pass.ps1`: the `passes.csv` cell is now `$maxLengthCell`, i.e. `$MaxLength` for a
`perf` ladder and an **empty** cell for `-Runner spec_decode`, whose argv had `-max_length` stripped
(and whose dry-run `ARGS` proves it: see below). The drafting length is the `-cs` in the `args` cell,
which is where `ratios.py`'s provenance table already shows it. Comment added above the row.
Pinned by `PassesCsvCells.test_drafting_row_writes_no_max_length_it_never_received` (the row formatter
must print the runner-aware cell, never `$MaxLength`; the blanking rule must exist). That test is a
source-shape pin, not a CSV unit test — there is no PowerShell test rig, and the writer only runs after
`Start-Process`.

## F5 — the `-Extra` paragraph promised a remedy the code does not print

`bench/windows/README.md:65` now says what `$regimeRemedy` does: the abort names the flag and the way
to change it — the delta switch where one exists (`-Model`, `-CacheSize`, `-MaxLength`, `-Split`,
`-Threads`), `edit state.json` for `-cq`, `-chunk_size`, `-ambs` and the drafting flags. Verified
against the deployed script (four aborts below).

## F6 — the drafting paragraph omitted `-dds`

`bench/windows/README.md`'s drafting paragraph now lists `dynamic_draft: true` → `-dds`. The state on
the box has `"dynamic_draft": false`, which is why the spec dry run below carries `-mtp -ndt 2` and no
`-dds` — the mapping is documented for when it is true, and the code path is `run_pass.ps1:201`.

---

## Verification

### Tests

```
$ python3 bench/test_gates.py -v
...
Ran 40 tests in 0.059s
OK            (rc=0, 40 of 40 "... ok")
```

34 before this dispatch, 40 now. New cases:

| test | finding |
|---|---|
| `PercentageGate.test_one_decimal_quote_matching_the_table_passes` | F1 (`0.9–1.0` vs a 0.92–0.99 table PASSes) |
| `PercentageGate.test_one_decimal_quote_outside_the_table_fails` | F1 (`0.87–1.1` **and** `0.9–1.1` FAIL, message names 0.92 and 0.99) |
| `PercentageGate.test_one_decimal_interval_is_read_as_a_band` | F1 (the regex reads `0.9–1.0` as one interval) |
| `PercentageGate.test_quote_against_a_range_column_names_the_ranges` | F7 ("matches no range in the report's table(s)", ranges listed, not "quote without a table") |
| `PercentageGate.test_a_table_range_read_as_a_quote_is_not_called_a_missing_table` | F7 (the third message; no self-contradiction) |
| `PassesCsvCells.test_drafting_row_writes_no_max_length_it_never_received` | F4 |

`test_interval_without_a_table_fails` was strengthened to require the words "no ratio table" — the half
of F7 that must not change. The F1 cases that already existed still pass unchanged
(`test_a_longer_decimal_is_not_truncated_to_a_band`, `test_one_decimal_ratio_column_is_recognised`,
`test_quote_matching_the_table_passes`, `test_quote_disagreeing_with_the_table_fails`,
`test_quote_of_the_whole_range_passes_across_a_missing_ratio`, `test_sub_block_range_does_not_pass`,
the sentence-final pair).

### Mutation evidence (copy in `/tmp/opencode/mut`, never in the repo)

The copy is `bench/{verify_task,test_gates,size_check}.py` + `eval/spec_decode.py` + `eval/prompts/`, so
`test_gates.py` resolves its `REPO` inside the copy. Baseline there: `Ran 16 tests … OK`.

Mutation A — revert F1's regex to `\d{2,}`:

```
FAIL: test_one_decimal_interval_is_read_as_a_band
FAIL: test_one_decimal_quote_matching_the_table_passes
FAIL: test_one_decimal_quote_outside_the_table_fails (quoted='0.87–1.1')
FAIL: test_one_decimal_quote_outside_the_table_fails (quoted='0.9–1.1')
Ran 16 tests … FAILED (failures=4)
```

and the failure body is the reproduced defect itself: `AssertionError: 0 != 1 : … verdict: all gates
passed` for `0.9–1.1` beside a 0.92–0.99 table.

Mutation B — restore the single "quote without a table" branch (F7's before-state):

```
FAIL: test_a_table_range_read_as_a_quote_is_not_called_a_missing_table
FAIL: test_interval_without_a_table_fails
FAIL: test_quote_against_a_range_column_names_the_ranges
Ran 16 tests … FAILED (failures=3)
```

Both mutations were reverted; the copy passes again (`Ran 16 tests … OK`).

### Gates on the real reports

```
$ python3 bench/verify_task.py --pre
pre-run gates
  PASS  state.json: model D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8, split 376, cs 131072, cq 2,2
  PASS  deployed state.json matches …/bench/windows/state.json (parsed, not bytes)
  WARN  installed engine blocks -or with a draft model (fine unless this task needs it)
  PASS  frozen expert profile present (4 file(s))
…
verdict    FITS (needs >= 1024 MiB VRAM headroom)
verdict: all gates passed                                        rc=0
```

```
$ python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md   # rc=1
  FAIL  rb-ab-166-20261029-202655: 1 of 49 CPU-split component(s) have no saved placement …
          -- CPU split experts (worker, dynamic): mtp.layers.0.mlp [152..512) of 512
  PASS  rb-ab-166-20261029-202655: arm rows present in arms.csv or spec_decode.csv
  FAIL  `±5 %` is a symmetric band; this project reports one-sided ratio intervals (C5) …
  FAIL  `0.87–1.10` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  FAIL  `0.47–1.04` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  INFO  `1.0–2.2` is not a ratio band (an endpoint is above 1.2) - not checked against the ratio table
  PASS  `0.96–1.09` matches this report's own ratio table
  FAIL  `0.6–0.8` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  … 3 more INFO ratios, 10 INFO percentages
verdict: CONTRADICTIONS FOUND
```

Diff against the pre-fix script (`git show dfa36468:bench/verify_task.py`) on the same report: F1 adds
one FAIL (`0.6–0.8`) and three INFO lines (`1.0–2.2`, `1.2–2.4`, `35.2–38.8`). Nothing that used to
fail now passes.

> **Superseded for `0.6–0.8` by wave 2 (`a30abcc8`).** That FAIL was a false positive: the range is
> followed by the unit `GB` ("Generator costs ~0.6–0.8 GB that never comes back"), so it is a host
> memory measurement, not a ratio quote. Wave 2's exclusion rule turns that line into an INFO and the
> report's FAIL count back to four — the two ratio bands stay FAILs. The numbers above are kept as the
> wave-1 record; see "Wave 2" at the end for the current output.

```
$ python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T05-memops.md   # rc=1
  FAIL  `±11 %` is a symmetric band; …
  FAIL  `±2 %` is a symmetric band; …
  FAIL  `0.915–1.034` is a range the report's own table prints, not a band quoted in prose - this
        gate compares a quote with a ratio column's min/max
  … 11 more of that shape …
  FAIL  `0.83–0.93` matches no range in the report's table(s) (ranges: 0.915–1.034, 0.850–0.862,
        0.946–0.971, 0.796–0.943 (+8 more)): this gate compares a quote with a ratio column's
        min/max, and those cells are ranges
  FAIL  `0.958–1.001` matches no range in the report's table(s) (… same ranges …)
  INFO  `17 %` / `11 %` are prose, not a ratio band - not checked
verdict: CONTRADICTIONS FOUND
```

Before the fix the same report printed **14 × "quote without a table"** (`rc=1`); now every one of
them names the ranges the report actually prints. The verdict is unchanged, as required.

> **Superseded by wave 3 (`09e3b084`).** Twelve of those fourteen FAILs were the table's own range
> cells (`T05-memops.md:18` and the rows below it), audited as if they were prose. Table rows are no
> longer scanned, so T05 now prints 4 FAILs — the two `±` bands and the two genuine prose claims
> (`medians 0.83–0.93`, `medians 0.958–1.001`), which stay checked. The block above is the wave-1
> record; see "Wave 3" at the end for the current output.

### `-DryRun` and the deployed copy

```
$ bench/windows/deploy.sh
state.json: unchanged
deployed to /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165

$ powershell.exe … Parser::ParseFile('…\v165\run_pass.ps1')
parse OK: C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1 (4913 tokens)

$ ls -1 $D/logs | wc -l ; ls -1 $D/results | wc -l      # before
106
144

$ powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
    "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-final -DryRun
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
rc=0

$ ls -1 $D/logs | wc -l ; ls -1 $D/results | wc -l      # after
106
144
```

Newest entries in both directories were `perf-val-spec-20261029-233352.{log,err.log}` and
`arms.csv`/`passes.csv` before and after; `engine-state/` still holds only `active` and `master`
(`FROZEN_STATE=master->active` is reported, not performed).

"Creates no directory" proved separately, on a scratch `-Root` (`C:\Users\yoho\AppData\Local\Temp\
v165-drytest`, deployed with `DEST=` then `logs/` and `results/` removed):

```
scratch after deploy+rm:
mtp-sidecar-2.3bpw.yaml mtp-sidecar-2.6bpw.yaml mtp-sidecar-3.1bpw.yaml perf.py prompts
run_build.ps1 run_pass.ps1 spec_decode.py state.json watch.ps1
DRYRUN runner=perf state=C:\Users\yoho\AppData\Local\Temp\v165-drytest\state.json
ARGS … ENV …
rc=0
scratch after dry run:  (identical list)
ls: cannot access '…/v165-drytest/logs': No such file or directory
ls: cannot access '…/v165-drytest/results': No such file or directory
```

The scratch directory was removed afterwards. A drafting dry run was also taken (F4's premise):

```
DRYRUN runner=spec_decode state=…\state.json
ARGS -m … -cs 40960 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16 -mtp -ndt 2
ENV … EXL3_MOE_CPU_SWAP=0 …
```

— no `-max_length` in a drafting argv, hence no `max_length` cell in its row.

`-Extra` aborts against the deployed script (F5's wording):

```
-Extra "-cq Q8"      ABORT: -Extra repeats -cq, which the regime owns (edit state.json)               exit 2
-Extra "-cs 40960"   ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit state.json)  exit 2
-Extra "-mcs 400"    ABORT: -Extra repeats -mcs, which the regime owns (use -Split, or edit state.json)      exit 2
-Extra "-ambs 4"     ABORT: -Extra repeats -ambs, which the regime owns (edit state.json)             exit 2
```

---

## What remains unexercised

- **The one-owner gate's dry-run skip is not staged.** No dummy `python perf.py` process was created
  and no server was stopped on this production box; the skip is verified by code read (the gate now
  sits inside `if (-not $DryRun)`), by the dry run above (`rc=0`, three lines, with `nvidia-smi`
  reachable and no pass running) and by `Parser::ParseFile`. The branch that fires — another pass owns
  the card while a dry run is attempted — has not been executed.
- **The gate still works for a real pass**: unchanged code, but also unrun here (a real pass needs the
  GPU, which this plan forbids).
- **`max_length` cell for a drafting run**: no real drafting pass was run, so no row with an empty
  cell exists yet; the pin test covers the source, not a written CSV line.
- **F1's precision rule on a live report**: exercised on fixtures and on T05/T06; no report in the
  tree quotes a one-decimal band beside a table today.
- Nothing was installed, killed, or measured on the box; no log, CSV or `engine-state/` file was
  created or removed.

## Concerns

- **The D6 cell's "two misquoted ratio bands" is correct** and needs no change: the third reading
  wave 1 produced (`0.6–0.8`) was the widened band reader auditing a **unit-bearing range**, so wave 2
  wrote it out as an exclusion rule rather than as a T06 defect — a range that carries a unit (`GB`,
  `t/s`, `ms`, `%`, …) is a measurement and prints INFO. A range with a unit outside the enumerated
  set still FAILs: the loud false contradiction is the side of the risk this gate takes deliberately.
- The new F4 test (`PassesCsvCells`) asserts on the source text of `run_pass.ps1`, because the CSV
  writer runs only after `Start-Process`. It is a pin against a revert, not a test of the writer.
- F1 widens what the gate reads: a report can gain a FAIL on a prose interval it used to ignore. That
  is the intended direction — every number in a quote is examined — but it will surprise whoever
  re-runs `--post` on the archived reports. Wave 3 bounds the other direction: the gate audits prose
  claims only, so a ratio table's own range cells produce no line at all.

---

# Wave 2 — a band that carries a unit is a measurement (`a30abcc8`)

Residual finding against wave 1: the widened `RATIO_INTERVAL` audited `wiki/plans/v165-reports/
T06-mtp.md:81` — "Generator costs ~0.6–0.8 GB that never comes back" — as a ratio quote and printed
`FAIL  `0.6–0.8` does not match this report's ratio table`. A gate that fails a correct sentence is a
gate an operator stops reading.

## Change

`bench/verify_task.py`:

- `RANGE_UNIT` — a range immediately followed (whitespace allowed) by `%` or one of `GB/s`, `MB/s`,
  `tok/s`, `it/s`, `t/s`, `GiB`, `MiB`, `KiB`, `GB`, `MB`, `KB`, `B`, `min`, `ms`, `µs`/`μs`/`us`,
  `Hz`, `W`, `h`, `s` is a measurement. `percentage_gate()` prints
  ``INFO  `0.6–0.8 GB` carries a unit - a measurement, not a ratio band`` and the verdict is
  untouched. The slice is printed as written (unit and spacing included), and the span is added to
  `checked` so the `%` of `0.92–0.99%` is not reported a second time as prose.
- The set is **enumerated from this repo's reports**, not "any following word": `GB` (T06:81,
  v165-STATE "0.34 GiB", the retune plan "1.1–2.1 GB") and `t/s` are what the archive actually uses,
  and a loose word rule would read "was 0.92–0.99 against the table" as unit-bearing and let a real
  ratio quote pass unchecked. The risk taken is the other one, and it is stated in the code comment:
  an unenumerated unit keeps its range a ratio band and can FAIL a correct sentence — a loud, visible,
  recoverable contradiction rather than a silent pass, which is the defect this gate exists to catch.
- `bench/test_gates.py`: `test_a_range_carrying_a_unit_is_a_measurement` (0.6–0.8 GB beside the
  ratio table → rc 0, INFO, no "does not match"), `test_a_percentage_suffixed_range_is_not_a_ratio_band`
  (`0.92–0.99%` → INFO, no FAIL), and `0.87–1.10` added to
  `test_one_decimal_quote_outside_the_table_fails` (a unitless two-decimal quote beside the same table
  still FAILs).

## Verification

```
$ python3 bench/test_gates.py -v
Ran 42 tests in 0.061s
OK                                    (40 before wave 2; 42 now)

$ python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md   # rc=1
  FAIL  rb-ab-166-20261029-202655: 1 of 49 CPU-split component(s) have no saved placement …
  PASS  rb-ab-166-20261029-202655: arm rows present in arms.csv or spec_decode.csv
  FAIL  `±5 %` is a symmetric band; this project reports one-sided ratio intervals (C5) …
  FAIL  `0.87–1.10` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  FAIL  `0.47–1.04` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  INFO  `1.0–2.2` is not a ratio band (an endpoint is above 1.2) - not checked against the ratio table
  PASS  `0.96–1.09` matches this report's own ratio table
  INFO  `0.6–0.8 GB` carries a unit - a measurement, not a ratio band; not checked against the ratio table
  INFO  `1.2–2.4` / `1.14–2.01` / `1.01–1.68` / `1.12–2.41` are not ratio bands (an endpoint is above 1.2)
  INFO  `35.2–38.8 t/s` carries a unit - a measurement, not a ratio band; not checked against the ratio table
  INFO  `13 %` / `1 %` / `35 %` ×2 / `33 %` / `84 %` ×2 / `40 %` / `42 %` are prose, not a ratio band
verdict: CONTRADICTIONS FOUND
```

Diff against the wave-1 output of the same command: exactly two lines change, and the FAIL count drops
from 5 to 4 —

```
- FAIL  `0.6–0.8` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
+ INFO  `0.6–0.8 GB` carries a unit - a measurement, not a ratio band; not checked against the ratio table
- INFO  `35.2–38.8` is not a ratio band (an endpoint is above 1.2) - not checked against the ratio table
+ INFO  `35.2–38.8 t/s` carries a unit - a measurement, not a ratio band; not checked against the ratio table
```

Mutation (copy in `/tmp/opencode/mut`, never in the repo) — the unit guard deleted, everything else
wave 2:

```
FAIL: test_a_percentage_suffixed_range_is_not_a_ratio_band
FAIL: test_a_range_carrying_a_unit_is_a_measurement
FAILED (failures=2)
# and the false FAIL is back, verbatim:
#   AssertionError: 1 != 0 : post-run gates
#     FAIL  `0.6–0.8` does not match this report's ratio table: the table's ratios run 0.92 to 0.99
```

The copy was restored from the repo and passes again (`PercentageGate` … OK).

```
$ python3 bench/verify_task.py --pre
verdict    FITS (needs >= 1024 MiB VRAM headroom)
verdict: all gates passed              rc=0
```

## Tracker

`wiki/plans/v165-critique-implementation.md`'s D6 cell already says "two misquoted ratio bands", which
is now the true count — nothing to revert and nothing to add. The exclusion rule lives in the code
comment above `RANGE_UNIT` and in this report, not as a note about a T06 defect.

## Unexercised after wave 2

- No unit outside the enumerated set appears in the tree's reports, so the "unenumerated unit still
  FAILs" side of the trade-off is stated in the comment and not exercised by a real report.
- Same as wave 1: the one-owner gate's dry-run skip, a real drafting row's empty `max_length` cell,
  and the swap gate's verdict branch remain code-read/pinned, not run.

---

# Wave 3 — the gate audits prose, not the table (`09e3b084`)

Residual finding: after wave 2 the gate still reported `CONTRADICTIONS FOUND` on a correct report,
because it scanned markdown table rows as if they were prose. `wiki/plans/v165-reports/T05-memops.md:18`
is `| 4096 | 0.974 | 0.915–1.034 | 0.856 | 0.850–0.862 |`; those ranges are the table, and twelve of
T05's 14 quoted-band lines were the table being audited against itself.

## Change

`bench/verify_task.py`:

- `table_line_spans()` returns the character spans of the report's markdown rows (`markdown_cells()`
  is the row test), `in_table_row()` asks whether a match starts inside one of them.
- `percentage_gate()` skips a match in a table row for both band scans — `SYMMETRIC_PCT` and
  `RATIO_INTERVAL` — before any message, so a range cell produces **no line at all**, not even INFO.
  A band printed in a table is the reference data; a band in prose is the claim checked against it.
- The third no-band message (`` `0.915–1.034` is a range the report's own table prints``) was removed:
  with rows excluded a table cell cannot reach that branch, and leaving it would be dead code. The two
  messages the F7 finding asked for remain (no table / no matching range). `table_ratio_ranges()` still
  supplies the range list for the second one.
- `PCT` (prose `%` mentions, INFO only, never affects the verdict) is unchanged — it never FAILed, so
  the dispatch's "keep everything else as it is" applies literally there.

`bench/test_gates.py`: `test_a_table_range_read_as_a_quote_is_not_called_a_missing_table` (whose subject
no longer exists) is replaced by two cases —

- `test_a_table_only_range_report_passes` — a report that is nothing but the range table → rc 0, and
  `0.915–1.034` / `0.850–0.862` appear nowhere in the output (no FAIL, no INFO);
- `test_the_same_range_quoted_in_prose_still_fails` — the same interval written in a sentence → rc 1,
  "matches no range in the report's table(s)".

`test_quote_against_a_range_column_names_the_ranges` (a different prose quote, `0.958–1.001`) still
passes unchanged.

## Verification

```
$ python3 bench/test_gates.py -v
Ran 43 tests in 0.062s
OK                                    (42 before wave 3; one test replaced by two)

$ python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T05-memops.md   # rc=1, 4 FAILs
  PASS  memops0-1-20261029-200246: 48 CPU-split component(s), every one from the saved profile
  PASS  memops0-2-20261029-201150: …   (and the other two runs)
  FAIL  `±11 %` is a symmetric band; this project reports one-sided ratio intervals (C5) …
  FAIL  `±2 %` is a symmetric band; this project reports one-sided ratio intervals (C5) …
  FAIL  `0.83–0.93` matches no range in the report's table(s) (ranges: 0.915–1.034, 0.850–0.862,
        0.946–0.971, 0.796–0.943 (+8 more)): this gate compares a quote with a ratio column's
        min/max, and those cells are ranges
  FAIL  `0.958–1.001` matches no range in the report's table(s) (… same ranges …)
  INFO  `17 %` is prose, not a ratio band - not checked
  INFO  `11 %` is prose, not a ratio band - not checked
verdict: CONTRADICTIONS FOUND
```

The 12 table-cell FAILs are gone (T05 went 16 FAILs → 4). The verdict stays `CONTRADICTIONS FOUND`, and
that is right: T05 makes four **prose** claims the gate cannot certify —

- `±11 %` (line 31) and `±2 %` (line 33) — the symmetric bands the C5 critique is about;
- `medians 0.83–0.93` (line 29) and `medians 0.958–1.001 at the measurement bands` (line 32) — genuine
  prose ratio claims whose table prints per-band *ranges*, so there is no column min/max to check them
  against.

```
$ python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md   # rc=1, 4 FAILs
  FAIL  rb-ab-166-20261029-202655: 1 of 49 CPU-split component(s) have no saved placement …
          -- CPU split experts (worker, dynamic): mtp.layers.0.mlp [152..512) of 512
  PASS  rb-ab-166-20261029-202655: arm rows present in arms.csv or spec_decode.csv
  FAIL  `±5 %` is a symmetric band; this project reports one-sided ratio intervals (C5) …
  FAIL  `0.87–1.10` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  FAIL  `0.47–1.04` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
  INFO  `0.6–0.8 GB` carries a unit - a measurement, not a ratio band; not checked against the ratio table
  INFO  `1.2–2.4` is not a ratio band (an endpoint is above 1.2) - not checked against the ratio table
  INFO  `35.2–38.8 t/s` carries a unit - a measurement, not a ratio band; not checked against the ratio table
  … 10 INFO percentages …
verdict: CONTRADICTIONS FOUND
```

The cold MTP line, `±5 %`, `0.87–1.10` and `0.47–1.04` are intact — both ratio FAILs sit in prose
(`T06-mtp.md:24-25`). The only lines wave 3 removes from T06 are table cells (`1.0–2.2`,
`0.96–1.09` — which used to PASS, `1.14–2.01`, `1.01–1.68`, `1.12–2.41`), all inside the checkpoint
table; no verdict changes.

Mutation (copy in `/tmp/opencode/mut`, never in the repo) — the two `in_table_row` guards deleted:

```
FAIL: test_a_table_only_range_report_passes
FAILED (failures=1)
# and the false FAILs are back:
#   AssertionError: 1 != 0 : post-run gates
#     FAIL  `0.915–1.034` matches no range in the report's table(s) (ranges: 0.915–1.034, …)
# and on the real report: T05 rc=1 with 16 FAILs again (4 prose + 12 table cells)
```

The copy was restored from the repo and passes again (`PercentageGate` … OK).

```
$ python3 bench/verify_task.py --pre
verdict    FITS (needs >= 1024 MiB VRAM headroom)
verdict: all gates passed              rc=0
```

## Unexercised after wave 3

- No report in the tree prints `±` inside a table row, so the symmetric-band exclusion is covered by
  the interval case and reasoned, not exercised by real data.
- A quote that straddles a table boundary (a claim started in prose and continued inside a row) is
  judged by where the match starts; no report does this.
- Unchanged from wave 1/2: the one-owner gate's dry-run skip, a real drafting row's empty
  `max_length` cell, and the swap gate's verdict branch remain code-read or pinned, not run.


