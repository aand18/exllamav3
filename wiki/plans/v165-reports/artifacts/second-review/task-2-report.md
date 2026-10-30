# Task 2 report — `run_pass.ps1`: `.draft`, `.knob_arms_disable_swap`, one `-cs`, `-DryRun`

Status: **DONE_WITH_CONCERNS**
Commit: `7c043454` — `bench: run_pass honours state.json draft+swap, emits one -cs, adds -DryRun`
(base `5134738b`, one file, +89 / −5)

## Read this first (the concern, with evidence)

The brief's expected perf argv (`-mtp -ndt 2`) is **not accepted by `perf.py`**. `eval/perf.py:224-229`
calls `model_init.add_args(parser, …)` **without** `add_draft_model_args = True`, and the installed
1.6.5 `model_init.py:129-136` registers `-mtp/-ndt/-sdh/-hdt/-dds` only inside
`if add_draft_model_args:`. `perf.py:237` is `parser.parse_args()` (strict). Measured on the box,
building the parser exactly as perf.py builds it:

```
$ /mnt/c/Users/yoho/Downloads/tabbyAPI/venv/Scripts/python.exe -c "import argparse; from exllamav3 import model_init; p = argparse.ArgumentParser(allow_abbrev=False); model_init.add_args(p, default_cache_size=32768, default_autosplit_max_batch_size=1, default_chunk_size=4096); o=[s for a in p._actions for s in a.option_strings]; print('perf.py parser registers:', [x for x in ('-mtp','-ndt','-sdh','-hdt','-dds') if x in o] or 'none'); print('cache flags:', [x for x in ('-cs','-cq','-max_length') if x in o])"
perf.py parser registers: none
cache flags: ['-cs', '-cq']

$ …same, with add_draft_model_args=True (spec_decode.py's call at eval/spec_decode.py:442-449)…
spec_decode.py parser registers: ['-mtp', '-ndt', '-sdh', '-hdt', '-dds']
```

(That probe only constructs an `argparse` parser: no model, no GPU, no file written. It is the same
class of process as the identity gates the brief sanctions.)

Consequence, if the brief is kept verbatim: `-Runner spec_decode` runs are fine, but the **next real
`-Runner perf` pass will die at `argparse` with `unrecognized arguments: -mtp -ndt 2`** (exit 2, status
`fail`). This is invisible to `-DryRun` by construction, which is why it is called out here rather
than left for a measurement run to find.

I implemented the brief as written (draft args for both runners, `-mtp -ndt 2` in the perf argv) and
did **not** touch `eval/perf.py`: the fix is one keyword in another file, and the plan already has a
later task editing `eval/perf.py` (Task 5). Ruling requested — either
`model_init.add_args(parser, …, add_draft_model_args = True)` in `eval/perf.py` (worth doing anyway:
otherwise no perf pass can have drafting on, and `-sd/--spec_dec` cannot be measuring what it says),
or this task should emit draft args only for `-Runner spec_decode`.

## What I implemented

All five requirements of the brief, nothing else.

1. **`.draft` → draft args** (`bench/windows/run_pass.ps1`, regime section). `$Draft = $ST.draft`,
   then `-mtp` for `mode: "mtp"`, `-ndt N` for `num_draft_tokens`, `-sdh N` / `-hdt N` only when
   `> 0`, `-dds` for `dynamic_draft: true`. `-dm` is never emitted: the MTP head rides the main
   model. `draft.readback` is information only — the comment says why (no per-run `model_init`
   switch at 1.6.5; the only readback control is `spec_decode.py`'s `-rb_ab`, which measures four
   arms in one process). A missing `.draft` keeps the old behaviour: the whole block is inside
   `if ($null -ne $Draft)`.

2. **`-CacheSize N` (0 = from state)**, replacing the unconditional `$CacheSize = $ST.cache_size`.
   `$stateCs = $ST.cache_size`, and for `-Runner spec_decode` it becomes
   `$ST.draft.spec_decode_cache_size` when the draft block, the field and a value `> 0` are present,
   else it stays `cache_size`. `-cs` is emitted once, from `$CacheSize`; the log header's `=== args`
   line and `passes.csv` both carry the used value (see the next point for which cell).

3. **`knob_arms_disable_swap: true` → `EXL3_MOE_CPU_SWAP=0`**, written into `$baseEnv` right after
   `$ST.env` is loaded, i.e. before `-Unset` and `-Env`, so an arm's delta still wins. It reaches
   `$envApplied` through the existing copy, so it lands in the log header and `passes.csv`.

4. **`-Extra` may not repeat a regime-owned flag.** The scan runs over the *assembled* argv (not just
   the `-Extra` tokens), which is what makes "exactly once" true regardless of which path added the
   copy. Table + abort message name the flag and its replacement:
   `-m → -Model`, `-cs → -CacheSize`, `-max_length → -MaxLength`, `-mcs → -Split`,
   `-mct → -Threads`, `-chunk_size/-ambs/-cq → edit state.json`. The `-Extra "-cq Q8"` usage example
   is replaced by `-Extra "-dr 3"` (a perf.py flag the regime does not own) and the usage paragraph
   now says the `-cq` spelling aborts by design.

5. **`-DryRun`**: after the state/model/file/one-owner/engine-identity gates, after the regime check,
   after the duplicate guard and after env assembly, it prints exactly

   ```
   DRYRUN runner=<perf|spec_decode> state=<state path>
   ARGS <argv joined by spaces>
   ENV <k=v sorted, spaces>
   ```

   and `exit 0` before the log path is even stamped, so there is no log, no CSV row and no watcher.
   `runner=` reports the runner actually selected (`$RunnerName`) rather than echoing `-Runner`, so a
   misspelled runner cannot make the line lie about which script would run.

Other edits forced by the brief: the header usage block gained `-DryRun`; the usage paragraph now
documents `-CacheSize` and the regime-owned rule; the `passes.csv` row writes `$CacheSize` into the
existing `max_length` cell (see below).

### The `max_length` cell

Requirement 2 says the used `-cs` value is "the value recorded in the log header and in `passes.csv`
(the `max_length` column already exists; do not add a column)". So the row's `max_length` cell is now
`$CacheSize`, with a comment saying why: one length column, no cache column, and spec_decode strips
`-max_length` from its argv entirely, so that cell is the one that must answer "how much context did
this run size for". The ladder length and every other flag stay in the `args` cell, and both
`passes.csv` readers (`bench/ratios.py:128-133`, `bench/verify_task.py:124-140`) only display that
field, so nothing downstream changes meaning. Not asked for, but visible: for a perf run with a
`-MaxLength` delta (e.g. `-MaxLength 262144`) the cell now shows the cache size and the ladder length
lives in `args` — flagging it in case you wanted the opposite (I followed the brief's wording).

## Verification — the brief's four commands, verbatim

`bench/windows/deploy.sh` first (required: the box only ever runs the deployed copy):

```
$ bench/windows/deploy.sh
state.json: unchanged
deployed to /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165
6d99e40f…  run_build.ps1
c77a5d21fc486d2b68ef947b4704cac8a8c9a9dc72c262a7aa34d4e6fac0ca72  run_pass.ps1
910f05a2…  watch.ps1
28d9afdd…  perf.py
7913608f…  state.json
```

The deployed hash equals the committed blob and the working-tree bytes
(`git diff --stat -- bench/windows/run_pass.ps1` empty, `git show HEAD:… | sha256sum` = `c77a5d21…`),
so the box runs exactly `7c043454`.

**1. `-Tag dryrun-perf -DryRun`**

```
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -mtp -ndt 2 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

`-cs 131072` once, `-mtp`, `-ndt 2`, no `-sdh`/`-hdt`/`-dds`, `EXL3_MOE_CPU_SWAP=0`. ✅

**2. `-Tag dryrun-spec -DryRun -Runner spec_decode`**

```
DRYRUN runner=spec_decode state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 40960 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16 -mtp -ndt 2
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

`-cs 40960` once (from `draft.spec_decode_cache_size`), no `-max_length` (stripped as before), and the
val-spec `-cs 131072 … -cs 40960` pair is gone. ✅

**3. `-Tag dryrun-dup -DryRun -Extra "-cs 40960"`**

```
ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit state.json)
EXIT=2
```

Bash reports the PowerShell exit code as `2`; no DRYRUN/ARGS lines are printed. ✅

**4. `-Tag dryrun-env -DryRun -Env "EXL3_MOE_CPU_SWAP=1"`**

```
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -mtp -ndt 2 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=1 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

The per-arm delta wins (`=1`), same argv. ✅

**Extra, requirement 1's fallback** (not one of the four; ran because a missing `.draft` block is the
one requirement the four cannot reach). State file is a copy of `state.json` with `.draft` deleted,
hosted in WSL and read over UNC so nothing is written on the box:

```
dr$ powershell.exe -NoProfile -ExecutionPolicy Bypass -File '…\run_pass.ps1' -Tag dryrun-nodraft -DryRun \
    -Runner spec_decode -State '\\wsl.localhost\Ubuntu-24.04\tmp\opencode\nodraft-state.json'
DRYRUN runner=spec_decode state=\\wsl.localhost\Ubuntu-24.04\tmp\opencode\nodraft-state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16
ENV … EXL3_MOE_CPU_SWAP=0 …
EXIT=0
```

No draft flags, and the cache falls back to `cache_size` 131072 even on `spec_decode`. ✅

### Nothing was written on the box

`logs/` and `results/` manifests (`find … -printf '%T@ %s %p'`, 252 files) are byte-identical before
the deploy+dry-runs and after all five:

```
$ find logs results -type f -printf '%T@ %s %p\n' | sort > /tmp/opencode/v165-after.txt
$ diff /tmp/opencode/v165-before.txt /tmp/opencode/v165-after.txt && echo "IDENTICAL: logs/ and results/ gained nothing"
IDENTICAL: logs/ and results/ gained nothing
logs files: 106  results files: 144        # unchanged counts
```

No watcher `watch-*.csv`, no `stop-*.flag`, no `passes.csv` row. The one thing a dry run does touch is
`engine-state/active` (mtime moves): env assembly refreshes the frozen profile from `master` — the
same thing every pass does, by the existing code, before the `-DryRun` exit. `master/` is untouched.

## Files changed

- `bench/windows/run_pass.ps1` — the only file in the commit.
- `.superpowers/sdd/v165-critique-fixes/task-2-report.md` — this report (untracked, as reports are).
- `wiki/plans/v165-critique-fixes.md` — pre-existing untracked, not mine, not committed.

## Self-review of the diff

- **Completeness**: the five requirements map to the five bullets above; every exact string the brief
  gave is used verbatim (`ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit
  state.json)`, the three `DRYRUN/ARGS/ENV` lines, `EXIT 2`).
- **Why-comments, not what-comments**: the `-cs` split comment cites the val-spec log; the swap
  comment explains that a drafter that accepts an extra token also moves the expert placement; the
  guard comment explains why the scan is on the assembled argv; the dry-run comment explains why it
  sits after the gates.
- **Discipline**: no new flags beyond `-CacheSize` and `-DryRun`, no refactor of the run/parse/verdict
  path, no touch of `eval/perf.py`, `state.json`, `deploy.sh` or `watch.ps1`. `-Unset`/`-Env` ordering
  untouched (that ordering is what lets `-Env` win, as requirement 3 assumes).
- **Edge cases checked**: missing `.draft` (run above); `spec_decode_cache_size` absent or `0`
  (`-gt 0` guard falls back to `cache_size`); `-CacheSize` given explicitly (param non-zero wins);
  `-sdh`/`-hdt`/`-dds` absent from argv when the state says `0`/`false`; `-max_length` still stripped
  for `spec_decode`; duplicate guard fires on `-cs` and cannot fire on the regime's own single copy.
- **No dead code**: `$RunnerName` is used only by `-DryRun` but is the truthful value there;
  `$stateCs` collapses to one assignment.
- **Testing**: five dry runs (four briefed + the fallback), plus the two parser probes; no GPU pass, no
  model load, no process started or killed on the box beyond the scripts' own `python -c` gates.

## Concerns

1. **`perf.py` cannot accept the draft args** — the headline concern at the top of this report: the
   next real `-Runner perf` pass aborts at `argparse` until `eval/perf.py` passes
   `add_draft_model_args = True`. Needs a ruling (it is a one-keyword change in a file this task was
   not given).
2. **`passes.csv`'s `max_length` cell now means the cache size.** Follows the brief's parenthetical;
   for a perf run whose `-MaxLength` differs from the regime cache the ladder length is only in
   `args`. If you meant "keep `$MaxLength`, and the cache size is already recorded via `args`", that
   is a one-line revert.
3. **`-Extra` can still smuggle a flag the runner rejects** (`-or`, `-nds`, …): the guard covers only
   the eight regime-owned flags, as specified. A misspelled `--cache_size` also slips past it — the
   brief lists short spellings only.
4. **`bench/windows/README.md` is stale for `run_pass.ps1`** ("The `production` regime is fixed in
   `run_pass.ps1`: model 2.05bpw, `-mcs 360 …`"), and it does not mention `-DryRun`/`-CacheSize`. Not
   in the brief, so untouched — worth a task if the README is meant to be the harness contract.
5. Nit: the dry-run's `state=` prints the path as given, so the fallback run shows the UNC form. That
   is the state file that was read, which is the point of the line.

---

# Fix report (round 2) — three findings against `7c043454`

Status: **DONE**
Fix commit: `89f0b787` — `bench: draft args are spec_decode-only, max_length cell restored, dry run
writes nothing` (on top of `7c043454`, same one file, +25 / −19).
Deployed copy == worktree == HEAD blob: `c4a00dbe613a185d51d3d57d340a4c6d9f9d09d6b12b4c601ed2ac8acd6d5522`
(`sha256sum bench/windows/run_pass.ps1`, `sha256sum …/_agent/v165/run_pass.ps1`,
`git show HEAD:bench/windows/run_pass.ps1 | sha256sum` — all three equal).

Re-read the regenerated brief first. One residual inconsistency to note rather than act on: the
brief's requirement 1 and 5 now carry the rulings, but its closing "Expected:" paragraph (brief
lines 68-70) still reads `perf dry-run has -cs 131072 once, -mtp, -ndt 2` — the pre-ruling text. I
implemented the rulings (requirement 1 wins; your message says the same), so the perf dry run now
carries **no** draft flags and the spec dry run carries `-mtp -ndt 2`.

## What changed

1. **Draft args are `-Runner spec_decode` only.** The block condition is now
   `if ($null -ne $Draft -and $Runner -eq "spec_decode")`; the comment states why: `eval/perf.py`
   calls `model_init.add_args()` without `add_draft_model_args` and unpacks four values from
   `model_init.init()`, so `-mtp`/`-ndt` in a pp/tg argv aborts at argparse, and a pp/tg ladder
   measures the main model, so drafting is not part of its regime. The `-cs` substitution is
   untouched (it was already spec_decode-only).

2. **`passes.csv`'s `max_length` cell is `$MaxLength` again.** The overload comment is replaced by
   the one line "The cache the run used is in the args cell (-cs N), not here: this cell is the
   ladder length." Column count and header unchanged, and the row block is now byte-identical to the
   one that produced the existing rows (see the check below).

3. **`-DryRun` no longer performs the frozen-profile restore.** The `master → active` copy (which
   deletes and repopulates `engine-state/active`) and the directory creation are wrapped in
   `if (-not $DryRun)`; `XDG_CACHE_HOME` is still set and `FROZEN_STATE=master->active` is still
   recorded in `$envApplied`, so the printed `ENV` is exactly what a real pass would carry. The
   dry-run banner comment now says this instead of the older, weaker claim.

## Verification, re-run in full

`bench/windows/deploy.sh` (deployed `run_pass.ps1` sha `c4a00dbe…`, `state.json: unchanged`), then:

**1. `-Tag dryrun-perf -DryRun`** — no draft flags, `-cs 131072` once:

```
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

No `-mtp`, no `-ndt`, no `-sdh`/`-hdt`/`-dds`; `-cs 131072` once; `EXL3_MOE_CPU_SWAP=0` present;
`FROZEN_STATE=master->active` still reported even though the copy was skipped. ✅

**2. `-Tag dryrun-spec -DryRun -Runner spec_decode`** — `-mtp -ndt 2`, `-cs 40960` once:

```
DRYRUN runner=spec_decode state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 40960 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16 -mtp -ndt 2
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

✅

**3. `-Tag dryrun-dup -DryRun -Extra "-cs 40960"`** — still exit 2, named flag:

```
ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit state.json)
EXIT=2
```

✅

**4. `-Tag dryrun-env -DryRun -Env "EXL3_MOE_CPU_SWAP=1"`** — the arm's delta still wins:

```
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=1 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

✅

**5. `passes.csv` unchanged in column count.** Two independent checks, both static (a dry run must
not write a row, so this cannot be observed from an actual row):

- the row block is byte-identical to the version at base `5134738b` (the one that produced the rows
  already on the box), which proves both the column count and that the cell is `$MaxLength`:

  ```
  $ git show 5134738b:bench/windows/run_pass.ps1 | sed -n '/^\$row = /,+3p' > /tmp/opencode/rb.txt
  $ sed -n '/^\$row = /,+3p' bench/windows/run_pass.ps1                > /tmp/opencode/rn.txt
  $ diff /tmp/opencode/rb.txt /tmp/opencode/rn.txt && echo "row block byte-identical to base 5134738b"
  row block byte-identical to base 5134738b
  ```
- header and row widths agree, and neither changed since base:

  ```
  $ python3 -c '…regex over run_pass.ps1…'
  header columns: 23
  ['tag','run_id','finished_utc','regime','status','engine','ext','elapsed_s','max_length',
   'min_free_vram_mb','min_free_ram_mb','peak_proc_ws_mb','samples','breach','env','split','threads',
   'args','harness_sha256','log','runner','peak_alloc_mb','peak_reserved_mb']
  row placeholders: 23 max index: 22
  $ git diff 5134738b -- bench/windows/run_pass.ps1 | grep -c 'finished_utc'
  0
  ```

  The cache size the run used is still recorded — in `args` (`-cs 131072` / `-cs 40960`) and in the
  log header's `=== args` line, which is what requirement 2 now asks for. ✅

**6. A dry run leaves `engine-state/` untouched.** Method: a full manifest of the whole `engine-state`
tree — every entry's type, mtime (`%T@`, sub-second) and size, directories included, plus sha256 of
every file — taken after deploy-time was already past and before any dry run, then compared after
all four. Directory mtimes are in the manifest on purpose: the restore is `Remove-Item -Recurse` +
`Copy-Item`, which preserves file mtimes but re-times the directories, so a directory timestamp is
the cheapest witness that the copy ran.

```
$ cd /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165
$ find engine-state -printf '%y %T@ %s %p\n' | sort > /tmp/opencode/es-before.txt        # 15 entries
$ find engine-state -type f -exec sha256sum {} + | sort -k2 > /tmp/opencode/es-hash-before.txt  # 8 files
… deploy + the four dry runs above …
$ find engine-state -printf '%y %T@ %s %p\n' | sort | diff /tmp/opencode/es-before.txt -
IDENTICAL: nothing under engine-state/ was created, deleted, rewritten or retimed
$ find engine-state -type f -exec sha256sum {} + | sort -k2 | diff /tmp/opencode/es-hash-before.txt -
IDENTICAL hashes
$ stat -c '%y %n' engine-state/active
2026-10-30 01:40:37.630915800 +0100 engine-state/active      # unchanged (pre-round timestamp)
```

For contrast, the same manifest moved on every run of the previous round (`active` was re-timed to
01:39/01:40 by the restore), so the comparison is known to be sensitive to the thing it tests. ✅

**7. `logs/` and `results/` still gained nothing** — same 252-file manifest as before the first round:

```
$ find logs results -type f -printf '%T@ %s %p\n' | sort | diff /tmp/opencode/v165-before.txt -
IDENTICAL: logs/ and results/ gained nothing
logs files: 106  results files: 144
```

(`/tmp/opencode/v165-before.txt` sha256 `787815ba…`, taken before round 1; nine dry runs later the
manifest is unchanged.) ✅

## Self-review of the fix

- **1**: perf argv is now exactly what `perf.py` can parse — `-m -cs -chunk_size -ambs -cq
  -max_length -mcs -mct -sd`; the draft flags exist only where `add_draft_model_args=True` is passed.
  `-cs` logic untouched.
- **2**: the row block is byte-identical to base, so no CSV consumer sees anything new; comment says
  where the cache size lives.
- **3**: the only writes in the `engine-state/` path are now behind `if (-not $DryRun)`; both the
  master branch and the no-master branch (directory creation) are guarded. Real passes are
  unaffected: with `-DryRun` absent the code is the original three statements.
- Nothing else changed: no new parameters, no new flags, `-Env`/`-Unset` ordering untouched, no
  edits to `eval/*`, `state.json`, `deploy.sh`, `watch.ps1`.

## Open items

1. `eval/perf.py` still cannot enable drafting (`add_draft_model_args` not passed; `-sd/--spec_dec`
   only selects seqlens 1..4), so today no pp/tg pass has drafting on regardless of `state.json`.
   Out of scope here and no longer a break — the perf argv no longer asks for it — but it is the
   follow-up if pp/tg is meant to be measured with MTP.
2. `bench/windows/README.md` still describes the old fixed regime; untouched.
3. Persisted the `perf.py` flag-surface finding to Engram (obs 345, judged `related` to the earlier
   PowerShell `-Model` collision note).

---

# Fix report (round 3) — spelling-proof guard, scoped swap, `-StaticSwap`

Status: **DONE**
Fix commit: `8c69b5ec` — `bench: spelling-proof regime guard, scoped EXL3_MOE_CPU_SWAP=0 with
-StaticSwap` (on top of `89f0b787`, same one file).
Deployed == worktree == HEAD blob: `5966f78e34540890e9fbc57ddbffd8a5f492b6936196c33c6972a38788b87182`
(`sha256sum` of all three).

Re-read the regenerated brief (requirements 3, 4 and the verification block changed). Two notes on
it, neither acted on:

- The last verification command (`brief:83-84`) is `-Tag dryrun-env -DryRun -Env
  "EXL3_MOE_CPU_SWAP=1"` — no `-Runner spec_decode` — while the "Expected" text says "the `-Env` run
  **on top of spec_decode**". I ran both; both show `=1`, and only the spec_decode one exercises the
  override of the automatically applied `0`.
- `-StaticSwap` had to be added to the `param` block, so the file also gained that one parameter; the
  brief's requirement 3 asks for the switch by name.

## What changed

1. **Spelling-proof duplicate guard** (requirement 4). The regime's argv is now built into
   `$regimeArgs` (base flags + the state's draft args when emitted) and kept separate from the deltas.
   A `$regimeSpellings` table maps every spelling argparse accepts to the canonical short flag —
   `--model_dir/-m`, `--cache_size/-cs`, `--cache_quant/-cq`, `--moe_cpu_split/-mcs`,
   `--moe_cpu_threads/-mct`, `--max_length/-max_length`, `--chunk_size/-chunk_size`,
   `--autosplit_max_batch_size/-ambs`, `--mtp/-mtp`, `--num_draft_tokens/-ndt`,
   `--sparse_draft_head/-sdh`, `--hybrid_draft_tokens/-hdt`, `--dynamic_draft/-dds` — and
   `Get-RegimeFlag` strips a `=value` suffix and lowercases before the lookup, returning `$null` for
   values (`131072`, `2,2`, `D:\llms\…`, `mtp-sidecar-3.1bpw.yaml`). The owned set is the set of
   canonical flags present in `$regimeArgs`; the scan runs over the `-Extra` tokens **before** the
   spec_decode `-max_length` strip, so a restatement there aborts instead of being eaten. The abort
   names the canonical flag and its remedy from `$regimeRemedy` (draft flags and the no-parameter
   flags say "edit state.json"; `-m/-cs/-max_length/-mcs/-mct` name `-Model/-CacheSize/-MaxLength/
   -Split/-Threads`).

2. **`EXL3_MOE_CPU_SWAP=0` scoped** (requirement 3, controller ruling). New `[switch] $StaticSwap`.
   The env is applied only when `$ST.knob_arms_disable_swap` is true **and**
   (`$Runner -eq "spec_decode"` or `$StaticSwap`); a plain ladder run emits nothing and keeps the
   engine default. Override order unchanged (`-Unset`/`-Env` still come after), and the rule is
   documented in the file header and in the code comment (why: 44 archived rows were measured with
   placement on; the T4 thresholds and the T10 ladder are read against that series).

3. **Comment on the `num_draft_tokens -gt 0` guard** (reviewer Minor 4): a state value of 0 is
   treated as unset because the engine default is 4, and "no `-ndt` in the argv" is how that shows.
   No behaviour change.

Header usage block gained the `-StaticSwap` example and now says "restate" (not "repeat") and that
the guard is spelling-proof.

## Verification — the full brief block, plus the two readings of the `-Env` case

`bench/windows/deploy.sh` → deployed `run_pass.ps1` sha `5966f78e…`, `state.json: unchanged`.

**1. `-Tag dryrun-perf -DryRun`** — `-cs 131072` once, no draft flags, **no** `EXL3_MOE_CPU_SWAP`:

```
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

**2. `-Tag dryrun-perfstatic -DryRun -StaticSwap`** — the switch adds the static placement:

```
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

**3. `-Tag dryrun-spec -DryRun -Runner spec_decode`** — `-mtp -ndt 2`, `-cs 40960` once, `=0`:

```
DRYRUN runner=spec_decode state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 40960 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16 -mtp -ndt 2
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

**4-6. The three `-cs` spellings** (`dryrun-dup`, `dryrun-dup2`, `dryrun-dup3`) — all exit 2, same
canonical flag:

```
== -Extra "-cs 40960"
ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit state.json)
EXIT=2
== -Extra "-cs=40960"
ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit state.json)
EXIT=2
== -Extra "--cache_size 40960"
ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit state.json)
EXIT=2
```

**7. `-Tag dryrun-dup4 -DryRun -Runner spec_decode -Extra "-ndt 4"`** — the state-emitted draft flag
is owned too:

```
ABORT: -Extra repeats -ndt, which the regime owns (edit state.json)
EXIT=2
```

**8. `-Tag dryrun-env -DryRun -Env "EXL3_MOE_CPU_SWAP=1"`** (the brief's literal, perf):

```
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=1 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

**9. `-Runner spec_decode -Env "EXL3_MOE_CPU_SWAP=1"`** (the Expected text's wording — the delta wins
over the automatically applied `0`):

```
DRYRUN runner=spec_decode state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 40960 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16 -mtp -ndt 2
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=1 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
EXIT=0
```

**Beyond the block** (guard is spelling-proof, so these were worth checking):

```
== spec_decode + -Extra "--num_draft_tokens=4"      (long alias, =value form)
ABORT: -Extra repeats -ndt, which the regime owns (edit state.json)          EXIT=2
== spec_decode + -Extra "-max_length 8192"          (guard runs before the strip: it used to be eaten)
ABORT: -Extra repeats -max_length, which the regime owns (use -MaxLength, or edit state.json)   EXIT=2
== -Extra "-cq Q8"                                  (the usage example that must abort by design)
ABORT: -Extra repeats -cq, which the regime owns (edit state.json)           EXIT=2
== -Runner spec_decode -Extra "-or mtp-sidecar-3.1bpw.yaml -dr 3"   (legit extras still pass)
ARGS … -mtp -ndt 2 -or mtp-sidecar-3.1bpw.yaml -dr 3                         EXIT=0
```

**Nothing written.** `logs/`, `results/` and `engine-state/` all still match the manifests taken
before the deploy, after eleven dry runs this round:

```
$ find logs results -type f -printf '%T@ %s %p\n' | sort | diff /tmp/opencode/lr3-before.txt -
IDENTICAL                                      # 106 + 144 files, same manifest as round 1 (787815ba…)
$ find engine-state -printf '%y %T@ %s %p\n' | sort | diff /tmp/opencode/es3-before.txt -
IDENTICAL                                      # 15 entries incl. directory mtimes
$ find engine-state -type f -exec sha256sum {} + | sort -k2 | diff /tmp/opencode/es3-hash-before.txt -
IDENTICAL                                      # 8 files
```

## Self-review of round 3

- Guard: owned set is derived from `$regimeArgs`, not hardcoded, so a state change that starts
  emitting `-sdh`/`-hdt`/`-dds` automatically makes those guardable — no second list to keep in sync.
  `Get-RegimeFlag` returns `$null` for values, so `-or mtp-sidecar-3.1bpw.yaml`, `-dr 3` and a
  negative-looking value cannot be mistaken for a flag. `-dm`/`-or` stay available to an arm (the
  state never emits them).
- The scan sits before the strip loop (requirement 4's ordering) and before the `-short`/`-sd`/skip
  flags, none of which are regime-owned.
- Scoping: `$StaticSwap` is a switch, so `-StaticSwap` is `$true` and absent is `$false`; the
  condition reads once, in the env section, before `-Unset`/`-Env`. Real passes without the switch
  are byte-for-byte the same behaviour as round 2.
- Style: hashtables + one small function, matching the file's existing `[ordered]@{}` idiom; every
  new block says why; no dead code (`$regimeOwned` was replaced, not left behind).
- Not done, deliberately: the `-Extra` guard does not police flags outside the regime set (e.g. `-or`
  typos) or abbreviations of long forms (`allow_abbrev=False` rejects those at argparse anyway).

## Open items (unchanged)

1. `eval/perf.py` still cannot enable drafting (`add_draft_model_args` not passed) — a follow-up, not a
   break; the perf argv no longer asks for drafting.
2. `bench/windows/README.md` still describes the old fixed regime and does not mention
   `-DryRun`/`-CacheSize`/`-StaticSwap`.
