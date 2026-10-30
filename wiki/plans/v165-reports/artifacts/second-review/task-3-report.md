# Task 3 report — `spec_decode.py`: one prompt reader, swap gate, `alloc peak` last; `perf.py`: `alloc peak` last

Status: DONE_WITH_CONCERNS (one brief item is factually impossible as written; see Concerns §1)
Commit: `c6daa77d eval: one prompt reader, swap gate on multi-arm runs, alloc peak last`
Branch: `165-win-perf`, worktree `/home/dev/.local/share/opencode/worktree/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf`

## What was implemented

### 1. One prompt reader (`eval/spec_decode.py`)

`read_prompt(filename)` was extracted verbatim from the parsing that only `measure()` did: it opens
`prompts/<filename>`, decodes every stringified `tool_calls[i].function.arguments` with
`json.loads`, and returns the parsed dict. Both callers now use it:

- `measure()` — the inline `open`/`json.load`/decode block is deleted, replaced by
  `data = read_prompt(filename)` (its now-unused `path = …` local went with it);
- `warm_corpus()` — `open`/`json.load` replaced by `data = read_prompt(filename)`.

One implementation, no second copy. The docstring says why the two passes had to converge (the
warm pass used to template a string where the measured pass templated a dict on the `Agentic, *`
workloads).

### 2. Swap gate

Three new functions before `main()`:

- `swap_refusal(arms: int, dynamic: bool) -> str | None` — pure; returns `None` when `arms <= 1` or
  `not dynamic`, otherwise a multi-line message whose remedy line contains `EXL3_MOE_CPU_SWAP=0`
  and whose third line names `-swap_ok`. No model, no CUDA, no globals.
- `dynamic_placement(model, config)` — `dynamic` as the brief defines it:
  `getattr(getattr(config, "infer_params", None), "moe_cpu_swap_modules", None) or []`, plus
  `getattr(model, "modules", None) or []`, then `any(getattr(m, "_split_dynamic", False) …)`.
  Every attribute access is `getattr`-guarded.
- `count_arms(args)` — mirrors the arm blocks in `main()`: baseline unless `-nbl`, +1 for the
  n-gram arm, +1 for the draft arm (+4 with `-rb_ab`).

In `main()`, immediately after `model_init.init(...)` (placement is fixed at load, so that is the
first moment it can be read):

```python
refusal = swap_refusal(count_arms(args), dynamic_placement(model, config))
if refusal:
    if args.swap_ok:
        print(f"{col_yellow}WARNING: {refusal}{col_default}", file = sys.stderr)
    else:
        print(f"{col_red}ERROR: {refusal}{col_default}", file = sys.stderr)
        sys.exit(2)
```

New flag, next to the other run-scope flags:

```python
parser.add_argument("-swap_ok", "--swap_ok", action = "store_true", help = "Measure a multi-arm comparison with dynamic expert placement (EXL3_MOE_CPU_SWAP) still on. The arms do not generate the same tokens, so the ratios are exploration only")
```

Engine attributes confirmed read-only against the installed engine
(`/mnt/c/Users/yoho/Downloads/tabbyAPI/venv/Lib/site-packages/exllamav3`):

- `modules/block_sparse_mlp_cpu.py:830` sets `self._split_dynamic = os.environ.get("EXL3_MOE_CPU_SWAP", "1") != "0" and not self.tid2eid_key` inside `load_cpu_split()`; it only returns `True` after that point, so `_split_dynamic` true means a split really is dynamic;
- `:504-507` creates `self.config.infer_params.moe_cpu_swap_modules` and `:513` appends only when `_split_dynamic`; `:943` reads `ip.moe_cpu_swap_modules`, so `config.infer_params` is the right home;
- `:955` comment: a placement change "visibly derails a generation in progress".

### 3. `alloc peak` last

- `eval/perf.py`: the print moved from between the prefill block and `if not args.skip_gen:` to the
  last statement of `main()` (after generation), with a comment saying why (`-spf` probes).
- `eval/spec_decode.py`: the same line added as the last statement of `main()`, identical wording
  and units.

Wording unchanged everywhere: ` -- alloc peak allocated=<n> MiB reserved=<n> MiB`, which
`bench/windows/run_pass.ps1:523` matches with `alloc peak allocated=([0-9.]+) MiB reserved=([0-9.]+) MiB`.

### 4. `bench/test_gates.py` (new, stdlib `unittest`)

Two `TestCase`s, one per subject, `python3 bench/test_gates.py`:

- `ReadPrompt.test_every_prompt_parses` — every file in `eval/prompts/` parses and yields a dict with
  a `messages` list.
- `ReadPrompt.test_stringified_arguments_are_decoded` — on `agentic_code_05.json`: the raw file's
  arguments are strings, `read_prompt` returns dicts, same count.
- `ReadPrompt.test_both_callers_use_the_one_reader` — `ast` over `eval/spec_decode.py`: both
  `measure()` and `warm_corpus()` call `read_prompt` and neither calls `open()` nor an attribute
  `load()` (i.e. no second copy of the parsing).
- `SwapRefusal.test_one_arm_is_allowed` — `swap_refusal(1, True)` and `swap_refusal(0, True)` are `None`.
- `SwapRefusal.test_multi_arm_with_dynamic_placement_is_refused` — `swap_refusal(2, True)` is not
  `None` and contains `EXL3_MOE_CPU_SWAP=0`.
- `SwapRefusal.test_multi_arm_with_static_placement_is_allowed` — `swap_refusal(2, False)` is `None`.
- `SwapRefusal.test_arm_count_follows_the_arm_blocks` — `count_arms` → 1 baseline-only, 0 `-nbl`,
  2 draft arm, 5 `-rb_ab`.

The runners import torch/tabulate/the installed engine at module scope and none of those exists in
the WSL tree (the engine lives in the Windows venv), so the file installs a stub for each *absent*
dependency before importing `eval/spec_decode.py` (`install_stubs()`), and imports the real module
when it is installed. `alloc peak` placement is checked by inspection (§Verification), not by a test,
because the brief scoped it to inspection.

## Commands and output (all captured live)

```
$ python3 -m py_compile eval/spec_decode.py eval/perf.py && echo PY_COMPILE OK
PY_COMPILE OK

$ python3 bench/test_gates.py
test_both_callers_use_the_one_reader (__main__.ReadPrompt.test_both_callers_use_the_one_reader) ... ok
test_every_prompt_parses (__main__.ReadPrompt.test_every_prompt_parses) ... ok
test_stringified_arguments_are_decoded (__main__.ReadPrompt.test_stringified_arguments_are_decoded) ... ok
test_arm_count_follows_the_arm_blocks (__main__.SwapRefusal.test_arm_count_follows_the_arm_blocks) ... ok
test_multi_arm_with_dynamic_placement_is_refused (__main__.SwapRefusal.test_multi_arm_with_dynamic_placement_is_refused) ... ok
test_multi_arm_with_static_placement_is_allowed (__main__.SwapRefusal.test_multi_arm_with_static_placement_is_allowed) ... ok
test_one_arm_is_allowed (__main__.SwapRefusal.test_one_arm_is_allowed) ... ok
----------------------------------------------------------------------
Ran 7 tests in 0.010s

OK
```

### Mutation checks (tests really fail on the regressions, then restored)

Bypassing the reader from `warm_corpus` (the original defect, restored to `open`/`json.load`):

```
Fail: test_both_callers_use_the_one_reader
  AssertionError: 'read_prompt' not found in {'open', 'Job', 'GreedySampler'} :
  warm_corpus() must read prompts through read_prompt()
Ran 7 tests ... FAILED (failures=1)
```

Weakening the gate (`if arms <= 1 or dynamic: return None`):

```
Failures: test_multi_arm_with_dynamic_placement_is_refused, test_multi_arm_with_static_placement_is_allowed
Ran 7 tests ... FAILED (failures=2)
```

`cp /tmp/opencode/sd.bak eval/spec_decode.py && diff -q …` → identical, suite green again.

### Wire-level behaviour of the gate (scratch script, stubbed model, no GPU, not committed)

Fake `model_init.init` returning a `config.infer_params.moe_cpu_swap_modules` holding a module with
`_split_dynamic = True`, then running the file's `__main__` block:

```
argv=['-ngram_min', '3', '-nwarm']              -> exit 2
   stderr: ERROR: dynamic expert placement is active (EXL3_MOE_CPU_SWAP) and this run compares 2 arms. …
argv=['-ngram_min', '3', '-nwarm', '-swap_ok']  -> continues (fails later on the stubbed Generator)
   stderr: WARNING: dynamic expert placement is active (EXL3_MOE_CPU_SWAP) and this run compares 2 arms. …
argv=['-nwarm']                                 -> continues, no stderr (one arm stays allowed)
```

Flag parsing/dest check: `PARSED swap_ok=True`; help shows `-swap_ok, --swap_ok`.

### Deploy and hash comparison

```
$ bench/windows/deploy.sh
state.json: unchanged
deployed to /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165
… sha256 of every .ps1/perf.py/state.json …

$ sha256sum eval/perf.py eval/spec_decode.py \
    /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/perf.py \
    /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/spec_decode.py
b270a0e962e2413da38b3b2a373bdbcfffe92eb3e4b9e67463d63417474a99b2  eval/perf.py
333c4f24a20c77b7123e55cb3b963680c86f4c6444ac886322090669f713034e  eval/spec_decode.py
b270a0e962e2413da38b3b2a373bdbcfffe92eb3e4b9e67463d63417474a99b2  .../v165/perf.py
333c4f24a20c77b7123e55cb3b963680c86f4c6444ac886322090669f713034e  .../v165/spec_decode.py
```

Deployed copies carry the change (identical hashes). No process was started or killed; no model
loaded; no benchmark run.

### No `alloc peak` precedes generation in the deployed copies (inspection)

```
$ grep -n "alloc peak" .../v165/perf.py .../v165/spec_decode.py
.../v165/perf.py:221:    print(f" -- alloc peak allocated=…"
.../v165/spec_decode.py:509:    print(f" -- alloc peak allocated=…"

$ grep -n "measure_generate(args, model, cache)" .../v165/perf.py
215:        generate_results = measure_generate(args, model, cache)
```

`perf.py`: the only `alloc peak` (221) sits after the last `measure_generate` call (215) and is the
last statement of `main()`. `spec_decode.py`: line 509 is after the last `measure(...)` call (431)
and after the last `arm_line(...)` call (420); nothing in `main()` follows it.

## Files changed

- `eval/spec_decode.py` — `read_prompt()` extracted and used by both callers; `count_arms()`,
  `dynamic_placement()`, `swap_refusal()` added; gate + `-swap_ok` in `main()`/argparse; `alloc peak`
  line at the end of `main()`.
- `eval/perf.py` — `alloc peak` print moved to the end of `main()` (3 lines moved, comment added).
- `bench/test_gates.py` — new offline gate suite (7 tests, 2 `TestCase`s).

Not touched: `bench/windows/run_pass.ps1`, `bench/verify_task.py`, `bench/windows/deploy.sh`,
`arm_line()` and its `ARM …` output, the `alloc peak` wording, `bench/windows/state.json`.

`git status` after the commit: only the pre-existing untracked `wiki/plans/v165-critique-fixes.md`.

## Self-review

Completeness — all three defects addressed; the brief's five verification bullets are all covered
with captured output (py_compile, test suite, deploy + grep).

Quality — the gate is two pure functions plus a thin wiring block; the reader has one
implementation and both callers go through it; no parsing is duplicated and the `ast` test enforces
that going forward.

Discipline — no refactors beyond the brief: the only collateral edits are the deletion of the
now-dead `path` local in `measure()` and one `-swap_ok` argparse entry. `arm_line()` output, the
`alloc peak` wording/units, and the per-category table are untouched.

Testing — which test catches which regression:

- reader bypassed in `warm_corpus`/`measure` → `ReadPrompt.test_both_callers_use_the_one_reader`
  (demonstrated: fails);
- gate deleted → `SwapRefusal.test_multi_arm_with_dynamic_placement_is_refused` (AttributeError on
  the missing function);
- gate weakened to allow dynamic multi-arm → same test plus
  `test_multi_arm_with_static_placement_is_allowed` if the condition is inverted (demonstrated:
  fails);
- `count_arms` losing the `-nbl`/`-rb_ab`/draft-arm terms → `test_arm_count_follows_the_arm_blocks`
  (this is the input that decides whether the gate can fire at all);
- `read_prompt` losing the decode → `test_stringified_arguments_are_decoded`.

## Concerns

1. **The brief's `read_prompt` test target does not carry the case it describes.**
   `eval/prompts/agentic_code_01.json` has **no** `tool_calls` at all (0 in all four messages), so it
   cannot exercise "returns dict-valued `function.arguments`". I kept it covered (it is parsed by
   `test_every_prompt_parses`) and put the decode assertion on `agentic_code_05.json`, the smallest
   workload that does carry stringified arguments (4 tool calls, all `str`). Workloads with them:
   agentic_code_05/10/20/29, agentic_curl_05/10/15/16. If you want the literal file tested for
   decoding, the corpus needs a different fixture, not a code change.
2. Minor line references in the brief are stale: the `alloc peak` regex is at
   `run_pass.ps1:523` (not 372), and `warm_corpus()` is at `eval/spec_decode.py:176-191` (not
   184-191). Content matched the description in both cases.
3. The `alloc peak` line is skipped on the "No results to display" early return in `spec_decode.py`
   (a run that measured no arm). Kept as-is: it is a degenerate invocation, and hoisting the print
   above the table would reintroduce a "watermark before the report" ordering.
4. The refusal/warning goes to **stderr**. `run_pass.ps1` appends the `.err` log into the run log
   before parsing (`Get-Content $err | Add-Content $log`), so the text is in the log and the incident
   is diagnosable, but `passes.csv`'s verdict for a spec_decode runner is not keyed on the exit code
   ("no pp/tg rows" only fails a `perf` runner) — a refused spec_decode pass would land as `ok` with
   no arms. That verdict gap is Task 2/4 territory; I did not touch it.
5. Harness interaction to be aware of: `run_pass.ps1` already sets `EXL3_MOE_CPU_SWAP=0` for
   `-Runner spec_decode` when `state.knob_arms_disable_swap` is true, so normal arms will not trip
   the gate. A deliberate `-Env "EXL3_MOE_CPU_SWAP=1"` arm will now be refused (exit 2) unless
   `-Extra "-swap_ok"` is passed — which is the intended meaning of the escape hatch.
6. `-swap_ok` is annotated because the brief spelled the signature `swap_refusal(arms: int, dynamic:
   bool) -> str | None`; `count_arms()`/`dynamic_placement()` carry no annotations, matching the
   file's existing style. If the reviewers prefer no annotations anywhere, dropping them is a
   one-line change with no functional effect.

---

# Requirement 4 (added after review) — an arm-less `spec_decode` pass must not read `ok`

Commit: `db1029e3 bench: a spec_decode pass with no ARM rows is fail, not ok`
Status: DONE_WITH_CONCERNS

## Brief state

The brief was not regenerated as stated: `.superpowers/sdd/v165-critique-fixes/task-3-brief.md` is
unchanged (mtime 02:08, before this report at 02:16; still 42 lines, still "Three defects", no
requirement 4, verification list still the offline five). Requirement 4 was implemented from the
text in the assignment message, which is self-contained. Nobody needs to re-do anything, but the
file on disk does not contain what the message says it contains.

## What changed (one hunk, `bench/windows/run_pass.ps1`)

```powershell
# --- Verdict -----------------------------------------------------------------------------------
$status = "ok"
if ($breach) { $status = "breach" }
elseif ($traceback) { $status = "fail" }
elseif ($Runner -eq "perf" -and $ppRows.Count -eq 0 -and $tgRows.Count -eq 0) { $status = "fail" }
# A drafting pass that measured no arm is a failure, for the same reason the ladder rule above exists:
# the venv python's exit code is unreadable through the uv shim, so the verdict has to come from the
# evidence. spec_decode.py now exits 2 at the swap gate before any arm runs, and a pass that dies
# before its first ARM line is indistinguishable from one that ran - both leave no rows, and "ok with
# no arms" reads as a measured comparison that does not exist.
elseif ($Runner -eq "spec_decode" -and $armRows.Count -eq 0) { $status = "fail" }
```

The exact condition: `$Runner -eq "spec_decode" -and $armRows.Count -eq 0`. Precedence, unchanged:

1. `$breach` → `breach` (watcher floor crossed, BREACH file, or sampled minimum under a floor);
2. else `$traceback` → `fail` (a `Traceback (most recent call last)` in the runner's own output,
   before the `=== pass` header, so the header cannot fake it);
3. else the perf rule (`$Runner -eq "perf"` and zero pp rows and zero tg rows) → `fail`;
4. else the new arm rule → `fail`;
5. else `ok`.

So a breach still outranks a missing row, a traceback still outranks it, and the two no-evidence
rules are mutually exclusive by `$Runner` (their relative order is therefore immaterial). The perf
line is byte-identical to before; the new line is an added sibling `elseif`. Nothing else in the
file was touched — `-DryRun`, argv construction, env application, parsing and persistence are
untouched, and `$armRows` is the ArrayList the `^ARM\s+…` regex loop already fills
(`run_pass.ps1:485`, before the verdict).

Task 4's `verify_task.py --post` arm-row gate stays the archive-side check of the same case; this is
the pass-time side.

## Commands and output

Parse check on the **deployed** copy (not the tree copy):

```
$ powershell.exe -NoProfile -Command '$errs = @(); $null = [System.Management.Automation.Language.Parser]::ParseFile(
    "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1", [ref]$null, [ref]$errs); "parse errors: " + $errs.Count'
parse errors: 0
```

Deployed copy carries the change:

```
$ sha256sum bench/windows/run_pass.ps1 /mnt/c/.../\_agent/v165/run_pass.ps1
9c3a61463cec97ded9d50b9c962bb912025223b3238989d038e257abc8e4fdec  bench/windows/run_pass.ps1
9c3a61463cec97ded9d50b9c962bb912025223b3238989d038e257abc8e4fdec  /mnt/c/.../\_agent/v165/run_pass.ps1
```

`-DryRun` on each runner (both exit 0; no log, no CSV row, no watcher, GPU untouched):

```
$ … run_pass.ps1 -Tag dryperf -Runner perf -DryRun
DRYRUN runner=perf state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 131072 -mcs 376 -mct 16 -sd
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
exit=0

$ … run_pass.ps1 -Tag dryspec -Runner spec_decode -DryRun
DRYRUN runner=spec_decode state=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\state.json
ARGS -m D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 -cs 40960 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16 -mtp -ndt 2
ENV EXL3_HOST_MEM_RESERVE_MB=0 EXL3_MOE_ARENA_HUGEPAGE=0 EXL3_MOE_CPU_PIN=1 EXL3_MOE_CPU_SWAP=0 EXL3_MOE_CPU_SWIZZLE=1 EXL3_MOE_MEMOPS=0 FROZEN_STATE=master->active PYTHONUTF8=1 XDG_CACHE_HOME=C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\engine-state\active
exit=0
```

Argv/env path unchanged, mechanically:

- `git diff -U0 -- bench/windows/run_pass.ps1` has one hunk header, `@@ -533,0 +534,6 @@` — six
  lines added after line 533; no line at or before 533 is modified or removed;
- the `-DryRun` exit is at line 366 in both `git show HEAD:bench/windows/run_pass.ps1` and the
  deployed copy, and `diff` of lines 364-373 between the two is empty — the argv/env code path is
  byte-identical, so it cannot have changed;
- the spec_decode DryRun also confirms Task 2's `EXL3_MOE_CPU_SWAP=0` scoping is what a harness
  drafting pass gets, i.e. the new swap gate will not fire in a normal arm set.

## What is NOT exercised (honest limits)

- The verdict path itself ran nowhere. `$status` is computed after `Start-Process` returns, i.e.
  only in a real pass, so the new `elseif` has never executed. This was verified by parse + code
  read, not by a run.
- Specifically unexercised: a live `-Runner spec_decode` pass that exits 2 at the swap gate (would
  now write `status=fail`, `args` cell with the frozen `EXL3_MOE_CPU_SWAP=1`, and no
  spec_decode.csv rows); a spec_decode pass killed during its baseline arm (also `fail` now, where
  before it could read `ok`); and the ordering against a simultaneous breach (code-read only:
  `breach` still wins).
- The next real spec_decode pass is the first execution. The cheapest deliberate exercise is a
  one-arm-cost dry experiment once the card is free: `-Runner spec_decode -Env
  "EXL3_MOE_CPU_SWAP=1"` should refuse at the gate and land `fail` in passes.csv. Not run here (no
  GPU work in this task).
- Task 4's `verify_task.py --post` arm-row gate will catch the same case from the archive side, so a
  missed verdict is still caught downstream once that exists.

## Files changed

- `bench/windows/run_pass.ps1` — one added `elseif` in the verdict block plus a 5-line comment
  (the only hunk). `bench/verify_task.py`, `watch.ps1`, `state.json`, `deploy.sh` untouched.
- Deployed: `C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1`, hash `9c3a6146…` = tree.

---

# Review fix round — mutation-proofed coverage, and a false fail in the requirement-4 rule

Commit: `25d7a6e0 bench: spec_decode fail needs no ARM line and no table row`
Status: DONE

Four findings from the review of `c6daa77d` + `db1029e3`. No runner logic changed: the only
production change is the requirement-4 condition (finding 3); findings 1, 2 and 4 are test coverage.

## Finding 1 — the gate's wiring into `main()` was untested

New `SwapRefusal.test_main_wires_the_gate_in`, modelled on `ReadPrompt.test_both_callers_use_the_one_reader`
(`bench/test_gates.py:139-155`): it parses `eval/spec_decode.py`, finds the single `main()`, and
asserts

- `swap_refusal`, `dynamic_placement` and `count_arms` are all called in it (the decision is not
  hard-wired, the placement is read rather than assumed, the arm count is not a constant);
- exactly one `sys.exit(...)` call exists, and it sits inside an `if` whose test mentions `refusal`
  (a deleted guard, or an exit moved out of it, is a run that reports a ratio it should have refused);
- `args.swap_ok` is consulted, so the escape hatch cannot be dropped while the refusal stays.

Mutation-proved on a mirror (`/tmp/opencode/mirror`, repo files untouched):

```
MUTATION A: deleted the gate block from main(), helpers kept
AssertionError: 'swap_refusal' not found in {'reset_arm', 'tabulate', 'print_stats', …} :
  main() must ask swap_refusal()
FAILED (failures=1)   [test_main_wires_the_gate_in]
```

Bonus mutation in the same block, `sys.exit(2)` removed but the two prints kept:

```
MUTATION C: refusal prints but no longer exits
AssertionError: 0 != 1 : the refusal path must exit, and exit once
FAILED (failures=1)   [test_main_wires_the_gate_in]
```

## Finding 2 — `test_arm_count_follows_the_arm_blocks` never set `s_ngram_match_min`

The test now walks the n-gram arm as well: `-ngram_min 3` beside a baseline → 2 arms, and with
`-nbl` → 1 arm. Mirror mutation:

```
MUTATION B: count_arms() no longer counts the n-gram arm
AssertionError: 1 != 2
FAILED (failures=1)   [test_arm_count_follows_the_arm_blocks]
```

That mutation is the one that mattered: a baseline + `-ngram_min 3` run prints `N-gram (greedy)`
beside `Baseline`, i.e. exactly the arm-to-arm ratio the gate forbids, and it counted as one arm.

## Finding 3 — the requirement-4 condition had a false fail

`$armRows.Count -eq 0` alone is not "no evidence": a legitimate single-arm n-gram pass
(`-nbl -ngram_min 3`) calls `arm_line()` for neither arm (`eval/spec_decode.py:364` baseline,
`:420` draft), so it emits no `ARM` line — but it does print the per-category table, which the
harness parses into `$sdRows` (`run_pass.ps1:484` collection, `:514` rows). Corrected condition:

```powershell
elseif ($Runner -eq "spec_decode" -and $armRows.Count -eq 0 -and $sdRows.Count -eq 0) { $status = "fail" }
```

A swap-gate refusal exits 2 before the first workload, so it produces neither an `ARM` line nor a
table row and still fails; the one-arm n-gram pass no longer does. The comment above the line now
names that case (why both sets are required to be empty), and the wording "measured no arm" is gone
because it was the false assumption.

Verification, and its limits:

- deployed copy parses cleanly:

  ```
  $ powershell.exe -NoProfile -Command '$errs = @(); $null = [System.Management.Automation.Language.Parser]::ParseFile(
      "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1", [ref]$null, [ref]$errs); "parse errors: " + $errs.Count'
  parse errors: 0
  ```
- deployed hash `f1e86f6cab16dd7dca506e406d92151af6d108801de1af118b31a131430884df` = tree copy;
- `-DryRun` on the affected runner still prints the same argv/env and exits 0
  (`ARGS … -mtp -ndt 2`, `ENV … EXL3_MOE_CPU_SWAP=0 …`), and `git diff -U0` for this round is one
  hunk in the verdict block (`@@ -534,6 +534,7 @@`) — nothing above it, including the `-DryRun`
  exit at :366, changed;
- code-read precedence (unchanged): `breach` → `traceback` → perf no-rows → spec_decode no-evidence
  → `ok`;
- **still unexercised:** the verdict path only runs after `Start-Process`, so neither the new
  `elseif` nor the corrected condition has executed. Untested live: a swap-gate refusal landing
  `status=fail`, a killed baseline arm landing `fail`, and the both-empty case in general. Task 4's
  `verify_task.py --post` arm-row gate is the archive-side check of the same class.

## Finding 4 — `dynamic_placement()` had no test

`SwapRefusal.test_dynamic_placement_reads_the_registry_and_the_modules`, all fakes, no engine
import: registry absent (`infer_params` without the attribute) → False; registry with
`_split_dynamic = False` → False; registry with `_split_dynamic = True` → True; `infer_params = None`
→ False; and a dynamic module present only in `model.modules` with `infer_params = None` → True.
Mirror mutation:

```
MUTATION D: dynamic_placement() ignores model.modules
FAIL: test_dynamic_placement_reads_the_registry_and_the_modules
FAILED (failures=1)
```

## Suite and state

`python3 bench/test_gates.py -v` → **9 tests, OK** (was 7):

```
test_both_callers_use_the_one_reader                        ok
test_every_prompt_parses                                    ok
test_stringified_arguments_are_decoded                      ok
test_arm_count_follows_the_arm_blocks                       ok
test_dynamic_placement_reads_the_registry_and_the_modules   ok
test_main_wires_the_gate_in                                 ok
test_multi_arm_with_dynamic_placement_is_refused            ok
test_multi_arm_with_static_placement_is_allowed             ok
test_one_arm_is_allowed                                     ok
Ran 9 tests in 0.015s — OK
```

Repo working tree after the commit: clean except the pre-existing untracked
`wiki/plans/v165-critique-fixes.md`. Mirror deleted after use; no repo file was mutated to prove a
test.
