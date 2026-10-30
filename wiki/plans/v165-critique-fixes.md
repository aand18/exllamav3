# v165 second-review fixes — plan

Source: session `ses_e762fc6aaffeqwXTkhBgyOdswh`, message
`msg_18a21ae37001YArivM6a5bJ4zw` ("Qwen3.8-Flash-Next retuning plan critique"), reviewed against
the implementer's report (`ses_e761a5aa0ffeBB1dp8g8lYdWya`, `msg_18a04de20001iGNMGh2WCdCcDY`) and
its skill proposal (`msg_18a123bb3001Zw9IGKxG3M8W98`).

Operator instruction: implement **every** item in that review. The review's verdicts bind:

- Part 1 — six code defects: they are defects, fix all six.
- Part 2 — skills: **do not create** `benchmark-harness-integrity`, **do not create**
  `fit-before-measure`, **reject** `integrating-external-critique`; **extend only**
  `measurement-protocol`.

## Spec (binding requirements, transcribed from the review)

1. `run_pass.ps1` must honour `state.json`'s `.draft` block and `.knob_arms_disable_swap`
   (draft args, `-cs` for the drafting runner, `EXL3_MOE_CPU_SWAP=0`), and must not emit a
   duplicate `-cs`.
2. `eval/spec_decode.py`: `warm_corpus()` must deserialize stringified
   `tool_calls[i].function.arguments` exactly as `measure()` does, and dynamic expert swaps must
   not silently contaminate a multi-arm comparison.
3. `bench/verify_task.py`: the warm-placement gate must check **every** `CPU split experts` line
   (the MTP head is line 2 of `perf-mtp-ref-318-20261029-222244.log` and is cold); the percentage
   gate must check a number instead of printing PASS on any `%`; `--pre` must pass `--draft` to
   `size_check.py`.
4. `bench/windows/deploy.sh` must not leave a stale deployed `state.json` (split brain with the
   repo copy that `verify_task --pre` validates).
5. `alloc peak` must be printed **after** generation, in both `eval/perf.py` and
   `eval/spec_decode.py`.
6. `measurement-protocol` gains the four stateful caches, the fit-before-measure check and the
   `verify_task --pre`/`--post` gate — and nothing else is created.

## Global constraints

- **No GPU measurement pass in this plan.** Nothing loads a model. Verification is offline:
  PowerShell `-DryRun`, stdlib `unittest` fixtures, synthetic logs/reports, `py_compile`.
- The Windows host is production: no installs, no process kills, no deleting logs or results.
- The repo is the source of truth; `bench/windows/*` and `eval/perf.py`, `eval/spec_decode.py`
  reach the box **only** through `bench/windows/deploy.sh`. Never edit a deployed copy.
- No new skill files and no new skill directories. One edit to
  `~/.config/opencode/skills/measurement-protocol/SKILL.md`.
- Python 3.12, stdlib only (no pytest on this box), keep each file's existing style and idioms.
- One atomic commit per task, message style `bench: …` / `skill: …` / `plan: …` as fits the repo.
- Report files live in the SDD workspace; the full report goes to the file, the reply is short.

---

## Task 1 — `deploy.sh` ships `state.json` every time

`bench/windows/deploy.sh:25-27` installs `state.json` only when the destination has none. Once the
box has a copy, a regime change in the repo is silently not deployed, while
`bench/verify_task.py --pre` validates the repo copy — the exact repo-vs-deployed split brain the
script exists to prevent.

Requirements:

- Always `install -m 0666 "$HERE/state.json" "$DEST/state.json"`.
- Report which happened: print `state.json: updated` when the bytes changed, `state.json: unchanged`
  otherwise (`cmp -s` against a temporary copy, or hash before/after — your choice, keep it POSIX
  `bash` with `set -euo pipefail`).
- Replace the old comment with the new rule: the repo is the source of truth, a regime change is
  recorded and deployed, never edited on the box.
- Add `state.json` to the closing `sha256sum` line.

Verification (offline, do this):

```
rm -rf /tmp/opencode/deploy-test
DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh            # state.json: updated
DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh            # state.json: unchanged
sed -i 's/"split": 376/"split": 372/' bench/windows/state.json    # then revert!
DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh            # state.json: updated, dest split 372
git checkout -- bench/windows/state.json
DEST=/tmp/opencode/deploy-test bench/windows/deploy.sh
```

The repo's `state.json` must be byte-identical to `HEAD` when you finish (`git diff --stat` clean
for that file).

## Task 2 — `run_pass.ps1`: `.draft`, `.knob_arms_disable_swap`, one `-cs`, `-DryRun`

Evidence (already confirmed, do not re-litigate): `perf-val-spec-20261029-233352.log:160-163` shows
`-cs 131072 … -mtp -cs 40960`, `ARM … acc=1.74/4.00` (4 draft tokens while the state says 2), and an
`env` line with **no** `EXL3_MOE_CPU_SWAP`. `run_pass.ps1` never references `$ST.draft` or
`$ST.knob_arms_disable_swap`.

Exact engine surface at 1.6.5 (verified in the installed `model_init.py`):

| state.json field | flag |
|---|---|
| `draft.mode: "mtp"` | `-mtp` (also `-dm <model>` is **not** used; the MTP head rides the main model) |
| `draft.num_draft_tokens` | `-ndt N` |
| `draft.sparse_draft_head` (>0) | `-sdh N` |
| `draft.hybrid_draft_tokens` (>0) | `-hdt N` |
| `draft.dynamic_draft: true` | `-dds` |
| `draft.spec_decode_cache_size` | the `-cs` value **for `-Runner spec_decode`** |
| `draft.readback` | **no CLI surface at 1.6.5** (only `-rb_ab`, which measures four arms); keep it informational and say so in the comment |

Requirements:

1. Build the draft args from `$ST.draft` when present; a missing `.draft` block must keep the old
   behaviour (no draft flags). Draft args are emitted **only for `-Runner spec_decode`** —
   `eval/perf.py` registers no draft-model arguments (`add_draft_model_args` is not passed) and
   unpacks four values from `model_init.init`, so `-mtp`/`-ndt` in a perf argv aborts at argparse.
   A pp/tg ladder measures the main model; drafting is not part of its regime.
2. Cache size: add `-CacheSize <int>` (0 = from state). For `-Runner perf` the state value is
   `$ST.cache_size`; for `-Runner spec_decode` it is `$ST.draft.spec_decode_cache_size`, falling
   back to `$ST.cache_size` when the draft block or the field is absent. `-cs` appears **exactly
   once** in the emitted argv, and the value that is used is the value recorded in the log header
   and in the `args` column of `passes.csv` (do not overload the `max_length` column: a column named
   for one quantity must not carry another, and the existing CSV header cannot grow a column
   without rewriting the series).
3. `knob_arms_disable_swap: true` → `EXL3_MOE_CPU_SWAP=0` in the regime env, applied **before**
   `-Env`/`-Unset` so an arm's delta can still override it (that ordering is already how `-Env`
   works — keep it). Record it in `$envApplied` so it lands in the log header and `passes.csv`.
   **Scope:** the field is named for knob arms, and the pp/tg ladder is not one — 44 archived rows
   were measured with the engine default (dynamic placement), so forcing static placement on every
   ladder run would silently break the series the T4 thresholds and the T10 ladder are read against.
   Apply it automatically for `-Runner spec_decode` (the arm-to-arm drafting comparisons) and for
   any run that passes a new `-StaticSwap` switch; a perf ladder run without `-StaticSwap` keeps the
   engine default. Prove both with dry runs.
4. `-Extra` may not restate a flag the regime owns. The guard must be spelling-proof, because
   argparse accepts more spellings than a token scan sees: `-cs=40960` and `--cache_size 40960` both
   set the cache size while a `-eq` scan counts one `-cs`. Build the owned set from the regime-built
   argv (so the state-derived draft flags `-mtp/-ndt/-sdh/-hdt/-dds` are included when emitted),
   strip a `=value` suffix, map every long alias to its short name, and abort `exit 2` naming the
   canonical flag — `ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, or edit
   state.json)`. Run the guard **before** the spec_decode `-max_length` strip loop, so a duplicate
   there aborts instead of being silently eaten. Update the usage examples that show
   `-Extra "-cq Q8"` — that spelling now aborts by design.
5. `-DryRun` switch: after all gates (state file, model present, engine identity) and after env
   assembly, print exactly three lines and `exit 0` **before** `Start-Process`:

   ```
   DRYRUN runner=<perf|spec_decode> state=<state path>
   ARGS <argv joined by spaces>
   ENV <k=v sorted, spaces>
   ```

   It must create no log, no CSV row, no watcher, and it must not perform the frozen-profile
   restore (the `master → active` copy writes to `engine-state/`; set the variable and report it
   without copying). It must not require a free GPU.

Verification (offline, on this box, no model load):

```
bench/windows/deploy.sh
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-perf -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-perfstatic -DryRun -StaticSwap
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-spec -DryRun -Runner spec_decode
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-dup -DryRun -Extra "-cs 40960"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-dup2 -DryRun -Extra "-cs=40960"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-dup3 -DryRun -Extra "--cache_size 40960"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-dup4 -DryRun -Runner spec_decode -Extra "-ndt 4"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dryrun-env -DryRun -Runner spec_decode -Env "EXL3_MOE_CPU_SWAP=1"
```

Expected: the plain perf dry-run has `-cs 131072` once, **no** draft flags, and **no**
`EXL3_MOE_CPU_SWAP` in `ENV` (the ladder keeps the engine default); `-StaticSwap` adds
`EXL3_MOE_CPU_SWAP=0`; the spec dry-run has `-mtp -ndt 2`, `-cs 40960` once and
`EXL3_MOE_CPU_SWAP=0`; all four `-Extra` runs exit 2 with the canonical flag named (`-cs` three
times, `-ndt` once); the `-Env` run on top of spec_decode shows `EXL3_MOE_CPU_SWAP=1` (delta wins).
Paste the outputs into the report. Confirm `logs/`, `results/` and `engine-state/` gained nothing.

## Task 3 — `spec_decode.py`: one prompt reader, swap gate, `alloc peak` last; `perf.py`: `alloc peak` last

Three defects, one commit (two files, same "the runner must not lie" theme):

1. `warm_corpus()` (`eval/spec_decode.py:184-191`) reads the prompt JSON without the
   `tool_calls[i].function.arguments` deserialization that `measure()` does at `:52-58`. On the
   `Agentic, *` workloads that carry stringified arguments it crashes or misformats. Extract one
   helper — `read_prompt(filename)` returning the parsed dict with `arguments` already decoded — and
   use it from **both** `measure()` and `warm_corpus()`. One implementation, no second copy.
2. Dynamic expert placement is decided at load (`EXL3_MOE_CPU_SWAP`, read in
   `block_sparse_mlp_cpu.py:830`) and the engine's own comment says a placement change "visibly
   derails a generation in progress". With it on, `Baseline` and the drafting arm are not the same
   text (measured: `perf-val-spec-…log:61` vs `:103` produce different greedy tokens from the same
   prompt). A measured arm-to-arm ratio is then meaningless.

   Add a hard gate: after `model_init.init`, determine whether dynamic placement is active
   (`config.infer_params.moe_cpu_swap_modules`, any module with `_split_dynamic` true — guard every
   attribute with `getattr`). If it is active **and** the run compares more than one arm
   (`not args.no_baseline` + n-gram arm + draft arm + `-rb_ab`), print a loud error naming the
   remedy (`EXL3_MOE_CPU_SWAP=0`) and exit 2. A one-arm run stays allowed. Provide an escape hatch
   flag `-swap_ok` (`--swap_ok`) that downgrades it to a printed warning, for deliberate
   exploration only. Make the decision a pure function
   (`swap_refusal(arms: int, dynamic: bool) -> str | None`) so it is testable without a model.
3. `alloc peak` (`eval/perf.py:209`) prints before `measure_generate`, so a `-spf` probe reports
   allocation before generation and `passes.csv` records `0,0` for spec_decode (`passes.csv:45`).
   Move the print to the **end** of `main()` in `eval/perf.py`, and add the equivalent print at the
   end of `main()` in `eval/spec_decode.py` (`alloc peak allocated=<n> MiB reserved=<n> MiB`,
   `torch.cuda.max_memory_allocated/reserved`). Keep the exact wording the parser in
   `run_pass.ps1:372` matches.
4. **The gate must have teeth in the verdict.** `bench/windows/run_pass.ps1` currently marks a run
   `fail` only for a *perf* pass with no pp/tg rows; a `spec_decode` run that exits 2 at the swap
   gate (or dies for any reason before its arms) still lands as `status=ok` with no arms — the same
   "verdict from an exit code" defect the harness already fixed once. Extend the evidence rule: a
   `-Runner spec_decode` run with zero `ARM` rows is `fail`. Do not touch anything else in that
   file (Task 2 owns it otherwise).

   Verification: `powershell.exe -NoProfile -Command` parse check of the file, a `-DryRun` to prove
   the new argv/env path is unaffected, and a code-read argument in the report naming the exact
   condition. The verdict path itself needs a real pass to exercise — say so in the report and in
   the tracker rather than claiming it was run.

Verification (offline):

- `python3 -m py_compile eval/spec_decode.py eval/perf.py`.
- Create `bench/test_gates.py` (stdlib `unittest`, run as `python3 bench/test_gates.py`; Task 4 adds
  its own cases to the same file) with: `read_prompt` on a prompt that actually carries tool calls
  (`agentic_code_01.json` has none — use the smallest `agentic_*.json` that does, and parse every
  file in `eval/prompts/` as a separate case) returns dict-valued
  `function.arguments` and every prompt file in `eval/prompts/` parses; `swap_refusal(1, True) is
  None`, `swap_refusal(2, True)` returns a message containing `EXL3_MOE_CPU_SWAP=0`,
  `swap_refusal(2, False) is None`.
- `python3 bench/test_gates.py` passes.
- `deploy.sh` then `grep -n "alloc peak" .../v165/perf.py .../v165/spec_decode.py` and demonstrate by
  inspection that no `alloc peak` print precedes `measure_generate` in the deployed copy.

## Task 4 — `verify_task.py`: per-line warm check, a real percentage gate, `--draft`, deployed-state check

Evidence: `perf-mtp-ref-318-20261029-222244.log` has 49 `CPU split experts` lines, 48 ending
`placement from saved profile`, line 2 (`mtp.layers.0.mlp`) cold — the current substring test returns
PASS on that exact run. `verify_task.py:152-154` prints PASS for any `%` in the report without
reading a number.

Requirements:

1. Warm gate: parse **every** line matching `CPU split experts`; every one must contain
   `placement from saved profile`. Report the offending lines verbatim (there is at least one, or
   the log is empty → FAIL). No `CPU split experts` line at all → FAIL (not a placement-having run).
2. Percentage gate. Replace the no-op with these rules, applied to the report text:
   - `±N %` (symmetric band) → **FAIL**: the project reports one-sided ratio intervals (C5), point at
     the ratio table.
   - A decimal interval (`0.92–0.99`, `0.92-0.99`, `(0.95 - 0.99)`) → find the ratio table in the
     report (markdown rows whose last cell is a ratio like `0.92`), take its ratio column's min/max
     at 2 decimals, and **FAIL** unless the quoted interval equals that min/max (±0.005). Quote both
     numbers in the failure message.
   - A decimal interval with no ratio table in the report → **FAIL** ("quote without a table:
     regenerate with `bench/ratios.py`").
   - Other percentage mentions (acceptance rates, "+15 % faster" prose) → print `  INFO  …` and do
     **not** affect the verdict. Never print PASS for an unchecked number.
3. `--pre` must pass `--draft mtp` to `size_check.py` when `state["active"]["draft"]["mode"] == "mtp"`.
4. Split-brain gate in `--pre`: add `--deployed PATH` (default `os.path.dirname(RESULTS_DEFAULT)`,
   i.e. `_agent/v165`; empty string disables). If `<deployed>/state.json` exists and differs from
   `--state`, FAIL with `run bench/windows/deploy.sh`. **Compare parsed JSON, not raw bytes or
   hashes** — `core.autocrlf=true` means a plain `git checkout -- state.json` writes CRLF while the
   blob is LF, so a byte comparison would fail spuriously on an innocent re-checkout. Missing file →
   print `INFO` and pass.
5. Tests: `bench/test_gates.py`, stdlib `unittest`, no network, no box paths — use `tempfile`
   fixtures and `--results`/`--logs`/`--deployed` overrides. Cases: warm-everywhere log PASS; the
   real 48-warm/1-cold shape FAIL and the failing line named; report quoting `0.92–0.99` beside a
   table whose ratios are 0.92–0.99 PASS; report quoting `(0.95-0.99)` beside that table FAIL;
   `±5 %` FAIL; interval with no table FAIL; a plain "acceptance 41 %" line does not change the
   verdict and prints INFO; deployed state differing from the repo FAIL, absent → pass. Also cover
   `--draft` pass-through by asserting the `size_check` argv built for an MTP state (extract a small
   function `size_check_argv(state)` so it is testable).
6. Arm-row gate: a cited run whose `passes.csv` row says `runner=spec_decode` and whose `run_id` has
   **no** rows in `arms.csv` (or, when that file is absent, `spec_decode.csv`) is a FAIL — a run that
   died before producing an arm must not be citable as a measurement. Test with a fixture whose
   `passes.csv` and `arms.csv` disagree.

Verification: `python3 bench/test_gates.py -v` (all pass); `python3 bench/verify_task.py --pre`
against the repo state ends with `verdict: all gates passed`; `python3 bench/verify_task.py --post
--report wiki/plans/v165-reports/T06-mtp.md` demonstrates the per-line gate (it may legitimately
FAIL on the cold reference — record the actual output in the report, do not "fix" the report).

## Task 5 — `measurement-protocol`: stateful caches, fit before measure, pre/post gates

The controller runs the RED baseline (five fresh-context samples against the current skill) and
hands you the transcripts. You write **only** the skill edit; you do not re-run the samples.

`~/.config/opencode/skills/measurement-protocol/SKILL.md` (71 lines) already covers floors, run
naming, one owner, spread-first, the recorded-number confound and "discard the first run". Add, in
the existing section order, and nothing more:

1. **Stateful caches** — replace the single "discard the first run" sentence's scope with a table of
   the four hidden states and how each bites:
   - routed-expert frequency profile (`expert_stats`): first run cold, later runs inherit its
     permutation, and every run rewrites it;
   - sparse-head token profile (`token_stats`): shared across checkpoints **and** `-sdh` sizes,
     written only while an `sdh>0` generator runs;
   - OS file cache for the streamed 50 GB ngram table: first arm in a process pays cold disk reads
     for exactly those prompts (the 6.99 t/s "baseline");
   - allocator + dynamic swap sweeps: placement mutates every 128 steps, so arm 2 starts somewhere
     else than arm 1.
   Generalisation to state: enumerate the stateful caches, freeze them, restore per arm, and never
   divide by a baseline measured before the freeze. Note that `XDG_CACHE_HOME` pins both profiles at
   1.6.5 without an engine patch, and that a baseline in the same process is not a control.
2. **Fit before measure** — one short section: compute geometry from the checkpoint headers first
   (`bench/size_check.py`: VRAM, pinned host arena, minimum `-cs`) instead of spending a 60 GB load
   to discover the answer; plot both curves (VRAM and host arena vs the knob) before choosing an
   operating point, because on this box every split failed one side and the window was empty; name
   every guard that enforces a margin (watchdog, engine host-memory guard, allocator) so two of them
   do not double-charge the same GB; a "does not fit" verdict states the **phase** (load, warmup,
   prefill band, decode).
3. **The gates before and after** — `bench/verify_task.py --pre` before the first pass of a task and
   `--post` on the report before the task is called done, and what each gate proves.

Constraints: third person; no narrative of this session; no invented numbers; keep the existing
content intact (the file may grow to ~95 lines, not more); the `description` frontmatter is unchanged
(it already triggers on this workflow); no new files, no new directories.

Verification: `wc -l` within budget; the controller runs the GREEN samples and reports which of the
five behaviours changed. If a sample still divides by a cold baseline or skips the fit check, you
get the transcripts back and tighten the wording.

## Task 6 — tracker and harness docs: record the second review and what changed

`wiki/plans/v165-critique-implementation.md` currently claims things that were false (I5: "`run_pass.ps1`
reads … draft args from it"; C3: "Set `EXL3_MOE_CPU_SWAP=0` for knob arms … ✅ (protocol + harness)").

Add a section **C. Second review (code + skills)** with one row per item: D1–D6 the six defects
(claim, evidence, fix, commit), and S1–S4 the four skill proposals with the review's verdict
(`rejected`, `folded into measurement-protocol`, `extended`, `not created`) and one line of why.
Correct the two false ✅ claims in place (they now hold only after this plan's commits — say which).
Keep the existing legend and style; do not delete rows.

Also fix the documentation that these changes invalidated — they are commands an operator would
copy and run:

- `bench/windows/README.md` still shows `-Extra "-cq Q8"` (aborts by design now) and describes the
  v1.6.4 fixed regime, and it does not document `-DryRun`, `-CacheSize`, `-StaticSwap`, or the rule
  that draft args come from `state.json` and are spec_decode-only. Correct it to what the code does
  now — same voice, no new sections unless the file already has a section for it.
- `wiki/plans/v165-reports/T06-mtp.md` records `-Extra "-rb_ab -mtp"` and the retune plan records
  `-Extra "-mtp -ndt 2"`; both abort now. Do **not** rewrite the historical reports. Instead add one
  line to the tracker's C section naming those two documents and the replacement spelling
  (`state.json` carries the draft args; `-Extra` carries only the delta).

Record in the same section what remains unrun (the T6 resume, T7 sidecars after T8, the T10 ladder
on the frozen profile, T13's 262144-with-drafter and hugepage measurements) so the tracker still
states the honest 🟡 boundary.

Verification: the tracker names each commit `git log --oneline` shows for tasks 1-4 and the skill
edit; every command in the edited README section is one the current `run_pass.ps1` accepts (check
each flag against its `param` block or run `-DryRun` once); `grep -c "✅"` is consistent with the rows
you changed.

## Parked after the final review (do not re-dispatch)

The whole-branch review triaged these as "can follow": none of them can turn a wrong number into an
accepted one. Recorded here because the execution ledger that held them is scratch.

- `deploy.sh`: lowercase `state_sha_before` beside `HERE`/`REPO`/`DEST`; the before-hash swallows a
  read error (fails safe, prints `updated`); `install` rewrites an identical file (mtime churn);
  `state.json: …` prints before the remaining installs.
- `run_pass.ps1`: `-StaticSwap` is inert when `knob_arms_disable_swap` is false (the state is the
  authority; the help text overpromises); the `-max_length → use -MaxLength` remedy is a dead end
  under `-Runner spec_decode` (that flag is stripped); the new `-St` prefix now abbreviates
  `-StaticSwap` rather than `-State` (no caller uses abbreviations); an inherited OS
  `EXL3_MOE_CPU_SWAP` is invisible in the header (pre-existing).
- `spec_decode.py` / tests: the AST gate test pins the call but not its position or polarity;
  `getattr(m, "_split_dynamic", False)`'s default is unpinned; the `-swap_ok` warning still tells
  you to pass `-swap_ok`; the gate ignores the draft model's placement; `-temp` arms are not
  counted (the count is a lower bound, and any printed ratio still needs >= 2 arms).
- `verify_task.py`: the precision rule rounds the table value to the quote's decimals, so a 1-dp
  band certifies within 0.05 per endpoint; a unit-carrying range is excluded by an enumerated list,
  so an unenumerated unit still FAILs (the loud side); `T05-memops.md` keeps four genuine FAILs
  because its table prints per-band ranges rather than a ratio column — a report in that style
  cannot have its prose ratio claims certified, and the gate fails closed.
- `-Runner` accepts any string; a typo silently runs `perf.py` (the row records `runner=perf`).
- The skill's floors table (200 MB / 2 GB) is the protocol's constant; the harness's `state.json`
  currently enforces 100/1200. Reconciling them is a regime decision, not a doc fix.
