# Task 6 report — tracker section C, and the harness documentation it invalidated

Status: **DONE**

Commits (two, one file each — the repo splits `bench:` for the harness from `plan:` for
`wiki/plans/`, and the brief allows the split; the README lands first because C's documentation
bullet says the README already describes the harness):

```
dfa36468 plan: tracker records the second review - six defects fixed, one skill extended
7c8754e2 bench: README documents the current harness - state regime, -DryRun, drafting
```

Range: `1c5b9670..dfa36468`, 2 files, +165 / −33. Nothing else touched: no code, no `state.json`, no
report numbers, no historical report edited (`git diff --stat 1c5b9670..HEAD` lists only
`bench/windows/README.md` and `wiki/plans/v165-critique-implementation.md`).

## 1. Tracker — `wiki/plans/v165-critique-implementation.md` (+95 / −2)

New `## C. Second review (code + skills)`, placed between B and "Conflicts resolved" so the three
item tables sit together; C's intro says the sections after it ("Conflicts resolved", "Verification
performed", "Not exercised yet") describe the **first** pass and C records what the second review
changed on top. That is the one structural decision: lettered section, existing sections untouched
in place.

Contents, in order:

- **Intro** — source session/message, the report and skill proposal it was read against, the
  Part-1/Part-2 split (six defects fixed; four skill verdicts binding), the commit range
  `79926691..1c5b9670`, and the pointer to what follows.
- **D1–D6 table** — `# | Defect (the review's claim) | Evidence | Fix | Commit | Status`:
  - D1 `deploy.sh` state.json shipped only when absent → `5134738b`, ✅ (four scratch deploys).
  - D2 `run_pass.ps1` ignored `.draft` / `.knob_arms_disable_swap`, second `-cs`, default 4 draft
    tokens → `7c043454`, `89f0b787`, `8c69b5ec`, ✅ (dry run quoted).
  - D3 `warm_corpus()` without the `tool_calls[].function.arguments` decode → `c6daa77d`, ✅.
  - D4 dynamic placement contaminating multi-arm ratios, and the verdict teeth that stop a refusal
    landing as `ok` → `c6daa77d`, `db1029e3`, `25d7a6e0`, 🟡 (gate tested offline; a model must load).
  - D5 `alloc peak` printed before generation → `c6daa77d`, 🟡 (only a real pass writes the value).
  - D6 `verify_task.py` gates as no-ops → `ffec2e50`, `1c5b9670`, ✅ (34 tests; real artefacts).
- **The two false claims, corrected in place** (no row deleted):
  - C3 Status `✅ (protocol + harness)` → `✅ (protocol; the harness half was false until
    7c043454 + 8c69b5ec — see D2/D4)`.
  - I5 Status `✅` → `✅ (draft args and the drafting cache only since 7c043454/89f0b787 — see D2)`.
  - plus the paragraph naming what makes each true: C3 by `7c043454` (env written) and `8c69b5ec`
    (scoped to `-Runner spec_decode` / `-StaticSwap`), I5 by `7c043454` (the `.draft` block) and
    `89f0b787` (draft args spec_decode-only; `-CacheSize` carries the drafting cache).
- **S1–S4 table** — `# | Proposal | Verdict | Why` with the review's words: S1
  `benchmark-harness-integrity` **not created**; S2 `fit-before-measure` **folded into
  `measurement-protocol`**; S3 extend `measurement-protocol` **extended** (only skill change made);
  S4 `integrating-external-critique` **rejected**.
- **`### S3 — what the skill now says, and the evidence it was needed`** — 71 → 99 lines,
  frontmatter `description` unchanged, no commit (skills dir is not a git repo); the three sections
  (fit comes before the pass; the four-row stateful-cache table incl. `token_stats` keyed on the
  tokenizer hash + vocab size and the 51.8 GB ngram table; the gates before/after with exit 1
  semantics); the RED/GREEN numbers (old skill 0/4 `size_check.py`, 0/4 `verify_task.py --pre/--post`,
  4/4 on state and baseline; new skill 5/5 and 5/5, no regression; the discarded contaminated RED
  rep explained so 0/4 vs 5/5 is not a puzzle).
- **`### Documentation these commits changed, and the spellings that now abort`** — the README
  correction, then the one line the brief asked for: `wiki/plans/v165-reports/T06-mtp.md:10`
  (`-Extra "-rb_ab -mtp"`) and `wiki/plans/2026-10-29-v165-retune.md:206,511` (`-Extra "-mtp -ndt 2"`
  / `-Extra "-rb_ab -mtp"`) abort now, are **not** rewritten, and `state.json` carries the draft args
  so the replacement spelling is `-Extra "-rb_ab"` — `-Extra` carries only the delta.
- **`### Still unrun after this plan (the honest 🟡 boundary)`** — T6 resume (first real
  `-Runner spec_decode` pass at `-mtp -ndt 2`/`-cs 40960`, first swap-gate opportunity, first
  alloc peaks), T7 sidecars after T8 (`1385306e` / C12), T10 ladder on the frozen profile (needs
  C2's warm re-measure), T13's 262144-with-drafter and the hugepage measurements.

## 2. `bench/windows/README.md` (+70 / −31)

No new sections (the brief's rule); every change is a corrected paragraph, table row or example in a
section that already existed:

- **Inventory** — `run_pass.ps1` now says it runs either runner and prints `ARM` lines; `deploy.sh`
  says what it actually ships (scripts, `state.json`, sidecar specs, `perf.py`, `spec_decode.py`,
  prompts); `BREACH-<tag>-<utc>.txt` in the `watch.ps1` row.
- **Floors** — 200 MB / 2048 MB → **100 MB / 1200 MB**, the operator's numbers, with their source
  (`floors` in `state.json` for a pass, `-MinFreeVramMB`/`-MinFreeRamMB` for a build). This was
  false before my edit, not merely stale: `run_pass.ps1` reads the state floors (100/1200) and
  `run_build.ps1` defaults to 100/1200.
- **Deploy** — one paragraph: `deploy.sh` ships `state.json` every run, prints `updated`/`unchanged`,
  and `verify_task.py --pre` fails on a repo-vs-box disagreement.
- **Run a pass** — `_agent/v164` → `_agent/v165` in the example; flags table gained `-CacheSize`,
  `-Model`, `-Runner`, `-StaticSwap`, `-DryRun`; `-Threads`/`-Split` defaults are now "0 takes it
  from `state.json`" instead of the dead 16/360; the `-Extra "-cq Q8"` row is replaced by the
  regime-owned rule (spellings included); `-Skip*` marked perf-only; `-Tag` documents the run id.
- **Regime paragraph** — was "fixed in `run_pass.ps1`: model 2.05bpw, `-mcs 360 … -cs 262144`", is
  now: the regime lives in `state.json` (active: 3.18bpw, `-mcs 376 -mct 16 -cs 131072`…, env list),
  every switch is a delta, `-Regime production` is an alias.
- Two further paragraphs in the same section: the frozen-profile restore (`XDG_CACHE_HOME` ←
  `frozen_state_master`, `-NoFrozenState`) and the drafting rule (`-Runner spec_decode` only; flags
  from `state.json`; `-cs` = `draft.spec_decode_cache_size`; a `perf.py` ladder cannot carry them;
  `-Extra "-rb_ab"` is a delta, `-Extra "-rb_ab -mtp"` is refused).
- **`-DryRun`** — one paragraph plus the two example commands, both verified.
- **Outputs** — `passes.csv`/`pp.csv`/`tg.csv` columns now include `run_id`, `log`, `runner`,
  `peak_alloc_mb`, `peak_reserved_mb`; `spec_decode.csv` and `arms.csv` rows added; `watch-` and
  `BREACH-` file names carry the UTC stamp; `logs/perf-<tag>-<utc>.log` + `.err.log`; `builds.csv`
  columns corrected to `run_id … log`.
- **`status`** — the three values plus what `fail` means now (traceback, a perf pass with no pp/tg
  rows, a drafting pass with neither an `ARM` line nor a table row), and why the verdict is evidence-
  based.
- **Relationship to the plan** — `2026-10-24-v164-retune.md` → `2026-10-29-v165-retune.md`; the
  v164-specific task narrative replaced by what is true of v165 (Task 0 deploys and records ground
  truth; later tasks take flag combinations only), with the new switches in the list.

## Verification

All offline. No GPU work, no model load, no process started on the box: the only Windows processes
were `-DryRun`s (which exit before `Start-Process`) and the `python -c` identity probes the harness
runs itself.

1. **Every commit in C exists and is the right one.** `git log --oneline 79926691..1c5b9670` lists
   the nine commits; each SHA named in C appears exactly once:

   ```
   5134738b 1   7c043454 1   89f0b787 1   8c69b5ec 1   c6daa77d 1
   db1029e3 1   25d7a6e0 1   ffec2e50 1   1c5b9670 1
   ```

2. **Every switch in the README is one `run_pass.ps1` accepts.** Extracted every `-Flag` in the
   switch table and cross-checked against the `param` block:

   ```
   table switches: ['-Tag', '-MaxLength', '-Env', '-Unset', '-Threads', '-Split', '-CacheSize',
                    '-Model', '-Runner', '-Extra', '-StaticSwap', '-DryRun', '-NoSpecDec',
                    '-SkipPrefill', '-Prof']
   not a parameter: []
   ```

3. **The commands themselves, dry-run on the box** (the deployed `run_pass.ps1` is byte-identical to
   HEAD: sha256 `f1e86f6cab16…` on both, so no redeploy was needed):

   ```
   $ …run_pass.ps1 -Tag armA-1 -MaxLength 262144 -DryRun          # the example, + -DryRun
   DRYRUN runner=perf state=…\_agent\v165\state.json
   ARGS -m D:\llms\…-3.18bpw_h6_ng8 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2 -max_length 262144
        -mcs 376 -mct 16 -sd
   ENV  … (no EXL3_MOE_CPU_SWAP) …
   EXIT=0

   $ …run_pass.ps1 -Tag dry-perf -DryRun
   ARGS … -cs 131072 … -max_length 131072 -mcs 376 -mct 16 -sd            EXIT=0

   $ …run_pass.ps1 -Tag dry-spec -Runner spec_decode -DryRun
   DRYRUN runner=spec_decode …
   ARGS … -cs 40960 -chunk_size 4096 -ambs 2 -cq 2,2 -mcs 376 -mct 16 -mtp -ndt 2
   ENV  … EXL3_MOE_CPU_SWAP=0 …                                           EXIT=0
   ```

   and the three claims the new text makes about `-Extra`, each reproduced:

   ```
   -Extra "-rb_ab"          → appended (`… -mtp -ndt 2 -rb_ab`),            EXIT=0
   -Extra "-rb_ab -mtp"     → ABORT: -Extra repeats -mtp, which the regime owns (edit state.json)
   -Extra "-cq Q8"          → ABORT: -Extra repeats -cq, which the regime owns (edit state.json)
   -Extra "--cache_size 40960" → ABORT: -Extra repeats -cs, which the regime owns (use -CacheSize, …)
   -Extra "-dr 3"           → appended (`… -dr 3 -sd`),                     EXIT=0
   ```

4. **Nothing was written by any of it.** After a marker file, nine dry runs, then:

   ```
   $ find /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165 -newer <marker>
   (no output)
   ```

   `logs/` still holds 106 files and `results/` 144; newest log is still
   `perf-val-spec-20261029-233352.log`; `engine-state/` still holds only `active/` and `master/`.

5. **The D6 status claim re-checked here, not quoted from another report:**

   ```
   $ python3 bench/verify_task.py --pre | tail -1
   verdict: all gates passed                                          exit 0
   $ python3 bench/verify_task.py --post --report wiki/plans/v165-reports/T06-mtp.md
   FAIL  rb-ab-166-20261029-202655: 1 of 49 CPU-split component(s) have no saved placement …
         -- CPU split experts (worker, dynamic): mtp.layers.0.mlp [152..512) of 512
   FAIL  `±5 %` is a symmetric band …
   FAIL  `0.87–1.10` does not match this report's ratio table: the table's ratios run 0.96 to 1.09
   FAIL  `0.47–1.04` does not match …
   verdict: CONTRADICTIONS FOUND                                       exit 1
   $ grep -c "CPU split experts" …/perf-mtp-ref-318-20261029-222244.log
   49                                                                  # line 2 = mtp.layers.0.mlp, cold
   $ python3 bench/test_gates.py
   Ran 34 tests … OK
   ```

6. **`✅` count is consistent with the rows changed.** 22 lines → 26: 1 legend + 25 status cells.
   The +4 are D1, D2, D3, D6; C3 and I5 keep their `✅` (edited in place, one cell each) and D4/D5
   are the two 🟡 the "still unrun" list explains. No `⚠️`/🟡 cell was converted to ✅.

7. **The section's tables are well formed** — cell counts per row: C 5, I 4, D 6, S 4.

8. **Historical documents untouched** — `T06-mtp.md` and `2026-10-29-v165-retune.md` are not in
   `git diff --stat 1c5b9670..HEAD`.

9. **Line endings preserved** — README still all-CRLF (147/147), tracker still all-LF (178/178),
   matching their state before the edit; the HEAD blobs are LF in both cases (`core.autocrlf`).

10. **The skill edit re-read, not taken on trust** — `wc -l` 99; section headings are the original
    six plus "The fit comes before the pass" (after the floors) and "The gates, before and after"
    (before "Reading a result"); line 3's `description` matches the text that was already there.

## Files changed

- `wiki/plans/v165-critique-implementation.md` — section C, C3 and I5 status cells (+95 / −2).
- `bench/windows/README.md` — the corrections listed above (+70 / −31).
- `.superpowers/sdd/v165-critique-fixes/task-6-report.md` — this report (untracked, as reports are).

## Self-review

- The brief's five requirements are each met and each is visible in the diff: C with D1–D6 and
  S1–S4; the two false ✅ corrected in place with the commits that make them true; the README
  corrected to the code (no new sections, no invented switches); the two historical documents named
  with the replacement spelling instead of rewritten; the unrun remainder stated.
- Every factual claim in C that this box can check was checked here (commit list, D1 line range,
  D2 log lines 160–163 and the missing `EXL3_MOE_CPU_SWAP`, D6's 49/48/1 shape, `--post` behaviour,
  34 tests, skill line count). D3/D5's line numbers are quoted from the plan's transcription of the
  review and from the task reports, not re-derived — they are pointers to pre-fix code, so they are
  the least load-bearing numbers in the table.
- The README now teaches only behaviour the deployed script exhibits: I ran each example, including
  the two negative cases the new `-Extra` row claims, and the one positive `-Extra` case.
- Deliberately not done: rewriting either historical document; adding a regime table for a second
  regime (there is none); documenting `-Root`/`-State`/`-NoFrozenState`/`-Short`/`-MinFree*` beyond
  where they matter (the file is a contract for the common path, and every one of those is named
  where it is used); committing the untracked plan file `wiki/plans/v165-critique-fixes.md` (not
  mine to commit).

## Concerns

1. **D-numbering is chronological, not the review's order.** The brief fixed the count (six) but not
   the order; I grouped by commit (deploy, run_pass, prompt reader, swap contamination, alloc peak,
   verify gates) because that is what the Commit column can prove. If the parent wants the review's
   own Part-1 order, the rows renumber without changing a fact.
2. **"Names the switch that sets it" needed qualifying.** Draft flags have no delta switch — the
   harness's own remedy is "edit state.json" — so the `-Extra` row says "names the flag with the
   switch or state field that sets it", which is what all three abort messages do.
3. **Two floor statements in `2026-10-29-v165-retune.md` are inconsistent and left alone** (:69 says
   "200 MB free VRAM / 2048 MB free RAM", :156 says 100/1200, matching `state.json`). It is plan
   text, not a command, and the brief's only instruction about that document was the `-Extra`
   spelling — but a reader following :69 will be surprised that a 150 MB-free pass is **not** a
   breach. Same class of drift: `watch.ps1`'s own `-MinFreeVramMB`/`-MinFreeRamMB` defaults are still
   200/2048; every sanctioned caller passes 100/1200 explicitly, so only a hand-run `watch.ps1`
   would use the old floors.
4. **The two amend passes during self-review changed the SHAs** (`0d93d849`/`67011fa6` were
   superseded by `7c8754e2`/`dfa36468`) so that the final pair stays one-file-per-commit. Only my
   own, unpushed commits were rewritten.
5. **D2's ✅ rests on the dry run plus the parser probe**, not on a log line from a real drafting
   pass; the first such pass is the T6 resume, which the "still unrun" list now names explicitly.
   D4 and D5 are 🟡 for the same reason.
