# Task 5 report — `measurement-protocol`: fit before measure, the two gates, stateful caches

Status: **DONE_WITH_CONCERNS** (one concern, at the end: the brief file did not exist; the dispatch
message carried its full content and it matches the plan and ledger, so the work proceeded).

File edited: `/home/dev/.config/opencode/skills/measurement-protocol/SKILL.md`
No new files, no new directories, no other skill touched. Not in git — no commit for this task.

## Line count

71 → **95** (`wc -l`), i.e. exactly at the ≤95 cap with no slack. Frontmatter `description` line is
byte-identical (verified with `diff` on line 3 against the saved original).

## What was added, and what each line is for

### 1. `## The fit comes before the pass` — 9 lines, placed after the floors and before "The run names what it measured"

Targets **F1** (0/4 computed a fit or reached for `bench/size_check.py` before launching). This is the
only section in the file that states a precondition on *launching* rather than on interpreting a
number, and its position is deliberate: the floors say what a run must hold, the fit says whether the
configuration can run at all.

- **"No GPU pass launches until `bench/size_check.py` agrees on the configuration"** — the whole of
  F1 in one clause: condition (no pass), action (run the tool), tool named by path. A fresh agent
  cannot read this and launch first.
- **the list after the colon** — `-mcs` trunk VRAM, pinned host arena, `-cs` cache, minimum `-cs`,
  `--draft mtp`. These are the four answers the tool gives (confirmed from its `--help` and its
  actual output), so the reader knows the tool answers *this* question rather than a tok/s question.
  `--draft mtp` is named because the regime under test drafts through the MTP head: without it the
  tool understates VRAM by the head's footprint (337 MiB trunk + 303 MiB cache on the current state),
  which is large enough to flip a verdict held at 1024 MiB of headroom.
- **"Plot both curves … before naming an operating point"** — the plan's item 2, and the reason the
  fit is not a single-point check. The second half states the *mechanism* (VRAM wants a higher `-mcs`,
  host RAM a lower one) because that is what makes one point insufficient; the repo records the
  numbers (`wiki/plans/2026-10-29-v165-retune.md`: VRAM needs `-mcs` ≥ ~404, RAM ≤ ~410), the skill
  gives the shape.
- **"The autosplit's ~2 GiB reserve is already in the verdict; charging it again rejects a fit"** —
  the plan's "two guards must not double-charge the same GB", compressed to the failure it prevents.
  `size_check.py`'s `fits` already subtracts `--reserve-mb` (2048) and requires `--headroom-mb`
  (1024) free; subtracting the 200 MB watchdog floor on top of that is how a valid configuration gets
  rejected before it is tried. Verified in the tool's source (`size_check.py:120-123,197-199`).

### 2. The stateful-cache table in `## A recorded number is not a control` — 7 lines net

Targets **F3**, as *reference* only, per the RED result (4/4 already freeze the profile, discard the
cold run, and notice the sweeps). Four rows, and each cell is a fact the repo does not show:

- **`expert_stats`** — a run inherits the previous run's ranking and rewrites it; saved every 4096
  decode steps, only by a session past 64 steps (`block_sparse_mlp_cpu.py:78,153,154`). The existing
  measured figure (41.5 and 48.7 tok/s against 96-120 warm) moves into this row rather than being
  dropped.
- **`token_stats`** — keyed on the sha1 of `tokenizer.json` **and** the vocab size, so every quant of
  a model and every model sharing that tokenizer reads and extends one file; written only while a
  generator with a sparse draft head runs (`sparse_head.py:46-60`). This is the one cache whose scope
  an agent cannot guess from the knob names, and it is misread as per-checkpoint by default.
- **PLE n-gram table** — 51.8 GB (verified on this box) streamed from disk, so the first arm in a
  process pays cold reads for exactly the prompts it touches. This is why "the first arm is slower"
  is not necessarily a cold-GPU effect.
- **dynamic placement** — `EXL3_MOE_CPU_SWAP` (default 1) decides at load; with it on, placement
  moves while decoding (a sweep every `EXL3_MOE_CPU_SWAP_INTERVAL`, default 128, or a promote per
  step where spares exist) and each move is a numeric step for that expert, so arm 2 starts somewhere
  else than arm 1. Verified in ``_split_swap_tick`` and its own comment.

The lead-in keeps the original instruction — "the first run after a restart is discarded" — as a
one-line statement, because the table explains *why* without saying *do it*; the instruction is the
part that carries behaviour.

### 3. `## The gates, before and after` — 8 lines, placed before "Reading a result"

Targets **F2** (0/4 mentioned `bench/verify_task.py --pre` / `--post`).

- **"`bench/verify_task.py --pre` runs before the first pass of a task"** — the timing, which is the
  failure: four of four started the A/B without it. The list after the colon (model, deployed
  `state.json`, installed engine, frozen profile, the fit) says what it *proves*, so the gate reads
  as a precondition rather than a formality, and `the fit` ties it back to section 1 — the fit is not
  a separate chore, `--pre` runs it.
- **"`bench/verify_task.py --post --report <file>` runs before any number is quoted"** — the second
  timing, at the point where the RED samples would have gone straight to a verdict.
- **the checks that follow** (cited runs `status=ok` with `breach=0`, every `CPU split experts` line
  warm, quoted bands equal to the report's table, arm rows for a cited spec_decode run) — the four
  things the gate actually rejects, in one line each compressed into a clause. This is the part that
  makes the gate worth running: a quoted ratio band is the artifact most likely to be wrong.
- **"Exit 1 means the report contradicts its logs: replace the claim, not the gate"** — the verdict
  semantics the brief asked for, plus the one wrong reaction to a FAIL (weakening the gate).

## Deliberately not written

- **Freeze/restore per arm, dry-run diff, interleaving the arms, mechanism counters before tok/s,
  refusing last night's rows** — F3 and F4 were 4/4 in the RED run and the file already carries the
  regime sentence and the recorded-number section. Repeating them would grow the file without moving
  behaviour, and would read as lecturing.
- **`XDG_CACHE_HOME` pins both profiles without an engine patch** (plan item 1) — it is not in the
  verified-facts list handed down with this task, so it was not written. If it is wanted, one clause
  fits in the `token_stats` row.
- **"A does-not-fit verdict names the phase"** (plan item 2) — also not in the verified-facts list,
  and it is the first thing to go when the line budget bound: F1 is about *reaching* the tool, not
  about how a refusal is reported.
- **The tool's own rationale and flag table** (`~1 s instead of a 60 GB load`, `--json`,
  `--headroom-mb`, `--reserve-mb` defaults) — already in `size_check.py --help`; the skill only
  needed to put the tool in the sequence.
- **Anything from `bench/windows/README.md`** — checked line by line: the README covers the floors,
  `run_pass.ps1` switches, the output CSVs, and four traps (uv shim, WSL `sys.path`, rich output,
  decimal commas). None of the new text restates any of it; `size_check.py` and `verify_task.py` are
  not even in the README's inventory.

## Correction made during the edit

The plan's Task 5 spec says "on this box every split failed one side and the window was empty". The
repo's own record says something narrower: at 262144 no split fixes both sides, the window is
`-mcs` ~404-410 wide with only a few hundred MB of worst-case margin, and run-to-run variance is
~±450 MB VRAM (`wiki/plans/2026-10-29-v165-retune.md:108-113`). The skill states the sliver and the
opposite-direction mechanism rather than "empty", because "empty" is not what the evidence says.

## Diff against the original

Original saved to `/tmp/opencode/measurement-protocol.orig.md` before editing.

```diff
--- /tmp/opencode/measurement-protocol.orig.md
+++ /home/dev/.config/opencode/skills/measurement-protocol/SKILL.md
@@ -19,6 +19,15 @@
 A run that reaches a floor is stopped by the watchdog and reported as a result. The floors are
 constants, not a per-run choice, and they are the reason the harness exists.
 
+## The fit comes before the pass
+
+No GPU pass launches until `bench/size_check.py` agrees on the configuration - trunk VRAM at the
+`-mcs`, the pinned host arena, the cache at the `-cs`, the minimum `-cs`, `--draft mtp` when MTP drafts.
+
+- Plot both curves - VRAM and host arena against the same knob - before naming an operating point:
+  VRAM wants a higher `-mcs`, host RAM a lower one, and the window between them is a sliver.
+- The autosplit's ~2 GiB reserve is already in the verdict; charging it again rejects a fit.
+
 ## The run names what it measured
 
 A number is evidence only when the artifact says which code and which process produced it. Before the
@@ -60,8 +69,23 @@
 same session, back to back, with the same harness.
 
-Discard the first run after a restart: the expert cache is cold and the number is ~2.5x off (41.5 and
-48.7 tok/s for run 1, against 96-120 warm).
+State outlives the process, so the first run after a restart is discarded:
+
+| state | what it does across runs |
+| --- | --- |
+| routed-expert profile (`expert_stats`) | ... |
+| sparse-head token profile (`token_stats`) | ... |
+| PLE n-gram table | ... |
+| dynamic placement | ... |
+
+## The gates, before and after
+
+`bench/verify_task.py --pre` runs before the first pass of a task and proves the run matches the
+brief: model, deployed `state.json`, installed engine, frozen profile, the fit.
+`bench/verify_task.py --post --report <file>` runs before any number is quoted and proves the report
+against its own evidence: cited runs `status=ok` with `breach=0`, every `CPU split experts` line
+warm, quoted bands equal to the report's table, arm rows for a cited spec_decode run. Exit 1 means
+the report contradicts its logs: replace the claim, not the gate.
 
 ## Reading a result
```

Table row bodies elided here; the authoritative diff is
`git diff --no-index /tmp/opencode/measurement-protocol.orig.md <skill>`.

## Verification performed

- `wc -l` → 95 (cap 95).
- `diff` of the frontmatter `description` line against the original → identical.
- Markdown shape of the new table: header + separator + 4 rows, 2 columns per row.
- No new non-table prose line exceeds the file's existing maximum width (104).
- The two facts quoted as numbers were re-checked on this box: `ngram_embedding.safetensors` is
  51,840,275,600 bytes; `verify_task.py --pre` on the current state prints the fit (19812 MiB of
  24564 used → FITS) and `verdict: all gates passed`, which is what the gates section claims it does.
- `git status` in the worktree: unchanged except the pre-existing untracked plan file.
- GREEN samples not run, per instruction.

## Concern

`.superpowers/sdd/v165-critique-fixes/task-5-brief.md` does not exist. The dispatch message carried
the brief's full content (RED results, verified facts, constraints, report format), and it agrees
with `progress.md` lines 57-63 and with the plan's Task 5 section, so the edit was made from that. If
a separate brief file was intended with content beyond the dispatch message, it was never read.

## Key Learnings:

1. `token_stats` is keyed on the tokenizer sha1 plus vocab size, so its profile is shared across every
   quant of a model — a per-checkpoint freeze is not enough to isolate an arm.
2. The autosplit's 2 GiB reserve and the 1 GiB headroom are already inside `size_check.py`'s verdict;
   re-charging the watchdog floor on top turns a valid fit into a rejection.
3. The v165 record shows the VRAM/RAM window as a 404-410 sliver, not an empty window — worth
   correcting wherever that claim is repeated.

---

# Addendum — review-mandated rules added (95 → 99 lines, cap raised to 100)

Two rules the review required were missing because the earlier 95-line cap forced them out. The cap
is now 100; both were added inside "The fit comes before the pass", nothing else changed.

## Line count

95 → **99** (`wc -l`), one line under the new cap. Frontmatter `description` re-checked byte-identical.

## 1. A "does not fit" verdict names its phase (2 lines)

```
- A "does not fit" verdict names the phase: load, warmup, a prefill band, decode. A split that loads
  and then dies in the 131072 band did not fail to load; the claim has to name the band.
```

Phrased as a condition on the verdict, not as a story: the phase list is the enumeration the claim
must draw from, and the second sentence gives the failure the rule prevents — a split that loads and
then dies in a band is not a load failure. This is the wrong claim the project had to withdraw
(`wiki/plans/2026-10-29-v165-retune.md:691`: "368 loads but dies in long prefill"), stated as a rule.

## 2. Every margin is charged once (3 lines, absorbing the previous autosplit-reserve bullet)

```
- The watchdog floors, the engine's host guard (`EXL3_HOST_MEM_RESERVE_MB`, default 2048) and the
  autosplit's ~2 GiB reserve each enforce a margin. Name the one that is authoritative for each
  resource and charge it once: two guards holding the same GB abort a load that would have survived.
```

The old one-line bullet ("the autosplit's ~2 GiB reserve is already in the verdict; charging it again
rejects a fit") named one guard and one consequence; it is now the third item in a list of three and
the point is the *choice* between them. Three facts were verified before writing: `EXL3_HOST_MEM_RESERVE_MB`
defaults to 2048 and is the host-RAM guard for large CPU allocations (`exllamav3/util/memory.py:594,624`,
`doc/env_vars.md:688`); the autosplit's 2 GiB reserve is charged against the card, not host RAM
(`size_check.py:120-123,197-199`) — hence "per resource" rather than "host RAM" in the wording; and
the production regime resolves the choice by setting the guard to 0 because the watchdog holds the
floor (`wiki/plans/v165-STATE.md:137`).

## Diff

Only the fit section grew; the cache table, the gates section and every original sentence are
untouched. Full diff against `/tmp/opencode/measurement-protocol.orig.md`:

```diff
@@ -19,6 +19,19 @@
 A run that reaches a floor is stopped by the watchdog and reported as a result. The floors are
 constants, not a per-run choice, and they are the reason the harness exists.
 
+## The fit comes before the pass
+
+No GPU pass launches until `bench/size_check.py` agrees on the configuration - trunk VRAM at the
+`-mcs`, the pinned host arena, the cache at the `-cs`, the minimum `-cs`, `--draft mtp` when MTP drafts.
+
+- Plot both curves - VRAM and host arena against the same knob - before naming an operating point:
+  VRAM wants a higher `-mcs`, host RAM a lower one, and the window between them is a sliver.
+- The watchdog floors, the engine's host guard (`EXL3_HOST_MEM_RESERVE_MB`, default 2048) and the
+  autosplit's ~2 GiB reserve each enforce a margin. Name the one that is authoritative for each
+  resource and charge it once: two guards holding the same GB abort a load that would have survived.
+- A "does not fit" verdict names the phase: load, warmup, a prefill band, decode. A split that loads
+  and then dies in the 131072 band did not fail to load; the claim has to name the band.
+
 ## The run names what it measured
```

(The cache-table and gates hunks are unchanged from the report above.)

## Checks run after the edit

- `wc -l` → 99 (cap 100).
- `diff` of lines 1-4 against the original → frontmatter identical.
- Non-table prose widths in the new block: 96, 104, 98, 92, 97, 97, 100, 100, 88 — the 104 matches
  the widest pre-existing prose line in the file (line 60 of the original), so the wrap convention holds.
- Full-file read for stale sentences: "the fit" in the gates section still resolves to this section;
  the floors section still says the floors are constants on every run, so "name the authoritative
  guard" cannot be read as "drop the floors"; nothing in the table or gates sections contradicts the
  new phase rule.

## Deliberately not added

The regime's own resolution (guard `0`, watchdog authoritative) is left out of the skill: it is in
`state.json` and the run header, and the rule — choose and charge once — is what generalises.
