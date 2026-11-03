# Plan — T29: make `feat/adaptive-tier` runnable at HEAD, verify it on the box, and fold the re-measurement into T16

## Context

The branch's four-arm campaign (T16 §7) was run on a file-level install of the branch's patch. An
independent review (T28) found the coverage instrument blind to promotion (fixed on the branch,
commit `c6d034d4`), and a re-measurement (`t29`, one session, four interleaved passes, artefact
`/tmp/opencode/t29-pin-unset.md`) then found that with the pin budget **unset** (driver limit, 30 of
34 chunks) and the fixed instrument, arm C's coverage is **72.3 % against a 31.7 % control** —
+40.6 points at a repeat spread of 0.008, with 8,669 promotions and 42 of 48 layers promotable.

That run also proved **branch HEAD `c6d034d4` cannot run**: two defects the re-measurement hit and
worked around in the installed copy only.

| defect | symptom |
|---|---|
| `torch.cuda.Event(device = d)` in `_TierStats.step` | `TypeError: Event.__new__() got an unexpected keyword argument 'device'` on torch 2.11.0+cu130 — every pass with `EXL3_MOE_TIER_STATS=1` fails after the prefill ladder |
| `pinned_layers()` walks `layer_auxloc` one nesting level short | `TypeError: '<' not supported between instances of 'tuple' and 'int'` in the tier line, i.e. the pin report crashes the moment a pinned arena exists |

## Global constraints

- Do **not** modify `/home/dev/.local/share/opencode/worktree/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf`
  (it owns the deployment), the deployed harness `/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/`,
  or the pristine backup `/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/adaptive-t27/backup/`.
- Never deploy this branch's harness over `_agent/v165`. Arms run from that directory, with env deltas only.
- The box serves live traffic. A GPU task stops the served model to free VRAM and must leave the box
  as it found it: pristine engine files back in site-packages (hashes
  `ec103eb9522d162e86e872de8ea42d289452060f0f73ffd7179c2acdff7bc952` and
  `f7342c9d57f767af68425b60ae98b911528e9e60481de5caeb71793541d31d71`), server restarted with
  `C:\Users\yoho\Downloads\tabbyAPI\start_tuned.ps1`, and `/v1/models` answering 200.
- A pass is judged by its row in `results/passes.csv` and its parsed `PASS/FLOORS/PP/TG_S1` block,
  never by a process exit code (the venv `python.exe` is a uv shim).
- No regime flag in `-Env` (`-mcs`/`-cs`/`-cq`/`-chunk_size`/`-ambs`/draft flags): the harness aborts.
- Commit messages state the measurement or the defect, not the activity. One logical change per commit.
- Evidence rule: every claim in a report cites `file:line`, a log line, or a CSV row that the author read.

## Task 1 — make HEAD runnable: the two runtime defects, with the tests that would have caught them

Files: `exllamav3/modules/block_sparse_mlp_cpu.py`, `exllamav3/model/moe_cpu_host.py`,
`tests/moe_cpu/` (new or existing test file).

1. **`_TierStats.step`'s event.** `torch.cuda.Event()` takes no `device` keyword on the installed
   torch. Construct the event under the device whose stream records it (`with torch.cuda.device(d):`)
   so a multi-GPU split records each event on its own device, and leave a comment saying why the
   kwarg is not used. Check whether `_PromoteState`/`_split_promote_step` has the same latent pattern
   and fix it there only if the fix is the same one-liner; otherwise record it in the report as a
   separate finding, do not refactor.
2. **`MoeCpuHost.pinned_layers()`.** The nesting is wrong: `layer_auxloc[li]` is a per-expert list of
   per-projection locations, so the current `for proj in ...: {loc[0] for loc in proj ...}` iterates
   tuples and compares them against ints. Walk it with the shape the worker actually sends (the
   reviewer's finding names the shape; derive it from `register_layer`/the ack path in
   `moe_cpu_host.py` and from `_split_promote_verify`, which already walks `layer_auxloc`
   correctly — copy that walk). Return the layer indices, ascending.
3. **Tests, `nogpu` where possible.** A test that `pinned_layers()` returns the right layers for a
   two-layer structure with mixed pinned/unpinned chunks (including a `None` aux entry, which the
   walk must tolerate), and a test that the tier line's pin phrase contains the promotable range when
   some layers are stranded. For the event: if a CPU-only test cannot construct the object, say so in
   the report and rely on Task 2's GPU pass; do not add a mock that tests nothing.
4. Run `pytest tests/moe_cpu -q` on the box's venv **against the source tree** (copy the worktree to a
   Windows path first, as the report's §9 describes; `pytest` from a WSL path mixes the import path).
   Everything green except the clusters T16 §9 lists as pre-existing.
5. Commit; report the test command and its summary line.

## Task 2 — prove HEAD runs on the box (GPU, one short pass)

Prerequisite: Task 1 merged into the branch.

1. Stop the served model; confirm ~0 MiB VRAM used.
2. Copy the branch's `block_sparse_mlp_cpu.py` and `moe_cpu_host.py` into the venv's site-packages;
   record `git rev-parse HEAD` and both `sha256sum`s.
3. Prove the patch is live (`_TIER_STATS_EVERY == 128`, `hasattr(_TierStats, '_refresh')`,
   `hasattr(MoeCpuHost, 'pinned_layers')`).
4. Run **one** short pass, `-Runner perf -MaxLength 4096`, with
   `-Env "EXL3_MOE_TIER_STATS=1,EXL3_MOE_PINNED_ARENA=1,EXL3_MOE_CPU_PROMOTE=16"`. Acceptance: the row
   is `status=ok` and `breach=0`, the log contains at least one ` -- expert tier:` line, and that line
   carries the pin phrase with `N of 48 layer(s) promotable (0..K)` — i.e. both defects are gone and
   the fixed instrument reports.
5. Restore: pristine files back (verify both hashes), server restarted, `/v1/models` 200, VRAM recorded.
6. Report: tag, run id, status/breach, the exact tier line, the venv hashes before/after, the HTTP status.

## Task 3 — fold the re-measurement into T16

File: `wiki/plans/v165-reports/T16-adaptive-tier.md` only.

1. Replace §7's verdict with what the two campaigns now support: the gate's coverage half is **met at
   the driver-limit pin** (+40.6 points, 31.7 → 72.3 %, repeat spread 0.008, 8,669 promotions, 42/48
   layers promotable) and **failed at the 8 GiB budget** (which stranded 37 of 48 layers); decode is
   unresolved (C +1.1 % inside a 4.6 % repeat spread); prefill +18 % / +15.7 % in the t29 arms.
2. Add a §12 with the four t29 passes (run ids, pooled and mean coverage, promotable range,
   promotions, hook ms/step, PP/TG) and the provenance: branch HEAD, the two installed-file hashes,
   the harness hash, the session order A1 C1 C2 A2.
3. Correct §10's entry for the event fix and the `pinned_layers` fix: both were found by the
   re-measurement, not by the branch's own tests, and both are now on the branch (Task 1's commit).
4. Update §11's ordered list: drop items the re-measurement settled, keep the decode question (with
   the repeat count it needs), the policy arm, `EXL3_MOE_STREAM_MIN_ROWS`, and the budget move.
5. `bench/verify_task.py --post --report wiki/plans/v165-reports/T16-adaptive-tier.md --cites-failure t27-A-base-20261103-121148`
   must pass from the 165-win-perf worktree; paste the gate output into §8.
