# exllamav3 1.5.1 → 1.5.4 update plan (handoff) — fork rebase + tabbyAPI upgrade

Status quo (verified 2026-10-04): fork base 1.5.1, upstream latest
1.5.4 (3 releases behind). Our branch `wip/kvarn-cache`: 60 files,
+17k/−115 vs `origin/fork-overview`. Known conflict surface from
`git diff v1.5.1..v1.5.4`: `attn.py` (+144), `triton_paged.py`
(+302), `dispatch.py` (+4), new `smem.py`, DSA/MLA rework.
Relevant upstream wins: 1.5.2 transient-VRAM fixes, 1.5.0
nondeterminism removal, 1.5.0/1.5.1 MoE opts, Flash-Next TP.

Two independent tracks (either order, never cross-compare numbers
across versions). Do Track A fully before Track B only because B
is destructive (venv replacement) — not because they depend on
each other.

## Track A — fork rebase (our branch onto v1.5.4)

### A0. Prep (no code, 10 minutes)

1. `git status --short` must show ONLY untracked spikes/bats
   (tracked tree clean). If not: STOP, report.
2. `git push origin wip/kvarn-cache` (ensure remote backup current).
3. Safety copy (never rebase the live branch directly):
   `git checkout -b wip/kvarn-r154 wip/kvarn-cache &&
   git push origin wip/kvarn-r154`.
4. `git fetch upstream tag v1.5.4` (verify: `git rev-parse
   v1.5.4` prints a sha). Do NOT touch `master`,
   `origin/fork-overview`, or any remote except pushing your
   own `wip/kvarn-r154` branch. Moving fork-overview is the
   maintainer's job, not yours.

### A1. Rebase + conflicts

0. **Do NOT run step 1 as originally written.** `git rebase --onto v1.5.4
   origin/fork-overview` is silently destructive: `fork-overview` is
   upstream `6b84a21` + 15 docs commits (`AGENTS.md`, `BRANCHES.md`, the
   README fork header, `doc/local-build.md`), and rebasing onto bare
   `v1.5.4` drops that whole layer **with no conflict** — the diff just
   shows those files "no longer in our diff", which reads like a clean
   win. Build the base first, then rebase onto it:

   ```sh
   git checkout -b wip/fork-docs-154 v1.5.4
   git rebase --onto v1.5.4 6b84a21 origin/fork-overview   # fork docs onto 1.5.4
   git checkout wip/kvarn-r154
   git rebase --onto wip/fork-docs-154 v1.5.4              # our diff onto that
   ```

   Verify: `git diff --stat wip/fork-docs-154 origin/fork-overview -- AGENTS.md
   BRANCHES.md doc/local-build.md` must be empty, and `git merge-base
   --is-ancestor v1.5.4 HEAD` must pass.

   Expect conflicts in: `exllamav3/modules/attn.py`,
   `exllamav3/modules/attention_fn/dispatch.py`, possibly
   `triton_paged.py`, `model/*`, `model_init.py`.
2. New files are ours alone (no conflict possible, keep all):
   `exllamav3/cache/kvarn.py`, `qsa.py`,
   `modules/attention_fn/kvarn_triton.py`, `qsa_indexer.py`,
   `bc_attn.py`, all `tests/test_kvarn_*.py`, all `eval/*kvarn*`.
3. Conflict rule (memorize): KEEP BOTH — upstream's new code +
   our additions. Concretely per file:
   - `dispatch.py`: keep upstream's 4 added lines AND our
     `_try_kvarn_online_decode` call, `_kvarn_past_*` wiring,
     `_ov_dec_n_mirror` write-back clear. If upstream refactored
     `attn_dispatch`'s signature, adapt our call sites to the new
     signature (do not revert their refactor).
   - `attn.py`: keep upstream's changes AND our
     `decode_flash_attn` hooks (`get_for_device` calls,
     `block_table`/`cache_seqlens` handling). If upstream moved
     code you hook into, move the hook with it.
     **Actual outcome (1.5.4):** the only conflict is
     `_autosplit_layer`. Upstream **moved** the
     `isinstance(layer, QSAPlanes)` check OUT of `_autosplit_layer`
     and into its two callers (`autosplit_prepare`,
     `autosplit_extra_measure`). Take upstream's shape and re-insert
     ONLY the kvarn guard — keeping the old guard as well
     double-returns. `decode_flash_attn` did not conflict.
   - `triton_paged.py`, `model/*`, `model_init.py`: keep both
     sides; if a hunk is purely upstream bookkeeping around our
     untouched code, take upstream.
   - `__init__.py`, `cache/__init__.py`: keep both (usually
     disjoint import lines).
4. `git add` resolved files ONLY (explicit paths, never `-A`),
   `git rebase --continue`. If a conflict is incomprehensible:
   `git rebase --abort`, record the file + hunk in the plan
   file §8, STOP, report (do not guess-merge).
5. After rebase completes: `python3 -m py_compile` on every
   touched file. Then `venv/bin/python -m pytest
   tests/test_kvarn_cpu.py tests/test_kvarn_tail_cpu.py
   tests/test_kvarn_widths_cpu.py tests/test_kvarn_m4_cpu.py
   tests/test_kvarn_m5_cpu.py tests/test_kvarn_triton.py -q`
   must be **85 passed / 14 skipped** (CUDA tests skip here). The
   original "12 skipped" in this plan was stale: all 14 skips are
   `needs CUDA + triton` in `tests/test_kvarn_triton.py`, and those
   files are byte-identical before and after the rebase.
6. **Audit for content loss by diffing TREES, not diffs.** Comparing
   `git diff v1.5.1 <old>` against `git diff <new-base> HEAD` reports
   a phantom multi-thousand-line "deficit" that is only reindentation
   and duplicated blank lines. Use `git diff <old-branch> HEAD`
   (old tree vs new tree) and then, per file, ask whether each removed
   line is absent from upstream `v1.5.4` too (upstream removed it) or
   still present there (we dropped it).

### A1.5. Rebuild exllamav3_ext on the box (REQUIRED, was missing)

v1.5.4's `loader/safetensors.py:1020` calls
`ext.stloader_deferred_batch`, which neither the Sep-25 box `.pyd`
nor tabbyAPI's 1.4.9 `.pyd` (verified: zero symbol matches)
export. Without a fresh binary every box run dies at model load.
Read `doc/local-build.md` FIRST (VsDevCmd trap, arch-list syntax,
scratch-copy rule — all binding here).

1. Build on the BOX (Windows + CUDA), never in WSL, never in a
   working checkout: copy the rebased tree (minus `.git`,
   `build/`, `*.pyd/obj`) to a scratch dir and build there.
2. Toolchain: x64 Native Tools prompt (or manual MSVC/SDK PATH
   per local-build.md — `VsDevCmd.bat` does NOT propagate
   reliably). Check at build time: `nvcc --version` (13.x
   present), cmake (VS generator — no ninja on box, budget
   accordingly), `pip show torch` in the tabbyAPI venv (match
   the CUDA major; minor skew tolerated per doc).
3. Target: `exllamav3_ext` only, sm_89 (+PTX), ~190 TUs. Budget
   1–3h wall; run detached with a log file; do not parallelize
   past RAM (44GB free is plenty, keep 2GB system-RAM floor).
4. Install WITHOUT touching the venv: place the fresh `.pyd`
   in a shadow dir (e.g. box-side `ext-154/`) and PREPEND it
   via `PYTHONPATH` in every test bat (Python resolves the
   shadow copy before the venv's 1.4.9 one). NEVER overwrite
   `venv\...\exllamav3_ext.cp312-win_amd64.pyd` — the tabbyAPI
   server depends on it. Same for any Sep-25 box `.pyd`.
   **The shadow dir must NOT be inside the mirror tree root.**
   `sys.path[0]` is the *script's own directory*, so a `.pyd`
   sitting in the tree root silently wins over `PYTHONPATH` — this
   produced a false "the venv is already on 1.5.4" reading that
   looked like a completed upgrade when nothing had been installed.
   Always confirm `e.__file__` from a **neutral cwd** (`C:\`), never
   from the repo dir.
5. Verify before any model load: fresh python,
   `import exllamav3_ext as e;
   assert hasattr(e, 'stloader_deferred_batch')`, plus print
   which file was loaded (`e.__file__` must be the shadow
   copy). Then one tiny load (512 ctx) before the battery.
6. Track B is UNAFFECTED by this section (the pip 1.5.4 wheel
   ships its own binary). A wrong/ABI-mismatched binary fails
   loudly at import — that is the fail-closed behavior; on any
   import error STOP, do not retry with flags.

### A2. Box re-validation (full gate battery — numbers WILL shift)

Upstream changed perf-relevant paths (prefill, MoE, VRAM,
nondeterminism), so old ledger numbers are references, not
targets. Do NOT "fix" code to reproduce old numbers.

1. Sync box mirror (`cp` + `unix2dos`, same as always).
2. KLD parity 8k (`PARITY=1`, expect green; NEW asserts from
   upstream are fine, note them).
3. tg pair 64k graph (protocol v3, hot counts) + KLD + peaks.
4. Compare vs ledger: record new baselines in a dated ledger
   paragraph (old vs new, with the upstream version noted).
   Drift <5%: proceed. Drift ≥5% either direction: investigate
   ONE level (which path moved, Kineto if cheap), then record —
   do not chase it in this task.
5. PARITY 8k green required before push. Push ONLY
   `wip/kvarn-r154` (never force-push `wip/kvarn-cache`;
   the maintainer merges).
6. **Write each gate's log to a per-setting filename**
   (`a2_ntok<N>_parity<P>.log`), never a single `a2.log`. A shared
   name means the last run overwrites the earlier evidence, and the
   overwritten numbers then survive only in the operator's transcript.
   Each log should also carry a provenance header (tree version, which
   `.pyd` loaded, `smem.py`, `CacheLayer_kvarn` importable) so a row
   can never be attributed to the wrong build.
7. **When a number lands inside the noise band, re-run it and report
   the range.** A single sample that sits near the baseline is not
   evidence of "no change" — quoting the pessimistic sample of a noisy
   measurement understates the result. See the ledger's 64k tg row,
   where one pass read 51.4/51.7 and a repeat read 52.2/52.2 against
   a 52.2 baseline.
8. **Validate a cross-setting comparison against a control, not by
   reasoning.** A `PARITY=1` number cannot be compared to a
   `PARITY=0` baseline — but the ledger already contains a population
   of historical `PARITY=1` validation runs, and landing inside it is
   the actual proof.

### A3. Done means (Track A)

`wip/kvarn-r154` on v1.5.4, CPU suite green, box gates green,
ledger re-baselined with dated paragraph, pushed, maintainer
notified for merge. Untracked spikes only in tree.

## Track B — tabbyAPI 1.4.9 → 1.5.4 (independent, destructive)

tabbyAPI runs a PREBUILT wheel in its own venv
(`C:\Users\yoho\Downloads\tabbyAPI\venv`); nothing on our
branch affects it until this track runs. All work on the box,
not WSL, in an x64 Native Tools prompt (see
`tabbyAPI/MEMOPS_FIX_PROMPT.md` §Environment for toolchain).

### B0. Compat check (before touching the venv)

1. Read `tabbyAPI/backends/exllamav3/` for version pins/checks
   (`1.4.9` strings, `__version__` asserts, API imports). List
   every exllamav3 API the backend imports.
2. Diff the imported APIs against 1.5.4 (pip download the 1.5.4
   wheel to a SCRATCH dir, do not install; compare signatures
   of imported functions). Breaking change found: STOP, report
   the exact import + change (do not shim it yourself).
3. Record current serving behavior as the rollback baseline:
   one short generation on the Flash-Next production config
   (log kept), plus `pip freeze | findstr exllamav3` output.

### B1. Upgrade with rollback ready

1. BACKUP FIRST: copy `venv\Lib\site-packages\exllamav3*` to
   `venv\exl3_149_backup\` (precedent exists:
   `exl3_148_backup`). Verify the backup imports (spot-check
   file count matches source). No backup = no upgrade.
2. Install the 1.5.4 wheel (exact release asset `+cuXX.torch2.XX`
   matching the venv's torch/CUDA — check torch first; wrong CUDA
   build = brokenestensibly working install, so match it exactly).
   **The venv has NO pip** (`python -m pip` → "No module named pip",
   `Scripts\pip.exe` absent); `start.bat` bootstraps via `uv`. Use:

   ```bat
   uv pip install --python venv\Scripts\python.exe --no-deps --reinstall <wheel>
   ```

   `--no-deps` is load-bearing: the wheel pins `torch==2.11.0`, and
   letting the resolver run risks upgrading torch under a live server.
   Confirm the interpreter is the venv's, not system `C:\Python312`.
3. Smoke: import exllamav3, print `__version__` (expect 1.5.4),
   load the Flash-Next production config, one short greedy
   generation. Any failure: STOP, do not debug forward.
   Two traps here: `config.yml` pins `host: 0.0.0.0` / `port: 5000`
   and CLI `--port` does **not** override it, so poll 5000; and auth
   needs `Authorization: Bearer <api_key>` from `api_tokens.yml`
   (regenerated on each boot). Flash-Next is a **reasoning** model, so
   budget `max_tokens` ≥ 512 or the whole budget goes to
   `reasoning_content`, `content` comes back `null`, and
   `finish_reason: length` — that is not a failure.

### B2. Re-validate serving (their metrics, their harness)

1. Re-run their perf battery (`logs/perf` + `logs/mtp` pattern
   from `tabbyAPI/PERF_FINDINGS.md`, same workload): expect
   shifts (1.5.0/1.5.1 MoE opts + 1.5.2 VRAM should HELP, but
   record, don't assume). Compare vs B0 baseline.
   `bench.ps1` / `mtp_sweep.ps1` point at a **separate upstream
   checkout** (`Downloads\exllamav3`) that does not exist on this box.
   Point `$PERF`/`$SD` at our mirror instead and change nothing else —
   `perf.py` and `spec_decode.py` there are byte-identical to v1.5.4
   upstream, so the comparison stays fair. **Keep the harness in
   PowerShell**: `spec_decode.py`'s `-single "Agentic, code"` is one
   argv token containing a space, and batch re-splits it under every
   form (quoted, caret, `=`-joined, `%~2`). The failure is silent —
   argparse drops the workload and every arm exits 0 with an empty
   result table. Zero exit + no rows is a FAILED run.
   Copy `bench.ps1`'s per-arm env discipline exactly: clear
   `EXL3_MOE_MEMOPS`/`_STREAM_T`/`_STREAM_BATCH_EXPERTS`, then set only
   that arm's keys, or arms inherit each other's env and every
   comparison is void.
2. Confirm: tuned env still respected (`start_tuned.ps1` vars
   take effect — spot-check one, e.g. offload behavior),
   `cpu_moe_offload_layers: 38` still valid (VRAM re-measure!),
   131k cache still fits, MTP draft still engages.
3. MEMOPS note: 1.5.x may have changed memop behavior — test
   `MEMOPS=1` vs `0` once (10% gap on 1.4.9); if 1 closed it,
   say so loudly (kills a workaround + maybe the upstream issue
   draft in `memops_win_issue.md`).
   Verify the unset case really is `MEMOPS=1` before labelling it:
   `moe_cpu_host.py` reads `os.environ.get("EXL3_MOE_MEMOPS", "1") != "0"`.
   Run the A/B **order-controlled** (both arm orders) — a single pass
   ordering can manufacture or hide a gap this size.
4. Rollback (if anything is red and not trivially explained):
   copy `exl3_149_backup\*` back over `site-packages`, re-run
   B0 smoke, confirm `__version__` reads 1.4.9 again. Report.

### B3. Done means (Track B)

1.5.4 serving with B0-comparable-or-better perf battery,
VRAM re-measured, MEMOPS verdict recorded, backup retained
(do NOT delete `exl3_149_backup` for a month). Report to
maintainer; `PERF_FINDINGS.md` updated ONLY if they own it
(ask before editing their docs).

## Outcome (2026-10-05) — what actually happened

Both tracks ran to completion. Full numbers and evidence live in
`doc/kvarn-4090.md` (§ "v1.5.4 re-baseline" and § "tabbyAPI serving on
exllamav3 1.5.4"). Headlines, including the ones that contradict the
plan's expectations:

**Track A** — rebased 320 commits onto v1.5.4 (branch `wip/kvarn-r154`,
3 conflicts, keep-both). CPU suite 85 passed / 14 skipped. Box gates
green. KLD **unchanged** (median 1e-6, same-top 100.00% at 8k and 64k,
digit-identical across all 4 runs); prefill +36% @8k / +17% @64k from
upstream work; decode **flat at matched parity** (64k: 52.2/52.2 against a
52.2 baseline). Pushed.

**Track B** — 26/26 backend symbols resolve at 1.5.4 with 1.4.9-identical
signatures, 0 call-site breaks. Installed via `uv --no-deps`. Serving
green: mcl 38 honored, MTP drafting engaged, a 151k-token prompt completes
past the 131k cache, VRAM 20546 MiB used / 3593 free idle.

Two findings that invert the plan's expectations:

1. **The MEMOPS workaround is still mandatory — the plan's hope that 1.5.x
   closed it is not supported.** Within 1.5.4, `MEMOPS=0` is **29%** faster
   than default (four runs, both arm orders, disjoint ranges). The
   "widened from 11.5%" framing rests on tabbyAPI's archived September logs
   for the 1.4.9 side, not on a run of ours — see the ledger's
   "MEMOPS: what the flag actually switches" section for the mechanism, the
   per-claim strength split, and what may not be quoted. The session-solid
   fact to pass on: **on 1.5.4, default settings decode at ~22 it/s and
   `MEMOPS=0` at ~29 it/s on this workload.** `memops_win_issue.md` and
   `MEMOPS_FIX_PROMPT.md` stay **open**; production is unaffected only
   because it already sets `MEMOPS=0`.
2. **Drafting got materially better**: r01's MTP-vs-baseline speedup went
   **1.07x → 1.27–1.31x**, with acceptance rates unchanged to ±0.06
   tokens/draft.

Plan defects found and corrected above (A1 step 1's destructive rebase,
the stale "12 skipped", A1.5's shadow-dir/`sys.path[0]` trap, B1's
pip→uv, B2's harness-portability and order-control gaps) — each is now
written back here so the next agent does not rediscover it.

**Still open, needs a human:** whether tabbyAPI's owner publishes the
MEMOPS finding in `PERF_FINDINGS.md` (their doc), the maintainer merge of
`wip/kvarn-r154`, and pruning `exl3_149_backup` after a month.

## Global non-goals + guards (binding)

No kernel work, no perf optimization, no sampler work, no
changing what the numbers mean. Hands-off everything outside
your track; `git status` + `git diff --stat` before every
commit (Track A), explicit per-file `git add`. No force-push
except your own `wip/kvarn-r154` after a rebase (and only
that branch, only when the rebase restarts). Never touch
`master`, `fork-overview`, another process's files, or box
RAM/VRAM guard rails (200MB VRAM kill + 2GB RAM floor stand
for every box run in both tracks).
