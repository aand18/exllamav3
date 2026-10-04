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

1. On `wip/kvarn-r154`: `git rebase --onto v1.5.4
   origin/fork-overview` (replays our diff onto 1.5.4).
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
   must be 85 passed / 12 skipped (CUDA tests skip here).

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
2. `pip install --upgrade` the 1.5.4 wheel (exact release asset
   `+cuXX.torch2.XX` matching the venv's torch/CUDA — check
   `pip show torch` first; wrong CUDA build = brokenestensibly
   working install, so match it exactly).
3. Smoke: import exllamav3, print `__version__` (expect 1.5.4),
   load the Flash-Next production config, one short greedy
   generation. Any failure: STOP, do not debug forward.

### B2. Re-validate serving (their metrics, their harness)

1. Re-run their perf battery (`logs/perf` + `logs/mtp` pattern
   from `tabbyAPI/PERF_FINDINGS.md`, same workload): expect
   shifts (1.5.0/1.5.1 MoE opts + 1.5.2 VRAM should HELP, but
   record, don't assume). Compare vs B0 baseline.
2. Confirm: tuned env still respected (`start_tuned.ps1` vars
   take effect — spot-check one, e.g. offload behavior),
   `cpu_moe_offload_layers: 38` still valid (VRAM re-measure!),
   131k cache still fits, MTP draft still engages.
3. MEMOPS note: 1.5.x may have changed memop behavior — test
   `MEMOPS=1` vs `0` once (10% gap on 1.4.9); if 1 closed it,
   say so loudly (kills a workaround + maybe the upstream issue
   draft in `memops_win_issue.md`).
4. Rollback (if anything is red and not trivially explained):
   copy `exl3_149_backup\*` back over `site-packages`, re-run
   B0 smoke, confirm `__version__` reads 1.4.9 again. Report.

### B3. Done means (Track B)

1.5.4 serving with B0-comparable-or-better perf battery,
VRAM re-measured, MEMOPS verdict recorded, backup retained
(do NOT delete `exl3_149_backup` for a month). Report to
maintainer; `PERF_FINDINGS.md` updated ONLY if they own it
(ask before editing their docs).

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
