# Plan: adopt r154 as wip/kvarn-cache + review fixes (handoff)

Goal: single line of truth — `wip/kvarn-cache` becomes the 1.5.4-based
tree (currently `wip/kvarn-r154`), plus the two review fixes. No
merge (histories rewritten — merging fights every line twice).

## 0. Preconditions (STOP if any fails)

1. `git status --short` shows ONLY `??` lines (tracked tree clean).
2. Record tips: `git rev-parse wip/kvarn-cache wip/kvarn-r154`
   (write both shas into §6 outcome log at the end).
3. Confirm `origin/wip/kvarn-r154` equals local r154 tip
   (`git rev-parse origin/wip/kvarn-r154`); if behind, `git push
   origin wip/kvarn-r154` first. You must never move a ref
   someone else advanced under you.

## 1. Cherry-pick the spike harness (only port)

`git checkout wip/kvarn-r154` (stay here for everything below).
`git cherry-pick 45e60f9` — pure file add
(`eval/_spike23_q5.py`); expect zero conflicts. If the file
already exists on r154: `git cherry-pick --skip` + note it,
do not duplicate. Verify: `python3 -m py_compile
eval/_spike23_q5.py`, then full CPU suite
(`venv/bin/python -m pytest tests/test_kvarn_cpu.py
tests/test_kvarn_tail_cpu.py tests/test_kvarn_widths_cpu.py
tests/test_kvarn_m4_cpu.py tests/test_kvarn_m5_cpu.py
tests/test_kvarn_triton.py -q`): bar is 85 passed / 14 skipped,
exact. Anything else: STOP, report (do not "fix" tests).

## 2. Review fix A — window_right/sink_key0 decline gate

File: `exllamav3/modules/attention_fn/dispatch.py`, function
`_try_kvarn_online_decode`, next to the existing
`if window_size not in (None, -1): return None` gate. Add:

```python
    if window_right != 0 or sink_key0:
        return None
```

But FIRST read the function signature: if it does not receive
`window_right`/`sink_key0` (expected — the 1.5.4 params were
added to `attn_dispatch`, not to this helper), thread them
through from the `attn_dispatch` call site (which HAS them as
locals) as new trailing args. Do NOT reorder existing args.
Fail-closed direction only: decline, never attempt to honor
them (the kernels have no windowed path).

Verification proportionate to 2 lines: `py_compile` + full CPU
suite green (import check) + re-read the gate in place. Twin:
N/A by design — no CUDA here and the gate is a decline (fault
needle: a test would need the full dispatch import chain;
declines are verified by code read + the PARITY suite on box,
which exercises every gate each run). Say so in the commit
message, do not fake a test.

Commit (atomic): `Perf: decline kvarn online arm on
window_right/sink_key0 (merge gap from 1.5.4)`.

## 3. Review fix B — eval/ helper manifest (no deletions)

New file `eval/README.md` (nothing else touched): for every
`_spike*`, `_probe*`, `run_*.bat`-adjacent helper in `eval/`,
one line: name → what backs which ledger number (or `DEAD:`
+ why, e.g. superseded by landed work). Determine DEAD only
from ledger/plan references (cites the commit that superseded
it); when in doubt mark LIVE. Delete nothing, move nothing —
this task only maps. Commit (atomic):
`Docs: eval helper manifest (live vs dead spikes)`.

## 4. Move the pointer (only after §1–§3 green + pushed)

1. `git push origin wip/kvarn-r154` (r154 tip incl. §1–§3).
2. Fresh backup of the old line:
   `git branch wip/kvarn-cache-pre154 <recorded-sha-from-§0>`
   + `git push origin wip/kvarn-cache-pre154`. Verify both
   pushes (`git rev-parse origin/...` equals local).
3. `git checkout wip/kvarn-cache && git reset --hard
   wip/kvarn-r154 && git push --force-with-lease
   origin wip/kvarn-cache`. The lease is load-bearing: if the
   push is rejected, someone advanced the ref — STOP, report,
   do not `--force`.
4. Leave `wip/kvarn-r154`, `wip/kvarn-r154-backup`, and
   `wip/kvarn-cache-pre154` in place for a month. Delete
   nothing in this task.

## 5. Explicitly NOT in this task

Porting the other 9 kvarn-cache-only commits (7 superseded/
moot, 2 already covered better on r154 — see review
2026-10-05; porting stale 1.5.1-allocator numbers into a
1.5.4 ledger would corrupt it). Re-measuring anything on box.
Merging (forbidden here). Touching `master`,
`fork-overview`, `wip/fork-docs-154`, the venv, or tabbyAPI.
Force-push anywhere except §4.3's single leased push.

## 6. Done means + outcome log

`wip/kvarn-cache` == old r154 tip (verify: `git rev-parse`
both, equal; `git log --oneline -2` shows the §1–§3 commits),
remote matches (`force-with-lease` accepted, no rejection),
CPU suite re-run green AFTER the move (cheap, proves the ref
points at a working tree), outcome appended here (§7) with the
§0 shas + new tip. Then report; the maintainer owns whatever
runs next on the moved branch.
