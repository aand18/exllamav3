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

### §4.3 re-test 2026-10-05 (maintainer run, real push, not dry-run)

`git reset --hard wip/kvarn-r154` + real
`git push --force-with-lease origin wip/kvarn-cache` → **rejected**:
`push declined due to repository rule violations`. The block is real
(ruleset, not lease, not auth). Corollaries now proven: (1) `push
--dry-run` does NOT evaluate rulesets — a dry-run success earlier the
same day meant nothing; never cite dry-runs as clearance. (2) Local
was reset back to `584ec09` immediately, refs coherent. The three
options in the outcome log stand; the pointer is unmoved pending one
of them.

## 7. Outcome II — move executed 2026-10-05 (maintainer relaxed rule)

Ruleset 24036645 (`deletion`, `non_fast_forward`) disabled
2026-10-05 (full JSON backed up first), pointer moved
(`584ec09..3c5e84f`, force-with-lease accepted), ruleset
re-enabled and re-verified (`active`, both rules present via
fresh GET). CPU suite on moved tip: 85 passed / 14 skipped.
Refs: `wip/kvarn-cache` == `origin/wip/kvarn-cache` ==
`3c5e84f`; `wip/kvarn-cache-pre154` == `584ec09` (backup);
`wip/kvarn-r154` + backup retained.

## 6. Done means + outcome log

`wip/kvarn-cache` == old r154 tip (verify: `git rev-parse`
both, equal; `git log --oneline -2` shows the §1–§3 commits),
remote matches (`force-with-lease` accepted, no rejection),
CPU suite re-run green AFTER the move (cheap, proves the ref
points at a working tree), outcome appended here (§7) with the
§0 shas + new tip. Then report; the maintainer owns whatever
runs next on the moved branch.

## 7. Outcome (2026-10-05)

**§0 shas (recorded):** `wip/kvarn-cache` = `584ec09`, `wip/kvarn-r154`
= `08bb2e9`. `origin/wip/kvarn-r154` matched local, so no push was needed
to satisfy §0.3.

**New tip:** `wip/kvarn-r154` = `5892f6c`.

| step | result |
|---|---|
| §0 preconditions | all 3 pass |
| §1 cherry-pick `45e60f9` | **skipped, correctly** — already present |
| §2 window_right/sink_key0 gate | done, `10efc54` |
| §3 eval manifest | completed, `5892f6c` |
| §4.1 push r154 | done |
| §4.2 `wip/kvarn-cache-pre154` | done, pushed, verified |
| §4.3 move the pointer | **BLOCKED — see below** |
| §6 CPU suite after the move | 85 passed / 14 skipped on both refs |

### §1 was already satisfied, and the plan's own rule covers it

`eval/_spike23_q5.py` was already on r154, added by `9785256` — the rebase
had replayed `45e60f9` under a new sha. Verified **byte-identical**
(`md5 5031134b45eb3b6b96751a9c9cc0760e` both sides) rather than assuming,
then `cherry-pick --skip` per the plan's instruction, with AST parse and the
85/14 suite green.

### §4.3 is structurally impossible: a ruleset forbids the force-push

The push was **rejected**, and per §4.3 the instruction on rejection is
STOP — do not `--force`. Cause, confirmed read-only via the GitHub API:

```
ruleset 24036645 "protect wip/kvarn-cache"
  enforcement: active   target: branch
  rules: deletion, non_fast_forward
  bypass_actors: []            <-- nobody, admin included
```

```
remote: - Cannot force-push to this branch
 ! [remote rejected] wip/kvarn-cache -> wip/kvarn-cache
        (push declined due to repository rule violations)
```

Note this is **not** a lease failure. A stale lease reports
`(stale info)`; this is a rules rejection, and it would reject a plain
`--force` identically. `gh api .../branches/wip%2Fkvarn-cache/protection`
returns 404 (not classic branch protection — it is a ruleset), so the
protection is invisible to the older API and easy to misdiagnose as
"nothing is protected, retry harder".

The plan's §5 carve-out ("Force-push anywhere except §4.3's single leased
push") and its whole §4.3 design assume this push is available. It is not:
`wip/kvarn-cache` carries a deliberate `non_fast_forward` rule, so the
prescribed mechanism is forbidden by a control the owner put there on
purpose. Circumventing it — `--force`, dropping the ruleset, or swapping in a
merge, which §0 explicitly forbids — is not mine to do.

**Local state left coherent, not diverged.** `reset --hard` had already run
before the rejection, leaving local `wip/kvarn-cache` at `5892f6c` while the
remote sat at `584ec09`. A local ref that disagrees with its remote is a
hidden trap (the next plain push fails as non-fast-forward with no obvious
cause), so local was reset back to `584ec09`. All three refs now agree with
their remotes:

| ref | sha | role |
|---|---|---|
| `wip/kvarn-cache` | `584ec09` | unchanged; the pointer move is pending |
| `wip/kvarn-cache-pre154` | `584ec09` | backup of the pre-move line (also the §0 sha) |
| `wip/kvarn-r154` | `5892f6c` | the completed 1.5.4 tree + §1–§3 |

Nothing was lost: the old line is on `wip/kvarn-cache-pre154` (pushed) and
the new one on `wip/kvarn-r154` (pushed).

### To finish §4.3, a maintainer picks one

1. **Relax the ruleset, then push** — edit ruleset `24036645` to drop
   `non_fast_forward` (or add a bypass actor), then:
   `git push --force-with-lease origin wip/kvarn-r154:wip/kvarn-cache`
2. **Push it themselves** — anyone who can edit the ruleset can run the
   force-push directly; the local ref is already correct and green.
3. **Change the plan** — if non-fast-forward on `wip/kvarn-cache` is
   intended to be permanent, then "single line of truth" needs a different
   mechanism (the owner merges, or `wip/kvarn-cache` is renamed and replaced
   by a fresh ref). That is a design call, not a mechanical one.

`wip/kvarn-r154`, `wip/kvarn-r154-backup` and `wip/kvarn-cache-pre154` are
all left in place per §4.4. Nothing was deleted. Box, venv and tabbyAPI
untouched (§5).
