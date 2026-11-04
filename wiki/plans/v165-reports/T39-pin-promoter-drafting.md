# T39 — the pin and the promoter on the drafting path: withdraw

Campaign of 2026-11-04, one session, server stopped, scratch harness root
(`C:\Users\yoho\Downloads\_t4check`) running the repo's `eval/spec_decode.py` — the one that resets
expert placement between arms (T31/Task 4). Engine: the branch's Python files, file-level in the venv
(`2d65860…`, `a8a9090…`, `ee16ca7…`, `7b9b705a…`).

## Question

T32/T33 measured a large decode gain in a **no-drafter, dynamic-placement** regime and t34 measured
none in the **drafter, static-placement** regime — but there the promotion path was inert
(`SWAP=0` ⇒ spares 0). This campaign measures the combination production could actually adopt:
DFlash2 drafting **with** dynamic placement, the pinned arena, and the fixed promoter, against
today's static path.

## Arms

| arm | configuration |
|---|---|
| **A** | `EXL3_MOE_CPU_SWAP=0` — today's production path |
| **B** | `SWAP=1` + `EXL3_MOE_PINNED_ARENA=1` + `EXL3_MOE_PINNED_ARENA_GIB=24` + `-swap_ok` — dynamic placement, pinned arena, fixed promoter (4 spare slots/layer) |

Order **A B B A**, all `status=ok`:

| run | run id | elapsed | corpus draft tok/s | acc |
|---|---|---|---|---|
| A1 | `t39-A1-static-20261104-192312` | 1 047 s | 55.02 | 2.04 / 2.84 |
| B1 | `t39-B1-dynamic-pin-20261104-194054` | 1 063 s | 57.81 | 2.09 / 2.86 |
| B2 | `t39-B2-dynamic-pin-20261104-195852` | 1 104 s | 56.97 | 2.06 / 2.81 |
| A2 | `t39-A2-static-20261104-201730` | 1 080 s | 53.91 | 2.02 / 2.86 |

## The metric that decides

`bench/spec_primary.py`, the agentic-weighted rule fixed before the t34 data was read (12 workloads;
pass needs the ratio at or above `max(1.02, 1 + same-arm spread)` with **every** primary category at or
above 1.00; withdraw fires on a primary below 1.00 **or any primary category below 0.98**):

```
primary  control 68.60  treat 68.58  ratio 0.9997 (-0.03 %)  floor 1.046 (spread 4.65 % / 0.69 %)
  Agentic; code    66.98 ->  67.34   +0.53 %   acc 2.79 -> 2.79
  Agentic; curl    65.41 ->  67.34   +2.94 %   acc 2.69 -> 2.55
  Coding           75.56 ->  72.32   -4.28 %   acc 2.61 -> 2.69
corpus   ratio 1.0536   acc delta -0.023
secondary worst: Translation +7.2 %, Creative +13.5 %, Translation (reasoning) +15.7 %
verdict  WITHDRAW
```

**Verdict: withdraw.** Two independent grounds: the weighted comparator is flat inside A's own 4.65 %
spread, and `Coding` falls 4.28 %, past the 0.98 line on its own. The whole-corpus gain (+5.4 %) comes
from Creative and Translation — workloads this deployment does not serve, and precisely the reason the
rule weights by workload instead of averaging categories.

## What this does and does not say

- **It says** the pin + promoter + dynamic placement, as a package, does not pay on the drafting path for
  agentic traffic. The pre-committed tree (recorded before the data) therefore restores production and
  closes the thread.
- **It does not attribute** the loss. The arm changed three things at once; the decomposition designed for
  this (`B0` = dynamic + pin without the promoter) was not run, because the tree spends no further
  passes on a losing arm. Anyone reviving this should run it first.
- **The prefill finding stands** (T32/T33: +16–22 % on streamed prefill) and remains mostly
  prefix-cached in production, so it is not a deployment reason either.
- **A correction on the record:** an earlier note quoted B1's "+5.1 %". That was the *unweighted corpus
  mean*; on the workload-weighted primary the same run is below the control. The corpus number is
  reported here only to show where the gain came from.

## How to re-run

```
# scratch root: a copy of _agent/v165 with eval/spec_decode.py (the one with reset_placement)
run_pass.ps1 -Tag t39-A1 -Root <scratch> -Runner spec_decode -Env "EXL3_MOE_CPU_SWAP=0"
run_pass.ps1 -Tag t39-B1 -Root <scratch> -Runner spec_decode \
  -Env "EXL3_MOE_CPU_SWAP=1,EXL3_MOE_PINNED_ARENA=1,EXL3_MOE_PINNED_ARENA_GIB=24" -Extra "-swap_ok"
# then:
python3 bench/spec_primary.py --csv <scratch>/results/spec_decode.csv --control t39-A --treat t39-B
```

Post gate: `bench/verify_task.py --post --report wiki/plans/v165-reports/T39-pin-promoter-drafting.md`
(run from the `165-win-perf` worktree).
