# T15 — cache modes at 262k, measured against each other

The operator moved production from Q2 to Q4 on 2026-11-02 because the Q2 cache was hurting answer
accuracy on real traffic. This task measures what that costs, and what the next step costs, in one
session so the numbers are comparable.

## Configurations

All at `cache_size` 262144, `max_batch_size` 2, DFlash2 BF16 window 4 (`-ndt 4`), the frozen
profile, and the same corpus. Only the cache mode and its split floor change.

| arm | cache | split | state file | why |
|---|---|---|---|---|
| A | Q4 `"4,4"` | 381 | `state-q4.json` | the deployed configuration |
| B | Q2 `"2,2"` | 378 | `state-q2.json` | same-session reference, the mode it replaced |
| C | FP16 | ~448 | `state-fp16.json` | the cache-accuracy ceiling; floor still to be found |

Split floors are engine-verified at 262144: 376 and 378 refuse at Q4, 381 loads; Q2 loads at 378.

## Decision rule

- **Acceptance** is the accuracy signal the operator cares about: draft tokens accepted per round.
  Q2 and Q4 differ if the intervals separate; the Q2 arm is the control.
- **Decode tok/s** decides the cost. If Q4 lands within a couple of percent of Q2, the accuracy came
  free and the question closes.
- FP16 is worth deploying only if it buys acceptance beyond Q4. At ~440–448 split it moves about 60
  experts to the CPU.

## Runs

| run id | arm | status |
|---|---|---|
| `t14-q4-262k-corpus-*` | A | running |
| `t14-q2-262k-corpus-*` | B | queued |
| `t14-fp16-262k-corpus-*` | C | queued, floor check first |

The corpus is the same runner that produced the 58.54 tok/s Q2 figure in T14, so A and B compare
directly with it and with each other.

## Results

To fill when the arms finish.

## Harness wart found while setting up

`state.json` has no field for the draft **model** path. `run_pass.ps1` emits `-mtp` for
`draft.mode = mtp` and never emits `-dm`, because the state was written when the MTP head rode the
main model. Production now uses the DFlash2 sidecar, so every DFlash2 run passes the path through
`-Extra "-dm <dir>"`, and a run that forgets it silently drafts with the MTP head while the state
still says `spec_decode`. Fix: add `draft.draft_model` and emit `-dm` from the regime, then set
`state.json` to the production truth. Not changed mid-campaign, because A/B/C must share one harness.
