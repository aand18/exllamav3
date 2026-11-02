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
| `t14-q4-262k-corpus-*` | A | **ok**, 1068 s at split 389 |
| `t14-q2-262k-corpus-*` | B | **ok**, 1049 s at split 378 |
| `t14-fp16-262k-corpus-*` | C | **ok**, 1207 s at split 448 |

**A and B both ran 8 experts above the server's dynamic floor**, because `spec_decode` sets
`EXL3_MOE_CPU_SWAP=0` and static placement needs more. Q4 at 381 refuses in this harness (25 s) while
the server loads it at 381 with dynamic placement; Q4 needs 389 static, Q2 keeps 378. The static tax
applies to both arms, so the ratio stands, but the absolute numbers are not the server's numbers.

The corpus is the same runner that produced the 58.54 tok/s Q2 figure in T14, so A and B compare
directly with it and with each other.

## Results — three cache modes, one session (2026-11-02)

| arm | corpus tok/s | vs Q2 | baseline tok/s | acceptance | min free VRAM | peak working set |
|---|---|---|---|---|---|---|
| Q2 at 378 | **54.78** | — | 40.65 | 2.39/4.00 | 777 MB | 41 650 MB |
| Q4 at 389 | **52.96** | 0.969 | 39.67 | 2.34/4.00 | 1777 MB | 42 676 MB |
| FP16 at 448 | **46.56** | 0.846 | 37.17 | 2.39/4.00 | 2057 MB | 47 916 MB |

Median per-category ratios against Q2: **Q4 0.969** (n=8, 0.836–1.028) and **FP16 0.846**
(0.798–0.949). FP16's Coding band is 0.814 and Agentic code 0.832.

FP16 costs 15 % against Q2 and 12 % against Q4, and it moves 6.3 GB more into the host arena: its
peak working set is 47.9 GB, and available RAM fell to 2094 MB at the worst sample. That is the same
range the retired 2048 MB floor used to fire at.

Median per-category ratio **0.969** (n=8, range 0.836–1.028), from `bench/ratios.py` over the two
runs. The categories the operator cares about:

| category | Q2 tok/s | Q4 tok/s | ratio | Q2 acc | Q4 acc |
|---|---|---|---|---|---|
| Coding | 76.96 | 75.32 | 0.979 | 2.76 | 2.76 |
| Agentic code | 72.24 | 70.00 | 0.969 | 3.18 | 3.09 |
| Agentic curl | 60.44 | 58.39 | 0.966 | 2.33 | 2.41 |
| Translation (reasoning) | 41.08 | 34.36 | 0.836 | 1.74 | 1.42 |

**Q4 costs about 3 %, not the 19 % the first catalog predicted.** The expert count said 1.7 %; the
measured 3.1 % is that plus the run-to-run spread.

**Acceptance is flat, and that is expected.** Acceptance measures how often the drafter and the trunk
agree on a token, not whether the trunk recalled the right paragraph. A cache that hurts long-context
accuracy leaves token-level agreement untouched, so this number can neither confirm nor refute the
operator's observation. The probe for that is retrieval at length, not acceptance.

## Accuracy: retrieval at length

`-Runner longctx -xl` on Q4 at split 389 (`t14-q4-262k-longctx-20261102-095950`, 1197 s):
**16/16 facts**, the same score Q2 reached at 165 k on 10-31 (`t13-378-xl`). Every one of the eleven
jobs retrieved its fact, including the inserted paragraph inside 165 k tokens of Pride and Prejudice
and the chapter-title list.

So the cache does not separate on this probe either. Q2 and Q4 both saturate it.

**A transcript diff cannot answer the remaining question.** The outputs differ a lot between the two
runs (ic_zoomer 5,917 characters on Q4 against 34,060 on Q2; pp_chap 41,705 against 64,341), which is
sampling variance, not retrieval: the runner samples, so one draw per mode carries no attribution. A
controlled answer would be greedy decoding on the same prompts in both modes, two runs — not run here.

What the evidence supports: on every probe available in this harness, **Q2 and Q4 are
indistinguishable**, and Q4 costs 3 %. That makes Q4 cheap insurance rather than a measured gain. The
operator's own traffic remains the only instrument that separated them, which is why the switch was
made on that observation rather than on a number from here.

## Harness wart found while setting up

`state.json` has no field for the draft **model** path. `run_pass.ps1` emits `-mtp` for
`draft.mode = mtp` and never emits `-dm`, because the state was written when the MTP head rode the
main model. Production now uses the DFlash2 sidecar, so every DFlash2 run passes the path through
`-Extra "-dm <dir>"`, and a run that forgets it silently drafts with the MTP head while the state
still says `spec_decode`. Fix: add `draft.draft_model` and emit `-dm` from the regime, then set
`state.json` to the production truth. Not changed mid-campaign, because A/B/C must share one harness.
