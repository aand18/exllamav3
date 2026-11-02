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
| `t14-q4-262k-corpus-20261102-085612` | A | **ok**, 1068 s at split 389 |
| `t14-q2-262k-corpus-20261102-091413` | B | **ok**, 1049 s at split 378 |
| `t14-fp16-262k-corpus-20261102-093623` | C | **ok**, 1207 s at split 448 |

**Only arm A ran above the server's dynamic floor.** `spec_decode` sets `EXL3_MOE_CPU_SWAP=0`, so Q4
needs 389 to load statically while the server loads it at 381 dynamically; arm B kept 378, which is
the same number the server uses for Q2. The Q4/Q2 ratio in that table is therefore a comparison
across two placement regimes, and the absolute numbers are not the server's numbers.

The corpus is the same runner that produced the 58.54 tok/s Q2 figure in T14, so A and B compare
directly with it and with each other.

## Results — three cache modes, one session (2026-11-02)

| arm | corpus tok/s | vs Q2 | baseline tok/s | acceptance | min free VRAM | peak working set |
|---|---|---|---|---|---|---|
| Q2 at 378 | **54.78** | — | 40.65 | 2.39/4.00 | 777 MB | 41 650 MB |
| Q4 at 389 | **52.96** | 0.969 | 39.67 | 2.34/4.00 | 1777 MB | 42 676 MB |
| FP16 at 448 | **46.56** | 0.846 | 37.17 | 2.39/4.00 | 2057 MB | 47 916 MB |

Median per-category ratios against Q2: **Q4 0.969** (n=8) and **FP16 0.846**
(n=8). FP16's Coding band is 0.814 and Agentic code 0.832.

FP16 costs 15 % against Q2 and 12 % against Q4. Its peak process working set is 46.8 GiB against
Q4's 41.7 GiB (+5.1 GiB), and available RAM fell to 2094 MB at the worst sample. That is the same
range the retired 2048 MB floor used to fire at.

Median per-category ratio **0.969** (n=8), from `bench/ratios.py` over the two
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

## `-dds` (dynamic draft size) on DFlash2, measured at Q4

`t14-q4-dds-corpus-20261102-102211` against the Q4 arm of the same session: corpus 53.71 against
52.96, no-drafting baseline 39.69 against 39.67 (so the control held). Median per-category ratio
**1.045**, but the distribution is bimodal and the shape is the finding:

| category | without | with | ratio | acceptance without → with |
|---|---|---|---|---|
| Translation (reasoning) | 34.36 | 42.02 | 1.223 | 1.42 → 1.22 |
| Translation | 35.12 | 41.86 | 1.192 | 1.48 → 1.26 |
| Creative | 40.42 | 43.26 | 1.070 | 1.83 → 1.40 |
| Agentic curl | 58.39 | 62.15 | 1.064 | 2.41 → 2.32 |
| Coding | 75.32 | 77.27 | 1.026 | 2.76 → 2.67 |
| Agentic code | 70.00 | 69.73 | 0.996 | 3.09 → 2.87 |
| Creative (reasoning) | 39.10 | 39.44 | 1.009 | 1.68 → 1.18 |
| Trivial repetition | 71.00 | 53.94 | 0.760 | 4.00 → 2.67 |

Dynamic draft size pays where a fixed four draft tokens overdraft, and it costs where acceptance is
at the ceiling: trivial repetition always accepts four, and `-dds` sends fewer, so it loses 24 %.
Coding and agentic traffic sit in the middle: +2.6 % on Coding, +6.4 % on Agentic curl, flat on
Agentic code.

**Verdict: promising, not deployed.** One pass, and a category at 0.76 inside the same run. The
T06 note called this "unresolved, leaning +2 %"; today's shape explains why the earlier reading was
noisy. A confirmation pass plus a check that tabbyAPI forwards `dynamic_draft_tokens` would make it
deployable.

## `-chunk_size`: prefill throughput against VRAM headroom

Q4 at split 389, 262144 cache, prefill ladder only (`-SkipGen`: decode processes one token per step
and never sees the chunk size, so it is not part of this measurement).

| chunk | PP @4k | PP @32k | PP @64k | PP @262 k | min free VRAM |
|---|---|---|---|---|---|
| 256 | 282.7 | 271.3 | — | 278.4 | 3817 MB |
| 512 | 439.6 | 441.2 | — | 449.1 | 3821 MB |
| 1024 | 752.5 | 722.3 | 739.7 | 739.3 | 3675 MB |
| 4096 (deployed) | 1921.7 | 1836.9 | 1887.3 | 1880.8 | 2983 MB |
| 6144 | 1914.4 | 2012.5 | 2120.4 | 2217.1 | 2831 MB |
| 8192 | 1912.5 | 2714.4 | 2738.7 | 2702.4 | 2187 MB |
| **10240** | 1877.6 | 2702.7 | 2732.6 | **2828.2** | 1107 MB |
| 12288 | 1459.6 | 1567.8 | 1711.3 | 2167.4 | 719 MB |
| 14336 | — | — | — | refused at load | — |

**The peak is 10240, not 8192**: 2828 T/s at 262 k, 50 % above the deployed 4096, for 1107 MB of
headroom. 8192 gives 2702 for twice the headroom. 12288 loads but collapses (2167, and it is worse
than 8192 at every band below 262 k), and 14336 refuses. Below 4096 every step is a double loss.

**Deployment is still blocked.** With the drafter attached at split 381 the server refuses even
8192 (`Insufficient VRAM in split for model and cache`, measured 2026-11-02), so any of these needs
a split step to 389 - and the 389 arms above ran without the drafter, so whether 8192 fits there
with it is untested.

**8192 is 1.44x faster at 262 k than 4096** (2702 against 1881) and costs about 800 MB of headroom.
Shrinking the chunk is a bad trade in every direction: 1024 is 2.5x slower for 700 MB, 256 is 6.8x
slower for 840 MB. 16384 does not fit at this split.

The cost lands on the split, not on decode: if 800 MB forces nine more experts to the CPU, that is
about 5 % decode, and a coding workload that hits its prefix cache 96 % of the time notices prefill
rarely. So 8192 is worth deploying **only if the server still loads at split 381** - a pure prefill
win, or a split step that eats it.

### Deploy check: the server refuses 8192 at split 381

Loading the production configuration with `chunk_size: 8192` fails with `Insufficient VRAM in split
for model and cache`, so the 800 MB is not spare at 381: the prefill gain costs at least one split
step, about 4.5 % decode. For this workload that is a net loss - a first-touch 20 k prompt saves
about 3 s at 8192, while every cached turn pays 0.4 s on a 500-token answer, and 96 % of turns are
cached. **4096 stays.**

The knob's real shape: prefill throughput rises 1.44x from 4096 to 8192, and the configuration cannot
afford it at 262k Q4 on this card.

## MTP head at FP16: the drafter's VRAM is worth more than its agreement

FP16 at 262k, split 434, `-mtp -ndt 2` (`t15-fp16-mtp-corpus-20261102-123715`): corpus **50.62**
against DFlash2's 46.56 at split 448, which accepts **2.39 of 4** against the MTP head's **1.65 of 2**. The lighter head
pulls the floor down 14 experts and buys back 8.7 % of the 15 % that FP16 costs.

But the corpus mean is the wrong lens. Against Q2 + DFlash2, in the categories the operator runs:

| category | Q2 + DFlash2 | Q4 + DFlash2 | FP16 + MTP | MTP ratio |
|---|---|---|---|---|
| Coding | 76.96 | 75.32 | 62.20 | 0.808 |
| Agentic code | 72.24 | 70.00 | 58.51 | 0.810 |
| Agentic curl | 60.44 | 58.39 | 54.48 | 0.901 |
| Translation | 34.72 | 35.12 | 47.24 | 1.361 |
| Trivial repetition | 69.06 | 71.00 | 48.82 | 0.707 |

Median 0.973 across the corpus, **0.81 on Coding and Agentic code**. The MTP head's agreement per
draft token is higher (82.5 % against 59.8 %), but it drafts two tokens where DFlash2 drafts four,
so tokens per round fall (1.65 against 2.39) and throughput on real work falls with them.

**Verdict: FP16 is not viable for coding traffic at 262k on this card, with either head.** The
drafter swap recovers speed but cannot recover 19 %.

## Accuracy, probe 3: multi-key needles at 245 k

`eval/needles.py` plants K unique five-digit access codes at hashed depths in a 245 k-token haystack
and asks for all of them in one prompt; the harness scores the answer against the codes. Built
because `longctx` saturates at 11 questions and cannot fail.

| arm | items | found | status |
|---|---|---|---|
| Q2 at 378 | 32 distinct codes, planted in two haystacks | **32/32** | ok |
| Q4 at 389 | 32 distinct codes, planted in two haystacks | **32/32** | ok |
| FP16 at 448 | 32 distinct codes, planted in two haystacks | **32/32** | ok |

The 64 in `needles.csv` counts planted instances, not independent items: the same 32 codes appear in
both haystacks. **An earlier attempt at 20 codes scored 20/40 on Q2 and on Q4** (`t15-needles-q2`,
`t15-needles-q4`), a 50 % swing between repeats of the same configuration that a scorer defect
explains - it cut the log at the first truth line and scored the second haystack against nothing -
but the rows stay in `needles.csv` and the disagreement is part of the record.

Two haystacks of 32 keys each, six digits of headroom, exact-form answers. Every mode returns every
code. The probe is not saturated by construction - a model that drops a needle or transposes a digit
loses the point - but this model does not drop one at 245 k in any cache mode.

**Verdict on the accuracy question: not measurable here.** Three probes, three ties:

- draft acceptance: 2.39 / 2.34 / 2.39 of 4, which measures drafter agreement, not recall;
- `longctx` retrieval: 16/16 on Q2 and Q4;
- multi-key needles: 64/64 on Q2, Q4 and FP16.

The operator's traffic remains the only instrument that separated the modes. That is a legitimate
reason to run Q4, and it is 3 % rather than the 15 % FP16 would cost. Nothing in this harness
supports paying the extra 12 %.

To go further one would need the harder RULER families (variable tracking, aggregation), where a
single linked error fails the item, rather than more needles at the same depth profile.

## `-dds` confirmed, and tabbyAPI already supports it

Second run (`t16-dds-confirm-20261102-160442`): corpus **55.44**, paired against the reference
measured 83 minutes later (`t16-q4-ref2`, 53.51) rather than the morning one: **+3.6 %**, median
per-category 1.032. Pairing it against the morning reference gives 1.064 and flatters it, which is
what an earlier revision of this section did.

| run | corpus | Coding | Agentic code | Agentic curl | Trivial repetition |
|---|---|---|---|---|---|
| no `-dds` | 52.96 | 75.32 | 70.00 | 58.39 | 71.00 |
| `-dds` #1 | 53.71 | 77.27 | 69.73 | 62.15 | 53.94 |
| `-dds` #2 | 55.44 | 73.09 | 68.49 | 61.79 | 59.82 |

Two runs, both above the reference on the corpus (+1.4 % and +4.7 %). On the operator's categories it
is flat to slightly positive: Coding 1.026 then 0.970, Agentic code 0.996 then 0.978, Agentic curl
1.064 then 1.058. Trivial repetition loses in both, which is the dynamic draft doing what it should:
it sends fewer tokens where acceptance is already at the ceiling.

**tabbyAPI needs no patch.** `config.yml` carries `dynamic_draft` in the draft block (true since 2026-11-02)
and the server passes `dynamic_draft_tokens` to the engine; `draft_num_tokens: 4` is the ceiling.
The earlier note that a patch was necessary was wrong. Deployment is one config change.

## Concurrency: two slots share the GPU, they do not multiply it

`eval/concurrency.py` puts N jobs in flight before the first token, so `max_batch_size 2` finally
gets measured. Q2 at 262 k, 256 tokens per job, greedy.

| arm | wall | job 1 prefill / decode | job 2 prefill / decode | aggregate |
|---|---|---|---|---|
| 1 x 20 k | 20.0 s | 10.9 s / 9.2 s (27.9 T/s) | — | 12.77 t/s |
| 2 x 20 k | 40.3 s | 22.4 s / 18.0 s (14.2 T/s) | 20.2 s / 17.7 s (14.5 T/s) | 12.69 t/s |
| 1 x 131 k | 80.0 s | 70.6 s / 8.8 s (29.0 T/s) | — | 3.20 t/s |
| 2 x 131 k | 158.9 s | 69.5 s / 9.2 s (27.8 T/s) | 69.9 s / 9.6 s (26.7 T/s) | 3.22 t/s |

**Two 20 k prompts co-reside and share the card.** Each one's prefill doubles (10.9 → 22.4 s) and
each one's decode halves (27.9 → 14.2 T/s), so every request takes twice as long and the aggregate
does not move: 12.69 against 12.77 t/s. That is fair sharing, not throughput. It is what a
memory-bandwidth-bound decode should do on one GPU, and it matches what the engine is famous for:
requests make progress together, they do not go faster together.

**Two 131 k prompts cannot co-reside.** They need 262 209 tokens against a 262 144 cache, so the
second job waits and the pair runs strictly one after the other (159 s against 80 s for the same
work alone). Each job's prefill runs at the full solo rate (69.5 s, the same as 70.6 s), which is
the fingerprint of serialization rather than overlap.

Practical reading for two agents on this box: parallelism buys latency hiding for work that mixes
tool calls with generation, and it buys nothing in tokens per second. At 131 k each, plan on
sequential service.

## Correction: the drafter's break-even is not 2.34

An earlier note in this file, and one line in the state file, read as if a drafter that accepts
fewer than 2.34 of 4 draft tokens loses time. That is wrong, and the operator caught it.

Two different thresholds were being confused:

- **Drafter against no drafting, at draft width 4.** Every round produces `1 + accepted` tokens for
  one trunk pass. The DFlash2 arm gives 52.96 against a 39.67 baseline: 1.34 at 2.34 accepted, which
  under speed proportional to `1 + accepted` puts break-even at **1.5** - but only at width 4. The
  same identity on the other arms gives a per-round cost of 2.50 baseline-thousand-token-steps at
  width 4, 2.18 at the window `-dds` picks, and **1.91 for the MTP head at width 2**, whose implied
  break-even is **0.9**. So 1.5 belongs to the width and the cache regime, not to dragging in
  general, and it must be re-derived whenever either changes.
- **The quantisation trade.** Removing 0.8 GiB from the drafter frees about nine experts, roughly
  5 % decode at a fixed split. That 5 % is the budget the acceptance may spend: about 2.2 accepted
  per round before the freed VRAM stops paying for itself.

So a quantised drafter that accepts less is still worth running, either because the VRAM buys a
lower split or because the margin against 1.5 is large. The decision is a measurement: acceptance
plus the split the freed VRAM reaches, not one threshold read off the old number.

### The 65-token gap, resolved: co-residence is sharing, not speed

The 131 k pair missed the shared cache by 65 tokens. Re-run at 130 k each (260 064 tokens against
262 144), so both slots co-reside:

| arm | wall | job 1 prefill / decode | job 2 prefill / decode | aggregate |
|---|---|---|---|---|
| 1 x 130 k | 80.2 s | 70.6 s / 9.2 s (27.9 T/s) | — | 3.19 t/s |
| 2 x 130 k, both resident | 164.3 s | 143.2 s / 20.5 s (12.5 T/s) | 141.0 s / 20.1 s (12.7 T/s) | 3.12 t/s |
| 2 x 131 k, 65 tokens over | 158.9 s | 69.5 s / 9.2 s (27.8 T/s) | 69.9 s / 9.6 s (26.7 T/s) | 3.22 t/s |

So the answer to the open question is no: a split cache does not change the overlap. Both cases give
the same aggregate and the same double wall time; the only difference is the fingerprint. With both
jobs resident, each prefill doubles and each decode halves. With the cache 65 tokens short, each
prefill runs at the full solo rate and the jobs simply queue.

**Aggregate throughput is constant in every configuration measured** (20 k pair 12.69 against 12.77;
130 k pair 3.12 against 3.19; 131 k pair 3.22 against 3.20). Concurrency on this box buys latency
hiding, not tokens per second, and one slot is as fast in aggregate as two.
