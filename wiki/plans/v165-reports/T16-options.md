# T16 — the options table

One place to choose from. Every row is a configuration this box has run, or a compromise that the
measurements rule out. Corpus arms are `spec_decode` at 262 144 tokens unless the row says otherwise,
DFlash2 window 4, Qwen3.8-Flash-Next `3.18bpw_h6_ng8`.

## What is on the table

| option | cache | split | corpus t/s | vs deployed | draft accepted | min free VRAM | peak working set | min avail RAM | evidence |
|---|---|---|---|---|---|---|---|---|---|
| **deployed** | Q4 262k | 381 (dynamic) | — | — | — | 1681 MB | 42.6 GB | 9074 MB | server boot, no harness row |
| Q4 262k, static floor | Q4 262k | 389 | **52.96** | reference | 2.34/4 | 1777 MB | 41.7 GB | 4652 MB | `t14-q4-262k-corpus-20261102-085612` |
| Q2 262k | Q2 262k | 378 | **54.78** | +3.4 % | 2.39/4 | 777 MB | 40.7 GB | 6880 MB | `t14-q2-262k-corpus-20261102-091413` |
| FP16 262k | FP16 262k | 448 | **46.56** | −12.1 % | 2.39/4 | 2057 MB | 46.8 GB | 2094 MB | `t14-fp16-262k-corpus-20261102-093623` |
| FP16 262k, MTP head | FP16 262k | 434 | **50.62** | −4.4 % | 1.65/2 | 657 MB | 46.2 GB | 2773 MB | `t15-fp16-mtp-corpus-20261102-123715` |
| FP16 131k | FP16 131k | 408 | pending | — | — | — | — | — | running |
| `-dds` on Q4, run 1 | Q4 262k | 389 | **53.71** | +1.4 % | 1.95/2.74 | 1777 MB | 41.7 GB | 6638 MB | `t14-q4-dds-corpus-20261102-102211` |
| `-dds` on Q4, run 2 | Q4 262k | 389 | **55.44** | +4.7 % | 2.07/2.90 | 1805 MB | 41.6 GB | 5632 MB | `t16-dds-confirm-20261102-160442` |
| chunk 8192 | Q4 262k | 389 | refused at 381 | — | — | — | — | — | `t15-chunk8192-pp` at 389, load refusal at 381 |

Two slots are deployed. Aggregate throughput is flat in every pair measured (12.77 against 12.69 at
20 k; 3.19 against 3.12 at 130 k), so the second slot buys latency hiding and nothing else.

## What each compromise costs

- **Q2 instead of Q4**: about 3 % faster on paper, and no probe in this harness can see the
  difference in answer quality. The operator's traffic is the only evidence that Q2 hurt, which is
  why Q4 is deployed.
- **FP16 instead of Q4**: about 12 % slower, 6.3 GB more host working set, and still no measurable
  quality gain on three probes. Not worth it on this card.
- **FP16 with the MTP head**: recovers half of the FP16 tax by dropping the split 14 experts, but its
  two-token drafts lose 19 % on Coding and Agentic code. Not viable for this traffic.
- **`-dds`**: two runs above the reference on the corpus and on Agentic curl, flat on Coding, and
  clearly negative on Trivial repetition. Deployed, because the workload that matters does not
  repeat.
- **Two slots** (deployed): aggregate throughput is flat in every pair measured, so the second slot
  buys latency hiding for two agents and nothing in tokens per second.
- **Chunk 8192**: 44 % faster prefill, refused at the deployed split; the split step it would need
  costs more decode than the prefill returns on this traffic.

## What is not measured, and why it matters

- **Q4 against Q2 at one split.** Q4 was measured at 389 and Q2 at 378. Eight experts is about 4.3 %
  by the campaign's per-expert coefficient, which is larger than the 3.4 % gap, so the cache cost is
  not separated from the placement cost. The run that settles it is Q2 at 389.
- **FP16 at 131k**: the last option with real value; running.
- **Production acceptance.** The corpus accepts 58.5 % of drafts; the live log shows 39.6 %, which is
  1.58 of 4 against a break-even near 1.5. Every drafting number here is priced on a friendlier
  workload than production.

## Evidence status

`bench/options_table.py` regenerates the measured rows from `passes.csv` and `arms.csv`, and
`bench/verify_task.py --post` gates the three reports this table draws on (T13, T14, T15).
