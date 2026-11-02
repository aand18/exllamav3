# T17 — what the two audits found, and what was done

Two clean-context auditors reviewed this campaign against its own data on 2026-11-02: one on
arithmetic and units, one on statistical inference. Both were told to assume error. This file records
what they found and the disposition, so the record survives the session.

## The finding that matters most

**Basis drift.** Nearly every number traces to a real CSV cell, and the arithmetic on top is what goes
wrong: a KV-only figure is differenced against a KV-plus-recurrent one, a static split floor against
a dynamic one, a size-check model value tabulated beside a watcher reading, a MiB cell printed as GB
in one document and GiB in another, a per-category pair quoted as a corpus pair, and a coefficient
measured on a day the campaign itself calls an outlier carried into the regime the box actually runs.

The proof that it is not pedantry: the cost of the Q4 cache, the number the production configuration
rests on, appeared in the record as **1.7 %**, **3.4 %**, **0.5 %** and **0.3 %** for the same three
experts, each "supported by evidence".

## Corrections applied

| what the record said | what the data says | where |
|---|---|---|
| FP16 cache "6.94 GiB … Q2 costs 2.09 GiB, matches measured" | 6.94 GiB is KV only; 2.09 is total. Like-for-like: KV 1.875 (Q2), 6.94 (FP16) | T14 |
| "+4.5 GiB RAM" for FP16 at 262k | +5.0 GiB (37.6 against 32.6) | T14 |
| rows 5/6 "−10 %" and "−26 %" at 0.56 %/expert | 22 × 0.56 = 12.3 %, 62 × 0.56 = 34.7 %; measured arm is −15.0 % vs Q2 | T14 |
| "46–53 T/s live" | the table's own buckets span 42.0–56.9 | T14 |
| "A and B both ran 8 experts above the floor" | only A did; B kept 378, the server's own Q2 number | T15 |
| MTP "2.39 of 4 against 1.65 of 2" | reversed: MTP 1.65/2, DFlash2 2.39/4 | T15 |
| "prefill doubles from 4096 to 8192" | 1.44× | T15 |
| "moves 6.3 GB into the host arena … 47.9 GB" | +5.1 GiB of process working set (46.8 against 41.7 GiB) | T15 |
| "`dynamic_draft: false`" | true since 2026-11-02 | T15 |
| "5.2 GB more working set (47.9 against 42.7 GB)" | +5.1 GiB (46.8 against 41.7 GiB) | T16 |
| "2.84 against 2.76" and "46.91 against 49.00" as corpus | both Coding-only; corpus is 2.35 against 2.34 and 38.49 against 39.67 | T16 |
| 0.75 GiB "pushes eleven experts" | 8.8 at the tool's 87.6 MiB/expert; 11 is the static split gap | T16 |
| "1.7 % decode" for the deployed Q4 | same-day dynamic ladder: 0.8 % for eight experts, so under 1 % for three | STATE |
| load "31.3 s" | 32.3 s for the 17:48 boot | STATE |

## Open, with the measurement that would settle each

- **The served configuration has never been measured with the drafter attached.** Every drafting arm
  runs static placement; production runs dynamic. Settle with a Q4@381 and a Q2@378 `spec_decode`
  pair, draft-only, against a `perf` ladder at the same split.
- **The split floors are quoted across placement modes.** Q2's 378 comes from static probes; Q4's 381
  from a dynamic load. Whether Q2 loads lower than 378 dynamically has never been probed (382–388 were
  never tried at Q4 either, so 389 is a split that worked, not a floor).
- **The needles tie is at ceiling.** 32 distinct codes found in every mode; a probe that cannot fail
  bounds a difference at one item in 32. Needs a task family where the reference misses one — RULER
  variable tracking or aggregation — plus a degraded-cache positive control.
- **Production acceptance is a context effect, not a workload effect.** 39.6 % live against 58.5 %
  corpus tracks prompt length (64.7 % below 16 k, 39.5 % above 200 k). One acceptance arm with a
  ~100 k filler prefix settles it and re-prices the drafter at production length.
- **Two identical Q4 references differ by 1.0 % on the corpus and 3.2 % on Coding.** Every 1–4 %
  claim in this campaign is of that order. Per-category σ needs one more repeat pair.

## Minor, recorded not fixed

Three cells in T13's huge-page table (PP 1843.10 and 1836.44, TG median 38.95, PP median 1832), the
"1393 MB" watcher minimum that the watch file does not contain, "−cs 262144" for a table whose runs
are all `-cs 131072`, the `-sdh` miss-rate and probe counts attributed to the wrong arm, and the
`BREACH-server262.txt` citation pointing at the second breach rather than the first.
