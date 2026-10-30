# T11 — cache staging pages, recurrent checkpoints, Q2 sanity

Date: 2026-10-30. Engine win-perf `1.6.5`. Regime: 3.18bpw, split 376, `-cs 262144` where the task
asks for long context, `-cq 2,2` (operator-fixed; no mode ladder), `-NoSpecDec`, frozen profile per
run. Cache mode is a decision, not a candidate: Q2 is operator-fixed, so Step 1–2 measure the knobs
that make that mode affordable rather than remeasuring the mode itself.

## Step 1 — `EXL3_QC_STAGING_PAGES`: inert at this chunk size (mechanism, not a null)

| arm (run id) | PP 65536 | PP 131072 | PP 262144 | TG 261888 | peak_alloc | peak_reserved | min free VRAM |
|---|---|---|---|---|---|---|---|
| `t11-sp64-…` | **1886.42** | **1867.22** | **1850.08** | 39.08 | 20618 | 20794 | 2883 MB |
| `t11-sp256-…` (default) | 1843.58 | 1830.45 | 1828.03 | 39.35 | 20618 | 20794 | 2883 MB |
| `t11-sp512-…` | 1847.04 | 1845.80 | 1834.54 | 38.75 | 20618 | 20794 | 2883 MB |
| `t11-sp1024-…` | 1839.30 | 1832.45 | 1822.84 | 39.09 | 20618 | 20794 | 2883 MB |

The four arms' allocator watermarks are **byte-identical** (20618 allocated, 20794 reserved, 2883 MB
free), and the pp ordering is a ~1 % spread across a 16× change in the knob. The engine says why:

```python
npps_w = block_table.shape[1]                      # pages the window needs, per call
w = min(npps_w, max(1, max(_qc_staging_pages, staging_pages or 0)))   # the knob is a CAP
kd = torch.empty((bsz * w, page_size, n_kvh, head_dim), ...)          # per-call transient
```

At `-chunk_size 4096` a prefill call spans 4096 tokens = **16 pages of 256**, so `npps_w` is 16 and
every tested cap (64…1024) is above it: `w = npps_w` in all four arms, the same scratch, the same
kernels. The knob only binds when the per-call window exceeds the cap — a chunk size of 64k or more.

**The repeat confirms it.** Two arms re-run with the knob at the two ends:

| arm | PP 131072 | PP 262144 | mean (two passes) |
|---|---|---|---|
| `t11-sp256` and `t11-sp256-r2-…` | 1830.45 / 1856.29 | 1828.03 / 1848.12 | 1843.4 / 1838.1 |
| `t11-sp64` and `t11-sp64-r2-…` | 1867.22 / 1837.92 | 1850.08 / 1819.35 | 1852.6 / 1834.7 |

The 16× knob moves the mean by +0.5 % at 131072 and −0.2 % at 262144, while the same arm varies
0.7 % peak-to-peak between its own two passes. The ordering in the first pass does not survive.

**Verdict: the knob is inert at the campaign's chunk size, and the ~1 % pp differences are noise.**
It is not a lever to ship at `-chunk_size 4096`, and the three-page "PP 64 is fastest" reading that
the raw numbers invite would be a fabrication. If chunk size is ever raised, the cap becomes
relevant and this table is the calibration for it.

*(Consequence worth carrying: a bigger `-chunk_size` is the knob that would make both the staging cap
and the per-call overhead matter. It is not in T11's scope and the regime fixes chunk 4096; recorded
as an unexplored lever rather than a result.)*

## Step 2 — recurrent state in fp32 (`-rs32`)

| arm (run id) | PP 262144 | TG 261888 | peak_alloc | peak_reserved | min free VRAM |
|---|---|---|---|---|---|
| `t12-rs32-…` | pending | | | | |
| `t13-262k-perf-20261030-080210` (fp16 default) | 1839.54 | 39.32 | 20618 | 20794 | 2855 MB |

## Step 3 — Q2 retrieval sanity at length

Wired first: `eval/longctx.py` is now a harness runner (`-Runner longctx`), because "run it some
other way" is how a measurement stops being recorded. The script names each test and the prompt
length it exercised and prints one `LONGCTX job=… prompt_tokens=… out_chars=…` line per test; the
harness parses them into `results/longctx.csv` and marks a run with none as `fail`. Retrieval runs
get `EXL3_MOE_CPU_SWAP=0` like drafting runs (a placement sweep can flip the greedy token an answer
depends on) and carry no draft args. `deploy.sh` ships the script and its `eval_texts/` corpus, whose
largest file is 700 KB (~177 k tokens) for the `-xl` arm.

Scoring is a reader's job, aided — not replaced — by `bench/longctx_facts.py`: it strips the
references the script prints beside each answer (they contain the very facts being looked for, so
leaving them in makes every check pass), splits the three Q&A answers at their own questions, and
checks whether the **context-only facts** came back. Summary tests are checked at plot level (a
summary may paraphrase the names away); Q&A at fact level. Output length is printed beside every row
because a truncated answer is an instrument limit, not a retrieval failure.

### Standard set, cache 65536, Q2 — 36.7 k tokens of context

`t11-q2-std-…` (`status=ok`, 293 s, floors clean).

| test | prompt tokens | facts | evidence |
|---|---|---|---|
| `ic_sum` | 13939 | 2/2 | Holmes named, plot (exposure, vitriol, the marriage) |
| `ic_french` | 13992 | 2/2 | names the translated paragraph and quotes French text |
| `ic_zoomer` | 13963 | 1/1 | identifies the rewritten passages |
| `vm_sum` | 36741 | 2/2 | "a time-displaced fix-it man … transforming it into a safer FTL drive" |
| `vm_q1` | 36764 | 2/2 | Milsom as the unpredictable variable |
| `vm_q2` | 36755 | 2/2 | **"rewires Icarus's control turret … safely returns from faster-than-light flight"** |
| `vm_q3` | 36761 | 1/1 | interstellar travel as the outcome |
| `vm_pony` | 36942 | 1/1 | finds the inserted passage |
| `vm_char` | 36741 | 1/1 | **14 of the 16 named characters** |
| **total** | | **14/14** | |

The `vm_q2` answer is the strongest evidence that this is retrieval and not prior knowledge: the
question asks what Milsom does to the *control turret*, a detail that exists nowhere but inside the
36.7 k-token context, and the answer names the mechanism exactly.

### Long set, cache 262144, Q2 — 165 k tokens of context

`t11-q2-xl-…` (`status=ok`, 1459 s, floors clean: min free VRAM 2931 MB, RAM 8929 MB).

| test | prompt tokens | facts | evidence |
|---|---|---|---|
| `ic_sum` | 13939 | 2/2 | Holmes plus the plot |
| `ic_french` | 13992 | 2/2 | identifies and quotes the French paragraph |
| `ic_zoomer` | 13963 | 1/1 | identifies the rewritten passages |
| `vm_sum` | 36741 | 2/2 | the time-displaced man and the FTL outcome |
| `vm_q1` | 36764 | 2/2 | Milsom as the unpredictable variable |
| `vm_q2` | 36755 | 2/2 | the turret rewrite |
| `vm_q3` | 36761 | 1/1 | interstellar travel |
| `vm_pony` | 36942 | 1/1 | finds the inserted passage |
| `vm_char` | 36741 | 1/1 | 14 of 16 names, in a clean 1.3 k-char final answer |
| `pp_mod` | **165471** | 1/1 | **quotes the inserted passage verbatim** — the Terra/Jorblax/Icarus paragraph — and names it as unrelated sci-fi |
| `pp_chap` | **165466** | 1/1 | per-chapter titles |
| **total** | | **16/16** | |

The `pp_mod` answer is the decisive one for this step: in 165 471 tokens of Pride & Prejudice with a
Q2 cache, the model recovers a *single inserted paragraph* verbatim and says what it is. A cache
quantisation that had degraded long-range attention could not do that.

### Verdict — Q2 retrieval holds at both lengths

| arm | context exercised | context-only facts | floors |
|---|---|---|---|
| `t11-q2-std` (cache 65536) | up to 36 942 tokens | **14/14** | clean |
| `t11-q2-xl` (cache 262144) | up to 165 471 tokens | **16/16** | clean, 2931 MB VRAM free |

No retrieval failure is visible at either length, so **Q2 can stay the operator's fixed cache mode**.
Two limits on that sentence, stated rather than buried:

- **This is "Q2 is fine", not "Q2 equals FP16".** A Q8/FP16 control at 262144 would need ~8.4 GiB of
  cache and does not fit this card at split 376, so the comparison the plan asks for ("vs a
  short-context reference") is answered *across context length at Q2*, not across cache modes.
- **The two arms differ in generation cap as well as context** — the script gives every test 1024
  tokens without `-xl` and 16384 with it — so the `-xl` arm has more room to surface a fact and its
  `out_chars` include long reasoning blocks. The fact check is a presence instrument, not a grade:
  the 48 k-char `vm_char` row is 16 k tokens of thinking around a clean 1.3 k-char answer.

A third limit belongs to the instrument: `bench/longctx_facts.py` can prove a fact came back, not
that an answer is good. The prose is in `wiki/plans/v165-reports/artifacts/logs/` and a reader can
disagree with any row above.
