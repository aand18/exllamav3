# T40/T41 — the PLE row cache costs more than it saves; the draft-window model is close but unterminated

Two arms measured back to back in one session (2026-11-04), server stopped, scratch harness root
`C:\Users\yoho\Downloads\_t4check` running the repo's `eval/spec_decode.py`, branch engine file-level in
the venv. Order `A B B A` per arm.

## t40 — PLE row cache (`EXL3_NGRAM_ROW_CACHE_ROWS=1048576` + a 1M-row pin)

The pin set was built from this repo's own text (`bench/ngram_pin_manifest.py` → `ngram_hot_rows.py`):
1 048 576 rows, 162 MB, covering **98.3 %** of that corpus's lookups. B1's identity header confirms the
arm (`EXL3_NGRAM_ROW_CACHE_PIN=…`, `…_ROWS=1048576`, `…_STATS=1`).

| run | status | elapsed | corpus draft tok/s | acc |
|---|---|---|---|---|
| `t40-A1-cacheoff-20261104-203604` | **breach** | 483 s | 55.02 | 2.04 / 2.84 |
| `t40-B1-cachepin-20261104-204424` | ok | 1 215 s | **54.90** | 2.02 / 2.80 |
| `t40-B2-cachepin-20261104-210456` | ok | 1 213 s | **55.51** | 2.04 / 2.86 |
| `t40-A2-cacheoff-20261104-212526` | ok | 1 035 s | **56.49** | 2.03 / 2.80 |

- The cache arm is **~1.3 tok/s below** the one valid control (55.21 vs 56.49, −2.3 %), and each cached
  pass costs **~180 s more** wall clock (the 162 MB pinned set is read at open).
- `spec_primary.py` cannot judge it: the control has a single valid run (A1 was killed by the RAM floor,
  caused by concurrent I/O on the box — recorded as an operator rule, not an arm property).
- **Direction is consistent and negative.** A second control would formalise it; none is quoted as a verdict.

## t41 — draft-window cost model (`EXL3_DRAFT_COST_MODEL=1`)

| run | elapsed | corpus draft tok/s | acc |
|---|---|---|---|
| `t41-A1-costmodel-off-20261104-214258` | 1 088 s | 54.56 | 2.07 / 2.86 |
| `t41-B1-costmodel-on-20261104-220123` | 1 053 s | **55.85** | 2.09 / 2.90 |
| `t41-B2-costmodel-on-20261104-221914` | 1 056 s | **56.23** | 2.02 / 2.85 |
| `t41-A2-costmodel-off-20261104-223707` | 1 062 s | 55.26 | 2.02 / 2.82 |

`bench/spec_primary.py`:

```
primary  control 69.33  treat 70.65  ratio 1.0190 (+1.90 %)  floor 1.020 (spread 1.53 % / 0.69 %)
  Agentic; code   68.00 -> 69.72   +2.54 %   acc 2.81 -> 2.75
  Agentic; curl   66.31 -> 67.22   +1.38 %   acc 2.77 -> 2.65
  Coding          75.59 -> 76.75   +1.54 %   acc 2.73 -> 2.72
corpus   ratio 1.0205   acc delta -0.070
verdict  INCONCLUSIVE
```

**Inconclusive, and narrowly so:** every primary category is positive (+1.4…+2.5 %), but the weighted
ratio lands at **1.0190 against a 1.020 floor** — 0.1 points short — and the acceptance delta (−0.070)
is outside the ±0.05 band. Under the rule as fixed, that is not a pass, and it is not reported as one.
It is the only arm in this campaign that trends positive on the workloads this box serves; the honest
next step is a repeat (two more passes per arm) rather than a deployment.

## What this changes

- The row cache does not pay on this box: the served path's NVMe reads are already hidden well enough
  that a 98 %-hit cache adds open cost and loses ~2 %. It stays in the tree, default-off.
- The draft-window cost model is the one open lead. It needs a repeat at the same bar, not a
  reinterpretation.
- Both arms' code is default-off; the deployed engine was restored to the pristine wheel
  (`328a82d8…`, `75ce3c64…`, `2d36c221…`, `02d82227…`) and the server restarted on port 5000.

## How to re-run

```
run_pass.ps1 -Tag t41-B1 -Root C:\Users\yoho\Downloads\_t4check -Runner spec_decode \
  -Env "EXL3_MOE_CPU_SWAP=0,EXL3_DRAFT_COST_MODEL=1"
python3 bench/spec_primary.py --csv <scratch>/results/spec_decode.csv --control t41-A --treat t41-B
```

## Repeat with four runs per arm (t43) — the t41 gain did not reproduce

The t41 result's own rule demanded a repeat; a second campaign ran the **reverse** order (`B A A B`, where
t41 ran `A B B A`) and added the missing t40 control. All five passes `status=ok`
(`t43-B3-…`, `t43-A3-…`, `t43-A4-…`, `t43-B4-…`, `t40-A3-cacheoff-…`).

| arm | corpus draft tok/s, four runs | mean |
|---|---|---|
| t41 control (A) | 54.56, 55.26, 56.88, 56.54 | 55.81 |
| t41 cost model (B) | 55.85, 56.23, 56.16, 56.26 | 56.13 |

```
primary  control 70.10  treat 70.47  ratio 1.0053 (+0.53 %)  floor 1.043 (spread 4.26 % / 1.20 %)
  Agentic; code   +0.73 %   Agentic; curl  +0.89 %   Coding  -0.21 %
verdict INCONCLUSIVE
```

**The +1.90 % of the first two-run pass shrank to +0.53 %**, and `Coding` — positive in the first pass —
is now −0.21 %. The A arm's own spread grew to 4.26 %, so the first pass's floor of 1.020 was a narrow
window on a noisy arm. Honest reading: **the draft-window model is not demonstrated; its effect is inside
the noise of this harness.** It stays default-off and needs a different design of experiment (more decode
reps per cell, not more passes) before it is worth another campaign.

The same repeat gave t40 its second valid control, and the verdict is now a formal one:

```
primary  control 70.66  treat 69.60  ratio 0.9850 (-1.50 %)  floor 1.033 (spread 3.29 % / 2.32 %)
  Agentic; code  -2.19 %   Agentic; curl  +1.16 %   Coding  -3.51 %   acc -0.100
verdict WITHDRAW
```

**The PLE row cache costs 1.5 % on the workloads this box serves** and ~180 s per pass at open. That is
the item's answer: not a win, and not a close call.
