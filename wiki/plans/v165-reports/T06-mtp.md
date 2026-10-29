# T06 — MTP drafting knobs

Date: 2026-10-29. Engine stock 1.6.5. Runner: `eval/spec_decode.py` (sha
`204fac5f…`) through `run_pass.ps1 -Runner spec_decode`, so the watcher, floors, run-id logging and
`passes.csv` apply as for any pass.

## Step 1 — batched vs serial readback (`-rb_ab`), screening model

Run `rb-ab-166-20261029-202655` (1949 s, status ok, floors clean: min VRAM 7615 MB, min RAM
19460 MB). Invocation: `-Extra "-rb_ab -mtp"` — **`-rb_ab` alone measures only the baseline**
(the drafting section is skipped without a draft model; a first attempt cost 375 s and produced
no arm).

Per-arm mean over the eight workload categories (8 categories × up to 1024 tokens each):

| arm | mean t/s | vs baseline |
|---|---|---|
| Baseline (no drafting) | 45.76 | 1.000 |
| MTP, batched r1 | 44.06 | 0.963 |
| MTP, serial r1 | 47.82 | 1.045 |
| MTP, serial r2 | 48.29 | 1.055 |
| MTP, batched r2 | 49.69 | 1.086 |

Within-arm agreement: serial r1/r2 median category ratio **1.003** (range 0.87–1.10); batched
r1/r2 median **0.937** (range 0.47–1.04, driven by a "Trivial repetition" outlier of 0.47).

**Verdict: no readback verdict.** The two batched rounds differ by 13 % while the two serial
rounds differ by 1 %, so the arms overlap their own noise and neither beats the baseline
convincingly. The default (**batched, `_NO_BATCH_READBACK = False`) stands** — it is what the
plan expects, and there is no evidence here to change it. Revisit only if a later arm shows a
large effect that needs the last few percent explained.

## The screening model cannot decide drafting knobs (measured)

| checkpoint | body | MTP head | acceptance observed |
|---|---|---|---|
| 1.66bpw_h4_ng3 (screening) | **1.91 bpw** | 4.01 bpw | 1.0–2.2 of 4 drafts → drafting break-even (0.96–1.09× vs baseline) |
| 3.68bpw_h6_ng8 (production) | **4.00 bpw** | 6.01 bpw | to be measured |

The two-model protocol screens **load-bound** knob ratios on the 1.66 model; drafting acceptance
is not load-bound, it is a *quality* property of the body/head pair. A 1.91 bpw body is noisier
than the head that predicts its tokens, so acceptance collapses and every drafting knob looks
neutral. **Ruling: T6 steps 2–5 and all of T7 run on the 3.68 production model.** Cost if wrong:
the drafting series costs production-model load times (~60 s extra per run) instead of screening
load times.

## Cache sizing for the drafting arms

`spec_decode.py` measures short workloads (KBs of context) but `model_init` sizes the cache from
`-cs`; the drafting arms therefore run with `-cs 32768` instead of the regime's 262144. Two caches
exist while drafting (target + draft), so the regime value would cost ~4.2 GB of VRAM for context
that never appears in a workload. Knob *ratios* stay comparable because every drafting arm uses
the same `-cs`. Whether the deployed server can hold 262144 **and** a drafter on the 3.68 model is
a separate fit question, carried to T13.

## Drafting on the production model costs a second CPU arena

First attempt (`rb-ab-368`, split 400, `-cs 32768`) breached at 33 s: free RAM fell to **447 MB**
(floor 1200) during the load. The log shows why — the MTP head is loaded as a second component
with its own worker: `CPU MoE worker started: 1 layers`, because **the MTP head inherits the
main `-mcs`** (model_init shares one `Config` under `use_mtp`; only the *offload*-style knob has a
draft-specific twin, `-dmcl`, which does not apply to the split path). So drafting adds ~0.9 GB
of pinned arena on top of the body's 43 GB.

Ruling: the drafting series runs at **`-Split 384` + `-cs 65536`** (`-cs 32768` is too small:
`Job requires 139 pages (only 128 available)` — a 1024-token generation over a prompt needs
≥35584 tokens of cache) — 16 fewer body experts on the
CPU (≈1.7 GB host RAM back, ≈1.7 GB more VRAM) and a 64k cache instead of 262144 (≈1.5 GB more
VRAM, and the workloads are short). Every drafting arm shares those two constants, so knob ratios
stay comparable; the deployment consequence (a drafter needs ~2 GB more VRAM or a lower split)
goes to T13.
