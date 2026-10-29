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

Ruling: the drafting series runs at **`-Split 392` + `-cs 65536`** (384 freed enough host
RAM — 2872 MB — but the two 64k caches plus the head ran VRAM to 17 MB; 392 buys 0.86 GB of VRAM
for 0.86 GB of RAM, and the RAM floor still has room) (`-cs 32768` is too small:
`Job requires 139 pages (only 128 available)` — a 1024-token generation over a prompt needs
≥35584 tokens of cache) — 16 fewer body experts on the
CPU (≈1.7 GB host RAM back, ≈1.7 GB more VRAM) and a 64k cache instead of 262144 (≈1.5 GB more
VRAM, and the workloads are short). Every drafting arm shares those two constants, so knob ratios
stay comparable; the deployment consequence (a drafter needs ~2 GB more VRAM or a lower split)
goes to T13.

## Drafting leaks ~700 MB of host RAM per arm in one process (measured)

The split-392 attempt died at 32 s (RAM 1165 MB) and the split-388 attempt ran 1765 s before
dying (RAM 974 MB). The watcher trace from the 29-minute run shows the shape:

| phase (min free RAM per eighth of the run) | 2517 | 5346 | 4796 | 3711 | 2756 | 2272 | 1808 | 1229 | 974 |
|---|---|---|---|---|---|---|---|---|---|
| free VRAM (same blocks) | 3333 | 2625 | 2431 | 2431 | 2427 | 2427 | 2427 | 2427 | 2427 |

VRAM is flat; host RAM falls monotonically — every further `Generator` built for another arm
costs ~0.6–0.8 GB that never comes back. So a four-arm `-rb_ab` run cannot finish on this box at
the 3.68 model's arena size, and that is a **deployment caveat** too: a long-lived server that
builds generators repeatedly grows the same way.

**Protocol change (ruling):** drafting arms run **one arm per process** — no `-rb_ab`, each run
measures the baseline and one arm, and the verdict is the in-process in-run ratio. Same knobs,
same constants (`-Split 388`, `-cs 40960`), ~6–8 min per arm instead of 30. Cost if wrong: the
readback A/B stays unmeasured (default kept, which the plan expects anyway), and cross-arm
comparisons rely on each arm's own baseline instead of one shared baseline.

## Drafting works on the production model: +35 % median (3.68, measured 2026-10-29)

Two-arm run (`mtp-dnt3-368`, baseline + MTP `dnt 3`, split 388, `-cs 40960`, 1457 s, floors clean:
min VRAM 2573 MB, min RAM 2139 MB):

| category | baseline | MTP | ratio | acceptance |
|---|---|---|---|---|
| Agentic, code | 33.84 | 38.53 | 1.14× | 1.87/4 |
| Agentic, curl | 33.57 | 41.41 | 1.23× | 1.78/4 |
| Coding | 32.02 | 41.34 | 1.29× | 2.21/4 |
| Creative | 28.95 | 49.74 | 1.72× | 1.19/4 |
| Creative (reasoning) | **6.99** | 49.58 | **7.09×** | 1.33/4 |
| Translation | 31.27 | 43.77 | 1.40× | 1.64/4 |
| Translation (reasoning) | 37.00 | 46.99 | 1.27× | 1.52/4 |
| Trivial repetition | 33.62 | 43.02 | 1.28× | 2.40/4 |

- **Median ratio 1.35×** (+35 % tg). Acceptance 1.2–2.4 of 4 — modest per draft, but the window is
  cheap enough to pay.
- **Outlier flagged, not used:** "Creative (reasoning)" baseline ran at 6.99 t/s where every other
  baseline sits at 29–37 t/s, so its 7.09× is a baseline pathology (the drafting arm was normal at
  49.58). Excluding it, mean baseline 32.9 → MTP 43.6 t/s = **+33 %**, consistent with the median.
- This is the opposite verdict from the screening model (break-even there, 1.91 bpw body), which is
  why the drafting series moved to the production model. The 1.66 result stands as the reason, not
  as a measurement of the knobs.

**The series continues on the swapped-in 3.18bpw model** (operator decision, same day): the 3.68
numbers above establish that drafting is worth measuring; the knob verdicts must come from the
model that will be deployed.
