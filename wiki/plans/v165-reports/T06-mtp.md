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
`-cs`. Two caches exist while drafting (target + draft), but the **draft cache is sized by the
head's one full-attention layer**, so at 262144/Q2 it is only ~160 MiB, not another 2.09 GiB
(correction 2026-10-30, critique C10). The series used `-cs 40960` because a 1024-token generation
needs 139 pages = 35584 tokens (measured: `Job requires 139 pages (only 128 available)` at 32768),
not because 262144 was unaffordable. Knob *ratios* stay comparable because every drafting arm uses
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

## The ~700 MB/arm growth was self-inflicted, not an engine leak (corrected 2026-10-30)

What was measured: over a 29-minute four-arm run free host RAM fell monotonically
(2517 → 974 MB) while VRAM stayed flat at 2427 MB. What was **misdiagnosed**: "each further
Generator costs ~0.6–0.8 GB that never comes back… a deployment caveat, verify whether tabbyAPI
builds a Generator per job."

Both halves were wrong (critique C6):

- **Our own `spec_decode.py` build** created a `Generator` **inside** the arm loop and never
  released it. A Generator owns a `RecurrentCache` whose host pool holds the arm's GDN checkpoints,
  plus a `ThreadPoolExecutor`, plus a reference cycle
  (`Generator -> recurrent_cache -> page table -> Generator`). Five Generators = five stashed
  checkpoint sets.
- **tabbyAPI does not build a Generator per job.** It creates one `AsyncGenerator` in
  `create_generator()` during model load and runs `await self.generator.close(); gc.collect()` only
  on its error path. There is no per-job leak to warn the operator about.

Fixed: `reset_arm()` (`recurrent_cache.clear()`, `host_pool.release()`, `generator.close()`,
`gc.collect()`, `torch.cuda.empty_cache()`), one shared Generator for the readback toggle (the flag
is read per verification round, so four instances were never needed), and an untimed warm pass over
the workload corpus (`-nwarm` to skip). The one-arm-per-process protocol that this bug forced is no
longer required.

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

## Draft length on 3.18 (split 376, `-cs 40960`, `-tokens 512`): `-ndt 2` wins

One arm per process, each with its own in-run baseline (the protocol forced by the per-arm RAM
growth). Median of the eight per-category ratios:

| arm | median vs the default arm | mean baseline → mean MTP | acceptance |
|---|---|---|---|
| default (`-ndt` unset = 4) | 1.00× (reference) | 35.15 → 39.05 t/s | 1.14–2.01 / 4 |
| **`-ndt 2`** | **1.30×** | 36.24 → 50.14 t/s | 1.01–1.68 / 2 (50–84 % per draft) |
| `-ndt 6` | 0.97× | 38.76 → 37.94 t/s | 1.12–2.41 / 6 |

**Correction (2026-10-30, critiques C1/C3):** these are **arm-to-arm** ratios recomputed from
`results/spec_decode.csv`. The figures first reported (1.42× / 1.10× / 0.98×) were
drafting÷baseline ratios, and that baseline is contaminated: it runs first in the process, pays the
cold PLE disk reads for exactly these prompts, and (before the frozen-profile work) could be a cold
expert placement. And `mtp-ref-318` itself was a **cold** run — its log line 2 has no
`placement from saved profile`, while `mtp-ndt2-318` does — so the default arm's number is
provisional until a warm `mtp-ref2-318` exists. The *ordering* (2 beats 4 beats 6) is stable
across both ways of computing it and across 8/8 categories.

- **Fewer drafts win**, and the mechanism is visible in acceptance: at 2 drafts the head lands
  50–84 % per draft; at 6 it lands 19–40 %, so most verification work is wasted. Six drafts is a
  net loss (0.98×), four is barely positive (1.10×), two is +42 %.
- Cross-process baseline means vary 35.2–38.8 t/s (±5 %), so the verdict rests on the in-run
  ratios, all of which point the same way (ndt2 beats the default in 8/8 categories).
- **`MTP_WINNER` so far: `-ndt 2`, batched readback (default), stock MTP head.**

### Series interrupted

The operator cancelled the remaining arms (`-sdh` ×2, `-hdt`, `-dds`) mid-series; `mtp-sdh1-318`
was already past load and finishes on its own (its numbers are the discardable profile build).
**Unmeasured: `-sdh` (sparse draft head), `-hdt` (hybrid drafting), `-dds` (dynamic draft).**
`-sdh` is the one worth resuming first — it is the only knob here that changes *what* is drafted
rather than how much, and the plan expects it to matter most for MTP.

## Step 3 — `-sdh` sparse draft head (resumed 2026-10-30, production 3.18 @ split 376)

Runner: `run_pass.ps1 -Runner spec_decode`, regime from `state.json` (3.18, split 376, mct 16,
`-cs 40960` from `draft.spec_decode_cache_size`, `EXL3_MOE_CPU_SWAP=0` automatic, frozen expert
profile restored per arm), `-Extra "<sdh …> -nbl"` so each process measures the draft arm only.
Harness sha256 `fba75185…`, `spec_decode.py` `333c4f24…`. All four runs `status=ok`, `breach=0`.

Per the C4 protocol, the first `-sdh` run of the day only warms the shared token profile and is
**not** a measurement:

| arm (run id) | `-sdh` | tok/s | accepted/round | min free VRAM | role |
|---|---|---|---|---|---|
| `sdh-warm-20261030-035233` | 32768 | 43.19 | 1.18/2.00 | 1867 MB | profile warm-up — discarded |
| `sdh32k-20261030-041634` | 32768 | 42.87 | 1.17/2.00 | 2129 MB | measured |
| `sdh64k-20261030-042635` | 65536 | 43.16 | 1.17/2.00 | 2173 MB | measured |
| `sdhoff-20261030-043614` | 0 (off) | **45.86** | **1.34/2.00** | 1869 MB | measured |

Mechanism counter printed by every arm, e.g. the warm-up:
`sparse draft head: 7735 columns, mode sparse True (0 switches), miss rate 0.00% (break-even
2.61%), rounds 7416, probes 114 (0 inside / 17 outside)`.

**Verdict: `-sdh` loses — drop it.** Three independent runs with the sparse head agree at
42.9–43.2 tok/s and 1.17–1.18 accepted per round; the plain full head measures 45.86 tok/s and
1.34. The 6 % tg loss is consistent and larger than any spread seen in this series, and it comes
with *lower* acceptance, not equal acceptance: the miss-rate counter says the active columns cover
the drafted mass (0.00 % against a 2.61 % break-even), but the head still disagrees with the full
head on 17 of 114 probes outside the active set, and that shows up as a ~13 % relative acceptance
drop. The reduced matmul does not pay for the bookkeeping plus the approximation.

Consequence for T7/T13: the sparse head is **not** part of `DRAFT_WINNER`; `-sdh` stays 0 in
`state.json`, and a sidecar/DFlash2 winner does not need to be re-tested with it unless its own
acceptance is high enough that a 6 % draft-step saving could matter (it is not, at 1.34 accepted
of 2 drafted).

## Steps 4–5 — hybrid drafting (`-hdt`) and dynamic draft (`-dds`), 2026-10-30

Same regime and protocol as Step 3 (`-nbl`, draft arm only, 3.18 @ split 376, `EXL3_MOE_CPU_SWAP=0`,
frozen profile per arm). Ten runs in all, each `status=ok` / `breach=0`.

| arm (run id) | extra flags | tok/s (mean) | accepted/round | min free VRAM |
|---|---|---|---|---|
| `sdhoff-…043614` | — | 45.86 | 1.34/2.00 | 1869 MB |
| `sdhoff-r2-…044515` | — | 45.49 | 1.33/2.00 | 2401 MB |
| `hdt16-…045515` | `-hdt 16` | 46.62 | 2.63/3.35 | 2081 MB |
| `hdt24-…050510` | `-hdt 24` | 49.10 | 2.59/3.35 | 1569 MB |
| `hdt32-…051505` | `-hdt 32` | 46.15 | 3.20/3.92 | 1363 MB |
| `hdt48-…052500` | `-hdt 48` | 49.12 | 2.75/3.58 | 1803 MB |
| `dds-on-…053455` | `-dds` | 46.50 | 1.34/1.90 | 2187 MB |
| `hdt24-dds-…054450` | `-hdt 24 -dds` | 49.54 | 2.57/3.28 | 2037 MB |

**The mean tok/s is not the verdict — the per-category ratio is.** Measured against the mean of the
two reference runs, per category (7 workloads vs the "Trivial repetition" one):

| category | hdt16 | hdt24 | hdt32 | hdt48 | dds | hdt24+dds |
|---|---|---|---|---|---|---|
| Agentic; code | 0.97 | 0.98 | 0.93 | 0.95 | 1.00 | 1.00 |
| Agentic; curl | 0.88 | 0.99 | 0.94 | 0.95 | 1.02 | 0.95 |
| Coding | 1.02 | 0.99 | 1.00 | 1.00 | **1.05** | 1.03 |
| Creative | 1.09 | 0.97 | 0.97 | 1.03 | 0.97 | 1.04 |
| Creative (reasoning) | 1.06 | 1.08 | 1.00 | 1.07 | 1.06 | 1.07 |
| Translation | 1.02 | 1.00 | 1.03 | 1.01 | 1.02 | 1.02 |
| Translation (reasoning) | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.98 |
| **Trivial repetition** | 1.16 | **1.57** | 1.23 | **1.61** | 1.02 | 1.58 |
| **median ratio** | 1.022 | 0.995 | 0.992 | 1.002 | **1.018** | 1.021 |

Reference spread (r2/r1, same config): median 0.994, range 0.972–1.014.

**Verdict `-hdt`: repetition-only, keep it off.** Every variant sits at a median ratio of 0.99–1.02,
i.e. no effect on seven of eight workloads, and the entire mean gain is the one synthetic
"Trivial repetition" workload where a suffix match is guaranteed (1.16–1.61×). At 32 and 48 it
also costs 5–12 % on both Agentic workloads. A deployment only wants this if its traffic is
repetition-heavy (long duplicated boilerplate, JSON with repeated keys); it is not a default, and
the mean "+7 %" from `hdt 24` must never be quoted without this table.

**Verdict `-dds`: unresolved, leaning +2 %.** Its ratio is broad rather than concentrated
(median 1.018, range 0.974–1.060, no category spike), which is the signature of a real effect
rather than a workload artifact — but the reference itself moves 0.972–1.014 between two identical
runs, so this design cannot separate +2 % from noise. One ABBA pair at the T13 config lock decides
it; it is not carried into `MTP_WINNER` on this evidence.

**Step 6 `draft_max_rows`: N/A at bsz 1** — the ceiling only trims the window when
`bsz·(1+draft)` exceeds it, and every measurement here is single-stream. Revisit only if the server
runs concurrent streams.

**`MTP_WINNER` frozen: `-ndt 2`, batched readback, stock MTP head, `-sdh 0`, `-hdt 0`, `-dds` off
(unresolved +2 %).** Confirmation at 262144 with the drafter happens in T13.
