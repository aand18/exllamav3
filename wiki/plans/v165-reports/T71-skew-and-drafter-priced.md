# T71 — Argon's two remaining factors, priced: both are negative. Campaign stops.

Argon's second review named exactly two factors it thought were still open, and asked for one free
check before stopping. Both are now priced from data already on disk. Neither survives.

## 1. Per-layer split skew (its Proposal 3b) — +0.8 % at best, so below its own stop rule

The skew is real: mean distinct CPU experts at rows=5 ranges **14.1 (L39) to 31.9 (L0)**, mean 20.15
over 48 layers. But the slot budget is fixed at 6 288 GPU slots, so helping the neediest layers means
starving the least needy, and the proportional model prices the exchange:

| slots/layer moved between the 5 neediest and 5 least-needy | net misses/layer | µs/round | % of a round |
|---|---:|---:|---:|
| 20 | +0.07 | +125 | +0.3 % |
| 40 | +0.14 | +249 | +0.5 % |
| **60** | **+0.21** | +374 | **+0.8 %** |
| 100 | +0.35 | +623 | +1.3 % |

(The sign is "more misses and more time" — moving slots to the neediest layers is the wrong direction
in this model because the elasticities nearly cancel; at best a targeted allocation buys **0.8 %**.)

Argon's stated threshold was **≥3 misses/layer (≥0.11 ms/layer)** to be worth pursuing. The measured
ceiling is **0.2 misses/layer**. **Fails its own gate → stop all MoE work.**

## 2. BF16 drafter (its second factor) — the acceptance gain is priced in CPU experts

The BF16 drafter accepts **2.07** against the exl3's **1.88** (T61/T63) — **+6.6 % tokens per round**
for unchanged `cmb`. But it does not fit at 262 144 with split 381: measured today, that combination
refuses to load in 35 s. It needs roughly 6–8 more CPU experts per layer, and each costs 37 µs × 48:

| split needed | round time | token gain | **net** |
|---|---:|---:|---:|
| 387 (+6 CPU experts) | +23 % | +6.6 % | **−16.1 %** |
| 389 (+8) | +30 % | +6.6 % | **−23.6 %** |

The drafter's acceptance win is spent many times over buying the VRAM back in CPU experts. **Negative
by a wide margin**, and that is before the exl3 drafter's own agentic-curl penalty is recovered.

## The conclusion, with the whole basis in one place

`tok/s = (1 + acc) / Σ_layers t_layer`, and both terms are now measured and closed:

- **`t_layer`** = `cmb` (0.68 ms = 20.5 distinct CPU experts × 33.9 µs) + GPU work (0.27 ms). The
  µs/expert law is measured over 1 014 545 layer-rounds (T66). Its two factors are closed: experts
  per layer (T62 +3, T69 +8, T66 placement, T67 pinning — four attempts, all zero) and experts per
  token (T70, `-ndt 3` withdraws at Agentic-code −5.82 %).
- **`acc`** is closed by price: the better drafter costs more CPU experts than its acceptance is
  worth (T71 §2), and the window that would amortise it is the wrong size (T70).

**Stop.** The remaining options are outside this campaign's reach: a 2 bpw-class expert
representation (one K per layer in the engine; operator decision) or a different engine. Recorded in
`v165-STATE.md`.

## Evidence base

Per-layer skew: `t70A-ndt4-262k-20261111-055730`, `t70D-ndt4-262k-20261111-061433`,
`t70E-ndt4-262k-20261111-070757` (distinct counters, rows=5 only, 48 layers).

BF16 pricing: acceptance 2.07 vs 1.88 from `t61A-bf16-131k-20261110-065833` /
`t61D-bf16-131k-20261110-074755` against `t61B-exl3-131k-20261110-071514` /
`t61C-exl3-131k-20261110-073132`; the split cost from the µs/expert law (T66) and the load refusal
measured in T68's first attempt.

No new run was made for this report: it prices two mechanisms from existing CSVs, which is why the
post gate finds no run of its own.
