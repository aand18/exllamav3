# T70 — the draft window is not a lever: the 4th draft earns more than it costs

Pre-registered in `2026-11-11-T70-draft-window.md`. Basis: `spec_decode`, cache 262144 (production
width), split 381, exl3 drafter (the served one), static placement, pin 24 GiB, slots 4, `-dds` on so
the ceiling is the only variable.

## Runs

| arm | `-ndt` | run id | draft tok/s | accepted |
|---|---|---:|---:|---:|
| reference | 4 | `t70A-ndt4-262k-20261111-055730` | 57.01 | 1.85 |
| reference | 4 | `t70D-ndt4-262k-20261111-061433` | 56.76 | 1.92 |
| **treat** | **3** | `t70B-ndt3-262k-20261111-063421` | **56.98** | **1.64** |
| **treat** | **3** | `t70C-ndt3-262k-20261111-065103` | **56.95** | **1.64** |
| reference | 4 | `t70E-ndt4-262k-20261111-070757` | 57.29 | 1.88 |

Means: **57.02 (ndt 4) vs 56.97 (ndt 3)** — corpus ratio **0.9990**, i.e. flat.

Agentic-weighted primary: **71.72 → 69.27, ratio 0.9657 (−3.43 %)** against a 1.050 floor, with
**Agentic-code −5.82 %** (acceptance 2.81 → 2.25), Coding −2.54 %, Agentic-curl −0.87 %.
**Verdict: `WITHDRAW`.**

## The arm tested what it claimed

The `-dstats`/distinct distribution confirms the window moved: **ndt 3 ran 100 % of its drafted
rounds at 4 rows** (41 378 of 41 378, no 5-row rounds at all), ndt 4 ran 44 548 at 5 rows plus a
`-dds`-truncated tail at 4 and 3. And the cost fell exactly as the model predicted:

| window | distinct CPU experts | `cmb` | `moe` |
|---|---:|---:|---:|
| rows 5 (ndt 4) | 20 | 0.677 | 0.863 |
| **rows 4 (ndt 3)** | **18** | **0.570** | **0.741** |

A 4-row round is **11–16 % cheaper** in the MoE window. It still loses, because it yields fewer
tokens: 2.64 versus 2.88 per round on the corpus and **3.25 versus 3.81 on Agentic-code**.

## Why the arithmetic was right and the lever still fails

The pre-registered break-even was `acc₃ > 1.73`. Measured corpus `acc₃` = **1.64** — below it, and the
sign of the result follows. On the category that matters most the gap is wider: dropping the fourth
draft costs **0.56 accepted tokens** on Agentic-code to save ~10 % of a round.

So the fourth draft position is the most valuable part of the window, not the marginal one. That is
the opposite of the intuition that a tail position is nearly free — it is where tool-call and code
structure gets predicted.

## What this closes

Both factors of the measured decode cost are now closed by measurement:

1. **Experts per layer** (20 at the drafted window): placement (T66), +3 experts (T62), +8 (T69),
   pinning 24→30 chunks (T67) — all zero on `cmb`.
2. **Experts per token** (this task): the window cannot shrink without losing more acceptance than it
   saves.

With `cmb` = 20 × 33.9 µs = 0.68 ms/layer of DDR5 read against 0.27 ms of GPU work, and both factors
frozen, **decode at this model, cache width and split is exhausted**: no configuration change, no
placement policy, no kernel and no drafting knob moves it. The remaining options are a different
expert representation (2 bpw class — operator decision, engine holds one K per layer) or a different
engine.
