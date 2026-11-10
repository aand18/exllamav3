# T63 — the per-domain table: drafting pays least on Agentic-curl, and both recent arms cost it

Written after T61/T62 were reported on *aggregate* tok/s. The aggregate hid the sign: the campaign's
own rule (`bench/spec_primary.py`) weights **Agentic; code ×5, Agentic; curl ×4, Coding ×3**, and on
that weighting both arms are regressions on the domain that matters for agentic traffic.

Basis: `spec_decode`, 131 072 decode, static placement, pin on, `-cq 4,4`, `-mct 16`, `-ambs 2`,
`-ndt 4 -dds`. **Two cache allocations, reported in separate blocks — never pooled:** 40960 (the
`spec_decode` measurement cache, T61's ABBA) and 262144 (the production cache, T62's split arm).

## Block A — cache 40960, BF16 drafter vs exl3 drafter (T61)

| category | BF16 t/s | exl3 t/s | Δ | BF16 acc | exl3 acc | drafting gain |
|---|---:|---:|---:|---:|---:|---|
| **Agentic; code** | 70.09 | 72.00 | **+2.73 %** | 2.83 | 2.84 | 1.65× → 1.65× |
| **Agentic; curl** | 69.06 | 65.09 | **−5.74 %** | 2.67 | **2.19** | **1.63× → 1.51×** |
| **Coding** | 77.16 | 80.20 | **+3.95 %** | 2.64 | 2.44 | 1.57× → 1.59× |
| Creative | 46.33 | 49.78 | +7.45 % | 1.63 | 1.46 | 1.28× → 1.36× |
| Creative (reasoning) | 43.27 | 42.44 | −1.92 % | 1.47 | 1.08 | 1.16× → 1.14× |
| Translation | 43.83 | 45.03 | +2.76 % | 1.27 | 1.02 | 1.20× → 1.25× |
| Translation (reasoning) | 44.25 | 44.79 | +1.22 % | 1.29 | 1.02 | 1.18× → 1.20× |
| Trivial repetition | 62.54 | 67.70 | +8.27 % | 2.74 | 3.02 | 1.64× → 1.76× |

`bench/spec_primary.py` on this pair: **primary 71.51 → 71.75, ratio 1.0033 (+0.33 %) against a
1.024 floor; Agentic-curl −5.74 % with acceptance 2.67 → 2.19; verdict `WITHDRAW`.**

## Block B — cache 262144, split 381 vs split 378, exl3 drafter (T62)

| category | 381 t/s | 378 t/s | Δ | 381 acc | 378 acc | drafting gain |
|---|---:|---:|---:|---:|---:|---|
| **Agentic; code** | 74.12 | 74.37 | **+0.34 %** | 2.75 | 2.75 | 1.69× → 1.69× |
| **Agentic; curl** | 65.78 | 61.17 | **−7.01 %** | 2.09 | 1.88 | **1.55× → 1.39×** |
| **Coding** | 81.38 | 83.05 | **+2.05 %** | 2.56 | 2.40 | 1.65× → 1.68× |
| Creative | 46.91 | 47.01 | +0.21 % | 1.34 | 1.52 | 1.27× → 1.22× |
| Creative (reasoning) | 43.58 | 45.53 | +4.47 % | 1.11 | 1.43 | 1.18× → 1.20× |
| Translation | 44.97 | 45.09 | +0.26 % | 1.04 | 1.03 | 1.25× → 1.19× |
| Translation (reasoning) | 43.41 | 44.32 | +2.10 % | 1.00 | 1.01 | 1.15× → 1.17× |
| Trivial repetition | 68.37 | 68.25 | −0.18 % | 3.08 | 2.87 | 1.82× → 1.79× |

Agentic-weighted primary: **0.9862 (−1.38 %)**. `spec_primary.py` refuses a formal verdict (one
run on the 378 arm, so no spread), and the control itself moves 6.5 % on Agentic-curl between its
two runs (67.85 vs 63.71) — so the −7.01 % is **inside that category's own control spread** and
the arm is **unresolved on curl**, not proven worse. It is certainly not better.

## What the table says, in one place

1. **The trunk's own speed differs by domain** (baseline, no drafting): Agentic-code 42.6–44.1,
   Coding 49.1–50.0, Translation 36.0–37.8, Creative 36.3–38.4 t/s. Drafting multiplies whatever
   the trunk gives, so an aggregate mean mixes workloads with a 1.4× spread in trunk speed.
2. **Drafting gains are domain-shaped, not uniform.** Best: Trivial repetition 1.64–1.82×, Agentic
   code 1.65–1.69×, Coding 1.57–1.68×. Worst: Creative-reasoning 1.16× down to 1.14×, Translation
   1.25× down to 1.15×, **Agentic-curl 1.63× down to 1.39×** (each range read off the two blocks
   above, gain columns).
3. **Agentic-curl is the fragile one.** It has the lowest acceptance of the three agentic-weighted
   categories (2.67 → 2.19 → 1.88 → 2.09 depending on drafter and split) and the largest response to
   both changes. For a coding agent, that workload is the tool-calling loop, i.e. the one that runs
   most often.
4. **Both recent arms spend their gain on the wrong side of the weighting.** exl3 drafter: +2.73 %
   agentic-code, **−5.74 % agentic-curl**. 378: +0.34 % agentic-code, **−7.01 % agentic-curl**.

## Consequences for the served configuration

- **The served drafter (exl3) is a `WITHDRAW` under the campaign's own agentic-weighted rule** at
  cache 40960. It was never weighed this way before; the acceptance cost (2.67 → 2.19 on curl) is
  what drives it, not tok/s.
- **Do not move to split 378.** Its only argument was freed VRAM, and on the agentic weighting it
  reads −1.38 % (unresolved, but with no upside anywhere near the ×5/×4/×3 categories).
- **If agentic traffic is the target, the drafter is the lever to re-examine** — a BF16 drafter costs
  ~0.75 GB but buys +0.48 accepted drafts on Agentic-curl. That trade was never measured against the
  split it would force; it needs the operator and one arm at 262144.

## Reporting rule this session should have followed

`spec_primary.py` exists for exactly this and takes two minutes: **quote the primary ratio and the
three agentic categories, then the corpus mean as a secondary line.** Aggregate corpus tok/s stays in
the report as a cross-check only.

## Runs

Block A (cache 40960): `t61A-bf16-131k-20261110-065833`, `t61D-bf16-131k-20261110-074755` (BF16);
`t61B-exl3-131k-20261110-071514`, `t61C-exl3-131k-20261110-073132` (exl3).

Block B (cache 262144): `t62A-381-262k-20261110-085520`, `t62D-381-262k-20261110-091303` (381);
`t62E-378-262k-20261110-093222` (378).

The split arms that refused to load (`t62B`, `t62C`, `t62F`, `t62G`) contribute no rows and are not
part of this table. All eight cited runs are `status=ok` with `breach=0`.
