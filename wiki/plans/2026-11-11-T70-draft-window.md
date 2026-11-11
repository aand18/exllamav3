# T70 pre-registration — the draft window as a CPU-byte lever

## Why this is the last mechanism worth testing

Decode is `20 CPU experts/layer × 33.9 µs × 48 ≈ 0.69 ms/layer` of DDR5 read (T66/T69). Distinct CPU
experts scale with the verify window, measured at three independent sessions:

| rows | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| distinct experts (median) | 8 | 12 | 16 | 19 | **21** |

`rows = 1 + num_draft_tokens`. So a smaller draft window is **fewer CPU experts per round** — the one
lever that touches the measured cost that is not bytes-per-expert (2 bpw, operator decision) and not
expert placement (closed: T62, T66, T67, T69).

## The arithmetic the test must beat

Round time from the T58/T66 phase medians (× 48 layers + mixer ≈ 0.10 ms):

| window | moe/layer | round ≈ 48×(moe+mixer) | tokens/round at corpus acceptance |
|---|---:|---:|---|
| ndt 4 (rows 5) | 0.883 | 47.2 ms | 1 + 2.0 = 3.0 |
| ndt 3 (rows 4) | 0.796 | 43.0 ms | 1 + acc₃ |
| ndt 2 (rows 3) | 0.646 | 35.8 ms | 1 + acc₂ |

Break-even for `ndt 3` against `ndt 4`: `1 + acc₃ > 3.0 × 43.0/47.2 = 2.73`, i.e. **acc₃ > 1.73**.
Acceptance decays with position, so the fourth draft is worth ~0.4–0.5 accepted tokens; acc₃ ≈ 1.5–1.6
would make this **lose**. The measurement decides it; the arithmetic says it is close enough to be worth
one campaign and no closer.

## Design

`-Runner spec_decode`, cache 262144 (production width), split 381, exl3 drafter, `EXL3_MOE_CPU_SWAP=0`,
pin 24 GiB, slots 4, `-dds` retained (the deployed setting) so the ceiling is the only variable.
`-dstats` writes per-round (position, window, accepted) so the row distribution is measured, not inferred.

| arm | `-ndt` | role |
|---|---|---|
| `t70A` | 4 | reference (production) |
| `t70B` | 3 | the lever |
| `t70C` | 3 | repeat |
| `t70D` | 4 | closing reference |

## Pre-registered verdict

- **Deployable** if the agentic-weighted primary (`spec_primary.py`) is **≥ +2.0 %** AND the `-dstats`
  rows show the window actually moving to 4 rows for the majority of drafted rounds (otherwise the arm
  did not test what it claims).
- **Null/withdraw** otherwise — and specifically withdraw if `acc₃ < 1.73`, which the arithmetic
  predicts.
- Acceptance must stay within 0.08 of the reference; a larger drop means the smaller window changed
  the generation, not the cost structure.
- A floor breach or missing arm rows is `fail`, not a result.

## What a null closes

If `ndt 3` loses, then **both factors of the DRAM cost are closed** — experts per layer (four
independent attempts) and experts per token (this one) — and the campaign's remaining decode work is
exhausted at this model, cache width and split. The next honest step would be a different
representation (2 bpw class) or a different engine, both operator decisions.
