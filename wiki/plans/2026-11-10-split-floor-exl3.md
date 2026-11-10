# T62 pre-registration — the split floor under the served (exl3) drafter

## Question

Production serves `Qwen3.8-Flash-Next-DFlash2-exl3` (472 MB) at **split 381**. The 381 floor was
founded 2026-11-02 with the **BF16** drafter (1.2 GB) resident. T61 measured the exl3 drafter
freeing **728–940 MB** of VRAM at the same split (floor 2687 → 3415/3627 MB). If that headroom
buys GPU experts, the static floor should move down.

## Regime

`-Runner spec_decode`, **`-CacheSize 262144`** (the production cache allocation — the floor is a
VRAM property of the cache, not of `max_length`), `-MaxLength 131072` (decode ladder), exl3
drafter via `_t4check/state-exl3.json`, pin on, `-cq 4,4`, `-mct 16`, `-ambs 2`, `-ndt 4 -dds`.
Arms differ only in `-Split`.

## Ladder

Geometry first (`bench/size_check.py`, exl3 drafter, cs 262144, Q4, slots 2, `fit_headroom_mb 512`):

| split | trunk VRAM | draft | TOTAL | free | verdict |
|---:|---:|---:|---:|---:|---|
| 381 | 17551 | 471 | 20997 | **1519** | FITS |
| 378 | 17814 | 471 | 21260 | 1256 | FITS |
| 375 | 18077 | 471 | 21523 | 993 | FITS |
| 372 | 18339 | 471 | 21786 | **730** | FITS |

One expert is **87.7 MiB** of trunk (381→378 costs 263 MiB for 3). The exl3 drafter frees **680 MiB**
at 381 versus the BF16 record (839 MiB free → 1519 MiB), i.e. ~7.7 experts; 372 is the lowest split
the 512 MiB headroom rule allows, so it is the extreme of the reachable range and the arm to test
first. The gate is geometry, the answer is the engine's own floor at load.

Arms (ABBA on the extreme, so a two-reference mean exists):

| arm | split | what it answers |
|---|---:|---|
| `t62A-381` | 381 | same-session reference (the served value) |
| `t62B-372` | 372 | the most GPU experts the headroom can hold |
| `t62C-372` | 372 | repeat |
| `t62D-381` | 381 | closing reference (ABBA integrity) |

If 372 refuses to load, the refusal fixes the floor from above and the ladder repeats one step
higher (378, then 375) — a refusal is a result, not a failure.

## Pre-registered verdict rule

- **Deployable** only if a lower split **loads, completes, and beats the pair of 381 references
  by ≥ +2.0 % draft tok/s** (the T14 coefficient prices an expert near the floor at ~1 %, so a
  real move needs ≥2 experts' worth).
- **Null** if the mean of the two 381 arms is within ±2.0 % of the best lower split: the freed
  VRAM is real but not spendable on experts at this cache width.
- **Refused** arms are recorded with their log line; they do not count as a win.
- Acceptance must stay within 0.08 of the 381 mean (the golden text differs per split, so small
  movement is expected; a large drop means the arm measured a different generation).

## Bases stated up front

- tok/s: drafted arm, `spec_decode` corpus mean over 8 categories, 131072 decode, cache allocated
  at 262144.
- Expert cost near the floor: ~1 % per expert, T14 dynamic ladder (not the static 0.56 %/expert,
  which is 5.6× too large on this box).
- Spread: identical-config arms differ up to 3.3 % of draft tok/s (T57), which is why the
  threshold is 2 % on a two-reference mean rather than a single pair.
