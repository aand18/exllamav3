| category | Baseline | MTP (rb=batched r1, greedy) | MTP (rb=serial r1, greedy) | MTP (rb=serial r2, greedy) | MTP (rb=batched r2, greedy) |
|---|---|---|---|---|---|
| Agentic, code | 47.05 t/s | 50.25 t/s, 1.07x, 1.74/4.00 acc/draft | 52.42 t/s, 1.11x, 1.84/4.00 acc/draft | 51.79 t/s, 1.10x, 1.82/4.00 acc/draft | 53.00 t/s, 1.13x, 1.92/4.00 acc/draft |
| Agentic, curl | 46.54 t/s | 48.19 t/s, 1.04x, 1.69/4.00 acc/draft | 45.39 t/s, 0.98x, 1.59/4.00 acc/draft | 44.01 t/s, 0.95x, 1.49/4.00 acc/draft | 46.37 t/s, 1.00x, 1.66/4.00 acc/draft |
| Coding | 52.31 t/s | 60.91 t/s, 1.16x, 2.03/4.00 acc/draft | 63.02 t/s, 1.20x, 2.21/4.00 acc/draft | 63.36 t/s, 1.21x, 2.14/4.00 acc/draft | 62.30 t/s, 1.19x, 2.01/4.00 acc/draft |
| Creative | 45.85 t/s | 38.85 t/s, 0.85x, 1.31/4.00 acc/draft | 41.89 t/s, 0.91x, 1.20/4.00 acc/draft | 44.65 t/s, 0.97x, 1.22/4.00 acc/draft | 46.03 t/s, 1.00x, 1.31/4.00 acc/draft |
| Creative (reasoning) | 43.85 t/s | 38.83 t/s, 0.89x, 1.06/4.00 acc/draft | 37.18 t/s, 0.85x, 0.94/4.00 acc/draft | 42.62 t/s, 0.97x, 1.16/4.00 acc/draft | 38.47 t/s, 0.88x, 0.93/4.00 acc/draft |
| Translation | 44.31 t/s | 46.16 t/s, 1.04x, 1.62/4.00 acc/draft | 49.10 t/s, 1.11x, 1.64/4.00 acc/draft | 51.93 t/s, 1.17x, 1.64/4.00 acc/draft | 51.30 t/s, 1.16x, 1.65/4.00 acc/draft |
| Translation (reasoning) | 43.62 t/s | 45.42 t/s, 1.04x, 1.40/4.00 acc/draft | 51.67 t/s, 1.18x, 1.67/4.00 acc/draft | 47.02 t/s, 1.08x, 1.40/4.00 acc/draft | 49.03 t/s, 1.12x, 1.51/4.00 acc/draft |
| Trivial repetition | 42.56 t/s | 23.90 t/s, 0.56x, 2.25/4.00 acc/draft | 41.87 t/s, 0.98x, 1.51/4.00 acc/draft | 40.91 t/s, 0.96x, 1.32/4.00 acc/draft | 51.03 t/s, 1.20x, 1.78/4.00 acc/draft |

| arm | mean t/s | vs baseline |
|---|---|---|
| Baseline | 45.76 | 1.000 |
| MTP (rb=batched r1, greedy) | 44.06 | 0.963 |
| MTP (rb=serial r1, greedy) | 47.82 | 1.045 |
| MTP (rb=serial r2, greedy) | 48.29 | 1.055 |
| MTP (rb=batched r2, greedy) | 49.69 | 1.086 |
