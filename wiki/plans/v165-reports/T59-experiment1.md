# T59 — Experiment 1: the roundprof occupancy read

Post-hoc analysis of the T58 `roundprofT58*.csv` / `.distinct.csv` pair (131072,
spec_decode, static, production regime, pin on). No new GPU run was needed; the brackets
already landed the data. Numbers are per-layer stream ms, medians unless written "mean" —
never mixed.

## The window in ms (rows = verify-window rows)

| rows | moe | sub | ker | cmb | moe − (sub+ker+cmb) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.500 | 0.040 | 0.079 | 0.357 | ~0.024 |
| 3 | 0.646 | 0.046 | 0.091 | 0.486 | ~0.023 |
| 4 | 0.796 | 0.050 | 0.096 | 0.631 | ~0.018 |
| 5 | 0.883 | 0.047 | 0.117 | 0.696 | ~0.023 |

At rows=5 (the drafted window), the MoE window median is 0.883 ms/layer and
**the wait (`cmb`) is 0.696 ms of it — 79 %**. `ker` is 0.117 ms, `sub` 0.047 ms, and the
unaccounted remainder is ~0.023 ms — no launch-gap layer to chase; the brackets leave no
that matches Experiment 2's pre-registered pass mark (moe median down ≥ 0.10/layer with
graph capture of the fused issue + collect pair: 2 launches/layer, one doorbell wait, one
fold).

The `ker` rows-scaling read stands: rows=1 median 0.061 ms → rows=5 0.117 ms, a 1.9× rise
for a 5× window, i.e. kernel time is mostly fixed cost per launch. That is the read the
plan pre-registered as pointing to Experiment 3 — and the grouped-kernel experiment has
since measured its verdict (rows=1 0.0914 ms > 0.061 pass mark; rows=5 0.0646 ms ≤ 0.082
mark; compaction 0.26 ms; **REJECT**).

## What this closes

- **Experiment 2 (one graph) is demoted to a falsifier, not a build:** the pass mark is
  small compared to the wait slice it can't address. Its only honest role now is measuring
  whether any of the 0.696 ms cmb slice is doorbell-serialization inside the stream rather
  than worker lateness. A cheaper probe is a graph-capture A/B at rows=5 only.
- **Experiment 4 (weight reuse) does not matter to the window:** ker is 13 % of the window
  at rows=5; even halving it buys 0.06 ms/layer.
- **The remaining lever is the exposed wait itself** (0.696 ms median at rows=5): helped
  only by a CPU-completion overlap mechanism that fires per round and is cheaper than the
  wait it hides. Per-round thread-group paging was rejected against that scale in T57,
  but a per-round "one shared wait" shape remains the mechanism family the numbers point
  at.

## Stopping point

Experiments 1 → 3 closed. The graph route is demoted. The next session's open item is the
per-round shared-wait overlap (not built; not pre-registered yet) or accepting the
0.68 ms as the exposed floor of the current design.

NOT MEASURED: an `nsys` gate-overlap trace (the brackets show no gap layer to explain;
a trace would confirm, not redirect). The 0.205 → 0.183 ms slots A/B stays the standing
`sub` number (median 0.049 ms).

Run ids, same-session: `t58A-slots4-131k-20261108-214726`, `t58B-slots8-131k-20261108-220339`,
`t58C-slots8-131k-20261108-221944`, `t58D-slots4-131k-20261108-223615`
(basis: T58, 131072 spec_decode).

Verified and committed (`e9aa18ee` pushed after `verify --post`).
