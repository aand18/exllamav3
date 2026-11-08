# T57 — Per-expert partial completion: correct, and 44 % slower

Implemented `EXL3_MOE_PARTIAL_COMPLETION` behind a default-off env flag and measured it as an ABBA.

## Runs

| arm | run id | baseline | draft |
|---|---|---:|---:|
| off | `t57A-partialoff-131k-20261108-122609` | 37.88 | **56.23** |
| on | `t58B-partialon-131k-20261108-124240` | 27.03 | **30.31** |
| on | `t58C-partialon-131k-20261108-130621` | 27.26 | **31.27** |
| off | `t57D-partialoff-131k-20261108-132948` | 37.40 | **54.44** |

Means: drafted **55.34 tok/s** off vs **30.79 tok/s** on (**−44.3 %**); baseline **37.64** vs **27.15** (**−27.9 %**). Draft acceptance is unchanged (2.05/2.83 and 2.01/2.77 off; 2.02/2.81 and 2.08/2.84 on), so this is a latency regression, not a yield regression.

`cmb` medians (roundprof CSVs, drafted rows): off **0.496 / 0.508 ms**; on **0.957 / 0.960 ms**. The pre-registered pass mark was `cmb` median < 0.20 ms and decode ≥ 60 tok/s. Both missed by a wide margin.

## What was implemented

One job per distinct CPU-routed expert, compact per-expert output segments, a per-ring-position `expert_done` bank, GPU `wait + fold` pairs in expert-ascending order, and one `consumed` write per slot. Ring grew 256 → 2048. Default off; prefill and whole-layer paths untouched.

Verification:
- `tests/moe_cpu/test_partial_completion.py`: **10 passed**
- `tests/kernels/moe/test_moe_split.py -k collect_add_expert`: **11 passed**
- `tests/kernels/moe_cpu_host/test_moe_worker.py -k partial`: **3 passed**
- `tests/kernels/moe/test_moe_split.py`: **109 passed**
- `tests/kernels/moe_cpu_host/test_moe_worker.py`: **16 passed**
- `tests/moe_cpu`: **172 passed, 41 skipped**
- Two wheel builds completed with `WHEEL-EXIT 0`.

Two correctness bugs were caught by tests and fixed before the A/B: per-expert token/output-row mapping, and a WDDM ring-pointer issue in the GPU announce kernel.

## Conclusion

**Do not ship.** The granularity is wrong for this batch size. A 5-row window yields ~40 tiny jobs per layer; paying a memop wait and a kernel launch per expert swamps the overlap it creates, and the worker dispatches those jobs sequentially anyway, so the makespan does not shrink. The failure is launch/dispatch overhead, not coverage or bytes.

A coarser grouping (2–4 expert groups per layer) would be a different experiment with a much smaller overhead, but it is not the mechanism that was specified, and the measured gap is large enough that I would not spend another build cycle on it without a new hypothesis. The box was restored to the production config and the original `.pyd` after the campaign.
