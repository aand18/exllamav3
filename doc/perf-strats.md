# KVarN perf strategies (2026-10-02, ranked)

Goal remainder: pp 3.6s vs 2.8s Bee (engine GEMMs + O(n) seal math),
tg@128k 35.2 (O(n) serve). Scope: K4V4 only. Merge target: easy
upstream merge (atomic commits, twins, env-gated safe defaults,
fallbacks reproduce legacy).

## Shipped (reference)

- Seal-direct fast path: pp 5.2 -> 4.0s. Hierarchical serve: tg@64k
  34 -> 35.1. Decode graphs: tg 48.2 -> 57.8 @8k. Host trims.
- Rejected with reasons (do not re-propose without new evidence):
  MMA floor (neutral), fused tail-QK (SIMT vs MMA loss), inductor
  seal (inexact), flat-32 groups (-4%), tail v2 (+0.1%).

## 1. Async seals (PARKED 2026-10-02 -- prize too small for risk)

Audit (subagent A, stored `wiki/reports/2026-10-02-async-seal-audit.md`):
premise weakened -- inter-chunk get_kv DOES read sealed records
(dequant for attention), safe-under-lag only via staging-pin +
fences (copy_page TOCTOU, serve fence, pressure overflow to inline
sync at ~512 tokens). Refined design (rows-stash, no slot growth)
is sound but the prize re-estimated at +1-3% (streams hide device,
not host dispatch, which dominates) for medium risk (fences,
TOCTOU, pressure) + merge debt (new env var + fallback matrix).
PARKED: revisit only if stream infra exists for other reasons.
Superseded by: nothing (pp remainder stays fragmented).

## 2. Hand-fused seal Triton kernel (pp, +2-3%, high effort)

Sinkhorn (16 iters torch elementwise) + RTN + pack in 1-2 kernels
with EXACT torch op order (bit-identical by construction, twin
gated). Inductor failed (maxdiff 16.0); hand-written avoids that
by replicating order, not discovering it. Kills ~300 launches /
layer / chunk. Design queued (subagent B).

## 3. Prefill graphs, subregions (pp, +2-5%, medium effort)

Per feasibility study: graph WHT + vectorized store-scatter per
chunk size; seal core stays eager. Whole-chunk graph rejected
(dozens of shapes, pressure storms). Queued behind 1-2.

## Deliberately not pursued

- Speculative decoding (model feature, not cache).
- Eviction/windowing (breaks the exactness contract).
- 256-row groups (breaks v6 record format + Bee compat).
- K4V2 (scope call, 2026-10-01).
- tail-QK v2, tail fusion (traffic crumbs, <0.5%).

## Upstream-merge notes

Mergeable as-is: env-gated perf paths with safe defaults,
fallback == legacy behavior, twin per kernel change, atomic
one-cut commits, no API churn. NOT for upstream: eval spikes /
probes (`eval/_spike*`, `_probe*` stay local), PTIMES/DEBUG_HASH
hot-path instrumentation (gate or drop at merge time), `wiki/`
(process docs, keep on fork). Merge prep when wanted: rebase to
clean stack, drop scaffolding, PR summary with gate table.
