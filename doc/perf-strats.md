# KVarN perf strategies (2026-10-02, ranked)

Goal remainder: pp 3.6s vs 2.8s Bee (engine GEMMs + O(n) seal math),
tg@128k 35.2 (O(n) serve). Scope: K4V4 only. Merge target: easy
upstream merge (atomic commits, twins, env-gated safe defaults,
fallbacks reproduce legacy).

## Shipped (reference)

- Seal-direct fast path: pp 5.2 -> 4.0s. Hierarchical serve: tg@64k
  34 -> 35.1. Decode graphs: tg 48.2 -> 57.8 @8k. Host trims.
- Shipped neutral, kept as structural unlocks (2026-10-02/03):
  combine WHT-split (`90074b7`, 47.5 vs 47.8), n-mirror (`c50905e`,
  47.9 vs 47.8), longskip (`633660b`, 47.8 vs 47.9). All green
  (twin/KLD/PARITY), all ~0%: the step is dispatch-throughput-bound,
  not sync- or conversion-bound (see tg wall below).
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

## tg@64k wall (2026-10-03, measured)

tg@64k graph 47.9 (21ms/step). Online-path Kineto, 5 steps: device
~16.4ms (MoE exl3 GEMV/GEMM ~8ms inherent, serve 4.5ms, tail 0.8ms,
combine 0.28ms), host cpu_op wall 23.4ms = the full wall (~3000
aten calls/step, no fat host function, syncs are cheap polls).
Device ceiling with a free host: 16.4ms = ~61 tok/s. Three
host/dispatch cuts measured neutral (above): piecemeal is done.

## 4. Tail + merge fusion (tg device, +0.5-1%, S effort)

Tail-reduce (0.8ms) + torch gap + merge in one kernel on the `(qh,)`
grid; bmm stays torch (per `2026-09-30-fused-decode.md:24`). Kills
1 launch/layer + ~50KB/layer round-trips. Do first: smallest,
stacks with the split combine (stable merge input layout).

## 5. Sample-in-graph (tg host, ~+0.5ms, M effort)

Argmax/sample inside the captured graph; host receives only the
finished token. Kills the terminal step-boundary serialization
(sampler is already 1-sync; this removes the orchestration around
it). Needs sampler + capture interlock design.

## 6. Whole-layer / whole-step graphs or torch.compile (tg host,
biggest prize, L effort)

The only attack on the actual bottleneck (dispatch throughput):
capture the entire layer (attn + MLP + norms + recurrent) or the
whole model step so ~3000 dispatches collapse into replays.
Scope: all 48 layers incl. GDN/recurrent + MoE routing, static
shapes/addresses, control flow (seal/evict/pressure fallbacks stay
eager). Risk medium-high (staleness bugs; replay itself is
bit-exact, failures mostly loud). Prize: the 5-7ms host gap,
i.e. the road from 48 toward 61 tok/s.

## 7. Serve traffic/compute surgery (tg device 4.5ms, HIGH RISK)

Serve reads + dequantizes all 64k KV/layer/step at ~220GB/s
effective. Cutting traffic/compute is the biggest device prize,
but perf can go BACKWARDS: flat-32 lesson (-4% from CTA
underfill). Requires occupancy-preserving redesign + a Kineto
iteration loop per attempt. Attempt only with dedicated box time;
do not interleave with 4-6.

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
