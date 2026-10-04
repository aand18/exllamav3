# tg@64k serve-path audit (2026-10-02, read-only)

Branch `wip/kvarn-cache` in `/home/dev/exllamav3/.worktrees/kvarn-cache`, HEAD `5df2190`
(post-graphs-v2, protocol v3). Scope: KVarN decode serve path at 64k, K4V4 only.
Method: code read only, no GPU runs. Builds on (does not re-derive):
hierarchical cap-128 shipped (`3647f54`), graphs-v2 sublattice replay
(`wiki/patterns/cuda-graphs-decode.md`), Kineto @64k serve/combine/tail split.

## Context numbers (evidence, not new measurement)

- `doc/kvarn-4090.md:218-221`: tg kvarn4 48.2 eager / 55.8 graph @8k,
  46.8 / 54.0 @16k, 43.4 / 47.0 @64k vs Bee flat 44.0. 131072: 35.2 graph
  (first measurement, Bee unknown).
- `doc/kvarn-4090.md:430`: slope 64k->128k 47.0 -> 35.2 (-25% for 2x ctx, O(n) serve).
- `doc/kvarn-4090.md:609-611`: Decode-step Kineto @64k (26ms wall, 17ms device):
  serve 4.0ms (23%) + combine 1.2ms (7%, serial NB loop) + tail 0.8ms;
  ~9ms wall is bubbles (host gaps/syncs/allocator), not kernels.
- `doc/kvarn-4090.md:340-342`: Kineto @8k (5 steps): kvarn kernels 1.9ms
  (serve 0.74 + tail 0.78 + combine 0.16 + wht 0.23); host ~19ms.
- `doc/perf-strats.md:3-4`: goal remainder tg@128k 35.2 (O(n) serve).
- Graphs gain shrinks with ctx: +16% @8k, +15% @16k, +8% @64k
  (from the table above) — host share shrinks, device O(n) dominates at 64k.
  Graphs did not change device work, only dispatch.

## What the graph replay covers vs leaves eager

Covered (replayed kernels, `dispatch.py:378-392` in `_graph_capture`):
`qwht` (`:379`) -> `serve` (serve kernel + combine, `:380-384`) ->
`tail_gather` (`:385-387`) -> `bmm` + `mul` (`:388-389`) ->
`tail_reduce` (`:390`) -> `merge` (`:391`). Static bufs + precomputed views
(`_graph_bufs` `:187-213`, `_graph_tail_bufs` `:216-233`); twin bar maxabs 0.0.

Left eager per step (`_try_kvarn_graph_decode` `:236-341`):
- store `update_kv_direct` `:257` (WHT 1 + store_row 1 launch, 1 `status.tolist`
  sync — `kvarn_triton.py:502`) runs before the bucket check, always eager.
- Static-input copies `:293-299` (`Qbuf/Qfbuf/nbuf/tpos/btbuf` `copy_` + 2 aranges),
  bucket lookup `(gc,R)` `:303`, buffer-identity verify (`is`-check `:318-324`),
  `replay()` `:325`, post-replay flag due read every 128 `:329-335`.
- Fallbacks: status 1/2, cert fail `:277-279`, miss/bad bucket `:307-313`,
  shape change `:274-276`, any capture throw `:337-341`. Seal path untouched.

Consequence: under replay the ~9ms bubble pool is *reduced* (8 launches +
~40 dispatches + 2-3 syncs/layer collapse to 1 replay + store), but device
time of serve + combine + tail + merge is *identical*. The next +5-10% must
come from device work or the one remaining per-layer sync (store status),
not more host trimming.

## Current serve + combine + tail + merge (all refs HEAD)

Serve kernel `_kvarn_online_serve_kernel` (`kvarn_triton.py:1531-1765`):
grid `(kvh, groups)` (`:1858`), `num_warps=4, num_stages=1` (`:1869`).
Each program covers CPG chunks x 128 rows with one online state
(`:1565-1571`, `for t0 in range(8*CPG)`, TOK=16 `:1564`). Hierarchical
`_kvarn_serve_groups` (`:1768-1775`): `gc if gc<=64 else min(gc,128)`;
at 64k gc~512 -> groups=128, CPG=4, ~512 CTAs (kvh=4 class). Includes
uniform-group metadata fast path (`:1592-1605`, `:1635-1649`, `:1724-1738`),
nibble K4 / quad K4V2 paths (`:1609-1620`, `:1690-1708`), exact-direct tail
select (`:1651-1658`, `:1743-1746`), MMA QK `tl.dot` (`:1666`) + transposed EV
(`:1750-1752`), sticky flag (`:1660-1661`). Partials strided by GROUPS
(`:1759-1765`), buffers `_ov_serve_m/l` `(kvh,qpad,groups)` fp32 +
`_ov_serve_acc` `(kvh,qpad,groups,hd)` fp32 (`:1826-1832`) — O(n) capped at
128 (was gc): 4x less partials traffic at 64k, occupancy preserved.

Combine `_kvarn_online_combine_kernel` (`:1017-1074`): one program per q-head
(`:1029-1031`), grid `(qh,)` (`:1872-1874`), `num_warps=1` *required* (`:1028`,
`:1874`) because the out-WHT is folded in (`:1058-1074`, via `_fwht128_block`
`:62-82` which is single-warp-only on sm_89/triton 3.8). Serial
`for b in tl.range(NB)` (`:1046-1051`) with one-hot `where(nboff==b)` selects
(`:1048`) + per-b `acc` load (`:1049-1050`); NB bound tightened (skip NBPAD
tail, `:1042-1045`). NB=groups=128 at 64k (was gc=512 pre-hierarchical).
m/l reload O(n) floats/head (`:1035-1037`); den==0 NaN guard (`:1055`).

Merge `_kvarn_online_merge_kernel` (`:1077-1119`), launcher `:1122-1149`:
one program per q-head, `num_warps=4` (`:1148`), GC-padded loads
(`:1101-1106`), body-stats recompute (`:1107-1108`) + tail merge
(`:1110-1118`). Reads O(GC) m/l per head (tiny: ~128 floats), un-normalizes
`out_b` by den with NO extra WHT (`cacd7af` fix, `:1126-1128`).

Tail gather (`:1236-1278`, launcher `:1281-1313`, grid `(R,kvh)` `:1310-1312`)
+ tail reduce (`:1153-1197`, launcher `:1200-1232`, grid `(kvh*qpk)` `:1229-1231`,
`num_warps=4`): serial `for r in tl.range(R)` (`:1193-1196`) but R O(1)
(`sink+tail_eff`, bucket key `:268-269`); exrev-partitioned mask in-kernel
(`:1183-1185`); bmm `Qh@Kt` + `mul scale` stay torch (`dispatch.py:388-389`,
`562-563`) with persistent `st` R-keyed cache (`:552-561`).

QWHT (`:1317-1364`, launcher `:1399-1419`, grid `(qh,)` `:1417-1418`,
`num_warps=1`): already fused convert+WHT, persistent bufs
(`_kvarn_online_buffers` `:1368-1396`, mirrored `_ov_serve_*`).

## Is the serial combine still the top item under replay?

No — serve device is the top *cost* (4.0ms vs 1.2ms vs 0.8ms @64k), but
combine is the top *leverage*. Reasons:
1. Combine's 1.2ms is 7% of wall pre-graph; after bubbles shrink under replay
   its share of the replayed sublattice rises (~10-12% of device sublattice).
2. `num_warps=1` infects the whole reduce: the serial loop cannot tensorize
   and the folded WHT forces 1 warp on memory-bound reduce work.
3. O(n) scaling is unchanged by hierarchical: NB 512->128 helped (loop iters
   4x fewer) but `for b in range(NB)` + m/l reloads still grow with ctx
   (`doc/kvarn-4090.md:568-569`: 97us@8k -> ~1.5ms/layer@128k pre-hierarchical
   trajectory; hierarchical divides by ~4, slope remains).
4. Serve itself is parallelism-healthy now (512 CTAs saturate 144 SMs);
   further traffic cuts without occupancy loss require combine-side changes
   (fewer partials -> fewer CTAs -> underfill, the flat-32 lesson).

So: fix combine structure first (unlocks parallel reduce + WHT split),
then serve traffic/compute, then the store-sync remainder.

## Ranked candidates (gain x effort x merge-risk; K4V4, one cut/commit, twin each)

| # | Cut | Exp. gain @64k | Effort | Merge-risk | Why this rank |
|---|-----|----------------|--------|------------|---------------|
| 1 | Combine WHT-split + parallel reduce | +3-5% tg | M (1 kernel -> 2 launches, `_kvarn_online_combine_kernel:1017-1074` split) | Low (domain unchanged; fp32 assoc noise only, allclose-gated; twin: existing combine twin + WHT twin compose) | Removes the single-warp serial bottleneck without touching serve grid/CPG/occupancy. Reduce phase goes `num_warps=4` + tree/one-hot-free gather; WHT reuses proven `_kvarn_wht_hd_kernel`. Extra pass over `(qh,hd)` ~16KB/head-class is noise vs 8MB acc traffic. Fits one-cut + easy-merge (new kernel, env-gated fallback to old combine). |
| 2 | Tail_reduce + merge fusion | +1-2% tg | S (same `(qh,)` grid; `tail_num`+`out_b`+stats stay on-chip) | Low (blockers enumerated in `2026-09-30-fused-decode.md:24`: bmm stays torch, exrev mask + bf16 path move in-kernel; PARITY twin asserts on `tail_m/den/num` intermediates — keep them as debug-only outs) | Easiest fusion on the board; kills 1 launch + ~50KB/layer round-trips + torch gap. Small alone, stacks with #1 (both end at `(qh,hd)` fp16 out). Do after #1 so merge input layout is stable. |
| 3 | Serve + combine fusion (acc never hits DRAM) | +5-10% tg *if* done without occupancy loss | L | High (grid mismatch `(kvh,groups)` vs `(qh,)` per `2026-09-30-fused-decode.md:23`; warp mismatch 4 vs 1; flag sync in between; CPG/hierarchical interplay; large-kernel review burden) | Biggest prize (~8MB@64k R+W/layer, `2026-09-30-fused-decode.md:20`), but violates one-cut unless staged (#1 first, then fuse reduce into serve epilogue per-subgroup). Do NOT attempt as a single commit. Stage: #1 -> per-CTA partial-WHT -> cross-CTA single reduce launch. Needs fresh Kineto before/after @64k/128k + 128k VRAM check (`_ov_serve_acc` ~65MB/layer@128k pre-cap, `:674-675`). |
| 4 | Store `status.tolist` defer / device-predicate | +1-3% tg (host; larger when graphs expose it) | M | Medium (status drives 0/2/1 control flow `kvarn_triton.py:482-483,502`, cannot speculate blindly; seal must not lag — needs KLD + needle, not just twins) | Only per-layer DtoH left on the replayed path (flag already periodic `:104-114`, `sync_flag=False` `:384`; store status still syncs every layer every step). Options: 1-step-delayed read (Spec A pattern, keep sync until green) or device-side predicate + eager-check every K steps. Measure first: PTIMES-gated sync counter to confirm share under replay. |
| 5 | Serve dequant/MMA micro-work | ~0% (proven) | — | — | Do NOT chase: MMA floor neutral + reverted (`doc/kvarn-4090.md:310-318`, `2a0b382`); loop-invariant hoisting evaluated and SKIPPED (loads not dominant `:612-613`); nibble/quad + uniform paths already shipped (`:1609-1620,:1690-1708,:1592-1605`). Any new proposal needs a micro-A/B showing dots/loads bind (they currently do not). |
| 6 | Tighter hierarchical cap (128 -> 64/32) | negative (proven) | — | — | Do NOT do: flat-32 lost 4% (`doc/kvarn-4090.md:259-264`, `doc/perf-strats.md:14`); serve is parallelism-bound, not traffic-bound. 128 = 512 CTAs saturates 144 SMs; fewer groups underfills. Revisit only with a new occupancy model + hot-cache A/B. |

Notes on effort scale: S = single kernel + twin, 1 commit. M = kernel +
launcher + twin + KLD-8k/64k. L = multi-kernel restructure + Kineto + needle.

What "tensorize/parallelize the combine" concretely means:
- Split `:1025-1057` (reduce: m/l/acc -> `row` fp32, `num_warps=4`, parallel
  tree or blocked gather replacing the `where(nboff==b)` one-hot `:1048`)
  from `:1058-1074` (WHT: `row` -> original domain, `num_warps=1`, call the
  existing head kernel instead of inlining). Two launches, both small-grid.
- Bee reference (`doc/kvarn-4090.md:679-685`): fixed-64 splits + parallel
  reduce + 3 launches + zero syncs. Our analog after #1: fixed subgroup
  count (already 128) + parallel reduce + WHT-only second launch. Do not copy
  Bee's 64-token split literally — our CPG=4 chunks are 512 rows/super-chunk;
  the reduce granularity is subgroups (128), not tokens.
- Keep `den==0 -> 0.0` (`:1055`) and `-inf/0.0` padding (`:1035-1037`) verbatim;
  they are the NaN-proof contract the merge kernel mirrors (`:1093-1096`).

## Dead ends (do not re-propose without new evidence)

- Flat-32 groups: -4% (`8e500b9`, `doc/kvarn-4090.md:259-264`). Cause:
  128 CTAs underfill 144 SMs. Cap stays 128.
- Split-parallel S=2048 serve (`3100ca3`, reverted `312eefb`): slower at 8k
  (31.3 vs 43.6) and 64k (30.6 vs 36.1), KLD-identical. Lesson: serve is
  throughput-bound on total work, not parallelization-starved
  (`doc/kvarn-4090.md:623-632`).
- MMA floor QPAD>=8 (`561588c`, reverted `2a0b382`): twins green but hot-cache
  neutral (K4V4 39.4->39.4), doubled serve-acc VRAM. Dots are not the bind.
- Fused tail-QK / tail v2: SIMT-vs-MMA loss / +0.1% (`doc/perf-strats.md:13-14`).
- Inductor seal: maxdiff 16.0 inexact (queued hand-fused kernel instead,
  `doc/perf-strats.md:29-35`).
- Whole-step / whole-chunk graphs: infeasible (data-dependent branches;
  dozens of shapes, pressure storms). Per-layer sublattice only
  (`wiki/patterns/cuda-graphs-decode.md:8-14`, `2026-09-30-prefill-graphs.md:25`).
- Async seals: PARKED (premise weakened: inter-chunk get_kv DOES read sealed
  records; prize re-estimated +1-3% for medium risk — `2026-10-02-async-seal-audit.md`,
  `doc/perf-strats.md:16-27`).

## Open questions (need GPU or decision, blocking follow-ups marked *)

1. *Under-replay device breakdown: re-run Kineto @64k with graphs ON and
   attribute serve/combine-tail/bmm/tail_reduce/merge + store remainder.
   Pre-graph split (4.0/1.2/0.8ms) post-dates replay; without a replayed
   profile the #1 vs #4 order is inferred from launch counts, not measured.
2. Combine split twin bar: exact (maxabs 0.0) or allclose (~1e-7 fp32 assoc)?
   Combine currently claims bit-identical NB-tighten (`:1042-1045`); CPG
   already accepts assoc noise (tree depth 2, allclose-gated `:1570`).
   Decide before #1 so the twin is not over-strict.
3. Store-status deferral safety: is a 1-step-delayed seal acceptable to the
   exactness contract (KLD same-top 100% + needle 6/6), or must status stay
   synchronous forever (`dispatch.py:507-509` comment says cannot speculate)?
   Blocks #4 design.
4. 128k occupancy: at 128k gc~1024 -> groups=128, CPG=8, iters 64/program.
   Does the 8-deep super-chunk regress vs 64k CPG=4 (register pressure /
   `num_stages=1` still best, `121673c`/`1fc2d4b` were 64k-era)? Blocks #3 sizing.
5. Merge GC-stride invariant: serve stores by GROUPS, merge/combine read
   GROUPS (`dispatch.py:580-584`, `kvarn_triton.py:1757-1758`). Any #1/#2/#3
   change must keep stride==count; the `_serve_groups()` single source
   (`:1768-1775`) is the guard — add an assert in merge/tail paths if touched.
6. Non-goals confirmed: 256-row groups (breaks v6 + Bee compat), eviction/
   windowing (breaks exactness), speculative decoding (model feature),
   K4V2 revival (scope call 2026-10-01) — all per `doc/perf-strats.md:43-50`.

## Compliance notes for the next cut

- One cut per commit, K4V4 vehicle, twin per kernel change, gates: CPU suite,
  PARITY=1, KLD-8k + KLD-64k identical (same-top 100%, mean <1e-4 @64k),
  warmed tg @8k/64k back-to-back pairs (protocol v2/v3: hot triton cache, run
  twice, expandable_segments at 64k+), VRAM, table, commit (`doc/kvarn-4090.md:637-642`).
- Easy-merge shape: env-gated fallback to legacy, no hot-path `print`/`traceback`
  (`dispatch.py:334,338,340,609,611` flagged in `2026-10-02-merge-audit.md`),
  const-ify 64/128 (`kvarn_triton.py:1822` equiv. now `:1775`), drop
  `DEBUG_HASH` prints (`:1846-1857,:1876-1881`) before any upstream prep.
- Suggested next commit: #1 (combine WHT-split + parallel reduce), env-gated,
  with combine twin (maxabs-or-allclose per Q2) + existing WHT twin reuse.
