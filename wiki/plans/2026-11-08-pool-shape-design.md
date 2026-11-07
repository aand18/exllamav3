# T2 — Design: one pool, per-layer shape, exchange instead of admission

Status: design, no code. 2026-11-08, for T3 of `2026-11-07-adaptive-tier-shape.md`. Line numbers are
`exllamav3/modules/block_sparse_mlp_cpu.py` unless a file is named. Everything here is **default-off**:
env unset ⇒ the served config (`bench/windows/state.json:4-47`) behaves exactly as today.

## 0. What already exists (corrects a T50 premise)

"The shipped box has no adaptive placement at all" (T50 §3.5) holds for the **promote path** only. The
**sweep** is live: `_split_dynamic` defaults on (`:1354`), `spares = 0` without `-mcpa` (`:1365-1367`), so
`_split_swap_tick` skips the promote branch (`:1482-1485`) and arms the sweep (`:1486-1497`). And
`_split_swap_experts` **is** the protocol §3 asks for — re-read the incoming expert into the coldest
resident's slot in place, demote by map update (`:1543-1613`, flip `:1612`): slot count fixed, nothing
evicted. Cadence 128 steps (`:1486`), budget 64 swaps for the whole model (`:141`) ≈ Strata's 96/round.
**T3 is three deltas on a deployed mechanism**: per-layer pool, a cadence that spends the budget, a swap
that stops reading the checkpoint.

## 1. Per-layer slot vector

**Format: env `EXL3_MOE_CPU_SPLIT_VEC`, a JSON list of ints**, read once at `exllamav3/model/config.py:49`
beside `EXL3_MOE_CPU_SPLIT`, stored as `infer_params.moe_cpu_split_vec`, positional over split modules in
load order. Env over config key: an arm is one `-Env` line in `run_pass.ps1` and revert is unset (plan Q1).
`moe_cpu_split` stays a **scalar** — `model.py:614` and `:702` call `int(...)` and truthiness on it.

**Input-only inside the module — confirmed.** `split_k = int(getattr(ip, "moe_cpu_split", 0))` (`:971`) is
read per module, and every consumer derives locally: `first = num_experts - split_k` (`:1314`),
`wfirst = first - spares` (`:1369`), aux slice `ls[wfirst:]` (`:1411`), worker tail (`:1426-1428`),
`cpu_split_first = first` (`:1451`). Nothing compares modules; `_PromoteState` (`:200`) and `_TierStats`
(`:447-453`) already build `first` as a per-row column, and `moe_lookahead.cpu_slot` reads the module's own
(`moe_lookahead.py:472`). Replace `:971` with `vec[i] if i < len(vec) else scalar`, `i =
ip.moe_cpu_split_assigned` (exists, `:996`).

**Not input-only across the repo** — three uniformity assumptions:

| place | assumption | fix |
|---|---|---|
| `:141-148` | one swap budget for all layers; early layers eat the remainder | per-layer share `ceil(budget · first_L / Σfirst)`, global cap kept |
| `bench/size_check.py:82` | `gpu_share = (E - mcs)/E` for every layer | pass `--mcs min(vec)` (low over-estimates VRAM, high under-estimates) |
| `moe_lookahead.py:479-482` | no spares ⇒ nothing moves, worker slot = `id - first` | maintain `_split_mp` in exchange mode (§3) |

**Validation.** Per element at `:971`: integer, `1 <= k <= E-1` — a `0` is rejected because it drops the
module out of the sweep registry, so that layer could never adapt (upper bound is the existing cap, `:982`);
`EXL3_MOE_CPU_SPLIT_LAYERS` truncates layers before `first` derives (`:977-979`), so the length is the
post-truncation count. Per vector: one tally on the first `_split_swap_tick` (`:1460`, where `reg[0] is
self`) — `pool vector: 48 layers, Σ 6288 slots, min 64 max 256, scalar 131` — the line `verify_task.py
--post` cites. **Σ = 6,288**: 48 MoE layers (T50:19, `:115`) × (512 − 381) = 48 × 131 (arithmetic; 381 =
`state.json:7`, 512 = the profile key `…|512`, T50:40).

**Malformed or stale ⇒ fall back to the scalar, print once, never raise**: a bad vector must cost a uniform
split, not a failed load on a serving box. Trigger: not JSON / not a list / wrong length (stale — only
knowable after the load, hence the tally; prints `fallback:length 47 != 48`) / Σ ≠ 6,288 / any element
outside `[1, E-1]`. Record `ip.moe_cpu_pool_vector = "fallback:<reason>"` so the gate can tell
chosen-uniform from accidental-uniform.

## 2. Allocation rule

`m_L` = layer L's summed decayed hit count from the saved profile (`_expert_stats_load`, `:68-75`; the file
of T50 §2, Σ = 82,274 over 49 entries — the 49th is unexplained and must be understood before Σm is a
denominator).

**A — proportional to mass.** `s_L = round(6288 · m_L / Σm)`, then floor/cap, then largest remainder to
force Σs = 6,288. Uniform is 131 slots and 82,274/48 = 1,714 hits per layer; a layer with 4,000 hits gets
`6288 × 4000/82274 = 305.7` → 305 → capped 256; one with 900 gets `6288 × 900/82274 = 68.8` → 68; slots
the caps free go to the next-largest remainders (all arithmetic). Its exact form is marginal, not
proportional: take slots one at a time by descending individual count across all 48×512 cells — the global
top-6,288 of the profile, i.e. literally Strata's "one list across the whole model" (T50 §3), ~15 lines of
numpy on the same JSON. It dominates the rounded form: uniformly, a hot layer's 131st expert holds a slot
that its 132nd–250th out-count.

**B — proportional to observed miss mass.** Same arithmetic, `m_L` = worker-computed evaluations, and here
**the cap is the rule**: a layer carrying 12 % of misses wants `6288 × 0.12 = 754.6`, capped to 256 — a 3×
oversubscription, so the cap decides nearly everything. **The instrument already computes this and discards
it**: `_TierStats` accumulates per-layer worker-computed deltas masked to worker-resident experts (`:520`,
`:528-530`), but `report(detail≥2)` prints only the worst 6 (`:586-590`) and zeroes the array (`:592`). T3
needs ~5 lines — a cumulative copy, all 48 emitted; `EXL3_MOE_TIER_STATS=2` already means per-layer detail,
so no new knob.

**Recommendation: A in its marginal (global top-K) form, floor 64, cap 256.** (i) A load-time input, so
placement is reproducible and `reset_placement` has a target. (ii) B is a feedback loop — slots lower a
layer's misses, which lowers its next allocation; per-session oscillation is the last thing a tok/s arm
needs. (iii) Miss mass describes one conversation: the static-profile bet T50 already won. Use B once,
offline, as a saved second-pass profile if A's arm leaves headroom — never adaptively. Floor 64 = 0.49×
uniform (keeps the ratio test meaningful, layer's CPU share off the critical path); cap 256 = 1.95×.

## 3. Exchange instead of spare-admission

**Build on the sweep; do not extend the promote path.** The promote path is async only because both
destinations are *unmapped* (free slots popped at `:358`/`:378`, filled at `:710`) — that is what spares
are for. An exchange writes into a slot a live expert is routed to, so a forward during the copy reads
another expert's bytes: the copy must be quiesced, which the sweep already is (`:127-136`, all-device sync
before any layer moves). Async would need a third staging slot per layer ≈ 1.95 MB (`:158`) × 48 ≈ 94 MB of
**new VRAM** → **out of scope, not proposed**.

Hence `EXL3_MOE_CPU_PROMOTE` stays 0; `_split_promote_step`, `_split_pending`, `_split_free_gpu` and
`_split_free_cpu` are untouched and unused here. A boundary exchange has nothing in flight, so no pending
protocol is needed.

Hysteresis/floor **reuse as-is**: `SWAP_HYST=2.0` and `FLOOR=8.0 × hist.sum()/num_experts` (`:1512-1522`),
the sorted-zip pairing (`:1525-1531`), the post-sweep halving (`:1540`). Caveat: the absolute floor scales
with the accumulation window and a faster cadence shortens it, so re-derive it once on the box — else the
sweep quietly becomes a no-op (safe, but it voids the arm).

Cadence, with the arithmetic that sets it: overhead = swaps × cost / (interval × step). 96 swaps at 386 µs
is 37.1 ms; at 25 ms/step (40 tok/s, `t49-A-ref128k`) and interval 64 that is `37.1/1600 = 2.3 %` of decode,
1.2 % at 128. Arm `SWAP_MAX=96`, `SWAP_INTERVAL=64` as deltas. **Do not lower `INTERVAL` blindly**: at
`:1495-1497` a sweep 4 × interval overdue runs *inline, mid-generation*, which the code's own comment
(`:1489-1492`) says derails the text. T3 must make that bound explicit — `EXL3_MOE_CPU_SWAP_INLINE_MAX`,
default `4 × interval` (today's behaviour) — so a served arm can set 0.

`reset_placement` (`:728`, `:839-913`) is correct in exchange mode **because** it re-reads the checkpoint
for every slot whose current tenant (from the *current* map, `:733-740`) is not the load-map expert: after
an exchange `inv[g] = r_hot ≠ g` and `inv[first+c] = r_cold ≠ r`, so both directions are restored
(`:746-760`, `:765-785`). With `spares = 0`, `gpu_first = first` (`:741`) covers every GPU slot and the
"spare slot is unreachable" case (`:743-745`) never arises; `_split_map_initialise` (`:696-713`) rebuilds at
`spares = 0` and skips the free/pending lists.

One break: `moe_lookahead.cpu_slot` (`moe_lookahead.py:459-482`) assumes "no spares ⇒ nothing moves", so
exchange mode must maintain `_split_mp` at `spares = 0` (a host mirror of the map the sweep already builds
locally, `:1513`), or a prefetch reads a demoted expert's stale slot. Lookahead is off in the served config:
a correctness fix, not a T5 blocker.

## 4. Cheap swap — gated on T1b, not assumed

The arena API **already exists**: `MoeCpuHost.promote_expert` (`moe_cpu_host.py:1884-1918`) DMAs the trellis
block out of `arena_views` (`:1891`, raw `cuda_memcpy_async` `:1901`), repacks grouped/planar layouts in
flight (`:1907`), and copies `suh`/`svh`/`bias` from their GPU copies (`:1910-1915`); `demote_expert`
(`:1921-1967`) mirrors it into the arena and both aux copies (`:1947-1959`). So it is wiring: in
`_split_swap_experts`, replace the `stc.get_tensor` reads (`:1571-1573`) with `promote_expert` on the promo
stream and the blocking `install_expert` round trip (`:1586`) with `demote_expert`, then sync both events.

Requirements, all checkable pre-arm: `-mcpa` on (`:1365`); `promote_ready(layer, cpu_local)` true in **both**
directions (`moe_cpu_host.py:1788-1798`) — the pin can stop short and `pinned_layers()` (`:1801`) reports the
reach; `host.aux[layer]` current for that worker slot (source of the aux copies); and the existing shape
guard kept (`:1556-1568`), since `promote_expert` has no shape check and would corrupt a mixed-width slot.
Cost: **unmeasured**. Two 1.95 MB DMAs at 26.8 GB/s ≈ 73 µs each (`:158-159`, the code's own estimate) ⇒
~146 µs + repack, against an unmeasured sweep swap that today does two checkpoint reads (parent `:1571`,
child `moe_cpu_host.py:1986`) plus a blocking IPC round trip (`:1970-1978`). T1b must time a **sweep** swap:
the 386 µs figure is the promote path and does not transfer.

## 5. Failure modes, rollback

1. **A layer refuses the split** (`:1317-1336` non-mul1 / bad `trellis_rate` / mixed bias presence, or
   `:992` `num_local_experts != num_experts`): it loads whole on the GPU, so Σ first is short and **VRAM
   rises above the uniform baseline** — at 262 144 an OOM, not a slowdown. Read the tally before any pass.
2. **Mixed quant widths in a layer**: skipped by design (`:1556-1558`, `:1562-1568`), so the pool freezes
   silently there — T3 prints skipped pairs per layer.
3. **Mid-generation placement change**: preserved via `moe_cpu_swap_pending` → the generator's idle hook
   (`generator.py:1199-1203` → `run_pending_swap_sweeps` `:115-150`). Hazard: the `4 × interval` inline
   fallback (§3).
4. **Drafter churn**: a placement change is a numeric step for every touched expert and "visibly derails a
   generation in progress" (`:1489-1492`); Strata recorded the same (T50:96-97, plan Q4). One sweep per
   boundary, and T5 reports draft acceptance beside tok/s so a null is attributable.
5. **Vector drift**: Σ ≠ 6,288 ⇒ VRAM differs from the regime the arm was sized for (§1, `verify_task --pre`).

**Rollback**: every gate defaults to today — vector unset ⇒ scalar `moe_cpu_split`; `INLINE_MAX` unset ⇒
`4 × interval`; arena wiring gated on `-mcpa`; promote path untouched and off. Revert = unset three vars,
`reset_placement` (map **and** weights, `:716-786`), reload, box restored per `state.json`.

## 6. Test plan for T3 (invariants)

Synthetic split modules, no checkpoint, no device: `split_module()` in
`tests/testlib/test_reset_placement.py:33-45`, plus `tests/moe_cpu/test_tier_stats.py:60`,
`test_promote_visibility.py:71`.

1. **Σ slots constant**: for a 48-layer vector, `Σ (m.num_experts - m.cpu_split_first) == 6288`, and every
   layer's `_split_map` sorts to `range(num_experts)` (`:1535` does this per layer; extend across layers,
   before and after N sweeps).
2. **A promotion never reduces static residents**: `count(mp[r] < first) == first` per layer before and
   after every sweep; in exchange mode also `_split_spares == 0` and no free-slot lists.
3. **`reset_placement` restores the load map**: write a post-exchange map, reset, assert equality with a
   freshly initialised module and that `moved` counts the moved slots.
4. **Malformed vector falls back**: not JSON / wrong length / Σ ≠ 6,288 / element 0 / element `> E-1` — each
   falls back to the scalar with exactly one printed line.
5. **Mixed widths skip**: shapes differ ⇒ `_split_swap_experts` returns False, `nswaps` unchanged, map still
   a permutation.
6. **Existing suite green**: all 18 files of `tests/moe_cpu` pass with the passed/skipped count unchanged —
   take the baseline from the box, not memory (last recorded 124 passed, 63 skipped).

## 7. Not verified here

- Per-layer profile masses (the JSON is on the Windows box): §2's numbers are arithmetic on T50's
  Σ = 82,274, not measured per-layer values; the 49th profile entry (T50:41) is unexplained.
- The sweep's real per-swap cost (§4); whether `generator.py:1199` idles often enough under tabbyAPI
  traffic for a 64-interval cadence to fire once per generation — the served arm must log sweeps per pass
  before its tok/s is read.
