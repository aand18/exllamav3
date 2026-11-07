# T51 — At 262 144 cells, adaptive placement raises coverage 22 → 57 % and moves nothing

Two passes, one session, 2026-11-07. `run_pass.ps1` production regime, `-dr 2`, pin on
(`EXL3_MOE_PINNED_ARENA=1`, `_GIB=24`), `EXL3_MOE_CPU_PROMOTE=0`, `EXL3_MOE_TIER_STATS=1`,
`EXL3_MOE_CPU_SWAP_DEBUG=1`. The arms differ in one variable: `t51-A` keeps the sweep on (the engine
default, and what the served box runs), `t51-B` sets `EXL3_MOE_CPU_SWAP=0` (`-StaticSwap`).

Runs: `t51-A-servedcov-262k-20261107-161729` (400 s, `status=ok`), `t51-B-sweepoff-262k-20261107-164709`
(393 s, `status=ok`). Floors: min free VRAM 2 199 MB in both, `breach=0` both.

## What the instrument saw

`t51-A` printed **49 tier samples**. Pooled over them, coverage is **33.94 %** (5 629 920 evaluations,
3 719 049 on the CPU). The trajectory is the finding:

```
22.3 24.1 25.0 26.6 24.0 25.7 25.5 26.2 32.9 ... 34.1 35.8 30.3 35.8 43.5 35.5 30.3 38.6 39.7
26.6 42.7 44.3 28.5 40.6 50.3 39.6 47.6 57.0 48.3 43.0 57.5
```

**First sample 22.3 %, last 57.5 %**, with **512 swaps across 11 sweeps** (64, 64, 54, 4, 7, 22, 59, 64,
64, 64, 46). On the single-row (480-evaluation) steps, CPU evaluations fell from **387 to 290 per step**
(−25 %) as coverage climbed 19 → 40 %. The sweep is not inert, and it is moving real work off the CPU.

The pre-registered prediction for this pass was **26–32 % pooled** (T50 §5). It is **missed** on the high
side, and the reason is instructive: 31.76 % was a *static* reading. The served regime starts there and
adapts upward within one pass.

## What the arms measured

Single-stream decode (`tg.csv`, `s1`), tok/s:

| context | adaptive (sweep on) | static (sweep off) | delta |
|---|---|---|---|
| 4 096 | 33.23 | 37.47 | −11.3 % |
| 32 768 | 35.21 | 37.00 | −4.8 % |
| 65 536 | 31.92 | 37.49 | −14.9 % |
| 131 072 | 33.41 | 37.80 | −11.6 % |
| **261 888** | **37.40** | **38.62** | **−3.2 %** |
| median over 12 bands | **34.48** | **36.37** | **−5.2 %** |

Prefill (`pp.csv`): median **2 174.6 vs 2 166.1 tok/s (+0.4 %)**, and +0.4 % at the 262 144 band — flat.

**So: a 35-point coverage swing, worth nothing on decode, and the point estimates favour the static arm.**
The 65 536 and 4 096 cells are dips that also appear in earlier passes (compare `t48-B`'s 32.09 at 32 768),
so the median is the honest summary and the long band is the like-for-like cell: **−3.2 % at 261 888**.

## The confound, stated

`t51-B` printed **zero tier samples**. The tier-stats tick lives inside the same path that the sweep arms
(`block_sparse_mlp_cpu.py:1474-1485`), so **with `EXL3_MOE_CPU_SWAP=0` the coverage instrument does not
emit at all**. Two consequences:

1. `t51-B`'s coverage is **inferred** from `t51-A`'s first sample (~22 %, the static load map), not measured.
2. The adaptive arm pays the instrument's own cost and the static arm does not. **The instrument's cost is
   unmeasured** (it was never run `=0`; T31 §686-689), so part of the −5.2 % may be instrumentation rather
   than the sweep.

Neither confound rescues the mechanism: with the instrument subsidising the *slow* arm, the true sweep cost
can only be smaller, and coverage still bought nothing.

## Verdict

**Coverage is not the critical path at 262 144.** Raising it from ~22 % to ~57 % — three quarters of the gap
to Strata's 82.77 % — changes decode by nothing measurable and costs a few percent at most. Whatever the
decode round is waiting on, it is not host expert evaluations at this context.

That closes the coverage family for deployment, on evidence rather than on a null arm:

- pin (`+23-25 % prefill`, 0 decode) — shipped for prefill only;
- promoter / pool / cadence — the mechanism works, coverage rises, decode does not follow;
- KV→experts at 32 k and 128 k (+5.5 %, +4.4 %) — those arms bought **GPU residency**, not coverage, which
  is why they moved and this did not;
- host KV tier — closed earlier (no context, no VRAM).

The remaining gap to Strata is therefore **GPU-side per-row work**: the fork processes a 5-row window in
~52 ms against Strata's ~25.4 ms for ~4 rows, and no coverage figure explains that. The fork has **no
per-stage GPU instrument** (G11 in the gap ledger), which is now the binding limitation — the next
measurement is a round-level GPU phase split, not another placement arm.

## Basis and caveats

- One pass per arm, `-dr 2` (two draws averaged per cell); the spread on this box is 1–3 % on a cell, so
  the −3.2 % at the long band is at the edge and the −5.2 % median is a point estimate, not a proven cost.
  The claim that survives either reading is the null: **no gain**.
- Both arms load placement from the **frozen** state dir (`XDG_CACHE_HOME=…\engine-state\active`), which is
  not the profile the live server uses (`…\.cache\exllamav3\expert_stats`, written 2026-11-04). Placement
  differs between pass and server; the *comparison* is unaffected because both arms share it.
- 35 of 48 layers are promotable under the 24 GiB pin (chunks 0–34); the other 13 can never adapt. A larger
  pin budget would change that and is a host-RAM question, not a VRAM one.
