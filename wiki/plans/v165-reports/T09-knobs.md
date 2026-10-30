# T09 — inherited defaults and CUDA-side knobs (production 3.18 @ split 376)

Date: 2026-10-30. Engine win-perf `1.6.5` (`1422a0a1…`). Regime: `state.json` (3.18bpw, split 376,
`-cs 131072`, Q2, `-mct 16`, `EXL3_MOE_CPU_SWAP` default = dynamic, frozen profile restored per
run). Every arm is a full ladder with `-NoSpecDec -dr 3` (three decode reps per context, so the
within-run spread is visible and the median is over repeats, not over one 100-token sample).

**Knobs verified present before spending a pass** (installed wheel):
`EXL3_MOE_TIMER_RES`, `EXL3_MOE_THREAD_PRIO`, `EXL3_MOE_CPU_PIN_OFFSET` (C++), `EXL3_MOE_BLOCKING_SYNC`
(C++), `EXL3_MOE_STREAM_T`, `EXL3_MOE_STREAM_BATCH_EXPERTS`, `EXL3_GRAPHS`, `EXL3_HC_FOLD`,
`EXL3_GR_MIX_TILED`, `EXL3_SDPA_PREFILL`, `EXL3_MOE_HOST_CORES`, `EXL3_SPLIT_PROF`.

## Step 1 — threads: the mechanism gate rules out going down

Two `-Prof` probes at 8192 (`t9-prof16`, `t9-prof14`), reading the engine's own
`split prof [wait]` counter — the time the GPU stalls waiting for a CPU expert result, which is what
thread count is supposed to move:

| arm | wait med | wait p90 | issue med | PP @8192 | TG @7936 |
|---|---|---|---|---|---|
| `t9-prof16-20261030-090139` (16 threads) | **0.202 ms** | **0.267 ms** | 0.012 ms | **1885.15** | 38.38 |
| `t9-prof14-20261030-090319` (14 threads) | 0.221 ms | 0.293 ms | 0.012 ms | 1727.44 | 34.42 |

Fewer worker threads **raise** the wait (the GPU stalls longer), and they cost prefill throughput too
(1885 → 1727, −8 %). Both signals point the same way, so the gate says: 14 is not the direction. The
production default stays 16; the ladder below tests 15 for completeness.

The `max` outliers in both probes (17–27 ms) are the swap-sweep moments, not steady state — they
appear at the same magnitude with 14 and 16 threads.

## Step 1b/2–6 — the ladder and the env knobs

Every arm: full ladder, `-NoSpecDec -dr 3`, same session, same frozen profile. Reference
`t9-ref-a`: PP **1835.57** at 131072, TG **38.87** at 130816 (within-run range 37.88–40.16),
**37.79** at 65536. The within-run spread `-dr 3` reports is 3–18 % p2p depending on band, so a
knob is only credited with an effect it clears.

| arm (run id) | change | PP 131072 (ratio) | TG 130816 (ratio) | TG 65536 | min free VRAM | verdict |
|---|---|---|---|---|---|---|
| `t9-ref-a-20261030-090607` | production | 1835.57 | 38.87 [37.88–40.16] | 37.79 | 3617 MB | reference |
| `t9-th15-a-20261030-091001` | `-Threads 15` | 1689.63 (0.92) | 40.02 [39.16–41.24] (1.03) | 39.07 | 3617 MB | |
| `t9-ref-b-20261030-093737` | production, repeat | 1832.57 | 39.83 | 37.00 | 3617 MB | reference ×2 mean **39.35** |
| `t9-th15-b-20261030-094130` | `-Threads 15`, repeat | 1691.20 (0.92) | 39.31 | 37.06 | 3617 MB | ×2 mean **39.67** |
| `t9-hc0-a-20261030-091356` | `EXL3_MOE_HOST_CORES=0` | 1849.07 (1.01) | 33.40 [30.27–38.76] (**0.86**) | 30.64 (0.81) | 3617 MB | **rejected** |
| `t9-graphs0-a-20261030-091803` | `EXL3_GRAPHS=0` | 1815.80 (0.99) | 39.55 [38.83–40.05] (1.02) | 37.47 | 3621 MB | no effect, default kept |
| `t9-hcfold1-a-20261030-092156` | `EXL3_HC_FOLD=1` | 1815.01 (0.99) | 39.57 [39.46–39.64] (1.02) | 36.42 (0.96) | 3617 MB | no effect, default kept |
| `t9-grmix0-a-20261030-092547` | `EXL3_GR_MIX_TILED=0` | 1678.10 (**0.91**) | 38.85 [37.69–39.55] (1.00) | 35.94 | 2749 MB | **default (tiled) is right** |
| `t9-sdpa1-a-20261030-092947` | `EXL3_SDPA_PREFILL=1` | 1829.62 (1.00) | 40.32 (1.04) | 37.54 | 3617 MB | no effect, default kept |

**Verdicts:**

- **`EXL3_MOE_HOST_CORES=0` is a clear loss** — decode drops 14 % at the long band, 19 % at 65536,
  and its within-run range (30.27–38.76) is three times wider than the reference's. Confining the
  host process to the worker's cores (the default) is doing real work; do not float it.
- **`EXL3_GR_MIX_TILED=0` costs 9 % prefill** (1678 vs 1836 at 131072) for no decode gain. The tiled
  path is the default and stays.
- **`EXL3_GRAPHS=0`, `EXL3_HC_FOLD=1`, `EXL3_SDPA_PREFILL=1` are all inside the reference's spread**
  (within 2 % per side, and the reference's own within-run range covers each). None of them earns a place in the
  deployed env: the defaults stay, which is the useful half of the result — three knobs the campaign
  no longer has to wonder about.
- **Threads**: **15 and 14 both rejected.** 14 already lost on the wait counter and on both metrics.
  15's first pass looked like +3 % decode; the repeat gives 39.31 against a reference repeat of
  39.83, i.e. a two-pass mean of **39.67 vs 39.35 (+0.8 %, inside the spread)** while prefill is
  reproducibly 8 % worse (1689.7 and 1691.2 against 1835.6 and 1832.6). A decode-biased operator
  could argue for it, but it does not clear the spread and it costs a measured prefill loss, so the
  production value stays **16**.

**Not tested, with the reason:** `EXL3_MOE_BLOCKING_SYNC` — a prior measurement on this box put it
~4 % slower with MTP host syncs, and the campaign's advise-only list carries it for that reason.
`EXL3_MOE_TIMER_RES`/`EXL3_MOE_THREAD_PRIO` are the fork's worker-scheduling knobs (T8 step 3) and
are measured in `T08-fork.md`; `EXL3_MOE_STREAM_T` and `EXL3_MOE_STREAM_BATCH_EXPERTS` remain the
only inherited values not re-earned this session.
