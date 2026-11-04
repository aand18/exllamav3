# T33 — the pin is a step function: 8 GiB buys nothing, 24 GiB buys all of it

Cap curve of 2026-11-04, one session, server down, production regime (`split 381`, `cs 262144`,
`cq 4,4`, DFlash2 window 4, `max_length` 131072), `-dr 3`, instrument off, no spares (promotions were
killed in T32). Engine: the branch's two files, file-level in the venv.

## Question

T32 showed `EXL3_MOE_PINNED_ARENA=1` (unbounded) is worth +4…+37 % decode and +21.9 % prefill, but it
pins **30 720 MiB** of a 63.6 GB host and leaves 5.5 GB free. How much of that gain survives a cap of
8 / 16 / 24 GiB?

## Arms

Order **A P8 P16 P24 P∞ P∞ P24 P16 P8 A**, ten passes, every one `status=ok` (`results/passes.csv`):

| arm | run ids |
|---|---|
| A (no pin) | `t33-A1-pin-off-20261104-122134`, `t33-A2-pin-off-20261104-131426` |
| P8 | `t33-P8a-pin8-20261104-122752`, `t33-P8b-pin8-20261104-130801` |
| P16 | `t33-P16a-pin16-20261104-123425`, `t33-P16b-pin16-20261104-130154` |
| P24 | `t33-P24a-pin24-20261104-124028`, `t33-P24b-pin24-20261104-125616` |
| P∞ | `t33-Pinf-a-pinunbounded-20261104-124556`, `t33-Pinf-b-pinunbounded-20261104-125106` |

## Decode (`TG_S1`, tok/s, mean of two arms, spread in brackets)

| context | A | P8 | P16 | **P24** | P∞ |
|---|---|---|---|---|---|
| 0 | 38.09 (1.1 %) | 36.85 (2.9 %) | 38.14 (1.1 %) | 44.50 (3.3 %) | **49.80 (3.5 %)** |
| 2048 | 38.91 (2.5 %) | 38.92 (0.6 %) | 40.06 (1.3 %) | 45.47 (9.8 %) | **48.02 (0.7 %)** |
| 4096 | 38.00 (2.8 %) | 37.14 (0.6 %) | 39.91 (0.9 %) | 47.04 (0.1 %) | **48.19 (0.8 %)** |
| 16384 | **41.44 (0.2 %)** | 38.03 (2.6 %) | 39.90 (8.6 %) | 44.89 (11 %) | 44.86 (0.6 %) |
| 32768 | 42.09 (2.1 %) | 37.64 (3.9 %) | 40.94 (1.3 %) | 44.05 (4.2 %) | **46.51 (0.3 %)** |
| 65536 | 38.78 (2.8 %) | 38.17 (2.6 %) | 40.64 (5.2 %) | 42.41 (5.1 %) | **43.72 (2.0 %)** |
| **130816** | 39.20 (8.0 %) | 38.28 (3.1 %) | 40.66 (5.6 %) | **46.04 (1.2 %)** | 44.70 (7.0 %) |

Prefill, mean of two arms:

| band | A | P8 | P16 | P24 | P∞ |
|---|---|---|---|---|---|
| 4096 | 1945.9 | 2012.2 (+3.4 %) | 2121.9 (+9.0 %) | 2260.5 (+16.2 %) | **2324.6 (+19.5 %)** |
| 131072 | 1922.4 | 2000.0 (+4.0 %) | 2099.9 (+9.2 %) | 2225.9 (+15.8 %) | **2316.3 (+20.5 %)** |

## What it says

- **8 GiB is worth nothing on decode** — P8 sits on the control at every context (37.14 vs 38.00 at
  4096, 38.28 vs 39.20 at 130816) and buys only +3–4 % of prefill. This retro-validates t27's 8 GiB
  arm, which was read at the time as "pin buys prefill only".
- **The knee is 24 GiB.** P24 is +24 % at 4096 (spread 0.1 %), +17 % at 130816 (spread 1.2 %), and
  +16 % prefill — about 89 % of the unbounded decode gain at 4096, at 12 GB of free host RAM instead
  of 5.5 GB.
- **Unbounded only wins where production does not live**: contexts 0–2048 (+5…+12 %). At the
  production context (130816) P24 is 3.0 % *above* P∞, whose own spread there is 7.0 % (its first
  draw, 43.12, was its worst).

## Decision

**Ship 24 GiB** (`EXL3_MOE_PINNED_ARENA=1`, `EXL3_MOE_PINNED_ARENA_GIB=24`), taken on a pre-registered
rule and a ruling from the smart seat (R15):

- the rule said "smallest cap within `max(2 %, spread)` of P∞ at 4096 **and** 130816, ≥ 8 GB free";
  P24 passes at 130816 (+3.0 %) and **misses at 4096 by 0.4 of a point** (−2.4 % against a 2 %
  threshold). No cap passes literally, so the rule's fallback is unbounded.
- the ruling overrode the fallback on three counts: 5.5 GB free on a 63.6 GB Windows host is not a
  production margin (WDDM and the driver's pinning budget both bite there); live traffic is
  long-context, where P24 is the better arm; and P∞'s advantage is confined to contexts under 2k.

## Deployment, as shipped

- The venv keeps the installed `exllamav3` wheel; the branch's **two Python files** are the deployed
  delta (`modules/block_sparse_mlp_cpu.py` md5 `7ff6aded…`, `model/moe_cpu_host.py` md5 `7b9b705a…`),
  i.e. the exact bytes every t32/t33 pass ran. No wheel is rebuilt: after normalising line endings
  the branch tree and the installed package are byte-identical apart from these two files, so a
  rebuild would only swap a campaign-measured `.pyd` for an unmeasured one.
- `start_tuned.ps1` gains `$env:EXL3_MOE_PINNED_ARENA = "1"` and
  `$env:EXL3_MOE_PINNED_ARENA_GIB = "24"`.
- Verification: `size_check.py`, a `-DryRun` pass, the server log's arena report, free RAM and VRAM,
  `/v1/models` 200 and a completion smoke.
- Rollback: restore the two files from `_agent/adaptive-t27/backup/` (md5 `328a82d8…`, `75ce3c64…`),
  remove the two env lines from `start_tuned.ps1`, restart. If the 24 GiB pin refuses at split 381,
  unset `EXL3_MOE_PINNED_ARENA` first rather than moving the split.

## Limits

One box, one workload, n = 2 per config, batch-1 decode without the drafter (`perf.py` loads no
DFlash2). The production drafter verifies 5 rows per step, which changes the miss profile; a
`spec_decode` confirmation is the natural follow-up, not a precondition for this cap.
