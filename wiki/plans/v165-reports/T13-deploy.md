# T13 — lock the config, deploy, verify

Date: 2026-10-30. Engine win-perf `1.6.5` (`1422a0a1…`). This report measures what the deployment
needs to know and writes the configuration **as a proposal**; nothing in `config.yml` or
`start_tuned.ps1` was changed, and the server was not restarted (production service — operator's
decision).

## Step 2b — 262144 with the winning drafter: it does not fit at split 376

`t13-262k-dflash-20261030-080138`: **`status=fail` after 25 s**,

```
RuntimeError: Insufficient VRAM in split for model and cache
```

The refusal is at load, before any measurement — the engine sized the trunk, the 262144 Q2 cache and
DFlash2's 1150 MB of weights plus its draft cache, and would not fit them at `-mcs 376`.

`bench/size_check.py --cs 262144 --draft <DFlash2>` predicts the same and names the price of fitting:

| split | VRAM resident | free after the autosplit's reserve | fits |
|---|---|---|---|
| 376 | 21666 MiB | 850 MiB | **no** |
| 384 | 20965 MiB | 1551 MiB | yes, thin |
| 392 | 20264 MiB | 2252 MiB | yes |
| 400 | 19563 MiB | 2953 MiB | yes |

**262144 without a drafter fits comfortably** (`t13-262k-perf-20261030-080210`, `ok`, 317 s): PP
**1839.54** at the 262144 band (flat against 1846 at 131072), TG **39.32** at 261888, min free VRAM
**2855 MB**, `ALLOC 20618/20794`. So the long-context question and the drafter question are
**mutually exclusive at this split**: the cache costs ~1.2 GB and the drafter ~1.4 GB, and the card
has room for one.

| configuration | fits | PP @262144 | decode |
|---|---|---|---|
| 131072 + DFlash2, split 376 | yes | — (drafter arm is spec_decode) | 56.57 tok/s mean, 2.40/4 accepted |
| 262144, no drafter, split 376 | yes | 1839.54 | 39.32 @261888 |
| 262144 + DFlash2, split 376 | **no** | — | — |
| 262144 + DFlash2, split 392 | **yes** — `t13-262k-dflash-s392-20261030-081052`, `ok`, 605 s, min free VRAM 2749 MB | 56.57 → **54.32** tok/s (−4 %), acceptance 2.40 → **2.35**/4 | |

**The 392 arm answers the trade explicitly:** doubling the context costs 4 % of drafting speed
(54.32 vs 56.57) and a thin 2.7 GB VRAM margin, and buys a full 262144 window. Both configurations
are now measured; the choice is the operator's product decision, not an unknown.

**Recommendation to the operator:** for a coding/agentic box, **131072 + DFlash2 at split 376** —
the drafter is worth +22 % mean and costs no context the campaign measured as reproducible
(262144 was a coin flip at split 400 on 3.68 and only clears today at 376 *without* a drafter). If
262144 is a hard product requirement, the drafter must come off, or the split must rise to 392 and
absorb the offload cost.

## Step 2c — `EXL3_MOE_ARENA_HUGEPAGE` for production: no measured benefit

| arm (`EXL3_MOE_ARENA_HUGEPAGE=1`, split 376, `-NoSpecDec`) | TG 130816 | TG 65536 | PP 131072 | min free VRAM |
|---|---|---|---|---|
| `t13-hugepage-20261030-080735` | 37.94 | 35.89 | 1850.08 | 3109 MB |
| `t13-hugepage-r2-20261030-082104` | 39.57 | 38.95 | 1843.10 | 3617 MB |
| `t13-hugepage-r3-20261030-082401` | 39.69 | 37.72 | 1836.44 | 3617 MB |
| **median (=1)** | **39.57** | 38.95 | 1843 | |
| reference (`=0`, `t10-s376-…` r1/r2/r3) | 40.38 / 36.30 / 40.37 | 38.41 / 38.79 / 38.21 | 1823–1841 | 3617 MB |
| **median (=0)** | **40.37** | 38.41 | 1832 | |

**The 1.5.4-era "12–19 % tg from large pages" claim does not reproduce on 1.6.5.** Three passes
each: `=1` median 39.57 against `=0` median 40.37 — 2 % *lower*, with the two ranges overlapping
almost entirely (37.94–39.69 vs 36.30–40.38). The one pass that looked 500 MB tighter on VRAM did
not repeat either. **Recommendation: ship the engine default (`EXL3_MOE_ARENA_HUGEPAGE=0`) and stop
carrying the claim** — the measurement purity argument for `0` stands, and there is no tg to buy by
flipping it.

## Step 1/2 — the production configuration is stale in three load-bearing fields

`C:\Users\yoho\Downloads\tabbyAPI\config.yml` today:

| field | value | measured regime |
|---|---|---|
| `model.model_name` | `Qwen3.8-Flash-Next-exl3-2.05bpw` | **deleted in T2** — the server cannot load its configured model |
| `model.cache_mode` | `FP16` | operator-fixed **Q2** (`-cq 2,2`) |
| `model.cpu_moe_split_experts` | `360` | **376** — 360 does not load the 3.18 model |
| `model.cache_size` | `262144` | 131072 with a drafter |
| `draft_model.draft_mode` | `mtp`, `draft_num_tokens: 3`, `dynamic_draft: true` | DFlash2 w4, or MTP `-ndt 2` |

**Proposed patch (not applied):**

```yaml
model:
  model_name: Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8
  cache_size: 131072        # 262144 only if the drafter is off or the split is 392
  cache_mode: Q2
  cpu_moe_split_experts: 376
  cpu_moe_threads: 16
draft_model:
  draft_mode: model                     # DFlash2; use `mtp` + draft_num_tokens 2 for a translation mix
  draft_model_name: Qwen3.8-Flash-Next-DFlash2
  draft_num_tokens: 4
  dynamic_draft: false
  draft_cache_mode: Q2
```

plus, in `start_tuned.ps1`, the environment the measurements ran under:
`EXL3_MOE_MEMOPS=0`, `EXL3_HOST_MEM_RESERVE_MB=0`, `EXL3_MOE_CPU_PIN=1`,
`EXL3_MOE_CPU_SWIZZLE=1`, `EXL3_MOE_ARENA_HUGEPAGE=0`, and the watcher floors 200 MB VRAM /
2048 MB RAM.

## Step 3/4 — boot under the watcher, verify

Not done: it requires restarting the production server, which is the operator's call. Every number
this report quotes comes from the offline harness (`perf.py` / `spec_decode.py`), never from the
server, so the boot test is the one remaining step of T13 and it is a deployment action, not a
measurement.
