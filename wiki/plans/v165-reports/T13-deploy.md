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

plus, in `start_tuned.ps1`:

```
EXL3_MOE_MEMOPS=0
EXL3_MOE_THREAD_PRIO=1      # T8: +11 % decode, +2 % prefill, two passes disjoint from the reference
EXL3_MOE_CPU_PIN=1
EXL3_MOE_CPU_SWIZZLE=1
EXL3_HOST_MEM_RESERVE_MB=0
EXL3_MOE_ARENA_HUGEPAGE=0   # T13 2c: three passes each, no benefit from 1
```

and the watcher floors 200 MB VRAM / 2048 MB RAM. `EXL3_MOE_TIMER_RES` and
`EXL3_MOE_CPU_PIN_OFFSET=8` are **not** deployed (both measured neutral); `EXL3_MOE_BLOCKING_SYNC`
stays off for the reason recorded in T8.

## Step 3/4 — booted under the watcher (operator instruction, 2026-10-31)

Operator: *"Start best variant, I wanna test it."* Applied: the fully measured configuration,
recorded verbatim in `artifacts/deployed/config.yml` and `artifacts/deployed/start_tuned.ps1`.

| field | applied | why |
|---|---|---|
| `model.model_name` | `…-3.18bpw_h6_ng8` | production model; the previous value named a **deleted** checkpoint |
| `model.cache_size` | `131072` | the measured length (262144 needs split 392 with a drafter) |
| `model.cache_mode` | `"2,2"` | Q2 — see the schema finding below |
| `model.cpu_moe_split_experts` | `376` | T10 ladder winner |
| `draft_model` | `model` / DFlash2 / 4 tokens / `"2,2"` / `dynamic_draft: false` | T7 winner |
| env | `MEMOPS=0`, `THREAD_PRIO=1`, `CPU_PIN=1`, `SWIZZLE=1`, `HOST_MEM_RESERVE_MB=0`, `HUGEPAGE=0` | T5/T8/T13 |

**Finding: tabbyAPI has no `Q2` cache mode.** `cache_mode` validates against
`literal['FP16','Q8','Q6','Q4'] | pattern '^[2-8]\s*,\s*[2-8]$'`, so the measured Q2 regime is spelled
**`"2,2"`** — the name is rejected at startup with a pydantic `literal_error`, which is how this was
found (the first boot attempt died before loading anything). `draft_cache_mode` takes the same pair.

Boot: `main.py` directly (not `start.py`, which re-checks dependencies) from the tabbyAPI directory
with the venv python, logs to `logs/server-<stamp>.log`. **Model loaded in 34.1 s, warmup 20.6 s,
serving on `0.0.0.0:4099`** (a configured port, not the 5000 default).

First request, from the Windows side:

```
GET  /v1/models        -> 200, id=Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8, n_ctx=131072
POST /v1/chat/completions
     23 prompt tokens + 64 completion tokens in 2.2 s (36.5 tok/s)
     accepted_prediction_tokens=37, rejected_prediction_tokens=71   <- the drafter is live
```

`c-api` note: `curl` **from WSL** to the Windows listener gets `400 Invalid HTTP request received`
for a well-formed HTTP/1.1 request (GET included), while the same request via PowerShell returns
200 — the WSL→Windows hop breaks the framing for this listener. Verification and any scripted
client should go through PowerShell (`Invoke-RestMethod`) or the host IP from a Windows shell, not
`curl` from WSL, or the server will look broken when it is not.

Floor watcher: `watch-server.csv` in the v165 results (floors **200 MB VRAM / 2048 MB RAM**), first
samples 3025 MB VRAM free, 11674 MB RAM free, tree 38 GB. `stop-server.flag` in the same directory
ends it.

## Step 5 — 262144 deployed at the split floor (2026-10-31, operator: "max VRAM use, 200 MB headroom")

The 392 recommendation above was a step, not a search. Asked to use as much VRAM as the card allows
while holding the 200 MB floor, the split was swept instead of stepped:

| split | 262144 + DFlash2 | result |
|---|---|---|
| 377 | — | **refuses at load** (`Insufficient VRAM in split for model and cache`, 25 s) |
| **378** | corpus: **58.54 tok/s**, 2.46/4 accepted, **763 MB** min free VRAM | **deployed** |
| 392 | corpus: 54.32 tok/s, 2.35/4 accepted, 2749 MB min free | previous recommendation |

**378 is the engine's floor, not the headroom's.** The watcher never came near 200 MB — the engine's
own load-time accounting refuses at 377 while ~1 GB was still free, so "200 MB of headroom" is not
reachable by lowering the split further. The lowest usable split buys **+7.8 %** over 392 (58.54 vs
54.32 tok/s on the same corpus, same cache, same drafter) and it is the last performance the split
can give.

Note the measurement that changed the answer: a one-workload 128-token smoke said 378 ≈ 392
(76.99 vs 77.03 tok/s on Coding); the full 22-workload corpus said +7.8 %. The smoke was noise.

Floors and shapes at 378 / 262144:

| run | workload | min free VRAM | min free RAM | verdict |
|---|---|---|---|---|
| `t13-378-corpus` | 22 workloads × 1024 tok | **763 MB** | 7779 MB | ok |
| `t13-378-xl` | 165 k-token prompts, 16 k-token generations | 1545 MB | 7015 MB | ok, 16/16 retrieval facts |

Deployed: `cache_size: 262144`, `cpu_moe_split_experts: 378`, `max_seq_len: 262144` (which also
silences the clamp warning). Boot 31.7 s, warmup 18.5 s.

**End-to-end through the API** (the deployment-shaped check):

| request | prompt | prefill | decode | drafter |
|---|---|---|---|---|
| cold | **165 433 tokens** | 95.8 s → **1726 tok/s** | 56.6 tok/s | 130 / 280 accepted |
| same prompt again | 165 433 (165 376 cached) | prompt cache hit | **503 tokens in 9.5 s** | 345 / 632 accepted |

The answer was correct — it located the inserted passage "between Chapters XXIX and XXX" and quoted
its first two sentences — and the floor watcher recorded **1393 MB** free VRAM at the minimum, seven
times the floor.
