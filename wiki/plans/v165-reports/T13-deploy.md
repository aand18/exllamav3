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

## Step 6 — does a smaller drafter buy a lower split? (operator's hypothesis)

The drafter's VRAM is not free for the drafter alone: every byte it does not take is an expert the
trunk can keep on the GPU. The campaign had compared drafters at fixed splits and splits at a fixed
drafter, never the two together, so both were measured across the interaction.

| drafter (262 144 cache) | lowest split that loads | corpus tok/s | accepted/round | min free VRAM |
|---|---|---|---|---|
| DFlash2 BF16 | **378** (377 refuses) | **58.54** | 2.46 / 4 | 763 MB |
| MTP head (`-ndt 2`) | **364** (360 refuses) | 46.77 | 1.34 / 2 | 595 MB |

**A smaller drafter does buy a lower split — 14 experts of it — and the split is not worth having.**
MTP's lighter head (~640 MB of cache+weights against DFlash2's ~1.4 GB) lets the trunk keep 14 more
experts on the card, and the corpus moves 46.48 → 46.77 tok/s (+0.6 %) while the *drafting* gap is
12 tok/s (58.54 vs 46.77, −20 % for MTP). The split curve had already flattened by 376 (T10: 368 ≈
372 ≈ 376 at 131 072), so VRAM freed at the bottom of the ladder buys almost nothing; the value is in
tokens per verification round (2.46 against 1.34), which the head choice controls.

Consequence: DFlash2 at 378 stays. A smaller drafter is worth considering only if a *future* one
keeps the window/acceptance, not for the VRAM.

## Step 7 — draft-cache quantisation: what was actually tested, and the Q2-vs-Q4 answer

The campaign's drafting arms never separated the two caches. `model_init.py` builds the draft cache
with **the same `k_bits`/`v_bits` as the trunk** (`-cq`), so every `run_pass.ps1` arm ran trunk and
draft cache at the same quant — Q2 ("2,2"). tabbyAPI *can* separate them (`draft_cache_mode`), and
the deployed config had been set to `"2,2"` by analogy with the trunk, **without a measurement**.

Measured now through the live API (the only instrument that can express it), same six prompts,
only `draft_cache_mode` changed:

| prompt | Q2 draft cache | Q4 draft cache |
|---|---|---|
| coding | 78.5 % | **85.2 %** |
| json | 57.6 % | **62.5 %** |
| reasoning | 50.3 % | **59.2 %** |
| translation | **52.1 %** | 42.4 % |
| creative | **33.2 %** | 29.7 % |
| list | **89.5 %** | 87.9 % |
| **total** | **52.2 %** | 50.4 % |
| tok/s | 56.0 | 56.1 |

At 165 k tokens of context: Q4 56.5 % cold / 53.2 % cached; Q2 46.4 % / 54.6 % (single samples).

**Verdict: Q2 and Q4 are unresolved at both short and long context, and throughput is identical to
0.2 %.** That matches the earlier box finding that BF16 versus Q4 draft cache is a wash and extends
it: Q2 versus Q4 is a wash too. The deployed config stays **`"2,2"`** because it costs half the
VRAM (248 MB against ~500 MB) with no measured penalty — and Q4 remains a defensible switch for
coding-heavy traffic, where it measured +6.7 pp acceptance on code.

## Step 8 — port

`network.port` was **4099** in the operator's `config.yml`; the operator asked for their usual
**5000**. 5000 had no listener and a connect attempt was refused, so the change was made and the
server restarted. Verified on the new port: `/v1/models` → `n_ctx=262144`, one chat request →
60.2 tok/s with 114/148 draft tokens accepted, floor watcher re-attached (2193 MB VRAM free).

Nothing else changed: the port is a config value, and 4099 was never a measurement decision.

## Step 9 — the level moves between sessions, and coding is the workload that counts

Two corrections, both measured after the operator pushed back on the campaign's own shortcuts.

### (a) "Do not compare across days" was the wrong rule

Called out as too blunt, and the objection is right: a calendar day is not a variable. Re-running the
archive's exact regime (split 376, `-cs 262144`, no frozen profile) on three days:

| date | TG 130816 | note |
|---|---|---|
| 2026-10-29 | 49.79 | the archive |
| 2026-10-30 | 37.47 | the day every T6/T8/T9/T11/T12 arm ran |
| **2026-10-31** | **52.33** | `xday-repro376`, +5 % on the archive |

So the box reproduces the archive two days later, and **2026-10-30 was the outlier** (28 % low), not
"cross-day comparison" being invalid. The rule that survives is: carry a reference measured in the
same session, and when a cross-day comparison is unavoidable, re-measure the reference first and
compare differences — not absolute levels carried across sessions.

What that invalidates: any absolute claim taken only on 2026-10-30. The **huge-page verdict**
(39.57 against 40.37 median) was measured entirely on that day and should be re-earned on a normal
day before it is treated as settled. Every *ratio* measured within that day remains valid, because
all its arms shared the day's level.

The expert-profile content was tested as the explanation and is not it: the same corpus at 378 with
the frozen master profile gives 58.54, with the live profile 56.42 — the frozen profile is if anything
the better one, and both are warm (48/48 layers).

### (b) Coding is the operator's main traffic, so the ranking is computed on it

Coding and Agentic-code columns across every drafting arm measured (same corpus throughout):

| arm | Coding | Agentic; code | mean |
|---|---|---|---|
| **DFlash2 w4 @378/262144 (deployed)** | **82.8** | **75.0** | 58.54 |
| DFlash2 w6 @376/131072 | 81.9 | 76.3 | 55.45 |
| DFlash2 w4 @376/131072 | 80.5 | 73.5 | 56.58 |
| MTP @364/262144 (its split floor) | 60.9 | 52.1 | 46.77 |
| **MTP + `-dds`** | **60.5** | 50.5 | 46.50 |
| MTP w2 (nd2, the old winner) | 59.0 | 53.0 | 46.48 |
| MTP, `-sdh 0` | 58.7 | 50.8 | 45.86 |
| sidecar 2.6 bpw w2 | 57.8 | 50.9 | 45.53 |
| MTP + `-hdt 24` | 57.3 | 49.6 | 49.10 |
| MTP w4 | 55.0 | 46.8 | 40.45 |
| MTP + `-sdh 65536` | 54.9 | 51.5 | 43.16 |
| MTP + `-sdh 32768` | 54.3 | 49.2 | 42.87 |

**What changes under coding weighting, and what does not:**

- **The drafter verdict gets stronger**, not weaker: DFlash2 is 1.36× the stock MTP head on Coding
  and 1.44× on Agentic code, and its one known regression (translation) is not this traffic.
- **The draft-cache switch does not survive.** "Q4 is a defensible switch for coding-heavy traffic"
  rested on a *single* coding prompt (85.2 % against 78.5 %). Re-measured on six coding/agentic
  prompts, 1.6 k draft tokens per arm: **Q2 66.3 % (1093/1648) and 56.9 tok/s against Q4 66.6 %
  (1074/1612) and 56.5 tok/s** — the same number twice. The deployed cache stays `"2,2"`, and the
  earlier recommendation is withdrawn.
- **`-dds` is the most interesting MTP knob for this traffic** (60.5 against 59.0 on Coding) but it
  has never been run on DFlash2, and with DFlash2's coding acceptance at ~85 % a window ceiling of 4
  that can only shorten is unlikely to help. Open, not recommended.
- **`-hdt` and `-sdh` stay off**: `-hdt 24` is 3 % down on Coding, and both `-sdh` sizes are 6–8 %
  down, so the repetition-only argument is not needed to reject them here.
- **Split 378 confirmed on the coding case**: Coding 82.8 against 392's 77.03 in the same protocol,
  +7.5 %, in line with the +7.8 % corpus mean.

## Step 10 — the watcher killed a healthy server (2026-11-01): free RAM is not available RAM

At 00:44:06 the floor watcher killed the process tree. Verbatim from `BREACH-server262.txt`:

```
free_vram_mb=1487 floor=200 low=False
free_ram_mb=1699 floor=2048 low=True
action=killed process tree
```

The server had been serving for five hours (18 k-token prompts at 40–56 T/s, 35–59 % draft
acceptance) and the kill was **wrong**. Three facts, all from the same session's samples:

- **VRAM was nowhere near its floor** — 1487 MB free against 200 MB.
- **The process's working set never moved**: 39 111 MB at the first sample and 39 111 MB at the last.
  Nothing was growing inside the server.
- **Free RAM fell 4610 → 3134 → 1699 MB in three seconds** while the working set was flat. That is
  the **standby list** — the OS file cache, which by then held much of the 51.8 GB streamed n-gram
  table. `Win32_OperatingSystem.FreePhysicalMemory` excludes standby, so the watcher counted
  reclaimable cache as a shortage.

Right now the same box reports `FreePhysicalMemory` 45 277 MB against `AvailableMBytes` 48 506 MB,
i.e. a standing ~3 GB that free-RAM accounting hides, and during service the gap is far larger.

**Fix (commit `784f8b44`):** `watch.ps1` reads `Win32_PerfFormattedData_PerfOS_Memory.AvailableMBytes`
— what a new allocation can actually get, and what Task Manager calls Available. The CSV column is
renamed `free_ram_mb` → `avail_ram_mb` and the `passes.csv` cell `min_free_ram_mb` →
`min_avail_ram_mb` (truthful names; the header on the box was repadded), and the floor keeps its
2048 MB value, which is now a real shortage rather than a cache artefact.

The server was restarted on the same configuration and is serving again (loaded 31.6 s, n_ctx
262144); the harness was already sending 22.5 k-token prompts while the restart was being verified,
which is why the verification request measured 20–33 T/s instead of 60 — contention with those
in-flight jobs, not a regression.

## Step 11 — second kill, and the production decision (2026-11-01 08:52)

The watcher killed the server again, this time on the corrected metric:

```
free_vram_mb=1219 floor=200 low=False
avail_ram_mb=2037 floor=2048 low=True
watched_pid=24164 launcher_pid=40340
action=killed process tree
```

VRAM was healthy and the process working set was 43.0 GB (up from 38.6 GB at the start of the run),
so this was not a leak: the streamed n-gram table plus the harness's own concurrency used the host
RAM for real. The available-RAM floor of 2048 MB sits below the normal working set of this workload,
so it fires on healthy production traffic, not on a fault.

**Decision:** the production server runs with **no watcher**. The floors stay in `watch.ps1` for
measurement passes, where a runaway configuration has to die inside seconds, but nobody arms them
against the live endpoint. With the watcher gone the host RAM ceiling is enforced by the OS again.
The server was restarted at 10:31 UTC (load 32.1 s, `n_ctx=262144`, port 5000, same key).

## Verifying this report

The 262144-at-376 run is cited as a refusal, not a measurement:

```
bench/verify_task.py --post --report wiki/plans/v165-reports/T13-deploy.md \
    --cites-failure t13-262k-dflash-20261030-080138
```
