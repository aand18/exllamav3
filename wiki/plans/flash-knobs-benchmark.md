# Flash-Next knob benchmark plan (handoff) — pp/tg/VRAM/RAM/boot time

Goal: measured impact of every tunable on 3.05bpw Flash-Next serving
(48 MoE layers, MTP head), so `config.yml` stops being tuned by lore.
Metrics per knob setting: boot time (process start → first token),
pp tok/s, tg tok/s, VRAM peak + min-free, sys-RAM free before/after.

## A. Where you are (read this first, it is all load-bearing)

- **Machine:** RTX 4090 24GB + Ryzen 7950X3D (16C) + 64GB RAM,
  Windows 11 + WSL2. GPU headless (display on iGPU), optimally
  cooled (no thermal variance — do not blame thermals).
- **Repos:** tabbyAPI (server) at
  `C:\Users\yoho\Downloads\tabbyAPI` (own git repo — NEVER commit
  there; `config.yml` edits stay working-tree + `.bak`, see §0).
  Docs commits go ONLY to
  `/home/dev/exllamav3/.worktrees/kvarn-cache`, branch
  `wip/kvarn-cache` (verify with `git branch --show-current`
  before every commit).
- **How to run box commands:** from WSL, `cmd.exe /c "..."` for
  reads; write `.bat` files for sequences (unix2dos every bat or
  it silently misparses). Never use bare `curl|wget` to stdout.
- **Model:** `D:\llms\Qwen3.8-Flash-Next-exl3-3.05bpw` (52.5GB).
  Server starts via `start_tuned.ps1` — ALWAYS use it, never
  plain `start.bat` (the tuned env is load-bearing: THREADS=8,
  PIN/SWIZZLE/ZERO_COPY=1, MEMOPS=0, STREAM_T=6/BATCH=48).
- **Production baseline** (from `config.yml`, verify each in the
  startup log before trusting): `cpu_moe_offload_layers: 38`,
  `cache_size: 262144`, `cache_mode: 5,4`, `chunk_size: 4096`,
  `max_batch_size: 2`, `draft_mode: mtp`, `draft_num_tokens: 5`,
  `dynamic_draft: true`, `cuda_malloc_async: True` (differs from
  upstream default False — do not "fix", it is measured state).
- **Known numbers (do not re-measure):** MTP draft payoff +
  `draft_num_tokens` sweep (in `PERF_FINDINGS.md`); MEMOPS 0-vs-1
  gap ~29% (`MEMOPS=0` stands); box spread ±25% between processes
  (fragmentation, NOT thermals/clocks) → §Guards quoting rules.

## 0. Lock the baseline + instrument (no knob changes)

1. Read `tabbyAPI/PERF_FINDINGS.md` fully + `tabbyAPI/config.yml`
   model section. **REVISED 2026-10-05 — the original single-workload
   line was wrong; there is no one command.** Knobs split three ways
   by what their harness can actually reach (the split
   `PERF_FINDINGS.md` already documents at lines 24-27 and 75-77,
   which this plan originally failed to carry over):

   | Phase | Harness | Server | Covers |
   |---|---|---|---|
   | A | `eval/perf.py` raw forwards (`bench.ps1` flags) | **STOPPED** | `cpu_moe_offload_layers` `-mcl`, `cpu_moe_threads` `-mct`, `cpu_moe_split_experts` `-mcs`, `cache_mode` `-cq`, `cache_size` `-cs`, `chunk_size` `-chunk_size`, `max_batch_size` `-ambs`, all `EXL3_MOE_*` env. Reports pp/tg + polled VRAM/RAM. |
   | B | `eval/spec_decode.py` MTP (`mtp_sweep.ps1` flags) | **STOPPED** | `draft_num_tokens` `-ndt`, `dynamic_draft` `-dds`, draft-off. Reports tg t/s + acc/draft, greedy and temp-1.0. |
   | C | live server via `start_tuned.ps1` + `config.yml` | RUNNING | everything server-only: boot time (start → ready → first token), `max_batch_size` concurrency, `warmup`, `recurrent_checkpoint_interval`, `cuda_malloc_async`, `draft_cache_mode`, `cache_size` as-shipped. Plus end-to-end validation of Phase A/B winners. |

   Evidence: `bench.ps1:17-18` maps `-mcl -mct -cq -cs -chunk_size
   -ambs -max_length`; `mtp_sweep.ps1:11-13` maps `-ndt -dds -tokens
   256 -single "agentic, code"`; `exllamav3/model_init.py:64` adds
   `-mcs` so split-experts is offline-testable too.
   `bench.ps1:2` requires the full 24 GB VRAM, so Phases A/B hold the
   server stopped; Phase C is the only phase that boots it.

   **`draft_cache_mode` is server-only** — `model_init.py` builds
   `draft_cache` with no corresponding CLI flag (lines 320-360), so
   §1.5's draft-cache sweep runs in Phase C, not B. Everything else
   in §1.5 stays in Phase B.

   Perf.py prints a pp/tg ladder, not one number: `Length N: X
   tokens/s` and `Context N: S=1 Y tokens/s`. Quote the
   agent-follow-up context (256-4096) for pp and `Context 0` for tg;
   state which context each pp figure came from.

   Only Phase C produces boot time — offline harnesses load the model
   inside the measured process and their load time is not boot time.
2. `cp config.yml config.yml.bak-<date>` before the first edit.
   Every knob change = edit + server restart + grep the startup
   log for the intended value (settings fail silently on typo).
   If the server is live-serving users, confirm restart windows
   with the maintainer first (restarts drop connections).
3. Warmed server (2 throwaway generations), 3 reps, median.
   Boot: `Measure-Command { <start-cmd> }` to the ready line.
   VRAM per run: `nvidia-smi --query-gpu=memory.used,
   memory.free --format=csv` before/after + peak from the
   server log. RAM per run: PowerShell
   `[math]::Round((Get-CimInstance Win32_OperatingSystem).
   FreePhysicalMemory/1024)` before/after. **÷1024 for MB —
   the original said `/1MB`, which is GB** (1MB is 1048576), so
   it printed ~46 where the guards mean ~46000 MB. Guards are
   stated in MB and FreePhysicalMemory is KB.
4. Run the baseline battery once → the reference row. All deltas
   vs this row, same day. If the box state changed (reboot,
   driver, server version), re-run baseline, never reuse old.

### 0.5 Depth tiering (added 2026-10-05 — full depth on all ~35 settings was 8+ h)

| Tier | Depth | Cost | Purpose |
|---|---|---|---|
| 0 smoke | every setting, 1 file × 1 rep greedy | ~1 min | kills guard-trips, crashes, >10% losers. Enforces the §1 STOP rule. |
| 1 screen | survivors: 2 warmup + 2 files × 2 reps, greedy | ~2 min | resolves >5% effects. Boot/VRAM/RAM need 1 rep only (near-deterministic). Skip the temp arm here — it only adds sampling variance. |
| 2 full | ~5-7 finalists + baseline: full 2 warmup + 5 files × 3 reps × 2 arms | ~12 min | **the only tier that quotes deltas.** |

Re-run the baseline at every tier boundary — tiers use different
workloads, so cross-tier numbers are not comparable. Any setting
still inside noise after Tier 1 is promoted to Tier 2; clear losers
stop at Tier 0/1 with their one bad row as proof (ladder rule).

Phase A maps tiers onto perf.py flags instead of file counts, since
perf.py measures a length ladder, not a prompt set:

- Tier 0 → `-spf -max_length 1024 -dr 1` (decode only, one context)
- Tier 1 → `-short -max_length 4096 -dr 2`
- Tier 2 → `-short -sd -max_length 32768 -dr 3` (full `bench.ps1` BASE)

Phase B maps them onto spec_decode.py `-single` categories, because
`-single` filters by CATEGORY, not by file — there is no single-file
selector (verified in `spec_decode.py:194-200`, category list at
`spec_decode.py:17-38`):

- Tier 0 → `-single "Coding"` (3 small files, real prompts)
- Tier 1 → `-single "Creative (reasoning)"` (3 files, thinking ON)
- Tier 2 → `-single "agentic, code"` + `-temp` (5 files, both arms)

Every tier uses real prompts on purpose. The cheapest category is
`"Trivial repetition"` (1 file, 531 B), but a prompt that EOSes after a
handful of tokens yields a meaningless t/s, and Tier 0 exists to detect
a >10% loser — it cannot do that on a degenerate measurement. Tier 1
keeps thinking ON because `dynamic_draft` trims drafts on
low-confidence reasoning content, so a non-thinking tier would measure
the wrong behaviour for the dyn arms.

**Rep reporting deviation from §0.3, deliberate.** §0.3 asks for a
median of 3 reps. Cross-process medians are unusable here: §A records
±25% process-to-process spread (fragmentation), which would swamp the
>5% effects under test. Instead reps run INSIDE one process —
perf.py `-dr N` reports mean plus min-max in a single load. Report
that mean and quote the min-max as the spread. This keeps §Guards'
single-process A/B discipline and is strictly more stable than the
median it replaces. Phase B has no in-process rep flag, so its reps
are separate processes: quote all three and the spread.

### 0.8 Test protocol decided 2026-10-06 (supersedes parts of §0/§0.5)

**KVaRN is out of scope.** The operator may not ship it, so nothing here may
depend on it. `model_init.py:288` gates the KVaRN cache on
`cq.startswith("kvarn")` — it is the only path that passes `kvarn_kwargs`. A
bare `k,v` pair (`5,4`, `2,2`, `4,4`, `8,8`) takes the plain-`Cache` branch.
So: use bare pairs only, never a `kvarn*` preset, and treat `-kvt`,
`-kvt_type`, `-kvsk`, `-kvsv` as inert and out of scope. Already-correct
measurements stay valid; every `cache_mode` row in the report is a plain pair.

**Sustained conversation replay is the primary metric** (§0.3's "1 request per
boot" is retired). Protocol: replay `agentic_code_10.json` progressively — turn
*i* sends `messages[0..i]` — so one boot yields a real
`prompt → answer → prompt` series with a growing, realistically warm prefix.

- Compare **the same turn index** across configs. Turn cost scales with prompt
  length (turns ran 11k → 16k prompt tokens), so a median across heterogeneous
  turns is not a stable metric — measured spread was 24–36%, which is prompt
  difficulty, not config.
- Primary estimator: **median of per-turn ratios** (config B ÷ config A at the
  same turn index). Prompt difficulty cancels by construction.
- Use per-turn **wallSec** as the raw signal; it is far more reproducible than
  parsed tg (combo turn 4: 7.02 s / 7.55 s across runs vs tg 43.4–55.6).
- `pp` from the server log is **unreliable in this mode** (parsed 102 where the
  real value is ~1200–1600). Use tg + wallSec; get pp from `-Rotate` instead.
- Boot time is still reported — it gates *testing* throughput — but as a cost
  to minimize, not a target.

**Cache sweeps are symmetric-only.** Do not test asymmetric `k,v` pairs. The
arms measured so far were `2,2` / `4,4` / `8,8` — all symmetric — so nothing in
the report needs retracting. The one asymmetric value in the data is `5,4`,
which is the **production baseline** and is measured as-is because it is what
actually ships; it is not a candidate. Note the recommended change
(`5,4 → 2,2`) is symmetric, but it is also a *quality* change
(asymmetric k=5/v=4 vs symmetric k=2/v=2) that remains unvalidated here — KLD
owns that. KVaRN presets stay excluded entirely (§0.8 above).

**Recorded metrics — per-process first, system-wide retained.** Both old
metrics were system-wide, so they silently attributed the desktop, the RDP
session, and any other process to our result. Record, per run:

| metric | scope | notes |
|---|---|---|
| `procRamWSMB` | our process tree | max resident working set. **exact** |
| `procPrivateMB` | our process tree | max commit charge. **exact** |
| `procVramMB` | our process | `nvidia-smi` per-process GPU bytes |
| `procCpuPct` | our process tree | % of all logical cores, peak |
| `gpuWattsPeak` / `gpuLimitW` | board | power draw |
| `gpuUtilPeakPct` | board | SM utilisation |
| `sysRamFreeMinMB` / `sysRamUsedPeakMB` | system | **keep** — feeds the RAM guard |
| `vramPeak` / `vramMinFree` | system | feeds the VRAM guard |

Two honest limits, both labelled in the output rather than smoothed over:
- **Per-process VRAM is usually `[N/A]` under WDDM**, so it is opt-in
  (`$KBWantProcVram`) and reported as unavailable instead of being faked from a
  system-wide delta.
- `PercentProcessorTime` is already a per-logical-core percentage, so the tree
  sums to `cores × 100` at saturation. Divide by core count — do **not**
  multiply by 100 again (that reported 5415% for a real ~54% load).

All fields come from **one** consolidated probe per poll (single `nvidia-smi`
query for memory+power+utilisation, single `Win32_OperatingSystem`, single
`Win32_PerfFormattedData_PerfProc_Process`), and the guards read that same
sample rather than re-querying.

**Determinism mode (operator: determinism worth a prefill penalty).** The DSA
staged-prefill path switches numerics on a context-length threshold:
`EXL3_DSA_QC_STAGE` (default `1`) selects a gather-once fp16 transient
(~3.9× faster attention at 16k) instead of the online dequantize path, gated
by `0 < pool_len <= EXL3_DSA_QC_STAGE_MAX_ENTRIES` (default 1M entries, ~1.2 GB
transient). `pool_len` is the *actual context*, so past that cap the same
config silently takes the other path — different numerics, different speed.
With a non-KVaRN cache this path is live, so it applies to this box.

- `EXL3_DSA_QC_STAGE=0` pins the online path: deterministic, no cliff, and
  costs the prefill speedup. **Use it for any run where A/B determinism
  matters more than wall-clock** — screening passes may use the default.
- `EXL3_DSA_QC_STAGE_MAX_ENTRIES` raises the threshold at ~1.2 GB VRAM per 1M
  entries. At 260k the default cap is a live risk on a config already near
  ~1 GB free, so either raise it or pin `EXL3_DSA_QC_STAGE=0` before the
  long-context passes.

**No seed parameter exists** and none is needed: `GreedySampler` is
`ArgmaxSampler` (`SS_Argmax`, no RNG), and the API accepts no `seed` field. The
whole battery already runs `temperature: 0.0`, so output token selection is
deterministic; residual spread is numerical (kernel/reduction-order, tied to
the ~830 MB VRAM swing on identical configs). Consequence: **draft-acceptance
differences under ~3pp are noise**, not quality signal.

### 0.9 Second-tier cache and checkpoint knobs (investigated 2026-10-06, ALL UNTESTED)

Found while auditing the config schema. None are in §1 and none have been
measured. All are confirmed real config keys (`common/config_models.py`) with
working implementations — behaviour below is read from the code, not assumed.
Ordered by how much they should matter for 128k/260k serving.

#### 1. `sysmem_kv_cache` — currently `0`. **MEASURED INFEASIBLE on this box.**

**Do not enable.** Both test arms were killed by the plan's own RAM guard
during load:

| setting | outcome |
|---|---|
| `8192` (8 GB) | `SERVER NOT READY: RAM free 492 MB during load` |
| `24576` (24 GB) | `SERVER NOT READY: RAM free 88 MB during load` |

Monotonic and severe. The pinned second-tier cache is allocated **eagerly**, so
it comes straight out of the ~37 GB the CPU-offloaded experts already require.
64 GB total does not cover both.

This also **corrects an assumption the rest of this plan rested on.** I had
treated VRAM as the scarce resource and system RAM as having headroom
(~34 GB used of 64 GB at rest). That is wrong once the model is loaded: free
RAM looks comfortable while idle and is nearly exhausted during load, because
the expert host arena, the offloaded vision tower and pinned buffers all land
together. **RAM is the binding constraint on this box, not VRAM** — which also
means the ~1 GB VRAM margin on `mcs380` is not the thing most likely to break
first at long context.

Its real doc (`generator/generator.py:122`), for the record:

`memory:` section, MB, default `0`. I first misread this as a VRAM-relief
lever; it is not. Its real doc (`generator/generator.py:122`):

> Complete K/V pages **evicted from the GPU cache** are stored there and
> restored **on prompt-cache hits instead of being recomputed by prefill**.

Mechanism: `CPUPageCache` attaches to the page table as `pagetable.cpu_tier`
(`generator.py:227-231`). Pinned system memory. **Not supported in
tensor-parallel mode** (fine — single GPU).

Why it matters here: the server log shows real prompt-cache reuse
(`prompt 17,830 tokens, 62% cached, 6,822 new in 5.56 s`). At 260k prompt
against `cache_size 262144`, pages **will** evict. Today an evicted page that is
needed again must be **re-prefilled**; with this set it is **restored**. So it
is a recompute-cost lever, and the cost it removes grows with context length —
exactly the regime we are heading into. Currently it does nothing at all.

#### 2. `recurrent_checkpoint_interval_pp` — prompt-ingestion checkpoint grid

Default `32768`. Only for models with recurrent states (this one: 48 GDN
layers). Its doc states the trade explicitly:

> With the default, a long prompt is only checkpointed near its end and an
> early edit costs a full re-prefill; with a denser grid the replay cost
> becomes proportional to the distance from the edit to the end of the prompt.

**Each checkpoint costs one recurrent state of system RAM — 148 MiB for this
class of model — bounded by `sysmem_recurrent_cache`.**

Note the character of this knob: it does **not** move tok/s. It sets what a
mid-conversation *edit* costs. That is exactly the agentic-coding-with-long-
context workload, so the number is worth knowing even though it cannot be
tuned for speed. Must be a multiple of 256, rounded up to a multiple of
`chunk_size`. Interacts directly with #3.

#### 3. `sysmem_recurrent_cache` — currently `8192` (8 GB)

Caps the total recurrent checkpoint RAM for #2. Also holds recurrent cache in
sysmem generally (`model.py:1022`). More budget ⇒ denser checkpoint grid
possible ⇒ cheaper edits, at the cost of system RAM. Pairs with #2.

#### 4. `recurrent_checkpoint_interval` — decode-side grid

Default is architecture-determined (`None` → engine default). I tested **512**
only (+1.7% pp, +800 MB VRAM) and never the other direction. Note `512` made
prefill *better* and VRAM *worse*, which is the opposite of the usual
"denser grid costs more" intuition and is worth confirming.

#### 5. `output_chunking` — currently `true`

`max_rq_tokens = chunk_size if output_chunking else None` (`model.py:450`).
Doc: "Maximum number of tokens before job is requeued… limits how many new
pages are allocated in the cache for the job in any one round and allows a
single job to use the full cache size **without limiting concurrency for other
jobs**." So turning it **off** allocates the whole completion at once. Never
toggled. Relevant only under concurrency — and `max_batch_size` is 2.

#### 6. `sysmem_multimodal_cache` — default `1024`, absent from `config.yml`

Now coupled to the standing `vision_offload: true` decision. `ImageEmbeddingCache`
is an LRU **bounded by embedding storage, not entry count** (`vision.py:34`),
evicting oldest-first and never dropping entries the current request already
resolved. If this workload never sends images, it is dead weight — but note it
is a *budget*, so it likely reserves nothing until populated. Low priority.

#### Explicitly not pursuing

- `draft_confidence` — **not in the config schema** (CLI-only), out of scope.
- `cpu_moe_threads` — the config-side twin of `EXL3_MOE_CPU_THREADS`. The env
  var is already proven and lives in `start_tuned.ps1`; no reason to move it.
- `max_seq_len` — already equals the model default (262,144); not a knob we
  need to set, and it is the same number as `cache_size`.

**Suggested order:** ~~`sysmem_kv_cache` first~~ — **ruled out, see #1.**
Next: the #2/#3 checkpoint pair for edit-replay cost at long context, but note
#1's lesson — #3 (`sysmem_recurrent_cache`) buys RAM, and RAM is what ran out,
so raising the checkpoint grid is capped by the same constraint. Then
`recurrent_checkpoint_interval` at 4096, which is a VRAM-side knob and so
should still be affordable.

### 0.7 Long-context requirement (added 2026-10-05 — do this LAST, after everything else)

The operator needs **128k minimum, 260k desirable** context. Everything in
§1/§2 above is measured at ~17k prompt tokens (the `agentic_code_*` files are
11k–30k), which is only 6.8% of the 262144-token `cache_size`. A knob that
wins at 17k can still lose at 128k+, so short-context results are a screen,
not a verdict.

**Run order, strictly in this order — this is the last work, not parallel to it:**

1. Finish §1, §2, §3 as written (short context).
2. **2x context pass (~35k)** on the *promising* knobs only. Pick the workload
   by prompt length, not by category: synthesize prompts to ~35k tokens from
   the `agentic_code_*` conversations by extending the message history. Test
   the top ~4 knobs plus the baseline — do not re-run the full ladder.
3. **Full-context validation** of the survivors at 128k, then 260k if 128k is
   healthy. This is the only tier that decides what goes in `config.yml`.

**Why the short-context VRAM numbers are more transferable than they look.**
`cache_size` is the KV cache *allocated at load* ("Size of the key/value
cache to allocate, in tokens"), so the ~20.5 GB steady-state peak already
includes the full 262144-token allocation regardless of how much context a
request actually uses. Longer context therefore costs mostly **prefill
time**, which `chunk_size` bounds, plus transient attention working set — not
a large new steady-state VRAM claim. The knobs most exposed to context length
are `recurrent_checkpoint_interval` (checkpoints near the end of a prompt, so
a 128k prompt hits very different paths than a 17k one),
`cache_mode` (quality and re-use at length), and anything with VRAM headroom
to spare — the winner below has ~1 GB, which is the real risk.

**How the long-context prompts are built** (`eval/_kb_mklongctx.py`).
`eval/perf.py` and `eval/spec_decode.py` select workloads by CATEGORY, not
length, and the longest real conversation is `agentic_code_29` (~30k prompt
tokens) — there is nothing to point at for 35k/128k/260k, so prompts are
synthesised by recycling the five `agentic_code_*` histories.

Three measured facts that drive the design:

- The five files hold only **36,326 text tokens combined**, so 260k means
  cycling the histories ~7× (the synthesiser tags recycled turns `[pass N]`
  so they are not verbatim duplicates).
- **Tools dominate the prompt.** `agentic_code_10` is 7,808 text tokens but the
  server logged **17,830** prompt tokens — the 11 tool schemas cost ~10k. Tools
  are per-request, attached once, and must not be counted per message.
- `cache_size: 262144` **is** 256Ki tokens, so a "260k" prompt is only ~2.1k
  under the ceiling. Leave headroom for the 256 generated tokens.

Prefill measurement needs **cold prefixes**: repeating one long prompt makes
every rep after the first ~100% prefix-cached and measures nothing. So the
synthesiser emits **distinct variants** (different start offsets into the cycle)
and Phase C sends them via `-Rotate`, one request per variant. Ladder built:
`synth_25k_*` (~35k prompt), `synth_118k_*` (~128k), `synth_248k_*` (~258k).

**Calibrate against the server log, not the estimate.** The synthesiser reports
text tokens; the server reports true prompt tokens. Read the latter and adjust
the target before trusting a length.

**Known trap, already paid once:** a config that loads under `eval/perf.py`
can fail to boot the live server. The live server sits **+1.7 to +2.5 GB**
above the raw-forward harness on the same config (`mcl38`: 18858 MB offline
vs 20546–21378 MB live; the live figure varies ~830 MB run-to-run on an
identical config, so the offset's own spread is comparable to its magnitude).
That offset is NOT server framework overhead — it is components `perf.py`
never builds: the server log shows `Loading vision modules 30/30` and
`Loading draft modules 3/3`, plus the MTP draft head, its draft KV cache and
serving buffers.

**Do NOT decompose that offset from parameter counts — it was tried and it is
wrong.** The vision tower (depth 27, hidden 1152, inter 4304, 987 tensors) is
unquantized fp16, which naively suggests ~1.1 GB. But a measured 2×2 of
`vision_offload` × `warmup` shows `vision_offload: true` frees only
**~150–190 MB**, not ~1.1 GB — so most of that tower was never resident in
VRAM. The residual is unattributed; do not repeat the arithmetic.

Likewise **CUDA graphs are not a VRAM cost.** The working hypothesis was that
`warmup: true` captures graphs and that this costs persistent VRAM, eating the
vision saving. The 2×2 refutes it: holding either key fixed, `warmup: true`
*reduces* peak VRAM by ~40–80 MB. The real cost of `warmup` is entirely in
boot time: **+12 s** (53.8 → 65.7 s, consistent in both columns), buying
~1–2 s (5–8%) on the first real request.

Because the offset is large and config-independent, any setting that fits
offline by a thin margin will NOT fit live. `mcs360` loaded under `perf.py`
(22486 MB) and then failed to boot the server at all — still true after
`vision_offload: true` freed 190 MB. **Every candidate config MUST get a live
Phase C boot before it is proposed, no matter what Phase A said about it** —
and the safe offline margin is ~2.5 GB, not the 200 MB guard.

### 0.6 Premise corrections from the 2026-10-05 battery (measured, do not re-litigate)

Four things this plan asserted turned out to be wrong on the box. They are
recorded here so the next reader inherits the measurement, not the lore.

1. **Thread count: "low values score higher" is backwards.** Tier 0 tg tok/s
   by `EXL3_MOE_CPU_THREADS`: 1 → 7.07, 2 → 11.38, 4 → 20.25, 8 → 24.69,
   12 → 30.33, 16 → 31.74, 24 → 30.68. Monotone up to 16, then it turns over
   at 24 (the core count). §1.6's "12 / 16 only as follow-up if low values
   don't resolve a trend" fires in the *upward* direction. This also
   contradicts `PERF_FINDINGS.md`, which saw 16 peak ~32 t/s but rejected it
   on **variance** ("swings 20–32"), not on the mean — so 16 likely wins on
   median and loses on spread, which only Tier 2 (`-dr 3`) can settle.
2. **The offload ladder's productive direction is DOWN.** `mcl34` was +22.3%
   and `mcl36` +19.1% over `mcl38`; 40/42 were −6.6%/−8.9%. §1.1 only listed
   36/40/42, so 34/32 were added by following the trend.
3. **`mcl34` is UNSAFE, not fast.** It peaked at **203 MB free VRAM** — 3 MB
   above the 200 MB guard — and died with `cudaErrorLaunchFailure` in
   `decode_flash_attn` during the `-short` prefill sweep. `mcl32` never loads
   at all: the engine raises `Insufficient VRAM in split for model and cache`.
   §1.3's "mark UNSAFE not slow" applies. The usable boundary is **36**.
4. **MTP max is not 4.** §1.5 said "4 is believed to be this model's MTP max".
   exllamav3 1.5.4 runs `-ndt 5` clean (`logs/mtp-154/r06-ndt5.log`,
   acc/draft 5.00), and production ships 5. The sweep ceiling is at least 5.

Two more harness-level facts, both of which cost real runs:

- **perf.py flakes with `ZeroDivisionError` on the `-short` sweep** (length-0
  timer resolution; `PERF_FINDINGS.md` already noted it for 1.4.9 and it
  survives into 1.5.4). It is benign — relaunch. It hit roughly half of all
  Tier 1 arms. `Invoke-KBRun` retries *only* that signature and never retries
  a guard trip or a CUDA failure.
- **`spec_decode.py` reads nothing from `config.yml`.** Phase B must lead with
  the production flags (`-mcl 38 -cq 5,4 -cs 262144 -chunk_size 4096
  -ambs 2`) exactly as `mtp_sweep.ps1:11-13` does, or it tries to fit all
  52.5 GB on the GPU and dies on the VRAM check.

Cross-process baseline drift, measured on an unchanged config: **24.69 vs
26.80 tok/s (−8%)** in two runs. So Tier 0 deltas are same-process only, and
the ">10% worse" stop rule has a ±8% noise floor — it can only reliably kill
large losers (PIN −19.8% qualifies; the −6…−8% cluster does not). In-process
spreads are far tighter (0.8–4.5% at `-dr 2`), which is the whole reason
§0.5 replaced cross-process medians with in-process reps.

## 1. Knob sweep — one at a time, in this order

For each: change ONLY it, restart, 3 reps, all six
metrics, restore before the next. STOP a knob early if its first
rep is >10% worse AND trips a guard (record + move on).
Ladder rule: sweep in the listed order and STOP the series at
the first setting that is obviously worse than baseline on the
primary metric (pp/tg down >5% median with no guard headroom
gained) — do not burn runs marching further down a degrading
direction. The series result is then "baseline stands, boundary
at <last-good value>", recorded with the one bad row as proof.

1. `cpu_moe_offload_layers`: 38 → 36 / 40 / 42. (VRAM↔RAM↔speed
   frontier; fewer offloaded = faster until the VRAM guard
   bites. Interacts with everything — first for a reason.)
2. `cache_mode`: baseline 5,4 → 2,2 → 4,4 → 8,8. (Full ladder
   ordered low→high; quality NOT measured here — label all rows
   perf-only; KLD owns quality separately. 2,2 may trip guards —
   that is data, record + move on.)
3. `chunk_size`: 4096 → 2048 / 8192. (8192 needs guard headroom
   — abort past it, mark UNSAFE not slow.)
4. `max_batch_size`: 2 → 1. (Single-stream ceiling vs contended
   reality. Do not "recommend" 1 on speed alone — it halves
   serving capacity; report both numbers.)
5. Draft: `draft_num_tokens` 5 → 4 / 3 / off; `dynamic_draft`
   on/off. (4 is believed to be this model's MTP max — confirm
   from the server log/error on 5+ rather than assuming; cap the
   sweep at the confirmed max. Narrow: MTP sweep exists; only
   re-probe gaps.)
   `draft_cache_mode`: **Phase C only** (no offline CLI flag, see
   §0.1). Sweep LOWEST first (Q2 if offered, else Q4
   → Q8 → FP16 — enumerate from the sample). Prior finding: low
   draft-cache quants showed no slowdown, so press downward for
   VRAM until quality or speed moves, then stop.
6. Env threads/streams: `EXL3_MOE_CPU_THREADS` 8 → 1 / 2 / 4
   first (experience says low values score higher on LLM work
   and saturation rarely pays on this CPU; 12 / 16 only as
   follow-up if low values don't resolve a trend);
   `STREAM_T` 6 → 3 / 12 with
   `STREAM_BATCH_EXPERTS` 48 → 24. (Env read at server start —
   restart required, re-verify with a settings dump.)
7. Ablations (one each, back to baseline between):
   `EXL3_MOE_CPU_PIN` 1→0, `SWIZZLE` 1→0, `ZERO_COPY` 1→0.
   (Keeps `start_tuned.ps1` honest — drop zero-effect lines.)
8. ADDED (in `config.yml`, commented — uncomment to test):
   `cpu_moe_split_experts` + `cpu_moe_threads` COMBINED
   (mutually exclusive with `cpu_moe_offload_layers` — set ONE,
   never both; finer + overlaps own-GPU-compute);
   `warmup` true/false (boot time vs cold-start stability);
   `recurrent_checkpoint_interval` (36 GDN layers);
   `cuda_malloc_async` True→False (upstream default; fork in
   allocator behavior — affects every fragmentation finding).

SKIP with reasons (do not re-litigate): `MEMOPS` (29% gap
measured); `PYTORCH_CUDA_ALLOC_CONF` (Windows no-op); TP
(single GPU); rope_* (model-driven); `ngram_ram` (tens of GB
RAM vs the 2GB floor rule).

## 2. Interactions (only the top-2 §1 winners)

Combine once. Combo underperforms the sum by >30% → record
the interaction (usually VRAM-headroom contention), keep the
single best. No three-way combos in this task.

## 3. Report + recommendation

One table: knob → boot/pp/tg/VRAM/RAM deltas vs baseline with
3-rep spreads + guard numbers attached to every claim. Then a
concrete `config.yml` DIFF PROPOSAL (not an edit: settings +
values + expected gain + risk per line) AND a
`start_tuned.ps1` verdict per env line (keep/drop). If a knob
was skipped per §1 STOP rule, it still gets a row (with why).

## Guards (binding, every run)

- VRAM kill under 200MB free; RAM abort unless ≥2GB free
  before, kill at 1GB during (swap poisons the machine —
  reboot territory, not a slow run).
- Restarts between values; startup-log verification every time.
- No cross-day/clock deltas (re-run baseline on state change);
  warmed, 3 reps, median; single-process discipline for A/B.
- Hands-off everything outside the knob list (no kernel,
  sampler, prefill, MoE-math, record-format, or dispatch work;
  no `master`/fork-overview/rebases; no force-push anywhere).
- Docs commits only in the kvarn-cache worktree on
  `wip/kvarn-cache`: `git status` + `git diff --stat` first,
  explicit per-file `git add`, atomic commits, push. Production
  `config.yml` is NEVER committed (proposed only). Spikes and
  all box logs stay untracked.
