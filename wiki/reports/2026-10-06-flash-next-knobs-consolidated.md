# Flash-Next knob battery — consolidated results (2026-10-06)

Every tunable on 3.05bpw Flash-Next serving (48 MoE layers, MTP head),
RTX 4090 24 GB + 7950X3D (16C/32T, 128 MB L3), Win11 + WSL2.

This is the **readable summary**. The full run-by-run record, including seven
retractions of my own earlier claims, is the audit trail in
`2026-10-06-flash-next-knob-battery.md`. Where the two disagree, the audit trail
shows how the number was reached; this document shows what is currently believed.

**Scope tested:** 33 offline arms (Phase A), 10 draft arms (Phase B), 19 live-server
arms (Phase C), plus 5 prompt-length stages and 3 content categories.

---

## How to read this

Every finding carries a confidence marker. They mean:

| marker | meaning |
|---|---|
| **HIGH** | interleaved reference in the same window, replicated, spread well under the effect |
| **MED** | measured, but n small, single pair, or one tier only |
| **LOW** | screen only (Tier 0/1), or confounded, or superseded |
| **SCREEN** | deliberately not measured to plan depth; recorded so it is not re-tested blindly |

Nothing below is marked HIGH unless the reference arm was measured in the same
time window as the candidate. That single rule caught more errors than any other
part of the protocol.

---

## 1. The recommendation

```yaml
model:
  #cpu_moe_offload_layers: 38        # mutually exclusive with the line below
  cpu_moe_split_experts: 380
  cache_mode: 2,2                    # unchanged from proposal; perf-neutral, +VRAM
draft_model:
  draft_num_tokens: 3                # 5 -> 3
  dynamic_draft: true                # unchanged
  draft_cache_mode: Q4               # unchanged (deliberate, confirmed)
memory:
  cuda_malloc_async: True            # unchanged
```

Plus one line in `start_tuned.ps1`: `EXL3_MOE_CPU_THREADS=8` → `16`.

| change | gain | where it pays | cost | confidence |
|---|---|---|---|---|
| `thr16` | +26.1% offline decode, +35.4% live tg | short/decode-bound prompts | **none (no VRAM)** | **HIGH** |
| `mcs380` | +14% prefill T/s at 130k; +1.36× combined at short ctx | 130k prefill | ~2.5 GB VRAM | **HIGH** (130k) / MED (short ctx) |
| `ndt3` | +7.2% tg | everywhere | none | **HIGH** |
| `cache_mode 2,2` | ~0 tg, +2.8 GB VRAM | — | none | **HIGH** (null) |

**Fallback order if VRAM ever binds:** drop `mcs380`, keep `thr16`. It costs no
VRAM and carries the largest decode win in the battery.

---

## 2. Adopted

### 2.1 `EXL3_MOE_CPU_THREADS: 8 → 16` — HIGH

Largest single win, and free.

| tier | tg | vs base | evidence |
|---|---|---|---|
| Phase A Tier 2 | 32.36 / 32.41 | **+26.1%** | n=2, replicates to 0.15% |
| live, sustained turn-matched | 46.6 tok/s | **+35.4%** | n=4, non-overlapping ranges |
| at 130k (prefill-bound) | — | **1.026×** | prefill unchanged — decode only |

Rejected directions: `thr24` regressed. `thr1/2/4` all scored below 8. Peak is 16.

### 2.2 `cpu_moe_split_experts: 380` — HIGH at 130k, MED elsewhere

`-mcs N` moves the **tail N** routed experts to the CPU, so **lower is faster and
larger in footprint** — VRAM and speed fall together.

| mcs | offline peak | offline min-free | decode tg0 | live-boot tested | verdict |
|---|---|---|---|---|---|
| 360 | 23308 | 831 | 30.00 | yes — **will not boot** | rejected |
| **375** | ~2506 *(interp.)* | — | — | yes — boots, **967 MB free** | rejected: **0.3% faster (noise)** |
| **380** | 21074 | 3065 | 28.66 | yes — 1479–1543 MB free | **ADOPT** |
| 390 | 20064 | 4075 | 26.68 | **never** | rejected: slower *and* smaller |
| 405 | 18540 | 5599 | 23.60 | **never** | rejected: slower *and* smaller |

`380` is the floor: `375` boots but buys nothing measurable for 35% less VRAM
margin; `360` does not boot at all. The boundary is measured, not inferred —
live min-free at 375 is 967 MB versus 1479–1543 MB at 380.

Combined with `thr16` at 130k: **1.157× turn-matched**, and per-half attribution
is clean (see §5.2).

### 2.3 `draft_num_tokens: 5 → 3` — HIGH

Acceptance fell but **quality does not**, so acceptance is only a speed
constraint. Swept 3→10 with the baseline bracketed (32.08 / 33.85 t/s):

| setting | tg t/s | vs base | acc | drafted |
|---|---|---|---|---|
| **`ndt3`** | **35.34** | **+7.2%** | 2.30 | 2.96 |
| `ndt5` (prod) | 32.08 / 33.85 | ref | 2.76 / 3.00 | 4.23 / 4.38 |
| `ndt6` | 33.13 | +0.5% | 3.11 | 4.77 |
| `ndt7` | 32.68 | −0.9% | 3.13 | 4.89 |
| `ndt8` | 30.81 | −6.5% | 3.17 | 5.29 |
| `ndt10` | 31.06 | −5.8% | 3.31 | 5.57 |

**Acceptance rises monotonically (2.76 → 3.31); speed does not.** Speed peaks at
`ndt3` and decays past `ndt6`. So the ceiling above 5 is *closed* — it is not a
speed win, the opposite of what the earlier "ceiling is above 6" note implied.

> Caveat: losslessness of speculative drafting is theory plus the operator's
> statement, **not measured here**. Phase C ran temperature 0 only; the Phase B
> `-temp` arm was never run. Confidence in the *direction* is HIGH; in the
> temp>0 guarantee it is theoretical.

### 2.4 `cache_mode: 2,2` — HIGH (as a null result)

−2.1% decode at Tier 2 (inside noise), +5.2% pp4096, and **+998 MB VRAM**. Kept as
VRAM relief, not as a speed change.

---

## 3. Rejected — measured and worse

### 3.1 Guard-trips and hard failures

| setting | outcome | confidence |
|---|---|---|
| `cpu_moe_offload_layers: 32` | `Insufficient VRAM in split for model and cache` | HIGH |
| `cpu_moe_offload_layers: 34` | **UNSAFE** — 129 MB free (guard 200), `cudaErrorLaunchFailure` in `decode_flash_attn` | HIGH |
| `mcs 360` | loads offline (23308 MB), **will not boot the server** | HIGH |
| `mcs 300` / `340` | VRAM/RAM refused | HIGH |
| `mcs 500` | refused | HIGH |
| `sysmem_kv_cache: 8192` | **RAM guard: 492 MB free during load** | HIGH |
| `sysmem_kv_cache: 24576` | **RAM guard: 88 MB free during load** | HIGH |

`cpu_moe_offload_layers: 34` is the dangerous one: it is *faster* (Tier 0 +22.3%)
and trips the guard. It is unsafe, not slow.

### 3.2 Measured slower

| setting | effect | confidence |
|---|---|---|
| MTP off | **−15.3% tg** | HIGH |
| `dynamic_draft` off (static) | −7.1% tg despite *higher* acceptance (2.92/5.00 vs 2.78/4.29) | MED |
| `EXL3_MOE_CPU_PIN: 0` | −13.0% to −19.8% | MED |
| `EXL3_MOE_CPU_SWIZZLE: 0` | −3.6% to −11.2% | MED |
| `EXL3_MOE_STREAM_T: 12` | **pp256 −40%** | HIGH |
| `draft_cache_mode: 2,2` | −23% tg on code; **slower in all 3 categories** | MED |
| `draft_cache_mode: 3,3` | +4.7% slower | MED (n=1) |
| `ndt7` / `ndt8` / `ndt10` | −0.9% / −6.5% / −5.8% | MED |
| `thr24` | regressed | LOW |
| `chunk_size: 8192` | +4.7 GB VRAM, no speed gain | LOW |

### 3.3 Rejected after investigation

| setting | why not | confidence |
|---|---|---|
| `recurrent_checkpoint_interval: 8192` | 1.013 ratio (noise), VRAM unchanged | MED |
| `recurrent_checkpoint_interval: 512` | +1.7% pp for **+800 MB VRAM** | MED |
| `recurrent_checkpoint_interval_pp: 8192` | works (6.7% cheaper edits) but costs 2.3 GiB RAM — the binding constraint | MED |

`recurrent_checkpoint_interval_pp` deserves a note because it is the one knob
where **quality of use, not speed, was the objective**. On the default 32768 grid
an early edit costs essentially a full re-prefill (0.7% saved); at 8192 it costs
6.7% less, confirmed by the server log (11% cached / 127,719 new vs 6% /
135,911 — exactly one checkpoint interval). Not adopted because ~6.7% on edits
is bought with RAM that `sysmem_kv_cache` already proved scarce.

---

## 4. Neutral — measured, nothing to gain

Given equal weight deliberately: these were real measurements that produced no
action, and listing them prevents re-testing.

| knob | result | confidence |
|---|---|---|
| `recurrent_checkpoint_interval` (4096) | ratio 1.013, VRAM unchanged | MED |
| `EXL3_MOE_ZERO_COPY: 0` | +2.5% to +3.5% — straddles zero | **LOW** (never run at Tier 2; prior work says +3.3%) |
| `EXL3_MOE_STREAM_T: 3` | +1.5% tg, neutral pp | LOW |
| `EXL3_MOE_STREAM_BATCH_EXPERTS: 24` | +2.9% tg, noise | LOW |
| `cache_mode 4,4` / `8,8` | neither beat 5,4 | LOW |
| `warmup: true` | **+12 s boot**, no sustained gain, *reduces* VRAM ~40–80 MB | HIGH |
| `vision_offload: false` | ~1.1 GB VRAM, no speed change | HIGH |
| `max_batch_size: 1` | +1.9% tg — **but halves serving capacity** | LOW |
| `draft_cache_mode: Q8` | 1.001 — neutral, and costs 178 MB | MED (n=1) |
| `draft_cache_mode: FP16` | 1.021 — neutral, costs 384 MB | MED (n=2) |
| `draft_num_tokens 4` | +1.9% tg | LOW |

**`draft_cache_mode` ladder, in full.** Monotone in VRAM, flat in speed down to
Q8, real penalty below. The VRAM it buys is not needed on this box (RAM binds),
so there is no reason to move in either direction. `Q4` was chosen deliberately
to recover VRAM without hurting tg — the ladder *confirms* that choice.

**Skipped per plan (SCREEN):** `EXL3_MOE_MEMOPS` (29% per plan, +10% per
`PERF_FINDINGS` — sources disagree), `PYTORCH_CUDA_ALLOC_CONF`, tensor parallel,
rope scaling, `ngram_ram`, `output_chunking`, `sysmem_multimodal_cache`.
`draft_confidence` is CLI-only, not in the config schema.

---

## 5. Long context — the gain is not uniform

Measured at five lengths, 11k to 259k prompt tokens (up to **96–99% of
`cache_size 262144`**). The operator runs 200k+ prompts, so the right-hand column
is the one that matters most.

| prompt len | combo vs baseline | what improved | confidence |
|---|---|---|---|
| 11–16k | **1.36×** | decode | HIGH |
| 55–62k | ~1.00× | — | MED |
| **130k** | **1.157×** | prefill +14% T/s | **HIGH** |
| 224k | 1.096× | prefill | **LOW** — single pair, see §5.3 |
| **250k** | **~1.00× (parity)** | nothing | MED |

**No length showed a regression.** But the honest summary is not "wins
everywhere": the combo wins where the bottleneck is decode (short prompts) or
mid-length prefill (130k), and is at parity at the top of the range.

### 5.1 Where the win comes from — measured at 130k

Same interleaved batch, prefill isolated from the server log:

| arm | 130k | prefill T/s | role |
|---|---|---|---|
| `mcs380` alone | **1.135×** | 1507 → 1725 (**+14%**) | the long-context lever |
| `thr16` alone | 1.026× | 1521 (unchanged) | decode only |
| combo | **1.161×** | 1749 | both |

### 5.2 The offline tier disagrees with the live server — and the offline tier is right to be doubted

| tier | `thr16` | `mcs380` | sum of singles | combo | vs sum |
|---|---|---|---|---|---|
| Tier 2 offline | +26.1% | +10.0% | +36.1% | **+13.4%** | **−63% sub-additive** |
| live sustained | — | — | — | **1.36× turn-matched** | — |

On decode-bound work the two changes are **substitutes** — more worker threads
and fewer CPU experts relieve the same bottleneck, so adding both relieves it
once. That corroborates the live server, where combo tg (51.0) sat *between* the
two singles (46.6, 37.0).

The live sustained metric wins because it is production-shaped. But this is the
second time an offline per-step gain failed to transfer live (`mcs375`: predicted
+2.4% decode, delivered 0.3%), so offline ladders should be treated as screens.

### 5.3 Why 224k and 250k are not trustworthy at face value

Prefill T/s at 250k across every boot run:

| arm | state | pp T/s | n |
|---|---|---|---|
| baseline | normal | **1510** | 9 (1509–1519, tight) |
| baseline | fast excursion | **1715** | 1 |
| combo | slow | **1526** | 10 (1520–1536, tight) |
| combo | fast | **1678** | 7 |

**Both arms enter a fast/slow machine state.** Matched *within* state the combo is
at parity: 1.011× (slow vs normal), 0.978× (fast vs fast). The apparent
1.11–1.13× wins came from comparing the combo's fast state against the baseline's
normal state.

**130k is the only long-context length with state-matched evidence.** The 224k
figure rests on a single interleaved pair and carries the same suspicion.

The state is **not** paging, clock, or VRAM — all measured and flat:
pagefile reads (two of three slow boots had *near-zero* reads while the fast boot
had the third-highest rate), CPU `% Processor Performance` (113.9–114.4% of
nominal, fully overlapping), available RAM, WDDM `SharedUsage`, run order
(`combo,combo,combo` → fast,fast,slow, not alternating), and config application
(server log identical). **Cause still unidentified.**

**VRAM is comfortable throughout:** min-free 1479–1543 MB at 96–99% cache
utilisation, and the spread does *not* widen with context.

---

## 6. Draft / MTP by content category

Acceptance (drafted-token acceptance, `Q4` baseline, generations ≥200 tok):

| category | tools | acceptance |
|---|---|---|
| agentic, curl | 29 | **75.4%** |
| code (`agentic_code_10`) | 11 | 68.7% |
| prose (`translate_02`) | **0** | **61.0%** |

`draft_cache_mode: 2,2` acceptance loss tracks baseline acceptance: curl
75.4% → 57.3% (−18.1pp), prose 61.0% → 47.2% (−13.8pp). Slower in **all three**
categories (code ~1.30×, curl 1.143×, prose 1.098×).

> **Confound, unresolved:** prose and tool count are entangled — `translate_02`
> has 0 tools, the others have 11 and 29. The milder prose result is as
> consistent with "untooled prompts degrade less" as with "prose degrades less",
> and no fixture provides prose-with-tools. Confidence MED; the direction (2,2 is
> worse everywhere) is HIGH.

The operator's workload is **code, tool-heavy**, so the code rows are the relevant
ones and the evidence base is representative rather than a narrow screen.

---

## 7. Memory findings

- **RAM, not VRAM, is the binding constraint.** Free RAM looks comfortable at rest
  (~49 GB of 64 GB) but is nearly gone during load — the CPU-offloaded experts
  need a ~34 GB large-page arena. `sysmem_kv_cache` failed on the RAM guard twice.
- The live server sits **+1.7 to +2.5 GB above** `eval/perf.py` on the same config,
  because the server builds components `perf.py` never does (vision modules, draft
  modules, MTP head + draft KV cache, serving buffers). **A config that fits
  offline by a thin margin will not fit live** — safe offline margin is ~2.5 GB,
  not the 200 MB guard. This is why `mcs360` fails live.
- **Do not decompose that offset from parameter counts** — measured twice and wrong.
  `vision_offload: true` frees only ~150–190 MB, not the ~1.1 GB the vision tower's
  unquantized fp16 weights suggest. Most of that tower was never VRAM-resident.
- **CUDA graphs are not a VRAM cost.** `warmup: true` *reduces* peak VRAM by
  ~40–80 MB. Its cost is entirely the +12 s boot.

---

## 8. Measurement methodology — the part that changed most the answers

| rule | why it exists |
|---|---|
| **Interleave the reference arm.** A ratio is only valid against an arm measured in the same window. | Session drift is ~10%; drift between consecutive boots is 0.07%. A carried-forward baseline is invisible to any same-batch check and produced two wrong 130k verdicts. |
| **Check the baseline is stable across its own interleaved runs** before quoting any ratio. | The first 250k batches looked conclusive until this was done. |
| **Compare turn-matched prompts**, not medians over different prompts. | Prompt difficulty otherwise dominates. |
| **For two-role workloads (cold prefill then replay), use the within-boot ratio**, never absolute second-turn wall-clock. | Identical requests ran 20% apart across boots, which would have handed a candidate a spurious 1.22×. |
| **Never quote a ratio before the last arm finishes.** | 250k was committed as 1.144× from boot 1 of 2; it settled at ~1.06×, then resolved to parity. |
| **Verify time alignment before believing any time-correlated claim.** | Two separate false correlations (a "cache pressure" theory and a pagefile correlation) came from windows that did not map to the phase being discussed. The jsonl row timestamp is the **end** of the run. |
| **Any knob whose benefit depends on a learned predictor must be swept across content categories.** | Code is not the best case for MTP acceptance — curl is. |

### Prompt-length calibration

The synthesiser's `--target` is **text** tokens; the server counts prompt tokens,
which are far larger because every message carries template scaffolding:

```
actual ≈ 0.2383 × text_tokens + 294.4 × n_messages + 9,110     (max error 0.35%)
```

Verified to 0.07% at one point, 0.01% on a later batch. The single-variable form
`text × 1.797 + 10,000` agrees at short lengths and **diverges near the ceiling**
— always predict every variant with both terms and reject a batch if any exceeds
`cache_size` minus generation.

---

## 9. Open — what this does not establish

1. **The 250k fast/slow machine state is unidentified.** Both arms enter it; it is
   not paging, clock, VRAM, run order, or config. Operators should match on it
   before quoting any 250k ratio.
2. **224k is a single pair** and may be another cross-state comparison.
3. **Losslessness of speculative drafting at temp > 0 is theoretical here** —
   Phase C ran temperature 0 only; the `-temp` arm was never run.
4. **`ZERO_COPY` is unmeasured at every tier** — it straddles zero in both
   directions across sources.
5. **Content vs tool count is confounded** in the cross-category check.
6. **Tier 2 covers only the 5 Phase A finalists.** Every other Phase A row is a
   Tier 1 screen and is not quote-quality.
7. **`cache_mode` and draft-length output quality were never assessed** — KLD owns
   those. `cache_mode 2,2` is a perf-only and VRAM-only line here.

---

## 10. Production state

`config.yml` was **never modified** and **never committed**. It is restored after
every arm; pristine md5 `0fe01cc8f2e1e4cd2de7b1a1648ecb4f`. All commits are
docs-and-tooling on `wip/kvarn-cache`.

Note on that md5: it is the *pristine* config. The per-arm files
`config.yml.kb-<arm>` are snapshots taken **before** that arm ran — the name
identifies the arm about to run, not a config with it applied. Reading one as
"the applied config" is a trap that cost one wrong correction during this work.