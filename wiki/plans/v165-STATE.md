# v1.6.5 retune — live state

Config options and swap recipes: `wiki/plans/v165-reports/T14-config-catalog.md`. Per-task evidence:
the other reports in that directory. The operator owns the box: ask before the first GPU task.
Rollback wheels: 1.5.4 at
`/mnt/c/Users/yoho/Downloads/exl3-wheel-154/exllamav3-1.5.4+cu132.torch2.11.0-cp312-cp312-win_amd64.whl`,
1.6.4 at `C:\Users\yoho\Downloads\exl3-build-164\src\dist\exllamav3-1.6.4-cp312-cp312-win_amd64.whl`.

## Deployment now (2026-11-01)

The served endpoint, verified on the box:

- `Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8`, `cache_size` 262144, cache `"4,4"`, split **381**,
  DFlash2 BF16 window 4, `EXL3_MOE_THREAD_PRIO=1`, `max_batch_size` 2.
- Port 5000, API key unchanged (`api_tokens.yml`, mtime 2026-08-26). Launcher `start_tuned.ps1`,
  load 36 s with vision enabled.
- **Vision is on** since 2026-11-03: `vision: true`, `vision_offload: true`. The weights are the 987
  `model.visual.*` tensors already inside the exl3 checkpoint at `vision_bits: 6` - no separate
  projector is needed, and the engine has no GGUF loader at all, so the `mmproj-…BF16.gguf` in
  `D:\llms` is unused by this stack. They stream from system RAM, so the GPU is unchanged (21 834 MiB
  used, 2 305 MB free at boot, chunk 8192, split 381).
- Drafter is the **4-bit exl3 DFlash2** (`Qwen3.8-Flash-Next-DFlash2-exl3`, 472 MB against 1.2 GB),
  which is what lets chunk 8192 fit at split 381; prefill 1661 → 2289 T/s on a 19 k prompt.
- **No watcher runs against the live endpoint.** The floors stay in `watch.ps1` for measurement
  passes, where a runaway configuration must die within seconds (T13 step 11).
- Numbers: corpus 58.54 tok/s, 2.46/4 accepted, min free VRAM 763 MB — measured on the Q2 cache.
  The Q4 corpus run has not been done. Live per-bucket decode and prefill (also Q2): T14 section A.
  API proof on Q2: 165,433-token prompt prefilled at 1726 tok/s, cached repeat 503 tokens in 9.5 s.
- Split floors are engine-verified at 262144: Q2 needs 378, **Q4 needs 381** (376 and 378 refuse).
  The Q4 cache costs 3 more CPU experts. The same-day dynamic ladder puts eight experts at 0.8 %,
  so three are worth under 1 %; the older "1.7 %" came from a static cross-day coefficient and does
  not apply to the served placement.
- Exact files: `wiki/plans/v165-reports/artifacts/deployed/{config.yml,start_tuned.ps1}`. Config
  backups sit beside the live `config.yml` as `config.yml.bak-*`.

Two tabbyAPI facts that each cost a boot or a debugging session: there is no `Q2` cache mode, so Q2
is spelled `"2,2"`; and `curl` from WSL to the Windows listener returns `400 Invalid HTTP request
received` for a well-formed request, while PowerShell gets 200.

## Frozen winners

- **Arm A reference (T4, 2026-10-29):** 3.68bpw, split 400, 131072, Q2, no drafting,
  guard 0, huge pages off. A1 PP 1637.69@4096 / 1632.40@131072; A2 1602.18 / 1642.92.
  **Acceptance threshold (A2/A1): pp +-2 % at 4096–131072 (small bands +-5–10 %), tg +-11 % worst
  band (65536), typically +-5 %.** Report `T04-armA.md`; ratios in `artifacts/analysis/`.
- **MEMOPS_WINNER: 0** (T5, ABBA 2026-10-29: memops=1 is 8–17 % slower on tg at every
  context, medians 0.83–0.93, pp neutral-to-worse; reproduces the 1.4.9 finding). Deployment
  value unchanged; no 262144 confirmation (nothing to deploy changes).
- **MTP_WINNER (frozen 2026-10-30): `-ndt 2` + batched readback + stock MTP head, `-sdh 0`,
  `-hdt 0`, `-dds` off.** On 3.18 at split 376: ndt2 median **1.30× vs the default arm** (arm-to-arm;
  50–84 % acceptance per draft), default (4) 1.00×, ndt6 0.97×. The earlier 1.42×/1.10× figures were
  drafting÷baseline ratios, i.e. contaminated by the cold-baseline problem (critique C3).
  - `-sdh 32768`/`65536`: **negative** — 42.9 / 43.2 tok/s and 1.17 accepted/round against 45.9 and
    1.34 off, despite a 0.00 % miss rate (break-even 2.61 %) and 17/114 out-of-window probe
    disagreements. Dropped.
  - `-hdt 16/24/32/48`: **repetition-only** — median per-category ratio 0.99–1.02, with the whole
    mean gain coming from the synthetic "Trivial repetition" workload (1.16–1.61×) and a 5–12 %
    loss on both Agentic workloads at 32/48. Off by default; a lever for repetition-heavy traffic.
  - `-dds`: **unresolved, leaning +2 %** — median 1.018 across categories with no spike, against a
    reference that itself moves 0.972–1.014 between identical runs. One ABBA pair at the T13 lock
    decides it.
  - `draft_max_rows`: N/A at bsz 1 (single stream everywhere in this campaign).
  Report `T06-mtp.md` steps 3–5 with the per-category table.
- **Correction (2026-10-30): there is no engine leak and no tabbyAPI per-job caveat (critique C6).**
  The ~700 MB/arm growth was self-inflicted: our `spec_decode.py` build created a `Generator` per
  arm and never cleared `recurrent_cache`/`host_pool` (a Generator -> recurrent_cache -> page table
  reference cycle plus a thread pool; each arm stashed GDN checkpoints). Fixed with `reset_arm()`
  and one shared Generator. tabbyAPI creates **one** `AsyncGenerator` in `create_generator()` at
  model load and calls `close()` + `gc.collect()` on its error path — never per job.
- MTP readback (history): inconclusive on the screening model
  (T06 report) — **default batched (`_NO_BATCH_READBACK=False`) stands**; knobs move to the
  production model, because acceptance is a body/head quality property: 1.66 model = 1.91 bpw
  body / 4.01 bpw head → 1.0–2.2 of 4 accepted (break-even), 3.68 = 4.00 / 6.01 → measuring now.
- **DRAFT_WINNER: DFlash2 BF16, window 4** (T7, 2026-10-30). 56.57 tok/s and 2.40/4.00 accepted
  against the stock MTP head's 46.48 and 1.35/2.00 — **+22 % mean, median per-category ratio
  1.258** — measured in one session at split 376 with the frozen profile (`t7-ref` vs
  `t7-dflash2-w4`). Per category: agentic code 1.39, coding 1.36, repetition 1.36, agentic curl
  1.32, creative 1.19, **translation 0.88–0.90** (acceptance 37 % there). One drafter, one
  traffic mix: deploy it for coding/agentic traffic, keep the stock MTP head for a
  translation-heavy box. Windows: 4 > 6 (55.45) and 8 crashes — the drafter's `block_size` is 8 and
  the generator asks for `window + 1`, so the usable maximum is 7 (engine defect, reported).
  Sidecar heads closed negative (2.6 bpw: acceptance identical at w2, both tok/s and acceptance
  worse at w4). Report `T07-drafter.md`.
- CODE_WINNER (stock vs win-perf): **fork kept, null on the code delta** (T8). ABBA `t8-stock-a1` /
  `t8-fork-b1` / `t8-fork-b2` / `t8-stock-a2`: pp 1.00/0.97, tg 1.04/0.99, inside the ladder's own
  ±3–5 % jitter. Kept for the sidecar path (`-or`) and the worker knobs, all default-off.
- THREADS/HOST_CORES: **defaults stand, plus `EXL3_MOE_THREAD_PRIO=1`** (T8 step 3: 43.68 vs 39.35
  mean at 130816, **+11 %** decode and +2 % prefill, bracketed by two reference-level arms). T9:
  `HOST_CORES=0` costs 14–19 % decode, `GR_MIX_TILED=0` costs 9 % prefill, `-Threads 15` is +0.8 %
  decode inside the spread for 8 % less prefill, `-Threads 14` loses on both; `EXL3_GRAPHS=0`,
  `EXL3_HC_FOLD=1` and `EXL3_SDPA_PREFILL=1` all sit inside the reference's spread.
- **SPLIT: 376, confirmed 2026-10-30 on a warm ladder** (T10). Four arms at 131072, all `ok`:
  380 TG 36.76 / 376 40.38 / 372 39.93 / 368 38.43 at the 130816 band, PP flat (1824–1849).
  **368 now completes the full pass** — `perf.py`'s `empty_cache()` between bands removed the
  allocator-scratch death; the "does not load / cannot run long context" claim is closed.
  The 376 confirm pair spreads 36.30–40.38 (10 % peak-to-peak), so 368/372/376 are **tied on
  decode** and 380 is unprovable; **376 is chosen on the median plus headroom** (3617 MB free vs
  3081 at 368, and the winning drafter costs ~1.4 GB). The old "384 → 376 is worth ~15 %" claim does
  not survive the warm measurement (380 → 376 is 9 % at one pass each, inside the spread).
  Report `T10-placement.md`. Step 5 (pinned arena / promote-on-miss) not run this session.
- CACHE_MODE: **Q2 (2,2) — operator decision, frozen**
- STAGING_PAGES: **inert at `-chunk_size 4096`** (T11). `EXL3_QC_STAGING_PAGES` is a cap and the
  per-call need is 16 pages, so the allocator watermarks are byte-identical; `-rs32` costs ~0.7 GB
  VRAM for nothing. Q2 retrieval: 14/14 facts at 36.9 k and 16/16 at 165.5 k. A Q8/FP16 control
  does not fit at 262144, so this is "Q2 is fine", not "Q2 equals FP16".
- PLE/GDN: **defaults are load-bearing** (T12). `PLE_SUB_CHUNK=0` costs ~0.9 GB VRAM and 2 %
  prefill; `ngram_ram` (`-ngr`) drove free RAM to 637 MB in 27 s and was killed; `-rs32` costs VRAM
  for no gain; `BC_GDN=0` costs 15 % decode; `GDN_SUB_CHUNK=1024` is 2–4 % down.

## Harness

Repo source of truth: `bench/windows/` on `165-win-perf`. Deploy target:
`C:/Users/yoho/Downloads/tabbyAPI/_agent/v165/` via
`DEST=/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165 bench/windows/deploy.sh`;
always pass `-Root "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165"`. T0 done 2026-10-29:
`run_pass.ps1` carries the `-Model` override (commit `09923498`; the plan's literal placement
was a no-op — `$Model`/`$MODEL` are the same case-insensitive variable — fixed as a conditional
default) and `-cq 2,2` in the production regime. Deployed `run_pass.ps1` sha256 `01adc66f…`. `perf.py` unchanged v1.6.4→v1.6.5, sha256
`246e721a62d7c5238de4b7296adb0b26cd9f42a734b9f648436b35b9b6a29e04`.

### Harness v2 (2026-10-30, commits `5134738b…09e3b084`) — read before any measurement

The second review (tracker section C, plan `wiki/plans/v165-critique-fixes.md`) fixed six defects and
changed what a command means. Deployed files now hash-match the repo (verify with `sha256sum`):
`run_pass.ps1` `fba75185…`, `watch.ps1` `910f05a2…`, `perf.py` `b270a0e9…`,
`spec_decode.py` `333c4f24…`.

- **`state.json` is the regime, verbatim.** `run_pass.ps1` reads model, split, threads, cache,
  cache quant, env, floors **and `active.draft`** (`-mtp`, `-ndt N`, `-sdh N`, `-hdt N`, `-dds`).
  An arm passes only a delta. `-CacheSize` replaces the cache size; a drafting run defaults to
  `draft.spec_decode_cache_size` (40960) and a ladder run to `cache_size` (131072).
- **Draft args are `spec_decode`-only.** `perf.py` registers no draft-model args (it unpacks four
  values from `model_init.init`), so a pp/tg ladder never carries `-mtp`/`-ndt`.
- **`EXL3_MOE_CPU_SWAP=0` is automatic for `-Runner spec_decode`** and opt-in for a pp/tg run via
  `-StaticSwap`. A perf knob ladder keeps the engine default (dynamic) unless the task says
  otherwise — that is what the archived rows and the deployed server run.
- **`-Extra` may not restate a regime flag**, in any spelling (`-cs=…`, `--cache_size …`): the run
  aborts exit 2 naming the flag. `-DryRun` prints `DRYRUN`/`ARGS`/`ENV` and touches nothing, so a
  spelling can be checked while the box is busy.
- **Verdicts come from evidence.** A run with a traceback, a floor breach, no pp/tg rows (perf), or
  no `ARM`/table rows (spec_decode) is `fail`, never `ok`.
- **Gates are mandatory** per the measurement protocol: `bench/verify_task.py --pre` before the
  first pass of a task and `--post --report <file>` before a number is quoted. `--post` checks every
  cited run id is `ok` with `breach=0`, every `CPU split experts` line is warm, quoted bands match
  the report's own table, and a cited drafting run has arm rows.
- **`alloc peak` moved to the end** of `perf.py` and `spec_decode.py`, so `passes.csv` records the
  watermark after generation.
- **`deploy.sh` ships `state.json` on every run** and prints `updated`/`unchanged`.

Consequence for the campaign: every arm in T7–T13 must be described as a delta against
`state.json`, and a knob arm that needs a different split/threads/cache passes `-Split`/`-Threads`/
`-CacheSize`, never raw `-Extra`.

## Regime (measured baseline, 3.18bpw production model)

- cache: 262144, **Q2 (`"2,2"`) — operator-fixed 2026-10-29, used in every pass**, chunk 4096,
  batch 2. The measurement series itself ran 131072, because 262144 was a coin flip on VRAM at
  split 376; the 378 floor settled that.
- MoE: split **378** at 262144 with DFlash2 (377 refuses), **376** at 131072 (T10 warm ladder). The
  3.68-era baseline was 400: 360 and 372 did not load that model, and 392 died at the 65536 band.
  16 threads, `EXL3_MOE_MEMOPS=0` (T5), `EXL3_MOE_CPU_PIN=1`, `EXL3_MOE_CPU_SWIZZLE=1`.
- host RAM and determinism: `EXL3_HOST_MEM_RESERVE_MB=0` and `EXL3_MOE_ARENA_HUGEPAGE=0`, so page
  backing cannot vary between runs. Watcher floors **200 MB VRAM / 2048 MB available RAM**, armed
  for measurement passes only.
- drafting: DFlash2 BF16 window 4 (DRAFT_WINNER), with `-sdh 0`, `-hdt 0` and `-dds` on (`dynamic_draft: true`, ceiling 4). The two `-dds` arms sit +1.4 % and +3.6 % above the reference nearest each; the sign repeats, the size does not resolve.
- NGRAM: ng8 table present, `ngram_ram` off (T12), vision off.

## Superseded (read this before quoting an old number)

Each line replaces an earlier decision that the history file still records. Current values are above.

- Production model **3.68bpw → 3.18bpw_h6_ng8** (operator, 2026-10-29): 54.5 GB of weights, the ng8
  table hardlinked in.
- Split **400 → 378 at 262144, 376 at 131072**. The 400 was the 3.68 measurement baseline.
- Measurement length **131072 → 262144**, after the 378 floor removed the coin flip.
- Port **4099 → 5000**; the key never changed.
- Drafter **MTP draft block 3 → DFlash2 BF16 window 4** (T7).
- Watcher floors **100 MB/1200 MB → 200 MB/2048 MB available**, and **no watcher in production**
  (T13 steps 10–11).
- **Q4-for-coding withdrawn** (T13): a coding-only re-test showed no gain, so Q2 stays.
- **"262k + FP16 is impossible" retracted**: 6.94 GiB of KV, fits at split ~440 (T14).

## VRAM relief valve (standing rule)

When a component (drafter, cache, staging) does not fit: raise `-mcs` (more experts to system
RAM) and record the tg cost of the offload in the same row; only if the offload itself breaches
the RAM floor is the component shrunk instead.

## Advise-only (do not change without a task)

- large pages / `SeLockMemoryPrivilege` (no privilege on this box)
- WDDM memop handshake (measured 8–11% slower on 1.4.9)
- `EXL3_MOE_BLOCKING_SYNC` (~4% slower with MTP host syncs)
- `EXL3_MOE_ZERO_COPY` (gone), KVaRN (operator decision)
- QSA attention cap 2048

## Open items

- **Q4 against Q2 at one split: settled 2026-11-02.** Q2 at 389 runs 52.70 tok/s against Q4's
  52.96 with identical 2.34/4 acceptance (median per-category 0.993,
  `t16-q2-389-corpus-20261102-170933`). The cache width costs VRAM, not speed: the earlier 3.4 % gap
  was the eleven experts Q4's extra 0.75 GiB pushes to the CPU.
- **Production acceptance is lower than the corpus.** The corpus accepts 58.5 % of drafts; the live
  log of 760,980 drafts accepts 39.6 %, which is 1.58 of 4 against a break-even near 1.5. Every
  drafting number is priced on a friendlier workload. Measuring it needs archived production
  prompts, which the box does not keep.
- `-dds` is deployed (`dynamic_draft: true`, ceiling `draft_num_tokens: 4`). Two arms beat the
  reference on the corpus; a third reference run would make the pair a true ABBA.
- **Cache precision is bounded, not proven equal** (T15): acceptance is flat and non-diagnostic,
  `longctx` saturates at 16/16 on both arms, and the needle probe returns 32/32 on all three modes.
  All three probes sit at ceiling, so they bound a difference at about one missed item in 32. A probe
  that can fail - RULER variable tracking or aggregation, with a degraded-cache positive control -
  is what would settle it.
- FP16 at 131 k measured (`t16-fp16-131k-corpus-20261102-164829`): 50.64 tok/s, Coding ratio 0.998 against Q4 at 262 k, but 3.0 of its 4.4-point gap is the trunk, so it is not the cache-free precise option it first looked like.
- Huge pages: deployed value is zero and the day noise is 28 %, so a "normal day" re-run would not
  resolve anything. Closed as not worth measuring.
- `ambs 1` versus 2: the concurrency pairs are flat in aggregate (T15), so a single-slot arm buys
  latency, not throughput. Low value.
- DFlash2 quantisation remains the last speed compromise: ~0.8 GB, about nine experts, and the
  drafter's own break-even near 1.5 accepted per round. Needs the Windows converter and a window.

## Closing audit (2026-11-02)

A clean-context auditor checked the campaign against its own record. Closed by evidence: the Q4/Q2
comparison at one split (`t16-q2-389-…`), the post gate on T13–T16, `state.json` pinned to the
deployed copy by `test_gates.ProductionState`, and the drafter path emitted from the regime. Fixed
after the audit: two mistyped cells and an unsourced row in `T16`, three compromises stated more
favour-ably than the data, `size_check_argv` leaving the sidecar out of the fit pre-gate, and the
pre-gate now honouring a declared `fit_headroom_mb` because the tool's 1024 MiB default is stricter
than the engine's own accounting (the served config boots with 839 MiB).

Still open and disclosed: production acceptance (1.58/4 against a corpus 58.5 %), per-category noise
(two identical references differ 3.3 % on Coding), and the served placement never measured with the
drafter attached.

## Provenance

Dated record and the measurements behind each winner:
`git show 305f75fb:wiki/plans/v165-STATE.md`, plus the per-task reports in
`wiki/plans/v165-reports/`. Earlier revisions: `git log -p wiki/plans/v165-STATE.md`.

