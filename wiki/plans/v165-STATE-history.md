# v1.6.5 retune — history (dated record)

Superseded material, kept whole for provenance. **Do not quote a number from this file without
checking `v165-STATE.md` first**: a later entry overrides an earlier one, and the live file carries
the current verdict. Entries are in the order they were written, oldest first.

## Code base (verified 2026-10-29)

- upstream `v1.6.5` = `cdc785db`; stock worktree `/home/dev/exllamav3/.worktrees/v165-stock`
  (branch `v165-stock`).
- fork rebase: branch `165-win-perf`, tag `win-perf-1.6.5` = v1.6.5 + Windows commits (worker
  knobs `5cef00e1`, .gitignore, Specter docs, harness `68eea9dc`, plans). Rebased clean onto
  v1.6.5; the four knobs (`EXL3_MOE_TIMER_RES/_THREAD_PRIO/_BLOCKING_SYNC/_CPU_PIN_OFFSET`) are
  wired in the rewritten pool code. Build worktree: `/home/dev/exllamav3/.worktrees/v165-win-perf`
  (detached at the tag).
- installed engine (verified T3, 2026-10-29): **1.6.5 stock** — built from the `v165-stock`
  worktree (wheel `exl3-build-165\src\dist\exllamav3-1.6.5-cp312-cp312-win_amd64.whl`, sha256
  `5337b620…`), installed via Windows `uv.exe`, verified from a Windows cwd (package + `.pyd`
  from venv site-packages). Report `wiki/plans/v165-reports/T03-build.md`.
- **T1 verified 2026-10-29** (`wiki/plans/v165-reports/T01-codebase.md`): tag
  `win-perf-1.6.5` = `d0f55f71` is NOT the branch tip — `165-win-perf` = `5835bb6b`, 3
  commits ahead (harness fix `09923498` + plan commits; bench/wiki only, engine code
  identical at the tag). 13/14 regime defaults confirmed at v1.6.5. **Mismatch:**
  `EXL3_NO_BATCH_READBACK` is not an env knob at v1.6.5 (or v1.6.4/fork) — it is a
  hardcoded test hook `_NO_BATCH_READBACK = False` in `generator/generator.py:32`
  (batched readback ON by default). T6's A/B is an in-process toggle between windows
  (plan corrected, commit `f85274c2`).

## Models (operator decision 2026-10-29; T2 downloads, then old 2.05bpw deleted)

- **production:** `D:\llms\Qwen3.8-Flash-Next-exl3-3.68bpw_h6_ng8`
  (turboderp/Qwen3.8-Flash-Next-exl3@3.68bpw_h6_ng8; 62.2 GB model + 51.8 GB
  `ngram_embedding.safetensors`)
- **screening:** `D:\llms\Qwen3.8-Flash-Next-exl3-1.66bpw_h4_ng3` (same repo @1.66bpw_h4_ng3,
  30.2 GB, its ng3 table NOT downloaded; the ng8 table is NTFS-hardlinked in). Fast loads,
  more experts in VRAM — screening ratios only, never absolute tok/s; placement/cache/drafter-
  VRAM arms and all confirmations run on the production model.
- **T2 status 2026-10-29: COMPLETE (steps 1–5)** — both downloads complete (3.68 dir 107 GB
  incl. ng8; 1.66 dir 29 GB; hardlink verified, link count 2); step 5 deletions done 2026-10-29
  (`2.05bpw` + `3.05bpw` deleted, D: ~179 GB free). **Step 4 load smokes DONE** (report
  `T02-models.md`): both models load and generate — `smoke368-4k` PP 1535 t/s @4k / TG 27–36 t/s,
  `smoke166-4k` PP 2254 / TG 37–55, breach=0 both.
  **Ruling — Arm A split = 400, not 360:** the regime default `-mcs 360` is a leftover from the
  deleted 2.05bpw model and does **not** load the 3.68 model (`Insufficient VRAM in split for
  model and cache`; `-lv` shows `layers.43` missing by 912 MiB at 360, `layers.45` at 372).
  At 400 it loads with 2216 MiB free after load. Cost if wrong: Arm A's absolute tok/s reflects a
  slightly more CPU-heavy split than optimal; every arm shares the split, so ratios stay valid,
  and T10's ladder searches 376…400 (not upward) to recover it.
  **RAM consequence:** at split 400 the CPU MoE arena is **43.02 GB locked**; total RAM 63.6 GB,
  and this box's resident `unbiased-strata.exe` normally holds 36–41 GB — the tuned config and
  that server cannot both be resident. Carry into T13.
  **Harness quirk:** a `-SkipPrefill -SkipGen` smoke always ends `status=fail` (the harness
  treats zero PP/TG rows as failure); judge load smokes by the log, and prefer a real 4096 pass.
  **Ngram check:** the loader prints nothing about the table (host-side, `prefer_cpu`); the
  hardlink is proven by the table file carrying the `head_offsets`/`head_vocab_sizes`/
  `layer_multipliers`/`head_bias` tensors that `_load_aux` requires with `optional=False` — the
  1.66 dir has no other copy, and it loaded. Table streams from disk (never in RSS).
- **Fit on this box, measured 2026-10-29 (the campaign's hard constraint):** RESOLVED. The
  3.68bpw model at 262144/Q2 runs full length with **`-Split 400` + `EXL3_HOST_MEM_RESERVE_MB=0` +
  the operator's floors (100 MB VRAM / 1200 MB RAM)**. Verified pass (`probe400`, 258 s):
  PP 1619@4096 … 1636@262144 t/s, TG 47–56 t/s, min free VRAM 503 MB, min RAM 5203 MB. Earlier
  kills were thresholds, not the model: 360/372 VRAM OoM at load; 392 killed at the 65536 band
  (23 MB VRAM); 400 killed by the old 2048 MB RAM floor / the engine guard (the load transient
  sits at ~2.0 GB). Q2 cache measured without a load = **2.093 GiB** at 262144/2 slots (KV 1.875
  + GDN states 0.218), so the binding constraint is the pinned arena, not KV.
  **Correction (2026-10-30):** the earlier claim that "the transient dips ~2 GB deeper for
  `max_length >= 131072`" was wrong — the loader never sees `max_length`, and same-split traces at
  4096 vs 131072 dip 8063 vs 8382 MB. The 2 GB difference between `smoke368-4k` and `probe400` was
  **background host-RAM drift over 90 minutes**, not `max_length`. Floors plus a frozen profile,
  never a `max_length` theory, are what make runs comparable.
  **Deployment consequence for T13 (corrected 2026-10-30):** on the **3.18** model at split 376
  the CPU arena is **32.9 GiB** (not the 43 GB carried over from 3.68), so the box *can* host it
  alongside more of the system than the old math suggested — still not a resident 36–41 GB
  inference server. `EXL3_HOST_MEM_RESERVE_MB=0` and the 100/1200 floors stay part of the served
  configuration.
  **MTP geometry (critique C10):** the head is **one** full-attention layer with **zero** GDN
  layers, so its draft cache at 262144/Q2 is **~160 MiB** (1.875 GiB / 12), its arena ~0.7 GiB and
  its GPU weights ~0.34 GiB — not the 1.1–2.1 GB claimed earlier. `bench/size_check.py` puts
  262144 + MTP at split 376 at **~20.9 GiB of a 24 GiB card with 1664 MiB free after the
  autosplit's 2 GiB reserve**: 262144 + drafter **fits** and must be *measured* in T13, not
  assumed impossible. (`perf.py -sd` allocates no second cache at all; only `spec_decode.py`
  creates a draft cache, sized by the head's one layer.)
  **tabbyAPI (critique C11):** `draft_mode: mtp` sets `draft_config = self.config`, so the head
  inherits the trunk's `moe_cpu_split`; `draft_mode: model` (DFlash2 / a separate drafter) builds a
  fresh config and does **not** forward `draft_moe_cpu_offload` / `cpu_moe_threads` — a T13 item if
  a separate drafter wins.
- **Log durability (operator requirement 2026-10-29, commit `94f4c305`):** every run now gets
  `runId = <Tag>-<UTC yyyymmdd-HHmmss>` in its log/err/watcher filenames and refuses to
  overwrite; CSVs carry `run_id` (+ the log filename in `passes.csv`);
  `bench/windows/archive_logs.sh` copies logs/CSVs/BREACH into
  `wiki/plans/v165-reports/artifacts/` (git-tracked) with stamped CSV snapshots.
- Rollback: 1.5.4 wheel on disk (path above) AND the 1.6.4 wheel at
  `C:\Users\yoho\Downloads\exl3-build-164\src\dist\exllamav3-1.6.4-cp312-cp312-win_amd64.whl`
  (earlier "not on disk" was wrong — checked the wrong dist dir).

## Production model swap 2026-10-29 (operator decision)

- **Production model: `3.18bpw_h6_ng8`, not 3.68bpw_h6_ng8.** Operator, 2026-10-29: *"since we're
  running with no headroom grab 3.18bpw_h6_ng8 (reuse local ngram table), its 8gb lighter. Use
  that instead of 3.6bpw."*
- Dir `D:\llms\Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8`: 7 shards, **54.5 GB of weights**
  (3.68 = 58.1 GB), ngram table **not downloaded** — the existing 51.8 GB `ng8` table is
  hardlinked in, exactly as for the screening model.
- Consequences, in order:
  1. **Arm A must be re-measured** on 3.18 (the acceptance threshold is derived from the model
     under test; the band spread may differ with a smaller GPU slice).
  2. **The split ladder re-founds on 3.18** (T10): fewer bits per expert means more experts fit
     in VRAM at the same split, so the arena can shrink — which is the headroom this swap buys.
  3. MEMOPS (T5) carries over: it is an engine path, not a per-checkpoint property, and its effect
     was 8–17 % with tight band spreads.
  4. The drafting series re-runs on 3.18; the 3.68 acceptance run stays as a data point.
- 3.68 stays on disk for now (rollback); its weights are ~58 GB and can be deleted once 3.18 is
  proven, which is an operator call.
- **Downloaded + staged 2026-10-29:** `hf download` (the `huggingface-cli` shim is deprecated and
  exits with a warning in this environment) → 7 shards, 54.6 GB, no table; the ng8 table is
  hardlinked in and now has **3 links** (3.68, 1.66, 3.18 dirs, one 51.84 GB file). D: 121 GB free.
  Structure verified against 3.68: same arch (`Qwen4ExpForConditionalGeneration`), 48 layers,
  512 experts, `moe_intermediate_size` 640, **6203 `mtp.*` keys** (same head shape), 6 PLE keys.
  Load smoke pending (GPU was busy with the last 3.68 drafting arm).

## Non-GPU prep done 2026-10-29

- **T3 build: DONE 2026-10-29** (`wiki/plans/v165-reports/T03-build.md`). Six watcher-gated
  attempts: five RAM-floor breaches (build peak ~4 GB at MAX_JOBS=2, ~3.5 GB at 1; the RAM
  holder was `unbiased-strata.exe` 41.5 GB — the server serving this session), then
  `build-stock6` OK at 2199 s (MAX_JOBS=2) after the operator restarted it (8.6 GB free).
  Harness quirk: `exit=-1` is the cmd-exit-code-read-empty trap — trust `WHEEL-EXIT 0` in
  `build_wheel.log` + the `BUILD_WHEEL` line.
  Src staged: `C:\Users\yoho\Downloads\exl3-build-165\src` (v165-stock copy) and
  `exl3-build-165-winperf\src`; recipes fixed (`SCRATCH=…\src`, MAX_JOBS=2).
  Install when built: Windows-side `uv.exe` (`C:\Users\yoho\.local\bin\uv.exe`) — the WSL uv
  cannot resolve a Windows venv interpreter (`--python ...\python.exe` → "No virtual
  environment found"). Verify from a Windows cwd that `exllamav3.__file__` is the venv path.
- **T6 step 0:** `spec_decode.py` deployed to `_agent/v165/`, sha256
  `204fac5fb61fcc2e2367f80614dd8bf06d7f7bca8974bb1ea83af08a8836c9d1` — the branch copy
  (`eval/spec_decode.py` on `165-win-perf`) carries a `-rb_ab` patch: measures the draft arm
  ABBA over `generator._NO_BATCH_READBACK` in one process (T6 step 1). The stock-worktree
  copy (unpatched, sha `2902dec0…`) is superseded.
- **T7 prep:** `D:\llms\Qwen3.8-Flash-Next-DFlash2\` (config.json + 1.206 GB model.safetensors +
  patches, assets excluded) downloaded. **Step-2 dir copy superseded by a no-copy path** (see
  sidecar prep below): `-or` override YAML shadows the head from one file, base dir untouched.
- **T7 sidecar prep DONE 2026-10-29 (non-GPU):** all four HermiHg checkpoints staged at
  `D:\llms\mtp-sidecars\`, sha256-verified against the HF repo LFS oids — 1.8bpw `3614dd45…`,
  2.3bpw `3a962ea0…`, 2.6bpw `998f5de2…`, 3.1bpw `f701abeb…`. Header-only load check vs the
  3.68 production base: **tensor key sets identical (6203 `mtp.*` each, zero diff)** — the shadow
  covers the whole head. Loader mechanism verified in code (`safetensors.py:add_tensor_files`:
  later file wins per key, prints ` !! Overriding <key> …` — that line in the load log IS the
  step-2 shadow check). NTFS/drvfs glob order verified alphabetical: `model-*` < `mtp-*` <
  `ngram_embedding`. **Confounder to record in T07-drafter:** the sidecar shadows 14 small
  norm/hyper-connection/router tensors with BF16 copies where the base checkpoint holds F16
  (sidecar copied bf16-exact from the BF16 source) — a small numeric delta beyond the
  requantized linears. **No-copy arm path (2026-10-29):** the existing `-or/--override` YAML
  mechanism routes every `mtp*` key to a sidecar stc (`VariantSafetensorsCollection.find_stc`;
  `SafetensorsCollection` accepts a single `.safetensors` path) — specs at
  `bench/windows/mtp-sidecar-{2.3,2.6,3.1}bpw.yaml`, deployed by `deploy.sh`; base dir stays
  untouched, no 55 GB copy. Needs the one-line `model_init.py` assert relaxation on this branch
  (`-or` + `-mtp` was blocked by the separate-draft-model guard; with `--mtp` the draft config
  IS the model config, so the override covers the MTP component). NOT in the `build-winperf`
  wheel compiling at prep time — rebuild after the patch lands, or fall back to the dir-copy
  path on the unpatched wheel. tabbyAPI forwards no `-or`: production deployment of a winning
  sidecar is the dir-copy (or strip-MTP) path, decided at T13.
- **T8 build staged:** `C:\Users\yoho\Downloads\exl3-build-165-winperf\src` = worktree at
  `f85274c2` (tag `win-perf-1.6.5`; engine code identical to branch tip) + fixed `_build.bat`.
  Build AFTER the stock build finishes (run_build aborts on a concurrent `bdist_wheel`).
- **T8 wheel BUILT 2026-10-29 (patched):** `build-winperf` ran clean 14:40→15:16 (unpatched),
  then rebuilt incremental (2 s, C++ untouched) after applying `1385306e` to the staged src
  (CRLF preserved; baseline verified == `f85274c2`). Wheel
  `C:\Users\yoho\Downloads\exl3-build-165-winperf\src\dist\exllamav3-1.6.5-cp312-cp312-win_amd64.whl`
  sha256 `1422a0a1e67eaf88efd6cddfaa200ded32e64a772b32db7509f7128d16b4d60f` — patched assert
  confirmed inside the wheel; it **supersedes the unpatched `d3eaf6c9…` build at the same path**.
  **NOT installed** — the installed engine is stock 1.6.5 (T3, DONE: built, installed via
  Windows `uv.exe`, verified from a Windows cwd; report `T03-build.md`); T8 installs this wheel.
  T8 note: this wheel carries the `-or` load-time patch on top of the four worker knobs —
  load-time only, no runtime delta, so the stock-vs-winperf comparison stays clean.
  Harness quirk: `run_build.ps1` reported
  `exit=-1` because `Start-Process -PassThru` couldn't read the `cmd.exe` exit code; the
  recipe's `WHEEL-EXIT 0` in `build_wheel.log` is the authoritative verdict, no BREACH file.
- **T13 tabbyAPI decision (recommended 2026-10-29, pending operator confirm):** do NOT patch
  tabbyAPI for `-or` — it builds its own Config/Model (no `model_init` path), and the loader's
  alphabetical glob already gives production the sidecar for free: deployment = model dir
  containing `mtp-*.safetensors` (+ optional strip-MTP of the base shards, reclaims ~1.9 GB
  from shard 8; with a stripped base a missing sidecar fails the load instead of silently
  falling back). `-or` stays benchmark-only. Only patch still possible: the 3-kwarg drafting
  forward (`-sdh`/`-hdt`/`draft_max_rows`), and only if T6/T7 freeze a winner using them.
- **Merge + deploy 2026-10-29:** `optimized-mtp-heads` merged into `165-win-perf` (`1479e6ae`),
  tag `win-perf-1.6.5` moved to the merge tip (the tag source now includes `1385306e`, matching
  the built wheel). Sidecar YAMLs deployed to `_agent/v165/` (2.6bpw sha `dbd1a6ac…`, matches
  repo). Trap: `deploy.sh` checks out CRLF (`core.autocrlf=true`) and dies with
  `/usr/bin/env: 'bash\r'` — strip CRs (`sed -i 's/\r$//'`) before running it from WSL.
- **T13 tabbyAPI: UPDATED 2026-10-29** (operator: "We haven't started. Update it"):
  `be74bf0` → `15484e2` (55 commits, ff-only, clean tree; my earlier GitHub-API path query
  had silently failed — upstream DID change the backend). Now forwards **`sparse_draft_head`,
  `draft_max_rows`, `hybrid_draft_tokens`, `draft_confidence`** in the draft config — the
  3-kwarg patch is OFF the table entirely. pyproject pins exllamav3 1.6.5 (matches the
  installed local build; venv deps unchanged). Verified: `main.py --help` +
  `import backends.exllamav3.model` OK against engine 1.6.5; launch is `python main.py`
  from the repo dir (the installed `tabbyapi` dist is a metadata stub — `import tabbyapi`
  failing is normal). `-or` decision unchanged: benchmark-only; production sidecar =
  dir-copy (+ optional strip-MTP).
- **T13 step 2 pre-researched (tabbyAPI `be74bf0`, `backends/exllamav3/model.py:1014`):**
  the `AsyncGenerator(...)` call forwards `num_draft_tokens`, `dynamic_draft_tokens`,
  `ngram_match_min` + checkpoint kwargs. NOT forwarded: `sparse_draft_head` (-sdh),
  `hybrid_draft_tokens` (-hdt), `draft_max_rows`; the `_NO_BATCH_READBACK` module flag is
  unreachable without a patch. If `DRAFT_WINNER` uses -sdh/-hdt, deployment needs a 3-kwarg
  tabbyAPI patch — decide at T13.

## Session drift 2026-10-30 (why today's absolutes are lower)

The first perf pass of this session (`t8-stock-a1`, 2026-10-30 05:50) measured **TG 38.38 tok/s at
130816** on the regime the archive used yesterday, against **49.79** for `split376-131k` and 48.87
for `armA2-318`. PP is unchanged (1825 vs 1828 at 131072), so it is decode-side only.

Isolated, each in its own pass:

| variable changed | TG @130816 |
|---|---|
| `t8-stock-a1` (state regime: `-cs 131072`, frozen profile) | 38.38 |
| `t8-cs262` (archive's `-cs 262144`, frozen profile) | 41.25 |
| `t8-repro226` (`-cs 262144`, `-NoFrozenState` — yesterday's exact regime) | 37.47 |

So the cache size explains ~7 % and the frozen profile nothing; the remaining ~15 % is not the
regime. Ruled out while the passes ran: GPU SM 2895–2910 MHz (power 90–240 W of 480, 39–47 °C), CPU
pinned at 4201 MHz, host load ≤ 83 % during decode and ≤ 3 % idle, no other GPU process, identical
load banner (48-layer worker, 16 threads, same host affinity LPs), same `perf.py` ladder. The MTP
head is not in a perf argv (draft args are spec_decode-only), and PP — which exercises the same
weights — is bit-for-bit at yesterday's level.

**Consequence, and the rule for this session:** every number taken today is compared **only inside
this session**. Yesterday's absolutes (and the 49.8 that chose split 376) are not a control, per the
protocol's own rule. Any arm whose verdict rests on a cross-day comparison must be re-measured today.
The cause is still unknown and is itself a finding: it is recorded here rather than absorbed, and a
future session that reproduces 49 tok/s must say which of the two levels its arms came from.

## Tasks

See `wiki/plans/2026-10-29-v165-retune.md` (T0–T13). Done before 2026-10-30: T0
(`T00-preconditions.md`), T1 (`T01-codebase.md`), T2 (`T02-models.md`), T3 (`T03-build.md`),
T4 (`T04-armA.md` — reference + thresholds), T5 (`T05-memops.md` — MEMOPS_WINNER=0).

### Progress 2026-10-30 (this session, all on the win-perf wheel unless noted)

- **T6 done** (`T06-mtp.md`): `-sdh` negative (6 % slower, acceptance 1.17 vs 1.34); `-hdt`
  repetition-only (median per-category 0.99–1.02, mean is one synthetic workload); `-dds`
  unresolved (+2 % broad, inside the reference's own 0.972–1.014 spread). `MTP_WINNER` frozen.
- **T8 done for the code delta** (report pending): win-perf wheel `1422a0a1…` installed over stock
  1.6.5, ABBA `t8-stock-a1` / `t8-fork-b1` / `t8-fork-b2` / `t8-stock-a2` — **null** on pp and tg
  (pp 1.00/0.97, tg 1.04/0.99, inside the ladder's own ±3–5 % jitter). Fork kept (needed for the
  sidecar path, knobs default-off). **T8 step 3 (the four worker knobs) not yet run.**
- **T7 done** (`T07-drafter.md`): reference `t7-ref` 46.48 tok/s / 1.35 accepted. The `-or` sidecar
  path **crashed** (engine bug: `bias_keys()` read `stc.tensor_file_map`, absent on
  `VariantSafetensorsCollection`; fixed in the repo at `330ce0cd`, not in the installed wheel), so
  the sidecar ran through a hardlink dir copy `…-mtpopt` (the production shape). Sidecar 2.6 bpw:
  42.90 / 45.53 tok/s across two runs, acceptance **identical** — closed negative; window 4 made it
  worse (38.67 vs 40.45, 1.69 vs 1.73 accepted). **DFlash2 BF16 at window 4 wins**: 56.57 tok/s,
  2.40/4.00, median per-category 1.258 (translation the one regression, 0.88–0.90). Window 8 is out
  of contract (drafter `block_size` 8, generator asks window+1) and crashed — engine defect #2.
- Harness: `passes.csv` header repadded to its 23 real columns (55 rows now readable); `ratios.py`
  labels pre-`runner` rows `perf`; `verify_task --post --cites-failure <run_id>` added so a report
  can cite the run that proved a defect without it reading as a measurement.
- **T10 done** (`T10-placement.md`): warm ladder 380/376/372/368 all ok — **368 now completes the
  full 131072 pass** (`empty_cache()` between bands removed the allocator death); 376 wins on the
  median (40.38) with 372 (39.93) and 368 (38.43) tied inside a 10 % spread, and 376 chosen for the
  3617 MB headroom the drafter needs.
- **T13 measured, deployment not applied** (`T13-deploy.md`): 262144 + DFlash2 does **not** fit at
  split 376 (`Insufficient VRAM`, 25 s); it fits at **392** and costs 4 % (54.32 vs 56.57). 262144
  *without* a drafter fits at 376 (PP 1839.54 @262144, TG 39.32 @261888). Huge pages: three passes
  each, `=1` median 39.57 vs `=0` 40.37 — **no benefit**, the 12–19 % claim is retired. The
  production `config.yml` is stale (deleted 2.05bpw model, FP16 cache, split 360); the corrected
  block is written in `T13-deploy.md` as a proposal and **not applied** — restarting the server is
  the operator's call.
- **T8 step 3 done** (`T08-fork.md`): **`EXL3_MOE_THREAD_PRIO=1` is +11 % decode** (43.68 vs 39.35
  mean at 130816, two passes within 0.6 % of each other and disjoint from both reference passes,
  bracketed by two other arms at the reference level) and +2 % prefill — the campaign's largest lever
  after the drafter. `EXL3_MOE_TIMER_RES=1` and `EXL3_MOE_CPU_PIN_OFFSET=8` are neutral;
  `EXL3_MOE_BLOCKING_SYNC` stays untested by prior measurement. The fork wheel is kept (null on the
  code delta, needed for `-or`); **the deployed env gains one line: `EXL3_MOE_THREAD_PRIO=1`.**
- **T9 done** (`T09-knobs.md`): all CUDA-side defaults stand. `EXL3_MOE_HOST_CORES=0` costs 14–19 %
  decode, `EXL3_GR_MIX_TILED=0` costs 9 % prefill, `-Threads 15` is +0.8 % decode (inside the spread)
  for 8 % less prefill, `-Threads 14` loses on the wait counter and both metrics; `EXL3_GRAPHS=0`,
  `EXL3_HC_FOLD=1`, `EXL3_SDPA_PREFILL=1` are all inside the reference's spread.
- **T11 done** (`T11-cache.md`): `EXL3_QC_STAGING_PAGES` is **inert at `-chunk_size 4096`** — it is a
  cap (`w = min(npps_w, max(_qc_staging_pages, …))`) and the per-call need is 16 pages, below every
  tested value; the allocator watermarks are byte-identical and the two-pass repeat kills the 1 %
  ordering. `-rs32` costs ~0.7 GB VRAM for nothing. **Step 3 done, and it required wiring first:**
  `eval/longctx.py` is now a harness runner (`-Runner longctx`, `results/longctx.csv`, zero
  `LONGCTX` rows = `fail`, `EXL3_MOE_CPU_SWAP=0`, corpus shipped by `deploy.sh`), scored by
  `bench/longctx_facts.py` (references stripped, Q&A split per question, context-only facts).
  **Q2 retrieval holds: 14/14 facts at 36.9 k tokens and 16/16 at 165.5 k**, the decisive one being
  a verbatim recovery of a single inserted paragraph inside 165 k tokens of Pride & Prejudice.
  Limit: a Q8/FP16 control at 262144 does not fit this card, so this is "Q2 is fine", not "Q2 equals
  FP16".
- **T12 done** (`T12-ple-gdn.md`): `EXL3_PLE_SUB_CHUNK=0` costs ~0.9 GB VRAM and 2 % prefill (default
  1024 and 512 are level); **`ngram_ram` (`-ngr`) is not available** — the 51.8 GB table drove free
  RAM to 637 MB in 27 s and the watcher killed it; `-rs32` costs VRAM for no gain; `EXL3_BC_GDN=0`
  costs 15 % decode; `EXL3_GDN_SUB_CHUNK=1024` is 2–4 % down. No lever here — and three load-bearing
  defaults confirmed.
- Not started: nothing in the measurement plan. **T13 steps 3–4 done 2026-10-31** on operator
  instruction ("start best variant, I wanna test it"): the fully measured configuration is applied
  and the server is **live on `0.0.0.0:4099`** (model 3.18 @ split 376, cache 131072, Q2, DFlash2
  w4, `EXL3_MOE_THREAD_PRIO=1`), boot 34 s, first request 36.5 tok/s with the drafter accepting 37
  of 108 draft tokens on a trivial prompt, floor watcher running (`watch-server.csv`, 200/2048).
  **Updated 2026-10-31 (operator: 262k ctx, max VRAM use):** the server now runs
  **cache 262144 at split 378** — the *lowest* split that loads at that cache (377 refuses; the
  engine's accounting binds ~1 GB before the 200 MB floor does). Full corpus: **58.54 tok/s**,
  2.46/4 accepted, **min free VRAM 763 MB**; 165 k-token prompts: 1545 MB min free, 16/16 retrieval
  facts. That is **+7.8 %** over the earlier 392 recommendation (54.32 tok/s, 2749 MB free) — and a
  one-workload 128-token smoke had said they were equal, so the corpus run decided it.
  End-to-end through the API: 165 433-token prompt prefill **1726 tok/s** (95.8 s), decode 56.6
  tok/s, cached repeat 503 tokens in 9.5 s, min free VRAM 1393 MB, correct answer.
  **Two follow-ups 2026-10-31:** (a) the drafter's VRAM vs the split was measured together for the
  first time — MTP's lighter head loads at split **364** (DFlash2's floor is 378) but the corpus moves
  only 46.48 → 46.77 tok/s (+0.6 %) while the drafting gap is 12 tok/s, so **the split curve is flat
  there and a smaller drafter is not worth it**; (b) the **draft-cache quant was never separated from
  the trunk's** — `model_init` uses one `-cq` for both, so every arm was Q2/Q2. Measured through the
  API instead: Q2 52.2 % vs Q4 50.4 % acceptance on six prompts (mixed signs, identical tok/s) and
  46.4/54.6 % vs 56.5/53.2 % at 165 k — **unresolved either way**, so `"2,2"` stays (half the VRAM).
  **2026-11-01:** the floor watcher killed the server at 00:44 after five healthy hours — it floored
  on *free* RAM (1699 MB) while the working set had not moved (39 111 MB flat) and VRAM was fine
  (1487 MB); the shortfall was the standby list holding the streamed n-gram table. `watch.ps1` now
  floors on **available** RAM (`AvailableMBytes`), column renamed `avail_ram_mb`, passes.csv cell
  `min_avail_ram_mb` (commit `784f8b44`). Server restarted on the same config and serving.
  **2026-11-01 08:52:** killed a second time, on the corrected metric — available RAM 2037 MB
  against the 2048 MB floor, working set 43.0 GB, VRAM fine. That is the workload's normal footprint,
  not a fault. **The production server now runs with no watcher**; the floors stay for measurement
  passes only. Restarted 10:31 UTC, load 32.1 s, `n_ctx=262144`.
  Exact files: `wiki/plans/v165-reports/artifacts/deployed/{config.yml,start_tuned.ps1}`.
  **Two deployment findings:** tabbyAPI has no `Q2` cache mode — it validates
  `FP16|Q8|Q6|Q4` or a numeric pair, so Q2 is spelled `"2,2"` (the first boot died on it); and
  `curl` from WSL to the Windows listener returns `400 Invalid HTTP request received` for a
  well-formed request (GET included) while PowerShell gets 200, so scripted clients must not be
  WSL curl.
