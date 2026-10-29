# v1.6.5 retune — live state

Owner of the box: none yet — ask the operator before the first GPU task. Rollback wheel:
`/mnt/c/Users/yoho/Downloads/exl3-wheel-154/exllamav3-1.5.4+cu132.torch2.11.0-cp312-cp312-win_amd64.whl`.

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
  **Deployment consequence for T13:** the tuned config needs ~43 GB of host RAM for the CPU
  arena, so it cannot share the box with a resident 36–41 GB inference server — and
  `EXL3_HOST_MEM_RESERVE_MB=0` plus 100/1200 floors must be part of the served configuration.
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

## Harness

Repo source of truth: `bench/windows/` on `165-win-perf`. Deploy target:
`C:/Users/yoho/Downloads/tabbyAPI/_agent/v165/` via
`DEST=/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165 bench/windows/deploy.sh`;
always pass `-Root "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165"`. T0 done 2026-10-29:
`run_pass.ps1` carries the `-Model` override (commit `09923498`; the plan's literal placement
was a no-op — `$Model`/`$MODEL` are the same case-insensitive variable — fixed as a conditional
default) and `-cq 2,2` in the production regime. Deployed `run_pass.ps1` sha256 `01adc66f…`. `perf.py` unchanged v1.6.4→v1.6.5, sha256
`246e721a62d7c5238de4b7296adb0b26cd9f42a734b9f648436b35b9b6a29e04`.

## Regime (production baseline, 3.68bpw model)

- cache: 262144, **Q2 (`-cq 2,2`) — operator-fixed 2026-10-29, used in every pass**, chunk
  4096, batch 2
- MoE: **split 400** (not 360 — 360/372 do not load the 3.68 model; 392 dies at the 65536 band),
  16 threads, `EXL3_MOE_MEMOPS=0` (T5 re-decides), `EXL3_MOE_CPU_PIN=1`, `EXL3_MOE_CPU_SWIZZLE=1`
- host RAM / determinism: `EXL3_HOST_MEM_RESERVE_MB=0` (the watcher holds the floor) and
  `EXL3_MOE_ARENA_HUGEPAGE=0` (page backing must not vary between runs); watcher floors
  **100 MB VRAM / 1200 MB RAM** (operator 2026-10-29)
- **measurement length 131072**; 262144 runs but is not reproducible (coin flip on VRAM), so it
  is a single final confirmation only
- drafting: MTP draft block 3 provisional; T6/T7 settle `DRAFT_WINNER` (stock MTP / HermiHg
  sidecar / DFlash2), then every tg row carries it
- NGRAM: table present (ng8), `ngram_ram` off (T12 tests), vision: off

## Frozen winners (fill as tasks complete)

- **Arm A reference (T4, 2026-10-29):** 3.68bpw, split 400, 131072, Q2, no drafting,
  guard 0, huge pages off. A1 PP 1637.69@4096 / 1632.40@131072; A2 1602.18 / 1642.92.
  **Acceptance threshold (A2/A1): pp +-2 % at 4096–131072 (small bands +-5–10 %), tg +-11 % worst
  band (65536), typically +-5 %.** Report `T04-armA.md`; ratios in `artifacts/analysis/`.
- **MEMOPS_WINNER: 0** (T5, ABBA 2026-10-29: memops=1 is 8–17 % slower on tg at every
  context, medians 0.83–0.93, pp neutral-to-worse; reproduces the 1.4.9 finding). Deployment
  value unchanged; no 262144 confirmation (nothing to deploy changes).
- MTP_WINNER (readback/dnt/sdh/hdt/dds): readback arm inconclusive on the screening model
  (T06 report) — **default batched (`_NO_BATCH_READBACK=False`) stands**; knobs move to the
  production model, because acceptance is a body/head quality property: 1.66 model = 1.91 bpw
  body / 4.01 bpw head → 1.0–2.2 of 4 accepted (break-even), 3.68 = 4.00 / 6.01 → measuring now.
- DRAFT_WINNER (stock / sidecar-2.6 / DFlash2 + window + sdh): — (T7)
- CODE_WINNER (stock vs win-perf): — (T8)
- THREADS/HOST_CORES: — (T9)
- SPLIT: — (T10)
- CACHE_MODE: **Q2 (2,2) — operator decision, frozen**
- STAGING_PAGES: — (T11)
- PLE/GDN: — (T12)

## Advise-only (do not change without a task)

- large pages / `SeLockMemoryPrivilege` (no privilege on this box)
- WDDM memop handshake (measured 8–11% slower on 1.4.9)
- `EXL3_MOE_BLOCKING_SYNC` (~4% slower with MTP host syncs)
- `EXL3_MOE_ZERO_COPY` (gone), KVaRN (operator decision)
- QSA attention cap 2048

## VRAM relief valve (standing rule)

When a component (drafter, cache, staging) does not fit: raise `-mcs` (more experts to system
RAM) and record the tg cost of the offload in the same row; only if the offload itself breaches
the RAM floor is the component shrunk instead.

## Tasks

See `wiki/plans/2026-10-29-v165-retune.md` (T0–T13). T0 done 2026-10-29
(`wiki/plans/v165-reports/T00-preconditions.md`); harness commit `09923498`.
T1 done 2026-10-29 (`wiki/plans/v165-reports/T01-codebase.md`) — pass, with the
`EXL3_NO_BATCH_READBACK` flag for T6. T2 done (`T02-models.md`), T3 done (`T03-build.md`),
**T4 done (`T04-armA.md`)** — reference arm + acceptance thresholds frozen. T5 done (`T05-memops.md`) — MEMOPS_WINNER=0.
T6 in flight (`T06-mtp.md`): readback settled as inconclusive (default kept), drafting knobs now
measured on the 3.68 model with `-cs 32768` (two caches exist while drafting; workloads are
short). Next: T7 (drafter), T8 (code delta), T9–T13.

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
