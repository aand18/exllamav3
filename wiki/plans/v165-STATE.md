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
- installed engine (verified T0, 2026-10-29): **1.6.4** — not the expected 1.5.4 (a 1.6.4 wheel
  is already installed; torch 2.11.0+cu130). The 1.5.4 wheel path above is the only 1.5.4
  artifact; T3 installs stock 1.6.5 from source regardless.
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
- **T2 status 2026-10-29: steps 1–3, 5 done** — both downloads complete (3.68 dir 107 GB incl.
  ng8; 1.66 dir 29 GB; hardlink verified, link count 2); step 5 deletions done 2026-10-29
  (`2.05bpw` + `3.05bpw` deleted, D: 171 GB free). Step 4 (load smoke, GPU) pending.
- Rollback: 1.5.4 wheel on disk (path above) AND the 1.6.4 wheel at
  `C:\Users\yoho\Downloads\exl3-build-164\src\dist\exllamav3-1.6.4-cp312-cp312-win_amd64.whl`
  (earlier "not on disk" was wrong — checked the wrong dist dir).

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
- MoE: split 360 provisional (was tuned for the deleted 2.05bpw; T10 re-founds, ladder up),
  16 threads, `EXL3_MOE_MEMOPS=0` (T5 re-decides), `EXL3_MOE_CPU_PIN=1`, `EXL3_MOE_CPU_SWIZZLE=1`
- drafting: MTP draft block 3 provisional; T6/T7 settle `DRAFT_WINNER` (stock MTP / HermiHg
  sidecar / DFlash2), then every tg row carries it
- NGRAM: table present (ng8), `ngram_ram` off (T12 tests), vision: off

## Frozen winners (fill as tasks complete)

- MEMOPS_WINNER: — (T5)
- MTP_WINNER (readback/dnt/sdh/hdt/dds): — (T6)
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
`EXL3_NO_BATCH_READBACK` flag for T6. Next: T2 (steps 4–5, GPU).

## Non-GPU prep done 2026-10-29

- **T3 build: DEFERRED — machine has no RAM headroom.** Two attempts under the watcher
  (`build-stock` MAX_JOBS=4: breach at 368 s, free RAM 1999; `build-stock2` MAX_JOBS=2: breach
  at 96 s, free RAM 1713 — the operator's workload holds ~62/65 GB RAM + 23.5 GB VRAM right
  now). The watcher killed both cleanly; nothing of the operator's was touched. Build both
  wheels when the operator frees the machine (stock first, then winperf — builds serialize).
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
- **T13 step 2 pre-researched (tabbyAPI `be74bf0`, `backends/exllamav3/model.py:1014`):**
  the `AsyncGenerator(...)` call forwards `num_draft_tokens`, `dynamic_draft_tokens`,
  `ngram_match_min` + checkpoint kwargs. NOT forwarded: `sparse_draft_head` (-sdh),
  `hybrid_draft_tokens` (-hdt), `draft_max_rows`; the `_NO_BATCH_READBACK` module flag is
  unreachable without a patch. If `DRAFT_WINNER` uses -sdh/-hdt, deployment needs a 3-kwarg
  tabbyAPI patch — decide at T13.
