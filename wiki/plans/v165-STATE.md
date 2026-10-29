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
  (batched readback ON by default). T6's env A/B is impossible; needs a source patch
  or the arm drops — decide before T6.

## Models (operator decision 2026-10-29; T2 downloads, then old 2.05bpw deleted)

- **production:** `D:\llms\Qwen3.8-Flash-Next-exl3-3.68bpw_h6_ng8`
  (turboderp/Qwen3.8-Flash-Next-exl3@3.68bpw_h6_ng8; 62.2 GB model + 51.8 GB
  `ngram_embedding.safetensors`)
- **screening:** `D:\llms\Qwen3.8-Flash-Next-exl3-1.66bpw_h4_ng3` (same repo @1.66bpw_h4_ng3,
  30.2 GB, its ng3 table NOT downloaded; the ng8 table is NTFS-hardlinked in). Fast loads,
  more experts in VRAM — screening ratios only, never absolute tok/s; placement/cache/drafter-
  VRAM arms and all confirmations run on the production model.
- **T2 status 2026-10-29: steps 1–3 done** — both downloads complete (3.68 dir 107 GB incl.
  ng8; 1.66 dir 29 GB; hardlink verified, link count 2). Steps 4 (load smoke, GPU) and 5
  (delete `2.05bpw` 59 GB + `3.05bpw` 80 GB) pending. Disk after downloads: ~37 GB free.
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
