# v1.6.5 retune — live state

Owner of the box: none yet — ask the operator before the first GPU task. Rollback wheel:
`/mnt/c/Users/yoho/Downloads/exl3-wheel-154/exllamav3-1.5.4+cu132.torch2.11.0-cp312-cp312-win_amd64.whl`.

## Code base (verified 2026-10-29)

- upstream `v1.6.5` = `cdc785db`; stock worktree `/home/dev/exllamav3/.worktrees/v165-stock`
  (branch `v165-stock`).
- fork rebase: branch `165-win-perf`, tag `win-perf-1.6.5` = v1.6.5 + 4 commits
  (worker knobs `5cef00e1`, .gitignore, Specter docs, harness `68eea9dc`). Rebased clean onto
  v1.6.5; the four knobs (`EXL3_MOE_TIMER_RES/_THREAD_PRIO/_BLOCKING_SYNC/_CPU_PIN_OFFSET`) are
  wired in the rewritten pool code. Build worktree: `/home/dev/exllamav3/.worktrees/v165-win-perf`
  (detached at the tag).
- installed engine (expected, confirm in T0): 1.5.4.

## Harness

Repo source of truth: `bench/windows/` on `win-perf-1.6.5`. Deploy target:
`C:/Users/yoho/Downloads/tabbyAPI/_agent/v165/` via
`DEST=/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165 bench/windows/deploy.sh`;
always pass `-Root "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165"`.
`perf.py` unchanged v1.6.4→v1.6.5, sha256 `246e721a62d7c5238de4b7296adb0b26cd9f42a734b9f648436b35b9b6a29e04`.

## Regime (production, 2026-10-29 baseline)

- model: Qwen3.8-Flash-Next exl3 2.05 bpw (`D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw`), arch `qwen4_exp`
- cache: 262144, FP16 (quantized allowed from T8), chunk 4096, batch 2
- MoE: split 360 / 512, 16 threads, `EXL3_MOE_MEMOPS=0` (T4 re-decides), `EXL3_MOE_CPU_PIN=1`,
  `EXL3_MOE_CPU_SWIZZLE=1`
- MTP: draft block 3 (`-sd`)
- NGRAM: off, vision: off

## Frozen winners (fill as tasks complete)

- MEMOPS_WINNER: — (T4)
- CODE_WINNER (stock vs win-perf): — (T5)
- THREADS/HOST_CORES: — (T6)
- SPLIT: — (T7)
- CACHE_MODE + STAGING_PAGES: — (T8)
- DRAFT (readback/sdh/dnt/hdt/dds + head: stock / sidecar-2.6 / DFlash2): — (T9, T9b)
- PLE/GDN: — (T10)

## Advise-only (do not change without a task)

- large pages / `SeLockMemoryPrivilege` (no privilege on this box)
- WDDM memop handshake (measured 8–11% slower on 1.4.9)
- `EXL3_MOE_BLOCKING_SYNC` (~4% slower with MTP host syncs)
- `EXL3_MOE_ZERO_COPY` (gone), KVaRN (operator decision)
- QSA attention cap 2048

## Tasks

See `wiki/plans/2026-10-29-v165-retune.md`. T0 pending.
