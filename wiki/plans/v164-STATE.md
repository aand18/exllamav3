# v1.6.4 retune — live state

Owner of the box: this session (2026-10-24). Rollback wheel:
`/mnt/c/Users/yoho/Downloads/exl3-wheel-154/exllamav3-1.5.4+cu132.torch2.11.0-cp312-cp312-win_amd64.whl`.

## Harness

Deployed from `bench/windows/` (repo) to `C:/Users/yoho/Downloads/tabbyAPI/_agent/v164/`.
Validated 2026-10-24 with `run_pass.ps1 -Tag smoke2 -MaxLength 4096 -SkipGen -SkipWarmup -NoSpecDec`
(status `ok`, engine 1.5.4, ext `exllamav3_ext.cp312-win_amd64.pyd`, floors held, no breach).
Harness sha256: `f264438c7a1cdce715ebda1809dc3dd1226e773049f5ca5ffc953afd9b2a4958`.

## Regime (production, 2026-10-24 baseline)

- model: Qwen3.8-Flash-Next exl3 2.05 bpw (`D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw`)
- cache: 262144, FP16, chunk 4096, batch 2
- MoE: split 360 / 512, 16 threads, `EXL3_MOE_MEMOPS=0`, `EXL3_MOE_CPU_PIN=1`, `EXL3_MOE_CPU_SWIZZLE=1`
- MTP: draft block 3 (`-sd`)
- NGRAM: off, vision: off

## Advise-only (do not change without a task)

- `EXL3_MOE_MEMOPS` (Task 5 decides)
- `EXL3_MOE_STREAM_T`, `EXL3_MOE_STREAM_BATCH_EXPERTS` (Task 6)
- QSA attention cap 2048
- KVaRN (out)

## Tasks

See `wiki/plans/2026-10-24-v164-retune.md`. Task 0 (deploy + self-check + state) done.
Task 1 (worktrees) and Task 2 (stock baseline) pending.
