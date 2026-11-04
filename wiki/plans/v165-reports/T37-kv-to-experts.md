# T37 — moving resident KV into GPU experts: the arm breaches the VRAM floor

Campaign of 2026-11-04, one session, server stopped, production regime otherwise unchanged
(`split 381`, `cs 262144`, `cq 4,4`, DFlash2 window 4), `-dr 3`, the branch engine's Python files
installed file-level in the venv (md5 `2d65860…`, `a8a9090…`, `ee16ca7…`, `7b9b705a…`).

## What the arm was meant to test

T31's M1 lever: at `cache_size 262144` the cache holds **2 911 MiB** of VRAM, 2 688 of them KV. The
Lever drops the cache to `32768`, moves the cold prefix pages to a 2 GB host KV second tier
(`-ccs 2.0`), and spends the freed VRAM on **27 more GPU experts per layer** (`split 381 → 354`,
i.e. 131 → 158 of 512). The question was whether more resident experts beat a long VRAM cache at the
context this box actually serves.

## Geometry, before spending the card

`bench/size_check.py` on the checkpoint's own tensor shapes:

| configuration | trunk | cache | total | free |
|---|---|---|---|---|
| production `-mcs 381 -cs 262144` | 17 551 MiB | 2 911 MiB | 20 462 MiB | 2 054 MiB |
| arm `-mcs 354 -cs 32768` | 19 916 MiB | 559 MiB | 20 475 MiB | 2 041 MiB |

The model called the move **VRAM-neutral**, which is what the M1 arithmetic predicted: 2 352 MiB of
KV out, 2 365 MiB of experts in.

## What happened

Order **A B B A**: **A** = production (`-cs 262144 -mcs 381`), **B** = `-cs 32768 -mcs 354 -ccs 2.0`,
both `-MaxLength 131072 -Extra "-ccs 2.0 -dr 3"` for B and `-dr 3` for A.

| run | status | elapsed | note |
|---|---|---|---|
| `t37-A1-production-20261104-183412` | **ok** | 391 s | reference: pp 131072 **1899.21**, tg 130816 **36.22** |
| `t37-B1-kv32k-mcs354-20261104-184516` | **breach** | 222 s | `min_free_vram_mb=23` against the 200 MB floor; pp ladder reached 32768, tg reached 2048 |
| `t37-B2-kv32k-mcs354` | aborted | — | one-owner guard: a pytest process from the fix-round gate was running |
| `t37-A2-production` | aborted | — | same guard, same second |

The watcher killed B after 222 s with **23 MiB** of VRAM free, where the size model said 2 041 MiB.
Roughly **2 GB** the model does not account for.

## Where the model is wrong, and what it cannot know

- `size_check.py` does not model `-ccs` (the host KV second tier) at all: there is no argument for it
  and no term for it. Whether a host tier costs VRAM (a staging ring, a pinned spill buffer) is
  exactly the unknown that made this arm look free.
- The QSA indexer planes are documented as **not** scaling with cache width, so the arm's 559 MiB
  cache figure is itself optimistic by the plane bytes at `32768` (the model computes 336 MiB of KV
  plus recurrent 223, and treats the planes as part of the scaling term elsewhere).
- The 27 extra experts' aux copies (bias / `suh` / `svh` per layer) are not in the trunk term either.

None of those three is measured here. What is measured is the outcome: **with `-ccs 2.0` and
`split 354` at this regime, the configuration does not fit the floor.**

## Consequence

- The **KV→experts move is not deployable on this evidence**, and the arm that would test it cannot
  run at the production context until the host-KV mechanism's VRAM cost is known.
- The cheapest next step is not another 5-minute pass but a **five-line fix to `size_check.py`**: take
  `--ccs` and either model its VRAM cost from the first measurement or refuse to certify a
  configuration that uses it ("unknown cost, do not certify"), so the pre-flight cannot promise FIT
  for a configuration it cannot price.
- The M1 arithmetic itself is untouched: 131 → 158 experts is still +27/layer for 2 365 MiB, and the
  prefill and decode numbers of the arm are unknown, not negative.

## How to re-run

```
bench/size_check.py --model /mnt/d/llms/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 --mcs 381 --cs 262144 --cq 4,4
bench/size_check.py --model /mnt/d/llms/Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8 --mcs 354 --cs 32768  --cq 4,4
run_pass.ps1 -Tag t37-B1 -CacheSize 32768 -Split 354 -MaxLength 131072 -Extra "-ccs 2.0 -dr 3"
```

The gate for this report: `bench/verify_task.py --post --report wiki/plans/v165-reports/T37-kv-to-experts.md`
(run from the `165-win-perf` worktree, where the harness's CSVs are resolved).
