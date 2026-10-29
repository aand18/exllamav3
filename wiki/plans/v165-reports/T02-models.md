# T02 — model swap and load smokes

Date: 2026-10-29. GPU: RTX 4090 24564 MiB, idle at start (0 MiB used, 0 % util).
Harness: deployed `_agent/v165` (post-`94f4c305`, run-id log naming).

## Steps 1–3, 5 (non-GPU, done earlier today)

- 3.68bpw_h6_ng8 downloaded: 8 shards (58.1 GB) + `ngram_embedding.safetensors` (49 GB) + configs.
- 1.66bpw_h4_ng3 downloaded **without** its own ngram table (29 GB).
- ng8 table NTFS-hardlinked into the screening dir: `stat` → **2 links, 51 840 275 600 bytes**
  (same file as the production copy, no second copy on disk).
- Deleted `…-exl3-2.05bpw` and `…-exl3-3.05bpw`; D: ~179 GB free.
  **Deviation, already recorded:** deletions ran before this smoke passed, so there is no
  fallback model on disk. If a model fails to load, stop and report.

## Step 4 — load smokes

### The plan's command does not load: `-mcs 360` is a leftover from the deleted 2.05bpw model

`run_pass.ps1 … -MaxLength 4096 -SkipPrefill -SkipGen -SkipWarmup -NoSpecDec` at the regime's
default split failed in 33 s:

```
RuntimeError: Insufficient VRAM in split for model and cache
```

With `-lv` the loader named the module and the size of the miss:

| split (`-mcs`) | GPU experts/layer | result |
|---|---|---|
| 360 | 152 | OoM at `layers.43`: needs 912 MiB, 1.28 GiB driver-free, 20.47 GiB allocated by torch |
| 372 | 140 | OoM at `layers.45`, same 912 MiB request |
| 392 | 120 | loads and runs, but only **211 MB VRAM free** and 2088 MB RAM free (floors: 200 / 2048) — too tight to stand |
| 400 | 112 | **loads** — "largest measured transient 934 MiB, 2216 MiB free after load" |

The 3.68 model carries ~1.8× the bits of the 2.05bpw model that 360 was tuned on, so it needs
fewer experts on the card. **Arm A split = 400** (the plan's own T4 escape hatch: "record the
split that made it fit as the Arm A split"). The load boundary lies between 372 and 400 — T10's
ladder should search 376…400, not upward.

### Harness quirk: a skip-everything smoke can never report `ok`

`run_pass.ps1` marks a pass `fail` when it has no PP and no TG rows, so the plan's
`-SkipPrefill -SkipGen -SkipWarmup` smoke always ends `status=fail` even on a clean load. The
load verdict therefore comes from the log (no traceback, `free after load`, arena size), and the
smokes below add a real 4096-token pass so the harness can also agree.

### Results — both models load and generate

| pass | model | split | status | PP @4096 | TG (S=1) | min free VRAM | min free RAM | breach |
|---|---|---|---|---|---|---|---|---|
| `smoke368-4k` | 3.68bpw | 400 | ok | 1535.41 t/s | 27–36 t/s | 3329 MB | 5392 MB | 0 |
| `smoke166-4k` | 1.66bpw | 360 | ok | 2254.25 t/s | 37–55 t/s | 8731 MB | 30949 MB | 0 |

`smoke368c` (load-only, split 400): CPU MoE arena **43.02 GB** locked in RAM, engine warmup 16/16
in 68 s, 2216 MiB VRAM free after load.

### Ngram table check (the plan asked for a load-log line; the loader prints none)

`ngram_embedding` is host-side (`modules/ngram_embedding.py`, `prefer_cpu`, optional disk
streaming) and silent in the log. The proof that the screening dir's **hardlink** is what loads is
stronger than a log line: the table file itself carries the aux tensors the loader requires
unconditionally — `head_offsets`, `head_vocab_sizes`, `layer_multipliers`, `head_bias` (read with
`optional=False`, so a missing file raises). The 1.66 dir holds no other copy of those tensors,
and it loaded, so the hardlinked table was opened and used. Corroborated by RAM: the table never
appears in RSS (1.66 pass kept 30 GB free), i.e. it streams from disk.

## Consequence worth flagging for deployment (T13)

At split 400 the CPU arena is **43 GB** of locked RAM. Total system RAM is 63.6 GB, and this
box's resident inference server (`unbiased-strata.exe`) normally holds 36–41 GB. So the tuned
production config and that server cannot both be resident — the 3.68bpw model at 262144 ctx does
not fit a 24 GB card without ~43 GB of CPU-resident experts.

### The model sits on a knife edge in *both* resources

`armA-1` (first attempt, same split that had just loaded) died at 32 s in the engine's
**host-memory guard**, not on VRAM:

```
CPU MoE expert arena chunk 39 (40064 MiB in total) needs 1024 MiB of host memory,
but only 2965 MiB is available and EXL3_HOST_MEM_RESERVE_MB = 2048 MiB is kept free.
```

It missed by **107 MB**. The watcher trace shows why the numbers are so tight: the 43 GB arena
lives in the engine's separate CPU-MoE worker process, so free RAM drops from 50.9 GB at start to
3.3 GB at the last chunk while the watched `perf.py` process shows only 3.2 GB working set.

Both constraints are already at their edge, and they pull opposite ways: fewer CPU experts (lower
`-mcs`) frees host RAM but overruns VRAM, more CPU experts (higher `-mcs`) frees VRAM but overruns
host RAM. Ruling: keep `-mcs 400` and set **`EXL3_HOST_MEM_RESERVE_MB=1024`** in the production
regime (it is a load-safety threshold, not a perf knob). The harness watcher independently
enforces a 2 GB free-RAM floor and kills the tree on breach, so the engine default double-counted
the same margin; one enforced margin remains. Cost if wrong: a pass is killed by the watcher
instead of aborting cleanly — the machine is not at risk.

## Log durability (operator requirement, 2026-10-29)

`run_pass.ps1` used to `Remove-Item perf-<Tag>.log` before each run, so a repeated tag destroyed
the earlier evidence. Fixed in `94f4c305`:

- every run gets `runId = <Tag>-<UTC yyyymmdd-HHmmss>`; log, err and watcher CSV names carry it,
  and the harness aborts rather than overwrite;
- `passes.csv` / `pp.csv` / `tg.csv` gain a `run_id` column, `passes.csv` gains the log filename,
  so every row points at its evidence;
- `bench/windows/archive_logs.sh` copies logs, CSVs and BREACH files into
  `wiki/plans/v165-reports/artifacts/` (git-tracked) and snapshots the cumulative CSVs with a
  UTC stamp. The T2-era smoke logs and the pre-`run_id` CSVs are archived there.
