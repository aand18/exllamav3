# Windows-side measurement harness

Everything that measures this box lives here, in the repo, versioned. One place to fix a bug, one
place to read the contract, no new `.ps1` per benchmark.

The box is a Windows 11 / RTX 4090 / 7950X3D / 64 GB machine running tabbyAPI and exllamav3. The
engine is built as a wheel for the tabbyAPI venv, so every measurement is "install a wheel, boot the
model, run `eval/perf.py`". These scripts make that repeatable.

## Inventory

| File | Purpose | Status |
|---|---|---|
| `watch.ps1` | samples free VRAM/RAM every second, writes a CSV, kills the process tree and writes `BREACH-<tag>.txt` when a floor is crossed | in use, mandatory |
| `run_pass.ps1` | one `perf.py` pass in a named regime, under `watch.ps1`, parsed into CSVs, prints the `HARNESS/PASS/FLOORS/PP/TG_S1` block | in use |
| `run_build.ps1` | one wheel build under `watch.ps1`, records the wheel and its size | in use |
| `deploy.sh` | copies this directory plus `eval/perf.py` to the directory the box runs from | in use |
| `legacy/` | the v1.4.9-era scripts, kept for provenance | superseded, see `legacy/README.md` |

## Floors: non-negotiable

Every run — pass, build, boot, deployment smoke test — holds **200 MB free VRAM** and **2048 MB
free system RAM** at every sample. `watch.ps1` is the only sanctioned way to enforce that, and a run
whose report has no `FLOORS` line is not evidence. On breach the watcher kills the process tree,
writes `results/BREACH-<tag>.txt`, and `run_pass.ps1` returns status `breach` (exit 4): the numbers
are void, not "a bit low".

## Deploy

```bash
bench/windows/deploy.sh                      # repo -> C:/Users/yoho/Downloads/tabbyAPI/_agent/v165
DEST=/mnt/c/other bench/windows/deploy.sh    # elsewhere
```

The deployed copy is an artifact. Edit here, deploy, then run. Both scripts print
`HARNESS sha256=...` and record it in `results/passes.csv`, so a number can always be traced to the
exact script that produced it.

## Run a pass

From WSL, with a Windows path (PowerShell will not take `/mnt/c/...` for `-File`):

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v164\run_pass.ps1" -Tag armA-1 -MaxLength 262144
```

| Switch | Meaning |
|---|---|
| `-Tag` | names every output file; one tag per pass, unique |
| `-MaxLength` | highest band measured. `131072` screens (~10 min), `262144` confirms (~20-25 min) |
| `-Env "K=V,K=V"` | overrides the regime environment |
| `-Unset "K,K"` | removes inherited variables |
| `-Threads N` | replaces the regime's `-mct` (default 16) |
| `-Split N` | replaces the regime's `-mcs` (default 360) |
| `-Extra "-cq Q8"` | appends raw `perf.py` flags (`-cq`, `-rcs`, …) |
| `-NoSpecDec` | skip the S=1..4 decode grid (prefill-only decisions) |
| `-SkipPrefill` / `-SkipGen` / `-SkipWarmup` | `perf.py`'s `-spf` / `-sg` / `-swu` |
| `-Prof` | turn the v1.4.9-era profilers on (diagnosis, not tok/s) |

The `production` regime is fixed in `run_pass.ps1`: model 2.05bpw, `-mcs 360 -mct 16 -cs 262144
-chunk_size 4096 -ambs 2`, decode at S=1..4, `EXL3_MOE_MEMOPS=0`, `EXL3_MOE_CPU_PIN=1`,
`EXL3_MOE_CPU_SWIZZLE=1`, profilers off. Add a regime by editing the table rather than by adding
flags to a command line.

`perf.py` measures prefill at `256, 512, … max_length` and decode at `0, 256, … max_length-256`
with `S=1..4`; `S=1` is the tg number, `S=2..4` are the per-step cost of MTP verification, not an
acceptance rate (that is `eval/spec_decode.py`).

## Outputs

| Path | Contents |
|---|---|
| `results/passes.csv` | `tag, finished_utc, regime, status, engine, ext, elapsed_s, max_length, min_free_vram_mb, min_free_ram_mb, peak_proc_ws_mb, samples, breach, env, split, threads, args, harness_sha256` |
| `results/pp.csv` | `tag, band, pp_tok_s` |
| `results/tg.csv` | `tag, context, s1, s2, s3, s4` |
| `results/watch-<tag>.csv` | `t_iso, free_vram_mb, used_vram_mb, free_ram_mb, proc_ws_mb` |
| `results/builds.csv` | `tag, finished_utc, src, exit, elapsed_s, wheel, wheel_mb, harness_sha256` |
| `logs/perf-<tag>.log` | `perf.py` stdout+stderr, identity header appended last (partial metrics survive a kill) |
| `results/BREACH-<tag>.txt` | floor violation: values, action taken |

`status` is `ok`, `fail` (traceback or no metrics) or `breach`.

## Traps this harness encodes

All four were measured on this box on 2026-10-24; each one silently corrupts a benchmark.

1. **The venv `python.exe` is a uv shim.** `Start-Process` returns the shim (3 MB working set), which
   spawns the real interpreter (`C:\Python312\python.exe`) as its child. Watching or killing the
   shim measures/kills nothing, and the shim's exit code is unreadable. `watch.ps1 -MatchCmd`
   resolves the real process by command line; `run_pass.ps1` derives the verdict from parsed
   metrics plus a traceback scan instead of an exit code.
2. **A WSL working directory leaks into Python's import path.** Launch PowerShell from
   `/home/dev/exllamav3` and `import exllamav3` resolves through `\\wsl.localhost\...` to the WSL
   source tree, mixing a 1.4.9 package with the installed extension. `run_pass.ps1` changes to a
   Windows directory before any Python call and **aborts** if `exllamav3.__file__` is not the venv's
   `site-packages`.
3. **`rich` progress output.** Batches are interleaved with ANSI redraws and `\r`; the parser strips
   ANSI, splits on carriage returns, and matches the band lines without anchoring.
4. **Decimal commas.** The box can run with a comma-decimal locale; every number written to CSV is
   formatted with the invariant culture.

## Relationship to the plan

`wiki/plans/2026-10-24-v164-retune.md` drives these scripts task by task (one subagent per task).
Task 0 deploys and self-checks the harness, Task 2 validates a pass end to end on the installed
engine before any build, and Tasks 3-13 only take `-Tag`/`-MaxLength`/`-Env`/`-Threads`/`-Split`/
`-Extra` combinations — no new scripts.
