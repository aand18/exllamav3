# Windows-side measurement harness

Everything that measures this box lives here, in the repo, versioned. One place to fix a bug, one
place to read the contract, no new `.ps1` per benchmark.

The box is a Windows 11 / RTX 4090 / 7950X3D / 64 GB machine running tabbyAPI and exllamav3. The
engine is built as a wheel for the tabbyAPI venv, so every measurement is "install a wheel, boot the
model, run `eval/perf.py`". These scripts make that repeatable.

## Inventory

| File | Purpose | Status |
|---|---|---|
| `watch.ps1` | samples free VRAM/RAM every second, writes a CSV, kills the process tree and writes `BREACH-<tag>-<utc>.txt` when a floor is crossed | in use, mandatory |
| `run_pass.ps1` | one `perf.py` pp/tg pass, one `spec_decode.py` drafting pass or one `longctx.py` retrieval pass, in the regime `state.json` names, under `watch.ps1`, parsed into CSVs; prints the `HARNESS/PASS/FLOORS/PP/TG_S1` block, one `ARM` line per drafting arm and one `LONGCTX` line per retrieval test | in use |
| `run_build.ps1` | one wheel build under `watch.ps1`, records the wheel and its size | in use |
| `deploy.sh` | copies the scripts, `state.json`, the sidecar specs, `eval/perf.py`, `eval/spec_decode.py`, `eval/longctx.py` and both corpora (`prompts/`, `eval_texts/`) to the directory the box runs from | in use |
| `legacy/` | the v1.4.9-era scripts, kept for provenance | superseded, see `legacy/README.md` |

## Floors: non-negotiable

Every run — pass, build, boot, deployment smoke test — holds **100 MB free VRAM** and **1200 MB
free system RAM** at every sample: the operator's floors, `floors` in `state.json` for a pass and
`-MinFreeVramMB`/`-MinFreeRamMB` for a build. `watch.ps1` is the only sanctioned way to enforce
that, and a run whose report has no `FLOORS` line is not evidence. On breach the watcher kills the
process tree, writes `results/BREACH-<tag>-<utc>.txt`, and `run_pass.ps1` returns status `breach`
(exit 4): the numbers are void, not "a bit low".

## Deploy

```bash
bench/windows/deploy.sh                      # repo -> C:/Users/yoho/Downloads/tabbyAPI/_agent/v165
DEST=/mnt/c/other bench/windows/deploy.sh    # elsewhere
```

The deployed copy is an artifact. Edit here, deploy, then run. Both scripts print
`HARNESS sha256=...` and record it in `results/passes.csv`, so a number can always be traced to the
exact script that produced it.

`deploy.sh` ships `state.json` on every run and prints `state.json: updated` or
`state.json: unchanged`: the repo copy is the regime, so a box left on an earlier deploy measures a
different pass from the one the plan describes, and `bench/verify_task.py --pre` fails on the
disagreement rather than on the measurement.

## Run a pass

From WSL, with a Windows path (PowerShell will not take `/mnt/c/...` for `-File`):

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag armA-1 -MaxLength 262144
```

| Switch | Meaning |
|---|---|
| `-Tag` | names every output file and the run id (`<tag>-<UTC>`); one tag per pass, unique |
| `-MaxLength` | highest band measured; `0` takes `max_length` from `state.json`. `131072` screens (~10 min), `262144` confirms (~20-25 min) |
| `-Env "K=V,K=V"` | overrides the regime environment |
| `-Unset "K,K"` | removes inherited variables |
| `-Threads N` | replaces the regime's `-mct`; `0` takes `threads` from `state.json` |
| `-Split N` | replaces the regime's `-mcs`; `0` takes `split` from `state.json` |
| `-CacheSize N` | replaces the regime's `-cs`; `0` takes `cache_size` from `state.json`, or `draft.spec_decode_cache_size` for a drafting run |
| `-Model PATH` | replaces the regime's `-m`; empty takes `model` from `state.json` |
| `-Runner perf` / `-Runner spec_decode` / `-Runner longctx` | one `perf.py` pp/tg ladder (default), one `spec_decode.py` drafting pass, or one `longctx.py` retrieval pass |
| `-Extra "..."` | appends raw runner flags (`-dr 3`, `-rb_ab`, …). It may not restate a flag the regime owns — `-m`, `-cs`, `-cq`, `-chunk_size`, `-ambs`, `-max_length`, `-mcs`, `-mct` and, on a drafting run, `-mtp`/`-ndt`/`-sdh`/`-hdt`/`-dds` — in any spelling (`-cs=40960`, `--cache_size 40960` included); a restatement aborts, naming the flag and the way to change it: the delta switch where one exists (`-Model`, `-CacheSize`, `-MaxLength`, `-Split`, `-Threads`), `edit state.json` for the rest (`-cq`, `-chunk_size`, `-ambs` and the drafting flags) |
| `-StaticSwap` | applies `state.json`'s `knob_arms_disable_swap` to this pass (`EXL3_MOE_CPU_SWAP=0`); a drafting run gets it without asking |
| `-DryRun` | prints the `DRYRUN`/`ARGS`/`ENV` lines and exits 0: no log, no CSV row, no watcher, no frozen-profile restore, no output directory, no GPU, and no one-owner check — so it works while another pass or the server owns the card |
| `-NoSpecDec` | skip the S=1..4 decode grid (prefill-only decisions) |
| `-SkipPrefill` / `-SkipGen` / `-SkipWarmup` | `perf.py`'s `-spf` / `-sg` / `-swu` (a `perf.py` pass only) |
| `-Prof` | turn the v1.4.9-era profilers on (diagnosis, not tok/s) |

Every switch is a delta: the regime lives in `state.json` — model, `split`, `threads`,
`cache_size`, `cache_quant`, `env`, `floors`, `draft`. The active regime is 3.18bpw, decode at
S=1..4, with `-mcs 376 -mct 16 -cs 131072 -chunk_size 4096 -ambs 2 -cq 2,2`, `EXL3_MOE_MEMOPS=0`,
`EXL3_MOE_CPU_PIN=1`, `EXL3_MOE_CPU_SWIZZLE=1`, `EXL3_HOST_MEM_RESERVE_MB=0`,
`EXL3_MOE_ARENA_HUGEPAGE=0` and profilers off. `-Regime production` is accepted as an alias for the
state; a regime change is an edit to that file, not a flag added to a command line.

Before every run `run_pass.ps1` points `XDG_CACHE_HOME` at `frozen_state_dir` and refreshes it from
`frozen_state_master`, so each arm starts from the same routed-expert and token profile instead of
inheriting what the previous run learned (`-NoFrozenState` opts out).

Drafting is `-Runner spec_decode` only, and its flags come from `state.json` too: `draft.mode: mtp`
becomes `-mtp`, `num_draft_tokens` becomes `-ndt`, `sparse_draft_head`/`hybrid_draft_tokens` are
passed when above zero, `dynamic_draft: true` becomes `-dds`, and a drafting run's cache is
`draft.spec_decode_cache_size` (`-cs 40960`, not the ladder's 131072). A `perf.py` ladder carries no
draft flags at all — `eval/perf.py` registers none, so `-mtp` in its argv aborts at argparse.
`-Extra` carries only what the state does not: on a drafting run `-Extra "-rb_ab"` is a delta,
`-Extra "-rb_ab -mtp"` is refused.

`perf.py` measures prefill at `256, 512, … max_length` and decode at `0, 256, … max_length-256`
with `S=1..4`; `S=1` is the tg number, `S=2..4` are the per-step cost of MTP verification, not an
acceptance rate (that is `eval/spec_decode.py`).

Retrieval is `-Runner longctx`, and it is a **quality** runner: `eval/longctx.py` feeds whole stories
(14k tokens for Sherlock, 38k for "The Variable Man", 177k for the `-xl` Pride & Prejudice arm) and
asks for summaries, translations, Q&A and inserted-passage detection. Its answers are prose, so no
script grades them — the harness records the run (one `LONGCTX job=<name> prompt_tokens=<n>
out_chars=<k>` line per test, into `results/longctx.csv`), refuses to call a run `ok` without them,
and the reader scores the transcript against the references the script prints. A retrieval run gets
`EXL3_MOE_CPU_SWAP=0` like a drafting run (a placement sweep can flip the greedy token an answer
depends on), and carries no draft args.

A spelling can be checked while the card is busy: `-DryRun` runs every gate that can refuse a pass
except the one-owner check (a dry run starts nothing, so another pass or the tabbyAPI server may own
the card), prints the argv and environment, and exits before anything is created or started.

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dry-perf -DryRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File \
  "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag dry-spec -Runner spec_decode -DryRun
```

## Outputs

| Path | Contents |
|---|---|
| `results/passes.csv` | `tag, run_id, finished_utc, regime, status, engine, ext, elapsed_s, max_length, min_free_vram_mb, min_free_ram_mb, peak_proc_ws_mb, samples, breach, env, split, threads, args, harness_sha256, log, runner, peak_alloc_mb, peak_reserved_mb` |
| `results/pp.csv` | `tag, run_id, band, pp_tok_s` |
| `results/tg.csv` | `tag, run_id, context, s1, s2, s3, s4` |
| `results/spec_decode.csv` | `tag, run_id, category, arm, tok_s, ratio, acc, draft` — a drafting pass's per-category table |
| `results/arms.csv` | `tag, run_id, arm, tok_s, acc, draft, rounds` — the `ARM` lines, one row per arm |
| `results/longctx.csv` | `tag, run_id, job, prompt_tokens, out_chars` — the `LONGCTX` lines, one row per retrieval test |
| `results/watch-<tag>-<utc>.csv` | `t_iso, free_vram_mb, used_vram_mb, free_ram_mb, proc_ws_mb` |
| `results/builds.csv` | `tag, run_id, finished_utc, src, exit, elapsed_s, wheel, wheel_mb, harness_sha256, log` — one row per build, and `log` names `logs/build-<tag>-<utc>.log` |
| `logs/perf-<tag>-<utc>.log` | runner stdout+stderr, identity header appended last (partial metrics survive a kill); `.err.log` beside it keeps the raw stderr |
| `results/BREACH-<tag>-<utc>.txt` | floor violation: values, action taken |

`status` is `ok`, `fail` or `breach`. `fail` is a traceback, a `perf` pass with no pp/tg rows, or a
drafting pass with neither an `ARM` line nor a table row: the uv shim's exit code is unreadable, so
the verdict comes from the evidence. Every run is stamped with its UTC start time, so a repeated tag
never overwrites an earlier log and a `passes.csv` row always points at the file it came from.

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

`wiki/plans/2026-10-29-v165-retune.md` drives these scripts task by task (one subagent per task).
Task 0 deploys the harness and records the ground truth before any measurement, and the later tasks
only take `-Tag`/`-MaxLength`/`-Env`/`-Threads`/`-Split`/`-CacheSize`/`-Runner`/`-StaticSwap`/
`-Extra` combinations — no new scripts.
