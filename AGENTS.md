# AGENTS.md — exllamav3 fork, Windows perf branch

This fork is the measurement and serving stack for one production box: Windows, one 24 GiB card,
63.6 GB of RAM. Read these before you act:

- `wiki/plans/v165-STATE.md` — what is deployed now, the frozen winners, open items
- `wiki/plans/v165-reports/T14-config-catalog.md` — swappable configs, split floors, swap recipes
- `bench/windows/README.md` — harness reference: runners, CSV columns, flags

## The box serves live traffic

- Ask before the first GPU task, before a restart, and before a change to the served config.
- Keep the port at 5000 and the API key unchanged (`api_tokens.yml`). The operator's harness must
  never need a reconfiguration.
- The floors in `watch.ps1` belong to measurement passes, where a runaway configuration must die
  within seconds. Production runs unwatched (T13 steps 10–11).
- Install only the wheels the plan calls for.

## Measurement rules

- Run `bench/size_check.py` before any pass. It answers what a configuration costs in about a
  second, from the checkpoint's own tensor shapes.
- `state.json` is the regime. An arm passes a delta (`-Split`, `-Threads`, `-CacheSize`) and never a
  restated regime flag in `-Extra`. `-DryRun` checks a spelling while the box is busy.
- Draft arguments belong to `spec_decode`; `perf.py` registers none.
- Run `bench/verify_task.py --pre` before the first pass and `--post --report <file>` before you
  quote a number.
- A verdict comes from evidence. A traceback, a floor breach or missing rows makes a run `fail`.
- Compare arms inside one session and carry a same-session reference. A cross-day absolute is not a
  control.

## Traps that each cost a session

- `deploy.sh` from WSL dies with `/usr/bin/env: 'bash\r'`. Strip the CRs first.
- `Start-Process -PassThru` reads an empty exit code. Trust `WHEEL-EXIT 0` in the build log.
- A `-SkipPrefill -SkipGen` smoke always ends `status=fail`. Judge a load by the log.
- `curl` from WSL to the Windows listener returns 400 for a valid request. Use PowerShell.
- tabbyAPI has no `Q2` cache mode. Q2 is spelled `"2,2"`.
- The cache is cheaper than it looks and stranger than it looks: KV lives on 12 of 48 layers, and
  fp16 indexer planes ride along at every width. `bench/size_check.py` derives the bytes from the
  shapes; never scale a cost from one measured width.

## Where work goes

- One report per task at `wiki/plans/v165-reports/TNN-*.md`. The live summary is `v165-STATE.md`.
- A commit message states the measurement or the defect, not the activity.
