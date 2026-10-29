# T03 — stock v1.6.5 wheel: build, install, verify

Date: 2026-10-29. No GPU used (build is CPU/RAM; watcher enforced floors throughout).

## Staging

- `cp -a /home/dev/exllamav3/.worktrees/v165-stock → C:\Users\yoho\Downloads\exl3-build-165\src`
- Recipe `bench/windows/legacy/build_wheel.bat → C:\Users\yoho\Downloads\exl3-build-165\_build.bat`,
  with `SCRATCH=C:\Users\yoho\Downloads\exl3-build-165\src` (the check is `%SCRATCH%\setup.py` —
  pointing SCRATCH at the parent aborts `ABORT: no setup.py`), and `L=` pointed at the parent so
  `run_build.ps1` finds `build_wheel.log` where it expects it.
- Toolchain lines confirmed in the log: MSVC 14.44.35207, CUDA v13.3 nvcc, `TORCH_CUDA_ARCH_LIST=8.9`.

## Build attempts (watcher RAM floor)

| tag | MAX_JOBS | outcome |
|---|---|---|
| build-stock | 4 | aborted at 0 s — SCRATCH staging bug (above) |
| build-stock | 4 | BREACH 368 s: free RAM 1999 < 2048 (operator workload up, ~9.8 GB free at start) |
| build-stock2 | 2 | BREACH 96 s: free RAM 1713 |
| build-stock3 | 2 | BREACH 68 s: free RAM 2026 (operator freed some RAM; build peak ~3.8 GB observed) |
| build-stock4 | 1 | BREACH 64 s: free RAM 2045 — breach by 3 MB |
| build-stock5 | 1 | BREACH 66 s |
| **build-stock6** | **2** | **OK 2199 s** after operator restarted the RAM holder (`unbiased-strata.exe` 41.5→36.4 GB; 8.6 GB free) |

RAM holder diagnosis (top working sets): `unbiased-strata.exe` 41.5 GB (the inference server
serving the agent session), `vmmem` 6.2 GB. The build's peak at MAX_JOBS=2 is ~4 GB; at MAX_JOBS=1
still ~3.5 GB (single `cicc` on `moe_mul1.cpp` dominates). Breaches killed only the build tree.

Harness quirk: `BUILD tag=build-stock6 exit=-1` — `cmd.exe /c <bat>` exit code read back empty
(`$null` → -1 in `run_build.ps1`); the recipe's own log shows `WHEEL-EXIT 0 29/10/2026 14:38:06`
and the wheel exists. Treat `exit=-1` + `BUILD_WHEEL name=…` + `WHEEL-EXIT 0` as success.

## Wheel

- `C:\Users\yoho\Downloads\exl3-build-165\src\dist\exllamav3-1.6.5-cp312-cp312-win_amd64.whl`
- 16,275,036 bytes; sha256 `5337b6209506dc078e8fd30b55fe7309fea3df008437d8eb122389cafc87d14d`
- `builds.csv` row: `build-stock6,2026-10-29T14:38:07,…,-1,2199,exllamav3-1.6.5-…whl,16`

## Install + verify

WSL `uv` cannot resolve a Windows venv interpreter (`--python …\python.exe` → "No virtual
environment found"); used the Windows-side `C:\Users\yoho\.local\bin\uv.exe` via powershell:

```
uv pip install --python C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe --no-deps --reinstall <wheel>
 - exllamav3==1.6.4 (from file:///…/exl3-build-164/src/dist/exllamav3-1.6.4-cp312-cp312-win_amd64.whl)
 + exllamav3==1.6.5 (from file:///…/exl3-build-165/src/dist/exllamav3-1.6.5-cp312-cp312-win_amd64.whl)
```

From cwd `C:\Users\yoho` (never a UNC cwd):

```
1.6.5
C:\Users\yoho\Downloads\tabbyAPI\venv\Lib\site-packages\exllamav3\__init__.py
C:\Users\yoho\Downloads\tabbyAPI\venv\Lib\site-packages\exllamav3_ext.cp312-win_amd64.pyd
```

## Rollback

- 1.6.4 wheel: `C:\Users\yoho\Downloads\exl3-build-164\src\dist\exllamav3-1.6.4-cp312-cp312-win_amd64.whl`
- 1.5.4 wheel: `C:\Users\yoho\Downloads\exl3-wheel-154\exllamav3-1.5.4+cu132.torch2.11.0-cp312-cp312-win_amd64.whl`
- Command (either): Windows `uv.exe` as above with `--no-deps --reinstall <wheel>`.

## Verdict

**PASS.** Engine in the tabbyAPI venv is stock v1.6.5 built from the `v165-stock` worktree
(upstream `cdc785db`). The winperf fork wheel build (`build-winperf`, src staged at tag
`win-perf-1.6.5`) was launched right after; result recorded in T08 prep.
