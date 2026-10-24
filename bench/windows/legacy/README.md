# Legacy harness scripts (kept for provenance)

These are the scripts the previous rounds of tuning actually ran. They are **not** used by the
v1.6.4 retune — `bench/windows/run_pass.ps1` replaces them — but they are kept because the numbers
in Engram were produced by them, and because two of their ideas became the current contract.

Do not edit them. If a measurement needs a new shape, add a regime to `run_pass.ps1`.

| File | Date | What it did | Status |
|---|---|---|---|
| `bench_one.ps1` | 2026-09-11 | one pass of `eval/spec_decode.py` on the 3.05 bpw model (`-mcl 36 -mct 8 -cs 131072`), with `EXL3_MOE_MEMOPS`, `_FLUSH`, `_SUBMIT`, `_LOG`, `EXL3_MOE_HANDOFF_PROF`, `EXL3_MOE_ZERO_COPY` and the worker knobs as parameters; log per name, no watcher | superseded. It produced the v1.4.9 memop/handshake/zero-copy logs (`bench_*.log` in `_agent/`), incl. the "handshake is 8-11% slower" result (#51). Its knobs include `EXL3_MOE_ZERO_COPY`, which no longer exists, and it writes `EXL3_MOE_STREAM_T=6` / `_BATCH_EXPERTS=48` unconditionally. |
| `poll_bench.ps1` | 2026-09-11 | polls a `bench_<name>.log` for `BENCH_EXIT` every 2 min | superseded. Long runs use the shell tool's background mode; no polling. |
| `specter_a.ps1` | 2026-10-23 | production mirror on the installed 1.5.4 (2.05 bpw, `-mcs 360 -mct 16 -cs 262144`), A1 baseline then A2 with `EXL3_MOE_PINNED_ARENA=1` | superseded, historically important: the source of the pinned-arena result (prefill 2417-2441 -> 2906-2979 tok/s, +21%, decode unresolved, #260). |
| `specter_pinned_abba.ps1` | 2026-10-23 | the same comparison re-run as ABBA (unpinned, pinned, pinned, unpinned) because the first pair was order-confounded | superseded; its ABBA shape is now the contract in `run_pass.ps1`. |
| `specter_regime.ps1` | 2026-10-23 | named-regime harness: a `$REGIMES` table, engine/extension identity in the log, a lock file against a second pass, and a header appended **after** the pass so partial metrics survive a kill | superseded by `run_pass.ps1`, which keeps the regime table, the identity header, the header-last ordering and the single-owner lock, and adds the VRAM/RAM watcher, the parser and CSVs. It pins 1.4.9 env values (`STREAM_T=6`, `BATCH_EXPERTS=48`, `MEMOPS=0`) and runs `perf.py` from the v1.4.9 clone. |
| `build_wheel.bat` | 2026-10-23 | the Windows wheel recipe: MSVC 14.44.35207 + SDK 10.0.26100.0 set by hand, CUDA 13.3, `TORCH_CUDA_ARCH_LIST=8.9`, `MAX_JOBS=4`, `CUDAHOSTCXX=cl.exe`, `DISTUTILS_USE_SDK=1`, `setup.py bdist_wheel` (no pip/wheel in the venv) | **still authoritative as a recipe**. `run_build.ps1` wraps it: copy it next to a source tree as `_build.bat`, set `SCRATCH` to that tree, run under the watcher. |

## What the previous rounds learned, encoded in the current harness

- A number is only comparable to a reference measured in the same window (drift ~10% per session,
  0.07% between consecutive boots, #157) -> ABBA in every comparison.
- Machine state must be sampled during the run, not assumed at the start -> `watch.ps1`.
- The identity of the engine must be in the artifact, not in the chat -> `engine`, `ext`, and now
  the harness hash in every log header and CSV row.
- Partial output must survive a kill -> header written last.
- One owner of the box -> refuse to start when another pass or server is alive.
