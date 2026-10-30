# T08 — fork delta (`win-perf-1.6.5`) and its four worker knobs

Date: 2026-10-30. Engine installed during this task: win-perf `1.6.5` wheel
`exllamav3-1.6.5-cp312-cp312-win_amd64.whl` sha256 `1422a0a1e67eaf88efd6cddfaa200ded32e64a772b32db7509f7128d16b4d60f`,
installed over stock `1.6.5` with Windows `uv.exe pip install --no-deps --reinstall`, verified from
a Windows cwd (`exllamav3.version.__version__` = 1.6.5, package and `.pyd` under
`venv\Lib\site-packages`).

**Installation verified by content, not by the installer's word:** the compiled extension carries
the three C++ knob strings (`EXL3_MOE_CPU_PIN_OFFSET`, `EXL3_MOE_THREAD_PRIO`,
`EXL3_MOE_BLOCKING_SYNC`) — 1 occurrence each in the 19 472 896-byte `.pyd`, and 0 in the stock
`.pyd` after reinstalling stock for the A/B below. `model_init.py:213` is the `mtp`/`draft_model_dir`
exclusion, and the old `assert not draft_model_dir, "Tensor overrides not supported…"` is gone, so
`-or` with a draft model loads (`verify_task.py --pre --expect-or` passes).

## Step 2 — the code delta alone: a null

ABBA at 131072, no fork knob set, same regime, same session, stock → fork → fork → stock
(each transition an install + verification):

| pair | PP 4096 | PP 65536 | PP 131072 | TG 65536 | TG 130816 |
|---|---|---|---|---|---|
| `t8-stock-a1-20261030-055050` (stock) | 1741.87 | 1839.21 | 1825.14 | 40.88 | 38.38 |
| `t8-fork-b1-20261030-060712` (fork) | 1765.20 | 1848.44 | 1833.53 | 38.82 | 39.98 |
| `t8-fork-b2-20261030-061117` (fork) | 1860.98 | 1825.54 | 1813.46 | 40.24 | 40.92 |
| `t8-stock-a2-20261030-061534` (stock) | 1924.78 | 1868.52 | 1835.26 | 39.50 | 38.96 |

Ratios: PP 1.00 then 0.97 at the long bands; TG 1.04 then 0.99. **Inside the ladder's own jitter
(3–5 % either way), so the fork is a null at defaults** — which is what it should be: the four knobs are
default-off and the wheel differs only in load-path code plus the `-or` relaxation.

**Decision: keep the fork.** It is required for the sidecar path (T7), it costs nothing measurable,
and its knobs are where the value is (below).

## Step 3 — the knobs, two passes each

Reference: the T9 session reference (`t9-ref-a` / `t9-ref-b`, same regime, same `-NoSpecDec -dr 3`
protocol): TG **38.87 / 39.83** at 130816, **37.79 / 37.00** at 65536, PP 1835.57 / 1798.08 at 131072.

| arm (run id) | TG 130816 | TG 65536 | PP 131072 | vs reference | verdict |
|---|---|---|---|---|---|
| `t8-timer1-a-20261030-094540`, `-b-20261030-102233` | 38.72 / 38.48 | 36.92 / 36.92 | 1820.75 / 1831.89 | 0.98 | no effect |
| `t8-prio1-a-20261030-095326`, `-b-20261030-103116` | **43.75 / 43.60** | **40.77 / 40.71** | 1847.63 / 1870.42 | **1.11** | **WIN** |
| `t8-pinoff8-a-20261030-100049`, `-b-20261030-103959` | 39.08 / 38.39 | 36.13 / 38.11 | 1800.64 / 1821.02 | 0.98 | no effect |

- **`EXL3_MOE_THREAD_PRIO=1` is worth 11 % of decode** — 43.68 mean against 39.35, the two passes
  within 0.6 % of each other and **disjoint from both reference passes**, at every context band
  (+11 % at 32768, +9 % at 65536, +11 % at 130816) and 2 % *better* on prefill. This is the
  campaign's largest single lever after the DFlash2 drafter, and the first fork knob that earns a
  place in the deployed environment.

  **Two checks that the number is real, because a +11 % single-session arm deserves them.** (1) The
  three knob arms ran back to back — `timer1` 38.60, `prio1` **43.68**, `pinoff8` 38.74 — so neither
  a monotonic drift nor a warm-up curve can produce it; the arm is bracketed by two others at the
  reference level. (2) The mechanism is **not** a shorter mean wait: a `-Prof` probe with the knob on
  (`t8-prio1-prof-20261030-100942`) prints `split prof [wait]` med **0.196 / 0.209 ms** against the
  baseline's **0.202 / 0.210 ms**, i.e. unchanged, with the same p90. The gain is therefore in the
  scheduling tail the profiler does not print, not in the median the counter reports — recorded as
  an open mechanism question, not as an explanation.
- **`EXL3_MOE_TIMER_RES=1` and `EXL3_MOE_CPU_PIN_OFFSET=8` are neutral** (0.98 each, inside the
  reference's spread, no band moves consistently). Neither goes into production.
- **`EXL3_MOE_BLOCKING_SYNC` is not tested**: a prior measurement on this box put it ~4 % slower with
  MTP host syncs, and the campaign's advise-only list carries it for that reason — recorded as a
  decision, not an omission.

## Step 4 — code base decision

**Serve the win-perf wheel.** It is a measured null on the code delta, it unblocks the `-or` path,
and its `EXL3_MOE_THREAD_PRIO=1` is a +11 % decode knob that the stock wheel does not have. The
deployed environment should gain one line (`EXL3_MOE_THREAD_PRIO=1`); the other three knobs stay at
their defaults.

**Carry-forward to T13:** that line is a *config* change (`start_tuned.ps1`), so it is written into
the T13 proposal and not applied here.
