# T0 — Preconditions and ground-truth snapshot (2026-10-29)

Executed on branch `165-win-perf` in worktree
`/home/dev/.local/share/opencode/worktree/3a6c9fa6df7b22161dba7be0a7f274f329396456/165-win-perf`.

## Step 1 — Ownership

Operator statement, 2026-10-29 (recorded verbatim): **the 4090 is currently in use and will be
freed when the plan needs it.** Task 0 needs no GPU, so it proceeded. The GPU was indeed busy
during this task (23.4 GB VRAM used by a non-python process; see Step 4). Ask again before the
first GPU task.

## Step 2 — Harness edit (`bench/windows/run_pass.ps1`)

Two edits per the plan:

(a) `[string] $Model = ""` added to `param(...)`; the `$MODEL` default made conditional on it.
(b) `"-cq", "2,2",` inserted before `"-max_length"` in the `production` regime `Args`.

**Deviation from the plan's literal placement (bug found and fixed):** the plan said to add
`if ($Model -ne "") { $MODEL = $Model }` *after* the line
`$MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw"`. PowerShell variable names are
case-insensitive: `$Model` and `$MODEL` are the **same variable**, so the default assignment
clobbers the bound `-Model` value before the override line runs. The first bogus-path test
proved it (the pass ran with the default 2.05bpw model — see the failed test below). The
working form is a conditional default:

```powershell
if ($Model -ne "") { $MODEL = $Model } else { $MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw" }
```

Final diff (committed):

```diff
@@ -46,6 +46,7 @@ param(
     [int] $Threads = 16,
     [int] $Split = 360,
     [string] $Extra = "",
+    [string] $Model = "",
     [string] $Root = "C:\Users\yoho\Downloads\tabbyAPI\_agent\v164",
     [switch] $Short,
@@ -62,6 +63,7 @@ $inv = [System.Globalization.CultureInfo]::InvariantCulture

 $PY    = "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe"
-$MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw"
+if ($Model -ne "") { $MODEL = $Model } else { $MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw" }
 $LOGS  = Join-Path $Root "logs"
@@ -113,7 +115,7 @@
     "production" = @{
-        Args = @("-m", $MODEL, "-cs", "262144", "-chunk_size", "4096", "-ambs", "2", "-max_length", "$MaxLength")
+        Args = @("-m", $MODEL, "-cs", "262144", "-chunk_size", "4096", "-ambs", "2", "-cq", "2,2", "-max_length", "$MaxLength")
```

### Bogus-path test — attempt 1 (FAILED, exposed the case-collision bug)

```
$ powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag t0-bogus -Model "D:\llms\does-not-exist-t0" -MaxLength 4096 -SkipPrefill -SkipGen -SkipWarmup -NoSpecDec -Root "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165"
HARNESS sha256=AAB12627583CF730B28C64C20D54A7570E04800AC062CF2CA1C67498D1866EF2
PASS tag=t0-bogus regime=production status=breach elapsed_s=5 engine=1.6.4 ext=exllamav3_ext.cp312-win_amd64.pyd
GPU  NVIDIA GeForce RTX 4090, 24564 MiB, 596.36
FLOORS min_free_vram_mb=61 min_free_ram_mb=7303 peak_proc_ws_mb=2595 samples=5 breach=1
CSV  C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\results\passes.csv ; ...
BREACH tag=t0-bogus floors violated; see C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\results\BREACH-t0-bogus.txt
exit=4
```

The `-Model` override had not taken effect, so the script loaded the default 2.05bpw model on a
busy GPU; the watcher killed the process tree within 5 s (breach floor 200 MB, free VRAM 61 MB —
the operator's workload, not the pass, owns that VRAM):

```
BREACH tag=t0-bogus at 2026-10-29T05:24:16
free_vram_mb=61 floor=200 low=True
free_ram_mb=7303 floor=2048 low=False
watched_pid=19796 launcher_pid=15564
action=killed process tree
```

The `t0-bogus` row in `passes.csv` is a harness self-test artifact — **not a measurement**; do
not use its numbers. Nothing was left running (watcher killed the tree; Step 4's process list
confirms).

### Bogus-path test — attempt 2 (after the fix, PASSES)

```
$ powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\run_pass.ps1" -Tag t0-bogus2 -Model "D:\llms\does-not-exist-t0" -MaxLength 4096 -SkipPrefill -SkipGen -SkipWarmup -NoSpecDec -Root "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165"
ABORT: model not found: D:\llms\does-not-exist-t0
test_exit=2
```

The override reaches the check. `-cq 2,2` appears in the recorded args of every production-regime
pass (see the `t0-bogus` args row: `-cq 2,2` present).

### Commit

```
[165-win-perf 09923498] bench: -Model override and Q2 cache in the production regime
 1 file changed, 3 insertions(+), 2 deletions(-)
```

## Step 3 — Deploy the harness

Adaptations (recorded):

1. The worktree has `core.autocrlf=true`: working-tree files are CRLF, committed blobs are LF.
   `bench/windows/deploy.sh` itself would not execute (CRLF shebang: `/usr/bin/env: 'bash\r'`).
   It was run via a CR-stripped copy placed in `bench/windows/` (so `BASH_SOURCE` resolves the
   same), deleted immediately after. No repo file was changed.
2. `eval/perf.py` in the working tree was CRLF (sha `0c6f940f…`); the plan's expected sha
   `246e721a…` is the LF blob. Re-checked the working-tree copy out with
   `git -c core.autocrlf=input checkout` (after `rm`) so the deploy ships the LF bytes;
   `git status` stays clean for it (autocrlf normalizes on diff).
3. First deploy run accidentally omitted the `DEST=` prefix and deployed to the retired
   `_agent/v164` directory (harmless: that plan is dead; its deployed copies are now the current
   harness). Re-run with the correct DEST follows.

```
$ DEST=/mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165 bench/windows/deploy.sh
deployed to /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165
c8b714dfb71983baf3d548d42e083f3c84a84f5c8d26873ac5736943f5f7f2b6  /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/run_build.ps1
01adc66f5c9d88d184e6f3673191d19ebe5d4a11de70ec706389e154b3f98ce2  /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/run_pass.ps1
68e2f3691471c50f59e93e052a333e4b8f84db8863f4c187c95cd44a6a530cd7  /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/watch.ps1
246e721a62d7c5238de4b7296adb0b26cd9f42a734b9f648436b35b9b6a29e04  /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/v165/perf.py
```

`perf.py` sha256 = `246e721a62d7…` — **matches** the plan's expected value. (The `.ps1` shas are
of the CRLF working-tree copies; the deployed `run_pass.ps1` sha `01adc66f…` is the fixed
version — the harness hash in later passes will show it.)

## Step 4 — No server, no stray pass

```
$ powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | ? { $_.Name -match '^python' } | select ProcessId,CommandLine | fl"
ProcessId   : 14300
CommandLine : "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe" -c "import exllamav3.version as v; print(v.__version__)"
ProcessId   : 28004
CommandLine : "C:\Python312\python.exe" -c "import exllamav3.version as v; print(v.__version__)"
```

Only the two processes of the Step 5 version probe itself (uv shim + real interpreter). No
tabbyAPI server, no perf.py. The GPU user is a **non-python** process (23.4 GB VRAM used at
05:25) — the operator's workload; nothing was killed.

## Step 5 — Installed engine

```
$ powershell.exe -NoProfile -Command "cd C:\Users\yoho\Downloads\tabbyAPI; .\venv\Scripts\python.exe -c 'import exllamav3.version as v; print(v.__version__)'"
1.6.4
```

**Found: 1.6.4 — NOT the 1.5.4 the plan expected.** A 1.6.4 wheel is already installed (identity
from the pass log: package `C:\Users\yoho\Downloads\tabbyAPI\venv\Lib\site-packages\exllamav3\__init__.py`,
extension `exllamav3_ext.cp312-win_amd64.pyd`, torch `2.11.0+cu130 / cuda 13.0`, GPU
`NVIDIA GeForce RTX 4090, 24564 MiB, driver 596.36`). The 1.5.4 rollback wheel path in
`v165-STATE.md` no longer matches the installed engine; T3's stock-1.6.5 install supersedes it
anyway.

## Step 6 — Production snapshot

`config.yml` copied to `C:/Users/yoho/Downloads/tabbyAPI/_agent/config.v165-baseline.yml`.

```
91b6e23b318f151f0f77f8ff6e4782cc  /mnt/c/Users/yoho/Downloads/tabbyAPI/config.yml
91b6e23b318f151f0f77f8ff6e4782cc  /mnt/c/Users/yoho/Downloads/tabbyAPI/_agent/config.v165-baseline.yml
```

Keys (line numbers in config.yml):

| key | value |
|---|---|
| `cache_size` | 262144 (line 129) |
| `cache_mode` | **FP16** (line 135) — the Q2 operator decision is not yet in config.yml |
| `cpu_moe_split_experts` | 360 (line 174) |
| `cpu_moe_threads` | 16 (line 179) |
| `chunk_size` | 4096 (line 205) |
| `max_batch_size` | 2 (line 217) |
| `ngram_ram` | false (line 185) |
| `vision` | false (line 247) |
| draft block | `draft_mode: mtp`, `draft_num_tokens: 3`, `dynamic_draft: true`, `draft_cache_mode: Q4`, `draft_model_name:` (empty) (lines 341–381) |

`start_tuned.ps1` env block (verbatim):

```powershell
$env:EXL3_MOE_CPU_THREADS = "16"
$env:EXL3_MOE_CPU_PIN = "1"
$env:EXL3_MOE_CPU_SWIZZLE = "1"
$env:EXL3_MOE_MEMOPS = "0"
$env:EXL3_MOE_ZERO_COPY = "1"
$env:EXL3_MOE_STREAM_T = "6"
$env:EXL3_MOE_STREAM_BATCH_EXPERTS = "48"
Start-Process -FilePath "C:\Users\yoho\Downloads\tabbyAPI\start.bat" -WorkingDirectory "C:\Users\yoho\Downloads\tabbyAPI" -RedirectStandardOutput "C:\Users\yoho\Downloads\tabbyAPI\logs\server-final.log"
```

(Note: `EXL3_MOE_ZERO_COPY` is gone in 1.6.x; T13 removes it.)

## Step 7 — Watcher self-check

```
$ powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\watch.ps1" -Tag selfcheck -MaxSeconds 6 -OutDir "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\results"
WATCH tag=selfcheck samples=6 watched_pid=0 min_free_vram_mb=739 min_free_ram_mb=9022 peak_proc_ws_mb=0 breach=0
exit=0
```

`results/watch-selfcheck.csv` present:

```
t_iso,free_vram_mb,used_vram_mb,free_ram_mb,proc_ws_mb
2026-10-29T05:25:40,739,23400,9032,0
2026-10-29T05:25:41,739,23400,9034,0
2026-10-29T05:25:42,739,23400,9034,0
2026-10-29T05:25:43,739,23400,9034,0
2026-10-29T05:25:44,739,23400,9029,0
2026-10-29T05:25:45,739,23400,9022,0
```

## Step 8 — STATE file

`wiki/plans/v165-STATE.md` already existed (created with the plan); updated instead of created.

## Verdict

T0 done. Harness edited, verified, committed (`09923498`); v165 deploy directory live with
`perf.py` sha `246e721a…`; engine on the box is **1.6.4**; production config snapshotted
(md5 `91b6e23b…`); watcher healthy. GPU still owned by the operator's workload — confirm it is
free before T2's smoke passes.

**Next: T1.**
