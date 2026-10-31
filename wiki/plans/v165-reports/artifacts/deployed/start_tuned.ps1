# Deployed environment for the v165 measured winner (2026-10-31).
#
# Every line here was measured on this box and is recorded in
# wiki/plans/v165-reports/ (T08-fork, T09-knobs, T11-cache, T12-ple-gdn, T13-deploy).
# Removed versus the previous script, each with a reason:
#   EXL3_MOE_ZERO_COPY      - the knob is gone from 1.6.5 (T1 finding).
#   EXL3_MOE_STREAM_T=6     - a 1.5.4 pin; never re-earned on 1.6.5 (T9 carried it as untested).
#   EXL3_MOE_STREAM_BATCH_EXPERTS=48 - same.
# Added:
#   EXL3_MOE_THREAD_PRIO=1  - T8: +11 % decode (43.68 vs 39.35 tok/s at 130816, two passes
#                             disjoint from both references), +2 % prefill, no VRAM cost.
#   EXL3_HOST_MEM_RESERVE_MB=0 / EXL3_MOE_ARENA_HUGEPAGE=0 - the measured regime: the watcher
#                             holds the floor, and huge pages measured no benefit (T13 2c).
$env:EXL3_MOE_CPU_THREADS = "16"
$env:EXL3_MOE_CPU_PIN = "1"
$env:EXL3_MOE_CPU_SWIZZLE = "1"
$env:EXL3_MOE_MEMOPS = "0"
$env:EXL3_MOE_THREAD_PRIO = "1"
$env:EXL3_HOST_MEM_RESERVE_MB = "0"
$env:EXL3_MOE_ARENA_HUGEPAGE = "0"

$logDir = "C:\Users\yoho\Downloads\tabbyAPI\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")

# main.py is the API itself; start.py would re-check/upgrade dependencies on a production box.
$p = Start-Process -FilePath "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe" `
    -ArgumentList "main.py" `
    -WorkingDirectory "C:\Users\yoho\Downloads\tabbyAPI" `
    -RedirectStandardOutput "$logDir\server-$stamp.log" `
    -RedirectStandardError  "$logDir\server-$stamp.err.log" `
    -PassThru
Write-Output ("SERVER pid=" + $p.Id + " log=$logDir\server-$stamp.log")
