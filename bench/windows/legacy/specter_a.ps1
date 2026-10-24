# Specter arm A — M1/M2/M3 on the INSTALLED engine (exllamav3 1.5.4), production mirror.
#
# Zero install changes: this only loads the model and times it. Nothing is installed,
# nothing is rebuilt, no config file is touched.
#
# Mirrors production config.yml as it stands 2026-10-23: 2.05 bpw, split mode with 360 of
# 512 experts on CPU, 16 worker threads, 262144 cache, FP16 cache, chunk 4096, batch 2.
# EXL3_MOE_ZERO_COPY is deliberately NOT set: the deployed engine has no such knob (verified
# against site-packages), so start_tuned.ps1's copy of it is a no-op left over from 1.4.9.
#
# Profilers are on so the log carries the numbers the plan is gated on:
#   EXL3_MOE_CPU_PROF        worker per-phase ms per 512 jobs   -> M1 host us/expert
#   EXL3_SPLIT_PROF          issue/wait percentiles             -> submit path
#   EXL3_MOE_SUBMIT_PROF     submission timing
#   EXL3_MOE_CPU_SWAP_DEBUG  routed/step, miss %, hook cost     -> per-swap cost, swap counts
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File specter_a.ps1

$ErrorActionPreference = "Continue"
$PY    = "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe"
$PERF  = "C:\Users\yoho\Downloads\exllamav3\eval\perf.py"
$MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw"
$LOG   = "C:\Users\yoho\Downloads\tabbyAPI\logs\perf"
New-Item -ItemType Directory -Force $LOG | Out-Null

if (-not (Test-Path $MODEL)) { Write-Host "ABORT: model not found: $MODEL"; exit 2 }
if (-not (Test-Path $PY))    { Write-Host "ABORT: python not found: $PY"; exit 2 }

$BASE = @("-m", $MODEL, "-mcs", "360", "-mct", "16", "-cs", "262144",
          "-chunk_size", "4096", "-ambs", "2", "-max_length", "32768", "-short", "-sd")

# Production env (start_tuned.ps1) minus the dead knob
$env:EXL3_MOE_MEMOPS                = "0"
$env:EXL3_MOE_CPU_PIN               = "1"
$env:EXL3_MOE_CPU_SWIZZLE           = "1"
$env:EXL3_MOE_STREAM_T              = "6"
$env:EXL3_MOE_STREAM_BATCH_EXPERTS  = "48"
# Probes
$env:EXL3_MOE_CPU_PROF              = "1"
$env:EXL3_MOE_CPU_SWAP_DEBUG        = "1"
$env:EXL3_SPLIT_PROF                = "1"
$env:EXL3_MOE_SUBMIT_PROF           = "1"

function Run([string]$name, [string[]]$envs)
{
    Remove-Item "env:EXL3_MOE_PINNED_ARENA" -ErrorAction SilentlyContinue
    foreach ($e in $envs) { $k, $v = $e -split "=", 2; Set-Item "env:$k" $v }
    $out = Join-Path $LOG "$name.out.log"
    $err = Join-Path $LOG "$name.err.log"
    Write-Host "=== $name  start $(Get-Date -Format s)" -ForegroundColor Cyan
    $t0 = Get-Date
    $p = Start-Process -FilePath $PY -ArgumentList (@($PERF) + $BASE) -NoNewWindow -Wait -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    # worker threads are a child process: their profiler output lands in the stderr file
    Get-Content $err -ErrorAction SilentlyContinue | Add-Content $out
    Write-Host "=== $name  done  exit=$($p.ExitCode)  ${dt}s  -> $out" -ForegroundColor Cyan
}

# A1: baseline (production mirror)
Run "specter-a1-baseline" @()

# A2: same, with the pinned expert arena — the one lever that needs no code.
#     1.5.4 has no fallback when page-locking fails, so this may fail at load; the log
#     will say so, which is itself the answer for this engine.
Run "specter-a2-pinned-arena" @("EXL3_MOE_PINNED_ARENA=1")

Write-Host "ALL DONE $(Get-Date -Format s)"
