# Specter arm F confirmation: EXL3_MOE_PINNED_ARENA on the INSTALLED engine (exllamav3 1.5.4).
#
# Why this exists: the first pass (specter-a1/a2) ran unpinned THEN pinned, so the order is
# confounded with clock/thermal state. S=1 decode swings 24-32 it/s inside a single run, which is
# wider than any effect we could hope to see there, so decode cannot be settled by one pair. What
# the first pair DID show is clean and mechanism-consistent: streamed prefill moved from
# 2417-2441 tok/s (spread +-1.5%) to 2906-2979 tok/s (spread +-1.2%), i.e. +21%, and the pinned
# arena's whole purpose is to let streamed prefill DMA straight out of the worker arena instead of
# copying through the stager. This run repeats it as ABBA so both sides are measured before and
# after the configuration switch, on the same regime.
#
# One owner of the hardware: nothing else may run (no build, no server, no WSL work) while this
# executes. Four runs at ~8 min each is ~31 min.
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File specter_pinned_abba.ps1

$ErrorActionPreference = "Continue"
$PY    = "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe"
$PERF  = "C:\Users\yoho\Downloads\exllamav3\eval\perf.py"
$MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw"
$LOG   = "C:\Users\yoho\Downloads\tabbyAPI\logs\perf"
New-Item -ItemType Directory -Force $LOG | Out-Null

if (-not (Test-Path $MODEL)) { Write-Host "ABORT: model not found"; exit 2 }

# Identical regime to specter-a1/a2, so the four runs are comparable with the first pair.
$ARGS = @($PERF, "-m", $MODEL, "-mcs", "360", "-mct", "16", "-cs", "262144",
          "-chunk_size", "4096", "-ambs", "2", "-max_length", "32768", "-short", "-sd")

$env:EXL3_MOE_MEMOPS               = "0"
$env:EXL3_MOE_CPU_PIN              = "1"
$env:EXL3_MOE_CPU_SWIZZLE          = "1"
$env:EXL3_MOE_STREAM_T             = "6"
$env:EXL3_MOE_STREAM_BATCH_EXPERTS = "48"
$env:EXL3_MOE_CPU_PROF             = "1"
$env:EXL3_MOE_CPU_SWAP_DEBUG       = "1"
$env:EXL3_SPLIT_PROF               = "1"
$env:EXL3_MOE_SUBMIT_PROF          = "1"

function Run([string]$name, [bool]$pinned)
{
    if ($pinned) { $env:EXL3_MOE_PINNED_ARENA = "1" }
    else         { Remove-Item "env:EXL3_MOE_PINNED_ARENA" -ErrorAction SilentlyContinue }
    $out = Join-Path $LOG "$name.out.log"
    $err = Join-Path $LOG "$name.err.log"
    $t0 = Get-Date
    Write-Host "=== $name  pinned=$pinned  start $(Get-Date -Format s)" -ForegroundColor Cyan
    $p = Start-Process -FilePath $PY -ArgumentList $ARGS -NoNewWindow -Wait -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    Get-Content $err -ErrorAction SilentlyContinue | Add-Content $out
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host "=== $name  done  exit=$($p.ExitCode)  ${dt}s" -ForegroundColor Cyan
}

# ABBA: the configuration changes twice, so a drift in box state shows up as a difference
# between the two members of the same pair rather than as the effect being measured.
Run "pinned-abba-1-unpinned" $false
Run "pinned-abba-2-pinned"   $true
Run "pinned-abba-3-pinned"   $true
Run "pinned-abba-4-unpinned" $false

Remove-Item "env:EXL3_MOE_PINNED_ARENA" -ErrorAction SilentlyContinue
Write-Host "ALL DONE $(Get-Date -Format s)"
