# Ticket 01 - one measurement harness, named regimes.
#
# Every later measurement runs through this script, so no two of them can drift into different
# flags, environment or log paths. A regime is a name, not a set of remembered arguments, and every
# run records which regime it was plus which engine and extension binary actually loaded.
#
# Usage
#   # one pass of the production-mirror regime
#   powershell -NoProfile -ExecutionPolicy Bypass -File specter_regime.ps1 -Tag arena-base
#
#   # four passes with EXL3_MOE_PINNED_ARENA switching twice (unpinned, pinned, pinned, unpinned)
#   powershell -NoProfile -ExecutionPolicy Bypass -File specter_regime.ps1 -Tag arena -Mode abba `
#              -Toggle EXL3_MOE_PINNED_ARENA
#
#   # list the regimes and what they set, without running anything
#   powershell -NoProfile -ExecutionPolicy Bypass -File specter_regime.ps1 -ListRegimes
#
# The box has one owner: while a pass runs, nothing else should be using the GPU, the CPU pool or
# system RAM. The script does not enforce that, it only refuses to start a second pass of its own.

param(
    [string] $Regime = "production-mirror",
    [string] $Tag    = "run",
    [ValidateSet("plain", "abba")] [string] $Mode = "plain",
    [string] $Toggle = "",          # environment variable to switch in ABBA mode
    [switch] $ListRegimes
)

$ErrorActionPreference = "Continue"

$PY    = "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe"
$PERF  = "C:\Users\yoho\Downloads\exllamav3\eval\perf.py"
$MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw"
$LOG   = "C:\Users\yoho\Downloads\tabbyAPI\logs\perf"
$LOCK  = Join-Path $LOG ".regime-running"

# ---------------------------------------------------------------------------------------------
# Regimes. Each one is the whole comparison regime: model, flags, environment and profiler set.
# A regime change is a deliberate act; adding knobs to a run ad hoc is how two measurements stop
# being comparable.
# ---------------------------------------------------------------------------------------------
$REGIMES = @{
    # The deployed configuration as of 2026-10-23: 2.05 bpw, per-layer split with the tail 360 of
    # 512 experts on the CPU, 16 worker threads, 262144 cache, FP16 cache, chunk 4096, batch 2.
    # profilers are part of the regime because their numbers are read before tok/s.
    "production-mirror" = @{
        Model  = $MODEL
        Flags  = @("-m", $MODEL, "-mcs", "360", "-mct", "16", "-cs", "262144",
                   "-chunk_size", "4096", "-ambs", "2", "-max_length", "32768", "-short", "-sd")
        Env    = @{
            EXL3_MOE_MEMOPS               = "0"
            EXL3_MOE_CPU_PIN              = "1"
            EXL3_MOE_CPU_SWIZZLE          = "1"
            EXL3_MOE_STREAM_T             = "6"
            EXL3_MOE_STREAM_BATCH_EXPERTS = "48"
            EXL3_MOE_CPU_PROF             = "1"
            EXL3_MOE_CPU_SWAP_DEBUG       = "1"
            EXL3_SPLIT_PROF               = "1"
            EXL3_MOE_SUBMIT_PROF          = "1"
        }
    }

    # Same, but graphs off and a routing capture written. Needed whenever the capture hook is in
    # play: it is not stream-capture safe, and the module says so rather than corrupting ids.
    "production-mirror-capture" = @{
        Model  = $MODEL
        Flags  = @("-m", $MODEL, "-mcs", "360", "-mct", "16", "-cs", "262144",
                   "-chunk_size", "4096", "-ambs", "2", "-max_length", "32768", "-spf", "-sd")
        Env    = @{
            EXL3_MOE_MEMOPS               = "0"
            EXL3_MOE_CPU_PIN              = "1"
            EXL3_MOE_CPU_SWIZZLE          = "1"
            EXL3_MOE_STREAM_T             = "6"
            EXL3_MOE_STREAM_BATCH_EXPERTS = "48"
            EXL3_MOE_CPU_PROF             = "1"
            EXL3_MOE_CPU_SWAP_DEBUG       = "1"
            EXL3_SPLIT_PROF               = "1"
            EXL3_MOE_SUBMIT_PROF          = "1"
            EXL3_GRAPHS                   = "0"
            EXL3_ROUTE_CAPTURE            = (Join-Path $LOG "capture-production-mirror.jsonl")
            EXL3_ROUTE_CAPTURE_RECORDS    = "16384"
        }
    }
}

if ($ListRegimes)
{
    foreach ($name in ($REGIMES.Keys | Sort-Object))
    {
        $r = $REGIMES[$name]
        Write-Host "== $name" -ForegroundColor Cyan
        Write-Host ("   flags: " + ($r.Flags -join " "))
        foreach ($k in ($r.Env.Keys | Sort-Object)) { Write-Host ("   env:   {0}={1}" -f $k, $r.Env[$k]) }
    }
    exit 0
}

if (-not $REGIMES.ContainsKey($Regime))
{
    Write-Host ("ABORT: unknown regime '{0}'. Known: {1}" -f $Regime, (($REGIMES.Keys | Sort-Object) -join ", "))
    exit 2
}
$R = $REGIMES[$Regime]
New-Item -ItemType Directory -Force $LOG | Out-Null

if ($Mode -eq "abba" -and -not $Toggle)
{
    Write-Host "ABORT: -Mode abba needs -Toggle <ENV_VAR> to know what to switch"
    exit 2
}

# One owner: refuse to start a pass while another pass of this harness is running.
if (Test-Path $LOCK)
{
    Write-Host ("ABORT: a pass is already running (stale lock? {0})" -f $LOCK)
    exit 3
}

# What is actually installed - recorded so a later reader never has to guess which engine a log
# came from. The extension path matters: a stray .pyd in the working directory can win over the
# installed one, so the path is evidence, not decoration.
$engine = (& $PY -c "import exllamav3.version as v; print(v.__version__)" 2>&1) -join " "
$ext    = (& $PY -c "import exllamav3.ext as e; print(e.__file__)" 2>&1) -join " "
$ext    = $ext.Trim() -replace ".*[/\\](exllamav3_ext[^/\\]*)$", '$1'

function Invoke-Pass([string] $name, [bool] $toggled)
{
    if ($toggled) { Set-Item "env:$Toggle" "1" }
    else          { Remove-Item "env:$Toggle" -ErrorAction SilentlyContinue }

    foreach ($k in $R.Env.Keys) { Set-Item "env:$k" $R.Env[$k] }

    $out = Join-Path $LOG "$name.out.log"
    $err = Join-Path $LOG "$name.err.log"
    $started = Get-Date
    $p = Start-Process -FilePath $PY -ArgumentList (@($PERF) + $R.Flags) -NoNewWindow -Wait -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    $elapsed = [int]((Get-Date) - $started).TotalSeconds
    Get-Content $err -ErrorAction SilentlyContinue | Add-Content $out

    # Header appended after the pass: it records the regime and the engine, which is what makes the
    # log attributable later. Written last so a killed pass still leaves partial metrics.
    $header = @(
        "",
        "=== pass        : $name",
        "=== regime      : $Regime",
        "=== mode        : $Mode",
        ("=== toggled     : {0}={1}" -f $(if ($Toggle) { $Toggle } else { "-" }), $toggled),
        "=== engine      : $engine",
        "=== extension   : $ext",
        ("=== env         : " + (($R.Env.Keys | Sort-Object | ForEach-Object { "$_=$($R.Env[$_])" }) -join " ")),
        "=== started     : $($started.ToString('s'))",
        "=== elapsed s   : $elapsed",
        "=== exit        : $($p.ExitCode)"
    )
    Add-Content -Path $out -Value $header
    Write-Host ("=== {0}  exit={1}  {2}s  regime={3}" -f $name, $p.ExitCode, $elapsed, $Regime) -ForegroundColor Cyan
}

New-Item -ItemType File -Force $LOCK | Out-Null
try
{
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    if ($Mode -eq "plain")
    {
        Invoke-Pass "$stamp-$Regime-$Tag" $false
    }
    else
    {
        # ABBA: the configuration switches twice, so drift in box state shows up as a difference
        # between the two members of a pair instead of as the effect being measured.
        Invoke-Pass "$stamp-$Regime-$Tag-1-off" $false
        Invoke-Pass "$stamp-$Regime-$Tag-2-on"  $true
        Invoke-Pass "$stamp-$Regime-$Tag-3-on"  $true
        Invoke-Pass "$stamp-$Regime-$Tag-4-off" $false
        Remove-Item "env:$Toggle" -ErrorAction SilentlyContinue
    }
}
finally
{
    Remove-Item $LOCK -ErrorAction SilentlyContinue
    Write-Host ("done $(Get-Date -Format s)")
}
