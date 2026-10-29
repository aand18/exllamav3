# v1.6.4 retune - one measurement pass through the sanctioned regime, under the watcher.
#
# WHY THIS EXISTS: every arm of the retune must be the same commands, the same environment model,
# the same parser and the same floors, or two numbers stop being comparable. This script is the
# only sanctioned way to produce a pp/tg number for the retune. It is deployed from
# bench/windows/ in the repo; never edit the deployed copy.
#
# It does NOT install anything, does NOT touch config.yml, and does NOT start the server. It loads
# the model offline through perf.py (a copy of eval/perf.py; that file is byte-identical in v1.5.4
# and v1.6.4, and it imports the INSTALLED engine).
#
# TWO TRAPS THIS SCRIPT GUARDS AGAINST (both measured 2026-10-24):
#   1. The tabbyAPI venv's python.exe is a uv shim: Start-Process returns the shim, the real
#      interpreter is its child. The watcher is told to resolve the real process by command line.
#   2. When PowerShell is launched from a WSL directory, its working directory is a UNC path and
#      `import exllamav3` resolves to the WSL source tree instead of site-packages. The script
#      changes to a Windows directory first and refuses to run if the imported package is not the
#      installed one.
#
# Usage:
#   run_pass.ps1 -Tag armA-a
#   run_pass.ps1 -Tag memops1-a -Env "EXL3_MOE_MEMOPS=1" -MaxLength 131072
#   run_pass.ps1 -Tag t14-a -Threads 14 -MaxLength 131072
#   run_pass.ps1 -Tag split320 -Split 320 -MaxLength 262144
#   run_pass.ps1 -Tag q8 -Extra "-cq Q8" -MaxLength 131072
#   run_pass.ps1 -Tag x -Unset "EXL3_MOE_MEMOPS,EXL3_MOE_CPU_PIN"
#
# -Env is a comma-separated KEY=VALUE list applied AFTER the regime's own environment, so it can
# override a regime value. -Unset removes variables from the inherited environment. -Threads and
# -Split replace the regime's -mct and -mcs. -Extra appends raw perf.py flags.
#
# Outputs (in -OutDir):
#   logs/perf-<Tag>-<utc>.log    perf.py stdout + stderr + an identity header appended last
#   logs/perf-<Tag>-<utc>.err.log  raw stderr (kept; the .log has it appended too)
#   results/passes.csv           one row per pass: tag, run_id, status, engine, floors, env, args,
#                                harness hash, log file
#   results/pp.csv               one row per prefill band: tag, run_id, band, pp_tok_s
#   results/tg.csv               one row per generation context: tag, run_id, context, s1..s4 it/s
#   results/watch-<Tag>-<utc>.csv  watcher samples
#   stdout                       HARNESS / PASS / GPU / FLOORS / PP / TG_S1 / CSV lines
#
# Every run is stamped with its UTC start time: a repeated tag gets a new run id, so no earlier log
# or watcher trace is ever overwritten. passes.csv carries the run id and log filename, so a row
# always points at the evidence that produced it.

param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [string] $Regime = "production",
    [string] $Env = "",
    [string] $Unset = "",
    [int] $MaxLength = 262144,
    [int] $Threads = 16,
    [int] $Split = 360,
    [string] $Extra = "",
    [string] $Model = "",
    [string] $Root = "C:\Users\yoho\Downloads\tabbyAPI\_agent\v164",
    [switch] $Short,
    [string] $Runner = "perf",   # perf.py (pp/tg ladder) or spec_decode.py (drafting arms)
    [switch] $NoSpecDec,
    [switch] $SkipPrefill,
    [switch] $SkipGen,
    [switch] $SkipWarmup,
    [switch] $Prof,
    [int] $MinFreeVramMB = 100,
    [int] $MinFreeRamMB = 1200
)

$ErrorActionPreference = "Continue"
$inv = [System.Globalization.CultureInfo]::InvariantCulture

$PY    = "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe"
if ($Model -ne "") { $MODEL = $Model } else { $MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw" }
$LOGS  = Join-Path $Root "logs"
$RES   = Join-Path $Root "results"
if ($Runner -eq "spec_decode") {
    $PERF = Join-Path $Root "spec_decode.py"
    $MATCH_CMD = "spec_decode\.py"
} else {
    $PERF = Join-Path $Root "perf.py"
    $MATCH_CMD = "perf\.py"
}
$WATCH = Join-Path $Root "watch.ps1"

New-Item -ItemType Directory -Force $LOGS, $RES | Out-Null
Set-Location $Root   # trap 2: a WSL launch leaves a UNC working directory behind

foreach ($f in @($PY, $PERF, $WATCH)) {
    if (-not (Test-Path $f)) { Write-Output "ABORT: missing $f (deploy with bench/windows/deploy.sh)"; exit 2 }
}
if (-not (Test-Path $MODEL)) { Write-Output "ABORT: model not found: $MODEL"; exit 2 }

# One owner of the box: refuse to start while another pass or a server is running.
$other = @(Get-CimInstance Win32_Process |
           Where-Object { $_.Name -match '^python' -and $_.ProcessId -ne $PID -and
                          $_.CommandLine -match 'perf\.py|spec_decode\.py|tabbyAPI' })
if ($other.Count -gt 0) {
    Write-Output "ABORT: another pass is running:"
    $other | ForEach-Object { Write-Output ("  " + $_.ProcessId + " " + $_.CommandLine) }
    exit 3
}

# --- Identity of what is under test ------------------------------------------------------------
$engine = ((& $PY -c "import exllamav3.version as v; print(v.__version__)" 2>&1) -join " ").Trim()
$pkg    = ((& $PY -c "import exllamav3; print(exllamav3.__file__)" 2>&1) -join " ").Trim()
$ext    = ((& $PY -c "import torch, exllamav3_ext as e; print(e.__file__)" 2>&1) -join " ").Trim()
if ($ext -match '\.pyd$') { $extName = [System.IO.Path]::GetFileName($ext) } else { $extName = "n/a" }
$torch  = ((& $PY -c "import torch; print(torch.__version__ + ' / cuda ' + str(torch.version.cuda))" 2>&1) -join " ").Trim()
$gpu    = ((& nvidia-smi.exe --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1) -join " ").Trim()
$freeAtStart = -1
$smiLine = (& nvidia-smi.exe --query-gpu=memory.free --format=csv,nounits,noheader 2>$null | Select-Object -First 1)
if ($smiLine) { $freeAtStart = [int]$smiLine.Trim() }

if ($pkg -notmatch '\\venv\\Lib\\site-packages\\exllamav3\\__init__\.py$') {
    Write-Output ("ABORT: imported engine is not the installed one: " + $pkg)
    exit 2
}
$harnessHash = (Get-FileHash -Path $PSCommandPath -Algorithm SHA256).Hash

# --- Regimes -----------------------------------------------------------------------------------
# production = the deployed regime of 2026-10-24, expressed for v1.6.4: model 2.05 bpw, split mode
# with the tail 360 of 512 experts on the CPU, 16 worker threads, 262144 cache in FP16, chunk 4096,
# batch 2. Environment: only the three knobs production actually relies on plus the host-memory
# guard. EXL3_MOE_MEMOPS=0 is the
# DEPLOYED value, not the engine default (1.6.4 defaults to 1); Task 5 of the plan decides it.
# Deliberately NOT set: EXL3_MOE_STREAM_T and EXL3_MOE_STREAM_BATCH_EXPERTS. Production's
# start_tuned.ps1 carries the v1.4.9 values (6 / 48), which are no longer the defaults (per-device /
# 24) - Task 6 tests them as knobs instead of baking them in.
# Profilers are off by default: they cost throughput and the retune wants tok/s, not mechanism.
$REGIMES = @{
    "production" = @{
        Args = @("-m", $MODEL, "-cs", "262144", "-chunk_size", "4096", "-ambs", "2", "-cq", "2,2", "-max_length", "$MaxLength")
        Env  = [ordered]@{
            EXL3_MOE_MEMOPS      = "0"
            EXL3_MOE_CPU_PIN     = "1"
            EXL3_MOE_CPU_SWIZZLE = "1"
            # Host-memory guard, not a perf knob. The 3.68bpw model at split 400 offloads a 43.02 GB
            # expert arena to a separate worker process, leaving ~2.0 GB of host RAM available at
            # the deepest chunk; the engine's default reserve (2048 MB) aborts the load there, and
            # so does 1024 MB. The watcher holds the enforced floor (1200 MB, operator 2026-10-29),
            # so the engine guard is off and one margin is charged once.
            EXL3_HOST_MEM_RESERVE_MB = "0"
            # Determinism, not a perf knob. The OS grants a variable slice of the 42.9 GB CPU arena
            # as large pages (measured 0.87 GB in A1 vs 2.08 GB in A2, same config), and CPU expert
            # gathers run faster through huge pages - that alone moved tg 12-19 % between two
            # identical passes. Regular pages for every run make the arms comparable; any loss
            # against a lucky huge-page run belongs to the measurement, not to the knobs.
            EXL3_MOE_ARENA_HUGEPAGE = "0"
        }
    }
}

if (-not $REGIMES.ContainsKey($Regime)) {
    Write-Output ("ABORT: unknown regime '" + $Regime + "'. Known: " + (($REGIMES.Keys | Sort-Object) -join ", "))
    exit 2
}
$R = $REGIMES[$Regime]
$argsList = @($R.Args)
$argsList += @("-mcs", "$Split", "-mct", "$Threads")
if ($Extra -ne "") {
    foreach ($tok in ($Extra -split '\s+')) { if ($tok -ne "") { $argsList += $tok } }
}
if ($Short)          { $argsList += "-short" }
if (-not $NoSpecDec -and $Runner -eq "perf") { $argsList += "-sd" }
if ($Runner -eq "spec_decode") {
    # -max_length is perf.py's ladder length; spec_decode.py takes model_init args plus its own
    # a fixed-size array has no RemoveAt: rebuild without the flag and its value
    $keep = @()
    for ($j = 0; $j -lt $argsList.Count; $j++) {
        if ($argsList[$j] -eq "-max_length") { $j++; continue }
        $keep += $argsList[$j]
    }
    $argsList = $keep
}
if ($SkipPrefill)    { $argsList += "-spf" }
if ($SkipGen)        { $argsList += "-sg" }
if ($SkipWarmup)     { $argsList += "-swu" }

# --- Environment -------------------------------------------------------------------------------
foreach ($k in $R.Env.Keys) { Set-Item "env:$k" $R.Env[$k] }
$envApplied = [ordered]@{}
foreach ($k in $R.Env.Keys) { $envApplied[$k] = $R.Env[$k] }
if ($Prof) {
    foreach ($k in @("EXL3_MOE_CPU_PROF", "EXL3_MOE_CPU_SWAP_DEBUG", "EXL3_SPLIT_PROF", "EXL3_MOE_SUBMIT_PROF")) {
        Set-Item "env:$k" "1"; $envApplied[$k] = "1"
    }
}
if ($Unset -ne "") {
    foreach ($k in ($Unset -split ",")) {
        $k = $k.Trim()
        if ($k -ne "") { Remove-Item "env:$k" -ErrorAction SilentlyContinue; if ($envApplied.Contains($k)) { $envApplied.Remove($k) } }
    }
}
if ($Env -ne "") {
    foreach ($pair in ($Env -split ",")) {
        $kv = $pair.Trim()
        if ($kv -eq "") { continue }
        $i = $kv.IndexOf("=")
        if ($i -lt 1) { Write-Output ("ABORT: -Env entry needs KEY=VALUE, got: " + $kv); exit 2 }
        $k = $kv.Substring(0, $i).Trim()
        $v = $kv.Substring($i + 1).Trim()
        Set-Item "env:$k" $v
        $envApplied[$k] = $v
    }
}

# --- Run ---------------------------------------------------------------------------------------
# Every run gets its own log file: the UTC start time is part of the name, so a repeated tag never
# overwrites earlier evidence, and a log can be matched to a passes.csv row by filename. Nothing is
# deleted here -- logs are the analysis record and outlive the session that made them.
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$runId = "$Tag-$stamp"
$log = Join-Path $LOGS ("perf-" + $runId + ".log")
$err = Join-Path $LOGS ("perf-" + $runId + ".err.log")
if ((Test-Path $log) -or (Test-Path $err)) {
    Write-Output ("ABORT: log already exists for run id " + $runId + " -- refusing to overwrite")
    exit 2
}

$header = @(
    "",
    "=== pass     : $Tag",
    "=== run id   : $runId",
    "=== regime   : $Regime",
    "=== engine   : $engine",
    "=== package  : $pkg",
    "=== extension: $ext",
    "=== torch    : $torch",
    "=== gpu      : $gpu",
    "=== free vram at start (MiB): $freeAtStart",
    "=== harness  : $harnessHash",
    "=== args     : " + ($argsList -join " "),
    "=== env      : " + (($envApplied.Keys | Sort-Object | ForEach-Object { "$_=$($envApplied[$_])" }) -join " "),
    "=== started  : " + (Get-Date).ToString("s")
)

$t0 = Get-Date
$p = Start-Process -FilePath $PY -ArgumentList (@($PERF) + $argsList) -NoNewWindow -PassThru `
                   -RedirectStandardOutput $log -RedirectStandardError $err
$w = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$WATCH`"",
        "-Tag", $runId, "-TargetPid", $p.Id, "-MatchCmd", $MATCH_CMD,
        "-MinFreeVramMB", $MinFreeVramMB, "-MinFreeRamMB", $MinFreeRamMB,
        "-OutDir", "`"$RES`"")
$p.WaitForExit()
$elapsed = [int]((Get-Date) - $t0).TotalSeconds
New-Item -ItemType File -Force (Join-Path $RES ("stop-" + $runId + ".flag")) | Out-Null
$w.WaitForExit(60000) | Out-Null

# perf.py writes progress to stdout and worker profiler output to stderr; the header goes in last
# so a killed pass still leaves its partial metrics, and the file stays tail-able while it runs.
Get-Content $err -ErrorAction SilentlyContinue | Add-Content -Path $log -Encoding ASCII
$header += @("=== elapsed s: $elapsed")
$header | Add-Content -Path $log -Encoding ASCII

# --- Floors from the watcher samples -----------------------------------------------------------
$minVram = -1; $minRam = -1; $peakWs = 0; $samples = 0
$wCsv = Join-Path $RES ("watch-" + $runId + ".csv")
if (Test-Path $wCsv) {
    foreach ($row in (Get-Content $wCsv -ErrorAction SilentlyContinue | Select-Object -Skip 1)) {
        $c = @($row -split ",")
        if ($c.Count -lt 5) { continue }
        $samples++
        $fv = [int]$c[1]; $fr = [int]$c[3]; $ws = [int]$c[4]
        if ($fv -ge 0 -and ($minVram -lt 0 -or $fv -lt $minVram)) { $minVram = $fv }
        if ($fr -ge 0 -and ($minRam  -lt 0 -or $fr -lt $minRam))  { $minRam  = $fr }
        if ($ws -gt $peakWs) { $peakWs = $ws }
    }
}
$breach = Test-Path (Join-Path $RES ("BREACH-" + $runId + ".txt"))
# Belt and braces: a killed pass must never read as ok. If the watcher's floor was crossed at all
# (it writes BREACH and kills the tree), or the sampled minimum sits under a floor, this is a
# breach even when the BREACH file is missing for any reason.
if (-not $breach -and ($minVram -ge 0 -and $minVram -lt $MinFreeVramMB)) { $breach = $true }
if (-not $breach -and ($minRam -ge 0 -and $minRam -lt $MinFreeRamMB)) { $breach = $true }

# --- Parse -------------------------------------------------------------------------------------
$ppRows = New-Object System.Collections.ArrayList
$tgRows = New-Object System.Collections.ArrayList
$ansi = [char]27 + "\[[0-9;]*[A-Za-z]"
$raw = (Get-Content $log -Raw -ErrorAction SilentlyContinue)
if ($null -eq $raw) { $raw = "" }

# Exit code of the venv python is not readable (uv shim; measured 2026-10-24), so the verdict comes
# from the evidence: metrics parsed, no traceback, floors held. This must run AFTER $raw is loaded:
# it used to sit above, read $null every time, and let a crashed run report ok whenever the runner
# produced no pp/tg rows (found 2026-10-29 when a spec_decode run lost its prompts directory).
$traceback = $false
$perfPart = $raw
$marker = [regex]::Match($raw, '\r?\n=== pass     :')
if ($marker.Success) { $perfPart = $raw.Substring(0, $marker.Index) }
if ([regex]::IsMatch($perfPart, 'Traceback \(most recent call last\)')) { $traceback = $true }
foreach ($rawLine in (($raw -replace "`r", "`n") -split "`n")) {
    $l = $rawLine -replace $ansi, ""
    if ($l -match 'Length\s+(\d+):\s+([0-9.]+)\s+tokens/s') {
        [void]$ppRows.Add(@([int]$Matches[1], [double]::Parse($Matches[2], $inv)))
        continue
    }
    if ($l -match 'Context\s+(\d+):\s+(.+)$') {
        $ctx = [int]$Matches[1]
        $existing = $null
        foreach ($r in $tgRows) { if ($r[0] -eq $ctx) { $existing = $r; break } }
        if ($null -eq $existing) {
            $existing = @($ctx, @{ 1 = 0.0; 2 = 0.0; 3 = 0.0; 4 = 0.0 })
            [void]$tgRows.Add($existing)
        }
        foreach ($m in [regex]::Matches($Matches[2], 'S=(\d)\s+([0-9.]+)')) {
            $existing[1][[int]$m.Groups[1].Value] = [double]::Parse($m.Groups[2].Value, $inv)
        }
    }
}
$ppRows = @($ppRows | Where-Object { $_[1] -gt 0 })
$tgRows = @($tgRows | Where-Object { $_[1][1] -gt 0 })

# --- Verdict -----------------------------------------------------------------------------------
$status = "ok"
if ($breach) { $status = "breach" }
elseif ($traceback) { $status = "fail" }
elseif ($Runner -eq "perf" -and $ppRows.Count -eq 0 -and $tgRows.Count -eq 0) { $status = "fail" }

# --- Persist -----------------------------------------------------------------------------------
$passesCsv = Join-Path $RES "passes.csv"
if (-not (Test-Path $passesCsv)) {
    Set-Content -Path $passesCsv -Value "tag,run_id,finished_utc,regime,status,engine,ext,elapsed_s,max_length,min_free_vram_mb,min_free_ram_mb,peak_proc_ws_mb,samples,breach,env,split,threads,args,harness_sha256,log,runner" -Encoding ASCII
}
$envStr = (($envApplied.Keys | Sort-Object | ForEach-Object { "$_=$($envApplied[$_])" }) -join " ")
$argStr = (($argsList -join " ") -replace ",", ";")
$breachInt = 0
if ($breach) { $breachInt = 1 }
$row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20}" -f `
    $Tag, $runId, (Get-Date).ToString("s"), $Regime, $status, $engine, $extName, $elapsed, $MaxLength, `
    $minVram, $minRam, $peakWs, $samples, $breachInt, $envStr, $Split, $Threads, $argStr, $harnessHash, `
    ("perf-" + $runId + ".log"), $Runner
Add-Content -Path $passesCsv -Value $row -Encoding ASCII

$ppCsv = Join-Path $RES "pp.csv"
if (-not (Test-Path $ppCsv)) { Set-Content -Path $ppCsv -Value "tag,run_id,band,pp_tok_s" -Encoding ASCII }
foreach ($r in $ppRows) {
    Add-Content -Path $ppCsv -Value ("{0},{1},{2},{3}" -f $Tag, $runId, $r[0], ([string]::Format($inv, "{0:F2}", $r[1]))) -Encoding ASCII
}

$tgCsv = Join-Path $RES "tg.csv"
if (-not (Test-Path $tgCsv)) { Set-Content -Path $tgCsv -Value "tag,run_id,context,s1,s2,s3,s4" -Encoding ASCII }
foreach ($r in $tgRows) {
    $s = $r[1]
    Add-Content -Path $tgCsv -Value ("{0},{1},{2},{3},{4},{5},{6}" -f $Tag, $runId, $r[0],
        [string]::Format($inv, "{0:F2}", $s[1]), [string]::Format($inv, "{0:F2}", $s[2]),
        [string]::Format($inv, "{0:F2}", $s[3]), [string]::Format($inv, "{0:F2}", $s[4])) -Encoding ASCII
}

# --- Summary block (paste this into the task report) -------------------------------------------
Write-Output ("HARNESS sha256=$harnessHash")
Write-Output ("PASS tag=$Tag regime=$Regime status=$status elapsed_s=$elapsed engine=$engine ext=$extName")
Write-Output ("GPU  $gpu")
Write-Output ("FLOORS min_free_vram_mb=$minVram min_free_ram_mb=$minRam peak_proc_ws_mb=$peakWs samples=$samples breach=$breachInt")
if ($ppRows.Count -gt 0) {
    $parts = @()
    foreach ($r in $ppRows) { $parts += ("{0}={1}" -f $r[0], [string]::Format($inv, "{0:F2}", $r[1])) }
    Write-Output ("PP   " + ($parts -join " "))
}
if ($tgRows.Count -gt 0) {
    $parts = @()
    foreach ($r in $tgRows) { $parts += ("{0}={1}" -f $r[0], [string]::Format($inv, "{0:F2}", $r[1][1])) }
    Write-Output ("TG_S1 " + ($parts -join " "))
}
Write-Output ("CSV  " + $passesCsv + " ; " + $ppCsv + " ; " + $tgCsv + " ; " + $wCsv)

if ($status -eq "breach") {
    Write-Output ("BREACH tag=$Tag floors violated; see " + (Join-Path $RES ("BREACH-" + $Tag + ".txt")))
    exit 4
}
if ($status -eq "fail") { exit 5 }
exit 0
