# v1.6.4 retune - one wheel build under the watcher.
#
# WHY THIS EXISTS: the build is a ~33-minute run that holds ~8-10 GB of system RAM with MAX_JOBS=4.
# It gets the same floor enforcement as a measurement, and it records the source tree, the engine
# version it built and the wheel it produced, so a later reader never has to guess which tree a
# wheel came from. Deployed from bench/windows/; never edit the deployed copy.
#
# Usage:
#   run_build.ps1 -Tag build-stock -Src "C:\Users\yoho\Downloads\exl3-build-164\src"
#
# Expects the recipe next to the source: <Src>\..\_build.bat, copied from
# _a15_build.bat (see doc/win-build.md), which cd's into <Src>, sets MSVC/SDK/CUDA by hand and runs
# `setup.py bdist_wheel --dist-dir dist`.
#
# Output: <Src>\..\build_wheel.log (the recipe's own log), results/builds.csv, watcher CSV, and a
# BUILD block on stdout.

param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [Parameter(Mandatory = $true)][string] $Src,
    [string] $Root = "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165",
    [int] $MinFreeRamMB = 1200,
    [int] $MinFreeVramMB = 100
)

$ErrorActionPreference = "Continue"

$PY    = "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe"
$WATCH = Join-Path $Root "watch.ps1"
$RES   = Join-Path $Root "results"
New-Item -ItemType Directory -Force $RES | Out-Null

$Src = (Resolve-Path $Src).Path
$bat = Join-Path (Split-Path $Src -Parent) "_build.bat"
if (-not (Test-Path $bat)) { Write-Output "ABORT: no build recipe at $bat"; exit 2 }
if (-not (Test-Path $PY))  { Write-Output "ABORT: no venv python at $PY"; exit 2 }

$log = Join-Path (Split-Path $Src -Parent) "build_wheel.log"
Remove-Item $log -ErrorAction SilentlyContinue

# Every build gets its own copy of the recipe's log in the results directory: the recipe rewrites
# build_wheel.log in place, so without this a later build in the same src tree would destroy the
# earlier one's evidence (operator requirement, 2026-10-29).
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$runId = "$Tag-$stamp"

$other = @(Get-CimInstance Win32_Process |
           Where-Object { $_.Name -match '^(python|cl|nvcc)' -and $_.ProcessId -ne $PID -and
                          $_.CommandLine -match 'perf\.py|spec_decode\.py|tabbyAPI|bdist_wheel' })
if ($other.Count -gt 0) {
    Write-Output "ABORT: another pass or build is running:"
    $other | ForEach-Object { Write-Output ("  " + $_.ProcessId + " " + $_.CommandLine) }
    exit 3
}

$harnessHash = (Get-FileHash -Path $PSCommandPath -Algorithm SHA256).Hash
$t0 = Get-Date
Write-Output ("HARNESS sha256=$harnessHash")
Write-Output ("BUILD_START tag=$Tag src=$Src recipe=$bat at " + $t0.ToString("s"))
Set-Location (Split-Path $Src -Parent)   # never let a UNC working directory (WSL launch) leak in

$p = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$bat`"") -NoNewWindow -PassThru
$w = Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -PassThru -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$WATCH`"",
        "-Tag", $runId, "-TargetPid", $p.Id,
        "-MinFreeVramMB", $MinFreeVramMB, "-MinFreeRamMB", $MinFreeRamMB,
        "-OutDir", "`"$RES`"")
$p.WaitForExit()
$exit = $p.ExitCode
if ($null -eq $exit -or "$exit" -eq "") { $exit = -1 }
$elapsed = [int]((Get-Date) - $t0).TotalSeconds
New-Item -ItemType File -Force (Join-Path $RES ("stop-" + $runId + ".flag")) | Out-Null
$w.WaitForExit(60000) | Out-Null
$breach = Test-Path (Join-Path $RES ("BREACH-" + $runId + ".txt"))

# Keep this build's recipe log: the recipe rewrites build_wheel.log in place, so the copy in the
# results directory is what makes two builds in one src tree distinguishable afterwards.
$buildLog = Join-Path $RES ("build-" + $runId + ".log")
if (Test-Path $log) { Copy-Item $log $buildLog -Force }

$wheel = Get-ChildItem -Path (Join-Path $Src "dist") -Filter "*.whl" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime | Select-Object -Last 1
$wheelName = "NONE"
$wheelSize = 0
$wheelPath = "-"
if ($wheel) { $wheelName = $wheel.Name; $wheelSize = [int]($wheel.Length / 1MB); $wheelPath = $wheel.FullName }

if ($breach) {
    Write-Output ("BUILD_BREACH tag=$Tag run_id=$runId see " + (Join-Path $RES ("BREACH-" + $runId + ".txt")))
    Write-Output ("BUILD tag=$Tag exit=$exit elapsed_s=$elapsed wheel=$wheelName wheel_mb=$wheelSize breach=1")
    exit 4
}

$csv = Join-Path $RES "builds.csv"
if (-not (Test-Path $csv)) { Set-Content -Path $csv -Value "tag,run_id,finished_utc,src,exit,elapsed_s,wheel,wheel_mb,harness_sha256,log" -Encoding ASCII }
$row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f $Tag, $runId, (Get-Date).ToString("s"), $Src, $exit, $elapsed, $wheelName, $wheelSize, $harnessHash, ("build-" + $runId + ".log")
Add-Content -Path $csv -Value $row -Encoding ASCII

Write-Output ("BUILD tag=$Tag run_id=$runId exit=$exit elapsed_s=$elapsed src=$Src")
Write-Output ("BUILD_WHEEL name=$wheelName mb=$wheelSize path=$wheelPath")
Write-Output ("BUILD_LOG " + $log)
if ($exit -ne 0) { exit $exit }
exit 0
