# v1.6.4 retune - resource watcher and floor enforcer.
#
# WHY THIS EXISTS: every run on this box (measurement, build, boot, deployment smoke test) must
# hold two floors at every sample, not just at the start: 100 MB free VRAM and 1200 MB free system
# RAM (operator decision 2026-10-29: the 3.68bpw load transient cannot survive tighter floors
# RAM. This script is the only sanctioned way to hold them, so that no run can quietly swap the
# machine into a stall and make the numbers look like a config effect.
#
# WHY -MatchCmd EXISTS: the tabbyAPI venv's python.exe is a uv shim. Start-Process returns the shim,
# which immediately spawns the real interpreter (C:\Python312\python.exe) and stays behind at ~3 MB
# of working set. Watching the shim measures nothing, and killing it orphans the interpreter. With
# -MatchCmd the watcher resolves the real process by its command line (one pass runs at a time, so
# the match is unambiguous) and watches/kills that.
#
# Usage, measurement (run_pass.ps1 starts this for you):
#   watch.ps1 -Tag memops1-a -TargetPid <launcher pid> -MatchCmd "perf\.py"
# Usage, build (cmd.exe is the real process, no shim):
#   watch.ps1 -Tag build-stock -TargetPid <cmd pid>
# Usage, blind sampling:
#   watch.ps1 -Tag idle-check -MaxSeconds 10
#
# Ends when any of: the watched process exits, stop-<Tag>.flag appears, or MaxSeconds elapses.
# On a floor breach it writes BREACH-<Tag>.txt and kills the watched process tree.
#
# Outputs (in -OutDir):
#   watch-<Tag>.csv      t_iso,free_vram_mb,used_vram_mb,free_ram_mb,proc_ws_mb
#                        (proc_ws_mb is the sum over the watched process AND its descendants:
#                         the CPU-MoE expert arena lives in a child worker)
#   BREACH-<Tag>.txt     only on breach: time, measured values, action taken
#   stop-<Tag>.flag      created by the caller to end the watch
#   stdout summary       one WATCH line, parseable with a regex

param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [int] $TargetPid = 0,
    [string] $MatchCmd = "",
    [int] $IntervalMs = 1000,
    [int] $MinFreeVramMB = 200,
    [int] $MinFreeRamMB = 2048,
    [int] $MaxSeconds = 0,
    [int] $ResolveTimeoutSec = 60,
    [string] $OutDir = "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165\results"
)

$ErrorActionPreference = "Continue"

New-Item -ItemType Directory -Force $OutDir | Out-Null
$csv        = Join-Path $OutDir "watch-$Tag.csv"
$stopPath   = Join-Path $OutDir "stop-$Tag.flag"
$breachPath = Join-Path $OutDir "BREACH-$Tag.txt"
Remove-Item $csv, $stopPath, $breachPath -ErrorAction SilentlyContinue
Set-Content -Path $csv -Value "t_iso,free_vram_mb,used_vram_mb,free_ram_mb,proc_ws_mb" -Encoding ASCII

$minVram    = [int]::MaxValue
$minRam     = [int]::MaxValue
$peakWs     = 0
$samples    = 0
$breach     = $false
$watchPid   = $TargetPid
$resolved   = ($MatchCmd -eq "")
$t0         = Get-Date

function Resolve-WatchedPid {
    if ($MatchCmd -eq "") { return $TargetPid }
    $cands = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -match $MatchCmd })
    if ($cands.Count -eq 0) { return 0 }
    $child = @($cands | Where-Object { $_.ParentProcessId -eq $TargetPid })
    if ($child.Count -gt 0) { return [int]$child[0].ProcessId }
    $sorted = @($cands | Sort-Object CreationDate)
    return [int]$sorted[$sorted.Count - 1].ProcessId
}

while ($true)
{
    if (Test-Path $stopPath) { break }
    if ($MaxSeconds -gt 0)
    {
        if ([int]((Get-Date) - $t0).TotalSeconds -ge $MaxSeconds) { break }
    }

    if (-not $resolved)
    {
        $pid2 = Resolve-WatchedPid
        if ($pid2 -gt 0) { $watchPid = $pid2; $resolved = $true }
        elseif ([int]((Get-Date) - $t0).TotalSeconds -ge $ResolveTimeoutSec) { break }
        else { Start-Sleep -Milliseconds 200; continue }
    }

    $freeVram = -1
    $usedVram = -1
    $line = (& nvidia-smi.exe --query-gpu=memory.free,memory.used --format=csv,nounits,noheader 2>$null | Select-Object -First 1)
    if ($line)
    {
        $parts = @($line -split "," | ForEach-Object { $_.Trim() })
        if ($parts.Count -ge 2)
        {
            $freeVram = [int]$parts[0]
            $usedVram = [int]$parts[1]
        }
    }

    $freeRam = -1
    try
    {
        $os = Get-CimInstance Win32_OperatingSystem -Property FreePhysicalMemory -ErrorAction Stop
        $freeRam = [int]($os.FreePhysicalMemory / 1024)
    }
    catch { }

    # Working set of the WHOLE tree, not just the launcher: the 30-43 GB CPU-MoE expert arena lives
    # in a child worker process, so watching perf.py alone reported ~3.7 GB for a 45 GB run and hid
    # every host-side growth (critique/improvement I3c).
    $ws = 0
    if ($watchPid -gt 0)
    {
        try {
            [void](Get-Process -Id $watchPid -ErrorAction Stop)
        }
        catch { break }   # the watched process exited: a pass is over
        try {
            $all = Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, WorkingSetSize -ErrorAction Stop
            $children = @{}
            foreach ($pr in $all) {
                if (-not $children.ContainsKey([int]$pr.ParentProcessId)) { $children[[int]$pr.ParentProcessId] = @() }
                $children[[int]$pr.ParentProcessId] += $pr
            }
            $queue = New-Object System.Collections.Queue
            $queue.Enqueue($watchPid)
            $seen = @{}
            while ($queue.Count -gt 0) {
                $cur = [int]$queue.Dequeue()
                if ($seen.ContainsKey($cur)) { continue }
                $seen[$cur] = $true
                foreach ($pr in $all) {
                    if ([int]$pr.ProcessId -eq $cur) { $ws += [int]($pr.WorkingSetSize / 1MB) }
                }
                if ($children.ContainsKey($cur)) { foreach ($ch in $children[$cur]) { $queue.Enqueue([int]$ch.ProcessId) } }
            }
        } catch { }
    }

    $samples++
    if ($freeVram -ge 0 -and $freeVram -lt $minVram) { $minVram = $freeVram }
    if ($freeRam  -ge 0 -and $freeRam  -lt $minRam)  { $minRam  = $freeRam }
    if ($ws -gt $peakWs) { $peakWs = $ws }

    $row = "{0},{1},{2},{3},{4}" -f (Get-Date).ToString("s"), $freeVram, $usedVram, $freeRam, $ws
    Add-Content -Path $csv -Value $row -Encoding ASCII

    $vramLow = ($freeVram -ge 0 -and $freeVram -lt $MinFreeVramMB)
    $ramLow  = ($freeRam  -ge 0 -and $freeRam  -lt $MinFreeRamMB)
    if ($vramLow -or $ramLow)
    {
        $breach = $true
        $msg = @(
            ("BREACH tag={0} at {1}" -f $Tag, (Get-Date).ToString("s")),
            ("free_vram_mb={0} floor={1} low={2}" -f $freeVram, $MinFreeVramMB, $vramLow),
            ("free_ram_mb={0} floor={1} low={2}" -f $freeRam, $MinFreeRamMB, $ramLow),
            ("watched_pid={0} launcher_pid={1}" -f $watchPid, $TargetPid)
        )
        if ($watchPid -gt 0)
        {
            try { & taskkill.exe /PID $watchPid /T /F | Out-Null; $msg += "action=killed process tree" }
            catch { $msg += "action=kill failed" }
        }
        else { $msg += "action=none (no pid)" }
        Set-Content -Path $breachPath -Value $msg -Encoding ASCII
        break
    }

    Start-Sleep -Milliseconds $IntervalMs
}

$minVramOut = if ($minVram -eq [int]::MaxValue) { -1 } else { $minVram }
$minRamOut  = if ($minRam  -eq [int]::MaxValue) { -1 } else { $minRam }
$breachInt  = if ($breach) { 1 } else { 0 }
Write-Output "WATCH tag=$Tag samples=$samples watched_pid=$watchPid min_free_vram_mb=$minVramOut min_free_ram_mb=$minRamOut peak_proc_ws_mb=$peakWs breach=$breachInt"
if ($breach) { Write-Output ("BREACH_FILE=" + $breachPath) }
exit $(if ($breach) { 4 } else { 0 })
