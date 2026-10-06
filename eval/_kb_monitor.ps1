# Monitor the box while a Phase C boot runs, so a slow/fast difference can be
# attributed instead of guessed at.
#
# Why: the 250k stage is bimodal (prefill pp median 1670-1741 vs 1520-1525, a
# ~10% step) and the mode is constant WITHIN a boot but differs BETWEEN boots.
# Anything that varies per boot and moves prefill throughput is a candidate.
# Candidates logged here:
#   - pagefile I/O        (Memory\PageReadsPersec / PageWritesPersec)
#   - physical headroom   (Memory\AvailableMBytes, PercentCommittedBytesInUse)
#   - WDDM shared memory  (GPUAdapterMemory SharedUsage) -> VRAM spill indicator
#   - CPU clock behaviour (Processor Performance % of nominal) -> boost / thermal
#
# Usage: powershell -NoProfile -File _kb_monitor.ps1 -Out <csv> -Minutes 20
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Minutes = 20,
    [int]$IntervalSec = 2
)

$ErrorActionPreference = 'SilentlyContinue'

function Get-GpuSharedMB {
    $g = Get-CimInstance -Namespace root/cimv2 `
        -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory
    if (-not $g) { return 0 }
    [math]::Round((($g | Measure-Object -Property SharedUsage -Sum).Sum) / 1MB, 1)
}

"ts,availMB,commitPct,pageReads,pageWrites,gpuSharedMB,cpuPerfPct" | Set-Content $Out
$end = (Get-Date).AddMinutes($Minutes)
while ((Get-Date) -lt $end) {
    $m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    # "% Processor Performance" = achieved clock / nominal clock. The
    # Win32_PerfFormattedData_* Processor class reports 0 for this on this box;
    # Get-Counter is the one that works (idle reads ~117, i.e. boost).
    $perf = -1
    try {
        $s = (Get-Counter '\Processor Information(_Total)\% Processor Performance' `
              -ErrorAction Stop).CounterSamples[0]
        $perf = [math]::Round($s.CookedValue, 1)
    } catch { }
    $line = "{0},{1},{2},{3},{4},{5},{6}" -f `
        (Get-Date -Format 'HH:mm:ss'),
        [math]::Round($m.AvailableBytes / 1MB, 0),
        $m.PercentCommittedBytesInUse,
        $m.PageReadsPersec,
        $m.PageWritesPersec,
        (Get-GpuSharedMB),
        $perf
    $line | Add-Content $Out
    Start-Sleep -Seconds $IntervalSec
}
Write-Host ("monitor done -> {0} ({1} samples)" -f $Out, ((Get-Content $Out).Count - 1))