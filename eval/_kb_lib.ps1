# _kb_lib.ps1 -- shared harness library for the Flash-Next knob battery
# (wiki/plans/flash-knobs-benchmark.md). Dot-source this; do not run it.
#
# Why PowerShell and not batch: spec_decode.py's -single takes
# "agentic, code", which CONTAINS A SPACE. A PowerShell array element keeps
# it one argv token; every batch form tried re-splits on the space and
# argparse rejects the stray "code". Same reason _b2_mtp_sweep.ps1 is PS.
#
# Everything writes under logs\kb\ and stays untracked. config.yml is never
# written by this library except by _kb_phase_c.ps1, which edits it in the
# working tree only and never commits.

$KB = @{
    Tabby   = 'C:\Users\yoho\Downloads\tabbyAPI'
    Py      = 'C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe'
    X3      = 'C:\Users\yoho\Downloads\exllamav3-kvarn'
    Perf    = 'C:\Users\yoho\Downloads\exllamav3-kvarn\eval\perf.py'
    Spec    = 'C:\Users\yoho\Downloads\exllamav3-kvarn\eval\spec_decode.py'
    Prompts = 'C:\Users\yoho\Downloads\exllamav3-kvarn\eval\prompts'
    Model   = 'D:\llms\Qwen3.8-Flash-Next-exl3-3.05bpw'
    LogRoot = 'C:\Users\yoho\Downloads\tabbyAPI\logs\kb'
}

# Guards, verbatim from the plan's Guards section.
$KBGuard = @{
    RamFreeMinBeforeMB = 2048   # abort unless >= 2GB free before the run
    RamFreeKillMB      = 1024   # kill at 1GB during the run
    VramFreeKillMB     = 200    # kill under 200MB free during the run
    PollSeconds        = 2
}

# Baseline env == start_tuned.ps1 exactly. Every arm starts from this and
# overrides single keys, so "baseline" means production, not bench.ps1's
# lighter env (which omits MEMOPS/ZERO_COPY/STREAM_T).
$KBBaseEnv = @{
    EXL3_MOE_CPU_THREADS          = '8'
    EXL3_MOE_CPU_PIN              = '1'
    EXL3_MOE_CPU_SWIZZLE          = '1'
    EXL3_MOE_MEMOPS               = '0'
    EXL3_MOE_ZERO_COPY            = '1'
    EXL3_MOE_STREAM_T             = '6'
    EXL3_MOE_STREAM_BATCH_EXPERTS = '48'
}

# Determinism mode (plan §0.8). The DSA staged-prefill path
# (EXL3_DSA_QC_STAGE, default 1) gathers+dequantizes once into an fp16 transient
# for ~3.9x faster attention at 16k, but it is gated on
# pool_len <= EXL3_DSA_QC_STAGE_MAX_ENTRIES (1M entries, ~1.2 GB transient) and
# pool_len is the ACTUAL context -- so past that cap the same config silently
# takes the online path, with different numerics. Setting this to 0 pins one
# path at the cost of the prefill speedup. Opt-in per run via -Deterministic.
# Per-process VRAM needs a second nvidia-smi query type and returns [N/A] under
# WDDM, so it is opt-in rather than paid on every poll.
$KBWantProcVram = $false

$KBDeterminismEnv = @{
    EXL3_DSA_QC_STAGE = '0'
}

# Current production model settings, from the live config.yml.
$KBBaseFlags = @('-m', $KB.Model, '-mcl', '38', '-cq', '5,4', '-cs', '262144',
                 '-chunk_size', '4096', '-ambs', '2')
# NOTE: no -mct. Threads are controlled purely by EXL3_MOE_CPU_THREADS so
# the offline knob and the live knob are the same knob. model_init defaults
# -mct to None and falls back to the env var.

# ---------------------------------------------------------------- measurements

function Get-KBRamMB {
    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
}

function Get-KBVram {
    $o = & nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>$null
    if (-not $o) { return $null }
    $p = ($o -split ',') | ForEach-Object { $_.Trim() }
    [pscustomobject]@{ UsedMB = [int]$p[0]; FreeMB = [int]$p[1] }
}

function Get-KBStamp { (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') }

# ---- per-process memory (plan §0.8): the operator wants OUR process's usage,
# not the whole box. nvidia-smi memory.used and FreePhysicalMemory are both
# system-wide, so they attribute the desktop, the RDP session and any other
# process to our result. Two caveats, handled explicitly rather than silently:
#   - RAM: exact. WorkingSet64 is resident, PrivateMemorySize64 is commit.
#   - VRAM: on WDDM, nvidia-smi per-process used_gpu_memory usually reports
#     [N/A] (observed on this box). When it does, we fall back to the
#     system-wide delta and LABEL it, so the number is never mistaken for a
#     per-process figure.

function Get-KBDescendants([int]$RootPid) {
    $all = @{}
    foreach ($p in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
        $all[[int]$p.ProcessId] = [int]$p.ParentProcessId
    }
    $tree = @($RootPid)
    for ($i = 0; $i -lt $tree.Count; $i++) {
        foreach ($kv in $all.GetEnumerator()) {
            if ($kv.Value -eq $tree[$i] -and $tree -notcontains $kv.Key) { $tree += $kv.Key }
        }
    }
    return $tree
}

function Get-KBGpuWatts {
    $o = & nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits 2>$null
    if (-not $o) { return $null }
    $p = ($o -split ',') | ForEach-Object { $_.Trim() }
    return [pscustomobject]@{ Watts = [double]$p[0]; LimitW = [double]$p[1] }
}

# Instantaneous CPU utilisation of OUR process tree, percent of all logical
# cores. Normalled by logical core count so it is comparable across the 16C/32T
# box and does not read as "3200%" when every core is busy.
function Get-KBLogicalCores {
    try { return [int](Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).NumberOfLogicalProcessors }
    catch { return 0 }
}

# ---- one consolidated probe (plan §0.8) -------------------------------------
# The monitor loop used to issue 3 separate `nvidia-smi` calls (vram, power,
# compute-apps) plus TWO Win32_OperatingSystem queries per tick -- CIM is slow
# enough that this dominated the poll. Now every field the battery records comes
# from ONE GPU query, ONE OS query and ONE perf query, and the guard reads the
# same sample instead of re-querying.
function Get-KBSample([int]$ProcId = 0) {
    $s = [ordered]@{
        vramUsedMB = $null; vramFreeMB = $null; gpuWatts = $null; gpuLimitW = $null
        gpuUtilPct = $null; procVramMB = $null
        sysRamFreeMB = $null; sysRamTotalMB = $null
        procRamWSMB = $null; procRamPrivMB = $null; procCpuPct = $null; procCount = 0
    }

    # ONE gpu-level query: memory + power + utilisation together.
    $g = & nvidia-smi --query-gpu=memory.used,memory.free,power.draw,power.limit,utilization.gpu `
                      --format=csv,noheader,nounits 2>$null
    if ($g) {
        $p = ($g -split ',') | ForEach-Object { $_.Trim() }
        if ($p.Count -ge 5) {
            $s.vramUsedMB = [int]$p[0]; $s.vramFreeMB = [int]$p[1]
            if ($p[2] -match '[\d.]+') { $s.gpuWatts = [double]$p[2] }
            if ($p[3] -match '[\d.]+') { $s.gpuLimitW = [double]$p[3] }
            if ($p[4] -match '[\d.]+') { $s.gpuUtilPct = [double]$p[4] }
        }
    }

    # ONE OS query, reused for the guard and for the system-wide RAM peak.
    $os = Get-CimInstance Win32_OperatingSystem
    if ($os) {
        $s.sysRamFreeMB = [math]::Round($os.FreePhysicalMemory / 1024, 0)
        $s.sysRamTotalMB = [math]::Round($os.TotalVisibleMemorySize / 1024, 0)
    }

    if ($ProcId -gt 0) {
        $tree = @(Get-KBDescendants $ProcId)
        $s.procCount = $tree.Count
        $ws = 0; $priv = 0
        foreach ($q in $tree) {
            $p2 = Get-Process -Id $q -ErrorAction SilentlyContinue
            if ($p2) { $ws += $p2.WorkingSet64; $priv += $p2.PrivateMemorySize64 }
        }
        $s.procRamWSMB = [math]::Round($ws / 1MB, 0)
        $s.procRamPrivMB = [math]::Round($priv / 1MB, 0)

        # ONE perf query for the whole tree, instead of Get-Counter per PID.
        $cores = Get-KBLogicalCores
        if ($cores -gt 0) {
            try {
                $rows = Get-CimInstance Win32_PerfFormattedData_PerfProc_Process -ErrorAction Stop |
                        Where-Object { $tree -contains [int]$_.IDProcess -and $_.Name -ne 'Idle' }
                $cpu = ($rows | Measure-Object -Property PercentProcessorTime -Sum).Sum
                # PercentProcessorTime is already a percentage PER LOGICAL CORE,
                # so the tree sums to (cores * 100) at saturation. Dividing by
                # cores yields percent-of-machine directly -- do NOT multiply by
                # 100 again, which reported 5415% for a genuinely ~54% load.
                if ($cpu) { $s.procCpuPct = [math]::Round($cpu / $cores, 1) }
            } catch { }
        }

        # Per-process VRAM is a SEPARATE nvidia-smi query type and usually N/A
        # under WDDM, so it is only attempted when explicitly wanted.
        if ($KBWantProcVram) {
            $a = & nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>$null
            if ($a) {
                foreach ($r in $a) {
                    $c = ($r -split ',') | ForEach-Object { $_.Trim() }
                    if ($c.Count -ge 2 -and $c[1] -ne '[N/A]' -and $c[1] -ne '' -and [int]$c[0] -eq $ProcId) {
                        $s.procVramMB = [int]$c[1]
                    }
                }
            }
        }
    }
    return [pscustomobject]$s
}

# Strip ANSI so the print()s in perf.py / spec_decode.py parse as plain text.
function Get-KBPlain([string]$path) {
    if (-not (Test-Path $path)) { return '' }
    (Get-Content $path -Raw) -replace "$([char]27)\[[0-9;]*m", ''
}

# ------------------------------------------------------------------- env setup

function Set-KBEnv([hashtable]$overrides, [switch]$Deterministic) {
    # Clear every EXL3_* first so an arm can never inherit a stale knob
    # from a previous arm -- the single biggest source of silent A/B errors.
    # Widened from EXL3_MOE_* to EXL3_* so the determinism keys clear too.
    Get-ChildItem env: | Where-Object { $_.Name -like 'EXL3_*' } |
        ForEach-Object { Remove-Item "env:$($_.Name)" -ErrorAction SilentlyContinue }
    $env:PYTHONPATH = ''
    foreach ($k in $KBBaseEnv.Keys) { Set-Item "env:$k" $KBBaseEnv[$k] }
    if ($Deterministic) {
        foreach ($k in $KBDeterminismEnv.Keys) { Set-Item "env:$k" $KBDeterminismEnv[$k] }
    }
    if ($overrides) {
        foreach ($k in $overrides.Keys) {
            if ($null -eq $overrides[$k]) {
                Remove-Item "env:$k" -ErrorAction SilentlyContinue
            } else {
                Set-Item "env:$k" $overrides[$k]
            }
        }
    }
}

function Get-KBEnvSig {
    # Built via [Environment]::GetEnvironmentVariable, not "$env:$_" -- a
    # dynamic $env: reference is a parse error in PowerShell (needs ${env:$_}).
    $parts = @()
    $keys = @($KBBaseEnv.Keys) + @($KBDeterminismEnv.Keys)
    foreach ($k in ($keys | Sort-Object -Unique)) {
        $parts += ('{0}={1}' -f $k, [Environment]::GetEnvironmentVariable($k))
    }
    return ($parts -join ' ')
}

# ----------------------------------------------------------------------- runs

function Invoke-KBRun {
    <#
      Runs one python measurement under the guards, polling VRAM/RAM while it
      runs and killing it the moment a guard trips. Returns a row object; also
      appends it to the phase's results.jsonl.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Script,
        [string[]]$PyArgs = @(),
        [hashtable]$EnvSet = $null,
        [int]$TimeoutMin = 45,
        [int]$Retries = 2,
        # Log file stem. MUST differ per tier: the name alone collided across
        # tiers (a07 at tier 1 overwrote a07 at tier 0), which let a later tier
        # silently replace an earlier tier's numbers.
        [string]$LogTag = '',
        # Pin the DSA prefill path (EXL3_DSA_QC_STAGE=0) for A/B runs where
        # determinism matters more than wall-clock. See plan 0.8.
        [switch]$Deterministic,
        [string]$Jsonl = $null
    )

    $row = [ordered]@{
        name       = $Name
        started    = Get-KBStamp
        script     = $Script
        args       = ($PyArgs -join ' ')
        env        = $null
        status     = 'pending'
        guard      = $null
        exit       = $null
        ramBefore  = $null
        ramAfter   = $null
        vramBefore = $null
        vramAfter  = $null
        vramPeak   = $null
        vramMinFree = $null
        elapsedSec = $null
        out        = $null
        err        = $null
        note       = $null
        procRamPeakMB    = $null   # our process tree, max working set
        procPrivatePeakMB = $null   # our process tree, max commit charge
        procVramPeakMB   = $null   # per-process if WDDM allows, else $null
        procVramPerProcess = $null
        sysRamFreeMinMB    = $null   # system-wide free RAM, min during run
        sysRamUsedPeakMB  = $null   # system-wide used RAM (total - free), peak
        cpuPctPeak         = $null   # our tree, % of all logical cores, peak
        gpuWattsPeak       = $null   # board power, peak
        gpuWattsLimit      = $null
        gpuUtilPeakPct     = $null   # SM utilisation, peak
    }

    Set-KBEnv $EnvSet -Deterministic:$Deterministic
    $row.env = Get-KBEnvSig
    $row.ramBefore = Get-KBRamMB
    $v0 = Get-KBVram
    if ($v0) { $row.vramBefore = $v0.FreeMB }

    Write-Host ("  {0,-34} ram={1}MB vramFree={2}MB" -f $Name, $row.ramBefore,
                $(if ($v0) { $v0.FreeMB } else { '?' }))

    # Pre-run guard: abort before burning a load, not after.
    if ($row.ramBefore -lt $KBGuard.RamFreeMinBeforeMB) {
        $row.status = 'ABORT_RAM_PRE'
        $row.guard  = "ram free $($row.ramBefore)MB < $($KBGuard.RamFreeMinBeforeMB)MB before run"
        Write-Host "    ABORT (guard): $($row.guard)" -ForegroundColor Red
        Write-KBRow $row $Jsonl
        return [pscustomobject]$row
    }

    if (-not (Test-Path $KB.LogRoot)) { New-Item -ItemType Directory -Force $KB.LogRoot | Out-Null }
    $stem = if ($LogTag) { $LogTag } else { $Name }
    $out = Join-Path $KB.LogRoot "$stem.out"
    $err = Join-Path $KB.LogRoot "$stem.err"
    $row.out = $out
    $row.err = $err

    # Start-Process joins ArgumentList with spaces and adds no quotes, so any
    # token containing a space must be quoted here or it re-splits in the child.
    $argList = @($Script) + @($PyArgs | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    })

    # perf.py has a documented ZeroDivisionError flake on the -short sweep
    # (PERF_FINDINGS.md: "Length-0 ZeroDivisionError on reruns (timer
    # resolution) -- just relaunch"). Retry ONLY that. A guard trip or a CUDA
    # failure is real signal and must never be retried into a pass.
    $maxAttempts = 1 + $Retries
    $row.attempts = 0
    $cores = Get-KBLogicalCores

    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $peakUsed = 0; $minFree = [int]::MaxValue
        $peakProcRam = 0; $peakProcPriv = 0; $peakProcVram = 0
        $sysRamMin = [int]::MaxValue; $sysRamUsedPeak = 0
        $cpuPeak = 0.0; $wPeak = 0.0; $utilPeak = 0.0
        $trip = $null
        $row.attempts = $attempt

        $sw = [Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath $KB.Py -ArgumentList $argList `
                              -WorkingDirectory 'C:\' -NoNewWindow -PassThru `
                              -RedirectStandardOutput $out -RedirectStandardError $err
        # Touch .Handle immediately: without it PowerShell does not retain the
        # process handle and $proc.ExitCode comes back $null after the child exits.
        $null = $proc.Handle

        while ($true) {
            if ($proc.HasExited) { break }
            Start-Sleep -Seconds $KBGuard.PollSeconds
            if ($proc.HasExited) { break }

            # ONE consolidated probe feeds the guards AND every recorded field.
            $s = Get-KBSample $proc.Id
            if ($s.vramUsedMB -gt $peakUsed) { $peakUsed = $s.vramUsedMB }
            if ($s.vramFreeMB -lt $minFree) { $minFree = $s.vramFreeMB }
            if ($s.procRamWSMB -gt $peakProcRam) { $peakProcRam = $s.procRamWSMB }
            if ($s.procRamPrivMB -gt $peakProcPriv) { $peakProcPriv = $s.procRamPrivMB }
            if ($s.procVramMB -and $s.procVramMB -gt $peakProcVram) { $peakProcVram = $s.procVramMB }
            if ($s.sysRamFreeMB -lt $sysRamMin) { $sysRamMin = $s.sysRamFreeMB }
            if ($s.sysRamTotalMB -and ($s.sysRamTotalMB - $s.sysRamFreeMB) -gt $sysRamUsedPeak) {
                $sysRamUsedPeak = $s.sysRamTotalMB - $s.sysRamFreeMB
            }
            if ($s.procCpuPct -and $s.procCpuPct -gt $cpuPeak) { $cpuPeak = $s.procCpuPct }
            if ($s.gpuWatts -and $s.gpuWatts -gt $wPeak) { $wPeak = $s.gpuWatts }
            if ($s.gpuUtilPct -and $s.gpuUtilPct -gt $utilPeak) { $utilPeak = $s.gpuUtilPct }
            $row.gpuWattsLimit = $s.gpuLimitW
            $row.procVramPerProcess = [bool]$s.procVramMB

            if ($s.vramFreeMB -lt $KBGuard.VramFreeKillMB) {
                $trip = "VRAM free $($s.vramFreeMB)MB < $($KBGuard.VramFreeKillMB)MB"
                break
            }
            if ($s.sysRamFreeMB -lt $KBGuard.RamFreeKillMB) {
                $trip = "RAM free $($s.sysRamFreeMB)MB < $($KBGuard.RamFreeKillMB)MB"
                break
            }

            if ($sw.Elapsed.TotalMinutes -gt $TimeoutMin) {
                $trip = "TIMEOUT after $TimeoutMin min"
                break
            }
        }

        if ($trip) {
            # Kill the whole tree: python spawns worker threads, not children,
            # but taskkill /T is the reliable way to be sure nothing survives.
            & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null
            $sw.Stop()
            $row.elapsedSec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
            $row.vramPeak   = $peakUsed
            $row.vramMinFree = if ($minFree -eq [int]::MaxValue) { $null } else { $minFree }
            $row.procRamPeakMB = $peakProcRam
            $row.procPrivatePeakMB = $peakProcPriv
            $row.procVramPeakMB = if ($peakProcVram -gt 0) { $peakProcVram } else { $null }
            $row.sysRamFreeMinMB = if ($sysRamMin -eq [int]::MaxValue) { $null } else { $sysRamMin }
            $row.cpuPctPeak = if ($cpuPeak -gt 0) { $cpuPeak } else { $null }
            $row.gpuWattsPeak = if ($wPeak -gt 0) { [math]::Round($wPeak, 1) } else { $null }
            $row.gpuUtilPeakPct = if ($utilPeak -gt 0) { [math]::Round($utilPeak, 0) } else { $null }
            $row.guard  = $trip
            $row.status = 'GUARD_KILL'
            Write-Host "    GUARD KILL: $trip" -ForegroundColor Red
            break
        }

        $row.exit = $proc.ExitCode
        $row.status = if ($proc.ExitCode -eq 0) { 'OK' } else { 'FAIL' }
        $sw.Stop()
        $row.elapsedSec = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        $row.vramPeak   = $peakUsed
        $row.vramMinFree = if ($minFree -eq [int]::MaxValue) { $null } else { $minFree }
        $row.procRamPeakMB = $peakProcRam
        $row.procPrivatePeakMB = $peakProcPriv
        $row.procVramPeakMB = if ($peakProcVram -gt 0) { $peakProcVram } else { $null }
        $row.sysRamFreeMinMB = if ($sysRamMin -eq [int]::MaxValue) { $null } else { $sysRamMin }
        $row.sysRamUsedPeakMB = [math]::Round($sysRamUsedPeak, 0)
        $row.cpuPctPeak = if ($cpuPeak -gt 0) { $cpuPeak } else { $null }
        $row.gpuWattsPeak = if ($wPeak -gt 0) { [math]::Round($wPeak, 1) } else { $null }
        $row.gpuUtilPeakPct = if ($utilPeak -gt 0) { [math]::Round($utilPeak, 0) } else { $null }

        if ($row.status -eq 'OK') { break }

        $errTxt = if (Test-Path $err) { Get-Content $err -Raw } else { '' }
        $isFlake = $errTxt -match 'ZeroDivisionError'
        if ($isFlake -and $attempt -lt $maxAttempts) {
            Write-Host ("    known perf.py timer flake (attempt {0}/{1}) -- relaunching" -f `
                        $attempt, $maxAttempts) -ForegroundColor Yellow
            Start-Sleep -Seconds 5
            continue
        }
        if ($isFlake) { $row.note = 'ZeroDivisionError flake persisted through all retries' }
        break
    }

    $v1 = Get-KBVram
    if ($v1) { $row.vramAfter = $v1.FreeMB }
    $row.ramAfter = Get-KBRamMB

    Write-Host ("    -> {0} {1}s peakUsed={2}MB minFree={3}MB ramAfter={4}MB attempts={5}" -f
                $row.status, $row.elapsedSec, $row.vramPeak, $row.vramMinFree, $row.ramAfter, $row.attempts)
    Write-Host ("       ours: ramWS={0}MB priv={1}MB vram={2}" -f `
                $row.procRamPeakMB, $row.procPrivatePeakMB, `
                $(if ($row.procVramPeakMB) { "$($row.procVramPeakMB)MB" } else { 'n/a(WDDM)' }))
    Write-Host ("       sys: ramFreeMin={0}MB usedPeak={1}MB | cpu={2}% gpu={3}W util={4}%" -f `
                $row.sysRamFreeMinMB, $row.sysRamUsedPeakMB, `
                $(if ($row.cpuPctPeak) { $row.cpuPctPeak } else { 'n/a' }), `
                $(if ($row.gpuWattsPeak) { $row.gpuWattsPeak } else { 'n/a' }), `
                $(if ($row.gpuUtilPeakPct) { $row.gpuUtilPeakPct } else { 'n/a' }))

    Write-KBRow $row $Jsonl
    return [pscustomobject]$row
}

function Write-KBRow($row, [string]$Jsonl) {
    if (-not $Jsonl) { return }
    $line = $row | ConvertTo-Json -Compress -Depth 6
    Add-Content -Path $Jsonl -Value $line -Encoding utf8
}

# --------------------------------------------------------------------- parsers

# perf.py prints a ladder, not one number:
#   "Length   4096:  1234.56 tokens/s"
#   "Context     0: S=1   26.50 tokens/s [26.10 - 26.90],   S=2  14.37 it/s [...]"
# Two traps: the unit flips to "it/s" when -sd is on (spec-decode seqlens), and
# one Context line carries EVERY seqlen, so each line needs a match-all loop.
# Returns @{ Prefill = @{len = tps}; Gen = @{ctx = @{seqlen = {Tps,Min,Max}}} }
function ConvertFrom-KBPerf([string]$path) {
    $txt = Get-KBPlain $path
    $pre = @{}; $gen = @{}
    $seqlenRe = 'S=(\d+)\s+([\d.]+)\s+(?:tokens|it)/s(?:\s*\[([\d.]+)\s*-\s*([\d.]+)\])?'
    foreach ($line in ($txt -split "`n")) {
        if ($line -match '^\s*Length\s+(\d+):\s+([\d.]+)\s+tokens/s') {
            $pre[[int]$Matches[1]] = [double]$Matches[2]
        }
        elseif ($line -match '^\s*Context\s+(\d+):') {
            $ctx = [int]$Matches[1]
            if (-not $gen.ContainsKey($ctx)) { $gen[$ctx] = @{} }
            foreach ($mm in [regex]::Matches($line, $seqlenRe)) {
                $e = @{ Tps = [double]$mm.Groups[2].Value; Min = $null; Max = $null }
                if ($mm.Groups[3].Success) {
                    $e.Min = [double]$mm.Groups[3].Value
                    $e.Max = [double]$mm.Groups[4].Value
                }
                $gen[$ctx][[int]$mm.Groups[1].Value] = $e
            }
        }
    }
    return @{ Prefill = $pre; Gen = $gen }
}

# spec_decode.py prints a markdown table, arm order taken from the header:
#   | Category      |       MTP (greedy) |   MTP (temp 1) |
#   | Agentic, code | 36.57 t/s, 3.63/4.80 acc/draft | ... |
# Returns @{ '<cat>' = @{ '<arm>' = @{Tps;Acc;Draft} } }
function ConvertFrom-KBSpec([string]$path) {
    $txt = Get-KBPlain $path
    $arms = @(); $out = @{}
    $inTable = $false
    foreach ($line in ($txt -split "`n")) {
        $t = $line.Trim()
        if ($t -notmatch '^\|') { continue }
        $cells = ($t.Trim('|') -split '\|') | ForEach-Object { $_.Trim() }
        if ($cells[0] -match '^:?-{2,}') { continue }          # separator row
        if (-not $inTable) {
            if ($cells[0] -eq 'Category') { $arms = $cells[1..($cells.Count - 1)]; $inTable = $true }
            continue
        }
        $cat = $cells[0]
        if (-not $out.ContainsKey($cat)) { $out[$cat] = @{} }
        for ($i = 0; $i -lt $arms.Count -and ($i + 1) -lt $cells.Count; $i++) {
            $c = $cells[$i + 1]
            $e = @{ Tps = $null; Acc = $null; Draft = $null }
            if ($c -match '([\d.]+)\s*t/s') { $e.Tps = [double]$Matches[1] }
            if ($c -match '([\d.]+)/([\d.]+)\s*acc/draft') {
                $e.Acc = [double]$Matches[1]; $e.Draft = [double]$Matches[2]
            }
            if ($null -ne $e.Tps) { $out[$cat][$arms[$i]] = $e }
        }
    }
    return $out
}

# Short one-line summary of a perf.py log, for the eye during long batteries.
function Get-KBPerfSummary($parsed) {
    $tg = $null
    if ($parsed.Gen.ContainsKey(0) -and $parsed.Gen[0].ContainsKey(1)) { $tg = $parsed.Gen[0][1] }
    $ppShort = $parsed.Prefill[256]
    $ppChunk = $parsed.Prefill[4096]
    "tg0={0} pp256={1} pp4096={2}" -f
        $(if ($tg) { "{0:N2}{1}" -f $tg.Tps, $(if ($tg.Min) { " [{0:N2}-{1:N2}]" -f $tg.Min, $tg.Max } else { '' }) } else { 'NA' }),
        $(if ($ppShort) { "{0:N2}" -f $ppShort } else { 'NA' }),
        $(if ($ppChunk) { "{0:N2}" -f $ppChunk } else { 'NA' })
}