# _kb_phase_a.ps1 -- knob battery, PHASE A: engine knobs via eval/perf.py.
#
# Server must be STOPPED (bench.ps1:2 wants the full 24 GB VRAM). Knobs arrive
# as CLI flags / env, NOT config.yml, so nothing in this file edits production.
#
# Tier flag sets (see plan §0.5):
#   0  -spf -max_length 1024 -dr 1              decode only, one short ladder
#   1  -short -max_length 4096 -dr 2            + short-prefill sweep, 2 reps
#   2  -short -sd -max_length 32768 -dr 3       full bench.ps1 BASE, 3 reps
#
# Reps run INSIDE one process (perf.py -dr) and print mean + min-max. That
# replaces the plan's cross-process median on purpose: §A records +-25%
# process-to-process spread from fragmentation, which would swamp a >5% effect.
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File .\_kb_phase_a.ps1 -Tier 0

param(
    [int]$Tier = 0,
    [string]$Only = '',      # comma list of arm names
    [string]$Jsonl = '',
    # Pin the DSA prefill path (EXL3_DSA_QC_STAGE=0). Determinism over
    # wall-clock; see plan §0.8. Use for A/B runs, not for speed screens.
    [switch]$Deterministic
)

. "$PSScriptRoot\_kb_lib.ps1"

$Jsonl = if ($Jsonl) { $Jsonl } else { Join-Path $KB.LogRoot 'phaseA.jsonl' }
if (-not (Test-Path $KB.LogRoot)) { New-Item -ItemType Directory -Force $KB.LogRoot | Out-Null }

$tierFlags = @{
    0 = @('-spf', '-max_length', '1024', '-dr', '1')
    1 = @('-short', '-max_length', '4096', '-dr', '2')
    2 = @('-short', '-sd', '-max_length', '32768', '-dr', '3')
}
if (-not $tierFlags.ContainsKey($Tier)) { throw "Tier must be 0, 1 or 2 (got $Tier)" }

# ---------------------------------------------------------------------- arms
# Flags is a hashtable of flag -> value that REPLACES the matching base pair.
# Drop lists base flags to remove entirely (mcs* arms: -mcl must be gone, the
# two modes are mutually exclusive).
$arms = @(
    @{ n = 'a00-baseline'; knob = 'baseline';    flag = $null; env = $null; drop = @() }
    # 1 cpu_moe_offload_layers 38 -> 36 / 40 / 42. Fewer offloaded = faster
    # until the VRAM guard bites; interacts with everything, so it goes first.
    @{ n = 'a01-mcl36'; knob = 'cpu_moe_offload_layers'; flag = @{ '-mcl' = '36' }; env = $null; drop = @() }
    @{ n = 'a02-mcl40'; knob = 'cpu_moe_offload_layers'; flag = @{ '-mcl' = '40' }; env = $null; drop = @() }
    @{ n = 'a03-mcl42'; knob = 'cpu_moe_offload_layers'; flag = @{ '-mcl' = '42' }; env = $null; drop = @() }
    # Tier 0 showed 36 = +9.7% vs 38 and 40/42 clearly worse, so the ladder's
    # live direction is DOWNWARD. The plan listed only 36/40/42; the ladder rule
    # says follow the trend, so extend until the VRAM guard bites.
    @{ n = 'a24-mcl34'; knob = 'cpu_moe_offload_layers'; flag = @{ '-mcl' = '34' }; env = $null; drop = @() }
    @{ n = 'a25-mcl32'; knob = 'cpu_moe_offload_layers'; flag = @{ '-mcl' = '32' }; env = $null; drop = @() }
    # 2 cache_mode 5,4 -> 2,2 / 4,4 / 8,8. PERF ONLY -- KLD owns quality.
    @{ n = 'a04-cq22'; knob = 'cache_mode'; flag = @{ '-cq' = '2,2' }; env = $null; drop = @() }
    @{ n = 'a05-cq44'; knob = 'cache_mode'; flag = @{ '-cq' = '4,4' }; env = $null; drop = @() }
    @{ n = 'a06-cq88'; knob = 'cache_mode'; flag = @{ '-cq' = '8,8' }; env = $null; drop = @() }
    # 3 chunk_size 4096 -> 2048 / 8192. 8192 needs guard headroom or it is UNSAFE.
    @{ n = 'a07-chunk2048'; knob = 'chunk_size'; flag = @{ '-chunk_size' = '2048' }; env = $null; drop = @() }
    @{ n = 'a08-chunk8192'; knob = 'chunk_size'; flag = @{ '-chunk_size' = '8192' }; env = $null; drop = @() }
    # 4 max_batch_size 2 -> 1. Halves serving capacity; report both numbers.
    @{ n = 'a09-ambs1'; knob = 'max_batch_size'; flag = @{ '-ambs' = '1' }; env = $null; drop = @() }
    # 6 env threads, low first: experience says low values score higher on LLM
    # work and saturation rarely pays on this CPU.
    @{ n = 'a10-thr1'; knob = 'EXL3_MOE_CPU_THREADS'; flag = $null; env = @{ EXL3_MOE_CPU_THREADS = '1' }; drop = @() }
    @{ n = 'a11-thr2'; knob = 'EXL3_MOE_CPU_THREADS'; flag = $null; env = @{ EXL3_MOE_CPU_THREADS = '2' }; drop = @() }
    @{ n = 'a12-thr4'; knob = 'EXL3_MOE_CPU_THREADS'; flag = $null; env = @{ EXL3_MOE_CPU_THREADS = '4' }; drop = @() }
    @{ n = 'a13-thr12'; knob = 'EXL3_MOE_CPU_THREADS'; flag = $null; env = @{ EXL3_MOE_CPU_THREADS = '12' }; drop = @() }
    # Tier 0 resolved the thread trend and it points UP, not down: 1/2/4/8 =
    # 7.07/11.38/20.25/26.80 tok/s, monotone. That is the OPPOSITE of the plan's
    # stated prior ("low values score higher"), so the plan's own rule fires --
    # "12 / 16 only as follow-up if low values don't resolve a trend" -- and the
    # follow-up goes upward. thr12 read +13.2%; test 16 and 24 (24 is the core
    # count, the point past which oversubscription should finally bite).
    @{ n = 'a26-thr16'; knob = 'EXL3_MOE_CPU_THREADS'; flag = $null; env = @{ EXL3_MOE_CPU_THREADS = '16' }; drop = @() }
    @{ n = 'a27-thr24'; knob = 'EXL3_MOE_CPU_THREADS'; flag = $null; env = @{ EXL3_MOE_CPU_THREADS = '24' }; drop = @() }
    # 6 env streams
    @{ n = 'a14-st3'; knob = 'EXL3_MOE_STREAM_T'; flag = $null; env = @{ EXL3_MOE_STREAM_T = '3' }; drop = @() }
    @{ n = 'a15-st12'; knob = 'EXL3_MOE_STREAM_T'; flag = $null; env = @{ EXL3_MOE_STREAM_T = '12' }; drop = @() }
    @{ n = 'a16-be24'; knob = 'EXL3_MOE_STREAM_BATCH_EXPERTS'; flag = $null; env = @{ EXL3_MOE_STREAM_BATCH_EXPERTS = '24' }; drop = @() }
    # 7 ablations, one at a time
    @{ n = 'a17-pin0'; knob = 'EXL3_MOE_CPU_PIN'; flag = $null; env = @{ EXL3_MOE_CPU_PIN = '0' }; drop = @() }
    @{ n = 'a18-swz0'; knob = 'EXL3_MOE_CPU_SWIZZLE'; flag = $null; env = @{ EXL3_MOE_CPU_SWIZZLE = '0' }; drop = @() }
    @{ n = 'a19-zc0'; knob = 'EXL3_MOE_ZERO_COPY'; flag = $null; env = @{ EXL3_MOE_ZERO_COPY = '0' }; drop = @() }
    # 8 split-experts + threads COMBINED. -mcs N offloads the TAIL N experts of
    # EVERY layer. Model has 48 layers x 512 experts, so mcl=38 offloads
    # 38*512=19456 experts; spread over 48 layers that is N~405 for the same
    # parameter volume, i.e. the VRAM-matched comparison. -mcl MUST be dropped.
    @{ n = 'a20-mcs405'; knob = 'cpu_moe_split_experts'; flag = @{ '-mcs' = '405' }; env = $null; drop = @('-mcl') }
    @{ n = 'a21-mcs300'; knob = 'cpu_moe_split_experts'; flag = @{ '-mcs' = '300' }; env = $null; drop = @('-mcl') }
    @{ n = 'a22-mcs500'; knob = 'cpu_moe_split_experts'; flag = @{ '-mcs' = '500' }; env = $null; drop = @('-mcl') }
    @{ n = 'a23-mcs405-thr2'; knob = 'cpu_moe_split_experts+threads'; flag = @{ '-mcs' = '405' }; env = @{ EXL3_MOE_CPU_THREADS = '2' }; drop = @('-mcl') }
    # Second pass INSIDE the window. First pass bracketed it: mcs300 was
    # VRAM-refused and mcs500 host-RAM-refused, leaving only 405 (which is
    # VRAM-matched to mcl38: 405/512 = 79.1% vs 38/48 = 79.2% CPU-side).
    # The two limits pull opposite ways -- lower N keeps MORE experts on the GPU
    # (VRAM ceiling) and FEWER in the host arena (RAM relief) -- so the band is
    # narrow and worth bisecting rather than declaring the knob dead at 405.
    # Operator reports ~360 working, so start there and bracket both ways.
    @{ n = 'a28-mcs360'; knob = 'cpu_moe_split_experts'; flag = @{ '-mcs' = '360' }; env = $null; drop = @('-mcl') }
    @{ n = 'a29-mcs380'; knob = 'cpu_moe_split_experts'; flag = @{ '-mcs' = '380' }; env = $null; drop = @('-mcl') }
    @{ n = 'a30-mcs340'; knob = 'cpu_moe_split_experts'; flag = @{ '-mcs' = '340' }; env = $null; drop = @('-mcl') }
    @{ n = 'a31-mcs390'; knob = 'cpu_moe_split_experts'; flag = @{ '-mcs' = '390' }; env = $null; drop = @('-mcl') }
    # Plan section 2: the two Tier 1 winners combined once. 380 rather than 360
    # because the 0.29 tg/s between them is inside noise while the extra 1.4 GB
    # of VRAM headroom is not.
    @{ n = 'a32-mcs380-thr16'; knob = 'cpu_moe_split_experts x threads'; flag = @{ '-mcs' = '380' }; env = @{ EXL3_MOE_CPU_THREADS = '16' }; drop = @('-mcl') }
)

if ($Only) {
    $want = $Only -split ',' | ForEach-Object { $_.Trim() }
    # @() matters: Where-Object returning a single object unrolls it to a scalar
    # hashtable, and .Count on a hashtable is its KEY count (5), not 1.
    $arms = @($arms | Where-Object { $want -contains $_.n })
}

# --------------------------------------------------------------- flag assembly
function Get-KBArgs($arm) {
    $flags = [System.Collections.ArrayList]@($KBBaseFlags)
    foreach ($d in $arm.drop) {
        for ($i = 0; $i -lt $flags.Count; $i++) {
            if ($flags[$i] -eq $d) {
                $flags.RemoveAt($i)                      # the flag NAME
                # Removing the name shifts its VALUE into slot $i, and leaving
                # that behind produces "unrecognized arguments: 38". Drop it.
                if ($i -lt $flags.Count) { $flags.RemoveAt($i) }
                $i--
            }
        }
    }
    if ($arm.flag) {
        foreach ($k in ($arm.flag.Keys | Sort-Object)) {
            # replace an existing pair if the base already carries this flag
            $found = $false
            for ($i = 0; $i -lt $flags.Count; $i++) {
                if ($flags[$i] -eq $k) { $flags[$i + 1] = $arm.flag[$k]; $found = $true; break }
            }
            if (-not $found) { [void]$flags.Add($k); [void]$flags.Add($arm.flag[$k]) }
        }
    }
    # Assert strict name/value alternation on the BASE portion only -- an
    # orphaned value is a silent A/B corruption that would otherwise surface
    # late as an argparse error. Tier flags are excluded because they include
    # store-true flags (-spf, -short, -sd) that legitimately take no value.
    if ($flags.Count % 2 -ne 0) { throw ("arm {0}: odd base flag count {1}" -f $arm.n, $flags.Count) }
    for ($i = 0; $i -lt $flags.Count; $i += 2) {
        if ($flags[$i] -notlike '-*') { throw ("arm {0}: '{1}' at slot {2} is not a flag -> {3}" -f $arm.n, $flags[$i], $i, (@($flags) -join ' ')) }
    }
    return (@($flags) + $tierFlags[$Tier])
}

Write-Host ("==== PHASE A tier {0} : {1} arms, flags [{2}] ====" -f
            $Tier, $arms.Count, ($tierFlags[$Tier] -join ' ')) -ForegroundColor Cyan
Write-Host ("date={0}  venv exllamav3 from wheel; perf.py from {1}" -f (Get-KBStamp), $KB.Perf)
$v0 = Get-KBVram
Write-Host ("pre-battery: vram free={0}MB  ram free={1}MB" -f $v0.FreeMB, (Get-KBRamMB))
Write-Host ("guards: vram kill <{0}MB free, ram kill <{1}MB, ram abort pre <{2}MB`n" -f
            $KBGuard.VramFreeKillMB, $KBGuard.RamFreeKillMB, $KBGuard.RamFreeMinBeforeMB)

$results = @()
foreach ($arm in $arms) {
    Write-Host ("`n--- {0}  [{1}]" -f $arm.n, $arm.knob) -ForegroundColor Yellow
    $args = Get-KBArgs $arm
    $row = Invoke-KBRun -Name $arm.n -Script $KB.Perf -PyArgs $args -EnvSet $arm.env -Deterministic:$Deterministic -Jsonl $Jsonl
    $row | Add-Member -NotePropertyName knob -NotePropertyValue $arm.knob
    $row | Add-Member -NotePropertyName tier -NotePropertyValue $Tier
    if ($row.status -eq 'OK') {
        $p = ConvertFrom-KBPerf $row.out
        Write-Host ("    {0}" -f (Get-KBPerfSummary $p))
        # -NotePropertyName/-NotePropertyValue explicitly: "-NoteProperty X Y"
        # is ambiguous in PS 5.1 (it wants a hashtable).
        $row | Add-Member -NotePropertyName parsed -NotePropertyValue (Get-KBPerfSummary $p)
        if ($p.Prefill.ContainsKey(256)) {
            $row | Add-Member -NotePropertyName pp256 -NotePropertyValue $p.Prefill[256]
        }
        if ($p.Prefill.ContainsKey(4096)) {
            $row | Add-Member -NotePropertyName pp4096 -NotePropertyValue $p.Prefill[4096]
        }
        if ($p.Gen.ContainsKey(0) -and $p.Gen[0].ContainsKey(1)) {
            $g = $p.Gen[0][1]
            $row | Add-Member -NotePropertyName tg0 -NotePropertyValue $g.Tps
            if ($null -ne $g.Min) {
                $row | Add-Member -NotePropertyName tg0min -NotePropertyValue $g.Min
                $row | Add-Member -NotePropertyName tg0max -NotePropertyValue $g.Max
            }
        }
    } else {
        Write-Host "    no parse: $($row.status) $($row.guard)" -ForegroundColor Red
    }
    # Re-append: Invoke-KBRun wrote the jsonl row BEFORE the parsed metrics were
    # attached above, so without this the jsonl carries no tg0/pp at all. The
    # report's dedupe keeps the last row per arm, so this append supersedes.
    Write-KBRow $row $Jsonl
    $results += $row
}

# ------------------------------------------------------------------- summary
Write-Host ("`n==== PHASE A tier {0} summary ====" -f $Tier) -ForegroundColor Cyan
Write-Host ("{0,-20} {1,-9} {2,10} {3,10} {4,10} {5,10} {6,10} {7,9}" -f `
            'arm', 'status', 'tg0', 'd%', 'pp256', 'pp4096', 'vramPk', 'ramAft')
$base = $results | Where-Object { $_.name -eq 'a00-baseline' } | Select-Object -First 1
$btg = if ($base -and $base.PSObject.Properties['tg0']) { $base.tg0 } else { $null }
$bpp = if ($base -and $base.PSObject.Properties['pp4096']) { $base.pp4096 } else { $null }
foreach ($r in $results) {
    $tg = if ($r.PSObject.Properties['tg0']) { $r.tg0 } else { $null }
    $pp = if ($r.PSObject.Properties['pp4096']) { $r.pp4096 } else { $null }
    $d = if ($btg -and $tg) { '{0:+0.0;-0.0;0.0}' -f (100 * ($tg / $btg - 1)) } else { '  NA' }
    $dp = if ($bpp -and $pp) { '{0:+0.0;-0.0;0.0}' -f (100 * ($pp / $bpp - 1)) } else { '  NA' }
    Write-Host ("{0,-20} {1,-9} {2,10} {3,10} {4,10} {5,10} {6,10} {7,9}" -f `
                $r.name, $r.status,
                $(if ($tg) { '{0:N2}' -f $tg } else { '-' }),
                $d,
                $(if ($r.PSObject.Properties['pp256'] -and $r.pp256) { '{0:N1}' -f $r.pp256 } else { '-' }),
                $(if ($pp) { '{0:N1}' -f $pp } else { '-' }),
                $r.vramPeak,
                $r.ramAfter)
    if ($r.status -ne 'OK') { Write-Host ("    guard={0}" -f $r.guard) -ForegroundColor Red }
}
Write-Host ("`nrows -> {0}" -f $Jsonl)