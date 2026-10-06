# _kb_phase_b.ps1 -- knob battery, PHASE B: draft knobs via eval/spec_decode.py.
#
# Server must be STOPPED. Knobs arrive as CLI flags (-ndt / -dds), NOT
# config.yml, so nothing here edits production. draft_cache_mode is NOT in this
# phase: model_init.py builds draft_cache with no CLI flag (plan §0.1), so it
# is Phase C only.
#
# Tier flag sets (plan §0.5). -single filters by CATEGORY, not by file
# (spec_decode.py:194-200), so tiers pick categories, not file counts:
#   0  -single "Coding"                 3 small files, real prompts, ~fast
#   1  -single "Creative (reasoning)"   3 files, THINKING ON, medium
#   2  -single "agentic, code" + -temp  5 files, THINKING ON, both arms
# Every tier uses real prompts: a category that EOSes after a handful of tokens
# would produce a meaningless t/s and cannot detect a 10% loser.
#
# Draft arms pass -nbl so each run measures ONLY its own draft arm; the two
# reference points are b00 (production) and b06 (draft off). Passing -nbl with
# no -dm would measure nothing at all.
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File .\_kb_phase_b.ps1 -Tier 0

param(
    [int]$Tier = 0,
    [string]$Only = '',
    [string]$Jsonl = '',
    # Print the argv each arm would run and exit. Phase B had never been run,
    # and an argument mistake costs a wasted model load to discover -- same
    # class of bug that made the first -mcs batch emit "unrecognized arguments".
    [switch]$DryRun,
    [switch]$Deterministic
)

. "$PSScriptRoot\_kb_lib.ps1"

$Jsonl = if ($Jsonl) { $Jsonl } else { Join-Path $KB.LogRoot 'phaseB.jsonl' }
if (-not (Test-Path $KB.LogRoot)) { New-Item -ItemType Directory -Force $KB.LogRoot | Out-Null }

$tierFlags = @{
    0 = @('-tokens', '256', '-single', 'Coding')
    1 = @('-tokens', '256', '-single', 'Creative (reasoning)')
    2 = @('-tokens', '256', '-single', 'agentic, code', '-temp')
}
if (-not $tierFlags.ContainsKey($Tier)) { throw "Tier must be 0, 1 or 2 (got $Tier)" }

# ---------------------------------------------------------------------- arms
# MTP auto-engages when -dm equals -m (mtp_sweep.ps1:2). -dds == dynamic_draft.
$arms = @(
    @{ n = 'b00-baseline'; knob = 'baseline'; extra = @('-ndt', '5', '-dds') }
    @{ n = 'b01-ndt4-dyn'; knob = 'draft_num_tokens'; extra = @('-ndt', '4', '-dds', '-nbl') }
    @{ n = 'b02-ndt3-dyn'; knob = 'draft_num_tokens'; extra = @('-ndt', '3', '-dds', '-nbl') }
    # Ceiling probe. The plan believed MTP max was 4 and said to confirm from the
    # log rather than assume; the 1.5.4 logs already show ndt 5 works
    # (logs/mtp-154/r06-ndt5.log, acc/draft 5.00), so this arm looks for the real
    # ceiling instead. Tier 0 only -- if 6 is rejected the run fails loudly.
    @{ n = 'b03-ndt6-probe'; knob = 'draft_num_tokens (ceiling probe)'; extra = @('-ndt', '6', '-dds', '-nbl') }
    @{ n = 'b04-ndt5-static'; knob = 'dynamic_draft'; extra = @('-ndt', '5', '-nbl') }
    @{ n = 'b05-ndt3-static'; knob = 'draft_num_tokens x dynamic'; extra = @('-ndt', '3', '-nbl') }
    # draft off: no -dm at all, so the Baseline arm is what gets measured.
    @{ n = 'b06-draft-off'; knob = 'draft off'; extra = @() }
    # Ceiling sweep above 6. ndt6 beat ndt5 on BOTH tg and acceptance, which is
    # the signature of a knob still on its rising side, so the optimum may be
    # above 6. Operator states drafting does not affect quality, so acceptance
    # only constrains SPEED -- which means the sweep should go up, not down.
    @{ n = 'b07-ndt7-dyn'; knob = 'draft_num_tokens (ceiling probe)'; extra = @('-ndt', '7', '-dds', '-nbl') }
    @{ n = 'b08-ndt8-dyn'; knob = 'draft_num_tokens (ceiling probe)'; extra = @('-ndt', '8', '-dds', '-nbl') }
    @{ n = 'b09-ndt10-dyn'; knob = 'draft_num_tokens (ceiling probe)'; extra = @('-ndt', '10', '-dds', '-nbl') }
)

if ($Only) {
    $want = $Only -split ',' | ForEach-Object { $_.Trim() }
    $arms = @($arms | Where-Object { $want -contains $_.n })
}

# Mirror production before the draft flags. mtp_sweep.ps1:11-13 leads with
# -mcl/-cq/-cs/-chunk_size for exactly this reason: spec_decode.py reads NOTHING
# from config.yml, so without them it would try to fit all 52.5 GB on the GPU and
# die on the VRAM check. Caught by -DryRun before spending a model load.
function Get-KBArgs($arm) {
    $args = @($KBBaseFlags)
    # -dm == -m is what makes MTP auto-engage. b06-draft-off omits it entirely,
    # which is what makes its measured arm the no-drafting Baseline.
    if ($arm.n -ne 'b06-draft-off') { $args += @('-dm', $KB.Model) }
    $args += $arm.extra
    $args += $tierFlags[$Tier]
    return $args
}

Write-Host ("==== PHASE B tier {0} : {1} arms, flags [{2}] ====" -f
            $Tier, $arms.Count, ($tierFlags[$Tier] -join ' ')) -ForegroundColor Cyan
$v0 = Get-KBVram
Write-Host ("date={0}  pre: vram free={1}MB  ram free={2}MB" -f (Get-KBStamp), $v0.FreeMB, (Get-KBRamMB))

if ($DryRun) {
    Write-Host "`n-- DRY RUN: argv per arm (no model load) --" -ForegroundColor Cyan
    foreach ($arm in $arms) {
        $args = Get-KBArgs $arm
        $q = @($args | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } })
        Write-Host ("  {0,-18}" -f $arm.n)
        Write-Host ("      {0}" -f ($q -join ' ')) -ForegroundColor DarkGray
    }
    Write-Host "`n-- DRY RUN complete --"
    return
}

$results = @()
foreach ($arm in $arms) {
    Write-Host ("`n--- {0}  [{1}]" -f $arm.n, $arm.knob) -ForegroundColor Yellow
    # -dm == -m is what makes MTP auto-engage. b06-draft-off omits it entirely,
    # which is what makes its measured arm the no-drafting Baseline.
    $args = Get-KBArgs $arm

    $row = Invoke-KBRun -Name $arm.n -Script $KB.Spec -PyArgs $args -EnvSet $null -Deterministic:$Deterministic -Jsonl $Jsonl
    $row | Add-Member -NotePropertyName knob -NotePropertyValue $arm.knob
    $row | Add-Member -NotePropertyName tier -NotePropertyValue $Tier
    if ($row.status -eq 'OK') {
        $s = ConvertFrom-KBSpec $row.out
        $armsFound = @()
        foreach ($cat in $s.Keys) {
            foreach ($armName in $s[$cat].Keys) {
                $e = $s[$cat][$armName]
                $armsFound += ('{0}/{1}={2:N2}t/s{3}' -f $cat, $armName, $e.Tps,
                    $(if ($null -ne $e.Acc) { ' {0:N2}/{1:N2}acc' -f $e.Acc, $e.Draft } else { '' }))
                $row | Add-Member -NotePropertyName ("tps_" + $armName) -NotePropertyValue $e.Tps
                if ($null -ne $e.Acc) {
                    $row | Add-Member -NotePropertyName ("acc_" + $armName) -NotePropertyValue $e.Acc
                    $row | Add-Member -NotePropertyName ("draft_" + $armName) -NotePropertyValue $e.Draft
                }
            }
        }
        Write-Host ("    {0}" -f ($armsFound -join '  |  '))
    } else {
        Write-Host "    no parse: $($row.status) $($row.guard)" -ForegroundColor Red
    }
    # Re-append so the jsonl carries the parsed t/s + acc; see the same note in
    # _kb_phase_a.ps1. The report dedupes on arm name.
    Write-KBRow $row $Jsonl
    $results += $row
}

# ------------------------------------------------------------------- summary
Write-Host ("`n==== PHASE B tier {0} summary ====" -f $Tier) -ForegroundColor Cyan
Write-Host ("{0,-20} {1,-9} {2,10} {3,8} {4,10} {5,10} {6,9}" -f `
            'arm', 'status', 'tps', 'd%', 'acc', 'draft', 'vramPk')
$base = $results | Where-Object { $_.name -eq 'b00-baseline' } | Select-Object -First 1
$btps = $null
if ($base) {
    $bp = $base.PSObject.Properties | Where-Object { $_.Name -like 'tps_*' } | Select-Object -First 1
    if ($bp) { $btps = $bp.Value }
}
foreach ($r in $results) {
    $tp = $r.PSObject.Properties | Where-Object { $_.Name -like 'tps_*' } | Select-Object -First 1
    $tps = if ($tp) { $tp.Value } else { $null }
    $d = if ($btps -and $tps) { '{0:+0.0;-0.0;0.0}' -f (100 * ($tps / $btps - 1)) } else { '  NA' }
    $ap = $r.PSObject.Properties | Where-Object { $_.Name -like 'acc_*' } | Select-Object -First 1
    $dp = $r.PSObject.Properties | Where-Object { $_.Name -like 'draft_*' } | Select-Object -First 1
    Write-Host ("{0,-20} {1,-9} {2,10} {3,8} {4,10} {5,10} {6,9}" -f `
                $r.name, $r.status,
                $(if ($tps) { '{0:N2}' -f $tps } else { '-' }), $d,
                $(if ($ap) { '{0:N2}' -f $ap.Value } else { '-' }),
                $(if ($dp) { '{0:N2}' -f $dp.Value } else { '-' }),
                $r.vramPeak)
    if ($r.status -ne 'OK') { Write-Host ("    guard={0}" -f $r.guard) -ForegroundColor Red }
}
Write-Host ("`nrows -> {0}" -f $Jsonl)