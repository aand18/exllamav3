# _kb_selftest.ps1 -- validate the _kb_lib parsers against ALREADY-CAPTURED logs.
# Costs no box time: every input here was measured in a previous session, so a
# parser regression shows up before the battery burns hours.
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File .\_kb_selftest.ps1

. "$PSScriptRoot\_kb_lib.ps1"

$fail = 0
function Check([string]$what, $got, $expect) {
    $ok = "$got" -eq "$expect"
    if (-not $ok) { $script:fail++ }
    $tag = if ($ok) { 'ok  ' } else { 'FAIL' }
    Write-Host ("{0} {1,-46} got={2} expect={3}" -f $tag, $what, $got, $expect) -ForegroundColor $(if ($ok) { 'Green' } else { 'Red' })
}

Write-Host "`n--- env baseline ---" -ForegroundColor Cyan
Set-KBEnv $null
$sig = Get-KBEnvSig
Check 'threads' $env:EXL3_MOE_CPU_THREADS '8'
Check 'memops'  $env:EXL3_MOE_MEMOPS '0'
Check 'zerocopy' $env:EXL3_MOE_ZERO_COPY '1'
Check 'stream_t' $env:EXL3_MOE_STREAM_T '6'
Check 'batch_experts' $env:EXL3_MOE_STREAM_BATCH_EXPERTS '48'
# 7 EXL3_MOE_* baseline keys + 1 determinism key (EXL3_DSA_QC_STAGE), which
# Get-KBEnvSig reports whether or not determinism mode is on, so a
# deterministic row is distinguishable from a default one.
Check 'sig key count' (($sig -split ' ').Count) '8'

Write-Host "`n--- env override + clear round-trip ---" -ForegroundColor Cyan
Set-KBEnv @{ EXL3_MOE_CPU_THREADS = '2'; EXL3_MOE_ZERO_COPY = $null }
Check 'threads overridden' $env:EXL3_MOE_CPU_THREADS '2'
Check 'zerocopy removed'   ([string]$env:EXL3_MOE_ZERO_COPY) ''
Check 'pin still base'     $env:EXL3_MOE_CPU_PIN '1'
Set-KBEnv $null
Check 'threads restored'   $env:EXL3_MOE_CPU_THREADS '8'

Write-Host "`n--- live sensors ---" -ForegroundColor Cyan
$v = Get-KBVram
$r = Get-KBRamMB
Write-Host ("  vram free={0}MB used={1}MB ; ram free={2}MB" -f $v.FreeMB, $v.UsedMB, $r)
if ($null -eq $v -or $v.FreeMB -le 0) { $fail++; Write-Host 'FAIL vram probe' -ForegroundColor Red }
if ($null -eq $r -or $r -le 0) { $fail++; Write-Host 'FAIL ram probe' -ForegroundColor Red }

Write-Host "`n--- perf.py parser vs logs/perf-154/03-tg-repeat.log ---" -ForegroundColor Cyan
$log = 'C:\Users\yoho\Downloads\tabbyAPI\logs\perf-154\03-tg-repeat.log'
if (Test-Path $log) {
    $p = ConvertFrom-KBPerf $log
    Write-Host "  prefill lengths: $(($p.Prefill.Keys | Sort-Object) -join ',')"
    Write-Host "  gen contexts:     $(($p.Gen.Keys | Sort-Object) -join ',')"
    Write-Host "  summary: $(Get-KBPerfSummary $p)"
    # -spf -sd -dr 4 => decode only, seqlens 1..4, 4 reps (min-max printed)
    if ($p.Gen.Count -eq 0) { $fail++; Write-Host 'FAIL no gen rows parsed' -ForegroundColor Red }
    if (-not ($p.Gen[0].ContainsKey(1))) { $fail++; Write-Host 'FAIL no S=1 at ctx 0' -ForegroundColor Red }
    if ($null -eq $p.Gen[0][1].Min) { $fail++; Write-Host 'FAIL no min-max spread with -dr 4' -ForegroundColor Red }
    if (-not ($p.Gen[0].ContainsKey(4))) { $fail++; Write-Host 'FAIL missing S=4 (spec-decode seqlen)' -ForegroundColor Red }
} else { $fail++; Write-Host "FAIL missing $log" -ForegroundColor Red }

Write-Host "`n--- perf.py parser vs a -short -sd log (full ladder) ---" -ForegroundColor Cyan
$log2 = 'C:\Users\yoho\Downloads\tabbyAPI\logs\perf-154\01-base.log'
if (Test-Path $log2) {
    $p2 = ConvertFrom-KBPerf $log2
    $has256 = $p2.Prefill.ContainsKey(256)
    $has4096 = $p2.Prefill.ContainsKey(4096)
    Check 'prefill has length 256'  $has256 $true
    Check 'prefill has length 4096' $has4096 $true
    Write-Host "  pp256=$($p2.Prefill[256]) pp4096=$($p2.Prefill[4096])"
    Write-Host "  summary: $(Get-KBPerfSummary $p2)"
} else { $fail++; Write-Host "FAIL missing $log2" -ForegroundColor Red }

Write-Host "`n--- spec_decode.py parser vs logs/mtp-154/r11-ndt5-dyn.log ---" -ForegroundColor Cyan
$s1 = 'C:\Users\yoho\Downloads\tabbyAPI\logs\mtp-154\r11-ndt5-dyn.log'
if (Test-Path $s1) {
    $s = ConvertFrom-KBSpec $s1
    # r11 = -nbl -ndt 5 -dds => only the MTP (greedy) arm
    Write-Host "  categories: $($s.Keys -join ', ')"
    $armNames = $s['Agentic, code'].Keys
    Write-Host "  arms: $($armNames -join ', ')"
    # log tail reads: 36.57 t/s, 3.63/4.80 acc/draft
    Check 'ndt5-dyn tps'   $s['Agentic, code']['MTP (greedy)'].Tps    '36.57'
    Check 'ndt5-dyn acc'   $s['Agentic, code']['MTP (greedy)'].Acc    '3.63'
    # compare numerically: PowerShell stringifies 4.80 as 4.8
    Check 'ndt5-dyn draft' ([math]::Round($s['Agentic, code']['MTP (greedy)'].Draft, 2)) '4.8'
} else { $fail++; Write-Host "FAIL missing $s1" -ForegroundColor Red }

Write-Host "`n--- spec_decode.py parser vs r01 (baseline + MTP arms) ---" -ForegroundColor Cyan
$s2 = 'C:\Users\yoho\Downloads\tabbyAPI\logs\mtp-154\r01-ndt3-dyn.log'
if (Test-Path $s2) {
    $s = ConvertFrom-KBSpec $s2
    $arms = $s['Agentic, code'].Keys
    Write-Host "  arms: $($arms -join ', ')"
    if ($arms.Count -lt 2) { $fail++; Write-Host 'FAIL expected Baseline + MTP arms' -ForegroundColor Red }
    if ($s['Agentic, code'].ContainsKey('Baseline')) {
        Check 'baseline acc is null (no drafting)' ([string]$s['Agentic, code']['Baseline'].Acc) ''
    }
} else { $fail++; Write-Host "FAIL missing $s2" -ForegroundColor Red }

Write-Host "`n=== selftest $(if ($fail -eq 0) { 'PASS' } else { "FAIL ($fail)" }) ===" -ForegroundColor $(if ($fail -eq 0) { 'Green' } else { 'Red' })
exit $fail