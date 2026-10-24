param([string]$Name, [int]$Tries = 8, [int]$SleepSec = 120)
$LOG = "C:\Users\yoho\Downloads\tabbyAPI\_agent\bench_$Name.log"
for ($i = 0; $i -lt $Tries; $i++) {
    Start-Sleep -Seconds $SleepSec
    if (Test-Path $LOG) {
        $tail = Get-Content $LOG | Select-Object -Last 4
        if ($tail -match "BENCH_EXIT") { $tail; return }
    }
}
Write-Output "STILL RUNNING after $($Tries * $SleepSec)s:"
Get-Content $LOG | Select-Object -Last 3
