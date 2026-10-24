param([string]$Name, [string]$Memops, [string]$Flush, [string]$Submit, [string]$ZeroCopy = "0", [string]$Omp = "", [string]$Eager = "0", [string]$TimerRes = "0", [string]$Prio = "0", [string]$BlockSync = "0", [string]$LargePage = "0", [string]$Pin = "1", [string]$PinOffset = "")
$PY = "C:\Users\yoho\Downloads\tabbyAPI\venv\Scripts\python.exe"
$SD = "C:\Users\yoho\Downloads\exllamav3\eval\spec_decode.py"
$MODEL = "D:\llms\Qwen3.8-Flash-Next-exl3-3.05bpw"
$LOG = "C:\Users\yoho\Downloads\tabbyAPI\_agent\bench_$Name.log"
$env:EXL3_MOE_CPU_THREADS = "8"
if ($Pin -eq "") { $Pin = "1" }
$env:EXL3_MOE_CPU_PIN = $Pin
if ($PinOffset -ne "") { $env:EXL3_MOE_CPU_PIN_OFFSET = $PinOffset }
$env:EXL3_MOE_CPU_SWIZZLE = "1"
$env:EXL3_MOE_STREAM_T = "6"
$env:EXL3_MOE_STREAM_BATCH_EXPERTS = "48"
$env:EXL3_MOE_MEMOPS = $Memops
$env:EXL3_MOE_MEMOPS_FLUSH = $Flush
$env:EXL3_MOE_MEMOPS_SUBMIT = $Submit
$env:EXL3_MOE_MEMOPS_LOG = "1"
$env:EXL3_MOE_HANDOFF_PROF = "1"
$env:EXL3_MOE_ZERO_COPY = $ZeroCopy
if ($Omp -ne "") { $env:OMP_NUM_THREADS = $Omp; $env:MKL_NUM_THREADS = $Omp }
if ($Eager -eq "1") { $env:CUDA_MODULE_LOADING = "EAGER" }
$env:EXL3_MOE_TIMER_RES = $TimerRes
$env:EXL3_MOE_THREAD_PRIO = $Prio
$env:EXL3_MOE_BLOCKING_SYNC = $BlockSync
$env:EXL3_MOE_ARENA_LARGEPAGES = $LargePage
if ($LargePage -eq "1") { $env:EXL3_MOE_ARENA_DEBUG = "1" }
Set-Location C:\Users\yoho\Downloads\tabbyAPI
"VRAM free MiB: $((nvidia-smi.exe --query-gpu=memory.free --format=csv,nounits | Select-Object -Last 1))" | Tee-Object $LOG
"START $Name MEMOPS=$Memops FLUSH=$Flush SUBMIT=$Submit ZC=$ZeroCopy OMP=$Omp EAGER=$Eager TIMER=$TimerRes PRIO=$Prio BLOCKSYNC=$BlockSync LARGEPAGE=$LargePage $(Get-Date -Format o)" | Tee-Object $LOG -Append
& $PY $SD -m $MODEL -dm $MODEL -mcl 36 -mct 8 -cq 5,4 -cs 131072 -chunk_size 4096 -tokens 256 -temp -single "agentic, code" -nbl -ndt 5 -dds 2>&1 | Tee-Object $LOG -Append
"BENCH_EXIT=$LASTEXITCODE END $Name $(Get-Date -Format o)" | Tee-Object $LOG -Append
