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
#   run_pass.ps1 -Tag dry -DryRun
#   run_pass.ps1 -Tag memops1-a -Env "EXL3_MOE_MEMOPS=1" -MaxLength 131072
#   run_pass.ps1 -Tag t14-a -Threads 14 -MaxLength 131072
#   run_pass.ps1 -Tag split320 -Split 320 -MaxLength 262144
#   run_pass.ps1 -Tag static-a -StaticSwap
#   run_pass.ps1 -Tag reps3 -Extra "-dr 3" -MaxLength 131072
#   run_pass.ps1 -Tag x -Unset "EXL3_MOE_MEMOPS,EXL3_MOE_CPU_PIN"
#
# -Env is a comma-separated KEY=VALUE list applied AFTER the regime's own environment, so it can
# override a regime value. -Unset removes variables from the inherited environment. -Threads,
# -Split and -CacheSize replace the regime's -mct, -mcs and -cs. -Extra appends raw perf.py flags,
# but may not restate one the regime owns (-m, -cs, -chunk_size, -ambs, -cq, -max_length, -mcs, -mct,
# and the state's drafting flags -mtp/-ndt/-sdh/-hdt/-dds): the guard is spelling-proof, so
# "-cs=40960" and "--cache_size 40960" abort like "-cs 40960" (a second -cs is how val-spec
# 2026-10-29 came to log one cache size and run another), and "-Extra \"-cq Q8\"" aborts by design.
# Dynamic expert placement is on unless the run is a drafting one or asks for -StaticSwap: the state
# field knob_arms_disable_swap scopes EXL3_MOE_CPU_SWAP=0 to the knob arms, because the archived
# pp/tg ladder was measured with placement on and must stay comparable to itself. -DryRun prints the
# argv and environment a pass would use and exits before anything is written or created - the output
# directories included - and without the one-owner refusal, so a spelling can be checked while the
# card is busy.
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
    [int] $MaxLength = 0,      # 0 = take from state.json
    [int] $Threads = 0,        # 0 = take from state.json
    [int] $Split = 0,          # 0 = take from state.json
    [int] $CacheSize = 0,      # 0 = take from state.json (spec_decode: draft.spec_decode_cache_size)
    [int] $ChunkSize = 0,      # 0 = take from state.json, then 4096: prefill chunk, the pp/VRAM lever
    [string] $Extra = "",
    [string] $Model = "",      # "" = take from state.json
    [string] $State = "",      # "" = <Root>\state.json
    [switch] $NoFrozenState,   # do not point XDG_CACHE_HOME at the frozen profile dir
    [switch] $StaticSwap,      # turn dynamic expert placement off for a pp/tg run (spec_decode: always)
    [string] $Root = "C:\Users\yoho\Downloads\tabbyAPI\_agent\v165",
    [switch] $DryRun,          # print argv+env and exit 0 before any log, CSV row or watcher exists
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

# --- State: the one place a regime lives --------------------------------------------------------
# state.json carried the model/split/threads/cache/env/floors. The parameters are deltas, so an arm
# passes only what it changes and a model swap edits one file instead of every plan task (critique
# C13/C15: three stale v164 defaults and a hardcoded -cs outlived two model swaps).
$statePath = if ($State -ne "") { $State } else { Join-Path $Root "state.json" }
if (-not (Test-Path $statePath)) {
    Write-Output ("ABORT: no state.json at " + $statePath + " (pass -State, or deploy with deploy.sh)")
    exit 2
}
$ST = (Get-Content $statePath -Raw | ConvertFrom-Json).active
if ($null -eq $ST -or $null -eq $ST.model) { Write-Output ("ABORT: " + $statePath + " has no .active.model"); exit 2 }

if ($Model -eq "")   { $Model = [string]$ST.model }
$MODEL = $Model
if ($Split -eq 0)    { $Split = [int]$ST.split }
if ($Threads -eq 0)  { $Threads = [int]$ST.threads }
if ($MaxLength -eq 0){ $MaxLength = [int]$ST.max_length }
# The cache under test is not one number. A spec_decode pass sizes its cache for a short generation
# (a 1024-token round needs 139 pages = 35584 tokens, so the pp/tg ladder's 131072 would be a
# different allocation) and the pp/tg ladder sizes it for its measurement length. -cs was cache_size
# for both runners and -Extra appended a second one on top: val-spec 2026-10-29 logged "-cs 131072
# ... -mtp -cs 40960", so the recorded regime and the cache the process used were not one number.
$ChunkSize   = if ($ChunkSize -ne 0) { $ChunkSize } elseif ($null -ne $ST.chunk_size) { [int]$ST.chunk_size } else { 4096 }
$CacheQuant  = [string]$ST.cache_quant
$Draft       = $ST.draft
$stateCs     = [int]$ST.cache_size
if ($Runner -eq "spec_decode" -and $null -ne $Draft -and $null -ne $Draft.spec_decode_cache_size -and
    [int]$Draft.spec_decode_cache_size -gt 0) {
    $stateCs = [int]$Draft.spec_decode_cache_size
}
if ($CacheSize -eq 0) { $CacheSize = $stateCs }
if (-not $PSBoundParameters.ContainsKey("MinFreeVramMB")) { $MinFreeVramMB = [int]$ST.floors.min_free_vram_mb }
if (-not $PSBoundParameters.ContainsKey("MinFreeRamMB"))  { $MinFreeRamMB  = [int]$ST.floors.min_free_ram_mb }
$LOGS  = Join-Path $Root "logs"
$RES   = Join-Path $Root "results"
if ($Runner -eq "spec_decode") {
    $PERF = Join-Path $Root "spec_decode.py"
    $MATCH_CMD = "spec_decode\.py"
    $RunnerName = "spec_decode"   # what -DryRun reports: the runner actually selected
} elseif ($Runner -eq "concurrent") {
    # Two jobs in flight: the only runner that exercises max_batch_size, which every other runner
    # leaves at one job because it enqueues a single prompt.
    $PERF = Join-Path $Root "concurrency.py"
    $MATCH_CMD = "concurrency\.py"
    $RunnerName = "concurrent"
} elseif ($Runner -eq "needles") {
    # Multi-key needle retrieval: K access codes planted at hashed depths, one prompt asks for all of
    # them, and the hit count is the score. longctx saturates at 16/16 in every cache mode, so it
    # cannot answer a cache-precision question; this runner can.
    $PERF = Join-Path $Root "needles.py"
    $MATCH_CMD = "needles\.py"
    $RunnerName = "needles"
} elseif ($Runner -eq "longctx") {
    # Retrieval at length: eval/longctx.py reads stories from eval_texts/ and prints one LONGCTX
    # line per test plus a count. Its verdict is a reader's, but the run has to prove which tests
    # ran and what context each one saw.
    $PERF = Join-Path $Root "longctx.py"
    $MATCH_CMD = "longctx\.py"
    $RunnerName = "longctx"
} else {
    $PERF = Join-Path $Root "perf.py"
    $MATCH_CMD = "perf\.py"
    $RunnerName = "perf"
}
$WATCH = Join-Path $Root "watch.ps1"

Set-Location $Root   # trap 2: a WSL launch leaves a UNC working directory behind

foreach ($f in @($PY, $PERF, $WATCH)) {
    if (-not (Test-Path $f)) { Write-Output "ABORT: missing $f (deploy with bench/windows/deploy.sh)"; exit 2 }
}
if (-not (Test-Path $MODEL)) { Write-Output "ABORT: model not found: $MODEL"; exit 2 }

# One owner of the box: refuse to start while another pass or a server is running. Skipped under
# -DryRun, which starts nothing: before this the gate ran first, so a dry run exited 3 for any
# running pass or for the tabbyAPI server itself (its venv path contains tabbyAPI) and the argv
# could not be checked while the card was busy - the one case this script documents as supported.
if (-not $DryRun) {
    $other = @(Get-CimInstance Win32_Process |
               Where-Object { $_.Name -match '^python' -and $_.ProcessId -ne $PID -and
                              $_.CommandLine -match 'perf\.py|spec_decode\.py|tabbyAPI' })
    if ($other.Count -gt 0) {
        Write-Output "ABORT: another pass is running:"
        $other | ForEach-Object { Write-Output ("  " + $_.ProcessId + " " + $_.CommandLine) }
        exit 3
    }
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

# --- Regime ------------------------------------------------------------------------------------
# Built from state.json. The v1.6.4 hardcoded block (model 2.05bpw, -cs 262144, -mcs 360) is gone:
# it survived two model swaps and silently disagreed with the plans. "production" is accepted as an
# alias so older invocations keep working; anything else is an error.
if ($Regime -ne "production" -and $Regime -ne "state") {
    Write-Output ("ABORT: unknown regime '" + $Regime + "'. Known: state (alias: production)")
    exit 2
}
$draftArgs = @()
if ($null -ne $Draft -and $Runner -eq "spec_decode") {
    # Drafting args belong to the drafting runner alone: eval/perf.py calls model_init.add_args()
    # without add_draft_model_args and unpacks four values from model_init.init(), so -mtp/-ndt in a
    # pp/tg argv aborts at argparse - and a pp/tg ladder measures the main model, so drafting is not
    # part of its regime. The state names a drafting MODE, not a sidecar: the MTP head rides the main
    # model, so -dm is never emitted. -sdh/-hdt are passed only when non-zero because 0 is the
    # engine's own default (full head / hybrid drafting off), and draft.readback has no per-run
    # model_init switch at 1.6.5 - the only readback control is spec_decode.py's -rb_ab, which
    # measures all four arms in one process - so it stays information here. A pass that measured
    # acc=1.74/4.00 on four draft tokens while state.json said 2 is what this block exists to make
    # impossible.
    if ([string]$Draft.mode -eq "mtp") { $draftArgs += "-mtp" }
    # 0 (or absent) is treated as unset rather than sent as "-ndt 0": the engine's default is 4 draft
    # tokens, so a state value of 0 would silently run 4, and the way to see that from the outside is
    # the absence of -ndt in the argv (a real 0 is not a drafting regime anyone can measure).
    if ($null -ne $Draft.num_draft_tokens -and [int]$Draft.num_draft_tokens -gt 0) {
        $draftArgs += @("-ndt", "$([int]$Draft.num_draft_tokens)")
    }
    if ($null -ne $Draft.sparse_draft_head -and [int]$Draft.sparse_draft_head -gt 0) {
        $draftArgs += @("-sdh", "$([int]$Draft.sparse_draft_head)")
    }
    if ($null -ne $Draft.hybrid_draft_tokens -and [int]$Draft.hybrid_draft_tokens -gt 0) {
        $draftArgs += @("-hdt", "$([int]$Draft.hybrid_draft_tokens)")
    }
    if ($Draft.dynamic_draft -eq $true) { $draftArgs += "-dds" }
}
# The regime's own argv, kept apart from the deltas: the duplicate guard below has to see exactly what
# state.json decided, including the draft flags when they were emitted, and it has to see it BEFORE the
# spec_decode -max_length strip can eat a restated copy.
# FP16 is a cache mode with no flag: the engine's quantised cache asserts 2..8 bits, so -cq 16,16
# aborts at CacheLayer_quant's own assert (measured 2026-11-02: AssertionError after a 3 s load). The
# word the server's config.yml uses is "FP16", so the state may say it and the argv says nothing.
$cqArgs = @()
if ($CacheQuant -ne "" -and $CacheQuant -notmatch '^(?i)fp16$') { $cqArgs = @("-cq", $CacheQuant) }
$regimeArgs = @("-m", $MODEL, "-cs", "$CacheSize", "-chunk_size", "$ChunkSize", "-ambs", "2") + $cqArgs +
              @("-max_length", "$MaxLength", "-mcs", "$Split", "-mct", "$Threads")
$regimeArgs += $draftArgs

# Every spelling argparse accepts for the flags above, mapped to the canonical short name the regime
# and the abort message use. A token scan that only knows "-cs" counts one flag where the engine takes
# the value twice and keeps the last one: "-cs=40960" and "--cache_size 40960" both set the cache size
# (allow_abbrev=False does not stop the =value form or a long alias), which is the val-spec
# 2026-10-29 mismatch in another spelling.
$regimeSpellings = @{
    "-m" = "-m";                  "--model_dir" = "-m"
    "-cs" = "-cs";                "--cache_size" = "-cs"
    "-cq" = "-cq";                "--cache_quant" = "-cq"
    "-mcs" = "-mcs";              "--moe_cpu_split" = "-mcs"
    "-mct" = "-mct";              "--moe_cpu_threads" = "-mct"
    "-max_length" = "-max_length"; "--max_length" = "-max_length"
    "-chunk_size" = "-chunk_size"; "--chunk_size" = "-chunk_size"
    "-ambs" = "-ambs";            "--autosplit_max_batch_size" = "-ambs"
    "-mtp" = "-mtp";              "--mtp" = "-mtp"
    "-ndt" = "-ndt";              "--num_draft_tokens" = "-ndt"
    "-sdh" = "-sdh";              "--sparse_draft_head" = "-sdh"
    "-hdt" = "-hdt";              "--hybrid_draft_tokens" = "-hdt"
    "-dds" = "-dds";              "--dynamic_draft" = "-dds"
}
# How to change an owned flag properly. A flag with no delta parameter is a state.json edit, and the
# message says so instead of offering a parameter that does not exist.
$regimeRemedy = @{
    "-m"          = "use -Model, or edit state.json"
    "-cs"         = "use -CacheSize, or edit state.json"
    "-max_length" = "use -MaxLength, or edit state.json"
    "-mcs"        = "use -Split, or edit state.json"
    "-mct"        = "use -Threads, or edit state.json"
    "-chunk_size" = "-ChunkSize"
    "-ambs"       = "edit state.json"
    "-cq"         = "edit state.json"
    "-mtp"        = "edit state.json"
    "-ndt"        = "edit state.json"
    "-sdh"        = "edit state.json"
    "-hdt"        = "edit state.json"
    "-dds"        = "edit state.json"
}

# Canonical short flag named by an argv token, or $null for a value ("131072", "2,2", "D:\llms\...").
# The =value form is stripped before the lookup so "-cs=40960" and "--cache_size=40960" resolve too.
function Get-RegimeFlag([string] $tok) {
    if (-not $tok.StartsWith("-")) { return $null }
    $name = ($tok.Split("=")[0]).ToLowerInvariant()
    if ($regimeSpellings.ContainsKey($name)) { return $regimeSpellings[$name] }
    return $null
}

# -Extra may not restate a flag the regime owns. Only the regime decides those; a restatement is how
# the recorded argv and the running process stopped describing the same pass, so it aborts instead.
$owned = @{}
foreach ($tok in $regimeArgs) {
    $flag = Get-RegimeFlag $tok
    if ($null -ne $flag) { $owned[$flag] = $true }
}
$argsList = $regimeArgs   # the emitted argv starts as the regime's own; -Extra appends to a copy
if ($Extra -ne "") {
    foreach ($tok in ($Extra -split '\s+')) {
        if ($tok -eq "") { continue }
        $flag = Get-RegimeFlag $tok
        if ($null -ne $flag -and $owned.ContainsKey($flag)) {
            Write-Output ("ABORT: -Extra repeats " + $flag + ", which the regime owns (" + $regimeRemedy[$flag] + ")")
            exit 2
        }
        $argsList += $tok
    }
}
if ($Short)          { $argsList += "-short" }
if (-not $NoSpecDec -and $Runner -eq "perf") { $argsList += "-sd" }
if ($Runner -ne "perf") {
    # -max_length is perf.py's ladder length; spec_decode.py and longctx.py take model_init args
    # plus their own, and a fixed-size array has no RemoveAt: rebuild without the flag and its value
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
$baseEnv = [ordered]@{}
if ($ST.env) { foreach ($pr in $ST.env.PSObject.Properties) { $baseEnv[$pr.Name] = [string]$pr.Value } }
# Drafting arms are read with dynamic expert placement OFF (state.knob_arms_disable_swap): a drafter
# that accepts an extra token also changes which experts stay resident, so a number measured with
# both knobs moving says nothing about either. The field is named for knob arms and the pp/tg ladder
# is not one: 44 archived ladder rows were measured with placement on, so forcing it off for every
# ladder run would silently break the series the T4 thresholds and the T10 ladder are read against.
# Hence the scope - automatic for -Runner spec_decode (the arm-to-arm drafting comparisons) and for a
# run that asks for it with -StaticSwap; a plain ladder run keeps the engine default. It lands in the
# regime env, before -Unset/-Env, so an arm can still override it (-Env "EXL3_MOE_CPU_SWAP=1") and
# that delta is measured, not assumed; $envApplied copies $baseEnv below, which puts it in the log
# header and in passes.csv.
# A retrieval run is a quality comparison, so placement must be frozen for it too: a sweep mid-
# generation changes rounding and can flip the greedy token the answer depends on.
if ($ST.knob_arms_disable_swap -eq $true -and ($Runner -eq "spec_decode" -or $Runner -eq "longctx" -or $StaticSwap)) {
    $baseEnv["EXL3_MOE_CPU_SWAP"] = "0"
}
foreach ($k in $baseEnv.Keys) { Set-Item "env:$k" $baseEnv[$k] }
$envApplied = [ordered]@{}
foreach ($k in $baseEnv.Keys) { $envApplied[$k] = $baseEnv[$k] }

# Frozen hidden state (critique C1/C2, improvement I2): both engine profiles - the routed-expert
# frequencies and the sparse-head token frequencies - live under XDG_CACHE_HOME. Pointing it at a
# frozen snapshot makes every arm start from the same placement instead of "whoever ran first pays
# the cold run and everybody after inherits its profile".
$frozen = [string]$ST.frozen_state_dir
$master = [string]$ST.frozen_state_master
if ($frozen -ne "" -and -not $NoFrozenState) {
    if ($master -ne "" -and (Test-Path $master)) {
        # Start every run from the SAME profile: without this, arm N inherits whatever placement the
        # previous run learned and the arms are not comparable (critique C1/C2, improvement I2).
        # A -DryRun reports the same FROZEN_STATE and XDG_CACHE_HOME without doing it: the restore
        # deletes and repopulates engine-state/active, and a dry run writes nothing.
        if (-not $DryRun) {
            if (Test-Path $frozen) { Remove-Item $frozen -Recurse -Force -ErrorAction SilentlyContinue }
            New-Item -ItemType Directory -Force $frozen | Out-Null
            Copy-Item -Path (Join-Path $master "*") -Destination $frozen -Recurse -Force -ErrorAction SilentlyContinue
        }
        $envApplied["FROZEN_STATE"] = "master->active"
    } else {
        if (-not $DryRun) { New-Item -ItemType Directory -Force $frozen | Out-Null }
    }
    Set-Item "env:XDG_CACHE_HOME" $frozen
    $envApplied["XDG_CACHE_HOME"] = $frozen
}
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

# --- Dry run ------------------------------------------------------------------------------------
# The argv and environment a real pass would use, printed after every gate that can refuse a run: an
# argv checked against a different engine, model or package than the one on the box proves nothing.
# The frozen-profile restore above is skipped under -DryRun, so nothing under engine-state/ is
# rewritten; the one-owner gate above is skipped too, and this exit lands before the output
# directories are created, before the log path is stamped and before Start-Process: no log, no CSV
# row, no watcher trace, and the GPU is never touched - usable while the card is busy.
if ($DryRun) {
    Write-Output ("DRYRUN runner=" + $RunnerName + " state=" + $statePath)
    Write-Output ("ARGS " + ($argsList -join " "))
    Write-Output ("ENV " + (($envApplied.Keys | Sort-Object | ForEach-Object { "$_=$($envApplied[$_])" }) -join " "))
    exit 0
}

# --- Run ---------------------------------------------------------------------------------------
# The output directories are made here, not with the state load: creating two directories is a write,
# and a -DryRun returns above having made none.
New-Item -ItemType Directory -Force $LOGS, $RES | Out-Null
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

# spec_decode.py prints a per-category table; without this every drafting arm lived only in the log
# and no CSV analysis was possible (critique C13b). Both the rendered table and the machine-readable
# "ARM ..." lines are captured, so acceptance and latency are separable afterwards (improvement I4).
$sdRows = New-Object System.Collections.ArrayList
$armRows = New-Object System.Collections.ArrayList
$sdHeader = $null
foreach ($rawLine in (($raw -replace "`r", "`n") -split "`n")) {
    $l = ($rawLine -replace $ansi, "").Trim()
    if ($l -match '^ARM\s+(\S+)\s+tok_s=([0-9.]+)(?:\s+acc=([0-9.]+)/([0-9.]+))?(?:\s+rounds=(\d+))?') {
        [void]$armRows.Add(@($Matches[1], [double]::Parse($Matches[2], $inv),
                             $(if ($Matches[3] -ne "") { [double]::Parse($Matches[3], $inv) } else { -1.0 }),
                             $(if ($Matches[4] -ne "") { [double]::Parse($Matches[4], $inv) } else { -1.0 }),
                             $(if ($Matches[5] -ne "") { [int]$Matches[5] } else { 0 })))
        continue
    }
    if ($l -match '^\|\s*Category\s*\|') {
        $cells = @($l.Trim('|').Split('|') | ForEach-Object { $_.Trim() })
        $sdHeader = @($cells[1..($cells.Count - 1)])
        continue
    }
    if ($null -ne $sdHeader -and $l -match '^\|') {
        $cells = @($l.Trim('|').Split('|') | ForEach-Object { $_.Trim() })
        if ($cells.Count -ne ($sdHeader.Count + 1)) { continue }
        if ($cells[1] -match '^-+$' -or $cells[0] -match '^-+$') { continue }
        for ($c = 1; $c -lt $cells.Count; $c++) {
            $cell = $cells[$c]
            if ($cell -notmatch '([0-9.]+)\s+t/s') { continue }
            $tps = [double]::Parse($Matches[1], $inv)
            $ratio = 0.0; $acc = -1.0; $draft = -1.0
            if ($cell -match '([0-9.]+)x') { $ratio = [double]::Parse($Matches[1], $inv) }
            if ($cell -match '([0-9.]+)/([0-9.]+)\s+acc/draft') {
                $acc = [double]::Parse($Matches[1], $inv); $draft = [double]::Parse($Matches[2], $inv)
            }
            [void]$sdRows.Add(@($cells[0], $sdHeader[$c - 1], $tps, $ratio, $acc, $draft))
        }
    }
}

# Retrieval runs: one LONGCTX line per test, parsed so a run that produced none is a failure rather
# than an `ok` row with no evidence (the same rule the drafting runner follows).
$lcRows = New-Object System.Collections.ArrayList
foreach ($rawLine in (($raw -replace "`r", "`n") -split "`n")) {
    $l = ($rawLine -replace $ansi, "").Trim()
    if ($l -match '^LONGCTX job=(\S+)\s+prompt_tokens=(\d+)\s+out_chars=(\d+)') {
        [void]$lcRows.Add(@($Matches[1], [int]$Matches[2], [int]$Matches[3]))
    }
}

# Needle runs: the pairs are printed by the runner, and the hit is computed here against the answer
# text so a run carries its own score instead of needing a second pass over the log.
$ndRows = New-Object System.Collections.ArrayList
# Strip EVERY truth line, not just the first one: with -nt 2 the answers and the truth lines
# interleave, so cutting at the first NEEDLE line scored the second haystack against nothing and
# reported 20/40 for a run that found 40/40 (measured 2026-11-02).
$answerText = ((($raw -replace "`r", "`n") -split "`n") | Where-Object { $_ -notmatch '^NEEDLE' }) -join "`n"
foreach ($rawLine in (($raw -replace "`r", "`n") -split "`n")) {
    $l = ($rawLine -replace $ansi, "").Trim()
    if ($l -match '^NEEDLE name=(\S+)\s+code=(\d+)$') {
        $nm = $Matches[1]; $cd = $Matches[2]
        $hit = 0
        if ($answerText.Contains($cd)) { $hit = 1 }
        [void]$ndRows.Add(@($nm, $cd, $hit))
    }
}

# Concurrency runs: one summary line per run, with the aggregate throughput that says whether two
# jobs cost one wall time or two.
$ccRows = New-Object System.Collections.ArrayList
foreach ($rawLine in (($raw -replace "`r", "`n") -split "`n")) {
    $l = ($rawLine -replace $ansi, "").Trim()
    if ($l -match '^CONC jobs=(\d+) prompt_tokens=(\d+)\.\.(\d+) new_tokens=(\d+) wall_s=([0-9.]+) agg_tps=([0-9.]+)') {
        [void]$ccRows.Add(@([int]$Matches[1], [int]$Matches[2], [int]$Matches[3],
                            [int]$Matches[4], [double]::Parse($Matches[5], $inv),
                            [double]::Parse($Matches[6], $inv)))
    }
}

# torch allocator watermarks (improvement I3b): driver-free VRAM says nothing about how much of the
# card PyTorch is holding in cached blocks, which is what actually kills runs at the floor
$peakAlloc = 0.0
$peakReserved = 0.0
foreach ($m in [regex]::Matches($raw, 'alloc peak allocated=([0-9.]+) MiB reserved=([0-9.]+) MiB')) {
    $a = [double]::Parse($m.Groups[1].Value, $inv); $r2 = [double]::Parse($m.Groups[2].Value, $inv)
    if ($a -gt $peakAlloc) { $peakAlloc = $a }
    if ($r2 -gt $peakReserved) { $peakReserved = $r2 }
}

# --- Verdict -----------------------------------------------------------------------------------
$status = "ok"
if ($breach) { $status = "breach" }
elseif ($traceback) { $status = "fail" }
elseif ($Runner -eq "perf" -and $ppRows.Count -eq 0 -and $tgRows.Count -eq 0) { $status = "fail" }
# A drafting pass that produced no evidence at all is a failure, for the same reason the ladder rule
# above exists: the venv python's exit code is unreadable through the uv shim, so the verdict has to
# come from the evidence. spec_decode.py now exits 2 at the swap gate before any workload runs, and a
# pass that dies before its first row leaves neither an ARM line nor a table row. Both are required to
# be absent: a single-arm n-gram pass (-nbl -ngram_min 3) prints the per-category table but no ARM
# line, because arm_line() only runs for the baseline and draft arms, and that pass did measure.
elseif ($Runner -eq "spec_decode" -and $armRows.Count -eq 0 -and $sdRows.Count -eq 0) { $status = "fail" }
elseif ($Runner -eq "longctx" -and $lcRows.Count -eq 0) { $status = "fail" }
elseif ($Runner -eq "needles" -and $ndRows.Count -eq 0) { $status = "fail" }
elseif ($Runner -eq "concurrent" -and $ccRows.Count -eq 0) { $status = "fail" }

# --- Persist -----------------------------------------------------------------------------------
$passesCsv = Join-Path $RES "passes.csv"
if (-not (Test-Path $passesCsv)) {
    Set-Content -Path $passesCsv -Value "tag,run_id,finished_utc,regime,status,engine,ext,elapsed_s,max_length,min_free_vram_mb,min_avail_ram_mb,peak_proc_ws_mb,samples,breach,env,split,threads,args,harness_sha256,log,runner,peak_alloc_mb,peak_reserved_mb" -Encoding ASCII
}
$envStr = (($envApplied.Keys | Sort-Object | ForEach-Object { "$_=$($envApplied[$_])" }) -join " ")
$argStr = (($argsList -join " ") -replace ",", ";")
$breachInt = 0
if ($breach) { $breachInt = 1 }
# The cache the run used is in the args cell (-cs N), not here: this cell is the ladder length.
# A drafting pass never receives -max_length (stripped above), so its cell stays empty instead of
# carrying a state value the process did not get - ratios.py prints that column as provenance, and
# the drafting length it did use is the -cs in the args cell.
$maxLengthCell = $MaxLength
if ($Runner -eq "spec_decode") { $maxLengthCell = "" }
$row = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22}" -f `
    $Tag, $runId, (Get-Date).ToString("s"), $Regime, $status, $engine, $extName, $elapsed, $maxLengthCell, `
    $minVram, $minRam, $peakWs, $samples, $breachInt, $envStr, $Split, $Threads, $argStr, $harnessHash, `
    ("perf-" + $runId + ".log"), $Runner, ([string]::Format($inv, "{0:F0}", $peakAlloc)), ([string]::Format($inv, "{0:F0}", $peakReserved))
Add-Content -Path $passesCsv -Value $row -Encoding ASCII

$ppCsv = Join-Path $RES "pp.csv"
if (-not (Test-Path $ppCsv)) { Set-Content -Path $ppCsv -Value "tag,run_id,band,pp_tok_s" -Encoding ASCII }
foreach ($r in $ppRows) {
    Add-Content -Path $ppCsv -Value ("{0},{1},{2},{3}" -f $Tag, $runId, $r[0], ([string]::Format($inv, "{0:F2}", $r[1]))) -Encoding ASCII
}

$sdCsv = Join-Path $RES "spec_decode.csv"
if (-not (Test-Path $sdCsv)) {
    Set-Content -Path $sdCsv -Value "tag,run_id,category,arm,tok_s,ratio,acc,draft" -Encoding ASCII
}
foreach ($r in $sdRows) {
    Add-Content -Path $sdCsv -Value ("{0},{1},{2},{3},{4},{5},{6},{7}" -f $Tag, $runId, ($r[0] -replace ",", ";"), ($r[1] -replace ",", ";"),
        ([string]::Format($inv, "{0:F2}", $r[2])), ([string]::Format($inv, "{0:F3}", $r[3])),
        ([string]::Format($inv, "{0:F2}", $r[4])), ([string]::Format($inv, "{0:F2}", $r[5]))) -Encoding ASCII
}
$armCsv = Join-Path $RES "arms.csv"
if (-not (Test-Path $armCsv)) {
    Set-Content -Path $armCsv -Value "tag,run_id,arm,tok_s,acc,draft,rounds" -Encoding ASCII
}
foreach ($r in $armRows) {
    Add-Content -Path $armCsv -Value ("{0},{1},{2},{3},{4},{5},{6}" -f $Tag, $runId, $r[0],
        ([string]::Format($inv, "{0:F2}", $r[1])), ([string]::Format($inv, "{0:F2}", $r[2])),
        ([string]::Format($inv, "{0:F2}", $r[3])), $r[4]) -Encoding ASCII
}

$lcCsv = Join-Path $RES "longctx.csv"
if (-not (Test-Path $lcCsv)) { Set-Content -Path $lcCsv -Value "tag,run_id,job,prompt_tokens,out_chars" -Encoding ASCII }
foreach ($r in $lcRows) {
    Add-Content -Path $lcCsv -Value ("{0},{1},{2},{3},{4}" -f $Tag, $runId, $r[0], $r[1], $r[2]) -Encoding ASCII
}

$ccCsv = Join-Path $RES "concurrent.csv"
if (-not (Test-Path $ccCsv)) {
    Set-Content -Path $ccCsv -Value "tag,run_id,jobs,prompt_min,prompt_max,new_tokens,wall_s,agg_tps" -Encoding ASCII
}
foreach ($r in $ccRows) {
    Add-Content -Path $ccCsv -Value ("{0},{1},{2},{3},{4},{5},{6},{7}" -f $Tag, $runId, $r[0], $r[1], $r[2], $r[3],
        ([string]::Format($inv, "{0:F1}", $r[4])), ([string]::Format($inv, "{0:F2}", $r[5]))) -Encoding ASCII
}

$ndCsv = Join-Path $RES "needles.csv"
if (-not (Test-Path $ndCsv)) { Set-Content -Path $ndCsv -Value "tag,run_id,name,code,hit" -Encoding ASCII }
foreach ($r in $ndRows) {
    Add-Content -Path $ndCsv -Value ("{0},{1},{2},{3},{4}" -f $Tag, $runId, $r[0], $r[1], $r[2]) -Encoding ASCII
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
Write-Output ("FLOORS min_free_vram_mb=$minVram min_avail_ram_mb=$minRam peak_proc_ws_mb=$peakWs samples=$samples breach=$breachInt")
if ($peakAlloc -gt 0) { Write-Output ("ALLOC  peak_allocated_mb=" + [string]::Format($inv, "{0:F0}", $peakAlloc) + " peak_reserved_mb=" + [string]::Format($inv, "{0:F0}", $peakReserved)) }
foreach ($r in $armRows) { Write-Output ("ARM {0} tok_s={1} acc={2}/{3} rounds={4}" -f $r[0], [string]::Format($inv, "{0:F2}", $r[1]), [string]::Format($inv, "{0:F2}", $r[2]), [string]::Format($inv, "{0:F2}", $r[3]), $r[4]) }
if ($ppRows.Count -gt 0) {
    $parts = @()
    foreach ($r in $ppRows) { $parts += ("{0}={1}" -f $r[0], [string]::Format($inv, "{0:F2}", $r[1])) }
    Write-Output ("PP   " + ($parts -join " "))
}
if ($ccRows.Count -gt 0) {
    foreach ($r in $ccRows) {
        Write-Output ("CONCURRENT jobs=" + $r[0] + " wall_s=" + [string]::Format($inv, "{0:F1}", $r[4]) +
                      " agg_tps=" + [string]::Format($inv, "{0:F2}", $r[5]) + " new_tokens=" + $r[3])
    }
}
if ($ndRows.Count -gt 0) {
    $hits = 0
    foreach ($r in $ndRows) { if ($r[2] -eq 1) { $hits++ } }
    Write-Output ("NEEDLES n=" + $ndRows.Count + " hits=" + $hits)
}
if ($lcRows.Count -gt 0) {
    $parts = @()
    foreach ($r in $lcRows) { $parts += ("{0}:{1}tok/{2}ch" -f $r[0], $r[1], $r[2]) }
    Write-Output ("LONGCTX " + $lcRows.Count + " test(s) " + ($parts -join " "))
}
if ($tgRows.Count -gt 0) {
    $parts = @()
    foreach ($r in $tgRows) { $parts += ("{0}={1}" -f $r[0], [string]::Format($inv, "{0:F2}", $r[1][1])) }
    Write-Output ("TG_S1 " + ($parts -join " "))
}
Write-Output ("CSV  " + $passesCsv + " ; " + $ppCsv + " ; " + $tgCsv + " ; " + $sdCsv + " ; " + $armCsv + " ; " + $lcCsv + " ; " + $wCsv)

if ($status -eq "breach") {
    Write-Output ("BREACH tag=$Tag run=$runId floors violated; see " + (Join-Path $RES ("BREACH-" + $runId + ".txt")))
    exit 4
}
if ($status -eq "fail") { exit 5 }
exit 0
