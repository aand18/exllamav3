# _kb_phase_c.ps1 -- knob battery, PHASE C: the LIVE server.
#
# The only phase that boots tabbyAPI, so it is the only one that can measure
# boot time, and the only one that can reach the server-only knobs
# (max_batch_size concurrency, warmup, recurrent_checkpoint_interval,
# cuda_malloc_async, draft_cache_mode). Also used to validate Phase A/B
# winners end-to-end.
#
# Config edits are working-tree only and always restored from a per-arm backup.
# config.yml is NEVER committed.
#
# Env discipline: the tuned env is read straight out of start_tuned.ps1 by
# Get-KBTunedEnv rather than hardcoded here, so this script can never drift from
# the production launcher. start_tuned.ps1 itself is NOT invoked, because it
# redirects to a single fixed server-final.log and these runs need one log per
# arm; the env it sets is applied verbatim instead.
#
# Run:  powershell -NoProfile -ExecutionPolicy Bypass -File .\_kb_phase_c.ps1 -Arm c00-baseline

param(
    [string]$Arm = 'c00-baseline',
    [int]$Concurrent = 1,
    [switch]$FullProtocol,
    # Apply the arm's config edits, print them, restore, exit. No server boot.
    # Set-KBConfig matches lines by regex and config.yml has commented AND live
    # occurrences of some keys (max_batch_size is both); a silent miss would
    # invalidate the arm, and settings fail silently on a typo.
    [switch]$DryRun,
    # SUSTAINED conversation after warmup, all on ONE boot. This is the metric
    # the operator actually cares about: production perf is many requests against
    # a long-lived server, so reps belong WITHIN a boot, not across reboots.
    # Boot time and first-request TTFT are reported but demoted -- one restart is
    # amortized over hours of serving, while request N is the steady state.
    #
    # -Sustained is the number of REQUESTS (turns), not conversations. The block
    # replays a multi-turn conversation in order (prompt -> answer -> prompt),
    # which is what production actually looks like: each turn extends the same
    # conversation, so the prefix cache stays warm the way it does in real use
    # AND prompt content varies turn to turn. -Rotate prompts starts a fresh
    # conversation per turn instead, so every request pays a cold prefix and pp
    # is measured rather than being skipped as ~100% cached.
    [int]$Sustained = 0,
    [string[]]$Conversation = @(),
    [switch]$Rotate
)

. "$PSScriptRoot\_kb_lib.ps1"

# ---------------------------------------------------------------- standing config
# Operator directive 2026-10-06: ALWAYS vision_offload: true and warmup: true.
# Applied to EVERY Phase C arm as a floor, before the arm's own cfg, so the
# battery measures knobs on top of the config that will actually ship.
#   vision_offload: true -- vision tower (~1.1 GB, unquantized fp16, the
#     largest single reason the live server sits above the offline harness)
#     moves to pinned host memory. Trades vision speed for VRAM; this workload
#     never sends images, so that trade is free here.
#   warmup: true -- +12 s measured boot for kernel/graph capture at load.
$KBStandingCfg = [ordered]@{
    vision_offload = 'true'
    warmup         = 'true'
}
# key -> anchor line, for keys absent from config.yml.
$KBStandingAnchor = @{ vision_offload = 'vision' }
# Applied first so an arm can still override a standing key explicitly.
$KBStandingOrder = @('vision_offload', 'warmup')

# PS 5.1 does not resolve System.Net.Http types until the assembly is loaded.
# Without this the boot probe dies with "Unable to find type [Net.Http.HttpClient]"
# AFTER the server is already up, which is how the first run leaked a server.
Add-Type -AssemblyName System.Net.Http

$KB.Config   = Join-Path $KB.Tabby 'config.yml'
# Per-BOOT log, not per-arm. It was "$Arm.server.log", so every boot of the same
# arm overwrote the last one. That is why the 250k fast/slow investigation could
# never see the CPU MoE arena reservation line for more than one boot -- it is the
# only per-boot server property still correlated with the mode (34.09GB slow vs
# 34.63GB fast) and the log holding it was destroyed on the next boot.
$KB.ServerLog = Join-Path $KB.LogRoot ("{0}.{1}.server.log" -f $Arm, (Get-Date -Format 'HHmmss'))
$KB.Jsonl    = Join-Path $KB.LogRoot 'phaseC.jsonl'
$script:BootProps = @()
if (-not (Test-Path $KB.LogRoot)) { New-Item -ItemType Directory -Force $KB.LogRoot | Out-Null }

# ------------------------------------------------------------------- config

# Parse start_tuned.ps1 so the tuned env has exactly one source of truth.
function Get-KBTunedEnv {
    $f = Join-Path $KB.Tabby 'start_tuned.ps1'
    $h = @{}
    foreach ($line in (Get-Content $f)) {
        if ($line -match '^\s*\$env:(EXL3_MOE_\w+)\s*=\s*"?([^"]+)"?\s*$') { $h[$Matches[1]] = $Matches[2].Trim() }
    }
    if ($h.Count -eq 0) { throw "parsed no env out of $f -- refusing to run with an untuned env" }
    return $h
}

# Set one config.yml key. Prefers the UNCOMMENTED occurrence (the live value);
# falls back to a commented one and uncomments it. Throws when the key is
# absent -- settings fail silently on a typo, so a silent no-op here would
# invalidate the whole arm.
function Set-KBConfig([string]$key, [string]$value, [string]$Anchor = '') {
    $lines = Get-Content $KB.Config
    $live = -1; $commented = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^(\s*)#?\s*$([regex]::Escape($key))\s*:") {
            if ($lines[$i] -match "^\s*$([regex]::Escape($key))\s*:") { $live = $i } else { $commented = $i }
        }
    }
    $idx = if ($live -ge 0) { $live } elseif ($commented -ge 0) { $commented } else { -1 }
    if ($idx -lt 0) {
        # Absent key: insert it after its anchor so it lands in the right
        # section. Appending at EOF would put a model-section key under
        # `memory:` and the loader would ignore it -- a silent no-op, which is
        # exactly what this function exists to prevent.
        if (-not $Anchor) { throw "key '$key' not found in config.yml and no -Anchor given" }
        $a = -1
        for ($i = 0; $i -lt $lines.Count; $i++) {
            # Accept a COMMENTED anchor too: several keys in config.yml exist only
            # as `#key:` (recurrent_checkpoint_interval is one), and the new key
            # must land beside its sibling, not at EOF.
            if ($lines[$i] -match "^\s*#?\s*$([regex]::Escape($Anchor))\s*:") { $a = $i; break }
        }
        if ($a -lt 0) { throw "key '$key' absent and anchor '$Anchor' not found" }
        $indent = ([regex]::Match($lines[$a], '^(\s*)')).Groups[1].Value
        $lines = @($lines[0..$a]) + @("$indent$key`: $value") + @($lines[($a + 1)..($lines.Count - 1)])
        Set-Content -Path $KB.Config -Value $lines
        return @{ key = $key; value = $value; line = ($a + 2); wasLive = $false; inserted = $true }
    }
    $indent = ([regex]::Match($lines[$idx], '^(\s*)')).Groups[1].Value
    $lines[$idx] = "$indent$key`: $value"
    Set-Content -Path $KB.Config -Value $lines
    return @{ key = $key; value = $value; line = ($idx + 1); wasLive = ($live -ge 0) }
}

function Set-KBConfigDrop([string]$key) {
    # Comment a key out. Required for cpu_moe_split_experts, which the engine
    # rejects alongside cpu_moe_offload_layers
    # (backends/exllamav3/model.py:208).
    $lines = Get-Content $KB.Config
    $idx = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^(\s*)#?\s*$([regex]::Escape($key))\s*:") { $idx = $i; break }
    }
    if ($idx -lt 0) { throw "cannot drop '$key': not found in config.yml" }
    $indent = ([regex]::Match($lines[$idx], '^(\s*)')).Groups[1].Value
    if ($lines[$idx] -notmatch '^\s*#') { $lines[$idx] = "$indent#$($lines[$idx].Trim())" }
    Set-Content -Path $KB.Config -Value $lines
    return @{ key = $key; value = '(commented out)'; line = ($idx + 1) }
}

function Get-KBConfigValues([string[]]$keys) {
    $out = @{}
    foreach ($k in $keys) {
        foreach ($line in (Get-Content $KB.Config)) {
            if ($line -match "^\s*$([regex]::Escape($k))\s*:\s*(.*?)\s*$") { $out[$k] = $Matches[1] }
        }
    }
    return $out
}

# ------------------------------------------------------------- server control

function Start-KBServer([hashtable]$envOverrides) {
    # start_tuned.ps1's env, then the arm's overrides on top.
    $e = Get-KBTunedEnv
    Get-ChildItem env: | Where-Object { $_.Name -like 'EXL3_MOE_*' } |
        ForEach-Object { Remove-Item "env:$($_.Name)" -ErrorAction SilentlyContinue }
    foreach ($k in $e.Keys) { Set-Item "env:$k" $e[$k] }
    if ($envOverrides) { foreach ($k in $envOverrides.Keys) { Set-Item "env:$k" $envOverrides[$k] } }

    $log = $KB.ServerLog
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $KB.Py
    $psi.Arguments = 'main.py --config config.yml'
    $psi.WorkingDirectory = $KB.Tabby
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $proc = New-Object Diagnostics.Process
    $proc.StartInfo = $psi
    $sb = New-Object Text.StringBuilder
    $onOut = { if ($EventArgs.Data -ne $null) { [void]$Event.MessageData.AppendLine($EventArgs.Data) } }
    Register-ObjectEvent -InputObject $proc -EventName OutputDataReceived -Action $onOut -MessageData $sb | Out-Null
    Register-ObjectEvent -InputObject $proc -EventName ErrorDataReceived  -Action $onOut -MessageData $sb | Out-Null

    # True process start, before any interpreter work. The server log's first
    # timestamp is already several seconds late (torch import), so boot time
    # measured from the log alone understates it.
    $t0 = Get-Date
    [void]$proc.Start()
    $proc.BeginOutputReadLine(); $proc.BeginErrorReadLine()

    $peak = 0; $minFree = [int]::MaxValue
    $deadline = (Get-Date).AddMinutes(25)
    $ready = $false; $trip = $null
    while ((Get-Date) -lt $deadline -and -not $proc.HasExited) {
        Start-Sleep -Seconds $KBGuard.PollSeconds
        if ($proc.HasExited) { break }
        $v = Get-KBVram
        if ($v) {
            if ($v.UsedMB -gt $peak) { $peak = $v.UsedMB }
            if ($v.FreeMB -lt $minFree) { $minFree = $v.FreeMB }
            if ($v.FreeMB -lt $KBGuard.VramFreeKillMB) { $trip = "VRAM free $($v.FreeMB)MB during load"; break }
        }
        $r = Get-KBRamMB
        if ($r -lt $KBGuard.RamFreeKillMB) { $trip = "RAM free ${r}MB during load"; break }
        $txt = $sb.ToString()
        if ($txt -match 'Serving OAI API on') { $ready = $true; break }
    }

    $sb.ToString() | Set-Content -Path $log -Encoding utf8
    if ($trip) {
        try { & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {}
        return [pscustomobject]@{ proc = $proc; sb = $sb; ready = $false; guard = $trip; t0 = $t0; peak = $peak; minFree = $minFree }
    }
    if (-not $ready) {
        if (-not $proc.HasExited) { try { & taskkill /PID $proc.Id /T /F 2>&1 | Out-Null } catch {} }
        return [pscustomobject]@{ proc = $proc; sb = $sb; ready = $false; guard = 'NOT_READY'; t0 = $t0; peak = $peak; minFree = $minFree }
    }

    $readyAt = Get-Date
    $r = [pscustomobject]@{
        proc = $proc; sb = $sb; ready = $true; guard = $null
        t0 = $t0; readyAt = $readyAt
        bootReadySec = [math]::Round(($readyAt - $t0).TotalSeconds, 1)
        peak = $peak; minFree = $minFree
    }
    return $r
}

function Stop-KBServer($srv) {
    if ($srv -and -not $srv.proc.HasExited) { try { & taskkill /PID $srv.proc.Id /T /F 2>&1 | Out-Null } catch {} }
    Start-Sleep -Seconds 3
    if ($srv) { Get-EventSubscriber | Unregister-Event -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

# Boot -> first token, measured with a streaming probe so the timing stops at
# the first content delta rather than at end-of-response.
function Measure-KBFirstToken($srv) {
    $hdr = Get-KBApiHeaders
    $body = @{ model = $KB.ServedModel; messages = @(@{ role = 'user'; content = 'Reply with the single word: ready' })
               max_tokens = 8; temperature = 0.0; stream = $true } | ConvertTo-Json -Depth 6
    $h = [Net.Http.HttpClient]::new()
    $req = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "http://127.0.0.1:5000/v1/chat/completions")
    foreach ($k in $hdr.Keys) { [void]$req.Headers.TryAddWithoutValidation($k, $hdr[$k]) }
    $req.Content = [Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $resp = $h.SendAsync($req, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    $stream = $resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
    $reader = New-Object IO.StreamReader($stream)
    $firstTokenSec = $null
    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        if ($line -match '"content"\s*:\s*"[^"]') { $firstTokenSec = [math]::Round($sw.Elapsed.TotalSeconds, 2); break }
    }
    $sw.Stop()
    $h.Dispose()
    if ($null -eq $firstTokenSec) { $firstTokenSec = [math]::Round($sw.Elapsed.TotalSeconds, 2) }
    return $firstTokenSec
}

function Get-KBApiHeaders {
    $raw = Get-Content (Join-Path $KB.Tabby 'api_tokens.yml') -Raw
    $h = @{}
    if ($raw -match 'api_key:\s*(\S+)') { $h['Authorization'] = "Bearer $($Matches[1])" }
    return $h
}

# Replay the real captured agentic_code request bodies. They are already OAI
# request shapes (messages/tools/tool_choice), so they go over the wire as-is.
function Get-KBRequestBodies([string[]]$files, [int]$maxTokens, [double]$temp) {
    $out = @()
    foreach ($f in $files) {
        $p = Resolve-KBPrompt $f
        if (-not (Test-Path $p)) { throw "missing prompt $p" }
        $d = Get-Content $p -Raw | ConvertFrom-Json
        $out += [pscustomobject]@{
            file = $f
            body = @{
                model = $KB.ServedModel
                messages = $d.messages
                tools = $d.tools
                tool_choice = $d.tool_choice
                max_tokens = $maxTokens
                temperature = $temp
                top_p = $(if ($temp -gt 0) { 0.95 } else { 1.0 })
                stream = $false
                chat_template_kwargs = @{ enable_thinking = $true }
            }
        }
    }
    return $out
}

function Invoke-KBRequest($b, $hdr) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        # Send BYTES, not the string. Invoke-WebRequest derives Content-Length
        # from the STRING length but writes encoded BYTES, so any non-ASCII
        # desynchronises them and the server reads a truncated body ->
        # 400 "There was an error parsing the body". ConvertFrom-Json unescapes
        # \uXXXX into real characters, so agentic_code_10 came out 74223 bytes
        # but 74193 chars. agentic_code_01 was pure ASCII and worked, which is
        # exactly why this looked content-dependent rather than systemic.
        $json = $b.body | ConvertTo-Json -Depth 12 -Compress
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/v1/chat/completions' -Method Post `
             -Body $bytes -Headers $hdr `
             -ContentType 'application/json' -TimeoutSec 900 -UseBasicParsing
        $sw.Stop()
        $j = $r.Content | ConvertFrom-Json
        return [pscustomobject]@{
            file = $b.file; ok = $true; status = $r.StatusCode
            wallSec = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            promptTokens = $j.usage.prompt_tokens; completionTokens = $j.usage.completion_tokens
            finish = $j.choices[0].finish_reason
        }
    } catch {
        $sw.Stop()
        # Print BOTH: Exception.Message is the client-side wrapper,
        # ErrorDetails.Message carries the server's JSON error body, which is
        # the only place a 400's reason shows up (the server logs nothing for a
        # request rejected before job creation, so the log looks empty).
        $msg = $_.Exception.Message
        $detail = $null
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $detail = $_.ErrorDetails.Message }
        Write-Host ("    FAIL {0}: {1}" -f $b.file, $msg) -ForegroundColor Red
        if ($detail) { Write-Host ("      server said: {0}" -f ($detail.Substring(0, [math]::Min(400, $detail.Length)))) -ForegroundColor Red }
        return [pscustomobject]@{ file = $b.file; ok = $false; status = 0
                                  wallSec = [math]::Round($sw.Elapsed.TotalSeconds, 2)
                                  err = $msg; errDetail = $detail }
    }
}

# The server already prints every metric per request:
#   #2 chat/completions: 33 tokens generated at 24.3 T/s · prompt 182,518 tokens,
#   none cached, 182,518 new in 101.0 s (1,807 T/s) · first token 101.0 s,
#   total 102.3 s · draft 24/45 accepted (53%)
function ConvertFrom-KBServerLog([string]$path) {
    $txt = Get-KBPlain $path
    # tabbyAPI's logger hard-wraps at ~70 chars, so one logical record spans
    # several physical lines ("... 256 tokens generated at 33.7 T/s ·" then an
    # indented "prompt 17,830 tokens, 62% cached, 6,822 new in 5.56 s"). Parse
    # line-by-line and every field after the first wrap is invisible. Rejoin
    # continuations first: a new record always starts with a timestamp.
    $entries = @()
    $buf = $null
    foreach ($line in ($txt -split "`n")) {
        if ($line -match '^\s*\d{2}:\d{2}:\d{2}\.\d{3}\s+(INFO|WARN|ERROR|DEBUG)') {
            if ($buf) { $entries += $buf }
            $buf = $line
        } elseif ($null -ne $buf) {
            $buf = $buf + ' ' + $line.Trim()
        }
    }
    if ($buf) { $entries += $buf }

    $reqs = @()
    $meta = @{}
    foreach ($line in $entries) {
        if ($line -match 'exllamav3 version:\s*(\S+)') { $meta.version = $Matches[1] }
        if ($line -match 'cache_size\s+([\d,]+)\s*tokens') { $meta.cacheSize = $Matches[1] }
        if ($line -match 'Model loaded in\s+([\d.]+)\s*s') { $meta.loadSec = [double]$Matches[1] }
        if ($line -match 'Configured backend:\s*(\S+)') { $meta.backend = $Matches[1] }
        if ($line -match 'draft_num_tokens|draft_mode') { $meta.draftCfg = $line.Trim() }
        if ($line -match 'Serving OAI API on\s+(\S+)') {
            if ($line -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)') { $meta.readyTs = $Matches[1] }
        }
        if ($line -match 'chat/completions:\s*(\d+)\s+tokens generated at\s+([\d.]+)\s*T/s') {
            $e = @{ id = [int]$Matches[1]; genTokens = [int]$Matches[1]; tgTps = [double]$Matches[2] }
            if ($line -match 'new in\s+([\d,.]+)\s*s') { $e.ppSec = [double]($Matches[1] -replace ',', '') }
            # Prefill rate is only printed when `new` exceeds 1000 tokens (drops
            # the rate entirely for small turns), so gate on it -- otherwise a
            # rate-less turn still parses and the median lands on noise.
            if ($line -match '([\d,]+)\s+new in' -and [int]($Matches[1] -replace ',', '') -gt 1000) {
                if ($line -match '\(([\d,]+)\s+T/s\)') { $e.ppTps = [double]($Matches[1] -replace ',', '') }
            }
            if ($line -match 'first token\s+([\d.]+)\s*s') { $e.firstTokenSec = [double]$Matches[1] }
            if ($line -match 'total\s+([\d.]+)\s*s') { $e.totalSec = [double]$Matches[1] }
            if ($line -match 'draft\s+(\d+)/(\d+)\s+accepted') { $e.draftAcc = [int]$Matches[1]; $e.draftOf = [int]$Matches[2] }
            if ($line -match 'temperature:\s*([\d.]+)') { $e.temp = [double]$Matches[1] }
            $reqs += $e
        }
    }
    return @{ Meta = $meta; Requests = $reqs }
}

# ----------------------------------------------------------------------- arms
# Config knobs are applied to config.yml; Env knobs to the process env only.
$arms = @{
    'c00-baseline'     = @{ cfg = @{}; env = @{} }
    'c01-ambs1'        = @{ cfg = @{ max_batch_size = '1' }; env = @{} }
    'c02-warmup'       = @{ cfg = @{ warmup = 'true' }; env = @{} }
    'c03-rci512'       = @{ cfg = @{ recurrent_checkpoint_interval = '512' }; env = @{} }
    'c04-rci4096'      = @{ cfg = @{ recurrent_checkpoint_interval = '4096' }; env = @{} }
    'c05-mallocasync-off' = @{ cfg = @{ cuda_malloc_async = 'False' }; env = @{} }
    'c06-dcm22'        = @{ cfg = @{ draft_cache_mode = '2,2' }; env = @{} }
    'c07-dcm33'        = @{ cfg = @{ draft_cache_mode = '3,3' }; env = @{} }
    'c08-dcmQ8'        = @{ cfg = @{ draft_cache_mode = 'Q8' }; env = @{} }
    'c09-dcmFP16'      = @{ cfg = @{ draft_cache_mode = 'FP16' }; env = @{} }
    # env ablations re-measured live, to confirm Phase A's offline verdict holds
    'c10-pin0'         = @{ cfg = @{}; env = @{ EXL3_MOE_CPU_PIN = '0' } }
    'c11-zc0'          = @{ cfg = @{}; env = @{ EXL3_MOE_ZERO_COPY = '0' } }
    'c12-thr2'         = @{ cfg = @{}; env = @{ EXL3_MOE_CPU_THREADS = '2' } }
    # The Tier 1 winner, validated end-to-end on the real server rather than
    # only through raw forwards. Offline and live share the env knob, so this is
    # the check that the +12.8% survives the server's own scheduling.
    'c13-thr16'        = @{ cfg = @{}; env = @{ EXL3_MOE_CPU_THREADS = '16' } }
    'c14-thr24'        = @{ cfg = @{}; env = @{ EXL3_MOE_CPU_THREADS = '24' } }
    'c15-offload36'    = @{ cfg = @{ cpu_moe_offload_layers = '36' }; env = @{} }
    # Split-experts wins once N drops below the VRAM-matched point: 360 reached
    # 29.98 tg offline (mcl34's speed) with 1653 MB free where mcl34 tripped the
    # guard at 129 MB. 380 gives up 0.29 tg/s (noise) for 1.4 GB more headroom.
    'c16-mcs380'       = @{ cfg = @{ cpu_moe_split_experts = '380' }; cfgDrop = @('cpu_moe_offload_layers'); env = @{} }
    'c17-mcs360'       = @{ cfg = @{ cpu_moe_split_experts = '360' }; cfgDrop = @('cpu_moe_offload_layers'); env = @{} }
    # The only value between 360 (measured: will not boot live) and 380 (boots,
    # 565-1365MB projected live headroom). mcs N = TAIL N experts on CPU, so
    # LOWER is both faster AND larger in footprint -- offline min-free falls
    # 3065 (380) -> 2506 (375, interpolated) -> 1948 (370, interpolated).
    # Projected live headroom at 375 is -9..590MB, i.e. right at the 200MB guard,
    # and 370 is negative. 375 is therefore the only refinement worth testing.
    'c26-mcs375'       = @{ cfg = @{ cpu_moe_split_experts = '375' }; cfgDrop = @('cpu_moe_offload_layers'); env = @{} }
    'c27-mcs390'       = @{ cfg = @{ cpu_moe_split_experts = '390' }; cfgDrop = @('cpu_moe_offload_layers'); env = @{} }
    # Plan section 2: combine the top-2 winners once.
    'c18-combo'        = @{ cfg = @{ cpu_moe_split_experts = '380' }; cfgDrop = @('cpu_moe_offload_layers'); env = @{ EXL3_MOE_CPU_THREADS = '16' } }

    # 2x2 isolation of the two standing keys. vision_offload moved VRAM the
    # WRONG way (20546 -> 21136 MB peak) and warmup captures CUDA graphs, which
    # cost persistent VRAM -- so neither can be judged without the other held
    # fixed. Standing config is (vision_offload=true, warmup=true); these two
    # arms flip one key each, and c21 flips both back to the original defaults.
    # Arms override standing keys because standing is applied first.
    'c19-vo-on-warm-off' = @{ cfg = @{ warmup = 'false' }; env = @{} }
    'c20-vo-off-warm-on' = @{ cfg = @{ vision_offload = 'false' }; env = @{} }
    'c21-both-off'       = @{ cfg = @{ vision_offload = 'false'; warmup = 'false' }; env = @{} }

    # §0.9 second-tier KV cache. Currently 0, i.e. OFF. Its real mechanism
    # (generator/generator.py:122) is NOT VRAM relief: complete K/V pages
    # EVICTED FROM THE GPU CACHE are held in pinned sysmem and RESTORED ON
    # PROMPT-CACHE HITS INSTEAD OF BEING RECOMPUTED BY PREFILL. So the payoff is
    # a cheaper cache hit, and it only appears where pages actually evict -- at
    # 11k-17k prompt tokens nothing evicts, so the interesting test is at long
    # context. These arms exist to measure it in VRAM/RAM terms first.
    'c22-sysmem-kv-8g'   = @{ cfg = @{ sysmem_kv_cache = '8192' }; env = @{} }
    'c23-sysmem-kv-24g'  = @{ cfg = @{ sysmem_kv_cache = '24576' }; env = @{} }
    # §0.9 checkpoint pair. Each checkpoint costs 148 MiB of sysmem for this
    # 48-recurrent-layer model, so rci_pp density is bought with RAM. Default
    # is 32768; 8192 gives a denser grid and cheaper mid-conversation edits.
    'c24-rci-pp-8192'    = @{ cfg = @{ recurrent_checkpoint_interval_pp = '8192' }; env = @{} }
    # Untested direction for the decode-side grid: I only measured 512, which
    # made prefill better AND VRAM worse -- opposite to the usual
    # denser-grid-costs-more intuition, so the other side needs confirming.
    'c25-rci-4096'       = @{ cfg = @{ recurrent_checkpoint_interval = '4096' }; env = @{} }
}
if (-not $arms.ContainsKey($Arm)) { throw "unknown arm '$Arm'. known: $($arms.Keys -join ', ')" }
$spec = $arms[$Arm]

# Files: Tier-2 = all 5 agentic_code, both arms, 3 reps. Otherwise one file,
# greedy only, which is enough for boot/VRAM/RAM and a first look.
if ($FullProtocol) {
    $files = @('agentic_code_01.json', 'agentic_code_05.json', 'agentic_code_10.json',
               'agentic_code_20.json', 'agentic_code_29.json')
    $temps = @(0.0, 1.0); $reps = 3
} else {
    $files = @('agentic_code_10.json'); $temps = @(0.0); $reps = 1
}

# Sustained turns: one real multi-turn agentic-code conversation by default.
# A value containing a path separator is used verbatim, which is how the
# synthesised long-context prompts (_kb_mklongctx.py, plan §0.7) are fed in
# from logs\kb\longctx\.
if ($Conversation.Count -eq 0) { $Conversation = @('agentic_code_10.json') }
# A caller passing several paths through a shell often lands them here as ONE
# comma-joined string rather than an array (Windows quoting does not split on
# commas). Split defensively so a long-context run does not quietly measure
# nothing -- and fail fast on a bad path rather than after a server boot.
$Conversation = @($Conversation | ForEach-Object { $_ -split ',' } |
                  ForEach-Object { $_.Trim() } | Where-Object { $_ })

function Resolve-KBPrompt([string]$name) {
    if ($name -match '[\\/]') { return $name }
    return (Join-Path $KB.Prompts $name)
}

$KB.ServedModel = (Get-KBConfigValues @('model_name')).model_name
# Validate every sustained prompt BEFORE booting. Discovering a bad path after a
# 65 s boot wastes the run; the 35k pass lost two boots to exactly this.
foreach ($cf in $Conversation) {
    $cp = Resolve-KBPrompt $cf
    if (-not (Test-Path $cp)) { throw "conversation prompt missing: $cp" }
    Write-Host ("conv prompt OK: {0} ({1:N0} bytes)" -f $cf, (Get-Item $cp).Length)
}
Write-Host ("==== PHASE C arm {0} ====" -f $Arm) -ForegroundColor Cyan
Write-Host ("served model: {0}" -f $KB.ServedModel)
Write-Host ("files={0} temps={1} reps={2} concurrent={3} fullProtocol={4}" -f `
            ($files -join ','), ($temps -join ','), $reps, $Concurrent, [bool]$FullProtocol)

$v0 = Get-KBVram; $r0 = Get-KBRamMB
Write-Host ("pre: vram free={0}MB ram free={1}MB" -f $v0.FreeMB, $r0)
if ($r0 -lt $KBGuard.RamFreeMinBeforeMB) { throw "ABORT (guard): ram free ${r0}MB < $($KBGuard.RamFreeMinBeforeMB)MB before run" }

# ---- backup config.yml, apply this arm's knobs, and report the intended values
$cfgBak = "$($KB.Config).kb-$Arm"
Copy-Item $KB.Config $cfgBak -Force
$applied = @()
$srv = $null
try {
    foreach ($k in $KBStandingOrder) {
        $a = if ($KBStandingAnchor.ContainsKey($k)) { $KBStandingAnchor[$k] } else { '' }
        $applied += Set-KBConfig $k $KBStandingCfg[$k] -Anchor $a
    }
    # Keys absent from config.yml but present in config_sample.yml need an anchor
    # so they land in the right section instead of being appended at EOF (where
    # the loader would ignore them -- a silent no-op). Set-KBConfig throws
    # rather than guessing.
    $KBKeyAnchor = @{
        recurrent_checkpoint_interval_pp = 'recurrent_checkpoint_interval'
        sysmem_multimodal_cache          = 'sysmem_kv_cache'
    }
    foreach ($k in ($spec.cfg.Keys | Sort-Object)) {
        $a = if ($KBKeyAnchor.ContainsKey($k)) { $KBKeyAnchor[$k] } else { '' }
        $applied += Set-KBConfig $k $spec.cfg[$k] -Anchor $a
    }
    if ($spec.cfgDrop) { foreach ($k in $spec.cfgDrop) { $applied += Set-KBConfigDrop $k } }
    Write-Host ("config edits: {0}" -f $(if ($applied) { ($applied | ForEach-Object { "$($_.key)=$($_.value)@line$($_.line)" }) -join ' ' } else { 'none (baseline)' }))

    if ($DryRun) {
        Write-Host "`n-- DRY RUN: config.yml after edits --" -ForegroundColor Cyan
        foreach ($k in ($applied | ForEach-Object { $_.key })) {
            $hits = @(Select-String -Path $KB.Config -Pattern "^\s*$([regex]::Escape($k))\s*:" )
            Write-Host ("  {0}: {1} uncommented occurrence(s)" -f $k, $hits.Count) -ForegroundColor DarkGray
            foreach ($h in $hits) { Write-Host ("      line {0}: {1}" -f $h.LineNumber, $h.Line.Trim()) }
        }
        Write-Host "`n-- full model/draft/memory section diff --" -ForegroundColor Cyan
        $now = Get-Content $KB.Config
        $orig = Get-Content $cfgBak
        Compare-Object $orig $now | ForEach-Object {
            $s = if ($_.SideIndicator -eq '=>') { 'NEW' } else { 'OLD' }
            Write-Host ("  {0}: {1}" -f $s, $_.InputObject.Trim()) -ForegroundColor DarkGray
        }
        Write-Host "`n-- DRY RUN complete (config.yml will be restored) --"
        return
    }

    $srv = Start-KBServer $spec.env
    if (-not $srv.ready) {
        Write-Host ("SERVER NOT READY: {0}" -f $srv.guard) -ForegroundColor Red
    } else {
        $ft = Measure-KBFirstToken $srv
        Write-Host ("boot: t0->ready {0}s ; ready->first token {1}s ; BOOT->FIRST TOKEN {2}s ; peakVRAM {3}MB minFree {4}MB" -f `
                    $srv.bootReadySec, $ft, [math]::Round($srv.bootReadySec + $ft, 1), $srv.peak, $srv.minFree)

        $hdr = Get-KBApiHeaders
        # FIRST real request, BEFORE the plan's 2 throwaway warmups (SS0.3).
        # This is the only point where warmup:true can show its value: it moves
        # kernel compile / autotune / graph capture into load, so the first
        # request should stop paying for them. Measuring it after the
        # throwaways -- which is what the harness did originally -- masks the
        # effect completely and makes warmup look like pure +12s boot cost.
        $firstProbe = (Get-KBRequestBodies @('agentic_code_10.json') 256 0.0)[0]
        $fr = Invoke-KBRequest $firstProbe $hdr
        Write-Host ("  FIRST-REQUEST (pre-warmup, DEMOTED metric): ok={0} {1}s ptok={2} ctok={3}" -f `
                    $fr.ok, $fr.wallSec, $fr.promptTokens, $fr.completionTokens)
        if ($fr.ok) {
            $rowFirst = @{ wallSec = $fr.wallSec }
        } else { $rowFirst = @{ wallSec = $null; err = $fr.err } }

        # 2 throwaway generations (plan §0.3 warmed server) before any measurement.
        $warm = Get-KBRequestBodies @('agentic_code_01.json') 64 0.0
        foreach ($w in $warm) { $r = Invoke-KBRequest $w $hdr; Write-Host ("  warmup {0}: {1} {2}s" -f $w.file, $r.ok, $r.wallSec) }

        $sustainedTps = @()
        if ($Sustained -gt 0 -and $Concurrent -le 1) {
            $mode = if ($Rotate) { 'rotate (cold prefix per turn)' } else { 'conversation (growing prefix)' }
            Write-Host ("`n  -- SUSTAINED x{0} turns, mode={1} (PRIMARY metric) --" -f $Sustained, $mode) -ForegroundColor Cyan

            # Build the turn list: a sequence of request bodies.
            $turns = @()
            if ($Rotate) {
                # One conversation per turn: each request is independent, so its
                # prefix is cold and pp is genuinely measured.
                for ($i = 1; $i -le $Sustained; $i++) {
                    $pf = $Conversation[[Math]::Min($i - 1, $Conversation.Count - 1)]
                    $turns += ,(Get-KBRequestBodies @($pf) 256 0.0)[0]
                }
            } else {
                # Progressive replay of ONE conversation: turn i sends messages
                # [0..i], so each turn extends the history (prompt -> answer ->
                # prompt). Prefix cache stays warm like real use, prompt content
                # still changes per turn, and prefill grows with the history.
                $src = $Conversation[0]
                $pd = Get-Content (Resolve-KBPrompt $src) -Raw | ConvertFrom-Json
                $msgs = @($pd.messages)
                # Walk forward to each user/tool turn; that index is the cut point
                # for "everything before this is history, model answers next".
                $cuts = @()
                for ($i = 0; $i -lt $msgs.Count; $i++) {
                    if ($msgs[$i].role -eq 'user' -or $msgs[$i].role -eq 'tool') { $cuts += $i }
                }
                foreach ($c in ($cuts | Select-Object -First $Sustained)) {
                    $hist = @()
                    for ($j = 0; $j -le $c; $j++) { $hist += $msgs[$j] }
                    $turns += [pscustomobject]@{
                        file = "$src#$c"
                        body = @{
                            model = $KB.ServedModel; messages = $hist; tools = $pd.tools
                            tool_choice = $pd.tool_choice; max_tokens = 256; temperature = 0.0
                            top_p = 1.0; stream = $false
                            chat_template_kwargs = @{ enable_thinking = $true }
                        }
                    }
                }
            }

            $ti = 0
            foreach ($tb in $turns) {
                $ti++
                $rs = Invoke-KBRequest $tb $hdr
                $sustainedTps += [pscustomobject]@{ i = $ti; file = $tb.file; ok = $rs.ok; wallSec = $rs.wallSec }
                Write-Host ("     turn {0}/{1} [{2}]: ok={3} {4}s ptok={5}" -f `
                            $ti, $turns.Count, $tb.file, $rs.ok, $rs.wallSec, $rs.promptTokens)
            }
            $Sustained = $turns.Count
        }

        $reqs = @()
        for ($rep = 1; $rep -le $reps; $rep++) {
            foreach ($t in $temps) {
                foreach ($f in $files) {
                    $b = (Get-KBRequestBodies @($f) 256 $t)[0]
                    if ($Concurrent -le 1) {
                        $r = Invoke-KBRequest $b $hdr
                        Write-Host ("  rep{0} temp{1} {2}: ok={3} {4}s ptok={5} ctok={6}" -f `
                                    $rep, $t, $f, $r.ok, $r.wallSec, $r.promptTokens, $r.completionTokens)
                        $reqs += $r
                    } else {
                        $bs = @($f | ForEach-Object { (Get-KBRequestBodies @($_) 256 $t)[0] })
                        $sw = [Diagnostics.Stopwatch]::StartNew()
                        $jobs = @($bs | ForEach-Object { Start-Job -ScriptBlock {
                            param($bodyJson, $hdrJson)
                            $h = ConvertFrom-Json $hdrJson
                            try {
                                # Encode to bytes in the child too -- same
                                # Content-Length desync as the serial path.
                                $jb = [System.Text.Encoding]::UTF8.GetBytes($bodyJson)
                                $resp = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/v1/chat/completions' -Method Post `
                                       -Body $jb -Headers $h -ContentType 'application/json' -TimeoutSec 900 -UseBasicParsing
                                $j = $resp.Content | ConvertFrom-Json
                                [pscustomobject]@{ ok = $true; ctok = $j.usage.completion_tokens }
                            } catch { [pscustomobject]@{ ok = $false; ctok = 0; err = $_.Exception.Message } }
                        } -ArgumentList ($_.body | ConvertTo-Json -Depth 12 -Compress), ($hdr | ConvertTo-Json -Compress) })
                        $null = Wait-Job $jobs
                        $out = @($jobs | Receive-Job)
                        Remove-Job $jobs -Force
                        $sw.Stop()
                        $totTok = ($out | Measure-Object -Property ctok -Sum).Sum
                        $aggTps = if ($sw.Elapsed.TotalSeconds -gt 0) { [math]::Round($totTok / $sw.Elapsed.TotalSeconds, 2) } else { 0 }
                        Write-Host ("  rep{0} temp{1} CONCURRENT x{2}: {3} tok in {4}s = {5} tok/s aggregate" -f `
                                    $rep, $t, $bs.Count, $totTok, [math]::Round($sw.Elapsed.TotalSeconds, 1), $aggTps)
                        $reqs += [pscustomobject]@{ file = "$($bs.Count)x$f"; ok = ($out.ok -notcontains $false)
                                                   wallSec = [math]::Round($sw.Elapsed.TotalSeconds, 2)
                                                   completionTokens = $totTok; aggregateTps = $aggTps }
                    }
                }
            }
        }

        # flush the event-captured server log, then read the metrics out of it
        Start-Sleep -Seconds 3
        $srv.sb.ToString() | Set-Content -Path $KB.ServerLog -Encoding utf8
        # Capture the load-time properties that are only in the server log and are
        # the leading suspect for the 250k fast/slow mode. Recorded per boot
        # because they differ BETWEEN boots while the config does not.
        $logText = $srv.sb.ToString()
        foreach ($pat in @('CPU MoE arena: ([\d\.]+) GB of ([\d\.]+) GB',
                           'CPU MoE worker started: ([^\r\n]+)',
                           'CPU split experts \(worker, dynamic\): ([^\r\n]+)')) {
            $m = [regex]::Match($logText, $pat)
            if ($m.Success) {
                Write-Host ("  boot-prop: {0}" -f $m.Value.Trim()) -ForegroundColor DarkGray
                $script:BootProps += @{ pat = $pat.Substring(0, [Math]::Min(20, $pat.Length)); val = $m.Value.Trim() }
            }
        }
        $parsed = ConvertFrom-KBServerLog $KB.ServerLog
        Write-Host ("`nserver meta: ver={0} load={1}s cache={2} backend={3}" -f `
                    $parsed.Meta.version, $parsed.Meta.loadSec, $parsed.Meta.cacheSize, $parsed.Meta.backend)
        Write-Host ("{0,-6} {1,8} {2,9} {3,9} {4,10} {5,11}" -f 'id', 'genTok', 'tgTps', 'ppTps', 'firstTok', 'draftAcc')
        foreach ($q in $parsed.Requests) {
            Write-Host ("{0,-6} {1,8} {2,9} {3,9} {4,10} {5,11}" -f $q.id, $q.genTokens, $q.tgTps,
                        $(if ($q.ppTps) { $q.ppTps } else { '-' }), $(if ($q.firstTokenSec) { $q.firstTokenSec } else { '-' }),
                        $(if ($null -ne $q.draftAcc) { "$($q.draftAcc)/$($q.draftOf)" } else { '-' }))
        }

        $vramAfter = Get-KBVram; $ramAfter = Get-KBRamMB

        # Steady-state distribution from the sustained block, parsed out of the
        # server log (authoritative per-request tg/pp). Median + min-max is what
        # production perf means here; a single sample cannot.
        $sustTgs = @(); $sustPps = @()
        $rowSustTg = $null; $rowSustTgMin = $null; $rowSustTgMax = $null
        $rowSustPp = $null; $spread = $null
        if ($Sustained -gt 0 -and $parsed.Requests.Count -gt 0) {
            # Drop the probe/warmup entries: keep the LAST $Sustained requests,
            # which are exactly the sustained block issued above.
            $tail = @($parsed.Requests | Select-Object -Last $Sustained)
            foreach ($q in $tail) {
                if ($q.tgTps) { $sustTgs += $q.tgTps }
                # pp only where the server actually printed a rate (see the
                # >1000-token gate in ConvertFrom-KBServerLog). Averaging in
                # rate-less turns is what produced the bogus "pp median=221".
                if ($q.ppTps) { $sustPps += $q.ppTps }
            }
            if ($sustTgs.Count -gt 0) {
                $st = $sustTgs | Sort-Object
                $sp = $sustPps | Sort-Object
                $mid = [int](($st.Count - 1) / 2)
                $rowSustTg = [math]::Round($st[$mid], 2)
                $rowSustTgMin = [math]::Round($st[0], 2)
                $rowSustTgMax = [math]::Round($st[-1], 2)
                if ($sp.Count -gt 0) {
                    $rowSustPp = [math]::Round($sp[[int](($sp.Count - 1) / 2)], 0)
                }
                $spread = if ($rowSustTgMin -gt 0) {
                    [math]::Round(100 * ($rowSustTgMax - $rowSustTgMin) / $rowSustTgMin, 1)
                } else { $null }
                Write-Host ("`n  SUSTAINED tg median={0} min={1} max={2} spread={3}%  pp median={4}" -f `
                            $rowSustTg, $rowSustTgMin, $rowSustTgMax, $spread, $rowSustPp) -ForegroundColor Green
            }
        }
        $row = [ordered]@{
            arm = $Arm; ts = Get-KBStamp
            # PRIMARY: steady-state production perf, sustained within one boot.
            sustainedN = $Sustained
            sustTgMedian = $rowSustTg; sustTgMin = $rowSustTgMin
            sustTgMax = $rowSustTgMax; sustTgSpreadPct = $spread
            sustPpMedian = $rowSustPp
            # DEMOTED: boot and first-request numbers. Kept because they are
            # cheap and occasionally diagnostic, not because they drive config.
            firstReqWallSec = $rowFirst.wallSec
            configApplied = (($applied | ForEach-Object { "$($_.key)=$($_.value)" }) -join ' ')
            envOverrides = (($spec.env.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ' ')
            bootReadySec = $srv.bootReadySec; readyToFirstTokenSec = $ft
            bootToFirstTokenSec = [math]::Round($srv.bootReadySec + $ft, 1)
            loadSec = $parsed.Meta.loadSec; cacheSize = $parsed.Meta.cacheSize; backend = $parsed.Meta.backend
            exl3Version = $parsed.Meta.version
            vramPeak = $srv.peak; vramMinFreeDuringLoad = $srv.minFree; vramFreeAfter = $vramAfter.FreeMB
            ramFreeBefore = $r0; ramFreeAfter = $ramAfter
            requests = $reqs.Count; serverRequests = $parsed.Requests.Count
            # per-boot load-time properties, e.g. CPU MoE arena reservation
            bootProps = @($script:BootProps | ForEach-Object { $_.val })
            log = $KB.ServerLog
        }
        Add-Content -Path $KB.Jsonl -Value ($row | ConvertTo-Json -Compress -Depth 6) -Encoding utf8
        Write-Host ("`nrow -> {0}" -f $KB.Jsonl)
    }
} finally {
    # Server FIRST: cleanup lived only in the success branch, so any throw after
    # the server came up left it running and holding ~20 GB VRAM. The next arm
    # then aborts on the pre-run guard for no real reason.
    if ($srv) { Stop-KBServer $srv }
    # config.yml is restored no matter how this arm ends
    Copy-Item $cfgBak $KB.Config -Force
    Write-Host ("config.yml restored from {0}" -f (Split-Path $cfgBak -Leaf))
}

Write-Host ("`npost: vram free={0}MB ram free={1}MB" -f (Get-KBVram).FreeMB, (Get-KBRamMB))